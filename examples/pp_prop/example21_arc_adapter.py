"""Real BrainCell and ARC bridge for the Example 21 evolution lifecycle.

The coordinator in :mod:`example21_evolve` owns policy and durability.  This
module owns direct ARC bytes, compiled model execution, sparse mutation, Muon
continuation state, and format-1 checkpoint identities.  Evaluation corpus
access is deliberately lazy and occurs only through ``evaluation_manifest`` or
``evaluate_terminal``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

_TOPOLOGY_ARRAYS = (
    "neuron_ids",
    "dale_codes",
    "owner_codes",
    "mechanism_codes",
    "neuron_count",
    "integration_substeps",
    "input_indptr",
    "input_indices",
    "recurrent_indptr",
    "recurrent_indices",
)
_PARAMETER_ARRAYS = (
    "input_values",
    "recurrent_values",
    "readout_weight",
    "readout_bias",
)
_OPTIMIZER_ARRAYS = (
    "input_m1",
    "input_m2",
    "recurrent_m1",
    "recurrent_m2",
    "readout_weight_m1",
    "readout_weight_m2",
    "readout_bias_m1",
    "readout_bias_m2",
    "input_step",
    "recurrent_step",
    "readout_step",
)
_EXPECTED_UPDATES = 128


@dataclass(frozen=True)
class CheckpointIdentities:
    """Independent identities and persistent size for one checkpoint.

    Parameters
    ----------
    topology_sha256, parameters_sha256, optimizer_sha256 : str
        Stable SHA-256 identities for the three continuation components.
    persistent_bytes : int
        Sum of all format-1 array bytes before container compression.
    """

    topology_sha256: str
    parameters_sha256: str
    optimizer_sha256: str
    persistent_bytes: int


@dataclass(frozen=True)
class SupervisedQuery:
    """One immutable encoded target-bearing ARC query.

    Parameters
    ----------
    task_id : str
        Stable ARC task identifier.
    query_index : int
        Query position in the source task.
    events, advances : numpy.ndarray
        Fixed 705-event input and its Boolean advance mask.
    target : numpy.ndarray
        Direct target color grid.
    """

    task_id: str
    query_index: int
    events: np.ndarray
    advances: np.ndarray
    target: np.ndarray


@dataclass
class _Runtime:
    model: Any
    learner: Any
    trainer: Any
    topology: Any
    optimizer: Any
    readout_bias: np.ndarray


@dataclass(frozen=True)
class _ScoredRuntime:
    score: Any
    owners: tuple[tuple[int, ...], ...]
    owner_codes: np.ndarray
    neuron_scores: np.ndarray
    source_scores: np.ndarray
    target_scores: np.ndarray
    edge_scores: np.ndarray


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_digest(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        if name not in arrays:
            raise ValueError(f"Checkpoint is missing {name}; provide format-1 arrays.")
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def checkpoint_identities(
    arrays: Mapping[str, np.ndarray],
) -> CheckpointIdentities:
    """Return partitioned identities for format-1 continuation arrays.

    Parameters
    ----------
    arrays : mapping
        Complete format-1 checkpoint arrays.

    Returns
    -------
    CheckpointIdentities
        Topology, parameter, optimizer, and persistent-byte evidence.
    """

    values = {name: np.asarray(value) for name, value in arrays.items()}
    required = set(_TOPOLOGY_ARRAYS + _PARAMETER_ARRAYS + _OPTIMIZER_ARRAYS)
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(
            "Checkpoint identity requires complete format-1 arrays; missing "
            + ", ".join(missing)
            + "."
        )
    return CheckpointIdentities(
        _array_digest(values, _TOPOLOGY_ARRAYS),
        _array_digest(values, _PARAMETER_ARRAYS),
        _array_digest(values, _OPTIMIZER_ARRAYS),
        sum(int(value.nbytes) for value in values.values()),
    )


def _log_cross_entropy(logits: np.ndarray, label: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not 0 <= int(label) < values.size:
        raise ValueError("Cross-entropy label is outside its one-dimensional logits.")
    if not np.all(np.isfinite(values)):
        return math.inf
    maximum = float(np.max(values))
    return float(
        maximum + np.log(np.sum(np.exp(values - maximum))) - values[int(label)]
    )


def direct_query_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    decode_prediction: Callable[[np.ndarray], np.ndarray],
) -> tuple[bool, float]:
    """Measure direct exactness and canonical shape-and-cell loss.

    Parameters
    ----------
    logits : numpy.ndarray
        One shape request and 30 row requests with shape ``(31, 360)``.
    target : numpy.ndarray
        Target grid with integer colors zero through nine.
    decode_prediction : callable
        Direct Example 21 request decoder.

    Returns
    -------
    tuple
        Exact-grid Boolean and height plus width plus mean valid-cell
        cross-entropy.  Non-finite output returns ``(False, inf)``.
    """

    values = np.asarray(logits)
    expected = np.asarray(target)
    if values.shape != (31, 360):
        raise ValueError("Direct ARC logits must have shape (31, 360).")
    if (
        expected.ndim != 2
        or not 0 < expected.shape[0] <= 30
        or not 0 < expected.shape[1] <= 30
        or not np.issubdtype(expected.dtype, np.integer)
        or np.any((expected < 0) | (expected > 9))
    ):
        raise ValueError("Direct ARC target must be a 1 through 30 integer color grid.")
    if not np.all(np.isfinite(values)):
        return False, math.inf
    height, width = expected.shape
    shape_loss = _log_cross_entropy(values[0, :30], height - 1)
    shape_loss += _log_cross_entropy(values[0, 30:60], width - 1)
    row_values = values[1:, 60:].reshape((30, 30, 10))
    cell_losses = [
        _log_cross_entropy(row_values[row, column], int(expected[row, column]))
        for row in range(height)
        for column in range(width)
    ]
    prediction = np.asarray(decode_prediction(values))
    exact = bool(
        prediction.shape == expected.shape
        and np.issubdtype(prediction.dtype, np.integer)
        and np.array_equal(prediction, expected)
    )
    return exact, float(shape_loss + np.mean(cell_losses))


def owner_codes(task_neuron_scores: np.ndarray) -> NDArray[np.int16]:
    """Encode unique, shared, and inactive task ownership per neuron.

    Parameters
    ----------
    task_neuron_scores : numpy.ndarray
        Nonnegative task-by-neuron evidence.

    Returns
    -------
    numpy.ndarray
        Signed 16-bit owner indices.  ``-1`` means inactive and ``-2`` means
        that multiple tasks tie for maximum positive evidence.
    """

    scores = np.asarray(task_neuron_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError("Task ownership requires a nonempty task-by-neuron array.")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Task ownership evidence must be finite and nonnegative.")
    result = np.full(scores.shape[1], -1, dtype=np.int16)
    maximum = np.max(scores, axis=0)
    for neuron in np.flatnonzero(maximum > 0):
        owners = np.flatnonzero(scores[:, neuron] == maximum[neuron])
        result[neuron] = int(owners[0]) if len(owners) == 1 else -2
    return result


def _normalise(values: np.ndarray) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(result), initial=0.0))
    return np.zeros_like(result) if scale == 0.0 else result / scale


class Example21ArcAdapter:
    """Bridge the real Example 21 BrainCell implementation to its coordinator.

    Parameters
    ----------
    arc_root : path-like
        Raw ARC root containing ``data/training`` and ``data/evaluation``.
    contracts_module, structural_module, model_module : module, optional
        Dependency injection points for isolated tests.  Production callers
        omit them and receive lazy imports of the repository implementations.
    """

    def __init__(
        self,
        arc_root: str | os.PathLike[str],
        *,
        contracts_module: Any | None = None,
        structural_module: Any | None = None,
        model_module: Any | None = None,
    ) -> None:
        self.arc_root = Path(arc_root)
        self._contracts_value = contracts_module
        self._structural_value = structural_module
        self._model_value = model_module
        self._raw_manifests: dict[str, Any] = {}
        self._manifests: dict[str, Any] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._queries: dict[str, tuple[SupervisedQuery, ...]] = {}
        self._evidence_by_checkpoint: dict[str, _ScoredRuntime] = {}
        self._parent_by_checkpoint: dict[str, str | None] = {}
        self._temporary_paths: set[Path] = set()

    def _contracts(self) -> Any:
        if self._contracts_value is None:
            from examples.pp_prop import arc_contracts

            self._contracts_value = arc_contracts
        return self._contracts_value

    def _evolve(self) -> Any:
        from examples.pp_prop import example21_evolve

        return example21_evolve

    def _structural(self) -> Any:
        if self._structural_value is None:
            from examples.pp_prop import example21_structural

            self._structural_value = example21_structural
        return self._structural_value

    def _model(self) -> Any:
        if self._model_value is not None:
            return self._model_value
        expected = Path(__file__).with_name("21-braincell-arc.py").resolve()
        main = sys.modules.get("__main__")
        main_path = getattr(main, "__file__", None)
        if main_path is not None and Path(main_path).resolve() == expected:
            self._model_value = main
            return main
        module_name = "examples.pp_prop._example21_braincell_arc_runtime"
        existing = sys.modules.get(module_name)
        if existing is not None:
            self._model_value = existing
            return existing
        specification = importlib.util.spec_from_file_location(module_name, expected)
        if specification is None or specification.loader is None:
            raise ImportError(f"Cannot load the Example 21 model from {expected}.")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        self._model_value = module
        return module

    def _raw_manifest(self, role: str) -> Any:
        if role not in self._raw_manifests:
            contracts = self._contracts()
            raw_role = "practice" if role == "training" else "evaluation"
            self._raw_manifests[role] = contracts.load_corpus_manifest(
                self.arc_root,
                raw_role,
                allow_evaluation=role == "evaluation",
            )
        return self._raw_manifests[role]

    def _load_tasks(self, role: str) -> dict[str, Any]:
        if role in self._tasks:
            return self._tasks[role]
        contracts = self._contracts()
        raw = self._raw_manifest(role)
        raw_role = "practice" if role == "training" else "evaluation"
        tasks: dict[str, Any] = {}
        for source in raw.sources:
            task = contracts.load_task(
                self.arc_root,
                source.task_id,
                raw_role,
                allow_evaluation=role == "evaluation",
                manifest=raw,
            )
            missing = tuple(
                index for index, target in enumerate(task.targets) if target is None
            )
            if missing:
                raise ValueError(
                    f"ARC {role} task {source.task_id} has missing target queries "
                    f"{missing}; direct exact scoring requires every query target."
                )
            tasks[source.task_id] = task
        self._tasks[role] = tasks
        return tasks

    def _manifest(self, role: str) -> Any:
        if role in self._manifests:
            return self._manifests[role]
        evolve = self._evolve()
        raw = self._raw_manifest(role)
        tasks = self._load_tasks(role)
        query_order = tuple(
            (source.task_id, query_index)
            for source in raw.sources
            for query_index in range(len(tasks[source.task_id].targets))
        )
        manifest = evolve.CorpusManifest(
            role=role,
            task_ids=tuple(source.task_id for source in raw.sources),
            source_digests=tuple(source.source_sha256 for source in raw.sources),
            query_order=query_order,
        )
        manifest.validate()
        self._manifests[role] = manifest
        return manifest

    def training_manifest(self) -> Any:
        """Return all 400 sorted training tasks and supervised queries.

        Returns
        -------
        CorpusManifest
            Complete training-only evolution manifest.
        """

        return self._manifest("training")

    def evaluation_manifest(self) -> Any:
        """Open and return the terminal-only 400-task evaluation manifest.

        Returns
        -------
        CorpusManifest
            Complete held-out evaluation manifest.
        """

        return self._manifest("evaluation")

    def _encoded_queries(self, role: str) -> tuple[SupervisedQuery, ...]:
        if role in self._queries:
            return self._queries[role]
        contracts = self._contracts()
        tasks = self._load_tasks(role)
        records = []
        for task_id, query_index in self._manifest(role).query_order:
            task = tasks[task_id]
            events, advances = contracts.encode_episode(task, query_index)
            target = np.asarray(task.targets[query_index], dtype=np.int32)
            copied = (
                np.asarray(events, dtype=np.float32),
                np.asarray(advances, dtype=bool),
                np.array(target, dtype=np.int32, copy=True),
            )
            for value in copied:
                value.setflags(write=False)
            records.append(
                SupervisedQuery(task_id, query_index, copied[0], copied[1], copied[2])
            )
        self._queries[role] = tuple(records)
        return self._queries[role]

    @staticmethod
    def _mutation_kind(stage: str, arm: str) -> str:
        allowed = {
            ("edge", "add"): "edge-add",
            ("edge", "prune"): "edge-prune",
            ("edge-revisit", "add"): "edge-add",
            ("edge-revisit", "prune"): "edge-prune",
            ("compression-edge", "prune"): "edge-prune",
            ("neuron", "add"): "neuron-add",
            ("neuron", "prune"): "neuron-prune",
            ("compression-neuron", "prune"): "neuron-prune",
            ("dale", "excitatory"): "dale-excitatory",
            ("dale", "inhibitory"): "dale-inhibitory",
        }
        try:
            return allowed[(stage, arm)]
        except KeyError as error:
            raise ValueError(
                f"unsupported evolution arm {stage}/{arm}; use the coordinator stage order."
            ) from error

    def _topology_optimizer_from_arrays(
        self, arrays: Mapping[str, np.ndarray]
    ) -> tuple[Any, Any, np.ndarray]:
        structural = self._structural()
        neuron_count = int(np.asarray(arrays["neuron_count"]))
        integration_substeps = int(np.asarray(arrays["integration_substeps"]))
        mechanism_codes = np.asarray(arrays["mechanism_codes"], dtype=np.uint8)
        if integration_substeps != 1:
            raise ValueError(
                "Example 21 cannot execute non-default integration substeps; "
                "use integration_substeps=1."
            )
        if np.any(mechanism_codes != 0):
            raise ValueError(
                "Example 21 cannot execute deferred mechanism codes; "
                "use zero mechanism_codes."
            )
        input_indptr = np.asarray(arrays["input_indptr"], dtype=np.int32)
        recurrent_indptr = np.asarray(arrays["recurrent_indptr"], dtype=np.int32)
        topology = structural.SparseTopology(
            np.repeat(np.arange(441, dtype=np.int32), np.diff(input_indptr)),
            np.asarray(arrays["input_indices"], dtype=np.int32),
            np.asarray(arrays["input_values"], dtype=np.float32),
            np.repeat(
                np.arange(neuron_count, dtype=np.int32), np.diff(recurrent_indptr)
            ),
            np.asarray(arrays["recurrent_indices"], dtype=np.int32),
            np.asarray(arrays["recurrent_values"], dtype=np.float32),
            np.asarray(arrays["readout_weight"], dtype=np.float32),
            np.asarray(arrays["dale_codes"], dtype=np.int8),
            tuple(() for _ in range(neuron_count)),
            np.asarray(arrays["owner_codes"], dtype=np.int16),
            np.asarray(arrays["neuron_ids"], dtype=np.int32),
        )
        optimizer = structural.StructuralAdam(
            np.asarray(arrays["readout_weight_m1"], dtype=np.float32),
            np.asarray(arrays["readout_weight_m2"], dtype=np.float32),
            np.asarray(arrays["input_m1"], dtype=np.float32),
            np.asarray(arrays["input_m2"], dtype=np.float32),
            np.asarray(arrays["recurrent_m1"], dtype=np.float32),
            np.asarray(arrays["recurrent_m2"], dtype=np.float32),
            bias_first=np.asarray(arrays["readout_bias_m1"], dtype=np.float32),
            bias_second=np.asarray(arrays["readout_bias_m2"], dtype=np.float32),
            input_step=int(np.asarray(arrays["input_step"])),
            recurrent_step=int(np.asarray(arrays["recurrent_step"])),
            readout_step=int(np.asarray(arrays["readout_step"])),
        )
        topology, optimizer = structural.canonicalize_topology_and_optimizer(
            topology, optimizer
        )
        if not structural.validate_topology_dale(topology):
            raise ValueError(
                "Checkpoint violates an accepted Dale sign; recover the last valid parent."
            )
        return topology, optimizer, np.asarray(arrays["readout_bias"], dtype=np.float32)

    def _runtime_from_state(
        self, topology: Any, optimizer: Any, readout_bias: np.ndarray
    ) -> _Runtime:
        module = self._model()
        structural = self._structural()
        topology, optimizer = structural.canonicalize_topology_and_optimizer(
            topology, optimizer
        )
        model = module.BrainCellArcModel(topology)
        model.readout_bias.value = np.asarray(readout_bias, dtype=np.float32)
        learner = module.compile_pp_prop_model(model)
        trainer = module.PPPropEpisodeTrainer(
            learner,
            {
                "input": model.input_weight.value,
                "recurrent": model.recurrent_weight.value,
            },
        )
        trainer.muon_groups = structural.initialize_muon_groups(trainer, optimizer)
        trainer.updates = np.asarray(
            max(
                int(optimizer.input_step),
                int(optimizer.recurrent_step),
                int(optimizer.readout_step),
            ),
            dtype=np.int32,
        )
        trainer._sync_compiled_parameters()
        return _Runtime(
            model,
            learner,
            trainer,
            topology,
            optimizer,
            np.asarray(readout_bias, dtype=np.float32),
        )

    def _runtime_from_checkpoint(self, path: str | os.PathLike[str]) -> _Runtime:
        arrays = self._model().load_checkpoint(path)
        topology, optimizer, bias = self._topology_optimizer_from_arrays(arrays)
        return self._runtime_from_state(topology, optimizer, bias)

    def _fresh_runtime(self) -> _Runtime:
        module = self._model()
        structural = self._structural()
        model = module.BrainCellArcModel()
        learner = module.compile_pp_prop_model(model)
        trainer = module.PPPropEpisodeTrainer(
            learner,
            {
                "input": model.input_weight.value,
                "recurrent": model.recurrent_weight.value,
            },
        )
        topology = structural.topology_from_model(model)
        optimizer = structural.optimizer_from_muon_groups(trainer)
        return _Runtime(
            model,
            learner,
            trainer,
            topology,
            optimizer,
            np.asarray(model.readout_bias.value, dtype=np.float32),
        )

    @staticmethod
    def _peak_device_memory_bytes() -> int | None:
        try:
            import jax

            values = []
            for device in jax.devices():
                statistics = device.memory_stats()
                if statistics:
                    value = statistics.get(
                        "peak_bytes_in_use", statistics.get("bytes_in_use")
                    )
                    if value is not None:
                        values.append(int(value))
            return max(values) if values else None
        except (ImportError, RuntimeError, TypeError, ValueError):
            return None

    def _resources(
        self,
        arrays: Mapping[str, np.ndarray],
        path: str | os.PathLike[str],
    ) -> Any:
        evolve = self._evolve()
        structural = self._structural()
        identities = checkpoint_identities(arrays)
        peak_host = structural._peak_process_resident_memory_bytes()
        return evolve.ResourceUsage(
            persistent_bytes=identities.persistent_bytes,
            checkpoint_bytes=Path(path).stat().st_size,
            neurons=int(np.asarray(arrays["neuron_count"])),
            recurrent_edges=len(np.asarray(arrays["recurrent_values"])),
            peak_host_ram_bytes=peak_host,
            device_memory_bytes=self._peak_device_memory_bytes(),
        )

    def _snapshot(
        self,
        *,
        candidate_id: str,
        path: str | os.PathLike[str],
        arrays: Mapping[str, np.ndarray],
        score: Any,
        topology_changed: bool,
    ) -> Any:
        evolve = self._evolve()
        destination = Path(path)
        identities = checkpoint_identities(arrays)
        return evolve.CandidateSnapshot(
            candidate_id=candidate_id,
            checkpoint_path=str(destination.resolve()),
            checkpoint_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            topology_sha256=identities.topology_sha256,
            parameters_sha256=identities.parameters_sha256,
            optimizer_sha256=identities.optimizer_sha256,
            score=score,
            resources=self._resources(arrays, destination),
            topology_changed=bool(topology_changed),
        )

    def _candidate_path(self, output_dir: Path, candidate_id: str) -> Path:
        safe = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in candidate_id
        )
        directory = output_dir / ".candidates"
        directory.mkdir(parents=True, exist_ok=True)
        path = (directory / f"{safe}.npz").resolve()
        self._temporary_paths.add(path)
        return path

    @staticmethod
    def _provenance_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.provenance.json")

    def _record_lineage(
        self, candidate: Any, parent_checkpoint_sha256: str | None
    ) -> None:
        candidate_sha256 = candidate.checkpoint_sha256
        if not _is_sha256(candidate_sha256):
            raise ValueError(
                "Candidate checkpoint identity must be one SHA-256 digest."
            )
        if parent_checkpoint_sha256 is not None and not _is_sha256(
            parent_checkpoint_sha256
        ):
            raise ValueError(
                "Parent checkpoint identity must be one SHA-256 digest or None."
            )
        checkpoint_path = Path(candidate.checkpoint_path).resolve()
        if checkpoint_path.parent.name != ".candidates":
            self._parent_by_checkpoint[candidate_sha256] = parent_checkpoint_sha256
            return
        if not checkpoint_path.is_file():
            raise ValueError(
                "Cannot durably record lineage for a missing candidate checkpoint."
            )
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != candidate_sha256:
            raise ValueError(
                "Cannot durably record lineage for changed candidate checkpoint bytes."
            )
        provenance = self._provenance_path(checkpoint_path)
        temporary = provenance.with_name(f"{provenance.name}.tmp")
        payload = {
            "candidate_checkpoint_sha256": candidate_sha256,
            "candidate_id": candidate.candidate_id,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "version": 1,
        }
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, provenance)
        self._parent_by_checkpoint[candidate_sha256] = parent_checkpoint_sha256

    def _persisted_lineage(
        self,
        candidate: Any,
        source: Path,
        destination: Path,
    ) -> str | None:
        source = source.resolve()
        expected_directory = (
            destination.resolve().parent.parent / ".candidates"
        ).resolve()
        if source.parent != expected_directory:
            raise ValueError(
                "Selected candidate is outside the coordinator-owned candidate directory."
            )
        provenance = self._provenance_path(source)
        if not provenance.is_file():
            raise ValueError(
                "Selected candidate has no durable lineage provenance; reject it."
            )
        try:
            payload = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Selected candidate lineage provenance is unreadable; reject it."
            ) from error
        required = {
            "candidate_checkpoint_sha256",
            "candidate_id",
            "parent_checkpoint_sha256",
            "version",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError(
                "Selected candidate lineage provenance has an invalid schema."
            )
        parent_sha256 = payload["parent_checkpoint_sha256"]
        if (
            payload["version"] != 1
            or payload["candidate_checkpoint_sha256"] != candidate.checkpoint_sha256
            or payload["candidate_id"] != candidate.candidate_id
            or (
                parent_sha256 is not None
                and (
                    not isinstance(parent_sha256, str) or not _is_sha256(parent_sha256)
                )
            )
        ):
            raise ValueError(
                "Selected candidate lineage provenance does not match the candidate."
            )
        return parent_sha256

    def _write_runtime(
        self,
        runtime: _Runtime,
        *,
        role: str,
        candidate_id: str,
        path: Path,
        topology_changed: bool,
    ) -> Any:
        structural = self._structural()
        scored = self._score_runtime(runtime, role)
        runtime.model.owner_codes = np.asarray(scored.owner_codes, dtype=np.int16)
        optimizer = structural.optimizer_from_muon_groups(runtime.trainer)
        arrays = structural.checkpoint_arrays(
            runtime.model, optimizer, {"owners": scored.owners}
        )
        if not structural.validate_topology_dale(
            self._topology_optimizer_from_arrays(arrays)[0]
        ):
            raise ValueError(
                "Trained candidate violates a Dale sign; reject this candidate."
            )
        self._model().write_checkpoint(path, arrays)
        snapshot = self._snapshot(
            candidate_id=candidate_id,
            path=path,
            arrays=arrays,
            score=scored.score,
            topology_changed=topology_changed,
        )
        self._evidence_by_checkpoint[snapshot.checkpoint_sha256] = scored
        return snapshot

    def initialize(self, config: Any, output_dir: Path) -> Any:
        """Build, directly score, and stage the initial untyped BrainCell.

        Parameters
        ----------
        config : PipelineConfig
            Evolution caps and the required Muon policy.
        output_dir : pathlib.Path
            Run artifact directory.

        Returns
        -------
        CandidateSnapshot
            Initial fully scored format-1 continuation.
        """

        if config.optimizer != "muon" or int(config.updates) != _EXPECTED_UPDATES:
            raise ValueError(
                "Production evolution requires Muon and exactly 128 updates per block."
            )
        self.training_manifest()
        runtime = self._fresh_runtime()
        path = self._candidate_path(Path(output_dir), "initial")
        candidate = self._write_runtime(
            runtime,
            role="training",
            candidate_id="initial",
            path=path,
            topology_changed=False,
        )
        self._record_lineage(candidate, None)
        return candidate

    def restore(self, candidate: Any) -> Any:
        """Verify a persisted checkpoint without changing its recorded score.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Durable coordinator handoff.

        Returns
        -------
        CandidateSnapshot
            Identity-verified continuation with refreshed engineering resources.
        """

        path = Path(candidate.checkpoint_path)
        if not path.is_file():
            raise ValueError(
                f"Accepted checkpoint is missing from {path}; recover it before resume."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != candidate.checkpoint_sha256:
            raise ValueError(
                "Accepted checkpoint bytes differ from run state; reject this resume."
            )
        arrays = self._model().load_checkpoint(path)
        identities = checkpoint_identities(arrays)
        if (
            identities.topology_sha256 != candidate.topology_sha256
            or identities.parameters_sha256 != candidate.parameters_sha256
            or identities.optimizer_sha256 != candidate.optimizer_sha256
        ):
            raise ValueError(
                "Accepted checkpoint components differ from run state; reject this resume."
            )
        return self._snapshot(
            candidate_id=candidate.candidate_id,
            path=path,
            arrays=arrays,
            score=candidate.score,
            topology_changed=candidate.topology_changed,
        )

    def persist(
        self,
        candidate: Any,
        destination: Path,
        *,
        parent_checkpoint_sha256: str | None,
        stage_id: str,
    ) -> Any:
        """Atomically copy one selected format-1 candidate to its lineage path.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Selected temporary or durable continuation.
        destination : pathlib.Path
            Coordinator-owned stage checkpoint path.
        parent_checkpoint_sha256 : str or None
            Parent identity recorded by the coordinator.
        stage_id : str
            Stable stage identity.

        Returns
        -------
        CandidateSnapshot
            Same continuation identities at the durable destination.
        """

        source = Path(candidate.checkpoint_path)
        if (
            hashlib.sha256(source.read_bytes()).hexdigest()
            != candidate.checkpoint_sha256
        ):
            raise ValueError(
                "Selected temporary checkpoint changed before persistence; reject it."
            )
        if parent_checkpoint_sha256 is not None and not _is_sha256(
            parent_checkpoint_sha256
        ):
            raise ValueError(
                "Parent checkpoint identity must be one SHA-256 digest or None."
            )
        if (
            not stage_id
            or destination.name != f"{stage_id}.npz"
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in stage_id
            )
        ):
            raise ValueError(
                "Stage identity must be safe and match its checkpoint destination."
            )
        durable_parent = self._persisted_lineage(candidate, source, destination)
        marker = object()
        remembered_parent = self._parent_by_checkpoint.get(
            candidate.checkpoint_sha256, marker
        )
        if durable_parent != parent_checkpoint_sha256 or (
            remembered_parent is not marker and remembered_parent != durable_parent
        ):
            raise ValueError(
                "Selected candidate ancestry does not match its declared parent checkpoint."
            )
        arrays = self._model().load_checkpoint(source)
        self._model().write_checkpoint(destination, arrays, parent=source)
        persisted = self._snapshot(
            candidate_id=candidate.candidate_id,
            path=destination,
            arrays=arrays,
            score=candidate.score,
            topology_changed=candidate.topology_changed,
        )
        evidence = self._evidence_by_checkpoint.pop(candidate.checkpoint_sha256, None)
        if evidence is not None:
            self._evidence_by_checkpoint[persisted.checkpoint_sha256] = evidence
        self._parent_by_checkpoint.pop(candidate.checkpoint_sha256, None)
        resolved_source = source.resolve()
        resolved_source.unlink(missing_ok=True)
        self._provenance_path(resolved_source).unlink(missing_ok=True)
        self._temporary_paths.discard(resolved_source)
        return persisted

    def attest_pending(
        self,
        candidate: Any,
        *,
        parent_checkpoint_sha256: str,
        stage_id: str,
    ) -> Any:
        """Verify and re-register one journal-recovered temporary child.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Journal-validated child whose temporary checkpoint still exists.
        parent_checkpoint_sha256 : str
            Exact durable parent identity recovered by the coordinator.
        stage_id : str
            Stable pending stage identity recovered by the coordinator.

        Returns
        -------
        CandidateSnapshot
            Byte- and component-verified unchanged temporary child.
        """

        if not _is_sha256(parent_checkpoint_sha256):
            raise ValueError("Pending parent identity must be one SHA-256 digest.")
        if not stage_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in stage_id
        ):
            raise ValueError("Pending stage identity must be nonempty and safe.")
        source = Path(candidate.checkpoint_path).resolve()
        if source.parent.name != ".candidates":
            raise ValueError(
                "Recovered child is outside a coordinator-owned candidate directory."
            )
        restored = self.restore(candidate)
        provenance = self._provenance_path(source)
        if provenance.is_file():
            expected_destination = (
                source.parent.parent / "checkpoints" / f"{stage_id}.npz"
            )
            recorded_parent = self._persisted_lineage(
                restored, source, expected_destination
            )
            if recorded_parent != parent_checkpoint_sha256:
                raise ValueError(
                    "Recovered child provenance differs from its journal parent."
                )
        self._record_lineage(restored, parent_checkpoint_sha256)
        return restored

    def discard(self, attempt: Any) -> None:
        """Remove only adapter-owned unselected candidate checkpoints.

        Parameters
        ----------
        attempt : CandidateAttempt
            Completed, blocked, or failed sibling result.
        """

        candidate = getattr(attempt, "candidate", None)
        if candidate is None:
            return
        path = Path(candidate.checkpoint_path).resolve()
        tracked = path in self._temporary_paths
        output_dir = path.parent.parent
        restart_owned = bool(
            path.parent.name == ".candidates"
            and (
                (output_dir / "run-state.json").is_file()
                or (output_dir / "checkpoints").is_dir()
            )
        )
        if tracked or restart_owned:
            if not path.is_file():
                self._provenance_path(path).unlink(missing_ok=True)
                self._temporary_paths.discard(path)
                self._evidence_by_checkpoint.pop(candidate.checkpoint_sha256, None)
                self._parent_by_checkpoint.pop(candidate.checkpoint_sha256, None)
                return
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != candidate.checkpoint_sha256
            ):
                raise ValueError(
                    "Candidate cleanup path or SHA-256 is invalid; preserve it for recovery."
                )
            path.unlink(missing_ok=True)
            self._provenance_path(path).unlink(missing_ok=True)
            self._temporary_paths.discard(path)
            self._evidence_by_checkpoint.pop(candidate.checkpoint_sha256, None)
            self._parent_by_checkpoint.pop(candidate.checkpoint_sha256, None)

    def _score_runtime(self, runtime: _Runtime, role: str) -> _ScoredRuntime:
        import brainstate
        import jax
        import jax.numpy as jnp

        module = self._model()
        structural = self._structural()
        manifest = self._manifest(role)
        queries = self._encoded_queries(role)
        event_values = jnp.asarray(
            np.stack([record.events for record in queries]), dtype=jnp.float32
        )
        advance_values = jnp.asarray(
            np.stack([record.advances for record in queries]), dtype=bool
        )
        before_parameters = jax.tree_util.tree_map(
            jnp.array, runtime.trainer.parameters
        )

        def evaluate(events: Any, advances: Any) -> tuple[Any, Any]:
            runtime.model.reset_episode(runtime.learner)
            voltages, spikes = module.run_event_sequence(
                runtime.model,
                events,
                advances,
                return_spikes=True,
            )
            features = jnp.tanh((voltages[-31:] + 65.0) / 20.0)
            logits = (
                features @ runtime.model.readout_weight.value
                + runtime.model.readout_bias.value
            )
            return logits, jnp.mean(jnp.abs(spikes), axis=0)

        score_all = brainstate.transform.jit(
            lambda events, advances: brainstate.transform.for_loop(
                evaluate, events, advances
            )
        )
        logits, activities = score_all(event_values, advance_values)
        after_parameters = jax.tree_util.tree_map(jnp.array, runtime.trainer.parameters)
        if not bool(
            jax.tree_util.tree_all(
                jax.tree_util.tree_map(
                    jnp.array_equal, before_parameters, after_parameters
                )
            )
        ):
            raise RuntimeError(
                "Direct ARC scoring changed trainable parameters; reject this score."
            )
        logits_array = np.asarray(logits)
        activity_array = np.asarray(activities, dtype=np.float64)
        exact_by_task: dict[str, list[bool]] = {
            task_id: [] for task_id in manifest.task_ids
        }
        loss_by_task: dict[str, list[float]] = {
            task_id: [] for task_id in manifest.task_ids
        }
        activity_by_task: dict[str, list[np.ndarray]] = {
            task_id: [] for task_id in manifest.task_ids
        }
        for record, output, activity in zip(
            queries, logits_array, activity_array, strict=True
        ):
            exact, loss = direct_query_metrics(
                output, record.target, module.decode_prediction
            )
            exact_by_task[record.task_id].append(exact)
            loss_by_task[record.task_id].append(loss)
            activity_by_task[record.task_id].append(activity)
        task_exact = tuple(
            bool(exact_by_task[task_id]) and all(exact_by_task[task_id])
            for task_id in manifest.task_ids
        )
        task_loss = tuple(
            float(np.mean(loss_by_task[task_id])) for task_id in manifest.task_ids
        )
        optimizer = structural.optimizer_from_muon_groups(runtime.trainer)
        finite = bool(
            np.all(np.isfinite(logits_array))
            and np.all(np.isfinite(activity_array))
            and runtime.trainer.optimizer_is_finite()
            and all(math.isfinite(value) for value in task_loss)
        )
        score = self._evolve().ScoreSnapshot(
            task_ids=manifest.task_ids,
            task_exact=task_exact,
            task_loss=task_loss,
            finite=finite,
        )

        readout_mass = np.sum(
            np.abs(np.asarray(runtime.model.readout_weight.value)), axis=1
        )
        task_neuron = np.stack(
            [
                np.mean(np.stack(activity_by_task[task_id]), axis=0)
                * (1.0 + _normalise(readout_mass))
                for task_id in manifest.task_ids
            ]
        )
        task_neuron = np.stack([_normalise(row) for row in task_neuron])
        owners = structural.task_owners(task_neuron)
        codes = owner_codes(task_neuron)

        topology = structural.topology_from_model(runtime.model)
        neuron_gradient = np.sum(np.abs(optimizer.neuron_first), axis=1)
        neuron_gradient += np.bincount(
            topology.input_target,
            weights=np.abs(optimizer.input_first),
            minlength=topology.neuron_count,
        )
        recurrent_gradient = np.abs(optimizer.recurrent_first)
        neuron_gradient += np.bincount(
            topology.recurrent_source,
            weights=recurrent_gradient,
            minlength=topology.neuron_count,
        )
        neuron_gradient += np.bincount(
            topology.recurrent_target,
            weights=recurrent_gradient,
            minlength=topology.neuron_count,
        )
        activity_score = np.max(task_neuron, axis=0)
        neuron_score = 0.5 * _normalise(activity_score)
        neuron_score += 0.5 * _normalise(neuron_gradient)
        effective = np.abs(structural.effective_topology_recurrent_values(topology))
        transmission = effective * activity_score[topology.recurrent_source]
        edge_score = 0.5 * _normalise(transmission)
        edge_score += 0.5 * _normalise(recurrent_gradient)
        source_score = neuron_score + np.bincount(
            topology.recurrent_source,
            weights=edge_score,
            minlength=topology.neuron_count,
        )
        target_score = neuron_score + np.bincount(
            topology.recurrent_target,
            weights=edge_score,
            minlength=topology.neuron_count,
        )
        return _ScoredRuntime(
            score,
            owners,
            codes,
            np.asarray(neuron_score, dtype=np.float64),
            np.asarray(source_score, dtype=np.float64),
            np.asarray(target_score, dtype=np.float64),
            np.asarray(edge_score, dtype=np.float64),
        )

    @staticmethod
    def _episode_payload(record: SupervisedQuery) -> dict[str, np.ndarray]:
        target = np.asarray(record.target, dtype=np.int32)
        event_count = record.events.shape[0]
        request_mask = np.zeros(event_count, dtype=bool)
        request_mask[-31:] = True
        request_kind = np.zeros(event_count, dtype=np.int32)
        request_kind[-31] = 1
        request_kind[-30:] = 2
        target_shape = np.broadcast_to(
            np.asarray(target.shape, dtype=np.int32) - 1,
            (event_count, 2),
        ).copy()
        padded_rows: np.ndarray = np.zeros((30, 30), dtype=np.int32)
        padded_rows[: target.shape[0], : target.shape[1]] = target
        target_rows = np.zeros((event_count, 30), dtype=np.int32)
        target_rows[-30:] = padded_rows
        target_valid = np.zeros((event_count, 30), dtype=np.float32)
        start = event_count - 30
        target_valid[start : start + target.shape[0], : target.shape[1]] = 1.0
        return {
            "events": np.asarray(record.events, dtype=np.float32),
            "advance_mask": np.asarray(record.advances, dtype=bool),
            "loss_mask": request_mask,
            "request_kind": request_kind,
            "target_shape": target_shape,
            "target_rows": target_rows,
            "target_valid_mask": target_valid,
        }

    def _run_update_block(self, runtime: _Runtime, schedule: Any) -> None:
        if (
            len(schedule.entries) != _EXPECTED_UPDATES
            or schedule.cursor_end - schedule.cursor_start != _EXPECTED_UPDATES
        ):
            raise ValueError(
                "Production PP-Prop blocks require exactly 128 scheduled queries."
            )
        import brainstate
        import jax
        import jax.numpy as jnp

        module = self._model()
        query_lookup = {
            (record.task_id, record.query_index): record
            for record in self._encoded_queries("training")
        }
        try:
            records = tuple(
                query_lookup[(entry.task_id, entry.query_index)]
                for entry in schedule.entries
            )
        except KeyError as error:
            raise ValueError(
                "Update schedule references a query outside the training manifest."
            ) from error
        payloads = tuple(self._episode_payload(record) for record in records)
        stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *payloads)

        def step_fn(
            event: Any,
            advance: Any,
            request_kind: Any,
            target_shape: Any,
            target_rows: Any,
            target_valid_mask: Any,
        ) -> tuple[Any, Any]:
            def advancing() -> tuple[Any, Any]:
                runtime.learner(event)
                features = runtime.model.readout_features()
                logits = (
                    features @ runtime.model.readout_weight.value
                    + runtime.model.readout_bias.value
                )
                loss = module._supervised_request_loss(
                    logits,
                    target_shape,
                    target_rows,
                    target_valid_mask,
                    request_kind,
                )
                return loss, features

            return cast(
                tuple[Any, Any],
                brainstate.transform.cond(
                    advance,
                    advancing,
                    lambda: (
                        jnp.asarray(0.0),
                        jnp.zeros(
                            (runtime.model.readout_weight.value.shape[0],),
                            dtype=jnp.float32,
                        ),
                    ),
                ),
            )

        def direct_grad_fn(
            *,
            aux: Any,
            request_kind: Any,
            target_shape: Any,
            target_rows: Any,
            target_valid_mask: Any,
            mask: Any,
            **_: Any,
        ) -> Any:
            return module._direct_readout_gradients(
                aux,
                target_shape,
                target_rows,
                target_valid_mask,
                request_kind,
                mask,
                runtime.model.readout_weight.value,
                runtime.model.readout_bias.value,
            )

        def update(payload: Mapping[str, Any]) -> Any:
            runtime.trainer.reset_episode()
            return runtime.trainer.update_episode(
                **payload,
                step_fn=step_fn,
                direct_grad_fn=direct_grad_fn,
            )

        before = int(np.asarray(runtime.trainer.updates))
        run = brainstate.transform.jit(
            lambda values: brainstate.transform.for_loop(update, values)
        )
        run(stacked)
        completed = int(np.asarray(runtime.trainer.updates)) - before
        if completed != _EXPECTED_UPDATES:
            raise RuntimeError(
                f"Compiled PP-Prop block completed {completed} updates; expected 128."
            )
        if not runtime.trainer.optimizer_is_finite():
            raise RuntimeError(
                "Compiled PP-Prop block produced non-finite Muon state; reject it."
            )

    @staticmethod
    def _trainer_update_count(runtime: _Runtime) -> int:
        value = np.asarray(runtime.trainer.updates)
        if value.shape != () or not np.issubdtype(value.dtype, np.integer):
            raise RuntimeError(
                "Candidate trainer update count must be one integer scalar."
            )
        count = int(value)
        if count < 0:
            raise RuntimeError("Candidate trainer update count must be nonnegative.")
        return count

    @classmethod
    def _executed_update_delta(cls, runtime: _Runtime, before: int) -> int:
        delta = cls._trainer_update_count(runtime) - before
        if delta < 0 or delta > _EXPECTED_UPDATES:
            raise RuntimeError(
                "Candidate trainer executed-update delta must be from 0 to 128."
            )
        return delta

    def train_parent(self, parent: Any, schedule: Any, context: Any) -> Any:
        """Continue the accepted checkpoint for one compiled 128-update block.

        Parameters
        ----------
        parent : CandidateSnapshot
            Immutable accepted continuation.
        schedule : UpdateSchedule
            Exact shared 128-query training schedule.
        context : StageContext
            Stage identity and output directory.

        Returns
        -------
        CandidateSnapshot
            Directly scored trained continuation.
        """

        restored = self.restore(parent)
        runtime = self._runtime_from_checkpoint(restored.checkpoint_path)
        self._run_update_block(runtime, schedule)
        candidate_id = f"{context.stage_id}-training"
        path = self._candidate_path(context.output_dir, candidate_id)
        candidate = self._write_runtime(
            runtime,
            role="training",
            candidate_id=candidate_id,
            path=path,
            topology_changed=False,
        )
        self._record_lineage(candidate, parent.checkpoint_sha256)
        return candidate

    def _parent_evidence(self, parent: Any, runtime: _Runtime) -> _ScoredRuntime:
        evidence = self._evidence_by_checkpoint.get(parent.checkpoint_sha256)
        if evidence is None:
            evidence = self._score_runtime(runtime, "training")
            if evidence.score.task_ids != parent.score.task_ids:
                raise ValueError(
                    "Parent direct score does not match the training manifest; reject it."
                )
            self._evidence_by_checkpoint[parent.checkpoint_sha256] = evidence
        return evidence

    def _edge_prune(
        self, topology: Any, optimizer: Any, evidence: _ScoredRuntime
    ) -> tuple[Any, Any]:
        structural = self._structural()
        count = structural.mutation_count(len(topology.recurrent_value))
        if count >= len(topology.recurrent_value):
            raise ValueError(
                "Edge pruning would remove every recurrent edge; retain this parent."
            )
        keep: np.ndarray = np.ones(len(topology.recurrent_value), dtype=bool)
        order = np.lexsort((np.arange(len(evidence.edge_scores)), evidence.edge_scores))
        keep[order[:count]] = False
        candidate = structural.SparseTopology(
            topology.input_source.copy(),
            topology.input_target.copy(),
            topology.input_value.copy(),
            topology.recurrent_source[keep],
            topology.recurrent_target[keep],
            topology.recurrent_value[keep],
            topology.readout.copy(),
            topology.dale.copy(),
            tuple(topology.mechanisms),
            None if topology.owner_codes is None else topology.owner_codes.copy(),
            None if topology.neuron_ids is None else topology.neuron_ids.copy(),
        )
        mapped = structural.StructuralAdam(
            optimizer.neuron_first.copy(),
            optimizer.neuron_second.copy(),
            optimizer.input_first.copy(),
            optimizer.input_second.copy(),
            optimizer.recurrent_first[keep],
            optimizer.recurrent_second[keep],
            bias_first=optimizer.bias_first.copy(),
            bias_second=optimizer.bias_second.copy(),
            input_step=optimizer.input_step,
            recurrent_step=optimizer.recurrent_step,
            readout_step=optimizer.readout_step,
        )
        candidate, mapped = structural.canonicalize_topology_and_optimizer(
            candidate, mapped
        )
        return candidate, mapped

    def _edge_add(
        self,
        topology: Any,
        optimizer: Any,
        evidence: _ScoredRuntime,
        config: Any,
    ) -> tuple[Any, Any]:
        structural = self._structural()
        count = structural.mutation_count(len(topology.recurrent_value))
        if len(topology.recurrent_value) + count > int(config.max_recurrent_edges):
            raise ValueError(
                "Edge addition exceeds the configured recurrent-edge cap; retain the parent."
            )
        existing = set(
            zip(
                topology.recurrent_source.tolist(),
                topology.recurrent_target.tolist(),
            )
        )
        pairs = structural.select_connection_additions(
            topology.neuron_count,
            existing,
            evidence.source_scores,
            evidence.target_scores,
            count,
        )
        candidate = structural.add_recurrent_connections(topology, pairs)
        mapped = structural.grow_adam_for_connections(optimizer, len(pairs))
        candidate, mapped = structural.canonicalize_topology_and_optimizer(
            candidate, mapped
        )
        return candidate, mapped

    def _neuron_prune(
        self, topology: Any, optimizer: Any, evidence: _ScoredRuntime
    ) -> tuple[Any, Any]:
        structural = self._structural()
        count = structural.mutation_count(topology.neuron_count)
        if count >= topology.neuron_count:
            raise ValueError(
                "Neuron pruning would remove the complete network; retain the parent."
            )
        alive: np.ndarray = np.ones(topology.neuron_count, dtype=bool)
        order = np.lexsort(
            (np.arange(len(evidence.neuron_scores)), evidence.neuron_scores)
        )
        alive[order[:count]] = False
        candidate, mapped, _ = structural.compact(topology, alive, optimizer)
        return candidate, mapped

    def _neuron_add(
        self,
        topology: Any,
        optimizer: Any,
        evidence: _ScoredRuntime,
        config: Any,
    ) -> tuple[Any, Any]:
        structural = self._structural()
        count = structural.mutation_count(topology.neuron_count)
        if topology.neuron_count + count > int(config.max_neurons):
            raise ValueError(
                "Neuron addition exceeds the configured neuron cap; retain the parent."
            )
        connected = set(
            zip(
                topology.recurrent_source.tolist(),
                topology.recurrent_target.tolist(),
            )
        )
        donors: list[int] = []
        for donor in structural.stable_rank(evidence.neuron_scores, descending=True):
            if evidence.neuron_scores[donor] <= 0:
                break
            if all(
                (donor, selected) not in connected
                and (selected, donor) not in connected
                for selected in donors
            ):
                donors.append(int(donor))
            if len(donors) == count:
                break
        if len(donors) != count:
            raise ValueError(
                "Neuron addition has insufficient unconnected positive donors; "
                "retain the parent."
            )
        donor_indices = np.asarray(donors, dtype=np.int32)
        input_degree = np.bincount(
            topology.input_target, minlength=topology.neuron_count
        )
        incoming_degree = np.bincount(
            topology.recurrent_target, minlength=topology.neuron_count
        )
        outgoing_degree = np.bincount(
            topology.recurrent_source, minlength=topology.neuron_count
        )
        added_input = int(np.sum(input_degree[donor_indices]))
        added_recurrent = int(
            np.sum(incoming_degree[donor_indices] + outgoing_degree[donor_indices])
        )
        if len(topology.recurrent_value) + added_recurrent > int(
            config.max_recurrent_edges
        ):
            raise ValueError(
                "Neuron addition exceeds the configured recurrent-edge cap; "
                "retain the parent."
            )
        structural.enforce_biological_connection_ceiling(
            topology.neuron_count + count,
            len(topology.input_value) + added_input,
            len(topology.recurrent_value) + added_recurrent,
        )
        candidate, selected_donors = structural.add_twin_neurons(
            topology, evidence.neuron_scores, required=count
        )
        if tuple(selected_donors) != tuple(donors):
            raise RuntimeError(
                "Neuron donor preflight differs from construction; reject the candidate."
            )
        mapped = structural.grow_adam_for_twins(optimizer, topology, candidate)
        candidate, mapped = structural.canonicalize_topology_and_optimizer(
            candidate, mapped
        )
        return candidate, mapped

    def _dale_assignment(
        self,
        topology: Any,
        optimizer: Any,
        evidence: _ScoredRuntime,
        parent_id: str,
        sign: int,
    ) -> tuple[Any, Any]:
        from examples.pp_prop.dale_candidates import (
            DaleMeasurements,
            measure_dale_candidates,
        )

        structural = self._structural()
        effective = np.asarray(
            structural.effective_topology_recurrent_values(topology), dtype=float
        )
        recurrent_gradient = np.bincount(
            topology.recurrent_source,
            weights=np.abs(optimizer.recurrent_first),
            minlength=topology.neuron_count,
        )
        lesion = np.bincount(
            topology.recurrent_source,
            weights=np.abs(effective)
            * evidence.neuron_scores[topology.recurrent_source],
            minlength=topology.neuron_count,
        )
        ownership = np.where(evidence.owner_codes >= 0, 1.0, 0.0)
        measurements = DaleMeasurements(
            parent_id=parent_id,
            rows=np.asarray(topology.recurrent_source, dtype=np.int32),
            weights=effective,
            activity=np.asarray(evidence.neuron_scores, dtype=float),
            gradient_mass=np.asarray(recurrent_gradient, dtype=float),
            task_ownership=np.asarray(ownership, dtype=float),
            lesion_evidence=np.asarray(lesion, dtype=float),
            type_signs=np.asarray(topology.dale, dtype=np.int8),
        )
        selection = measure_dale_candidates(measurements)
        indices = selection.excitatory if sign == 1 else selection.inhibitory
        candidate = structural.assign_dale_type(topology, indices, sign)
        candidate, mapped = structural.canonicalize_topology_and_optimizer(
            candidate, optimizer
        )
        return candidate, mapped

    def _mutate(
        self,
        kind: str,
        topology: Any,
        optimizer: Any,
        evidence: _ScoredRuntime,
        parent: Any,
        config: Any,
    ) -> tuple[Any, Any]:
        if kind == "edge-add":
            return self._edge_add(topology, optimizer, evidence, config)
        if kind == "edge-prune":
            return self._edge_prune(topology, optimizer, evidence)
        if kind == "neuron-add":
            return self._neuron_add(topology, optimizer, evidence, config)
        if kind == "neuron-prune":
            return self._neuron_prune(topology, optimizer, evidence)
        if kind == "dale-excitatory":
            return self._dale_assignment(
                topology,
                optimizer,
                evidence,
                parent.checkpoint_sha256,
                1,
            )
        if kind == "dale-inhibitory":
            return self._dale_assignment(
                topology,
                optimizer,
                evidence,
                parent.checkpoint_sha256,
                -1,
            )
        raise ValueError(f"Unknown mutation kind {kind}; reject this candidate.")

    def run_candidate(
        self,
        parent: Any,
        arm: str,
        schedule: Any,
        context: Any,
    ) -> Any:
        """Build, train, and score one immutable-parent structural sibling.

        Parameters
        ----------
        parent : CandidateSnapshot
            Stage parent that remains byte-identical.
        arm : str
            Coordinator arm name.
        schedule : UpdateSchedule
            Shared 128-query schedule for both siblings.
        context : StageContext
            Stage, caps, and temporary output location.

        Returns
        -------
        CandidateAttempt
            Completed, blocked, or failed arm with its literal update count.
        """

        evolve = self._evolve()
        kind = self._mutation_kind(context.stage, arm)
        restored = self.restore(parent)
        parent_path = Path(restored.checkpoint_path)
        parent_bytes = parent_path.read_bytes()
        runtime = self._runtime_from_checkpoint(parent_path)
        candidate_path: Path | None = None
        candidate: Any | None = None
        candidate_runtime: _Runtime | None = None
        updates_before: int | None = None
        try:
            evidence = self._parent_evidence(parent, runtime)
        except Exception as error:
            if parent_path.read_bytes() != parent_bytes:
                raise RuntimeError(
                    "Candidate evidence changed its immutable parent checkpoint."
                ) from error
            return evolve.CandidateAttempt.failed(
                arm,
                f"{type(error).__name__}: {error}",
                executed_updates=0,
            )
        try:
            topology, optimizer = self._mutate(
                kind,
                runtime.topology,
                runtime.optimizer,
                evidence,
                parent,
                context.config,
            )
        except ValueError as error:
            if parent_path.read_bytes() != parent_bytes:
                raise RuntimeError(
                    "Blocked mutation changed its immutable parent checkpoint."
                ) from error
            return evolve.CandidateAttempt.blocked(arm, str(error), executed_updates=0)
        except Exception as error:
            if parent_path.read_bytes() != parent_bytes:
                raise RuntimeError(
                    "Failed mutation changed its immutable parent checkpoint."
                ) from error
            return evolve.CandidateAttempt.failed(
                arm,
                f"{type(error).__name__}: {error}",
                executed_updates=0,
            )
        try:
            if topology.neuron_count > int(context.config.max_neurons):
                return evolve.CandidateAttempt.blocked(
                    arm,
                    "Candidate exceeds the configured neuron cap.",
                    executed_updates=0,
                )
            if len(topology.recurrent_value) > int(context.config.max_recurrent_edges):
                return evolve.CandidateAttempt.blocked(
                    arm,
                    "Candidate exceeds the configured recurrent-edge cap.",
                    executed_updates=0,
                )
            try:
                self._structural().enforce_biological_connection_ceiling(
                    topology.neuron_count,
                    len(topology.input_value),
                    len(topology.recurrent_value),
                )
            except ValueError as error:
                return evolve.CandidateAttempt.blocked(
                    arm, str(error), executed_updates=0
                )
            candidate_runtime = self._runtime_from_state(
                topology, optimizer, runtime.readout_bias
            )
            updates_before = self._trainer_update_count(candidate_runtime)
            self._run_update_block(candidate_runtime, schedule)
            executed_updates = self._executed_update_delta(
                candidate_runtime, updates_before
            )
            candidate_id = f"{context.stage_id}-{arm}"
            candidate_path = self._candidate_path(context.output_dir, candidate_id)
            candidate = self._write_runtime(
                candidate_runtime,
                role="training",
                candidate_id=candidate_id,
                path=candidate_path,
                topology_changed=True,
            )
            self._record_lineage(candidate, parent.checkpoint_sha256)
            return evolve.CandidateAttempt.completed(
                arm, candidate, executed_updates=executed_updates
            )
        except Exception as error:  # noqa: BLE001 - isolate one failed candidate
            if candidate_path is not None:
                candidate_path.unlink(missing_ok=True)
                self._provenance_path(candidate_path).unlink(missing_ok=True)
                self._provenance_path(candidate_path).with_name(
                    f"{self._provenance_path(candidate_path).name}.tmp"
                ).unlink(missing_ok=True)
                self._temporary_paths.discard(candidate_path.resolve())
            if candidate is not None:
                self._evidence_by_checkpoint.pop(candidate.checkpoint_sha256, None)
                self._parent_by_checkpoint.pop(candidate.checkpoint_sha256, None)
            executed_updates = (
                0
                if candidate_runtime is None or updates_before is None
                else self._executed_update_delta(candidate_runtime, updates_before)
            )
            return evolve.CandidateAttempt.failed(
                arm,
                f"{type(error).__name__}: {error}",
                executed_updates=executed_updates,
            )
        finally:
            if parent_path.read_bytes() != parent_bytes:
                raise RuntimeError(
                    "Structural sibling changed its immutable parent checkpoint."
                )

    def render_topology(self, candidate: Any, output_path: Path) -> None:
        """Render stable IDs, owners, Dale labels, and recurrent edges.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Selected executed checkpoint.
        output_path : pathlib.Path
            Destination PNG, normally a coordinator temporary sibling.
        """

        restored = self.restore(candidate)
        structural = self._structural()
        topology = structural.topology_from_checkpoint(
            self._model(), restored.checkpoint_path
        )
        structural.plot_topology(
            topology,
            output_path,
            title=(
                "Example 21 accepted BrainCell topology "
                f"({restored.score.exact_count}/400 training tasks)"
            ),
        )

    def evaluate_terminal(self, candidate: Any, manifest: Any) -> Mapping[str, object]:
        """Directly score the unchanged terminal checkpoint on held-out ARC.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Final accepted training checkpoint.
        manifest : CorpusManifest
            Coordinator-requested held-out manifest.

        Returns
        -------
        mapping
            Exact count, per-task exact flags and losses, and finiteness.
        """

        expected = self.evaluation_manifest()
        if manifest.digest != expected.digest:
            raise ValueError(
                "Terminal evaluation manifest differs from the declared ARC corpus."
            )
        restored = self.restore(candidate)
        before = Path(restored.checkpoint_path).read_bytes()
        runtime = self._runtime_from_checkpoint(restored.checkpoint_path)
        try:
            scored = self._score_runtime(runtime, "evaluation")
        finally:
            if Path(restored.checkpoint_path).read_bytes() != before:
                raise RuntimeError(
                    "Terminal evaluation changed the accepted checkpoint; reject the result."
                )
        return {
            "task_count": len(scored.score.task_ids),
            "strict_task_pass_at_1_count": scored.score.exact_count,
            "task_ids": list(scored.score.task_ids),
            "task_exact": list(scored.score.task_exact),
            "task_loss": list(scored.score.task_loss),
            "mean_unresolved_task_loss": scored.score.unresolved_loss,
            "finite": scored.score.finite,
        }
