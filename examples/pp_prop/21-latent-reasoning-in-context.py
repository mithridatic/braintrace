"""21 - Standard ARC with pp-prop and recurrent latent spiking effort.

This example keeps the observable contract of BDH-CQ--ordinary ARC tasks,
exact ranked candidates, and selectable latent effort--while using the neuron,
synapse, sparse-operator, and pp-prop stack established by Examples 18--20.
The paper's private architecture, private data, and training recipe are not
available.  This is a repository-native experiment, not a reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

try:
    from examples.pp_prop.latent_workspace_analysis import (
        OutputLogits,
        aggregate_arc_metrics,
        analyze_latent_trajectory,
        compare_control_trajectories,
        decode_candidates,
        score_query_candidates,
    )
    from examples.pp_prop.latent_workspace_model import (
        ModelConfig,
        LatentWorkspaceModel,
        arc_loss_per_example,
        compile_pp_prop,
        expand_compact_logits,
        parameter_snapshot,
        run_selected_packed_stream,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcPair,
        ArcTask,
        associative_memory_feature_indices,
        DatasetSource,
        EncodedQueryEpisode,
        LoadedDataset,
        RowEventConfig,
        assert_no_evaluation_leakage,
        augment_training_task,
        canonical_task_fingerprint,
        encode_arc_query_episode,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
        load_dataset_source,
        smoke_loaded_dataset,
    )
except ModuleNotFoundError:
    from latent_workspace_analysis import (
        OutputLogits,
        aggregate_arc_metrics,
        analyze_latent_trajectory,
        compare_control_trajectories,
        decode_candidates,
        score_query_candidates,
    )
    from latent_workspace_model import (
        ModelConfig,
        LatentWorkspaceModel,
        arc_loss_per_example,
        compile_pp_prop,
        expand_compact_logits,
        parameter_snapshot,
        run_selected_packed_stream,
    )
    from latent_workspace_task import (
        ArcPair,
        ArcTask,
        associative_memory_feature_indices,
        DatasetSource,
        EncodedQueryEpisode,
        LoadedDataset,
        RowEventConfig,
        assert_no_evaluation_leakage,
        augment_training_task,
        canonical_task_fingerprint,
        encode_arc_query_episode,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
        load_dataset_source,
        smoke_loaded_dataset,
    )


DeviceName = Literal["cpu", "gpu"]
PrimaryCandidateMode = Literal["model_only"]
CHECKPOINTS = (0, 8, 16, 32)
TRAINING_EFFORTS = (8, 16, 32)
EVALUATION_ARM_ORDER = (
    "intact",
    "repeat_intact",
    "no_context",
    "shuffled_demonstrations",
    "slot_ablation",
)
STATE_RMS_TOLERANCE = 1e-6
APPROVED_TRAINING_SOURCES = frozenset(
    {
        "arc-agi-1 training",
        "re-arc",
        "conceptarc",
        "arc-heavy",
        "arc-gen100k",
    }
)
APPROVED_EVALUATION_SOURCES = frozenset({"arc-agi-1 evaluation", "arc-task-gen"})
CLAIM_BOUNDARY = (
    "Claim boundary: Example 21 instantiates the paper's public ARC task, "
    "ranked-candidate, and variable-effort contract with BrainPy LIF neurons, "
    "BrainTrace sparse synapses, and pp-prop. The paper's private data, model "
    "dimensions, internal update rules, and training recipe were unavailable. "
    "This is not a reproduction, makes no paper-score or inference-cost claim, "
    "and asserts no agreement between pp-prop and a BPTT gradient oracle."
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    """Configure data, one-model training, frozen evaluation, and artifacts.

    Parameters
    ----------
    source_manifest : pathlib.Path or None
        JSON declaration of public local ARC sources. Full scientific runs
        require at least one training and one evaluation source.
    output_dir : pathlib.Path
        Directory for ``result.json``, text report, plot, and resolved manifest.
    device : {"cpu", "gpu"}
        Requested fail-closed JAX backend.
    seed : int
        BrainState random seed for parameters, topology, scheduling, and
        training-only augmentation.
    neuron_count, recurrent_edges : int
        Physical LIF population and exact directed sparse-edge count.
    readout_width, color_rank : int
        Shared readout bottleneck and CP rank of the full-grid color head.
    context_memory_width : int
        Associative workspace width. Zero selects the byte-compatible legacy
        reservoir; positive values up to 128 opt into ``S_K/H_r``.
    memory_decay : float
        Associative memory self-decay in the closed interval ``[0, 1]``.
    max_demonstrations, max_grid_size : int
        Static lossless ARC row-event capacities.
    latent_steps : int
        Maximum zero-input recurrent effort. Must be at least 32.
    training_updates : int
        Number of pp-prop/Adam updates shared across effort lengths.
    training_chunk_size : int
        Number of updates staged on device at once. ``0`` stages the whole
        schedule in one chunk, reproducing an unchunked run exactly. Any other
        value must divide ``training_updates`` so that every chunk compiles to
        the same scan length.
    learning_rate, clip_norm : float
        Adam rate and global gradient clipping norm.
    balanced_color_loss : bool
        Whether each target color contributes equal total valid-cell weight.
        The default retains the legacy uniform valid-cell objective.
    ablation_slot : int
        Deterministic 64-neuron slot used by the frozen ablation control.
    evaluation_task_limit : int or None
        Development-only task cap. Any cap disqualifies a full scientific run.
    smoke : bool
        Whether results use embedded fixtures and are plumbing-only.
    structural_only : bool
        Instantiate and run without optimization; never scientific evidence.
    primary_candidate_mode : {"model_only"}
        Fail-closed primary ARC scoring mode. Only candidates decoded from the
        model may occupy submitted pass@2 slots.
    """

    source_manifest: pathlib.Path | None = None
    output_dir: pathlib.Path = pathlib.Path("var/example21")
    device: DeviceName = "gpu"
    seed: int = 2108
    neuron_count: int = 2048
    recurrent_edges: int = 16384
    readout_width: int = 128
    color_rank: int = 16
    context_memory_width: int = 0
    memory_decay: float = 1.0
    max_demonstrations: int = 10
    max_grid_size: int = 30
    latent_steps: int = 32
    training_updates: int = 96
    training_chunk_size: int = 0
    learning_rate: float = 1e-4
    clip_norm: float = 1.0
    balanced_color_loss: bool = False
    ablation_slot: int = 0
    evaluation_task_limit: int | None = None
    smoke: bool = False
    structural_only: bool = False
    primary_candidate_mode: PrimaryCandidateMode = "model_only"

    def __post_init__(self) -> None:
        for name, minimum in (
            ("seed", 0),
            ("neuron_count", 64),
            ("recurrent_edges", 1),
            ("readout_width", 1),
            ("color_rank", 1),
            ("context_memory_width", 0),
            ("max_demonstrations", 1),
            ("max_grid_size", 1),
            ("latent_steps", 32),
            ("training_updates", 0),
            ("training_chunk_size", 0),
            ("ablation_slot", 0),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=minimum)
            )
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.primary_candidate_mode != "model_only":
            raise ValueError("primary_candidate_mode must be 'model_only'")
        if self.neuron_count % 64:
            raise ValueError("neuron_count must be divisible by 64")
        if self.context_memory_width > 128:
            raise ValueError("context_memory_width must be at most 128")
        if self.recurrent_edges > self.neuron_count * (self.neuron_count - 1):
            raise ValueError("recurrent_edges exceeds directed no-self capacity")
        if self.max_grid_size != 30:
            raise ValueError("max_grid_size must be 30 for standard ARC")
        if self.ablation_slot >= self.neuron_count // 64:
            raise ValueError("ablation_slot exceeds the configured 64-neuron slots")
        if self.evaluation_task_limit is not None:
            object.__setattr__(
                self,
                "evaluation_task_limit",
                _integer(
                    self.evaluation_task_limit, "evaluation_task_limit", minimum=1
                ),
            )
        object.__setattr__(
            self, "learning_rate", _positive_real(self.learning_rate, "learning_rate")
        )
        object.__setattr__(
            self, "clip_norm", _positive_real(self.clip_norm, "clip_norm")
        )
        object.__setattr__(
            self, "memory_decay", _unit_interval(self.memory_decay, "memory_decay")
        )
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.source_manifest is not None:
            object.__setattr__(
                self, "source_manifest", pathlib.Path(self.source_manifest)
            )
        if not self.structural_only and self.training_updates < len(TRAINING_EFFORTS):
            raise ValueError(
                "training_updates must expose one model to 8, 16, and 32 steps"
            )
        if (
            self.training_chunk_size
            and self.training_updates
            and self.training_updates % self.training_chunk_size
        ):
            raise ValueError("training_chunk_size must divide training_updates")

    @classmethod
    def smoke_config(
        cls,
        *,
        output_dir: pathlib.Path = pathlib.Path("var/example21-smoke"),
        device: DeviceName = "cpu",
        seed: int = 2108,
        context_memory_width: int = 0,
        memory_decay: float = 1.0,
        balanced_color_loss: bool = False,
    ) -> "ExperimentConfig":
        """Return a reduced complete-pipeline configuration.

        Parameters
        ----------
        output_dir : pathlib.Path
            Artifact directory for the smoke run.
        device : {"cpu", "gpu"}
            Requested JAX backend.
        seed : int
            Deterministic model, schedule, and augmentation seed.
        context_memory_width : int
            Optional associative workspace width; zero retains legacy mode.
        memory_decay : float
            Associative memory decay in ``[0, 1]``.
        balanced_color_loss : bool
            Whether to balance valid-cell color loss by present target class.

        Returns
        -------
        ExperimentConfig
            A 128-neuron, 1,024-edge, three-update plumbing-only run.
        """
        return cls(
            output_dir=output_dir,
            device=device,
            seed=seed,
            neuron_count=128,
            recurrent_edges=1024,
            readout_width=32,
            color_rank=4,
            context_memory_width=context_memory_width,
            memory_decay=memory_decay,
            balanced_color_loss=balanced_color_loss,
            max_demonstrations=4,
            training_updates=3,
            learning_rate=5e-4,
            smoke=True,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe configuration mapping."""
        result = asdict(self)
        result["source_manifest"] = (
            None if self.source_manifest is None else str(self.source_manifest)
        )
        result["output_dir"] = str(self.output_dir)
        result["checkpoints"] = list(CHECKPOINTS)
        result["training_efforts"] = list(TRAINING_EFFORTS)
        return result


@dataclass(frozen=True)
class _OriginTask:
    source_name: str
    role: str
    task: ArcTask


@dataclass(frozen=True)
class _ExperimentData:
    training: tuple[_OriginTask, ...]
    evaluation: tuple[_OriginTask, ...]
    loaded: tuple[LoadedDataset, ...]
    plumbing_only: bool


@dataclass(frozen=True)
class _TrainingTensors:
    events: np.ndarray
    advances: np.ndarray
    heights: np.ndarray
    widths: np.ndarray
    colors: np.ndarray
    masks: np.ndarray
    efforts: np.ndarray
    task_fingerprints: tuple[str, ...]
    base_task_fingerprints: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    held_out_demonstration_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _EvaluationRecord:
    source_name: str
    task_key: str
    encoded: EncodedQueryEpisode


def _devices_for(platform: DeviceName) -> list[jax.Device]:
    return list(jax.devices(platform))


def _device_memory_stats(device: jax.Device) -> dict[str, int]:
    try:
        return {
            str(key): int(value)
            for key, value in (device.memory_stats() or {}).items()
            if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        }
    except (AttributeError, RuntimeError):
        return {}


def _resolve_device(platform: DeviceName) -> tuple[jax.Device, dict[str, object]]:
    try:
        devices = _devices_for(platform)
    except RuntimeError as error:
        devices = []
        detail = f": {error}"
    else:
        detail = ""
    if not devices:
        raise RuntimeError(
            f"requested JAX {platform} backend is unavailable{detail}; "
            "choose --device cpu explicitly only for a reduced run"
        )
    device = devices[0]
    return device, {
        "requested": platform,
        "platform": str(device.platform),
        "id": int(device.id),
        "kind": str(getattr(device, "device_kind", device)),
        "memory_stats": _device_memory_stats(device),
    }


def _source_declarations(path: pathlib.Path) -> tuple[DatasetSource, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read source manifest {path}: {error}") from error
    values = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("source manifest must contain a nonempty 'sources' list")
    declarations: list[DatasetSource] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"sources[{index}] must be an object")
        required = {"name", "role", "version", "path", "license_reference"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"sources[{index}] is missing {missing}")
        source_path = pathlib.Path(str(value["path"]))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        declarations.append(
            DatasetSource(
                name=value["name"],
                role=value["role"],
                version=value["version"],
                path=str(source_path),
                license_reference=value["license_reference"],
                format=value.get("format", "auto"),
                exclude_fingerprints=tuple(value.get("exclude_fingerprints", ())),
            )
        )
    return tuple(declarations)


def _load_data(config: ExperimentConfig) -> _ExperimentData:
    if config.smoke or (config.structural_only and config.source_manifest is None):
        fixture = smoke_loaded_dataset()
        origins = tuple(
            _OriginTask(fixture.manifest.source.name, "fixture", task)
            for task in fixture.tasks
        )
        return _ExperimentData(origins, origins, (fixture,), True)
    if config.source_manifest is None:
        raise ValueError("full runs require --source-manifest")
    declarations = _source_declarations(config.source_manifest)
    loaded = tuple(load_dataset_source(source) for source in declarations)
    assert_no_evaluation_leakage(item.manifest for item in loaded)
    training = tuple(
        _OriginTask(item.manifest.source.name, item.manifest.source.role, task)
        for item in loaded
        if item.manifest.source.role == "train"
        for task in item.tasks
    )
    evaluation = tuple(
        _OriginTask(item.manifest.source.name, item.manifest.source.role, task)
        for item in loaded
        if item.manifest.source.role == "evaluation"
        for task in item.tasks
    )
    if not config.structural_only and not training:
        raise ValueError("scientific training requires at least one train-role source")
    if not evaluation:
        raise ValueError("evaluation requires at least one evaluation-role source")
    return _ExperimentData(training, evaluation, loaded, False)


def _row_config(config: ExperimentConfig) -> RowEventConfig:
    return RowEventConfig(
        max_demonstrations=config.max_demonstrations,
        max_grid_size=config.max_grid_size,
    )


def _packed_events(
    encoded: EncodedQueryEpisode, config: ExperimentConfig
) -> np.ndarray:
    total = encoded.events.shape[0] + config.latent_steps
    result = np.zeros((total, encoded.events.shape[1]), dtype=np.float32)
    result[: encoded.events.shape[0]] = encoded.events
    return result


def _demonstration_advance_width(
    encoded: EncodedQueryEpisode, row_config: RowEventConfig
) -> int:
    """Return the advancing row count shared by every demonstration block.

    Demonstration blocks are a fixed ``max_grid_size`` rows wide whatever the
    grid heights are, so advancing a whole block spends the unused rows as
    all-zero membrane-leak steps.  This is the per-episode maximum occupied
    height instead: it never drops an encoded row, and because
    :func:`_derange_task` rotates the outputs it is identical for the intact and
    deranged encodings, which keeps the ``shuffled_demonstrations`` control on a
    byte-identical schedule.
    """
    valid = encoded.events[:, row_config.valid_slice.start] > 0.0
    return max(
        (int(valid[start:stop].sum()) for start, stop in encoded.demonstration_spans),
        default=0,
    )


def _packed_advances(
    encoded: EncodedQueryEpisode,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> np.ndarray:
    """Build a matched context/padding/latent state-advance schedule."""
    total = encoded.events.shape[0] + config.latent_steps
    advances = np.zeros((total,), dtype=np.bool_)
    width = _demonstration_advance_width(encoded, row_config)
    for start, _stop in encoded.demonstration_spans:
        advances[start : start + width] = True
    advances[encoded.query_start : encoded.query_stop] = True
    advances[encoded.query_stop : encoded.query_stop + config.latent_steps] = True
    return advances


def _effort_schedule(updates: int, rng: brainstate.random.RandomState) -> np.ndarray:
    base = np.resize(np.asarray(TRAINING_EFFORTS, dtype=np.int32), updates)
    order = np.asarray(rng.permutation(updates), dtype=np.int32)
    return base[order]


def _empty_training_tensors() -> _TrainingTensors:
    empty = np.zeros((0,), dtype=np.float32)
    return _TrainingTensors(empty, empty, empty, empty, empty, empty, empty, ())


def _compact_training_stream(
    encoded: EncodedQueryEpisode,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Move every semantic training advance into one static-shape prefix.

    Returns the compact event tensor, its prefix-only advance schedule, and the
    compact index of the query-terminal checkpoint.  The gathered rows retain
    the order and physical semantics of :func:`_packed_advances`; only frozen
    layout positions move behind the final latent tick, where no later loss can
    consume their eligibility trace.
    """
    padded = _packed_events(encoded, config)
    padded_advances = _packed_advances(encoded, config, row_config)
    active_indices = np.flatnonzero(padded_advances)
    query_terminal = encoded.query_stop - 1
    compact_query = np.flatnonzero(active_indices == query_terminal)
    if compact_query.size != 1:
        raise ValueError("query terminal must be one semantic training advance")

    compact = np.zeros_like(padded)
    compact[: active_indices.size] = padded[active_indices]
    advances = np.zeros_like(padded_advances)
    advances[: active_indices.size] = True
    query_checkpoint = int(compact_query[0])
    latent_count = int(active_indices.size) - query_checkpoint - 1
    if latent_count != config.latent_steps:
        raise ValueError("compact training prefix has the wrong latent length")
    return compact, advances, query_checkpoint


def _without_official_test_targets(task: ArcTask) -> ArcTask:
    """Return a task whose official queries cannot leak labels into training."""

    return ArcTask(
        train=task.train,
        test=tuple(ArcPair(pair.input, None) for pair in task.test),
        task_id=task.task_id,
    )


def _training_row(
    origin: _OriginTask,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    rng: brainstate.random.RandomState,
    *,
    effort: int,
    plumbing_only: bool,
) -> dict[str, Any]:
    """Encode one augmented leave-one-demonstration-out training update."""

    base_task = _without_official_test_targets(origin.task)
    task = (
        base_task
        if plumbing_only
        else augment_training_task(base_task, rng, role="train")
    )
    episodes = leave_one_demonstration_out_episodes(task)
    held_out_index = int(np.asarray(rng.randint(0, len(episodes))))
    encoded = encode_arc_query_episode(episodes[held_out_index], row_config)
    if encoded.target is None:
        raise ValueError(
            f"training fold {task.task_id or encoded.task_fingerprint}:"
            f"{held_out_index} lacks a target"
        )
    sequence, advances, query_checkpoint = _compact_training_stream(
        encoded, config, row_config
    )
    mask = np.zeros((sequence.shape[0],), dtype=np.float32)
    terminal = query_checkpoint + effort
    if effort > config.latent_steps or terminal >= int(np.count_nonzero(advances)):
        raise ValueError("terminal effort exceeds packed sequence capacity")
    depth_count = effort + 1
    mask[query_checkpoint : terminal + 1] = np.float32(1.0 / depth_count)
    target = encoded.target
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[: target.height, : target.width] = target.as_array()
    return {
        "events": sequence[:, None, :],
        "advances": advances[:, None],
        "heights": target.height,
        "widths": target.width,
        "colors": padded[None],
        "masks": mask,
        "task_fingerprints": canonical_task_fingerprint(task),
        "base_task_fingerprints": canonical_task_fingerprint(base_task),
        "source_names": origin.source_name,
        "held_out_demonstration_index": held_out_index,
    }


def _stacked_chunk(rows: list[dict[str, Any]], efforts: np.ndarray) -> _TrainingTensors:
    def column(name: str) -> list[Any]:
        return [row[name] for row in rows]

    return _TrainingTensors(
        events=np.stack(column("events")),
        advances=np.stack(column("advances")),
        heights=np.asarray(column("heights"), dtype=np.int32)[:, None],
        widths=np.asarray(column("widths"), dtype=np.int32)[:, None],
        colors=np.stack(column("colors")),
        masks=np.stack(column("masks")),
        efforts=efforts,
        task_fingerprints=tuple(column("task_fingerprints")),
        base_task_fingerprints=tuple(column("base_task_fingerprints")),
        source_names=tuple(column("source_names")),
        held_out_demonstration_indices=tuple(column("held_out_demonstration_index")),
    )


def _training_chunks(
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> Iterator[_TrainingTensors]:
    """Yield the training schedule in fixed-size, device-sized chunks.

    The effort and task draws stay up front at full ``training_updates`` size
    and the per-update walk visits updates in schedule order, so the random
    stream — and therefore every produced tensor — is independent of how the
    schedule is chunked.
    """
    if config.structural_only:
        yield _empty_training_tensors()
        return
    if not data.training:
        raise ValueError("training data is empty")
    rng = brainstate.random.RandomState(config.seed + 1000)
    efforts = _effort_schedule(config.training_updates, rng)
    task_indices = np.asarray(
        rng.randint(0, len(data.training), size=config.training_updates), dtype=np.int32
    )
    size = config.training_chunk_size or config.training_updates
    rows: list[dict[str, Any]] = []
    for update_index, task_index in enumerate(task_indices):
        rows.append(
            _training_row(
                data.training[int(task_index)],
                config,
                row_config,
                rng,
                effort=int(efforts[update_index]),
                plumbing_only=data.plumbing_only,
            )
        )
        if len(rows) == size:
            start = update_index + 1 - size
            yield _stacked_chunk(rows, efforts[start : update_index + 1])
            rows = []


_CHUNK_ARRAY_FIELDS = (
    "events",
    "advances",
    "heights",
    "widths",
    "colors",
    "masks",
    "efforts",
)
_CHUNK_METADATA_FIELDS = (
    "task_fingerprints",
    "base_task_fingerprints",
    "source_names",
    "held_out_demonstration_indices",
)


def _concatenated_chunks(chunks: list[_TrainingTensors]) -> _TrainingTensors:
    arrays = {
        name: np.concatenate([getattr(chunk, name) for chunk in chunks])
        for name in _CHUNK_ARRAY_FIELDS
    }
    metadata = {
        name: tuple(value for chunk in chunks for value in getattr(chunk, name))
        for name in _CHUNK_METADATA_FIELDS
    }
    return _TrainingTensors(**arrays, **metadata)


def _prepare_training(
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> _TrainingTensors:
    """Materialise the whole training schedule.

    Retained for tests and inspection. The run path uses
    :func:`_training_chunks` so that peak memory does not scale with
    ``training_updates``.
    """
    chunks = list(_training_chunks(data, config, row_config))
    return chunks[0] if len(chunks) == 1 else _concatenated_chunks(chunks)


def _model_config(
    config: ExperimentConfig, row_config: RowEventConfig, *, batch_size: int
) -> ModelConfig:
    arguments: dict[str, object] = {
        "input_width": row_config.input_width,
        "batch_size": batch_size,
        "neuron_count": config.neuron_count,
        "recurrent_edges": config.recurrent_edges,
        "max_latent_steps": config.latent_steps,
        "readout_width": config.readout_width,
        "color_rank": config.color_rank,
        "seed": config.seed,
    }
    if config.context_memory_width > 0:
        features = associative_memory_feature_indices(row_config)
        arguments.update(
            {
                "context_memory_width": config.context_memory_width,
                "memory_decay": config.memory_decay,
                "demonstration_phase_index": row_config.phase_slice.start,
                "query_phase_index": row_config.phase_slice.start + 1,
                "input_side_valid_index": row_config.side_valid_slice.start,
                "output_side_valid_index": row_config.side_valid_slice.start + 1,
                "memory_key_indices": features.key_indices,
                "memory_value_indices": features.value_indices,
            }
        )
    return ModelConfig(**arguments)


def _memory_architecture_report(
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    training_batch_size: int,
    evaluation_batch_size: int,
) -> dict[str, object]:
    """Describe the selected reasoning mode and dense context-state cost."""
    training_batch_size = _integer(
        training_batch_size, "training_batch_size", minimum=1
    )
    evaluation_batch_size = _integer(
        evaluation_batch_size, "evaluation_batch_size", minimum=1
    )
    enabled = config.context_memory_width > 0
    if enabled:
        features = associative_memory_feature_indices(row_config)
        key_width = len(features.key_indices)
        value_width = len(features.value_indices)
    else:
        key_width = 0
        value_width = 0
    bytes_per_example = config.context_memory_width**2 * np.dtype(np.float32).itemsize
    return {
        "reasoning_mode": ("associative_workspace" if enabled else "legacy_reservoir"),
        "context_memory_width": config.context_memory_width,
        "memory_decay": config.memory_decay,
        "raw_key_feature_width": key_width,
        "raw_value_feature_width": value_width,
        "context_memory_bytes_per_example": bytes_per_example,
        "context_memory_bytes_training_batch": (
            bytes_per_example * training_batch_size
        ),
        "context_memory_bytes_evaluation_batch": (
            bytes_per_example * evaluation_batch_size
        ),
    }


def _model_memory_report(model: LatentWorkspaceModel) -> dict[str, object]:
    """Return the model-owned associative representation provenance."""
    return model.associative_memory_report().to_dict()


def _make_model(
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    batch_size: int,
    device: jax.Device,
) -> LatentWorkspaceModel:
    with jax.default_device(device), brainstate.random.seed_context(config.seed):
        return LatentWorkspaceModel(
            _model_config(config, row_config, batch_size=batch_size)
        )


def _copy_parameters(
    source: LatentWorkspaceModel, target: LatentWorkspaceModel
) -> None:
    source_states = source.states(brainstate.ParamState)
    target_states = target.states(brainstate.ParamState)
    if tuple(source_states.keys()) != tuple(target_states.keys()):
        raise ValueError("training and evaluation parameter paths differ")
    for source_state, target_state in zip(
        source_states.values(), target_states.values(), strict=True
    ):
        target_state.value = jax.tree.map(jnp.array, source_state.value)


def _tree_digest(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        for leaf in jax.tree.leaves(values[key]):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _compiler_evidence(learner: Any) -> dict[str, object]:
    report = getattr(learner, "report", None)
    if report is None:
        return {
            "available": False,
            "counts": {},
            "etrace_weights": [],
            "excluded_weights": [],
            "diagnostics": [],
        }

    def path_text(path: object) -> str:
        if isinstance(path, (tuple, list)):
            return ".".join(str(part) for part in path)
        return str(path)

    def enum_text(value: object) -> str:
        return str(getattr(value, "value", value))

    diagnostics = []
    for record in report.diagnostics:
        item: dict[str, object] = {
            "kind": enum_text(record.kind),
            "level": enum_text(record.level),
            "message": str(record.message),
        }
        if hasattr(record, "weight_path"):
            item["weight_path"] = path_text(record.weight_path)
        if hasattr(record, "hidden_paths"):
            item["hidden_paths"] = [path_text(path) for path in record.hidden_paths]
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            path_classification = context.get("path_classification")
            if isinstance(path_classification, dict):
                item["path_classification_by_hidden_state"] = {
                    path_text(path): enum_text(classification)
                    for path, classification in path_classification.items()
                }
        diagnostics.append(item)
    warning_count = sum(item["level"] == "warning" for item in diagnostics)
    error_count = sum(item["level"] == "error" for item in diagnostics)
    return {
        "available": True,
        "counts": {
            "hidden_groups": len(report.hidden_groups),
            "etrace_weights": len(report.etrace_weights),
            "excluded_weights": len(report.excluded_weights),
            "warnings": warning_count,
            "errors": error_count,
        },
        "etrace_weights": [
            {
                "parameter": path_text(path),
                "hidden_group_indices": [int(index) for index in groups],
            }
            for path, groups in report.etrace_weights
        ],
        "excluded_weights": [
            {"parameter": path_text(path), "reason": str(reason)}
            for path, reason in report.excluded_weights
        ],
        "diagnostics": diagnostics,
    }


def _parameter_change_evidence(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, object]]:
    if before.keys() != after.keys():
        raise ValueError("parameter paths changed during optimization")
    result: dict[str, dict[str, object]] = {}
    for path in before:
        before_leaves = jax.tree.leaves(before[path])
        after_leaves = jax.tree.leaves(after[path])
        if len(before_leaves) != len(after_leaves):
            raise ValueError(f"parameter structure changed during optimization: {path}")
        squared = 0.0
        for before_leaf, after_leaf in zip(before_leaves, after_leaves, strict=True):
            delta = np.asarray(after_leaf, dtype=np.float64) - np.asarray(
                before_leaf, dtype=np.float64
            )
            squared += float(np.sum(delta * delta))
        before_sha = _tree_digest({path: before[path]})
        after_sha = _tree_digest({path: after[path]})
        result[path] = {
            "l2_delta": math.sqrt(squared),
            "changed": before_sha != after_sha,
            "sha256_before": before_sha,
            "sha256_after": after_sha,
        }
    return result


@dataclass(frozen=True)
class _TrainingSchedule:
    """Per-update training metadata accumulated across chunks, in order."""

    efforts: tuple[int, ...] = ()
    task_fingerprints: tuple[str, ...] = ()
    base_task_fingerprints: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    held_out_demonstration_indices: tuple[int, ...] = ()

    def extended(self, chunk: _TrainingTensors) -> "_TrainingSchedule":
        return _TrainingSchedule(
            efforts=self.efforts + tuple(int(value) for value in chunk.efforts),
            task_fingerprints=self.task_fingerprints + tuple(chunk.task_fingerprints),
            base_task_fingerprints=(
                self.base_task_fingerprints + tuple(chunk.base_task_fingerprints)
            ),
            source_names=self.source_names + tuple(chunk.source_names),
            held_out_demonstration_indices=(
                self.held_out_demonstration_indices
                + tuple(chunk.held_out_demonstration_indices)
            ),
        )


def _train_chunks(
    chunks: Iterable[_TrainingTensors],
    train_all: Any,
) -> tuple[list[float], _TrainingSchedule]:
    """Stage each chunk on device in turn and accumulate the whole schedule.

    The per-update work stays inside the single compiled ``train_all`` scan;
    this loop only walks a handful of data-staging steps so that peak device
    memory tracks the chunk size rather than ``training_updates``.
    """
    losses: list[float] = []
    schedule = _TrainingSchedule()
    sequence_length: int | None = None
    for chunk in chunks:
        if sequence_length is None:
            sequence_length = int(chunk.events.shape[1])
        elif int(chunk.events.shape[1]) != sequence_length:
            raise ValueError("training chunks disagree on packed sequence length")
        losses.extend(
            float(value)
            for value in np.asarray(
                train_all(
                    jnp.asarray(chunk.events),
                    jnp.asarray(chunk.advances),
                    jnp.asarray(chunk.heights),
                    jnp.asarray(chunk.widths),
                    jnp.asarray(chunk.colors),
                    jnp.asarray(chunk.masks),
                )
            )
        )
        schedule = schedule.extended(chunk)
    if len(losses) != len(schedule.efforts):
        raise ValueError("training losses and effort schedule disagree in length")
    return losses, schedule


def _train_model(
    model: LatentWorkspaceModel,
    chunks: Iterable[_TrainingTensors],
    config: ExperimentConfig,
) -> dict[str, object]:
    learner = compile_pp_prop(model)
    compiler_report = _compiler_evidence(learner)
    compiler = {
        "pp_prop_compiled": True,
        "learner_type": type(learner).__name__,
        "event_and_advance_arguments": True,
        "compiler_report": compiler_report,
        "compiled_parameter_paths": [
            ".".join(str(part) for part in path)
            if isinstance(path, tuple)
            else str(path)
            for path in getattr(learner, "param_states", {}).keys()
        ],
    }
    if config.structural_only:
        model.reset_state()
        learner.reset_state(batch_size=model.config.batch_size)
        return {
            "performed": False,
            "reason": "structural_only",
            "one_shared_model": True,
            "supervised_depths": "0..effort",
            "depth_weighting": "uniform_unit_sum_per_update",
            "balanced_color_loss": config.balanced_color_loss,
            **compiler,
            "optimizer_updates_by_effort": {
                str(value): 0 for value in TRAINING_EFFORTS
            },
            "losses": [],
        }
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(learner.param_states)
    before_snapshot = parameter_snapshot(model)
    before = _tree_digest(before_snapshot)
    rank = model.config.color_rank

    @brainstate.transform.jit
    def train_all(events, advances, heights, widths, colors, masks):
        def train_one(inputs):
            sequence, advance, target_height, target_width, target_colors, mask = inputs
            model.reset_state()
            learner.reset_state(batch_size=model.config.batch_size)

            def step_loss(event, advance_gate):
                compact = learner(event, advance_gate)
                return jnp.mean(
                    arc_loss_per_example(
                        compact,
                        target_height,
                        target_width,
                        target_colors,
                        color_rank=rank,
                        class_balanced_colors=config.balanced_color_loss,
                    )
                )

            gradients, objective = learner.etrace_grad(
                sequence,
                advance,
                step_fn=step_loss,
                mask=mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            return objective

        return brainstate.transform.for_loop(
            train_one, (events, advances, heights, widths, colors, masks)
        )

    losses, schedule = _train_chunks(chunks, train_all)
    after_snapshot = parameter_snapshot(model)
    after = _tree_digest(after_snapshot)
    counts = Counter(schedule.efforts)
    sample_records = [
        {
            "source": source,
            "base_task_fingerprint": base_fingerprint,
            "augmented_task_fingerprint": augmented_fingerprint,
            "episode_kind": "leave_one_demonstration_out",
            "held_out_demonstration_index": int(held_out_index),
            "maximum_supervised_depth": int(effort),
        }
        for source, base_fingerprint, augmented_fingerprint, held_out_index, effort in zip(
            schedule.source_names,
            schedule.base_task_fingerprints,
            schedule.task_fingerprints,
            schedule.held_out_demonstration_indices,
            schedule.efforts,
            strict=True,
        )
    ]
    return {
        "performed": True,
        "one_shared_model": True,
        "one_shared_optimizer_state": True,
        **compiler,
        "supervised_depths": "0..effort",
        "depth_weighting": "uniform_unit_sum_per_update",
        "per_update_depth_weight_sum": 1.0,
        "balanced_color_loss": config.balanced_color_loss,
        "loss_weights": {"height": 1.0, "width": 1.0, "valid_cell_color": 1.0},
        "optimizer_updates_by_effort": {
            str(value): int(counts[value]) for value in TRAINING_EFFORTS
        },
        "losses": np.asarray(losses, dtype=np.float64).tolist(),
        "effort_schedule": list(schedule.efforts),
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameters_moved": before != after,
        "parameter_changes": _parameter_change_evidence(
            before_snapshot, after_snapshot
        ),
        "training_task_fingerprints": list(schedule.task_fingerprints),
        "training_episode_kind": "leave_one_demonstration_out",
        "sampled_base_task_count": len(set(schedule.base_task_fingerprints)),
        "sampled_base_fold_count": len(
            set(
                zip(
                    schedule.base_task_fingerprints,
                    schedule.held_out_demonstration_indices,
                    strict=True,
                )
            )
        ),
        "sampling_with_replacement": True,
        "training_samples": sample_records,
    }


def _origin_task_key(origin: _OriginTask) -> str:
    fingerprint = canonical_task_fingerprint(origin.task)
    task_name = origin.task.task_id or fingerprint[:12]
    return f"{origin.source_name}:{task_name}:{fingerprint}"


def _evaluation_records(
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> tuple[_EvaluationRecord, ...]:
    origins = data.evaluation
    if config.evaluation_task_limit is not None:
        origins = origins[: config.evaluation_task_limit]
    records: list[_EvaluationRecord] = []
    for task_index, origin in enumerate(origins):
        task_key = _origin_task_key(origin)
        for query_index in range(len(origin.task.test)):
            encoded = encode_query_episode(
                origin.task,
                query_index,
                row_config,
                task_index=task_index,
            )
            if encoded.target is None:
                raise ValueError(
                    f"evaluation query {task_key}:{query_index} lacks target"
                )
            records.append(_EvaluationRecord(origin.source_name, task_key, encoded))
    if not records:
        raise ValueError("evaluation produced no scored queries")
    return tuple(records)


def _derange_task(task: ArcTask) -> ArcTask | None:
    if len(task.train) < 2:
        return None
    outputs = tuple(pair.output for pair in task.train)
    return ArcTask(
        train=tuple(
            ArcPair(pair.input, outputs[(index + 1) % len(outputs)])
            for index, pair in enumerate(task.train)
        ),
        test=task.test,
        task_id=task.task_id,
    )


def _arm_sequences(
    records: Sequence[_EvaluationRecord],
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    arm: Literal["intact", "no_context", "shuffled"],
    source_tasks: Sequence[_OriginTask],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    sequences: list[np.ndarray] = []
    advance_rows: list[np.ndarray] = []
    query_stops: list[int] = []
    metadata: list[dict[str, object]] = []
    task_lookup = {_origin_task_key(origin): origin.task for origin in source_tasks}
    for record in records:
        encoded = record.encoded
        arm_encoded = encoded
        detail: dict[str, object] = {"available": True, "timing_matched": True}
        if arm == "no_context":
            events = np.array(encoded.events, copy=True)
            events[: encoded.query_start] = 0.0
            packed = _packed_events(encoded, config)
            packed[: encoded.events.shape[0]] = events
        elif arm == "shuffled":
            changed = _derange_task(task_lookup[record.task_key])
            if changed is None:
                packed = _packed_events(encoded, config)
                detail = {
                    "available": False,
                    "reason": "fewer than two demonstrations",
                    "timing_matched": True,
                }
            else:
                arm_encoded = encode_query_episode(
                    changed,
                    encoded.query_index,
                    row_config,
                    task_index=encoded.task_index,
                )
                packed = _packed_events(arm_encoded, config)
                detail["timing_matched"] = (
                    arm_encoded.query_start == encoded.query_start
                    and arm_encoded.query_stop == encoded.query_stop
                )
                if np.array_equal(
                    arm_encoded.events[: arm_encoded.query_start],
                    encoded.events[: encoded.query_start],
                ):
                    detail = {
                        "available": False,
                        "reason": (
                            "rotation leaves demonstration associations unchanged"
                        ),
                        "timing_matched": bool(detail["timing_matched"]),
                    }
        else:
            packed = _packed_events(encoded, config)
        sequences.append(packed)
        advance_rows.append(_packed_advances(encoded, config, row_config))
        query_stops.append(encoded.query_stop)
        metadata.append(detail)
    stacked = np.stack(sequences, axis=1)
    advances = np.stack(advance_rows, axis=1)
    return stacked, advances, np.asarray(query_stops, dtype=np.int32), metadata


def _gather_window(
    packed, query_stops: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compact = np.asarray(packed.compact_logits)
    spikes = np.asarray(packed.spikes)
    voltage = np.asarray(packed.voltage)
    feedforward_current = np.asarray(packed.feedforward_current)
    recurrent_current = np.asarray(packed.recurrent_current)
    offsets = np.arange(0, max(CHECKPOINTS) + 1, dtype=np.int32)[:, None]
    indices = query_stops[None, :] - 1 + offsets
    batch = np.arange(query_stops.size, dtype=np.int32)[None, :]
    return (
        compact[indices, batch],
        spikes[indices, batch],
        voltage[indices, batch],
        feedforward_current[indices, batch],
        recurrent_current[indices, batch],
    )


def _score_windows(
    compact: np.ndarray,
    records: Sequence[_EvaluationRecord],
    color_rank: int,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Score only model-decoded candidates at every frozen checkpoint."""

    checkpoint_compact = compact[np.asarray(CHECKPOINTS, dtype=np.int32)]
    expanded = expand_compact_logits(jnp.asarray(checkpoint_compact), color_rank)
    height = np.asarray(expanded.height)
    width = np.asarray(expanded.width)
    colors = np.asarray(expanded.colors)
    metrics: dict[str, dict[str, object]] = {}
    query_details: dict[str, list[dict[str, object]]] = {}
    for effort_index, effort in enumerate(CHECKPOINTS):
        scores = []
        details = []
        for query_index, record in enumerate(records):
            logits = OutputLogits(
                height[effort_index, query_index],
                width[effort_index, query_index],
                colors[effort_index, query_index],
            )
            candidates = decode_candidates(logits)
            candidate_payloads: list[dict[str, object]] = []
            for candidate in candidates:
                payload = dict(candidate.to_dict())
                provenance = payload.get("provenance", "model")
                if provenance != "model":
                    raise ValueError(
                        "primary scoring rejected non-model candidate provenance "
                        f"{provenance!r}"
                    )
                payload["provenance"] = "model"
                candidate_payloads.append(payload)
            target = record.encoded.target
            assert target is not None
            score = score_query_candidates(
                [candidate.grid for candidate in candidates],
                target.as_array(),
                task_id=record.task_key,
                query_index=record.encoded.query_index,
            )
            scores.append(score)
            details.append(
                {
                    "task_id": record.task_key,
                    "query_index": record.encoded.query_index,
                    "primary_candidate_mode": "model_only",
                    "candidates": candidate_payloads,
                    "score": score.to_dict(),
                }
            )
        metrics[str(effort)] = aggregate_arc_metrics(scores)
        query_details[str(effort)] = details
    return metrics, query_details


def _trajectory_reports(
    compact: np.ndarray,
    spikes: np.ndarray,
    voltage: np.ndarray,
    feedforward_current: np.ndarray,
    recurrent_current: np.ndarray,
    records: Sequence[_EvaluationRecord],
    color_rank: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    expanded = expand_compact_logits(jnp.asarray(compact), color_rank)
    height = np.asarray(expanded.height)
    width = np.asarray(expanded.width)
    colors = np.asarray(expanded.colors)
    reports: list[dict[str, object]] = []
    for query_index, record in enumerate(records):
        target = record.encoded.target
        assert target is not None
        report = analyze_latent_trajectory(
            height[:, query_index],
            width[:, query_index],
            colors[:, query_index],
            spikes[:, query_index],
            voltage[:, query_index],
            feedforward_current=feedforward_current[:, query_index],
            recurrent_current=recurrent_current[:, query_index],
            target=target.as_array(),
            task_id=record.task_key,
            query_index=record.encoded.query_index,
            step_indices=np.arange(compact.shape[0]),
        )
        report["task_id"] = record.task_key
        report["query_index"] = record.encoded.query_index
        reports.append(report)

    pair_count = min(256, len(records)) if len(records) > 1 else 0
    pair_left = np.arange(pair_count, dtype=np.int32)
    pair_right = (pair_left * 131 + 17) % len(records) if pair_count else pair_left
    if pair_count:
        pair_right = np.where(
            pair_right == pair_left, (pair_right + 1) % len(records), pair_right
        )

    def distribution(values: np.ndarray) -> dict[str, float] | None:
        if values.size == 0:
            return None
        return {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
        }

    aggregate: list[dict[str, object]] = []
    for step in range(compact.shape[0]):
        rows = [report["steps"][step] for report in reports]
        if pair_count:
            pair_spike = np.mean(
                spikes[step, pair_left] != spikes[step, pair_right], axis=1
            )
            scale = math.sqrt(spikes.shape[2])
            pair_voltage = (
                np.linalg.norm(
                    voltage[step, pair_left] - voltage[step, pair_right], axis=1
                )
                / scale
            )
            pair_feedforward = (
                np.linalg.norm(
                    feedforward_current[step, pair_left]
                    - feedforward_current[step, pair_right],
                    axis=1,
                )
                / scale
            )
            pair_recurrent = (
                np.linalg.norm(
                    recurrent_current[step, pair_left]
                    - recurrent_current[step, pair_right],
                    axis=1,
                )
                / scale
            )
        else:
            pair_spike = np.asarray([], dtype=np.float64)
            pair_voltage = np.asarray([], dtype=np.float64)
            pair_feedforward = np.asarray([], dtype=np.float64)
            pair_recurrent = np.asarray([], dtype=np.float64)
        aggregate.append(
            {
                "step": step,
                "mean_firing_rate": float(
                    np.mean([row["firing_rate"] for row in rows])
                ),
                "mean_spike_count": float(
                    np.mean([row["spike_count"] for row in rows])
                ),
                "mean_voltage_l2": float(np.mean([row["voltage_l2"] for row in rows])),
                "mean_feedforward_current_l2": float(
                    np.mean([row["feedforward_current_l2"] for row in rows])
                ),
                "mean_recurrent_current_l2": float(
                    np.mean([row["recurrent_current_l2"] for row in rows])
                ),
                "mean_predictive_entropy": float(
                    np.mean([row["predictive_entropy"] for row in rows])
                ),
                "mean_changed_cell_fraction": (
                    None
                    if step == 0
                    else float(np.mean([row["changed_cell_fraction"] for row in rows]))
                ),
                "converged_fraction": float(
                    np.mean([row["converged"] for row in rows])
                ),
                "near_silence_fraction": float(
                    np.mean([row["near_silence"] for row in rows])
                ),
                "near_saturation_fraction": float(
                    np.mean([row["near_saturation"] for row in rows])
                ),
                "unique_state_hashes": len({row["state_sha256"] for row in rows}),
                "pair_sample_count": pair_count,
                "pair_sampling": "deterministic modular query pairs",
                "pairwise_spike_hamming_fraction": distribution(pair_spike),
                "pairwise_voltage_rms_distance": distribution(pair_voltage),
                "pairwise_feedforward_current_rms_distance": distribution(
                    pair_feedforward
                ),
                "pairwise_recurrent_current_rms_distance": distribution(pair_recurrent),
            }
        )
    return reports, aggregate


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _control_summary(
    name: str,
    intact: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    control: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    records: Sequence[_EvaluationRecord],
    color_rank: int,
    intact_metrics: dict[str, dict[str, object]],
    metadata: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    if metadata is None:
        metadata = tuple({"available": True, "timing_matched": True} for _ in records)
    if len(metadata) != len(records):
        raise ValueError("control metadata must match the evaluation records")
    applicable = np.asarray(
        [bool(item.get("available", False)) for item in metadata], dtype=np.bool_
    )
    applicable_indices = np.flatnonzero(applicable)
    applicable_records = tuple(records[index] for index in applicable_indices)

    def subset(window: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return tuple(value[:, applicable_indices] for value in window)

    applicable_intact = subset(intact)
    applicable_control = subset(control)
    if applicable_records:
        metrics, control_checkpoint_queries = _score_windows(
            applicable_control[0],
            applicable_records,
            color_rank,
        )
        matched_intact_metrics, intact_checkpoint_queries = _score_windows(
            applicable_intact[0],
            applicable_records,
            color_rank,
        )
        if (
            len(applicable_records) == len(records)
            and matched_intact_metrics != intact_metrics
        ):
            raise ValueError("recomputed intact control metrics are inconsistent")
        candidate_match_count_by_effort: dict[str, int] = {}
        candidates_match_by_effort: dict[str, bool] = {}
        for effort in CHECKPOINTS:
            key = str(effort)
            intact_rows = intact_checkpoint_queries[key]
            control_rows = control_checkpoint_queries[key]
            if len(intact_rows) != len(control_rows):
                raise ValueError("matched control candidate rows differ in length")
            match_count = int(
                sum(
                    intact_row["candidates"] == control_row["candidates"]
                    for intact_row, control_row in zip(
                        intact_rows, control_rows, strict=True
                    )
                )
            )
            candidate_match_count_by_effort[key] = match_count
            candidates_match_by_effort[key] = match_count == len(intact_rows)
        decoded_candidates_match = bool(all(candidates_match_by_effort.values()))
    else:
        metrics = {}
        matched_intact_metrics = {}
        candidate_match_count_by_effort = {}
        candidates_match_by_effort = {}
        decoded_candidates_match = None

    if applicable_records:
        intact_spikes = applicable_intact[1]
        intact_voltage = applicable_intact[2]
        control_spikes = applicable_control[1]
        control_voltage = applicable_control[2]
        comparison = compare_control_trajectories(
            intact_spikes.transpose(0, 2, 1).reshape(intact_spikes.shape[0], -1),
            intact_voltage.transpose(0, 2, 1).reshape(intact_voltage.shape[0], -1),
            control_spikes.transpose(0, 2, 1).reshape(control_spikes.shape[0], -1),
            control_voltage.transpose(0, 2, 1).reshape(control_voltage.shape[0], -1),
            control_name=name,
            intact_scores=matched_intact_metrics,
            control_scores=metrics,
            intact_synaptic_currents={
                "feedforward": applicable_intact[3]
                .transpose(0, 2, 1)
                .reshape(applicable_intact[3].shape[0], -1),
                "recurrent": applicable_intact[4]
                .transpose(0, 2, 1)
                .reshape(applicable_intact[4].shape[0], -1),
            },
            control_synaptic_currents={
                "feedforward": applicable_control[3]
                .transpose(0, 2, 1)
                .reshape(applicable_control[3].shape[0], -1),
                "recurrent": applicable_control[4]
                .transpose(0, 2, 1)
                .reshape(applicable_control[4].shape[0], -1),
            },
        )
        comparison["state_byte_identical_by_step"] = [
            all(
                _array_bytes_equal(
                    applicable_intact[state_index][step_index],
                    applicable_control[state_index][step_index],
                )
                for state_index in range(1, 5)
            )
            for step_index in range(applicable_intact[0].shape[0])
        ]
    else:
        comparison = {
            "control_name": name,
            "available": False,
            "causally_null_at_measured_precision": None,
            "interpretation": f"{name} had no applicable evaluation queries.",
        }
    per_query_null = [
        bool(
            all(
                _array_bytes_equal(
                    intact[state_index][:, index],
                    control[state_index][:, index],
                )
                for state_index in range(1, 5)
            )
        )
        for index in applicable_indices
    ]
    timing_matched_applicable = int(
        sum(
            bool(item.get("timing_matched", False))
            for item, is_applicable in zip(metadata, applicable, strict=True)
            if is_applicable
        )
    )
    result: dict[str, object] = {
        "metrics_by_effort": metrics,
        "trajectory_comparison": comparison,
        "decoded_candidates_match_intact": decoded_candidates_match,
        "decoded_candidates_match_intact_by_effort": candidates_match_by_effort,
        "decoded_candidate_match_query_count_by_effort": (
            candidate_match_count_by_effort
        ),
        "causally_null_query_count": int(sum(per_query_null)),
        "byte_identical_query_count": int(sum(per_query_null)),
        "query_count": len(records),
        "applicable_query_count": int(applicable_indices.size),
        "available_query_count": int(applicable_indices.size),
        "unavailable_query_count": int(len(records) - applicable_indices.size),
        "timing_matched_applicable_query_count": timing_matched_applicable,
        "intervention_metadata": list(metadata),
    }
    result["timing_matched_query_count"] = int(
        sum(bool(item.get("timing_matched", False)) for item in metadata)
    )
    return result


def _state_tolerance_summary(
    intact: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    step_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    leading_shape = intact[0].shape[:2]
    for name, left, right in zip(
        ("compact", "spikes", "voltage", "feedforward_current", "recurrent_current"),
        intact,
        candidate,
        strict=True,
    ):
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError(f"matched {name} windows must have equal rank-3 shapes")
        if left.shape[:2] != leading_shape:
            raise ValueError(
                f"matched {name} windows must share step and query dimensions"
            )
    if leading_shape[1] < 1:
        raise ValueError("matched state windows must contain at least one query")
    if step_indices is None:
        indices = np.arange(intact[0].shape[0], dtype=np.int32)
    else:
        indices = np.asarray(step_indices, dtype=np.int32)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("step_indices must be a nonempty rank-1 sequence")
        if np.any(indices < 0) or np.any(indices >= intact[0].shape[0]):
            raise ValueError("step_indices exceed the matched windows")

    selected_spikes = intact[1][indices]
    candidate_spikes = candidate[1][indices]
    spike_difference = selected_spikes != candidate_spikes
    spike_hamming_by_query = np.count_nonzero(spike_difference, axis=(0, 2)).astype(
        np.int64
    )
    spike_hamming = int(np.sum(spike_hamming_by_query))
    numeric_names = (
        "compact_logits",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    )
    per_step_query_rms: dict[str, list[list[float]]] = {}
    per_query_maximum_rms: dict[str, list[float]] = {}
    per_query_maximum_absolute: dict[str, list[float]] = {}
    maximum_absolute: dict[str, float] = {}
    for name, state_index in zip(numeric_names, (0, 2, 3, 4), strict=True):
        delta = np.asarray(
            candidate[state_index][indices], dtype=np.float64
        ) - np.asarray(intact[state_index][indices], dtype=np.float64)
        rms = np.sqrt(np.mean(delta * delta, axis=2))
        query_maximum = np.max(rms, axis=0)
        absolute_query_maximum = np.max(np.abs(delta), axis=(0, 2))
        per_step_query_rms[name] = rms.tolist()
        per_query_maximum_rms[name] = query_maximum.tolist()
        per_query_maximum_absolute[name] = absolute_query_maximum.tolist()
        maximum_absolute[name] = float(np.max(np.abs(delta)))
    maximum_rms = {
        name: float(max(values)) for name, values in per_query_maximum_rms.items()
    }
    intact_dtype_by_state = {
        name: str(intact[state_index].dtype)
        for name, state_index in zip(numeric_names, (0, 2, 3, 4), strict=True)
    }
    candidate_dtype_by_state = {
        name: str(candidate[state_index].dtype)
        for name, state_index in zip(numeric_names, (0, 2, 3, 4), strict=True)
    }
    required_float32_dtypes = bool(
        all(value == "float32" for value in intact_dtype_by_state.values())
        and candidate_dtype_by_state == intact_dtype_by_state
    )
    state_byte_identical_by_query = np.asarray(
        [
            all(
                _array_bytes_equal(
                    intact[state_index][indices, query_index],
                    candidate[state_index][indices, query_index],
                )
                for state_index in range(1, 5)
            )
            for query_index in range(leading_shape[1])
        ],
        dtype=np.bool_,
    )
    compact_byte_identical_by_query = np.asarray(
        [
            _array_bytes_equal(
                intact[0][indices, query_index],
                candidate[0][indices, query_index],
            )
            for query_index in range(leading_shape[1])
        ],
        dtype=np.bool_,
    )
    within_tolerance_by_query = spike_hamming_by_query == 0
    for values in per_query_maximum_rms.values():
        within_tolerance_by_query &= np.asarray(values) <= STATE_RMS_TOLERANCE
    within_tolerance_by_query &= required_float32_dtypes
    within_tolerance_query_count = int(np.count_nonzero(within_tolerance_by_query))
    query_count = int(leading_shape[1])
    return {
        "evaluated_steps": indices.astype(int).tolist(),
        "query_count": query_count,
        "state_byte_identical": bool(np.all(state_byte_identical_by_query)),
        "state_byte_identical_by_query": state_byte_identical_by_query.tolist(),
        "state_byte_identical_by_step": [
            all(
                _array_bytes_equal(
                    intact[state_index][step_index],
                    candidate[state_index][step_index],
                )
                for state_index in range(1, 5)
            )
            for step_index in indices
        ],
        "compact_logits_byte_identical": bool(np.all(compact_byte_identical_by_query)),
        "compact_logits_byte_identical_by_query": (
            compact_byte_identical_by_query.tolist()
        ),
        "within_declared_tolerance": within_tolerance_query_count == query_count,
        "within_tolerance_by_query": within_tolerance_by_query.tolist(),
        "within_tolerance_query_count": within_tolerance_query_count,
        "declared_per_query_axis_rms_tolerance": STATE_RMS_TOLERANCE,
        "spike_hamming_count": spike_hamming,
        "spike_hamming_count_by_query": spike_hamming_by_query.tolist(),
        "per_step_query_rms": per_step_query_rms,
        "per_query_maximum_rms": per_query_maximum_rms,
        "maximum_rms": maximum_rms,
        "per_query_maximum_absolute": per_query_maximum_absolute,
        "maximum_absolute": maximum_absolute,
        "intact_dtype_by_state": intact_dtype_by_state,
        "candidate_dtype_by_state": candidate_dtype_by_state,
        "required_float32_dtypes": required_float32_dtypes,
    }


def _checkpoint_zero_gate_summary(
    intact_metrics: dict[str, dict[str, object]],
    control: dict[str, object],
    numeric_summary: dict[str, object],
) -> dict[str, bool]:
    state_within_tolerance = numeric_summary.get("within_declared_tolerance") is True
    candidate_matches = control.get("decoded_candidates_match_intact_by_effort")
    decoded_candidates_exact = bool(
        isinstance(candidate_matches, dict) and candidate_matches.get("0") is True
    )
    control_metrics = control.get("metrics_by_effort")
    metrics_exact = bool(
        isinstance(control_metrics, dict)
        and "0" in intact_metrics
        and "0" in control_metrics
        and control_metrics["0"] == intact_metrics["0"]
    )
    return {
        "state_within_tolerance": state_within_tolerance,
        "decoded_candidates_exact": decoded_candidates_exact,
        "metrics_exact": metrics_exact,
        "matched": bool(
            state_within_tolerance and decoded_candidates_exact and metrics_exact
        ),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _associative_evaluation_diagnostics(
    enabled: bool,
    intact: tuple[np.ndarray, np.ndarray, np.ndarray],
    controls: dict[
        str,
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray],
            Sequence[dict[str, object]],
        ],
    ],
) -> dict[str, object]:
    """Summarize bounded pairing-sensitive ``S_K``/read/workspace evidence."""
    if not enabled:
        return {
            "available": False,
            "complete": True,
            "reason": "legacy_reservoir_has_no_associative_state",
        }

    def validate(
        name: str, window: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        workspace, memory_read, context_memory = map(np.asarray, window)
        if workspace.ndim != 3 or memory_read.ndim != 3:
            raise ValueError(f"{name} workspace/read diagnostics must be rank three")
        if workspace.shape[:2] != memory_read.shape[:2]:
            raise ValueError(f"{name} workspace/read checkpoints must align")
        if context_memory.ndim != 3 or context_memory.shape[0] != workspace.shape[1]:
            raise ValueError(f"{name} context memory batch must align")
        if context_memory.shape[1] != context_memory.shape[2]:
            raise ValueError(f"{name} context memory must be square")
        if memory_read.shape[2] != context_memory.shape[1]:
            raise ValueError(f"{name} memory read width must match context memory")
        if context_memory.shape[1] < 1:
            raise ValueError(f"{name} associative diagnostics must have positive width")
        for array in (workspace, memory_read, context_memory):
            if (
                not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} associative diagnostics must be finite")
        return workspace, memory_read, context_memory

    intact_workspace, intact_read, intact_memory = validate("intact", intact)
    depth_count, query_count = intact_workspace.shape[:2]

    def l2_by_depth(value: np.ndarray) -> list[float]:
        norms = np.linalg.norm(value.astype(np.float64), axis=2)
        return np.mean(norms, axis=1).tolist()

    def comparison(
        name: str,
        window: tuple[np.ndarray, np.ndarray, np.ndarray],
        metadata: Sequence[dict[str, object]],
    ) -> dict[str, object]:
        workspace, memory_read, context_memory = validate(name, window)
        if workspace.shape != intact_workspace.shape:
            raise ValueError(f"{name} workspace shape must match intact")
        if memory_read.shape != intact_read.shape:
            raise ValueError(f"{name} memory read shape must match intact")
        if context_memory.shape != intact_memory.shape:
            raise ValueError(f"{name} context memory shape must match intact")
        if len(metadata) != query_count:
            raise ValueError(f"{name} metadata must match the query count")
        applicable = np.asarray(
            [bool(item.get("available", True)) for item in metadata],
            dtype=np.bool_,
        )
        applicable_count = int(np.count_nonzero(applicable))
        memory_delta = context_memory.astype(np.float64) - intact_memory.astype(
            np.float64
        )
        memory_l2 = np.linalg.norm(memory_delta.reshape(query_count, -1), axis=1)
        memory_rms = np.sqrt(np.mean(memory_delta * memory_delta, axis=(1, 2)))
        memory_changed = np.asarray(
            [
                not _array_bytes_equal(intact_memory[index], context_memory[index])
                for index in range(query_count)
            ],
            dtype=np.bool_,
        )

        def trajectory_delta(
            left: np.ndarray, right: np.ndarray
        ) -> tuple[list[float], list[int], int]:
            delta = right.astype(np.float64) - left.astype(np.float64)
            l2 = np.linalg.norm(delta, axis=2)
            changed = np.any(left != right, axis=2) & applicable[None, :]
            if applicable_count:
                mean_l2 = np.mean(l2[:, applicable], axis=1).tolist()
            else:
                mean_l2 = [0.0] * depth_count
            return (
                mean_l2,
                np.count_nonzero(changed, axis=1).astype(int).tolist(),
                int(np.count_nonzero(np.any(changed, axis=0))),
            )

        read_l2, read_changed, read_changed_any = trajectory_delta(
            intact_read, memory_read
        )
        workspace_l2, workspace_changed, workspace_changed_any = trajectory_delta(
            intact_workspace, workspace
        )
        zero_memory = np.asarray(
            [
                np.count_nonzero(context_memory[index]) == 0
                for index in range(query_count)
            ]
        )
        context_memory_exact = _array_bytes_equal(intact_memory, context_memory)
        memory_read_exact = _array_bytes_equal(intact_read, memory_read)
        workspace_exact = _array_bytes_equal(intact_workspace, workspace)
        return {
            "applicable_query_count": applicable_count,
            "context_memory_changed_applicable_query_count": int(
                np.count_nonzero(memory_changed & applicable)
            ),
            "context_memory_l2_by_query": memory_l2.tolist(),
            "context_memory_rms_by_query": memory_rms.tolist(),
            "context_memory_sha256_by_query": [
                _array_sha256(context_memory[index]) for index in range(query_count)
            ],
            "context_memory_zero_query_count": int(np.count_nonzero(zero_memory)),
            "memory_read_mean_l2_by_depth": read_l2,
            "memory_read_changed_query_count_by_depth": read_changed,
            "memory_read_changed_at_any_depth_applicable_query_count": (
                read_changed_any
            ),
            "workspace_carrier_mean_l2_by_depth": workspace_l2,
            "workspace_carrier_changed_query_count_by_depth": workspace_changed,
            "workspace_carrier_changed_at_any_depth_applicable_query_count": (
                workspace_changed_any
            ),
            "context_memory_byte_identical_to_intact": context_memory_exact,
            "memory_read_byte_identical_to_intact": memory_read_exact,
            "workspace_carrier_byte_identical_to_intact": workspace_exact,
            "byte_identical_to_intact": bool(
                context_memory_exact and memory_read_exact and workspace_exact
            ),
        }

    expected_controls = {
        "repeat_intact",
        "no_context",
        "shuffled_demonstrations",
        "slot_ablation",
    }
    if set(controls) != expected_controls:
        raise ValueError("associative controls are incomplete")
    control_reports = {
        name: comparison(name, *controls[name]) for name in EVALUATION_ARM_ORDER[1:]
    }
    repeat_report = control_reports["repeat_intact"]
    repeat_exact = bool(
        repeat_report["context_memory_byte_identical_to_intact"]
        and repeat_report["memory_read_byte_identical_to_intact"]
    )
    no_context_zero = (
        control_reports["no_context"]["context_memory_zero_query_count"] == query_count
    )
    shuffled_report = control_reports["shuffled_demonstrations"]
    shuffled_applicable = int(shuffled_report["applicable_query_count"])
    shuffled_pairing_sensitive = bool(
        shuffled_applicable > 0
        and int(shuffled_report["context_memory_changed_applicable_query_count"])
        == shuffled_applicable
        and int(
            shuffled_report["memory_read_changed_at_any_depth_applicable_query_count"]
        )
        == shuffled_applicable
    )
    return {
        "available": True,
        "complete": bool(
            repeat_exact and no_context_zero and shuffled_pairing_sensitive
        ),
        "depth_count": depth_count,
        "query_count": query_count,
        "intact_context_memory_sha256_by_query": [
            _array_sha256(intact_memory[index]) for index in range(query_count)
        ],
        "intact_context_memory_frobenius_norm_by_query": np.linalg.norm(
            intact_memory.astype(np.float64).reshape(query_count, -1), axis=1
        ).tolist(),
        "intact_memory_read_mean_l2_by_depth": l2_by_depth(intact_read),
        "intact_workspace_carrier_mean_l2_by_depth": l2_by_depth(intact_workspace),
        "repeat_intact_exact": repeat_exact,
        "no_context_memory_exactly_zero": no_context_zero,
        "shuffled_pairing_sensitive_for_every_applicable_query": (
            shuffled_pairing_sensitive
        ),
        "controls": control_reports,
    }


def _compile_evaluation_arm(
    model: LatentWorkspaceModel,
    selected_indices: jax.Array,
    slots: jax.Array,
):
    """Compile the common device-side driver for one selected evaluation arm."""
    selected_indices = jnp.asarray(selected_indices)
    slots = jnp.asarray(slots)

    @brainstate.transform.jit(
        inline=False,
        name="example21_evaluation_arm",
    )
    def run_arm(events, advances, gates):
        packed = run_selected_packed_stream(
            model,
            events,
            selected_indices,
            reset=True,
            advance_gates=advances,
            ablation_slots=slots,
            ablation_gates=gates,
        )
        return (
            packed.compact_logits,
            packed.spikes,
            packed.voltage,
            packed.feedforward_current,
            packed.recurrent_current,
            packed.memory_read,
            packed.final_context_memory,
        )

    return run_arm


def _evaluate(
    trained_model: LatentWorkspaceModel,
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    device: jax.Device,
) -> dict[str, object]:
    records = _evaluation_records(data, config, row_config)
    batch_size = len(records)
    model = _make_model(config, row_config, batch_size=batch_size, device=device)
    _copy_parameters(trained_model, model)
    before = _tree_digest(parameter_snapshot(model))
    intact_events, intact_advances, query_stops, intact_meta = _arm_sequences(
        records, config, row_config, arm="intact", source_tasks=data.evaluation
    )
    no_context_events, no_context_advances, no_context_stops, no_context_meta = (
        _arm_sequences(
            records, config, row_config, arm="no_context", source_tasks=data.evaluation
        )
    )
    shuffled_events, shuffled_advances, shuffled_stops, shuffled_meta = _arm_sequences(
        records, config, row_config, arm="shuffled", source_tasks=data.evaluation
    )
    if not np.array_equal(query_stops, no_context_stops) or not np.array_equal(
        query_stops, shuffled_stops
    ):
        raise ValueError("control query boundaries are not matched")

    slots = np.full((batch_size,), config.ablation_slot, dtype=np.int32)
    inactive_gates = np.zeros((intact_events.shape[0], batch_size), dtype=np.bool_)
    selected_indices = (
        query_stops[None, :]
        - 1
        + np.arange(max(CHECKPOINTS) + 1, dtype=np.int32)[:, None]
    )
    run_device_arm = _compile_evaluation_arm(
        model,
        jnp.asarray(selected_indices),
        jnp.asarray(slots),
    )
    arm_wall_seconds: dict[str, float] = {}

    def run_arm(
        name: str,
        events: np.ndarray,
        advances: np.ndarray,
        gates: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        arm_started = time.perf_counter()
        packed = run_device_arm(
            jnp.asarray(events),
            jnp.asarray(advances),
            jnp.asarray(gates),
        )
        window = tuple(np.asarray(value) for value in packed)
        arm_wall_seconds[name] = time.perf_counter() - arm_started
        del packed
        physical = window[:5]
        associative = (physical[2], window[5], window[6])
        return physical, associative

    intact, intact_associative = run_arm(
        "intact", intact_events, intact_advances, inactive_gates
    )
    repeat_intact, repeat_associative = run_arm(
        "repeat_intact", intact_events, intact_advances, inactive_gates
    )
    intact_metrics, checkpoint_queries = _score_windows(
        intact[0], records, config.color_rank
    )
    channel_attribution = _channel_attribution(checkpoint_queries)
    trajectories, aggregate_trajectory = _trajectory_reports(
        *intact, records, config.color_rank
    )

    repeat_result = _control_summary(
        "repeat_intact",
        intact,
        repeat_intact,
        records,
        config.color_rank,
        intact_metrics,
        intact_meta,
    )
    repeat_match = _state_tolerance_summary(intact, repeat_intact)
    repeat_metrics_exact = repeat_result["metrics_by_effort"] == intact_metrics
    repeat_predictions_exact = bool(repeat_result["decoded_candidates_match_intact"])
    repeat_reproducible = bool(
        repeat_match["within_declared_tolerance"]
        and repeat_metrics_exact
        and repeat_predictions_exact
    )
    repeat_comparison = repeat_result["trajectory_comparison"]
    repeat_comparison["state_byte_identical_all_steps"] = repeat_match[
        "state_byte_identical"
    ]
    repeat_comparison["state_byte_identical_by_step"] = repeat_match[
        "state_byte_identical_by_step"
    ]
    repeat_comparison["within_declared_reproducibility_tolerance"] = repeat_reproducible
    repeat_comparison["causally_null_at_measured_precision"] = repeat_reproducible
    repeat_comparison["interpretation"] = (
        "repeat_intact is reproducible within the declared tolerance; spikes, "
        "decoded candidates, and metrics are exact while logit/state/current RMS "
        f"differences are at most {STATE_RMS_TOLERANCE:.1e} on the feature axis "
        "for logits and neuron axis for physical state at every checkpoint/query."
        if repeat_reproducible
        else "repeat_intact exceeded the declared reproducibility tolerance."
    )
    repeat_result["causally_null_query_count"] = repeat_match[
        "within_tolerance_query_count"
    ]
    repeat_result["within_tolerance_query_count"] = repeat_match[
        "within_tolerance_query_count"
    ]
    del repeat_intact

    no_context, no_context_associative = run_arm(
        "no_context", no_context_events, no_context_advances, inactive_gates
    )
    no_context_result = _control_summary(
        "no_context",
        intact,
        no_context,
        records,
        config.color_rank,
        intact_metrics,
        no_context_meta,
    )
    del no_context

    shuffled, shuffled_associative = run_arm(
        "shuffled_demonstrations",
        shuffled_events,
        shuffled_advances,
        inactive_gates,
    )
    shuffled_result = _control_summary(
        "shuffled_demonstrations",
        intact,
        shuffled,
        records,
        config.color_rank,
        intact_metrics,
        shuffled_meta,
    )
    del shuffled

    gates = inactive_gates.copy()
    gates[query_stops, np.arange(batch_size)] = True
    ablated, ablated_associative = run_arm(
        "slot_ablation", intact_events, intact_advances, gates
    )
    ablation_result = _control_summary(
        f"slot_ablation_{config.ablation_slot}",
        intact,
        ablated,
        records,
        config.color_rank,
        intact_metrics,
        intact_meta,
    )
    pre_intervention_match = _state_tolerance_summary(
        intact, ablated, step_indices=(0,)
    )
    ablation_checkpoint_zero = _checkpoint_zero_gate_summary(
        intact_metrics, ablation_result, pre_intervention_match
    )
    associative_diagnostics = _associative_evaluation_diagnostics(
        config.context_memory_width > 0,
        intact_associative,
        {
            "repeat_intact": (repeat_associative, intact_meta),
            "no_context": (no_context_associative, no_context_meta),
            "shuffled_demonstrations": (shuffled_associative, shuffled_meta),
            "slot_ablation": (ablated_associative, intact_meta),
        },
    )
    del (
        intact_associative,
        repeat_associative,
        no_context_associative,
        shuffled_associative,
        ablated_associative,
    )
    del ablated
    after = _tree_digest(parameter_snapshot(model))
    return {
        "query_count": batch_size,
        "task_count": len({record.task_key for record in records}),
        "same_frozen_parameter_bytes": before == after,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "primary_candidate_mode": config.primary_candidate_mode,
        "metrics_by_effort": intact_metrics,
        "channel_attribution": channel_attribution,
        "checkpoint_queries": checkpoint_queries,
        "query_trajectories": trajectories,
        "aggregate_trajectory": aggregate_trajectory,
        "associative_memory_diagnostics": associative_diagnostics,
        "determinism": {
            "same_control_capable_execution_path": True,
            "state_rms_tolerance": STATE_RMS_TOLERANCE,
            "spike_tolerance": "exact identity",
            "metric_absolute_tolerance": 0.0,
            "repeat_intact_state_byte_identical": repeat_match["state_byte_identical"],
            "repeat_intact_compact_logits_byte_identical": repeat_match[
                "compact_logits_byte_identical"
            ],
            "repeat_intact_within_tolerance": repeat_match["within_declared_tolerance"],
            "repeat_intact_metrics_exact": repeat_metrics_exact,
            "repeat_intact_decoded_candidates_exact": repeat_predictions_exact,
            "repeat_intact_numeric_evidence": repeat_match,
            "slot_ablation_checkpoint_zero_byte_identical": pre_intervention_match[
                "state_byte_identical"
            ],
            "slot_ablation_checkpoint_zero_state_within_tolerance": (
                ablation_checkpoint_zero["state_within_tolerance"]
            ),
            "slot_ablation_checkpoint_zero_decoded_candidates_exact": (
                ablation_checkpoint_zero["decoded_candidates_exact"]
            ),
            "slot_ablation_checkpoint_zero_metrics_exact": ablation_checkpoint_zero[
                "metrics_exact"
            ],
            "slot_ablation_checkpoint_zero_within_tolerance": (
                ablation_checkpoint_zero["matched"]
            ),
            "slot_ablation_checkpoint_zero_numeric_evidence": pre_intervention_match,
        },
        "controls": {
            "repeat_intact": repeat_result,
            "no_context": no_context_result,
            "shuffled_demonstrations": shuffled_result,
            "slot_ablation": ablation_result,
            "truncation": {
                "checkpoints": list(CHECKPOINTS),
                "uses_one_continuous_intact_trajectory": True,
            },
        },
        "execution": {
            "arm_order": list(EVALUATION_ARM_ORDER),
            "selected_arm_driver": "brainstate.transform.jit",
            "jit_name": "example21_evaluation_arm",
            "jit_inline": False,
            "sequential_separate_arms": True,
            "repeat_intact_cached": False,
            "wall_seconds_by_arm": arm_wall_seconds,
            "cold_intact_to_warm_repeat_ratio": (
                arm_wall_seconds["intact"] / arm_wall_seconds["repeat_intact"]
                if arm_wall_seconds["repeat_intact"] > 0.0
                else None
            ),
        },
    }


def _qualification(
    config: ExperimentConfig,
    data: _ExperimentData,
    training: dict[str, object],
    evaluation: dict[str, object],
    device_report: dict[str, object],
    model_report: dict[str, object],
) -> dict[str, object]:
    def finite_tree(value: object) -> bool:
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(item) for item in value)
        if isinstance(value, (bool, np.bool_)) or value is None:
            return True
        if isinstance(value, Real):
            return math.isfinite(float(value))
        return True

    required_metric_names = {
        "query_count",
        "task_count",
        "query_pass_at_1",
        "query_pass_at_2",
        "strict_task_pass_at_1",
        "strict_task_pass_at_2",
        "shape_accuracy_diagnostic",
        "valid_cell_pixel_accuracy_diagnostic",
    }

    def metrics_complete(value: object) -> bool:
        if not isinstance(value, dict) or set(value) != {
            str(checkpoint) for checkpoint in CHECKPOINTS
        }:
            return False
        return all(
            isinstance(row, dict)
            and required_metric_names <= row.keys()
            and int(row["query_count"]) == expected_query_count
            and int(row["task_count"]) == expected_task_count
            and finite_tree(row)
            for row in value.values()
        )

    expected_origins = data.evaluation
    if config.evaluation_task_limit is not None:
        expected_origins = expected_origins[: config.evaluation_task_limit]
    expected_task_count = len(expected_origins)
    expected_query_count = sum(len(origin.task.test) for origin in expected_origins)

    required_trajectory_names = {
        "step",
        "mean_firing_rate",
        "mean_spike_count",
        "mean_voltage_l2",
        "mean_feedforward_current_l2",
        "mean_recurrent_current_l2",
        "mean_predictive_entropy",
        "mean_changed_cell_fraction",
        "converged_fraction",
        "near_silence_fraction",
        "near_saturation_fraction",
        "unique_state_hashes",
        "pair_sample_count",
        "pairwise_spike_hamming_fraction",
        "pairwise_voltage_rms_distance",
        "pairwise_feedforward_current_rms_distance",
        "pairwise_recurrent_current_rms_distance",
    }
    aggregate = evaluation.get("aggregate_trajectory")
    aggregate_complete = bool(
        isinstance(aggregate, list)
        and len(aggregate) == max(CHECKPOINTS) + 1
        and all(
            isinstance(row, dict)
            and required_trajectory_names <= row.keys()
            and row.get("step") == index
            and finite_tree(row)
            for index, row in enumerate(aggregate)
        )
    )
    query_trajectories = evaluation.get("query_trajectories")
    query_count = int(evaluation.get("query_count", 0))
    required_query_step_names = {
        "step",
        "candidates",
        "changed_cell_count",
        "changed_cell_fraction",
        "predictive_entropy",
        "top_two_logit_margin",
        "spike_count",
        "firing_rate",
        "raster_active_indices",
        "voltage_mean",
        "voltage_std",
        "voltage_mean_absolute",
        "voltage_l2",
        "spike_hamming_displacement",
        "spike_hamming_fraction",
        "voltage_l2_displacement",
        "feedforward_current_mean_absolute",
        "feedforward_current_l2",
        "feedforward_current_l2_displacement",
        "recurrent_current_mean_absolute",
        "recurrent_current_l2",
        "recurrent_current_l2_displacement",
        "converged",
        "near_silence",
        "near_saturation",
        "state_sha256",
        "score",
    }

    def query_trajectory_complete(report: object) -> bool:
        if not isinstance(report, dict) or not isinstance(report.get("steps"), list):
            return False
        steps = report["steps"]
        return bool(
            report.get("step_count") == max(CHECKPOINTS) + 1
            and int(report.get("neuron_count", 0))
            == int(model_report.get("neuron_count", -1))
            and len(steps) == max(CHECKPOINTS) + 1
            and all(
                isinstance(row, dict)
                and required_query_step_names <= row.keys()
                and row.get("step") == index
                and isinstance(row.get("candidates"), list)
                and bool(row["candidates"])
                and finite_tree(row)
                for index, row in enumerate(steps)
            )
            and finite_tree(report)
        )

    query_trajectories_complete = bool(
        isinstance(query_trajectories, list)
        and len(query_trajectories) == query_count
        and all(query_trajectory_complete(report) for report in query_trajectories)
    )
    checkpoint_queries = evaluation.get("checkpoint_queries")
    checkpoint_queries_complete = bool(
        isinstance(checkpoint_queries, dict)
        and set(checkpoint_queries) == {str(checkpoint) for checkpoint in CHECKPOINTS}
        and all(
            isinstance(rows, list)
            and len(rows) == expected_query_count
            and all(
                isinstance(row, dict)
                and {
                    "task_id",
                    "query_index",
                    "primary_candidate_mode",
                    "candidates",
                    "score",
                }
                <= row.keys()
                and row["primary_candidate_mode"] == "model_only"
                and isinstance(row["candidates"], list)
                and bool(row["candidates"])
                and all(
                    isinstance(candidate, dict)
                    and candidate.get("provenance") == "model"
                    for candidate in row["candidates"]
                )
                and finite_tree(row)
                for row in rows
            )
            for rows in checkpoint_queries.values()
        )
    )

    def comparison_complete(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        current_distance = value.get("synaptic_current_l2_by_step")
        score_deltas = value.get("score_deltas_control_minus_intact")
        return bool(
            isinstance(value.get("causally_null_at_measured_precision"), bool)
            and len(value.get("state_byte_identical_by_step", ()))
            == max(CHECKPOINTS) + 1
            and len(value.get("spike_hamming_by_step", ())) == max(CHECKPOINTS) + 1
            and len(value.get("spike_hamming_fraction_by_step", ()))
            == max(CHECKPOINTS) + 1
            and len(value.get("voltage_l2_by_step", ())) == max(CHECKPOINTS) + 1
            and isinstance(current_distance, dict)
            and set(current_distance) == {"feedforward", "recurrent"}
            and all(
                len(current_distance[name]) == max(CHECKPOINTS) + 1
                for name in current_distance
            )
            and isinstance(score_deltas, dict)
            and {
                "32.query_pass_at_2",
                "32.valid_cell_pixel_accuracy_diagnostic",
            }
            <= score_deltas.keys()
            and finite_tree(value)
        )

    controls = evaluation.get("controls")

    def control_complete(name: str) -> bool:
        if not isinstance(controls, dict) or not isinstance(controls.get(name), dict):
            return False
        control = controls[name]
        total = int(control.get("query_count", -1))
        applicable = int(control.get("applicable_query_count", -1))
        unavailable = int(control.get("unavailable_query_count", -1))
        timing_matched = int(control.get("timing_matched_applicable_query_count", -1))
        control_metrics = control.get("metrics_by_effort")
        metrics_ok = (
            bool(
                isinstance(control_metrics, dict)
                and set(control_metrics)
                == {str(checkpoint) for checkpoint in CHECKPOINTS}
                and all(
                    isinstance(row, dict)
                    and required_metric_names <= row.keys()
                    and int(row["query_count"]) == applicable
                    and 1 <= int(row["task_count"]) <= expected_task_count
                    and finite_tree(row)
                    for row in control_metrics.values()
                )
            )
            if applicable > 0
            else control_metrics == {}
        )
        comparison_ok = (
            comparison_complete(control.get("trajectory_comparison"))
            if applicable > 0
            else isinstance(control.get("trajectory_comparison"), dict)
            and control["trajectory_comparison"].get("available") is False
        )
        candidate_matches = control.get("decoded_candidates_match_intact_by_effort")
        candidate_match_counts = control.get(
            "decoded_candidate_match_query_count_by_effort"
        )
        decoded_candidates_match = control.get("decoded_candidates_match_intact")
        candidate_summary_ok = (
            bool(
                isinstance(candidate_matches, dict)
                and set(candidate_matches)
                == {str(checkpoint) for checkpoint in CHECKPOINTS}
                and all(isinstance(value, bool) for value in candidate_matches.values())
                and isinstance(candidate_match_counts, dict)
                and set(candidate_match_counts) == set(candidate_matches)
                and all(
                    isinstance(value, Integral)
                    and not isinstance(value, bool)
                    and 0 <= int(value) <= applicable
                    for value in candidate_match_counts.values()
                )
                and all(
                    candidate_matches[key]
                    == (int(candidate_match_counts[key]) == applicable)
                    for key in candidate_matches
                )
                and isinstance(decoded_candidates_match, bool)
                and decoded_candidates_match == all(candidate_matches.values())
            )
            if applicable > 0
            else candidate_matches == {}
            and candidate_match_counts == {}
            and decoded_candidates_match is None
        )
        applicability_ok = (
            applicable == expected_query_count
            if name in ("repeat_intact", "no_context", "slot_ablation")
            else applicable > 0
        )
        return bool(
            total == query_count
            and total == expected_query_count
            and applicable >= 0
            and unavailable >= 0
            and applicable + unavailable == total
            and timing_matched == applicable
            and applicability_ok
            and metrics_ok
            and comparison_ok
            and candidate_summary_ok
        )

    required_controls_complete = all(
        control_complete(name)
        for name in (
            "repeat_intact",
            "no_context",
            "shuffled_demonstrations",
            "slot_ablation",
        )
    )
    truncation = controls.get("truncation", {}) if isinstance(controls, dict) else {}
    truncation_complete = bool(
        truncation.get("checkpoints") == list(CHECKPOINTS)
        and truncation.get("uses_one_continuous_intact_trajectory") is True
    )
    determinism = evaluation.get("determinism", {})
    required_numeric_states = {
        "compact_logits",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    }

    def numeric_evidence_complete(value: object, steps: Sequence[int]) -> bool:
        if not isinstance(value, dict):
            return False
        maximum_rms = value.get("maximum_rms")
        per_query_rms = value.get("per_query_maximum_rms")
        per_step_query_rms = value.get("per_step_query_rms")
        intact_dtypes = value.get("intact_dtype_by_state")
        candidate_dtypes = value.get("candidate_dtype_by_state")
        if not (
            isinstance(maximum_rms, dict)
            and set(maximum_rms) == required_numeric_states
            and isinstance(per_query_rms, dict)
            and set(per_query_rms) == required_numeric_states
            and isinstance(per_step_query_rms, dict)
            and set(per_step_query_rms) == required_numeric_states
            and isinstance(intact_dtypes, dict)
            and isinstance(candidate_dtypes, dict)
            and intact_dtypes == {name: "float32" for name in required_numeric_states}
            and candidate_dtypes == intact_dtypes
            and value.get("required_float32_dtypes") is True
        ):
            return False

        def tolerated(value: object) -> bool:
            return bool(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= STATE_RMS_TOLERANCE
            )

        def count_is(name: str, expected: int) -> bool:
            item = value.get(name)
            return bool(
                isinstance(item, Integral)
                and not isinstance(item, bool)
                and int(item) == expected
            )

        expected_steps = list(steps)
        per_query_ok = all(
            isinstance(values, list)
            and len(values) == expected_query_count
            and all(tolerated(item) for item in values)
            for values in per_query_rms.values()
        )
        per_step_query_ok = all(
            isinstance(rows, list)
            and len(rows) == len(expected_steps)
            and all(
                isinstance(row, list)
                and len(row) == expected_query_count
                and all(tolerated(item) for item in row)
                for row in rows
            )
            for rows in per_step_query_rms.values()
        )
        return bool(
            value.get("evaluated_steps") == expected_steps
            and count_is("query_count", expected_query_count)
            and value.get("within_declared_tolerance") is True
            and value.get("within_tolerance_by_query") == [True] * expected_query_count
            and count_is("within_tolerance_query_count", expected_query_count)
            and count_is("spike_hamming_count", 0)
            and value.get("spike_hamming_count_by_query") == [0] * expected_query_count
            and value.get("declared_per_query_axis_rms_tolerance")
            == STATE_RMS_TOLERANCE
            and all(tolerated(item) for item in maximum_rms.values())
            and per_query_ok
            and per_step_query_ok
        )

    repeat_control = (
        controls.get("repeat_intact", {}) if isinstance(controls, dict) else {}
    )
    repeat_candidate_counts = (
        repeat_control.get("decoded_candidate_match_query_count_by_effort", {})
        if isinstance(repeat_control, dict)
        else {}
    )
    repeat_candidate_flags = (
        repeat_control.get("decoded_candidates_match_intact_by_effort", {})
        if isinstance(repeat_control, dict)
        else {}
    )
    repeat_candidates_exact = bool(
        isinstance(repeat_candidate_counts, dict)
        and isinstance(repeat_candidate_flags, dict)
        and set(repeat_candidate_counts)
        == {str(checkpoint) for checkpoint in CHECKPOINTS}
        and set(repeat_candidate_flags) == set(repeat_candidate_counts)
        and all(
            isinstance(repeat_candidate_counts[key], Integral)
            and not isinstance(repeat_candidate_counts[key], bool)
            and int(repeat_candidate_counts[key]) == expected_query_count
            and repeat_candidate_flags[key] is True
            for key in repeat_candidate_counts
        )
        and repeat_control.get("decoded_candidates_match_intact") is True
    )
    repeat_metrics_exact = bool(
        isinstance(repeat_control, dict)
        and repeat_control.get("metrics_by_effort")
        == evaluation.get("metrics_by_effort")
    )
    repeat_numeric_exact = bool(
        isinstance(determinism, dict)
        and determinism.get("state_rms_tolerance") == STATE_RMS_TOLERANCE
        and determinism.get("spike_tolerance") == "exact identity"
        and isinstance(determinism.get("metric_absolute_tolerance"), Real)
        and not isinstance(determinism.get("metric_absolute_tolerance"), bool)
        and float(determinism["metric_absolute_tolerance"]) == 0.0
        and numeric_evidence_complete(
            determinism.get("repeat_intact_numeric_evidence"),
            range(max(CHECKPOINTS) + 1),
        )
    )
    repeatable = bool(
        isinstance(determinism, dict)
        and determinism.get("same_control_capable_execution_path") is True
        and determinism.get("repeat_intact_within_tolerance") is True
        and determinism.get("repeat_intact_metrics_exact") is True
        and determinism.get("repeat_intact_decoded_candidates_exact") is True
        and repeat_candidates_exact
        and repeat_metrics_exact
        and repeat_numeric_exact
    )
    slot_control = (
        controls.get("slot_ablation", {}) if isinstance(controls, dict) else {}
    )
    slot_candidate_counts = (
        slot_control.get("decoded_candidate_match_query_count_by_effort", {})
        if isinstance(slot_control, dict)
        else {}
    )
    slot_candidate_flags = (
        slot_control.get("decoded_candidates_match_intact_by_effort", {})
        if isinstance(slot_control, dict)
        else {}
    )
    slot_metrics = (
        slot_control.get("metrics_by_effort", {})
        if isinstance(slot_control, dict)
        else {}
    )
    intact_metrics = evaluation.get("metrics_by_effort")
    slot_checkpoint_zero_exact = bool(
        isinstance(slot_candidate_counts, dict)
        and isinstance(slot_candidate_flags, dict)
        and isinstance(slot_candidate_counts.get("0"), Integral)
        and not isinstance(slot_candidate_counts.get("0"), bool)
        and int(slot_candidate_counts["0"]) == expected_query_count
        and slot_candidate_flags.get("0") is True
        and isinstance(slot_metrics, dict)
        and isinstance(intact_metrics, dict)
        and slot_metrics.get("0") == intact_metrics.get("0")
    )
    slot_numeric_exact = bool(
        isinstance(determinism, dict)
        and numeric_evidence_complete(
            determinism.get("slot_ablation_checkpoint_zero_numeric_evidence"), (0,)
        )
    )
    ablation_matched = bool(
        isinstance(determinism, dict)
        and determinism.get("slot_ablation_checkpoint_zero_state_within_tolerance")
        is True
        and determinism.get("slot_ablation_checkpoint_zero_decoded_candidates_exact")
        is True
        and determinism.get("slot_ablation_checkpoint_zero_metrics_exact") is True
        and determinism.get("slot_ablation_checkpoint_zero_within_tolerance") is True
        and slot_checkpoint_zero_exact
        and slot_numeric_exact
    )
    evaluation_complete = bool(
        evaluation.get("primary_candidate_mode") == "model_only"
        and query_count > 0
        and query_count == expected_query_count
        and int(evaluation.get("task_count", 0)) == expected_task_count
        and metrics_complete(evaluation.get("metrics_by_effort"))
        and aggregate_complete
        and query_trajectories_complete
        and checkpoint_queries_complete
        and required_controls_complete
        and truncation_complete
        and finite_tree(evaluation)
    )

    compiler_report = training.get("compiler_report", {})
    compiler_counts = (
        compiler_report.get("counts", {}) if isinstance(compiler_report, dict) else {}
    )
    routed_paths = (
        {
            item.get("parameter")
            for item in compiler_report.get("etrace_weights", ())
            if isinstance(item, dict)
        }
        if isinstance(compiler_report, dict)
        else set()
    )
    plain_paths = (
        {
            item.get("parameter")
            for item in compiler_report.get("excluded_weights", ())
            if isinstance(item, dict)
        }
        if isinstance(compiler_report, dict)
        else set()
    )
    legacy_temporal_paths = {
        "ff_syn.comm.weight",
        "rec_syn.comm.weight",
    }
    plain_paths_expected = {
        "color_factor_head.weight",
        "height_head.weight",
        "readout_projection.weight",
        "width_head.weight",
    }
    associative_paths = {
        "memory_write_scale",
        "workspace_query_projection.weight",
        "memory_read_projection.weight",
    }
    memory_enabled = config.context_memory_width > 0
    routed_paths_expected = legacy_temporal_paths | (
        associative_paths if memory_enabled else set()
    )
    expected_parameter_paths = routed_paths_expected | plain_paths_expected
    route_classifications: dict[object, set[object]] = {}
    for item in (
        compiler_report.get("diagnostics", ())
        if isinstance(compiler_report, dict)
        else ()
    ):
        if not isinstance(item, dict) or item.get("kind") != "relation_included":
            continue
        classifications = item.get("path_classification_by_hidden_state")
        if not isinstance(classifications, dict) or not classifications:
            continue
        route_classifications.setdefault(item.get("weight_path"), set()).update(
            classifications.values()
        )
    associative_routes_direct = bool(
        not memory_enabled
        or all(
            route_classifications.get(path) == {"all_direct"}
            for path in associative_paths
        )
    )
    associative_diagnostics = evaluation.get("associative_memory_diagnostics")
    associative_diagnostics_complete = bool(
        not memory_enabled
        or (
            isinstance(associative_diagnostics, dict)
            and associative_diagnostics.get("available") is True
            and associative_diagnostics.get("complete") is True
            and associative_diagnostics.get("repeat_intact_exact") is True
            and associative_diagnostics.get("no_context_memory_exactly_zero") is True
            and associative_diagnostics.get(
                "shuffled_pairing_sensitive_for_every_applicable_query"
            )
            is True
            and int(associative_diagnostics.get("query_count", 0)) == query_count
            and int(associative_diagnostics.get("depth_count", 0))
            == max(CHECKPOINTS) + 1
        )
    )
    compiler_complete = bool(
        training.get("pp_prop_compiled") is True
        and isinstance(compiler_report, dict)
        and compiler_report.get("available") is True
        and int(compiler_counts.get("hidden_groups", 0)) >= 1
        and int(compiler_counts.get("errors", -1)) == 0
        and routed_paths == routed_paths_expected
        and plain_paths == plain_paths_expected
        and routed_paths | plain_paths == expected_parameter_paths
        and associative_routes_direct
    )
    full_scale = bool(
        model_report.get("neuron_count") == 2048
        and model_report.get("recurrent_edge_count") == 16384
        and model_report.get("slot_count") == 32
        and int(model_report.get("parameter_count", 0)) > 0
    )
    component_types = model_report.get("component_types", {})
    component_contract = bool(
        isinstance(component_types, dict)
        and component_types
        == {
            "neuron": "LIF",
            "feedforward_projection_wrapper": "AlignPostProj",
            "feedforward_projection": "Linear",
            "feedforward_synapse": "Expon",
            "feedforward_output": "CUBA",
            "recurrent_projection_wrapper": "AlignPostProj",
            "recurrent_projection": "SparseLinear",
            "recurrent_synapse": "Expon",
            "recurrent_output": "CUBA",
        }
    )
    gpu_complete = str(device_report.get("platform", "")).casefold() == "gpu"
    frozen = evaluation.get("same_frozen_parameter_bytes") is True
    structural_checks = {
        "actual_full_scale": full_scale,
        "physical_component_contract": component_contract,
        "actual_gpu_backend": gpu_complete,
        "pp_prop_compiler_routes": compiler_complete,
        "associative_routes_all_direct": associative_routes_direct,
        "associative_diagnostics_complete": associative_diagnostics_complete,
        "complete_frozen_evaluation": evaluation_complete,
        "frozen_parameters_unchanged": frozen,
        "repeat_intact_deterministic": repeatable,
        "slot_ablation_pre_intervention_matched": ablation_matched,
    }
    structural = all(structural_checks.values())

    training_counts = training.get("optimizer_updates_by_effort", {})
    mixed = bool(
        isinstance(training_counts, dict)
        and all(
            int(training_counts.get(str(effort), 0)) > 0 for effort in TRAINING_EFFORTS
        )
        and sum(int(training_counts.get(str(effort), 0)) for effort in TRAINING_EFFORTS)
        == config.training_updates
    )
    losses = training.get("losses")
    losses_complete = bool(
        isinstance(losses, list)
        and len(losses) == config.training_updates
        and finite_tree(losses)
    )
    parameter_changes = training.get("parameter_changes")
    temporal_paths_moved = bool(
        isinstance(parameter_changes, dict)
        and all(
            isinstance(parameter_changes.get(path), dict)
            and parameter_changes[path].get("changed") is True
            and float(parameter_changes[path].get("l2_delta", 0.0)) > 0.0
            and math.isfinite(float(parameter_changes[path]["l2_delta"]))
            for path in ("ff_syn.comm.weight", "rec_syn.comm.weight")
        )
    )
    all_parameter_changes_finite = bool(
        isinstance(parameter_changes, dict)
        and set(parameter_changes) == expected_parameter_paths
        and all(
            isinstance(item, dict)
            and item.get("changed") is True
            and float(item.get("l2_delta", 0.0)) > 0.0
            and math.isfinite(float(item["l2_delta"]))
            for item in parameter_changes.values()
        )
    )
    sources = [item.manifest.source for item in data.loaded]
    training_names = {
        str(source.name).casefold() for source in sources if source.role == "train"
    }
    evaluation_names = {
        str(source.name).casefold() for source in sources if source.role == "evaluation"
    }
    approved_sources = bool(
        training_names
        and evaluation_names
        and training_names <= APPROVED_TRAINING_SOURCES
        and evaluation_names <= APPROVED_EVALUATION_SOURCES
    )
    no_rejected_sources = all(
        len(getattr(item.manifest, "rejected", ())) == 0 for item in data.loaded
    )
    depth_supervision = bool(
        training.get("supervised_depths") == "0..effort"
        and training.get("depth_weighting") == "uniform_unit_sum_per_update"
        and training.get("per_update_depth_weight_sum", 1.0) == 1.0
    )
    if "supervised_depths" not in training:
        depth_supervision = training.get("terminal_supervision_only") is True
    associative_capability_status = (
        "associative_capability_gates_pending"
        if memory_enabled
        else "not_applicable_legacy"
    )
    scientific_checks = {
        "structural_qualification": structural,
        "not_smoke_or_structural_only": not config.smoke and not config.structural_only,
        "complete_evaluation_split": config.evaluation_task_limit is None,
        "approved_train_and_evaluation_sources": approved_sources,
        "no_rejected_source_records": no_rejected_sources,
        "not_plumbing_only": not data.plumbing_only,
        "one_model_one_optimizer_depth_supervision": bool(
            training.get("performed") is True
            and training.get("one_shared_model") is True
            and training.get("one_shared_optimizer_state") is True
            and depth_supervision
        ),
        "mixed_effort_update_schedule": mixed,
        "finite_loss_per_update": losses_complete,
        "parameters_moved": training.get("parameters_moved") is True,
        "temporal_synapses_moved": temporal_paths_moved,
        "all_parameter_groups_moved_with_finite_delta": all_parameter_changes_finite,
        "associative_capability_gates_complete": not memory_enabled,
    }
    scientific = all(scientific_checks.values())
    structural_messages = {
        "actual_full_scale": "actual model is not the required 2048-neuron/16384-edge scale",
        "physical_component_contract": "actual neuron, projection, synapse, or current-output component types do not match the declared substrate",
        "actual_gpu_backend": "actual evaluation backend is not GPU",
        "pp_prop_compiler_routes": "pp-prop compilation or feedforward/recurrent eligibility routing evidence is incomplete",
        "associative_routes_all_direct": "associative pp-prop routes are not all_direct",
        "associative_diagnostics_complete": "pairing-sensitive S_K, memory-read, or continuous-workspace diagnostics are incomplete",
        "complete_frozen_evaluation": "exact metrics, trajectories, or controls are incomplete or non-finite",
        "frozen_parameters_unchanged": "evaluation mutated frozen parameter bytes",
        "repeat_intact_deterministic": "same-run intact repeat exceeded the declared state/logit tolerance or changed exact candidates or metrics",
        "slot_ablation_pre_intervention_matched": "slot-ablation checkpoint zero exceeded the declared state/logit tolerance or changed exact candidates or effort-0 metrics",
    }
    scientific_messages = {
        "not_smoke_or_structural_only": "smoke fixtures or disabled optimization cannot be scientific evidence",
        "complete_evaluation_split": "evaluation_task_limit makes this a development subset",
        "approved_train_and_evaluation_sources": "approved train/evaluation source roles were not both present",
        "no_rejected_source_records": "source rejections were present",
        "not_plumbing_only": "embedded fixtures are plumbing-only",
        "one_model_one_optimizer_depth_supervision": "training did not retain one shared model, optimizer state, and normalized supervision at every depth 0..effort",
        "mixed_effort_update_schedule": "one shared model did not receive the complete 8/16/32 update schedule",
        "finite_loss_per_update": "one finite loss was not retained for every optimizer update",
        "parameters_moved": "training did not change parameter bytes",
        "temporal_synapses_moved": "feedforward and recurrent eligibility-routed synapses did not both move",
        "all_parameter_groups_moved_with_finite_delta": "not every parameter group moved with a finite delta",
        "associative_capability_gates_complete": "associative_capability_gates_pending",
    }
    reasons_not_structural = [
        structural_messages[name]
        for name, passed in structural_checks.items()
        if not passed
    ]
    reasons_not_scientific = list(reasons_not_structural)
    reasons_not_scientific.extend(
        scientific_messages[name]
        for name, passed in scientific_checks.items()
        if name != "structural_qualification" and not passed
    )
    return {
        "full_structural_qualification": structural,
        "full_scientific_qualification": scientific,
        "plumbing_only": data.plumbing_only,
        "associative_capability_status": associative_capability_status,
        "structural_checks": structural_checks,
        "scientific_checks": scientific_checks,
        "reasons_not_structural": reasons_not_structural,
        "reasons_not_scientific": reasons_not_scientific,
    }


def _parameter_count(values: dict[str, Any]) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(values)))


def _software_report() -> dict[str, object]:
    distributions = (
        "braintrace",
        "brainstate",
        "brainpy",
        "braintools",
        "jax",
        "jaxlib",
        "numpy",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
    }


def _implementation_report() -> dict[str, object]:
    directory = pathlib.Path(__file__).resolve().parent
    names = (
        pathlib.Path(__file__).name,
        "latent_workspace_task.py",
        "latent_workspace_analysis.py",
        "latent_workspace_model.py",
    )
    combined = hashlib.sha256()
    files: dict[str, str] = {}
    for name in names:
        payload = (directory / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[name] = digest
        combined.update(name.encode("utf-8"))
        combined.update(payload)
    return {
        "source_tree_sha256": combined.hexdigest(),
        "file_sha256": files,
        "source_revision": os.environ.get("EXAMPLE21_SOURCE_REVISION"),
        "source_dirty": os.environ.get("EXAMPLE21_SOURCE_DIRTY"),
    }


def _data_summary(
    data: _ExperimentData,
    manifests: Sequence[dict[str, object]],
    evaluation: dict[str, object],
) -> dict[str, object]:
    task_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    source_names: dict[str, list[str]] = {}
    for item in data.loaded:
        role = str(item.manifest.source.role)
        task_counts[role] += len(item.tasks)
        query_counts[role] += sum(len(task.test) for task in item.tasks)
        source_names.setdefault(role, []).append(str(item.manifest.source.name))
    canonical = json.dumps(
        list(manifests), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return {
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_count": len(manifests),
        "source_names_by_role": {
            role: sorted(names) for role, names in source_names.items()
        },
        "task_counts_by_role": dict(sorted(task_counts.items())),
        "query_counts_by_role": dict(sorted(query_counts.items())),
        "training_task_pool_count": len(data.training),
        "evaluated_task_count": int(evaluation["task_count"]),
        "evaluated_query_count": int(evaluation["query_count"]),
        "parsed_task_count": int(
            sum(int(manifest["parsed_task_count"]) for manifest in manifests)
        ),
        "valid_task_count": int(
            sum(int(manifest["valid_task_count"]) for manifest in manifests)
        ),
        "rejected_task_count": int(
            sum(int(manifest["rejected_task_count"]) for manifest in manifests)
        ),
        "duplicate_task_count": int(
            sum(int(manifest["duplicate_task_count"]) for manifest in manifests)
        ),
        "excluded_task_count": int(
            sum(int(manifest["excluded_task_count"]) for manifest in manifests)
        ),
        "split_overlap_check": "passed",
        "private_paper_data_available": False,
        "private_training_recipe_available": False,
    }


def _channel_attribution(
    checkpoint_queries: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, object]]:
    """Summarize the fail-closed model-only primary candidate channel.

    Parameters
    ----------
    checkpoint_queries
        Per-effort query records as written to ``result.json``.

    Returns
    -------
    dict
        One summary per effort, keyed by the effort as a string.
    """

    summary: dict[str, dict[str, object]] = {}
    for effort, details in checkpoint_queries.items():
        candidates = [
            candidate
            for item in details
            for candidate in item.get("candidates", ())
            if isinstance(candidate, Mapping)
        ]
        if any(candidate.get("provenance") != "model" for candidate in candidates):
            raise ValueError("primary attribution found non-model candidate provenance")
        pass_at_2 = sum(1 for item in details if item["score"]["pass_at_2"])
        summary[effort] = {
            "primary_candidate_mode": "model_only",
            "query_count": len(details),
            "submitted_model_candidate_count": len(candidates),
            "exact_by_model_candidates": pass_at_2,
            "exact_total": pass_at_2,
        }
    return summary


def _render_report(result: dict[str, object]) -> str:
    configuration = result.get("configuration", {})
    device = result.get("device", {})
    model = result.get("model", {})
    training = result.get("training", {})
    evaluation = result.get("evaluation", {})
    associative_diagnostics = evaluation.get("associative_memory_diagnostics", {})
    qualification = result.get("qualification", {})
    data_summary = result.get("data_summary", {})
    software = result.get("software", {})
    implementation = result.get("implementation", {})
    compiler_report = training.get("compiler_report", {"counts": {}, "diagnostics": []})
    compiler_counts = compiler_report.get("counts", {})
    runtime = float(result.get("runtime_seconds", 0.0))
    if training.get("performed") is True:
        training_line = (
            "Training: one parameter set and one Adam state; normalized uniform "
            "supervision at every depth 0..R; updates by maximum depth "
            f"{training.get('optimizer_updates_by_effort', {})}."
        )
    else:
        training_line = (
            "Training: optimization was not performed; reason="
            f"{training.get('reason', 'unreported')}; updates="
            f"{training.get('optimizer_updates_by_effort', {})}."
        )
    if training.get("performed") is True:
        plain_route_line = (
            "The plain routes received exact current-window gradients in this run; "
            "they do not carry temporal eligibility."
        )
    else:
        plain_route_line = (
            "Optimization was disabled, so no routes were trained in this run; when "
            "enabled, the plain routes receive exact current-window gradients and do "
            "not carry temporal eligibility."
        )
    lines = [
        "Example 21 - ARC latent reasoning with pp-prop",
        "",
        str(result.get("claim_boundary", CLAIM_BOUNDARY)),
        "",
        f"Seed: {configuration.get('seed', 'unreported')}",
        f"Runtime: {runtime:.3f} seconds",
        (
            f"Device: requested={device.get('requested', 'unreported')}, "
            f"actual={device.get('platform', 'unreported')} "
            f"({device.get('kind', 'unreported')})"
        ),
        (
            f"Device memory after training/evaluation: {device.get('memory_stats', {})}; "
            f"capture={device.get('memory_stats_capture', 'unreported')}."
        ),
        (
            "Implementation: source-tree SHA-256="
            f"{implementation.get('source_tree_sha256', 'unreported')}; "
            f"revision={implementation.get('source_revision') or 'unreported'}; "
            f"source_dirty={implementation.get('source_dirty') or 'unreported'}."
        ),
        (
            f"Software: Python {software.get('python', 'unreported')}; packages "
            f"{software.get('packages', {})}."
        ),
        f"Configuration: {configuration}",
        "",
        (
            f"Physical model: {model.get('neuron_count', 'unreported')} LIF neurons, "
            f"{model.get('recurrent_edge_count', 'unreported')} directed sparse edges, "
            f"{model.get('slot_count', 'unreported')} x 64-neuron analysis slots, "
            f"{model.get('parameter_count', 'unreported')} scalar parameters."
        ),
        (
            "Reasoning memory: mode="
            f"{model.get('reasoning_mode', 'unreported')}; width="
            f"{model.get('context_memory_width', 'unreported')}; decay="
            f"{model.get('memory_decay', 'unreported')}; raw key/value widths="
            f"{model.get('raw_key_feature_width', 'unreported')}/"
            f"{model.get('raw_value_feature_width', 'unreported')}; dense S bytes "
            "per-example/training-batch/evaluation-batch="
            f"{model.get('context_memory_bytes_per_example', 'unreported')}/"
            f"{model.get('context_memory_bytes_training_batch', 'unreported')}/"
            f"{model.get('context_memory_bytes_evaluation_batch', 'unreported')} bytes."
        ),
        (
            "Associative memory implementation: "
            f"{model.get('associative_memory_implementation', {})}."
        ),
        f"Physical component types: {model.get('component_types', {})}.",
        (
            f"Data manifest SHA-256: {data_summary.get('manifest_sha256', 'unreported')}; "
            f"sources={data_summary.get('source_names_by_role', {})}."
        ),
        (
            f"Splits: tasks={data_summary.get('task_counts_by_role', {})}; "
            f"queries={data_summary.get('query_counts_by_role', {})}; "
            f"evaluated={data_summary.get('evaluated_task_count', 'unreported')} tasks/"
            f"{data_summary.get('evaluated_query_count', 'unreported')} queries; "
            f"rejected={data_summary.get('rejected_task_count', 'unreported')}; "
            f"duplicates={data_summary.get('duplicate_task_count', 'unreported')}; "
            f"explicit exclusions={data_summary.get('excluded_task_count', 'unreported')}."
        ),
        training_line,
        (
            f"Training exposure: {training.get('sampled_base_task_count', 'unreported')} "
            "unique base tasks and "
            f"{training.get('sampled_base_fold_count', 'unreported')} unique "
            "leave-one-demonstration-out folds sampled with replacement="
            f"{training.get('sampling_with_replacement', 'unreported')} "
            f"from a {data_summary.get('training_task_pool_count', 'unreported')}-task pool."
        ),
        (
            f"Training movement: parameter bytes changed={training.get('parameters_moved', False)}; "
            f"per-group evidence={training.get('parameter_changes', {})}."
        ),
        (
            "Compiler: eligibility-trace temporal routes="
            f"{compiler_counts.get('etrace_weights', 0)} "
            f"({[item.get('parameter') for item in compiler_report.get('etrace_weights', [])]}), "
            "plain exact current-window reverse-mode routes="
            f"{compiler_counts.get('excluded_weights', 0)} "
            f"({[item.get('parameter') for item in compiler_report.get('excluded_weights', [])]}), "
            f"{compiler_counts.get('warnings', 0)} "
            "warnings, "
            f"{compiler_counts.get('errors', 0)} errors. {plain_route_line}"
        ),
        f"Evaluation execution: {evaluation.get('execution', {})}.",
        (
            "Associative evaluation diagnostics: "
            f"available={associative_diagnostics.get('available')}; "
            f"complete={associative_diagnostics.get('complete')}; "
            "repeat exact="
            f"{associative_diagnostics.get('repeat_intact_exact')}; "
            "no-context S zero="
            f"{associative_diagnostics.get('no_context_memory_exactly_zero')}; "
            "shuffled pairing-sensitive for every applicable query="
            f"{associative_diagnostics.get('shuffled_pairing_sensitive_for_every_applicable_query')}."
        ),
        "",
        "Frozen exact ARC results:",
    ]
    for effort in CHECKPOINTS:
        metrics = evaluation.get("metrics_by_effort", {}).get(str(effort))
        if metrics is None:
            lines.append(f"  effort {effort:>2}: unavailable")
            continue
        lines.append(
            f"  effort {effort:>2}: query pass@1={metrics['query_pass_at_1']:.4f}, "
            f"pass@2={metrics['query_pass_at_2']:.4f}; strict task pass@1="
            f"{metrics['strict_task_pass_at_1']:.4f}, pass@2="
            f"{metrics['strict_task_pass_at_2']:.4f}; shape diagnostic="
            f"{metrics['shape_accuracy_diagnostic']:.4f}, pixel diagnostic="
            f"{metrics['valid_cell_pixel_accuracy_diagnostic']:.4f}"
        )
    attribution = evaluation.get("channel_attribution", {})
    for effort in CHECKPOINTS:
        split = attribution.get(str(effort))
        if split is None:
            continue
        lines.append(
            f"  effort {effort:>2} primary channel: model_only; exact="
            f"{split['exact_by_model_candidates']} of {split['query_count']} queries; "
            "submitted model candidates="
            f"{split['submitted_model_candidate_count']}."
        )
    intact_metrics = evaluation.get("metrics_by_effort", {})
    if "0" in intact_metrics and "32" in intact_metrics:
        effort_zero = intact_metrics["0"]["query_pass_at_2"]
        effort_32 = intact_metrics["32"]["query_pass_at_2"]
        exact_count_32 = round(effort_32 * int(intact_metrics["32"]["query_count"]))
        direction = (
            "improved"
            if effort_32 > effort_zero
            else "worsened"
            if effort_32 < effort_zero
            else "tied"
        )
        lines.append(
            f"  Empirical outcome: effort 32 {direction} effort 0 on exact pass@2 "
            f"({effort_32:.4f} versus {effort_zero:.4f}); "
            f"{exact_count_32} effort-32 queries were exact within the scored set."
        )
    lines.extend(["", "Aggregate latent trajectory:"])
    trajectory = evaluation.get("aggregate_trajectory", [])
    for effort in CHECKPOINTS:
        if effort >= len(trajectory):
            lines.append(f"  step {effort:>2}: unavailable")
            continue
        row = trajectory[effort]
        lines.append(
            f"  step {effort:>2}: firing={row.get('mean_firing_rate', math.nan):.6f}; "
            f"Voltage L2={row.get('mean_voltage_l2', math.nan):.6f}; "
            f"feedforward-current L2={row.get('mean_feedforward_current_l2', math.nan):.6f}; "
            f"recurrent-current L2={row.get('mean_recurrent_current_l2', math.nan):.6f}; "
            f"entropy={row.get('mean_predictive_entropy', math.nan):.6f}; "
            f"changed-cell fraction={row.get('mean_changed_cell_fraction')}; "
            f"converged/silent/saturated={row.get('converged_fraction')}/"
            f"{row.get('near_silence_fraction')}/{row.get('near_saturation_fraction')}; "
            f"raw-byte state hashes={row.get('unique_state_hashes')}."
        )
        lines.append(
            f"           deterministic pair sample n={row.get('pair_sample_count', 0)}; "
            f"spike-Hamming={row.get('pairwise_spike_hamming_fraction')}; "
            f"voltage RMS={row.get('pairwise_voltage_rms_distance')}; "
            f"feedforward/recurrent current RMS="
            f"{row.get('pairwise_feedforward_current_rms_distance')}/"
            f"{row.get('pairwise_recurrent_current_rms_distance')}."
        )
    lines.extend(
        [
            "  Raw-byte hash counts report collisions only; pairwise distances, not hash uniqueness, test geometry.",
            "",
            "Frozen controls and deterministic repeat:",
        ]
    )
    controls = evaluation.get("controls", {})
    for name in (
        "repeat_intact",
        "no_context",
        "shuffled_demonstrations",
        "slot_ablation",
    ):
        control = controls.get(name)
        if not isinstance(control, dict):
            lines.append(f"  {name}: unavailable")
            continue
        applicable = int(control.get("applicable_query_count", 0))
        unavailable = int(control.get("unavailable_query_count", 0))
        comparison = control.get("trajectory_comparison", {})
        lines.append(
            f"  {name}: applicable={applicable}/{control.get('query_count', 0)}, "
            f"unavailable={unavailable}, timing-matched="
            f"{control.get('timing_matched_applicable_query_count', 0)}/{applicable}; "
            f"causally_null={comparison.get('causally_null_at_measured_precision')}; "
            f"null_queries={control.get('causally_null_query_count', 0)}/{applicable}; "
            f"byte-identical queries={control.get('byte_identical_query_count', 0)}/{applicable}."
        )
        if applicable:
            control_metrics = control.get("metrics_by_effort", {})
            for effort in CHECKPOINTS:
                row = control_metrics.get(str(effort))
                if row is None:
                    lines.append(f"    effort {effort:>2}: unavailable")
                    continue
                lines.append(
                    f"    effort {effort:>2}: pass@1={row['query_pass_at_1']:.4f}; "
                    f"pass@2={row['query_pass_at_2']:.4f}; shape="
                    f"{row['shape_accuracy_diagnostic']:.4f}; pixels="
                    f"{row['valid_cell_pixel_accuracy_diagnostic']:.4f}."
                )
            lines.append(
                "    state comparison: "
                f"{comparison.get('interpretation', 'unreported')}; "
                "aggregate score deltas control-minus-intact="
                f"{dict((key, value) for key, value in comparison.get('score_deltas_control_minus_intact', {}).items() if '.tasks.' not in key)}."
            )
            current_l2 = comparison.get("synaptic_current_l2_by_step", {})
            spike_fraction = comparison.get("spike_hamming_fraction_by_step", [])
            voltage_l2 = comparison.get("voltage_l2_by_step", [])
            feedforward_l2 = current_l2.get("feedforward", [])
            recurrent_l2 = current_l2.get("recurrent", [])
            if all(
                len(values) > max(CHECKPOINTS)
                for values in (
                    spike_fraction,
                    voltage_l2,
                    feedforward_l2,
                    recurrent_l2,
                )
            ):
                lines.append(
                    "    step-32 state deltas: spike-Hamming fraction="
                    f"{spike_fraction[max(CHECKPOINTS)]:.6f}; voltage L2="
                    f"{voltage_l2[max(CHECKPOINTS)]:.6f}; feedforward/recurrent "
                    f"current L2={feedforward_l2[max(CHECKPOINTS)]:.6f}/"
                    f"{recurrent_l2[max(CHECKPOINTS)]:.6f}."
                )
    determinism = evaluation.get("determinism", {})
    repeat_numeric = determinism.get("repeat_intact_numeric_evidence", {})
    ablation_numeric = determinism.get(
        "slot_ablation_checkpoint_zero_numeric_evidence", {}
    )

    def numeric_noise_line(label: str, evidence: object) -> str:
        if not isinstance(evidence, dict):
            return f"{label} numeric noise: unavailable."

        def formatted_values(value: object) -> dict[str, str]:
            if not isinstance(value, dict):
                return {}
            return {
                str(name): f"{float(number):.3e}"
                for name, number in value.items()
                if isinstance(number, Real) and not isinstance(number, bool)
            }

        steps = evidence.get("evaluated_steps")
        step_count = len(steps) if isinstance(steps, list) else "unreported"
        return (
            f"{label} numeric noise: queries={evidence.get('query_count', 'unreported')}; "
            f"steps={step_count}; spike mismatches="
            f"{evidence.get('spike_hamming_count', 'unreported')}; maximum RMS="
            f"{formatted_values(evidence.get('maximum_rms'))}; maximum absolute="
            f"{formatted_values(evidence.get('maximum_absolute'))}; dtypes="
            f"{evidence.get('intact_dtype_by_state', 'unreported')}; within tolerance="
            f"{evidence.get('within_declared_tolerance', 'unreported')}."
        )

    lines.extend(
        [
            "",
            (
                "Determinism gate: repeat intact byte-identical="
                f"{determinism.get('repeat_intact_state_byte_identical')}; "
                "compact logits byte-identical="
                f"{determinism.get('repeat_intact_compact_logits_byte_identical')}; "
                f"within tolerance={determinism.get('repeat_intact_within_tolerance')}; "
                f"decoded candidates exact={determinism.get('repeat_intact_decoded_candidates_exact')}; "
                f"metrics exact={determinism.get('repeat_intact_metrics_exact')}; "
                "slot-ablation checkpoint 0 matched="
                f"{determinism.get('slot_ablation_checkpoint_zero_within_tolerance')} "
                f"(byte-identical={determinism.get('slot_ablation_checkpoint_zero_byte_identical')}; "
                "state/logits within tolerance="
                f"{determinism.get('slot_ablation_checkpoint_zero_state_within_tolerance')}; "
                "decoded candidates exact="
                f"{determinism.get('slot_ablation_checkpoint_zero_decoded_candidates_exact')}; "
                "effort-0 metrics exact="
                f"{determinism.get('slot_ablation_checkpoint_zero_metrics_exact')}); "
                "per-query RMS tolerance (feature axis for logits; neuron axis "
                "for physical state)="
                f"{determinism.get('state_rms_tolerance', 'unreported')}; "
                f"metric absolute tolerance={determinism.get('metric_absolute_tolerance', 'unreported')}."
            ),
            numeric_noise_line("Repeat", repeat_numeric),
            numeric_noise_line("Ablation checkpoint-0", ablation_numeric),
        ]
    )
    compiler_warnings = [
        item
        for item in compiler_report.get("diagnostics", [])
        if item.get("level") == "warning"
    ]
    if compiler_warnings:
        lines.extend(["", "Compiler warnings (retained, not hidden):"])
        for item in compiler_warnings:
            lines.append(f"  - {item['message']}")
    lines.extend(
        [
            "",
            (
                "Qualification: structural="
                f"{qualification.get('full_structural_qualification', False)}, "
                "scientific="
                f"{qualification.get('full_scientific_qualification', False)}."
            ),
            f"Structural checks: {qualification.get('structural_checks', {})}.",
            f"Scientific checks: {qualification.get('scientific_checks', {})}.",
        ]
    )
    for reason in qualification.get("reasons_not_scientific", []):
        lines.append(f"  - {reason}")
    if qualification.get("full_scientific_qualification", False):
        interpretation = (
            "This run satisfies the declared full scientific protocol gates. It is "
            "not evidence of converged ARC training and not an architecture falsification."
        )
    elif qualification.get("full_structural_qualification", False):
        interpretation = (
            "This run satisfies the full structural protocol gates only; it is not "
            "scientific model-quality evidence."
        )
    else:
        interpretation = (
            "This artifact does not satisfy the full structural or scientific "
            "qualification gates."
        )
    lines.extend(["", f"Interpretation boundary: {interpretation}"])
    return "\n".join(lines) + "\n"


def _plot(result: dict[str, object], path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = result["evaluation"]["metrics_by_effort"]
    trajectory = result["evaluation"]["aggregate_trajectory"]
    efforts = np.asarray(CHECKPOINTS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(
        efforts,
        [metrics[str(value)]["strict_task_pass_at_1"] for value in efforts],
        marker="o",
        label="strict pass@1",
    )
    axes[0, 0].plot(
        efforts,
        [metrics[str(value)]["strict_task_pass_at_2"] for value in efforts],
        marker="o",
        label="strict pass@2",
    )
    axes[0, 0].set(title="Exact ARC quality", xlabel="latent steps", ylabel="rate")
    axes[0, 0].legend()
    steps = [row["step"] for row in trajectory]
    changed = [
        np.nan
        if row["mean_changed_cell_fraction"] is None
        else row["mean_changed_cell_fraction"]
        for row in trajectory
    ]
    axes[0, 1].plot(steps, changed, color="tab:blue", label="changed cells")
    axes[0, 1].set(
        title="Per-step output dynamics",
        xlabel="latent step",
        ylabel="changed-cell fraction",
    )
    entropy_axis = axes[0, 1].twinx()
    entropy_axis.plot(
        steps,
        [row["mean_predictive_entropy"] for row in trajectory],
        color="tab:orange",
        label="predictive entropy",
    )
    entropy_axis.set_ylabel("predictive entropy")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    entropy_handles, entropy_labels = entropy_axis.get_legend_handles_labels()
    axes[0, 1].legend(handles + entropy_handles, labels + entropy_labels)
    axes[1, 0].plot(
        steps,
        [row["mean_firing_rate"] for row in trajectory],
        color="tab:blue",
        label="firing rate",
    )
    axes[1, 0].plot(
        steps,
        [row["near_saturation_fraction"] for row in trajectory],
        color="tab:green",
        linestyle="--",
        label="saturated queries",
    )
    axes[1, 0].set(
        title="Spike and voltage dynamics",
        xlabel="latent step",
        ylabel="spike-derived fraction",
    )
    voltage_axis = axes[1, 0].twinx()
    voltage_axis.plot(
        steps,
        [row["mean_voltage_l2"] for row in trajectory],
        color="tab:red",
        label="voltage L2",
    )
    voltage_axis.set_ylabel("voltage L2")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    voltage_handles, voltage_labels = voltage_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles + voltage_handles, labels + voltage_labels)
    controls = result["evaluation"]["controls"]
    names = ["no_context", "shuffled_demonstrations", "slot_ablation"]
    exact_deltas = []
    diagnostic_deltas = []
    state_effects = []
    for name in names:
        comparison = controls.get(name, {}).get("trajectory_comparison", {})
        score_deltas = comparison.get("score_deltas_control_minus_intact", {})
        if "32.query_pass_at_2" not in score_deltas:
            exact_deltas.append(np.nan)
            diagnostic_deltas.append(np.nan)
            state_effects.append(np.nan)
            continue
        exact_deltas.append(score_deltas["32.query_pass_at_2"])
        diagnostic_deltas.append(
            score_deltas["32.valid_cell_pixel_accuracy_diagnostic"]
        )
        state_effects.append(
            comparison["spike_hamming_fraction_by_step"][max(CHECKPOINTS)]
        )
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.36
    axes[1, 1].bar(
        positions - width / 2, exact_deltas, width=width, label="pass@2 delta"
    )
    axes[1, 1].bar(
        positions + width / 2,
        diagnostic_deltas,
        width=width,
        label="pixel diagnostic delta",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        title="Control deltas at effort 32",
        ylabel="control minus intact",
        xticks=positions,
        xticklabels=names,
    )
    axes[1, 1].tick_params(axis="x", rotation=15)
    state_axis = axes[1, 1].twinx()
    state_axis.plot(
        positions,
        state_effects,
        color="black",
        marker="D",
        linestyle="none",
        label="spike-Hamming fraction",
    )
    state_axis.set_ylabel("state effect at step 32")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    state_handles, state_labels = state_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + state_handles, labels + state_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run training, frozen interventions, and evidence generation.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.

    Returns
    -------
    dict
        JSON-safe complete experiment evidence.
    """
    started = time.perf_counter()
    device, device_report = _resolve_device(config.device)
    data = _load_data(config)
    rows = _row_config(config)
    model = _make_model(config, rows, batch_size=1, device=device)
    training = _train_model(model, _training_chunks(data, config, rows), config)
    evaluation = _evaluate(model, data, config, rows, device)
    device_report["memory_stats"] = _device_memory_stats(device)
    device_report["memory_stats_capture"] = "after training and evaluation"
    manifests = [item.manifest.to_dict() for item in data.loaded]
    memory_architecture = _memory_architecture_report(
        config,
        rows,
        training_batch_size=model.config.batch_size,
        evaluation_batch_size=int(evaluation["query_count"]),
    )
    memory_implementation = _model_memory_report(model)
    memory_contract = {
        "mode": "reasoning_mode",
        "memory_width": "context_memory_width",
        "key_feature_width": "raw_key_feature_width",
        "value_feature_width": "raw_value_feature_width",
    }
    for implementation_name, architecture_name in memory_contract.items():
        if memory_implementation.get(implementation_name) != memory_architecture.get(
            architecture_name
        ):
            raise ValueError(
                "model and experiment associative-memory reports disagree on "
                f"{implementation_name}"
            )
    model_report = {
        "neuron_count": model.neuron_count,
        "recurrent_edge_count": model.recurrent_edge_count,
        "slot_count": model.slot_count,
        "neurons_per_slot": 64,
        "input_width": rows.input_width,
        "compact_output_width": model.config.compact_output_width,
        "color_rank": model.config.color_rank,
        "parameter_count": _parameter_count(parameter_snapshot(model)),
        "component_types": {
            "neuron": type(model.neu).__name__,
            "feedforward_projection_wrapper": type(model.ff_syn).__name__,
            "feedforward_projection": type(model.ff_syn.comm).__name__,
            "feedforward_synapse": type(model.ff_syn.syn).__name__,
            "feedforward_output": type(model.ff_syn.out).__name__,
            "recurrent_projection_wrapper": type(model.rec_syn).__name__,
            "recurrent_projection": type(model.rec_syn.comm).__name__,
            "recurrent_synapse": type(model.rec_syn.syn).__name__,
            "recurrent_output": type(model.rec_syn.out).__name__,
        },
        **memory_architecture,
        "associative_memory_implementation": memory_implementation,
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "claim_boundary": CLAIM_BOUNDARY,
        "configuration": config.to_dict(),
        "device": device_report,
        "model": model_report,
        "software": _software_report(),
        "implementation": _implementation_report(),
        "data_manifests": manifests,
        "data_summary": _data_summary(data, manifests, evaluation),
        "training": training,
        "evaluation": evaluation,
        "runtime_seconds": time.perf_counter() - started,
    }
    result["qualification"] = _qualification(
        config, data, training, evaluation, device_report, model_report
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "data_manifest.json"
    result_path = config.output_dir / "result.json"
    report_path = config.output_dir / "report.txt"
    figure_path = config.output_dir / "latent_reasoning.png"
    result["artifacts"] = {
        "data_manifest": str(manifest_path),
        "result": str(result_path),
        "report": str(report_path),
        "figure": str(figure_path),
    }
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(result), encoding="utf-8")
    _plot(result, figure_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=pathlib.Path)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("var/example21")
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--recurrent-edges", type=int, default=16384)
    parser.add_argument("--context-memory-width", type=int, default=0)
    parser.add_argument("--memory-decay", type=float, default=1.0)
    parser.add_argument("--training-updates", type=int, default=96)
    parser.add_argument("--training-chunk-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--balanced-color-loss", action="store_true")
    parser.add_argument("--evaluation-task-limit", type=int)
    parser.add_argument("--ablation-slot", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--structural-only", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        if args.neurons != 2048 or args.recurrent_edges != 16384:
            raise ValueError("--smoke owns its reduced neuron and edge scale")
        return ExperimentConfig.smoke_config(
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            context_memory_width=args.context_memory_width,
            memory_decay=args.memory_decay,
            balanced_color_loss=args.balanced_color_loss,
        )
    return ExperimentConfig(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        neuron_count=args.neurons,
        recurrent_edges=args.recurrent_edges,
        context_memory_width=args.context_memory_width,
        memory_decay=args.memory_decay,
        training_updates=args.training_updates,
        training_chunk_size=args.training_chunk_size,
        learning_rate=args.learning_rate,
        balanced_color_loss=args.balanced_color_loss,
        evaluation_task_limit=args.evaluation_task_limit,
        ablation_slot=args.ablation_slot,
        structural_only=args.structural_only,
    )


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run Example 21 from command-line arguments.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments excluding the executable name. ``None`` uses ``sys.argv``.

    Returns
    -------
    dict
        Structured result also written under ``--output-dir``.
    """
    config = _config_from_args(_parser().parse_args(argv))
    result = run_experiment(config)
    print(_render_report(result), end="")
    return result


if __name__ == "__main__":
    main()
