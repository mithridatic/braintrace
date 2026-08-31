"""Tests for the production Example 21 ARC evolution adapter."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.pp_prop import example21_arc_adapter as adapter


def test_public_api_has_numpy_style_docstrings():
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    public: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public.append((node.name, node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public.append((node.name, node))
            public.extend(
                (f"{node.name}.{method.name}", method)
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not method.name.startswith("_")
            )

    assert {name for name, _ in public} == {
        "CheckpointIdentities",
        "SupervisedQuery",
        "checkpoint_identities",
        "direct_query_metrics",
        "owner_codes",
        "Example21ArcAdapter",
        "Example21ArcAdapter.training_manifest",
        "Example21ArcAdapter.evaluation_manifest",
        "Example21ArcAdapter.initialize",
        "Example21ArcAdapter.restore",
        "Example21ArcAdapter.persist",
        "Example21ArcAdapter.attest_pending",
        "Example21ArcAdapter.discard",
        "Example21ArcAdapter.rescore",
        "Example21ArcAdapter.train_parent",
        "Example21ArcAdapter.run_candidate",
        "Example21ArcAdapter.render_topology",
        "Example21ArcAdapter.evaluate_terminal",
    }
    for name, node in public:
        docstring = ast.get_docstring(node)
        assert docstring, f"{name} has no docstring"
        if isinstance(node, ast.ClassDef):
            assert "\nParameters\n----------\n" in docstring, name
        else:
            parameters = [
                argument.arg
                for argument in (*node.args.posonlyargs, *node.args.args)
                if argument.arg not in {"self", "cls"}
            ]
            parameters.extend(argument.arg for argument in node.args.kwonlyargs)
            if parameters:
                assert "\nParameters\n----------\n" in docstring, name
            if not (
                isinstance(node.returns, ast.Constant) and node.returns.value is None
            ):
                assert "\nReturns\n-------\n" in docstring, name


@dataclass(frozen=True)
class _Source:
    task_id: str
    source_sha256: str


@dataclass(frozen=True)
class _Manifest:
    root: str
    role: str
    sources: tuple[_Source, ...]


@dataclass(frozen=True)
class _Task:
    task_id: str
    targets: tuple[np.ndarray | None, ...]


class _Contracts:
    """Small corpus double that records evaluation-capability use."""

    def __init__(
        self,
        root: Path,
        *,
        missing_target: bool = False,
        partial_target: bool = False,
    ):
        self.root = root
        self.calls: list[tuple[str, bool]] = []
        self.missing_target = missing_target
        self.partial_target = partial_target
        self.sources = tuple(
            _Source(f"{index:08x}", hashlib.sha256(str(index).encode()).hexdigest())
            for index in range(400)
        )

    def load_corpus_manifest(self, root, role="practice", *, allow_evaluation=False):
        assert Path(root) == self.root
        self.calls.append((role, allow_evaluation))
        return _Manifest(str(self.root), role, self.sources)

    def load_task(
        self,
        root,
        task_id,
        role="practice",
        *,
        allow_evaluation=False,
        manifest=None,
    ):
        assert Path(root) == self.root
        assert manifest is not None
        index = int(task_id, 16)
        if self.missing_target and index == 0:
            targets = (None,)
        elif self.partial_target and index == 0:
            targets = (np.asarray([[1]], dtype=np.uint8), None)
        elif index == 0:
            targets = (
                np.asarray([[1]], dtype=np.uint8),
                np.asarray([[2]], dtype=np.uint8),
            )
        else:
            targets = (np.asarray([[index % 10]], dtype=np.uint8),)
        return _Task(task_id, targets)

    @staticmethod
    def encode_episode(task, query_index):
        events = np.zeros((32, 441), dtype=bool)
        events[0, 0] = True
        advances = np.ones(32, dtype=bool)
        return events, advances


def _perfect_logits(target: np.ndarray) -> np.ndarray:
    logits = np.full((31, 360), -8.0, dtype=np.float32)
    height, width = target.shape
    logits[0, height - 1] = 8.0
    logits[0, 30 + width - 1] = 8.0
    rows = logits[1:, 60:].reshape((30, 30, 10))
    for row in range(height):
        for column in range(width):
            rows[row, column, int(target[row, column])] = 8.0
    return logits


def _decode(logits: np.ndarray) -> np.ndarray:
    height = int(np.argmax(logits[0, :30])) + 1
    width = int(np.argmax(logits[0, 30:60])) + 1
    colors = np.argmax(logits[1:, 60:].reshape((30, 30, 10)), axis=-1)
    return colors[:height, :width].astype(np.uint8)


def _minimal_checkpoint_arrays() -> dict[str, np.ndarray]:
    return {
        "neuron_ids": np.asarray([7], dtype=np.int32),
        "dale_codes": np.asarray([0], dtype=np.int8),
        "owner_codes": np.asarray([-1], dtype=np.int16),
        "mechanism_codes": np.asarray([0], dtype=np.uint8),
        "neuron_count": np.asarray(1, dtype=np.int32),
        "integration_substeps": np.asarray(1, dtype=np.int32),
        "input_indptr": np.concatenate(
            (np.asarray([0, 1], dtype=np.int32), np.ones(440, dtype=np.int32))
        ),
        "input_indices": np.asarray([0], dtype=np.int32),
        "input_values": np.asarray([0.5], dtype=np.float32),
        "input_m1": np.asarray([0.0], dtype=np.float32),
        "input_m2": np.asarray([0.0], dtype=np.float32),
        "recurrent_indptr": np.asarray([0, 1], dtype=np.int32),
        "recurrent_indices": np.asarray([0], dtype=np.int32),
        "recurrent_values": np.asarray([0.25], dtype=np.float32),
        "recurrent_m1": np.asarray([0.0], dtype=np.float32),
        "recurrent_m2": np.asarray([0.0], dtype=np.float32),
        "readout_weight": np.zeros((1, 360), dtype=np.float32),
        "readout_bias": np.zeros(360, dtype=np.float32),
        "readout_weight_m1": np.zeros((1, 360), dtype=np.float32),
        "readout_weight_m2": np.zeros((1, 360), dtype=np.float32),
        "readout_bias_m1": np.zeros(360, dtype=np.float32),
        "readout_bias_m2": np.zeros(360, dtype=np.float32),
        "input_step": np.asarray(0, dtype=np.int64),
        "recurrent_step": np.asarray(0, dtype=np.int64),
        "readout_step": np.asarray(0, dtype=np.int64),
    }


def test_training_manifest_contains_all_400_tasks_and_every_supervised_query(tmp_path):
    contracts = _Contracts(tmp_path)
    subject = adapter.Example21ArcAdapter(tmp_path, contracts_module=contracts)

    manifest = subject.training_manifest()

    assert manifest.role == "training"
    assert manifest.task_ids == tuple(source.task_id for source in contracts.sources)
    assert len(manifest.task_ids) == 400
    assert len(manifest.query_order) == 401
    assert manifest.query_order[:3] == (
        ("00000000", 0),
        ("00000000", 1),
        ("00000001", 0),
    )
    assert contracts.calls == [("practice", False)]


def test_evaluation_manifest_is_not_read_until_terminal_capability_is_requested(
    tmp_path,
):
    contracts = _Contracts(tmp_path)
    subject = adapter.Example21ArcAdapter(tmp_path, contracts_module=contracts)

    subject.training_manifest()
    assert contracts.calls == [("practice", False)]

    manifest = subject.evaluation_manifest()
    assert manifest.role == "evaluation"
    assert contracts.calls[-1] == ("evaluation", True)


def test_manifest_fails_closed_when_any_task_has_no_supervised_query(tmp_path):
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path, missing_target=True),
    )

    with pytest.raises(ValueError, match="missing target queries"):
        subject.training_manifest()


def test_manifest_rejects_one_missing_target_among_other_supervised_queries(tmp_path):
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path, partial_target=True),
    )

    with pytest.raises(ValueError, match=r"missing target queries \(1,\)"):
        subject.training_manifest()


def test_direct_query_metrics_use_shape_and_only_valid_target_cells():
    target = np.asarray([[2, 3], [4, 5]], dtype=np.uint8)
    logits = _perfect_logits(target)

    exact, loss = adapter.direct_query_metrics(logits, target, _decode)

    assert exact is True
    assert 0.0 < loss < 1e-4
    changed_padding = logits.copy()
    changed_padding[30, 60:] = 100.0
    assert adapter.direct_query_metrics(
        changed_padding, target, _decode
    ) == pytest.approx((True, loss))


def test_direct_query_metrics_reject_nonfinite_output_and_penalize_wrong_shape():
    target = np.asarray([[7]], dtype=np.uint8)
    perfect = _perfect_logits(target)
    wrong = perfect.copy()
    wrong[0, 0] = -8.0
    wrong[0, 1] = 8.0

    exact, loss = adapter.direct_query_metrics(wrong, target, _decode)
    assert exact is False
    assert loss > adapter.direct_query_metrics(perfect, target, _decode)[1]

    invalid = perfect.copy()
    invalid[0, 0] = np.nan
    exact, loss = adapter.direct_query_metrics(invalid, target, _decode)
    assert exact is False
    assert np.isinf(loss)


def test_owner_codes_distinguish_unique_shared_and_inactive_neurons():
    values = np.asarray(
        [
            [4.0, 2.0, 0.0, 3.0],
            [1.0, 2.0, 0.0, 1.0],
        ]
    )

    assert np.array_equal(
        adapter.owner_codes(values),
        np.asarray([0, -2, -1, 0], dtype=np.int16),
    )


def test_checkpoint_identities_partition_topology_parameters_and_muon_state():
    arrays = _minimal_checkpoint_arrays()
    first = adapter.checkpoint_identities(arrays)

    changed_optimizer = {name: value.copy() for name, value in arrays.items()}
    changed_optimizer["recurrent_m1"][0] = 3.0
    second = adapter.checkpoint_identities(changed_optimizer)
    assert second.topology_sha256 == first.topology_sha256
    assert second.parameters_sha256 == first.parameters_sha256
    assert second.optimizer_sha256 != first.optimizer_sha256

    changed_owner = {name: value.copy() for name, value in arrays.items()}
    changed_owner["owner_codes"][0] = 5
    third = adapter.checkpoint_identities(changed_owner)
    assert third.topology_sha256 != first.topology_sha256
    assert third.parameters_sha256 == first.parameters_sha256
    assert third.optimizer_sha256 == first.optimizer_sha256
    assert first.persistent_bytes == sum(value.nbytes for value in arrays.values())


def test_numeric_and_identity_contracts_reject_malformed_inputs():
    arrays = _minimal_checkpoint_arrays()
    arrays.pop("input_m1")
    with pytest.raises(ValueError, match="missing input_m1"):
        adapter.checkpoint_identities(arrays)
    with pytest.raises(ValueError, match="label is outside"):
        adapter._log_cross_entropy(np.zeros((2, 2)), 0)
    assert np.isinf(adapter._log_cross_entropy(np.asarray([np.nan]), 0))
    with pytest.raises(ValueError, match=r"shape \(31, 360\)"):
        adapter.direct_query_metrics(np.zeros((30, 360)), np.ones((1, 1)), _decode)
    with pytest.raises(ValueError, match="integer color grid"):
        adapter.direct_query_metrics(np.zeros((31, 360)), np.asarray([[11]]), _decode)
    with pytest.raises(ValueError, match="nonempty task-by-neuron"):
        adapter.owner_codes(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        adapter.owner_codes(np.asarray([[np.nan]]))
    assert np.array_equal(adapter._normalise(np.zeros(2)), np.zeros(2))
    assert np.array_equal(adapter._normalise(np.asarray([0.0, 2.0])), [0.0, 1.0])


def test_encoded_query_bank_is_immutable_complete_and_cached(tmp_path):
    contracts = _Contracts(tmp_path)
    subject = adapter.Example21ArcAdapter(tmp_path, contracts_module=contracts)

    first = subject._encoded_queries("training")
    second = subject._encoded_queries("training")

    assert first is second
    assert len(first) == 401
    assert not first[0].events.flags.writeable
    assert not first[0].advances.flags.writeable
    assert not first[0].target.flags.writeable
    assert subject.training_manifest() is subject.training_manifest()


def test_candidate_arm_names_fail_closed_before_model_work(tmp_path):
    contracts = _Contracts(tmp_path)
    subject = adapter.Example21ArcAdapter(tmp_path, contracts_module=contracts)

    with pytest.raises(ValueError, match="unsupported evolution arm"):
        subject._mutation_kind("edge", "excitatory")
    assert subject._mutation_kind("edge-revisit", "add") == "edge-add"
    assert subject._mutation_kind("dale", "inhibitory") == "dale-inhibitory"


def test_production_dependencies_and_hyphenated_model_load_lazily(tmp_path):
    subject = adapter.Example21ArcAdapter(tmp_path)

    contracts = subject._contracts()
    structural = subject._structural()
    model = subject._model()

    assert contracts.__name__ == "examples.pp_prop.arc_contracts"
    assert structural.__name__ == "examples.pp_prop.example21_structural"
    assert model.__name__ == "examples.pp_prop._example21_braincell_arc_runtime"
    assert subject._contracts() is contracts
    assert subject._structural() is structural
    assert subject._model() is model
    assert adapter.Example21ArcAdapter(tmp_path)._model() is model


def test_candidate_compile_failure_is_recorded_and_preserves_parent(
    tmp_path, monkeypatch
):
    parent_path = tmp_path / "parent.npz"
    parent_path.write_bytes(b"immutable-parent")
    parent = SimpleNamespace(
        checkpoint_path=str(parent_path),
        checkpoint_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    )
    topology = SimpleNamespace(
        neuron_count=2,
        input_value=np.ones(2),
        recurrent_value=np.ones(2),
    )
    runtime = SimpleNamespace(
        topology=topology,
        optimizer=object(),
        readout_bias=np.zeros(360),
    )
    structural = SimpleNamespace(enforce_biological_connection_ceiling=lambda *_: 4)
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
    )
    monkeypatch.setattr(subject, "restore", lambda value: value)
    monkeypatch.setattr(subject, "_runtime_from_checkpoint", lambda _: runtime)
    monkeypatch.setattr(subject, "_parent_evidence", lambda *_: object())
    monkeypatch.setattr(
        subject,
        "_mutate",
        lambda *_: (topology, object()),
    )
    monkeypatch.setattr(
        subject,
        "_runtime_from_state",
        lambda *_: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )
    context = SimpleNamespace(
        stage="edge",
        stage_id="r000-edge",
        output_dir=tmp_path,
        config=SimpleNamespace(max_neurons=8, max_recurrent_edges=8),
    )

    attempt = subject.run_candidate(parent, "add", object(), context)

    assert attempt.status == "failed"
    assert attempt.reason == "RuntimeError: compile failed"
    assert attempt.executed_updates == 0
    assert parent_path.read_bytes() == b"immutable-parent"


def test_persist_removes_selected_temporary_checkpoint(tmp_path, monkeypatch):
    source = (tmp_path / ".candidates" / "candidate.npz").resolve()
    source.parent.mkdir()
    source.write_bytes(b"candidate")
    destination = tmp_path / "checkpoints" / "r000-edge.npz"
    candidate = SimpleNamespace(
        candidate_id="candidate",
        checkpoint_path=str(source),
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        score=object(),
        topology_changed=True,
    )

    class _CheckpointModule:
        @staticmethod
        def load_checkpoint(path):
            assert Path(path) == source
            return {"sentinel": np.asarray(1)}

        @staticmethod
        def write_checkpoint(path, arrays, *, parent=None):
            assert arrays["sentinel"] == 1
            assert Path(parent) == source
            Path(path).parent.mkdir(parents=True)
            Path(path).write_bytes(b"persisted")

    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        model_module=_CheckpointModule,
    )
    subject._temporary_paths.add(source)
    subject._record_lineage(candidate, "0" * 64)
    persisted = SimpleNamespace(checkpoint_sha256="f" * 64)
    evidence = object()
    subject._evidence_by_checkpoint[candidate.checkpoint_sha256] = evidence
    monkeypatch.setattr(subject, "_snapshot", lambda **_: persisted)

    result = subject.persist(
        candidate,
        destination,
        parent_checkpoint_sha256="0" * 64,
        stage_id="r000-edge",
    )

    assert result is persisted
    assert destination.read_bytes() == b"persisted"
    assert not source.exists()
    assert not subject._provenance_path(source).exists()
    assert source not in subject._temporary_paths
    assert candidate.checkpoint_sha256 not in subject._evidence_by_checkpoint
    assert subject._evidence_by_checkpoint[persisted.checkpoint_sha256] is evidence


def test_snapshot_restore_verifies_bytes_components_and_resources(
    tmp_path, monkeypatch
):
    from examples.pp_prop import example21_evolve
    from examples.pp_prop import example21_structural as structural

    arrays = _minimal_checkpoint_arrays()
    path = tmp_path / "accepted.npz"
    path.write_bytes(b"format-one-container")

    class _CheckpointModule:
        @staticmethod
        def load_checkpoint(value):
            assert Path(value) == path
            return {name: item.copy() for name, item in arrays.items()}

    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
        model_module=_CheckpointModule,
    )
    monkeypatch.setattr(subject, "_peak_device_memory_bytes", lambda: 17)
    score = example21_evolve.ScoreSnapshot(("task",), (False,), (1.5,))

    snapshot = subject._snapshot(
        candidate_id="accepted",
        path=path,
        arrays=arrays,
        score=score,
        topology_changed=False,
    )
    restored = subject.restore(snapshot)

    assert restored.checkpoint_sha256 == snapshot.checkpoint_sha256
    assert restored.optimizer_sha256 == snapshot.optimizer_sha256
    assert restored.resources.persistent_bytes == sum(
        value.nbytes for value in arrays.values()
    )
    assert restored.resources.checkpoint_bytes == path.stat().st_size
    assert restored.resources.device_memory_bytes == 17

    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        subject.restore(snapshot)


def test_format_one_arrays_rebuild_canonical_topology_and_optimizer(
    tmp_path, monkeypatch
):
    from examples.pp_prop import example21_structural as structural

    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
    )
    arrays = _minimal_checkpoint_arrays()

    topology, optimizer, bias = subject._topology_optimizer_from_arrays(arrays)

    assert topology.neuron_count == 1
    assert np.array_equal(topology.neuron_ids, np.asarray([7]))
    assert optimizer.input_step == optimizer.recurrent_step == 0
    assert bias.shape == (360,)

    nondefault_substeps = {name: value.copy() for name, value in arrays.items()}
    nondefault_substeps["integration_substeps"] = np.asarray(2, dtype=np.int32)
    with pytest.raises(ValueError, match="cannot execute non-default"):
        subject._topology_optimizer_from_arrays(nondefault_substeps)
    nondefault_mechanism = {name: value.copy() for name, value in arrays.items()}
    nondefault_mechanism["mechanism_codes"][0] = 1
    with pytest.raises(ValueError, match="cannot execute deferred"):
        subject._topology_optimizer_from_arrays(nondefault_mechanism)

    monkeypatch.setattr(structural, "validate_topology_dale", lambda value: False)
    with pytest.raises(ValueError, match="violates an accepted Dale sign"):
        subject._topology_optimizer_from_arrays(arrays)


def test_runtime_builders_restore_muon_steps_and_fresh_state(tmp_path, monkeypatch):
    topology = SimpleNamespace(
        input_value=np.asarray([1.0]),
        recurrent_value=np.asarray([2.0]),
    )
    optimizer = SimpleNamespace(input_step=3, recurrent_step=5, readout_step=4)

    class _Model:
        def __init__(self, supplied=None):
            self.supplied = supplied
            self.input_weight = SimpleNamespace(value=np.asarray([1.0]))
            self.recurrent_weight = SimpleNamespace(value=np.asarray([2.0]))
            self.readout_bias = SimpleNamespace(value=np.zeros(360))

    class _Trainer:
        def __init__(self, learner, parameters):
            self.learner = learner
            self.parameters = parameters
            self.muon_groups = None
            self.updates = np.asarray(0)
            self.synced = False

        def _sync_compiled_parameters(self):
            self.synced = True

    module = SimpleNamespace(
        BrainCellArcModel=_Model,
        compile_pp_prop_model=lambda model: SimpleNamespace(model=model),
        PPPropEpisodeTrainer=_Trainer,
        load_checkpoint=lambda path: {"path": str(path)},
    )
    structural = SimpleNamespace(
        canonicalize_topology_and_optimizer=lambda first, second: (first, second),
        initialize_muon_groups=lambda trainer, state: {"restored": state},
        topology_from_model=lambda model: topology,
        optimizer_from_muon_groups=lambda trainer: optimizer,
    )
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
        model_module=module,
    )

    restored = subject._runtime_from_state(topology, optimizer, np.ones(360))
    fresh = subject._fresh_runtime()

    assert int(restored.trainer.updates) == 5
    assert restored.trainer.synced
    assert restored.trainer.muon_groups == {"restored": optimizer}
    assert np.array_equal(restored.model.readout_bias.value, np.ones(360))
    assert fresh.topology is topology
    assert fresh.optimizer is optimizer

    sentinel = object()
    monkeypatch.setattr(
        subject,
        "_topology_optimizer_from_arrays",
        lambda arrays: (topology, optimizer, np.zeros(360)),
    )
    monkeypatch.setattr(subject, "_runtime_from_state", lambda *values: sentinel)
    assert subject._runtime_from_checkpoint("checkpoint.npz") is sentinel


def test_device_memory_measurement_prefers_peak_and_fails_closed(monkeypatch):
    import jax

    devices = (
        SimpleNamespace(memory_stats=lambda: None),
        SimpleNamespace(memory_stats=lambda: {"unrelated": 7}),
        SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 11}),
        SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 23}),
    )
    monkeypatch.setattr(jax, "devices", lambda: devices)
    assert adapter.Example21ArcAdapter._peak_device_memory_bytes() == 23

    monkeypatch.setattr(
        jax, "devices", lambda: (_ for _ in ()).throw(RuntimeError("no backend"))
    )
    assert adapter.Example21ArcAdapter._peak_device_memory_bytes() is None


def test_runtime_checkpoint_writer_refreshes_owners_and_caches_evidence(
    tmp_path, monkeypatch
):
    from examples.pp_prop import example21_evolve

    arrays = _minimal_checkpoint_arrays()
    score = example21_evolve.ScoreSnapshot(("task",), (False,), (1.0,))
    scored = adapter._ScoredRuntime(
        score=score,
        owners=((0,),),
        owner_codes=np.asarray([0], dtype=np.int16),
        neuron_scores=np.asarray([1.0]),
        source_scores=np.asarray([1.0]),
        target_scores=np.asarray([1.0]),
        edge_scores=np.asarray([1.0]),
    )
    model = SimpleNamespace(owner_codes=None)
    runtime = adapter._Runtime(
        model=model,
        learner=object(),
        trainer=object(),
        topology=object(),
        optimizer=object(),
        readout_bias=np.zeros(360),
    )

    class _Module:
        @staticmethod
        def write_checkpoint(path, values):
            assert values is arrays
            Path(path).write_bytes(b"checkpoint")

    structural = SimpleNamespace(
        optimizer_from_muon_groups=lambda trainer: "optimizer",
        checkpoint_arrays=lambda candidate, optimizer, evidence: arrays,
        validate_topology_dale=lambda topology: True,
        _peak_process_resident_memory_bytes=lambda: 19,
    )
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
        model_module=_Module,
    )
    monkeypatch.setattr(subject, "_score_runtime", lambda *values, **options: scored)
    monkeypatch.setattr(
        subject,
        "_topology_optimizer_from_arrays",
        lambda values: (object(), None, None),
    )
    monkeypatch.setattr(subject, "_peak_device_memory_bytes", lambda: 29)
    path = tmp_path / "candidate.npz"

    snapshot = subject._write_runtime(
        runtime,
        role="training",
        candidate_id="candidate",
        path=path,
        topology_changed=True,
    )

    assert np.array_equal(model.owner_codes, np.asarray([0]))
    assert snapshot.resources.peak_host_ram_bytes == 19
    assert snapshot.resources.device_memory_bytes == 29
    assert subject._evidence_by_checkpoint[snapshot.checkpoint_sha256] is scored

    structural.validate_topology_dale = lambda topology: False
    with pytest.raises(ValueError, match="violates a Dale sign"):
        subject._write_runtime(
            runtime,
            role="training",
            candidate_id="invalid",
            path=tmp_path / "invalid.npz",
            topology_changed=True,
        )


def test_initialize_enforces_muon_128_and_stages_fresh_runtime(tmp_path, monkeypatch):
    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    with pytest.raises(ValueError, match="Muon and exactly 128"):
        subject.initialize(SimpleNamespace(optimizer="adam", updates=128), tmp_path)
    with pytest.raises(ValueError, match="Muon and exactly 128"):
        subject.initialize(SimpleNamespace(optimizer="muon", updates=64), tmp_path)

    runtime = object()
    staged = tmp_path / ".candidates" / "initial.npz"
    staged.parent.mkdir()
    staged.write_bytes(b"initial")
    result = SimpleNamespace(
        candidate_id="initial",
        checkpoint_path=str(staged),
        checkpoint_sha256=hashlib.sha256(staged.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(subject, "training_manifest", lambda: object())
    monkeypatch.setattr(subject, "_fresh_runtime", lambda: runtime)
    monkeypatch.setattr(subject, "_write_runtime", lambda *args, **kwargs: result)

    assert (
        subject.initialize(SimpleNamespace(optimizer="muon", updates=128), tmp_path)
        is result
    )
    assert (tmp_path / ".candidates" / "initial.npz").resolve() in (
        subject._temporary_paths
    )
    sanitized = subject._candidate_path(tmp_path, "unsafe / candidate")
    assert sanitized.name == "unsafe---candidate.npz"


def test_restore_and_persist_reject_missing_mismatched_or_invalid_lineage(
    tmp_path, monkeypatch
):
    arrays = _minimal_checkpoint_arrays()
    identities = adapter.checkpoint_identities(arrays)
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        model_module=SimpleNamespace(load_checkpoint=lambda path: arrays),
    )
    missing = SimpleNamespace(checkpoint_path=str(tmp_path / "missing.npz"))
    with pytest.raises(ValueError, match="is missing"):
        subject.restore(missing)

    path = tmp_path / "candidate.npz"
    path.write_bytes(b"candidate")
    candidate = SimpleNamespace(
        candidate_id="candidate",
        checkpoint_path=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        topology_sha256="0" * 64,
        parameters_sha256=identities.parameters_sha256,
        optimizer_sha256=identities.optimizer_sha256,
        score=object(),
        topology_changed=False,
    )
    with pytest.raises(ValueError, match="components differ"):
        subject.restore(candidate)

    candidate.checkpoint_sha256 = "1" * 64
    with pytest.raises(ValueError, match="changed before persistence"):
        subject.persist(
            candidate,
            tmp_path / "selected.npz",
            parent_checkpoint_sha256=None,
            stage_id="stage",
        )
    candidate.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Parent checkpoint identity"):
        subject.persist(
            candidate,
            tmp_path / "selected.npz",
            parent_checkpoint_sha256="short",
            stage_id="stage",
        )
    assert adapter._is_sha256("a" * 64)
    assert not adapter._is_sha256("A" * 64)
    candidate_dir = tmp_path / ".candidates"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / "candidate.npz"
    path.replace(candidate_path)
    candidate.checkpoint_path = str(candidate_path)
    subject._record_lineage(candidate, "a" * 64)
    with pytest.raises(ValueError, match="Stage identity"):
        subject.persist(
            candidate,
            tmp_path / "selected.npz",
            parent_checkpoint_sha256="a" * 64,
            stage_id="stage",
        )
    with pytest.raises(ValueError, match="ancestry"):
        subject.persist(
            candidate,
            tmp_path / "checkpoints" / "stage.npz",
            parent_checkpoint_sha256="b" * 64,
            stage_id="stage",
        )


def test_discard_removes_only_tracked_candidate_and_cached_evidence(tmp_path):
    from examples.pp_prop import example21_evolve

    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    subject.discard(example21_evolve.CandidateAttempt.blocked("add", "blocked"))
    path = subject._candidate_path(tmp_path, "rejected")
    path.write_bytes(b"candidate")
    candidate = SimpleNamespace(
        checkpoint_path=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    subject._evidence_by_checkpoint[candidate.checkpoint_sha256] = object()

    subject.discard(example21_evolve.CandidateAttempt.completed("add", candidate))

    assert not path.exists()
    assert candidate.checkpoint_sha256 not in subject._evidence_by_checkpoint

    unowned = tmp_path / "unowned.npz"
    unowned.write_bytes(b"not-adapter-owned")
    unowned_candidate = SimpleNamespace(
        checkpoint_path=str(unowned),
        checkpoint_sha256=hashlib.sha256(unowned.read_bytes()).hexdigest(),
    )
    subject.discard(
        example21_evolve.CandidateAttempt.completed("add", unowned_candidate)
    )
    assert unowned.read_bytes() == b"not-adapter-owned"


def test_fresh_adapter_restart_cleanup_requires_owned_run_and_matching_sha(tmp_path):
    from examples.pp_prop import example21_evolve

    output = tmp_path / "run"
    candidate_dir = output / ".candidates"
    candidate_dir.mkdir(parents=True)
    (output / "run-state.json").write_text("{}", encoding="utf-8")
    path = candidate_dir / "rejected.npz"
    path.write_bytes(b"restart-candidate")
    candidate = SimpleNamespace(
        checkpoint_path=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    attempt = example21_evolve.CandidateAttempt.completed("prune", candidate)
    fresh = adapter.Example21ArcAdapter(tmp_path, contracts_module=_Contracts(tmp_path))

    fresh.discard(attempt)
    assert not path.exists()
    fresh.discard(attempt)

    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 is invalid"):
        fresh.discard(attempt)
    assert path.exists()


def test_fresh_adapter_persists_pending_candidate_from_durable_lineage(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"
    source = output / ".candidates" / "r001-edge-add.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"candidate-checkpoint")
    destination = output / "checkpoints" / "r001-edge.npz"
    parent_sha256 = "a" * 64
    candidate = SimpleNamespace(
        candidate_id="r001-edge-add",
        checkpoint_path=str(source),
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        score=object(),
        topology_changed=True,
    )

    class _CheckpointModule:
        @staticmethod
        def load_checkpoint(path):
            assert Path(path) == source
            return {"sentinel": np.asarray(1)}

        @staticmethod
        def write_checkpoint(path, arrays, *, parent=None):
            assert arrays["sentinel"] == 1
            assert Path(parent) == source
            Path(path).parent.mkdir(parents=True)
            Path(path).write_bytes(b"persisted-after-restart")

    original = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        model_module=_CheckpointModule,
    )
    original._record_lineage(candidate, parent_sha256)
    provenance = original._provenance_path(source)
    assert provenance.is_file()

    fresh = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        model_module=_CheckpointModule,
    )
    persisted = SimpleNamespace(checkpoint_sha256="f" * 64)
    monkeypatch.setattr(fresh, "_snapshot", lambda **_: persisted)

    assert (
        fresh.persist(
            candidate,
            destination,
            parent_checkpoint_sha256=parent_sha256,
            stage_id="r001-edge",
        )
        is persisted
    )
    assert destination.read_bytes() == b"persisted-after-restart"
    assert not source.exists()
    assert not provenance.exists()


def test_attest_pending_verifies_components_and_registers_exact_journal_lineage(
    tmp_path, monkeypatch
):
    from examples.pp_prop import example21_evolve
    from examples.pp_prop import example21_structural as structural

    arrays = _minimal_checkpoint_arrays()
    source = tmp_path / "run" / ".candidates" / "r001-edge-add.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pending-child")

    class _CheckpointModule:
        @staticmethod
        def load_checkpoint(path):
            assert Path(path) == source
            return {name: value.copy() for name, value in arrays.items()}

    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
        model_module=_CheckpointModule,
    )
    monkeypatch.setattr(subject, "_peak_device_memory_bytes", lambda: None)
    score = example21_evolve.ScoreSnapshot(("task",), (False,), (1.0,))
    candidate = subject._snapshot(
        candidate_id="r001-edge-add",
        path=source,
        arrays=arrays,
        score=score,
        topology_changed=True,
    )
    parent_sha256 = "a" * 64

    attested = subject.attest_pending(
        candidate,
        parent_checkpoint_sha256=parent_sha256,
        stage_id="r001-edge",
    )

    assert attested.checkpoint_sha256 == candidate.checkpoint_sha256
    assert attested.topology_sha256 == candidate.topology_sha256
    assert attested.parameters_sha256 == candidate.parameters_sha256
    assert attested.optimizer_sha256 == candidate.optimizer_sha256
    assert subject._parent_by_checkpoint[candidate.checkpoint_sha256] == parent_sha256
    assert subject._provenance_path(source).is_file()
    assert (
        subject.attest_pending(
            candidate,
            parent_checkpoint_sha256=parent_sha256,
            stage_id="r001-edge",
        ).checkpoint_sha256
        == candidate.checkpoint_sha256
    )

    with pytest.raises(ValueError, match="journal parent"):
        subject.attest_pending(
            candidate,
            parent_checkpoint_sha256="b" * 64,
            stage_id="r001-edge",
        )

    with pytest.raises(ValueError, match="components differ"):
        subject.attest_pending(
            replace(candidate, topology_sha256="0" * 64),
            parent_checkpoint_sha256=parent_sha256,
            stage_id="r001-edge",
        )
    with pytest.raises(ValueError, match="Pending parent identity"):
        subject.attest_pending(
            candidate,
            parent_checkpoint_sha256="short",
            stage_id="r001-edge",
        )
    with pytest.raises(ValueError, match="Pending stage identity"):
        subject.attest_pending(
            candidate,
            parent_checkpoint_sha256=parent_sha256,
            stage_id="unsafe/stage",
        )


def test_lineage_provenance_rejects_missing_changed_and_malformed_candidates(tmp_path):
    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    outside_candidate = SimpleNamespace(
        candidate_id="outside",
        checkpoint_path=str(outside),
        checkpoint_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    subject._record_lineage(outside_candidate, None)
    assert not subject._provenance_path(outside).exists()

    with pytest.raises(ValueError, match="Candidate checkpoint identity"):
        subject._record_lineage(
            SimpleNamespace(
                candidate_id="bad",
                checkpoint_path=str(outside),
                checkpoint_sha256="short",
            ),
            None,
        )
    with pytest.raises(ValueError, match="Parent checkpoint identity"):
        subject._record_lineage(outside_candidate, "short")

    output = tmp_path / "run"
    source = output / ".candidates" / "child.npz"
    source.parent.mkdir(parents=True)
    missing = SimpleNamespace(
        candidate_id="child",
        checkpoint_path=str(source),
        checkpoint_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="missing candidate"):
        subject._record_lineage(missing, None)
    assert missing.checkpoint_sha256 not in subject._parent_by_checkpoint
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed candidate"):
        subject._record_lineage(missing, None)
    assert missing.checkpoint_sha256 not in subject._parent_by_checkpoint

    actual = SimpleNamespace(
        **(
            vars(missing)
            | {"checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    )
    destination = output / "checkpoints" / "stage.npz"
    with pytest.raises(ValueError, match="no durable lineage"):
        subject._persisted_lineage(actual, source, destination)
    provenance = subject._provenance_path(source)
    provenance.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        subject._persisted_lineage(actual, source, destination)
    provenance.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        subject._persisted_lineage(actual, source, destination)

    subject._record_lineage(actual, "a" * 64)
    with pytest.raises(ValueError, match="does not match"):
        subject._persisted_lineage(
            SimpleNamespace(**(vars(actual) | {"candidate_id": "different"})),
            source,
            destination,
        )
    with pytest.raises(ValueError, match="outside"):
        subject._persisted_lineage(actual, source, tmp_path / "stage.npz")


def test_neuron_addition_preflights_recurrent_growth_before_construction(tmp_path):
    structural, topology, optimizer, evidence = _small_structural_state()
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
    )
    config = SimpleNamespace(
        max_neurons=64,
        max_recurrent_edges=len(topology.recurrent_value) + 1,
    )

    with pytest.raises(ValueError, match="configured recurrent-edge cap"):
        subject._neuron_add(topology, optimizer, evidence, config)


def test_parent_training_and_evidence_cache_preserve_score_identity(
    tmp_path, monkeypatch
):
    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    parent = SimpleNamespace(
        checkpoint_path="parent.npz",
        checkpoint_sha256="b" * 64,
        score=SimpleNamespace(task_ids=("task",)),
    )
    runtime = object()
    staged = tmp_path / ".candidates" / "r000-train-training.npz"
    staged.parent.mkdir()
    staged.write_bytes(b"trained")
    result = SimpleNamespace(
        candidate_id="r000-train-training",
        checkpoint_path=str(staged),
        checkpoint_sha256=hashlib.sha256(staged.read_bytes()).hexdigest(),
    )
    context = SimpleNamespace(stage_id="r000-train", output_dir=tmp_path)
    monkeypatch.setattr(subject, "restore", lambda value: value)
    monkeypatch.setattr(subject, "_runtime_from_checkpoint", lambda path: runtime)
    monkeypatch.setattr(subject, "_run_update_block", lambda *values: None)
    monkeypatch.setattr(subject, "_write_runtime", lambda *args, **kwargs: result)

    assert subject.train_parent(parent, object(), context) is result

    scored = SimpleNamespace(score=SimpleNamespace(task_ids=("task",)))
    monkeypatch.setattr(subject, "_score_runtime", lambda *values, **options: scored)
    assert subject._parent_evidence(parent, runtime) is scored
    assert subject._parent_evidence(parent, runtime) is scored

    other = adapter.Example21ArcAdapter(tmp_path, contracts_module=_Contracts(tmp_path))
    monkeypatch.setattr(
        other,
        "_score_runtime",
        lambda *values, **options: SimpleNamespace(
            score=SimpleNamespace(task_ids=("other",))
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        other._parent_evidence(parent, runtime)


def _small_structural_state():
    from examples.pp_prop import example21_structural as structural

    count = 20
    recurrent_source = np.arange(count, dtype=np.int32)
    recurrent_target = (recurrent_source + 1) % count
    topology = structural.SparseTopology(
        input_source=np.arange(count, dtype=np.int32),
        input_target=np.arange(count, dtype=np.int32),
        input_value=np.linspace(0.1, 0.2, count, dtype=np.float32),
        recurrent_source=recurrent_source,
        recurrent_target=recurrent_target,
        recurrent_value=np.linspace(-0.2, 0.2, count, dtype=np.float32),
        readout=np.zeros((count, 360), dtype=np.float32),
        dale=np.zeros(count, dtype=np.int8),
        mechanisms=tuple(() for _ in range(count)),
        owner_codes=np.arange(count, dtype=np.int16),
        neuron_ids=np.arange(100, 100 + count, dtype=np.int32),
    )
    optimizer = structural.StructuralAdam(
        neuron_first=np.zeros((count, 360), dtype=np.float32),
        neuron_second=np.zeros((count, 360), dtype=np.float32),
        input_first=np.linspace(0.0, 1.0, count, dtype=np.float32),
        input_second=np.zeros(count, dtype=np.float32),
        recurrent_first=np.linspace(1.0, 0.0, count, dtype=np.float32),
        recurrent_second=np.zeros(count, dtype=np.float32),
        bias_first=np.zeros(360, dtype=np.float32),
        bias_second=np.zeros(360, dtype=np.float32),
        input_step=3,
        recurrent_step=3,
        readout_step=3,
    )
    evidence = adapter._ScoredRuntime(
        score=object(),
        owners=tuple((index,) for index in range(count)),
        owner_codes=np.arange(count, dtype=np.int16),
        neuron_scores=np.linspace(0.1, 1.0, count),
        source_scores=np.linspace(0.1, 1.0, count),
        target_scores=np.linspace(1.0, 0.1, count),
        edge_scores=np.linspace(0.0, 1.0, count),
    )
    return structural, topology, optimizer, evidence


def test_structural_mutations_preserve_state_and_apply_exact_five_percent(tmp_path):
    structural, topology, optimizer, evidence = _small_structural_state()
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
    )
    config = SimpleNamespace(max_neurons=64, max_recurrent_edges=64)

    edge_pruned, edge_optimizer = subject._edge_prune(topology, optimizer, evidence)
    edge_added, added_optimizer = subject._edge_add(
        topology, optimizer, evidence, config
    )
    neuron_pruned, neuron_optimizer = subject._neuron_prune(
        topology, optimizer, evidence
    )
    neuron_added, twin_optimizer = subject._neuron_add(
        topology, optimizer, evidence, config
    )

    assert len(edge_pruned.recurrent_value) == 19
    assert len(edge_optimizer.recurrent_first) == 19
    assert len(edge_added.recurrent_value) == 21
    assert len(added_optimizer.recurrent_first) == 21
    assert neuron_pruned.neuron_count == 19
    assert len(neuron_optimizer.neuron_first) == 19
    assert neuron_added.neuron_count == 21
    assert len(twin_optimizer.neuron_first) == 21
    assert set(topology.neuron_ids).issubset(neuron_added.neuron_ids)
    assert int(np.max(neuron_added.neuron_ids)) == 120


def test_dale_siblings_start_from_same_untyped_parent_and_keep_optimizer(tmp_path):
    structural, topology, optimizer, evidence = _small_structural_state()
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
    )

    excitatory, excitatory_optimizer = subject._dale_assignment(
        topology, optimizer, evidence, "a" * 64, 1
    )
    inhibitory, inhibitory_optimizer = subject._dale_assignment(
        topology, optimizer, evidence, "a" * 64, -1
    )

    assert np.count_nonzero(excitatory.dale == 1) == 1
    assert np.count_nonzero(inhibitory.dale == -1) == 1
    assert np.count_nonzero(topology.dale) == 0
    assert structural.validate_topology_dale(excitatory)
    assert structural.validate_topology_dale(inhibitory)
    assert np.array_equal(
        excitatory_optimizer.recurrent_first, optimizer.recurrent_first
    )
    assert np.array_equal(
        inhibitory_optimizer.recurrent_first, optimizer.recurrent_first
    )


def test_mutation_dispatch_reaches_every_structural_sibling(tmp_path, monkeypatch):
    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    expected = {
        "edge-add": "edge-added",
        "edge-prune": "edge-pruned",
        "neuron-add": "neuron-added",
        "neuron-prune": "neuron-pruned",
        "dale-excitatory": "dale-positive",
        "dale-inhibitory": "dale-negative",
    }
    monkeypatch.setattr(subject, "_edge_add", lambda *args: "edge-added")
    monkeypatch.setattr(subject, "_edge_prune", lambda *args: "edge-pruned")
    monkeypatch.setattr(subject, "_neuron_add", lambda *args: "neuron-added")
    monkeypatch.setattr(subject, "_neuron_prune", lambda *args: "neuron-pruned")
    monkeypatch.setattr(
        subject,
        "_dale_assignment",
        lambda *args: "dale-positive" if args[-1] == 1 else "dale-negative",
    )
    parent = SimpleNamespace(checkpoint_sha256="a" * 64)

    for kind, result in expected.items():
        assert (
            subject._mutate(kind, object(), object(), object(), parent, object())
            == result
        )
    with pytest.raises(ValueError, match="Unknown mutation kind"):
        subject._mutate("unknown", object(), object(), object(), parent, object())


def _candidate_harness(tmp_path, monkeypatch, name):
    root = tmp_path / name
    root.mkdir()
    parent_path = root / "parent.npz"
    parent_path.write_bytes(b"immutable-parent")
    parent = SimpleNamespace(
        candidate_id="parent",
        checkpoint_path=str(parent_path),
        checkpoint_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        score=SimpleNamespace(task_ids=("task",)),
    )
    topology = SimpleNamespace(
        neuron_count=2,
        input_value=np.ones(2),
        recurrent_value=np.ones(2),
    )
    runtime = SimpleNamespace(
        topology=topology,
        optimizer=object(),
        readout_bias=np.zeros(360),
    )
    structural = SimpleNamespace(enforce_biological_connection_ceiling=lambda *_: 4)
    subject = adapter.Example21ArcAdapter(
        root,
        contracts_module=_Contracts(root),
        structural_module=structural,
    )
    monkeypatch.setattr(subject, "restore", lambda candidate: candidate)
    monkeypatch.setattr(subject, "_runtime_from_checkpoint", lambda path: runtime)
    monkeypatch.setattr(subject, "_parent_evidence", lambda *args: object())
    monkeypatch.setattr(subject, "_mutate", lambda *args: (topology, object()))
    candidate_runtime = SimpleNamespace(
        trainer=SimpleNamespace(updates=np.asarray(11, dtype=np.int32))
    )
    monkeypatch.setattr(subject, "_runtime_from_state", lambda *args: candidate_runtime)

    def complete_update_block(runtime, _schedule):
        runtime.trainer.updates = np.asarray(
            int(np.asarray(runtime.trainer.updates)) + 128,
            dtype=np.int32,
        )

    monkeypatch.setattr(subject, "_run_update_block", complete_update_block)
    context = SimpleNamespace(
        stage="edge",
        stage_id="r001-edge",
        output_dir=root,
        config=SimpleNamespace(max_neurons=8, max_recurrent_edges=8),
    )
    return subject, parent, parent_path, runtime, topology, structural, context


@pytest.mark.parametrize(
    ("case", "status", "reason"),
    (
        ("evidence", "failed", "evidence failed"),
        ("mutation-block", "blocked", "cannot mutate"),
        ("mutation-fail", "failed", "mutation failed"),
        ("neuron-cap", "blocked", "neuron cap"),
        ("edge-cap", "blocked", "recurrent-edge cap"),
        ("biological-cap", "blocked", "biological cap"),
    ),
)
def test_run_candidate_classifies_independent_sibling_failures(
    tmp_path, monkeypatch, case, status, reason
):
    (
        subject,
        parent,
        parent_path,
        _runtime,
        topology,
        structural,
        context,
    ) = _candidate_harness(tmp_path, monkeypatch, case)
    if case == "evidence":
        monkeypatch.setattr(
            subject,
            "_parent_evidence",
            lambda *args: (_ for _ in ()).throw(RuntimeError("evidence failed")),
        )
    elif case == "mutation-block":
        monkeypatch.setattr(
            subject,
            "_mutate",
            lambda *args: (_ for _ in ()).throw(ValueError("cannot mutate")),
        )
    elif case == "mutation-fail":
        monkeypatch.setattr(
            subject,
            "_mutate",
            lambda *args: (_ for _ in ()).throw(RuntimeError("mutation failed")),
        )
    elif case == "neuron-cap":
        topology.neuron_count = 9
    elif case == "edge-cap":
        topology.recurrent_value = np.ones(9)
    else:
        structural.enforce_biological_connection_ceiling = lambda *args: (
            _ for _ in ()
        ).throw(ValueError("biological cap"))

    attempt = subject.run_candidate(parent, "add", object(), context)

    assert attempt.status == status
    assert reason in attempt.reason
    assert attempt.executed_updates == 0
    assert parent_path.read_bytes() == b"immutable-parent"


def test_run_candidate_completes_with_durable_lineage_and_cleans_partial_failure(
    tmp_path, monkeypatch
):
    subject, parent, parent_path, *_rest, context = _candidate_harness(
        tmp_path, monkeypatch, "success"
    )

    def write_candidate(_runtime, *, candidate_id, path, **kwargs):
        del kwargs
        path.write_bytes(b"trained-child")
        return SimpleNamespace(
            candidate_id=candidate_id,
            checkpoint_path=str(path),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            score=object(),
            topology_changed=True,
        )

    monkeypatch.setattr(subject, "_write_runtime", write_candidate)
    completed = subject.run_candidate(parent, "add", object(), context)

    assert completed.status == "completed"
    assert completed.executed_updates == 128
    child = completed.candidate
    child_path = Path(child.checkpoint_path)
    assert child_path.read_bytes() == b"trained-child"
    assert subject._provenance_path(child_path).is_file()
    assert subject._parent_by_checkpoint[child.checkpoint_sha256] == (
        parent.checkpoint_sha256
    )
    assert parent_path.read_bytes() == b"immutable-parent"

    failed_subject, failed_parent, _, *_failed_rest, failed_context = (
        _candidate_harness(tmp_path, monkeypatch, "partial-failure")
    )
    monkeypatch.setattr(failed_subject, "_write_runtime", write_candidate)

    def fail_after_partial_lineage(candidate, parent_sha256):
        failed_subject._parent_by_checkpoint[candidate.checkpoint_sha256] = (
            parent_sha256
        )
        failed_subject._evidence_by_checkpoint[candidate.checkpoint_sha256] = object()
        failed_subject._provenance_path(Path(candidate.checkpoint_path)).write_text(
            "{}", encoding="utf-8"
        )
        raise OSError("provenance sync failed")

    monkeypatch.setattr(failed_subject, "_record_lineage", fail_after_partial_lineage)
    failed = failed_subject.run_candidate(
        failed_parent, "add", object(), failed_context
    )

    assert failed.status == "failed"
    assert "provenance sync failed" in failed.reason
    assert failed.executed_updates == 128
    failed_path = failed_context.output_dir / ".candidates" / "r001-edge-add.npz"
    assert not failed_path.exists()
    assert not failed_subject._provenance_path(failed_path).exists()
    assert not failed_subject._temporary_paths
    assert not failed_subject._parent_by_checkpoint
    assert not failed_subject._evidence_by_checkpoint


def test_run_candidate_reports_literal_partial_updates_on_failure(
    tmp_path, monkeypatch
):
    subject, parent, parent_path, *_rest, context = _candidate_harness(
        tmp_path, monkeypatch, "partial-updates"
    )
    candidate_runtime = SimpleNamespace(
        trainer=SimpleNamespace(updates=np.asarray(41, dtype=np.int32))
    )
    monkeypatch.setattr(subject, "_runtime_from_state", lambda *args: candidate_runtime)

    def fail_after_partial_updates(runtime, _schedule):
        runtime.trainer.updates = np.asarray(78, dtype=np.int32)
        raise RuntimeError("compiled update failed")

    monkeypatch.setattr(subject, "_run_update_block", fail_after_partial_updates)

    attempt = subject.run_candidate(parent, "add", object(), context)

    assert attempt.status == "failed"
    assert attempt.reason == "RuntimeError: compiled update failed"
    assert attempt.executed_updates == 37
    assert parent_path.read_bytes() == b"immutable-parent"


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (np.asarray([1], dtype=np.int32), "integer scalar"),
        (np.asarray(True), "integer scalar"),
        (np.asarray(-1, dtype=np.int32), "nonnegative"),
    ),
)
def test_candidate_update_counter_rejects_invalid_state(updates, message):
    runtime = SimpleNamespace(trainer=SimpleNamespace(updates=updates))

    with pytest.raises(RuntimeError, match=message):
        adapter.Example21ArcAdapter._trainer_update_count(runtime)


@pytest.mark.parametrize("after", (10, 140))
def test_candidate_update_delta_rejects_decrease_or_overrun(after):
    runtime = SimpleNamespace(
        trainer=SimpleNamespace(updates=np.asarray(after, dtype=np.int32))
    )

    with pytest.raises(RuntimeError, match="delta must be from 0 to 128"):
        adapter.Example21ArcAdapter._executed_update_delta(runtime, 11)


def test_run_candidate_detects_parent_mutation_during_evidence(tmp_path, monkeypatch):
    subject, parent, parent_path, *_ = _candidate_harness(
        tmp_path, monkeypatch, "immutable"
    )

    def corrupt_parent(*args):
        del args
        parent_path.write_bytes(b"mutated")
        raise RuntimeError("evidence failed")

    monkeypatch.setattr(subject, "_parent_evidence", corrupt_parent)
    context = _[-1]
    with pytest.raises(RuntimeError, match="evidence changed"):
        subject.run_candidate(parent, "add", object(), context)


def test_compiled_direct_scorer_aggregates_queries_by_task_without_updates(
    tmp_path, monkeypatch
):
    import brainstate
    import jax.numpy as jnp

    neuron_count = 360
    records = []
    for index, (task_id, color) in enumerate((("a", 2), ("b", 7))):
        events = np.zeros((32, 441), dtype=np.float32)
        events[0, 0] = index
        records.append(
            adapter.SupervisedQuery(
                task_id,
                0,
                events,
                np.ones(32, dtype=bool),
                np.asarray([[color]], dtype=np.int32),
            )
        )
    weights = SimpleNamespace(value=jnp.eye(neuron_count, dtype=jnp.float32))
    bias = SimpleNamespace(value=jnp.zeros(360, dtype=jnp.float32))
    model = SimpleNamespace(
        readout_weight=weights,
        readout_bias=bias,
        reset_episode=lambda learner: None,
    )
    trainer = SimpleNamespace(
        parameters={"sentinel": jnp.asarray([1.0])},
        optimizer_is_finite=lambda: True,
    )
    runtime = adapter._Runtime(
        model=model,
        learner=object(),
        trainer=trainer,
        topology=object(),
        optimizer=object(),
        readout_bias=np.zeros(360, dtype=np.float32),
    )
    topology = SimpleNamespace(
        neuron_count=neuron_count,
        input_target=np.asarray([0], dtype=np.int32),
        recurrent_source=np.asarray([0], dtype=np.int32),
        recurrent_target=np.asarray([1], dtype=np.int32),
    )
    optimizer = SimpleNamespace(
        neuron_first=np.zeros((neuron_count, 360), dtype=np.float32),
        input_first=np.zeros(1, dtype=np.float32),
        recurrent_first=np.zeros(1, dtype=np.float32),
    )

    def run_event_sequence(_model, events, advances, *, return_spikes):
        del advances
        index = int(np.asarray(events)[0, 0])
        color = (2, 7)[index]
        features = np.full((31, 360), -0.5, dtype=np.float32)
        features[0, 0] = 0.5
        features[0, 30] = 0.5
        features[1, 60 + color] = 0.5
        voltage = np.arctanh(features) * 20.0 - 65.0
        voltage = np.vstack((np.zeros((1, neuron_count)), voltage))
        spikes = np.zeros((32, neuron_count), dtype=np.float32)
        spikes[:, index] = 1.0
        assert return_spikes
        return jnp.asarray(voltage), jnp.asarray(spikes)

    def eager_for_loop(function, *values):
        outputs = [
            function(*(value[index] for value in values))
            for index in range(len(values[0]))
        ]
        return tuple(
            jnp.stack([output[position] for output in outputs])
            for position in range(len(outputs[0]))
        )

    fake_module = SimpleNamespace(
        run_event_sequence=run_event_sequence,
        decode_prediction=_decode,
    )
    fake_structural = SimpleNamespace(
        optimizer_from_muon_groups=lambda value: optimizer,
        task_owners=lambda scores: tuple(
            tuple(np.flatnonzero(column == np.max(column)))
            for column in np.asarray(scores).T
        ),
        topology_from_model=lambda value: topology,
        effective_topology_recurrent_values=lambda value: np.asarray([0.5]),
    )
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=fake_structural,
        model_module=fake_module,
    )
    monkeypatch.setattr(
        subject,
        "_manifest",
        lambda role: SimpleNamespace(task_ids=("a", "b"), role=role),
    )
    monkeypatch.setattr(subject, "_encoded_queries", lambda role: tuple(records))
    monkeypatch.setattr(brainstate.transform, "jit", lambda function: function)
    monkeypatch.setattr(brainstate.transform, "for_loop", eager_for_loop)

    result = subject._score_runtime(runtime, "training")

    assert result.score.exact_count == 2
    assert result.score.finite
    assert result.score.task_ids == ("a", "b")
    assert all(loss > 0 for loss in result.score.task_loss)
    assert np.array_equal(result.owner_codes[:2], np.asarray([0, 1]))
    assert np.array_equal(trainer.parameters["sentinel"], np.asarray([1.0]))

    screened = subject._score_runtime(runtime, "training", task_ids=("a",))

    assert screened.score.task_ids == ("a",)
    assert screened.score.task_exact == result.score.task_exact[:1]
    assert screened.score.task_loss == result.score.task_loss[:1]
    assert screened.owner_codes.shape == result.owner_codes.shape
    assert np.array_equal(trainer.parameters["sentinel"], np.asarray([1.0]))
    with pytest.raises(ValueError, match="manifest-ordered members"):
        subject._score_runtime(runtime, "training", task_ids=("b", "a"))
    with pytest.raises(ValueError, match="manifest-ordered members"):
        subject._score_runtime(runtime, "training", task_ids=("z",))


def test_update_block_runs_one_compiled_128_query_schedule(tmp_path, monkeypatch):
    import brainstate
    import jax.numpy as jnp

    events = np.zeros((32, 441), dtype=np.float32)
    query = adapter.SupervisedQuery(
        "task",
        0,
        events,
        np.ones(32, dtype=bool),
        np.asarray([[3]], dtype=np.int32),
    )
    entries = tuple(SimpleNamespace(task_id="task", query_index=0) for _ in range(128))
    schedule = SimpleNamespace(entries=entries, cursor_start=7, cursor_end=135)
    calls = {"step": 0, "direct": 0}

    class _Trainer:
        def __init__(self):
            self.updates = np.asarray(4, dtype=np.int32)

        def reset_episode(self):
            return None

        def update_episode(self, **payload):
            loss, features = payload["step_fn"](
                payload["events"][-1],
                payload["advance_mask"][-1],
                payload["request_kind"][-1],
                payload["target_shape"][-1],
                payload["target_rows"][-1],
                payload["target_valid_mask"][-1],
            )
            payload["direct_grad_fn"](
                aux=jnp.stack([features]),
                request_kind=payload["request_kind"][-1:],
                target_shape=payload["target_shape"][-1:],
                target_rows=payload["target_rows"][-1:],
                target_valid_mask=payload["target_valid_mask"][-1:],
                mask=payload["loss_mask"][-1:],
            )
            self.updates += 1
            return loss, jnp.asarray(0.0)

        @staticmethod
        def optimizer_is_finite():
            return True

    model = SimpleNamespace(
        readout_weight=SimpleNamespace(value=jnp.zeros((2, 360))),
        readout_bias=SimpleNamespace(value=jnp.zeros(360)),
        readout_features=lambda: jnp.zeros(2),
    )

    def learner(event):
        calls["step"] += 1
        return event

    def direct_gradients(*args):
        del args
        calls["direct"] += 1
        return {}

    fake_module = SimpleNamespace(
        _supervised_request_loss=lambda *args: jnp.asarray(0.0),
        _direct_readout_gradients=direct_gradients,
    )
    runtime = adapter._Runtime(
        model=model,
        learner=learner,
        trainer=_Trainer(),
        topology=object(),
        optimizer=object(),
        readout_bias=np.zeros(360),
    )
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        model_module=fake_module,
    )
    monkeypatch.setattr(subject, "_encoded_queries", lambda role: (query,))

    def eager_loop(function, values):
        outputs = []
        length = len(next(iter(values.values())))
        for index in range(length):
            outputs.append(
                function({name: value[index] for name, value in values.items()})
            )
        return outputs

    monkeypatch.setattr(brainstate.transform, "jit", lambda function: function)
    monkeypatch.setattr(brainstate.transform, "for_loop", eager_loop)
    monkeypatch.setattr(
        brainstate.transform,
        "cond",
        lambda predicate, true, false: true() if bool(predicate) else false(),
    )

    subject._run_update_block(runtime, schedule)

    assert int(runtime.trainer.updates) == 132
    assert calls == {"step": 128, "direct": 128}

    with pytest.raises(ValueError, match="exactly 128"):
        subject._run_update_block(
            runtime,
            SimpleNamespace(entries=entries[:-1], cursor_start=0, cursor_end=127),
        )
    with pytest.raises(ValueError, match="outside the training manifest"):
        subject._run_update_block(
            runtime,
            SimpleNamespace(
                entries=tuple(
                    SimpleNamespace(task_id="missing", query_index=0)
                    for _ in range(128)
                ),
                cursor_start=0,
                cursor_end=128,
            ),
        )

    original_update = runtime.trainer.update_episode
    runtime.trainer.update_episode = lambda **payload: (
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    with pytest.raises(RuntimeError, match="completed 0 updates"):
        subject._run_update_block(runtime, schedule)
    runtime.trainer.update_episode = original_update
    runtime.trainer.optimizer_is_finite = lambda: False
    with pytest.raises(RuntimeError, match="non-finite Muon"):
        subject._run_update_block(runtime, schedule)


def test_render_topology_uses_identity_verified_checkpoint(tmp_path, monkeypatch):
    calls = {}
    verified = SimpleNamespace(
        checkpoint_path=str(tmp_path / "verified.npz"),
        score=SimpleNamespace(exact_count=17, task_ids=tuple(range(64))),
    )
    topology = object()
    model = object()

    def topology_from_checkpoint(model_value, path):
        calls["load"] = (model_value, path)
        return topology

    def plot_topology(value, path, *, title):
        calls["plot"] = (value, path, title)

    structural = SimpleNamespace(
        topology_from_checkpoint=topology_from_checkpoint,
        plot_topology=plot_topology,
    )
    subject = adapter.Example21ArcAdapter(
        tmp_path,
        contracts_module=_Contracts(tmp_path),
        structural_module=structural,
        model_module=model,
    )
    candidate = SimpleNamespace(checkpoint_path="unverified.npz")
    monkeypatch.setattr(subject, "restore", lambda value: verified)
    output = tmp_path / "topology.png"

    subject.render_topology(candidate, output)

    assert calls["load"] == (model, verified.checkpoint_path)
    assert calls["plot"] == (
        topology,
        output,
        "Example 21 accepted BrainCell topology (17/64 scored training tasks)",
    )


def test_terminal_evaluation_requires_manifest_and_preserves_checkpoint(
    tmp_path, monkeypatch
):
    from examples.pp_prop import example21_evolve

    path = tmp_path / "accepted.npz"
    path.write_bytes(b"accepted")
    candidate = SimpleNamespace(checkpoint_path=str(path))
    expected = SimpleNamespace(digest="evaluation-digest")
    runtime = object()
    score = example21_evolve.ScoreSnapshot(
        ("task-a", "task-b"),
        (True, False),
        (0.0, 2.0),
    )
    scored = SimpleNamespace(score=score)
    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    monkeypatch.setattr(subject, "evaluation_manifest", lambda: expected)
    monkeypatch.setattr(subject, "restore", lambda value: value)
    monkeypatch.setattr(subject, "_runtime_from_checkpoint", lambda value: runtime)
    monkeypatch.setattr(subject, "_score_runtime", lambda *args, **options: scored)

    with pytest.raises(ValueError, match="manifest differs"):
        subject.evaluate_terminal(candidate, SimpleNamespace(digest="other"))

    result = subject.evaluate_terminal(candidate, expected)
    assert result == {
        "task_count": 2,
        "strict_task_pass_at_1_count": 1,
        "task_ids": ["task-a", "task-b"],
        "task_exact": [True, False],
        "task_loss": [0.0, 2.0],
        "mean_unresolved_task_loss": 2.0,
        "finite": True,
    }
    assert path.read_bytes() == b"accepted"

    def mutate_and_fail(*args, **options):
        del args, options
        path.write_bytes(b"changed")
        raise ValueError("scoring failed")

    monkeypatch.setattr(subject, "_score_runtime", mutate_and_fail)
    with pytest.raises(RuntimeError, match="changed the accepted checkpoint"):
        subject.evaluate_terminal(candidate, expected)


def test_rescore_changes_only_the_score_scope_of_its_accepted_state(
    tmp_path, monkeypatch
):
    from examples.pp_prop import example21_evolve

    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    parent_path = tmp_path / "accepted.npz"
    parent_path.write_bytes(b"accepted-checkpoint")
    parent = _snapshot(
        example21_evolve,
        candidate_id="accepted",
        path=parent_path,
        task_ids=("a", "b", "c"),
    )
    scopes: list[tuple[str, ...] | None] = []

    def fake_write_runtime(runtime, **options):
        del runtime
        scopes.append(tuple(options["task_ids"] or ()))
        Path(options["path"]).write_bytes(b"rescored-checkpoint")
        scored_ids = tuple(options["task_ids"]) or ("a", "b", "c")
        return replace(
            parent,
            candidate_id=options["candidate_id"],
            checkpoint_path=str(options["path"]),
            checkpoint_sha256=hashlib.sha256(b"rescored-checkpoint").hexdigest(),
            score=example21_evolve.ScoreSnapshot(
                task_ids=scored_ids,
                task_exact=(False,) * len(scored_ids),
                task_loss=(1.0,) * len(scored_ids),
            ),
        )

    monkeypatch.setattr(subject, "restore", lambda candidate: candidate)
    monkeypatch.setattr(subject, "_runtime_from_checkpoint", lambda path: object())
    monkeypatch.setattr(subject, "_write_runtime", fake_write_runtime)
    monkeypatch.setattr(subject, "_record_lineage", lambda candidate, parent_sha: None)

    context = example21_evolve.StageContext(
        0,
        "round-screen",
        "r000-round-screen",
        tmp_path,
        example21_evolve.PipelineConfig(),
        score_task_ids=("a",),
    )
    attempt = subject.rescore(parent, context)

    assert scopes == [("a",)]
    assert attempt.name == "rescore"
    assert attempt.status == "completed"
    assert attempt.executed_updates == 0
    assert attempt.candidate.candidate_id == "r000-round-screen-rescore"
    assert attempt.candidate.score.task_ids == ("a",)
    assert attempt.candidate.topology_sha256 == parent.topology_sha256
    assert attempt.candidate.parameters_sha256 == parent.parameters_sha256
    assert attempt.candidate.optimizer_sha256 == parent.optimizer_sha256
    assert parent_path.read_bytes() == b"accepted-checkpoint"


def test_rescore_isolates_a_failed_scope_transition(tmp_path, monkeypatch):
    from examples.pp_prop import example21_evolve

    subject = adapter.Example21ArcAdapter(
        tmp_path, contracts_module=_Contracts(tmp_path)
    )
    parent_path = tmp_path / "accepted.npz"
    parent_path.write_bytes(b"accepted-checkpoint")
    parent = _snapshot(
        example21_evolve,
        candidate_id="accepted",
        path=parent_path,
        task_ids=("a", "b", "c"),
    )

    def failing_write_runtime(runtime, **options):
        del runtime
        Path(options["path"]).write_bytes(b"partial")
        raise ValueError("scoring failed")

    monkeypatch.setattr(subject, "restore", lambda candidate: candidate)
    monkeypatch.setattr(subject, "_runtime_from_checkpoint", lambda path: object())
    monkeypatch.setattr(subject, "_write_runtime", failing_write_runtime)

    context = example21_evolve.StageContext(
        0,
        "round-score",
        "r000-round-score",
        tmp_path,
        example21_evolve.PipelineConfig(),
    )
    attempt = subject.rescore(parent, context)

    assert attempt.status == "failed"
    assert "scoring failed" in attempt.reason
    assert attempt.candidate is None
    assert not (tmp_path / ".candidates" / "r000-round-score-rescore.npz").exists()
    assert parent_path.read_bytes() == b"accepted-checkpoint"


def _snapshot(example21_evolve, *, candidate_id, path, task_ids):
    return example21_evolve.CandidateSnapshot(
        candidate_id=candidate_id,
        checkpoint_path=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        topology_sha256="1" * 64,
        parameters_sha256="2" * 64,
        optimizer_sha256="3" * 64,
        score=example21_evolve.ScoreSnapshot(
            task_ids=task_ids,
            task_exact=(False,) * len(task_ids),
            task_loss=(1.0,) * len(task_ids),
        ),
        resources=example21_evolve.ResourceUsage(
            persistent_bytes=1_000,
            checkpoint_bytes=500,
            neurons=10,
            recurrent_edges=20,
        ),
        topology_changed=False,
    )
