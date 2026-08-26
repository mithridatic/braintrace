import hashlib
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "example21_structural", Path(__file__).with_name("example21_structural.py")
)
assert _SPEC is not None and _SPEC.loader is not None
structural = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(structural)


def test_evidence_scores_owners_ranking_and_twins_are_direct_and_stable():
    rows = np.array([[0.0, 2.0, 1.0], [0.0, 4.0, 4.0]])
    np.testing.assert_allclose(
        structural.normalize_task_rows(rows), [[0.0, 1.0, 0.5], [0.0, 1.0, 1.0]]
    )
    score = structural.neuron_contribution(rows, rows * 2, rows * 3)
    np.testing.assert_allclose(score, [0.0, 1.0, 1.0])
    assert structural.task_owners(np.array([[0, 1, 1], [0, 1, 0]])) == (
        (), (0, 1), (0,)
    )
    assert structural.stable_rank(np.array([1.0, 1.0, 0.0])) == (2, 0, 1)
    twins = structural.structural_twins(
        3,
        input_sources=((0,), (0,), (1,)),
        recurrent_incoming=((2,), (2,), ()),
        recurrent_outgoing=((2,), (2,), (0, 1)),
        dale_labels=(0, 0, 0),
        mechanisms=(("hh",), ("hh",), ("hh",)),
    )
    assert twins == ((0, 1), (2,))


def test_connection_contribution_uses_csr_row_as_source():
    indptr = np.array([0, 2, 3])
    values = np.array([2.0, -1.0, 4.0])
    spikes = np.array([[3.0, 5.0], [1.0, 0.0]])
    gradients = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    scores = structural.connection_contribution(indptr, values, spikes, gradients)
    np.testing.assert_allclose(scores, [0.45, 0.6, 0.75])


def test_structural_evidence_aggregates_real_sparse_edges_by_source_and_target():
    topology = structural.SparseTopology(
        np.array([0]), np.array([0]), np.array([1.0]),
        np.array([0, 0, 1]), np.array([1, 2, 0]), np.array([2.0, 1.0, 3.0]),
        np.ones((3, 1)), np.zeros(3), ((),) * 3,
    )
    evidence = structural.structural_evidence(
        topology,
        readout_effect=np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0]]),
        spikes=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
        gradient_mass=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
    )
    assert evidence["neuron_scores"].shape == (3,)
    assert evidence["connection_scores"].shape == (3,)
    assert evidence["owners"] == ((0,), (1,), (0,))
    with pytest.raises(ValueError, match="task-by-edge"):
        structural.structural_evidence(topology, np.zeros((1, 3)),
                                       np.zeros((1, 3)), np.zeros((1, 2)))


def test_pruning_masks_exact_ceiling_and_validation_guard():
    scores = np.arange(21.0)
    mask = structural.pruning_mask(scores, validation_strict=(True, False))
    assert np.flatnonzero(~mask).tolist() == [0, 1]
    with pytest.raises(ValueError, match="validation"):
        structural.pruning_mask(scores, validation_strict=(False, False))
    with pytest.raises(ValueError, match="positive"):
        structural.mutation_count(0)


def test_neuron_pruning_is_fail_closed_and_requires_one_score_per_neuron():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.ones((21, 1)), np.zeros(21), ((),) * 21,
    )
    with pytest.raises(ValueError, match="validation"):
        structural.prune_neurons(topology, np.arange(21.0), (False, False))
    with pytest.raises(ValueError, match="one score"):
        structural.prune_neurons(topology, np.arange(20.0), (True, False))
    mask = structural.prune_neurons(topology, np.arange(21.0), (True, False))
    assert np.flatnonzero(~mask).tolist() == [0, 1]


def test_compaction_remaps_sparse_edges_and_optimizer_rows():
    topology = structural.SparseTopology(
        input_source=np.array([0, 1, 1]), input_target=np.array([0, 1, 2]),
        input_value=np.array([1.0, 2.0, 3.0]),
        recurrent_source=np.array([0, 1, 2]), recurrent_target=np.array([1, 2, 0]),
        recurrent_value=np.array([4.0, 5.0, 6.0]),
        readout=np.arange(6.0).reshape(3, 2), dale=np.array([0, 1, -1]),
        mechanisms=(('a',), ('b',), ('c',)),
    )
    adam = structural.StructuralAdam(
        neuron_first=np.arange(6.0).reshape(3, 2),
        neuron_second=np.arange(6.0, 12.0).reshape(3, 2),
        input_first=np.array([10.0, 20.0, 30.0]), input_second=np.array([1.0, 2.0, 3.0]),
        recurrent_first=np.array([40.0, 50.0, 60.0]), recurrent_second=np.array([4.0, 5.0, 6.0]),
        step=7,
    )
    compact, mapped, reset = structural.compact(topology, np.array([True, False, True]), adam)
    assert compact.neuron_count == 2
    assert list(zip(compact.recurrent_source, compact.recurrent_target)) == [(1, 0)]
    np.testing.assert_array_equal(mapped.neuron_first, adam.neuron_first[[0, 2]])
    np.testing.assert_array_equal(mapped.recurrent_first, [60.0])
    assert mapped.step == 7 and reset


def test_neuron_mask_and_compaction_have_prediction_byte_identity():
    topology = structural.SparseTopology(
        input_source=np.array([0, 1]), input_target=np.array([0, 1]),
        input_value=np.array([1.0, 2.0]), recurrent_source=np.array([0, 1]),
        recurrent_target=np.array([1, 0]), recurrent_value=np.array([3.0, 4.0]),
        readout=np.array([[5.0], [6.0]]), dale=np.array([1, -1]),
        mechanisms=(("hh",), ("hh",)),
    )
    adam = structural.StructuralAdam(
        np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
        np.zeros(2), np.zeros(2), 2,
    )
    alive = np.array([True, False])
    masked = structural.mask_topology(topology, alive)
    compacted, _, _ = structural.compact(topology, alive, adam)
    assert structural.prediction_bytes_identical(
        masked, compacted,
        lambda value: value.readout.sum(axis=0),
        lambda value: value.readout.sum(axis=0),
    )


def test_recurrent_pruning_uses_exact_ceiling_and_preserves_other_arrays():
    topology = structural.SparseTopology(
        input_source=np.array([0]), input_target=np.array([0]), input_value=np.array([1.0]),
        recurrent_source=np.arange(21) % 3, recurrent_target=(np.arange(21) + 1) % 3,
        recurrent_value=np.arange(21.0), readout=np.ones((3, 1)),
        dale=np.ones(3), mechanisms=(("hh",),) * 3,
    )
    pruned, keep = structural.prune_recurrent(
        topology, np.arange(21.0), validation_strict=(False, True)
    )
    assert np.flatnonzero(~keep).tolist() == [0, 1]
    assert len(pruned.recurrent_value) == 19
    np.testing.assert_array_equal(pruned.input_value, topology.input_value)
    assert structural.prune_neurons(topology, [3.0, 2.0, 1.0], (True,)).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="validation"):
        structural.prune_neurons(topology, [1.0, 2.0, 3.0], (False,))
    with pytest.raises(ValueError, match="per neuron"):
        structural.prune_neurons(topology, [1.0], (True,))
    with pytest.raises(ValueError, match="validation"):
        structural.prune_recurrent(topology, np.arange(21.0), (False,))
    with pytest.raises(ValueError, match="per recurrent"):
        structural.prune_recurrent(topology, np.arange(2.0), (True,))


def test_real_model_snapshot_and_task_mass_helpers_are_sparse():
    class Value:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Csr:
        def __init__(self, indices, indptr):
            self.indices = np.asarray(indices)
            self.indptr = np.asarray(indptr)

    class Model:
        input_csr = Csr([0, 1, 1], [0, 2, 3])
        recurrent_csr = Csr([1, 0], [0, 1, 2])
        input_weight = Value([1.0, 2.0, 3.0])
        recurrent_weight = Value([4.0, 5.0])
        readout_weight = Value([[6.0], [7.0]])

    topology = structural.topology_from_model(Model())
    assert topology.neuron_count == 2
    np.testing.assert_array_equal(topology.input_source, [0, 0, 1])
    np.testing.assert_array_equal(topology.recurrent_source, [0, 1])
    mass = structural.task_gradient_mass({"recurrent": [[-1.0, 2.0], [3.0, -4.0]]}, "recurrent", 2)
    np.testing.assert_array_equal(mass, [[1.0, 2.0], [3.0, 4.0]])
    assert structural.task_gradient_mass({}, "missing", 2).shape == (2, 0)
    with pytest.raises(ValueError, match="task dimension"):
        structural.task_gradient_mass({"recurrent": [[1.0]]}, "recurrent", 2)
    assert structural.resident_tile_pairs(256) == 65_536
    with pytest.raises(ValueError, match="between"):
        structural.resident_tile_pairs(0)

    evidence = structural.structural_evidence(
        topology,
        readout_effect=np.array([[1.0, 2.0], [2.0, 0.0]]),
        spikes=np.array([[1.0, 0.0], [0.5, 2.0]]),
        gradient_mass=np.array([[3.0, 1.0], [2.0, 4.0]]),
    )
    assert evidence["neuron_scores"].shape == (2,)
    assert evidence["connection_scores"].shape == (2,)
    assert len(evidence["owners"]) == 2
    with pytest.raises(ValueError, match="task-by-neuron"):
        structural.structural_evidence(topology, [[1.0]], [[1.0]], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="task-by-edge"):
        structural.structural_evidence(
            topology, [[1.0, 2.0]], [[1.0, 2.0]], [[1.0]]
        )


def test_twin_addition_is_connected_splits_values_and_zeros_new_moments():
    topology = structural.SparseTopology(
        input_source=np.array([0]), input_target=np.array([0]), input_value=np.array([2.0]),
        recurrent_source=np.array([0]), recurrent_target=np.array([1]), recurrent_value=np.array([4.0]),
        readout=np.array([[6.0], [2.0]]), dale=np.array([1, 0]),
        mechanisms=(('hh',), ('hh',)),
    )
    grown, donors = structural.add_twin_neurons(
        topology, np.array([[3.0, 1.0]]), required=1
    )
    assert donors == (0,) and grown.neuron_count == 3
    assert (0, 2.0) in list(zip(grown.input_target, grown.input_value))
    np.testing.assert_array_equal(grown.readout[[0, 2]], [[3.0], [3.0]])
    assert (2, 1) in list(zip(grown.recurrent_source, grown.recurrent_target))
    np.testing.assert_array_equal(
        grown.recurrent_value[np.isin(grown.recurrent_source, [0, 2])], [2.0, 2.0]
    )
    adam = structural.StructuralAdam(
        np.ones((2, 1)), np.ones((2, 1)), np.ones(1), np.ones(1),
        np.ones(1), np.ones(1), 9,
    )
    mapped = structural.grow_adam_for_twins(adam, topology, grown)
    assert mapped.step == 9
    assert mapped.neuron_first[-1, 0] == 0
    assert np.all(mapped.recurrent_first[1:] == 0)
    with pytest.raises(ValueError, match="budget"):
        structural.add_twin_neurons(topology, [1.0, 0.0], required=0)
    with pytest.raises(ValueError, match="connected"):
        structural.add_twin_neurons(topology, [2.0, 1.0], required=2)


def test_twin_connectivity_guard_handles_numpy_edge_arrays():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([2, 3]), np.ones(2),
        np.ones((4, 1)), np.zeros(4), ((),) * 4,
    )
    grown, donors = structural.add_twin_neurons(topology, np.arange(4.0), required=2)
    assert donors == (3, 2)
    assert grown.neuron_count == 6


def test_tiled_connection_addition_is_global_stable_and_never_dense(monkeypatch):
    original_empty = np.empty
    def guarded_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == (300, 300):
            raise AssertionError("dense pair allocation")
        return original_empty(shape, *args, **kwargs)
    monkeypatch.setattr(np, "empty", guarded_empty)
    selected = structural.select_connection_additions(
        300, existing={(0, 1)}, source_evidence=np.arange(300.0, 0.0, -1.0),
        target_evidence=np.arange(300.0, 0.0, -1.0), required=3, tile_size=256,
    )
    assert selected == ((1, 0), (0, 2), (2, 0))
    assert structural.resident_tile_pairs(256) == 65536
    with pytest.raises(ValueError, match="256"):
        structural.resident_tile_pairs(257)


def test_connection_addition_appends_exact_items_and_zero_moments():
    topology = structural.SparseTopology(
        input_source=np.array([], dtype=int), input_target=np.array([], dtype=int),
        input_value=np.array([]), recurrent_source=np.array([0]),
        recurrent_target=np.array([1]), recurrent_value=np.array([2.0]),
        readout=np.ones((3, 1)), dale=np.zeros(3), mechanisms=((), (), ()),
    )
    grown = structural.add_recurrent_connections(topology, ((1, 2), (2, 0)))
    assert list(zip(grown.recurrent_source, grown.recurrent_target))[-2:] == [(1, 2), (2, 0)]
    np.testing.assert_array_equal(grown.recurrent_value[-2:], [0.0, 0.0])
    adam = structural.StructuralAdam(
        np.ones((3, 1)), np.ones((3, 1)), np.array([]), np.array([]),
        np.ones(1), np.ones(1), 4,
    )
    mapped = structural.grow_adam_for_connections(adam, 2)
    np.testing.assert_array_equal(mapped.recurrent_first, [1.0, 0.0, 0.0])
    typed = structural.add_recurrent_connections(topology, ((2, 1),), typed=True)
    np.testing.assert_allclose(np.log1p(np.exp(typed.recurrent_value[-1])), 1e-6)
    with pytest.raises(ValueError, match="distinct"):
        structural.add_recurrent_connections(topology, ((0, 1),))
    with pytest.raises(ValueError, match="65,536"):
        structural.select_connection_additions(2, set(), [1, 1], [1, 1], 1, 257)


def test_preclip_mass_is_taken_directly_from_real_etrace_boundary():
    class Learner:
        def etrace_grad(self, events, **kwargs):
            assert kwargs["return_value"]
            return {("model", "recurrent_weight"): np.array([2.0, -3.0])}, 7.0

    mass, loss = structural.preclip_gradient_mass(
        Learner(), np.ones((1, 2)), lambda value: value, 1, 3,
        reduction="sum",
    )
    np.testing.assert_array_equal(
        mass["model/recurrent_weight"], [[0.0, 0.0], [2.0, 3.0], [0.0, 0.0]]
    )
    assert loss == 7.0


def test_collect_model_evidence_keeps_preclip_mass_and_task_scores():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = type("CSR", (), {"indptr": np.array([0, 1]), "indices": np.array([0])})()
        recurrent_csr = type("CSR", (), {
            "indptr": np.array([0, 1, 2]), "indices": np.array([1, 0])
        })()
        input_weight = State([1.0])
        recurrent_weight = State([2.0, 3.0])
        readout_weight = State(np.ones((2, 1)))

    class Learner:
        def etrace_grad(self, events, **kwargs):
            return {("model", "recurrent_weight"): np.array([2.0, -3.0])}, np.array([4.0])

    result = structural.collect_model_evidence(
        Model(), Learner(), np.ones((1, 1)), lambda value: value,
        np.ones((2, 2)), np.ones((2, 2)), reduction="sum",
    )
    assert result["model_neurons"] == 2
    assert result["preclip_exceeds_clip"]
    np.testing.assert_array_equal(result["gradient_mass"], [[2.0, 3.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="recurrent weight"):
        structural.collect_model_evidence(
            Model(), type("L", (), {"etrace_grad": lambda *a, **k: ({}, 0)})(),
            np.ones((1, 1)), lambda value: value, np.ones((2, 2)), np.ones((2, 2)),
        )


def test_addition_driver_and_one_arm_execution_use_compiled_64_updates():
    class Transform:
        calls: ClassVar[list] = []

        @staticmethod
        def for_loop(function, values):
            Transform.calls.append(("for_loop", len(values)))
            return np.asarray([function(value) for value in values])

        @staticmethod
        def jit(function):
            def compiled(values):
                Transform.calls.append(("jit", len(values)))
                return function(values)
            return compiled

    ticks = []
    evidence = structural.execute_one_arm(
        "connection-add", (False, True), lambda: (object(), 2),
        lambda candidate: (True, True), updates=64, transform=Transform,
        update=lambda index: ticks.append(int(index)) or index,
        clock=iter((10.0, 12.5)).__next__,
    )
    assert ticks == list(range(64))
    assert Transform.calls == [("jit", 64), ("for_loop", 64)]
    assert evidence["promoted"] and evidence["mutated_item_count"] == 2
    assert evidence["elapsed_seconds"] == 2.5
    with pytest.raises(ValueError, match="64"):
        structural.run_addition_updates(Transform, lambda value: value, updates=63)
    with pytest.raises(ValueError, match="recognized"):
        structural.execute_one_arm("two-arms", (), lambda: (None, 0), lambda _: ())
    with pytest.raises(ValueError, match="compiled"):
        structural.execute_one_arm(
            "neuron-add", (False,), lambda: (None, 1), lambda _: (True,), updates=64
        )
    pruning = structural.execute_one_arm(
        "neuron-prune", (True,), lambda: (None, 1), lambda _: (True,),
        clock=iter((0.0, 1.0)).__next__,
    )
    assert not pruning["promoted"] and pruning["updates"] == 0


def test_artifact_is_canonical_and_records_environment(tmp_path):
    target = tmp_path / "arm.json"
    digest = structural.write_artifact(target, {"arm": "neuron-prune"})
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    document = json.loads(target.read_text())
    assert document["environment"]["seeds"] == [21, 22, 23]
    assert document["arm"] == "neuron-prune"


def test_arm_gate_requires_gain_no_regression_limits_and_fixed_updates():
    assert structural.promote_arm((False, True), (True, True), 299.0, "addition", 64)
    assert not structural.promote_arm((False, True), (True, False), 1.0, "addition", 64)
    assert not structural.promote_arm((False,), (True,), 301.0, "addition", 64)
    assert not structural.promote_arm((True,), (True,), 1.0, "pruning", 0)
    with pytest.raises(ValueError, match="64"):
        structural.promote_arm((False,), (True,), 1.0, "addition", 63)


def test_integrated_arm_rebuilds_model_remaps_adam_and_resets_eligibility():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        def __init__(self):
            self.input_csr = type("CSR", (), {
                "indptr": np.array([0, 1, 2]),
                "indices": np.array([0, 1]),
            })()
            self.recurrent_csr = type("CSR", (), {
                "indptr": np.array([0, 1, 2, 3]),
                "indices": np.array([1, 2, 0]),
            })()
            self.input_weight = State([1.0, 1.0])
            self.recurrent_weight = State([1.0, 1.0, 1.0])
            self.readout_weight = State(np.ones((3, 1)))
            self.reset_count = 0

        def reset_episode(self, learner):
            self.reset_count += 1

    def rebuild(topology, adam):
        model = Model()
        model.input_csr.indices = topology.input_target
        model.recurrent_csr.indices = topology.recurrent_target
        model.input_weight.value = topology.input_value
        model.recurrent_weight.value = topology.recurrent_value
        model.readout_weight.value = topology.readout
        return model, object()

    result = structural.run_integrated_arm(
        "connection-prune", Model, lambda model: object(),
        lambda model, learner: (False, True), rebuild,
        before_strict=(False, True),
        evidence={
            "neuron_scores": np.array([1.0, 1.0, 1.0]),
            "connection_scores": np.array([0.1, 0.2, 0.3]),
            "validation_strict": (False, True),
        },
        clock=iter((0.0, 1.0)).__next__,
    )
    assert result["real_model"]
    assert result["adam_remapped"]
    assert result["eligibility_reset"]
    assert result["mutated_item_count"] == 1
    assert not result["promoted"]


def test_real_arm_runner_covers_all_bounded_paths(monkeypatch, tmp_path):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1, 2, 3, 4]), indices=np.arange(4))
        recurrent_csr = SimpleNamespace(indptr=np.arange(5), indices=np.array([1, 2, 3, 0]))
        input_weight = State(np.ones(4))
        recurrent_weight = State(np.ones(4))
        readout_weight = State(np.ones((4, 1)))

    fake_module = SimpleNamespace(
        BrainCellArcModel=Model, TRAINING_TASK_IDS=("a",), VALIDATION_TASK_IDS=("b",)
    )
    fake_evidence = {
        "strict": [False, True], "neuron_scores": np.ones(4),
        "connection_scores": np.ones(4), "gradient_mass": np.ones(4),
        "preclip_gradient_mass": [], "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    monkeypatch.setattr(structural, "_load_example21_model", lambda: fake_module)
    monkeypatch.setattr(structural, "_fixed_task_evidence", lambda *args: fake_evidence)
    monkeypatch.setattr(structural, "_real_pp_prop_update", lambda *args: lambda index: index)
    monkeypatch.setattr(structural, "run_addition_updates", lambda *args, **kwargs: None)
    for arm in ("neuron-prune", "connection-prune", "neuron-add", "connection-add"):
        result = structural.measure_real_arm(arm)
        assert result["real_model"] and result["updates"] == (64 if arm.endswith("add") else 0)
    structural.main(["baseline", "--output", str(tmp_path / "baseline.json")])


def test_fixed_task_evidence_decodes_model_readout_not_neuron_voltage(monkeypatch):
    class Model:
        readout_weight = SimpleNamespace(value=np.ones((2, 360)))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.zeros(360)

    task = SimpleNamespace(targets=(np.zeros((1, 1), dtype=np.uint8),))
    module = SimpleNamespace(
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
        load_task=lambda *args: task,
        encode_episode=lambda *args: (np.ones((1, 441)), np.ones(1, dtype=bool)),
        run_event_sequence=lambda *args: np.zeros((1, 2)),
        decode_prediction=lambda value: (
            np.zeros((1, 1), dtype=np.uint8)
            if np.asarray(value).shape == (360,)
            else (_ for _ in ()).throw(AssertionError("decoded neuron voltage"))
        ),
        strict_task_pass_at_1=lambda predictions, targets: True,
    )
    monkeypatch.setattr(
        structural,
        "preclip_gradient_mass",
        lambda *args, **kwargs: ({"model/recurrent_weight": np.ones((1, 2))}, 0.0),
    )
    monkeypatch.setattr(
        structural,
        "topology_from_model",
        lambda model: SimpleNamespace(neuron_count=2),
    )
    monkeypatch.setattr(
        structural,
        "structural_evidence",
        lambda *args: {"neuron_scores": np.ones(2), "connection_scores": np.ones(2)},
    )
    evidence = structural._fixed_task_evidence(module, Model(), object(), "data")
    assert evidence["strict"] == [True, True]


def test_real_model_uses_canonical_validation_task_ids():
    module = structural._load_example21_model()
    assert module.VALIDATION_TASK_IDS == (
        "46f33fce", "3428a4f5", "d8c310e9", "09629e4f"
    )


def test_real_pruning_arm_records_closed_validation_gate(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.arange(2))
        recurrent_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.array([1, 0]))
        input_weight = State(np.ones(2))
        recurrent_weight = State(np.ones(2))
        readout_weight = State(np.ones((2, 1)))

    module = SimpleNamespace(
        BrainCellArcModel=Model,
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
    )
    evidence = {
        "strict": [False, False],
        "neuron_scores": np.ones(2),
        "connection_scores": np.ones(2),
        "gradient_mass": np.ones(2),
        "preclip_gradient_mass": [],
        "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(structural, "_fixed_task_evidence", lambda *args: evidence)
    result = structural.measure_real_arm("neuron-prune")
    assert result["pruning_blocked"]
    assert result["mutated_item_count"] == 0
    assert result["candidate_neurons"] == result["baseline_neurons"]
