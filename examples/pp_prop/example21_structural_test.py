import hashlib
import importlib.util
import json
from collections import namedtuple
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


def test_real_mask_compaction_identity_screens_fixed_tasks(monkeypatch):
    topology = structural.SparseTopology(
        input_source=np.array([0, 1]), input_target=np.array([0, 1]),
        input_value=np.array([1.0, 2.0]), recurrent_source=np.array([0, 1]),
        recurrent_target=np.array([1, 0]), recurrent_value=np.array([3.0, 4.0]),
        readout=np.array([[1.0], [2.0]]), dale=np.array([1, -1]),
        mechanisms=((), ()),
    )
    adam = structural.StructuralAdam(
        np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
        np.zeros(2), np.zeros(2), 0,
    )

    class Model:
        def __init__(self, candidate):
            self.candidate = candidate

        def reset_episode(self, learner):
            pass

        def readout(self):
            return self.candidate.readout

    task = SimpleNamespace(targets=[np.array([[1]], dtype=np.uint8)])
    module = SimpleNamespace(
        TRAINING_TASK_IDS=("train",), VALIDATION_TASK_IDS=("valid",),
        load_task=lambda *_args: task,
        encode_episode=lambda *_args: (np.zeros((1, 1)), np.ones(1, dtype=bool)),
        run_event_sequence=lambda *_args: None,
        decode_prediction=lambda value: np.asarray([value.sum()], dtype=np.uint8),
        strict_task_pass_at_1=lambda predictions, targets: True,
    )
    monkeypatch.setattr(
        structural, "_rebuild_real_candidate",
        lambda _module, candidate, _learner: (Model(candidate), object()),
    )
    result = structural._real_mask_compaction_identity(
        module, topology, adam, "data", alive=np.array([True, False])
    )
    assert result["prediction_bytes_identical"]
    assert result["strict_identical"]
    assert result["masked_neurons"] == 2
    assert result["compacted_neurons"] == 1


def test_real_pp_prop_update_builds_episode_driver():
    import jax.numpy as jnp

    class Learner:
        def etrace_evolve(self, events, return_outputs):
            return (events,)

    class Model:
        input_weight = SimpleNamespace(value=jnp.ones(1))
        recurrent_weight = SimpleNamespace(value=jnp.ones(1))

        def reset_episode(self, learner):
            self.reset = learner

    class Trainer:
        def __init__(self, learner, parameters):
            self.learner = learner
            self.parameters = parameters

        def update_episode(self, events, step_fn, loss_mask):
            return events.shape, loss_mask.shape, step_fn(events[0])

    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    update = structural._real_pp_prop_update(
        module, Model(), Learner(), {"events": [[1.0]], "advances": [True]}
    )
    shape, mask_shape, value = update(0)
    assert shape == (1, 1)
    assert mask_shape == (1,)
    assert value.shape == ()


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


def test_tiled_connection_addition_stops_after_score_upper_bound():
    stats = {}
    selected = structural.select_connection_additions(
        1024, existing=set(), source_evidence=np.arange(1024.0, 0.0, -1.0),
        target_evidence=np.arange(1024.0, 0.0, -1.0), required=2,
        tile_size=256, stats=stats,
    )
    assert selected == ((0, 1), (1, 0))
    assert stats["tile_stop_bound"] is True
    assert stats["tiles_evaluated"] < 16
    assert stats["max_resident_tile_pairs"] == 65_536


@pytest.mark.parametrize(
    ("source", "target", "required", "message"),
    [
        ([1.0], [1.0, 2.0], 1, "match"),
        ([np.inf, 1.0], [1.0, 2.0], 1, "finite"),
        ([-1.0, 1.0], [1.0, 2.0], 1, "nonnegative"),
        ([1.0, 2.0], [1.0, 2.0], 0, "at least one"),
    ],
)
def test_tiled_connection_addition_rejects_invalid_evidence(
    source, target, required, message
):
    with pytest.raises(ValueError, match=message):
        structural.select_connection_additions(
            2, set(), source, target, required
        )


def test_tiled_connection_addition_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="neuron count"):
        structural.select_connection_additions(0, set(), [], [], 1)
    with pytest.raises(ValueError, match="tile size"):
        structural.select_connection_additions(2, set(), [1, 1], [1, 1], 1, 0)


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


def test_real_rebuild_canonicalizes_added_recurrent_edges_and_values():
    topology = structural.SparseTopology(
        input_source=np.array([], dtype=int), input_target=np.array([], dtype=int),
        input_value=np.array([], dtype=float), recurrent_source=np.array([0, 1]),
        recurrent_target=np.array([1, 2]), recurrent_value=np.array([10.0, 20.0]),
        readout=np.ones((3, 360)), dale=np.zeros(3), mechanisms=((), (), ()),
    )
    grown = structural.add_recurrent_connections(topology, ((0, 0),))
    expected = [(0, 0, 0.0), (0, 1, 10.0), (1, 2, 20.0)]
    assert list(zip(grown.recurrent_source, grown.recurrent_target, grown.recurrent_value)) == expected
    adam = structural.StructuralAdam(
        np.zeros((3, 360)), np.zeros((3, 360)), np.array([]), np.array([]),
        np.array([100.0, 200.0]), np.array([1.0, 2.0]), 4,
    )
    mapped = structural.grow_adam_for_connections(
        adam, 1, topology=topology, candidate=grown
    )
    np.testing.assert_array_equal(mapped.recurrent_first, [0.0, 100.0, 200.0])
    np.testing.assert_array_equal(mapped.recurrent_second, [0.0, 1.0, 2.0])

    module = structural._load_example21_model()
    model = module.BrainCellArcModel(grown)
    sources = np.repeat(
        np.arange(len(model.recurrent_csr.indptr) - 1),
        np.diff(np.asarray(model.recurrent_csr.indptr)),
    )
    assert list(zip(sources, np.asarray(model.recurrent_csr.indices),
                   np.asarray(model.recurrent_csr.data))) == expected


def test_real_rebuild_preserves_nonzero_readout_bias_for_mask_and_compaction():
    topology = structural.SparseTopology(
        input_source=np.array([], dtype=int), input_target=np.array([], dtype=int),
        input_value=np.array([], dtype=float), recurrent_source=np.array([], dtype=int),
        recurrent_target=np.array([], dtype=int), recurrent_value=np.array([], dtype=float),
        readout=np.arange(720.0).reshape(2, 360), dale=np.zeros(2),
        mechanisms=((), ()), readout_bias=np.arange(360.0) + 0.5,
    )
    adam = structural.StructuralAdam(
        np.zeros((2, 360)), np.zeros((2, 360)), np.array([]), np.array([]),
        np.array([]), np.array([]), 1,
    )
    alive = np.array([False, True])
    masked = structural.mask_topology(topology, alive)
    compacted, _, _ = structural.compact(topology, alive, adam)
    module = structural._load_example21_model()
    masked_model = module.BrainCellArcModel(masked)
    compacted_model = module.BrainCellArcModel(compacted)
    np.testing.assert_array_equal(masked_model.readout_bias.value, topology.readout_bias)
    np.testing.assert_array_equal(compacted_model.readout_bias.value, topology.readout_bias)
    assert np.asarray(masked_model.readout()).tobytes() == np.asarray(
        compacted_model.readout()
    ).tobytes()


def test_fixed_task_evidence_sums_multi_query_preclip_mass(monkeypatch):
    import brainstate
    import jax.numpy as jnp

    class State:
        def __init__(self, value):
            self.value = jnp.asarray(value)

    class Learner:
        def etrace_evolve(self, events, return_outputs=True):
            return (events,)

        def etrace_grad(self, events, step_fn, **kwargs):
            del kwargs
            step_fn(events[0])
            value = events[0, 0]
            return {("model", "recurrent_weight"): jnp.array([value, 0.0])}, value

    class Model:
        readout_weight = State(np.ones((2, 360)))

        def reset_episode(self, learner):
            del learner

        def readout(self):
            return jnp.zeros(360)

    class Module:
        TRAINING_TASK_IDS = ("train",)
        VALIDATION_TASK_IDS = ("valid",)
        decode_prediction = staticmethod(lambda value: np.zeros((1, 1), dtype=np.uint8))
        strict_task_pass_at_1 = staticmethod(lambda predictions, targets: True)

    episodes = [
        {"task_id": "train", "task_index": 0, "events": np.array([[2.0]]),
         "advances": np.ones(1, dtype=bool), "target": np.ones((1, 1)),
         "target_vector": np.zeros(360)},
        {"task_id": "train", "task_index": 0, "events": np.array([[4.0]]),
         "advances": np.ones(1, dtype=bool), "target": np.ones((1, 1)),
         "target_vector": np.zeros(360)},
        {"task_id": "valid", "task_index": 1, "events": np.array([[8.0]]),
         "advances": np.ones(1, dtype=bool), "target": np.ones((1, 1)),
         "target_vector": np.zeros(360)},
    ]
    monkeypatch.setattr(
        structural, "topology_from_model",
        lambda model: SimpleNamespace(
            neuron_count=2, recurrent_value=np.ones(2),
            recurrent_source=np.array([0, 1]), recurrent_target=np.array([1, 0]),
        ),
    )
    monkeypatch.setattr(
        structural, "_run_fixed_episode_batch",
        lambda *args: (
            np.zeros((3, 1, 2)), np.zeros((3, 1, 2)), np.zeros((3, 360)),
        ),
    )
    monkeypatch.setattr(
        structural, "structural_evidence",
        lambda *args: {
            "neuron_scores": np.ones(2), "connection_scores": np.ones(2),
            "neuron_scores_by_task": np.ones((2, 2)),
            "connection_scores_by_task": np.ones((2, 2)), "owners": ((), ()),
        },
    )
    monkeypatch.setattr(brainstate.transform, "jit", lambda function: function)
    monkeypatch.setattr(
        brainstate.transform, "for_loop",
        lambda function, values: tuple(
            np.asarray(items) for items in zip(
                *(function(value) for value in values)
            )
        ),
    )
    result = structural._fixed_task_evidence(
        Module, Model(), Learner(), "data", episodes=episodes
    )
    np.testing.assert_array_equal(result["preclip_gradient_mass"], [[6.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(result["target_scores_by_task"][0], [16.0, 16.0])


def test_wrong_output_target_evidence_uses_l1_weights_not_voltage_effect():
    weights = np.array([[2.0, -3.0, 100.0], [-4.0, 5.0, -200.0]])
    output_mask = np.array([True, True, False])
    np.testing.assert_array_equal(
        structural._wrong_output_readout_evidence(weights, output_mask),
        [5.0, 9.0],
    )


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


def test_real_pp_prop_update_seeds_remapped_adam_state():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.adam_groups = {
                "readout": type("Adam", (), {"first": np.zeros(1), "second": np.zeros(1), "step": 0})(),
                "input": type("Adam", (), {"first": np.zeros(2), "second": np.zeros(2), "step": 0})(),
                "recurrent": type("Adam", (), {"first": np.zeros(3), "second": np.zeros(3), "step": 0})(),
            }

        def update_episode(self, *args, **kwargs):
            return 0.0, 0.0

    model = SimpleNamespace(input_weight=State([1, 1]), recurrent_weight=State([1, 1, 1]))
    model.reset_episode = lambda learner: None
    learner = SimpleNamespace()
    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    adam = structural.StructuralAdam(
        np.array([1.0]), np.array([2.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0]),
        np.array([6.0, 7.0, 8.0]), np.array([9.0, 10.0, 11.0]), step=4,
    )
    update = structural._real_pp_prop_update(
        module, model, learner,
        {"events": np.zeros((1, 2)), "advances": np.ones(1)}, adam,
    )
    update()
    trainer = update.trainer
    np.testing.assert_array_equal(trainer.adam_groups["readout"].first, [1])
    np.testing.assert_array_equal(trainer.adam_groups["readout"].second, [2])
    assert trainer.adam_groups["readout"].step == 4
    np.testing.assert_array_equal(trainer.adam_groups["input"].first, [2, 3])
    np.testing.assert_array_equal(trainer.adam_groups["recurrent"].second, [9, 10, 11])
    assert trainer.adam_groups["recurrent"].step == 4


def test_muon_groups_are_remapped_for_structural_candidate():
    state = SimpleNamespace(
        mu=np.arange(6, dtype=float).reshape(3, 2),
        count=np.array(7, dtype=np.int32),
    )
    mapped = structural.remap_muon_groups(
        {"readout_weight": state},
        {"readout_weight": (np.array([True, False, True]), (2, 2))},
    )
    np.testing.assert_array_equal(mapped["readout_weight"].mu, [[0, 1], [4, 5]])
    assert mapped["readout_weight"].count == 7


def test_muon_remap_traverses_plain_tuples_inside_named_states():
    import jax.numpy as jnp

    State = namedtuple("State", "inner")
    state = State((jnp.arange(3.0).reshape(3, 1),))
    mapped = structural.remap_muon_groups(
        {"input": state}, {"input": (np.array([True, False, True]), (4, 1))}
    )
    np.testing.assert_array_equal(
        mapped["input"].inner[0], [[0.0], [2.0], [0.0], [0.0]]
    )


def test_real_update_hands_remapped_muon_groups_to_trainer():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.muon_groups = {}

        def update_episode(self, *args, **kwargs):
            return 0.0, 0.0

    model = SimpleNamespace(
        input_weight=State([1, 1]), recurrent_weight=State([1, 1, 1])
    )
    model.reset_episode = lambda learner: None
    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    state = SimpleNamespace(mu=np.arange(6, dtype=float).reshape(3, 2))
    update = structural._real_pp_prop_update(
        module, model, SimpleNamespace(),
        {"events": np.zeros((1, 2)), "advances": np.ones(1)},
        muon_groups={"readout_weight": state},
        parameter_maps={"readout_weight": (np.array([True, False, True]), (2, 2))},
    )
    np.testing.assert_array_equal(update.trainer.muon_groups["readout_weight"].mu,
                                  [[0, 1], [4, 5]])


def test_real_mask_compaction_identity_reuses_pruning_episode_snapshot(monkeypatch):
    episodes = []

    class Module:
        TRAINING_TASK_IDS = ("train",)
        VALIDATION_TASK_IDS = ("valid",)

        def load_task(self, root, task_id, split):
            return SimpleNamespace(targets=(task_id,))

        def encode_episode(self, task, query_index):
            episodes.append((task.targets[0], query_index))
            return np.ones((1, 1)), np.ones(1, dtype=bool)

        def strict_task_pass_at_1(self, predictions, targets):
            return True

        def decode_prediction(self, value):
            return np.asarray(value)

        def run_event_sequence(self, model, events, advances):
            return None

    class Model:
        def __init__(self):
            self.readout_weight = SimpleNamespace(value=np.ones((2, 1)))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.ones(1)

    module = Module()
    topology = SimpleNamespace(
        neuron_count=2, input_source=np.array([0, 0]), input_target=np.array([0, 1]),
        input_value=np.ones(2), recurrent_source=np.array([0]),
        recurrent_target=np.array([1]), recurrent_value=np.ones(1),
        readout=np.ones((2, 1)), dale=np.ones(2), mechanisms=((), ()),
    )
    monkeypatch.setattr(structural, "_rebuild_real_candidate", lambda *args: (Model(), object()))
    structural._real_mask_compaction_identity(
        module, topology, structural.StructuralAdam(
            np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
            np.zeros(1), np.zeros(1),
        ), "data", alive=np.array([True, False]),
    )
    assert episodes == [("train", 0), ("valid", 0)]


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
    monkeypatch.setattr(structural, "_real_pp_prop_update", lambda *args, **kwargs: lambda index: index)
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


def test_real_arm_evaluates_rebuilt_candidate_model_after_mutation(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        def __init__(self, topology=None):
            self.is_candidate = topology is not None
            self.input_csr = SimpleNamespace(
                indptr=np.array([0, 1, 2]), indices=np.array([0, 1])
            )
            self.recurrent_csr = SimpleNamespace(
                indptr=np.array([0, 1, 2]), indices=np.array([1, 0])
            )
            self.input_weight = State(np.ones(2))
            self.recurrent_weight = State(np.ones(2))
            self.readout_weight = State(np.ones((2, 1)))

        def reset_episode(self, learner):
            pass

    module = SimpleNamespace(
        BrainCellArcModel=Model,
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
        compile_pp_prop_model=lambda model: object(),
    )
    evidence = {
        "strict": [False], "neuron_scores": np.array([1.0, 0.0]),
        "connection_scores": np.ones(2), "gradient_mass": np.ones(2),
        "preclip_gradient_mass": [], "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(
        structural, "_fixed_task_evidence",
        lambda _module, model, _learner, _root: {
            **evidence, "strict": [True] if model.is_candidate else [False]
        },
    )
    monkeypatch.setattr(structural, "_real_pp_prop_update", lambda *args, **kwargs: lambda _: None)
    monkeypatch.setattr(structural, "run_addition_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        structural,
        "_real_mask_compaction_identity",
        lambda *args, **kwargs: {
            "prediction_bytes_identical": True,
            "strict_identical": True,
            "masked_strict": [False],
            "compacted_strict": [False],
        },
        raising=False,
    )
    result = structural.measure_real_arm("neuron-add", data_root="data")
    assert result["candidate_neurons"] == 3
    assert result["after_strict"] == [True]
    assert result["mask_compaction"]["prediction_bytes_identical"] is True


def test_muon_remap_handles_nested_named_states_padding_and_shape_misses():
    State = namedtuple("State", "matrix")
    mapped = structural.remap_muon_groups(
        {"group": {"state": State(np.arange(6.0).reshape(3, 2)),
                    "scalar": np.array([7.0])}},
        {"group": (np.array([True, False, True]), (4, 2))},
    )
    np.testing.assert_array_equal(
        mapped["group"]["state"].matrix,
        [[0.0, 1.0], [4.0, 5.0], [0.0, 0.0], [0.0, 0.0]],
    )
    np.testing.assert_array_equal(mapped["group"]["scalar"], [7.0])
    untouched = object()
    assert structural.remap_muon_groups({"other": untouched}, {})["other"] is untouched


def test_structural_muon_maps_cover_pruning_and_growth_selectors():
    topology = structural.SparseTopology(
        np.array([0, 1, 2]), np.array([0, 1, 2]), np.ones(3),
        np.array([0, 1, 2]), np.array([1, 2, 0]), np.ones(3),
        np.ones((3, 1)), np.zeros(3), ((),) * 3,
    )
    adam = structural.StructuralAdam(
        np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(3), np.zeros(3),
        np.zeros(3), np.zeros(3),
    )
    alive = np.array([True, False, True])
    compacted, _, _ = structural.compact(topology, alive, adam)
    pruned = structural.structural_muon_parameter_maps(
        topology, compacted, "neuron-prune", alive
    )
    assert pruned["readout_weight"][0].tolist() == [True, False, True]
    recurrent, _ = structural.prune_recurrent(
        topology, np.array([0.0, 1.0, 2.0]), (True,)
    )
    connection = structural.structural_muon_parameter_maps(
        topology, recurrent, "connection-prune"
    )
    assert connection["recurrent"][0].tolist() == [False, True, True]
    grown, _ = structural.add_twin_neurons(topology, np.array([3.0, 1.0, 0.0]), 1)
    for arm, candidate in (("neuron-add", grown), ("connection-add", topology)):
        selectors = structural.structural_muon_parameter_maps(topology, candidate, arm)
        assert set(selectors) == {"input", "recurrent", "readout_weight", "readout_bias"}


def test_muon_growth_remap_preserves_canonical_recurrent_pair_identity():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([1, 2]), np.array([1.0, 2.0]),
        np.ones((3, 1)), np.zeros(3), ((),) * 3,
    )
    candidate = structural.add_recurrent_connections(topology, ((0, 0),))
    state = SimpleNamespace(mu=np.array([100.0, 200.0]))
    maps = structural.structural_muon_parameter_maps(
        topology, candidate, "connection-add"
    )
    mapped = structural.remap_muon_groups(
        {"recurrent": state}, {"recurrent": maps["recurrent"]}
    )
    np.testing.assert_array_equal(mapped["recurrent"].mu, [0.0, 100.0, 200.0])


def test_optimizer_state_proof_rejects_misaligned_sparse_pair_state():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([1, 2]), np.array([1.0, 2.0]),
        np.ones((3, 1)), np.zeros(3), ((),) * 3,
    )
    candidate = structural.add_recurrent_connections(topology, ((0, 0),))
    state = SimpleNamespace(mu=np.array([100.0, 200.0]))
    correct = structural.structural_muon_parameter_maps(
        topology, candidate, "connection-add"
    )["recurrent"]
    bad = (np.array([0, 1, -1]), correct[1], correct[2], correct[3], correct[4])
    assert structural.optimizer_state_proof(
        {"recurrent": state}, {"recurrent": correct}
    ) == {
        "parent_nonzero": True,
        "survivors_preserved": True,
        "new_items_zero": True,
    }
    assert structural.optimizer_state_proof(
        {"recurrent": state}, {"recurrent": bad}
    ) == {
        "parent_nonzero": True,
        "survivors_preserved": False,
        "new_items_zero": True,
    }


def test_real_update_hands_pair_aware_muon_state_to_trainer():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.muon_groups = {}

        def update_episode(self, *args, **kwargs):
            return 0.0, 0.0

    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([1, 2]), np.array([1.0, 2.0]),
        np.ones((3, 1)), np.zeros(3), ((),) * 3,
    )
    candidate = structural.add_recurrent_connections(topology, ((0, 0),))
    maps = structural.structural_muon_parameter_maps(
        topology, candidate, "connection-add"
    )
    update = structural._real_pp_prop_update(
        SimpleNamespace(PPPropEpisodeTrainer=Trainer),
        SimpleNamespace(
            input_weight=State([1.0]), recurrent_weight=State([1.0, 1.0, 1.0])
        ),
        SimpleNamespace(),
        {"events": np.zeros((1, 1)), "advances": np.ones(1)},
        muon_groups={"recurrent": SimpleNamespace(mu=np.array([100.0, 200.0]))},
        parameter_maps={"recurrent": maps["recurrent"]},
    )
    np.testing.assert_array_equal(
        update.trainer.muon_groups["recurrent"].mu, [0.0, 100.0, 200.0]
    )


def test_evidence_validates_explicit_task_shape_and_data_requirements(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1]), indices=np.array([0]))
        recurrent_csr = SimpleNamespace(indptr=np.array([0, 1, 2]), indices=np.array([1, 0]))
        input_weight = State([1.0])
        recurrent_weight = State([2.0, 3.0])
        readout_weight = State(np.ones((2, 1)))

    class Learner:
        def etrace_grad(self, events, **kwargs):
            return {("model", "recurrent_weight"): np.array([2.0, -3.0])}, np.array([4.0])

    kwargs = dict(
        model=Model(), learner=Learner(), events=np.ones((1, 1)),
        step_fn=lambda value: value, readout_effect=np.ones((2, 2)),
        spikes_by_task=np.ones((2, 2)), task_count=2, reduction="sum",
    )
    result = structural.collect_model_evidence(**kwargs)
    assert result["gradient_mass"].shape == (2, 2)
    with pytest.raises(ValueError, match="task-by-neuron"):
        structural.collect_model_evidence(**{**kwargs, "spikes_by_task": np.ones((1, 2))})
    with pytest.raises(ValueError, match="data-root"):
        structural._fixed_task_evidence(SimpleNamespace(), Model(), object(), None)
    with pytest.raises(ValueError, match="data-root"):
        structural._real_mask_compaction_identity(
            SimpleNamespace(), SimpleNamespace(neuron_count=1), object(), None
        )


def test_structural_edge_scores_keep_task_maxima():
    topology = structural.SparseTopology(
        np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float),
        np.array([0, 1]), np.array([1, 0]), np.ones(2),
        np.ones((2, 1)), np.zeros(2), ((),) * 2,
    )
    evidence = structural.structural_evidence(
        topology,
        np.ones((2, 2)),
        np.array([[1.0, 0.0], [0.0, 3.0]]),
        np.array([[1.0, 0.0], [0.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        evidence["connection_scores"],
        np.max(evidence["connection_scores_by_task"], axis=0),
    )


def test_addition_evidence_uses_first_failed_task_and_separate_target_signal():
    evidence = {
        "neuron_scores_by_task": np.array([
            [1.0, 4.0], [8.0, 2.0], [99.0, 101.0],
        ]),
        "task_spike_evidence": np.array([
            [1.0, 2.0], [3.0, 5.0], [300.0, 500.0],
        ]),
        "target_scores_by_task": np.array([
            [10.0, 20.0], [30.0, 40.0], [300.0, 400.0],
        ]),
    }
    result = structural.addition_selection_evidence(evidence, (False, True, True))
    np.testing.assert_array_equal(result["neuron_scores"], [1.0, 4.0])
    np.testing.assert_allclose(result["source_evidence"], [1.0, 2.0])
    np.testing.assert_allclose(result["target_evidence"], [10.0, 20.0])


def test_addition_evidence_fallback_uses_incident_signal_and_first_task():
    evidence = {
        "neuron_scores_by_task": np.ones((2, 2)),
        "task_spike_evidence": np.array([[1.0, 2.0], [3.0, 5.0]]),
        "incident_gradient_mass_by_task": np.array([[6.0, 7.0], [8.0, 9.0]]),
        "gradient_mass": np.ones((2, 2)),
        "task_readout_evidence": np.array([[10.0, 20.0], [30.0, 40.0]]),
    }
    result = structural.addition_selection_evidence(evidence, (True, True))
    np.testing.assert_array_equal(result["source_evidence"], [1.0, 2.0])
    np.testing.assert_array_equal(result["target_evidence"], [6.0, 7.0])
    with pytest.raises(ValueError, match="training count"):
        structural.addition_selection_evidence(evidence, (True, True), 0)


def test_measurement_keeps_task_max_scores_for_pruning(monkeypatch):
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
        "neuron_scores": np.array([9.0, 1.0]),
        "neuron_scores_by_task": np.array([[1.0, 9.0], [8.0, 2.0]]),
        "task_spike_evidence": np.ones((2, 2)),
        "target_scores_by_task": np.ones((2, 2)),
        "connection_scores": np.ones(2),
        "preclip_gradient_mass": [],
        "task_readout_evidence": [],
    }
    seen = {}
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(structural, "_fixed_task_evidence", lambda *args: evidence)
    def identity(*args, **kwargs):
        seen["alive"] = kwargs["alive"]
        return {"prediction_bytes_identical": True, "strict_identical": True}

    monkeypatch.setattr(structural, "_real_mask_compaction_identity", identity)
    result = structural.measure_real_arm("neuron-prune", data_root="data")
    np.testing.assert_array_equal(seen["alive"], [True, False])
    assert result["pruning_blocked"] is True


def test_readout_evidence_only_uses_wrong_output_fields():
    logits = np.zeros(360)
    logits[1] = 1.0
    logits[30] = 1.0
    logits[60] = 1.0
    wrong = structural._wrong_output_mask(logits, np.zeros((1, 1), dtype=np.uint8))
    assert np.all(wrong[:30])
    assert not np.any(wrong[30:])
    weights = np.ones((2, 360))
    weights[:, 30:60] = 100.0
    result = structural._readout_effect(
        np.ones((2, 2)), weights, output_mask=wrong
    )
    np.testing.assert_allclose(result, [np.tanh(3.3)] * 2)


def test_request_output_mask_checks_shape_and_all_target_columns():
    logits = np.zeros((31, 360), dtype=float)
    logits[0, 0] = logits[0, 31] = 1.0
    logits[1, 62] = 1.0
    logits[1, 73] = 1.0
    target = np.asarray([[2, 3]], dtype=np.uint8)
    wrong = structural._wrong_request_output_mask(logits, target)
    assert not np.any(wrong)
    logits[1, 73] = 0.0
    logits[1, 74] = 1.0
    wrong = structural._wrong_request_output_mask(logits, target)
    assert np.all(wrong[70:80])


def test_measurement_collects_post_warmup_parent_evidence(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        input_csr = SimpleNamespace(indptr=np.array([0, 1]), indices=np.array([0]))
        recurrent_csr = SimpleNamespace(indptr=np.array([0, 1]), indices=np.array([0]))
        input_weight = State(np.ones(1))
        recurrent_weight = State(np.ones(1))
        readout_weight = State(np.ones((1, 1)))

        def __init__(self, topology=None):
            self.warm_count = 0

    module = SimpleNamespace(
        BrainCellArcModel=Model,
        TRAINING_TASK_IDS=("train",),
        VALIDATION_TASK_IDS=("valid",),
        compile_pp_prop_model=lambda model: object(),
        load_task=lambda *args: SimpleNamespace(targets=(np.zeros((1, 1), dtype=np.uint8),)),
    )
    calls = []

    def fake_episodes(*args):
        return ["train", "valid"], [{"task_id": "train", "task_index": 0,
                 "events": np.ones((1, 1)), "advances": np.ones(1, dtype=bool),
                 "target": np.zeros((1, 1), dtype=np.uint8),
                 "target_vector": np.zeros(360)}]

    def fake_evidence(module_value, model, learner, data_root, *, episodes=None):
        calls.append(model.warm_count)
        return {
            "strict": [False, False], "neuron_scores": np.ones(1),
            "connection_scores": np.ones(1), "preclip_gradient_mass": [],
            "task_spike_evidence": [], "task_readout_evidence": [],
            "episodes": episodes or [], "training_episodes": episodes or [],
            "training_task_ids": ["train"], "task_ids": ["train", "valid"],
        }

    def fake_update(module_value, model, learner, evidence, *args, **kwargs):
        def update(_index):
            model.warm_count += 1
        update.trainer = SimpleNamespace(muon_groups={}, adam_groups={})
        return update

    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(structural, "_fixed_task_episodes", fake_episodes, raising=False)
    monkeypatch.setattr(structural, "_fixed_task_evidence", fake_evidence)
    monkeypatch.setattr(structural, "_real_pp_prop_update", fake_update)
    monkeypatch.setattr(structural, "_strict_task_screen", lambda *args, **kwargs: [False, False])
    monkeypatch.setattr(
        structural, "_real_mask_compaction_identity",
        lambda *args, **kwargs: {"prediction_bytes_identical": True, "strict_identical": True},
    )
    structural.measure_real_arm("neuron-prune", data_root="data")
    assert calls == [1]


def test_measurement_passes_selected_pruning_mask_when_validation_is_closed(monkeypatch):
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
        "strict": [False, False], "neuron_scores": np.array([1.0, 2.0]),
        "connection_scores": np.ones(2), "gradient_mass": np.ones(2),
        "preclip_gradient_mass": [], "task_spike_evidence": [],
        "task_readout_evidence": [],
    }
    seen = {}
    monkeypatch.setattr(structural, "_load_example21_model", lambda: module)
    monkeypatch.setattr(structural, "_fixed_task_evidence", lambda *args: evidence)
    def identity(*args, **kwargs):
        seen["alive"] = kwargs["alive"]
        return {"prediction_bytes_identical": True, "strict_identical": True}
    monkeypatch.setattr(
        structural, "_real_mask_compaction_identity", identity,
    )
    result = structural.measure_real_arm("neuron-prune", data_root="data")
    np.testing.assert_array_equal(seen["alive"], [False, True])
    assert result["pruning_blocked"] is True


def test_fixed_task_evidence_runs_episode_batch_through_transform(monkeypatch):
    import brainstate

    calls = []
    def spy_for_loop(function, values):
        calls.append(len(values))
        results = [function(value) for value in values]
        return tuple(np.asarray(items) for items in zip(*results))

    monkeypatch.setattr(brainstate.transform, "jit", lambda function: function)
    monkeypatch.setattr(
        brainstate.transform, "for_loop",
        spy_for_loop,
    )
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        readout_weight = State(np.ones((2, 360)))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.ones(360)

    class Module:
        TRAINING_TASK_IDS = ("train",)
        VALIDATION_TASK_IDS = ("valid",)

        load_task = lambda self, *_args: SimpleNamespace(
            targets=(np.zeros((1, 1), dtype=np.uint8),)
        )
        encode_episode = lambda self, *_args: (
            np.ones((2, 441)), np.ones(2, dtype=bool)
        )
        run_event_sequence_with_spikes = lambda self, model, events, advances: (
            np.zeros((2, 2)), np.ones((2, 2))
        )
        decode_prediction = lambda self, value: np.zeros((1, 1), dtype=np.uint8)
        strict_task_pass_at_1 = lambda self, predictions, targets: True

    monkeypatch.setattr(
        structural, "topology_from_model",
        lambda model: SimpleNamespace(neuron_count=2, recurrent_value=np.ones(2)),
    )
    monkeypatch.setattr(
        structural, "preclip_gradient_mass",
        lambda *args, **kwargs: (
            {"model/recurrent_weight": np.ones((2, 2))}, 0.0
        ),
    )
    monkeypatch.setattr(
        structural, "structural_evidence",
        lambda *args: {
            "neuron_scores": np.ones(2), "connection_scores": np.ones(2),
            "neuron_scores_by_task": np.ones((2, 2)),
            "connection_scores_by_task": np.ones((2, 2)), "owners": ((), ()),
        },
    )
    result = structural._fixed_task_evidence(Module(), Model(), object(), "data")
    assert result["strict"] == [True, True]
    assert calls, "fixed-task execution must use brainstate.transform.for_loop"


def test_episode_fallback_uses_direct_previous_spikes_and_zero_shape():
    module = SimpleNamespace(run_event_sequence=lambda *args: None)
    model = SimpleNamespace(
        readout_weight=SimpleNamespace(value=np.ones((3, 1)))
    )
    voltages, spikes = structural._run_episode(
        module, model, np.zeros((2, 4)), np.ones(2, dtype=bool)
    )
    assert voltages.shape == (1, 3)
    assert not np.any(spikes)

    module = SimpleNamespace(
        run_event_sequence=lambda *args: np.ones((2, 3))
    )
    model.previous_spikes = SimpleNamespace(value=np.array([1.0, 0.0, 1.0]))
    _, spikes = structural._run_episode(
        module, model, np.zeros((2, 4)), np.ones(2, dtype=bool)
    )
    np.testing.assert_array_equal(spikes, [[1.0, 0.0, 1.0]] * 2)


def test_strict_screen_uses_compiled_runner_branch():
    import jax.numpy as jnp

    class Module:
        run_event_sequence_with_spikes = staticmethod(
            lambda model, events, advances: (
                jnp.ones((2, 3)), jnp.zeros((2, 3))
            )
        )
        decode_prediction = staticmethod(lambda value: np.asarray([value[0]]))
        strict_task_pass_at_1 = staticmethod(lambda predictions, targets: True)

    class Model:
        def reset_episode(self, learner):
            pass

        def readout(self):
            return jnp.ones(3)

    episodes = [
        {"task_id": "a", "events": np.ones((2, 1)),
         "advances": np.ones(2, dtype=bool), "target": np.zeros((1, 1))},
        {"task_id": "b", "events": np.ones((2, 1)),
         "advances": np.ones(2, dtype=bool), "target": np.zeros((1, 1))},
    ]
    strict, prediction_bytes = structural._strict_task_screen(
        Module, Model(), object(), episodes, return_bytes=True
    )
    assert strict == [True, True]
    assert prediction_bytes


def test_strict_screen_uses_each_request_readout_not_final_row():
    import jax.numpy as jnp
    from examples.pp_prop.arc_contracts import (
        decode_prediction as arc_decode_prediction,
        strict_task_pass_at_1 as arc_strict_task_pass_at_1,
    )

    events = np.zeros((31, 441), dtype=bool)
    events[0, 4] = True
    events[1:, 6] = True
    events[1:, 81 + np.arange(30)] = True
    voltages = jnp.ones((31, 1)) * 0.0
    voltages = voltages.at[-1].set(-100.0)
    weights = np.zeros((1, 360), dtype=np.float32)
    weights[0, 0] = 1.0
    weights[0, 30] = 1.0
    weights[0, 62] = 1.0

    class Module:
        run_event_sequence_with_spikes = staticmethod(
            lambda model, values, advances: (voltages, jnp.zeros_like(voltages))
        )
        decode_prediction = staticmethod(arc_decode_prediction)
        strict_task_pass_at_1 = staticmethod(arc_strict_task_pass_at_1)

    class Model:
        readout_weight = SimpleNamespace(value=jnp.asarray(weights))
        readout_bias = SimpleNamespace(value=jnp.zeros(360))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return jnp.zeros(360)

    strict, _ = structural._strict_task_screen(
        Module, Model(), object(), [{
            "task_id": "a", "events": events, "advances": np.ones(31, dtype=bool),
            "target": np.asarray([[2]], dtype=np.uint8),
        }], return_bytes=True
    )
    assert strict == [True]


def test_fixed_evidence_rejects_missing_queries_and_training_queries(monkeypatch):
    monkeypatch.setattr(
        structural, "topology_from_model",
        lambda model: SimpleNamespace(neuron_count=1, recurrent_value=np.ones(1)),
    )

    class Module:
        TRAINING_TASK_IDS = ("train",)
        VALIDATION_TASK_IDS = ("valid",)
        load_task = staticmethod(
            lambda *_args: SimpleNamespace(targets=(None,))
        )

    with pytest.raises(ValueError, match="target query"):
        structural._fixed_task_evidence(Module, object(), object(), "data")

    class ValidationModule(Module):
        encode_episode = staticmethod(
            lambda _task, _query_index: (
                np.ones((1, 1)), np.ones(1, dtype=bool)
            )
        )

        @staticmethod
        def load_task(_root, task_id, _split):
            target = None if task_id == "train" else np.zeros((1, 1), dtype=np.uint8)
            return SimpleNamespace(targets=(target,))

    with pytest.raises(ValueError, match="training query"):
        structural._fixed_task_evidence(
            ValidationModule, object(), object(), "data"
        )


def test_fixed_evidence_batches_real_training_gradients(monkeypatch):
    import jax.numpy as jnp

    class State:
        def __init__(self, value):
            self.value = value

    class Learner:
        def etrace_evolve(self, events, return_outputs=True):
            return (events,)

        def etrace_grad(self, events, step_fn, **kwargs):
            del kwargs
            step_fn(events[0])
            return {("model", "recurrent_weight"): jnp.array([1.0, 2.0])}, jnp.array(3.0)

    class Model:
        readout_weight = State(jnp.ones((2, 360)))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return jnp.ones(360)

    class Module:
        TRAINING_TASK_IDS = ("train-a", "train-b")
        VALIDATION_TASK_IDS = ("valid-a",)

        @staticmethod
        def load_task(_root, task_id, _split):
            return SimpleNamespace(
                targets=(np.zeros((1, 1), dtype=np.uint8),), task_id=task_id
            )

        @staticmethod
        def encode_episode(_task, _query_index):
            return np.ones((2, 1)), np.ones(2, dtype=bool)

        @staticmethod
        def run_event_sequence_with_spikes(model, events, advances):
            del model, advances
            return jnp.ones((2, 2)) * events[0, 0], jnp.ones((2, 2))

        @staticmethod
        def decode_prediction(value):
            return np.zeros((1, 1), dtype=np.uint8)

        @staticmethod
        def strict_task_pass_at_1(predictions, targets):
            return True

    monkeypatch.setattr(
        structural, "topology_from_model",
        lambda model: SimpleNamespace(
            neuron_count=2, recurrent_value=np.ones(2),
            recurrent_source=np.array([0, 1]), recurrent_target=np.array([1, 0]),
        ),
    )
    result = structural._fixed_task_evidence(
        Module, Model(), Learner(), "data"
    )
    assert np.asarray(result["preclip_gradient_mass"]).shape == (3, 2)
    np.testing.assert_array_equal(result["preclip_gradient_mass"][2], [0.0, 0.0])


def test_fixed_evidence_collects_all_tasks_and_direct_spikes(monkeypatch):
    import jax.numpy as jnp

    calls = []

    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        readout_weight = State(np.ones((2, 360)))
        previous_spikes = State(np.zeros(2))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.ones(360)

    class Module:
        TRAINING_TASK_IDS = ("train-a", "train-b")
        VALIDATION_TASK_IDS = ("valid-a",)

        def load_task(self, root, task_id, split):
            calls.append(("load", task_id, split))
            return SimpleNamespace(
                task_id=task_id, targets=(np.zeros((1, 1), dtype=np.uint8),)
            )

        def encode_episode(self, task, query_index):
            return np.full((2, 2), self._task_number(task)), np.ones(2, dtype=bool)

        @staticmethod
        def _task_number(task):
            return ("train-a", "train-b", "valid-a").index(task.task_id) + 1

        def run_event_sequence_with_spikes(self, model, events, advances):
            return jnp.full((2, 2), -1.0), jnp.full((2, 2), events[0, 0])

        def decode_prediction(self, value):
            return np.zeros((1, 1), dtype=np.uint8)

        def strict_task_pass_at_1(self, predictions, targets):
            return True

    topology = SimpleNamespace(neuron_count=2, recurrent_value=np.ones(2))
    monkeypatch.setattr(structural, "topology_from_model", lambda model: topology)
    def fake_mass(learner, events, step_fn, task_index, task_count, **kwargs):
        rows = np.zeros((task_count, 2))
        rows[task_index] = task_index + 2.0
        return {"model/recurrent_weight": rows}, np.array([1.0])
    monkeypatch.setattr(structural, "preclip_gradient_mass", fake_mass)
    monkeypatch.setattr(
        structural,
        "structural_evidence",
        lambda topology, readout, spikes, gradient: {
            "neuron_scores": np.ones(2),
            "connection_scores": np.ones(2),
            "neuron_scores_by_task": np.ones((3, 2)),
            "connection_scores_by_task": np.ones((3, 2)),
            "owners": ((), ()),
        },
    )
    result = structural._fixed_task_evidence(Module(), Model(), object(), "data")
    assert result["strict"] == [True, True, True]
    assert np.asarray(result["task_spike_evidence"]).shape == (3, 2)
    assert np.asarray(result["preclip_gradient_mass"]).shape == (3, 2)
    assert np.asarray(result["task_readout_evidence"]).shape == (3, 2)
    assert result["task_ids"] == ["train-a", "train-b", "valid-a"]
    assert result["task_spike_evidence"][0] != result["task_spike_evidence"][1]


def test_request_loss_uses_every_target_column_and_zeroes_non_requests():
    import jax
    import jax.numpy as jnp

    from examples.pp_prop import arc_contracts

    event = np.zeros(441, dtype=np.float32)
    event[6] = 1.0
    event[81] = 1.0
    row_logits = jnp.zeros((30, 10)).at[0, 1].set(1.0).at[1, 2].set(2.0)
    logits = jnp.concatenate((jnp.zeros(60), row_logits.reshape(-1)))
    first = np.zeros((30, 30), dtype=np.int32)
    first[0, :2] = (1, 1)
    second = first.copy()
    second[0, 1] = 2
    shape = jnp.asarray((1, 2), dtype=jnp.int32)
    first_loss = structural._request_loss(
        event, logits, first, shape, jnp
    )
    second_loss = structural._request_loss(
        event, logits, second, shape, jnp
    )
    assert float(first_loss) != float(second_loss)
    expected = arc_contracts.request_loss(
        np.asarray(row_logits), first[0], request="row", valid_mask=np.arange(30) < 2
    )
    assert float(first_loss) == pytest.approx(expected)
    gradient = jax.grad(
        lambda values: structural._request_loss(event, values, first, shape, jnp)
    )(logits)
    assert np.any(np.abs(np.asarray(gradient[70:80])) > 0.0)
    assert float(structural._request_loss(
        np.zeros(441), logits, first, shape, jnp
    )) == 0.0
    padding = np.zeros(441, dtype=np.float32)
    assert float(structural._request_loss(
        padding, logits, second, shape, jnp
    )) == 0.0


def test_real_update_uses_indexed_training_schedule_and_target_loss():
    calls = []

    class Learner:
        def etrace_evolve(self, events, return_outputs):
            return (events,)

    class Model:
        input_weight = SimpleNamespace(value=np.ones(1))
        recurrent_weight = SimpleNamespace(value=np.ones(1))

        def reset_episode(self, learner):
            pass

        def readout(self):
            return np.ones(360)

    class Trainer:
        def __init__(self, learner, parameters):
            self.learner = learner
            self.parameters = parameters
            self.muon_groups = {}

        def update_episode(self, **kwargs):
            calls.append(kwargs)
            kwargs["step_fn"](kwargs["events"][0])
            assert kwargs["direct_grad_fn"] is not None
            return 0.0, 0.0

    module = SimpleNamespace(PPPropEpisodeTrainer=Trainer)
    update = structural._real_pp_prop_update(
        module, Model(), Learner(), {
            "training_episodes": [
                {"events": np.full((2, 1), 1.0), "advances": np.ones(2),
                 "target_vector": np.zeros(360)},
                {"events": np.full((2, 1), 2.0), "advances": np.ones(2),
                 "target_vector": np.ones(360)},
            ]
        },
    )
    update(0)
    update(1)
    assert [int(call["events"][0, 0]) for call in calls] == [1, 2]


def test_real_update_accepts_candidate_learning_rates():
    class Trainer:
        def __init__(self, learner, parameters):
            self.learning_rates = {"input": 1.0}

        def update_episode(self, **kwargs):
            return 0.0, 0.0

    model = SimpleNamespace(
        input_weight=SimpleNamespace(value=np.ones(1)),
        recurrent_weight=SimpleNamespace(value=np.ones(1)),
    )
    update = structural._real_pp_prop_update(
        SimpleNamespace(PPPropEpisodeTrainer=Trainer), model, SimpleNamespace(),
        {"events": np.zeros((1, 1)), "advances": np.ones(1)},
        learning_rates={"input": 0.01, "recurrent": 0.003, "readout": 0.03},
    )
    assert update.trainer.learning_rates["readout"] == 0.03


def test_direct_readout_gradients_cover_voltage_features_and_bias():
    import brainunit as u
    import jax.numpy as jnp

    model = SimpleNamespace(
        cell=SimpleNamespace(V=SimpleNamespace(value=jnp.zeros(2) * u.mV)),
        readout_weight=SimpleNamespace(value=jnp.ones((2, 3))),
        readout_bias=SimpleNamespace(value=jnp.zeros(3)),
    )
    gradients = structural._direct_readout_gradients(model, jnp.ones(3), jnp)
    assert set(gradients) == {("readout_weight",), ("readout_bias",)}
    assert gradients[("readout_weight",)].shape == (2, 3)
    assert gradients[("readout_bias",)].shape == (3,)
    assert jnp.all(jnp.isfinite(gradients[("readout_weight",)]))
    assert jnp.any(gradients[("readout_weight",)] != 0)


def test_optimizer_state_proof_requires_nonzero_survivors_and_zero_new_items():
    state = SimpleNamespace(mu=np.array([[1.0], [2.0]]), nu=np.array([[3.0], [4.0]]))
    result = structural.optimizer_state_proof(
        {"readout_weight": state},
        {"readout_weight": (np.array([True, False]), (3, 1))},
    )
    assert result == {
        "parent_nonzero": True,
        "survivors_preserved": True,
        "new_items_zero": True,
    }


def test_real_update_remaps_model_weight_aliases_for_growth():
    class State:
        def __init__(self, value):
            self.mu = np.asarray(value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.muon_groups = {}

        def update_episode(self, **kwargs):
            return 0.0, 0.0

    class Model:
        input_weight = SimpleNamespace(value=np.ones(3))
        recurrent_weight = SimpleNamespace(value=np.ones(2))

        def reset_episode(self, learner):
            pass

    update = structural._real_pp_prop_update(
        SimpleNamespace(PPPropEpisodeTrainer=Trainer), Model(), SimpleNamespace(),
        {"events": np.zeros((1, 1)), "advances": np.ones(1)},
        muon_groups={"input_weight": State([[1.0], [2.0]])},
        parameter_maps={"input": (np.array([True, False]), (3, 1))},
    )
    np.testing.assert_array_equal(
        update.trainer.muon_groups["input_weight"].mu,
        [[1.0], [0.0], [0.0]],
    )


def test_integrated_dispatch_covers_each_arm_and_rejects_unknown_arm():
    class State:
        def __init__(self, value):
            self.value = np.asarray(value)

    class Model:
        def __init__(self):
            self.input_csr = SimpleNamespace(indptr=np.array([0, 1, 2, 3]), indices=np.array([0, 1, 2]))
            self.recurrent_csr = SimpleNamespace(indptr=np.array([0, 1, 2, 3]), indices=np.array([1, 2, 0]))
            self.input_weight = State([1.0, 1.0, 1.0])
            self.recurrent_weight = State([1.0, 1.0, 1.0])
            self.readout_weight = State(np.ones((3, 1)))

    class Transform:
        @staticmethod
        def jit(function):
            return function

        @staticmethod
        def for_loop(function, values):
            return np.asarray([function(value) for value in values])

    evidence = {
        "neuron_scores": np.array([3.0, 1.0, 0.0]),
        "connection_scores": np.array([0.0, 1.0, 2.0]),
        "target_scores": np.array([0.0, 2.0, 1.0]),
        "validation_strict": (True,),
    }
    rebuild = lambda topology, adam: (SimpleNamespace(), object())
    for arm in ("neuron-prune", "connection-prune", "neuron-add", "connection-add"):
        result = structural.run_integrated_arm(
            arm, Model, lambda model: object(), lambda model, learner: (True,), rebuild,
            transform=Transform, update=lambda index: index,
            before_strict=(False,), evidence=evidence,
            clock=iter((0.0, 1.0)).__next__,
        )
        assert result["real_model"] and result["eligibility_reset"]
    with pytest.raises(ValueError, match="recognized"):
        structural.run_integrated_arm(
            "unknown", Model, lambda model: object(), lambda model, learner: (False,), rebuild,
            before_strict=(False,), evidence=evidence,
        )


def test_measurement_and_merge_commands_reject_invalid_arm_and_emit_metadata(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="recognized"):
        structural.measure_real_arm("unknown")
    monkeypatch.chdir(Path(__file__).parents[2])
    target = tmp_path / "merged.json"
    structural.main(["merge", "--output", str(target)])
    merged = json.loads(target.read_text())
    assert len(merged["arms"]) == 4
    assert merged["focused_tests"]["passed"] == 70
    assert merged["focused_tests"]["failed"] == 0
    assert merged["focused_tests"]["coverage_percent"] == 92.0
    assert "coverage run --branch" in merged["focused_tests"]["command"]
    assert merged["arm_controls"]["max_resident_tile_pairs"] == 65_536
