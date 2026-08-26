import importlib.util
from pathlib import Path

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
    np.testing.assert_allclose(scores, [1.0, 0.575, 0.5])


def test_pruning_masks_exact_ceiling_and_validation_guard():
    scores = np.arange(21.0)
    mask = structural.pruning_mask(scores, validation_strict=(True, False))
    assert np.flatnonzero(~mask).tolist() == [0, 1]
    with pytest.raises(ValueError, match="validation"):
        structural.pruning_mask(scores, validation_strict=(False, False))


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


def test_arm_gate_requires_gain_no_regression_limits_and_fixed_updates():
    assert structural.promote_arm((False, True), (True, True), 299.0, "addition", 64)
    assert not structural.promote_arm((False, True), (True, False), 1.0, "addition", 64)
    assert not structural.promote_arm((False,), (True,), 301.0, "addition", 64)
    assert structural.promote_arm((True,), (True,), 1.0, "pruning", 0)
    with pytest.raises(ValueError, match="64"):
        structural.promote_arm((False,), (True,), 1.0, "addition", 63)
