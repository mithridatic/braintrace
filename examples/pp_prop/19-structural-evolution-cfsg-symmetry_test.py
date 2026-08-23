"""Tests for the structural-evolution CFSG symmetry example."""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLE = pathlib.Path(__file__).resolve().with_name(
    "19-structural-evolution-cfsg-symmetry.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_pp_prop_cfsg", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topology_validation_and_directed_twin_partition():
    example = _load()
    rows = np.array([0, 1, 0, 1, 2, 2, 2])
    cols = np.array([1, 0, 2, 2, 0, 1, 3])
    neighbors = example._validate_topology(4, rows, cols)
    assert example._are_twins(neighbors, 0, 1)
    assert not example._are_twins(neighbors, 0, 2)
    assert example._twin_partition(neighbors).tolist() == [0, 0, 2, 3]


@pytest.mark.parametrize(
    ("n_rec", "rows", "cols", "message"),
    [
        (0, np.array([], dtype=int), np.array([], dtype=int), "positive"),
        (2, np.array([[0]]), np.array([1]), "one-dimensional"),
        (2, np.array([0]), np.array([], dtype=int), "aligned"),
        (2, np.array([0]), np.array([2]), "outside"),
        (2, np.array([0.0]), np.array([1.0]), "integers"),
        (2, np.array([0, 0]), np.array([1, 1]), "duplicate"),
    ],
)
def test_topology_validation_rejects_invalid_inputs(n_rec, rows, cols, message):
    example = _load()
    with pytest.raises(ValueError, match=f"(?i){message}"):
        example._validate_topology(n_rec, rows, cols)


def test_role_refinement_splits_equal_degree_directed_roles():
    example = _load()
    neighbors = example._validate_topology(4, np.array([0, 1, 2]), np.array([2, 2, 3]))
    colors = example._refine_roles(neighbors)
    assert colors[0] == colors[1]
    assert colors[2] != colors[3]


@pytest.mark.parametrize(
    ("size", "expected"),
    [(1, []), (2, ["Z2"]), (3, ["Z3", "Z2"]),
     (4, ["Z2", "Z2", "Z3", "Z2"]), (5, ["A5", "Z2"])],
)
def test_composition_factors_match_symmetric_groups(size, expected):
    assert _load()._composition_factors(size) == expected


def test_symmetry_description_has_expected_order_and_factors():
    log10_order, factors = _load()._describe_symmetry([3, 2, 1])
    assert log10_order == pytest.approx(np.log10(12.0))
    assert factors == "Z2^2 . Z3"


def test_label_profiles_do_not_hide_heterogeneous_edges_behind_a_mode():
    example = _load()
    rows = np.array([0, 0, 0, 1, 1, 1])
    cols = np.array([2, 3, 4, 2, 3, 4])
    profiles = example._neuron_label_profiles(
        5, rows, cols, np.array([0, 0, 1, 0, 0, 1])
    )
    class_of = np.array([0, 0, 2, 3, 4])
    degree = np.array([3, 3, 2, 2, 2])
    assert profiles[0] == frozenset({0, 1})
    assert example._orbit_attribution_split(class_of, profiles, degree) == (0, 1)


def test_orbit_split_counts_only_wired_single_label_classes_as_pure():
    example = _load()
    class_of = np.array([0, 0, 2, 2, 4, 4])
    profiles = [frozenset({1}), frozenset({1}), frozenset({0}),
                frozenset({1}), frozenset(), frozenset()]
    degree = np.array([1, 1, 1, 1, 0, 0])
    assert example._orbit_attribution_split(class_of, profiles, degree) == (1, 1)


def test_task_pair_overlap_filters_idle_edges_and_counts_unordered_pairs():
    example = _load()
    mass = np.array([[3.0, 0.0, 1.0], [2.0, 0.0, 4.0], [1.0, 0.0, 2.0]])
    matrix = example._task_pair_overlap(mass, np.array([3, 3, 1]), 3)
    assert matrix.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    ("mass", "labels", "n_tasks"),
    [(np.zeros((1, 1)), np.array([1]), 1),
     (np.zeros((2, 2)), np.array([2]), 2),
     (np.zeros((2, 1)), np.array([3]), 2)],
)
def test_task_pair_overlap_rejects_invalid_contracts(mass, labels, n_tasks):
    with pytest.raises(ValueError):
        _load()._task_pair_overlap(mass, labels, n_tasks)


def test_main_reports_both_arms_and_returns_example_18_result(monkeypatch, capsys):
    example = _load()
    arm = {"rows": np.array([0, 1]), "cols": np.array([1, 0]),
           "attribution": np.array([0, 0]),
           "task_mass": np.array([[1.0, 1.0], [0.0, 0.0]]),
           "trick_names": ["fetch", "roll_over"]}
    result = {"config": SimpleNamespace(n_rec=2), "evolve": arm, "control": arm}
    monkeypatch.setattr(example.EX18, "main", lambda argv: result)
    assert example.main(["--smoke"]) is result
    output = capsys.readouterr().out
    assert "arm=evolve" in output and "arm=control" in output


def test_sparse_analysis_does_not_allocate_a_dense_adjacency(monkeypatch):
    example = _load()
    original_zeros = example.np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == (100, 100):
            raise AssertionError("Dense square allocation. Update the fixture or expected result to satisfy this assertion.")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(example.np, "zeros", guarded_zeros)
    arm = {"rows": np.array([0, 1]), "cols": np.array([1, 0]),
           "attribution": np.array([0, 0]),
           "task_mass": np.array([[1.0, 1.0], [0.0, 0.0]])}
    example._symmetry_report("sparse", arm, 100, ["a", "b"])
