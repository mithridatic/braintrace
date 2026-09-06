"""Construction index work, exact discretization parity, and cache lifetime."""

import braincell
import brainunit as u
import numpy as np
import pytest

from .h01_anatomy_test import imported
from .h01_ei_cell import make_h01_ei_cell
from .h01_ei_circuit_test import arguments
from .h01_construction import H01Cell, _indexed_morphology


def test_h01_cell_does_not_rebuild_index_map_per_edge(imported, monkeypatch):
    owner = type(imported.morphology)
    original = owner._branch_index_map
    calls = []
    def counted(self, **kwargs):
        calls.append(kwargs.get("order", "default"))
        return original(self, **kwargs)
    monkeypatch.setattr(owner, "_branch_index_map", counted)
    args = arguments(imported)
    make_h01_ei_cell(imported, args["annotations"], polarity="E",
        regions=args["regions"]["E"], region_basis="Test partition", pop_size=(1,))
    assert calls.count("default") <= 2, calls


def test_cache_restores_lookup_and_tracks_new_branches(imported):
    morphology = imported.morphology
    expected = [b.index for b in morphology.branches]
    with _indexed_morphology(morphology):
        assert [b.index for b in morphology.branches] == expected
        with _indexed_morphology(morphology):
            assert morphology.root.index == expected[0]
        morphology.attach(parent=morphology.root,
            child_branch=braincell.Branch.from_lengths(lengths=[2.]*u.um, radii=[1., 1.]*u.um),
            child_name="extra")
        assert [b.index for b in morphology.branches] == list(range(len(expected)+1))
        for branch in morphology.branches:
            assert branch.index_by(order="depth") >= 0
        assert len(morphology.branch_by_order(order="depth")) == len(expected)+1
    assert "_branch_index" not in morphology.__dict__
    assert "branch_by_order" not in morphology.__dict__
    with pytest.raises(RuntimeError), _indexed_morphology(morphology):
        raise RuntimeError("construction failure")
    assert "_branch_index" not in morphology.__dict__
    assert "branch_by_order" not in morphology.__dict__


def test_channel_placement_does_not_rebuild_branch_tuple_per_compartment(imported, monkeypatch):
    cell = H01Cell(imported.morphology, cv_policy=braincell.MaxCVLen(.5*u.um))
    cell.paint(braincell.filter.AllRegion(), braincell.mech.Channel("IL", name="leak"))
    owner = type(imported.morphology)
    original, calls = owner.branch_by_order, []
    def counted(self, **kwargs):
        calls.append(kwargs.get("order", "default"))
        return original(self, **kwargs)
    monkeypatch.setattr(owner, "branch_by_order", counted)
    assert cell.n_cv > 20
    assert calls.count("default") <= 2, len(calls)


def test_cached_and_standard_discretization_are_identical(imported):
    cells = [kind(imported.morphology, cv_policy=braincell.MaxCVLen(2.*u.um),
                  pop_size=(1,)) for kind in (braincell.Cell, H01Cell)]
    standard, cached = cells
    assert len(standard.cvs) == len(cached.cvs)
    for left, right in zip(standard.cvs, cached.cvs):
        assert (left.id, left.branch_id, left.prox, left.dist) == (right.id, right.branch_id, right.prox, right.dist)
        np.testing.assert_array_equal(left.area.to_decimal(u.um**2), right.area.to_decimal(u.um**2))
    assert standard.cv_tree == cached.cv_tree
    assert standard.node_tree.nodes == cached.node_tree.nodes
    assert standard.node_tree.edges == cached.node_tree.edges
    assert standard.node_tree.root_node_id == cached.node_tree.root_node_id
    for field in ("cv_to_mid_node_id", "branch_endpoint_node_id"):
        np.testing.assert_array_equal(getattr(standard.node_tree, field), getattr(cached.node_tree, field))
    outputs = []
    for cell in cells:
        soma = imported.anatomy().soma_location()
        cell.place(soma, braincell.mech.StateProbe(field="v", name="voltage"))
        cell.place(soma, braincell.CurrentClamp(delay=0.*u.ms, durations=.003*u.ms, amplitudes=.001*u.nA))
        result = cell.run(dt=.001*u.ms, duration=.003*u.ms)
        outputs.append(np.asarray(result.traces["voltage"].to_decimal(u.mV)))
        assert "_branch_index" not in cell.morpho.__dict__
    np.testing.assert_array_equal(*outputs)
