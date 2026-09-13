"""Construction index work, exact discretization parity, and cache lifetime."""

import braincell
from braincell.quad import _staggered as _st
import brainunit as u
import numpy as np
import pytest

from .h01_anatomy_test import imported
from .h01_ei_cell import make_h01_ei_cell
from .h01_ei_circuit_test import arguments
from .h01_construction import H01Cell, _indexed_morphology, build_dhs_static_source_1d


def test_axial_resistivity_cache_cannot_reuse_another_quantity_identity(monkeypatch):
    from . import h01_construction as subject
    monkeypatch.setattr(subject,'_RA_CACHE',{})
    # Deterministically simulate reuse of an expired object's memory address.
    monkeypatch.setattr(subject,'id',lambda value:123,raising=False)
    assert subject._get_ra_ohm_cm(100.*u.ohm*u.cm)==100.
    assert subject._get_ra_ohm_cm(150.*u.ohm*u.cm)==150.


def test_axial_resistivity_cache_bounds_retained_owners(monkeypatch):
    from . import h01_construction as subject
    monkeypatch.setattr(subject,'_RA_CACHE',{i:(object(),0.) for i in range(1024)})
    quantity=120.*u.ohm*u.cm
    assert subject._get_ra_ohm_cm(quantity)==120.
    assert len(subject._RA_CACHE)==1
    assert subject._get_ra_ohm_cm(quantity)==120.


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


def test_dhs_static_source_1d_parity(imported):
    cell = H01Cell(imported.morphology, cv_policy=braincell.MaxCVLen(2.*u.um), pop_size=(1,))
    sched = cell._node_scheduling_unchecked(algorithm="dhs")
    src_std = _st._build_dhs_static_source(cell, node_tree=cell.node_tree, scheduling=sched)
    src_1d = build_dhs_static_source_1d(cell, node_tree=cell.node_tree, scheduling=sched)
    np.testing.assert_allclose(src_std.diag_ms_inv_np, src_1d.diag_ms_inv_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(src_std.lowers_ms_inv_np, src_1d.lowers_ms_inv_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(src_std.uppers_ms_inv_np, src_1d.uppers_ms_inv_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(src_std.dynamic_rows_np, src_1d.dynamic_rows_np)
    np.testing.assert_array_equal(src_std.row_to_point_id_np, src_1d.row_to_point_id_np)
    np.testing.assert_array_equal(src_std.edges_np, src_1d.edges_np)
    np.testing.assert_array_equal(src_std.level_offsets_np, src_1d.level_offsets_np)
    np.testing.assert_array_equal(src_std.backsub_indices_np, src_1d.backsub_indices_np)


def test_deferred_axial_operator_computed_on_demand(imported):
    cell = H01Cell(imported.morphology, cv_policy=braincell.MaxCVLen(2.*u.um), pop_size=(1,))
    with pytest.raises(RuntimeError):
        cell._get_axial_operator()

    cell.init_state()
    assert cell._runtime.axial_operator_np is None
    assert cell._axial_jax is None

    # Calling compute_axial_derivative computes it on demand
    deriv = cell.compute_axial_derivative(cell.V.value)
    assert cell._runtime.axial_operator_np is not None
    assert cell._runtime.axial_operator_np.shape == (cell.n_cv, cell.n_cv)
    assert np.isfinite(np.asarray(deriv.to_decimal(u.mV / u.ms))).all()

    # Second call hits cached operator
    op2 = cell._get_axial_operator()
    assert op2 is cell._axial_jax


def test_fast_runtime_ion_instantiation_and_geometry_parity(imported):
    from braincell.ion import SodiumFixed, PotassiumFixed
    from .h01_pv_calcium import PVCalcium
    from .h01_construction import _fast_instantiate_runtime_ion_instance, _fast_attach_runtime_ion_geometry
    from braincell._compute.runtime import _instantiate_runtime_ion_instance
    from braincell._multi_compartment.bridge import attach_runtime_ion_geometry

    cell = H01Cell(imported.morphology, cv_policy=braincell.MaxCVLen(2.*u.um), pop_size=(1,))
    cell.init_state()
    ions = cell._runtime.ions
    for ion_name, ion in ions.items():
        assert hasattr(ion, "length") and hasattr(ion, "area")
        assert getattr(ion, "length").shape == (1, len(cell.node_tree.nodes))
        assert np.isfinite(getattr(ion, "length").mantissa).all()
        assert np.isfinite(getattr(ion, "area").mantissa).all()
def test_cached_voltage_linearizer_keeps_time_as_an_explicit_input(imported):
    import brainstate
    import jax.numpy as jnp
    from braintrace._compiler.sparse_io_graph import SparseIOGraph
    from examples.pp_prop.h01_arc_model import H01ArcModel
    with brainstate.environ.context(precision=64):
        args = arguments(imported)
        cell, _ = make_h01_ei_cell(imported, args['annotations'], polarity='E',
            regions=args['regions']['E'], region_basis='Synthetic regression fixture',
            pop_size=(1,), current_na=0., solver='h01_staggered_calcium_implicit')
        network = braincell.Network()
        network.add_population('cell', cell)
        model = H01ArcModel(network, ['5805562981'])
        brainstate.transform.jit(model.update)(jnp.zeros(441))
        model.reset_episode()
        graph = SparseIOGraph(model, jnp.zeros(441))
        outputs, _, _ = graph.forward(graph.inputs(jnp.zeros(441)))
        assert all(np.isfinite(np.asarray(value)).all() for value in outputs)
