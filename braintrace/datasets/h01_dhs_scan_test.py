"""DHS equations against the installed kernels and a dense tree oracle."""
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.quad import _staggered as original
from .h01_anatomy_test import imported


def tree(parent):
    n = len(parent)
    depth = np.zeros(n, dtype=int)
    for child in range(1, n):
        depth[child] = depth[parent[child]] + 1
    levels = [np.flatnonzero(depth == d) for d in range(int(depth.max()), 0, -1)]
    edges = np.asarray([(c, parent[c]) for level in levels for c in level], dtype=np.int32).reshape(-1, 2)
    offsets = np.r_[0, np.cumsum([len(level) for level in levels])].astype(np.int32)
    lookup = np.r_[parent, n].astype(np.int32)
    jumps = original._build_backsub_indices(lookup, n_nodes=n)
    return edges, offsets, jumps


def test_original_trace_expands_with_depth():
    sizes = []
    for n in (9, 65):
        edges, offsets, _ = tree([0] + list(range(n-1)))
        trace = jax.make_jaxpr(lambda d, s, l: original.comp_triang_raw(d, s, l, l, edges, offsets))(
            jnp.ones((1, n+1)), jnp.ones((1, n+1)), jnp.ones(n+1))
        sizes.append(len(trace.jaxpr.eqns))
    assert sizes[1] == 8*sizes[0]


@pytest.mark.parametrize("parent", [[0], [0, 0, 1, 2, 3], [0, 0, 0, 1, 1, 2, 5]])
@pytest.mark.parametrize("quantity", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_solve_matches_original_and_dense(parent, quantity, batch):
    from .h01_dhs_scan import _prepare_levels, _triang, _backsub
    n = len(parent)
    edges, offsets, jumps = tree(parent)
    levels = _prepare_levels(edges, offsets, n)
    with brainstate.environ.context(precision=64):
        low = -np.linspace(.1, .3, n+1); low[0] = low[-1] = 0
        up = 1.3*low
        diag = np.broadcast_to(np.r_[np.linspace(2., 3., n), 1.], (batch, n+1)).copy()
        rhs = np.arange(batch*(n+1), dtype=float).reshape(batch, n+1)/10 - 2
        rhs[:, -1] = 0
        d, s, l, h = map(jnp.asarray, (diag, rhs, low, up))
        if quantity:
            d, s, l, h = d*u.UNITLESS, s*u.mV, l*u.UNITLESS, h*u.UNITLESS
        expected_d, expected_s = original.comp_triang_raw(d, s, l, h, edges, offsets)
        expected_v = original.comp_backsub_raw(expected_d, expected_s, l, jumps)
        def solve(d, s, l, h):
            td, ts = _triang(d, s, l, h, levels)
            return td, ts, _backsub(td, ts, l, jumps)
        got_d, got_s, got_v = brainstate.transform.jit(solve)(d, s, l, h)
        for actual, expected in ((got_d, expected_d), (got_s, expected_s), (got_v, expected_v)):
            assert u.get_unit(actual) == u.get_unit(expected)
            np.testing.assert_allclose(u.get_mantissa(actual), u.get_mantissa(expected), rtol=2e-14, atol=2e-14)
        dense = np.diag(diag[0])
        for c, p in edges:
            dense[c, p], dense[p, c] = low[c], up[c]
        np.testing.assert_allclose(u.get_mantissa(got_v), np.linalg.solve(dense, rhs.T).T, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(u.get_mantissa(got_d)[:, -1], 1.)
        np.testing.assert_array_equal(u.get_mantissa(got_s)[:, -1], 0.)


def test_scan_trace_does_not_expand_with_depth():
    from .h01_dhs_scan import _prepare_levels, _triang
    sizes = []
    for n in (9, 65):
        edges, offsets, _ = tree([0] + list(range(n-1)))
        levels = _prepare_levels(edges, offsets, n)
        trace = jax.make_jaxpr(lambda d, s, l: _triang(d, s, l, l, levels))(
            jnp.ones((1, n+1)), jnp.ones((1, n+1)), jnp.ones(n+1))
        sizes.append(len(trace.jaxpr.eqns))
    assert sizes[0] == sizes[1] and sizes[1] < 100


@pytest.mark.parametrize("role", ["E", "I"])
def test_active_cell_trace_matches_original(imported, role):
    from types import SimpleNamespace
    from . import h01_dhs_scan
    from .h01_ei_cell import make_h01_ei_cell
    from .h01_ei_cell_test import partition
    traces = {}
    with brainstate.environ.context(precision=64):
        for solver in ("staggered", "h01_staggered_scan"):
            cell, _ = make_h01_ei_cell(imported,
                SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=())),
                polarity=role, regions=partition(imported), region_basis="Synthetic fixture solver comparison.",
                solver=solver, current_na=1., delay_ms=.5, duration_ms=2.)
            result = cell.run(dt=.001*u.ms, duration=4.*u.ms)
            traces[solver] = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    assert np.isfinite(traces["h01_staggered_scan"]).all()
    assert np.ptp(traces["staggered"]) > 20.
    np.testing.assert_allclose(traces["h01_staggered_scan"], traces["staggered"], rtol=1e-10, atol=1e-9)


def test_circuit_events_and_conductance_match_original(imported):
    from .h01_ei_circuit import make_h01_ei_circuit
    from .h01_ei_circuit_test import arguments
    records = {}
    with brainstate.environ.context(precision=64):
        for solver in ("staggered", "h01_staggered_scan"):
            net, evidence = make_h01_ei_circuit(**arguments(imported), solver=solver, delay_ms=.1,
                inhibitory_weight_us=.0001, currents_na={"E": 1., "I": 1.})
            records[solver] = net.run(dt=.001*u.ms, duration=4.*u.ms, spike_recording="population")
            assert all(c["solver"] == solver for c in evidence["cells"].values())
    a, b = records.values()
    for role in ("E", "I"):
        assert np.asarray(a.spikes[role]).any()
        np.testing.assert_array_equal(a.spikes[role], b.spikes[role])
        for name, unit in (("voltage", u.mV), ("output_voltage", u.mV), ("synaptic_conductance", u.uS)):
            np.testing.assert_allclose(a.traces[role][name].to_decimal(unit), b.traces[role][name].to_decimal(unit),
                rtol=1e-10, atol=1e-9)
    assert np.asarray(a.traces["E"]["synaptic_conductance"].to_decimal(u.uS)).max() > 0
