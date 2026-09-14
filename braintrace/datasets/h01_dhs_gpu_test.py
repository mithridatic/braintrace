"""Single-kernel elimination and exact implicit derivative qualification."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .h01_dhs_gpu import eliminate
from .h01_dhs_scan import _prepare_levels, _triang_raw, _solve_raw
from .h01_dhs_scan_test import tree


def _case(batch=1, leaves=32):
    parent = [0, 0, 1, 2]+[3]*leaves
    n = len(parent)
    edges, offsets, jumps = tree(parent)
    levels = _prepare_levels(edges, offsets, n)
    d = jnp.broadcast_to(jnp.linspace(2., 3., n+1), (batch, n+1))
    rhs = jnp.sin(jnp.arange(batch*(n+1), dtype=jnp.float64)).reshape(batch, n+1)
    low = jnp.linspace(-.01, -.03, n+1).at[0].set(0.).at[-1].set(0.)
    return (d, rhs, low, 1.3*low), levels, jumps, edges


@pytest.mark.parametrize('batch,leaves', [(1, 32), (3, 32), (1, 7)])
def test_elimination_preserves_inputs_and_matches_scan(batch, leaves):
    with brainstate.environ.context(precision=64):
        args, levels, _, _ = _case(batch, leaves)
        before = tuple(np.array(x) for x in args)
        expected = jax.jit(lambda *args: _triang_raw(*args, levels))(*args)
        actual = jax.jit(lambda *args: eliminate(*args, levels))(*args)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(got[:, -1], want[:, -1])
        for value, snapshot in zip(args, before):
            np.testing.assert_array_equal(value, snapshot)


def test_implicit_gpu_derivatives_and_vmap_match_dense():
    with brainstate.environ.context(precision=64):
        args, levels, jumps, edges = _case()

        def sparse(d, rhs, low, up):
            return _solve_raw(d, rhs, low, up, levels, jumps, edges, use_gpu=True)

        def dense(d, rhs, low, up):
            matrix = jnp.diag(d[0]).at[edges[:, 0], edges[:, 1]].set(low[edges[:, 0]])
            matrix = matrix.at[edges[:, 1], edges[:, 0]].set(up[edges[:, 0]])
            return jnp.linalg.solve(matrix, rhs.T).T

        np.testing.assert_allclose(jax.jit(sparse)(*args), dense(*args), rtol=1e-12, atol=1e-12)
        for mode in (jax.jacfwd, jax.jacrev):
            actual = jax.jit(mode(sparse, argnums=(0, 1, 2, 3)))(*args)
            expected = jax.jit(mode(dense, argnums=(0, 1, 2, 3)))(*args)
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-12)
        d, rhs, low, up = args
        rows = jnp.stack((rhs, rhs*2, rhs*3))
        actual = jax.jit(jax.vmap(lambda value: sparse(d, value, low, up)))(rows)
        expected = jax.vmap(lambda value: dense(d, value, low, up))(rows)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_explicit_cpu_compilation_uses_portable_fallback():
    with brainstate.environ.context(precision=64):
        args, levels, _, _ = _case()
        cpu = jax.devices('cpu')[0]
        placed = jax.tree.map(lambda x: jax.device_put(x, cpu), args)
        cpu_levels = jax.tree.map(lambda x: jax.device_put(x, cpu), levels)
        actual = jax.jit(lambda *args: eliminate(*args, cpu_levels))(*placed)
        expected = jax.jit(lambda *args: _triang_raw(*args, cpu_levels))(*placed)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_missing_triton_extension_uses_scan(monkeypatch):
    import builtins
    from . import h01_dhs_gpu
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == 'jax.experimental.pallas':
            raise ModuleNotFoundError('Simulated CPU-only jaxlib without Triton')
        return original_import(name, *args, **kwargs)

    capability = getattr(h01_dhs_gpu, '_gpu_modules', None)
    if capability is not None:
        capability.cache_clear()
    try:
        with brainstate.environ.context(precision=64):
            args, levels, _, _ = _case()
            expected = jax.jit(lambda *args: _triang_raw(*args, levels))(*args)
            monkeypatch.setattr(builtins, '__import__', unavailable)
            actual = jax.jit(lambda *args: eliminate(*args, levels))(*args)
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    finally:
        if capability is not None:
            capability.cache_clear()


def test_real_branched_cell_sparse_updates_match_scan(tmp_path, monkeypatch):
    if jax.default_backend() != 'gpu':
        pytest.skip('Qualifies the CUDA learning path')
    import braincell
    import braintrace
    import brainunit as u
    from braincell.filter import AllRegion, RootLocation
    from braincell.mech import Channel, StateProbe
    from . import h01_calcium_solver, h01_dhs_scan
    from .h01_construction import H01Cell
    from examples.pp_prop.h01_arc_model import H01ArcModel
    from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter
    from examples.pp_prop.h01_muon import model_optimizer

    source = tmp_path/'branched.swc'
    source.write_text('1 0 0 0 0 2 -1\n2 0 10 0 0 2 1\n'+''.join(
        f'{i+3} 0 20 {i-16} 0 1 2\n' for i in range(32)))
    original_solve = h01_dhs_scan._solve_raw
    trainer_type = Example21ArcAdapter(tmp_path)._model().PPPropEpisodeTrainer

    def run(use_gpu):
        monkeypatch.setattr(h01_dhs_scan, '_solve_raw',
            lambda *args, **kwargs: original_solve(*args, use_gpu=use_gpu))
        cell = H01Cell(braincell.Morphology.from_swc(source), V_init=-65.*u.mV,
            pop_size=(1,), cv_policy=braincell.MaxCVLen(10*u.um), solver='h01_staggered_calcium_implicit')
        cell.paint(AllRegion(), Channel('IL', g_max=.1*u.mS/u.cm**2, E=-65*u.mV))
        cell.place(RootLocation(0.), StateProbe(field='v', name='voltage'))
        cell.place(RootLocation(0.), braincell.CurrentClamp(delay=0.*u.ms,
            durations=1.*u.ms, amplitudes=0.*u.nA))
        network = braincell.Network()
        network.add_population('cell', cell)
        model = H01ArcModel(network, ['cell'], dt_ms=.000625)
        assert cell._runtime.h01_dhs_pack[2][0].shape[1] == 32
        learner = braintrace.pp_prop.sparse(model)
        learner.compile_graph(jnp.zeros(441))
        parameters = dict(input=model.input_weight.value, recurrent=model.recurrent_weight.value,
            readout_weight=model.readout_weight.value, readout_bias=model.readout_bias.value)
        trainer = trainer_type(learner, parameters, optimizer_adapter=model_optimizer(model, parameters))

        def loss(event):
            learner(event)
            return jnp.mean(jnp.square(model.readout()))

        def update(index):
            return trainer.update_episode(jnp.ones((1, 441)), step_fn=loss)

        result = brainstate.transform.jit(lambda: brainstate.transform.for_loop(update, jnp.arange(2)))()
        return jax.tree.map(np.array, (result, trainer.parameters, trainer.muon_groups, learner.factors.value, cell.V.value.mantissa))

    with brainstate.environ.context(precision=64):
        expected = run(False)
        actual = run(True)
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-11)
