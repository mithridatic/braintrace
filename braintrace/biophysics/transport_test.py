"""Independent conservation, analytic and derivative oracles for diffusion."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .transport import DiffusionGraph


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def test_unequal_volumes_closed_form_and_conservation():
    graph = DiffusionGraph([1., 3.], [[0, 1]], [2.], dt_ms=.2)
    result = jax.jit(graph.step)(jnp.array([4., 0.]))
    expected = np.linalg.solve([[1.4, -.4], [-.4, 3.4]], [4., 0.])
    np.testing.assert_allclose(result.concentration, expected, atol=1e-12)
    np.testing.assert_allclose(np.dot([1., 3.], result.concentration), 4., atol=1e-12)
    assert result.valid and abs(result.balance_error) < 1e-12


def test_reservoir_sink_and_source_amount_balance():
    graph = DiffusionGraph([2.], np.empty((0, 2), dtype=int), [], dt_ms=.5,
                           boundary_conductance=[3.], uptake_per_ms=[.1])
    result = graph.step(jnp.array([1.]), source_amount_per_ms=jnp.array([2.]), reservoir_mm=jnp.array([4.]))
    np.testing.assert_allclose(result.concentration, [(2+1+6)/(2+1.5+.1)])
    assert result.valid and abs(result.balance_error) < 1e-12
    assert result.boundary_amount > 0 and result.removed_amount > 0


def test_diffusion_decay_converges_to_analytic_solution():
    def run(dt):
        graph = DiffusionGraph([1., 1.], [[0, 1]], [1.], dt_ms=dt)
        return brainstate.transform.scan(lambda c, _: (graph.step(c).concentration, None),
            jnp.array([2., 0.]), jnp.arange(round(1/dt)))[0]
    expected = np.array([1+np.exp(-2), 1-np.exp(-2)])
    assert np.linalg.norm(run(.005)-expected) < np.linalg.norm(run(.01)-expected)


def test_implicit_derivatives_match_dense_oracle():
    graph = DiffusionGraph([1., 3.], [[0, 1]], [2.], dt_ms=.2)
    f = lambda c: graph.step(c).concentration
    oracle = np.linalg.solve([[1.4, -.4], [-.4, 3.4]], np.diag([1., 3.]))
    c = jnp.array([4., 1.])
    np.testing.assert_allclose(jax.jacfwd(f)(c), oracle, atol=1e-11)
    np.testing.assert_allclose(jax.jacrev(f)(c), oracle, atol=1e-11)


def test_invalid_state_and_excessive_sink_are_reported_not_clipped():
    graph = DiffusionGraph([1.], np.empty((0, 2), dtype=int), [], dt_ms=1.)
    result = graph.step(jnp.ones(1), source_amount_per_ms=jnp.array([-2.]))
    assert not result.valid and result.concentration[0] < 0
    assert not graph.step(jnp.array([-1.])).valid
    assert not graph.step(jnp.ones(1), reservoir_mm=jnp.array([-1.])).valid
    with pytest.raises(ValueError, match="match volumes"):
        graph.step(jnp.ones(2))


@pytest.mark.parametrize('kwargs', [
    dict(volumes=[0.]), dict(volumes=[]), dict(edges=[[0, 0]]),
    dict(edges=[[0, 2]]), dict(edges=[[0., 1.]]),
    dict(edges=[[0, 1], [1, 0]], conductance=[1., 1.]),
    dict(conductance=[-1.]), dict(dt_ms=0.), dict(tolerance=0.),
    dict(max_iterations=0), dict(boundary_conductance=[-1., 0.]),
    dict(uptake_per_ms=[1.]),
])
def test_invalid_graph_rejected(kwargs):
    config = dict(volumes=[1., 1.], edges=[[0, 1]], conductance=[1.], dt_ms=.1)
    config.update(kwargs)
    with pytest.raises(ValueError):
        DiffusionGraph(**config)
