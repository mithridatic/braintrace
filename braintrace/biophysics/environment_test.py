"""Membrane exchange, boundary accounting, gradients and shared chemical state."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .environment import ChemicalEnvironment, MembraneMap, kir41_current_na
from .extracellular import MOLECULES_PER_MM_UM3, outward_current_amount_rate
from .transport import DiffusionGraph


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def make_environment(boundary=False):
    k = DiffusionGraph([2., 3.], [[0, 1]], [.3], dt_ms=.005,
                       boundary_conductance=[0., .4 if boundary else 0.])
    g = DiffusionGraph([2., 3.], [[0, 1]], [.2], dt_ms=.005,
                       boundary_conductance=[0., .2 if boundary else 0.], uptake_per_ms=[1/600, 1/600])
    return ChemicalEnvironment(k, g, MembraneMap([0, 0, 1], [5., 7., 4.], 2), [140., 140., 140.])


def test_shared_exchange_and_release_conserve_total_pools():
    env = make_environment()
    initial = jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    current = jnp.array([.1, -.05, .03])
    k, gaba = brainstate.transform.jit(env.update)(current, jnp.array([3000., 0.]))
    final = jnp.sum(k*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    np.testing.assert_allclose(final, initial, atol=1e-12)
    np.testing.assert_allclose(env.inside_potassium.value,
        140-.005*outward_current_amount_rate(current)/env.membranes.inside_volumes)
    np.testing.assert_allclose(jnp.sum(gaba*env.gaba_transport.volumes)+env.gaba_removed_amount.value,
                               3000/MOLECULES_PER_MM_UM3, atol=1e-12)
    assert gaba[1] > 0 and env.valid.value and env.tick.value == 1
    np.testing.assert_allclose(env.balance_error.value, 0., atol=1e-12)


def test_compiled_repeated_exchange_boundary_ledger_and_reset():
    env = make_environment(True)
    initial = jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    def step(_):
        return env.update(jnp.array([1., 0., 0.]), jnp.array([3000., 0.]))
    k, gaba = brainstate.transform.for_loop(step, jnp.arange(20))
    final = jnp.sum(k[-1]*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    np.testing.assert_allclose(final-initial, env.k_boundary_amount.value, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(gaba[-1]*env.gaba_transport.volumes)+env.gaba_removed_amount.value-
                               env.gaba_boundary_amount.value, env.gaba_released_amount.value, atol=1e-12)
    assert env.k_boundary_amount.value < 0 and env.gaba_boundary_amount.value < 0
    assert env.tick.value == 20 and env.valid.value
    assert not np.allclose(env.reversal_mv(), make_environment().reversal_mv(), rtol=1e-8)
    env.reset_state()
    np.testing.assert_array_equal(env.inside_potassium.value, env.initial_inside)
    np.testing.assert_array_equal(env.potassium.value, env.bath)
    assert env.tick.value == 0 and env.valid.value and not np.any(env.gaba.value)
    assert env.k_boundary_amount.value == env.gaba_removed_amount.value == env.gaba_released_amount.value == 0


def test_current_to_concentration_gradient_has_opposite_pool_signs():
    env = make_environment()
    def response(current):
        k = env.k_transport.step(env.bath, source_amount_per_ms=env.membranes.scatter(outward_current_amount_rate(current)))
        inside = env.initial_inside-env.dt_ms*outward_current_amount_rate(current)/env.membranes.inside_volumes
        return jnp.concatenate((k.concentration, inside))
    values = jnp.array([.1, -.02, .04])
    derivative = jax.jacrev(response)(values)
    epsilon = 1e-3
    fd = np.stack([(np.asarray(response(values+epsilon*axis))-np.asarray(response(values-epsilon*axis)))/(2*epsilon)
                   for axis in np.eye(3)], axis=1)
    np.testing.assert_allclose(derivative, fd, rtol=1e-6, atol=1e-10)
    assert np.all(np.asarray(derivative[:2]) > 0)
    assert np.all(np.diag(np.asarray(derivative[2:])) < 0)


def test_kir_units_and_invalid_area():
    from .astrocyte import kir41_current_ma_cm2
    density = kir41_current_ma_cm2(-80., -100., 2.5)
    np.testing.assert_allclose(kir41_current_na(-80., -100., 2.5, 100.), density)
    assert np.isnan(kir41_current_na(-80., -100., 2.5, 0.))


@pytest.mark.parametrize('indices,volumes,count', [([0.], [1.], 1), ([1], [1.], 1),
    ([0], [0.], 1), ([0], [1.], 0), ([[0]], [1.], 1)])
def test_invalid_membrane_maps(indices, volumes, count):
    with pytest.raises(ValueError):
        MembraneMap(indices, volumes, count)


def test_geometry_initialization_and_input_validation():
    env = make_environment()
    with pytest.raises(ValueError):
        ChemicalEnvironment(env.k_transport, DiffusionGraph([1.], np.empty((0, 2), int), [], dt_ms=.005),
                            env.membranes, [140.]*3)
    with pytest.raises(ValueError):
        ChemicalEnvironment(env.k_transport, env.gaba_transport, env.membranes, [0.]*3)
    sink = DiffusionGraph([2., 3.], [[0, 1]], [.3], dt_ms=.005, uptake_per_ms=[1., 0.])
    with pytest.raises(ValueError):
        ChemicalEnvironment(sink, env.gaba_transport, env.membranes, [140.]*3)
    with pytest.raises(ValueError):
        env.update(jnp.zeros(2), jnp.zeros(2))
    with pytest.raises(ValueError):
        env.update(jnp.zeros(3), jnp.zeros(3))
    env.update(jnp.zeros(3), jnp.array([-1., 0.]))
    assert not env.valid.value


def test_depleted_pool_is_rejected_without_clipping():
    env = make_environment()
    env.update(jnp.array([1e9, 0., 0.]), jnp.zeros(2))
    assert not env.valid.value and env.inside_potassium.value[0] < 0
    assert np.isnan(env.reversal_mv()[0])


def test_small_physical_current_is_not_lost_to_transport_tolerance():
    env = make_environment()
    current = jnp.array([1e-8, 0., 0.])
    before = env.potassium.value
    env.update(current, jnp.zeros(2))
    expected = env.dt_ms*jnp.sum(outward_current_amount_rate(current))
    gained = jnp.sum((env.potassium.value-before)*env.k_transport.volumes)
    np.testing.assert_allclose(gained, expected, rtol=1e-4, atol=1e-15)
