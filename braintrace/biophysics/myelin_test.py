"""Radial axon/sheath flux, node boundaries and source pump stoichiometry."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from .myelin import MyelinPotassium, linear_pump_ma_cm2, nak_pump_ma_cm2


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def test_source_pump_rates_and_stoichiometry():
    np.testing.assert_allclose(linear_pump_ma_cm2(jnp.array([55., 110., 220.])), [-.05, 0., .1])
    na, k = nak_pump_ma_cm2(-80., 10., 2.5, temperature_c=37.)
    expected = .06*70/120/4
    np.testing.assert_allclose([na, k], [3*expected, -2*expected])
    assert np.isnan(linear_pump_ma_cm2(-1.))
    assert np.isnan(nak_pump_ma_cm2(-200., 10., 2.5)[0])


def domain():
    return MyelinPotassium([10., 10.], [1., 1.1, 1.5, 2.], [1., .1, .1],
                           [True, False], 1, [50.], [140., 140., 140.])


def test_radial_exchange_balances_both_cells_and_boundary_without_axial_diffusion():
    model = domain()
    env = model.environment
    def amount():
        return jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    before = amount()
    k = brainstate.transform.jit(model.update)(jnp.array([.01, 0.]), jnp.array([-.003]))
    np.testing.assert_allclose(amount()-before, env.k_boundary_amount.value,
                               atol=8*np.finfo(float).eps*float(before))
    np.testing.assert_allclose(k[1], 2.5, atol=1e-13)
    assert not np.allclose(k[0], 2.5, atol=1e-8, rtol=0.)
    assert env.inside_potassium.value[0] < 140. and env.inside_potassium.value[2] > 140.
    assert env.valid.value and env.tick.value == 1
    model.reset_state()
    assert env.tick.value == 0 and np.allclose(env.potassium.value, 2.5)
    with pytest.raises(ValueError):
        model.update(jnp.ones(2), jnp.ones(2))


@pytest.mark.parametrize('mask,shell,volume', [([1, 0], 1, [50.]), ([True, False], 9, [50.]),
    ([True, False], 1, []), ([True, False], 1, [-1.])])
def test_invalid_sheath_configuration(mask, shell, volume):
    with pytest.raises(ValueError):
        MyelinPotassium([10., 10.], [1., 1.1, 2.], [1., .1], mask, shell, volume, [140.]*3)
