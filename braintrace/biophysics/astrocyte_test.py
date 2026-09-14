"""Buffer conservation, ER equilibrium and reaction derivative checks."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import json
from pathlib import Path

from .astrocyte import CalciumParameters, initial_calcium, calcium_derivative, calcium_reaction_step, kir41_current_ma_cm2


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def test_resting_buffers_and_er_equilibrium():
    p = CalciumParameters()
    state = initial_calcium(p)
    rate, er, pump = calcium_derivative(state, p, clamped_ip3_mm=p.ip3_resting_mm)
    np.testing.assert_allclose(rate, 0., atol=1e-16)
    np.testing.assert_allclose(er, 0., atol=1e-20)
    assert pump == 0
    updated, valid, residual = calcium_reaction_step(state, p, dt_ms=.005, clamped_ip3_mm=p.ip3_resting_mm)
    assert valid and residual < 1e-12
    np.testing.assert_allclose(updated, state, atol=1e-15)


def test_binding_preserves_buffer_and_calcium_amount():
    p = CalciumParameters(alpha=0.)
    state = initial_calcium(p).at[:4].set(.001)
    updated, valid, _ = calcium_reaction_step(state, p, dt_ms=.01, clamped_ip3_mm=.00005)
    assert valid
    old, new = state[:24].reshape(6, 4), updated[:24].reshape(6, 4)
    np.testing.assert_allclose(new[1]+new[2], old[1]+old[2], atol=1e-13)
    np.testing.assert_allclose(new[3]+new[4], old[3]+old[4], atol=1e-13)
    np.testing.assert_allclose(new[0]+new[2]+new[4], old[0]+old[2]+old[4], atol=1e-13)


def test_stimulation_pump_and_implicit_gradient():
    p = CalciumParameters()
    state = initial_calcium(p)
    base = calcium_derivative(state, p)[0][-1]
    stimulated, _, pump = calcium_derivative(state, p, glutamate_mm=.1, surface_to_volume=4.)
    assert stimulated[-1] > base and pump > 0
    f = lambda dose: calcium_reaction_step(state, p, dt_ms=.005, influx_mm_ms=jnp.ones(4)*dose)[0][0]
    dose, epsilon = 1e-4, 1e-7
    expected = (f(dose+epsilon)-f(dose-epsilon))/(2*epsilon)
    np.testing.assert_allclose(jax.grad(f)(dose), expected, rtol=1e-6)


def test_kir_formula_against_source_expression():
    v, ek, ko, g = -80., -100., 2.5, .4
    expected = .001*g*(v-ek*.81+14.83)*np.sqrt(ko/(1+np.exp((v-ek*.81+105.82)/19.23)))
    assert kir41_current_ma_cm2(v, ek, ko) == pytest.approx(expected)
    assert np.isnan(kir41_current_ma_cm2(v, ek, -1.))
    with pytest.raises(ValueError):
        CalciumParameters(stationary_mm=-1.)


def test_full_local_trajectory_against_independent_neuron():
    reference = json.loads((Path(__file__).resolve().parents[2]/'docs/biology/calcium-reference.json').read_text())
    p = CalciumParameters()
    expected = np.asarray(reference['state'])
    np.testing.assert_allclose(initial_calcium(p), expected[0], atol=1e-14)
    def run():
        def step(state, _):
            updated, valid, _ = calcium_reaction_step(state, p, dt_ms=reference['dt_ms'], glutamate_mm=.1)
            return updated, (updated, valid)
        return brainstate.transform.scan(step, initial_calcium(p), jnp.arange(len(expected)-1))[1]
    actual, valid = brainstate.transform.jit(run)()
    assert np.all(valid)
    np.testing.assert_allclose(actual, expected[1:], rtol=1e-7, atol=1e-12)
