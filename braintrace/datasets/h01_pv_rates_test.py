"""Independent NMODL values, singular limits, and numerical gate properties."""

import json
from pathlib import Path

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .h01_pv_rates import pv_rates


ORACLE = json.loads((Path(__file__).resolve().parents[2] /
                    "docs/evidence/h01-pv-rate-oracle.json").read_text())
MECHANISMS = tuple(ORACLE["rates"])


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_all_rates_match_independently_compiled_original_mechanisms(mechanism):
    with brainstate.environ.context(precision=64):
        actual = pv_rates(mechanism, np.array(ORACLE["voltage_mv"]), np.array(ORACLE["calcium_mm"]))
    for gate, (inf, tau) in actual.items():
        expected = ORACLE["rates"][mechanism][gate]
        np.testing.assert_allclose(inf, expected["inf"], rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(tau, expected["tau_ms"], rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_float32_rates_and_voltage_derivatives_remain_finite(mechanism):
    def flattened(v):
        return jnp.stack([x for values in pv_rates(mechanism, v).values() for x in values])
    voltage = jnp.linspace(-140., 100., 241, dtype=jnp.float32)
    values = jax.jit(jax.vmap(flattened))(voltage)
    gradient = jax.jit(jax.vmap(jax.jacfwd(flattened)))(voltage)
    assert np.isfinite(values).all()
    assert np.isfinite(gradient).all()
    assert (np.asarray(values)[:, 1::2] > 0).all()
    assert (np.asarray(values)[:, ::2] >= 0).all()
    assert (np.asarray(values)[:, ::2] <= 1).all()


@pytest.mark.parametrize("mechanism,pole", [("NaTg", -38.), ("NaTg", -56.),
    ("Nap", -38.), ("Nap", -17.), ("Nap", -64.4), ("Ca_HVA", -27.),
    ("Ih", -262.07471272583734)])
def test_removable_poles_are_continuous_without_voltage_perturbation(mechanism, pole):
    with brainstate.environ.context(precision=64):
        values = pv_rates(mechanism, jnp.array([pole-1e-6, pole, pole+1e-6]))
    for inf, tau in values.values():
        for array in (np.asarray(inf), np.asarray(tau)):
            assert np.isfinite(array).all()
            np.testing.assert_allclose(array[1], (array[0]+array[2])/2, rtol=1e-10, atol=1e-12)


def test_sk_preserves_source_cutoff_and_half_activation():
    with brainstate.environ.context(precision=64):
        inf, tau = pv_rates("SK", -70., jnp.array([0., .000099999, .0001, .00043]))["z"]
    inf = np.asarray(inf)
    np.testing.assert_array_equal(np.asarray(inf)[:2], 0.)
    assert 0 < inf[2] < .01
    assert float(inf[3]) == pytest.approx(.5)
    np.testing.assert_array_equal(tau, 1.)


def test_unknown_mechanism_is_rejected():
    with pytest.raises(ValueError, match="Unknown PV"):
        pv_rates("not_a_channel", -70.)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_targeted_gate_matches_full_dictionary(mechanism):
    with brainstate.environ.context(precision=64):
        full = pv_rates(mechanism, np.array(ORACLE["voltage_mv"]), np.array(ORACLE["calcium_mm"]))
        for gate, (exp_inf, exp_tau) in full.items():
            inf, tau = pv_rates(mechanism, np.array(ORACLE["voltage_mv"]), np.array(ORACLE["calcium_mm"]), gate=gate)
            np.testing.assert_array_equal(inf, exp_inf)
            np.testing.assert_array_equal(tau, exp_tau)
