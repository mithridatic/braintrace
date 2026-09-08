"""State dynamics and ionic current checks against compiled source rates."""

import json
from pathlib import Path

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell._base import IonInfo
from braincell.ion import Potassium

from .h01_pv_channels import make_pv_channel, PV_CHANNEL_NAMES


ORACLE = json.loads((Path(__file__).resolve().parents[2] /
                    "docs/evidence/h01-pv-rate-oracle.json").read_text())
POWERS = {"NaTg": (3, 1), "Nap": (3, 1), "K_P": (2, 1),
          "K_T": (4, 1), "Kv3_1": (1,), "Im": (1,), "Ih": (1,),
          "Ca_HVA": (2, 1), "Ca_LVA": (2, 1), "SK": (1,)}


def _ions(mechanism, calcium):
    def ion(e, ci=None):
        return IonInfo(E=e*u.mV, Ci=ci, Co=None, valence=1)
    if mechanism == "SK":
        return ion(-85.), ion(120., jnp.array([calcium])*u.mM)
    if mechanism == "Ih":
        return ()
    return (ion(50. if mechanism in ("NaTg", "Nap") else
                120. if mechanism.startswith("Ca_") else -85.),)


@pytest.mark.parametrize("mechanism", tuple(PV_CHANNEL_NAMES))
def test_clamped_gate_relaxation_and_current_from_independent_rates(mechanism):
    """Integrate channel derivatives; compare the full trace to exact relaxation."""
    start, end = (0, 5) if mechanism == "SK" else (3, 10)
    rates = ORACLE["rates"][mechanism]
    gates = tuple(rates)
    initial = np.array([rates[g]["inf"][start] for g in gates])
    target = np.array([rates[g]["inf"][end] for g in gates])
    tau = np.array([rates[g]["tau_ms"][end] for g in gates])
    dt = float(tau.min()/20.)
    steps = 100
    with brainstate.environ.context(precision=64):
        channel = make_pv_channel(mechanism, 1, 2.*u.mS/u.cm**2)
        v0 = jnp.array([ORACLE["voltage_mv"][start]])*u.mV
        v1 = jnp.array([ORACLE["voltage_mv"][end]])*u.mV
        ions0 = _ions(mechanism, ORACLE["calcium_mm"][start] if mechanism == "SK" else 1e-4)
        ions1 = _ions(mechanism, ORACLE["calcium_mm"][end] if mechanism == "SK" else 1e-4)
        channel.init_state(v0, *ions0)
        np.testing.assert_allclose([getattr(channel, g).value[0] for g in gates], initial, rtol=2e-12)
        channel.compute_derivative(v0, *ions0)
        for g in gates:
            np.testing.assert_allclose(getattr(channel, g).derivative.to_decimal(u.kHz), 0.)

        # RK4 is a test driver for a voltage clamp, not the proposed cell solver.
        def derivative(values):
            for index, gate in enumerate(gates):
                getattr(channel, gate).value = values[index:index+1]
            channel.compute_derivative(v1, *ions1)
            return jnp.stack([getattr(channel, g).derivative.to_decimal(u.kHz)[0] for g in gates])

        def step(_):
            values = jnp.stack([getattr(channel, g).value[0] for g in gates])
            k1 = derivative(values)
            k2 = derivative(values+dt*k1/2)
            k3 = derivative(values+dt*k2/2)
            k4 = derivative(values+dt*k3)
            updated = values+dt*(k1+2*k2+2*k3+k4)/6
            for index, gate in enumerate(gates):
                getattr(channel, gate).value = updated[index:index+1]
            return updated, channel.current(v1, *ions1).to_decimal(u.uA/u.cm**2)[0]

        observed, current = brainstate.transform.for_loop(step, jnp.arange(steps))
    times = dt*np.arange(1, steps+1)
    expected = target+(initial-target)*np.exp(-times[:, None]/tau)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-8)
    reversal = -45. if mechanism == "Ih" else float(ions1[0].E.to_decimal(u.mV))
    expected_current = 2.*np.prod(expected**np.array(POWERS[mechanism]), axis=1)*(reversal-float(v1.to_decimal(u.mV)[0]))
    np.testing.assert_allclose(current, expected_current, rtol=8e-6, atol=1e-9)
    assert np.all((np.asarray(observed) >= 0.) & (np.asarray(observed) <= 1.))
    if mechanism == "SK":
        assert channel.current_owner_type is Potassium


def test_unknown_channel_fails_before_state_allocation():
    with pytest.raises(ValueError, match="Unknown PV mechanism"):
        make_pv_channel("unknown", 1, 1.*u.mS/u.cm**2)
