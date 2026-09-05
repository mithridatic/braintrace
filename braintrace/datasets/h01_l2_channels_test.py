"""Independent source-rate checks for selected E and I channel transfers."""
import json
from pathlib import Path
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from .h01_l2_channels import L2_CHANNELS, l2_rates
from .h01_pv_channels import _CHANNELS
from .h01_pv_channels_test import _ions
from .h01_ei_profiles import get_ei_profile, channel_controls

ROOT = Path(__file__).resolve().parents[2]/"docs/evidence"


@pytest.mark.parametrize("role", ["E", "I"])
def test_all_gates_against_independent_neuron_rates(role):
    oracle = json.loads((ROOT/f"h01-ei-{role.lower()}-gate-oracle.json").read_text())
    with brainstate.environ.context(precision=64):
        for case in oracle["cases"]:
            mechanism = case["mechanism"]
            profile = get_ei_profile(role, mode=case["mode"])
            cls = (L2_CHANNELS if role == "E" else _CHANNELS)[mechanism]
            channel = cls(1, 2.*u.mS/u.cm**2, **channel_controls(profile, "soma", mechanism))
            v = jnp.array([case["voltage_mv"]])*u.mV
            ions = _ions("NaTg" if mechanism == "NaTs" else mechanism, case["calcium_mm"])
            # PV source shifts exact poles by 0.0001 mV; our analytic limit does not.
            tolerance = 5e-5 if role == "I" and case["voltage_mv"] in (-27., -38., -56.) else 2e-7
            channel.init_state(v, *ions)
            for gate, expected in case["gates"].items():
                getattr(channel, gate).value = jnp.array([case["state"]])
                np.testing.assert_allclose(getattr(channel, f"f_{gate}_inf")(v, *ions), expected["inf"],
                    atol=tolerance, rtol=tolerance, err_msg=str(case))
                np.testing.assert_allclose(getattr(channel, f"f_{gate}_tau")(v, *ions), expected["tau_ms"],
                    atol=tolerance, rtol=tolerance, err_msg=str(case))
            assert np.isfinite(channel.current(v, *ions).to_decimal(u.uA/u.cm**2)).all()


def test_unknown_l2_and_invalid_phase_controls():
    with pytest.raises(ValueError, match="Unknown L2"):
        l2_rates("bad", -80.)
    for kwargs in ({"m_open": 0.}, {"h_slope": np.nan}, {"h_close": np.inf}):
        with pytest.raises(ValueError, match="positive and finite"):
            L2_CHANNELS["NaTs"](1, 1.*u.mS/u.cm**2, **kwargs)


def test_nap_activation_is_instantaneous_first_power():
    with brainstate.environ.context(precision=64):
        channel = L2_CHANNELS["Nap"](1, 2.*u.mS/u.cm**2)
        v = jnp.array([-52.6])*u.mV
        ions = _ions("Nap", 1e-4)
        channel.init_state(v, *ions)
        channel.h.value = jnp.array([.25])
        np.testing.assert_allclose(channel.conductance_factor(v, *ions), .125)
        assert not hasattr(channel, "m")


@pytest.mark.parametrize("role", ["E", "I"])
def test_kv3_equilibrium_retains_source_opening_branch(role):
    with brainstate.environ.context(precision=64):
        profile = get_ei_profile(role)
        cls = (L2_CHANNELS if role == "E" else _CHANNELS)["Kv3_1"]
        ch = cls(1, 2.*u.mS/u.cm**2, **channel_controls(profile, "soma", "Kv3_1"))
        voltage = jnp.array([-60.])*u.mV
        ions = _ions("Kv3_1", 1e-4)
        ch.init_state(voltage, *ions)
        base_tau = 4./(1.+np.exp(-(-60.+46.56)/44.14))
        np.testing.assert_allclose(ch.f_m_tau(voltage, *ions), base_tau*(1. if role == "E" else .5))
        ch.compute_derivative(voltage, *ions)
        np.testing.assert_array_equal(ch.m.derivative.to_decimal(u.kHz), 0.)


def test_targeted_l2_rates_match_full_dictionary():
    with brainstate.environ.context(precision=64):
        vs = np.linspace(-100., 40., 20)
        for mech in list(L2_CHANNELS.keys()):
            full = l2_rates(mech, vs)
            for gate, (exp_inf, exp_tau) in full.items():
                inf, tau = l2_rates(mech, vs, gate=gate)
                np.testing.assert_array_equal(inf, exp_inf)
                np.testing.assert_array_equal(tau, exp_tau)
