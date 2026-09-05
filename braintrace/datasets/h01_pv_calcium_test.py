"""Calcium source units, state initialization, and mixed-ion routing."""

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Channel, Ion, MechanismProbe, CurrentProbe

from .h01_pv_calcium import PVCalcium
from . import h01_pv_channels  # Register source channels.


def test_calcium_influx_and_removal_follow_direct_analytic_trace():
    with brainstate.environ.context(precision=64):
        calcium = PVCalcium(1, decay=206.4043474695734*u.ms)
        voltage = jnp.array([-70.])*u.mV
        calcium.init_state(voltage)

        def step(index):
            current = jnp.where(index < 200, .2, 0.)*u.mA/u.cm**2
            value = calcium.Ci.value
            # Midpoint integration tests the derivative and State, with dt=0.5 ms.
            k1 = calcium.derivative(value, voltage, current)
            k2 = calcium.derivative(value+.25*u.ms*k1, voltage, current)
            calcium.Ci.value = value+.5*u.ms*k2
            return calcium.Ci.value.to_decimal(u.mM)

        observed = np.asarray(brainstate.transform.for_loop(step, jnp.arange(1600))).ravel()
    time = .5*np.arange(1, 1601)
    tau = 206.4043474695734
    influx = 10000*.2*.0005/(2*96485.33212331002*.1)
    loaded = influx*tau*(-np.expm1(-np.minimum(time, 100)/tau))
    expected = .0001+loaded*np.exp(-np.maximum(time-100, 0)/tau)
    np.testing.assert_allclose(observed, expected, rtol=4e-6, atol=1e-10)
    assert observed[199] == observed.max()


@pytest.mark.parametrize("current", [-.2, 0., .2])
def test_calcium_derivative_preserves_source_units_and_signed_flux(current):
    with brainstate.environ.context(precision=64):
        calcium = PVCalcium(1, decay=206.4043474695734*u.ms)
        v = np.array([-70.])*u.mV
        calcium.init_state(v)
        np.testing.assert_allclose(calcium.Ci.value.to_decimal(u.mM), 1e-4)
        expected_e = 8.31446261815324*307.15/(2*96485.33212331002)*np.log(2./1e-4)*1000
        np.testing.assert_allclose(calcium.E.to_decimal(u.mV), expected_e, rtol=1e-9)
        # Source outward current is the negative of BrainCell inward current.
        expected = 10000*current*.0005/(2*96485.33212331002*.1)-(.0007-.0001)/206.4043474695734
        actual = calcium.derivative(.0007*u.mM, v, current*u.mA/u.cm**2)
        np.testing.assert_allclose(actual.to_decimal(u.mM/u.ms), expected, rtol=1e-9)
        calcium.Ci.value = np.array([.0007])*u.mM
        assert float(calcium.E.to_decimal(u.mV)[0]) < expected_e
        calcium.reset_state(v)
        np.testing.assert_allclose(calcium.Ci.value.to_decimal(u.mM), 1e-4)


def test_assembled_calcium_and_sk_currents_have_separate_ion_owners():
    with brainstate.environ.context(precision=64):
        soma = braincell.Soma(lengths=np.array([10.])*u.um,
                             radii_proximal=np.array([5.])*u.um,
                             radii_distal=np.array([5.])*u.um)
        cell = braincell.Cell(braincell.Morphology(root_name="soma", root_branch=soma),
                              V_init=-40.*u.mV, solver="staggered")
        cell.paint(AllRegion(), Ion("H01PV_Calcium", name="calcium", decay=206.4043474695734*u.ms))
        cell.paint(AllRegion(), Ion("PotassiumFixed", name="potassium", E=-85.*u.mV))
        cell.paint(AllRegion(), Channel("H01PV_Ca_LVA", name="ca_channel", g_max=1.*u.mS/u.cm**2))
        cell.paint(AllRegion(), Channel("H01PV_SK", name="sk_channel", g_max=1.*u.mS/u.cm**2))
        for name, probe in {
            "ci": MechanismProbe(mechanism="calcium", field="Ci", name="ci"),
            "z": MechanismProbe(mechanism="sk_channel", field="z", name="z"),
            "ca_total": CurrentProbe(ion="calcium", name="ca_total"),
            "ca": CurrentProbe(mechanism="ca_channel", name="ca"),
            "k_total": CurrentProbe(ion="potassium", name="k_total"),
            "sk": CurrentProbe(mechanism="sk_channel", name="sk"),
        }.items():
            cell.place(RootLocation(.5), probe)
        result = cell.run(dt=.001*u.ms, duration=2.*u.ms)
        traces = {key: np.asarray(value.to_decimal(u.mM if key == "ci" else u.uA/u.cm**2))
                  if key != "z" else np.asarray(value) for key, value in result.traces.items()}
    ci = traces["ci"]
    assert np.isfinite(ci).all()
    assert ci[-1] > ci[0] >= 1e-4
    assert np.asarray(traces["z"])[-1] > np.asarray(traces["z"])[0]
    for total, mechanism in (("ca_total", "ca"), ("k_total", "sk")):
        np.testing.assert_allclose(traces[total], traces[mechanism], rtol=1e-12)
    assert np.all(traces["ca"] > 0)
    assert np.all(traces["sk"] < 0)
