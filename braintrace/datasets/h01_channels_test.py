"""Stateful channel integration and reference-compartment smoke checks."""

import braincell
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell._base import IonInfo

from .h01_channels import HumanSodium, HumanPotassium


def test_gate_initialization_currents_and_slow_recovery():
    with brainstate.environ.context(precision=64):
        v = np.array([-70.]) * u.mV
        for cls, reversal in ((HumanSodium, 68), (HumanPotassium, -86)):
            ion = IonInfo(E=reversal*u.mV, Ci=None, Co=None, valence=1)
            channel = cls(1, g_max=1*u.mS/u.cm**2)
            channel.init_state(v, ion)
            np.testing.assert_allclose(channel.h.value, channel.f_h_inf(v, ion))
            channel.reset_state(v, ion)
            channel.compute_derivative(v, ion)
            for gate in channel._iter_gates():
                st = getattr(channel, gate.name)
                np.testing.assert_allclose(st.derivative.to_decimal(u.kHz), 0.)
            current = channel.current(v, ion).to_decimal(u.uA/u.cm**2)
            assert np.sign(current[0]) == np.sign(reversal+70)
            if cls is HumanPotassium:
                channel.h2.value = np.array([0.])
                assert float(channel.f_h2_tau(v, ion)[0]) == pytest.approx(2789.7929428108187)
                channel.h2.value = np.array([1.])
                assert float(channel.f_h2_tau(v, ion)[0]) == 450.


from .h01_reference import make_reference_cell as reference_cell


def test_compiled_reference_compartment_fires():
    with brainstate.environ.context(precision=64):
        result = reference_cell().run(dt=.005*u.ms, duration=30*u.ms)
    v = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    assert np.isfinite(v).all()
    assert v.max() > 0
    assert v[-1] < -50
