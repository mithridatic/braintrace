"""Source Kir current sign, units and dependence on dynamic potassium."""

import brainstate
import brainunit as u
import numpy as np
from braincell.ion import PotassiumFixed

from .cable_kir import BraincellKir41
from .astrocyte import kir41_current_ma_cm2


def test_source_kir_density_and_dynamic_reversal():
    with brainstate.environ.context(precision=64):
        channel = BraincellKir41(2)
        voltage = np.array([-100., -70.])*u.mV
        # Use the runtime's registered IonInfo pytree for compiled arguments.
        ion = PotassiumFixed(2, E=np.array([-70., -100.])*u.mV, Co=np.array([10., 2.5])*u.mM).pack_info()
        current = brainstate.transform.jit(channel.current)(voltage, ion)
        expected = -kir41_current_ma_cm2(voltage.to_decimal(u.mV), ion.E.to_decimal(u.mV), ion.Co.to_decimal(u.mM))
        np.testing.assert_allclose(current.to_decimal(u.mA/u.cm**2), expected)
        assert current[0] > 0*u.mA/u.cm**2 and current[1] < 0*u.mA/u.cm**2
        assert not brainstate.graph.states(channel, brainstate.ParamState)


def test_dynamic_cable_voltage_matches_independent_neuron_reference():
    import json
    from pathlib import Path
    import braincell
    import jax.numpy as jnp
    from braincell.filter import AllRegion
    from braincell.mech import Channel, Ion
    from braintrace.datasets.h01_network_step import H01NetworkStep
    reference = json.loads((Path(__file__).resolve().parents[2]/'docs/biology/neuroglial-phase/kir-reference.json').read_text())
    with brainstate.environ.context(precision=64):
        branch = braincell.Branch(lengths=np.array([1.])*u.um,
            radii_proximal=np.array([1.])*u.um, radii_distal=np.array([1.])*u.um)
        cell = braincell.Cell(braincell.Morphology(root_name='glia', root_branch=branch),
            V_init=reference['initial_mv']*u.mV, pop_size=(1,), cv_policy=braincell.MaxCVLen(10*u.um))
        cell.paint(AllRegion(), braincell.CableProperty(membrane_capacitance=1.*u.uF/u.cm**2,
                                                       resting_potential=reference['initial_mv']*u.mV,
                                                       axial_resistivity=100.*u.ohm*u.cm))
        cell.paint(AllRegion(), Ion('PotassiumFixed', name='potassium', E=reference['reversal_mv']*u.mV,
                                   Ci=140.*u.mM, Co=10.*u.mM))
        cell.paint(AllRegion(), Channel('BraincellKir41', g_max=.4*u.mS/u.cm**2))
        network = braincell.Network()
        network.add_population('glia', cell)
        step = H01NetworkStep(network, dt_ms=reference['dt_ms'])
        def advance(_):
            step.update(sample_probes=False)
            return cell.V.value.to_decimal(u.mV).reshape(())
        actual = brainstate.transform.for_loop(advance, jnp.arange(len(reference['time_ms'])-1))
        np.testing.assert_allclose(actual, reference['voltage_mv'][1:], rtol=0, atol=.001)
