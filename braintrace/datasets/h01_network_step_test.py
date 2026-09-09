"""Native forward and reset checks for the reusable network step."""

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Channel, StateProbe, Synapse
from braincell.network import pairs

from .h01_network_step import H01NetworkStep


def _network(tmp_path, current_na=.001):
    swc = tmp_path / "cable.swc"
    swc.write_text("1 0 0 0 0 2 -1\n2 0 10 0 0 2 1\n3 0 20 0 0 1 2\n")
    cell = braincell.Cell(braincell.Morphology.from_swc(swc), V_init=-65.*u.mV,
                          V_th=-64.999*u.mV, pop_size=(1,), cv_policy=braincell.MaxCVLen(10*u.um))
    cell.paint(AllRegion(), Channel("IL", g_max=.1*u.mS/u.cm**2, E=-65*u.mV))
    cell.place(RootLocation(0.), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(0.), braincell.CurrentClamp(
        delay=0.*u.ms, durations=1.*u.ms, amplitudes=current_na*u.nA))
    network = braincell.Network()
    network.add_population("cell", cell)
    return network


def test_native_step_and_reset(tmp_path):
    network = _network(tmp_path)
    expected = network.run(dt=.005*u.ms, duration=.1*u.ms).traces
    network.reset_state()
    step = H01NetworkStep(network)
    result = brainstate.transform.for_loop(lambda _: step.update(), jnp.arange(20))
    np.testing.assert_allclose(result["cell"]["voltage"].to_decimal(u.mV),
                               expected["cell"]["voltage"].to_decimal(u.mV), atol=1e-10)
    assert int(step.tick.value) == 20
    step.reset_state()
    assert int(step.tick.value) == 0
    repeated = brainstate.transform.for_loop(lambda _: step.update(), jnp.arange(20))
    np.testing.assert_array_equal(result["cell"]["voltage"].mantissa,
                                  repeated["cell"]["voltage"].mantissa)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_dt(dt):
    with pytest.raises(ValueError, match="dt_ms"):
        H01NetworkStep(braincell.Network(), dt)


def test_empty_network():
    with pytest.raises(ValueError, match="empty"):
        H01NetworkStep(braincell.Network())


def _contact_network(tmp_path):
    network = _network(tmp_path)
    target = _network(tmp_path, current_na=0).populations["cell"].cell
    target.place(RootLocation(.25), Synapse("ExpSyn", name="contact", e=0.*u.mV,
                                          tau=2.*u.ms, weight=1.*u.uS))
    network.add_population("post", target)
    network.add_edges(name="edge", pre="cell", post="post", method=pairs([(0, 0)]))
    network.add_projection(name="contact", edges="edge", synapse="contact",
                           weight=.01*u.uS, delay=.01*u.ms)
    return network


def test_active_delayed_contact_matches_native_and_reset_clears_queue(tmp_path):
    network = _contact_network(tmp_path)
    native = network.run(dt=.005*u.ms, duration=.1*u.ms)
    assert np.any(native.spikes["cell"])
    assert float(native.traces["post"]["voltage"][-1].to_decimal(u.mV).reshape(())) > -65
    network.reset_state()
    step = H01NetworkStep(network)
    actual = brainstate.transform.for_loop(lambda _: step.update(), jnp.arange(20))
    np.testing.assert_allclose(actual["post"]["voltage"].to_decimal(u.mV),
                               native.traces["post"]["voltage"].to_decimal(u.mV), atol=1e-9)
    step.ring_buffers[0].value = u.math.ones_like(step.ring_buffers[0].value)
    step.reset_state()
    assert np.count_nonzero(step.ring_buffers[0].value.mantissa) == 0
    assert int(step.ring_cursors[0].value) == 0
