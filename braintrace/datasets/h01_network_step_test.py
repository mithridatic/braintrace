"""Native forward and reset checks for the reusable network step."""

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Channel, StateProbe

from .h01_network_step import H01NetworkStep


def _network(tmp_path):
    swc = tmp_path / "cable.swc"
    swc.write_text("1 0 0 0 0 2 -1\n2 0 10 0 0 2 1\n3 0 20 0 0 1 2\n")
    cell = braincell.Cell(braincell.Morphology.from_swc(swc), V_init=-65*u.mV, pop_size=(1,))
    cell.paint(AllRegion(), Channel("IL", g_max=.1*u.mS/u.cm**2, E=-65*u.mV))
    cell.place(RootLocation(0.), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(0.), braincell.CurrentClamp(
        delay=0.*u.ms, durations=1.*u.ms, amplitudes=.001*u.nA))
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
