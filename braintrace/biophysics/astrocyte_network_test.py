"""Spatial calcium transport, complete state resets and reservoir accounting."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest
import json
from pathlib import Path

from .astrocyte import CalciumParameters, initial_calcium, calcium_reaction_step
from .astrocyte_network import shell_transport, AstrocyteCalcium


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def test_source_annular_factors_and_bound_buffer_topology():
    graph = shell_transport([10., 10.], [2., 2.], [[0, 1]], dt_ms=.005, diffusion=.3)
    np.testing.assert_allclose(np.asarray(graph.volumes).reshape(2, 4).sum(axis=1), 10*np.pi)
    np.testing.assert_allclose(graph.conductance[:3], .3*10*np.pi*np.array([5., 3., 1.]))
    bound = shell_transport([10., 10.], [2., 2.], [[0, 1]], dt_ms=.005, diffusion=.05, radial=False)
    np.testing.assert_array_equal(bound.edges, [[0, 4], [1, 5], [2, 6], [3, 7]])


def test_spatial_spread_conserves_calcium_plus_bound_buffers():
    p = CalciumParameters(alpha=0., pump_velocity_um_ms=0.)
    model = AstrocyteCalcium([1., 1.], [2., 2.], [[0, 1]], parameters=p)
    model.state.value = model.state.value.at[0, 0].set(.001)
    volume = model.calcium_transport.volumes.reshape(2, 4)
    def amount():
        state = model.state.value
        return jnp.sum(volume*(state[:, :4]+state[:, 8:12]+state[:, 16:20]))
    before = amount()
    updated = brainstate.transform.jit(model.update)(jnp.zeros(2))
    assert model.valid.value and model.tick.value == 1
    assert updated[1, 0] > p.resting_mm
    np.testing.assert_allclose(amount(), before, atol=1e-12)
    assert model.er_amount.value == 0 and model.pumped_amount.value == 0
    model.reset_state()
    np.testing.assert_array_equal(model.state.value, model.initial)
    assert model.tick.value == 0


def test_reservoir_ledger_and_input_validation():
    model = AstrocyteCalcium([1.], [2.], np.empty((0, 2), dtype=int))
    before = np.asarray(model.state.value)
    volume = np.asarray(model.calcium_transport.volumes).reshape(1, 4)
    brainstate.transform.jit(model.update)(jnp.array([.1]), clamped_ip3_mm=jnp.array([.001]))
    after = np.asarray(model.state.value)
    change = np.sum(volume*((after[:, :4]+after[:, 8:12]+after[:, 16:20])-
                            (before[:, :4]+before[:, 8:12]+before[:, 16:20])))
    assert model.valid.value
    np.testing.assert_allclose(change, model.er_amount.value-model.pumped_amount.value, atol=1e-11)
    with pytest.raises(ValueError):
        model.update(jnp.zeros(2))
    with pytest.raises(ValueError):
        model.update(jnp.zeros(1), clamped_ip3_mm=jnp.zeros(2))


def test_ip3_clamp_is_visible_in_physical_state():
    p = CalciumParameters()
    result, valid, _ = calcium_reaction_step(initial_calcium(p), p, dt_ms=.005, clamped_ip3_mm=.001)
    assert valid
    assert result[24] == pytest.approx(.001)


@pytest.mark.parametrize('length,diameter,edges,diff', [([], [], [], .3),
    ([1.], [0.], np.empty((0, 2), dtype=int), .3),
    ([1.], [1.], [[0, 2]], .3), ([1.], [1.], np.empty((0, 2), dtype=int), -.1)])
def test_invalid_shell_geometry(length, diameter, edges, diff):
    with pytest.raises(ValueError):
        shell_transport(length, diameter, edges, dt_ms=.005, diffusion=diff)


def test_invalid_internal_step_count():
    with pytest.raises(ValueError, match='substeps'):
        AstrocyteCalcium([1.], [2.], np.empty((0, 2), int), substeps=0)


def test_spatial_trajectory_against_independent_neuron_and_time_refinement():
    reference = json.loads((Path(__file__).resolve().parents[2]/'docs/biology/calcium-spatial-reference.json').read_text())
    expected = np.asarray(reference['state'])
    def run(dt, steps):
        p = CalciumParameters(alpha=0., pump_velocity_um_ms=0.)
        model = AstrocyteCalcium([1.]*3, [2.]*3, [[0, 1], [1, 2]], dt_ms=dt, parameters=p, substeps=1)
        model.state.value = jnp.asarray(expected[0])
        def step(_):
            model.update(jnp.zeros(3))
            return model.state.value, model.valid.value
        result, valid = brainstate.transform.for_loop(step, jnp.arange(steps))
        assert np.all(valid)
        return np.asarray(result)
    coarse = run(.005, len(expected)-1)
    fine = run(.00125, 4*(len(expected)-1))[3::4]
    finer = run(.000625, 8*(len(expected)-1))[7::8]
    # All 25 state fields and both directions of spatial diffusion participate.
    # Compare source at its clock; qualify the finer selected clock separately.
    np.testing.assert_allclose(coarse, expected[1:], rtol=.05, atol=1e-9)
    np.testing.assert_allclose(fine, finer, rtol=.05, atol=1e-9)
    assert coarse[-1, 2, 0] > expected[0, 2, 0]
