"""Bounded cable-model interface and compiler integration checks."""

import brainstate
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_network_step_test import _network, _contact_network
from .h01_arc_model import H01ArcModel


def test_forward_padding_and_readout(tmp_path):
    model = H01ArcModel(_network(tmp_path, current_na=0), ["5805562981"])
    event = jnp.ones(441)
    before = np.asarray(model._soma())
    assert model.readout().shape == (360,)
    np.testing.assert_array_equal(brainstate.transform.jit(model.step)(event, False), [0.])
    np.testing.assert_array_equal(before, model._soma())
    assert model.interval(event, False).shape == (1,)
    with pytest.raises(ValueError, match="refinement"):
        model.interval(event, substeps=2)
    with pytest.raises(ValueError, match="manifest"):
        model.step(event, blocked_source=0)
    assert int(model.stepper.tick.value) == 0
    result = brainstate.transform.jit(model.step)(event)
    assert np.isfinite(result).all()
    assert int(model.stepper.tick.value) == 20
    model.reset_episode()
    assert int(model.stepper.tick.value) == 0
    np.testing.assert_array_equal(before, model._soma())


@pytest.mark.xfail(strict=True, raises=braintrace.NotSupportedError,
                   reason="pp-prop factorized traces reject cable position mixing; integration gate remains open")
@pytest.mark.parametrize("unroll", [16, 32])
def test_pp_prop_compiles_real_cable_state(tmp_path, unroll, record_property):
    model = H01ArcModel(_network(tmp_path), ["5805562981"])
    learner = braintrace.pp_prop(model, decay_or_rank=.99, vjp_method="single-step",
                                control_flow=braintrace.ControlFlowPolicy(scan_unroll_limit=unroll))
    try:
        learner.compile_graph(jnp.zeros(441))
    except braintrace.NotSupportedError as exc:
        assert "position-preserving" in str(exc)
        record_property("compiler_blocker", str(exc))
        raise
    assert learner is not None


@pytest.mark.parametrize("ids,dt,message", [
    ([], .005, "string"), ([12], .005, "string"),
    (["a", "a"], .005, "unique"), (["a", "b"], .005, "match"),
    (["a"], 0., "divide"), (["a"], float("nan"), "divide"),
    (["a"], .03, "divide"),
])
def test_invalid_model_manifest(tmp_path, ids, dt, message):
    with pytest.raises(ValueError, match=message):
        H01ArcModel(_network(tmp_path), ids, dt_ms=dt)


def test_contact_magnitudes_queue_and_eligibility_reset(tmp_path):
    from types import SimpleNamespace
    model = H01ArcModel(_contact_network(tmp_path), ["5805562981", "5965472721"])
    model.recurrent_weight.value = -jnp.ones(1)*.01
    result = brainstate.transform.jit(model.update)(jnp.zeros(441))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(model.contact_magnitude.value, [.01])
    assert isinstance(model.stepper.ring_buffers[0], brainstate.HiddenState)
    model.stepper.ring_buffers[0].value = u.math.ones_like(model.stepper.ring_buffers[0].value)
    states = brainstate.graph.states(model)
    before = {path: [np.array(u.get_mantissa(x)) for x in jax.tree_util.tree_leaves(state.value)]
              for path, state in states.items()}
    brainstate.transform.jit(model.step)(jnp.ones(441), False)
    for path, state in states.items():
        for expected, actual in zip(before[path], jax.tree_util.tree_leaves(state.value), strict=True):
            np.testing.assert_array_equal(expected, u.get_mantissa(actual), err_msg=str(path))
    calls = []
    model.reset_episode(SimpleNamespace(reset_state=lambda: calls.append(True)))
    assert calls == [True]
    assert np.count_nonzero(model.stepper.ring_buffers[0].value.mantissa) == 0
