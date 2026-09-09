"""Finite-window learning properties for sparse multicompartment IO factors."""

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._testing.oracle import bptt_param_gradients, chunked_online_param_gradients
from .sparse_pp_prop import SparsePPProp


class _Blocks(brainstate.nn.Module):
    def __init__(self, recurrence=.5):
        super().__init__()
        self.weight = brainstate.ParamState(jnp.array([[.2, .3], [.4, .5]]))
        self.readout = brainstate.ParamState(jnp.array([.7, .8]))
        self.a = brainstate.HiddenState(jnp.zeros(3))
        self.b = brainstate.HiddenState(jnp.zeros(5))
        self.recurrence = recurrence

    def update(self, x):
        y = braintrace.matmul(x, self.weight.value)
        self.a.value = jnp.tanh(self.recurrence*self.a.value + y[0])
        self.b.value = jnp.tanh(self.recurrence*self.b.value + y[1])
        return jnp.array([self.a.value[0], self.b.value[0]]) * self.readout.value


def _gradients(recurrence, decay):
    inputs = jnp.array([[.1, -.2], [.5, .1], [-.3, .2], [.2, .4]])
    return chunked_online_param_gradients(lambda: _Blocks(recurrence), inputs,
        algo_factory=lambda model: SparsePPProp(model, decay), chunk_size=1, compiled_scan=True)


def test_finite_window_memoryless_exactness_including_direct_readout():
    inputs = jnp.array([[.1, -.2], [.5, .1], [-.3, .2], [.2, .4]])
    expected = bptt_param_gradients(lambda: _Blocks(0.), inputs)
    got = _gradients(0., .8)
    for actual, wanted in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, wanted, rtol=1e-5, atol=1e-7)


def test_finite_window_gradient_depends_on_trace_decay():
    a, b = _gradients(.7, .2), _gradients(.7, .8)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(a))
    assert any(not np.allclose(x, y, rtol=1e-4, atol=1e-7)
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def test_trace_reset_and_nonzero_update_descends_controlled_loss():
    model = _Blocks(.5)
    learner = SparsePPProp(model, .5)
    sequence = jnp.array([[.3, .1], [.2, .4]])
    learner.compile_graph(sequence[0])
    params = model.states(brainstate.ParamState)
    def loss():
        return jnp.square(learner(braintrace.MultiStepData(sequence))).sum()
    grad, before = brainstate.transform.grad(loss, params, return_value=True)()
    assert any(np.linalg.norm(g) > 0 for g in jax.tree.leaves(grad))
    for path, state in params.items():
        state.value = state.value - .01*grad[path]
    model.a.value = jnp.zeros(3)
    model.b.value = jnp.zeros(5)
    learner.reset_state()
    assert int(learner.running_index.value) == 0
    assert all(not np.any(f) for f in learner.factors.value)
    assert float(loss()) < float(before)


def test_real_cable_finite_window_gradients_for_input_and_active_contact(tmp_path):
    from braintrace.datasets.h01_network_step_test import _contact_network
    from examples.pp_prop.h01_arc_model import H01ArcModel
    class ReadoutModel(H01ArcModel):
        def update(self, event):
            super().update(event)
            return self.readout()
    def factory():
        model = ReadoutModel(_contact_network(tmp_path), ['5805562981', '5965472721'])
        model.input_weight.value = jnp.ones_like(model.input_weight.value)*.01
        return model
    captured = {}
    sequence = jnp.ones((3, 441))
    gradients = chunked_online_param_gradients(factory, sequence,
        algo_factory=lambda model: SparsePPProp(model, .5), chunk_size=1, compiled_scan=True,
        initialize_model=False, after_init=lambda m, a: captured.update(model=m, learner=a))
    for name in ('input_weight', 'recurrent_weight'):
        gradient = gradients[(name,)]
        assert np.isfinite(gradient).all()
        assert np.linalg.norm(gradient) > 0
    model, learner = captured['model'], captured['learner']
    evaluate = brainstate.transform.jit(lambda: jnp.square(
        brainstate.transform.for_loop(model.update, sequence)).sum())
    model.reset_episode(learner)
    before_loss = float(evaluate())
    for name in ('input_weight', 'recurrent_weight'):
        state = getattr(model, name)
        before = np.array(state.value)
        gradient = gradients[(name,)]
        # Keep the controlled step inside the current contact-magnitude branch;
        # an unscaled step can cross zero and reflect through abs(weight).
        rate = 1e-3*jnp.max(jnp.abs(state.value))/(jnp.max(jnp.abs(gradient)) + 1e-20)
        state.value = state.value - rate*gradient
        assert not np.array_equal(before, state.value), name
    model.reset_episode(learner)
    assert float(evaluate()) < before_loss


def test_public_sparse_factory_padding_preserves_every_state():
    model = _Blocks()
    learner = braintrace.pp_prop.sparse(model, .5)
    learner.compile_graph(jnp.ones(2))
    learner(jnp.ones(2))
    states = brainstate.graph.states(learner)
    before = {path: jax.tree.map(np.array, state.value) for path, state in states.items()}
    result = brainstate.transform.jit(learner.update)(jnp.ones(2), False)
    np.testing.assert_array_equal(result, jnp.zeros(2))
    for path, state in states.items():
        for a, b in zip(jax.tree.leaves(before[path]), jax.tree.leaves(state.value)):
            np.testing.assert_array_equal(a, b)
