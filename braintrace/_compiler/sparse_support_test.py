"""Conservative program-derived support for multicompartment factor blocks."""

import brainstate
import jax
import jax.numpy as jnp

from .sparse_support import analyze_transition


def test_static_indexing_keeps_cells_separate_through_jit():
    @jax.jit
    def cell(v, current):
        return jnp.tanh(v + current)
    def transition(y, state):
        return cell(state[0], y[0]), cell(state[1], y[1])
    layout = analyze_transition(transition, jnp.zeros(2), (jnp.zeros(3), jnp.zeros(5)))
    assert layout.outputs == ((0,), (1,))
    assert layout.color_count == 1


def test_rematerialization_preserves_disconnected_dependency_support():
    @jax.checkpoint
    def transition(y, state):
        return state[0]+y[0], state[1]+y[1]
    layout = analyze_transition(transition, jnp.zeros(2), (jnp.zeros(3), jnp.zeros(5)))
    assert layout.outputs == ((0,), (1,))


def test_scan_preserves_cell_separation_and_closes_internal_coupling():
    def transition(y, state):
        def step(h, _):
            return (h[0] + y[0] + h[1], .5*h[1] + h[0], h[2] + y[1]), None
        return brainstate.transform.scan(step, state, jnp.arange(20))[0]
    state = (jnp.zeros(3), jnp.zeros(3), jnp.zeros(7))
    layout = analyze_transition(transition, jnp.zeros(2), state)
    assert layout.outputs == ((0,), (0,), (1,))


def test_cond_unions_possible_branches():
    def transition(y, h):
        return (jax.lax.cond(h[0][0] > 0, lambda: h[0] + y[0], lambda: h[0] + y[1]),)
    layout = analyze_transition(transition, jnp.zeros(2), (jnp.zeros(3),))
    assert layout.outputs == ((0, 1),)


def test_unknown_position_mixing_widens_rather_than_drops():
    def transition(y, h):
        return (jnp.ones((3, 2)) @ y + h[0],)
    layout = analyze_transition(transition, jnp.zeros(2), (jnp.zeros(3),))
    assert layout.outputs == ((0, 1),)


def test_stop_gradient_does_not_create_a_differentiable_dependency():
    layout = analyze_transition(lambda y, h: (h[0] + jax.lax.stop_gradient(y[0]),),
                                jnp.zeros(1), (jnp.zeros(2),))
    assert layout.outputs == ((),)
