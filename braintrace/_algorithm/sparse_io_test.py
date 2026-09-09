"""Exact local derivative checks for colored sparse IO-factor execution."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._compiler.sparse_influence import SparseInfluence
from .sparse_io import advance_factors, instant_factors, propagate_factors, contract_factors


def test_instantaneous_mixed_shape_columns_match_ad():
    layout = SparseInfluence.build([(3,), (2,)], [{0, 1}, {2}],
                                   [set(), set()], output_size=3)
    def tail(y):
        return (jnp.tanh(jnp.array([y[0], y[1], y[0] + 2*y[1]])),
                jnp.array([y[2]**2, 3*y[2]]))
    y = jnp.array([.2, -.4, .7])
    actual = brainstate.transform.jit(lambda v: instant_factors(tail, v, layout))(y)
    expected = jax.jacfwd(tail)(y)
    for got, full, indices in zip(actual, expected, layout.outputs):
        np.testing.assert_allclose(got, full[..., jnp.array(indices)], rtol=1e-6)


def test_recurrence_keeps_cross_block_cable_derivatives():
    layout = SparseInfluence.build([(2,), (1,), (1,)], [{0}, {1}, {2}],
                                   [{0, 1}, {0, 1}, {2}], output_size=3)
    state = (jnp.array([.2, .3]), jnp.array([.4]), jnp.array([.5]))
    factors = (jnp.array([[1., 2.], [3., 4.]]), jnp.array([[5., 6.]]), jnp.array([[7.]]))
    def transition(h):
        return (jnp.tanh(h[0] + h[1]), jnp.array([h[0].sum() + 2*h[1][0]]), h[2]**2)
    got = brainstate.transform.jit(lambda h, f: propagate_factors(transition, h, f, layout))(state, factors)
    # Compare every sparse column to an independently seeded full JVP.
    for output in range(3):
        seed = tuple(f[..., row.index(output)] if output in row else jnp.zeros_like(h)
                     for h, f, row in zip(state, factors, layout.outputs))
        _, derivative = jax.jvp(transition, (state,), (seed,))
        for actual, expected, row in zip(got, derivative, layout.outputs):
            if output in row:
                np.testing.assert_allclose(actual[..., row.index(output)], expected, rtol=1e-6)


def test_contraction_routes_to_original_outputs_and_sums_blocks():
    layout = SparseInfluence.build([(2,), ()], [{0, 2}, {2}],
                                   [set(), set()], output_size=4)
    factors = (jnp.array([[1., 2.], [3., 4.]]), jnp.array([5.]))
    cotangents = (jnp.array([2., 3.]), jnp.array(7.))
    result = contract_factors(factors, cotangents, layout)
    np.testing.assert_array_equal(result, [11., 0., 51., 0.])


def test_no_output_factors_preserve_empty_shapes():
    layout = SparseInfluence.build([(2,)], [set()], [set()], output_size=0)
    factors = instant_factors(lambda y: (jnp.ones(2),), jnp.zeros(0), layout)
    assert factors[0].shape == (2, 0)
    assert contract_factors(factors, (jnp.ones(2),), layout).shape == (0,)
    assert propagate_factors(lambda h: h, (jnp.ones(2),), factors, layout)[0].shape == (2, 0)
    assert advance_factors(lambda y, h: h, jnp.zeros(0), (jnp.ones(2),), factors, layout, .5)[0].shape == (2, 0)


def test_joint_jvp_equals_separate_propagated_and_injected_terms():
    layout = SparseInfluence.build([(2,), (1,)], [{0, 1}, {2}], [{0}, {1}], output_size=3)
    state, output = (jnp.array([.2, .3]), jnp.array([.5])), jnp.array([.1, .3, -.2])
    factors = (jnp.array([[1., 2.], [3., 4.]]), jnp.array([[5.]]))
    def transition(y, h):
        return jnp.tanh(h[0]+y[:2]), h[1]**2+y[2]
    expected_f = propagate_factors(lambda h: transition(output, h), state, factors, layout)
    expected_d = instant_factors(lambda y: transition(y, state), output, layout)
    actual = brainstate.transform.jit(lambda y, h, f: advance_factors(transition, y, h, f, layout, .7))(output, state, factors)
    for value, previous, injected in zip(actual, expected_f, expected_d):
        np.testing.assert_allclose(value, .7*previous+.3*injected, rtol=1e-6)


def test_invalid_derivative_layouts_fail_before_execution():
    layout = SparseInfluence.build([(2,)], [{0}], [set()], output_size=1)
    with pytest.raises(ValueError, match='output shape'):
        instant_factors(lambda y: (y,), jnp.ones(2), layout)
    with pytest.raises(ValueError, match='blocks'):
        propagate_factors(lambda h: h, (), (), layout)
    with pytest.raises(ValueError, match='shape'):
        propagate_factors(lambda h: h, (jnp.zeros(3),), (jnp.zeros((3, 1)),), layout)
    with pytest.raises(ValueError, match='blocks'):
        contract_factors((), (), layout)
    with pytest.raises(ValueError, match='shape'):
        contract_factors((jnp.zeros((3, 1)),), (jnp.zeros(3),), layout)
