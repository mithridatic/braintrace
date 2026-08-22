from __future__ import annotations

import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle import (
    assert_gradients_differ,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
)


def _reference(values, gate_input, gate_weight, gate_bias, output_weight, *, normalize, epsilon):
    if normalize:
        values = values / jnp.sqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + epsilon
        )
    gate = jax.nn.sigmoid(gate_input @ gate_weight + gate_bias)
    return (gate * values) @ output_weight


@pytest.mark.parametrize("normalize", [False, True])
def test_forward_matches_independent_reference(normalize: bool) -> None:
    values = jnp.asarray([[1.0, -2.0, 0.5], [-0.25, 3.0, 1.5]])
    gate_input = jnp.asarray([[0.2, -0.4], [0.7, 0.1]])
    gate_weight = jnp.asarray([[0.3, -0.2, 0.5], [-0.6, 0.1, 0.4]])
    gate_bias = jnp.asarray([0.05, -0.1, 0.2])
    output_weight = jnp.asarray(
        [[0.5, -0.2], [0.3, 0.7], [-0.4, 0.1]]
    )
    epsilon = 1e-5

    actual = braintrace.gated_projection(
        values,
        gate_input,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        output_weight=output_weight,
        normalize=normalize,
        epsilon=epsilon,
    )
    expected = _reference(
        values,
        gate_input,
        gate_weight,
        gate_bias,
        output_weight,
        normalize=normalize,
        epsilon=epsilon,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_jit_jvp_and_all_parameter_vjps_match_reference() -> None:
    values = jnp.asarray([[1.0, -2.0, 0.5]])
    gate_input = jnp.asarray([[0.2, -0.4]])
    parameters = (
        jnp.asarray([[0.3, -0.2, 0.5], [-0.6, 0.1, 0.4]]),
        jnp.asarray([0.05, -0.1, 0.2]),
        jnp.asarray([[0.5, -0.2], [0.3, 0.7], [-0.4, 0.1]]),
    )

    def actual(v, g, wg, bg, wo):
        return braintrace.gated_projection(
            v,
            g,
            gate_weight=wg,
            gate_bias=bg,
            output_weight=wo,
            normalize=True,
            epsilon=1e-5,
        )

    def expected(v, g, wg, bg, wo):
        return _reference(v, g, wg, bg, wo, normalize=True, epsilon=1e-5)

    operands = (values, gate_input, *parameters)
    tangents = tuple(jnp.full_like(value, 0.1) for value in operands)
    actual_primal, actual_tangent = jax.jvp(actual, operands, tangents)
    expected_primal, expected_tangent = jax.jvp(expected, operands, tangents)
    np.testing.assert_allclose(actual_primal, expected_primal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_tangent, expected_tangent, rtol=1e-5, atol=1e-6)

    actual_grads = jax.grad(lambda *args: jnp.square(actual(*args)).sum(), argnums=tuple(range(5)))(*operands)
    expected_grads = jax.grad(lambda *args: jnp.square(expected(*args)).sum(), argnums=tuple(range(5)))(*operands)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(jax.jit(actual)(*operands), expected_primal, rtol=1e-6)


def test_zero_gate_initialization_matches_linear_and_keeps_gate_gradient() -> None:
    values = jnp.asarray([[1.0, -2.0, 0.5]])
    gate_input = jnp.asarray([[0.2, -0.4]])
    baseline = jnp.asarray([[0.5, -0.2], [0.3, 0.7], [-0.4, 0.1]])
    gate_weight = jnp.zeros((2, 3))
    gate_bias = jnp.zeros(3)

    output = braintrace.gated_projection(
        values,
        gate_input,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        output_weight=2.0 * baseline,
        normalize=False,
        epsilon=1e-6,
    )
    np.testing.assert_allclose(output, values @ baseline, atol=1e-7)
    gate_grads = jax.grad(
        lambda wg, bg: jnp.square(
            braintrace.gated_projection(
                values,
                gate_input,
                gate_weight=wg,
                gate_bias=bg,
                output_weight=2.0 * baseline,
                normalize=False,
                epsilon=1e-6,
            )
        ).sum(),
        argnums=(0, 1),
    )(gate_weight, gate_bias)
    assert all(np.linalg.norm(np.asarray(gradient)) > 0.0 for gradient in gate_grads)


def test_zero_and_extreme_inputs_remain_finite() -> None:
    values = jnp.asarray([[0.0, 0.0], [1e20, -1e20]], dtype=jnp.float32)
    gate_input = jnp.asarray([[0.0], [1e20]], dtype=jnp.float32)
    output = braintrace.gated_projection(
        values,
        gate_input,
        gate_weight=jnp.asarray([[1e20, -1e20]], dtype=jnp.float32),
        gate_bias=jnp.zeros(2),
        output_weight=jnp.eye(2),
        normalize=True,
        epsilon=1e-5,
    )
    assert np.all(np.isfinite(np.asarray(output)))
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize(
    ("values", "gate_input", "kwargs", "error", "match"),
    [
        (jnp.ones(3), jnp.ones((1, 2)), {}, ValueError, "batched"),
        (jnp.ones((1, 3)), jnp.ones(2), {}, ValueError, "batched"),
        (jnp.ones((2, 3)), jnp.ones((1, 2)), {}, ValueError, "batch"),
        (jnp.ones((1, 3), dtype=jnp.int32), jnp.ones((1, 2)), {}, TypeError, "floating"),
        (jnp.ones((1, 3)), jnp.ones((1, 2)), {"epsilon": 0.0}, ValueError, "epsilon"),
    ],
)
def test_validation(values, gate_input, kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        braintrace.gated_projection(
            values,
            gate_input,
            gate_weight=jnp.ones((2, 3)),
            gate_bias=jnp.zeros(3),
            output_weight=jnp.ones((3, 4)),
            normalize=True,
            **kwargs,
        )


def test_dimensionful_operands_are_rejected() -> None:
    with pytest.raises(ValueError, match="dimensionless"):
        braintrace.gated_projection(
            jnp.ones((1, 3)) * u.mV,
            jnp.ones((1, 2)),
            gate_weight=jnp.ones((2, 3)),
            gate_bias=jnp.zeros(3),
            output_weight=jnp.ones((3, 4)),
            normalize=False,
            epsilon=1e-5,
        )


class _GatedCell(brainstate.nn.Module):
    def __init__(self, recurrent: bool):
        super().__init__()
        self.recurrent = recurrent
        self.projection = braintrace.nn.GatedProjection(
            3,
            2,
            3,
            gate_weight_init=jnp.asarray([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]),
            gate_bias_init=jnp.asarray([0.1, -0.2, 0.05]),
            output_weight_init=jnp.asarray(
                [[0.3, -0.2, 0.1], [0.4, 0.5, -0.3], [-0.1, 0.2, 0.6]]
            ),
        )
        self.hidden = brainstate.HiddenState(jnp.zeros((1, 3), dtype=jnp.float32))

    def update(self, x):
        drive = self.projection(x[:, :3], x[:, 3:])
        prior = 0.3 * self.hidden.value if self.recurrent else 0.0
        self.hidden.value = prior + jnp.tanh(drive)
        return self.hidden.value


def _cell_factory(recurrent: bool):
    return lambda: _GatedCell(recurrent)


def test_d_rtrl_matches_bptt_and_pp_prop_is_honestly_finite_window() -> None:
    inputs = jnp.asarray(
        [
            [[1.0, -2.0, 0.5, 0.2, -0.4]],
            [[-0.5, 1.5, 2.0, -0.1, 0.7]],
            [[0.25, 0.5, -1.0, 0.8, 0.3]],
        ],
        dtype=jnp.float32,
    )
    recurrent_factory = _cell_factory(True)
    bptt = bptt_param_gradients(recurrent_factory, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_rtrl = chunked_online_param_gradients(
            recurrent_factory,
            inputs,
            algo_factory=lambda model: braintrace.D_RTRL(model, vjp_method="multi-step"),
            chunk_size=1,
        )
        pp = chunked_online_param_gradients(
            recurrent_factory,
            inputs,
            algo_factory=lambda model: braintrace.pp_prop(
                model, decay_or_rank=0.8, vjp_method="multi-step"
            ),
            chunk_size=1,
        )
        independent_factory = _cell_factory(False)
        independent_bptt = bptt_param_gradients(independent_factory, inputs)
        one_step_pp = chunked_online_param_gradients(
            independent_factory,
            inputs,
            algo_factory=lambda model: braintrace.pp_prop(
                model, decay_or_rank=0.0, vjp_method="multi-step"
            ),
            chunk_size=1,
        )
    assert_param_gradients_close(d_rtrl, bptt, atol=3e-5)
    assert_param_gradients_close(one_step_pp, independent_bptt, atol=3e-5)
    assert_gradients_differ(pp, bptt, min_rel=1e-8)
