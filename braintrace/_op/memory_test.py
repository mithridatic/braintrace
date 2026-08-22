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
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
)


def _delta_reference(memory, x_key, x_value, parameters):
    key = 1.5 * (x_key @ parameters[0] + parameters[1])
    key = key / jnp.maximum(jnp.linalg.norm(key, axis=-1, keepdims=True), 1e-6)
    value = x_value @ parameters[2]
    beta = jax.nn.sigmoid(x_value @ parameters[3] + parameters[4])
    alpha = jnp.exp(-5.0 * jax.nn.sigmoid(x_value @ parameters[5] + parameters[6]))
    decayed = memory * alpha[:, None, :]
    error = value - jnp.einsum("bkv,bk->bv", decayed, key)
    return decayed + beta[:, :, None] * jnp.einsum("bk,bv->bkv", key, error)


def _delta_parameters():
    return (
        jnp.asarray([[0.4, -0.2], [0.1, 0.3]]),
        jnp.asarray([0.05, -0.1]),
        jnp.asarray([[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]]),
        jnp.asarray([[0.1], [-0.2], [0.3]]),
        jnp.asarray([0.2]),
        jnp.asarray([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.1]]),
        jnp.asarray([0.1, -0.2]),
    )


def _delta_call(memory, x_key, x_value, *parameters):
    return braintrace.delta_memory_update(
        memory,
        x_key,
        x_value,
        key_weight=parameters[0],
        key_bias=parameters[1],
        value_weight=parameters[2],
        beta_weight=parameters[3],
        beta_bias=parameters[4],
        retention_weight=parameters[5],
        retention_bias=parameters[6],
        key_scale=1.5,
        min_log_decay=-5.0,
    )


def test_delta_forward_jit_jvp_and_vjp_match_independent_reference() -> None:
    memory = jnp.asarray([[[0.2, -0.1], [0.4, 0.3]]])
    x_key = jnp.asarray([[0.5, -1.0]])
    x_value = jnp.asarray([[0.2, 0.7, -0.4]])
    parameters = _delta_parameters()
    operands = (memory, x_key, x_value, *parameters)

    expected = _delta_reference(memory, x_key, x_value, parameters)
    actual = _delta_call(*operands)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jax.jit(_delta_call)(*operands), expected, rtol=1e-6)
    tangents = tuple(jnp.full_like(value, 0.03) for value in operands)
    _, actual_jvp = jax.jvp(_delta_call, operands, tangents)
    _, expected_jvp = jax.jvp(
        lambda m, k, v, *p: _delta_reference(m, k, v, p), operands, tangents
    )
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-5, atol=2e-6)
    actual_grads = jax.grad(
        lambda *args: jnp.square(_delta_call(*args)).sum(),
        argnums=tuple(range(len(operands))),
    )(*operands)
    expected_grads = jax.grad(
        lambda m, k, v, *p: jnp.square(_delta_reference(m, k, v, p)).sum(),
        argnums=tuple(range(len(operands))),
    )(*operands)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=3e-5, atol=3e-6)


def test_delta_repeated_key_corrects_error_and_remains_finite() -> None:
    parameters = list(_delta_parameters())
    parameters[3] = jnp.zeros((3, 1))
    parameters[4] = jnp.asarray([20.0])
    parameters[5] = jnp.zeros((3, 2))
    parameters[6] = jnp.asarray([-20.0, -20.0])
    memory = jnp.zeros((1, 2, 2))
    key = jnp.asarray([[1.0, 0.0]])
    first_value = jnp.asarray([[1.0, 0.0, 0.0]])
    second_value = jnp.asarray([[0.0, 1.0, 0.0]])

    first = _delta_call(memory, key, first_value, *parameters)
    second = _delta_call(first, key, second_value, *parameters)
    target = second_value @ parameters[2]
    normalized_key = 1.5 * (key @ parameters[0] + parameters[1])
    normalized_key = normalized_key / jnp.linalg.norm(
        normalized_key, axis=-1, keepdims=True
    )
    retrieved = jnp.einsum("bkv,bk->bv", second, normalized_key)

    assert np.all(np.isfinite(np.asarray(second)))
    assert np.linalg.norm(np.asarray(retrieved - target)) < np.linalg.norm(
        np.asarray(jnp.einsum("bkv,bk->bv", first, normalized_key) - target)
    )


def test_delta_orthogonal_keys_preserve_associations_and_collisions_stay_finite() -> None:
    parameters = (
        jnp.eye(2),
        jnp.zeros(2),
        jnp.eye(2),
        jnp.zeros((2, 1)),
        jnp.asarray([20.0]),
        jnp.zeros((2, 2)),
        jnp.asarray([-20.0, -20.0]),
    )
    empty = jnp.zeros((1, 2, 2))
    key_a = jnp.asarray([[1.0, 0.0]])
    key_b = jnp.asarray([[0.0, 1.0]])
    value_a = jnp.asarray([[0.25, -0.75]])
    value_b = jnp.asarray([[1.5, 0.5]])

    first = _delta_call(empty, key_a, value_a, *parameters)
    orthogonal = _delta_call(first, key_b, value_b, *parameters)
    collision = _delta_call(first, key_a, value_b, *parameters)

    np.testing.assert_allclose(
        jnp.einsum("bkv,bk->bv", orthogonal, key_a),
        value_a,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        jnp.einsum("bkv,bk->bv", orthogonal, key_b),
        value_b,
        rtol=2e-5,
        atol=2e-5,
    )
    assert np.all(np.isfinite(np.asarray(collision)))


def test_memory_operators_zero_and_extreme_inputs_remain_finite() -> None:
    parameters = _delta_parameters()
    zero = _delta_call(
        jnp.zeros((2, 2, 2)),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 3)),
        *parameters,
    )
    huge = _delta_call(
        jnp.full((2, 2, 2), 1e6),
        jnp.full((2, 2), 1e6),
        jnp.full((2, 3), -1e6),
        *parameters,
    )
    situ = _situ_call(jnp.full((2, 2), 1e6), *_situ_parameters())

    assert zero.shape == huge.shape == (2, 2, 2)
    assert np.all(np.isfinite(np.asarray(zero)))
    assert np.all(np.isfinite(np.asarray(huge)))
    assert np.all(np.isfinite(np.asarray(situ)))


def _situ_reference(x, parameters):
    gate_pre = x @ parameters[0] + parameters[1]
    up_pre = x @ parameters[2] + parameters[3]
    gate = 4.0 * jnp.tanh(gate_pre / 4.0) * jax.nn.sigmoid(gate_pre)
    up = 25.0 * jnp.tanh(up_pre / 25.0)
    return (gate * up) @ parameters[4]


def _situ_parameters():
    return (
        jnp.asarray([[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]]),
        jnp.asarray([0.1, -0.2, 0.05]),
        jnp.asarray([[-0.1, 0.6, 0.2], [0.3, -0.4, 0.7]]),
        jnp.asarray([0.0, 0.1, -0.1]),
        jnp.asarray([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.6]]),
    )


def _situ_call(x, *parameters):
    return braintrace.situ_glu(
        x,
        gate_weight=parameters[0],
        gate_bias=parameters[1],
        up_weight=parameters[2],
        up_bias=parameters[3],
        output_weight=parameters[4],
    )


def test_situ_forward_jit_and_all_gradients_match_reference() -> None:
    x = jnp.asarray([[0.5, -1.2], [2.0, 0.3]])
    parameters = _situ_parameters()
    operands = (x, *parameters)
    expected = _situ_reference(x, parameters)
    np.testing.assert_allclose(_situ_call(*operands), expected, rtol=1e-6)
    np.testing.assert_allclose(jax.jit(_situ_call)(*operands), expected, rtol=1e-6)
    actual = jax.grad(
        lambda *args: jnp.square(_situ_call(*args)).sum(),
        argnums=tuple(range(len(operands))),
    )(*operands)
    reference = jax.grad(
        lambda value, *p: jnp.square(_situ_reference(value, p)).sum(),
        argnums=tuple(range(len(operands))),
    )(*operands)
    for left, right in zip(actual, reference, strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-6)


def test_memory_operators_reject_units_shapes_and_invalid_caps() -> None:
    parameters = _delta_parameters()
    with pytest.raises(ValueError, match="dimensionless"):
        _delta_call(jnp.zeros((1, 2, 2)) * u.mV, jnp.ones((1, 2)), jnp.ones((1, 3)), *parameters)
    with pytest.raises(ValueError, match="rank three"):
        _delta_call(jnp.zeros((2, 2)), jnp.ones((1, 2)), jnp.ones((1, 3)), *parameters)
    with pytest.raises(ValueError, match="key_scale"):
        braintrace.delta_memory_update(
            jnp.zeros((1, 2, 2)),
            jnp.ones((1, 2)),
            jnp.ones((1, 3)),
            key_weight=parameters[0], key_bias=parameters[1],
            value_weight=parameters[2], beta_weight=parameters[3],
            beta_bias=parameters[4], retention_weight=parameters[5],
            retention_bias=parameters[6], key_scale=0.0,
        )
    with pytest.raises(ValueError, match="gate_beta"):
        braintrace.situ_glu(
            jnp.ones((1, 2)),
            gate_weight=jnp.ones((2, 3)), gate_bias=jnp.zeros(3),
            up_weight=jnp.ones((2, 3)), up_bias=jnp.zeros(3),
            output_weight=jnp.ones((3, 2)), gate_beta=0.0,
        )


class _SiTUCell(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = braintrace.nn.SiTUGLU(
            2,
            3,
            2,
            gate_weight_init=_situ_parameters()[0],
            gate_bias_init=_situ_parameters()[1],
            up_weight_init=_situ_parameters()[2],
            up_bias_init=_situ_parameters()[3],
            output_weight_init=_situ_parameters()[4],
        )
        self.hidden = brainstate.HiddenState(jnp.zeros((1, 2), dtype=jnp.float32))

    def update(self, x):
        self.hidden.value = 0.2 * self.hidden.value + self.projection(x)
        return self.hidden.value


def test_situ_d_rtrl_matches_bptt_and_one_step_pp_prop() -> None:
    inputs = jnp.asarray([[[0.5, -1.2]], [[1.0, 0.3]]], dtype=jnp.float32)
    bptt = bptt_param_gradients(_SiTUCell, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_rtrl = chunked_online_param_gradients(
            _SiTUCell,
            inputs,
            algo_factory=lambda model: braintrace.D_RTRL(model, vjp_method="multi-step"),
            chunk_size=1,
        )
    assert_param_gradients_close(d_rtrl, bptt, atol=3e-5)
    for gradient in jax.tree.leaves(bptt):
        assert float(jnp.linalg.norm(u.get_mantissa(gradient))) > 0.0


class _DeltaCell(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        parameters = _delta_parameters()
        self.key_weight = brainstate.ParamState(parameters[0])
        self.key_bias = brainstate.ParamState(parameters[1])
        self.value_weight = brainstate.ParamState(parameters[2])
        self.beta_weight = brainstate.ParamState(parameters[3])
        self.beta_bias = brainstate.ParamState(parameters[4])
        self.retention_weight = brainstate.ParamState(parameters[5])
        self.retention_bias = brainstate.ParamState(parameters[6])
        self.memory = brainstate.HiddenState(jnp.zeros((1, 2, 2), dtype=jnp.float32))

    def update(self, x):
        self.memory.value = _delta_call(
            self.memory.value,
            x[..., :2],
            x,
            self.key_weight.value,
            self.key_bias.value,
            self.value_weight.value,
            self.beta_weight.value,
            self.beta_bias.value,
            self.retention_weight.value,
            self.retention_bias.value,
        )
        return self.memory.value


def test_delta_d_rtrl_matches_bptt_and_pp_prop_is_finite() -> None:
    inputs = jnp.asarray(
        [[[0.5, -1.2, 0.1]], [[1.0, 0.3, -0.4]]], dtype=jnp.float32
    )
    bptt = bptt_param_gradients(_DeltaCell, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_rtrl = chunked_online_param_gradients(
            _DeltaCell,
            inputs,
            algo_factory=lambda model: braintrace.D_RTRL(
                model, vjp_method="multi-step"
            ),
            chunk_size=1,
        )
        pp_prop = chunked_online_param_gradients(
            _DeltaCell,
            inputs,
            algo_factory=lambda model: braintrace.pp_prop(
                model, decay_or_rank=0.9, vjp_method="multi-step"
            ),
            chunk_size=1,
        )
    assert_param_gradients_close(d_rtrl, bptt, atol=5e-5)
    for gradient in jax.tree.leaves(bptt):
        assert float(jnp.linalg.norm(u.get_mantissa(gradient))) > 0.0
    for gradient in jax.tree.leaves(pp_prop):
        assert np.all(np.isfinite(np.asarray(u.get_mantissa(gradient))))
    bptt_flat = np.concatenate(
        [
            np.asarray(u.get_mantissa(leaf)).reshape(-1)
            for leaf in jax.tree.leaves(bptt)
        ]
    )
    pp_flat = np.concatenate(
        [
            np.asarray(u.get_mantissa(leaf)).reshape(-1)
            for leaf in jax.tree.leaves(pp_prop)
        ]
    )
    cosine = float(
        np.dot(bptt_flat, pp_flat)
        / (np.linalg.norm(bptt_flat) * np.linalg.norm(pp_flat))
    )
    relative_deviation = float(
        np.linalg.norm(pp_flat - bptt_flat) / np.linalg.norm(bptt_flat)
    )
    assert cosine >= 0.8
    assert relative_deviation < 1.0
