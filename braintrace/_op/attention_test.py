from __future__ import annotations

import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._op import (
    ETP_RULES_XY_TO_DW,
    GRADIENT_ENABLED_PRIMITIVES,
    etp_attention_residual_p,
)
from braintrace._testing.oracle import (
    assert_gradients_differ,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
)


def _reference(sources, query, mask=None, query_index=0, epsilon=None):
    sources = jnp.asarray(sources)
    query = jnp.asarray(query)
    if epsilon is None:
        epsilon = jnp.finfo(sources.dtype).eps
    if query.ndim == 1:
        selected = jnp.broadcast_to(query, sources.shape[:-2] + query.shape)
    else:
        selected = query[jnp.asarray(query_index)]
        selected = jnp.broadcast_to(selected, sources.shape[:-2] + (query.shape[-1],))
    keys = sources / jnp.sqrt(jnp.mean(jnp.square(sources), axis=-1, keepdims=True) + epsilon)
    logits = jnp.einsum("...sh,...h->...s", keys, selected)
    if mask is None:
        mask = jnp.ones(logits.shape, dtype=jnp.bool_)
    else:
        mask = jnp.broadcast_to(jnp.asarray(mask), logits.shape)
    shifted = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    shifted = shifted - jnp.max(shifted, axis=-1, keepdims=True)
    numerators = jnp.exp(shifted) * mask
    denominator = jnp.sum(numerators, axis=-1, keepdims=True)
    weights = jnp.where(denominator > 0, numerators / denominator, 0.0)
    return jnp.einsum("...s,...sh->...h", weights, sources), weights


def test_forward_and_weights_match_independent_reference() -> None:
    sources = jnp.asarray(
        [
            [[1.0, -2.0, 0.5], [3.0, 1.0, -1.0], [0.25, 4.0, 2.0]],
            [[-1.0, 2.0, 3.0], [2.0, -3.0, 0.5], [5.0, 1.0, -2.0]],
        ],
        dtype=jnp.float32,
    )
    query = jnp.asarray([[0.2, -0.4, 0.7], [-0.3, 0.8, 0.1]])
    mask = jnp.asarray([[True, False, True], [True, True, False]])
    indices = jnp.asarray([0, 1], dtype=jnp.int32)

    expected, expected_weights = _reference(sources, query, mask, indices)
    actual = braintrace.attention_residual(
        sources, query, source_mask=mask, query_index=indices
    )
    module = braintrace.nn.AttentionResidual(3, query_count=2)
    module.query.value = query

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        module.attention_weights(sources, source_mask=mask, query_index=indices),
        expected_weights,
        rtol=1e-6,
        atol=1e-6,
    )


def test_query_gradient_matches_jax_reference_and_inactive_query_is_zero() -> None:
    sources = jnp.asarray(
        [[[1.0, 0.0, 2.0], [-2.0, 1.0, 0.5], [0.5, 3.0, -1.0]]]
    )
    query = jnp.asarray([[0.1, -0.3, 0.4], [0.7, 0.2, -0.5]])
    cotangent = jnp.asarray([[0.5, -1.0, 2.0]])

    def loss(q, op):
        output = op(sources, q, query_index=1)
        return jnp.sum(output * cotangent)

    expected = jax.grad(lambda q: loss(q, lambda s, w, query_index: _reference(s, w, query_index=query_index)[0]))(query)
    actual = jax.grad(lambda q: loss(q, braintrace.attention_residual))(query)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual[0], 0.0)
    assert np.linalg.norm(np.asarray(actual[1])) > 0.0


def test_masks_zero_sources_and_convex_weights_remain_finite() -> None:
    sources = jnp.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 3.0], [-2.0, 5.0], [4.0, -1.0]],
        ],
        dtype=jnp.float32,
    )
    mask = jnp.asarray([[False, False, False], [True, False, True]])
    module = braintrace.nn.AttentionResidual(2)

    output = module(sources, source_mask=mask)
    weights = module.attention_weights(sources, source_mask=mask)

    assert np.all(np.isfinite(np.asarray(output)))
    assert np.all(np.isfinite(np.asarray(weights)))
    np.testing.assert_array_equal(output[0], 0.0)
    np.testing.assert_array_equal(weights[0], 0.0)
    np.testing.assert_allclose(weights[1].sum(), 1.0, atol=1e-7)
    assert np.all(np.asarray(weights) >= 0.0)
    assert np.all(np.asarray(weights) <= 1.0)
    expected_min = np.minimum(np.asarray(sources[1, 0]), np.asarray(sources[1, 2]))
    expected_max = np.maximum(np.asarray(sources[1, 0]), np.asarray(sources[1, 2]))
    assert np.all(np.asarray(output[1]) >= expected_min)
    assert np.all(np.asarray(output[1]) <= expected_max)

    all_masked_gradient = jax.grad(
        lambda query: jnp.sum(
            braintrace.attention_residual(
                sources[:1], query, source_mask=mask[:1]
            )
        )
    )(module.query.value)
    np.testing.assert_array_equal(all_masked_gradient, 0.0)


def test_zero_query_is_uniform_mean_and_has_nonzero_gradient() -> None:
    sources = jnp.asarray([[[1.0, 2.0], [5.0, -1.0], [-2.0, 4.0]]])
    module = braintrace.nn.AttentionResidual(2)

    np.testing.assert_allclose(module(sources), sources.mean(axis=-2), atol=1e-7)
    np.testing.assert_allclose(module.attention_weights(sources), 1.0 / 3.0, atol=1e-7)
    gradient = jax.grad(lambda q: jnp.square(
        braintrace.attention_residual(sources, q)
    ).sum())(module.query.value)
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_leading_dimensions_jit_and_vmap() -> None:
    sources = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5) / 17
    query = jnp.asarray([[0.1, -0.2, 0.3, -0.4, 0.5], [0.5, 0.4, -0.3, 0.2, -0.1]])
    indices = jnp.asarray([[0, 1, 0], [1, 0, 1]], dtype=jnp.int32)
    mask = jnp.asarray([True, True, False, True])

    expected, _ = _reference(sources, query, mask, indices)
    compiled = jax.jit(braintrace.attention_residual)(
        sources, query, source_mask=mask, query_index=indices
    )
    mapped = jax.vmap(
        lambda s, i: braintrace.attention_residual(s, query, query_index=i)
    )(sources, indices)

    np.testing.assert_allclose(compiled, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        mapped, _reference(sources, query, query_index=indices)[0], rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    ("sources", "query", "kwargs", "error", "match"),
    [
        (jnp.ones((2, 3), dtype=jnp.int32), jnp.ones(3), {}, TypeError, "floating"),
        (jnp.ones(3), jnp.ones(3), {}, ValueError, "rank"),
        (jnp.ones((2, 3)), jnp.ones(4), {}, ValueError, "hidden"),
        (jnp.ones((2, 3)), jnp.ones((2, 3, 1)), {}, ValueError, "query"),
        (jnp.ones((2, 3)), jnp.ones(3), {"source_mask": jnp.ones(2)}, TypeError, "boolean"),
        (jnp.ones((2, 3)), jnp.ones(3), {"source_mask": jnp.ones((3,), dtype=bool)}, ValueError, "mask"),
        (jnp.ones((2, 3)), jnp.ones((2, 3)), {"query_index": 2}, ValueError, "query_index"),
        (jnp.ones((2, 3)), jnp.ones(3), {"epsilon": 0.0}, ValueError, "epsilon"),
        (jnp.ones((0, 3)), jnp.ones(3), {}, ValueError, "source_count"),
        (jnp.ones((2, 0)), jnp.ones(0), {}, ValueError, "hidden_size"),
    ],
)
def test_validation(sources, query, kwargs, error, match) -> None:
    with pytest.raises(error, match=f"(?i){match}"):
        braintrace.attention_residual(sources, query, **kwargs)


def test_primitive_is_gradient_enabled_and_xy_rule_matches_vjp() -> None:
    assert etp_attention_residual_p in GRADIENT_ENABLED_PRIMITIVES
    sources = jnp.asarray([[[1.0, 2.0], [3.0, -1.0]]])
    query = jnp.asarray([[0.2, -0.4]])
    mask = jnp.ones((1, 2, 1), dtype=jnp.float32)
    selector = jnp.ones((1, 2, 1), dtype=jnp.float32)
    packed = jnp.concatenate((sources, selector, mask), axis=-1).reshape(1, -1)
    hidden = jnp.asarray([[0.7, -0.2]])
    params = {
        "hidden_size": 2,
        "source_count": 2,
        "query_count": 1,
        "epsilon": float(jnp.finfo(jnp.float32).eps),
    }
    rule = ETP_RULES_XY_TO_DW[etp_attention_residual_p]

    actual = rule(packed, hidden, {"query": query}, **params)["query"]
    _, pullback = jax.vjp(
        lambda q: etp_attention_residual_p.bind(packed, q, **params), query
    )
    expected = pullback(hidden)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class _IndependentAttentionCell(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = braintrace.nn.AttentionResidual(
            3, query_init=jnp.asarray([[0.2, -0.35, 0.15]], dtype=jnp.float32)
        )
        self.hidden = brainstate.HiddenState(jnp.zeros((1, 3), dtype=jnp.float32))

    def update(self, sources):
        self.hidden.value = jnp.tanh(self.mixer(sources))
        return self.hidden.value


def _cell_factory():
    return _IndependentAttentionCell()


class _RecurrentAttentionCell(_IndependentAttentionCell):
    def update(self, sources):
        self.hidden.value = 0.4 * self.hidden.value + jnp.tanh(self.mixer(sources))
        return self.hidden.value


def _recurrent_cell_factory():
    return _RecurrentAttentionCell()


@pytest.mark.parametrize("order", [1, 2])
def test_snap_rejects_unanchored_query_to_output_trace(order: int) -> None:
    model = _IndependentAttentionCell()
    sources = jnp.ones((1, 2, 3), dtype=jnp.float32)
    algorithm = braintrace.SnAp(model, n=order)

    with pytest.raises(NotImplementedError, match="etp_attention_residual"):
        algorithm.compile_graph(sources)


def test_d_rtrl_query_trace_matches_small_bptt_oracle() -> None:
    inputs = jnp.asarray(
        [
            [[[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]]],
            [[[-1.0, 0.5, 2.0], [4.0, 1.0, -3.0]]],
            [[[2.0, -1.0, 0.25], [-0.5, 3.0, 1.5]]],
        ],
        dtype=jnp.float32,
    )
    bptt = bptt_param_gradients(_recurrent_cell_factory, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        online = chunked_online_param_gradients(
            _recurrent_cell_factory,
            inputs,
            algo_factory=lambda model: braintrace.D_RTRL(model, vjp_method="multi-step"),
            chunk_size=1,
        )
    assert_param_gradients_close(online, bptt, atol=2e-5)


def test_pp_prop_exact_in_independent_finite_window_and_diverges_when_smoothed() -> None:
    inputs = jnp.asarray(
        [
            [[[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]]],
            [[[-1.0, 0.5, 2.0], [4.0, 1.0, -3.0]]],
            [[[2.0, -1.0, 0.25], [-0.5, 3.0, 1.5]]],
        ],
        dtype=jnp.float32,
    )
    bptt = bptt_param_gradients(_cell_factory, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = chunked_online_param_gradients(
            _cell_factory,
            inputs,
            algo_factory=lambda model: braintrace.pp_prop(
                model, decay_or_rank=0.0, vjp_method="multi-step"
            ),
            chunk_size=1,
        )
        recurrent_bptt = bptt_param_gradients(_recurrent_cell_factory, inputs)
        smoothed = chunked_online_param_gradients(
            _recurrent_cell_factory,
            inputs,
            algo_factory=lambda model: braintrace.pp_prop(
                model, decay_or_rank=0.8, vjp_method="multi-step"
            ),
            chunk_size=1,
        )
    assert_param_gradients_close(exact, bptt, atol=2e-5)
    assert_gradients_differ(smoothed, recurrent_bptt, min_rel=1e-7)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(smoothed))
