from __future__ import annotations

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace


def test_attention_residual_module_owns_zero_initialized_query_table() -> None:
    module = braintrace.nn.AttentionResidual(4, query_count=3)

    assert isinstance(module.query, brainstate.ParamState)
    assert module.query.value.shape == (3, 4)
    np.testing.assert_array_equal(module.query.value, 0.0)


def test_attention_weights_is_pure() -> None:
    module = braintrace.nn.AttentionResidual(2, query_count=2)
    module.query.value = jnp.asarray([[0.1, 0.2], [-0.3, 0.4]])
    before = np.asarray(module.query.value).copy()
    sources = jnp.asarray([[[1.0, 2.0], [3.0, -1.0]]])

    first = module.attention_weights(sources, query_index=1)
    second = module.attention_weights(sources, query_index=1)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(module.query.value, before)


@pytest.mark.parametrize(
    ("hidden_size", "query_count", "error"),
    [
        (True, 1, TypeError),
        (2, False, TypeError),
        (0, 1, ValueError),
        (2, 0, ValueError),
    ],
)
def test_attention_residual_module_validates_dimensions(
    hidden_size: int, query_count: int, error: type[Exception]
) -> None:
    with pytest.raises(error, match="positive integer"):
        braintrace.nn.AttentionResidual(hidden_size, query_count=query_count)


def test_public_exports() -> None:
    assert "attention_residual" in braintrace.__all__
    assert "AttentionResidual" in braintrace.nn.__all__
