# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Attention Residuals as a fused ETP primitive.

The operator implements the full Attention Residual equation over a source
axis. Sources are RMS-normalized only when used as keys; the returned values
are the original, unnormalized sources. The learned pseudo-query is the sole
trainable primitive input.

For pp-prop, the packed input trace retains every source position together
with its validity and query selection. The solve-time VJP is exact when the
trace contains one current step; longer histories retain pp-prop's usual
input/output factorization. D-RTRL stores the complete query-to-output
Jacobian, because every query coordinate can affect every output coordinate.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._typing import ArrayLike

from ._primitive import register_primitive
from ._registries import ETP_RULES_INSTANT_DRTRL, ETP_RULES_SOLVE_DRTRL

__all__ = ["attention_residual", "etp_attention_residual_p"]


def _masked_softmax(logits: Any, mask: Any) -> Any:
    """Return a finite masked softmax, including for an empty valid set."""
    floor = jnp.finfo(logits.dtype).min
    shifted = jnp.where(mask, logits, floor)
    shifted = shifted - jnp.max(shifted, axis=-1, keepdims=True)
    numerators = jnp.exp(shifted) * mask.astype(logits.dtype)
    denominator = jnp.sum(numerators, axis=-1, keepdims=True)
    safe_denominator = jnp.where(denominator > 0, denominator, 1)
    return numerators / safe_denominator


def _attention_from_packed(
    packed: Any,
    query: Any,
    *,
    hidden_size: int,
    source_count: int,
    query_count: int,
    epsilon: float,
) -> tuple[Any, Any]:
    """Return output and weights from the primitive's packed input."""
    packed = jnp.reshape(
        packed,
        (*packed.shape[:-1], source_count, hidden_size + query_count + 1),
    )
    sources = packed[..., :hidden_size]
    selector = packed[..., hidden_size : hidden_size + query_count]
    mask = packed[..., hidden_size + query_count] > 0.5
    selected_query = jnp.einsum("...sq,qh->...sh", selector, query)
    keys = sources / jnp.sqrt(
        jnp.mean(jnp.square(sources), axis=-1, keepdims=True) + epsilon
    )
    logits = jnp.sum(keys * selected_query, axis=-1)
    weights = _masked_softmax(logits, mask)
    output = jnp.sum(weights[..., :, None] * sources, axis=-2)
    return output, weights


def _attention_residual_impl(
    packed: Any,
    query: Any,
    *,
    hidden_size: int,
    source_count: int,
    query_count: int,
    epsilon: float,
) -> Any:
    return _attention_from_packed(
        packed,
        query,
        hidden_size=hidden_size,
        source_count=source_count,
        query_count=query_count,
        epsilon=epsilon,
    )[0]


def _attention_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    return {"query": 1}


def _attention_xy_to_dw(
    packed: Any,
    hidden_dim: Any,
    weights: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    """Evaluate the exact query VJP at the retained pp-prop input trace."""
    _, pullback = jax.vjp(
        lambda query: _attention_residual_impl(packed, query, **params),
        weights["query"],
    )
    return {"query": pullback(hidden_dim)[0]}


def _attention_init_pp(
    x_var: Any,
    y_var: Any,
    weight_vars: dict[str, Any],
    num_hidden_state: int,
) -> Any:
    del x_var, weight_vars
    return jnp.zeros(
        (*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype
    )


def _attention_init_drtrl(
    x_var: Any,
    y_var: Any,
    weight_vars: dict[str, Any],
    num_hidden_state: int,
) -> dict[str, Any]:
    del x_var
    query_aval = weight_vars["query"].aval
    dtype = jnp.result_type(query_aval.dtype, y_var.aval.dtype)
    batch = y_var.aval.shape[0]
    output_positions = y_var.aval.shape[1:]
    return {
        "query": jnp.zeros(
            (batch, *query_aval.shape, *output_positions, num_hidden_state),
            dtype=dtype,
        )
    }


def _attention_dt_to_t(
    hidden_dim: Any, trace: dict[str, Any], **params: Any
) -> dict[str, Any]:
    del params
    lifted = jnp.expand_dims(jnp.expand_dims(hidden_dim, axis=-2), axis=-2)
    return {"query": trace["query"] * lifted}


def _attention_instant_drtrl(
    packed: Any,
    df: Any,
    weights: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    jacobian = jax.jacrev(
        lambda query: _attention_residual_impl(packed, query, **params)
    )(weights["query"])
    query_to_output = jnp.moveaxis(jacobian, 0, -1)
    return {"query": query_to_output * df}


def _attention_solve_drtrl(
    dg_hidden: Any,
    trace: dict[str, Any],
    weights: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    del weights, params
    dg = u.get_mantissa(dg_hidden)
    return {
        "query": jnp.einsum("...o,...qho->...qh", dg, trace["query"])
    }


etp_attention_residual_p = register_primitive(
    "etp_attention_residual",
    _attention_residual_impl,
    batched=True,
    gradient_enabled=True,
    trainable_invars_fn=_attention_trainable_invars,
    x_invar_index=0,
)
etp_attention_residual_p.register_etp_rules(
    dt_to_t=_attention_dt_to_t,
    xy_to_dw=_attention_xy_to_dw,
    init_drtrl=_attention_init_drtrl,
    init_pp=_attention_init_pp,
    snap_anchor=lambda params: False,
)
ETP_RULES_INSTANT_DRTRL[etp_attention_residual_p] = _attention_instant_drtrl
ETP_RULES_SOLVE_DRTRL[etp_attention_residual_p] = _attention_solve_drtrl


def _floating_array(value: ArrayLike, name: str) -> Any:
    result = jnp.asarray(value)
    if not jnp.issubdtype(result.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype, got {result.dtype}. Ensure {name} has a floating dtype.")
    return result


def _epsilon_value(epsilon: float | None, dtype: Any) -> float:
    if epsilon is None:
        return float(jnp.finfo(dtype).eps)
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise TypeError("Epsilon must be a finite positive real scalar or None. Set Epsilon to a finite positive real scalar or None.")
    result = float(epsilon)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("Epsilon must be a finite positive real scalar. Set Epsilon to a finite positive real scalar.")
    return result


def _prepare_attention_inputs(
    sources: ArrayLike,
    query: ArrayLike,
    source_mask: ArrayLike | None,
    query_index: int | ArrayLike,
    epsilon: float | None,
) -> tuple[Any, Any, dict[str, Any]]:
    sources_array = _floating_array(sources, "sources")
    query_array = _floating_array(query, "query")
    if sources_array.ndim < 2:
        raise ValueError("Sources must have rank at least two. Ensure Sources has rank at least two.")
    if query_array.ndim not in (1, 2):
        raise ValueError("Query must have shape (hidden_size,) or (query_count, hidden_size). Ensure Query has shape (hidden_size,) or (query_count, hidden_size).")
    hidden_size = sources_array.shape[-1]
    if hidden_size == 0:
        raise ValueError("Sources hidden_size must be positive. Set Sources hidden_size to a positive value.")
    if query_array.shape[-1] != hidden_size:
        raise ValueError(
            "Query hidden dimension must match sources; got "
            f"{query_array.shape[-1]} and {hidden_size}. Set Query hidden dimension to match sources; got "
            f"{query_array.shape[-1]} and {hidden_size}."
        )
    if query_array.dtype != sources_array.dtype:
        query_array = query_array.astype(sources_array.dtype)
    if query_array.ndim == 1:
        query_array = query_array[None, :]
    query_count = query_array.shape[0]
    leading_shape = sources_array.shape[:-2]
    source_count = sources_array.shape[-2]
    if source_count == 0:
        raise ValueError("Sources source_count must be positive. Set Sources source_count to a positive value.")

    index = jnp.asarray(query_index)
    if not jnp.issubdtype(index.dtype, jnp.integer):
        raise TypeError("query_index must have an integer dtype. Ensure query_index has an integer dtype.")
    if index.ndim == 0:
        index = jnp.broadcast_to(index, leading_shape)
    elif index.shape != leading_shape:
        raise ValueError(
            f"query_index must be scalar or have shape {leading_shape}, got {index.shape}. Set query_index to scalar or have shape {leading_shape}."
        )
    if not isinstance(index, jax.core.Tracer):
        concrete_index = np.asarray(index)
        if np.any(concrete_index < 0) or np.any(concrete_index >= query_count):
            raise ValueError(
                f"query_index entries must be in [0, {query_count}), got {query_index!r}. Set query_index entries to values in [0, {query_count})."
            )

    if source_mask is None:
        mask = jnp.ones((*leading_shape, source_count), dtype=jnp.bool_)
    else:
        mask = jnp.asarray(source_mask)
        if mask.dtype != jnp.bool_:
            raise TypeError(f"source_mask must have boolean dtype, got {mask.dtype}. Ensure source_mask has boolean dtype.")
        if mask.shape == (source_count,):
            mask = jnp.broadcast_to(mask, (*leading_shape, source_count))
        elif mask.shape != (*leading_shape, source_count):
            raise ValueError(
                "source_mask must have shape "
                f"({source_count},) or {(*leading_shape, source_count)}, got {mask.shape}. Set source_mask to a value with shape "
                f"({source_count},) or {(*leading_shape, source_count)}."
            )

    selector = jax.nn.one_hot(index, query_count, dtype=sources_array.dtype)
    selector = jnp.broadcast_to(
        selector[..., None, :], (*leading_shape, source_count, query_count)
    )
    packed_sources = jnp.concatenate(
        (sources_array, selector, mask[..., None].astype(sources_array.dtype)), axis=-1
    )
    packed = jnp.reshape(packed_sources, (*leading_shape, -1))
    params = {
        "hidden_size": hidden_size,
        "source_count": source_count,
        "query_count": query_count,
        "epsilon": _epsilon_value(epsilon, sources_array.dtype),
    }
    return packed, query_array, params


def _attention_residual_weights(
    sources: ArrayLike,
    query: ArrayLike,
    *,
    source_mask: ArrayLike | None = None,
    query_index: int | ArrayLike = 0,
    epsilon: float | None = None,
) -> Any:
    """Pure diagnostic implementation returning source-axis weights."""
    packed, query_array, params = _prepare_attention_inputs(
        sources, query, source_mask, query_index, epsilon
    )
    return _attention_from_packed(packed, query_array, **params)[1]


def attention_residual(
    sources: ArrayLike,
    query: ArrayLike,
    *,
    source_mask: ArrayLike | None = None,
    query_index: int | ArrayLike = 0,
    epsilon: float | None = None,
) -> Any:
    r"""Mix residual sources with a learned pseudo-query.

    Parameters
    ----------
    sources : ArrayLike
        Floating source values shaped ``(..., source_count, hidden_size)``.
    query : ArrayLike
        One pseudo-query shaped ``(hidden_size,)`` or a query table shaped
        ``(query_count, hidden_size)``.
    source_mask : ArrayLike, optional
        Boolean validity mask shaped ``(source_count,)`` or
        ``(..., source_count)``. Default ``None`` keeps every source.
    query_index : int or ArrayLike, optional
        Scalar query index or an integer array matching the leading source
        dimensions. Default ``0``.
    epsilon : float, optional
        Positive RMS-normalization constant. ``None`` uses the source dtype's
        machine epsilon.

    Returns
    -------
    ArrayLike
        Convex weighted source sum shaped ``(..., hidden_size)``. An all-false
        mask returns exact zeros.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> sources = jnp.asarray([[1., 2.], [3., 4.]])
        >>> query = jnp.zeros(2)
        >>> braintrace.attention_residual(sources, query).tolist()
        [2.0, 3.0]
    """
    packed, query_array, params = _prepare_attention_inputs(
        sources, query, source_mask, query_index, epsilon
    )
    return etp_attention_residual_p.bind(packed, query_array, **params)
