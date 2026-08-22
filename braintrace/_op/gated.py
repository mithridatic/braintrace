# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r"""Fused gated projection with optional RMS normalization.

The primitive keeps the gate and output matrices in one ETP equation. This is
important for online learning: the gate parameters must not reach the recurrent
state through a second trainable primitive whose parameter ownership would make
the compiler reject or partially classify the path.

The pp-prop rule evaluates the complete parameter VJP at its retained input
trace. It is exact for a one-step finite window and inherits pp-prop's temporal
factorization for longer windows. The D-RTRL rules retain complete
parameter-to-output position Jacobians and are exact, with diagnostic-scale
memory cost.
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

__all__ = ["etp_gated_projection_p", "gated_projection"]

_PARAMETER_NAMES = ("gate_weight", "gate_bias", "output_weight")


def _gated_projection_impl(
    packed: Any,
    gate_weight: Any,
    gate_bias: Any,
    output_weight: Any,
    *,
    value_features: int,
    normalize: bool,
    epsilon: float,
) -> Any:
    values = packed[..., :value_features]
    gate_input = packed[..., value_features:]
    if normalize:
        values = values / jnp.sqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + epsilon
        )
    gate = jax.nn.sigmoid(gate_input @ gate_weight + gate_bias)
    return (gate * values) @ output_weight


def _trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    del params
    return {"gate_weight": 1, "gate_bias": 2, "output_weight": 3}


def _xy_to_dw(
    packed: Any,
    hidden_dim: Any,
    weights: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    def forward(current: dict[str, Any]) -> Any:
        return _gated_projection_impl(
            packed,
            current["gate_weight"],
            current["gate_bias"],
            current["output_weight"],
            **params,
        )

    _, pullback = jax.vjp(forward, weights)
    return jax.tree.map(u.get_mantissa, pullback(hidden_dim)[0])


def _init_pp(
    x_var: Any,
    y_var: Any,
    weight_vars: dict[str, Any],
    num_hidden_state: int,
) -> Any:
    del x_var, weight_vars
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _init_drtrl(
    x_var: Any,
    y_var: Any,
    weight_vars: dict[str, Any],
    num_hidden_state: int,
) -> dict[str, Any]:
    batch = y_var.aval.shape[0]
    output_positions = y_var.aval.shape[1:]
    dtype = jnp.result_type(
        x_var.aval.dtype,
        y_var.aval.dtype,
        *(weight_vars[name].aval.dtype for name in _PARAMETER_NAMES),
    )
    return {
        name: jnp.zeros(
            (batch, *weight_vars[name].aval.shape, *output_positions, num_hidden_state),
            dtype=dtype,
        )
        for name in _PARAMETER_NAMES
    }


def _dt_to_t(
    hidden_dim: Any,
    trace: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    del params
    result: dict[str, Any] = {}
    for name, parameter_rank in (
        ("gate_weight", 2),
        ("gate_bias", 1),
        ("output_weight", 2),
    ):
        lifted = hidden_dim
        for _ in range(parameter_rank):
            lifted = jnp.expand_dims(lifted, axis=-2)
        result[name] = trace[name] * lifted
    return result


def _instant_drtrl(
    packed: Any,
    df: Any,
    weights: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    def forward(current: dict[str, Any]) -> Any:
        return _gated_projection_impl(
            packed,
            current["gate_weight"],
            current["gate_bias"],
            current["output_weight"],
            **params,
        )

    jacobians = jax.jacrev(forward)(weights)
    return {
        name: jnp.moveaxis(jacobians[name], 0, -1) * df
        for name in _PARAMETER_NAMES
    }


def _solve_drtrl(
    dg_hidden: Any,
    trace: dict[str, Any],
    weights: dict[str, Any],
    **params: Any,
) -> dict[str, Any]:
    del weights, params
    dg = u.get_mantissa(dg_hidden)
    return {
        "gate_weight": jnp.einsum("...z,...gvz->...gv", dg, trace["gate_weight"]),
        "gate_bias": jnp.einsum("...z,...vz->...v", dg, trace["gate_bias"]),
        "output_weight": jnp.einsum(
            "...z,...voz->...vo", dg, trace["output_weight"]
        ),
    }


etp_gated_projection_p = register_primitive(
    "etp_gated_projection",
    _gated_projection_impl,
    batched=True,
    gradient_enabled=True,
    trainable_invars_fn=_trainable_invars,
    x_invar_index=0,
)
etp_gated_projection_p.register_etp_rules(
    dt_to_t=_dt_to_t,
    xy_to_dw=_xy_to_dw,
    init_drtrl=_init_drtrl,
    init_pp=_init_pp,
    snap_anchor=lambda params: False,
)
ETP_RULES_INSTANT_DRTRL[etp_gated_projection_p] = _instant_drtrl
ETP_RULES_SOLVE_DRTRL[etp_gated_projection_p] = _solve_drtrl


def _dimensionless_float(value: ArrayLike, name: str) -> Any:
    unit = u.get_unit(value)
    if not unit.is_unitless:
        raise ValueError(f"{name} must be dimensionless, got unit {unit}")
    result = jnp.asarray(u.get_mantissa(value))
    if not jnp.issubdtype(result.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype, got {result.dtype}")
    return result


def _epsilon(epsilon: float | None, dtype: Any) -> float:
    if epsilon is None:
        return float(jnp.finfo(dtype).eps)
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a finite positive real scalar or None")
    result = float(epsilon)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("epsilon must be a finite positive real scalar")
    return result


def gated_projection(
    values: ArrayLike,
    gate_input: ArrayLike,
    *,
    gate_weight: ArrayLike,
    gate_bias: ArrayLike,
    output_weight: ArrayLike,
    normalize: bool = False,
    epsilon: float | None = None,
) -> ArrayLike:
    r"""Apply a sigmoid-gated projection through one ETP primitive.

    Parameters
    ----------
    values : ArrayLike
        Batched value channels shaped ``(batch, value_features)``.
    gate_input : ArrayLike
        Batched gate features shaped ``(batch, gate_features)``.
    gate_weight : ArrayLike
        Trainable gate matrix shaped ``(gate_features, value_features)``.
    gate_bias : ArrayLike
        Trainable gate bias shaped ``(value_features,)``.
    output_weight : ArrayLike
        Trainable output matrix shaped ``(value_features, output_features)``.
    normalize : bool, optional
        RMS-normalize ``values`` before gating. Default ``False``.
    epsilon : float, optional
        Positive RMS stabilizer. ``None`` uses the value dtype epsilon.

    Returns
    -------
    ArrayLike
        Projected values shaped ``(batch, output_features)``.

    Raises
    ------
    TypeError
        If arrays are not floating or ``normalize`` is not boolean.
    ValueError
        If shapes disagree, an operand has physical units, or ``epsilon`` is
        not finite and positive.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> values = jnp.asarray([[1., 2.]])
        >>> gate_input = jnp.asarray([[3.]])
        >>> result = braintrace.gated_projection(
        ...     values,
        ...     gate_input,
        ...     gate_weight=jnp.zeros((1, 2)),
        ...     gate_bias=jnp.zeros(2),
        ...     output_weight=2. * jnp.eye(2),
        ... )
        >>> result.tolist()
        [[1.0, 2.0]]
    """
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    values_array = _dimensionless_float(values, "values")
    gate_array = _dimensionless_float(gate_input, "gate_input")
    gate_weight_array = _dimensionless_float(gate_weight, "gate_weight")
    gate_bias_array = _dimensionless_float(gate_bias, "gate_bias")
    output_weight_array = _dimensionless_float(output_weight, "output_weight")
    if values_array.ndim != 2 or gate_array.ndim != 2:
        raise ValueError("values and gate_input must be batched rank-two arrays")
    if values_array.shape[0] != gate_array.shape[0]:
        raise ValueError("values and gate_input must share the same batch size")
    value_features = values_array.shape[1]
    gate_features = gate_array.shape[1]
    if gate_weight_array.shape != (gate_features, value_features):
        raise ValueError(
            "gate_weight must have shape "
            f"({gate_features}, {value_features}), got {gate_weight_array.shape}"
        )
    if gate_bias_array.shape != (value_features,):
        raise ValueError(
            f"gate_bias must have shape ({value_features},), got {gate_bias_array.shape}"
        )
    if output_weight_array.ndim != 2 or output_weight_array.shape[0] != value_features:
        raise ValueError(
            "output_weight must have shape "
            f"({value_features}, output_features), got {output_weight_array.shape}"
        )
    packed = jnp.concatenate((values_array, gate_array), axis=-1)
    return etp_gated_projection_p.bind(
        packed,
        gate_weight_array,
        gate_bias_array,
        output_weight_array,
        value_features=value_features,
        normalize=bool(normalize),
        epsilon=_epsilon(epsilon, values_array.dtype),
    )
