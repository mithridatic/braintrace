# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

r"""Fused error-correcting memory update and SiTU-GLU operators.

Both primitives keep every trainable projection in one ETP equation. Their
pp-prop rules evaluate a complete parameter VJP at the retained input trace and
are exact for a one-step finite window; longer-window use is approximate under
pp-prop. D-RTRL retains full parameter-to-output position Jacobians.
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

__all__ = [
    "delta_memory_update",
    "etp_delta_memory_update_p",
    "etp_situ_glu_p",
    "situ_glu",
]

_DELTA_PARAMETERS = (
    "key_weight",
    "key_bias",
    "value_weight",
    "beta_weight",
    "beta_bias",
    "retention_weight",
    "retention_bias",
)
_SITU_PARAMETERS = (
    "gate_weight",
    "gate_bias",
    "up_weight",
    "up_bias",
    "output_weight",
)


def _dimensionless_float(value: ArrayLike, name: str) -> Any:
    unit = u.get_unit(value)
    if not unit.is_unitless:
        raise ValueError(f"{name} must be dimensionless, got unit {unit}. Set {name} to dimensionless.")
    result = jnp.asarray(u.get_mantissa(value))
    if not jnp.issubdtype(result.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype, got {result.dtype}. Ensure {name} has a floating dtype.")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real scalar. Set {name} to a finite positive real scalar.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar. Set {name} to a finite positive real scalar.")
    return result


def _negative_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite negative real scalar. Set {name} to a finite negative real scalar.")
    result = float(value)
    if not math.isfinite(result) or result >= 0.0:
        raise ValueError(f"{name} must be a finite negative real scalar. Set {name} to a finite negative real scalar.")
    return result


def _delta_impl(
    packed: Any,
    key_weight: Any,
    key_bias: Any,
    value_weight: Any,
    beta_weight: Any,
    beta_bias: Any,
    retention_weight: Any,
    retention_bias: Any,
    *,
    memory_key_width: int,
    memory_value_width: int,
    key_features: int,
    key_scale: float,
    min_log_decay: float,
    epsilon: float,
) -> Any:
    memory_elements = memory_key_width * memory_value_width
    memory = packed[..., :memory_elements].reshape(
        *packed.shape[:-1], memory_key_width, memory_value_width
    )
    x_key = packed[..., memory_elements : memory_elements + key_features]
    x_value = packed[..., memory_elements + key_features :]
    key = key_scale * (x_key @ key_weight + key_bias)
    key = key / jnp.maximum(
        jnp.linalg.norm(key, axis=-1, keepdims=True), epsilon
    )
    value = x_value @ value_weight
    beta = jax.nn.sigmoid(x_value @ beta_weight + beta_bias)
    alpha = jnp.exp(
        min_log_decay
        * jax.nn.sigmoid(x_value @ retention_weight + retention_bias)
    )
    decayed = memory * alpha[..., None, :]
    prediction = jnp.einsum("...kv,...k->...v", decayed, key)
    error = value - prediction
    return decayed + beta[..., :, None] * jnp.einsum(
        "...k,...v->...kv", key, error
    )


def _delta_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    del params
    return {name: index + 1 for index, name in enumerate(_DELTA_PARAMETERS)}


def _situ_impl(
    x: Any,
    gate_weight: Any,
    gate_bias: Any,
    up_weight: Any,
    up_bias: Any,
    output_weight: Any,
    *,
    gate_beta: float,
    up_beta: float,
) -> Any:
    gate_pre = x @ gate_weight + gate_bias
    up_pre = x @ up_weight + up_bias
    gate = gate_beta * jnp.tanh(gate_pre / gate_beta) * jax.nn.sigmoid(gate_pre)
    up = up_beta * jnp.tanh(up_pre / up_beta)
    return (gate * up) @ output_weight


def _situ_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    del params
    return {name: index + 1 for index, name in enumerate(_SITU_PARAMETERS)}


def _xy_to_dw_factory(impl: Any) -> Any:
    def rule(x: Any, hidden_dim: Any, weights: dict[str, Any], **params: Any) -> Any:
        def forward(current: dict[str, Any]) -> Any:
            return impl(x, **current, **params)

        _, pullback = jax.vjp(forward, weights)
        return jax.tree.map(u.get_mantissa, pullback(hidden_dim)[0])

    return rule


def _init_pp(
    x_var: Any,
    y_var: Any,
    weight_vars: dict[str, Any],
    num_hidden_state: int,
) -> Any:
    del x_var, weight_vars
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _init_drtrl_factory(parameter_names: tuple[str, ...]) -> Any:
    def rule(
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
            *(weight_vars[name].aval.dtype for name in parameter_names),
        )
        return {
            name: jnp.zeros(
                (
                    batch,
                    *weight_vars[name].aval.shape,
                    *output_positions,
                    num_hidden_state,
                ),
                dtype=dtype,
            )
            for name in parameter_names
        }

    return rule


def _dt_to_t_factory(
    parameter_ranks: dict[str, int], *, output_rank: int
) -> Any:
    def rule(hidden_dim: Any, trace: dict[str, Any], **params: Any) -> dict[str, Any]:
        del params
        result: dict[str, Any] = {}
        for name, parameter_rank in parameter_ranks.items():
            lifted = hidden_dim
            for _ in range(parameter_rank):
                lifted = jnp.expand_dims(lifted, axis=-(output_rank + 1))
            result[name] = trace[name] * lifted
        return result

    return rule


def _instant_drtrl_factory(
    impl: Any, parameter_names: tuple[str, ...], *, output_rank: int
) -> Any:
    def rule(
        x: Any,
        df: Any,
        weights: dict[str, Any],
        **params: Any,
    ) -> dict[str, Any]:
        def forward(current: dict[str, Any]) -> Any:
            return impl(x, **current, **params)

        jacobians = jax.jacrev(forward)(weights)
        return {
            name: jnp.moveaxis(
                jacobians[name],
                tuple(range(output_rank)),
                tuple(range(-output_rank, 0)),
            )
            * df
            for name in parameter_names
        }

    return rule


def _solve_drtrl_factory(
    parameter_names: tuple[str, ...], *, output_rank: int
) -> Any:
    def rule(
        dg_hidden: Any,
        trace: dict[str, Any],
        weights: dict[str, Any],
        **params: Any,
    ) -> dict[str, Any]:
        del weights, params
        dg = u.get_mantissa(dg_hidden)
        result: dict[str, Any] = {}
        for name in parameter_names:
            parameter_rank = trace[name].ndim - dg.ndim
            if output_rank == 1 and parameter_rank == 1:
                result[name] = jnp.einsum("...z,...az->...a", dg, trace[name])
            elif output_rank == 1 and parameter_rank == 2:
                result[name] = jnp.einsum("...z,...abz->...ab", dg, trace[name])
            elif output_rank == 2 and parameter_rank == 1:
                result[name] = jnp.einsum("...ij,...aij->...a", dg, trace[name])
            elif output_rank == 2 and parameter_rank == 2:
                result[name] = jnp.einsum("...ij,...abij->...ab", dg, trace[name])
            else:
                raise ValueError(
                    f"Unsupported output/parameter ranks {output_rank}/{parameter_rank}. Use a supported option or change the configuration."
                )
        return result

    return rule


etp_delta_memory_update_p = register_primitive(
    "etp_delta_memory_update",
    _delta_impl,
    batched=True,
    gradient_enabled=True,
    trainable_invars_fn=_delta_trainable_invars,
    x_invar_index=0,
)
etp_delta_memory_update_p.register_etp_rules(
    dt_to_t=_dt_to_t_factory(
        {
            "key_weight": 2,
            "key_bias": 1,
            "value_weight": 2,
            "beta_weight": 2,
            "beta_bias": 1,
            "retention_weight": 2,
            "retention_bias": 1,
        },
        output_rank=2,
    ),
    xy_to_dw=_xy_to_dw_factory(_delta_impl),
    init_drtrl=_init_drtrl_factory(_DELTA_PARAMETERS),
    init_pp=_init_pp,
    snap_anchor=lambda params: False,
)
ETP_RULES_INSTANT_DRTRL[etp_delta_memory_update_p] = _instant_drtrl_factory(
    _delta_impl, _DELTA_PARAMETERS, output_rank=2
)
ETP_RULES_SOLVE_DRTRL[etp_delta_memory_update_p] = _solve_drtrl_factory(
    _DELTA_PARAMETERS, output_rank=2
)

etp_situ_glu_p = register_primitive(
    "etp_situ_glu",
    _situ_impl,
    batched=True,
    gradient_enabled=True,
    trainable_invars_fn=_situ_trainable_invars,
    x_invar_index=0,
)
etp_situ_glu_p.register_etp_rules(
    dt_to_t=_dt_to_t_factory(
        {
            "gate_weight": 2,
            "gate_bias": 1,
            "up_weight": 2,
            "up_bias": 1,
            "output_weight": 2,
        },
        output_rank=1,
    ),
    xy_to_dw=_xy_to_dw_factory(_situ_impl),
    init_drtrl=_init_drtrl_factory(_SITU_PARAMETERS),
    init_pp=_init_pp,
    snap_anchor=lambda params: False,
)
ETP_RULES_INSTANT_DRTRL[etp_situ_glu_p] = _instant_drtrl_factory(
    _situ_impl, _SITU_PARAMETERS, output_rank=1
)
ETP_RULES_SOLVE_DRTRL[etp_situ_glu_p] = _solve_drtrl_factory(
    _SITU_PARAMETERS, output_rank=1
)


def delta_memory_update(
    memory: ArrayLike,
    x_key: ArrayLike,
    x_value: ArrayLike,
    *,
    key_weight: ArrayLike,
    key_bias: ArrayLike,
    value_weight: ArrayLike,
    beta_weight: ArrayLike,
    beta_bias: ArrayLike,
    retention_weight: ArrayLike,
    retention_bias: ArrayLike,
    key_scale: float = 1.0,
    min_log_decay: float = -5.0,
) -> ArrayLike:
    r"""Return an input-dependent error-correcting memory candidate.

    Parameters
    ----------
    memory : ArrayLike
        Batched memory shaped ``(batch, key_width, value_width)``.
    x_key, x_value : ArrayLike
        Batched key and value features.
    key_weight, key_bias, value_weight : ArrayLike
        Key and value projection parameters.
    beta_weight, beta_bias : ArrayLike
        Scalar write-strength projection parameters.
    retention_weight, retention_bias : ArrayLike
        Value-channel retention projection parameters.
    key_scale : float, optional
        Positive normalized-key scale. Default ``1``.
    min_log_decay : float, optional
        Negative lower log-retention bound. Default ``-5``.

    Returns
    -------
    ArrayLike
        Candidate memory with the same shape as ``memory``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> memory = jnp.zeros((1, 2, 2))
        >>> result = braintrace.delta_memory_update(
        ...     memory, jnp.ones((1, 2)), jnp.ones((1, 2)),
        ...     key_weight=jnp.eye(2), key_bias=jnp.zeros(2),
        ...     value_weight=jnp.eye(2), beta_weight=jnp.zeros((2, 1)),
        ...     beta_bias=jnp.zeros(1), retention_weight=jnp.zeros((2, 2)),
        ...     retention_bias=jnp.zeros(2),
        ... )
        >>> result.shape
        (1, 2, 2)
    """
    arrays = {
        name: _dimensionless_float(value, name)
        for name, value in (
            ("memory", memory),
            ("x_key", x_key),
            ("x_value", x_value),
            ("key_weight", key_weight),
            ("key_bias", key_bias),
            ("value_weight", value_weight),
            ("beta_weight", beta_weight),
            ("beta_bias", beta_bias),
            ("retention_weight", retention_weight),
            ("retention_bias", retention_bias),
        )
    }
    memory_array = arrays.pop("memory")
    x_key_array = arrays.pop("x_key")
    x_value_array = arrays.pop("x_value")
    if memory_array.ndim != 3 or x_key_array.ndim != 2 or x_value_array.ndim != 2:
        raise ValueError("Memory must be rank three and feature inputs rank two. Set Memory to rank three and feature inputs rank two.")
    if not (
        memory_array.shape[0] == x_key_array.shape[0] == x_value_array.shape[0]
    ):
        raise ValueError("Memory and feature inputs must share the batch size. Make Memory and feature inputs share the batch size.")
    key_width, value_width = memory_array.shape[1:]
    key_features = x_key_array.shape[1]
    value_features = x_value_array.shape[1]
    expected = {
        "key_weight": (key_features, key_width),
        "key_bias": (key_width,),
        "value_weight": (value_features, value_width),
        "beta_weight": (value_features, 1),
        "beta_bias": (1,),
        "retention_weight": (value_features, value_width),
        "retention_bias": (value_width,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}. Ensure {name} has shape {shape}.")
    packed = jnp.concatenate(
        (memory_array.reshape(memory_array.shape[0], -1), x_key_array, x_value_array),
        axis=-1,
    )
    return etp_delta_memory_update_p.bind(
        packed,
        *(arrays[name] for name in _DELTA_PARAMETERS),
        memory_key_width=key_width,
        memory_value_width=value_width,
        key_features=key_features,
        key_scale=_positive_real(key_scale, "key_scale"),
        min_log_decay=_negative_real(min_log_decay, "min_log_decay"),
        epsilon=float(jnp.finfo(memory_array.dtype).eps),
    )


def situ_glu(
    x: ArrayLike,
    *,
    gate_weight: ArrayLike,
    gate_bias: ArrayLike,
    up_weight: ArrayLike,
    up_bias: ArrayLike,
    output_weight: ArrayLike,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
) -> ArrayLike:
    r"""Apply a complete SiTU-GLU projection through one ETP primitive.

    Parameters
    ----------
    x : ArrayLike
        Batched input shaped ``(batch, input_width)``.
    gate_weight, up_weight : ArrayLike
        Input-to-hidden matrices with identical shape.
    gate_bias, up_bias : ArrayLike
        Hidden biases.
    output_weight : ArrayLike
        Hidden-to-output matrix.
    gate_beta, up_beta : float, optional
        Positive softcap magnitudes. Defaults are ``4`` and ``25``.

    Returns
    -------
    ArrayLike
        Batched output projection.
    """
    arrays = {
        name: _dimensionless_float(value, name)
        for name, value in (
            ("x", x),
            ("gate_weight", gate_weight),
            ("gate_bias", gate_bias),
            ("up_weight", up_weight),
            ("up_bias", up_bias),
            ("output_weight", output_weight),
        )
    }
    x_array = arrays.pop("x")
    if x_array.ndim != 2:
        raise ValueError("X must be a batched rank-two array. Set X to a batched rank-two array.")
    if arrays["gate_weight"].ndim != 2:
        raise ValueError("gate_weight must be rank two. Set gate_weight to rank two.")
    input_width, hidden_width = arrays["gate_weight"].shape
    expected = {
        "gate_weight": (input_width, hidden_width),
        "gate_bias": (hidden_width,),
        "up_weight": (input_width, hidden_width),
        "up_bias": (hidden_width,),
    }
    if x_array.shape[1] != input_width:
        raise ValueError(f"X must have final width {input_width}. Ensure X has final width {input_width}.")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}. Ensure {name} has shape {shape}.")
    if arrays["output_weight"].ndim != 2 or arrays["output_weight"].shape[0] != hidden_width:
        raise ValueError(
            "output_weight must have shape "
            f"({hidden_width}, output_width), got {arrays['output_weight'].shape}. Set output_weight to a value with shape "
            f"({hidden_width}, output_width)."
        )
    return etp_situ_glu_p.bind(
        x_array,
        *(arrays[name] for name in _SITU_PARAMETERS),
        gate_beta=_positive_real(gate_beta, "gate_beta"),
        up_beta=_positive_real(up_beta, "up_beta"),
    )
