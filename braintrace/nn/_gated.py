# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Neural-network wrapper for the fused gated projection."""

from __future__ import annotations

from collections.abc import Callable

import brainstate
import braintools
import jax

from braintrace._op.gated import gated_projection
from braintrace._typing import ArrayLike

_ZERO_INIT = braintools.init.ZeroInit()
_XAVIER_NORMAL_INIT = braintools.init.XavierNormal()


class GatedProjection(brainstate.nn.Module):
    r"""Sigmoid-gated projection with optional RMS-normalized values.

    Parameters
    ----------
    value_size : int
        Number of value channels.
    gate_size : int
        Number of gate-input features.
    output_size : int
        Number of projected output channels.
    normalize : bool, optional
        RMS-normalize values inside the fused primitive. Default ``False``.
    epsilon : float, optional
        Positive RMS stabilizer. ``None`` uses the value dtype epsilon.
    gate_weight_init, gate_bias_init : Callable or ArrayLike, optional
        Gate parameter initializers. Both default to exact zeros.
    output_weight_init : Callable or ArrayLike, optional
        Output-matrix initializer. Default is Xavier normal.
    param_type : type, optional
        ``ParamState`` subclass owning each trainable operand.
    name : str, optional
        Module name.

    Attributes
    ----------
    gate_weight, gate_bias, output_weight : brainstate.ParamState
        Parameters consumed together by :func:`braintrace.gated_projection`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> layer = braintrace.nn.GatedProjection(
        ...     2, 1, 2, output_weight_init=2. * jnp.eye(2)
        ... )
        >>> layer(jnp.asarray([[1., 2.]]), jnp.asarray([[3.]])).tolist()
        [[1.0, 2.0]]
    """

    __module__ = "braintrace.nn"

    def __init__(
        self,
        value_size: int,
        gate_size: int,
        output_size: int,
        *,
        normalize: bool = False,
        epsilon: float | None = None,
        gate_weight_init: ArrayLike | Callable[..., ArrayLike] = _ZERO_INIT,
        gate_bias_init: ArrayLike | Callable[..., ArrayLike] = _ZERO_INIT,
        output_weight_init: ArrayLike | Callable[..., ArrayLike] = _XAVIER_NORMAL_INIT,
        param_type: type = brainstate.ParamState,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        for dimension, dimension_name in (
            (value_size, "value_size"),
            (gate_size, "gate_size"),
            (output_size, "output_size"),
        ):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(f"{dimension_name} must be a positive integer. Set {dimension_name} to a positive integer.")
            if dimension <= 0:
                raise ValueError(f"{dimension_name} must be a positive integer. Set {dimension_name} to a positive integer.")
        self.value_size = value_size
        self.gate_size = gate_size
        self.output_size = output_size
        self.normalize = normalize
        self.epsilon = epsilon
        self.gate_weight = param_type(
            braintools.init.param(
                gate_weight_init, (gate_size, value_size), allow_none=False
            )
        )
        self.gate_bias = param_type(
            braintools.init.param(gate_bias_init, (value_size,), allow_none=False)
        )
        self.output_weight = param_type(
            braintools.init.param(
                output_weight_init, (value_size, output_size), allow_none=False
            )
        )

    def update(self, values: ArrayLike, gate_input: ArrayLike) -> ArrayLike:
        """Gate and project ``values`` using ``gate_input``.

        Parameters
        ----------
        values : ArrayLike
            Batched values shaped ``(batch, value_size)``.
        gate_input : ArrayLike
            Batched gate features shaped ``(batch, gate_size)``.

        Returns
        -------
        ArrayLike
            Projected values shaped ``(batch, output_size)``.
        """
        return gated_projection(
            values,
            gate_input,
            gate_weight=self.gate_weight.value,
            gate_bias=self.gate_bias.value,
            output_weight=self.output_weight.value,
            normalize=self.normalize,
            epsilon=self.epsilon,
        )

    def gate_activation(self, gate_input: ArrayLike) -> ArrayLike:
        """Return stop-gradient gate activations for diagnostics.

        Parameters
        ----------
        gate_input : ArrayLike
            Batched gate features shaped ``(batch, gate_size)``.

        Returns
        -------
        ArrayLike
            Sigmoid gate values shaped ``(batch, value_size)``. Parameters and
            inputs are detached so this diagnostic cannot create a second
            trainable path beside the fused ETP primitive.
        """
        return jax.nn.sigmoid(
            jax.lax.stop_gradient(gate_input)
            @ jax.lax.stop_gradient(self.gate_weight.value)
            + jax.lax.stop_gradient(self.gate_bias.value)
        )
