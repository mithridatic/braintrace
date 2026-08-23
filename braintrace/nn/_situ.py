# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Neural-network wrapper for the fused SiTU-GLU projection."""

from __future__ import annotations

from collections.abc import Callable

import brainstate
import braintools

from braintrace._op.memory import situ_glu
from braintrace._typing import ArrayLike

_ZERO_INIT = braintools.init.ZeroInit()
_XAVIER_NORMAL_INIT = braintools.init.XavierNormal()


class SiTUGLU(brainstate.nn.Module):
    r"""Softcapped SiLU gate times a softcapped up projection.

    Parameters
    ----------
    input_size, hidden_size, output_size : int
        Positive input, multiplicative hidden, and output widths.
    gate_beta, up_beta : float, optional
        Positive softcap magnitudes. Defaults are ``4`` and ``25``.
    gate_weight_init, up_weight_init, output_weight_init : Callable or ArrayLike
        Matrix initializers. Defaults are Xavier normal.
    gate_bias_init, up_bias_init : Callable or ArrayLike
        Bias initializers. Defaults are exact zeros.
    param_type : type, optional
        ``ParamState`` subclass owning each trainable operand.
    name : str, optional
        Module name.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> layer = braintrace.nn.SiTUGLU(2, 4, 3)
        >>> layer(jnp.ones((1, 2))).shape
        (1, 3)
    """

    __module__ = "braintrace.nn"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        gate_weight_init: ArrayLike | Callable[..., ArrayLike] = _XAVIER_NORMAL_INIT,
        gate_bias_init: ArrayLike | Callable[..., ArrayLike] = _ZERO_INIT,
        up_weight_init: ArrayLike | Callable[..., ArrayLike] = _XAVIER_NORMAL_INIT,
        up_bias_init: ArrayLike | Callable[..., ArrayLike] = _ZERO_INIT,
        output_weight_init: ArrayLike | Callable[..., ArrayLike] = _XAVIER_NORMAL_INIT,
        param_type: type = brainstate.ParamState,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)  # pyright: ignore[reportCallIssue]
        for size, size_name in (
            (input_size, "input_size"),
            (hidden_size, "hidden_size"),
            (output_size, "output_size"),
        ):
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError(f"{size_name} must be a positive integer. Set {size_name} to a positive integer.")
            if size <= 0:
                raise ValueError(f"{size_name} must be a positive integer. Set {size_name} to a positive integer.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gate_beta = gate_beta
        self.up_beta = up_beta
        self.gate_weight = param_type(
            braintools.init.param(
                gate_weight_init, (input_size, hidden_size), allow_none=False
            )
        )
        self.gate_bias = param_type(
            braintools.init.param(gate_bias_init, (hidden_size,), allow_none=False)
        )
        self.up_weight = param_type(
            braintools.init.param(
                up_weight_init, (input_size, hidden_size), allow_none=False
            )
        )
        self.up_bias = param_type(
            braintools.init.param(up_bias_init, (hidden_size,), allow_none=False)
        )
        self.output_weight = param_type(
            braintools.init.param(
                output_weight_init, (hidden_size, output_size), allow_none=False
            )
        )

    def update(self, x: ArrayLike) -> ArrayLike:
        """Apply the fused SiTU-GLU projection.

        Parameters
        ----------
        x : ArrayLike
            Batched input shaped ``(batch, input_size)``.

        Returns
        -------
        ArrayLike
            Output shaped ``(batch, output_size)``.
        """
        return situ_glu(
            x,
            gate_weight=self.gate_weight.value,
            gate_bias=self.gate_bias.value,
            up_weight=self.up_weight.value,
            up_bias=self.up_bias.value,
            output_weight=self.output_weight.value,
            gate_beta=self.gate_beta,
            up_beta=self.up_beta,
        )
