# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Attention Residual neural-network module."""

from __future__ import annotations

from collections.abc import Callable

import brainstate
import braintools

from braintrace._op.attention import (
    _attention_residual_weights,
    attention_residual,
)
from braintrace._typing import ArrayLike

_ZERO_QUERY_INIT = braintools.init.ZeroInit()


class AttentionResidual(brainstate.nn.Module):
    r"""Learned pseudo-query mixer over residual sources.

    Parameters
    ----------
    hidden_size : int
        Width of every source and pseudo-query.
    query_count : int, optional
        Number of independently learned pseudo-queries. Default ``1``.
    epsilon : float, optional
        RMS-normalization constant. ``None`` uses the source dtype's machine
        epsilon at call time.
    query_init : Callable or ArrayLike, optional
        Initializer for the ``(query_count, hidden_size)`` query table. The
        default is exact zeros, which produces uniform attention.
    param_type : type, optional
        ``ParamState`` subclass owning the query table.
    name : str, optional
        Module name.

    Attributes
    ----------
    query : brainstate.ParamState
        Learned pseudo-query table shaped ``(query_count, hidden_size)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> layer = braintrace.nn.AttentionResidual(2)
        >>> sources = jnp.asarray([[1., 2.], [3., 4.]])
        >>> layer(sources).tolist()
        [2.0, 3.0]
        >>> layer.attention_weights(sources).tolist()
        [0.5, 0.5]
    """

    __module__ = "braintrace.nn"

    def __init__(
        self,
        hidden_size: int,
        query_count: int = 1,
        *,
        epsilon: float | None = None,
        query_init: ArrayLike | Callable = _ZERO_QUERY_INIT,
        param_type: type = brainstate.ParamState,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)  # type: ignore[call-arg]
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("hidden_size must be a positive integer")
        if isinstance(query_count, bool) or not isinstance(query_count, int):
            raise TypeError("query_count must be a positive integer")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if query_count <= 0:
            raise ValueError("query_count must be a positive integer")
        self.hidden_size = hidden_size
        self.query_count = query_count
        self.epsilon = epsilon
        self.query = param_type(
            braintools.init.param(
                query_init, (query_count, hidden_size), allow_none=False
            )
        )

    def update(
        self,
        sources: ArrayLike,
        *,
        source_mask: ArrayLike | None = None,
        query_index: int | ArrayLike = 0,
    ) -> ArrayLike:
        """Return the learned convex mixture of ``sources``.

        Parameters
        ----------
        sources : ArrayLike
            Values shaped ``(..., source_count, hidden_size)``.
        source_mask : ArrayLike, optional
            Boolean source-validity mask.
        query_index : int or ArrayLike, optional
            Pseudo-query selected for each leading position. Default ``0``.

        Returns
        -------
        ArrayLike
            Mixed values shaped ``(..., hidden_size)``.
        """
        return attention_residual(
            sources,
            self.query.value,
            source_mask=source_mask,
            query_index=query_index,
            epsilon=self.epsilon,
        )

    def attention_weights(
        self,
        sources: ArrayLike,
        *,
        source_mask: ArrayLike | None = None,
        query_index: int | ArrayLike = 0,
    ) -> ArrayLike:
        """Return source-axis weights without mutating module state.

        Parameters
        ----------
        sources : ArrayLike
            Values shaped ``(..., source_count, hidden_size)``.
        source_mask : ArrayLike, optional
            Boolean source-validity mask.
        query_index : int or ArrayLike, optional
            Pseudo-query selected for each leading position. Default ``0``.

        Returns
        -------
        ArrayLike
            Convex weights shaped ``(..., source_count)``.
        """
        return _attention_residual_weights(
            sources,
            self.query.value,
            source_mask=source_mask,
            query_index=query_index,
            epsilon=self.epsilon,
        )
