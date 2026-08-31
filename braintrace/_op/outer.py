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

r"""Fused outer-product write primitive ``etp_outer_write_p``.

**Forward operation**

.. math::

    y_{bij} = c \, \varphi_k\!\big(x^{(k)}_b W_k + b_k\big)_i \;
                  \varphi_v\!\big(x^{(v)}_b W_v\big)_j

producing the rank-one write matrix of an associative memory
:math:`S_t = \lambda S_{t-1} + g_t \, y_t \odot \Sigma` directly, so the whole
key/value *coding* sits inside one ETP primitive.

**Why fused**

An outer product written as three separate operations -- two ``etp_mm``
projections feeding an ``einsum('bi,bj->bij')`` -- is rejected twice over.
The ``einsum`` lowers to ``dot_general``, which the position prover classifies
as mixing hidden positions (key position ``i`` reaches the whole row
``S[i, :]``), so pp-prop refuses the tail; and a weight whose only route to a
hidden state passes through *another* trainable ETP primitive violates the
non-parametric-tail invariant. Fusing the projections into the primitive makes
the compiler see a single ``weights -> y`` arrow whose ``y`` is already
hidden-shaped, leaving a genuinely elementwise tail (gate, decay, write scale,
side-validity mask) between ``y`` and ``S``.

**Two data inputs, one x-invar**

The registry gives a primitive one ``x``. The user-facing :func:`outer_write`
therefore passes ``x = concatenate([x_key, x_value], axis=-1)`` and the rules
split it at the static ``key_features`` parameter. This is exact rather than a
workaround: the IO-dim input trace is a linear elementwise low-pass filter, so
filtering the concatenation equals concatenating the filtered halves, and the
solve step needs *both* filtered inputs anyway.

**Approximation class**

Unlike ``etp_mm``, this primitive is nonlinear in ``x``, so it must **not**
defer its weight gradient to a solve-time VJP: the IO-dim solve path evaluates
``xy_to_dw`` at the low-pass-filtered ``x``, which would (i) evaluate
:math:`\varphi'_k` and :math:`\varphi_v` at a smoothed argument -- a Jensen-gap
error no other primitive carries -- and (ii) destroy the within-timestep
correlation between the key and value factors, since the two halves of ``x``
are filtered independently. For a memory whose purpose is *binding*, that
second error would silently erase the effect under study.

The primitive therefore registers a per-step factor rule
(:data:`~braintrace._op._registries.ETP_RULES_PP_DF_FACTORS`) that injects the
full postsynaptic factors

.. math::

    D_f^{W_k} = D_f^{b_k} = c \, \varphi'_k(p^t) \otimes \varphi_v(q^t),
    \qquad
    D_f^{W_v} = c \, \varphi_k(p^t) \otimes \varphi'_v(q^t)

into two y-shaped traces at *their own* timestep, leaving ``xy_to_dw`` a purely
linear contraction against the filtered ``x``. Composed with one step of
history the pair is exact (see
:func:`~braintrace._op.op_rule_oracle.assert_factored_rules_match_vjp`); the
residual approximation is then pp-prop's own :math:`\epsilon_f \otimes
\epsilon_x` collapse and nothing else.

Two trace entries suffice for three trainable inputs: ``key_bias`` shares
``key_weight``'s postsynaptic factor exactly and differs only in its
presynaptic factor, which is the constant one.

**D-RTRL: exact position-retaining traces**

A dense-style parameter-shaped trace cannot be exact for this primitive: one
key weight entry :math:`W_k[a, i]` influences the whole memory row
:math:`S[i, :]`, not a single position. The param-dim (D-RTRL) trace therefore
retains both position axes per slot,

.. math::

    \epsilon_{W_k} \in \mathbb{R}^{B \times A_k \times K \times V \times S},
    \quad
    \epsilon_{b_k} \in \mathbb{R}^{B \times K \times V \times S},
    \quad
    \epsilon_{W_v} \in \mathbb{R}^{B \times A_v \times K \times V \times S},

and the primitive registers the trace-structured override pair
(:data:`~braintrace._op._registries.ETP_RULES_INSTANT_DRTRL` /
:data:`~braintrace._op._registries.ETP_RULES_SOLVE_DRTRL`, the LoRA precedent)
so the instantaneous term lands per position and the solve step contracts the
axis each parameter does not have. With the position axes retained, the
diagonal recurrence loses nothing on an elementwise memory tail: for a model
whose hidden Jacobian is genuinely diagonal (the associative memory
:math:`S_t = \lambda S_{t-1} + g_t \, y_t \odot \Sigma` is), D-RTRL on this
primitive reproduces BPTT element-wise at any window length — it does **not**
rely on the sign-consistency premise behind pp-prop's rank-1 collapse, which
the cos/tanh factors violate by construction. The memory cost is the point of
this trace (≈``B·(A_k+A_v+1)·K·V·S`` floats): affordable at diagnostic scale,
deliberate at model scale.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import brainunit as u
import jax.numpy as jnp

from braintrace._typing import ArrayLike
from ._primitive import register_primitive
from ._registries import ETP_RULES_INSTANT_DRTRL, ETP_RULES_SOLVE_DRTRL

__all__ = [
    'etp_outer_write_p',
    'outer_write',
]

# Enumerated (not arbitrary-callable) nonlinearities, so the jaxpr equation
# parameters stay hashable, serialisable and stable across retraces.
_FORWARD: Dict[str, Callable[[Any], Any]] = {
    'cos_rff': jnp.cos,
    'tanh': jnp.tanh,
}

_DERIVATIVE: Dict[str, Callable[[Any], Any]] = {
    'cos_rff': lambda pre: -jnp.sin(pre),
    'tanh': lambda pre: 1.0 - jnp.square(jnp.tanh(pre)),
}


def _codes(x: Any, weights: Dict[str, Any], key_features: int, key_scale: float,
           key_nonlinearity: str, value_nonlinearity: str) -> tuple:
    """Return ``(pre_key, pre_value, key_code, value_code)`` for a packed ``x``.

    ``key_code`` carries ``key_scale``; ``value_code`` does not, so that every
    downstream product is scaled exactly once.
    """
    x_key = x[..., :key_features]
    x_value = x[..., key_features:]
    pre_key = x_key @ weights['key_weight'] + weights['key_bias']
    pre_value = x_value @ weights['value_weight']
    key_code = key_scale * _FORWARD[key_nonlinearity](pre_key)
    value_code = _FORWARD[value_nonlinearity](pre_value)
    return pre_key, pre_value, key_code, value_code


def _outer_write_impl(x: Any, key_weight: Any, key_bias: Any, value_weight: Any, *,
                      key_features: int, key_scale: float,
                      key_nonlinearity: str = 'cos_rff',
                      value_nonlinearity: str = 'tanh') -> Any:
    weights = {'key_weight': key_weight, 'key_bias': key_bias,
               'value_weight': value_weight}
    _, _, key_code, value_code = _codes(
        x, weights, key_features, key_scale, key_nonlinearity, value_nonlinearity)
    return jnp.einsum('bi,bj->bij', key_code, value_code)


def _outer_write_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return the fixed ``{key: invar_index}`` layout of this primitive."""
    return {'key_weight': 1, 'key_bias': 2, 'value_weight': 3}


def _outer_write_pp_df_factors(x: Any, weights: Dict[str, Any], *,
                               key_features: int, key_scale: float,
                               key_nonlinearity: str = 'cos_rff',
                               value_nonlinearity: str = 'tanh') -> Dict[str, Any]:
    r"""Per-step postsynaptic factors, one per df-trace entry.

    Parameters
    ----------
    x : Any
        The primitive's **raw current-step** packed input, shape
        ``(batch, key_features + value_features)``.
    weights : dict
        Current values of the three trainable inputs.

    Returns
    -------
    dict
        ``{'key': c·φ'_k(p) ⊗ φ_v(q), 'value': c·φ_k(p) ⊗ φ'_v(q)}``, each
        y-shaped ``(batch, key_out, value_out)``.

    Notes
    -----
    ``'key'`` is :math:`\partial y / \partial p` and ``'value'`` is
    :math:`\partial y / \partial q`, i.e. the Jacobians of the output with
    respect to the two pre-activations. Multiplying the injected
    :math:`\mathbf{D}_f^t` by these leaves exactly the presynaptic contraction
    for :func:`_outer_write_xy_to_dw`, with no nonlinearity evaluated at a
    filtered argument.
    """
    pre_key, pre_value, key_code, value_code = _codes(
        x, weights, key_features, key_scale, key_nonlinearity, value_nonlinearity)
    key_slope = key_scale * _DERIVATIVE[key_nonlinearity](pre_key)
    value_slope = _DERIVATIVE[value_nonlinearity](pre_value)
    return {
        'key': jnp.einsum('bi,bj->bij', key_slope, value_code),
        'value': jnp.einsum('bi,bj->bij', key_code, value_slope),
    }


def _outer_write_xy_to_dw(x: Any, hidden_dim: Dict[str, Any],
                          weights: Dict[str, Any], *, key_features: int,
                          key_scale: float = 1.0,
                          key_nonlinearity: str = 'cos_rff',
                          value_nonlinearity: str = 'tanh') -> Dict[str, Any]:
    r"""Contract the factored df traces with the filtered presynaptic input.

    Parameters
    ----------
    x : Any
        The filtered packed input, shape ``(batch, key_features + value_features)``.
    hidden_dim : dict
        The df traces produced by :func:`_outer_write_pp_df_factors` and carried
        by the algorithm: ``{'key': ..., 'value': ...}``, each y-shaped.
    weights : dict
        Current weight values. Unused -- every nonlinear factor was already
        applied at its own timestep, which is precisely the point.

    Returns
    -------
    dict
        Gradients keyed by trainable name, each shaped like its parameter.

    Notes
    -----
    ``key_bias`` reads the same ``'key'`` trace as ``key_weight``; its
    presynaptic factor is the constant one, so the contraction degenerates to a
    sum over the batch and value axes.
    """
    x_key = u.get_mantissa(x[..., :key_features])
    x_value = u.get_mantissa(x[..., key_features:])
    key_df = u.get_mantissa(hidden_dim['key'])
    value_df = u.get_mantissa(hidden_dim['value'])
    return {
        'key_weight': jnp.einsum('ba,bij->ai', x_key, key_df),
        'key_bias': jnp.einsum('bij->i', key_df),
        'value_weight': jnp.einsum('ba,bij->aj', x_value, value_df),
    }


def _outer_write_init_pp(x_var: Any, y_var: Any, weight_vars: Dict[str, Any],
                         num_hidden_state: int) -> Dict[str, Any]:
    r"""Allocate one zeroed y-shaped df trace per factor group.

    .. math::

        \boldsymbol{\epsilon}_f^{\text{key}},
        \boldsymbol{\epsilon}_f^{\text{value}} \in
        \mathbb{R}^{B \times K \times V \times n_{\text{state}}}
    """
    shape = (*y_var.aval.shape, num_hidden_state)
    dtype = y_var.aval.dtype
    return {'key': jnp.zeros(shape, dtype=dtype),
            'value': jnp.zeros(shape, dtype=dtype)}


def _outer_write_init_drtrl(x_var: Any, y_var: Any, weight_vars: Dict[str, Any],
                            num_hidden_state: int) -> Dict[str, Any]:
    r"""Allocate the exact position-retaining D-RTRL traces.

    Every entry keeps both memory position axes ``(key_out, value_out)`` per
    trace slot — a parameter-shaped trace cannot follow the diagonal
    recurrence here because one weight entry influences a whole memory row
    (see the module docstring):

    .. math::

        \epsilon_{W_k} \in \mathbb{R}^{B \times A_k \times K \times V \times S},
        \quad
        \epsilon_{b_k} \in \mathbb{R}^{B \times K \times V \times S},
        \quad
        \epsilon_{W_v} \in \mathbb{R}^{B \times A_v \times K \times V \times S}.

    Zero-initialised (:math:`\epsilon^0 = 0`); dtype via
    :func:`jax.numpy.result_type` of the participating avals, following the
    dense precedent.
    """
    batch = x_var.aval.shape[0]
    positions = y_var.aval.shape[1:]  # (Key_out, value_out)
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    key_in = weight_vars['key_weight'].aval.shape[0]
    value_in = weight_vars['value_weight'].aval.shape[0]
    return {
        'key_weight': jnp.zeros(
            (batch, key_in, *positions, num_hidden_state), dtype=dtype),
        'key_bias': jnp.zeros(
            (batch, *positions, num_hidden_state), dtype=dtype),
        'value_weight': jnp.zeros(
            (batch, value_in, *positions, num_hidden_state), dtype=dtype),
    }


def _outer_write_dt_to_t(hidden_dim: Any, trace: Dict[str, Any],
                         **params: Any) -> Dict[str, Any]:
    r"""Propagate :math:`\partial h / \partial y` through the retained traces.

    Because every trace keeps the full ``(key_out, value_out)`` position axes,
    the :math:`y \to` position link inside
    :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` is an elementwise
    broadcast — nothing is summed away and nothing approximated:

    .. math::

        \epsilon^{t}_{W, aij} = (\partial h / \partial y)_{ij}\,
                                \epsilon^{t-1}_{W, aij}, \qquad
        \epsilon^{t}_{b, ij}  = (\partial h / \partial y)_{ij}\,
                                \epsilon^{t-1}_{b, ij}.

    ``jnp.expand_dims(hidden_dim, axis=-3)`` inserts the input-feature axis
    before the position pair, valid in both executor contexts (batched trace
    update ``(B, K, V)`` against ``(B, A, K, V)``; batch-stripped solve
    ``(K, V)`` against ``(A, K, V)``) — the same trick as dense's ``axis=-2``.
    """
    lifted = jnp.expand_dims(hidden_dim, axis=-3)
    return {
        'key_weight': trace['key_weight'] * lifted,
        'key_bias': trace['key_bias'] * hidden_dim,
        'value_weight': trace['value_weight'] * lifted,
    }


def _outer_write_instant_drtrl(x: Any, df: Any, weights: Dict[str, Any], *,
                               key_features: int, key_scale: float,
                               key_nonlinearity: str = 'cos_rff',
                               value_nonlinearity: str = 'tanh') -> Dict[str, Any]:
    r"""Trace-structured instantaneous term, exact at the raw current-step ``x``.

    Parameters
    ----------
    x : Any
        Raw current-step packed input. The algorithm vmaps the batch and state
        axes away, so this rule sees a batch-free slice
        ``(key_features + value_features,)``.
    df : Any
        The ``y -> hidden`` tangent for this step, position-shaped
        ``(key_out, value_out)``.
    weights : dict
        Current values of the three trainable inputs.

    Returns
    -------
    dict
        Trace-structured slots: the full per-position Jacobian
        :math:`\partial h_{ij} / \partial W` with no contraction,

        .. math::

            \epsilon^{\text{inst}}_{W_k, aij} = x^{(k)}_a\,
                c\,\varphi'_k(p)_i\, \varphi_v(q)_j\, \mathrm{df}_{ij}, \qquad
            \epsilon^{\text{inst}}_{W_v, aij} = x^{(v)}_a\,
                c\,\varphi_k(p)_i\, \varphi'_v(q)_j\, \mathrm{df}_{ij},

        with ``key_bias`` sharing ``key_weight``'s postsynaptic factor at
        presynaptic factor one. Every nonlinearity is evaluated at the raw
        current-step ``x`` — the property the pp-prop path can only deliver
        through the factor hook, and the reason this trace is exact.
    """
    pre_key, pre_value, key_code, value_code = _codes(
        x, weights, key_features, key_scale, key_nonlinearity,
        value_nonlinearity)
    key_slope = key_scale * _DERIVATIVE[key_nonlinearity](pre_key)
    value_slope = _DERIVATIVE[value_nonlinearity](pre_value)
    key_df = df * jnp.einsum('...i,...j->...ij', key_slope, value_code)
    value_df = df * jnp.einsum('...i,...j->...ij', key_code, value_slope)
    x_v = u.get_mantissa(x)
    return {
        'key_weight': jnp.einsum(
            '...a,...ij->...aij', x_v[..., :key_features], key_df),
        'key_bias': key_df,
        'value_weight': jnp.einsum(
            '...a,...ij->...aij', x_v[..., key_features:], value_df),
    }


def _outer_write_solve_drtrl(dg_hidden: Any, trace: Dict[str, Any],
                             weights: Dict[str, Any],
                             **params: Any) -> Dict[str, Any]:
    r"""Contract the learning signal with each retained trace.

    Parameters
    ----------
    dg_hidden : Any
        The hidden-side learning signal, position-shaped
        ``(key_out, value_out)`` (the algorithm vmaps batch and state away).
    trace : dict
        The position-retaining traces from :func:`_outer_write_init_drtrl`.
    weights : dict
        Current weight values. Unused — every Jacobian factor already entered
        the trace at its own timestep, so no solve-time chaining remains.

    Returns
    -------
    dict
        Param-shaped gradients: the position axis each parameter does not
        have is summed here and nowhere else,

        .. math::

            \nabla W_k[a, i] = \sum_j \mathrm{dg}_{ij}\, \epsilon_{W_k, aij},
            \quad
            \nabla b_k[i] = \sum_j \mathrm{dg}_{ij}\, \epsilon_{b_k, ij},
            \quad
            \nabla W_v[a, j] = \sum_i \mathrm{dg}_{ij}\, \epsilon_{W_v, aij}.
    """
    dg = u.get_mantissa(dg_hidden)
    return {
        'key_weight': jnp.einsum(
            '...ij,...aij->...ai', dg, trace['key_weight']),
        'key_bias': jnp.einsum('...ij,...ij->...i', dg, trace['key_bias']),
        'value_weight': jnp.einsum(
            '...ij,...aij->...aj', dg, trace['value_weight']),
    }


etp_outer_write_p = register_primitive(
    'etp_outer_write',
    _outer_write_impl,
    batched=True,
    trainable_invars_fn=_outer_write_trainable_invars,
    x_invar_index=0,
)
etp_outer_write_p.register_etp_rules(
    dt_to_t=_outer_write_dt_to_t,
    xy_to_dw=_outer_write_xy_to_dw,
    init_drtrl=_outer_write_init_drtrl,
    init_pp=_outer_write_init_pp,
    pp_df_factors=_outer_write_pp_df_factors,
)
# Param-dim D-RTRL overrides (the LoRA precedent): the trace structure keeps
# the (key_out, value_out) position axes, so neither the instantaneous term
# nor the solve-time contraction can be expressed by xy_to_dw / dt_to_t alone.
# IO-dim (ES-D-RTRL) keeps using xy_to_dw with the factored df traces.
ETP_RULES_INSTANT_DRTRL[etp_outer_write_p] = _outer_write_instant_drtrl
ETP_RULES_SOLVE_DRTRL[etp_outer_write_p] = _outer_write_solve_drtrl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _require_dimensionless(value: ArrayLike, name: str) -> Any:
    """Return the mantissa of *value*, rejecting a dimensional quantity."""
    unit = u.get_unit(value)
    if not unit.is_unitless:
        raise ValueError(
            f'{name} must be dimensionless: outer_write applies '
            f'{tuple(_FORWARD)} nonlinearities to it, which are defined only '
            f'for dimensionless arguments; got unit {unit}.'
        )
    return u.get_mantissa(value)


def outer_write(
    x_key: ArrayLike,
    x_value: ArrayLike,
    *,
    key_weight: ArrayLike,
    key_bias: ArrayLike,
    value_weight: ArrayLike,
    key_scale: float = 1.0,
    key_nonlinearity: str = 'cos_rff',
    value_nonlinearity: str = 'tanh',
) -> ArrayLike:
    r"""ETP-aware fused key/value encoding and outer-product write.

    Computes :math:`y_{bij} = c\,\varphi_k(x^{(k)}W_k + b_k)_i\,
    \varphi_v(x^{(v)}W_v)_j` inside a single ETP primitive, so that the key and
    value *codings* participate in eligibility-trace learning even though the
    write itself is an outer product.

    Parameters
    ----------
    x_key : ArrayLike
        Key-side input, shape ``(batch, key_features)``.
    x_value : ArrayLike
        Value-side input, shape ``(batch, value_features)``. May be the same
        array as ``x_key``; the two are concatenated internally and split again
        by the trace rules.
    key_weight : ArrayLike
        Key projection, shape ``(key_features, key_out)``. Trainable.
    key_bias : ArrayLike
        Key phase offset, shape ``(key_out,)``. Trainable.
    value_weight : ArrayLike
        Value projection, shape ``(value_features, value_out)``. Trainable.
    key_scale : float, optional
        Scalar multiplying the key code (the random-Fourier normaliser
        :math:`\sqrt{2/W}` in the usual construction). Default ``1.0``.
    key_nonlinearity, value_nonlinearity : str, optional
        One of ``'cos_rff'`` or ``'tanh'``. Defaults ``'cos_rff'`` and
        ``'tanh'``.

    Returns
    -------
    ArrayLike
        The write matrix, shape ``(batch, key_out, value_out)``.

    Raises
    ------
    ValueError
        If an input is not batched, a feature width disagrees with its weight,
        a nonlinearity name is unknown, or an operand carries a physical unit.

    Notes
    -----
    Gating, decay, an elementwise write scale and side-validity masks belong
    *outside* this call: applied to the returned array they stay
    position-preserving on ``(batch, key_out, value_out)``, which is what keeps
    the tail from the primitive to the memory state elementwise.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> x_key = brainstate.random.randn(8, 5)
        >>> x_value = brainstate.random.randn(8, 4)
        >>> w_key = brainstate.random.randn(5, 6)
        >>> b_key = brainstate.random.randn(6)
        >>> w_value = brainstate.random.randn(4, 7)
        >>> write = braintrace.outer_write(
        ...     x_key, x_value, key_weight=w_key, key_bias=b_key,
        ...     value_weight=w_value, key_scale=0.5,
        ... )
        >>> print(write.shape)
        (8, 6, 7)
    """
    for name, nonlinearity in (('key_nonlinearity', key_nonlinearity),
                               ('value_nonlinearity', value_nonlinearity)):
        if nonlinearity not in _FORWARD:
            raise ValueError(
                f'{name} must be one of {sorted(_FORWARD)}; got {nonlinearity!r}. Set {name} to one of {sorted(_FORWARD)}; got {nonlinearity!r}.'
            )
    x_key_v = _require_dimensionless(x_key, 'x_key')
    x_value_v = _require_dimensionless(x_value, 'x_value')
    key_weight_v = _require_dimensionless(key_weight, 'key_weight')
    key_bias_v = _require_dimensionless(key_bias, 'key_bias')
    value_weight_v = _require_dimensionless(value_weight, 'value_weight')
    if x_key_v.ndim != 2 or x_value_v.ndim != 2:
        raise ValueError(
            f'outer_write requires batched 2-D inputs `(batch, features)`; got '
            f'x_key.ndim={x_key_v.ndim}, x_value.ndim={x_value_v.ndim}.'
        )
    if x_key_v.shape[0] != x_value_v.shape[0]:
        raise ValueError(
            f'x_key and x_value must share a batch axis; got '
            f'{x_key_v.shape[0]} != {x_value_v.shape[0]}.'
        )
    if key_weight_v.ndim != 2 or key_weight_v.shape[0] != x_key_v.shape[1]:
        raise ValueError(
            f'key_weight must have shape (key_features, key_out) with '
            f'key_features={x_key_v.shape[1]}; got {key_weight_v.shape}.'
        )
    if key_bias_v.shape != (key_weight_v.shape[1],):
        raise ValueError(
            f'key_bias must have shape ({key_weight_v.shape[1]},); '
            f'got {key_bias_v.shape}.'
        )
    if value_weight_v.ndim != 2 or value_weight_v.shape[0] != x_value_v.shape[1]:
        raise ValueError(
            f'value_weight must have shape (value_features, value_out) with '
            f'value_features={x_value_v.shape[1]}; got {value_weight_v.shape}.'
        )
    packed = jnp.concatenate([x_key_v, x_value_v], axis=-1)
    return etp_outer_write_p.bind(
        packed, key_weight_v, key_bias_v, value_weight_v,
        key_features=int(x_key_v.shape[1]),
        key_scale=float(key_scale),
        key_nonlinearity=key_nonlinearity,
        value_nonlinearity=value_nonlinearity,
    )
