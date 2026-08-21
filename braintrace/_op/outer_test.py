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

"""Tests for the fused outer-product write primitive ``etp_outer_write_p``.

The load-bearing property is *one-step exactness*: the per-step factor rule
composed with ``xy_to_dw`` must reproduce ``jax.vjp`` of the primitive's own
forward, element for element. That is what licenses the claim that the only
approximation left in the trace is pp-prop's own rank-1 collapse.
"""

from collections import namedtuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace import NotSupportedError
from braintrace._op import (
    ETP_RULES_DT_TO_T,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    get_pp_df_factors,
    get_trainable_invars,
    is_batched_primitive,
    is_etp_primitive,
)
from braintrace._op.op_rule_oracle import assert_factored_rules_match_vjp
from braintrace._op.outer import etp_outer_write_p, outer_write

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


_BATCH, _KEY_IN, _VALUE_IN, _KEY_OUT, _VALUE_OUT = 3, 5, 4, 6, 7
_SCALE = 0.25


def _operands(seed=0):
    """Return ``(x_key, x_value, weights)`` with reproducible values."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    return (
        jax.random.normal(keys[0], (_BATCH, _KEY_IN)),
        jax.random.normal(keys[1], (_BATCH, _VALUE_IN)),
        {
            'key_weight': jax.random.normal(keys[2], (_KEY_IN, _KEY_OUT)),
            'key_bias': jax.random.normal(keys[3], (_KEY_OUT,)),
            'value_weight': jax.random.normal(keys[0], (_VALUE_IN, _VALUE_OUT)),
        },
    )


def _params():
    return {
        'key_features': _KEY_IN,
        'key_scale': _SCALE,
        'key_nonlinearity': 'cos_rff',
        'value_nonlinearity': 'tanh',
    }


def _reference(x_key, x_value, weights):
    key_code = _SCALE * jnp.cos(
        x_key @ weights['key_weight'] + weights['key_bias'])
    value_code = jnp.tanh(x_value @ weights['value_weight'])
    return jnp.einsum('bi,bj->bij', key_code, value_code)


class TestForwardCorrectness:

    def test_matches_reference_outer_product(self):
        x_key, x_value, weights = _operands()
        got = outer_write(
            x_key, x_value,
            key_weight=weights['key_weight'],
            key_bias=weights['key_bias'],
            value_weight=weights['value_weight'],
            key_scale=_SCALE,
        )
        assert got.shape == (_BATCH, _KEY_OUT, _VALUE_OUT)
        np.testing.assert_allclose(
            got, _reference(x_key, x_value, weights), atol=1e-6)

    def test_zero_key_scale_makes_the_write_vanish(self):
        """The scale multiplies the whole write, so zero erases it exactly."""
        x_key, x_value, weights = _operands()
        got = outer_write(
            x_key, x_value,
            key_weight=weights['key_weight'],
            key_bias=weights['key_bias'],
            value_weight=weights['value_weight'],
            key_scale=0.0,
        )
        assert jnp.all(got == 0.0)

    def test_rejects_mismatched_feature_widths(self):
        x_key, x_value, weights = _operands()
        with pytest.raises(ValueError, match='key_weight'):
            outer_write(
                x_key, x_value,
                key_weight=weights['key_weight'][:-1],
                key_bias=weights['key_bias'],
                value_weight=weights['value_weight'],
                key_scale=_SCALE,
            )

    def test_rejects_unbatched_inputs(self):
        x_key, x_value, weights = _operands()
        with pytest.raises(ValueError, match='batched'):
            outer_write(
                x_key[0], x_value[0],
                key_weight=weights['key_weight'],
                key_bias=weights['key_bias'],
                value_weight=weights['value_weight'],
                key_scale=_SCALE,
            )

    def test_rejects_unknown_nonlinearity(self):
        x_key, x_value, weights = _operands()
        with pytest.raises(ValueError, match='key_nonlinearity'):
            outer_write(
                x_key, x_value,
                key_weight=weights['key_weight'],
                key_bias=weights['key_bias'],
                value_weight=weights['value_weight'],
                key_scale=_SCALE,
                key_nonlinearity='sigmoid',
            )

    def test_rejects_dimensional_inputs(self):
        """``cos`` / ``tanh`` demand dimensionless arguments; say so loudly
        rather than silently dropping the unit."""
        x_key, x_value, weights = _operands()
        with pytest.raises(ValueError, match='dimensionless'):
            outer_write(
                x_key * u.mA, x_value,
                key_weight=weights['key_weight'],
                key_bias=weights['key_bias'],
                value_weight=weights['value_weight'],
                key_scale=_SCALE,
            )


class TestRegistration:

    def test_is_a_batched_etp_primitive(self):
        """Registered ``batched=True`` so ``vmap`` never decomposes it out of
        eligibility-trace compilation."""
        assert is_etp_primitive(etp_outer_write_p)
        assert is_batched_primitive(etp_outer_write_p)

    def test_trainable_invar_layout(self):
        assert get_trainable_invars(etp_outer_write_p, _params()) == {
            'key_weight': 1, 'key_bias': 2, 'value_weight': 3,
        }

    def test_registers_the_per_step_factor_rule(self):
        assert get_pp_df_factors(etp_outer_write_p) is not None


class TestPPRules:

    def test_init_pp_allocates_one_y_shaped_trace_per_factor_group(self):
        rule = ETP_RULES_INIT_PP[etp_outer_write_p]
        out = rule(
            _fake_var((_BATCH, _KEY_IN + _VALUE_IN)),
            _fake_var((_BATCH, _KEY_OUT, _VALUE_OUT)),
            {
                'key_weight': _fake_var((_KEY_IN, _KEY_OUT)),
                'key_bias': _fake_var((_KEY_OUT,)),
                'value_weight': _fake_var((_VALUE_IN, _VALUE_OUT)),
            },
            num_hidden_state=2,
        )
        assert set(out) == {'key', 'value'}
        for trace in out.values():
            assert trace.shape == (_BATCH, _KEY_OUT, _VALUE_OUT, 2)
            assert jnp.all(trace == 0.0)

    def test_factor_rule_returns_y_shaped_multipliers(self):
        x_key, x_value, weights = _operands()
        factors = get_pp_df_factors(etp_outer_write_p)(
            jnp.concatenate([x_key, x_value], axis=-1), weights, **_params())
        assert set(factors) == {'key', 'value'}
        for factor in factors.values():
            assert factor.shape == (_BATCH, _KEY_OUT, _VALUE_OUT)

    def test_one_step_exactness_against_vjp(self):
        """Factor rule then ``xy_to_dw`` == ``jax.vjp`` of the forward.

        This is the whole point of the per-step injection: with no trace
        history, the factored path must return the true gradient, not an
        approximation of it. A rule that instead deferred the nonlinearity to
        solve time would fail here as soon as ``x`` is not constant.
        """
        x_key, x_value, weights = _operands(seed=1)
        x_packed = jnp.concatenate([x_key, x_value], axis=-1)
        hidden_dim = jax.random.normal(
            jax.random.PRNGKey(7), (_BATCH, _KEY_OUT, _VALUE_OUT))

        assert_factored_rules_match_vjp(
            factor_rule=get_pp_df_factors(etp_outer_write_p),
            xy_rule=ETP_RULES_XY_TO_DW[etp_outer_write_p],
            impl=lambda w: _reference(x_key, x_value, w),
            x=x_packed,
            hidden_dim=hidden_dim,
            weights=weights,
            params=_params(),
        )

    def test_zero_learning_signal_gives_exactly_zero_gradient(self):
        """A gated-off write contributes a zero cotangent; the rules must not
        manufacture gradient from the factors alone."""
        x_key, x_value, weights = _operands(seed=2)
        x_packed = jnp.concatenate([x_key, x_value], axis=-1)
        factors = get_pp_df_factors(etp_outer_write_p)(
            x_packed, weights, **_params())
        zero = jnp.zeros((_BATCH, _KEY_OUT, _VALUE_OUT))
        df = {name: factor * zero for name, factor in factors.items()}
        grads = ETP_RULES_XY_TO_DW[etp_outer_write_p](
            x_packed, df, weights, **_params())
        for name, grad in grads.items():
            assert jnp.all(grad == 0.0), name

    def test_gradient_shapes_match_their_parameters(self):
        x_key, x_value, weights = _operands(seed=3)
        x_packed = jnp.concatenate([x_key, x_value], axis=-1)
        factors = get_pp_df_factors(etp_outer_write_p)(
            x_packed, weights, **_params())
        hidden_dim = jnp.ones((_BATCH, _KEY_OUT, _VALUE_OUT))
        df = {name: factor * hidden_dim for name, factor in factors.items()}
        grads = ETP_RULES_XY_TO_DW[etp_outer_write_p](
            x_packed, df, weights, **_params())
        assert set(grads) == set(weights)
        for name, grad in grads.items():
            assert grad.shape == weights[name].shape


class TestUnsupportedAlgorithms:
    """v1 is pp-prop only; D-RTRL must fail loudly, never mis-trace."""

    def test_init_drtrl_raises_not_supported(self):
        with pytest.raises(NotSupportedError, match='D-RTRL'):
            ETP_RULES_INIT_DRTRL[etp_outer_write_p](
                _fake_var((_BATCH, _KEY_IN + _VALUE_IN)),
                _fake_var((_BATCH, _KEY_OUT, _VALUE_OUT)),
                {'key_weight': _fake_var((_KEY_IN, _KEY_OUT))},
                num_hidden_state=1,
            )

    def test_dt_to_t_raises_not_supported(self):
        with pytest.raises(NotSupportedError, match='D-RTRL'):
            ETP_RULES_DT_TO_T[etp_outer_write_p](
                jnp.ones((_BATCH, _KEY_OUT, _VALUE_OUT)), {}, **_params())
