from __future__ import annotations

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace


def test_module_owns_parameters_and_zero_gate_initialization() -> None:
    baseline = jnp.asarray([[0.5, -0.2], [0.3, 0.7], [-0.4, 0.1]])
    module = braintrace.nn.GatedProjection(
        3,
        2,
        2,
        output_weight_init=2.0 * baseline,
        name="gated",
    )

    assert module.name == "gated"
    assert isinstance(module.gate_weight, brainstate.ParamState)
    assert isinstance(module.gate_bias, brainstate.ParamState)
    assert isinstance(module.output_weight, brainstate.ParamState)
    np.testing.assert_array_equal(module.gate_weight.value, 0.0)
    np.testing.assert_array_equal(module.gate_bias.value, 0.0)
    values = jnp.asarray([[1.0, -2.0, 0.5]])
    gate_input = jnp.asarray([[0.2, -0.4]])
    np.testing.assert_allclose(module(values, gate_input), values @ baseline, atol=1e-7)
    np.testing.assert_array_equal(module.gate_activation(gate_input), 0.5)


@pytest.mark.parametrize(
    ("value_size", "gate_size", "output_size"),
    [(True, 2, 3), (2, False, 3), (2, 3, 0)],
)
def test_module_validates_dimensions(value_size, gate_size, output_size) -> None:
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        braintrace.nn.GatedProjection(value_size, gate_size, output_size)


def test_public_exports() -> None:
    assert "gated_projection" in braintrace.__all__
    assert "GatedProjection" in braintrace.nn.__all__
