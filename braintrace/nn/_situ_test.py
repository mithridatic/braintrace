from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import braintrace


def test_situ_glu_module_shapes_parameters_and_public_export() -> None:
    layer = braintrace.nn.SiTUGLU(2, 4, 3)
    output = layer(jnp.ones((5, 2)))

    assert output.shape == (5, 3)
    assert layer.gate_weight.value.shape == (2, 4)
    assert layer.gate_bias.value.shape == (4,)
    assert layer.up_weight.value.shape == (2, 4)
    assert layer.up_bias.value.shape == (4,)
    assert layer.output_weight.value.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(output)))
    assert braintrace.situ_glu is not None
    assert braintrace.delta_memory_update is not None


@pytest.mark.parametrize(
    ("args", "error"),
    [((0, 4, 3), ValueError), ((2, True, 3), TypeError), ((2, 4, -1), ValueError)],
)
def test_situ_glu_module_validates_sizes(args, error) -> None:
    with pytest.raises(error, match="positive integer"):
        braintrace.nn.SiTUGLU(*args)
