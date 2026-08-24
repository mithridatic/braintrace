"""Tests for the V22 spatial Conv-LIF ARC model."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

from examples.pp_prop.latent_workspace_task import RowEventConfig


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_spatial_model")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_spatial_config_fails_closed() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="input_width"):
        subject.SpatialModelConfig(input_width=8)
    with pytest.raises(ValueError, match="architecture_version"):
        subject.SpatialModelConfig(
            input_width=subject.MODEL_INPUT_WIDTH, architecture_version="old"
        )
    with pytest.raises(ValueError, match="positive"):
        subject.SpatialModelConfig(
            input_width=subject.MODEL_INPUT_WIDTH, spatial_channels=0
        )


def test_spatial_adapter_places_row_cells_and_decode_plane_exactly() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    event = np.zeros((row_config.input_width + 31,), dtype=np.float32)
    event[row_config.valid_slice.start] = 1.0
    event[row_config.row_index_slice.start + 2] = 1.0
    event[row_config.input_mask_slice.start + 4] = 1.0
    event[row_config.input_color_slice.start + 4 * 10 + 7] = 1.0

    features = np.asarray(subject.spatial_event_features(jnp.asarray(event)))

    assert features.shape == (30, 30, subject.SPATIAL_INPUT_CHANNELS)
    assert features[2, 4, subject.INPUT_MASK_CHANNEL] == 1.0
    assert features[2, 4, subject.INPUT_COLOR_CHANNEL.start + 7] == 1.0
    assert np.count_nonzero(features[..., subject.INPUT_COLOR_CHANNEL]) == 1

    decode = np.zeros_like(event)
    decode[row_config.input_width] = 1.0
    decode[row_config.input_width + 1 + 5] = 1.0
    decode_features = np.asarray(subject.spatial_event_features(jnp.asarray(decode)))
    assert np.all(decode_features[5, :, subject.DECODE_ROW_CHANNEL] == 1.0)
    assert np.count_nonzero(decode_features[..., subject.DECODE_ROW_CHANNEL]) == 30


def test_spatial_adapter_rejects_malformed_or_nonfinite_events() -> None:
    subject = _subject()
    with pytest.raises(ValueError, match="last dimension"):
        subject.spatial_event_features(jnp.zeros((subject.MODEL_INPUT_WIDTH - 1,)))
    with pytest.raises(ValueError, match="finite"):
        subject.validate_spatial_event(np.full((subject.MODEL_INPUT_WIDTH,), np.nan))


def test_conv_lif_model_emits_row_and_shape_logits_from_neural_state() -> None:
    subject = _subject()
    model = subject.SpatialARCConvLIF(
        subject.SpatialModelConfig(
            input_width=subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            seed=23,
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    event = jnp.zeros((1, subject.MODEL_INPUT_WIDTH), dtype=jnp.float32)
    event = event.at[0, subject.BASE_INPUT_WIDTH].set(1.0)
    event = event.at[0, subject.BASE_INPUT_WIDTH + 1].set(1.0)

    with brainstate.environ.context(dt=1.0 * u.ms):
        first = np.asarray(model(event))
        first_voltage = np.asarray(u.get_mantissa(model.membrane.value)).copy()
        second = np.asarray(model(event))
        second_voltage = np.asarray(u.get_mantissa(model.membrane.value)).copy()

    assert first.shape == (1, subject.OUTPUT_WIDTH)
    assert np.all(np.isfinite(first))
    assert first_voltage.tobytes() != second_voltage.tobytes()
    paths = {".".join(map(str, path)) for path in model.states(brainstate.ParamState)}
    assert any(path.startswith("input_conv") for path in paths)
    assert any(path.startswith("recurrent_conv") for path in paths)
    assert any(path.startswith("color_head") for path in paths)
