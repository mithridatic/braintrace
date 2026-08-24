"""Tests for the V41 continuous spatial recurrent ARC model."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_online_training import encode_online_episode
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
)


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_continuous_spatial_model"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task(test_output: tuple[tuple[int, ...], ...]) -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((1, 0),)), ArcGrid(((2, 0),))),
            ArcPair(ArcGrid(((3, 0),)), ArcGrid(((4, 0),))),
        ),
        test=(ArcPair(ArcGrid(((5, 0),)), ArcGrid(test_output)),),
        task_id="continuous-spatial-tiny",
    )


def test_v41_config_fails_closed() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="input_width"):
        subject.ContinuousSpatialConfig(input_width=8)
    with pytest.raises(ValueError, match="positive"):
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, spatial_channels=0
        )
    with pytest.raises(ValueError, match="retention"):
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, retention=1.0
        )
    with pytest.raises(ValueError, match="architecture_version"):
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, architecture_version="old"
        )


def test_v41_model_and_event_boundaries_fail_closed() -> None:
    subject = _subject()

    with pytest.raises(TypeError, match="input_width"):
        subject.ContinuousSpatialConfig(input_width=True)
    with pytest.raises(TypeError, match="seed"):
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, seed=True
        )
    with pytest.raises(ValueError, match="seed"):
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, seed=-1
        )
    with pytest.raises(TypeError, match="retention"):
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, retention=None
        )
    with pytest.raises(TypeError, match="config"):
        subject.ContinuousSpatialARC(object())
    with pytest.raises(ValueError, match="last dimension"):
        subject.validate_continuous_spatial_event(
            np.zeros((subject.MODEL_INPUT_WIDTH - 1,), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="finite"):
        subject.validate_continuous_spatial_event(
            np.full((subject.MODEL_INPUT_WIDTH,), np.nan, dtype=np.float32)
        )
    with pytest.raises(ValueError, match="last dimension"):
        subject.continuous_spatial_event_features(
            jnp.zeros((subject.MODEL_INPUT_WIDTH - 1,), dtype=jnp.float32)
        )

    model = subject.ContinuousSpatialARC(
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH, spatial_channels=1
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    with pytest.raises(ValueError, match="last dimension"):
        model(jnp.zeros((1, subject.MODEL_INPUT_WIDTH - 1), dtype=jnp.float32))


def test_v41_adapter_places_base_row_and_decode_selector_exactly() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    event = np.zeros((subject.MODEL_INPUT_WIDTH,), dtype=np.float32)
    event[row_config.valid_slice.start] = 1.0
    event[row_config.row_index_slice.start + 2] = 1.0
    event[row_config.input_mask_slice.start + 4] = 1.0
    event[row_config.input_color_slice.start + 4 * 10 + 7] = 1.0
    event[row_config.input_width] = 1.0
    event[row_config.input_width + 1 + 5] = 1.0

    features = np.asarray(subject.continuous_spatial_event_features(jnp.asarray(event)))

    assert features.shape == (30, 30, subject.SPATIAL_INPUT_CHANNELS)
    assert features[2, 4, subject.INPUT_MASK_CHANNEL] == 1.0
    assert features[2, 4, subject.INPUT_COLOR_CHANNEL.start + 7] == 1.0
    assert np.count_nonzero(features[..., subject.INPUT_COLOR_CHANNEL]) == 1
    assert np.all(features[5, :, subject.DECODE_ROW_CHANNEL] == 1.0)
    assert np.count_nonzero(features[..., subject.DECODE_ROW_CHANNEL]) == 30


def test_v41_model_uses_continuous_canvas_and_resets_exactly() -> None:
    subject = _subject()
    model = subject.ContinuousSpatialARC(
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            retention=0.8,
            seed=23,
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    event = jnp.zeros((1, subject.MODEL_INPUT_WIDTH), dtype=jnp.float32)
    event = event.at[0, subject.ROW_CONFIG.valid_slice.start].set(1.0)
    event = event.at[0, subject.ROW_CONFIG.row_index_slice.start].set(1.0)
    event = event.at[0, subject.ROW_CONFIG.input_mask_slice.start].set(1.0)
    event = event.at[0, subject.ROW_CONFIG.input_color_slice.start + 3].set(1.0)

    first = np.asarray(model(event))
    first_state = np.asarray(model.canvas.value).copy()
    second = np.asarray(model(event))
    second_state = np.asarray(model.canvas.value).copy()

    assert first.shape == (1, subject.OUTPUT_WIDTH)
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    assert np.any((first_state != 0.0) & (np.abs(first_state) < 1.0))
    assert first_state.tobytes() != second_state.tobytes()
    assert not hasattr(model, "membrane")
    assert not hasattr(model, "last_spikes")

    brainstate.nn.reset_all_states(model, batch_size=1)
    assert np.count_nonzero(np.asarray(model.canvas.value)) == 0


def test_v41_cell_head_is_spatially_local_and_checkpoint_owned() -> None:
    subject = _subject()
    model = subject.ContinuousSpatialARC(
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            seed=29,
        )
    )
    state = jnp.zeros((1, 30, 30, 2), dtype=jnp.float32)
    base = jnp.zeros((1, subject.MODEL_INPUT_WIDTH), dtype=jnp.float32)
    changed = base.at[0, subject.ROW_CONFIG.row_index_slice.start + 3].set(1.0)
    changed = changed.at[0, subject.ROW_CONFIG.input_mask_slice.start + 7].set(1.0)
    changed = changed.at[
        0, subject.ROW_CONFIG.input_color_slice.start + 7 * 10 + 6
    ].set(1.0)

    baseline = np.asarray(model._grid_logits(state, base))
    observed = np.asarray(model._grid_logits(state, changed))
    changed_cells = np.any(observed != baseline, axis=-1)

    assert changed_cells.shape == (1, 30, 30)
    assert np.count_nonzero(changed_cells) == 1
    assert changed_cells[0, 3, 7]
    paths = {str(path[0]) for path in model.states(brainstate.ParamState)}
    assert paths == {
        "input_conv",
        "recurrent_conv",
        "color_head",
        "height_head",
        "width_head",
    }


def test_v41_shape_heads_use_lossless_query_dimensions() -> None:
    subject = _subject()
    model = subject.ContinuousSpatialARC(
        subject.ContinuousSpatialConfig(
            input_width=subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            seed=31,
        )
    )
    state = jnp.zeros((1, 30, 30, 2), dtype=jnp.float32)
    first = jnp.zeros((1, subject.MODEL_INPUT_WIDTH), dtype=jnp.float32)
    first = first.at[0, subject.ROW_CONFIG.input_height_slice.start].set(1.0)
    first = first.at[0, subject.ROW_CONFIG.input_width_slice.start].set(1.0)
    second = jnp.zeros_like(first)
    second = second.at[0, subject.ROW_CONFIG.input_height_slice.start + 2].set(1.0)
    second = second.at[0, subject.ROW_CONFIG.input_width_slice.start + 3].set(1.0)

    first_logits = tuple(np.asarray(value) for value in model._shape_logits(state, first))
    second_logits = tuple(np.asarray(value) for value in model._shape_logits(state, second))

    assert first_logits[0].tobytes() != second_logits[0].tobytes()
    assert first_logits[1].tobytes() != second_logits[1].tobytes()


def test_v41_model_inputs_are_independent_of_held_out_target() -> None:
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    first = encode_online_episode(_task(((6, 0),)), 0, row_config)
    second = encode_online_episode(_task(((9, 9),)), 0, row_config)

    assert first.events.tobytes() == second.events.tobytes()
    assert first.target_rows.tobytes() != second.target_rows.tobytes()
