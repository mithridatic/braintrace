"""Tests for the V46 paired spatial recurrent ARC model."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_paired_spatial_model"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _event(subject, *, phase: str | None, color: int = 3):
    event = jnp.zeros(
        (1, subject.MAX_GRID_SIZE, subject.MAX_GRID_SIZE, subject.EVENT_CHANNELS),
        dtype=jnp.float32,
    )
    event = event.at[0, 14, 14, subject.INPUT_COLOR_CHANNEL].set(color)
    event = event.at[0, 14, 14, subject.INPUT_MASK_CHANNEL].set(1.0)
    if phase == "demo":
        event = event.at[..., subject.DEMO_PHASE_CHANNEL].set(1.0)
        event = event.at[0, 14, 14, subject.OUTPUT_COLOR_CHANNEL].set(7)
        event = event.at[0, 14, 14, subject.OUTPUT_MASK_CHANNEL].set(1.0)
    elif phase == "query":
        event = event.at[..., subject.QUERY_PHASE_CHANNEL].set(1.0)
    elif phase is not None:
        raise ValueError(phase)
    return event


def _zero_conv(layer) -> None:
    layer.weight.value = {
        name: jnp.zeros_like(value)
        for name, value in layer.weight.value.items()
    }


def test_v46_config_and_event_boundaries_fail_closed() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="spatial_channels"):
        subject.PairedSpatialConfig(spatial_channels=0)
    with pytest.raises(ValueError, match="retention"):
        subject.PairedSpatialConfig(retention=1.0)
    with pytest.raises(TypeError, match="seed"):
        subject.PairedSpatialConfig(seed=True)
    with pytest.raises(ValueError, match="refinement_steps"):
        subject.PairedSpatialConfig(refinement_steps=0)
    with pytest.raises(ValueError, match="architecture_version"):
        subject.PairedSpatialConfig(architecture_version="old")
    with pytest.raises(TypeError, match="config"):
        subject.PairedSpatialARC(object())
    with pytest.raises(ValueError, match="shape"):
        subject.validate_paired_spatial_event(
            np.zeros((30, 30, subject.EVENT_CHANNELS - 1), dtype=np.float32)
        )
    malformed = np.zeros((30, 30, subject.EVENT_CHANNELS), dtype=np.float32)
    malformed[..., subject.INPUT_COLOR_CHANNEL] = 10
    with pytest.raises(ValueError, match="input colour"):
        subject.validate_paired_spatial_event(malformed)
    malformed.fill(0)
    malformed[..., subject.DEMO_PHASE_CHANNEL] = 1.0
    malformed[..., subject.QUERY_PHASE_CHANNEL] = 1.0
    with pytest.raises(ValueError, match="exclusive"):
        subject.validate_paired_spatial_event(malformed)


def test_v46_demo_query_and_padding_phases_gate_separate_canvases() -> None:
    subject = _subject()
    model = subject.PairedSpatialARC(
        subject.PairedSpatialConfig(spatial_channels=2, refinement_steps=2, seed=23)
    )
    brainstate.nn.init_all_states(model, batch_size=1)

    demo_output = np.asarray(model(_event(subject, phase="demo")))
    demo_after_demo = np.asarray(model.demo_canvas.value).copy()
    query_after_demo = np.asarray(model.query_canvas.value).copy()
    query_output = np.asarray(model(_event(subject, phase="query")))
    demo_after_query = np.asarray(model.demo_canvas.value).copy()
    query_after_query = np.asarray(model.query_canvas.value).copy()
    model(_event(subject, phase=None))

    assert demo_output.shape == (1, subject.OUTPUT_WIDTH)
    assert query_output.shape == (1, subject.OUTPUT_WIDTH)
    assert np.any(demo_after_demo != 0.0)
    assert np.count_nonzero(query_after_demo) == 0
    assert demo_after_demo.tobytes() == demo_after_query.tobytes()
    assert np.any(query_after_query != 0.0)
    assert np.asarray(model.demo_canvas.value).tobytes() == demo_after_query.tobytes()
    assert np.asarray(model.query_canvas.value).tobytes() == query_after_query.tobytes()

    brainstate.nn.reset_all_states(model, batch_size=1)
    assert np.count_nonzero(np.asarray(model.demo_canvas.value)) == 0
    assert np.count_nonzero(np.asarray(model.query_canvas.value)) == 0


def test_v46_query_recurrence_expands_spatial_receptive_field() -> None:
    subject = _subject()
    model = subject.PairedSpatialARC(
        subject.PairedSpatialConfig(spatial_channels=1, refinement_steps=2, seed=29)
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    _zero_conv(model.query_input_conv)
    model.query_recurrent_conv.weight.value = {
        name: (
            jnp.ones_like(value)
            if name == "weight"
            else jnp.zeros_like(value)
        )
        for name, value in model.query_recurrent_conv.weight.value.items()
    }
    model.query_canvas.value = model.query_canvas.value.at[0, 14, 14, 0].set(0.5)
    query = _event(subject, phase="query", color=0)

    model(query)
    first = np.asarray(model.query_canvas.value).copy()
    model(query)
    second = np.asarray(model.query_canvas.value).copy()

    assert first[0, 14, 16, 0] == 0.0
    assert second[0, 14, 16, 0] != 0.0
    assert np.count_nonzero(second) > np.count_nonzero(first)


def test_v46_colour_logits_depend_on_both_recurrent_canvases() -> None:
    subject = _subject()
    model = subject.PairedSpatialARC(
        subject.PairedSpatialConfig(spatial_channels=2, seed=31)
    )
    event = _event(subject, phase="query")
    zeros = jnp.zeros((1, 30, 30, 2), dtype=jnp.float32)
    demo = zeros.at[0, 4, 5, 0].set(1.0)
    query = zeros.at[0, 4, 5, 1].set(1.0)

    baseline = np.asarray(model._color_logits(zeros, zeros, event))
    demo_changed = np.asarray(model._color_logits(demo, zeros, event))
    query_changed = np.asarray(model._color_logits(zeros, query, event))

    assert baseline.shape == (1, 30, 30, subject.COLOR_COUNT)
    assert baseline.tobytes() != demo_changed.tobytes()
    assert baseline.tobytes() != query_changed.tobytes()
    assert demo_changed.tobytes() != query_changed.tobytes()


def test_v46_every_answer_path_operator_is_checkpoint_owned() -> None:
    subject = _subject()
    model = subject.PairedSpatialARC(
        subject.PairedSpatialConfig(spatial_channels=2, seed=37)
    )

    paths = {str(path[0]) for path in model.states(brainstate.ParamState)}

    assert paths == {
        "demo_input_conv",
        "demo_recurrent_conv",
        "query_input_conv",
        "query_recurrent_conv",
        "color_head",
        "height_head",
        "width_head",
    }
    assert model.answer_head_version == "paired_spatial_grid_decoder_v46"
    assert model.proposal_source == "paired_spatial_model_logits"
