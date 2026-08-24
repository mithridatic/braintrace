"""Tests for the row-decoded online ARC recurrent model."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_model")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_config_rejects_schema_and_invalid_widths() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="architecture_version"):
        subject.OnlineModelConfig(input_width=8, architecture_version="old")
    with pytest.raises(TypeError, match="encoder_width"):
        subject.OnlineModelConfig(input_width=8, encoder_width=True)
    with pytest.raises(ValueError, match="positive"):
        subject.OnlineModelConfig(input_width=0)
    with pytest.raises(ValueError, match="at least two"):
        subject.OnlineModelConfig(input_width=8, recurrent_layers=1)
    with pytest.raises(TypeError, match="nonnegative"):
        subject.OnlineModelConfig(input_width=8, seed=1.5)
    with pytest.raises(TypeError, match="OnlineModelConfig"):
        subject.OnlineARCGRU(object())


def test_model_emits_one_row_and_shape_logits_per_step() -> None:
    subject = _subject()
    config = subject.OnlineModelConfig(
        input_width=12,
        encoder_width=8,
        hidden_width=16,
        recurrent_layers=2,
        seed=9,
    )
    model = subject.OnlineARCGRU(config)
    brainstate.nn.init_all_states(model, batch_size=3)

    output = np.asarray(model(jnp.ones((3, config.input_width), dtype=jnp.float32)))

    assert output.shape == (3, subject.OUTPUT_WIDTH)
    assert np.all(np.isfinite(output))
    assert model.config == config
    assert not any(
        token in name
        for name in vars(model)
        for token in ("retrieval", "associative", "template", "forest", "rule")
    )


def test_decode_instruction_changes_checkpoint_owned_output() -> None:
    subject = _subject()
    model = subject.OnlineARCGRU(
        subject.OnlineModelConfig(
            input_width=10,
            encoder_width=8,
            hidden_width=12,
            recurrent_layers=2,
            seed=11,
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    first = np.asarray(model(jnp.zeros((1, 10), dtype=jnp.float32)))
    brainstate.nn.reset_all_states(model, batch_size=1)
    event = jnp.zeros((1, 10), dtype=jnp.float32).at[0, -1].set(1.0)
    second = np.asarray(model(event))

    assert first.tobytes() != second.tobytes()
    parameter_paths = {
        ".".join(map(str, path))
        for path in model.states(brainstate.ParamState)
    }
    assert any(path.startswith("recurrent") for path in parameter_paths)
    assert any(path.startswith("row_color_head") for path in parameter_paths)
    assert any(path.startswith("height_head") for path in parameter_paths)
    assert any(path.startswith("width_head") for path in parameter_paths)


def test_split_step_logits_validates_last_dimension() -> None:
    subject = _subject()
    output = jnp.arange(subject.OUTPUT_WIDTH, dtype=jnp.float32)

    row, height, width = subject.split_step_logits(output)

    assert row.shape == (subject.MAX_GRID_SIZE, subject.COLOR_COUNT)
    assert height.shape == (subject.MAX_GRID_SIZE,)
    assert width.shape == (subject.MAX_GRID_SIZE,)
    with pytest.raises(ValueError, match="last dimension"):
        subject.split_step_logits(jnp.zeros((subject.OUTPUT_WIDTH - 1,)))


def test_model_rejects_wrong_event_width() -> None:
    subject = _subject()
    model = subject.OnlineARCGRU(
        subject.OnlineModelConfig(
            input_width=8,
            encoder_width=4,
            hidden_width=6,
            recurrent_layers=2,
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)

    with pytest.raises(ValueError, match="last dimension"):
        model(jnp.zeros((1, 7), dtype=jnp.float32))
