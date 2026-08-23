"""Tests for the direct recurrent ARC model."""

from __future__ import annotations

import importlib

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_direct_model")


def test_direct_gru_emits_complete_shape_and_cell_logits() -> None:
    subject = _subject()
    config = subject.DirectModelConfig(
        input_width=12,
        encoder_width=8,
        hidden_width=16,
        decoder_width=12,
        recurrent_layers=2,
        seed=7,
    )
    model = subject.DirectARCGRU(config)
    events = jnp.zeros((5, 2, 12), dtype=jnp.float32)
    events = events.at[:, :, 0].set(1.0)
    query = jnp.zeros((2, 30, 30, 11), dtype=jnp.float32)
    query = query.at[:, 0, 0, 3].set(1.0)
    query = query.at[:, 0, 0, 10].set(1.0)
    brainstate.nn.init_all_states(model, batch_size=2)

    height, width, colors = model.run(events, query)

    assert height.shape == (2, 30)
    assert width.shape == (2, 30)
    assert colors.shape == (2, 30, 30, 10)
    assert np.all(np.isfinite(np.asarray(height)))
    assert np.all(np.isfinite(np.asarray(width)))
    assert np.all(np.isfinite(np.asarray(colors)))


def test_direct_gru_is_seed_deterministic() -> None:
    subject = _subject()
    config = subject.DirectModelConfig(
        input_width=6,
        encoder_width=4,
        hidden_width=8,
        decoder_width=6,
        seed=13,
    )
    events = jnp.ones((3, 1, 6), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)

    first_model = subject.DirectARCGRU(config)
    brainstate.nn.init_all_states(first_model, batch_size=1)
    first = tuple(np.asarray(value) for value in first_model.run(events, query))
    second_model = subject.DirectARCGRU(config)
    brainstate.nn.init_all_states(second_model, batch_size=1)
    second = tuple(np.asarray(value) for value in second_model.run(events, query))

    for left, right in zip(first, second, strict=True):
        assert left.tobytes() == right.tobytes()


def test_cell_decoder_has_checkpoint_owned_cross_spatial_dependence() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=23,
        )
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    first_query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    first_query = first_query.at[0, 0, 0, 1].set(1.0)
    first_query = first_query.at[0, 0, 0, 10].set(1.0)
    first_query = first_query.at[0, 7, 9, 2].set(1.0)
    first_query = first_query.at[0, 7, 9, 10].set(1.0)
    second_query = first_query.at[0, 7, 9, 2].set(0.0)
    second_query = second_query.at[0, 7, 9, 4].set(1.0)

    first_colors = np.asarray(model.decode(hidden, first_query)[2])
    second_colors = np.asarray(model.decode(hidden, second_query)[2])

    assert (
        model.config.architecture_version
        == "query_shape_conditioned_attention_v3"
    )
    assert first_colors[0, 0, 0].tobytes() != second_colors[0, 0, 0].tobytes()


def test_shape_heads_have_checkpoint_owned_query_shape_dependence() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=29,
        )
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    one_cell = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    one_cell = one_cell.at[0, 0, 0, 0].set(1.0)
    one_cell = one_cell.at[0, 0, 0, 10].set(1.0)
    larger = one_cell.at[0, :2, :3, 0].set(1.0)
    larger = larger.at[0, :2, :3, 10].set(1.0)

    first_height, first_width, _ = model.decode(hidden, one_cell)
    second_height, second_width, _ = model.decode(hidden, larger)

    assert np.asarray(first_height).tobytes() != np.asarray(second_height).tobytes()
    assert np.asarray(first_width).tobytes() != np.asarray(second_width).tobytes()


@pytest.mark.parametrize(
    "changes",
    [
        {"input_width": 0},
        {"hidden_width": 0},
        {"recurrent_layers": 0},
        {"seed": -1},
        {"seed": True},
        {"architecture_version": "cross_spatial_attention_v2"},
    ],
)
def test_direct_model_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    subject = _subject()
    values = {
        "input_width": 12,
        "encoder_width": 8,
        "hidden_width": 16,
        "decoder_width": 12,
        "recurrent_layers": 1,
        "seed": 7,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        subject.DirectModelConfig(**values)
