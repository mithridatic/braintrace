"""Tests for the V42 task-gated neural operator bank."""

from __future__ import annotations

import importlib

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_expert_model")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _config(subject, **overrides):
    values = {
        "input_width": subject.MODEL_INPUT_WIDTH,
        "encoder_width": 5,
        "hidden_width": 16,
        "expert_count": 12,
        "recurrent_layers": 2,
        "seed": 23,
    }
    values.update(overrides)
    return subject.ExpertModelConfig(**values)


def test_v42_config_fails_closed() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="input_width"):
        subject.ExpertModelConfig(input_width=8)
    with pytest.raises(ValueError, match="expert_count"):
        _config(subject, expert_count=1)
    with pytest.raises(ValueError, match="hidden_width"):
        _config(subject, hidden_width=8, expert_count=12)
    with pytest.raises(ValueError, match="architecture_version"):
        _config(subject, architecture_version="old")
    with pytest.raises(TypeError, match="input_width"):
        subject.ExpertModelConfig(input_width=True)
    with pytest.raises(ValueError, match="encoder_width"):
        _config(subject, encoder_width=0)
    with pytest.raises(TypeError, match="seed"):
        _config(subject, seed=True)
    with pytest.raises(ValueError, match="seed"):
        _config(subject, seed=-1)
    with pytest.raises(ValueError, match="recurrent_layers"):
        _config(subject, recurrent_layers=1)
    with pytest.raises(TypeError, match="config"):
        subject.TaskGatedOnlineRNN(object())


def test_v42_supports_bound_small_grid_padding_and_extra_recurrence() -> None:
    subject = _subject()
    config = subject.ExpertModelConfig(
        input_width=subject.online_input_width(10, 3),
        encoder_width=5,
        hidden_width=16,
        expert_count=12,
        recurrent_layers=3,
        max_grid_size=3,
        seed=19,
    )
    model = subject.TaskGatedOnlineRNN(config)
    hidden = jnp.zeros((1, config.hidden_width), dtype=jnp.float32)
    event = jnp.zeros((1, config.input_width), dtype=jnp.float32)

    cells = np.asarray(model._cell_logits(hidden, event))
    height, width = (np.asarray(value) for value in model._shape_logits(hidden, event))

    assert config.query_mask_slice.stop == config.query_color_slice.start
    assert cells.shape == (1, 30, 10)
    assert height.shape == (1, 30)
    assert width.shape == (1, 30)
    assert len(model.recurrent) == 3


def test_v42_gate_is_normalized_and_activation_dependent() -> None:
    subject = _subject()
    model = subject.TaskGatedOnlineRNN(_config(subject))
    hidden = jnp.zeros((2, model.config.hidden_width), dtype=jnp.float32)
    changed = hidden.at[0, 3].set(4.0)

    baseline = np.asarray(model._expert_weights(hidden))
    observed = np.asarray(model._expert_weights(changed))

    assert baseline.shape == (2, model.config.expert_count)
    assert np.allclose(np.sum(baseline, axis=-1), 1.0)
    assert np.allclose(np.sum(observed, axis=-1), 1.0)
    assert baseline[0].tobytes() != observed[0].tobytes()
    assert baseline[1].tobytes() == observed[1].tobytes()


def test_v42_mixed_cell_logits_are_local_and_gate_dependent() -> None:
    subject = _subject()
    model = subject.TaskGatedOnlineRNN(_config(subject, seed=29))
    event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    changed_event = event.at[0, model.config.query_color_slice.start + 2].set(1.0)
    hidden = jnp.zeros((1, model.config.hidden_width), dtype=jnp.float32)
    changed_gate = hidden.at[0, 4].set(6.0)

    baseline = np.asarray(model._cell_logits(hidden, event))
    local = np.asarray(model._cell_logits(hidden, changed_event))
    gated = np.asarray(model._cell_logits(changed_gate, event))

    assert baseline.shape == (1, 30, 10)
    assert local[:, 0].tobytes() != baseline[:, 0].tobytes()
    assert local[:, 1:].tobytes() == baseline[:, 1:].tobytes()
    assert gated.tobytes() != baseline.tobytes()


def test_v42_compiler_has_no_recurrent_weight_to_weight_exclusion() -> None:
    subject = _subject()
    model = subject.TaskGatedOnlineRNN(_config(subject, seed=31))
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, model.config.input_width), dtype=jnp.float32),
        batch_size=1,
        vmap=False,
        decay_or_rank=0.9,
        vjp_method="single-step",
    )

    recurrent_exclusions = [
        path for path, _ in learner.report.excluded_weights if path[0] == "recurrent"
    ]
    kinds = {item.kind.value for item in learner.report.diagnostics}

    assert recurrent_exclusions == []
    assert "relation_excluded_weight_to_weight" not in kinds


def test_v42_model_emits_finite_fixed_layout_logits() -> None:
    subject = _subject()
    model = subject.TaskGatedOnlineRNN(_config(subject, seed=37))
    brainstate.nn.init_all_states(model, batch_size=2)

    output = np.asarray(
        model(jnp.zeros((2, model.config.input_width), dtype=jnp.float32))
    )

    assert output.shape == (2, subject.OUTPUT_WIDTH)
    assert np.all(np.isfinite(output))
    paths = {str(path[0]) for path in model.states(brainstate.ParamState)}
    assert paths == {"recurrent", "cell_color_head", "height_head", "width_head"}
    with pytest.raises(ValueError, match="last dimension"):
        model(jnp.zeros((2, model.config.input_width - 1), dtype=jnp.float32))
