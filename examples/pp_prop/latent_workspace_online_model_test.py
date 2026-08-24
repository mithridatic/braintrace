"""Tests for the row-decoded online ARC recurrent model."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_model")


def _config(subject, **overrides):
    max_demonstrations = overrides.pop("max_demonstrations", 1)
    max_grid_size = overrides.pop("max_grid_size", 2)
    return subject.OnlineModelConfig(
        input_width=subject.online_input_width(max_demonstrations, max_grid_size),
        max_demonstrations=max_demonstrations,
        max_grid_size=max_grid_size,
        **overrides,
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_config_rejects_schema_and_invalid_widths() -> None:
    subject = _subject()

    assert subject.ARCHITECTURE_VERSION == "online_task_patch_decoder_v39"

    with pytest.raises(ValueError, match="architecture_version"):
        _config(subject, architecture_version="old")
    with pytest.raises(TypeError, match="encoder_width"):
        _config(subject, encoder_width=True)
    with pytest.raises(ValueError, match="positive"):
        subject.OnlineModelConfig(input_width=0)
    with pytest.raises(ValueError, match="at least two"):
        _config(subject, recurrent_layers=1)
    with pytest.raises(TypeError, match="nonnegative"):
        _config(subject, seed=1.5)
    with pytest.raises(ValueError, match="layout"):
        subject.OnlineModelConfig(
            input_width=95, max_demonstrations=1, max_grid_size=2
        )
    with pytest.raises(TypeError, match="OnlineModelConfig"):
        subject.OnlineARCVanillaRNN(object())


def test_model_emits_one_row_and_shape_logits_per_step() -> None:
    subject = _subject()
    config = _config(
        subject,
        encoder_width=8,
        hidden_width=16,
        recurrent_layers=2,
        seed=9,
    )
    model = subject.OnlineARCVanillaRNN(config)
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
    model = subject.OnlineARCVanillaRNN(
        _config(
            subject,
            encoder_width=8,
            hidden_width=12,
            recurrent_layers=2,
            seed=11,
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    first = np.asarray(
        model(jnp.zeros((1, model.config.input_width), dtype=jnp.float32))
    )
    brainstate.nn.reset_all_states(model, batch_size=1)
    event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    event = event.at[0, -1].set(1.0)
    second = np.asarray(model(event))

    assert first.tobytes() != second.tobytes()
    parameter_paths = {
        ".".join(map(str, path))
        for path in model.states(brainstate.ParamState)
    }
    assert any(path.startswith("recurrent") for path in parameter_paths)
    assert any(path.startswith("cell_color_head") for path in parameter_paths)
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


def test_query_replay_slices_bind_mask_before_colours() -> None:
    subject = _subject()
    config = _config(subject, max_demonstrations=2, max_grid_size=3)

    assert config.query_mask_slice.stop == config.query_color_slice.start
    assert config.query_mask_slice.stop - config.query_mask_slice.start == 3
    assert config.query_color_slice.stop - config.query_color_slice.start == 30


def test_model_rejects_wrong_event_width() -> None:
    subject = _subject()
    model = subject.OnlineARCVanillaRNN(
        _config(
            subject,
            encoder_width=4,
            hidden_width=6,
            recurrent_layers=2,
        )
    )
    brainstate.nn.init_all_states(model, batch_size=1)

    with pytest.raises(ValueError, match="last dimension"):
        model(jnp.zeros((1, model.config.input_width - 1), dtype=jnp.float32))


def test_v32_removes_recurrent_weight_to_weight_exclusions() -> None:
    subject = _subject()
    legacy = braintrace.nn.GRUCell(3, 4)
    legacy_learner = braintrace.compile(
        legacy,
        braintrace.pp_prop,
        jnp.zeros((1, 3), dtype=jnp.float32),
        batch_size=1,
        vmap=False,
        decay_or_rank=0.9,
        vjp_method="single-step",
    )
    legacy_kinds = {item.kind.value for item in legacy_learner.report.diagnostics}
    assert "relation_excluded_weight_to_weight" in legacy_kinds

    model = subject.OnlineARCVanillaRNN(
        _config(
            subject,
            encoder_width=5,
            hidden_width=9,
            recurrent_layers=2,
            seed=23,
        )
    )
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


def test_shared_cell_decoder_is_local_and_task_conditioned() -> None:
    subject = _subject()
    config = _config(
        subject,
        encoder_width=5,
        hidden_width=9,
        recurrent_layers=2,
        seed=29,
    )
    model = subject.OnlineARCVanillaRNN(config)
    event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    changed_event = event.at[0, config.query_color_slice.start + 1].set(1.0)
    hidden_a = jnp.zeros((1, config.hidden_width), dtype=jnp.float32)
    hidden_b = jnp.full((1, config.hidden_width), 0.5, dtype=jnp.float32)

    base_a = np.asarray(model._cell_logits(hidden_a, event))
    changed_a = np.asarray(model._cell_logits(hidden_a, changed_event))
    base_b = np.asarray(model._cell_logits(hidden_b, event))
    changed_b = np.asarray(model._cell_logits(hidden_b, changed_event))

    assert base_a.shape == (1, subject.MAX_GRID_SIZE, subject.COLOR_COUNT)
    assert changed_a[:, 0].tobytes() != base_a[:, 0].tobytes()
    assert changed_a[:, 1:].tobytes() == base_a[:, 1:].tobytes()
    assert not np.allclose(changed_a[:, 0] - base_a[:, 0], changed_b[:, 0] - base_b[:, 0])
    parameter_paths = {path[0] for path in model.states(brainstate.ParamState)}
    assert {
        "cell_color_head",
        "height_head",
        "width_head",
    }.issubset(parameter_paths)


def test_query_patch_changes_only_one_cell_and_interacts_with_context() -> None:
    subject = _subject()
    config = _config(
        subject,
        encoder_width=5,
        hidden_width=9,
        recurrent_layers=2,
        seed=37,
    )
    model = subject.OnlineARCVanillaRNN(config)
    event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    changed_event = event.at[0, config.decode_patch_slice.start + 6].set(1.0)
    hidden_a = jnp.zeros((1, config.hidden_width), dtype=jnp.float32)
    hidden_b = jnp.full((1, config.hidden_width), 0.5, dtype=jnp.float32)

    base_a = np.asarray(model._cell_logits(hidden_a, event))
    changed_a = np.asarray(model._cell_logits(hidden_a, changed_event))
    base_b = np.asarray(model._cell_logits(hidden_b, event))
    changed_b = np.asarray(model._cell_logits(hidden_b, changed_event))

    assert config.decode_patch_slice.stop == config.input_width
    assert changed_a[:, 0].tobytes() != base_a[:, 0].tobytes()
    assert changed_a[:, 1:].tobytes() == base_a[:, 1:].tobytes()
    assert not np.allclose(
        changed_a[:, 0] - base_a[:, 0], changed_b[:, 0] - base_b[:, 0]
    )


def test_shape_decoder_interacts_query_dimensions_with_task_context() -> None:
    subject = _subject()
    config = _config(
        subject,
        encoder_width=5,
        hidden_width=9,
        recurrent_layers=2,
        seed=31,
    )
    model = subject.OnlineARCVanillaRNN(config)
    event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    changed_event = event.at[0, config.query_height_slice.start].set(1.0)
    changed_event = changed_event.at[0, config.query_width_slice.start + 1].set(1.0)
    hidden_a = jnp.zeros((1, config.hidden_width), dtype=jnp.float32)
    hidden_b = jnp.full((1, config.hidden_width), 0.5, dtype=jnp.float32)

    base_a = model._shape_logits(hidden_a, event)
    changed_a = model._shape_logits(hidden_a, changed_event)
    base_b = model._shape_logits(hidden_b, event)
    changed_b = model._shape_logits(hidden_b, changed_event)

    for first, second in zip(base_a, changed_a, strict=True):
        assert np.asarray(first).tobytes() != np.asarray(second).tobytes()
    for first, second, third, fourth in zip(
        base_a, changed_a, base_b, changed_b, strict=True
    ):
        assert not np.allclose(
            np.asarray(second - first), np.asarray(fourth - third)
        )
