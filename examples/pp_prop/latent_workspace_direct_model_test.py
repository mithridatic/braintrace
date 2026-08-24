"""Tests for the direct recurrent ARC model."""

from __future__ import annotations

import importlib

import brainstate
import jax
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

    assert model.config.architecture_version == "direct_spatial_recurrence_v19"
    assert first_colors[0, 0, 0].tobytes() != second_colors[0, 0, 0].tobytes()


def test_local_spatial_path_is_checkpoint_owned_and_changes_cell_logits() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=27,
        )
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, 4:6, 4:6, 8].set(1.0)
    query = query.at[0, 4:6, 4:6, 10].set(1.0)

    before = np.asarray(model.decode(hidden, query)[2])
    model.local_query_refinement.weight.value = jax.tree.map(
        lambda value: jnp.ones_like(value) * 0.25,
        model.local_query_refinement.weight.value,
    )
    after = np.asarray(model.decode(hidden, query)[2])

    assert before.tobytes() != after.tobytes()


def test_local_spatial_residual_has_exact_zero_initialization() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=28,
        )
    )

    for value in model.local_query_refinement.weight.value.values():
        np.testing.assert_array_equal(np.asarray(value), np.zeros_like(value))


def test_direct_query_color_head_has_scaled_identity_and_changes_logits() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=30,
        )
    )
    parameters = model.direct_query_color_head.weight.value
    np.testing.assert_array_equal(
        np.asarray(parameters["weight"]),
        np.eye(10, dtype=np.float32) * subject.DIRECT_QUERY_COLOR_INITIAL_SCALE,
    )
    np.testing.assert_array_equal(
        np.asarray(parameters["bias"]), np.zeros((10,), dtype=np.float32)
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, 3, 4, 7].set(1.0)
    query = query.at[0, 3, 4, 10].set(1.0)
    before = np.asarray(model.decode(hidden, query)[2])
    model.direct_query_color_head.weight.value = jax.tree.map(
        jnp.zeros_like, model.direct_query_color_head.weight.value
    )
    after = np.asarray(model.decode(hidden, query)[2])

    assert before.tobytes() != after.tobytes()


def test_demonstration_pair_encoder_changes_latent_conditioned_logits() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=32,
        )
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, :3, :3, 0].set(1.0)
    query = query.at[0, :3, :3, 10].set(1.0)
    inputs = jnp.zeros((1, 10, 30, 30, 10), dtype=jnp.float32)
    inputs = inputs.at[0, 0, :3, :3, 0].set(1.0)
    first_outputs = inputs.at[0, 0, 1, 1, 0].set(0.0)
    first_outputs = first_outputs.at[0, 0, 1, 1, 2].set(1.0)
    second_outputs = inputs.at[0, 0, 1, 1, 0].set(0.0)
    second_outputs = second_outputs.at[0, 0, 1, 1, 7].set(1.0)
    demonstration_valid = jnp.asarray([[True] + [False] * 9])
    first = np.asarray(
        model.decode(
            hidden,
            query,
            demonstration_inputs=inputs,
            demonstration_outputs=first_outputs,
            demonstration_grid_valid=demonstration_valid,
        )[2]
    )
    second = np.asarray(
        model.decode(
            hidden,
            query,
            demonstration_inputs=inputs,
            demonstration_outputs=second_outputs,
            demonstration_grid_valid=demonstration_valid,
        )[2]
    )

    assert first.tobytes() != second.tobytes()


def test_v19_spatial_parameter_groups_have_finite_nonzero_gradients() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=33,
        )
    )
    hidden = jnp.ones((1, 8), dtype=jnp.float32) * 0.25
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, :4, :4, 0].set(1.0)
    query = query.at[0, :4, :4, 10].set(1.0)
    inputs = jnp.zeros((1, 10, 30, 30, 10), dtype=jnp.float32)
    inputs = inputs.at[0, 0, :4, :4, 0].set(1.0)
    inputs = inputs.at[0, 0, 1:3, 1:3, 0].set(0.0)
    inputs = inputs.at[0, 0, 1:3, 1:3, 3].set(1.0)
    outputs = jnp.zeros_like(inputs)
    outputs = outputs.at[0, 0, :4, :4, 0].set(1.0)
    outputs = outputs.at[0, 0, 1:3, 1:3, 0].set(0.0)
    outputs = outputs.at[0, 0, 1:3, 1:3, 7].set(1.0)
    demonstration_valid = jnp.asarray([[True] + [False] * 9])
    expected_prefixes = {
        "demonstration_pair_projection",
        "demonstration_pair_refinement",
        "spatial_query_projection",
        "spatial_context_projection",
        "spatial_update",
        "spatial_color_head",
    }
    weights = model.states(brainstate.ParamState).filter(
        lambda path, _: str(path[0]) in expected_prefixes
    )

    def objective() -> jax.Array:
        colors = model.decode(
            hidden,
            query,
            demonstration_inputs=inputs,
            demonstration_outputs=outputs,
            demonstration_grid_valid=demonstration_valid,
        )[2]
        return jnp.mean(jnp.square(colors))

    gradients = brainstate.transform.grad(objective, weights)()
    observed_prefixes = set()
    for path in weights:
        prefix = str(path[0])
        if prefix not in expected_prefixes:
            continue
        observed_prefixes.add(prefix)
        leaves = jax.tree.leaves(gradients[path])
        assert leaves
        norm = sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves)
        assert np.isfinite(norm)
        assert norm > 0.0, path

    assert observed_prefixes == expected_prefixes


def test_demonstration_pair_encoder_rejects_invalid_input_shape() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=36,
        )
    )

    with pytest.raises(ValueError, match="demonstration_inputs"):
        model.decode(
            jnp.zeros((1, 8), dtype=jnp.float32),
            jnp.zeros((1, 30, 30, 11), dtype=jnp.float32),
            demonstration_inputs=jnp.zeros(
                (1, 9, 30, 30, 10), dtype=jnp.float32
            ),
            demonstration_outputs=jnp.zeros(
                (1, 10, 30, 30, 10), dtype=jnp.float32
            ),
            demonstration_grid_valid=jnp.zeros((1, 10), dtype=bool),
        )


def test_spatial_recurrence_uses_compiled_four_step_scan_and_changes_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=39,
        )
    )
    hidden = jnp.ones((1, 8), dtype=jnp.float32) * 0.125
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, 2:5, 3:6, 7].set(1.0)
    query = query.at[0, 2:5, 3:6, 10].set(1.0)
    observed_lengths: list[int] = []
    real_scan = brainstate.transform.scan

    def recording_scan(function, carry, xs, *args, **kwargs):
        observed_lengths.append(int(xs.shape[0]))
        return real_scan(function, carry, xs, *args, **kwargs)

    monkeypatch.setattr(brainstate.transform, "scan", recording_scan)
    before = np.asarray(model.decode(hidden, query)[2])
    model.spatial_color_head.weight.value = jax.tree.map(
        jnp.zeros_like, model.spatial_color_head.weight.value
    )
    after = np.asarray(model.decode(hidden, query)[2])

    assert observed_lengths == [subject.SPATIAL_RECURRENCE_STEPS] * 2
    assert before.tobytes() != after.tobytes()


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


def test_temporal_summary_parameters_change_executed_outputs() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=31,
        )
    )
    events = jnp.ones((4, 1, 6), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, :2, :2, 0].set(1.0)
    query = query.at[0, :2, :2, 10].set(1.0)
    brainstate.nn.init_all_states(model, batch_size=1)
    before = tuple(np.asarray(value) for value in model.run(events, query))
    model.temporal_summary_projection.weight.value = jax.tree.map(
        lambda value: value * 0.0,
        model.temporal_summary_projection.weight.value,
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    after = tuple(np.asarray(value) for value in model.run(events, query))

    assert any(
        left.tobytes() != right.tobytes()
        for left, right in zip(before, after, strict=True)
    )


def test_demo_shape_projection_changes_executed_shape_logits() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=37,
        )
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    first_shapes = jnp.zeros((1, subject.SHAPE_FEATURE_WIDTH), dtype=jnp.float32)
    second_shapes = first_shapes.at[0, 0].set(1.0)
    second_shapes = second_shapes.at[0, 67].set(1.0)

    first = model.decode(hidden, query, first_shapes)[:2]
    second = model.decode(hidden, query, second_shapes)[:2]

    assert hasattr(model, "demo_shape_projection")
    assert hasattr(model, "query_dimension_projection")
    assert any(
        np.asarray(left).tobytes() != np.asarray(right).tobytes()
        for left, right in zip(first, second, strict=True)
    )


def test_global_query_pattern_projection_changes_executed_colors() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=41,
        )
    )
    hidden = jnp.zeros((1, 8), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, :3, :3, 5].set(1.0)
    query = query.at[0, :3, :3, 10].set(1.0)
    before = np.asarray(model.decode(hidden, query)[2])
    model.global_query_pattern_projection.weight.value = jax.tree.map(
        lambda value: value * 0.0,
        model.global_query_pattern_projection.weight.value,
    )
    after = np.asarray(model.decode(hidden, query)[2])

    assert before.tobytes() != after.tobytes()


def test_temporal_memory_attention_changes_executed_colors() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            seed=43,
        )
    )
    events = jnp.zeros((5, 1, 6), dtype=jnp.float32)
    events = events.at[:, :, 0].set(1.0)
    events = events.at[:, :, 1:].set(
        jnp.arange(25, dtype=jnp.float32).reshape(5, 1, 5) / 25.0
    )
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    query = query.at[0, :3, :3, 2].set(1.0)
    query = query.at[0, :3, :3, 10].set(1.0)
    brainstate.nn.init_all_states(model, batch_size=1)
    before = np.asarray(model.run(events, query)[2])
    model.memory_value_projection.weight.value = jax.tree.map(
        lambda value: value * 0.0,
        model.memory_value_projection.weight.value,
    )
    brainstate.nn.init_all_states(model, batch_size=1)
    after = np.asarray(model.run(events, query)[2])

    assert before.tobytes() != after.tobytes()


def test_counted_schema_contains_no_demonstration_value_retrieval_leaves() -> None:
    subject = _subject()
    model = subject.DirectARCGRU(
        subject.DirectModelConfig(
            input_width=12,
            encoder_width=8,
            hidden_width=16,
            decoder_width=10,
            seed=47,
            memory_key_indices=(2, 3, 4),
            memory_value_indices=(5, 6, 7),
        )
    )
    paths = tuple(
        ".".join(map(str, path)) for path in model.states(brainstate.ParamState)
    )
    forbidden_attributes = (
        "associative_key_projection",
        "associative_value_projection",
        "relation_color_head",
        "whole_demo_direct_color_head",
        "whole_demo_value_projection",
        "local_demo_direct_color_head",
    )
    forbidden_path_tokens = ("associative_", "relation_", "whole_demo_", "local_demo_")

    assert all(not hasattr(model, name) for name in forbidden_attributes)
    assert not any(any(token in path for token in forbidden_path_tokens) for path in paths)


@pytest.mark.parametrize(
    "changes",
    [
        {"input_width": 0},
        {"hidden_width": 0},
        {"recurrent_layers": 0},
        {"seed": -1},
        {"seed": True},
        {"architecture_version": "metric_whole_demo_relation_v14"},
        {"memory_key_indices": (2,), "memory_value_indices": ()},
        {"memory_key_indices": (12,), "memory_value_indices": (2,)},
        {
            "memory_key_indices": (2, 3),
            "memory_value_indices": (4, 5),
            "memory_key_color_block_width": 3,
        },
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
