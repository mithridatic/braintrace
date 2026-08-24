"""Tests for target-isolated row-decoded PP-prop training."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import msgspec
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
)


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_training")


def _model_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_model")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task(output_color: int = 3) -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((1, 0),)), ArcGrid(((2, 0),))),
            ArcPair(ArcGrid(((4, 0),)), ArcGrid(((5, 0),))),
        ),
        test=(ArcPair(ArcGrid(((6, 0),)), ArcGrid(((output_color, 0),))),),
        task_id="online-tiny",
    )


def test_packing_is_lossless_ordered_and_target_independent() -> None:
    subject = _subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    first = subject.encode_online_episode(_task(3), 0, config)
    changed = subject.encode_online_episode(_task(9), 0, config)
    direct = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_training"
    ).encode_direct_episode(_task(3), 0, config)
    valid_rows = direct.events[direct.events[:, config.valid_slice.start] == 1.0]
    query_rows = direct.events[
        (direct.events[:, config.valid_slice.start] == 1.0)
        & (direct.events[:, config.phase_slice.start + 1] == 1.0)
    ]
    input_start = config.max_events - len(valid_rows)

    assert first.events.tobytes() == changed.events.tobytes()
    assert first.decode_mask.tobytes() == changed.decode_mask.tobytes()
    assert first.target_rows.tobytes() != changed.target_rows.tobytes()
    assert first.class_weights.tobytes() != changed.class_weights.tobytes()
    assert first.events.shape == (config.max_events + 30, config.input_width + 31)
    assert (
        first.events[input_start : config.max_events, : config.input_width].tobytes()
        == valid_rows.tobytes()
    )
    assert np.count_nonzero(first.events[:input_start]) == 0
    assert np.count_nonzero(
        first.events[input_start : config.max_events, config.input_width :]
    ) == 0
    decode = first.events[config.max_events :]
    assert np.all(decode[:, config.input_width] == 1.0)
    assert decode[: len(query_rows), : config.input_width].tobytes() == (
        query_rows.tobytes()
    )
    assert np.all(
        decode[:, config.input_height_slice]
        == query_rows[0, config.input_height_slice]
    )
    assert np.all(
        decode[:, config.input_width_slice]
        == query_rows[0, config.input_width_slice]
    )
    assert np.count_nonzero(
        decode[len(query_rows) :, config.input_mask_slice]
    ) == 0
    assert np.count_nonzero(
        decode[len(query_rows) :, config.input_color_slice]
    ) == 0
    assert np.array_equal(decode[:, config.input_width + 1 :], np.eye(30, dtype=np.float32))
    assert first.decode_mask.sum() == 30


def test_query_replay_changes_with_query_input_but_not_target() -> None:
    subject = _subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    base = _task(3)
    changed_input = ArcTask(
        train=base.train,
        test=(ArcPair(ArcGrid(((9, 0),)), base.test[0].output),),
        task_id=base.task_id,
    )
    changed_target = _task(9)

    first = subject.encode_online_episode(base, 0, config)
    input_changed = subject.encode_online_episode(changed_input, 0, config)
    target_changed = subject.encode_online_episode(changed_target, 0, config)

    assert first.events.tobytes() != input_changed.events.tobytes()
    assert first.events.tobytes() == target_changed.events.tobytes()
    assert first.target_rows.tobytes() == input_changed.target_rows.tobytes()
    assert first.target_rows.tobytes() != target_changed.target_rows.tobytes()


def test_stack_and_repeat_keep_time_major_targets_out_of_events() -> None:
    subject = _subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episode = subject.encode_online_episode(_task(), 0, config)

    batch = subject.stack_online_episodes((episode, episode))
    chunk = subject.repeat_online_batch(batch, updates=4)

    assert batch.events.shape == (config.max_events + 30, 2, config.input_width + 31)
    assert batch.target_rows.shape == (config.max_events + 30, 2, 30)
    assert batch.target_cell_mask.shape == batch.target_rows.shape
    assert batch.class_weights.shape == (config.max_events + 30, 2, 10)
    assert batch.decode_mask.shape == (config.max_events + 30,)
    assert chunk.events.shape[0] == 4
    assert chunk.target_heights.shape == (4, config.max_events + 30, 2)


def test_target_color_weights_are_bounded_and_emphasize_rare_colors() -> None:
    subject = _subject()
    colors = np.asarray([[0, 0], [0, 7]], dtype=np.int32)
    mask = np.ones_like(colors, dtype=np.float32)

    weights = subject.target_color_weights(colors, mask)

    assert weights[0] == pytest.approx(np.sqrt(4.0 / 3.0))
    assert weights[7] == pytest.approx(2.0)
    assert weights[7] > weights[0]
    assert np.all(weights[[1, 2, 3, 4, 5, 6, 8, 9]] == 0.0)
    assert np.all((weights == 0.0) | ((weights >= 0.5) & (weights <= 4.0)))
    with pytest.raises(ValueError, match="shape"):
        subject.target_color_weights(colors, mask[:1])


def test_online_step_loss_masks_cells_and_scores_shape() -> None:
    subject = _subject()
    model_subject = _model_subject()
    row_logits = jnp.full((1, 30, 10), -10.0).at[0, :, 0].set(10.0)
    height = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    width = jnp.full((1, 30), -10.0).at[0, 1].set(10.0)
    output = jnp.concatenate((row_logits.reshape(1, -1), height, width), axis=-1)
    target_row = jnp.zeros((1, 30), dtype=jnp.int32)
    target_mask = jnp.zeros((1, 30), dtype=jnp.float32).at[0, :2].set(1.0)

    correct = subject.online_step_loss(
        output,
        target_row,
        target_mask,
        jnp.asarray([0]),
        jnp.asarray([1]),
        jnp.ones((1, 10)),
    )
    wrong_output = output.at[0, model_subject.ROW_COLOR_WIDTH :].set(0.0)
    wrong = subject.online_step_loss(
        wrong_output,
        target_row,
        target_mask,
        jnp.asarray([0]),
        jnp.asarray([1]),
        jnp.ones((1, 10)),
    )

    assert float(correct) < float(wrong)


def test_balanced_step_loss_penalizes_rare_foreground_error_more() -> None:
    subject = _subject()
    model_subject = _model_subject()
    target = jnp.asarray([[0, 0, 0, 7] + [0] * 26])
    mask = jnp.zeros((1, 30)).at[0, :4].set(1.0)
    weights = jnp.asarray([[np.sqrt(4.0 / 3.0)] + [0.0] * 6 + [2.0] + [0.0] * 2])
    row = jnp.full((1, 30, 10), -10.0)
    row = row.at[0, :, 0].set(10.0).at[0, 3, 7].set(20.0)
    height = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    width = jnp.full((1, 30), -10.0).at[0, 3].set(10.0)
    correct = jnp.concatenate((row.reshape(1, -1), height, width), axis=-1)
    background_error = correct.at[0, 0].set(-20.0).at[0, 1].set(20.0)
    rare_offset = 3 * 10
    rare_error = correct.at[0, rare_offset + 7].set(-20.0)
    rare_error = rare_error.at[0, rare_offset + 1].set(20.0)
    dimensions = jnp.asarray([0])

    background_loss = subject.online_step_loss(
        background_error, target, mask, dimensions, jnp.asarray([3]), weights
    )
    rare_loss = subject.online_step_loss(
        rare_error, target, mask, dimensions, jnp.asarray([3]), weights
    )

    assert model_subject.ROW_COLOR_WIDTH == 300
    assert float(rare_loss) > float(background_loss)


def test_v27_loss_balances_present_colors_across_the_whole_grid() -> None:
    subject = _subject()
    steps = 10
    rows = jnp.zeros((steps, 1, 30), dtype=jnp.int32)
    rows = rows.at[-1, 0, 0].set(7)
    masks = jnp.zeros((steps, 1, 30), dtype=jnp.float32).at[:, 0, :10].set(1.0)
    dimensions = jnp.zeros((steps, 1), dtype=jnp.int32)
    widths = jnp.full((steps, 1), 9, dtype=jnp.int32)
    row_zero = jnp.full((1, 30, 10), -10.0).at[0, :, 0].set(10.0)
    row_seven = jnp.full((1, 30, 10), -10.0).at[0, :, 7].set(10.0)
    height = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    width = jnp.full((1, 30), -10.0).at[0, 9].set(10.0)
    zero = jnp.concatenate((row_zero.reshape(1, -1), height, width), axis=-1)
    seven = jnp.concatenate((row_seven.reshape(1, -1), height, width), axis=-1)
    zero_outputs = jnp.broadcast_to(zero, (steps, 1, 360))
    seven_outputs = jnp.broadcast_to(seven, (steps, 1, 360))

    color_mass = subject.whole_grid_color_mass(rows, masks)
    step_loss = jax.vmap(subject.whole_grid_online_step_loss)
    zero_loss = jnp.mean(
        step_loss(zero_outputs, rows, masks, dimensions, widths, color_mass)
    )
    seven_loss = jnp.mean(
        step_loss(seven_outputs, rows, masks, dimensions, widths, color_mass)
    )

    assert float(zero_loss) == pytest.approx(float(seven_loss), rel=1e-5)
    assert float(color_mass[0, 0, 0]) == pytest.approx(30.0 / (2.0 * 99.0))
    assert float(color_mass[-1, 0, 7]) == pytest.approx(15.0)
    assert np.count_nonzero(np.asarray(color_mass[..., 1:7])) == 0
    assert np.count_nonzero(np.asarray(color_mass[..., 8:])) == 0


def test_v28_loss_weights_color_eight_times_each_shape_component() -> None:
    subject = _subject()
    row = jnp.full((1, 30, 10), -10.0).at[0, :, 0].set(10.0)
    height = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    width = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    correct = jnp.concatenate((row.reshape(1, -1), height, width), axis=-1)
    color_error = correct.at[0, 0].set(-10.0).at[0, 1].set(10.0)
    height_offset = 300
    height_error = correct.at[0, height_offset].set(-10.0)
    height_error = height_error.at[0, height_offset + 1].set(10.0)
    width_offset = 330
    width_error = correct.at[0, width_offset].set(-10.0)
    width_error = width_error.at[0, width_offset + 1].set(10.0)
    target = jnp.zeros((1, 30), dtype=jnp.int32)
    mask = jnp.zeros((1, 30), dtype=jnp.float32).at[0, 0].set(1.0)
    dimension = jnp.zeros((1,), dtype=jnp.int32)
    mass = jnp.asarray([[1.0] + [0.0] * 9])

    baseline = subject.color_dominant_whole_grid_step_loss(
        correct, target, mask, dimension, dimension, mass
    )
    color_delta = subject.color_dominant_whole_grid_step_loss(
        color_error, target, mask, dimension, dimension, mass
    ) - baseline
    height_delta = subject.color_dominant_whole_grid_step_loss(
        height_error, target, mask, dimension, dimension, mass
    ) - baseline
    width_delta = subject.color_dominant_whole_grid_step_loss(
        width_error, target, mask, dimension, dimension, mass
    ) - baseline

    assert float(color_delta / height_delta) == pytest.approx(8.0, rel=1e-5)
    assert float(color_delta / width_delta) == pytest.approx(8.0, rel=1e-5)


def test_v29_mass_balances_gate_and_present_nonzero_colors_whole_grid() -> None:
    subject = _subject()
    rows = jnp.zeros((10, 1, 30), dtype=jnp.int32)
    rows = rows.at[-1, 0, 0].set(7).at[-1, 0, 1].set(8)
    masks = jnp.zeros_like(rows, dtype=jnp.float32).at[:, 0, :10].set(1.0)

    gate_mass, color_mass = subject.hierarchical_whole_grid_mass(rows, masks)

    assert gate_mass.shape == (10, 1, 2)
    assert color_mass.shape == (10, 1, 9)
    assert float(gate_mass[0, 0, 0]) == pytest.approx(30.0 / (2.0 * 98.0))
    assert float(gate_mass[0, 0, 1]) == pytest.approx(30.0 / (2.0 * 2.0))
    assert float(color_mass[0, 0, 6]) == pytest.approx(15.0)
    assert float(color_mass[0, 0, 7]) == pytest.approx(15.0)
    assert np.count_nonzero(np.asarray(color_mass[..., :6])) == 0
    assert np.count_nonzero(np.asarray(color_mass[..., 8:])) == 0

    all_background = jnp.zeros((2, 1, 30), dtype=jnp.int32)
    background_mask = jnp.zeros_like(all_background, dtype=jnp.float32)
    background_mask = background_mask.at[:, 0, :3].set(1.0)
    background_gate, background_color = subject.hierarchical_whole_grid_mass(
        all_background, background_mask
    )
    assert float(background_gate[0, 0, 0]) == pytest.approx(5.0)
    assert float(background_gate[0, 0, 1]) == 0.0
    assert np.count_nonzero(np.asarray(background_color)) == 0


def test_v30_gate_mass_uses_square_root_group_totals() -> None:
    subject = _subject()
    rows = jnp.zeros((10, 1, 30), dtype=jnp.int32)
    rows = rows.at[-1, 0, 0].set(7).at[-1, 0, 1].set(8)
    masks = jnp.zeros_like(rows, dtype=jnp.float32).at[:, 0, :10].set(1.0)

    gate_mass, color_mass = subject.sqrt_balanced_hierarchical_mass(rows, masks)
    _, equal_color_mass = subject.hierarchical_whole_grid_mass(rows, masks)
    background_total = float(gate_mass[0, 0, 0]) * 98.0
    foreground_total = float(gate_mass[0, 0, 1]) * 2.0

    assert background_total + foreground_total == pytest.approx(30.0)
    assert background_total / foreground_total == pytest.approx(7.0)
    assert np.asarray(color_mass).tobytes() == np.asarray(equal_color_mass).tobytes()

    all_background = jnp.zeros((2, 1, 30), dtype=jnp.int32)
    background_mask = jnp.zeros_like(all_background, dtype=jnp.float32)
    background_mask = background_mask.at[:, 0, :3].set(1.0)
    background_gate, background_color = subject.sqrt_balanced_hierarchical_mass(
        all_background, background_mask
    )
    assert float(background_gate[0, 0, 0]) == pytest.approx(5.0)
    assert float(background_gate[0, 0, 1]) == 0.0
    assert np.count_nonzero(np.asarray(background_color)) == 0


def test_v31_gate_mass_uses_fourth_root_group_totals() -> None:
    subject = _subject()
    rows = jnp.zeros((10, 1, 30), dtype=jnp.int32)
    rows = rows.at[-1, 0, 0].set(7).at[-1, 0, 1].set(8)
    masks = jnp.zeros_like(rows, dtype=jnp.float32).at[:, 0, :10].set(1.0)

    gate_mass, color_mass = subject.fourth_root_balanced_hierarchical_mass(
        rows, masks
    )
    _, equal_color_mass = subject.hierarchical_whole_grid_mass(rows, masks)
    background_total = float(gate_mass[0, 0, 0]) * 98.0
    foreground_total = float(gate_mass[0, 0, 1]) * 2.0

    assert background_total + foreground_total == pytest.approx(30.0)
    assert background_total / foreground_total == pytest.approx(np.sqrt(7.0))
    assert np.asarray(color_mass).tobytes() == np.asarray(equal_color_mass).tobytes()

    all_background = jnp.zeros((2, 1, 30), dtype=jnp.int32)
    background_mask = jnp.zeros_like(all_background, dtype=jnp.float32)
    background_mask = background_mask.at[:, 0, :3].set(1.0)
    background_gate, background_color = (
        subject.fourth_root_balanced_hierarchical_mass(
            all_background, background_mask
        )
    )
    assert float(background_gate[0, 0, 0]) == pytest.approx(5.0)
    assert float(background_gate[0, 0, 1]) == 0.0
    assert np.count_nonzero(np.asarray(background_color)) == 0


def test_v29_loss_excludes_background_conditional_logits_and_binds_weights() -> None:
    subject = _subject()
    model_subject = _model_subject()
    target = jnp.zeros((1, 30), dtype=jnp.int32).at[0, 0].set(1)
    mask = jnp.zeros((1, 30), dtype=jnp.float32).at[0, :2].set(1.0)
    dimension = jnp.zeros((1,), dtype=jnp.int32)
    gate_mass = jnp.ones((1, 2), dtype=jnp.float32)
    color_mass = jnp.ones((1, 9), dtype=jnp.float32)
    row = jnp.full((1, 30, 10), -100.0)
    row = row.at[0, 0, 0].set(10.0)
    row = row.at[0, 0, 1].set(5.0).at[0, 0, 2].set(-5.0)
    row = row.at[0, 1, 0].set(-10.0)
    height = jnp.full((1, 30), -100.0).at[0, 0].set(5.0).at[0, 1].set(-5.0)
    width = height.copy()
    baseline = jnp.concatenate((row.reshape(1, -1), height, width), axis=-1)

    background_changed = baseline.at[0, 10 + 1].set(-1000.0)
    background_changed = background_changed.at[0, 10 + 9].set(1000.0)
    baseline_loss = subject.hierarchical_whole_grid_step_loss(
        baseline, target, mask, dimension, dimension, gate_mass, color_mass
    )
    background_loss = subject.hierarchical_whole_grid_step_loss(
        background_changed,
        target,
        mask,
        dimension,
        dimension,
        gate_mass,
        color_mass,
    )
    assert float(background_loss) == pytest.approx(float(baseline_loss), rel=1e-6)

    gate_error = baseline.at[0, 0].set(-10.0)
    color_error = baseline.at[0, 1].set(-5.0).at[0, 2].set(5.0)
    height_error = baseline.at[0, model_subject.ROW_COLOR_WIDTH].set(-5.0)
    height_error = height_error.at[0, model_subject.ROW_COLOR_WIDTH + 1].set(5.0)
    width_offset = model_subject.ROW_COLOR_WIDTH + model_subject.MAX_GRID_SIZE
    width_error = baseline.at[0, width_offset].set(-5.0)
    width_error = width_error.at[0, width_offset + 1].set(5.0)

    def delta(output):
        return subject.hierarchical_whole_grid_step_loss(
            output, target, mask, dimension, dimension, gate_mass, color_mass
        ) - baseline_loss

    gate_delta = float(delta(gate_error))
    color_delta = float(delta(color_error))
    height_delta = float(delta(height_error))
    width_delta = float(delta(width_error))
    assert gate_delta == pytest.approx(color_delta, rel=1e-5)
    assert gate_delta / height_delta == pytest.approx(4.0, rel=1e-5)
    assert color_delta / width_delta == pytest.approx(4.0, rel=1e-5)


def test_pp_prop_compiler_descent_pilot_moves_all_parameter_groups() -> None:
    subject = _subject()
    model_subject = _model_subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episode = subject.encode_online_episode(_task(), 0, config)
    batch = subject.stack_online_episodes((episode, episode))
    model = model_subject.OnlineARCVanillaRNN(
        model_subject.OnlineModelConfig(
            input_width=config.input_width + 31,
            max_demonstrations=config.max_demonstrations,
            max_grid_size=config.max_grid_size,
            encoder_width=8,
            hidden_width=12,
            recurrent_layers=2,
            seed=13,
        )
    )
    trainer = subject.OnlinePPPropTrainer(
        model,
        batch_size=2,
        learning_rate=0.01,
        trace_decay=2.0 ** (-1.0 / 40.0),
    )
    before = subject.parameter_arrays(model)
    recurrent_before = {
        path: b"".join(
            np.ascontiguousarray(np.asarray(leaf)).tobytes()
            for leaf in jax.tree.leaves(state.value)
        )
        for path, state in model.states(brainstate.ParamState).items()
        if path[0] == "recurrent"
    }
    residual_before = {
        (path, index): np.ascontiguousarray(np.asarray(leaf)).tobytes()
        for path, state in model.states(brainstate.ParamState).items()
        if str(path[0]).startswith("query_")
        for index, leaf in enumerate(jax.tree.leaves(state.value))
    }

    @brainstate.transform.jit
    def forward(events):
        brainstate.nn.reset_all_states(model, batch_size=2)
        return brainstate.transform.for_loop(model, events)

    candidate_before = subject.decode_hierarchical_online_outputs(
        np.asarray(forward(jnp.asarray(batch.events)))[:, 0],
        episode.decode_mask,
        (),
    )

    losses, gradient_norms = trainer.train_chunk(
        subject.repeat_online_batch(batch, updates=5)
    )
    after = subject.parameter_arrays(model)
    recurrent_after = {
        path: b"".join(
            np.ascontiguousarray(np.asarray(leaf)).tobytes()
            for leaf in jax.tree.leaves(state.value)
        )
        for path, state in model.states(brainstate.ParamState).items()
        if path[0] == "recurrent"
    }
    residual_after = {
        (path, index): np.ascontiguousarray(np.asarray(leaf)).tobytes()
        for path, state in model.states(brainstate.ParamState).items()
        if str(path[0]).startswith("query_")
        for index, leaf in enumerate(jax.tree.leaves(state.value))
    }
    candidate_after = subject.decode_hierarchical_online_outputs(
        np.asarray(forward(jnp.asarray(batch.events)))[:, 0],
        episode.decode_mask,
        (),
    )

    assert np.asarray(losses).shape == (5,)
    assert np.all(np.isfinite(np.asarray(losses)))
    assert set(gradient_norms) == {
        "recurrent",
        "row_color",
        "height",
        "width",
    }
    assert all(np.isfinite(float(value)) and float(value) > 0.0 for value in gradient_norms.values())
    assert all(before[name].tobytes() != after[name].tobytes() for name in before)
    assert recurrent_before
    assert all(
        recurrent_before[path] != recurrent_after[path] for path in recurrent_before
    )
    assert not any(
        path[0] == "recurrent" for path, _ in trainer.learner.report.excluded_weights
    )
    assert residual_before
    assert all(
        residual_before[path] != residual_after[path] for path in residual_before
    )
    assert msgspec.json.encode(candidate_before["grid"]) != msgspec.json.encode(
        candidate_after["grid"]
    )
    assert trainer.algorithm == "pp_prop"
    assert trainer.vjp_method == "single-step"
    assert trainer.loss_version == (
        "fourth_root_gate_hierarchical_color_balanced_v31"
    )


def test_sampling_is_brainstate_deterministic_and_target_isolated() -> None:
    subject = _subject()
    direct = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_training"
    )
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    catalog = direct.leave_one_out_tasks(_task())

    first = subject.sample_online_training_chunk(
        catalog,
        config,
        brainstate.random.RandomState(31),
        updates=2,
        batch_size=2,
        augment=False,
    )
    second = subject.sample_online_training_chunk(
        catalog,
        config,
        brainstate.random.RandomState(31),
        updates=2,
        batch_size=2,
        augment=False,
    )

    assert first.events.tobytes() == second.events.tobytes()
    assert first.target_rows.tobytes() == second.target_rows.tobytes()
    with pytest.raises(TypeError, match="RandomState"):
        subject.sample_online_training_chunk(
            catalog, config, object(), updates=1, batch_size=1, augment=False
        )


def test_target_free_evaluation_is_deterministic() -> None:
    subject = _subject()
    model_subject = _model_subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episodes = (
        subject.encode_online_episode(_task(3), 0, config),
        subject.encode_online_episode(_task(9), 0, config),
    )
    model = model_subject.OnlineARCVanillaRNN(
        model_subject.OnlineModelConfig(
            input_width=config.input_width + 31,
            max_demonstrations=config.max_demonstrations,
            max_grid_size=config.max_grid_size,
            encoder_width=6,
            hidden_width=8,
            recurrent_layers=2,
            seed=37,
        )
    )

    first = subject.evaluate_online_model(model, episodes, batch_size=2)
    second = subject.evaluate_online_model(model, episodes, batch_size=2)

    assert first["candidate_sha256"] == second["candidate_sha256"]
    assert first["candidates"][0]["grid"] == first["candidates"][1]["grid"]
    assert first["query_count"] == 2
    assert first["task_count"] == 1
    assert first["diagnostics"]["shape_exact_count"] >= 0
    assert sum(first["diagnostics"]["predicted_color_counts"].values()) == sum(
        candidate["height"] * candidate["width"] for candidate in first["candidates"]
    )
    assert first["candidates"][0]["parameter_dependencies"]


def test_checkpoint_roundtrip_preserves_outputs_and_digest(tmp_path) -> None:
    subject = _subject()
    model_subject = _model_subject()
    row_config = RowEventConfig(max_demonstrations=1, max_grid_size=2)
    config = model_subject.OnlineModelConfig(
        input_width=row_config.input_width + 31,
        max_demonstrations=row_config.max_demonstrations,
        max_grid_size=row_config.max_grid_size,
        encoder_width=5,
        hidden_width=9,
        recurrent_layers=2,
        seed=17,
    )
    model = model_subject.OnlineARCVanillaRNN(config)
    path = tmp_path / "online.npz"

    digest = subject.save_online_checkpoint(model, path)
    loaded, metadata = subject.load_online_checkpoint(path)

    assert metadata["parameter_sha256"] == digest
    assert subject.parameter_digest(loaded) == digest
    inputs = jnp.ones((2, config.input_width), dtype=jnp.float32)
    brainstate.nn.init_all_states(model, batch_size=2)
    original = np.asarray(model(inputs))
    brainstate.nn.init_all_states(loaded, batch_size=2)
    restored = np.asarray(loaded(inputs))
    assert original.tobytes() == restored.tobytes()


@pytest.mark.parametrize("corruption", ["missing", "shape", "nonfinite", "schema"])
def test_checkpoint_rejects_exact_schema_corruption(tmp_path, corruption: str) -> None:
    subject = _subject()
    model_subject = _model_subject()
    row_config = RowEventConfig(max_demonstrations=1, max_grid_size=2)
    model = model_subject.OnlineARCVanillaRNN(
        model_subject.OnlineModelConfig(
            input_width=row_config.input_width + 31,
            max_demonstrations=row_config.max_demonstrations,
            max_grid_size=row_config.max_grid_size,
            encoder_width=5,
            hidden_width=9,
            recurrent_layers=2,
            seed=19,
        )
    )
    path = tmp_path / "online.npz"
    subject.save_online_checkpoint(model, path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = msgspec.json.decode(bytes(payload.pop("__metadata__")))
    if corruption == "missing":
        payload.pop("leaf_0000")
    elif corruption == "shape":
        payload["leaf_0000"] = payload["leaf_0000"].reshape(1, -1)
    elif corruption == "nonfinite":
        payload["leaf_0000"] = payload["leaf_0000"].copy()
        payload["leaf_0000"].flat[0] = np.nan
    else:
        metadata["schema_version"] = 999
    payload["__metadata__"] = np.frombuffer(
        msgspec.json.encode(metadata), dtype=np.uint8
    )
    with path.open("wb") as stream:
        np.savez(stream, **payload)

    with pytest.raises(ValueError, match="checkpoint"):
        subject.load_online_checkpoint(path)


def test_hierarchical_decode_uses_gate_then_nonzero_argmax_on_fixed_steps() -> None:
    subject = _subject()
    model_subject = _model_subject()
    time = 40
    outputs = np.zeros((time, model_subject.OUTPUT_WIDTH), dtype=np.float32)
    decode_mask = np.zeros((time,), dtype=np.bool_)
    decode_mask[5:35] = True
    outputs[5:35, model_subject.ROW_COLOR_WIDTH] = 10.0
    outputs[5:35, model_subject.ROW_COLOR_WIDTH + 30 + 1] = 10.0
    outputs[5:35, : model_subject.ROW_COLOR_WIDTH : 10] = -10.0
    outputs[5, 0] = 10.0
    outputs[5, 7] = 20.0
    outputs[5, 10 + 9] = 30.0

    candidate = subject.decode_hierarchical_online_outputs(outputs, decode_mask, ())

    assert candidate["height"] == 1
    assert candidate["width"] == 2
    assert candidate["grid"] == [[7, 0]]
    assert candidate["parameter_dependencies"] == []
    assert candidate["answer_head_version"] == "hierarchical_row_decoder_v29"
