"""Tests for compact target-free ARC adaptation banks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import brainstate
import braintools
import jax
import numpy as np
import pytest

try:
    from examples.pp_prop import latent_workspace_arc_adaptation as arc_adaptation
    from examples.pp_prop.latent_workspace_arc_adaptation import (
        ARC_ADAPTATION_CHECKPOINTS,
        ArcAdaptationFoldInputs,
        ArcContextStream,
        ArcTaskBankAdaptationResult,
        ArcTargetFreeTaskBank,
        build_arc_target_free_task_bank,
        compile_arc_task_local_adaptation_runner,
        synthesize_arc_evaluation_context,
        synthesize_arc_loo_context,
    )
    from examples.pp_prop.latent_workspace_adaptation import (
        snapshot_parameters,
    )
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        compile_pp_prop,
    )
    from examples.pp_prop.latent_workspace_refinement import RowRefinementLayout
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        associative_memory_feature_indices,
        encode_arc_query_episode,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
    )
except ImportError:
    import latent_workspace_arc_adaptation as arc_adaptation
    from latent_workspace_arc_adaptation import (
        ARC_ADAPTATION_CHECKPOINTS,
        ArcAdaptationFoldInputs,
        ArcContextStream,
        ArcTaskBankAdaptationResult,
        ArcTargetFreeTaskBank,
        build_arc_target_free_task_bank,
        compile_arc_task_local_adaptation_runner,
        synthesize_arc_evaluation_context,
        synthesize_arc_loo_context,
    )
    from latent_workspace_adaptation import snapshot_parameters
    from latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        compile_pp_prop,
    )
    from latent_workspace_refinement import RowRefinementLayout
    from latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        associative_memory_feature_indices,
        encode_arc_query_episode,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
    )


def _pair(input_color: int, output_color: int) -> ArcPair:
    return ArcPair(ArcGrid(((input_color,),)), ArcGrid(((output_color,),)))


def _task(*, test_outputs: tuple[int, ...] = (7, 8)) -> ArcTask:
    return ArcTask(
        train=(_pair(1, 4), _pair(2, 5), _pair(3, 6)),
        test=tuple(_pair(9 - index, color) for index, color in enumerate(test_outputs)),
        task_id="visible-task",
    )


def _tree_arrays(value: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(leaf) for leaf in jax.tree.leaves(value))


def _assert_parameter_snapshots_equal(left: Any, right: Any) -> None:
    assert tuple(record.path for record in left.records) == tuple(
        record.path for record in right.records
    )
    for left_record, right_record in zip(left.records, right.records, strict=True):
        assert left_record.tree_structure == right_record.tree_structure
        for left_leaf, right_leaf in zip(
            left_record.leaves, right_record.leaves, strict=True
        ):
            np.testing.assert_array_equal(left_leaf, right_leaf)


def _task_fold_inputs(
    bank: ArcTargetFreeTaskBank, task_index: int = 0
) -> ArcAdaptationFoldInputs:
    return jax.tree.map(lambda value: value[task_index], bank.fold_inputs)


def _shaped_task() -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(
                ArcGrid(((1, 2, 3), (4, 5, 6))),
                ArcGrid(((6, 5), (4, 3), (2, 1))),
            ),
            ArcPair(
                ArcGrid(((7,), (8,), (9,))),
                ArcGrid(((0, 1, 2),)),
            ),
            ArcPair(
                ArcGrid(((3, 3),)),
                ArcGrid(((4,), (4,))),
            ),
        ),
        test=(ArcPair(ArcGrid(((9, 8), (7, 6))), ArcGrid(((1,),))),),
        task_id="shaped",
    )


def _expected_context_advances(encoded: Any, rows: RowEventConfig) -> np.ndarray:
    valid = encoded.events[:, rows.valid_slice.start] > 0.5
    shared_width = max(
        (int(valid[start:stop].sum()) for start, stop in encoded.demonstration_spans),
        default=0,
    )
    advances = np.zeros((encoded.events.shape[0],), dtype=np.bool_)
    for start, _stop in encoded.demonstration_spans:
        advances[start : start + shared_width] = True
    advances[encoded.query_start : encoded.query_stop] = True
    return advances


def _row_layout(rows: RowEventConfig) -> RowRefinementLayout:
    return RowRefinementLayout(
        input_width=rows.input_width,
        event_valid_index=rows.valid_slice.start,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        normalized_start=rows.normalized_slice.start,
        row_index_start=rows.row_index_slice.start,
        input_height_start=rows.input_height_slice.start,
        input_width_start=rows.input_width_slice.start,
        output_height_start=rows.output_height_slice.start,
        output_width_start=rows.output_width_slice.start,
        input_mask_start=rows.input_mask_slice.start,
        output_mask_start=rows.output_mask_slice.start,
        input_color_start=rows.input_color_slice.start,
        output_color_start=rows.output_color_slice.start,
    )


def _tiny_row_model(rows: RowEventConfig) -> LatentWorkspaceModel:
    memory = associative_memory_feature_indices(rows)
    config = ModelConfig(
        input_width=rows.input_width,
        batch_size=1,
        neuron_count=64,
        recurrent_edges=64,
        max_latent_steps=60,
        readout_width=8,
        color_rank=2,
        context_memory_width=2,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        memory_key_indices=memory.key_indices,
        memory_value_indices=memory.value_indices,
        decoder_mode="row_refinement",
        refinement_steps=60,
        refinement_layout=_row_layout(rows),
        event_valid_index=rows.valid_slice.start,
        seed=91,
    )
    with brainstate.random.seed_context(91):
        return LatentWorkspaceModel(config)


def test_arc_bank_has_no_official_target_or_fingerprint_field() -> None:
    assert ArcTargetFreeTaskBank._fields == (
        "fold_inputs",
        "query_inputs",
        "query_shapes",
        "query_valid",
        "checkpoint_indices",
        "task_ordinals",
        "query_ordinals",
    )
    assert ArcAdaptationFoldInputs._fields == (
        "demonstration_inputs",
        "demonstration_input_shapes",
        "demonstration_outputs",
        "demonstration_output_shapes",
        "demonstration_valid",
        "held_out_demonstration_index",
        "fold_valid",
    )
    assert not any(
        "fingerprint" in field or "official" in field
        for field in ArcTargetFreeTaskBank._fields
    )


def test_official_target_mutation_is_byte_identical_at_prediction_boundary() -> None:
    rows = RowEventConfig(max_demonstrations=3)
    first = build_arc_target_free_task_bank((_task(test_outputs=(7, 8)),), rows)
    second = build_arc_target_free_task_bank((_task(test_outputs=(0, 1)),), rows)

    for left, right in zip(_tree_arrays(first), _tree_arrays(second), strict=True):
        np.testing.assert_array_equal(left, right)


def test_fold_order_and_demo_targets_are_only_in_fold_inputs() -> None:
    built = build_arc_target_free_task_bank(
        (_task(),), RowEventConfig(max_demonstrations=3)
    )
    folds = built.fold_inputs
    np.testing.assert_array_equal(folds.fold_valid[0], [True, True, True])
    np.testing.assert_array_equal(folds.held_out_demonstration_index[0], [0, 1, 2])
    np.testing.assert_array_equal(folds.demonstration_inputs[0, :, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(folds.demonstration_outputs[0, :, 0, 0], [4, 5, 6])
    np.testing.assert_array_equal(
        folds.demonstration_input_shapes[0], [[1, 1], [1, 1], [1, 1]]
    )
    np.testing.assert_array_equal(
        folds.demonstration_output_shapes[0], [[1, 1], [1, 1], [1, 1]]
    )
    assert not hasattr(built, "target")
    assert not hasattr(built, "task_fingerprint")


def test_padding_is_inert_for_folds_and_queries() -> None:
    short = ArcTask(
        train=(_pair(1, 2), _pair(3, 4)),
        test=(_pair(5, 6),),
        task_id="short",
    )
    built = build_arc_target_free_task_bank(
        (short, _task()), RowEventConfig(max_demonstrations=3)
    )
    folds = built.fold_inputs
    np.testing.assert_array_equal(folds.demonstration_valid[0], [True, True, False])
    assert not np.any(np.asarray(folds.demonstration_inputs[0, 2]))
    assert not np.any(np.asarray(folds.demonstration_outputs[0, 2]))
    np.testing.assert_array_equal(folds.fold_valid[0], [True, True, False])
    assert int(folds.held_out_demonstration_index[0, 2]) == -1

    np.testing.assert_array_equal(built.query_valid, [[True, False], [True, True]])
    assert not np.any(np.asarray(built.query_inputs[0, 1]))
    assert not np.any(np.asarray(built.query_shapes[0, 1]))
    assert int(built.query_ordinals[0, 1]) == -1
    np.testing.assert_array_equal(built.checkpoint_indices[0, 1], [0, 30, 60])


def test_multiquery_ordinals_and_checkpoint_indices_are_stable() -> None:
    built = build_arc_target_free_task_bank(
        (_task(),), RowEventConfig(max_demonstrations=3)
    )
    np.testing.assert_array_equal(built.task_ordinals, [0])
    np.testing.assert_array_equal(built.query_ordinals, [[0, 1]])
    assert ARC_ADAPTATION_CHECKPOINTS == (0, 30, 60)
    np.testing.assert_array_equal(
        built.checkpoint_indices,
        np.asarray([[[0, 30, 60], [0, 30, 60]]]),
    )
    np.testing.assert_array_equal(built.query_inputs[0, 0, 0, 0], 9)
    np.testing.assert_array_equal(built.query_inputs[0, 1, 0, 0], 8)


def test_compact_bank_projection_stays_well_below_dense_streams() -> None:
    built = build_arc_target_free_task_bank(
        (_task(),) * 4, RowEventConfig(max_demonstrations=3)
    )
    exact = sum(array.nbytes for array in _tree_arrays(built))
    dense_fold_bytes = 4 * 3 * 390 * 830 * np.dtype(np.float32).itemsize
    assert built.projected_bytes == exact
    assert built.projected_bytes < dense_fold_bytes // 20


def test_arc_bank_requires_two_demonstrations() -> None:
    one = ArcTask(
        train=(_pair(1, 2),),
        test=(_pair(3, 4),),
        task_id="one",
    )
    with pytest.raises(ValueError, match="at least two demonstrations"):
        build_arc_target_free_task_bank((one,), RowEventConfig(max_demonstrations=3))


@pytest.mark.parametrize(
    ("tasks", "rows", "latent_steps", "message"),
    [
        ((), RowEventConfig(max_demonstrations=3), 60, "non-empty"),
        ((_task(),), RowEventConfig(max_demonstrations=2), 60, "capacity"),
        ((_task(),), RowEventConfig(max_demonstrations=3), 59, "at least 60"),
        ((object(),), RowEventConfig(max_demonstrations=3), 60, "ArcTask"),
        ((_task(),), RowEventConfig(max_demonstrations=3), True, "integer"),
    ],
)
def test_arc_bank_rejects_invalid_static_contracts(
    tasks: Any, rows: RowEventConfig, latent_steps: int, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=f"(?i){message}"):
        build_arc_target_free_task_bank(tasks, rows, latent_steps=latent_steps)


def test_arc_bank_rejects_non_row_configuration() -> None:
    with pytest.raises(TypeError, match="RowEventConfig"):
        build_arc_target_free_task_bank((_task(),), object())  # type: ignore[arg-type]


def test_task_id_and_official_outputs_are_not_bank_inputs() -> None:
    rows = RowEventConfig(max_demonstrations=3)
    first = build_arc_target_free_task_bank((_task(),), rows)
    renamed = replace(_task(test_outputs=(0, 1)), task_id="renamed")
    second = build_arc_target_free_task_bank((renamed,), rows)

    for left, right in zip(_tree_arrays(first), _tree_arrays(second), strict=True):
        np.testing.assert_array_equal(left, right)


def test_compiled_evaluation_context_exactly_matches_public_encoder() -> None:
    task = _shaped_task()
    rows = RowEventConfig(max_demonstrations=3)
    bank = build_arc_target_free_task_bank((task,), rows)

    stream = synthesize_arc_evaluation_context(
        _task_fold_inputs(bank), bank.query_inputs[0, 0], bank.query_shapes[0, 0], rows
    )
    expected = encode_query_episode(task, 0, rows)

    assert isinstance(stream, ArcContextStream)
    np.testing.assert_array_equal(stream.events, expected.events)
    np.testing.assert_array_equal(
        stream.advances, _expected_context_advances(expected, rows)
    )
    assert int(stream.query_start) == expected.query_start
    assert int(stream.query_stop) == expected.query_stop
    assert bool(stream.valid)


@pytest.mark.parametrize("held_index", [0, 1, 2])
def test_compiled_loo_context_exactly_matches_public_encoder(
    held_index: int,
) -> None:
    task = _shaped_task()
    rows = RowEventConfig(max_demonstrations=3)
    bank = build_arc_target_free_task_bank((task,), rows)
    episode = leave_one_demonstration_out_episodes(task)[held_index]

    stream = synthesize_arc_loo_context(
        _task_fold_inputs(bank), np.int32(held_index), rows
    )
    expected = encode_arc_query_episode(episode, rows)

    np.testing.assert_array_equal(stream.events, expected.events)
    np.testing.assert_array_equal(
        stream.advances, _expected_context_advances(expected, rows)
    )
    assert int(stream.query_stop) == expected.query_stop
    assert bool(stream.valid)


def test_compiled_context_synthesis_is_jittable_and_target_free() -> None:
    rows = RowEventConfig(max_demonstrations=3)
    first = build_arc_target_free_task_bank((_task(test_outputs=(7, 8)),), rows)
    second = build_arc_target_free_task_bank((_task(test_outputs=(0, 1)),), rows)

    @jax.jit
    def synthesize(bank: ArcTargetFreeTaskBank) -> ArcContextStream:
        return synthesize_arc_evaluation_context(
            jax.tree.map(lambda value: value[0], bank.fold_inputs),
            bank.query_inputs[0, 0],
            bank.query_shapes[0, 0],
            rows,
        )

    left = synthesize(first)
    right = synthesize(second)
    for left_array, right_array in zip(
        _tree_arrays(left), _tree_arrays(right), strict=True
    ):
        np.testing.assert_array_equal(left_array, right_array)


def test_padded_loo_fold_materializes_as_inert_invalid_stream() -> None:
    short = ArcTask(
        train=(_pair(1, 2), _pair(3, 4)),
        test=(_pair(5, 6),),
        task_id="short",
    )
    rows = RowEventConfig(max_demonstrations=3)
    bank = build_arc_target_free_task_bank((short, _task()), rows)

    stream = synthesize_arc_loo_context(
        _task_fold_inputs(bank),
        bank.fold_inputs.held_out_demonstration_index[0, 2],
        rows,
    )

    assert not bool(stream.valid)
    assert not np.any(np.asarray(stream.events))
    assert not np.any(np.asarray(stream.advances))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("demonstration_inputs", np.zeros((2, 29, 30), np.uint8), "shape"),
        ("demonstration_outputs", np.zeros((2, 29, 30), np.uint8), "outputs"),
        ("demonstration_input_shapes", np.zeros((2, 3), np.uint8), "shape tensors"),
        ("demonstration_valid", np.ones((2, 1), np.bool_), "valid shape"),
        ("fold_valid", np.ones((2, 1), np.bool_), "schedule shape"),
        ("demonstration_inputs", np.zeros((2, 30, 30), np.float32), "integer"),
        ("demonstration_valid", np.ones((2,), np.int32), "boolean"),
    ],
)
def test_context_synthesis_rejects_malformed_compact_task_fields(
    field: str, replacement: np.ndarray, message: str
) -> None:
    rows = RowEventConfig(max_demonstrations=2)
    bank = build_arc_target_free_task_bank(
        (
            ArcTask(
                train=(_pair(1, 2), _pair(3, 4)),
                test=(ArcPair(ArcGrid(((5,),)), None),),
            ),
        ),
        rows,
    )
    task_inputs = _task_fold_inputs(bank)._replace(**{field: replacement})

    with pytest.raises((TypeError, ValueError), match=f"(?i){message}"):
        synthesize_arc_evaluation_context(
            task_inputs, bank.query_inputs[0, 0], bank.query_shapes[0, 0], rows
        )


@pytest.mark.parametrize(
    ("query_input", "query_shape", "message"),
    [
        (np.zeros((29, 30), np.uint8), np.asarray([1, 1]), "query_input"),
        (np.zeros((30, 30), np.uint8), np.asarray([1, 1, 1]), "query_shape"),
        (np.zeros((30, 30), np.float32), np.asarray([1, 1]), "integer"),
    ],
)
def test_context_synthesis_rejects_malformed_query_arrays(
    query_input: np.ndarray, query_shape: np.ndarray, message: str
) -> None:
    rows = RowEventConfig(max_demonstrations=2)
    bank = build_arc_target_free_task_bank(
        (
            ArcTask(
                train=(_pair(1, 2), _pair(3, 4)),
                test=(ArcPair(ArcGrid(((5,),)), None),),
            ),
        ),
        rows,
    )

    with pytest.raises(ValueError, match=f"(?i){message}"):
        synthesize_arc_evaluation_context(
            _task_fold_inputs(bank), query_input, query_shape, rows
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("query_inputs", np.zeros((1, 1, 29, 30), np.uint8), "query_inputs"),
        ("query_shapes", np.zeros((1, 1, 3), np.uint8), "query_shapes"),
        ("query_valid", np.ones((1, 1, 1), np.bool_), "query_valid shape"),
        ("query_valid", np.ones((1, 1), np.int32), "boolean"),
        ("checkpoint_indices", np.zeros((1, 1, 2), np.int32), "shape"),
        (
            "checkpoint_indices",
            np.asarray([[[0, 30, 59]]], np.int32),
            "exactly",
        ),
    ],
)
def test_arc_runner_bank_validation_fails_closed(
    field: str, replacement: np.ndarray, message: str
) -> None:
    rows = RowEventConfig(max_demonstrations=2)
    task = ArcTask(
        train=(_pair(1, 2), _pair(3, 4)),
        test=(ArcPair(ArcGrid(((5,),)), ArcGrid(((7,),))),),
    )
    bank = build_arc_target_free_task_bank((task,), rows)
    malformed = bank._replace(**{field: replacement})

    with pytest.raises((TypeError, ValueError), match=f"(?i){message}"):
        arc_adaptation._validate_arc_runner_bank(malformed, rows)


def test_arc_runner_bank_validation_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="ArcTargetFreeTaskBank"):
        arc_adaptation._validate_arc_runner_bank(object(), RowEventConfig())


def test_real_compiled_arc_runner_adapts_and_restores_target_free_tasks() -> None:
    rows = RowEventConfig(max_demonstrations=3)
    short = ArcTask(
        train=(_pair(1, 4), _pair(2, 5)),
        test=(ArcPair(ArcGrid(((3,),)), ArcGrid(((6,),))),),
        task_id="short-runner",
    )
    long = ArcTask(
        train=(_pair(1, 2), _pair(3, 4), _pair(5, 6)),
        test=(ArcPair(ArcGrid(((7,),)), ArcGrid(((8,),))),),
        task_id="long-runner",
    )
    bank = build_arc_target_free_task_bank((short, long), rows)
    mutated = build_arc_target_free_task_bank(
        (
            replace(short, test=(ArcPair(short.test[0].input, ArcGrid(((9,),))),)),
            replace(long, test=(ArcPair(long.test[0].input, ArcGrid(((0,),))),)),
        ),
        rows,
    )
    model = _tiny_row_model(rows)
    learner = compile_pp_prop(model)
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(learner.param_states)
    base = snapshot_parameters(model)
    runner = compile_arc_task_local_adaptation_runner(
        model,
        learner,
        optimizer,
        base_parameters=base,
        row_config=rows,
        latent_steps=60,
        clip_norm=1.0,
    )

    first = runner(bank)
    second = runner(mutated)

    assert isinstance(first, ArcTaskBankAdaptationResult)
    assert first.fold_losses.shape == (2, 3)
    assert first.fold_applied.shape == (2, 3)
    np.testing.assert_array_equal(
        first.fold_applied, [[True, True, False], [True, True, True]]
    )
    assert float(first.fold_losses[0, 2]) == 0.0
    assert np.all(np.isfinite(np.asarray(first.fold_losses)))
    assert first.checkpoint_outputs.shape == (2, 1, 3, 9060)
    np.testing.assert_array_equal(
        first.checkpoint_recorded,
        [[(True, True, True)], [(True, True, True)]],
    )
    np.testing.assert_array_equal(first.query_valid, bank.query_valid)
    for left, right in zip(_tree_arrays(first), _tree_arrays(second), strict=True):
        np.testing.assert_array_equal(left, right)
    restored = snapshot_parameters(model)
    _assert_parameter_snapshots_equal(restored, base)
    assert int(optimizer.step_count.value) == 0


def test_repeated_epochs_replay_each_fold_schedule_in_order() -> None:
    rows = RowEventConfig(max_demonstrations=3)
    task = ArcTask(
        train=(_pair(1, 2), _pair(3, 4), _pair(5, 6)),
        test=(ArcPair(ArcGrid(((7,),)), ArcGrid(((8,),))),),
        task_id="epoch-runner",
    )
    short = ArcTask(
        train=(_pair(1, 4), _pair(2, 5)),
        test=(ArcPair(ArcGrid(((3,),)), ArcGrid(((6,),))),),
        task_id="epoch-short",
    )
    bank = build_arc_target_free_task_bank((short, task), rows)
    model = _tiny_row_model(rows)
    learner = compile_pp_prop(model)
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(learner.param_states)
    base = snapshot_parameters(model)
    runner = compile_arc_task_local_adaptation_runner(
        model,
        learner,
        optimizer,
        base_parameters=base,
        row_config=rows,
        latent_steps=60,
        clip_norm=1.0,
        epochs=2,
    )

    result = runner(bank)

    assert result.fold_losses.shape == (2, 6)
    np.testing.assert_array_equal(
        result.fold_applied,
        [[True, True, False, True, True, False], [True] * 6],
    )
    assert np.all(np.isfinite(np.asarray(result.fold_losses)))
    _assert_parameter_snapshots_equal(snapshot_parameters(model), base)
    assert int(optimizer.step_count.value) == 0


@pytest.mark.parametrize("epochs", (0, -1, 1.5, True))
def test_runner_rejects_a_non_positive_epoch_count(epochs) -> None:
    rows = RowEventConfig(max_demonstrations=3)
    model = _tiny_row_model(rows)
    learner = compile_pp_prop(model)
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(learner.param_states)
    with pytest.raises(ValueError, match="(?i)epochs"):
        compile_arc_task_local_adaptation_runner(
            model,
            learner,
            optimizer,
            base_parameters=snapshot_parameters(model),
            row_config=rows,
            latent_steps=60,
            clip_norm=1.0,
            epochs=epochs,
        )


def _schedule_runner(model, learner, optimizer, rows, base, schedule):
    return compile_arc_task_local_adaptation_runner(
        model,
        learner,
        optimizer,
        base_parameters=base,
        row_config=rows,
        latent_steps=60,
        clip_norm=1.0,
        update_schedule=schedule,
    )


def _schedule_bank(rows):
    task = ArcTask(
        train=(_pair(1, 2), _pair(3, 4), _pair(5, 6)),
        test=(ArcPair(ArcGrid(((7,),)), ArcGrid(((8,),))),),
        task_id="schedule-runner",
    )
    return build_arc_target_free_task_bank((task,), rows)


@pytest.mark.parametrize("schedule", ("per_episode", "per_tick"))
def test_both_update_schedules_adapt_and_restore(schedule) -> None:
    rows = RowEventConfig(max_demonstrations=3)
    bank = _schedule_bank(rows)
    model = _tiny_row_model(rows)
    learner = compile_pp_prop(model)
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(learner.param_states)
    base = snapshot_parameters(model)

    result = _schedule_runner(model, learner, optimizer, rows, base, schedule)(bank)

    assert np.all(np.isfinite(np.asarray(result.fold_losses)))
    _assert_parameter_snapshots_equal(snapshot_parameters(model), base)
    assert int(optimizer.step_count.value) == 0


def test_the_two_schedules_reach_different_parameters() -> None:
    """Otherwise the option would be a formatting choice, not a schedule."""
    rows = RowEventConfig(max_demonstrations=3)
    bank = _schedule_bank(rows)
    reached = {}
    for schedule in ("per_episode", "per_tick"):
        model = _tiny_row_model(rows)
        learner = compile_pp_prop(model)
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(learner.param_states)
        base = snapshot_parameters(model)
        runner = _schedule_runner(model, learner, optimizer, rows, base, schedule)
        reached[schedule] = np.asarray(runner(bank).fold_losses)

    assert not np.allclose(reached["per_episode"], reached["per_tick"])


@pytest.mark.parametrize("schedule", ("", "per_step", "online", None))
def test_the_runner_rejects_an_unknown_update_schedule(schedule) -> None:
    rows = RowEventConfig(max_demonstrations=3)
    model = _tiny_row_model(rows)
    learner = compile_pp_prop(model)
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(learner.param_states)
    with pytest.raises(ValueError, match="update_schedule"):
        compile_arc_task_local_adaptation_runner(
            model,
            learner,
            optimizer,
            base_parameters=snapshot_parameters(model),
            row_config=rows,
            latent_steps=60,
            clip_norm=1.0,
            update_schedule=schedule,
        )


def test_arc_runner_source_uses_compiled_loops_without_python_model_loop() -> None:
    import ast
    import inspect

    source = inspect.getsource(compile_arc_task_local_adaptation_runner)
    tree = ast.parse(source)
    assert "brainstate.transform.jit" in source
    assert source.count("brainstate.transform.for_loop") >= 4
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
