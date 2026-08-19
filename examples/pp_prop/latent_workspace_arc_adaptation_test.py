"""Tests for compact target-free ARC adaptation banks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import numpy as np
import pytest

try:
    from examples.pp_prop.latent_workspace_arc_adaptation import (
        ARC_ADAPTATION_CHECKPOINTS,
        ArcAdaptationFoldInputs,
        ArcTargetFreeTaskBank,
        build_arc_target_free_task_bank,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
    )
except ImportError:
    from latent_workspace_arc_adaptation import (
        ARC_ADAPTATION_CHECKPOINTS,
        ArcAdaptationFoldInputs,
        ArcTargetFreeTaskBank,
        build_arc_target_free_task_bank,
    )
    from latent_workspace_task import ArcGrid, ArcPair, ArcTask, RowEventConfig


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
    with pytest.raises((TypeError, ValueError), match=message):
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
