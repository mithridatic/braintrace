from __future__ import annotations

from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pytest

from latent_workspace_refinement import (
    RowRefinementLayout,
    build_refinement_feedback_event,
    capture_query_rows,
    next_reasoning_index,
    refinement_output_logits,
    refinement_training_logits,
    scatter_answer_rows,
)
from latent_workspace_task import RowEventConfig


def _layout() -> RowRefinementLayout:
    return RowRefinementLayout(
        input_width=830,
        event_valid_index=0,
        demonstration_phase_index=1,
        query_phase_index=2,
        input_side_valid_index=13,
        output_side_valid_index=14,
        normalized_start=15,
        row_index_start=20,
        input_height_start=50,
        input_width_start=80,
        output_height_start=110,
        output_width_start=140,
        input_mask_start=170,
        output_mask_start=200,
        input_color_start=230,
        output_color_start=530,
    )


def _one_hot(index: int, width: int) -> np.ndarray:
    values = np.zeros((width,), dtype=np.float32)
    values[index] = 1.0
    return values


def test_layout_matches_the_production_row_event_contract() -> None:
    layout = _layout()
    row_config = RowEventConfig()

    assert layout.max_grid_size == 30
    assert layout.color_count == 10
    assert layout.row_width == 300
    assert layout.shape_width == 60
    assert layout.output_width == 9060
    assert layout.input_color_slice == slice(230, 530)
    assert layout.output_color_slice == slice(530, 830)
    assert layout.input_width == row_config.input_width
    assert layout.event_valid_index == row_config.valid_slice.start
    assert layout.demonstration_phase_index == row_config.phase_slice.start
    assert layout.query_phase_index == row_config.phase_slice.start + 1
    assert layout.input_side_valid_index == row_config.side_valid_slice.start
    assert layout.output_side_valid_index == row_config.side_valid_slice.start + 1
    assert layout.normalized_slice == row_config.normalized_slice
    assert layout.row_index_slice == row_config.row_index_slice
    assert layout.input_height_slice == row_config.input_height_slice
    assert layout.input_width_slice == row_config.input_width_slice
    assert layout.output_height_slice == row_config.output_height_slice
    assert layout.output_width_slice == row_config.output_width_slice
    assert layout.input_mask_slice == row_config.input_mask_slice
    assert layout.output_mask_slice == row_config.output_mask_slice
    assert layout.input_color_slice == row_config.input_color_slice
    assert layout.output_color_slice == row_config.output_color_slice


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"query_phase_index": 830}, "inside input_width"),
        ({"output_color_start": 531}, "contiguous"),
        ({"input_side_valid_index": 14}, "unique"),
        ({"input_width": 829}, "inside input_width"),
    ],
)
def test_layout_rejects_corrupt_or_ambiguous_feature_boundaries(
    updates: dict[str, int], message: str
) -> None:
    values = asdict(_layout())
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        RowRefinementLayout(**values)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"input_width": 0}, "positive"),
        ({"max_grid_size": 29}, "must be 30"),
        ({"color_count": 9}, "must be 10"),
        ({"event_valid_index": True}, "must be an integer"),
        ({"event_valid_index": 19}, "must not overlap"),
    ],
)
def test_layout_fails_closed_on_invalid_capacity_or_scalar_types(
    updates: dict[str, object], message: str
) -> None:
    values = asdict(_layout())
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        RowRefinementLayout(**values)


def test_capture_query_rows_ignores_demonstrations_and_output_targets() -> None:
    layout = _layout()
    query_grid = jnp.zeros((2, 30, 30, 10), dtype=jnp.float32)
    query_shape = jnp.zeros((2, 60), dtype=jnp.float32)
    events = np.zeros((2, layout.input_width), dtype=np.float32)

    events[0, layout.event_valid_index] = 1.0
    events[0, layout.query_phase_index] = 1.0
    events[0, layout.input_side_valid_index] = 1.0
    events[0, layout.row_index_start + 1] = 1.0
    events[0, layout.input_height_start + 2] = 1.0
    events[0, layout.input_width_start + 1] = 1.0
    events[0, layout.input_color_start + 2] = 1.0
    events[0, layout.input_color_start + 10 + 4] = 1.0
    events[0, layout.output_color_start + 9] = 99.0

    events[1, layout.event_valid_index] = 1.0
    events[1, layout.demonstration_phase_index] = 1.0
    events[1, layout.input_side_valid_index] = 1.0
    events[1, layout.output_side_valid_index] = 1.0
    events[1, layout.row_index_start] = 1.0
    events[1, layout.input_height_start] = 1.0
    events[1, layout.input_width_start] = 1.0
    events[1, layout.input_color_start + 7] = 1.0
    events[1, layout.output_color_start + 8] = 1.0

    captured_grid, captured_shape = capture_query_rows(
        query_grid,
        query_shape,
        jnp.asarray(events),
        jnp.asarray([True, True]),
        layout,
    )

    np.testing.assert_array_equal(captured_grid[0, 1, 0], _one_hot(2, 10))
    np.testing.assert_array_equal(captured_grid[0, 1, 1], _one_hot(4, 10))
    assert float(captured_grid[0].sum()) == 2.0
    np.testing.assert_array_equal(captured_shape[0, :30], _one_hot(2, 30))
    np.testing.assert_array_equal(captured_shape[0, 30:], _one_hot(1, 30))
    np.testing.assert_array_equal(captured_grid[1], query_grid[1])
    np.testing.assert_array_equal(captured_shape[1], query_shape[1])

    malformed = events[:1].copy()
    malformed[0, layout.demonstration_phase_index] = 1.0
    malformed_grid, malformed_shape = capture_query_rows(
        query_grid[:1],
        query_shape[:1],
        jnp.asarray(malformed),
        jnp.asarray([True]),
        layout,
    )
    np.testing.assert_array_equal(malformed_grid, query_grid[:1])
    np.testing.assert_array_equal(malformed_shape, query_shape[:1])


@pytest.mark.parametrize(
    ("grid_shape", "shape_shape", "event_shape", "advance_shape", "message"),
    [
        ((2, 29, 30, 10), (2, 60), (2, 830), (2,), "query_grid"),
        ((2, 30, 30, 10), (2, 59), (2, 830), (2,), "query_shape"),
        ((2, 30, 30, 10), (2, 60), (2, 829), (2,), "event"),
        ((2, 30, 30, 10), (2, 60), (2, 830), (2, 1), "advance"),
    ],
)
def test_capture_query_rows_rejects_incompatible_state_shapes(
    grid_shape: tuple[int, ...],
    shape_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
    advance_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        capture_query_rows(
            jnp.zeros(grid_shape),
            jnp.zeros(shape_shape),
            jnp.zeros(event_shape),
            jnp.zeros(advance_shape, dtype=jnp.bool_),
            _layout(),
        )


def test_feedback_event_contains_query_and_soft_answer_without_becoming_valid() -> None:
    layout = _layout()
    query_grid = np.zeros((1, 30, 30, 10), dtype=np.float32)
    query_grid[0, 2, 0, 3] = 1.0
    query_grid[0, 2, 1, 5] = 1.0
    query_shape = np.concatenate((_one_hot(2, 30), _one_hot(1, 30)))[None]
    answer_grid = np.full((1, 30, 30, 10), -1.0e9, dtype=np.float32)
    answer_grid[0, 2, :, 6] = 0.0
    answer_shape = np.full((1, 60), -1.0e9, dtype=np.float32)
    answer_shape[0, 3] = 0.0
    answer_shape[0, 30 + 4] = 0.0

    event = build_refinement_feedback_event(
        jnp.asarray(query_grid),
        jnp.asarray(query_shape),
        jnp.asarray(answer_grid),
        jnp.asarray(answer_shape),
        jnp.asarray([2], dtype=jnp.int32),
        layout,
    )
    event = np.asarray(event[0])

    assert event[layout.event_valid_index] == 0.0
    assert event[layout.demonstration_phase_index] == 0.0
    assert event[layout.query_phase_index] == 1.0
    assert event[layout.input_side_valid_index] == 1.0
    assert event[layout.output_side_valid_index] == 1.0
    np.testing.assert_allclose(
        event[layout.normalized_slice],
        np.asarray([3 / 30, 3 / 30, 2 / 30, 4 / 30, 5 / 30]),
        atol=1e-7,
    )
    np.testing.assert_array_equal(event[layout.row_index_slice], _one_hot(2, 30))
    np.testing.assert_array_equal(event[layout.input_height_slice], _one_hot(2, 30))
    np.testing.assert_array_equal(event[layout.input_width_slice], _one_hot(1, 30))
    np.testing.assert_array_equal(event[layout.output_height_slice], _one_hot(3, 30))
    np.testing.assert_array_equal(event[layout.output_width_slice], _one_hot(4, 30))
    np.testing.assert_array_equal(
        event[layout.input_mask_slice],
        np.asarray([1.0, 1.0] + [0.0] * 28, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        event[layout.output_mask_slice],
        np.asarray([1.0] * 5 + [0.0] * 25, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        event[layout.input_color_slice].reshape(30, 10)[0], _one_hot(3, 10)
    )
    np.testing.assert_array_equal(
        event[layout.input_color_slice].reshape(30, 10)[1], _one_hot(5, 10)
    )
    np.testing.assert_array_equal(
        event[layout.output_color_slice].reshape(30, 10)[7], _one_hot(6, 10)
    )


def test_feedback_event_zeros_sides_for_rows_beyond_each_learned_height() -> None:
    layout = _layout()
    query_grid = np.zeros((1, 30, 30, 10), dtype=np.float32)
    query_grid[0, 5, :, 3] = 1.0
    query_shape = np.concatenate((_one_hot(2, 30), _one_hot(7, 30)))[None]
    answer_grid = np.zeros((1, 30, 30, 10), dtype=np.float32)
    answer_grid[0, 5, :, 6] = 100.0
    answer_shape = np.full((1, 60), -1.0e9, dtype=np.float32)
    answer_shape[0, 3] = 0.0
    answer_shape[0, 30 + 8] = 0.0

    event = np.asarray(
        build_refinement_feedback_event(
            jnp.asarray(query_grid),
            jnp.asarray(query_shape),
            jnp.asarray(answer_grid),
            jnp.asarray(answer_shape),
            jnp.asarray([5], dtype=jnp.int32),
            layout,
        )[0]
    )

    assert event[layout.input_side_valid_index] == 0.0
    assert event[layout.output_side_valid_index] == 0.0
    assert not np.any(event[layout.input_mask_slice])
    assert not np.any(event[layout.output_mask_slice])
    assert not np.any(event[layout.input_color_slice])
    assert not np.any(event[layout.output_color_slice])


@pytest.mark.parametrize(
    ("answer_grid_shape", "answer_shape_shape", "row_shape", "message"),
    [
        ((1, 29, 30, 10), (1, 60), (1,), "answer_grid"),
        ((1, 30, 30, 10), (1, 59), (1,), "answer_shape"),
        ((1, 30, 30, 10), (1, 60), (1, 1), "row_indices"),
    ],
)
def test_feedback_event_rejects_incompatible_answer_state_shapes(
    answer_grid_shape: tuple[int, ...],
    answer_shape_shape: tuple[int, ...],
    row_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_refinement_feedback_event(
            jnp.zeros((1, 30, 30, 10)),
            jnp.zeros((1, 60)),
            jnp.zeros(answer_grid_shape),
            jnp.zeros(answer_shape_shape),
            jnp.zeros(row_shape, dtype=jnp.int32),
            _layout(),
        )


def test_scatter_and_index_wrap_update_one_row_per_batch_slot() -> None:
    answer_grid = jnp.zeros((2, 30, 30, 10), dtype=jnp.float32)
    rows = jnp.stack(
        (
            jnp.arange(300, dtype=jnp.float32),
            -jnp.arange(300, dtype=jnp.float32),
        )
    )
    indices = jnp.asarray([29, 4], dtype=jnp.int32)

    updated = scatter_answer_rows(answer_grid, rows, indices)

    np.testing.assert_array_equal(updated[0, 29], rows[0].reshape(30, 10))
    np.testing.assert_array_equal(updated[1, 4], rows[1].reshape(30, 10))
    assert int(np.count_nonzero(np.asarray(updated[0, :29]))) == 0
    assert int(np.count_nonzero(np.asarray(updated[1, :4]))) == 0
    assert int(np.count_nonzero(np.asarray(updated[1, 5:]))) == 0
    np.testing.assert_array_equal(
        next_reasoning_index(indices), jnp.asarray([0, 5], dtype=jnp.int32)
    )


@pytest.mark.parametrize(
    "indices",
    [jnp.asarray(3), jnp.zeros((2, 1), dtype=jnp.int32), jnp.asarray([1.5])],
)
def test_next_reasoning_index_requires_a_vector_of_integers(indices: object) -> None:
    with pytest.raises(ValueError, match="one-dimensional integer"):
        next_reasoning_index(indices)


@pytest.mark.parametrize(
    ("grid_shape", "row_shape", "index_shape", "message"),
    [
        ((2, 29, 30, 10), (2, 300), (2,), "answer_grid"),
        ((2, 30, 30, 10), (2, 299), (2,), "row_logits"),
        ((2, 30, 30, 10), (2, 300), (2, 1), "row_indices"),
    ],
)
def test_scatter_rejects_incompatible_row_state_shapes(
    grid_shape: tuple[int, ...],
    row_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        scatter_answer_rows(
            jnp.zeros(grid_shape),
            jnp.zeros(row_shape),
            jnp.zeros(index_shape, dtype=jnp.int32),
        )


def test_refinement_output_is_explicit_shape_plus_full_grid() -> None:
    shape = jnp.arange(2 * 60, dtype=jnp.float32).reshape(2, 60)
    grid = jnp.arange(2 * 30 * 30 * 10, dtype=jnp.float32).reshape(2, 30, 30, 10)

    output = refinement_output_logits(shape, grid)

    assert output.shape == (2, 9060)
    np.testing.assert_array_equal(output[:, :60], shape)
    np.testing.assert_array_equal(output[:, 60:], grid.reshape(2, 9000))


def test_refinement_training_output_keeps_full_grid_out_of_per_tick_carry() -> None:
    shape = jnp.arange(2 * 60, dtype=jnp.float32).reshape(2, 60)
    row = jnp.arange(2 * 300, dtype=jnp.float32).reshape(2, 300)

    output = refinement_training_logits(shape, row)

    assert output.shape == (2, 360)
    np.testing.assert_array_equal(output[:, :60], shape)
    np.testing.assert_array_equal(output[:, 60:], row)


@pytest.mark.parametrize(
    ("shape_shape", "row_shape", "message"),
    [
        ((2, 59), (2, 300), "answer_shape"),
        ((2, 60), (2, 299), "answer_row"),
        ((2, 60), (1, 300), "leading axes"),
    ],
)
def test_refinement_training_output_rejects_incompatible_shapes(
    shape_shape: tuple[int, ...], row_shape: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        refinement_training_logits(jnp.zeros(shape_shape), jnp.zeros(row_shape))


@pytest.mark.parametrize(
    ("shape_shape", "grid_shape", "message"),
    [
        ((2, 59), (2, 30, 30, 10), "answer_shape"),
        ((2, 60), (2, 29, 30, 10), "answer_grid"),
        ((2, 60), (1, 30, 30, 10), "leading axes"),
    ],
)
def test_refinement_output_rejects_incompatible_shapes(
    shape_shape: tuple[int, ...], grid_shape: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        refinement_output_logits(jnp.zeros(shape_shape), jnp.zeros(grid_shape))
