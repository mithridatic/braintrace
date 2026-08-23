"""Tests for the edit-rule ARC decoder.

The assertions that matter are the two the legacy decoder failed: that the
shape head can express a non-square output from one shared hidden (defect D2),
and that a colour cell can be *copied* from the query input rather than
generated from scratch (defect D1).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

try:
    from examples.pp_prop.latent_workspace_decoder import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        SHAPE_RULE_COUNT,
        SHAPE_SLOT_COUNT,
        color_cell_logits,
        decode_side_length,
        decoder_output_width,
        shape_axis_logits,
        split_decoder_logits,
    )
except ModuleNotFoundError:
    from latent_workspace_decoder import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        SHAPE_RULE_COUNT,
        SHAPE_SLOT_COUNT,
        color_cell_logits,
        decode_side_length,
        decoder_output_width,
        shape_axis_logits,
        split_decoder_logits,
    )

IDENTITY_RULE = 0
DOUBLE_RULE = 1
HALVE_RULE = 4
IDENTITY_OTHER_AXIS_RULE = len(("1/1", "2/1", "3/1", "4/1", "1/2", "1/3"))
ABSOLUTE_SLOT = SHAPE_RULE_COUNT


def _one_hot(side: int) -> jnp.ndarray:
    return jnp.asarray(np.eye(MAX_GRID_SIZE, dtype=np.float32)[side - 1])[None]


def _gate_on(slot: int) -> jnp.ndarray:
    logits = np.full((1, SHAPE_SLOT_COUNT), -20.0, dtype=np.float32)
    logits[0, slot] = 20.0
    return jnp.asarray(logits)


def _predicted_side(slot: int, height: int, width: int, *, predict_height: bool) -> int:
    logits = shape_axis_logits(
        _gate_on(slot),
        jnp.zeros((1, MAX_GRID_SIZE), dtype=jnp.float32),
        _one_hot(height),
        _one_hot(width),
        predict_height=predict_height,
    )
    return int(decode_side_length(logits)[0])


def test_compact_width_is_the_final_arc_output_size() -> None:
    assert decoder_output_width() == 2 * MAX_GRID_SIZE + 30 * 30 * COLOR_COUNT


def test_split_returns_the_shapes_the_existing_consumers_expect() -> None:
    compact = jnp.arange(2 * decoder_output_width(), dtype=jnp.float32).reshape(
        2, decoder_output_width()
    )

    height, width, colors = split_decoder_logits(compact)

    assert height.shape == (2, MAX_GRID_SIZE)
    assert width.shape == (2, MAX_GRID_SIZE)
    assert colors.shape == (2, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT)
    np.testing.assert_array_equal(
        jnp.concatenate((height, width, colors.reshape(2, -1)), axis=-1),
        compact,
    )


def test_split_rejects_a_legacy_width() -> None:
    with pytest.raises(ValueError, match="trailing axis"):
        split_decoder_logits(jnp.zeros((1, 1180), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("slot", "height", "width", "expected"),
    [
        (IDENTITY_RULE, 7, 3, 7),
        (DOUBLE_RULE, 7, 3, 14),
        (HALVE_RULE, 8, 3, 4),
        (IDENTITY_OTHER_AXIS_RULE, 7, 3, 3),
    ],
)
def test_deterministic_rules_map_input_sides_to_output_sides(
    slot: int, height: int, width: int, expected: int
) -> None:
    assert _predicted_side(slot, height, width, predict_height=True) == expected


def _halved(side: int) -> jnp.ndarray:
    return shape_axis_logits(
        _gate_on(HALVE_RULE),
        jnp.zeros((1, MAX_GRID_SIZE), dtype=jnp.float32),
        _one_hot(side),
        _one_hot(side),
        predict_height=True,
    )


def test_a_rule_that_cannot_divide_evenly_defers_instead_of_guessing() -> None:
    """An inexpressible rule contributes no mass and the other slots decide.

    Halving an even side is a definite answer; halving an odd one is not
    expressible at all, so the gated rule must not claim it.
    """
    even = jnp.exp(_halved(8))
    odd = jnp.exp(_halved(5))

    assert float(even[0, 3]) > 0.99  # Side 4, confidently
    assert float(jnp.max(odd)) < 0.5
    assert float(odd[0, 1]) < 0.5  # Never asserts side 2 for an odd input


def test_shared_gate_still_produces_a_non_square_shape() -> None:
    """The direct fix for defect D2.

    Two 30-way softmaxes off one shared hidden collapsed onto the same
    function, leaving 79.2% of predictions square. Here the *same* gate vector
    drives both axes and the prediction is still non-square, because each axis
    selects over a basis keyed to its own input side.
    """
    gate = _gate_on(IDENTITY_RULE)
    absolute = jnp.zeros((1, MAX_GRID_SIZE), dtype=jnp.float32)
    height_in, width_in = _one_hot(9), _one_hot(4)

    height = decode_side_length(
        shape_axis_logits(gate, absolute, height_in, width_in, predict_height=True)
    )
    width = decode_side_length(
        shape_axis_logits(gate, absolute, height_in, width_in, predict_height=False)
    )

    assert (int(height[0]), int(width[0])) == (9, 4)


def test_the_absolute_fallback_ignores_the_input_side() -> None:
    absolute = np.full((1, MAX_GRID_SIZE), -20.0, dtype=np.float32)
    absolute[0, 12] = 20.0

    predicted = decode_side_length(
        shape_axis_logits(
            _gate_on(ABSOLUTE_SLOT),
            jnp.asarray(absolute),
            _one_hot(3),
            _one_hot(3),
            predict_height=True,
        )
    )

    assert int(predicted[0]) == 13


def test_shape_logits_are_normalised_log_probabilities() -> None:
    logits = shape_axis_logits(
        jnp.zeros((1, SHAPE_SLOT_COUNT), dtype=jnp.float32),
        jnp.zeros((1, MAX_GRID_SIZE), dtype=jnp.float32),
        _one_hot(6),
        _one_hot(6),
        predict_height=True,
    )

    total = float(jnp.sum(jnp.exp(logits)))
    assert total == pytest.approx(1.0, abs=1e-5)


def _query_grid(grid: np.ndarray) -> jnp.ndarray:
    one_hot = np.zeros((1, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT), dtype=np.float32)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            one_hot[0, row, column, grid[row, column]] = 1.0
    return jnp.asarray(one_hot)


def _color_gate_on(component: int) -> jnp.ndarray:
    logits = np.full((1, MAX_GRID_SIZE, MAX_GRID_SIZE, 4), -20.0, dtype=np.float32)
    logits[..., component] = 20.0
    return jnp.asarray(logits)


def _decoded(
    component: int,
    grid: np.ndarray,
    output_shape: tuple[int, int],
    *,
    palette: jnp.ndarray | None = None,
    explicit: jnp.ndarray | None = None,
) -> np.ndarray:
    height, width = grid.shape
    colors = color_cell_logits(
        _color_gate_on(component),
        jnp.zeros((1, COLOR_COUNT), dtype=jnp.float32) if palette is None else palette,
        jnp.zeros((1, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT), dtype=jnp.float32)
        if explicit is None
        else explicit,
        _query_grid(grid),
        (
            jnp.asarray([output_shape[0]], dtype=jnp.int32),
            jnp.asarray([output_shape[1]], dtype=jnp.int32),
        ),
        (jnp.asarray([height], dtype=jnp.int32), jnp.asarray([width], dtype=jnp.int32)),
    )
    return np.asarray(jnp.argmax(colors, axis=-1))[0]


def test_copying_reproduces_the_query_grid_when_the_shape_is_unchanged() -> None:
    """The direct fix for defect D1: the output can reference the query input."""
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)

    decoded = _decoded(0, grid, (3, 3))

    np.testing.assert_array_equal(decoded[:3, :3], grid)


def test_copying_upscales_by_nearest_neighbour() -> None:
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    decoded = _decoded(0, grid, (4, 4))

    np.testing.assert_array_equal(
        decoded[:4, :4],
        np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.int32
        ),
    )


def test_copying_downscales_by_nearest_neighbour() -> None:
    grid = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])

    decoded = _decoded(0, grid.astype(np.int32), (2, 2))

    np.testing.assert_array_equal(
        decoded[:2, :2], np.array([[1, 2], [3, 4]], dtype=np.int32)
    )


def test_the_transposed_component_reproduces_a_transpose() -> None:
    grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    decoded = _decoded(1, grid, (3, 2))

    np.testing.assert_array_equal(decoded[:3, :2], grid.T)


def test_the_palette_component_ignores_the_query_input() -> None:
    palette = np.full((1, COLOR_COUNT), -20.0, dtype=np.float32)
    palette[0, 6] = 20.0
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    decoded = _decoded(2, grid, (2, 2), palette=jnp.asarray(palette))

    assert np.all(decoded == 6)


def test_the_explicit_component_emits_its_own_per_cell_colour() -> None:
    explicit = np.zeros(
        (1, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT), dtype=np.float32
    )
    explicit[0, 1, 1, 8] = 20.0
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    decoded = _decoded(3, grid, (2, 2), explicit=jnp.asarray(explicit))

    assert decoded[1, 1] == 8


def test_colour_logits_are_normalised_per_cell() -> None:
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    colors = color_cell_logits(
        jnp.zeros((1, MAX_GRID_SIZE, MAX_GRID_SIZE, 4), dtype=jnp.float32),
        jnp.zeros((1, COLOR_COUNT), dtype=jnp.float32),
        jnp.zeros((1, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT), dtype=jnp.float32),
        _query_grid(grid),
        (jnp.asarray([2], dtype=jnp.int32), jnp.asarray([2], dtype=jnp.int32)),
        (jnp.asarray([2], dtype=jnp.int32), jnp.asarray([2], dtype=jnp.int32)),
    )

    totals = np.asarray(jnp.sum(jnp.exp(colors), axis=-1))
    np.testing.assert_allclose(totals, np.ones_like(totals), atol=1e-5)


def test_colour_decoding_carries_a_batch_axis_independently() -> None:
    first = np.array([[1, 2], [3, 4]], dtype=np.int32)
    second = np.array([[5, 6], [7, 8]], dtype=np.int32)
    grids = jnp.concatenate((_query_grid(first), _query_grid(second)), axis=0)
    gate = jnp.concatenate((_color_gate_on(0), _color_gate_on(0)), axis=0)

    colors = color_cell_logits(
        gate,
        jnp.zeros((2, COLOR_COUNT), dtype=jnp.float32),
        jnp.zeros((2, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT), dtype=jnp.float32),
        grids,
        (jnp.asarray([2, 2], dtype=jnp.int32), jnp.asarray([2, 2], dtype=jnp.int32)),
        (jnp.asarray([2, 2], dtype=jnp.int32), jnp.asarray([2, 2], dtype=jnp.int32)),
    )

    decoded = np.asarray(jnp.argmax(colors, axis=-1))
    np.testing.assert_array_equal(decoded[0, :2, :2], first)
    np.testing.assert_array_equal(decoded[1, :2, :2], second)
