"""Tests for the per-cell ARC grid feature map."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from latent_workspace_cell_features import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    PAD_COLOR,
    PATCH_CELLS,
    PATCH_COLORS,
    _axis_period,
    _mirror_colours,
    _ray_evidence,
    cell_feature_width,
    cell_features,
)


def _padded(rows):
    grid = np.full((1, MAX_GRID_SIZE, MAX_GRID_SIZE), PAD_COLOR, np.int32)
    block = np.asarray(rows, np.int32)
    grid[0, : block.shape[0], : block.shape[1]] = block
    return jnp.asarray(grid), jnp.array([block.shape[0]]), jnp.array([block.shape[1]])


def _valid_mask(height, width):
    coordinates = jnp.arange(MAX_GRID_SIZE)
    return (coordinates[None, :, None] < height[:, None, None]) & (
        coordinates[None, None, :] < width[:, None, None]
    )


def test_feature_width_matches_the_produced_array():
    grid, height, width = _padded([[1, 2], [3, 4]])
    assert cell_features(grid, height, width).shape == (
        1,
        MAX_GRID_SIZE * MAX_GRID_SIZE,
        cell_feature_width(),
    )


def test_patch_block_reports_the_cell_and_its_out_of_grid_neighbours():
    grid, height, width = _padded([[7]])
    patch = cell_features(grid, height, width)[0, 0, : PATCH_CELLS * PATCH_COLORS]
    patch = patch.reshape(PATCH_CELLS, PATCH_COLORS)
    centre = PATCH_CELLS // 2
    assert int(jnp.argmax(patch[centre])) == 7
    # every other neighbour of a 1x1 grid is off-grid
    others = [int(jnp.argmax(patch[i])) for i in range(PATCH_CELLS) if i != centre]
    assert set(others) == {PAD_COLOR}


def test_mirror_block_reads_the_flipped_and_transposed_images():
    grid, height, width = _padded([[1, 2], [3, 4]])
    coordinates = jnp.arange(MAX_GRID_SIZE)
    rows = jnp.broadcast_to(coordinates[None, :, None], grid.shape)
    columns = jnp.broadcast_to(coordinates[None, None, :], grid.shape)
    mirrors = _mirror_colours(grid, rows, columns, height, width)
    assert [int(v) for v in mirrors[0, 0, 0]] == [3, 2, 4, 1]


def test_transpose_mirror_is_out_of_grid_when_the_grid_is_not_square():
    grid, height, width = _padded([[1, 2, 5]])
    coordinates = jnp.arange(MAX_GRID_SIZE)
    rows = jnp.broadcast_to(coordinates[None, :, None], grid.shape)
    columns = jnp.broadcast_to(coordinates[None, None, :], grid.shape)
    mirrors = _mirror_colours(grid, rows, columns, height, width)
    assert int(mirrors[0, 0, 0, 3]) == PAD_COLOR


@pytest.mark.parametrize(
    "rows,expected_row_period,expected_column_period",
    [
        ([[1, 2, 3, 1, 2, 3]], 1, 3),
        ([[1, 1], [2, 2], [1, 1], [2, 2]], 2, 1),
        ([[1, 2], [3, 4]], 2, 2),
    ],
)
def test_axis_period_finds_the_smallest_repeat(rows, expected_row_period, expected_column_period):
    grid, height, width = _padded(rows)
    valid = _valid_mask(height, width)
    assert int(_axis_period(grid, valid, height, 1)[0]) == expected_row_period
    assert int(_axis_period(grid, valid, width, 2)[0]) == expected_column_period


def test_rays_report_the_nearest_non_background_colour_and_distance():
    grid, height, width = _padded([[0, 0, 5, 0, 0]])
    valid = _valid_mask(height, width)
    background = jnp.array([0], jnp.int32)
    colours, distances = _ray_evidence(grid, valid, background)
    # cell (0, 4) looks left and finds the 5 two columns away
    assert int(colours[0, 0, 4, 0]) == 5
    assert int(distances[0, 0, 4, 0]) == 2
    # nothing to its right
    assert int(colours[0, 0, 4, 1]) == PAD_COLOR


def test_a_cell_does_not_see_itself_along_a_ray():
    grid, height, width = _padded([[5, 0, 0]])
    valid = _valid_mask(height, width)
    colours, _ = _ray_evidence(grid, valid, jnp.array([0], jnp.int32))
    assert int(colours[0, 0, 0, 0]) == PAD_COLOR
    assert int(colours[0, 0, 1, 0]) == 5


def test_features_are_finite_and_bounded_on_random_grids():
    rng = np.random.default_rng(0)
    grid = np.full((4, MAX_GRID_SIZE, MAX_GRID_SIZE), PAD_COLOR, np.int32)
    heights = rng.integers(1, MAX_GRID_SIZE + 1, size=4)
    widths = rng.integers(1, MAX_GRID_SIZE + 1, size=4)
    for i, (h, w) in enumerate(zip(heights, widths)):
        grid[i, :h, :w] = rng.integers(0, COLOR_COUNT, size=(h, w))
    features = cell_features(
        jnp.asarray(grid), jnp.asarray(heights), jnp.asarray(widths)
    )
    assert bool(jnp.isfinite(features).all())
    assert float(jnp.max(jnp.abs(features))) <= 1.0 + 1e-6


def test_padded_region_is_marked_invalid():
    grid, height, width = _padded([[1, 2], [3, 4]])
    features = cell_features(grid, height, width)
    # the validity scalar is the last feature
    flat = features[0, :, -1].reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
    assert float(flat[0, 0]) == 1.0
    assert float(flat[5, 5]) == 0.0


def test_features_are_translation_sensitive_but_shape_consistent():
    left, height, width = _padded([[3, 0], [0, 0]])
    right, _, _ = _padded([[0, 3], [0, 0]])
    a = cell_features(left, height, width)
    b = cell_features(right, height, width)
    assert not bool(jnp.allclose(a, b))
    assert a.shape == b.shape
