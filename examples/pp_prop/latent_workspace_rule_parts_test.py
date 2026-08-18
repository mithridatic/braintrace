"""Tests for the grid primitives shared by the verified rule families."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from latent_workspace_rule_parts import (  # noqa: E402
    COLOR_COUNT,
    DIHEDRAL_NAMES,
    apply_dihedral,
    background_color,
    connected_components,
    crop_to_mask,
    is_valid_grid,
    lattice_cells,
    periodic_fill,
    separator_indices,
    split_panels,
)


def test_dihedral_names_form_the_square_symmetry_group() -> None:
    """Eight distinct maps act on a grid whose cells are all different."""

    grid = np.arange(9, dtype=np.int32).reshape(3, 3)
    images = {apply_dihedral(grid, name).tobytes() for name in DIHEDRAL_NAMES}
    assert len(DIHEDRAL_NAMES) == 8
    assert len(images) == 8


@pytest.mark.parametrize("name", DIHEDRAL_NAMES)
def test_dihedral_maps_are_involutive_under_repetition(name: str) -> None:
    """Applying a map four times returns the original grid."""

    grid = np.arange(9, dtype=np.int32).reshape(3, 3)
    moved = grid
    for _ in range(4):
        moved = apply_dihedral(moved, name)
    assert np.array_equal(moved, grid)


def test_dihedral_rejects_unknown_name() -> None:
    """An unknown map name is a programming error, not a silent no-op."""

    with pytest.raises(ValueError, match="unknown dihedral map"):
        apply_dihedral(np.zeros((2, 2), np.int32), "shear")


def test_dihedral_preserves_non_square_shapes_correctly() -> None:
    """Rotations and transposes swap the axes of a non-square grid."""

    grid = np.arange(6, dtype=np.int32).reshape(2, 3)
    assert apply_dihedral(grid, "rot90").shape == (3, 2)
    assert apply_dihedral(grid, "transpose").shape == (3, 2)
    assert apply_dihedral(grid, "flip_horizontal").shape == (2, 3)


@pytest.mark.parametrize(
    ("grid", "expected"),
    [
        (np.zeros((1, 1), np.int32), True),
        (np.full((30, 30), 9, np.int32), True),
        (np.zeros((31, 1), np.int32), False),
        (np.zeros((1, 31), np.int32), False),
        (np.full((1, 1), COLOR_COUNT, np.int32), False),
        (np.full((1, 1), -1, np.int32), False),
        (np.zeros((0, 3), np.int32), False),
        (None, False),
    ],
)
def test_is_valid_grid_bounds(grid: np.ndarray | None, expected: bool) -> None:
    """Legality covers rank, both side limits, and the colour range."""

    assert is_valid_grid(grid) is expected


def test_background_color_is_the_mode_across_every_input() -> None:
    """The modal colour is pooled over inputs, never over outputs."""

    pairs = [
        (np.array([[0, 0, 3]], np.int32), np.full((1, 3), 7, np.int32)),
        (np.array([[3, 3, 3]], np.int32), np.full((1, 3), 7, np.int32)),
    ]
    assert background_color(pairs) == 3


def test_connected_components_respect_diagonality_and_colour() -> None:
    """Adjacency conventions change how many components a grid has."""

    grid = np.array([[1, 0], [0, 2]], np.int32)
    assert len(connected_components(grid, 0, diagonal=False, same_color=False)) == 2
    assert len(connected_components(grid, 0, diagonal=True, same_color=False)) == 1
    assert len(connected_components(grid, 0, diagonal=True, same_color=True)) == 2


def test_connected_components_cover_every_non_background_cell_once() -> None:
    """Masks partition the filled cells exactly."""

    grid = np.array([[1, 1, 0], [0, 2, 2], [3, 0, 2]], np.int32)
    masks = connected_components(grid, 0, diagonal=False, same_color=True)
    total = np.zeros(grid.shape, dtype=int)
    for mask in masks:
        total += mask.astype(int)
    assert np.array_equal(total > 0, grid != 0)
    assert total.max() <= 1


def test_crop_to_mask_returns_the_bounding_box() -> None:
    """Cropping keeps every row and column the mask touches."""

    grid = np.array([[0, 0, 0], [0, 5, 6], [0, 0, 0]], np.int32)
    assert crop_to_mask(grid, grid != 0).tolist() == [[5, 6]]


def test_separator_indices_finds_uniform_lines_only() -> None:
    """A line qualifies only when every one of its cells matches."""

    grid = np.array([[1, 2], [5, 5], [5, 6]], np.int32)
    assert separator_indices(grid, 0).tolist() == [1]
    assert separator_indices(grid, 1).tolist() == []


def test_split_panels_removes_separators_and_keeps_order() -> None:
    """Panels come back in row-major order without their separator lines."""

    grid = np.array([[1, 5, 2], [3, 5, 4]], np.int32)
    panels = split_panels(grid)
    assert [panel.tolist() for panel in panels] == [[[1], [3]], [[2], [4]]]


def test_split_panels_returns_the_grid_when_no_separator_exists() -> None:
    """A grid without an interior separator is its own single panel."""

    grid = np.array([[1, 2], [3, 4]], np.int32)
    assert len(split_panels(grid)) == 1


def test_split_panels_ignores_a_fully_uniform_grid() -> None:
    """Every line being uniform means no line is a separator."""

    grid = np.full((3, 3), 5, np.int32)
    assert len(split_panels(grid)) == 1


def test_lattice_cells_decomposes_a_clean_grid() -> None:
    """A single-colour lattice yields a rectangular cell array."""

    grid = np.array([[1, 5, 2], [5, 5, 5], [3, 5, 4]], np.int32)
    cells, color = lattice_cells(grid)
    assert color == 5
    assert [[cell.tolist() for cell in row] for row in cells] == [
        [[[1]], [[2]]],
        [[[3]], [[4]]],
    ]


def test_lattice_cells_rejects_mixed_separator_colours() -> None:
    """Two separator lines of different colours are not one lattice.

    A full-width separator row intersects every separator column, so a mixed
    pair can only arise between two lines of the *same* axis.
    """

    grid = np.array([[1, 2], [5, 5], [1, 2], [6, 6]], np.int32)
    assert lattice_cells(grid) is None


def test_lattice_cells_accepts_a_single_axis_lattice() -> None:
    """Separators on one axis alone still decompose the grid."""

    grid = np.array([[1, 5, 2], [6, 6, 6], [3, 5, 4]], np.int32)
    cells, color = lattice_cells(grid)
    assert color == 6
    assert (len(cells), len(cells[0])) == (2, 1)


def test_lattice_cells_rejects_a_grid_without_separators() -> None:
    """No uniform line means no lattice."""

    assert lattice_cells(np.array([[1, 2], [3, 4]], np.int32)) is None


def test_periodic_fill_repairs_the_smallest_exact_period() -> None:
    """A hole is filled from the tiling the known cells agree on."""

    grid = np.array([[1, 2, 1, 2], [1, 2, 9, 2]], np.int32)
    assert periodic_fill(grid, 9).tolist() == [[1, 2, 1, 2], [1, 2, 1, 2]]


def test_periodic_fill_rejects_inconsistent_grids() -> None:
    """No period explains the known cells, so no repair is proposed."""

    grid = np.array([[1, 2, 3], [4, 5, 9]], np.int32)
    assert periodic_fill(grid, 9) is None


def test_periodic_fill_requires_something_to_repair() -> None:
    """A grid with no hole, or nothing but holes, is not repairable."""

    assert periodic_fill(np.array([[1, 2]], np.int32), 9) is None
    assert periodic_fill(np.full((2, 2), 9, np.int32), 9) is None
