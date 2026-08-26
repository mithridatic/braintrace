"""Tests for the same-shape and object-level rule completions."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from latent_workspace_rule_edits import (
    CONNECTIVITY,
    EDIT_FAMILIES,
    family_border,
    family_fill_enclosed,
    family_gravity,
    family_object_rank_recolor,
    family_object_recolor,
    family_panel_combine,
    family_periodic_extend,
)


def _grid(rows: list[list[int]]) -> np.ndarray:
    """Return an ``int32`` grid from nested lists."""

    return np.asarray(rows, dtype=np.int32)


def _named(family, pairs) -> dict[str, object]:
    """Materialise a family into a name-to-callable mapping."""

    return dict(family(pairs))


def test_every_family_yields_uniquely_named_rules() -> None:
    """Names are the attribution key, so a family must not collide with itself."""

    pairs = [(_grid([[1, 0], [0, 2]]), _grid([[3, 0], [0, 3]]))]
    for family in EDIT_FAMILIES:
        names = [name for name, _ in family(pairs)]
        assert len(names) == len(set(names))


def test_connectivity_table_covers_all_four_conventions() -> None:
    """Diagonality and colour-matching are independent binary choices."""

    assert len(CONNECTIVITY) == 4
    assert len(set(CONNECTIVITY)) == 4


def test_object_recolour_fits_a_size_table() -> None:
    """Objects of equal size receive the same fitted colour."""

    source = _grid([[1, 0, 0], [0, 0, 0], [0, 1, 1]])
    target = _grid([[2, 0, 0], [0, 0, 0], [0, 3, 3]])
    rules = _named(family_object_recolor, [(source, target)])
    assert rules
    applied = [rule(source) for rule in rules.values()]
    assert any(np.array_equal(grid, target) for grid in applied if grid is not None)


def test_object_recolour_declines_a_shape_change() -> None:
    """A completion that edits in place cannot explain a shape change."""

    source = _grid([[1, 0], [0, 1]])
    target = _grid([[1]])
    assert _named(family_object_recolor, [(source, target)]) == {}


def test_object_rank_recolour_orders_by_size() -> None:
    """The largest object takes the rank-0 colour under a descending fit."""

    source = _grid([[1, 1, 0], [0, 0, 0], [0, 0, 1]])
    target = _grid([[2, 2, 0], [0, 0, 0], [0, 0, 3]])
    rules = _named(family_object_rank_recolor, [(source, target)])
    assert rules
    applied = [rule(source) for rule in rules.values()]
    assert any(np.array_equal(grid, target) for grid in applied if grid is not None)


def test_panel_combine_applies_a_boolean_operation() -> None:
    """Two panels merge cell-wise into a single painted mask."""

    source = _grid([[1, 5, 0], [0, 5, 0], [0, 5, 1]])
    target = _grid([[2], [0], [2]])
    rules = _named(family_panel_combine, [(source, target)])
    assert rules["pcomb:and:2"](source).tolist() == [[0], [0], [0]]
    assert rules["pcomb:or:2"](source).tolist() == [[2], [0], [2]]
    assert rules["pcomb:xor:2"](source).tolist() == [[2], [0], [2]]


def test_panel_combine_requires_exactly_two_equal_panels() -> None:
    """A grid that does not split into two matching panels yields nothing."""

    source = _grid([[1, 5, 0], [0, 5, 0], [0, 5, 1]])
    target = _grid([[2], [0], [2]])
    rules = _named(family_panel_combine, [(source, target)])
    assert rules["pcomb:and:2"](_grid([[1, 2], [3, 4]])) is None


@pytest.mark.parametrize(
    ("axis", "reverse", "expected"),
    [
        (0, False, [[0, 0], [1, 2]]),
        (0, True, [[1, 2], [0, 0]]),
    ],
)
def test_gravity_compacts_towards_one_edge(
    axis: int, reverse: bool, expected: list[list[int]]
) -> None:
    """Non-background cells fall to the chosen edge, preserving order."""

    source = _grid([[1, 0], [0, 2]])
    rules = _named(family_gravity, [(source, source)])
    assert rules[f"grav{axis}{int(reverse)}"](source).tolist() == expected


def test_border_padding_is_fitted_from_the_size_delta() -> None:
    """A symmetric growth of two per axis fits a one-wide pad."""

    source = _grid([[1]])
    target = _grid([[4, 4, 4], [4, 1, 4], [4, 4, 4]])
    rules = _named(family_border, [(source, target)])
    assert rules["pad1:4"](source).tolist() == target.tolist()


def test_border_padding_refuses_to_exceed_the_grid_limit() -> None:
    """A pad that would exceed 30 cells on a side proposes nothing."""

    source = _grid([[1]])
    target = _grid([[4, 4, 4], [4, 1, 4], [4, 4, 4]])
    rules = _named(family_border, [(source, target)])
    assert rules["pad1:4"](np.zeros((30, 30), np.int32)) is None


def test_border_trim_is_fitted_from_a_negative_delta() -> None:
    """A symmetric shrink of two per axis fits a one-wide trim."""

    source = _grid([[4, 4, 4], [4, 1, 4], [4, 4, 4]])
    target = _grid([[1]])
    rules = _named(family_border, [(source, target)])
    assert rules["trim1"](source).tolist() == [[1]]
    assert rules["trim1"](_grid([[1, 2], [3, 4]])) is None


def test_fill_enclosed_paints_only_unreachable_background() -> None:
    """Background touching the border is left alone."""

    source = _grid([[3, 3, 3], [3, 0, 3], [3, 3, 3]])
    target = _grid([[3, 3, 3], [3, 4, 3], [3, 3, 3]])
    rules = _named(family_fill_enclosed, [(source, target)])
    assert rules["fill0:4"](source).tolist() == target.tolist()
    assert "fill3:4" in rules


def test_fill_enclosed_declines_when_nothing_is_enclosed() -> None:
    """An open grid has no enclosure to fill."""

    source = _grid([[3, 3, 3], [3, 0, 3], [3, 3, 3]])
    target = _grid([[3, 3, 3], [3, 4, 3], [3, 3, 3]])
    rules = _named(family_fill_enclosed, [(source, target)])
    assert rules["fill0:4"](_grid([[0, 0], [0, 0]])) is None


def test_periodic_extension_tiles_to_a_constant_shape() -> None:
    """A fixed output shape is filled by repeating the input."""

    source = _grid([[1, 2]])
    target = _grid([[1, 2, 1], [1, 2, 1]])
    rules = _named(family_periodic_extend, [(source, target)])
    assert rules["pext:2x3"](source).tolist() == target.tolist()


def test_periodic_extension_scales_by_a_constant_ratio() -> None:
    """A fixed size ratio adapts the output shape to each input."""

    source = _grid([[1, 2]])
    target = _grid([[1, 2, 1, 2]])
    rules = _named(family_periodic_extend, [(source, target)])
    assert rules["pscale:1x2"](_grid([[3, 4, 5]])).shape == (1, 6)
