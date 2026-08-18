"""Tests for the panel-overlay, projection, and counting completions."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from latent_workspace_rule_shapes import (  # noqa: E402
    SHAPE_FAMILIES,
    family_color_rank,
    family_count_bar,
    family_palette_bar,
    family_panel_overlay,
    family_projection,
    family_uniform_fill,
)


def _grid(rows: list[list[int]]) -> np.ndarray:
    """Return an ``int32`` grid from nested lists.

    Fixtures below keep colour ``0`` in the clear majority of every input,
    because ``background_color`` is the mode of the inputs rather than a fixed
    colour. A small fixture that happens to be dominated by another colour
    silently changes what "background" means for the family under test.
    """

    return np.asarray(rows, dtype=np.int32)


def _named(family, pairs) -> dict[str, object]:
    """Materialise a family into a name-to-callable mapping."""

    return dict(family(pairs))


def test_every_family_yields_uniquely_named_rules() -> None:
    """Names are the attribution key, so a family must not collide with itself."""

    pairs = [(_grid([[1, 0], [0, 2]]), _grid([[3, 0], [0, 3]]))]
    for family in SHAPE_FAMILIES:
        names = [name for name, _ in family(pairs)]
        assert len(names) == len(set(names))


def test_panel_overlay_keeps_colours_and_respects_order() -> None:
    """Overlay preserves each panel's colours; order decides the winner."""

    source = _grid([[1, 5, 0], [0, 5, 2]])
    target = _grid([[1], [2]])
    rules = _named(family_panel_overlay, [(source, target)])
    assert rules["overlay:0:0"](source).tolist() == [[1], [2]]

    conflict = _grid([[1, 5, 3], [0, 5, 2]])
    assert rules["overlay:0:0"](conflict).tolist() == [[3], [2]]
    assert rules["overlay:1:0"](conflict).tolist() == [[1], [2]]


def test_panel_overlay_declines_mismatched_panels() -> None:
    """Panels of different shapes cannot be overlaid."""

    source = _grid([[1, 5, 0], [0, 5, 2]])
    target = _grid([[1], [2]])
    rules = _named(family_panel_overlay, [(source, target)])
    assert rules["overlay:0:0"](_grid([[1, 2], [3, 4]])) is None


def test_projection_collapses_one_axis() -> None:
    """Each row reduces to the single colour it contains."""

    source = _grid([[0, 3, 0], [2, 0, 2]])
    target = _grid([[3], [2]])
    rules = _named(family_projection, [(source, target)])
    assert rules["project:1:unique"](source).tolist() == [[3], [2]]
    assert rules["project:0:any_filled"](source).tolist() == [[1, 1, 1]]


def test_projection_declines_an_ambiguous_line() -> None:
    """A row holding two colours has no unique projection."""

    source = _grid([[0, 3, 0], [2, 0, 2]])
    target = _grid([[3], [2]])
    rules = _named(family_projection, [(source, target)])
    assert rules["project:1:unique"](_grid([[3, 2]])) is None


def test_colour_rank_maps_by_frequency() -> None:
    """The most frequent non-background colour takes the rank-0 output colour."""

    source = _grid([[1, 1, 2], [0, 0, 0]])
    target = _grid([[7, 7, 8], [0, 0, 0]])
    rules = _named(family_color_rank, [(source, target)])
    assert rules
    applied = [rule(source) for rule in rules.values()]
    assert any(
        grid is not None and np.array_equal(grid, target) for grid in applied
    )


def test_count_bar_length_tracks_the_counted_property() -> None:
    """A bar's length is the count, and its colour is fitted."""

    source = _grid([[1, 0, 1, 0, 0]])
    target = _grid([[4, 4]])
    rules = _named(family_count_bar, [(source, target)])
    assert rules["count:objects:0:4"](source).tolist() == [[4, 4]]
    assert rules["count:objects:0:4"](_grid([[1, 0, 1, 0, 1]])).tolist() == [[4, 4, 4]]


def test_count_bar_declines_an_empty_grid() -> None:
    """A count of zero is not a legal ARC grid side."""

    source = _grid([[1, 0, 1, 0, 0]])
    target = _grid([[4, 4]])
    rules = _named(family_count_bar, [(source, target)])
    assert rules["count:objects:0:4"](_grid([[0, 0]])) is None


def test_palette_bar_lists_colours_by_frequency() -> None:
    """The listing is ordered, deduplicated, and excludes the background."""

    source = _grid([[1, 1, 2], [0, 0, 0]])
    target = _grid([[1, 2]])
    rules = _named(family_palette_bar, [(source, target)])
    assert rules["palette:1:0"](source).tolist() == [[1, 2]]
    assert rules["palette:0:0"](source).tolist() == [[2, 1]]


def test_uniform_fill_selects_a_colour_from_the_input() -> None:
    """The fill colour is derived from the input, not fitted as a constant."""

    source = _grid([[1, 1, 2, 0, 0, 0, 0]])
    target = _grid([[1, 1, 1, 1, 1, 1, 1]])
    rules = _named(family_uniform_fill, [(source, target)])
    assert rules["uniform:mode:1"](source).tolist() == [[1] * 7]
    assert rules["uniform:rarest:1"](source).tolist() == [[2] * 7]


def test_uniform_fill_declines_a_blank_input() -> None:
    """With nothing but background there is no colour to choose."""

    source = _grid([[1, 1, 2, 0, 0, 0, 0]])
    target = _grid([[1, 1, 1, 1, 1, 1, 1]])
    rules = _named(family_uniform_fill, [(source, target)])
    assert rules["uniform:mode:1"](_grid([[0, 0]])) is None
