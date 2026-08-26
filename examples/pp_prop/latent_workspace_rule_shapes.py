"""Panel-overlay, projection, and counting completions for the rule channel.

These cover the shape-changing families the reduction/completion search does not
reach: overlaying two or more panels while keeping their colours, projecting a
grid onto one axis, mapping colours by frequency rank, and emitting a bar whose
length counts something in the input.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterator, Sequence

import numpy as np

from latent_workspace_rule_parts import (
    MAX_SIDE,
    background_color,
    connected_components,
    split_panels,
)

IntArray = np.ndarray
GridMap = Callable[[IntArray], "IntArray | None"]
DemoPairs = Sequence[tuple[IntArray, IntArray]]
NamedRule = tuple[str, GridMap]


def family_panel_overlay(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield colour-preserving overlays of every equally shaped panel.

    Unlike the boolean panel combination, this keeps each panel's own colours
    and only decides which panel wins where two disagree.
    """

    background = background_color(pairs)
    for reverse in (False, True):
        for empty in sorted({background, 0}):

            def overlay(
                grid: IntArray, reverse=reverse, empty=empty
            ) -> IntArray | None:
                panels = split_panels(grid)
                if len(panels) < 2:
                    return None
                shape = panels[0].shape
                if any(panel.shape != shape for panel in panels):
                    return None
                ordered = list(reversed(panels)) if reverse else panels
                merged = np.full(shape, empty, dtype=grid.dtype)
                for panel in ordered:
                    merged = np.where(panel != empty, panel, merged)
                return merged

            yield f"overlay:{int(reverse)}:{empty}", overlay


def family_projection(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield per-row and per-column summaries of a grid onto a single line."""

    background = background_color(pairs)
    for axis in (0, 1):
        for criterion in ("unique", "mode", "first_filled", "any_filled"):

            def project(
                grid: IntArray, axis=axis, criterion=criterion
            ) -> IntArray | None:
                lines = grid if axis == 1 else grid.T
                values: list[int] = []
                for line in lines:
                    filled = line[line != background]
                    if criterion == "unique":
                        distinct = np.unique(filled)
                        if distinct.size != 1:
                            return None
                        values.append(int(distinct[0]))
                    elif criterion == "mode":
                        if not filled.size:
                            return None
                        counts = collections.Counter(filled.tolist())
                        values.append(int(counts.most_common(1)[0][0]))
                    elif criterion == "first_filled":
                        if not filled.size:
                            return None
                        values.append(int(filled[0]))
                    else:
                        values.append(int(bool(filled.size)))
                column = np.asarray(values, dtype=grid.dtype)
                return column[:, None] if axis == 1 else column[None, :]

            yield f"project:{axis}:{criterion}", project


def family_color_rank(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield recolourings that map each colour by its frequency rank."""

    if any(source.shape != target.shape for source, target in pairs):
        return
    for descending in (True, False):
        for ignore_background in (True, False):
            background = background_color(pairs) if ignore_background else -1
            table: dict[int, int] = {}
            consistent = True
            for source, target in pairs:
                counts = collections.Counter(
                    value for value in source.ravel().tolist() if value != background
                )
                ordered = [
                    color
                    for color, _ in sorted(
                        counts.items(),
                        key=lambda item: (-item[1] if descending else item[1], item[0]),
                    )
                ]
                for rank, color in enumerate(ordered):
                    painted = np.unique(target[source == color])
                    if painted.size != 1 or table.setdefault(
                        rank, int(painted[0])
                    ) != int(painted[0]):
                        consistent = False
                        break
                if not consistent:
                    break
            if not consistent or not table:
                continue

            def recolor(
                grid: IntArray,
                descending=descending,
                background=background,
                table=table,
            ) -> IntArray | None:
                counts = collections.Counter(
                    value for value in grid.ravel().tolist() if value != background
                )
                ordered = [
                    color
                    for color, _ in sorted(
                        counts.items(),
                        key=lambda item: (-item[1] if descending else item[1], item[0]),
                    )
                ]
                painted = grid.copy()
                for rank, color in enumerate(ordered):
                    if rank not in table:
                        return None
                    painted[grid == color] = table[rank]
                return painted

            yield f"rank:{int(descending)}:{int(ignore_background)}", recolor


_COUNTERS: dict[str, Callable[[IntArray, int], int]] = {
    "objects": lambda grid, background: len(
        connected_components(grid, background, diagonal=False, same_color=False)
    ),
    "objects8": lambda grid, background: len(
        connected_components(grid, background, diagonal=True, same_color=False)
    ),
    "colors": lambda grid, background: len(
        [value for value in np.unique(grid).tolist() if value != background]
    ),
    "filled": lambda grid, background: int((grid != background).sum()),
}


def family_count_bar(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield a single-line bar whose length counts a property of the input."""

    background = background_color(pairs)
    if any(min(target.shape) != 1 for _, target in pairs):
        return
    palette = sorted({int(value) for _, target in pairs for value in np.unique(target)})
    for label, counter in _COUNTERS.items():
        for vertical in (False, True):
            for paint in palette:

                def bar(
                    grid: IntArray,
                    counter=counter,
                    vertical=vertical,
                    paint=paint,
                ) -> IntArray | None:
                    length = counter(grid, background)
                    if not 1 <= length <= MAX_SIDE:
                        return None
                    line = np.full((length, 1) if vertical else (1, length), paint)
                    return line.astype(grid.dtype)

                yield f"count:{label}:{int(vertical)}:{paint}", bar


def family_palette_bar(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield a line listing the input's colours ordered by frequency."""

    background = background_color(pairs)
    if any(min(target.shape) != 1 for _, target in pairs):
        return
    for descending in (True, False):
        for vertical in (False, True):

            def listing(
                grid: IntArray, descending=descending, vertical=vertical
            ) -> IntArray | None:
                counts = collections.Counter(
                    value for value in grid.ravel().tolist() if value != background
                )
                if not counts or len(counts) > MAX_SIDE:
                    return None
                ordered = [
                    color
                    for color, _ in sorted(
                        counts.items(),
                        key=lambda item: (-item[1] if descending else item[1], item[0]),
                    )
                ]
                line = np.asarray(ordered, dtype=grid.dtype)
                return line[:, None] if vertical else line[None, :]

            yield f"palette:{int(descending)}:{int(vertical)}", listing


def family_uniform_fill(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield a constant-colour grid whose colour is chosen from the input."""

    background = background_color(pairs)
    for criterion in ("mode", "rarest", "unique_non_background"):
        for keep_shape in (True, False):
            shapes = {target.shape for _, target in pairs}
            if not keep_shape and len(shapes) != 1:
                continue
            fixed = None if keep_shape else shapes.pop()

            def fill(
                grid: IntArray, criterion=criterion, fixed=fixed
            ) -> IntArray | None:
                counts = collections.Counter(
                    value for value in grid.ravel().tolist() if value != background
                )
                if not counts:
                    return None
                if criterion == "mode":
                    color = counts.most_common(1)[0][0]
                elif criterion == "rarest":
                    color = min(counts.items(), key=lambda item: (item[1], item[0]))[0]
                else:
                    if len(counts) != 1:
                        return None
                    color = next(iter(counts))
                shape = grid.shape if fixed is None else fixed
                return np.full(shape, color, dtype=grid.dtype)

            yield f"uniform:{criterion}:{int(keep_shape)}", fill


SHAPE_FAMILIES: tuple[Callable[[DemoPairs], Iterator[NamedRule]], ...] = (
    family_panel_overlay,
    family_projection,
    family_color_rank,
    family_count_bar,
    family_palette_bar,
    family_uniform_fill,
)
