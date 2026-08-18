"""Same-shape and object-level completions for the verified rule channel.

Each family is a generator over ``(name, apply)`` pairs fitted to demonstration
grids. Admission against the demonstrations is the caller's job, so a family may
over-propose freely: a proposal that does not reproduce every pair is discarded.

These are the completions whose output is an *edit* of the input rather than a
geometric re-expression of it: object recolouring, panel composition, gravity,
border edits, enclosure filling, and periodic extension.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterator, Sequence

import numpy as np

from latent_workspace_rule_parts import (
    COLOR_COUNT,
    MAX_SIDE,
    background_color,
    connected_components,
    crop_to_mask,
    split_panels,
)

IntArray = np.ndarray
GridMap = Callable[[IntArray], "IntArray | None"]
DemoPairs = Sequence[tuple[IntArray, IntArray]]
NamedRule = tuple[str, GridMap]

#: Connectivity conventions indexed by the ``variant`` argument.
CONNECTIVITY: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


def _components(grid: IntArray, background: int, variant: int) -> list[IntArray]:
    """Return connected-component masks under one connectivity convention."""

    diagonal, same_color = CONNECTIVITY[variant]
    return connected_components(
        grid, background, diagonal=diagonal, same_color=same_color
    )


def _object_key(grid: IntArray, mask: IntArray, property_name: str) -> object:
    """Return the fitted grouping key of one object under a named property."""

    if property_name == "size":
        return int(mask.sum())
    if property_name == "bbox":
        patch = crop_to_mask(mask.astype(np.int8), mask)
        return patch.shape
    if property_name == "shape":
        patch = crop_to_mask(mask.astype(np.int8), mask)
        return patch.tobytes() + bytes(patch.shape)
    if property_name == "color":
        values = np.unique(grid[mask])
        return int(values[0]) if values.size == 1 else None
    if property_name == "holes":
        patch = crop_to_mask(mask.astype(np.int8), mask)
        inverted = 1 - patch
        enclosed = _enclosed_mask(inverted.astype(bool))
        return int(
            len(
                connected_components(
                    enclosed.astype(np.int32), 0, diagonal=False, same_color=False
                )
            )
        )
    raise ValueError(f"unknown object property: {property_name!r}")


def _enclosed_mask(free: IntArray) -> IntArray:
    """Return the cells of ``free`` unreachable from the grid border."""

    height, width = free.shape
    reachable = np.zeros_like(free)
    stack = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if (row in (0, height - 1) or column in (0, width - 1)) and free[row, column]
    ]
    for row, column in stack:
        reachable[row, column] = True
    while stack:
        row, column = stack.pop()
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_column = row + delta_row, column + delta_column
            if not (0 <= next_row < height and 0 <= next_column < width):
                continue
            if reachable[next_row, next_column] or not free[next_row, next_column]:
                continue
            reachable[next_row, next_column] = True
            stack.append((next_row, next_column))
    return free & ~reachable


def _fit_object_table(
    pairs: DemoPairs, background: int, variant: int, property_name: str
) -> dict[object, int] | None:
    """Fit an object-property to output-colour table across all demonstrations."""

    table: dict[object, int] = {}
    for source, target in pairs:
        if source.shape != target.shape:
            return None
        masks = _components(source, background, variant)
        if not masks:
            return None
        covered = np.zeros(source.shape, dtype=bool)
        for mask in masks:
            key = _object_key(source, mask, property_name)
            if key is None:
                return None
            painted = np.unique(target[mask])
            if painted.size != 1:
                return None
            if table.setdefault(key, int(painted[0])) != int(painted[0]):
                return None
            covered |= mask
        if not np.array_equal(source[~covered], target[~covered]):
            return None
    return table


def family_object_recolor(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield same-shape recolourings of each object by a fitted property."""

    background = background_color(pairs)
    for variant in range(len(CONNECTIVITY)):
        for property_name in ("size", "bbox", "shape", "color", "holes"):
            table = _fit_object_table(pairs, background, variant, property_name)
            if not table:
                continue

            def recolor(
                grid: IntArray,
                variant=variant,
                property_name=property_name,
                table=table,
            ) -> IntArray | None:
                painted = grid.copy()
                for mask in _components(grid, background, variant):
                    key = _object_key(grid, mask, property_name)
                    if key not in table:
                        return None
                    painted[mask] = table[key]
                return painted

            yield f"objcol{variant}:{property_name}", recolor


def family_object_rank_recolor(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield same-shape recolourings of each object by its size rank."""

    background = background_color(pairs)
    for variant in range(len(CONNECTIVITY)):
        for descending in (True, False):
            table: dict[int, int] = {}
            consistent = True
            for source, target in pairs:
                if source.shape != target.shape:
                    consistent = False
                    break
                masks = _components(source, background, variant)
                ordered = sorted(
                    masks, key=lambda mask: int(mask.sum()), reverse=descending
                )
                covered = np.zeros(source.shape, dtype=bool)
                for rank, mask in enumerate(ordered):
                    painted = np.unique(target[mask])
                    if painted.size != 1 or table.setdefault(
                        rank, int(painted[0])
                    ) != int(painted[0]):
                        consistent = False
                        break
                    covered |= mask
                if not consistent or not np.array_equal(
                    source[~covered], target[~covered]
                ):
                    consistent = False
                    break
            if not consistent or not table:
                continue

            def recolor(
                grid: IntArray, variant=variant, descending=descending, table=table
            ) -> IntArray | None:
                masks = _components(grid, background, variant)
                ordered = sorted(
                    masks, key=lambda mask: int(mask.sum()), reverse=descending
                )
                painted = grid.copy()
                for rank, mask in enumerate(ordered):
                    if rank not in table:
                        return None
                    painted[mask] = table[rank]
                return painted

            yield f"objrank{variant}:{int(descending)}", recolor


def family_panel_combine(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield cell-wise boolean combinations of two equally shaped panels."""

    background = background_color(pairs)
    operations: dict[str, Callable[[IntArray, IntArray], IntArray]] = {
        "and": lambda a, b: a & b,
        "or": lambda a, b: a | b,
        "xor": lambda a, b: a ^ b,
        "nand": lambda a, b: ~(a & b),
        "nor": lambda a, b: ~(a | b),
        "xnor": lambda a, b: ~(a ^ b),
        "diff": lambda a, b: a & ~b,
        "rdiff": lambda a, b: b & ~a,
    }
    palette = sorted({int(v) for _, target in pairs for v in np.unique(target)})
    if len(palette) > 3:
        return
    for label, operation in operations.items():
        for paint in palette:
            if paint == background:
                continue

            def combine(
                grid: IntArray, operation=operation, paint=paint
            ) -> IntArray | None:
                panels = split_panels(grid)
                if len(panels) != 2 or panels[0].shape != panels[1].shape:
                    return None
                merged = operation(panels[0] != background, panels[1] != background)
                return np.where(merged, paint, background).astype(grid.dtype)

            yield f"pcomb:{label}:{paint}", combine


def family_gravity(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield edge-ward compaction of the non-background cells."""

    background = background_color(pairs)
    for axis in (0, 1):
        for reverse in (False, True):

            def fall(
                grid: IntArray, axis=axis, reverse=reverse
            ) -> IntArray | None:
                moved = grid if axis == 0 else grid.T
                out = np.full_like(moved, background)
                for index in range(moved.shape[1]):
                    line = moved[:, index]
                    values = line[line != background]
                    if reverse:
                        out[: values.size, index] = values
                    elif values.size:
                        out[moved.shape[0] - values.size :, index] = values
                return out if axis == 0 else np.ascontiguousarray(out.T)

            yield f"grav{axis}{int(reverse)}", fall


def family_border(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield fixed-width border addition and removal."""

    source, target = pairs[0]
    rows = target.shape[0] - source.shape[0]
    columns = target.shape[1] - source.shape[1]
    if rows == columns and rows > 0 and rows % 2 == 0:
        width = rows // 2
        for paint in range(COLOR_COUNT):

            def pad(grid: IntArray, width=width, paint=paint) -> IntArray | None:
                if grid.shape[0] + 2 * width > MAX_SIDE:
                    return None
                if grid.shape[1] + 2 * width > MAX_SIDE:
                    return None
                return np.pad(grid, width, constant_values=paint)

            yield f"pad{width}:{paint}", pad
    if rows == columns and rows < 0 and rows % 2 == 0:
        width = -rows // 2

        def trim(grid: IntArray, width=width) -> IntArray | None:
            if grid.shape[0] <= 2 * width or grid.shape[1] <= 2 * width:
                return None
            return np.ascontiguousarray(grid[width:-width, width:-width])

        yield f"trim{width}", trim


def family_fill_enclosed(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield filling of empty regions not reachable from the border.

    The colour treated as empty is not assumed to be the modal one. In a
    hole-filling task the hole is by construction the minority colour, so both
    the modal colour and colour ``0`` are tried as the empty marker.
    """

    palette = sorted({int(v) for _, target in pairs for v in np.unique(target)})
    for empty in sorted({background_color(pairs), 0}):
        for paint in palette:
            if paint == empty:
                continue

            def fill(grid: IntArray, empty=empty, paint=paint) -> IntArray | None:
                enclosed = _enclosed_mask(grid == empty)
                if not enclosed.any():
                    return None
                filled = grid.copy()
                filled[enclosed] = paint
                return filled

            yield f"fill{empty}:{paint}", fill


def family_periodic_extend(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield periodic extension of the input to a fitted constant output shape."""

    shapes = {target.shape for _, target in pairs}
    ratios = {
        (target.shape[0] / source.shape[0], target.shape[1] / source.shape[1])
        for source, target in pairs
    }
    if len(shapes) == 1:
        height, width = shapes.pop()

        def extend(grid: IntArray, height=height, width=width) -> IntArray | None:
            repeats = (
                -(-height // grid.shape[0]),
                -(-width // grid.shape[1]),
            )
            return np.ascontiguousarray(np.tile(grid, repeats)[:height, :width])

        yield f"pext:{height}x{width}", extend
    if len(ratios) == 1:
        ratio = ratios.pop()

        def scale(grid: IntArray, ratio=ratio) -> IntArray | None:
            height = int(round(grid.shape[0] * ratio[0]))
            width = int(round(grid.shape[1] * ratio[1]))
            if not 1 <= height <= MAX_SIDE or not 1 <= width <= MAX_SIDE:
                return None
            repeats = (-(-height // grid.shape[0]), -(-width // grid.shape[1]))
            return np.ascontiguousarray(np.tile(grid, repeats)[:height, :width])

        yield f"pscale:{ratio[0]:g}x{ratio[1]:g}", scale


def family_majority_cell(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield selection of the most or least frequent lattice cell."""

    for label, most in (("most", True), ("least", False)):

        def select(grid: IntArray, most=most) -> IntArray | None:
            panels = split_panels(grid)
            if len(panels) < 2:
                return None
            shape = panels[0].shape
            if any(panel.shape != shape for panel in panels):
                return None
            counts = collections.Counter(panel.tobytes() for panel in panels)
            chosen = (max if most else min)(counts.items(), key=lambda item: item[1])
            return next(
                panel.copy() for panel in panels if panel.tobytes() == chosen[0]
            )

        yield f"panel_freq:{label}", select


EDIT_FAMILIES: tuple[Callable[[DemoPairs], Iterator[NamedRule]], ...] = (
    family_object_recolor,
    family_object_rank_recolor,
    family_panel_combine,
    family_gravity,
    family_border,
    family_fill_enclosed,
    family_periodic_extend,
    family_majority_cell,
)
