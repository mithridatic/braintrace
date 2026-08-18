"""Same-shape cell rules for the verified rule channel.

Sixty-nine percent of the unsolved ARC-AGI-1 evaluation tasks keep the input
shape and edit it, so these families carry most of the channel's remaining
recall. Each is fitted to the demonstrations and admitted only by the caller's
exact-reproduction check.

Every family here declines to propose when the query would take it outside the
evidence it was fitted on: an unseen neighbourhood key, an empty source set, or
a shape change all return ``None`` rather than a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from latent_workspace_rule_parts import COLOR_COUNT, background_color

IntArray = np.ndarray
GridMap = Callable[[IntArray], "IntArray | None"]
DemoPairs = Sequence[tuple[IntArray, IntArray]]
NamedRule = tuple[str, GridMap]

#: Sentinel colour standing for a cell outside the grid.
OUT_OF_BOUNDS = COLOR_COUNT
_KEY_BASE = COLOR_COUNT + 1


def shift(grid: IntArray, row_delta: int, column_delta: int) -> IntArray:
    """Translate a grid by one offset, filling vacated cells as out-of-bounds.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.
    row_delta
        Downward shift in cells; negative shifts up.
    column_delta
        Rightward shift in cells; negative shifts left.

    Returns
    -------
    numpy.ndarray
        Shifted grid of the same shape, vacated cells set to
        :data:`OUT_OF_BOUNDS`.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> shift(np.array([[1, 2], [3, 4]]), 1, 0).tolist()
        [[10, 10], [1, 2]]

        >>> shift(np.array([[1, 2], [3, 4]]), 0, -1).tolist()
        [[2, 10], [4, 10]]
    """

    moved = np.full_like(grid, OUT_OF_BOUNDS)
    height, width = grid.shape
    if abs(row_delta) >= height or abs(column_delta) >= width:
        return moved
    source_rows = slice(max(0, -row_delta), height - max(0, row_delta))
    source_columns = slice(max(0, -column_delta), width - max(0, column_delta))
    target_rows = slice(max(0, row_delta), height - max(0, -row_delta))
    target_columns = slice(max(0, column_delta), width - max(0, -column_delta))
    moved[target_rows, target_columns] = grid[source_rows, source_columns]
    return moved


_NEIGHBOURHOODS: dict[str, tuple[tuple[int, int], ...]] = {
    "n4": ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    "n8": (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ),
}


def _neighbourhood_keys(
    grid: IntArray, offsets: tuple[tuple[int, int], ...]
) -> IntArray:
    """Encode each cell's neighbourhood colours as one integer key."""

    keys = np.zeros(grid.shape, dtype=np.int64)
    for power, (row_delta, column_delta) in enumerate(offsets):
        keys += shift(grid, row_delta, column_delta).astype(np.int64) * (
            _KEY_BASE**power
        )
    return keys


def _parity_keys(grid: IntArray, period: int) -> IntArray:
    """Encode each cell's colour together with its position modulo a period."""

    rows, columns = np.indices(grid.shape)
    return (
        grid.astype(np.int64) * period * period
        + (rows % period) * period
        + (columns % period)
    )


def _line_keys(grid: IntArray) -> IntArray:
    """Encode each cell's colour with the colour sets of its row and column."""

    row_mask = np.zeros(grid.shape[0], dtype=np.int64)
    column_mask = np.zeros(grid.shape[1], dtype=np.int64)
    for color in range(COLOR_COUNT):
        present = grid == color
        row_mask |= present.any(axis=1).astype(np.int64) << color
        column_mask |= present.any(axis=0).astype(np.int64) << color
    return (
        grid.astype(np.int64) * (1 << 20)
        + row_mask[:, None] * (1 << 10)
        + column_mask[None, :]
    )


KEY_BUILDERS: dict[str, Callable[[IntArray], IntArray]] = {
    "n4": lambda grid: _neighbourhood_keys(grid, _NEIGHBOURHOODS["n4"]),
    "n8": lambda grid: _neighbourhood_keys(grid, _NEIGHBOURHOODS["n8"]),
    "par2": lambda grid: _parity_keys(grid, 2),
    "par3": lambda grid: _parity_keys(grid, 3),
    "lines": _line_keys,
}


def family_local_rule(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield exact per-cell rules keyed on a cell's local or positional context.

    Every key seen in a demonstration is bound to the output colour observed
    there, and a conflicting binding rejects the whole rule. At apply time a key
    the demonstrations never showed produces no proposal, so the family cannot
    extrapolate past its own evidence.
    """

    if any(source.shape != target.shape for source, target in pairs):
        return
    for label, builder in KEY_BUILDERS.items():
        table: dict[int, int] = {}
        consistent = True
        for source, target in pairs:
            for key, value in zip(
                builder(source).ravel().tolist(), target.ravel().tolist()
            ):
                if table.setdefault(key, value) != value:
                    consistent = False
                    break
            if not consistent:
                break
        if not consistent or not table:
            continue

        def relabel(grid: IntArray, builder=builder, table=table) -> IntArray | None:
            flat = builder(grid).ravel().tolist()
            if any(key not in table for key in flat):
                return None
            return np.asarray(
                [table[key] for key in flat], dtype=grid.dtype
            ).reshape(grid.shape)

        yield f"cell:{label}", relabel


RAY_DIRECTIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((-1, 0),),
    "down": ((1, 0),),
    "left": ((0, -1),),
    "right": ((0, 1),),
    "vertical": ((-1, 0), (1, 0)),
    "horizontal": ((0, -1), (0, 1)),
    "cross": ((-1, 0), (1, 0), (0, -1), (0, 1)),
    "diagonal": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}


def family_rays(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield rays cast from every non-background cell to an edge or obstacle."""

    if any(source.shape != target.shape for source, target in pairs):
        return
    background = background_color(pairs)
    for label, directions in RAY_DIRECTIONS.items():
        for blocked in (False, True):

            def cast(
                grid: IntArray, directions=directions, blocked=blocked
            ) -> IntArray | None:
                height, width = grid.shape
                sources = np.argwhere(grid != background)
                if not sources.size or sources.shape[0] * 2 > height * width:
                    return None
                painted = grid.copy()
                for row, column in sources.tolist():
                    color = int(grid[row, column])
                    for row_delta, column_delta in directions:
                        step_row = row + row_delta
                        step_column = column + column_delta
                        while 0 <= step_row < height and 0 <= step_column < width:
                            if grid[step_row, step_column] != background:
                                if blocked:
                                    break
                            else:
                                painted[step_row, step_column] = color
                            step_row += row_delta
                            step_column += column_delta
                return painted

            yield f"ray:{label}:{int(blocked)}", cast


def family_connect(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield straight connections between aligned cells that share a colour."""

    if any(source.shape != target.shape for source, target in pairs):
        return
    background = background_color(pairs)
    palette = sorted({int(value) for _, target in pairs for value in np.unique(target)})
    for label, paint in [("same", None), *((str(c), c) for c in palette)]:

        def connect(grid: IntArray, paint=paint) -> IntArray | None:
            painted = grid.copy()
            drew = False
            for color in np.unique(grid).tolist():
                if int(color) == background:
                    continue
                positions = np.argwhere(grid == color).tolist()
                for index, (row, column) in enumerate(positions):
                    for other_row, other_column in positions[index + 1 :]:
                        if row == other_row:
                            low, high = sorted((column, other_column))
                            cells = [(row, value) for value in range(low + 1, high)]
                        elif column == other_column:
                            low, high = sorted((row, other_row))
                            cells = [(value, column) for value in range(low + 1, high)]
                        else:
                            continue
                        if not cells or any(grid[cell] != background for cell in cells):
                            continue
                        for cell in cells:
                            painted[cell] = color if paint is None else paint
                        drew = True
            return painted if drew else None

        yield f"connect:{label}", connect


def family_translate(pairs: DemoPairs) -> Iterator[NamedRule]:
    """Yield rigid translation of the whole grid, fitted from one demonstration."""

    if any(source.shape != target.shape for source, target in pairs):
        return
    background = background_color(pairs)
    source, target = pairs[0]
    height, width = source.shape
    for row_delta in range(-height + 1, height):
        for column_delta in range(-width + 1, width):
            if (row_delta, column_delta) == (0, 0):
                continue
            moved = shift(source, row_delta, column_delta)
            moved = np.where(moved == OUT_OF_BOUNDS, background, moved)
            if not np.array_equal(moved, target):
                continue

            def translate(
                grid: IntArray, row_delta=row_delta, column_delta=column_delta
            ) -> IntArray | None:
                shifted = shift(grid, row_delta, column_delta)
                return np.where(shifted == OUT_OF_BOUNDS, background, shifted).astype(
                    grid.dtype
                )

            yield f"move:{row_delta}:{column_delta}", translate


CELL_FAMILIES: tuple[Callable[[DemoPairs], Iterator[NamedRule]], ...] = (
    family_local_rule,
    family_rays,
    family_connect,
    family_translate,
)
