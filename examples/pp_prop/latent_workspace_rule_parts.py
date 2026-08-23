"""Grid primitives shared by the demonstration-verified rule families.

Pure NumPy helpers: dihedral maps, connected components, lattice and panel
decomposition, periodic repair. Nothing here inspects a query target, and
nothing here depends on the spiking substrate.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence

import numpy as np

IntArray = np.ndarray

#: Ordered dihedral maps of the square symmetry group, identity first.
DIHEDRAL_NAMES: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)

MAX_SIDE = 30
COLOR_COUNT = 10

_DIHEDRAL_OPS = {
    "identity": lambda grid: grid,
    "rot90": lambda grid: np.rot90(grid, k=-1),
    "rot180": lambda grid: np.rot90(grid, k=2),
    "rot270": lambda grid: np.rot90(grid, k=1),
    "flip_horizontal": np.fliplr,
    "flip_vertical": np.flipud,
    "transpose": lambda grid: grid.T,
    "anti_transpose": lambda grid: np.rot90(grid, k=2).T,
}


def apply_dihedral(grid: IntArray, name: str) -> IntArray:
    """Apply one named dihedral map to a grid.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.
    Name
        Entry of :data:`DIHEDRAL_NAMES`.

    Returns
    -------
    numpy.ndarray
        Transformed grid, always a fresh contiguous array.

    Raises
    ------
    ValueError
        If ``name`` is not a known dihedral map.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> apply_dihedral(np.array([[1, 2], [3, 4]]), "rot90").tolist()
        [[3, 1], [4, 2]]

        >>> apply_dihedral(np.array([[1, 2], [3, 4]]), "transpose").tolist()
        [[1, 3], [2, 4]]
    """

    operation = _DIHEDRAL_OPS.get(name)
    if operation is None:
        raise ValueError(f"Unknown dihedral map: {name!r}. Set the named field to one of the supported values, then rerun the operation.")
    return np.ascontiguousarray(operation(grid))


def is_valid_grid(grid: IntArray | None) -> bool:
    """Return whether a proposed array is a legal ARC grid.

    Parameters
    ----------
    grid
        Candidate array, or ``None``.

    Returns
    -------
    bool
        True when the grid is 2-D, within 1..30 on both sides, and uses only
        colours 0..9.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> is_valid_grid(np.array([[0, 9]]))
        True

        >>> is_valid_grid(np.zeros((0, 3), dtype=int))
        False
    """

    if grid is None or getattr(grid, "ndim", 0) != 2 or grid.size == 0:
        return False
    return (
        1 <= grid.shape[0] <= MAX_SIDE
        and 1 <= grid.shape[1] <= MAX_SIDE
        and int(grid.min()) >= 0
        and int(grid.max()) < COLOR_COUNT
    )


def background_color(pairs: Sequence[tuple[IntArray, IntArray]]) -> int:
    """Return the most common colour across every demonstration input.

    Parameters
    ----------
    pairs
        Ordered ``(input, output)`` demonstration grids.

    Returns
    -------
    int
        Modal colour, or ``0`` when no cells are present.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> background_color([(np.array([[0, 0, 3]]), np.array([[3]]))])
        0
    """

    counts: collections.Counter[int] = collections.Counter()
    for source, _ in pairs:
        counts.update(source.ravel().tolist())
    return int(counts.most_common(1)[0][0]) if counts else 0


def connected_components(
    grid: IntArray, background: int, *, diagonal: bool, same_color: bool
) -> list[IntArray]:
    """Label connected non-background regions of a grid.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.
    Background
        Colour treated as empty space.
    Diagonal
        Whether diagonal adjacency joins two cells.
    same_color
        Whether adjacency additionally requires equal colour.

    Returns
    -------
    list of numpy.ndarray
        Boolean masks, one per component, in row-major discovery order.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> grid = np.array([[1, 0], [0, 2]])
        >>> len(connected_components(grid, 0, diagonal=False, same_color=False))
        2

        >>> len(connected_components(grid, 0, diagonal=True, same_color=False))
        1
    """

    height, width = grid.shape
    filled = grid != background
    seen = np.zeros_like(filled)
    offsets = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if not diagonal
        else ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    )
    components: list[IntArray] = []
    for row in range(height):
        for column in range(width):
            if not filled[row, column] or seen[row, column]:
                continue
            mask = np.zeros_like(filled)
            stack = [(row, column)]
            seen[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                mask[current_row, current_column] = True
                for delta_row, delta_column in offsets:
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    if not (0 <= next_row < height and 0 <= next_column < width):
                        continue
                    if seen[next_row, next_column] or not filled[next_row, next_column]:
                        continue
                    if same_color and grid[next_row, next_column] != grid[current_row, current_column]:
                        continue
                    seen[next_row, next_column] = True
                    stack.append((next_row, next_column))
            components.append(mask)
    return components


def crop_to_mask(grid: IntArray, mask: IntArray) -> IntArray:
    """Crop a grid to the bounding box of a boolean mask.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.
    Mask
        Boolean array with the same shape as ``grid``.

    Returns
    -------
    numpy.ndarray
        The bounding-box crop.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> grid = np.array([[0, 0, 0], [0, 5, 0]])
        >>> crop_to_mask(grid, grid == 5).tolist()
        [[5]]
    """

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return np.ascontiguousarray(
        grid[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    )


def separator_indices(grid: IntArray, axis: int) -> IntArray:
    """Return indices of fully uniform lines along one axis.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.
    Axis
        ``0`` to scan rows, ``1`` to scan columns.

    Returns
    -------
    numpy.ndarray
        Indices whose entire line holds a single colour.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> grid = np.array([[1, 2], [5, 5]])
        >>> separator_indices(grid, 0).tolist()
        [1]
    """

    moved = grid if axis == 0 else grid.T
    uniform = (moved == moved[:, :1]).all(axis=1)
    return np.flatnonzero(uniform)


def split_panels(grid: IntArray) -> list[IntArray]:
    """Split a grid on uniform separator lines shared by rows and columns.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.

    Returns
    -------
    list of numpy.ndarray
        Non-empty panels in row-major order. A grid with no interior separator
        returns itself as the single panel.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> grid = np.array([[1, 5, 2], [3, 5, 4]])
        >>> [panel.tolist() for panel in split_panels(grid)]
        [[[1], [3]], [[2], [4]]]
    """

    panels = [grid]
    for axis in (0, 1):
        expanded: list[IntArray] = []
        for panel in panels:
            cuts = separator_indices(panel, axis)
            if cuts.size == 0 or cuts.size == panel.shape[axis]:
                expanded.append(panel)
                continue
            keep = np.ones(panel.shape[axis], dtype=bool)
            keep[cuts] = False
            groups = np.split(np.flatnonzero(keep), np.flatnonzero(np.diff(np.flatnonzero(keep)) > 1) + 1)
            for group in groups:
                if group.size:
                    expanded.append(np.ascontiguousarray(panel.take(group, axis=axis)))
        panels = expanded
    return [panel for panel in panels if panel.size]


def lattice_cells(grid: IntArray) -> tuple[list[list[IntArray]], int] | None:
    """Decompose a separator lattice into a rectangular array of cells.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.

    Returns
    -------
    tuple or None
        ``(cells, separator_colour)`` where ``cells`` is a rectangular list of
        lists, or ``None`` when the grid is not a clean lattice.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> grid = np.array([[1, 5, 2], [5, 5, 5], [3, 5, 4]])
        >>> cells, color = lattice_cells(grid)
        >>> color, len(cells), len(cells[0])
        (5, 2, 2)
    """

    row_cuts = separator_indices(grid, 0)
    column_cuts = separator_indices(grid, 1)
    if row_cuts.size == 0 and column_cuts.size == 0:
        return None
    if row_cuts.size == grid.shape[0] or column_cuts.size == grid.shape[1]:
        return None
    colors = {int(value) for value in grid[row_cuts].ravel()}
    colors |= {int(value) for value in grid[:, column_cuts].ravel()}
    if len(colors) != 1:
        return None
    row_groups = _index_groups(grid.shape[0], row_cuts)
    column_groups = _index_groups(grid.shape[1], column_cuts)
    if not row_groups or not column_groups:
        return None
    cells = [
        [
            np.ascontiguousarray(grid[np.ix_(rows, columns)])
            for columns in column_groups
        ]
        for rows in row_groups
    ]
    return cells, colors.pop()


def _index_groups(length: int, cuts: IntArray) -> list[IntArray]:
    """Return contiguous index runs of ``range(length)`` excluding ``cuts``."""

    keep = np.ones(length, dtype=bool)
    keep[cuts] = False
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return [group for group in np.split(indices, breaks) if group.size]


def periodic_fill(grid: IntArray, noise: int) -> IntArray | None:
    """Repair cells of one colour by the grid's smallest exact 2-D period.

    Parameters
    ----------
    grid
        Two-dimensional integer grid.
    Noise
        Colour treated as unknown and repaired.

    Returns
    -------
    numpy.ndarray or None
        Fully repaired grid, or ``None`` when no period explains every known
        cell or the repair leaves unknowns.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> grid = np.array([[1, 2, 1, 2], [1, 2, 9, 2]])
        >>> periodic_fill(grid, 9).tolist()
        [[1, 2, 1, 2], [1, 2, 1, 2]]
    """

    height, width = grid.shape
    known = grid != noise
    if not known.any() or known.all():
        return None
    rows, columns = np.indices(grid.shape)
    for period_height in range(1, height + 1):
        for period_width in range(1, width + 1):
            if period_height == height and period_width == width:
                continue
            keys = (rows % period_height) * period_width + (columns % period_width)
            palette = np.full(period_height * period_width, -1, dtype=np.int32)
            flat_keys = keys[known]
            flat_values = grid[known].astype(np.int32)
            palette[flat_keys] = flat_values
            if not np.array_equal(palette[flat_keys], flat_values):
                continue
            if (palette < 0).any():
                continue
            return np.ascontiguousarray(palette[keys].astype(grid.dtype))
    return None
