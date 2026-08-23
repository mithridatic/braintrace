"""Per-cell ARC grid features for the demonstration-fitted answer head.

The shipped answer head reads the LIF carrier plus the single query row being
transcribed, so no transformation with two-dimensional structure is
representable and the only reachable optimum is copying the input.  This module
supplies what that head is missing: for every cell of a grid, its local colour
patch, the colour statistics of its row, column and grid, the colours of its
mirror images under the grid's symmetries, the nearest non-background colour
along each of the four axis rays, the colour it would hold under the grid's
dominant period, and its position relative to the grid's edges.

Measured on the ARC-AGI-1 evaluation split, a per-cell head fitted on a task's
own demonstration cells answers 2 queries over 1 task from the patch and colour
statistics alone, and 10 queries over 8 tasks once the mirror, ray and period
blocks are present -- the difference between those two feature sets is the
entire reason this module exists.

Every block is index arithmetic, an equality reduction or a cumulative maximum,
so the whole map lowers into a compiled refinement sweep.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


MAX_GRID_SIZE = 30
COLOR_COUNT = 10
PAD_COLOR = 10
PATCH_COLORS = COLOR_COUNT + 1
PATCH_RADIUS = 2
PATCH_CELLS = (2 * PATCH_RADIUS + 1) ** 2
MAX_PERIOD = 12

_MIRROR_COUNT = 4
_PERIOD_VIEWS = 3
_RAY_COUNT = 4
_SCALAR_COUNT = 22
_STATISTIC_WIDTH = 60


def cell_feature_width() -> int:
    """Return the per-cell feature width produced by :func:`cell_features`.

    Returns
    -------
    int
        Total feature count for one cell.

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_cell_features import cell_feature_width
        >>> cell_feature_width()
        482
    """

    return (
        PATCH_CELLS * PATCH_COLORS
        + _STATISTIC_WIDTH
        + (_MIRROR_COUNT + _PERIOD_VIEWS + _RAY_COUNT) * PATCH_COLORS
        + _RAY_COUNT
        + _SCALAR_COUNT
    )


def _gather_cells(grid: jax.Array, rows: jax.Array, columns: jax.Array) -> jax.Array:
    """Read ``grid`` at integer coordinates, returning ``PAD_COLOR`` off-grid."""

    inside = (
        (rows >= 0)
        & (rows < MAX_GRID_SIZE)
        & (columns >= 0)
        & (columns < MAX_GRID_SIZE)
    )
    safe_rows = jnp.clip(rows, 0, MAX_GRID_SIZE - 1)
    safe_columns = jnp.clip(columns, 0, MAX_GRID_SIZE - 1)
    values = jnp.take_along_axis(
        grid.reshape(grid.shape[0], -1),
        (safe_rows * MAX_GRID_SIZE + safe_columns).reshape(grid.shape[0], -1),
        axis=1,
    ).reshape(rows.shape)
    return jnp.where(inside, values, PAD_COLOR)


def _local_patches(grid: jax.Array) -> jax.Array:
    """Return the ``(2r+1)^2`` neighbourhood colours of every cell."""

    offsets = jnp.arange(-PATCH_RADIUS, PATCH_RADIUS + 1)
    coordinates = jnp.arange(MAX_GRID_SIZE)
    rows = coordinates[None, :, None, None, None] + offsets[None, None, None, :, None]
    columns = (
        coordinates[None, None, :, None, None] + offsets[None, None, None, None, :]
    )
    rows = jnp.broadcast_to(
        rows,
        (grid.shape[0], MAX_GRID_SIZE, MAX_GRID_SIZE, 2 * PATCH_RADIUS + 1, 2 * PATCH_RADIUS + 1),
    )
    columns = jnp.broadcast_to(columns, rows.shape)
    return _gather_cells(grid, rows, columns).reshape(
        grid.shape[0], MAX_GRID_SIZE, MAX_GRID_SIZE, PATCH_CELLS
    )


def _colour_statistics(
    grid: jax.Array, valid: jax.Array, height: jax.Array, width: jax.Array
) -> jax.Array:
    """Return row, column and grid colour presence and normalised counts."""

    one_hot = jax.nn.one_hot(jnp.clip(grid, 0, COLOR_COUNT - 1), COLOR_COUNT)
    one_hot = one_hot * valid[..., None]
    row_counts = jnp.sum(one_hot, axis=2)
    column_counts = jnp.sum(one_hot, axis=1)
    grid_counts = jnp.sum(row_counts, axis=1)
    row_block = jnp.concatenate(
        [
            (row_counts > 0).astype(grid_counts.dtype),
            row_counts / jnp.maximum(width, 1)[:, None, None],
        ],
        axis=-1,
    )
    column_block = jnp.concatenate(
        [
            (column_counts > 0).astype(grid_counts.dtype),
            column_counts / jnp.maximum(height, 1)[:, None, None],
        ],
        axis=-1,
    )
    grid_block = jnp.concatenate(
        [
            (grid_counts > 0).astype(grid_counts.dtype),
            grid_counts / jnp.maximum(height * width, 1)[:, None],
        ],
        axis=-1,
    )
    shape = (grid.shape[0], MAX_GRID_SIZE, MAX_GRID_SIZE, 2 * COLOR_COUNT)
    return jnp.concatenate(
        [
            jnp.broadcast_to(row_block[:, :, None, :], shape),
            jnp.broadcast_to(column_block[:, None, :, :], shape),
            jnp.broadcast_to(grid_block[:, None, None, :], shape),
        ],
        axis=-1,
    )


def _mirror_colours(
    grid: jax.Array, rows: jax.Array, columns: jax.Array, height: jax.Array, width: jax.Array
) -> jax.Array:
    """Return the colours of a cell's images under the grid's symmetries."""

    flipped_rows = height[:, None, None] - 1 - rows
    flipped_columns = width[:, None, None] - 1 - columns
    square = (height == width)[:, None, None]
    transposed = _gather_cells(grid, columns, rows)
    return jnp.stack(
        [
            _gather_cells(grid, flipped_rows, columns),
            _gather_cells(grid, rows, flipped_columns),
            _gather_cells(grid, flipped_rows, flipped_columns),
            jnp.where(square, transposed, PAD_COLOR),
        ],
        axis=-1,
    )


def _axis_period(grid: jax.Array, valid: jax.Array, extent: jax.Array, axis: int) -> jax.Array:
    """Return the smallest shift up to ``MAX_PERIOD`` under which ``grid`` repeats."""

    shifts = jnp.arange(1, MAX_PERIOD + 1)
    coordinates = jnp.arange(MAX_GRID_SIZE)
    source = coordinates[None, :] + shifts[:, None]
    within = source < MAX_GRID_SIZE
    indices = jnp.clip(source, 0, MAX_GRID_SIZE - 1).reshape(-1)
    shifted = jnp.take(grid, indices, axis=axis)
    shifted_valid = jnp.take(valid, indices, axis=axis)
    batch = grid.shape[0]
    if axis == 1:
        block = (batch, MAX_PERIOD, MAX_GRID_SIZE, MAX_GRID_SIZE)
        base = grid[:, None]
        base_valid = valid[:, None]
        keep = within.reshape(1, MAX_PERIOD, MAX_GRID_SIZE, 1)
        reduce_axes = (2, 3)
    else:
        block = (batch, MAX_GRID_SIZE, MAX_PERIOD, MAX_GRID_SIZE)
        base = grid[:, :, None]
        base_valid = valid[:, :, None]
        keep = within.reshape(1, 1, MAX_PERIOD, MAX_GRID_SIZE)
        reduce_axes = (1, 3)
    shifted = shifted.reshape(block)
    shifted_valid = shifted_valid.reshape(block)
    comparable = shifted_valid & base_valid & keep
    matches = jnp.all((shifted == base) | ~comparable, axis=reduce_axes)
    admissible = matches & (shifts[None, :] < extent[:, None])
    first = jnp.argmax(admissible, axis=-1) + 1
    return jnp.where(jnp.any(admissible, axis=-1), first, extent)


def _ray_evidence(grid: jax.Array, valid: jax.Array, background: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return nearest non-background colour and distance along four rays."""

    marked = valid & (grid != background[:, None, None])
    coordinates = jnp.arange(MAX_GRID_SIZE)
    colours = []
    distances = []
    for axis, reverse in ((2, False), (2, True), (1, False), (1, True)):
        position = (
            coordinates.reshape(1, 1, MAX_GRID_SIZE)
            if axis == 2
            else coordinates.reshape(1, MAX_GRID_SIZE, 1)
        )
        source = jnp.where(marked, jnp.broadcast_to(position, grid.shape), -1)
        oriented = jnp.flip(source, axis=axis) if reverse else source
        oriented_grid = jnp.flip(grid, axis=axis) if reverse else grid
        running = jax.lax.cummax(oriented, axis=axis)
        shifted = jnp.concatenate(
            [
                jnp.full(
                    tuple(1 if index == axis else size for index, size in enumerate(grid.shape)),
                    -1,
                    dtype=running.dtype,
                ),
                jnp.take(running, jnp.arange(MAX_GRID_SIZE - 1), axis=axis),
            ],
            axis=axis,
        )
        found = shifted >= 0
        here = jnp.broadcast_to(position, grid.shape)
        if axis == 2:
            rows = jnp.broadcast_to(coordinates.reshape(1, MAX_GRID_SIZE, 1), grid.shape)
            colour = _gather_cells(oriented_grid, rows, jnp.maximum(shifted, 0))
        else:
            columns = jnp.broadcast_to(coordinates.reshape(1, 1, MAX_GRID_SIZE), grid.shape)
            colour = _gather_cells(oriented_grid, jnp.maximum(shifted, 0), columns)
        colour = jnp.where(found, colour, PAD_COLOR)
        distance = jnp.where(found, jnp.clip(here - shifted, 0, 31), 31)
        if reverse:
            colour = jnp.flip(colour, axis=axis)
            distance = jnp.flip(distance, axis=axis)
        colours.append(colour)
        distances.append(distance)
    return jnp.stack(colours, axis=-1), jnp.stack(distances, axis=-1)


def _scalar_block(
    grid: jax.Array,
    valid: jax.Array,
    rows: jax.Array,
    columns: jax.Array,
    height: jax.Array,
    width: jax.Array,
    row_period: jax.Array,
    column_period: jax.Array,
    background: jax.Array,
) -> jax.Array:
    """Return position, extent, period and background scalars per cell."""

    tall = height[:, None, None]
    wide = width[:, None, None]
    bottom = (tall - 1 - rows).astype(jnp.float32)
    right = (wide - 1 - columns).astype(jnp.float32)
    return jnp.stack(
        [
            rows / 29.0,
            columns / 29.0,
            bottom / 29.0,
            right / 29.0,
            (rows == 0).astype(jnp.float32),
            (rows == tall - 1).astype(jnp.float32),
            (columns == 0).astype(jnp.float32),
            (columns == wide - 1).astype(jnp.float32),
            jnp.minimum(rows, bottom) / 29.0,
            jnp.minimum(columns, right) / 29.0,
            ((rows + columns) % 2).astype(jnp.float32),
            ((rows - columns) % 2).astype(jnp.float32),
            (rows % 3).astype(jnp.float32) / 2.0,
            (columns % 3).astype(jnp.float32) / 2.0,
            jnp.broadcast_to(tall.astype(jnp.float32) / 30.0, rows.shape),
            jnp.broadcast_to(wide.astype(jnp.float32) / 30.0, rows.shape),
            jnp.broadcast_to(
                row_period[:, None, None].astype(jnp.float32) / 30.0, rows.shape
            ),
            jnp.broadcast_to(
                column_period[:, None, None].astype(jnp.float32) / 30.0, rows.shape
            ),
            (grid == background[:, None, None]).astype(jnp.float32),
            (rows == columns).astype(jnp.float32),
            (rows + columns == jnp.minimum(tall, wide) - 1).astype(jnp.float32),
            valid.astype(jnp.float32),
        ],
        axis=-1,
    )


def cell_features(grid: jax.Array, height: jax.Array, width: jax.Array) -> jax.Array:
    """Return per-cell features for a batch of padded ARC colour grids.

    Parameters
    ----------
    grid : jax.Array
        Integer colours shaped ``(batch, 30, 30)``, ``PAD_COLOR`` outside the
        grid's own ``height`` by ``width`` extent.
    height, width : jax.Array
        Integer extents shaped ``(batch,)``, each in 1--30.

    Returns
    -------
    jax.Array
        Float features shaped ``(batch, 900, cell_feature_width())``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from latent_workspace_cell_features import cell_features
        >>> grid = jnp.full((1, 30, 30), 10).at[0, :2, :2].set(
        ...     jnp.array([[1, 2], [3, 4]]))
        >>> features = cell_features(grid, jnp.array([2]), jnp.array([2]))
        >>> features.shape
        (1, 900, 482)
    """

    grid = jnp.asarray(grid, dtype=jnp.int32)
    height = jnp.asarray(height, dtype=jnp.int32)
    width = jnp.asarray(width, dtype=jnp.int32)
    batch = grid.shape[0]
    coordinates = jnp.arange(MAX_GRID_SIZE)
    rows = jnp.broadcast_to(
        coordinates[None, :, None], (batch, MAX_GRID_SIZE, MAX_GRID_SIZE)
    )
    columns = jnp.broadcast_to(
        coordinates[None, None, :], (batch, MAX_GRID_SIZE, MAX_GRID_SIZE)
    )
    valid = (rows < height[:, None, None]) & (columns < width[:, None, None])

    one_hot = jax.nn.one_hot(jnp.clip(grid, 0, COLOR_COUNT - 1), COLOR_COUNT)
    counts = jnp.sum(one_hot * valid[..., None], axis=(1, 2))
    background = jnp.argmax(counts, axis=-1).astype(jnp.int32)

    patches = jax.nn.one_hot(_local_patches(grid), PATCH_COLORS).reshape(
        batch, MAX_GRID_SIZE, MAX_GRID_SIZE, PATCH_CELLS * PATCH_COLORS
    )
    statistics = _colour_statistics(grid, valid, height, width)
    mirrors = jax.nn.one_hot(
        _mirror_colours(grid, rows, columns, height, width), PATCH_COLORS
    ).reshape(batch, MAX_GRID_SIZE, MAX_GRID_SIZE, _MIRROR_COUNT * PATCH_COLORS)

    row_period = _axis_period(grid, valid, height, 1)
    column_period = _axis_period(grid, valid, width, 2)
    period_views = jnp.stack(
        [
            _gather_cells(grid, rows % row_period[:, None, None], columns),
            _gather_cells(grid, rows, columns % column_period[:, None, None]),
            _gather_cells(
                grid,
                rows % row_period[:, None, None],
                columns % column_period[:, None, None],
            ),
        ],
        axis=-1,
    )
    periods = jax.nn.one_hot(period_views, PATCH_COLORS).reshape(
        batch, MAX_GRID_SIZE, MAX_GRID_SIZE, _PERIOD_VIEWS * PATCH_COLORS
    )

    ray_colours, ray_distances = _ray_evidence(grid, valid, background)
    rays = jax.nn.one_hot(ray_colours, PATCH_COLORS).reshape(
        batch, MAX_GRID_SIZE, MAX_GRID_SIZE, _RAY_COUNT * PATCH_COLORS
    )
    ray_scale = ray_distances.astype(jnp.float32) / 31.0

    scalars = _scalar_block(
        grid, valid, rows, columns, height, width, row_period, column_period, background
    )
    stacked = jnp.concatenate(
        [patches, statistics, mirrors, periods, rays, ray_scale, scalars], axis=-1
    )
    return stacked.reshape(batch, MAX_GRID_SIZE * MAX_GRID_SIZE, -1)
