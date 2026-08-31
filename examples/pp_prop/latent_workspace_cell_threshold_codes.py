"""Threshold codes that make the per-cell map wholly binary.

The demonstration-fitted answer head splits every feature against ``0.5``,
which reproduces a continuous split only where that split sits at ``0.5``.  A
census of the reference tree's splits over the ARC-AGI-1 evaluation split puts
16% of its impurity gain on thresholds ``0.5`` cannot express, and names where
that gain sits: the per-row, per-column and whole-grid colour counts, the four
ray distances, and the position scalars.  A ``row / 29`` scalar cannot answer
"is this row below the middle" under a ``0.5`` threshold, and a count ratio
cannot answer "does this colour fill at least three cells of the row".

This module codes exactly those quantities as ``quantity >= k`` indicators, so
one binary split reproduces one continuous split.  Measured on the evaluation
split, a per-task tree over the coded map answers 18 of 419 queries against 11
for the same tree over the same map thresholded at ``0.5``.

Every block is a comparison against a constant vector, so the whole map lowers
into the same compiled program as :mod:`latent_workspace_cell_features`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

try:
    from examples.pp_prop.latent_workspace_cell_features import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        cell_feature_width,
        cell_features,
    )
except ImportError:
    from latent_workspace_cell_features import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        cell_feature_width,
        cell_features,
    )


CELLS = MAX_GRID_SIZE * MAX_GRID_SIZE
MAX_RAY_DISTANCE = 31
RAY_SCALE_SLICE = slice(456, 460)
PERIOD_SCALAR_SLICE = slice(476, 478)

_POSITION_COUNT = 8
_PERIOD_COUNT = 2
_RAY_COUNT = 4
_RESIDUE_MODULUS = 3
_RESIDUE_COUNT = 2

COUNT_STEPS = tuple(range(1, 16))
GRID_COUNT_STEPS = (
    1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 150, 200,
    300, 450, 600, 900,
)
FRACTION_STEPS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def threshold_code_width() -> int:
    """Return the code width produced by :func:`threshold_codes`.

    Returns
    -------
    int
        Total threshold-code count for one cell.

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_cell_threshold_codes import threshold_code_width
        >>> threshold_code_width()
        1214
    """

    return (
        (_POSITION_COUNT + _PERIOD_COUNT) * MAX_GRID_SIZE
        + _RAY_COUNT * (MAX_RAY_DISTANCE + 1)
        + _RESIDUE_COUNT * _RESIDUE_MODULUS
        + 2 * COLOR_COUNT * len(COUNT_STEPS)
        + COLOR_COUNT * len(GRID_COUNT_STEPS)
        + 3 * COLOR_COUNT * len(FRACTION_STEPS)
    )


def binary_cell_feature_width() -> int:
    """Return the width of the wholly binary map.

    Returns
    -------
    int
        Thresholded feature count plus threshold-code count.

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_cell_threshold_codes import (
        ...     binary_cell_feature_width)
        >>> binary_cell_feature_width()
        1696
    """

    return cell_feature_width() + threshold_code_width()


def _codes(quantity: jax.Array, steps: jax.Array) -> jax.Array:
    """Return ``quantity >= step`` for every step, flattened per cell."""

    batch = quantity.shape[0]
    indicators = quantity[..., None] >= steps
    return indicators.astype(jnp.float32).reshape(batch, CELLS, -1)


def _colour_counts(
    grid: jax.Array, valid: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-row, per-column and whole-grid colour counts."""

    one_hot = jax.nn.one_hot(jnp.clip(grid, 0, COLOR_COUNT - 1), COLOR_COUNT)
    one_hot = one_hot * valid[..., None]
    row_counts = jnp.sum(one_hot, axis=2)
    column_counts = jnp.sum(one_hot, axis=1)
    return row_counts, column_counts, jnp.sum(row_counts, axis=1)


def _count_codes(
    grid: jax.Array, valid: jax.Array, height: jax.Array, width: jax.Array
) -> list[jax.Array]:
    """Return count and fraction codes for the three colour-count scopes.

    A count answers "how many cells of this row are red"; the matching fraction
    answers "what share of the row is red". Both appear because a task can key
    on either, and neither is derivable from the other under a fixed threshold
    once the grid's extent varies across demonstrations.
    """

    batch = grid.shape[0]
    row_counts, column_counts, grid_counts = _colour_counts(grid, valid)
    shape = (batch, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT)
    per_row = jnp.broadcast_to(row_counts[:, :, None, :], shape)
    per_column = jnp.broadcast_to(column_counts[:, None, :, :], shape)
    per_grid = jnp.broadcast_to(grid_counts[:, None, None, :], shape)
    counts = jnp.asarray(COUNT_STEPS, jnp.float32)
    grid_steps = jnp.asarray(GRID_COUNT_STEPS, jnp.float32)
    fractions = jnp.asarray(FRACTION_STEPS, jnp.float32)
    tall = jnp.maximum(height, 1)[:, None, None, None].astype(jnp.float32)
    wide = jnp.maximum(width, 1)[:, None, None, None].astype(jnp.float32)
    return [
        _codes(per_row, counts),
        _codes(per_column, counts),
        _codes(per_grid, grid_steps),
        _codes(per_row / wide, fractions),
        _codes(per_column / tall, fractions),
        _codes(per_grid / (tall * wide), fractions),
    ]


def _position_codes(
    rows: jax.Array, columns: jax.Array, height: jax.Array, width: jax.Array
) -> jax.Array:
    """Return ``>= k`` codes for the six positions and the two extents."""

    tall = height[:, None, None]
    wide = width[:, None, None]
    bottom = tall - 1 - rows
    right = wide - 1 - columns
    stacked = jnp.stack(
        [
            rows,
            columns,
            bottom,
            right,
            jnp.minimum(rows, bottom),
            jnp.minimum(columns, right),
            jnp.broadcast_to(tall, rows.shape),
            jnp.broadcast_to(wide, rows.shape),
        ],
        axis=-1,
    )
    return _codes(stacked, jnp.arange(MAX_GRID_SIZE))


def _scale_codes(features: jax.Array) -> list[jax.Array]:
    """Return period and ray-distance codes read back from the float map.

    Both quantities are already computed by
    :func:`latent_workspace_cell_features.cell_features`, which divides them by
    a constant; recovering the integer here keeps one implementation of the
    period search and the ray sweep rather than two that can drift apart.
    """

    periods = jnp.round(features[..., PERIOD_SCALAR_SLICE] * float(MAX_GRID_SIZE))
    distances = jnp.round(features[..., RAY_SCALE_SLICE] * float(MAX_RAY_DISTANCE))
    return [
        _codes(periods, jnp.arange(MAX_GRID_SIZE)),
        _codes(distances, jnp.arange(MAX_RAY_DISTANCE + 1)),
    ]


def threshold_codes(
    features: jax.Array, grid: jax.Array, height: jax.Array, width: jax.Array
) -> jax.Array:
    """Return binary threshold codes for every ordered quantity of a grid.

    Parameters
    ----------
    features : jax.Array
        The float map from
        :func:`latent_workspace_cell_features.cell_features`, shaped
        ``(batch, 900, cell_feature_width())``.
    grid : jax.Array
        Integer colours shaped ``(batch, 30, 30)``, padded outside the extent.
    height, width : jax.Array
        Integer extents shaped ``(batch,)``, each in 1--30.

    Returns
    -------
    jax.Array
        Binary codes shaped ``(batch, 900, threshold_code_width())``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from latent_workspace_cell_features import cell_features
        >>> from latent_workspace_cell_threshold_codes import threshold_codes
        >>> grid = jnp.full((1, 30, 30), 10).at[0, :2, :2].set(
        ...     jnp.array([[1, 2], [3, 4]]))
        >>> height, width = jnp.array([2]), jnp.array([2])
        >>> codes = threshold_codes(
        ...     cell_features(grid, height, width), grid, height, width)
        >>> codes.shape
        (1, 900, 1214)
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
    residues = jax.nn.one_hot(
        jnp.stack([rows % _RESIDUE_MODULUS, columns % _RESIDUE_MODULUS], axis=-1),
        _RESIDUE_MODULUS,
    ).reshape(batch, CELLS, _RESIDUE_COUNT * _RESIDUE_MODULUS)
    blocks = [
        _position_codes(rows, columns, height, width),
        *_scale_codes(features),
        residues,
        *_count_codes(grid, valid, height, width),
    ]
    return jnp.concatenate(blocks, axis=-1)


def binary_cell_features(
    grid: jax.Array, height: jax.Array, width: jax.Array
) -> jax.Array:
    """Return the wholly binary per-cell map for a batch of padded grids.

    Parameters
    ----------
    grid : jax.Array
        Integer colours shaped ``(batch, 30, 30)``, padded outside the extent.
    height, width : jax.Array
        Integer extents shaped ``(batch,)``, each in 1--30.

    Returns
    -------
    jax.Array
        Binary features shaped ``(batch, 900, binary_cell_feature_width())``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from latent_workspace_cell_threshold_codes import binary_cell_features
        >>> grid = jnp.full((1, 30, 30), 10).at[0, :2, :2].set(
        ...     jnp.array([[1, 2], [3, 4]]))
        >>> binary_cell_features(grid, jnp.array([2]), jnp.array([2])).shape
        (1, 900, 1696)
    """

    features = cell_features(grid, height, width)
    return jnp.concatenate(
        [
            (features > 0.5).astype(jnp.float32),
            threshold_codes(features, grid, height, width),
        ],
        axis=-1,
    )
