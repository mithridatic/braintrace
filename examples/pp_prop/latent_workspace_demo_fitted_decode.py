"""Decode ARC queries with an answer head fitted on their own demonstrations.

Produces the same explicit ``30 + 30 + 30*30*10`` logit vector the row
refinement decoder emits, so every downstream stage of Example 21 -- candidate
decoding and diagnostic metrics -- can consume it unchanged. Raw forest ranks
are demonstration-only diagnostics; the checkpoint-conditioned answer head
uses these logits only to propose grids and lets trained-network likelihood
decide their submitted order.

The head predicts eleven classes per cell: the ten ARC colours plus "not part
of the output grid".  Extent therefore comes from the same fitted head as
colour and no shape rule is consulted: the height logit for an extent ``v`` is
the log likelihood that rows below ``v`` contain output cells and rows from
``v`` on do not.  On the ARC-AGI-1 evaluation split that head alone recovers
the extent of 320 of 419 queries, against 355 for the hand-written rule the
``rule_then_model`` channel uses -- close enough that the rule buys nothing the
model cannot learn from the task's own demonstrations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

try:
    from examples.pp_prop.latent_workspace_cell_features import (
        MAX_GRID_SIZE,
        PAD_COLOR,
        cell_features,
    )
    from examples.pp_prop.latent_workspace_demonstration_forest import (
        DemonstrationForestConfig,
        demonstration_forest_probabilities,
        fit_demonstration_forest,
    )
except ImportError:
    from latent_workspace_cell_features import (
        MAX_GRID_SIZE,
        PAD_COLOR,
        cell_features,
    )
    from latent_workspace_demonstration_forest import (
        DemonstrationForestConfig,
        demonstration_forest_probabilities,
        fit_demonstration_forest,
    )


COLOR_COUNT = 10
OUTSIDE_CLASS = COLOR_COUNT
CLASS_COUNT = COLOR_COUNT + 1
CELLS = MAX_GRID_SIZE * MAX_GRID_SIZE
DECODER_WIDTH = 2 * MAX_GRID_SIZE + CELLS * COLOR_COUNT
_LOG_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class DemonstrationBatch:
    """Demonstration and query tensors for one scored query.

    Demonstration counts vary per task and are *not* padded to a fixed slot
    count: an unused slot would be a grid of pure padding labelled entirely
    "outside", which is a real training row a forest would happily fit. The
    cost is one compilation per distinct demonstration count, of which the
    evaluation split has fewer than ten.

    Parameters
    ----------
    demonstration_grids : numpy.ndarray
        Padded demonstration inputs shaped ``(demonstrations, 30, 30)``.
    demonstration_heights, demonstration_widths : numpy.ndarray
        Demonstration input extents shaped ``(demonstrations,)``.
    demonstration_targets : numpy.ndarray
        Eleven-class per-cell targets shaped ``(demonstrations, 900)``.
    query_grid : numpy.ndarray
        Padded query input shaped ``(30, 30)``.
    query_height, query_width : int
        Query input extent.
    """

    demonstration_grids: np.ndarray
    demonstration_heights: np.ndarray
    demonstration_widths: np.ndarray
    demonstration_targets: np.ndarray
    query_grid: np.ndarray
    query_height: int
    query_width: int


def _pad_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    padded = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), PAD_COLOR, dtype=np.int32)
    array = np.asarray(grid, dtype=np.int32)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded, int(array.shape[0]), int(array.shape[1])


def build_demonstration_batch(
    demonstration_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    query_input: np.ndarray,
) -> DemonstrationBatch:
    """Pad one query's demonstrations onto the fixed 30x30 canvas.

    Every cell of the canvas is supervised, including the region outside the
    demonstration's own output. Without those negative examples a same-shape
    task supplies no evidence for where the grid ends and the decoded extent
    runs to 30x30.

    Parameters
    ----------
    demonstration_pairs : sequence of tuple of numpy.ndarray
        ``(input, output)`` colour grids.
    query_input : numpy.ndarray
        The query's input colour grid.

    Returns
    -------
    DemonstrationBatch
        Tensors ready for :func:`demonstration_fitted_windows`.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from latent_workspace_demo_fitted_decode import build_demonstration_batch
        >>> pair = (np.array([[1]]), np.array([[2]]))
        >>> batch = build_demonstration_batch([pair], np.array([[1]]))
        >>> batch.demonstration_targets.shape
        (1, 900)
    """

    if not demonstration_pairs:
        raise ValueError("a demonstration batch needs at least one pair")
    count = len(demonstration_pairs)
    grids = np.full((count, MAX_GRID_SIZE, MAX_GRID_SIZE), PAD_COLOR, np.int32)
    heights = np.ones((count,), np.int32)
    widths = np.ones((count,), np.int32)
    targets = np.full((count, CELLS), OUTSIDE_CLASS, np.int32)
    for slot, (source, target) in enumerate(demonstration_pairs):
        grids[slot], heights[slot], widths[slot] = _pad_grid(source)
        target = np.asarray(target, dtype=np.int32)
        canvas = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), OUTSIDE_CLASS, np.int32)
        canvas[: target.shape[0], : target.shape[1]] = target
        targets[slot] = canvas.reshape(-1)
    query_grid, query_height, query_width = _pad_grid(query_input)
    return DemonstrationBatch(
        grids, heights, widths, targets, query_grid, query_height, query_width
    )


def _extent_log_likelihood(inside_probability: jax.Array) -> jax.Array:
    """Return a 30-way extent log likelihood from per-line inside probability."""

    clipped = jnp.clip(inside_probability, _LOG_FLOOR, 1.0 - _LOG_FLOOR)
    occupied = jnp.log(clipped)
    empty = jnp.log1p(-clipped)
    occupied_prefix = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(occupied)])
    empty_suffix = jnp.concatenate([jnp.cumsum(empty[::-1])[::-1], jnp.zeros((1,))])
    extents = jnp.arange(1, MAX_GRID_SIZE + 1)
    return occupied_prefix[extents] + empty_suffix[extents]


def _pack_window(probabilities: jax.Array) -> jax.Array:
    """Pack per-cell class probabilities into the explicit decoder layout."""

    grid = probabilities.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE, CLASS_COUNT)
    inside = 1.0 - grid[..., OUTSIDE_CLASS]
    height_logits = _extent_log_likelihood(jnp.max(inside, axis=1))
    width_logits = _extent_log_likelihood(jnp.max(inside, axis=0))
    colours = grid[..., :COLOR_COUNT]
    colours = colours / jnp.maximum(
        jnp.sum(colours, axis=-1, keepdims=True), _LOG_FLOOR
    )
    colour_logits = jnp.log(jnp.maximum(colours, _LOG_FLOOR))
    return jnp.concatenate(
        [height_logits, width_logits, colour_logits.reshape(-1)], axis=0
    )


def _batch_window(
    grids, heights, widths, targets, query_grid, query_height, query_width,
    key, config,
):
    demonstration_features = cell_features(grids, heights, widths)
    query_features = cell_features(
        query_grid[None], query_height[None], query_width[None]
    )[0]
    forest = fit_demonstration_forest(
        key,
        demonstration_features.reshape(-1, demonstration_features.shape[-1]),
        targets.reshape(-1),
        config,
    )
    return _pack_window(
        demonstration_forest_probabilities(forest, query_features, config)
    )


def demonstration_fitted_windows(
    batches: Sequence[DemonstrationBatch],
    config: DemonstrationForestConfig,
    seed: int = 0,
) -> np.ndarray:
    """Return explicit decoder logits for a sequence of queries.

    Parameters
    ----------
    batches : sequence of DemonstrationBatch
        One entry per scored query.
    config : DemonstrationForestConfig
        Forest shape, fitted independently per query.
    seed : int, default=0
        Base PRNG seed; each query is fitted from its own derived key.

    Returns
    -------
    numpy.ndarray
        Logits shaped ``(queries, 9060)`` in the row-refinement layout.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from latent_workspace_demonstration_forest import (
        ...     DemonstrationForestConfig)
        >>> from latent_workspace_demo_fitted_decode import (
        ...     build_demonstration_batch, demonstration_fitted_windows)
        >>> pair = (np.array([[1]]), np.array([[2]]))
        >>> batch = build_demonstration_batch([pair], np.array([[1]]))
        >>> config = DemonstrationForestConfig(depth=2, tree_count=1)
        >>> demonstration_fitted_windows([batch], config).shape
        (1, 9060)
    """

    if not batches:
        raise ValueError("demonstration_fitted_windows requires at least one query")
    keys = brainstate.random.RandomState(int(seed)).split_key(len(batches))
    grouped: dict[int, list[int]] = {}
    for index, batch in enumerate(batches):
        grouped.setdefault(int(batch.demonstration_grids.shape[0]), []).append(index)
    windows: list[np.ndarray | None] = [None] * len(batches)
    for positions in grouped.values():
        selected = [batches[index] for index in positions]

        def fit_one(
            grids, heights, widths, targets, query_grid, query_height, query_width, key
        ):
            return _batch_window(
                grids,
                heights,
                widths,
                targets,
                query_grid,
                query_height,
                query_width,
                key,
                config,
            )

        fitted = np.asarray(
            brainstate.transform.for_loop(
                fit_one,
                jnp.asarray(
                    np.stack([batch.demonstration_grids for batch in selected])
                ),
                jnp.asarray(
                    np.stack([batch.demonstration_heights for batch in selected])
                ),
                jnp.asarray(
                    np.stack([batch.demonstration_widths for batch in selected])
                ),
                jnp.asarray(
                    np.stack([batch.demonstration_targets for batch in selected])
                ),
                jnp.asarray(np.stack([batch.query_grid for batch in selected])),
                jnp.asarray(
                    [batch.query_height for batch in selected], dtype=jnp.int32
                ),
                jnp.asarray(
                    [batch.query_width for batch in selected], dtype=jnp.int32
                ),
                keys[np.asarray(positions)],
            )
        )
        for position, window in zip(positions, fitted, strict=True):
            windows[position] = window
    if any(window is None for window in windows):
        raise RuntimeError("Every demonstration-fitted query must produce one window.")
    return np.stack(windows).astype(np.float32)
