"""A demonstration-fitted decision forest over the per-cell feature map.

Example 21's shipped colour head is a bias-free linear map trained once across
tasks and then frozen, and its argmax is the query input on every overlapping
cell: copying is the only optimum it can reach.  The measurements in
``docs/specs/2026-08-22-example21-in-context-cell-decoder.md`` put the whole
remaining gap in the *learner*, not the features -- over one per-cell feature
map a gradient-fitted head answered 5 of 419 evaluation queries where an
axis-aligned tree answered 14.

This module is that tree, written in JAX so it fits on device beside the rest
of the harness and needs no new dependency.  Every level is a segment sum plus
an argmax, so a whole tree is a fixed number of array operations and the depth
sweep lowers into one compiled program.

Two properties matter downstream.  The eleventh class -- "not part of the
output grid" -- lets one fitted forest carry extent as well as colour, so no
shape rule is consulted anywhere.  And leaf distributions are read back along
the whole root-to-leaf path with a geometric weight, so a query cell that
reaches a leaf no demonstration cell reached backs off to its ancestors instead
of returning a uniform guess that would read as "inside the grid".
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import brainstate
import jax
import jax.numpy as jnp


COLOR_COUNT = 10
OUTSIDE_CLASS = COLOR_COUNT
CLASS_COUNT = COLOR_COUNT + 1
_EPSILON = 1e-9


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class DemonstrationForestConfig:
    """Static shape of a demonstration-fitted forest.

    Parameters
    ----------
    depth : int, default=10
        Levels of binary splits. Node arrays are sized ``2 ** depth``, so this
        is the one field that drives memory.
    tree_count : int, default=16
        Trees fitted per task. Each sees its own random feature subset, which
        is the only source of diversity -- there is no bootstrap, so every tree
        sees every demonstration cell.
    feature_fraction : float, default=0.6
        Fraction of features offered to each tree.
    backoff : float, default=0.05
        Geometric weight applied per level when reading a leaf back along its
        root-to-leaf path. Smaller values trust the deepest non-empty node
        more.
    class_count : int, default=11
        Ten ARC colours plus "not part of the output grid".

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_demonstration_forest import (
        ...     DemonstrationForestConfig)
        >>> DemonstrationForestConfig(depth=4, tree_count=2).tree_count
        2
    """

    depth: int = 10
    tree_count: int = 16
    feature_fraction: float = 0.6
    backoff: float = 0.05
    class_count: int = CLASS_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", _positive_integer(self.depth, "depth"))
        object.__setattr__(
            self, "tree_count", _positive_integer(self.tree_count, "tree_count")
        )
        object.__setattr__(
            self,
            "feature_fraction",
            _unit_interval(self.feature_fraction, "feature_fraction"),
        )
        object.__setattr__(self, "backoff", _unit_interval(self.backoff, "backoff"))
        object.__setattr__(
            self, "class_count", _positive_integer(self.class_count, "class_count")
        )


def _split_scores(
    positive: jax.Array, totals: jax.Array
) -> jax.Array:
    """Return the Gini split score of every feature at every node.

    Minimising the Gini-weighted child impurity is equivalent to maximising
    ``sum_c n_c^2 / n`` summed over the two children, which is what this
    returns. Degenerate splits -- every cell on one side -- score ``-inf`` so
    they are chosen only when a node offers nothing else.
    """

    negative = totals[:, :, None] - positive
    positive_size = jnp.sum(positive, axis=1)
    negative_size = jnp.sum(negative, axis=1)
    score = jnp.sum(jnp.square(positive), axis=1) / jnp.maximum(
        positive_size, _EPSILON
    ) + jnp.sum(jnp.square(negative), axis=1) / jnp.maximum(negative_size, _EPSILON)
    usable = (positive_size > 0.0) & (negative_size > 0.0)
    return jnp.where(usable, score, -jnp.inf)


def _fit_tree(
    binary: jax.Array,
    labels: jax.Array,
    offered: jax.Array,
    config: DemonstrationForestConfig,
) -> tuple[jax.Array, jax.Array]:
    """Grow one tree; return its per-level split features and class counts."""

    nodes = 2 ** config.depth
    classes = config.class_count
    one_hot = jax.nn.one_hot(labels, classes)
    penalty = jnp.where(offered, 0.0, -jnp.inf)

    def level(node: jax.Array, _):
        counts = jax.ops.segment_sum(
            one_hot, node, num_segments=nodes, indices_are_sorted=False
        )
        positive = jax.ops.segment_sum(
            binary, node * classes + labels, num_segments=nodes * classes
        ).reshape(nodes, classes, -1)
        chosen = jnp.argmax(_split_scores(positive, counts) + penalty, axis=-1)
        bit = jnp.take_along_axis(binary, chosen[node][:, None], axis=1)[:, 0]
        return node * 2 + bit.astype(jnp.int32), (chosen, counts)

    final, (splits, level_counts) = brainstate.transform.scan(
        level, jnp.zeros(binary.shape[0], jnp.int32), jnp.arange(config.depth)
    )
    leaves = jax.ops.segment_sum(one_hot, final, num_segments=nodes)
    return splits, jnp.concatenate([level_counts, leaves[None]], axis=0)


def fit_demonstration_forest(
    key: jax.Array,
    features: jax.Array,
    labels: jax.Array,
    config: DemonstrationForestConfig,
) -> tuple[jax.Array, jax.Array]:
    """Fit one forest on a single task's demonstration cells.

    Parameters
    ----------
    key : jax.Array
        PRNG key; each tree draws its own feature subset from it.
    features : jax.Array
        Demonstration input-cell features shaped ``(cells, feature_width)``.
        Values are compared against ``0.5``, so the map should be one-hot.
    labels : jax.Array
        Per-cell classes shaped ``(cells,)`` over ``class_count`` values.
    config : DemonstrationForestConfig
        Forest shape.

    Returns
    -------
    tuple of jax.Array
        Split features shaped ``(trees, depth, 2 ** depth)`` and per-level
        class counts shaped ``(trees, depth + 1, 2 ** depth, class_count)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax, jax.numpy as jnp
        >>> from latent_workspace_demonstration_forest import (
        ...     DemonstrationForestConfig, fit_demonstration_forest)
        >>> config = DemonstrationForestConfig(depth=2, tree_count=2)
        >>> splits, counts = fit_demonstration_forest(
        ...     jax.random.key(0), jnp.eye(4), jnp.arange(4), config)
        >>> splits.shape
        (2, 2, 4)
    """

    binary = (features > 0.5).astype(jnp.float32)
    width = binary.shape[-1]
    keep = max(1, int(round(config.feature_fraction * width)))

    def tree(tree_key):
        rank = jax.random.uniform(tree_key, (width,))
        offered = rank <= jnp.sort(rank)[keep - 1]
        return _fit_tree(binary, labels, offered, config)

    keys = jax.random.split(key, config.tree_count)
    return brainstate.transform.scan(lambda _, k: (None, tree(k)), None, keys)[1]


def demonstration_forest_probabilities(
    forest: tuple[jax.Array, jax.Array],
    features: jax.Array,
    config: DemonstrationForestConfig,
) -> jax.Array:
    """Score cells with a fitted forest.

    Every tree is read back along the whole root-to-leaf path, weighting level
    ``d`` by ``backoff ** (depth - d)``. A leaf no demonstration cell reached
    contributes nothing and the cell falls back to its deepest populated
    ancestor.

    Parameters
    ----------
    forest : tuple of jax.Array
        Output of :func:`fit_demonstration_forest`.
    features : jax.Array
        Query cell features shaped ``(cells, feature_width)``.
    config : DemonstrationForestConfig
        The same configuration the forest was fitted under.

    Returns
    -------
    jax.Array
        Class probabilities shaped ``(cells, class_count)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax, jax.numpy as jnp
        >>> from latent_workspace_demonstration_forest import (
        ...     DemonstrationForestConfig, demonstration_forest_probabilities,
        ...     fit_demonstration_forest)
        >>> config = DemonstrationForestConfig(depth=2, tree_count=2)
        >>> forest = fit_demonstration_forest(
        ...     jax.random.key(0), jnp.eye(4), jnp.arange(4), config)
        >>> demonstration_forest_probabilities(forest, jnp.eye(4), config).shape
        (4, 11)
    """

    splits, counts = forest
    binary = (features > 0.5).astype(jnp.float32)
    weights = config.backoff ** (config.depth - jnp.arange(config.depth + 1))

    def tree(carry, member):
        member_splits, member_counts = member

        def level(node, chosen):
            bit = jnp.take_along_axis(binary, chosen[node][:, None], axis=1)[:, 0]
            return node * 2 + bit.astype(jnp.int32), node

        final, path = brainstate.transform.scan(
            level, jnp.zeros(binary.shape[0], jnp.int32), member_splits
        )
        visited = jnp.concatenate([path, final[None]], axis=0)
        gathered = jnp.take_along_axis(
            member_counts, visited[..., None], axis=1
        )
        return carry + jnp.sum(gathered * weights[:, None, None], axis=0), None

    total = brainstate.transform.scan(
        tree, jnp.zeros((binary.shape[0], config.class_count)), (splits, counts)
    )[0]
    return total / jnp.maximum(jnp.sum(total, axis=-1, keepdims=True), _EPSILON)
