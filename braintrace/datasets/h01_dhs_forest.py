"""Tree-contraction schedules and solves over a forest of independent cable trees.

Spec: docs/specs/2026-09-16-h01-fused-population.md, section 2. The single-tree
contraction in :mod:`h01_dhs_contraction` requires one root at row 0 and
``n-1`` edges; a fused population is a forest, one tree per cell, whose rows
are concatenated with per-cell offsets. The Schur elimination is identical
(a removed node has at most one child, is never a root, and its two surviving
neighbours absorb its row exactly); only the termination and the root
back-substitution change. Schedules are built once on the host in NumPy.
"""

import numpy as np
import jax
import jax.numpy as jnp


def forest_roots(edges, n):
    """Return the root rows of a forest given its child/parent edge list.

    Parameters
    ----------
    edges : numpy.ndarray
        ``(n_edges, 2)`` child/parent pairs; every non-root row appears
        exactly once as a child.
    n : int
        Number of physical rows, excluding the sentinel.

    Returns
    -------
    numpy.ndarray
        Sorted root rows (rows that are never a child).
    """
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('Forest edges must be an (n_edges, 2) array')
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Forest edge indices are out of range')
    children = edges[:, 0] if len(edges) else np.zeros(0, dtype=np.int64)
    if len(np.unique(children)) != len(children):
        raise ValueError('Every non-root row must appear exactly once as a child')
    is_child = np.zeros(n, dtype=bool)
    is_child[children] = True
    return np.flatnonzero(~is_child).astype(np.int32)


def build_schedule(edges, n):
    """Build grouped Schur-contraction stages for a forest.

    Parameters
    ----------
    edges : numpy.ndarray
        ``(n_edges, 2)`` child/parent pairs over rows ``0..n-1``.
    n : int
        Number of physical rows; the sentinel row ``n`` is appended by the solve.

    Returns
    -------
    tuple
        ``(packs, roots)``: ``packs`` is a tuple of stage groups, each a
        ``(removed, parent, child)`` triple of ``(stages, width)`` int32 arrays
        padded with ``n``; ``roots`` the root rows. A forest of isolated rows
        has no stages.

    Notes
    -----
    Each stage removes a set of pairwise non-adjacent non-root nodes with at
    most one child; the parent absorbs the node's Schur complement and the
    child is re-parented. The stage count is the maximum over the trees, so
    the forest runs the deepest cell's schedule once for every cell.
    """
    edges = np.asarray(edges, dtype=np.int64)
    roots = forest_roots(edges, n)
    parent = np.arange(n, dtype=np.int64)
    parent[edges[:, 0]] = edges[:, 1]
    is_root = np.zeros(n, dtype=bool)
    is_root[roots] = True
    children = [[] for _ in range(n)]
    for child, ancestor in edges:
        children[ancestor].append(int(child))
    order = list(roots)
    for node in order:
        order.extend(children[node])
    if len(order) != n:
        raise ValueError('Forest must be acyclic with every row reachable from a root')
    active = np.ones(n, dtype=bool)
    stages = []
    while active.sum() > len(roots):
        ids = np.flatnonzero(active & ~is_root)
        counts = np.bincount(parent[ids], minlength=n)
        child = np.full(n, n, dtype=np.int64)
        child[parent[ids]] = ids
        depth = np.zeros(n, dtype=np.int64)
        for node in order:
            if active[node] and not is_root[node]:
                depth[node] = depth[parent[node]]+1
        selected = active & ~is_root & (counts <= 1) & (depth % 2 == 0)
        selected |= active & ~is_root & (counts == 0) & ~selected[parent]
        selected &= ~is_root
        removed = np.flatnonzero(selected)
        down = child[removed]
        stages.append((removed.astype(np.int32), parent[removed].astype(np.int32), down.astype(np.int32)))
        has_child = down < n
        parent[down[has_child]] = parent[removed[has_child]]
        active[removed] = False
    return _pack(stages, n), roots


def _pack(stages, n):
    if not stages:
        return ()
    largest = max(len(stage[0]) for stage in stages)
    groups, previous = [], None
    for stage in stages:
        width = len(stage[0])
        bucket = 0 if width > largest/4 else (1 if width > largest/64 else 2)
        if bucket != previous:
            groups.append([])
            previous = bucket
        groups[-1].append(stage)
    packs = []
    for group in groups:
        width = max(len(stage[0]) for stage in group)
        packed = tuple(np.full((len(group), width), n, dtype=np.int32) for _ in range(3))
        for index, stage in enumerate(group):
            for target, values in zip(packed, stage):
                target[index, :len(values)] = values
        for array in packed:
            array.flags.writeable = False
        packs.append(packed)
    return tuple(packs)


def solve(d, rhs, low, up, packs, roots):
    """Solve one forest system with grouped scans over independent eliminations.

    Parameters
    ----------
    d, rhs, low, up : arrays
        One-dimensional diagonal, right-hand side and edge coefficients over
        ``n+1`` rows (the last row is the independent sentinel). Not mutated.
    packs : tuple
        Stage groups from :func:`build_schedule`.
    roots : array
        Root rows from :func:`build_schedule`.

    Returns
    -------
    array
        Solution over ``n+1`` rows.
    """
    n = d.shape[0]-1
    roots = jnp.asarray(roots, dtype=jnp.int32)
    if not packs:
        return rhs/d

    def eliminate(carry, indices):
        d, rhs, low, up = carry
        removed, parent, child = indices
        diagonal, value = d[removed], rhs[removed]
        lower, upper = low[removed], up[removed]
        child_lower = jnp.where(child < n, low[child], 0.)
        child_upper = jnp.where(child < n, up[child], 0.)
        parent_target = jnp.where(removed < n, parent, n+1)
        child_target = jnp.where(child < n, child, n+1)
        removed_target = jnp.where(removed < n, removed, n+1)
        d = d.at[parent_target].add(-upper*lower/diagonal, mode='drop')
        rhs = rhs.at[parent_target].add(-upper*value/diagonal, mode='drop')
        d = d.at[child_target].add(-child_lower*child_upper/diagonal, mode='drop')
        rhs = rhs.at[child_target].add(-child_lower*value/diagonal, mode='drop')
        low = low.at[child_target].set(-child_lower*lower/diagonal, mode='drop')
        up = up.at[child_target].set(-upper*child_upper/diagonal, mode='drop')
        up = up.at[removed_target].set(child_upper, mode='drop')
        return (d, rhs, low, up), None

    for packed in packs:
        (d, rhs, low, up), _ = jax.lax.scan(eliminate, (d, rhs, low, up), packed)
    result = jnp.zeros_like(rhs).at[roots].set(rhs[roots]/d[roots]).at[-1].set(rhs[-1]/d[-1])

    def substitute(result, indices):
        removed, parent, child = indices
        values = (rhs[removed]-low[removed]*result[parent]-up[removed]*result[child])/d[removed]
        target = jnp.where(removed < n, removed, n+1)
        return result.at[target].set(values, mode='drop'), None

    for packed in reversed(packs):
        result, _ = jax.lax.scan(substitute, result, packed, reverse=True)
    return result


def dense_matrix(d, low, up, edges, n):
    """Assemble the dense forest operator for oracle checks.

    Parameters
    ----------
    d, low, up : arrays
        Row diagonal and the coefficients of each child row's edge to its parent
        (``low[c]`` multiplies the parent in row ``c``; ``up[c]`` multiplies the
        child in the parent's row).
    edges : numpy.ndarray
        Child/parent pairs.
    n : int
        Physical rows.

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` matrix.
    """
    matrix = np.diag(np.asarray(d, dtype=float)[:n])
    for child, parent in np.asarray(edges):
        matrix[child, parent] += float(low[child])
        matrix[parent, child] += float(up[child])
    return matrix
