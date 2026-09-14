"""Validated tree contraction schedules and compact implicit-solve kernels."""

from collections import OrderedDict
import hashlib
import weakref

import jax
import jax.numpy as jnp
import numpy as np

_CACHE = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LIMIT_BYTES = 32*1024**2
_CACHE_LIMIT_ENTRIES = 128


def _discard(key, owner=None):
    global _CACHE_BYTES
    entry = _CACHE.get(key)
    if entry is not None and (owner is None or entry[0] is owner):
        _CACHE_BYTES -= entry[4]
        del _CACHE[key]


def _build(edges, n):
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError('Contraction edge indices are out of range')
    if not np.array_equal(np.sort(edges[:, 0]), np.arange(1, n)):
        raise ValueError('Every nonroot node must appear exactly once as a child')
    parent = np.zeros(n, dtype=np.int32)
    parent[edges[:, 0]] = edges[:, 1]
    children = [[] for _ in range(n)]
    for child, ancestor in edges:
        children[ancestor].append(int(child))
    order = [0]
    for node in order:
        order.extend(children[node])
    if len(order) != n:
        raise ValueError('Contraction topology must be connected and acyclic')
    active = np.ones(n, dtype=bool)
    stages = []
    while active.sum() > 1:
        ids = np.flatnonzero(active)[1:]
        counts = np.bincount(parent[ids], minlength=n)
        child = np.full(n, n, dtype=np.int32)
        child[parent[ids]] = ids
        depth = np.zeros(n, dtype=np.int32)
        for node in order[1:]:
            if active[node]:
                depth[node] = depth[parent[node]]+1
        selected = active & (counts <= 1) & (depth % 2 == 0)
        selected[0] = False
        selected |= active & (counts == 0) & ~selected[parent]
        selected[0] = False
        removed = np.flatnonzero(selected).astype(np.int32)
        down = child[removed]
        stages.append((removed, parent[removed].copy(), down))
        has_child = down < n
        parent[down[has_child]] = parent[removed[has_child]]
        active[removed] = False
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


def prepare(edges, n):
    """Validate and cache a tree's grouped contraction schedule.

    Parameters
    ----------
    edges : numpy.ndarray
        Child/parent pairs for every nonroot node, with root zero.
    n : int
        Number of physical nodes, excluding the independent sentinel.

    Returns
    -------
    tuple of tuple
        Read-only NumPy schedule groups. Cache retention is bounded to 32 MiB
        and ends when the source edge array is released.

    Raises
    ------
    ValueError
        If edges do not describe one valid rooted tree.
    """
    global _CACHE_BYTES
    if not isinstance(edges, np.ndarray):
        raise ValueError('Contraction schedules require static NumPy edges')
    if not isinstance(n, (int, np.integer)) or n < 1 or edges.shape != (n-1, 2) or edges.dtype.kind not in 'iu':
        raise ValueError('Contraction requires an integer (n-1, 2) tree edge array')
    key = id(edges)
    digest = hashlib.blake2b(str((edges.shape, edges.dtype.str)).encode()+edges.tobytes(), digest_size=16).digest()
    entry = _CACHE.get(key)
    if entry is not None:
        if entry[0]() is edges and entry[1] == digest and entry[2] == n:
            _CACHE.move_to_end(key)
            return entry[3]
        _discard(key)
    packs = _build(edges, n)
    size = sum(array.nbytes for group in packs for array in group)
    if size <= _CACHE_LIMIT_BYTES:
        while _CACHE and (_CACHE_BYTES+size > _CACHE_LIMIT_BYTES or len(_CACHE) >= _CACHE_LIMIT_ENTRIES):
            _discard(next(iter(_CACHE)))
        owner = weakref.ref(edges, lambda ref: _discard(key, ref))
        _CACHE[key] = (owner, digest, n, packs, size)
        _CACHE_BYTES += size
    return packs


def solve(d, rhs, low, up, packs):
    """Solve a tree with grouped scans over independent Schur eliminations.

    Parameters
    ----------
    d, rhs, low, up : arrays
        One-dimensional diagonal, RHS and edge coefficients, including an
        independent sentinel row. Caller arrays are not mutated.
    packs : tuple of tuple
        Static schedule from :func:`prepare`.

    Returns
    -------
    array
        Solution including the independent sentinel row.

    Notes
    -----
    Physical callers use this inside implicit linear-solve callbacks. Removed
    rows retain their coefficients for reverse substitution, without a stacked
    whole-system history. No numerical clock or precision is changed.
    """
    n = d.shape[0]-1
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
    result = jnp.zeros_like(rhs).at[0].set(rhs[0]/d[0]).at[-1].set(rhs[-1]/d[-1])

    def substitute(result, indices):
        removed, parent, child = indices
        values = (rhs[removed]-low[removed]*result[parent]-up[removed]*result[child])/d[removed]
        target = jnp.where(removed < n, removed, n+1)
        return result.at[target].set(values, mode='drop'), None

    for packed in reversed(packs):
        result, _ = jax.lax.scan(substitute, result, packed, reverse=True)
    return result
