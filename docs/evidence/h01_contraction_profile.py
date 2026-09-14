"""Evidence-only parallel tree contraction; no model or learning substitution."""

import json
from pathlib import Path
import time

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_dhs_scan import _prepare_levels, _solve_raw


def schedule(parents):
    """Return independent elimination stages for a rooted tree.

    Parameters
    ----------
    parents : array-like
        Parent index for each node, with root zero its own parent.

    Returns
    -------
    list of tuple
        Removed nodes, surviving parents, and optional surviving children.
    """
    parent = np.asarray(parents, dtype=np.int32).copy()
    n = len(parent)
    active = np.ones(n, dtype=bool)
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    order = [0]
    for i in order:
        order.extend(children[i])
    assert len(order) == n
    stages = []
    while active.sum() > 1:
        ids = np.flatnonzero(active)[1:]
        counts = np.bincount(parent[ids], minlength=n)
        child = np.full(n, n, dtype=np.int32)
        child[parent[ids]] = ids
        depth = np.zeros(n, dtype=np.int32)
        for i in order[1:]:
            if active[i]:
                depth[i] = depth[parent[i]] + 1
        selected = active & (counts <= 1) & (depth % 2 == 0)
        selected[0] = False
        selected |= active & (counts == 0) & ~selected[parent]
        selected[0] = False
        removed = np.flatnonzero(selected).astype(np.int32)
        assert len(removed) and not selected[parent[removed]].any()
        down = child[removed]
        assert not selected[down[down < n]].any()
        stages.append((removed, parent[removed].copy(), down))
        has_child = down < n
        parent[down[has_child]] = parent[removed[has_child]]
        active[removed] = False
    return stages


def solve(d, rhs, low, up, stages):
    """Solve using compiled independent scalar Schur complements.

    Parameters
    ----------
    d, rhs, low, up : arrays
        Diagonal, RHS, and edge coefficients including the neutral sentinel.
    stages : list of tuple
        Static schedule from :func:`schedule`.

    Returns
    -------
    array
        Solution with zero in the sentinel position.
    """
    n = d.shape[0]-1
    saved = []
    # Static numerical stages are unrolled while tracing, not model time steps.
    for removed, parent, child in stages:
        diagonal, value = d[removed], rhs[removed]
        lower, upper = low[removed], up[removed]
        child_lower, child_upper = low[child], up[child]
        saved.append((diagonal, value, lower, child_upper))
        d = d.at[parent].add(-upper*lower/diagonal)
        rhs = rhs.at[parent].add(-upper*value/diagonal)
        target = jnp.where(child < n, child, n+1)
        d = d.at[target].add(-child_lower*child_upper/diagonal, mode='drop')
        rhs = rhs.at[target].add(-child_lower*value/diagonal, mode='drop')
        low = low.at[target].set(-child_lower*lower/diagonal, mode='drop')
        up = up.at[target].set(-upper*child_upper/diagonal, mode='drop')
    result = jnp.zeros_like(rhs).at[0].set(rhs[0]/d[0])
    for (removed, parent, child), (diagonal, value, lower, child_upper) in zip(reversed(stages), reversed(saved)):
        result = result.at[removed].set((value-lower*result[parent]-child_upper*result[child])/diagonal)
    return result


def main():
    """Check dense examples and report actual-tree synchronized timings.

    Returns
    -------
    None
        Writes one JSON record per verified anatomical tree to stdout.
    """
    with brainstate.environ.context(precision=64):
        for parents in ([0, 0, 1, 2, 3, 4], [0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 0]):
            n = len(parents)
            d = jnp.full(n+1, 4.).at[-1].set(1.)
            rhs = jnp.arange(n+1, dtype=jnp.float64).at[-1].set(0.)
            low = jnp.full(n+1, -.1).at[0].set(0.).at[-1].set(0.)
            up = low*1.3
            matrix = np.diag(np.asarray(d))
            for i in range(1, n):
                matrix[i, parents[i]] = low[i]
                matrix[parents[i], i] = up[i]
            stages = schedule(parents)
            contracted = lambda d, r, l, u: solve(d, r, l, u, stages)
            def dense(d, r, l, u):
                rows = jnp.arange(1, n)
                ancestor = jnp.asarray(parents[1:])
                matrix = jnp.diag(d).at[rows, ancestor].set(l[rows])
                matrix = matrix.at[ancestor, rows].set(u[rows])
                return jnp.linalg.solve(matrix, r)
            out = jax.jit(contracted)(d, rhs, low, up)
            np.testing.assert_allclose(out, np.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
            # The sentinel is neutral and excluded from the physical solution.
            for derivative in (jax.jacfwd, jax.jacrev):
                actual = derivative(lambda *x: contracted(*x)[:-1], argnums=(0, 1, 2, 3))(d, rhs, low, up)
                expected = derivative(lambda *x: dense(*x)[:-1], argnums=(0, 1, 2, 3))(d, rhs, low, up)
                for left, right in zip(actual, expected):
                    np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)
        print(json.dumps(dict(dense_oracles=3, derivative_modes=['jacfwd', 'jacrev'], status='pass')), flush=True)
        for path in sorted(Path('/tmp/h01-real-trees-20260914').glob('tree-*.npz')):
            data = np.load(path)
            edges, n = data['edges'], int(data['n_point'])
            parents = np.zeros(n, dtype=np.int32)
            parents[edges[:, 0]] = edges[:, 1]
            started = time.perf_counter()
            stages = schedule(parents)
            schedule_seconds = time.perf_counter()-started
            levels = _prepare_levels(edges, data['offsets'], n)
            d = jnp.asarray(4.+np.bincount(edges[:, 1], minlength=n+1)).at[-1].set(1.)
            rhs = jnp.sin(jnp.arange(n+1, dtype=jnp.float64)).at[-1].set(0.)
            low = jnp.full(n+1, -.001).at[0].set(0.).at[-1].set(0.)
            up = 1.3*low
            baseline = jax.jit(lambda r: _solve_raw(d[None], r[None], low, up, levels, data['jumps'], edges)[0])
            candidate = jax.jit(lambda r: solve(d, r, low, up, stages))
            expected = baseline(rhs)
            started = time.perf_counter()
            output = jax.block_until_ready(candidate(rhs))
            cold = time.perf_counter()-started
            np.testing.assert_allclose(output, expected, rtol=1e-11, atol=1e-12)
            timings = {}
            for name, call in [('baseline', baseline), ('contraction', candidate)]:
                started = time.perf_counter()
                jax.block_until_ready(call(rhs))
                timings[name] = time.perf_counter()-started
            print(json.dumps(dict(tree=path.name, nodes=n, stages=len(stages), schedule_seconds=schedule_seconds,
                cold_seconds=cold, max_error=float(jnp.max(jnp.abs(output-expected))), **timings)), flush=True)


if __name__ == '__main__':
    main()
