"""Compare supplied DHS scheduling batches with dependency-height groups."""

import argparse
import json
from pathlib import Path
import time

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_dhs_scan import _prepare_levels, _solve_raw, _triang_raw, _backsub_raw


def _regroup(edges, n):
    height = np.zeros(n+1, dtype=np.int32)
    for child, parent in edges:
        height[parent] = max(height[parent], height[child]+1)
    edge_height = height[edges[:, 0]]
    if not np.all(height[edges[:, 1]] > edge_height):
        raise ValueError('Input edge order is not bottom-up')
    order = np.argsort(edge_height, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(edge_height))].astype(np.int32)
    return edges[order], offsets


def _groups(edges, offsets, n):
    counts = np.diff(offsets)
    buckets = np.array([int(count-1).bit_length() for count in counts])
    boundaries = np.r_[0, np.flatnonzero(np.diff(buckets))+1, len(counts)]
    return tuple(_prepare_levels(edges[offsets[start]:offsets[end]],
        offsets[start:end+1]-offsets[start], n)
        for start, end in zip(boundaries[:-1], boundaries[1:]))


def _grouped_solve(d, rhs, low, up, groups, jumps, edges):
    children, parents = edges[:, 0], edges[:, 1]

    def matvec(value):
        result = (d*value).at[..., children].add(low[children]*value[..., parents])
        return result.at[..., parents].add(up[children]*value[..., children])

    def solve(_, rhs):
        diagonal = d
        for levels in groups:
            diagonal, rhs = _triang_raw(diagonal, rhs, low, up, levels)
        return _backsub_raw(diagonal, rhs, low, jumps)

    def transpose(_, rhs):
        diagonal = d
        for levels in groups:
            diagonal, rhs = _triang_raw(diagonal, rhs, up, low, levels)
        return _backsub_raw(diagonal, rhs, up, jumps)

    return jax.lax.custom_linear_solve(matvec, rhs, solve=solve, transpose_solve=transpose)


def main():
    """Compare exact linear solves on exported anatomical tree topology.

    Returns
    -------
    None
        Prints timing and numerical parity records for each exported tree.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    with brainstate.environ.context(precision=64):
        for path in sorted(args.directory.glob('tree-*.npz')):
            with np.load(path) as data:
                edges, offsets, jumps, n = data['edges'], data['offsets'], data['jumps'], int(data['n_point'])
            reordered, new_offsets = _regroup(edges, n)
            old = _prepare_levels(edges, offsets, n)
            groups = _groups(reordered, new_offsets, n)
            d = jnp.asarray(4.+np.bincount(edges[:, 1], minlength=n+1))[None]
            rhs = jnp.sin(jnp.arange(n+1, dtype=jnp.float64))[None]
            low = jnp.full(n+1, -.001).at[0].set(0.).at[-1].set(0.)
            baseline = jax.jit(lambda rhs: _solve_raw(d, rhs, low, 1.3*low, old, jumps, edges, use_gpu=False))
            candidate = jax.jit(lambda rhs: _grouped_solve(d, rhs, low, 1.3*low, groups, jumps, edges))
            jax.block_until_ready(baseline(rhs))
            jax.block_until_ready(candidate(rhs))
            started = time.perf_counter()
            expected = jax.block_until_ready(baseline(rhs))
            old_time = time.perf_counter()-started
            started = time.perf_counter()
            actual = jax.block_until_ready(candidate(rhs))
            new_time = time.perf_counter()-started
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            print(json.dumps(dict(tree=path.name, nodes=n,
                old_shape=old[0].shape, group_shapes=[group[0].shape for group in groups],
                padded_entries=sum(int(np.prod(group[0].shape)) for group in groups),
                old_s=old_time, new_s=new_time,
                max_error=float(jnp.max(jnp.abs(actual-expected))))), flush=True)


if __name__ == '__main__':
    main()
