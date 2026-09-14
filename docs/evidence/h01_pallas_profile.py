"""Experimental single-kernel DHS elimination; not a production backend."""

import argparse
import json
from pathlib import Path
import time

import brainstate
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import numpy as np

from braintrace.datasets.h01_dhs_scan import _prepare_levels, _triang_raw


def _eliminate(d, s, low, up, levels):
    children, parents, valid = levels
    width = children.shape[1]
    if width != 32:
        raise ValueError('Prototype requires the installed 32-wide schedule')

    def kernel(di, si, lr, ur, cr, pr, vr, dr, sr):
        batch = pl.program_id(0)
        lanes = jnp.arange(32)

        def step(index, unused):
            c = plt.load(cr.at[index, lanes])
            p = plt.load(pr.at[index, lanes])
            mask = plt.load(vr.at[index, lanes])
            multiplier = plt.load(ur.at[c])/plt.load(dr.at[batch, c])
            delta_d = -plt.load(lr.at[c])*multiplier
            delta_s = -plt.load(sr.at[batch, c])*multiplier
            plt.atomic_add(dr, (batch, p), delta_d, mask=mask)
            plt.atomic_add(sr, (batch, p), delta_s, mask=mask)
            plt.debug_barrier()
            return unused

        jax.lax.fori_loop(0, children.shape[0], step, None)

    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct(d.shape, d.dtype), jax.ShapeDtypeStruct(s.shape, s.dtype)),
        grid=(d.shape[0],), input_output_aliases={0: 0, 1: 1},
        compiler_params=plt.CompilerParams(num_warps=1, num_stages=1))(
            d, s, low, up, children, parents, valid)


def main():
    """Compare a single-kernel prototype to the existing elimination oracle.

    Returns
    -------
    None
        Prints timings and numerical errors for anatomical tree schedules.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    with brainstate.environ.context(precision=64):
        for path in sorted(args.directory.glob('tree-*.npz')):
            with np.load(path) as data:
                edges, offsets, n = data['edges'], data['offsets'], int(data['n_point'])
            levels = _prepare_levels(edges, offsets, n)
            d = jnp.asarray(4.+np.bincount(edges[:, 1], minlength=n+1))[None]
            rhs = jnp.sin(jnp.arange(n+1, dtype=jnp.float64))[None]
            low = jnp.full(n+1, -.001).at[0].set(0.).at[-1].set(0.)
            original = jax.jit(lambda d, rhs: _triang_raw(d, rhs, low, 1.3*low, levels))
            candidate = jax.jit(lambda d, rhs: _eliminate(d, rhs, low, 1.3*low, levels))
            expected = jax.block_until_ready(original(d, rhs))
            actual = jax.block_until_ready(candidate(d, rhs))
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
            started = time.perf_counter()
            jax.block_until_ready(original(d, rhs))
            old_time = time.perf_counter()-started
            started = time.perf_counter()
            actual = jax.block_until_ready(candidate(d, rhs))
            new_time = time.perf_counter()-started
            print(json.dumps(dict(tree=path.name, old_s=old_time, new_s=new_time,
                max_error=max(float(jnp.max(jnp.abs(a-b))) for a, b in zip(actual, expected)))), flush=True)


if __name__ == '__main__':
    main()
