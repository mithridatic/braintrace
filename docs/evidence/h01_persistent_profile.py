"""Bounded anatomical-tree comparison of grouped and persistent contraction."""

import json
from pathlib import Path
import time

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_dhs_contraction import prepare, solve as grouped
from h01_contraction_profile import schedule
from h01_persistent_contraction import solve


def main():
    """Print synchronized timings on saved anatomical trees.

    Returns
    -------
    None
        Prints JSON records after each numerical comparison.
    """
    with brainstate.environ.context(precision=64):
        for path in sorted(Path('/tmp/h01-real-trees-20260914').glob('tree-*.npz')):
            data = np.load(path)
            edges, n = data['edges'], int(data['n_point'])
            parents = np.zeros(n, dtype=np.int32)
            parents[edges[:, 0]] = edges[:, 1]
            stages, packs = schedule(parents), prepare(edges, n)
            d = jnp.asarray(4.+np.bincount(edges[:, 1], minlength=n+1))
            rhs = jnp.sin(jnp.arange(n+1, dtype=jnp.float64))
            low = jnp.full(n+1, -.001, dtype=jnp.float64)
            up = low*1.3
            baseline = jax.jit(lambda r: grouped(d, r, low, up, packs))
            candidate = jax.jit(lambda r: solve(d, r, low, up, stages))
            expected = jax.block_until_ready(baseline(rhs))
            started = time.perf_counter()
            result = jax.block_until_ready(candidate(rhs))
            cold = time.perf_counter()-started
            np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-12)
            timings = {}
            for name, call in [('grouped', baseline), ('persistent', candidate)]:
                samples = []
                for _ in range(5):
                    started = time.perf_counter()
                    jax.block_until_ready(call(rhs))
                    samples.append(time.perf_counter()-started)
                timings[name] = samples
            print(json.dumps(dict(tree=path.name, nodes=n, cold=cold,
                                  max_error=float(jnp.max(jnp.abs(result-expected))),
                                  **timings)), flush=True)


if __name__ == '__main__':
    main()
