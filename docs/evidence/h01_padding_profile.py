"""Bounded padded DHS elimination benchmark on an uneven synthetic tree."""

import json
import time

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_dhs_scan import _prepare_levels, _triang_raw
from braintrace.datasets.h01_dhs_scan_test import tree


def main():
    """Measure one cold and two synchronized warm elimination calls.

    Returns
    -------
    None
        Prints timings and numerical checksums as JSON.
    """
    parent = [0]+list(range(128))+[128]*4096
    edges, offsets, _ = tree(parent)
    with brainstate.environ.context(precision=64):
        levels = _prepare_levels(edges, offsets, len(parent))
        n = len(parent)+1
        d = jnp.full((1, n), 100.)
        s = jnp.ones((1, n))
        low = jnp.full(n, -.01)
        execute = jax.jit(lambda d, s: _triang_raw(d, s, low, low, levels))
        started = time.perf_counter()
        result = jax.block_until_ready(execute(d, s))
        cold = time.perf_counter()-started
        started = time.perf_counter()
        result = jax.block_until_ready(execute(d, s))
        warm1 = time.perf_counter()-started
        started = time.perf_counter()
        result = jax.block_until_ready(execute(d, s))
        warm2 = time.perf_counter()-started
        print(json.dumps(dict(cold_s=cold, warm_s=[warm1, warm2], nodes=len(parent),
            padded_entries=int(np.prod(levels[0].shape)), edges=len(edges),
            sums=[float(jnp.sum(x)) for x in result], device=str(jax.devices()[0]))))


if __name__ == '__main__':
    main()
