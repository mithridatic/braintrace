"""Bounded CPU probe of compiled spatial calcium cost; no physiological claim."""

import argparse
import json
import os
from pathlib import Path
import time

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import psutil

from braintrace.biophysics.astrocyte_network import AstrocyteCalcium


def main():
    """Measure construction, first compiled call and warmed physical stepping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', type=int, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.segments <= 1024 or not 1 <= args.steps <= 100:
        raise ValueError('This local calibration probe is bounded to 1024 segments and 100 steps')
    with brainstate.environ.context(precision=64):
        n = args.segments
        edges = np.column_stack((np.arange(n-1), np.arange(1, n)))
        begin = time.perf_counter()
        model = AstrocyteCalcium(np.ones(n), np.ones(n), edges)
        construction = time.perf_counter()-begin
        def advance():
            def step(_):
                model.update(jnp.full(n, .1))
            brainstate.transform.for_loop(step, jnp.arange(args.steps))
            return model.valid.value, model.state.value
        advance = brainstate.transform.jit(advance)
        begin = time.perf_counter()
        first = jax.block_until_ready(advance())
        cold = time.perf_counter()-begin
        begin = time.perf_counter()
        warmed = jax.block_until_ready(advance())
        warm = time.perf_counter()-begin
        report = dict(segments=n, steps_per_call=args.steps, dt_ms=.005,
            solver='coupled_backward_euler_newton_krylov', substeps=model.substeps,
            physical_ms_per_call=.005*args.steps, construction_seconds=construction,
            compile_and_first_seconds=cold, warmed_seconds=warm,
            physical_ms_per_wall_second=.005*args.steps/warm,
            rss_bytes=psutil.Process(os.getpid()).memory_info().rss,
            device=[str(d) for d in jax.devices()], valid=bool(first[0] & warmed[0]),
            qualification='synthetic linear geometry runtime probe only')
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
