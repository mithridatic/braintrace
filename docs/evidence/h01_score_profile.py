"""Bounded GPU scoring profile; fixture timings are not full evolution timings."""

import argparse
import cProfile
import json
import pstats
import resource
import time
from types import SimpleNamespace

import brainstate
import jax
import jax.numpy as jnp

from examples.pp_prop.h01_arc_execution import score_episode, score_queries
from examples.pp_prop.h01_arc_execution_test import ScoreModel, original_score


def main():
    """Profile one cold and two synchronized warm scoring calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=('original', 'optimized'), required=True)
    parser.add_argument('--events', type=int, default=2048)
    parser.add_argument('--queries', action='store_true', help='Include per-rescore JIT wrapper creation/reuse')
    args = parser.parse_args()
    with brainstate.environ.context(precision=64):
        model = ScoreModel()
        session = SimpleNamespace(model=model, learner=None)
        function = original_score if args.implementation == 'original' else score_episode
        execute = brainstate.transform.jit(lambda e, a: function(session, e, a))
        events = jnp.ones((args.events, 441), dtype=jnp.float64)
        advances = jnp.arange(args.events) % 3 != 0
        if args.queries:
            events, advances = events[None], advances[None]
            if args.implementation == 'original':
                def execute(e, a):
                    return brainstate.transform.jit(lambda x, m: brainstate.transform.for_loop(
                        lambda query, mask: original_score(session, query, mask), x, m))(e, a)
            else:
                execute = lambda e, a: score_queries(session, e, a)
        jax.block_until_ready((events, advances))
        profiler = cProfile.Profile()
        started = time.perf_counter()
        result = profiler.runcall(execute, events, advances)
        jax.block_until_ready(result)
        cold = time.perf_counter()-started
        started = time.perf_counter()
        jax.block_until_ready(execute(events, advances))
        warm1 = time.perf_counter()-started
        started = time.perf_counter()
        jax.block_until_ready(execute(events, advances))
        warm2 = time.perf_counter()-started
        print(json.dumps(dict(implementation=args.implementation, events=args.events, queries=args.queries,
            device=str(jax.devices()[0]), cold_s=cold, warm_s=[warm1, warm2],
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            device_memory=jax.devices()[0].memory_stats()), sort_keys=True), flush=True)
        pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(15)


if __name__ == '__main__':
    main()
