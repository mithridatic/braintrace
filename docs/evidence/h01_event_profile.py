"""Profile real H01 event execution at the installed qualified clock."""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import resource
import time

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_network import make_h01_network
from braintrace.datasets.h01_network_init import init_h01_network_states
from examples.h01_arc_probe import _sparse_learning_probe
from examples.pp_prop.h01_arc_model import H01ArcModel
from examples.pp_prop.h01_session import numerical_settings


def main():
    """Write a synchronized profile after every completed runtime phase.

    Returns
    -------
    None
        Writes JSON and cProfile files to the requested output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=int, choices=(4, 12, 40, 104), default=4)
    parser.add_argument('--forward-only', action='store_true')
    parser.add_argument('--export-trees-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', phases={}, cells=args.cells, settings=numerical_settings())

    def save():
        report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def phase(name, call):
        report['active_phase'] = name
        save()
        profiler = cProfile.Profile()
        started = time.perf_counter()
        try:
            result = profiler.runcall(call)
            jax.block_until_ready(result)
            report['phases'][name] = dict(seconds=time.perf_counter()-started, status='pass')
            return result
        finally:
            profiler.dump_stats(str(args.output/(name+'.prof')))
            with (args.output/(name+'.txt')).open('w') as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
            save()
            print(name, report['phases'].get(name, {'status': 'incomplete'}), flush=True)

    try:
        with brainstate.environ.context(precision=64):
            topology = json.loads(Path('docs/evidence/h01-verified-network.json').read_text())
            components = json.loads(Path('docs/evidence/h01-population-components-soma.json').read_text())
            network, evidence = phase('construction', lambda: make_h01_network(
                topology, H01Archive(args.cache/'proofread104.zip'), H01Annotations(args.cache),
                include_isolated=True, cells=args.cells, components=components,
                solver=report['settings']['solver']))
            report['compartments'] = sum(row['n_compartments'] for row in evidence['cells'].values())
            phase('initialization', lambda: init_h01_network_states(network))
            report['tree_padding'] = [dict(
                edges=len(cell._runtime.dhs_static_source_np.edges_np),
                levels=len(cell._runtime.dhs_static_source_np.level_offsets_np)-1,
                padded_entries=int(np.prod(cell._runtime.h01_dhs_pack[2][0].shape)))
                for cell in (pop.cell for pop in network.populations.values())]
            if args.export_trees_only:
                for index, population in enumerate(network.populations.values()):
                    source = population.cell._runtime.dhs_static_source_np
                    np.savez(args.output/f'tree-{index}.npz', edges=source.edges_np,
                        offsets=source.level_offsets_np, jumps=source.backsub_indices_np,
                        n_point=source.n_point)
                report['status'] = 'trees_exported'
                return
            model = phase('model_setup', lambda: H01ArcModel(network, evidence['simulated_cell_ids'],
                dt_ms=report['settings']['dt_ms'], checkpoint_substeps=report['settings']['checkpoint_substeps']))
            step = brainstate.transform.jit(model.step)
            cold = phase('cold_event', lambda: step(jnp.ones(441)))
            warm = phase('warm_event', lambda: step(jnp.ones(441)))
            report['finite_events'] = bool(np.isfinite(np.asarray((cold, warm))).all())
            report['event_voltages_mv'] = np.asarray((cold, warm)).tolist()
            if args.forward_only:
                report['status'] = 'forward_pass' if report['finite_events'] else 'fail'
                return
            model.reset_episode()
            _sparse_learning_probe(model, report, phase)
            report['status'] = 'pass' if report['finite_events'] and report['learning_probe']['finite'] else 'fail'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
