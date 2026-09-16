"""Compare cable timesteps across a spiking window of the Example 21 H01 network.

Spec: docs/specs/2026-09-16-h01-fused-population.md, deliverable 0. The evolve
profile (h01-evolve-event-profile.md, section 5) compared dt 0.005 ms against the
pinned 0.000625 ms over 2 ms of subthreshold ARC events only. This script drives
the same ``H01ArcModel`` (same ``build_network`` / ``init_h01_network_states`` /
``H01NetworkStep`` path the manifest pins) through a window in which the kept
cells fire: each cell's soma receives its donor primary current
(``kept-currents.json``, the keep-test probe) from ``--pulse-delay-ms`` to the end
of the window, held through the model's ``drive`` state instead of the ARC
encoder. Soma voltages are recorded at every cable substep so that spike times
are resolved at the finer clock's resolution.

``arm``
    GPU. One timestep; writes ``report.json`` and ``soma_mv.npz``.
``report``
    Pure. Joins arms against the reference arm: max |dV| per cell over the
    window, spike counts, spike-time shifts; writes the evidence JSON and MD.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

EVENT_MS = .1
SPIKE_THRESHOLD_MV = 0.   # braincell Cell.V_th default; the keep tests count output-site crossings of it


def spike_times(voltage_mv, dt_ms, threshold_mv=SPIKE_THRESHOLD_MV):
    """Upward threshold crossings of one soma trace.

    Parameters
    ----------
    voltage_mv : array-like
        Soma voltage per cable substep, in mV.
    dt_ms : float
        Substep length in ms.
    threshold_mv : float, optional
        Crossing threshold.

    Returns
    -------
    numpy.ndarray
        Times in ms of the first substep at or above the threshold after a substep below it.
    """
    trace = np.asarray(voltage_mv, dtype=float)
    above = trace >= threshold_mv
    crossings = np.flatnonzero(above[1:] & ~above[:-1])+1
    return crossings*float(dt_ms)


def compare_cells(reference_mv, reference_dt, other_mv, other_dt, cell_ids):
    """Per-cell voltage and spike comparison of two arms over a shared window.

    Parameters
    ----------
    reference_mv, other_mv : array-like
        ``(substeps, cells)`` soma voltages of each arm.
    reference_dt, other_dt : float
        Substep length of each arm in ms; ``other_dt`` must be a multiple of ``reference_dt``.
    cell_ids : sequence of str
        Cell identities in column order.

    Returns
    -------
    dict
        Per-cell max |dV| (sampled at the coarser clock), spike counts on both arms
        and the shift of each matched spike, plus window-wide maxima.
    """
    a, b = np.asarray(reference_mv, dtype=float), np.asarray(other_mv, dtype=float)
    ratio = other_dt/reference_dt
    if not np.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError('The other timestep must be an integer multiple of the reference')
    stride = int(round(ratio))
    n = min(len(a)//stride, len(b))
    sampled, coarse = a[stride-1:stride*n:stride], b[:n]
    delta = np.abs(sampled-coarse)
    cells = {}
    for index, identity in enumerate(cell_ids):
        ref_spikes = spike_times(a[:stride*n, index], reference_dt)
        oth_spikes = spike_times(b[:n, index], other_dt)
        matched = min(len(ref_spikes), len(oth_spikes))
        shifts = (oth_spikes[:matched]-ref_spikes[:matched]).tolist()
        cells[identity] = dict(max_abs_mv=float(delta[:, index].max()),
            reference_spikes=int(len(ref_spikes)), other_spikes=int(len(oth_spikes)),
            counts_equal=bool(len(ref_spikes) == len(oth_spikes)),
            reference_spike_times_ms=ref_spikes.tolist(), other_spike_times_ms=oth_spikes.tolist(),
            spike_shift_ms=shifts, max_abs_spike_shift_ms=(float(np.max(np.abs(shifts))) if shifts else None),
            reference_min_mv=float(a[:stride*n, index].min()), reference_max_mv=float(a[:stride*n, index].max()))
    return dict(window_ms=n*other_dt, compared_substeps=int(n), stride=stride, cells=cells,
        max_abs_mv=float(delta.max()), cells_spiking_reference=sum(c['reference_spikes'] > 0 for c in cells.values()),
        total_reference_spikes=sum(c['reference_spikes'] for c in cells.values()),
        total_other_spikes=sum(c['other_spikes'] for c in cells.values()),
        all_counts_equal=all(c['counts_equal'] for c in cells.values()),
        max_abs_spike_shift_ms=max((c['max_abs_spike_shift_ms'] or 0.) for c in cells.values()),
        within_1mv=bool(delta.max() <= 1.))


def verdict(comparisons):
    """Pick the coarsest timestep that holds the 1 mV / identical-count contract.

    Parameters
    ----------
    comparisons : dict
        ``dt_ms -> compare_cells(...) output`` for every non-reference arm.

    Returns
    -------
    dict
        The passing and failing timesteps and the coarsest passing one, if any.
    """
    passing = sorted((dt for dt, c in comparisons.items() if c['within_1mv'] and c['all_counts_equal']), reverse=True)
    failing = sorted((dt for dt in comparisons if dt not in passing), reverse=True)
    return dict(passing_dt_ms=passing, failing_dt_ms=failing,
                coarsest_passing_dt_ms=(passing[0] if passing else None),
                contract='max |dV| <= 1 mV on every cell and identical spike counts per cell')


def _arm_command(args):
    import brainstate
    import jax
    import jax.numpy as jnp
    import brainunit as u
    from braintrace.datasets.h01_network_init import init_h01_network_states
    from examples.pp_prop.h01_arc_adapter import H01ArcAdapter
    from examples.pp_prop.h01_arc_model import H01ArcModel
    from examples.pp_prop.h01_runtime import build_network

    args.output.mkdir(parents=True, exist_ok=True)
    substeps = int(round(EVENT_MS/args.dt_ms))
    events = int(round(args.window_ms/EVENT_MS))
    report = dict(status='running', arm=args.name, settings=dict(dt_ms=args.dt_ms, substeps=substeps,
        window_ms=args.window_ms, events=events, pulse_delay_ms=args.pulse_delay_ms,
        currents=str(args.currents), precision=args.precision, manifest=str(args.manifest)), stages={})

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def stage(name, call):
        started = time.perf_counter()
        result = call()
        jax.block_until_ready(result)
        report['stages'][name] = time.perf_counter()-started
        save()
        print(name, round(report['stages'][name], 3), 's', flush=True)
        return result

    try:
        with brainstate.environ.context(precision=args.precision):
            adapter = stage('manifest_load', lambda: H01ArcAdapter(args.arc_root, args.manifest))
            settings = adapter.document['settings']
            topology = adapter.initial_topology
            cells = topology.to_dict()['active_cells']
            currents = json.loads(Path(args.currents).read_text())
            amplitude = np.asarray([float(currents[identity]) for identity in cells])
            report['cells'] = cells
            report['current_na'] = amplitude.tolist()
            archive = stage('archive_open', adapter._archive)
            network, records = stage('build_network', lambda: build_network(topology, archive,
                solver=settings['solver'], max_cv_length_um=settings['max_cv_length_um']))
            report['compartments'] = int(sum(r['n_compartments'] for r in records.values()))
            del records
            stage('init_state', lambda: init_h01_network_states(network))
            model = stage('model_setup', lambda: H01ArcModel(network, cells, seed=settings['seed'],
                dt_ms=args.dt_ms, checkpoint_substeps=False))
            drive = jnp.asarray(amplitude)
            delay_substeps = int(round(args.pulse_delay_ms/args.dt_ms))

            def cable_step(_):
                model.drive.value = jnp.where(model.stepper.tick.value >= delay_substeps, drive, 0.)
                model.stepper.update(sample_probes=False)
                spikes = jnp.stack([jnp.sum(cell.spike.value > .5) for cell in model.stepper.cells])
                return model._soma(), spikes

            def event():
                return brainstate.transform.for_loop(cable_step, jnp.arange(substeps))

            step = brainstate.transform.jit(event)
            soma, spikes = stage('first_event_compile_and_run', step)
            soma_mv, output_spikes, per_event = [np.asarray(soma)], [np.asarray(spikes)], []
            for index in range(1, events):
                started = time.perf_counter()
                soma, spikes = jax.block_until_ready(step())
                per_event.append(time.perf_counter()-started)
                soma_mv.append(np.asarray(soma))
                output_spikes.append(np.asarray(spikes))
                if index % 50 == 0 or index == events-1:
                    trace = np.concatenate(soma_mv)
                    report['forward'] = dict(events=index+1, seconds_per_event_median=float(np.median(per_event)),
                        total_seconds=float(np.sum(per_event)))
                    report['soma_min_mv'] = float(trace.min())
                    report['soma_max_mv'] = float(trace.max())
                    save()
                    print('event', index, 'soma max', round(float(trace.max()), 2), flush=True)
            trace = np.concatenate(soma_mv)
            output = np.concatenate(output_spikes)
            np.savez_compressed(args.output/'soma_mv.npz', soma_mv=trace, output_spikes=output,
                                dt_ms=args.dt_ms, cells=np.asarray(cells))
            report['finite'] = bool(np.isfinite(trace).all())
            report['spikes'] = {identity: dict(soma_crossings=spike_times(trace[:, i], args.dt_ms).tolist(),
                                               output_site_events=int(output[:, i].sum()))
                                for i, identity in enumerate(cells)}
            report['status'] = 'pass' if report['finite'] else 'nonfinite'
    except Exception as error:   # noqa: BLE001 - the receipt must name the failure
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        save()


def build_report(arms, reference_dt):
    """Join arm receipts into the evidence document.

    Parameters
    ----------
    arms : dict
        ``dt_ms -> (report dict, soma voltages (substeps, cells))``.
    reference_dt : float
        The pinned timestep every other arm is compared against.

    Returns
    -------
    dict
        Comparisons per timestep and the verdict.
    """
    report, reference = arms[reference_dt]
    cells = report['cells']
    comparisons = {dt: compare_cells(reference, reference_dt, volts, dt, cells)
                   for dt, (_, volts) in arms.items() if dt != reference_dt}
    return dict(schema='h01-dt-spike-window-v1', reference_dt_ms=reference_dt, cells=cells,
        compartments=report.get('compartments'), settings={dt: r['settings'] for dt, (r, _) in arms.items()},
        forward={dt: r.get('forward') for dt, (r, _) in arms.items()},
        stages={dt: r.get('stages') for dt, (r, _) in arms.items()},
        reference_spikes={identity: len(spike_times(reference[:, i], reference_dt)) for i, identity in enumerate(cells)},
        comparisons={str(dt): value for dt, value in comparisons.items()}, verdict=verdict(comparisons))


def _markdown(document):
    lines = ['# H01 timestep across a spiking window (2026-09-16)', '',
             f"Reference dt {document['reference_dt_ms']} ms; {len(document['cells'])} cells, "
             f"{document['compartments']} compartments; window "
             f"{next(iter(document['comparisons'].values()))['window_ms']:g} ms. "
             'Script: `h01_dt_spike_window.py`; JSON: `h01-dt-spike-window.json`.', '',
             '| dt (ms) | s/event | max abs dV (mV) | cells spiking (ref) | spikes ref / other | counts equal | max spike shift (ms) | within 1 mV |',
             '| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for dt, c in document['comparisons'].items():
        forward = document['forward'].get(float(dt)) or document['forward'].get(dt) or {}
        lines.append(f"| {dt} | {forward.get('seconds_per_event_median', float('nan')):.3f} | {c['max_abs_mv']:.4f} | "
                     f"{c['cells_spiking_reference']} | {c['total_reference_spikes']} / {c['total_other_spikes']} | "
                     f"{c['all_counts_equal']} | {c['max_abs_spike_shift_ms']:.4f} | {c['within_1mv']} |")
    lines += ['', f"Verdict: {json.dumps(document['verdict'])}", '', '## Per cell', '']
    for dt, c in document['comparisons'].items():
        lines += [f'### dt {dt} ms', '', '| cell | max abs dV (mV) | spikes ref | spikes other | shifts (ms) |', '| --- | ---: | ---: | ---: | --- |']
        for identity, row in c['cells'].items():
            lines.append(f"| {identity} | {row['max_abs_mv']:.4f} | {row['reference_spikes']} | {row['other_spikes']} | "
                         f"{', '.join(f'{s:+.4f}' for s in row['spike_shift_ms'])} |")
        lines.append('')
    return '\n'.join(lines)+'\n'


def _report_command(args):
    arms = {}
    for directory in args.arms:
        directory = Path(directory)
        report = json.loads((directory/'report.json').read_text())
        with np.load(directory/'soma_mv.npz') as npz:
            arms[float(report['settings']['dt_ms'])] = (report, np.asarray(npz['soma_mv']))
    document = build_report(arms, args.reference_dt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2)+'\n')
    args.output.with_suffix('.md').write_text(_markdown(document))
    print(json.dumps(document['verdict']))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    arm = sub.add_parser('arm')
    arm.add_argument('--name', required=True)
    arm.add_argument('--manifest', type=Path, default=Path('docs/evidence/h01-keep-drop/arc-manifest/manifest.json'))
    arm.add_argument('--arc-root', type=Path, default=Path('var/arc-agi-1'))
    arm.add_argument('--currents', type=Path, default=Path('docs/evidence/h01-keep-drop/kept-currents.json'))
    arm.add_argument('--output', type=Path, required=True)
    arm.add_argument('--dt-ms', type=float, default=.000625)
    arm.add_argument('--window-ms', type=float, default=40.)
    arm.add_argument('--pulse-delay-ms', type=float, default=2.)
    arm.add_argument('--precision', type=int, choices=(32, 64), default=64)
    report = sub.add_parser('report')
    report.add_argument('--arms', type=Path, nargs='+', required=True)
    report.add_argument('--reference-dt', type=float, default=.000625)
    report.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    """Dispatch one subcommand.

    Parameters
    ----------
    argv : sequence of str, optional
        Command line; ``None`` reads ``sys.argv``.
    """
    args = _parser().parse_args(argv)
    if args.command == 'arm':
        _arm_command(args)
    else:
        _report_command(args)


if __name__ == '__main__':
    main()
