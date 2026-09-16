"""Compare every state variable of the 17-cell network across cable timesteps, in lockstep.

Spec: docs/specs/2026-09-16-h01-fused-population.md, lever 0 (J's amendment). Four
copies of the fused 17-cell network (kept manifest plus ``--contacts`` synthetic
contacts) run side by side through the 40 ms spiking window of
``h01_dt_spike_window.py``: the reference at dt 0.000625 ms and one copy at each of
0.005 / 0.0025 / 0.00125 ms. Every 0.005 ms all four are at the same simulated
time and every floating-point state of the forest (membrane voltage, every gating
variable of every mechanism, intracellular calcium, synaptic conductances) plus the
derived axial rate (the cable term of dV/dt, mV/ms) and synaptic currents are
compared element by element. Recorded, in the variable's own units:

- per-compartment difference frames (coarse minus reference, and dt 0.005 minus
  dt 0.0025) every ``--frame-ms`` for every variable, float32, in ``frames.npz``;
- per-site traces at every 0.005 ms step for every variable at the soma, the axon
  initial segment and the most distal dendritic compartment of every cell, for
  all four timesteps, in ``sites.npz``;
- spike times per cell per timestep, and an index table (max abs / max relative
  to the reference's range over the window / end-of-window difference), split
  between samples within +-1 ms of a spike of that cell and the rest, in
  ``report.json``.

``plot`` draws the overlaid traces and the raw difference traces per variable for
selected cells from ``sites.npz``.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EVENT_MS = .1
REFERENCE_DT = .000625
LADDER = (.005, .0025, .00125)


def _site_cvs(forest, records, cells):
    """Soma, AIS and most distal dendritic CV per cell (forest CV ids)."""
    cvs = forest.cvs
    offsets = forest.forest_offsets
    distance = np.zeros(len(cvs))
    for cv in cvs:   # ids are topologically ordered within each cell (parents first)
        if cv.parent_cv is not None:
            parent = cvs[cv.parent_cv]
            distance[cv.id] = distance[parent.id]+(float(parent.length.mantissa)+float(cv.length.mantissa))/2.
    sites = {}
    for index, identity in enumerate(cells):
        intervals = records[identity]['electrical_intervals']
        lo, hi, br = int(offsets.cv[index]), int(offsets.cv[index+1]), int(offsets.branch[index])
        def region_of(cv):
            for name, rows in intervals.items():
                for branch, a, b in rows:
                    if cv.branch_id-br == branch and a-1e-9 <= (cv.prox+cv.dist)/2 <= b+1e-9:
                        return name
            return None
        members = [(cv, region_of(cv)) for cv in cvs[lo:hi]]
        soma = int(forest.soma_cv_ids[index])
        axon = [cv.id for cv, region in members if region == 'axon']
        dend = [cv.id for cv, region in members if region in ('dend', 'apic')]
        ais = min(axon, key=lambda i: distance[i]) if axon else soma
        distal = max(dend, key=lambda i: distance[i]) if dend else soma
        sites[identity] = dict(soma=soma, ais=ais, distal=distal,
                               distance_um=dict(soma=float(distance[soma]), ais=float(distance[ais]), distal=float(distance[distal])))
    return sites


def _variables(model):
    """Named floating-point forest states with last axis n_cv or n_point, plus synapse conductances."""
    import brainstate
    import brainunit as u
    forest = model.forest
    n_cv, n_point = forest.n_cv, forest.runtime.n_point
    found = {}
    for path, state in brainstate.graph.states(forest).items():
        value = u.get_mantissa(state.value)
        if not hasattr(value, 'shape') or not np.issubdtype(np.asarray(value).dtype, np.floating):
            continue
        name = '.'.join(str(p) for p in path if p not in ('ion_channels', 'channels', '_channel'))
        if name in ('spike', '_current_time_state'):
            continue
        last = value.shape[-1] if value.ndim else None
        if last in (n_cv, n_point) and value.ndim == 2:
            found[name] = dict(state=state, axis='cv' if last == n_cv else 'point', unit=str(u.get_unit(state.value)))
        elif name.endswith('.g') and value.ndim == 2 and last == 1:
            found[name] = dict(state=state, axis='contact', unit=str(u.get_unit(state.value)))
    return found


def _axial_rate(forest):
    """Cable term of dV/dt per point row (mV/ms) from the DHS coefficients."""
    import jax.numpy as jnp
    import brainunit as u
    source = forest.runtime.dhs_static_source_np
    v = jnp.zeros(source.n_point).at[jnp.asarray(source.dynamic_rows_np)].set(
        forest.V.value.to_decimal(u.mV).reshape(-1))
    child, parent = source.edges_np[:, 0], source.edges_np[:, 1]
    rate = -jnp.asarray(source.diag_ms_inv_np)*v
    rate = rate.at[child].add(-jnp.asarray(source.lowers_ms_inv_np)[child]*v[parent])
    rate = rate.at[parent].add(-jnp.asarray(source.uppers_ms_inv_np)[child]*v[child])
    return rate[None, :]


def _synaptic_current(model):
    """Per-contact synaptic current (nA) = g (uS) * (V_post_point - E)."""
    import jax.numpy as jnp
    import brainunit as u
    forest = model.forest
    currents = []
    for block in model.stepper.setup.delivery_blocks:
        layout = forest.runtime.layouts[int(block.source.layout_id)]
        node = forest.runtime.runtime_nodes[layout.id]
        point = int(layout.point_index[0])
        cv = int(np.flatnonzero(forest.runtime.node_tree.cv_to_mid_node_id == point)[0])
        g = u.get_mantissa(node.g.value.in_unit(u.uS)).reshape(-1)[0]
        e = float(u.get_mantissa(node.E.in_unit(u.mV)).reshape(-1)[0]) if hasattr(node, 'E') else 0.
        currents.append(g*(forest.V.value.to_decimal(u.mV).reshape(-1)[cv]-e))
    return jnp.stack(currents)[None, :] if currents else jnp.zeros((1, 0))


def _build(args, dt_ms, settings, topology, archive, cells):
    from braintrace.datasets.h01_network_init import init_h01_network_states
    from examples.pp_prop.h01_arc_model import H01ArcModel
    from examples.pp_prop.h01_runtime import build_network
    network, records = build_network(topology, archive, solver=settings['solver'],
                                     max_cv_length_um=settings['max_cv_length_um'], fused=True)
    init_h01_network_states(network)
    model = H01ArcModel(network, cells, seed=settings['seed'], dt_ms=dt_ms, checkpoint_substeps=False)
    return model, records


def _run(args):
    import brainstate
    import jax
    import jax.numpy as jnp
    import brainunit as u
    from examples.pp_prop.h01_arc_adapter import H01ArcAdapter
    from docs.evidence.h01_dt_spike_window import spike_times

    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', settings=dict(reference_dt_ms=REFERENCE_DT, ladder_dt_ms=list(args.ladder),
        window_ms=args.window_ms, frame_ms=args.frame_ms, contacts=args.contacts, contact_weight_us=args.contact_weight_us,
        pulse_delay_ms=args.pulse_delay_ms, fused=True, precision=64), stages={})

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    started = time.perf_counter()
    with brainstate.environ.context(precision=64):
        adapter = H01ArcAdapter(args.arc_root, args.manifest)
        settings = adapter.document['settings']
        topology = adapter.initial_topology
        cells = topology.to_dict()['active_cells']
        for index in range(args.contacts):
            topology = topology.add_contact(cells[index % len(cells)], cells[(index+1) % len(cells)],
                                            stage='dt-state-check', weight_us=args.contact_weight_us)
        report['cells'] = cells
        report['contacts_table'] = {k: dict(pre=v['pre'], post=v['post'], reversal_mv=v['reversal_mv'], tau_ms=v['tau_ms'],
                                            delay_ms=v['delay_ms'], weight_us=v['initial_weight_us'])
                                    for k, v in topology.to_dict()['contacts'].items()}
        currents = json.loads(Path(args.currents).read_text())
        amplitude = jnp.asarray([float(currents[identity]) for identity in cells])
        archive = adapter._archive()
        dts = (REFERENCE_DT,)+tuple(args.ladder)
        models = {}
        for dt in dts:
            models[dt], records = _build(args, dt, settings, topology, archive, cells)
            print('built', dt, round(time.perf_counter()-started, 1), 's', flush=True)
        report['stages']['build_all'] = time.perf_counter()-started
        report['compartments'] = int(models[REFERENCE_DT].forest.n_cv)
        sites = _site_cvs(models[REFERENCE_DT].forest, records, cells)
        report['sites'] = sites
        report['forest_offsets'] = dict(cv=models[REFERENCE_DT].forest.forest_offsets.cv.tolist(),
                                        point=models[REFERENCE_DT].forest.forest_offsets.point.tolist())
        variables = {dt: _variables(model) for dt, model in models.items()}
        names = list(variables[REFERENCE_DT])
        report['variables'] = {name: dict(axis=variables[REFERENCE_DT][name]['axis'], unit=variables[REFERENCE_DT][name]['unit'],
                                          shape=list(np.shape(u.get_mantissa(variables[REFERENCE_DT][name]['state'].value))))
                               for name in names}
        report['variables']['axial_rate'] = dict(axis='point', unit='mV/ms', shape=[1, models[REFERENCE_DT].forest.runtime.n_point])
        report['variables']['syn_current'] = dict(axis='contact', unit='nA', shape=[1, args.contacts])
        forest0 = models[REFERENCE_DT].forest
        point_of_cv = np.asarray(forest0.runtime.node_tree.cv_to_mid_node_id)
        site_index = {}
        for identity, row in sites.items():
            for site, cv in ((k, row[k]) for k in ('soma', 'ais', 'distal')):
                site_index[(identity, site)] = dict(cv=int(cv), point=int(point_of_cv[cv]))
        site_cv = np.asarray([v['cv'] for v in site_index.values()])
        site_point = np.asarray([v['point'] for v in site_index.values()])
        report['site_order'] = [list(k) for k in site_index]
        coarse_step = max(dts)   # the lockstep clock: every model lands here
        ratios = {dt: int(round(coarse_step/dt)) for dt in dts}
        steps = int(round(args.window_ms/coarse_step))
        frame_every = int(round(args.frame_ms/coarse_step))
        delay_substeps = {dt: int(round(args.pulse_delay_ms/dt)) for dt in dts}

        def values(dt):
            model = models[dt]
            out = {}
            for name, var in variables[dt].items():
                out[name] = u.get_mantissa(var['state'].value)
            out['axial_rate'] = _axial_rate(model.forest)
            out['syn_current'] = _synaptic_current(model)
            return out

        def advance(dt):
            model = models[dt]
            def substep(_):
                model.drive.value = jnp.where(model.stepper.tick.value >= delay_substeps[dt], amplitude, 0.)
                model.stepper.update(sample_probes=False)
            brainstate.transform.for_loop(substep, jnp.arange(ratios[dt]))

        def gather(vals):
            out = {}
            for name, value in vals.items():
                axis = report['variables'][name]['axis']
                if axis == 'cv':
                    out[name] = value[0, site_cv]
                elif axis == 'point':
                    out[name] = value[0, site_point]
                else:
                    out[name] = value[0]
            return out

        def step(_):
            for dt in dts:
                advance(dt)
            current = {dt: values(dt) for dt in dts}
            return {str(dt): gather(current[dt]) for dt in dts}

        def chunk(count):
            return brainstate.transform.for_loop(step, jnp.arange(count))

        chunks = {}
        def snapshot():
            current = {dt: values(dt) for dt in dts}
            frame = {}
            for name in current[REFERENCE_DT]:
                frame[f'ref/{name}'] = np.asarray(current[REFERENCE_DT][name], dtype=np.float32)[0]
            for dt in args.ladder:
                for name in current[dt]:
                    frame[f'{dt}-vs-ref/{name}'] = np.asarray(current[dt][name]-current[REFERENCE_DT][name], dtype=np.float32)[0]
            for name in current[.005]:
                if .005 in dts and .0025 in dts:
                    frame[f'0.005-vs-0.0025/{name}'] = np.asarray(current[.005][name]-current[.0025][name], dtype=np.float32)[0]
            return frame
        frames, frame_times, site_traces = [], [], []
        frames.append(snapshot()); frame_times.append(0.)
        done = 0
        stage_started = time.perf_counter()
        while done < steps:
            count = min(frame_every, steps-done)
            if count not in chunks:
                chunks[count] = brainstate.transform.jit(lambda c=count: chunk(c))
            traces = jax.block_until_ready(chunks[count]())
            site_traces.append(jax.tree.map(np.asarray, traces))
            done += count
            frames.append(snapshot()); frame_times.append(done*coarse_step)
            if len(frames) % 20 == 0 or done == steps:
                print('step', done, '/', steps, round(time.perf_counter()-stage_started, 1), 's', flush=True)
                report['stages']['lockstep_seconds'] = time.perf_counter()-stage_started
                report['progress_steps'] = done
                save()
        report['stages']['lockstep_seconds'] = time.perf_counter()-stage_started
        merged = jax.tree.map(lambda *xs: np.concatenate(xs), *site_traces)
        site_time = (np.arange(steps)+1)*coarse_step
        end_values = {str(dt): {name: np.asarray(u.get_mantissa(v), dtype=np.float64)[0] for name, v in values(dt).items()} for dt in dts}
    np.savez_compressed(args.output/'sites.npz', time_ms=site_time, site_cv=site_cv, site_point=site_point,
        **{f'{dt}/{name}': merged[dt][name] for dt in merged for name in merged[dt]})
    stacked = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    np.savez_compressed(args.output/'frames.npz', time_ms=np.asarray(frame_times), **stacked)
    report['ranges'] = {key.split('/', 1)[1]: float(value.max()-value.min()) for key, value in stacked.items() if key.startswith('ref/')}
    stacked = {key: value for key, value in stacked.items() if not key.startswith('ref/')}
    np.savez_compressed(args.output/'end_values.npz', **{f'{dt}/{name}': value for dt, row in end_values.items() for name, value in row.items()})
    report['files'] = {name: dict(sha256=hashlib.sha256((args.output/name).read_bytes()).hexdigest(), bytes=(args.output/name).stat().st_size)
                       for name in ('sites.npz', 'frames.npz', 'end_values.npz')}
    soma_rows = [i for i, key in enumerate(site_index) if key[1] == 'soma']
    report['spikes'] = {str(dt): {cells[j]: spike_times(merged[str(dt)]['V'][:, row], coarse_step).tolist()
                                  for j, row in enumerate(soma_rows)} for dt in dts}
    report['index'] = index_table(report, stacked, np.asarray(frame_times), end_values, cells, sites, point_of_cv)
    report['status'] = 'pass'
    save()


def _cell_of_positions(axis, forest_offsets, size, sites, cells, point_of_cv):
    """Cell index per position of a variable axis (``-1`` where unknown)."""
    if axis == 'cv':
        offsets = forest_offsets['cv']
    elif axis == 'point':
        offsets = forest_offsets['point']
    else:
        return np.full(size, -1)
    owner = np.full(size, -1)
    for index in range(len(cells)):
        owner[offsets[index]:offsets[index+1]] = index
    return owner


def index_table(report, stacked, frame_times, end_values, cells, sites, point_of_cv):
    """Max abs / relative / end-of-window differences per variable and pair, split around spikes.

    Parameters
    ----------
    report : dict
        Run report with ``spikes`` (per dt per cell, ms), ``variables``, ``forest_offsets``.
    stacked : dict
        ``pair/variable -> (frames, positions)`` difference frames.
    frame_times : array
        Frame times in ms.
    end_values : dict
        ``dt -> variable -> (positions,)`` final values.
    cells, sites, point_of_cv
        Cell order, site table and CV-to-point map.

    Returns
    -------
    dict
        ``pair -> variable -> {near_spike, between_spikes, end, range}`` in the
        variable's own units.
    """
    offsets = report['forest_offsets']
    reference = end_values[str(REFERENCE_DT)]
    table = {}
    for key, frames in stacked.items():
        pair, name = key.split('/', 1)
        axis = report['variables'][name]['axis']
        owner = _cell_of_positions(axis, offsets, frames.shape[1], sites, cells, point_of_cv)
        near = np.zeros(frames.shape, dtype=bool)
        spikes = report['spikes'][str(REFERENCE_DT)]
        for index, identity in enumerate(cells):
            times = np.asarray(spikes.get(identity, []))
            if not len(times):
                continue
            close = np.any(np.abs(frame_times[:, None]-times[None, :]) <= 1., axis=1)
            near[close[:, None] & (owner[None, :] == index)] = True
        if axis == 'contact':
            any_close = np.any([np.any(np.abs(frame_times[:, None]-np.asarray(t)[None, :]) <= 1., axis=1)
                                for t in spikes.values() if len(t)], axis=0) if any(len(t) for t in spikes.values()) else np.zeros(len(frame_times), bool)
            near[:] = any_close[:, None]
        magnitude = np.abs(frames)
        span = float(report['ranges'][name]) if name in report.get('ranges', {}) else None
        def stat(mask):
            if not mask.any():
                return None
            value = float(magnitude[mask].max())
            return dict(max_abs=value, max_rel_to_range=(value/span if span else None))
        table.setdefault(pair, {})[name] = dict(unit=report['variables'][name]['unit'],
            near_spike=stat(near), between_spikes=stat(~near),
            end=dict(max_abs=float(magnitude[-1].max()), reference_end_min=float(reference[name].min()),
                     reference_end_max=float(reference[name].max())))
    return table


def _plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((args.output/'report.json').read_text())
    sites = np.load(args.output/'sites.npz')
    order = [tuple(k) for k in report['site_order']]
    time_ms = sites['time_ms']
    dts = [REFERENCE_DT]+list(report['settings']['ladder_dt_ms'])
    names = [n for n in report['variables'] if report['variables'][n]['axis'] != 'contact']
    contact_names = [n for n in report['variables'] if report['variables'][n]['axis'] == 'contact']
    args.plots.mkdir(parents=True, exist_ok=True)
    for identity in args.cells or report['cells']:
        for pair in [(dt, REFERENCE_DT) for dt in report['settings']['ladder_dt_ms']]+[(.005, .0025)]:
            fig, axes = plt.subplots(len(names), 3, figsize=(15, 2.2*len(names)), sharex=True, squeeze=False)
            for row, name in enumerate(names):
                for col, site in enumerate(('soma', 'ais', 'distal')):
                    index = order.index((identity, site))
                    ax = axes[row, col]
                    a = sites[f'{pair[1]}/{name}'][:, index]
                    b = sites[f'{pair[0]}/{name}'][:, index]
                    ax.plot(time_ms, a, lw=.8, label=f'dt {pair[1]}')
                    ax.plot(time_ms, b, lw=.8, ls='--', label=f'dt {pair[0]}')
                    twin = ax.twinx()
                    twin.plot(time_ms, b-a, lw=.6, color='crimson', alpha=.7)
                    twin.set_ylabel('diff', color='crimson', fontsize=7)
                    twin.tick_params(labelsize=6)
                    if row == 0:
                        ax.set_title(f'{identity} {site} (cv {report["sites"][identity][site]}, {report["sites"][identity]["distance_um"][site]:.0f} um)', fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f'{name} [{report["variables"][name]["unit"]}]', fontsize=7)
                    ax.tick_params(labelsize=6)
            axes[0, 0].legend(fontsize=7)
            axes[-1, 0].set_xlabel('ms')
            fig.suptitle(f'{identity}: dt {pair[0]} against dt {pair[1]} ms, overlay and raw difference (own units)', fontsize=10)
            fig.tight_layout()
            fig.savefig(args.plots/f'{identity}-dt{pair[0]}-vs-{pair[1]}.png', dpi=110)
            plt.close(fig)
    if contact_names:
        fig, axes = plt.subplots(len(contact_names), 1, figsize=(12, 2.5*len(contact_names)), squeeze=False)
        for row, name in enumerate(contact_names):
            ax = axes[row, 0]
            for dt in dts:
                ax.plot(time_ms, sites[f'{dt}/{name}'], lw=.7, label=f'dt {dt}')
            ax.set_ylabel(f'{name} [{report["variables"][name]["unit"]}]', fontsize=7)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plots/'contacts.png', dpi=110)
        plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--manifest', type=Path, default=Path('docs/evidence/h01-keep-drop/arc-manifest/manifest.json'))
    run.add_argument('--arc-root', type=Path, default=Path('var/arc-agi-1'))
    run.add_argument('--currents', type=Path, default=Path('docs/evidence/h01-keep-drop/kept-currents.json'))
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--ladder', type=float, nargs='+', default=list(LADDER))
    run.add_argument('--window-ms', type=float, default=40.)
    run.add_argument('--frame-ms', type=float, default=.5)
    run.add_argument('--pulse-delay-ms', type=float, default=2.)
    run.add_argument('--contacts', type=int, default=3)
    run.add_argument('--contact-weight-us', type=float, default=.05)
    plot = sub.add_parser('plot')
    plot.add_argument('--output', type=Path, required=True)
    plot.add_argument('--plots', type=Path, required=True)
    plot.add_argument('--cells', nargs='*', default=None)
    return parser


def main(argv=None):
    """Dispatch ``run`` or ``plot``.

    Parameters
    ----------
    argv : sequence of str, optional
        Command line; ``None`` reads ``sys.argv``.
    """
    args = _parser().parse_args(argv)
    if args.command == 'run':
        _run(args)
    else:
        _plot(args)


if __name__ == '__main__':
    main()
