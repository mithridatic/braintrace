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

``run --mode parity`` runs the per-cell model (``fused=False``) and the fused model
at the pinned timestep side by side instead of the timestep ladder, comparing the
same variables (per-cell states concatenated in forest order).
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


def _cell_variables(cell):
    """Named floating-point states of one BrainCell cell (last axis n_cv or n_point), plus synapse ``g``."""
    import brainstate
    import brainunit as u
    n_cv, n_point = cell.n_cv, cell.runtime.n_point
    names = {}
    for layout in cell.runtime.layouts:
        declaration = cell.runtime.layout_mechanisms[layout.id]
        names[f'layout_{layout.id}'] = getattr(declaration, 'instance_name', f'layout_{layout.id}')
    found = {}
    for path, state in brainstate.graph.states(cell).items():
        value = u.get_mantissa(state.value)
        if not hasattr(value, 'shape') or not np.issubdtype(np.asarray(value).dtype, np.floating):
            continue
        parts = [names.get(str(p), str(p)) for p in path if p not in ('ion_channels', 'channels', '_channel')]
        name = '.'.join(parts)
        if name in ('spike', '_current_time_state') or path[0] == '_source_cells':
            continue
        last = value.shape[-1] if value.ndim else None
        if last in (n_cv, n_point) and value.ndim == 2:
            found[name] = dict(state=state, axis='cv' if last == n_cv else 'point', unit=str(u.get_unit(state.value)))
        elif name.endswith('.g') and value.ndim == 2 and last == 1:
            found[name] = dict(state=state, axis='contact', unit=str(u.get_unit(state.value)))
    return found


def _variables(model):
    """Named floating-point forest states with last axis n_cv or n_point, plus synapse conductances."""
    return _cell_variables(model.forest)


class _PerCellView:
    """Read the per-cell model's states as forest-ordered arrays, variable by variable.

    Parameters
    ----------
    model : H01ArcModel
        Per-cell model (one population per cell, ``fused=False``).
    template : dict
        ``_variables`` of the fused model built from the same topology.
    offsets : dict
        Forest ``cv`` and ``point`` offsets.

    Notes
    -----
    A cell without a mechanism contributes zeros and a ``present`` mask of
    zeros for that variable; the fused forest holds those points at the
    canonical initial state with zero density, so they are excluded from the
    comparison.
    """

    def __init__(self, model, template, offsets):
        self.model, self.offsets = model, offsets
        self.cells = list(model.stepper.cells)
        self.per_cell = [_cell_variables(cell) for cell in self.cells]
        self.template = template
        self.present = {}
        for name, spec in template.items():
            if spec['axis'] == 'contact':
                continue
            key = 'cv' if spec['axis'] == 'cv' else 'point'
            mask = np.zeros(offsets[key][-1], dtype=bool)
            for index, table in enumerate(self.per_cell):
                if name in table:
                    mask[offsets[key][index]:offsets[key][index+1]] = True
            self.present[name] = mask

    def values(self):
        import jax.numpy as jnp
        import brainunit as u
        out = {}
        for name, spec in self.template.items():
            if spec['axis'] == 'contact':
                for table in self.per_cell:
                    if name in table:
                        out[name] = u.get_mantissa(table[name]['state'].value)
                        break
                continue
            pieces = []
            for index, (cell, table) in enumerate(zip(self.cells, self.per_cell)):
                size = cell.n_cv if spec['axis'] == 'cv' else cell.runtime.n_point
                pieces.append(u.get_mantissa(table[name]['state'].value)[0] if name in table else jnp.zeros(size))
            out[name] = jnp.concatenate(pieces)[None, :]
        out['axial_rate'] = jnp.concatenate([_axial_rate(cell)[0] for cell in self.cells])[None, :]
        out['syn_current'] = _synaptic_current(self.model)
        return out


def _axial_rate(cell):
    """Cable term of dV/dt per CV (mV/ms) from the DHS coefficients (midpoint rows only)."""
    import jax.numpy as jnp
    import brainunit as u
    forest = cell
    source = forest.runtime.dhs_static_source_np
    point_v = u.get_mantissa(forest._cv_to_point_unchecked(forest.V.value).in_unit(u.mV)).reshape(-1)
    v = point_v[jnp.asarray(source.row_to_point_id_np)]   # DHS rows carry the boundary (algebraic) points too
    child, parent = source.edges_np[:, 0], source.edges_np[:, 1]
    rate = -jnp.asarray(source.diag_ms_inv_np)*v
    rate = rate.at[child].add(-jnp.asarray(source.lowers_ms_inv_np)[child]*v[parent])
    rate = rate.at[parent].add(-jnp.asarray(source.uppers_ms_inv_np)[child]*v[child])
    return rate[jnp.asarray(source.dynamic_rows_np)][None, :]


def _synaptic_current(model):
    """Per-contact synaptic current (nA) = g (uS) * (V_post_point - E)."""
    import jax.numpy as jnp
    import brainunit as u
    currents = []
    for block in model.stepper.setup.delivery_blocks:
        cell = model.forest if model.forest is not None else model.stepper.network.populations[block.source.post_population].cell
        layout = cell.runtime.layouts[int(block.source.layout_id)]
        node = cell.runtime.runtime_nodes[layout.id]
        point = int(layout.point_index[0])
        cv = int(np.flatnonzero(cell.runtime.node_tree.cv_to_mid_node_id == point)[0])
        g = u.get_mantissa(node.g.value.in_unit(u.uS)).reshape(-1)[0]
        e = float(u.get_mantissa(node.e.in_unit(u.mV)).reshape(-1)[0]) if hasattr(node, 'e') else 0.
        currents.append(g*(cell.V.value.to_decimal(u.mV).reshape(-1)[cv]-e))
    return jnp.stack(currents)[None, :] if currents else jnp.zeros((1, 0))


def _build(args, dt_ms, settings, topology, archive, cells, fused=True):
    from braintrace.datasets.h01_network_init import init_h01_network_states
    from examples.pp_prop.h01_arc_model import H01ArcModel
    from examples.pp_prop.h01_runtime import build_network
    network, records = build_network(topology, archive, solver=settings['solver'],
                                     max_cv_length_um=settings['max_cv_length_um'], fused=fused)
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
        if args.mode == 'ladder':
            labels = {str(dt): dt for dt in (REFERENCE_DT,)+tuple(args.ladder)}
            reference = str(REFERENCE_DT)
            pairs = [(str(dt), reference) for dt in args.ladder]
            if '0.005' in labels and '0.0025' in labels:
                pairs.append(('0.005', '0.0025'))
        else:
            labels = {'percell': REFERENCE_DT, 'fused': REFERENCE_DT}
            reference, pairs = 'percell', [('fused', 'percell')]
        report['settings'].update(mode=args.mode, labels=labels, reference=reference, pairs=pairs)
        models = {}
        for label, dt in labels.items():
            models[label], records = _build(args, dt, settings, topology, archive, cells, fused=(label != 'percell'))
            print('built', label, round(time.perf_counter()-started, 1), 's', flush=True)
        report['stages']['build_all'] = time.perf_counter()-started
        fused_label = next(label for label in labels if label != 'percell')
        forest0 = models[fused_label].forest
        report['compartments'] = int(forest0.n_cv)
        sites = _site_cvs(forest0, records, cells)
        report['sites'] = sites
        report['forest_offsets'] = dict(cv=forest0.forest_offsets.cv.tolist(), point=forest0.forest_offsets.point.tolist())
        template = _variables(models[fused_label])
        views = {}
        for label, model in models.items():
            views[label] = _PerCellView(model, template, report['forest_offsets']) if label == 'percell' else None
        names = list(template)
        report['variables'] = {name: dict(axis=template[name]['axis'], unit=template[name]['unit'],
                                          shape=list(np.shape(u.get_mantissa(template[name]['state'].value))))
                               for name in names}
        report['variables']['axial_rate'] = dict(axis='cv', unit='mV/ms', shape=[1, forest0.n_cv])
        report['variables']['syn_current'] = dict(axis='contact', unit='nA', shape=[1, args.contacts])
        present = {name: mask for view in views.values() if view is not None for name, mask in view.present.items()}
        report['percell_absent_positions'] = {name: int((~mask).sum()) for name, mask in present.items() if not mask.all()}
        point_of_cv = np.asarray(forest0.runtime.node_tree.cv_to_mid_node_id)
        site_index = {}
        for identity, row in sites.items():
            for site, cv in ((k, row[k]) for k in ('soma', 'ais', 'distal')):
                site_index[(identity, site)] = dict(cv=int(cv), point=int(point_of_cv[cv]))
        site_cv = np.asarray([v['cv'] for v in site_index.values()])
        site_point = np.asarray([v['point'] for v in site_index.values()])
        report['site_order'] = [list(k) for k in site_index]
        coarse_step = max(labels.values())   # the lockstep clock: every model lands here
        ratios = {label: int(round(coarse_step/dt)) for label, dt in labels.items()}
        steps = int(round(args.window_ms/coarse_step))
        frame_every = int(round(args.frame_ms/coarse_step))
        delay_substeps = {label: int(round(args.pulse_delay_ms/dt)) for label, dt in labels.items()}

        def values(label):
            model = models[label]
            if views[label] is not None:
                return views[label].values()
            out = {}
            for name, var in _variables(model).items():
                out[name] = u.get_mantissa(var['state'].value)
            out['axial_rate'] = _axial_rate(model.forest)
            out['syn_current'] = _synaptic_current(model)
            return out

        def advance(label):
            model = models[label]
            def substep(_):
                model.drive.value = jnp.where(model.stepper.tick.value >= delay_substeps[label], amplitude, 0.)
                model.stepper.update(sample_probes=False)
            brainstate.transform.for_loop(substep, jnp.arange(ratios[label]))

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
            for label in labels:
                advance(label)
            current = {label: values(label) for label in labels}
            return {label: gather(current[label]) for label in labels}

        def chunk(count):
            return brainstate.transform.for_loop(step, jnp.arange(count))

        chunks = {}
        def snapshot():
            current = {label: values(label) for label in labels}
            frame = {}
            for name in current[reference]:
                frame[f'ref/{name}'] = np.asarray(current[reference][name], dtype=np.float32)[0]
            for other, base in pairs:
                for name in current[other]:
                    delta = np.asarray(current[other][name]-current[base][name], dtype=np.float32)[0]
                    if name in present:
                        delta = np.where(present[name], delta, 0.)
                    frame[f'{other}-vs-{base}/{name}'] = delta
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
        end_values = {label: {name: np.asarray(u.get_mantissa(v), dtype=np.float64)[0] for name, v in values(label).items()}
                      for label in labels}
    np.savez_compressed(args.output/'sites.npz', time_ms=site_time, site_cv=site_cv, site_point=site_point,
        **{f'{label}/{name}': merged[label][name] for label in merged for name in merged[label]})
    stacked = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    np.savez_compressed(args.output/'frames.npz', time_ms=np.asarray(frame_times), **stacked)
    report['ranges'] = {key.split('/', 1)[1]: float(value.max()-value.min()) for key, value in stacked.items() if key.startswith('ref/')}
    stacked = {key: value for key, value in stacked.items() if not key.startswith('ref/')}
    np.savez_compressed(args.output/'end_values.npz', **{f'{label}/{name}': value for label, row in end_values.items() for name, value in row.items()})
    report['files'] = {name: dict(sha256=hashlib.sha256((args.output/name).read_bytes()).hexdigest(), bytes=(args.output/name).stat().st_size)
                       for name in ('sites.npz', 'frames.npz', 'end_values.npz')}
    soma_rows = [i for i, key in enumerate(site_index) if key[1] == 'soma']
    report['spikes'] = {label: {cells[j]: spike_times(merged[label]['V'][:, row], coarse_step).tolist()
                                for j, row in enumerate(soma_rows)} for label in labels}
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
    reference = end_values[report['settings']['reference']]
    table = {}
    for key, frames in stacked.items():
        pair, name = key.split('/', 1)
        axis = report['variables'][name]['axis']
        owner = _cell_of_positions(axis, offsets, frames.shape[1], sites, cells, point_of_cv)
        near = np.zeros(frames.shape, dtype=bool)
        spikes = report['spikes'][report['settings']['reference']]
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
            masked = np.where(mask, magnitude, -1.)
            frame, position = np.unravel_index(int(np.argmax(masked)), masked.shape)
            value = float(masked[frame, position])
            return dict(max_abs=value, max_rel_to_range=(value/span if span else None),
                        at_ms=float(frame_times[frame]), at_position=int(position))
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
    dts = list(report['settings']['labels'])
    names = [n for n in report['variables'] if report['variables'][n]['axis'] != 'contact']
    contact_names = [n for n in report['variables'] if report['variables'][n]['axis'] == 'contact']
    args.plots.mkdir(parents=True, exist_ok=True)
    for identity in args.cells or report['cells']:
        for pair in [tuple(p) for p in report['settings']['pairs']]:
            fig, axes = plt.subplots(len(names), 3, figsize=(15, 2.2*len(names)), sharex=True, squeeze=False)
            for row, name in enumerate(names):
                for col, site in enumerate(('soma', 'ais', 'distal')):
                    index = order.index((identity, site))
                    ax = axes[row, col]
                    a = sites[f'{pair[1]}/{name}'][:, index]
                    b = sites[f'{pair[0]}/{name}'][:, index]
                    ax.plot(time_ms, a, lw=.8, label=str(pair[1]))
                    ax.plot(time_ms, b, lw=.8, ls='--', label=str(pair[0]))
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
            fig.suptitle(f'{identity}: {pair[0]} against {pair[1]}, overlay and raw difference (own units)', fontsize=10)
            fig.tight_layout()
            fig.savefig(args.plots/f'{identity}-{pair[0]}-vs-{pair[1]}.png', dpi=110)
            plt.close(fig)
    if contact_names:
        fig, axes = plt.subplots(len(contact_names), 1, figsize=(12, 2.5*len(contact_names)), squeeze=False)
        for row, name in enumerate(contact_names):
            ax = axes[row, 0]
            for dt in dts:
                ax.plot(time_ms, sites[f'{dt}/{name}'], lw=.7, label=str(dt))
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
    run.add_argument('--mode', choices=('ladder', 'parity'), default='ladder',
                     help='ladder: fused model at each timestep; parity: per-cell against fused at the pinned timestep')
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
