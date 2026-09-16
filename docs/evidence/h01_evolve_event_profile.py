"""Measure where the Example 21 H01 backend spends its time per ARC event.

Drives the existing ``build_network`` / ``init_h01_network_states`` /
``H01ArcModel`` / sparse pp-prop objects directly, stage by stage, without the
evolution loop, so each stage of one candidate arm gets a wall-clock number,
a device-memory number and a host-RSS number. Three subcommands:

``corpus``
    CPU only. Counts the advancing ARC events the coordinator consumes before
    and inside a 128-update block (``initialize`` and ``train_parent`` score
    the whole training corpus; only ``run_candidate`` honours
    ``--screen-tasks``).
``arm``
    GPU. Builds the kept network under one numerical setting (dt, substeps,
    ``max_cv_length_um``, precision, checkpoint rematerialisation), times the
    stages, steps the same ARC events, records soma voltages, captures one
    XLA profile of a steady-state event and, optionally, compiles the sparse
    pp-prop learner and times training updates.
``report``
    Pure. Joins arm receipts and the corpus counts into the evidence JSON and
    Markdown (voltage differences against the pinned arm, seconds per block,
    blocks per hour).

Nothing here changes a pinned source or a committed manifest; numerical
overrides are applied to the objects built in this process only.
"""

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import re
import time

import numpy as np

EVENT_MS = .1
RTX4090_PEAK_TFLOPS = dict(fp32=82.6, fp64=1.29)   # NVIDIA specification sheet, boost clock
SOURCE_CLASSES = (
    # (class, needles matched against "path::function"); first match wins, so
    # function-specific rules precede file-level rules.
    ('channel', ('_fast_ind_exp_euler', 'ind_exp_euler', '_exp_euler', 'h01_wilbers', 'h01_l2_channels',
                 'h01_pv_rates', 'h01_pv_channels', 'h01_channels', 'h01_pv_calcium', 'braincell/channel',
                 'braincell/ion', 'braincell/mech', 'braincell/synapse')),
    ('delivery', ('braincell/network/delivery', 'brainevent', 'ring_buffer')),
    ('axial-solve', ('h01_dhs_scan', 'h01_dhs_gpu', 'h01_dhs_contraction', 'quad/_staggered',
                     '_voltage_step', 'braincell/quad')),
    ('calcium-implicit', ('h01_calcium_solver', 'h01_calcium_implicit')),
    ('encoder-readout', ('h01_arc_model', 'braintrace/_op', 'braintrace/_etrace')),
    ('spike-output', ('h01_spike_output',)),
    ('cell-runtime', ('braincell/_multi_compartment', 'braincell/_compute', 'braincell/_base',
                      'braincell/_misc', 'h01_network_step')),
    ('loop-control', ('brainstate/transform', 'brainstate/_state', 'lax/control_flow')),
)


def source_class(path):
    """Map a Python source path to the physical stage that created an XLA op.

    Parameters
    ----------
    path : str
        Absolute or relative source file path from the HLO stack-frame table.

    Returns
    -------
    str
        One of the ``SOURCE_CLASSES`` names, or ``'other'``.
    """
    normalised = str(path).replace('\\', '/')
    for name, needles in SOURCE_CLASSES:
        if any(needle in normalised for needle in needles):
            return name
    return 'other'


def parse_hlo_frames(text):
    """Parse the FileNames / FileLocations / StackFrames tables of an HLO dump.

    Parameters
    ----------
    text : str
        ``Compiled.as_text()`` output.

    Returns
    -------
    dict
        ``frame_id -> ("path::function", parent frame id)`` for every stack
        frame. The dump prints ``parent_frame_id`` one higher than the 1-based
        parent it stores (a root prints its own id), so the stored value minus
        one is the parent and zero terminates the chain.
    """
    tables = {}
    current = None
    for line in text.splitlines():
        if line in ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'):
            current = tables.setdefault(line, {})
            continue
        if current is None or not line.strip():
            current = None if not line.strip() else current
            continue
        match = re.match(r'^(\d+) (.*)$', line)
        if match is None:
            current = None
            continue
        current[int(match.group(1))] = match.group(2)
    files = {key: value.strip('"') for key, value in tables.get('FileNames', {}).items()}
    functions = {key: value.strip('"') for key, value in tables.get('FunctionNames', {}).items()}
    locations = {}
    for key, value in tables.get('FileLocations', {}).items():
        found = re.search(r'file_name_id=(\d+)', value)
        function = re.search(r'function_name_id=(\d+)', value)
        locations[key] = ((files.get(int(found.group(1)), '') if found else '')+'::'
                          +(functions.get(int(function.group(1)), '') if function else ''))
    frames = {}
    for key, value in tables.get('StackFrames', {}).items():
        location = re.search(r'file_location_id=(\d+)', value)
        parent = re.search(r'parent_frame_id=(\d+)', value)
        frames[key] = (locations.get(int(location.group(1)), '') if location else '',
                       (int(parent.group(1))-1) if parent else 0)
    return frames


def frame_class(frames, frame_id, depth=64):
    """Return the first classified source along a stack-frame chain.

    Parameters
    ----------
    frames : dict
        Output of :func:`parse_hlo_frames`.
    frame_id : int
        Leaf stack frame identity from an instruction's metadata.
    depth : int, optional
        Bound on the parent walk.

    Returns
    -------
    tuple
        ``(class name, leaf file path)``.
    """
    leaf = frames.get(frame_id, ('', 0))[0]
    seen = set()
    while frame_id > 0 and frame_id in frames and frame_id not in seen and depth > 0:
        seen.add(frame_id)
        path, parent = frames[frame_id]
        found = source_class(path)
        if found != 'other':
            return found, leaf
        frame_id, depth = parent, depth-1
    return 'other', leaf


def parse_hlo_kernels(text):
    """Classify every fusion / kernel-emitting instruction of an HLO dump by source.

    Parameters
    ----------
    text : str
        ``Compiled.as_text()`` output.

    Returns
    -------
    dict
        ``instruction name -> (class, leaf location)``. Fusions take the
        majority class of their fused computation, command-buffer calls
        (CUDA-graph replays of many kernels) are their own class, and other
        instructions take their own frame's class.
    """
    frames = parse_hlo_frames(text)
    computations = {}
    current = None
    for line in text.splitlines():
        header = re.match(r'^(?:ENTRY )?%([^\s]+) \(', line)
        if header:
            current = computations.setdefault(header.group(1), [])
            continue
        if current is None or not line.startswith('  '):
            continue
        instruction = re.match(r'^\s+(?:ROOT )?%([^\s]+) = \S+ ([a-z\-]+)\(', line)
        if instruction is None:
            continue
        frame = re.search(r'stack_frame_id=(\d+)', line)
        calls = re.search(r'(?:calls|to_apply)=%([^\s,]+)', line)
        current.append((instruction.group(1), instruction.group(2),
                        int(frame.group(1)) if frame else 0, calls.group(1) if calls else None))
    result = {}
    for instructions in computations.values():
        for name, op, frame, calls in instructions:
            if op in ('parameter', 'constant', 'get-tuple-element', 'tuple', 'bitcast'):
                continue
            if calls is not None and calls.startswith('command_buffer'):
                # A CUDA-graph replay of many kernels; the trace times it as one event.
                result[name] = ('command-buffer', calls)
                continue
            if calls is not None and calls in computations:
                votes = Counter()
                leaf = ''
                for _, inner_op, inner_frame, _ in computations[calls]:
                    if inner_op in ('parameter', 'constant', 'bitcast', 'broadcast', 'iota'):
                        continue
                    found, inner_leaf = frame_class(frames, inner_frame)
                    votes[found] += 1
                    leaf = leaf or inner_leaf
                if votes:
                    result[name] = (votes.most_common(1)[0][0], leaf)
                    continue
            result[name] = frame_class(frames, frame)
    return result


def kernel_class(name, stats, hlo_classes):
    """Classify one profiled GPU event.

    Parameters
    ----------
    name : str
        Trace event name (kernel symbol or ``MemcpyD2D`` etc.).
    stats : dict
        Trace event stats (``hlo_op``, ``kernel_details`` ...).
    hlo_classes : dict
        Output of :func:`parse_hlo_kernels`.

    Returns
    -------
    str
        Class name; memcpys are ``'memcpy'`` and unknown kernels ``'other'``.
    """
    if name.startswith('Memcpy') or name.startswith('Memset'):
        return 'memcpy'
    op = stats.get('hlo_op') or name
    found = hlo_classes.get(op)
    if found is None:
        found = hlo_classes.get(re.sub(r'\.\d+$', '', op))
    if found is not None:
        return found[0]
    lowered = (op+' '+str(stats.get('name', ''))).lower()
    if 'scatter' in lowered:
        return 'axial-solve'
    if 'while' in lowered or 'cond' in lowered or 'select' in lowered:
        return 'loop-control'
    return 'other'


def summarize_kernels(events, hlo_classes, executions=1, substeps=160):
    """Aggregate profiled GPU events by class and count launches per substep.

    Parameters
    ----------
    events : sequence of tuple
        ``(name, start_ns, duration_ns, stats)`` for every GPU stream event.
    hlo_classes : dict
        Output of :func:`parse_hlo_kernels`.
    executions : int, optional
        Number of ARC events executed inside the trace window.
    substeps : int, optional
        Cable substeps per ARC event.

    Returns
    -------
    dict
        Per-class seconds and share, busy seconds, launches per event and per
        substep, and the time span of the traced kernels.
    """
    seconds = defaultdict(float)
    launches = Counter()
    ops = Counter()
    start, end = None, None
    for name, begin, duration, stats in events:
        found = kernel_class(name, stats, hlo_classes)
        seconds[found] += duration/1e9
        launches[found] += 1
        ops[(found, stats.get('hlo_op') or name)] += duration/1e9
        start = begin if start is None else min(start, begin)
        end = begin+duration if end is None else max(end, begin+duration)
    busy = sum(seconds.values())
    span = 0. if start is None else (end-start)/1e9
    total_launches = sum(launches.values())
    classes = {name: dict(seconds_per_event=seconds[name]/executions,
                          share_of_busy=(seconds[name]/busy if busy else 0.),
                          launches_per_event=launches[name]/executions)
               for name in sorted(seconds, key=seconds.get, reverse=True)}
    heaviest = [dict(cls=key[0], op=key[1], seconds_per_event=value/executions)
                for key, value in ops.most_common(12)]
    return dict(classes=classes, busy_seconds_per_event=busy/executions,
                span_seconds_per_event=span/executions,
                idle_fraction_of_span=(1.-busy/span) if span else None,
                launches_per_event=total_launches/executions,
                launches_per_substep=total_launches/executions/substeps,
                heaviest_ops=heaviest)


def block_arithmetic(seconds_per_event, events_per_update, updates=128, fixed_seconds=0.,
                     cap_seconds=3600., initial_events=0, initial_seconds_per_event=None):
    """Turn per-event seconds into per-block seconds and blocks per cap.

    Parameters
    ----------
    seconds_per_event : float
        Steady-state seconds per advancing ARC event under the learner.
    events_per_update : float
        Mean advancing events in one scheduled episode.
    updates : int, optional
        Updates per training block (128 in Example 21).
    fixed_seconds : float, optional
        Construction, initialisation and compile seconds paid before stepping.
    cap_seconds : float, optional
        Wall cap of the run.
    initial_events : int, optional
        Forward-only events scored before the first block (``initialize``).
    initial_seconds_per_event : float, optional
        Forward-only seconds per event for that scoring pass.

    Returns
    -------
    dict
        Seconds per update and per block, the pre-block scoring seconds and
        how many complete blocks fit under the cap after the fixed costs.
    """
    per_update = seconds_per_event*events_per_update
    per_block = per_update*updates
    initial = initial_events*(initial_seconds_per_event if initial_seconds_per_event is not None
                              else seconds_per_event)
    remaining = cap_seconds-fixed_seconds-initial
    return dict(seconds_per_update=per_update, seconds_per_block=per_block,
                initial_scoring_seconds=initial, remaining_after_fixed_and_initial=remaining,
                blocks_under_cap=max(0., remaining)/per_block if per_block else None,
                initial_scoring_fits_under_cap=bool(fixed_seconds+initial <= cap_seconds))


def voltage_difference(reference, other):
    """Max absolute soma voltage difference over a shared event window.

    Parameters
    ----------
    reference, other : array-like
        ``(events, cells)`` soma voltages in mV.

    Returns
    -------
    dict
        Overall max, per-cell max, and the window compared.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(other, dtype=float)
    n = min(len(a), len(b))
    if n == 0 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Voltage windows must share a cell axis and overlap')
    delta = np.abs(a[:n]-b[:n])
    return dict(events=int(n), max_abs_mv=float(delta.max()),
                per_cell_max_abs_mv=delta.max(axis=0).tolist(),
                final_event_abs_mv=delta[-1].tolist())


def corpus_denominators(queries, schedule_entries, screen_ids):
    """Count the advancing ARC events the coordinator consumes per stage.

    Parameters
    ----------
    queries : sequence
        Encoded training queries with ``task_id`` and boolean ``advances``.
    schedule_entries : sequence
        ``build_update_schedule`` entries with ``task_id`` and ``query_index``.
    screen_ids : sequence of str
        Task identities of the ``--screen-tasks`` subset.

    Returns
    -------
    dict
        Corpus totals, block totals and screen totals in advancing events.
    """
    lookup = {(q.task_id, q.query_index): int(np.asarray(q.advances).sum()) for q in queries}
    corpus = np.array(list(lookup.values()))
    block = np.array([lookup[(e.task_id, e.query_index)] for e in schedule_entries])
    screen = [count for (task, _), count in lookup.items() if task in set(screen_ids)]
    return dict(training_queries=int(len(corpus)), training_tasks=len({q.task_id for q in queries}),
        corpus_advancing_events=int(corpus.sum()), corpus_padding_events=int(len(corpus)*705-corpus.sum()),
        mean_advancing_per_query=float(corpus.mean()),
        block_updates=int(len(block)), block_advancing_events=int(block.sum()),
        block_mean_advancing=float(block.mean()) if len(block) else None,
        screen_task_ids=list(screen_ids), screen_queries=len(screen), screen_advancing_events=int(sum(screen)),
        stacked_events_bytes_float64=int(len(corpus)*705*441*8))


def _rss_mb():
    import resource   # POSIX only; the arm subcommand runs on the Linux box
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.


def _device_memory():
    import jax
    stats = jax.devices()[0].memory_stats() or {}
    return dict(bytes_in_use=stats.get('bytes_in_use'), peak_bytes_in_use=stats.get('peak_bytes_in_use'))


def _corpus_command(args):
    from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter
    from examples.pp_prop.example21_evolve import PipelineConfig, build_update_schedule, screen_task_ids
    started = time.perf_counter()
    adapter = Example21ArcAdapter(args.arc_root)
    manifest = adapter.training_manifest()
    queries = adapter._encoded_queries('training')
    encode_seconds = time.perf_counter()-started
    schedule = build_update_schedule(manifest, 0, 128)
    config = PipelineConfig(screen_tasks=args.screen_tasks)
    result = corpus_denominators(queries, schedule.entries, screen_task_ids(manifest, config))
    result.update(encode_seconds=encode_seconds, screen_tasks_flag=args.screen_tasks,
                  block_unique_tasks=len({e.task_id for e in schedule.entries}),
                  schedule_first_entry=dict(task_id=schedule.entries[0].task_id,
                                            query_index=schedule.entries[0].query_index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


def _single_cell_topology(topology, index):
    from examples.pp_prop.h01_topology import H01Topology
    doc = topology.to_dict()
    cell = doc['active_cells'][index]
    doc['active_cells'] = [cell]
    doc['instances'] = {cell: doc['instances'][cell]}
    doc['contacts'], doc['active_contacts'] = {}, []
    return H01Topology.from_dict(doc)


def _profile_calls(call, directory, hlo_text, substeps, executions):
    """Trace ``executions`` calls and classify every GPU stream event.

    ``substeps`` is 160 (or 20) when ``call`` is a whole ARC event and 1 when
    it is one cable substep; the summary is always per executed call.
    """
    import jax
    from jax.profiler import ProfileData
    directory.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(directory))
    for _ in range(executions):
        jax.block_until_ready(call())
    jax.profiler.stop_trace()
    path = sorted(glob.glob(str(directory/'**'/'*.xplane.pb'), recursive=True))[-1]
    data = ProfileData.from_file(path)
    events, lines = [], []
    for plane in data.planes:
        if 'GPU' not in plane.name:
            continue
        for line in plane.lines:
            lines.append(line.name)
            if not line.name.startswith('Stream'):
                continue
            for item in line.events:
                events.append((item.name, item.start_ns, item.duration_ns, dict(item.stats)))
    summary = summarize_kernels(events, parse_hlo_kernels(hlo_text), executions=executions, substeps=substeps)
    summary['gpu_lines'] = lines
    summary['xplane'] = path
    return summary


def _arm_command(args):
    import brainstate
    import jax
    import jax.numpy as jnp
    from braintrace.datasets.h01 import H01Archive
    from braintrace.datasets.h01_network_init import init_h01_network_states
    from examples.pp_prop.h01_arc_adapter import H01ArcAdapter
    from examples.pp_prop.h01_arc_model import H01ArcModel
    from examples.pp_prop.h01_runtime import build_network
    from examples.pp_prop.example21_evolve import build_update_schedule

    args.output.mkdir(parents=True, exist_ok=True)
    substeps = int(round(EVENT_MS/args.dt_ms))
    report = dict(status='running', arm=args.name, settings=dict(dt_ms=args.dt_ms, substeps=substeps,
        max_cv_length_um=args.max_cv_length_um, precision=args.precision,
        checkpoint_substeps=not args.no_checkpoint, cell_index=args.cell_index), stages={}, memory={})

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def stage(name, call):
        report['active_stage'] = name
        save()
        started = time.perf_counter()
        result = call()
        jax.block_until_ready(result)
        report['stages'][name] = time.perf_counter()-started
        report['memory'][name] = dict(host_peak_rss_mb=_rss_mb(), **_device_memory())
        save()
        print(name, round(report['stages'][name], 3), 's', flush=True)
        return result

    try:
        with brainstate.environ.context(precision=args.precision):
            adapter = stage('manifest_load', lambda: H01ArcAdapter(args.arc_root, args.manifest))
            settings = adapter.document['settings']
            topology = adapter.initial_topology
            if args.cell_index is not None:
                topology = _single_cell_topology(topology, args.cell_index)
            if args.mutate:
                cells = topology.to_dict()['active_cells']
                topology = (topology.add_contact(cells[0], cells[1], stage='profile') if args.mutate == 'add-contact'
                            else topology.clone(cells[0], stage='profile'))
                report['mutation'] = dict(kind=args.mutate, cells=len(topology.to_dict()['active_cells']),
                                          contacts=len(topology.to_dict()['active_contacts']))
            archive = stage('archive_open', adapter._archive)
            network, records = stage('build_network', lambda: build_network(topology, archive,
                solver=settings['solver'], max_cv_length_um=args.max_cv_length_um))
            report['cells'] = len(topology.to_dict()['active_cells'])
            report['compartments'] = int(sum(r['n_compartments'] for r in records.values()))
            del records
            stage('init_state', lambda: init_h01_network_states(network))
            model = stage('model_setup', lambda: H01ArcModel(network, topology.to_dict()['active_cells'],
                seed=settings['seed'], dt_ms=args.dt_ms, checkpoint_substeps=not args.no_checkpoint))
            manifest = adapter.training_manifest()
            entry = build_update_schedule(manifest, 0, 128).entries[0]
            query = next(q for q in adapter._encoded_queries('training')
                         if (q.task_id, q.query_index) == (entry.task_id, entry.query_index))
            advancing = np.asarray(query.events[query.advances], dtype=np.float64)
            report['event_source'] = dict(task_id=query.task_id, query_index=query.query_index,
                                          advancing_events=int(len(advancing)))
            stream = [jnp.asarray(advancing[i % len(advancing)]) for i in range(args.events)]
            step = brainstate.transform.jit(model.update)
            first = stage('first_update_compile_and_run', lambda: step(stream[0]))
            voltages = [np.asarray(first, dtype=float).tolist()]
            compiled = stage('compile_object', lambda: step.compile(stream[0]))
            cost = compiled.cost_analysis() or {}
            report['cost_analysis'] = {key: float(value) for key, value in cost.items()
                                       if isinstance(value, (int, float)) and '{' not in key}
            hlo_text = compiled.as_text()
            (args.output/'update.hlo.txt').write_text(hlo_text)
            per_event = []
            for index in range(1, args.events):
                started = time.perf_counter()
                voltage = jax.block_until_ready(step(stream[index]))
                per_event.append(time.perf_counter()-started)
                voltages.append(np.asarray(voltage, dtype=float).tolist())
                if index % 20 == 0 or index == args.events-1:
                    report['forward'] = dict(events=index, seconds_per_event_mean=float(np.mean(per_event)),
                        seconds_per_event_median=float(np.median(per_event)), seconds_per_event_min=float(np.min(per_event)),
                        seconds_per_event_max=float(np.max(per_event)), total_seconds=float(np.sum(per_event)))
                    report['memory']['steady_state'] = dict(host_peak_rss_mb=_rss_mb(), **_device_memory())
                    np.save(args.output/'voltages_mv.npy', np.asarray(voltages))
                    save()
            report['voltages_mv_first_20'] = voltages[:20]
            report['finite'] = bool(np.isfinite(np.asarray(voltages)).all())
            seconds = report['forward']['seconds_per_event_median']
            updates = report['compartments']*substeps
            flops = report['cost_analysis'].get('flops')
            report['throughput'] = dict(compartment_updates_per_second=updates/seconds,
                flops_per_event=flops, gflops_per_second=(flops/seconds/1e9) if flops else None,
                fraction_of_fp64_peak=(flops/seconds/1e12/RTX4090_PEAK_TFLOPS['fp64']) if flops else None,
                fraction_of_fp32_peak=(flops/seconds/1e12/RTX4090_PEAK_TFLOPS['fp32']) if flops else None)
            save()
            if args.profile:
                report['profile_event'] = stage('profile_trace_event', lambda: _profile_calls(
                    lambda: step(stream[0]), args.output/'trace-event', hlo_text, substeps, 1))
                save()
            if args.profile_substep:
                substep = brainstate.transform.jit(lambda: model.stepper.update(sample_probes=False))
                stage('substep_compile_and_run', substep)
                substep_hlo = substep.compile().as_text()
                (args.output/'substep.hlo.txt').write_text(substep_hlo)
                timings = []
                for _ in range(20):
                    started = time.perf_counter()
                    jax.block_until_ready(substep())
                    timings.append(time.perf_counter()-started)
                report['substep_alone'] = dict(seconds_median=float(np.median(timings)), seconds_min=float(np.min(timings)),
                    while_loops_in_event_hlo=hlo_text.count(' while('), while_loops_in_substep_hlo=substep_hlo.count(' while('))
                report['profile_substep'] = stage('profile_trace_substep', lambda: _profile_calls(
                    substep, args.output/'trace-substep', substep_hlo, 1, args.profile_substep))
                report['profile_substep']['seconds_per_event_from_substeps'] = (
                    report['profile_substep']['busy_seconds_per_event']*substeps)
                save()
            if args.recompile_check:
                _recompile_stages(args, report, stage, save, adapter, model, settings, query)
            if args.learner:
                _learner_stages(args, report, stage, save, adapter, topology, model, settings, query)
            if args.score_query:
                _score_stages(args, report, stage, adapter, model, query)
        report['status'] = 'pass' if report.get('finite', False) else 'nonfinite'
    except Exception as error:   # noqa: BLE001 - the receipt must name the failure
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        save()


def _learner_stages(args, report, stage, save, adapter, topology, model, settings, query):
    import brainstate
    import braintrace
    import jax
    import jax.numpy as jnp
    from examples.pp_prop.h01_arc_execution import update_episode
    from examples.pp_prop.h01_muon import model_optimizer
    from examples.pp_prop.h01_session import H01Session
    module = adapter._model()
    learner = braintrace.pp_prop.sparse(model, settings['decay'], max_bytes=settings['factor_limit_bytes'])
    try:
        stage('learner_compile_graph', lambda: learner.compile_graph(jnp.zeros(441)))
    except Exception as error:   # noqa: BLE001 - recorded, the arm continues forward-only
        report['learner_refused'] = repr(error)
        return
    layout = learner.graph.layout
    report['eligibility'] = dict(elements=int(layout.elements), bytes=int(layout.nbytes),
                                 colors=int(layout.color_count), state_blocks=len(layout.shapes))
    values = dict(input=model.input_weight.value, recurrent=model.recurrent_weight.value,
                  readout_weight=model.readout_weight.value, readout_bias=model.readout_bias.value)
    optimizer = stage('optimizer_setup', lambda: model_optimizer(model, values))
    trainer = module.PPPropEpisodeTrainer(learner, values, optimizer_adapter=optimizer)
    events = np.asarray(query.events[query.advances], dtype=np.float64)

    def loss(event):
        learner(event)
        return jnp.mean(jnp.square(model.readout()))

    train = brainstate.transform.jit(lambda ev: trainer.update_episode(ev, step_fn=loss))
    report['learner'] = {}
    for count in args.train_events:
        batch = jnp.asarray(events[:count])
        model.reset_episode(learner)
        cold = stage(f'learner_update_{count}_events_compile_and_run', lambda: train(batch))
        warm = []
        for _ in range(args.train_steps):
            model.reset_episode(learner)
            started = time.perf_counter()
            jax.block_until_ready(train(batch))
            warm.append(time.perf_counter()-started)
        report['learner'][str(count)] = dict(events=count, cold_seconds=report['stages'][f'learner_update_{count}_events_compile_and_run'],
            warm_seconds=warm, warm_median=float(np.median(warm)), seconds_per_event_naive=float(np.median(warm))/count,
            finite=bool(np.isfinite(np.asarray(cold)).all()), updates=int(trainer.updates))
        report['memory'][f'learner_{count}'] = dict(host_peak_rss_mb=_rss_mb(), **_device_memory())
        save()
    counts = sorted(int(k) for k in report['learner'])
    if len(counts) >= 2:
        lo, hi = counts[0], counts[-1]
        slope = (report['learner'][str(hi)]['warm_median']-report['learner'][str(lo)]['warm_median'])/(hi-lo)
        report['learner']['seconds_per_event_slope'] = float(slope)
        report['learner']['fixed_seconds_per_update_intercept'] = float(report['learner'][str(lo)]['warm_median']-slope*lo)
    if args.real_update:
        session = H01Session(topology, model, learner, trainer, dict(settings), tuple(adapter.document['assets']))
        payload = adapter._episode_payload(query)
        real = brainstate.transform.jit(lambda p: update_episode(session, p, module))
        model.reset_episode(learner)
        stacked = jax.tree.map(jnp.asarray, payload)
        stage('real_update_episode_compile_and_run', lambda: real(stacked))
        warm = []
        for _ in range(args.real_update):
            model.reset_episode(learner)
            started = time.perf_counter()
            jax.block_until_ready(real(stacked))
            warm.append(time.perf_counter()-started)
        report['real_update'] = dict(advancing_events=int(query.advances.sum()), padded_events=int(len(query.advances)),
            warm_seconds=warm, warm_median=float(np.median(warm)),
            seconds_per_advancing_event=float(np.median(warm))/int(query.advances.sum()))
        report['memory']['real_update'] = dict(host_peak_rss_mb=_rss_mb(), **_device_memory())
        save()


def _score_stages(args, report, stage, adapter, model, query):
    import brainstate
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from examples.pp_prop.h01_arc_execution import score_episode
    session = SimpleNamespace(model=model, learner=None)
    events = jnp.asarray(query.events, dtype=jnp.float64)
    advances = jnp.asarray(query.advances)
    score = brainstate.transform.jit(lambda e, a: score_episode(session, e, a))
    stage('score_query_compile_and_run', lambda: score(events, advances))
    warm = []
    for _ in range(args.score_query):
        started = time.perf_counter()
        jax.block_until_ready(score(events, advances))
        warm.append(time.perf_counter()-started)
    report['score_query'] = dict(advancing_events=int(query.advances.sum()), padded_events=int(len(query.advances)),
        warm_seconds=warm, warm_median=float(np.median(warm)),
        seconds_per_advancing_event=float(np.median(warm))/int(query.advances.sum()))


def _arm_row(report):
    stages = report.get('stages', {})
    forward = report.get('forward', {})
    learner = report.get('learner', {})
    memory = report.get('memory', {})
    peak_device = max((m.get('peak_bytes_in_use') or 0) for m in memory.values()) if memory else None
    peak_host = max((m.get('host_peak_rss_mb') or 0) for m in memory.values()) if memory else None
    first_learner = next((stages[k] for k in stages
                          if k.startswith('learner_update_') and k.endswith('_compile_and_run')), None)
    return dict(arm=report.get('arm'), settings=report.get('settings'), status=report.get('status'),
        cells=report.get('cells'), compartments=report.get('compartments'),
        stage_seconds=dict(manifest_and_archive=stages.get('manifest_load', 0.)+stages.get('archive_open', 0.),
            build_network=stages.get('build_network'), init_state=stages.get('init_state'),
            model_setup=stages.get('model_setup'),
            first_update_compile_and_run=stages.get('first_update_compile_and_run'),
            learner_compile_graph=stages.get('learner_compile_graph'),
            learner_first_update_compile_and_run=first_learner,
            real_update_compile_and_run=stages.get('real_update_episode_compile_and_run'),
            score_query_compile_and_run=stages.get('score_query_compile_and_run')),
        forward_seconds_per_event=forward.get('seconds_per_event_median'), forward_events=forward.get('events'),
        learner_seconds_per_event=learner.get('seconds_per_event_slope',
            next((v['seconds_per_event_naive'] for k, v in learner.items() if k.isdigit()), None)),
        learner_fixed_seconds_per_update=learner.get('fixed_seconds_per_update_intercept'),
        learner_refused=report.get('learner_refused'),
        real_update=report.get('real_update'), score_query=report.get('score_query'),
        throughput=report.get('throughput'), cost_analysis=report.get('cost_analysis'),
        eligibility=report.get('eligibility'), substep_alone=report.get('substep_alone'),
        profile_substep=report.get('profile_substep'),
        profile_event=report.get('profile_event') or report.get('profile'),
        recompile_check=report.get('recompile_check'), learner_compile_graph_is=report.get('learner_compile_graph_is'),
        peak_device_bytes=peak_device, peak_host_rss_mb=peak_host, memory=memory, finite=report.get('finite'))


FIXED_STAGES = ('manifest_and_archive', 'build_network', 'init_state', 'model_setup',
                'first_update_compile_and_run', 'learner_compile_graph', 'learner_first_update_compile_and_run')


def build_report(arms, corpus, reference='pinned', cap_seconds=3600.):
    """Join arm receipts and corpus counts into the evidence document.

    Parameters
    ----------
    arms : dict
        ``arm name -> (report dict, voltages array or None)``.
    corpus : dict
        Output of :func:`corpus_denominators` (plus its extras).
    reference : str, optional
        Arm whose voltages every other arm is compared against.
    cap_seconds : float, optional
        Wall cap of the evolution run being explained.

    Returns
    -------
    dict
        Rows per arm, voltage differences, and block arithmetic per arm.
    """
    rows = {name: _arm_row(report) for name, (report, _) in arms.items()}
    reference_cells = arms[reference][0].get('cells') if reference in arms else None
    base = arms.get(reference, (None, None))[1]
    voltages = {}
    for name, (report, volts) in arms.items():
        if base is None or volts is None or name == reference or report.get('cells') != reference_cells:
            continue
        voltages[name] = voltage_difference(base, volts)
    arithmetic = {}
    for name, row in rows.items():
        forward, learner = row['forward_seconds_per_event'], row['learner_seconds_per_event']
        if forward is None or row['cells'] != reference_cells:
            continue
        fixed = sum(v for k, v in row['stage_seconds'].items() if v is not None and k in FIXED_STAGES)
        real = row.get('real_update') or {}
        per_event = real.get('seconds_per_advancing_event', learner)
        arithmetic[name] = dict(fixed_seconds=fixed, forward_seconds_per_event=forward,
            learner_seconds_per_event=learner, learner_measured=learner is not None,
            per_event_basis=('real 705-event update_episode' if real else
                             'truncated synthetic update slope' if learner is not None else 'forward only'),
            screen_scoring_seconds=corpus['screen_advancing_events']*forward,
            corpus_scoring_seconds=corpus['corpus_advancing_events']*forward,
            corpus_scoring_fits_under_cap=bool(fixed+corpus['corpus_advancing_events']*forward <= cap_seconds))
        if per_event is None:
            continue
        arithmetic[name].update(
            with_initialize_scoring=block_arithmetic(per_event, corpus['block_mean_advancing'], fixed_seconds=fixed,
                cap_seconds=cap_seconds, initial_events=corpus['corpus_advancing_events'],
                initial_seconds_per_event=forward),
            block_only=block_arithmetic(per_event, corpus['block_mean_advancing'], fixed_seconds=fixed,
                                        cap_seconds=cap_seconds),
            block_seconds=per_event*corpus['block_advancing_events'],
            blocks_per_hour_steady=3600./(per_event*corpus['block_advancing_events']))
    return dict(schema='h01-evolve-event-profile-v1', reference_arm=reference, cap_seconds=cap_seconds,
                corpus=corpus, arms=rows, voltage_differences_vs_reference=voltages, arithmetic=arithmetic)


def _report_command(args):
    arms = {}
    for directory in args.arms:
        directory = Path(directory)
        report = json.loads((directory/'report.json').read_text())
        volts = (np.load(directory/'voltages_mv.npy') if (directory/'voltages_mv.npy').exists()
                 else np.asarray(report['voltages_mv_first_20']) if report.get('voltages_mv_first_20') else None)
        arms[report['arm']] = (report, volts)
    corpus = json.loads(Path(args.corpus).read_text())
    document = build_report(arms, corpus, reference=args.reference, cap_seconds=args.cap_seconds)
    document['single_cell_long_windows'] = {}
    for pair in args.long_pairs:
        left, right = pair.split(':')
        a, b = arms[left][1], arms[right][1]
        document['single_cell_long_windows'][pair] = dict(voltage_difference(a, b),
            cell_index=arms[left][0]['settings'].get('cell_index'),
            reference_dt_ms=arms[left][0]['settings']['dt_ms'], other_dt_ms=arms[right][0]['settings']['dt_ms'],
            window_ms=min(len(a), len(b))*EVENT_MS,
            reference_min_mv=float(np.min(a)), reference_max_mv=float(np.max(a)))
    args.output.write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps(document['arithmetic'], indent=1))


def _recompile_stages(args, report, stage, save, adapter, model, settings, query):
    """Call the same learner update repeatedly and count XLA compiles per call."""
    import logging
    import brainstate
    import braintrace
    import jax
    import jax.numpy as jnp
    from examples.pp_prop.h01_muon import model_optimizer
    module = adapter._model()
    learner = braintrace.pp_prop.sparse(model, settings['decay'], max_bytes=settings['factor_limit_bytes'])
    stage('learner_compile_graph', lambda: learner.compile_graph(jnp.zeros(441)))
    report['learner_compile_graph_is'] = ('braintrace ETP compiler: jaxpr trace of the model plus eligibility-graph '
        'analysis on the host (graph_executor.compile_graph); no XLA executable is built here')
    values = dict(input=model.input_weight.value, recurrent=model.recurrent_weight.value,
                  readout_weight=model.readout_weight.value, readout_bias=model.readout_bias.value)
    trainer = module.PPPropEpisodeTrainer(learner, values, optimizer_adapter=model_optimizer(model, values))
    events = np.asarray(query.events[query.advances], dtype=np.float64)

    def loss(event):
        learner(event)
        return jnp.mean(jnp.square(model.readout()))

    train = brainstate.transform.jit(lambda ev: trainer.update_episode(ev, step_fn=loss))
    records = []

    class Counter_(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Counter_()
    logging.getLogger().addHandler(handler)
    count = args.recompile_check
    batch = jnp.asarray(events[:count])
    calls = []
    with jax.log_compiles(True):
        for index in range(args.train_steps):
            before = len(records)
            model.reset_episode(learner)
            started = time.perf_counter()
            jax.block_until_ready(train(batch))
            elapsed = time.perf_counter()-started
            new = records[before:]
            compiles = [m for m in new if 'ompil' in m]
            calls.append(dict(call=index+1, seconds=elapsed, compile_messages=len(compiles),
                              messages=[m[:160] for m in compiles][:10]))
            report['recompile_check'] = dict(events=count, calls=calls)
            save()
            print('recompile call', index+1, round(elapsed, 2), 's', len(compiles), 'compile messages', flush=True)
    logging.getLogger().removeHandler(handler)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    corpus = sub.add_parser('corpus')
    corpus.add_argument('--arc-root', type=Path, default=Path('var/arc-agi-1'))
    corpus.add_argument('--screen-tasks', type=int, default=2)
    corpus.add_argument('--output', type=Path, required=True)
    arm = sub.add_parser('arm')
    arm.add_argument('--name', required=True)
    arm.add_argument('--manifest', type=Path, default=Path('docs/evidence/h01-keep-drop/arc-manifest/manifest.json'))
    arm.add_argument('--arc-root', type=Path, default=Path('var/arc-agi-1'))
    arm.add_argument('--output', type=Path, required=True)
    arm.add_argument('--dt-ms', type=float, default=.000625)
    arm.add_argument('--max-cv-length-um', type=float, default=10.)
    arm.add_argument('--precision', type=int, choices=(32, 64), default=64)
    arm.add_argument('--no-checkpoint', action='store_true')
    arm.add_argument('--events', type=int, default=21)
    arm.add_argument('--cell-index', type=int, default=None)
    arm.add_argument('--mutate', choices=('add-contact', 'clone'), default=None,
                     help='apply one grow mutation to the manifest topology before building (recompile cost)')
    arm.add_argument('--profile', action='store_true', help='trace one whole ARC event (CUPTI may drop events)')
    arm.add_argument('--profile-substep', type=int, default=0, help='trace this many single cable substeps')
    arm.add_argument('--learner', action='store_true')
    arm.add_argument('--train-events', type=int, nargs='*', default=(8, 2))
    arm.add_argument('--train-steps', type=int, default=5)
    arm.add_argument('--real-update', type=int, default=0, help='warm real 705-event update_episode repeats')
    arm.add_argument('--score-query', type=int, default=0, help='warm real score_episode repeats')
    arm.add_argument('--recompile-check', type=int, default=0,
                     help='call one learner update of this many events --train-steps times under jax.log_compiles')
    report = sub.add_parser('report')
    report.add_argument('--arms', type=Path, nargs='+', required=True, help='arm output directories')
    report.add_argument('--corpus', type=Path, required=True)
    report.add_argument('--reference', default='pinned')
    report.add_argument('--cap-seconds', type=float, default=3600.)
    report.add_argument('--long-pairs', nargs='*', default=(),
                        help='reference:other arm names for long single-cell windows')
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
    if args.command == 'corpus':
        _corpus_command(args)
    elif args.command == 'arm':
        _arm_command(args)
    elif args.command == 'report':
        _report_command(args)


if __name__ == '__main__':
    main()
