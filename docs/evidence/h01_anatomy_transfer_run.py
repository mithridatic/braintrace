"""Drive one production-imported H01 cell with its deployed donor's recorded step protocol.

The cell is built exactly as ``make_h01_network`` builds it (same importer, component
selection, electrical partition, donor profile, CV policy and solver); only the somatic
pulse timing and amplitude follow the donor recording's protocol instead of the population's
assumed 1 nA / 2-5 ms probe. The output is one bounded physiological reading on retained
human anatomy: -20 mV crossing count inside the pulse, first-spike latency, finiteness.

``--ramp-na MAX`` replaces the step by a linear ramp from 0 at the pulse onset to MAX at the
pulse end (a BrainCell ``FunctionClamp`` evaluated inside the compiled step), reading the
rheobase (current at the first -20 mV crossing at the output site) and, where the trace
stays above -20 mV until the ramp ends, the block current.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import braincell
import brainstate
import brainunit as u
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"docs"/"evidence"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braintrace.datasets.h01 import ARCHIVE_SHA256  # noqa: E402
from braintrace.datasets.h01_annotations import H01Annotations  # noqa: E402
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell  # noqa: E402
from braintrace.datasets.h01_network import _regions, _register_cell  # noqa: E402
from braintrace.datasets.h01_network_init import init_h01_network_states, process_rss_mb  # noqa: E402
from examples.pp_prop.h01_runtime import open_archive  # noqa: E402
from h01_pv_human_datums import spike_datums  # noqa: E402

BASIS = ("Source soma/axon/AIS samples extend halfway along adjacent edges; remaining cable gets "
         "dendrite properties. Myelin is not modeled. This is the existing inferred electrical "
         "partition, not measured channel placement.")
BLOCK_HOLD_MS = 20.


def crossings(time_ms, voltage_mv, pulse_ms, detection_mv=-20.):
    """Count -20 mV rise crossings inside the pulse; first latency after onset; finiteness."""
    time_ms, voltage_mv = np.asarray(time_ms, float), np.asarray(voltage_mv, float)
    finite = bool(np.isfinite(voltage_mv).all())
    mask = (time_ms >= pulse_ms[0]) & (time_ms < pulse_ms[1])
    events = spike_datums(time_ms[mask], np.nan_to_num(voltage_mv[mask], nan=-1e3), detection_mv) if mask.any() else []
    first = events[0]["rise_crossing_ms"]-pulse_ms[0] if events else None
    return dict(count=len(events), first_spike_ms=first, finite=finite,
                crossing_times_ms=[e["rise_crossing_ms"] for e in events])


def windows(time_ms, voltage_mv, pulse_ms, rest_ms=100., return_ms=200., detection_mv=-20.):
    """Keep-rule observations outside the pulse: pre/post-pulse crossings, rest and return levels.

    ``rest_mean_mv`` and ``rest_sd_mv`` summarise the ``rest_ms`` before onset; ``return_mv``
    averages the 10 ms ending ``return_ms`` after the pulse and is None when the trace ends
    before that.
    """
    time_ms, voltage_mv = np.asarray(time_ms, float), np.asarray(voltage_mv, float)
    clean = np.nan_to_num(voltage_mv, nan=-1e3)
    before, after = time_ms < pulse_ms[0], time_ms >= pulse_ms[1]
    pre = len(spike_datums(time_ms[before], clean[before], detection_mv)) if before.any() else 0
    post = len(spike_datums(time_ms[after], clean[after], detection_mv)) if after.any() else 0
    rest = (time_ms >= pulse_ms[0]-rest_ms) & before
    return_end = pulse_ms[1]+return_ms
    available = bool(len(time_ms) and time_ms[-1] >= return_end)
    back = (time_ms >= return_end-10.) & (time_ms <= return_end)
    return dict(pre_pulse_count=int(pre), rest_mean_mv=float(voltage_mv[rest].mean()) if rest.any() else None,
                rest_sd_mv=float(voltage_mv[rest].std()) if rest.any() else None,
                return_mv=float(voltage_mv[back].mean()) if available and back.any() else None,
                return_available=available, post_pulse_count=int(post))


def count_verdict(model, registered, repeat_counts):
    """Exact/within-one/rejected against a registered count; band verdict against repeats."""
    difference = abs(model-registered)
    exact = "held" if difference == 0 else ("missed_within_one" if difference <= 1 else "rejected")
    band = None
    if repeat_counts:
        band = "held" if min(repeat_counts) <= model <= max(repeat_counts) else "missed"
    return dict(exact=exact, repeat_band=band)


def ramp_current_na(time_ms, pulse_ms, max_na):
    """Injected ramp current: 0 before onset, linear to ``max_na`` at the pulse end, 0 after."""
    time_ms = np.asarray(time_ms, float)
    on, off = pulse_ms
    inside = (time_ms >= on) & (time_ms < off)
    return np.where(inside, max_na*(time_ms-on)/(off-on), 0.)


def ramp_clamp(pulse_on_ms, pulse_ms, max_na):
    """BrainCell point mechanism delivering the linear ramp inside the pulse window.

    Parameters
    ----------
    pulse_on_ms, pulse_ms : float
        Ramp onset and width; the current is 0 outside ``[onset, onset+width)``.
    max_na : float
        Current at the end of the window.

    Returns
    -------
    braincell.FunctionClamp
        Evaluated by the compiled runtime at every step; no per-step Python loop.
    """
    on, width, top = float(pulse_on_ms), float(pulse_ms), float(max_na)
    if not np.isfinite([on, width, top]).all() or width <= 0. or on < 0. or top <= 0.:
        raise ValueError("Ramp needs a finite non-negative onset, positive width and positive maximum.")

    def ramp(t):
        elapsed = u.Quantity(t).to_decimal(u.ms)-on
        inside = u.math.logical_and(elapsed >= 0., elapsed < width)
        return u.Quantity(u.math.where(inside, elapsed/width*top, 0.), u.nA)

    return braincell.FunctionClamp(ramp)


def rise_times(time_ms, voltage_mv, detection_mv=-20.):
    """Interpolated upward crossing times, complete events or not."""
    above = voltage_mv >= detection_mv
    rises = np.flatnonzero(~above[:-1] & above[1:])
    return [float(time_ms[i]+(detection_mv-voltage_mv[i])*(time_ms[i+1]-time_ms[i])/(voltage_mv[i+1]-voltage_mv[i]))
            for i in rises]


def ramp_readings(time_ms, voltage_mv, pulse_ms, max_na, detection_mv=-20., hold_ms=BLOCK_HOLD_MS):
    """Rheobase and block current of a ramp run from the output-site trace.

    ``rheobase_na`` is the ramp current at the first -20 mV rise crossing inside the window.
    ``block_na`` is the current at the last rise crossing when the trace then stays at or
    above -20 mV, with no further crossing, for at least ``hold_ms`` before the ramp ends;
    otherwise None (spiking continued to the end, or it stopped without depolarisation block).
    """
    time_ms, voltage_mv = np.asarray(time_ms, float), np.asarray(voltage_mv, float)
    clean = np.nan_to_num(voltage_mv, nan=-1e3)
    on, off = pulse_ms
    mask = (time_ms >= on) & (time_ms < off)
    rises = [t for t in (rise_times(time_ms[mask], clean[mask], detection_mv) if mask.any() else [])
             if on <= t < off]
    result = dict(ramp_max_na=float(max_na), ramp_spike_times_ms=rises, ramp_count=len(rises),
                  rheobase_na=None, block_na=None, finite=bool(np.isfinite(voltage_mv).all()),
                  block_rule=f"last crossing, then >= {detection_mv:g} mV with no crossing for >= {hold_ms:g} ms to the ramp end")
    if not rises:
        return result
    result["rheobase_na"] = float(ramp_current_na(rises[0], pulse_ms, max_na))
    last = rises[-1]
    tail = (time_ms > last) & (time_ms < off)
    if off-last >= hold_ms and tail.any() and bool((clean[tail] >= detection_mv).all()):
        result["block_na"] = float(ramp_current_na(last, pulse_ms, max_na))
    return result


def usable_features(time_ms, voltage_mv, pulse_ms, label):
    """Usable-tier per-cycle table and QC view of one trace, when its analysis stack imports."""
    try:
        from h01_usable_tier import cycle_table, qc_view
    except Exception as exc:  # noqa: BLE001 - optional analysis dependencies (h5py) may be absent
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    clean = np.nan_to_num(np.asarray(voltage_mv, float), nan=-1e3)
    table = cycle_table(time_ms, clean, pulse_ms, label)
    return dict(available=True, cycles=table, view=qc_view(table, pulse_ms))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args, emit):
    archive = open_archive(args.archive, args.archive_sha256)
    annotations = H01Annotations(args.cache)
    if args.tags:
        tags = tuple(args.tags)
        annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=tags))
    record = next(c for c in json.loads(args.components.read_text())["cells"] if c["cell_id"] == args.cell)
    imported = archive.load(args.cell, component=record["largest_component"])
    step_na = 0. if args.ramp_na is not None else args.current_na
    cell, evidence = make_h01_ei_cell(imported, annotations, polarity=args.polarity, donor=args.donor,
                                      regions=_regions(imported), region_basis=BASIS,
                                      current_na=step_na, delay_ms=args.pulse_on_ms,
                                      duration_ms=args.pulse_ms, max_cv_length_um=args.max_cv_um,
                                      pop_size=(1,), solver=args.solver)
    if args.ramp_na is not None:
        cell.place(imported.anatomy().soma_location(), ramp_clamp(args.pulse_on_ms, args.pulse_ms, args.ramp_na))
        evidence["ramp_max_na"] = args.ramp_na
    network = braincell.Network(name="h01_transfer")
    _register_cell(network, args.cell, cell, evidence, imported, {}, step_na, emit)
    evidence.update(component=record["largest_component"], source_sha256=imported.source_sha256,
                    component_nodes=len(imported.source_rows))
    return network, evidence


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--polarity", required=True, choices=["E", "I"])
    drive = parser.add_mutually_exclusive_group(required=True)
    drive.add_argument("--current-na", type=float, help="Step amplitude over the pulse window.")
    drive.add_argument("--ramp-na", type=float, help="Ramp from 0 at the pulse onset to this current at its end.")
    parser.add_argument("--pulse-on-ms", type=float, required=True)
    parser.add_argument("--pulse-ms", type=float, required=True)
    parser.add_argument("--duration-ms", type=float, required=True)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--solver", default="h01_staggered_calcium_implicit")
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--archive", type=Path, default=Path(".cache/h01/proofread104.zip"))
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256, help="Expected digest of --archive.")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Source tags for a cell the annotation cache lacks (e.g. pyramidal L3).")
    parser.add_argument("--components", type=Path, default=Path("docs/evidence/h01-population-components.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registered-count", type=int, default=None, help="Human recorded count at this input.")
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=None)
    parser.add_argument("--donor-model-count", type=int, default=None, help="Donor fit's count on its own anatomy.")
    args = parser.parse_args(argv)
    if args.current_na is not None and args.registered_count is None:
        parser.error("--registered-count is required with --current-na")
    if args.ramp_na is not None and not (np.isfinite(args.ramp_na) and args.ramp_na > 0.):
        parser.error("--ramp-na must be finite and positive")
    return args


def summarize(args, arrays, evidence):
    """Readings of one run from its saved arrays; the ramp branch adds rheobase and block."""
    pulse = (args.pulse_on_ms, args.pulse_on_ms+args.pulse_ms)
    observed = crossings(arrays["time_ms"], arrays["output_voltage"], pulse)
    soma = crossings(arrays["time_ms"], arrays["voltage"], pulse)
    observed_windows = windows(arrays["time_ms"], arrays["output_voltage"], pulse)
    ramp = None
    verdict = donor_verdict = None
    if args.ramp_na is not None:
        ramp = ramp_readings(arrays["time_ms"], arrays["output_voltage"], pulse, args.ramp_na)
        ramp["soma_site"] = ramp_readings(arrays["time_ms"], arrays["voltage"], pulse, args.ramp_na)
    else:
        verdict = count_verdict(observed["count"], args.registered_count, args.repeat_counts)
        if args.donor_model_count is not None:
            donor_verdict = count_verdict(observed["count"], args.donor_model_count, None)
    label = f"ramp {args.ramp_na:g} nA" if args.ramp_na is not None else f"{args.current_na:g} nA"
    return dict(
        cell=args.cell, donor=args.donor, polarity=args.polarity, component=evidence["component"],
        source_sha256=evidence["source_sha256"], component_nodes=evidence["component_nodes"],
        n_compartments=evidence["n_compartments"], dt_ms=args.dt_ms, duration_ms=args.duration_ms,
        solver=args.solver, max_cv_um=args.max_cv_um, current_na=args.current_na, pulse_ms=list(pulse),
        ramp_max_na=args.ramp_na, ramp=ramp, drive="ramp" if args.ramp_na is not None else "step",
        registered_human_count=args.registered_count, repeat_counts=args.repeat_counts,
        donor_model_count_on_own_anatomy=args.donor_model_count,
        output_site=observed, soma_site=soma, verdict_vs_human=verdict, verdict_vs_donor_model=donor_verdict,
        rest_before_pulse_mv=float(arrays["output_voltage"][int(args.pulse_on_ms/args.dt_ms)-1]),
        **observed_windows,
        usable_tier=usable_features(arrays["time_ms"], arrays["output_voltage"], pulse, label),
        peak_mv=float(np.nanmax(arrays["output_voltage"])), min_mv=float(np.nanmin(arrays["output_voltage"])),
        tags=args.tags)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    emit = lambda message: print(f"[{time.perf_counter()-started:.1f}s] {message}", flush=True)
    with brainstate.environ.context(precision=64):
        network, evidence = build(args, emit)
        evidence["construction_seconds"] = time.perf_counter()-started
        emit("Initializing")
        init = init_h01_network_states(network, progress=emit, heartbeat_seconds=30.)
        emit("Compiling and stepping")
        run_started = time.perf_counter()
        result = network.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms, spike_recording="population")
        run_seconds = time.perf_counter()-run_started
        time_ms = np.asarray(result.time.to_decimal(u.ms))+args.dt_ms
        traces = result.traces["cell_"+args.cell]
        arrays = {"time_ms": time_ms}
        for name, trace in traces.items():
            arrays[name] = np.asarray(trace.to_decimal(u.mV))[:, 0]
        arrays["events"] = np.asarray(result.spikes["cell_"+args.cell])
    if args.ramp_na is not None:
        arrays["ramp_current_na"] = ramp_current_na(time_ms, (args.pulse_on_ms, args.pulse_on_ms+args.pulse_ms), args.ramp_na)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    summary = summarize(args, arrays, evidence)
    summary.update(
        construction_seconds=evidence["construction_seconds"], init_seconds=init["init_seconds_by_population"],
        run_seconds=run_seconds, steps=int(len(time_ms)), peak_rss_mb=process_rss_mb(peak=True),
        inputs=dict(archive=str(args.archive), archive_sha256=sha256(args.archive),
                    expected_archive_sha256=args.archive_sha256, components_sha256=sha256(args.components)),
        traces_sha256=sha256(args.output.with_suffix(".npz")),
        scope="One deployed donor on retained production-imported H01 anatomy under the donor recording's "
              "step protocol (or a ramp to the stated maximum). A physiological transfer reading; not a "
              "measurement of the H01 donor's own cell.")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    if summary["ramp"] is not None:
        emit(f"ramp to {args.ramp_na:g} nA: rheobase {summary['ramp']['rheobase_na']} nA, "
             f"block {summary['ramp']['block_na']} nA, {summary['ramp']['ramp_count']} crossings")
    else:
        emit(f"count {summary['output_site']['count']} vs human {args.registered_count}: {summary['verdict_vs_human']}")


if __name__ == "__main__":
    main()
