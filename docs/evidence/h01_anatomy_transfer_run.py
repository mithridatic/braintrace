"""Drive one production-imported H01 cell with its deployed donor's recorded step protocol.

The cell is built exactly as ``make_h01_network`` builds it (same importer, component
selection, electrical partition, donor profile, CV policy and solver); only the somatic
pulse timing and amplitude follow the donor recording's protocol instead of the population's
assumed 1 nA / 2-5 ms probe. The output is one bounded physiological reading on retained
human anatomy: -20 mV crossing count inside the pulse, first-spike latency, finiteness.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import braincell
import brainstate
import brainunit as u
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"docs"/"evidence"))

from braintrace.datasets.h01 import H01Archive  # noqa: E402
from braintrace.datasets.h01_annotations import H01Annotations  # noqa: E402
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell  # noqa: E402
from braintrace.datasets.h01_network import _regions, _register_cell  # noqa: E402
from braintrace.datasets.h01_network_init import init_h01_network_states, process_rss_mb  # noqa: E402
from h01_pv_human_datums import spike_datums  # noqa: E402

BASIS = ("Source soma/axon/AIS samples extend halfway along adjacent edges; remaining cable gets "
         "dendrite properties. Myelin is not modeled. This is the existing inferred electrical "
         "partition, not measured channel placement.")


def crossings(time_ms, voltage_mv, pulse_ms, detection_mv=-20.):
    """Count -20 mV rise crossings inside the pulse; first latency after onset; finiteness."""
    time_ms, voltage_mv = np.asarray(time_ms, float), np.asarray(voltage_mv, float)
    finite = bool(np.isfinite(voltage_mv).all())
    mask = (time_ms >= pulse_ms[0]) & (time_ms < pulse_ms[1])
    events = spike_datums(time_ms[mask], np.nan_to_num(voltage_mv[mask], nan=-1e3), detection_mv) if mask.any() else []
    first = events[0]["rise_crossing_ms"]-pulse_ms[0] if events else None
    return dict(count=len(events), first_spike_ms=first, finite=finite,
                crossing_times_ms=[e["rise_crossing_ms"] for e in events])


def count_verdict(model, registered, repeat_counts):
    """Exact/within-one/rejected against a registered count; band verdict against repeats."""
    difference = abs(model-registered)
    exact = "held" if difference == 0 else ("missed_within_one" if difference <= 1 else "rejected")
    band = None
    if repeat_counts:
        band = "held" if min(repeat_counts) <= model <= max(repeat_counts) else "missed"
    return dict(exact=exact, repeat_band=band)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args, emit):
    archive = H01Archive(args.cache/"proofread104.zip")
    annotations = H01Annotations(args.cache)
    record = next(c for c in json.loads(args.components.read_text())["cells"] if c["cell_id"] == args.cell)
    imported = archive.load(args.cell, component=record["largest_component"])
    cell, evidence = make_h01_ei_cell(imported, annotations, polarity=args.polarity, donor=args.donor,
                                      regions=_regions(imported), region_basis=BASIS,
                                      current_na=args.current_na, delay_ms=args.pulse_on_ms,
                                      duration_ms=args.pulse_ms, max_cv_length_um=args.max_cv_um,
                                      pop_size=(1,), solver=args.solver)
    network = braincell.Network(name="h01_transfer")
    _register_cell(network, args.cell, cell, evidence, imported, {}, args.current_na, emit)
    evidence.update(component=record["largest_component"], source_sha256=imported.source_sha256,
                    component_nodes=len(imported.source_rows))
    return network, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--polarity", required=True, choices=["E", "I"])
    parser.add_argument("--current-na", type=float, required=True)
    parser.add_argument("--pulse-on-ms", type=float, required=True)
    parser.add_argument("--pulse-ms", type=float, required=True)
    parser.add_argument("--duration-ms", type=float, required=True)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--solver", default="h01_staggered_calcium_implicit")
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--components", type=Path, default=Path("docs/evidence/h01-population-components.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registered-count", type=int, required=True, help="Human recorded count at this input.")
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=None)
    parser.add_argument("--donor-model-count", type=int, default=None, help="Donor fit's count on its own anatomy.")
    args = parser.parse_args()
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
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    pulse = (args.pulse_on_ms, args.pulse_on_ms+args.pulse_ms)
    observed = crossings(arrays["time_ms"], arrays["output_voltage"], pulse)
    soma = crossings(arrays["time_ms"], arrays["voltage"], pulse)
    verdict = count_verdict(observed["count"], args.registered_count, args.repeat_counts)
    summary = dict(
        cell=args.cell, donor=args.donor, polarity=args.polarity, component=evidence["component"],
        source_sha256=evidence["source_sha256"], component_nodes=evidence["component_nodes"],
        n_compartments=evidence["n_compartments"], dt_ms=args.dt_ms, duration_ms=args.duration_ms,
        solver=args.solver, max_cv_um=args.max_cv_um, current_na=args.current_na, pulse_ms=list(pulse),
        registered_human_count=args.registered_count, repeat_counts=args.repeat_counts,
        donor_model_count_on_own_anatomy=args.donor_model_count,
        output_site=observed, soma_site=soma, verdict_vs_human=verdict,
        verdict_vs_donor_model=(count_verdict(observed["count"], args.donor_model_count, None)
                                if args.donor_model_count is not None else None),
        rest_before_pulse_mv=float(arrays["output_voltage"][int(args.pulse_on_ms/args.dt_ms)-1]),
        peak_mv=float(np.nanmax(arrays["output_voltage"])), min_mv=float(np.nanmin(arrays["output_voltage"])),
        construction_seconds=evidence["construction_seconds"], init_seconds=init["init_seconds_by_population"],
        run_seconds=run_seconds, steps=int(len(time_ms)), peak_rss_mb=process_rss_mb(peak=True),
        inputs=dict(archive_sha256=sha256(args.cache/"proofread104.zip"), components_sha256=sha256(args.components)),
        traces_sha256=sha256(args.output.with_suffix(".npz")),
        scope="One deployed donor on retained production-imported H01 anatomy under the donor recording's "
              "step protocol. A physiological transfer reading; not a measurement of the H01 donor's own cell.")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    emit(f"count {observed['count']} vs human {args.registered_count}: {verdict}")


if __name__ == "__main__":
    main()
