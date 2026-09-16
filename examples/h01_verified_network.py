"""Export verified H01 connectivity and optionally build or run its circuit."""

import argparse
import json
import time
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_connectivity import prepare_connectivity
from braintrace.datasets.h01_network import make_h01_network, plan_h01_cells
from braintrace.datasets.h01_network_init import heartbeat, init_h01_network_states, process_rss_mb


def main():
    """Export topology, then optionally construct and run the incident cells.

    Returns
    -------
    None
        Writes JSON provenance and NumPy adjacency or trace arrays.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--audit", type=Path, default=Path("docs/evidence/h01-resolved-edge-list.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", type=Path, help="Reuse a prepared topology instead of projecting again.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--disconnected", action="store_true", help="Deprecated alias of --control disconnected.")
    parser.add_argument("--control", default="ei", choices=["ei", "e_only", "i_only", "disconnected"],
                        help="Projection control: which presynaptic Dale signs keep their projections.")
    parser.add_argument("--include-isolated", action="store_true",
                        help="Also build cells without constructible contacts using the component inventory selections.")
    parser.add_argument("--cells", type=int, help="Build only the first N cells of the evidence cell_order.")
    parser.add_argument("--components", type=Path, default=Path("docs/evidence/h01-population-components-soma.json"),
                        help="Component inventory used for isolated cells.")
    parser.add_argument("--init-only", action="store_true",
                        help="Construct, run init_state cell by cell (timed, peak RSS recorded), then exit without compiling or stepping.")
    parser.add_argument("--heartbeat-s", type=float, default=60.,
                        help="Heartbeat period (elapsed s, RSS) while init_state or compile+run is otherwise silent.")
    parser.add_argument("--duration-ms", type=float, default=0.)
    parser.add_argument("--dt-ms", type=float, default=.000625)  # qualified step, docs/evidence/h01-timestep-ladder.json
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--current-na", type=float, default=0., help="Same assumed soma pulse for each incident cell.")
    parser.add_argument("--currents-json", type=Path,
                        help="JSON {cell_id: nA} of per-cell soma pulse amplitudes; listed cells override --current-na.")
    parser.add_argument("--pulse-delay-ms", type=float, default=2., help="Soma pulse onset shared by all cells.")
    parser.add_argument("--pulse-duration-ms", type=float, default=3., help="Soma pulse length shared by all cells.")
    parser.add_argument("--solver", default="h01_staggered_scan", choices=["h01_staggered_scan", "staggered", "h01_staggered_calcium_implicit"])
    args = parser.parse_args()
    if args.disconnected:
        if args.control not in ("ei", "disconnected"):
            parser.error("--disconnected conflicts with --control "+args.control)
        args.control = "disconnected"
    if args.cells is not None and args.cells < 1:
        parser.error("--cells must be positive")
    if not np.isfinite([args.duration_ms, args.dt_ms, args.current_na]).all() or args.duration_ms < 0 or args.dt_ms <= 0:
        parser.error("duration must be nonnegative, dt positive, and all inputs finite")
    if (not np.isfinite([args.pulse_delay_ms, args.pulse_duration_ms]).all() or args.pulse_delay_ms < 0
            or args.pulse_duration_ms <= 0):
        parser.error("pulse onset must be nonnegative and pulse length positive")
    overrides = json.loads(args.currents_json.read_text(encoding="utf-8")) if args.currents_json else {}
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in overrides.values()):
        parser.error("every --currents-json amplitude must be a finite number")
    if args.init_only and args.duration_ms:
        parser.error("--init-only excludes --duration-ms")
    if not args.heartbeat_s > 0:
        parser.error("--heartbeat-s must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with brainstate.environ.context(precision=64):
        topology = json.loads(args.topology.read_text()) if args.topology else prepare_connectivity(args.cache, args.audit)
        args.output.with_suffix(".json").write_text(json.dumps(topology, indent=2)+"\n")
        count = len(topology["nodes"])
        counts = np.zeros((count, count), dtype=np.int32)
        signed_weights = np.zeros((count, count))
        runnable_counts = np.zeros_like(counts)
        for edge in topology["contacts"]:
            i, j = edge["pre_index"], edge["post_index"]
            counts[i, j] += 1
            signed_weights[i, j] += .01 if edge["dale_sign"] == 1 else -.02
            if edge["construction_ready"]:
                runnable_counts[i, j] += 1
        np.savez_compressed(args.output.with_suffix(".npz"),
            cell_ids=np.array([n["cell_id"] for n in topology["nodes"]]),
            dale_sign=np.array([n["dale_sign"] for n in topology["nodes"]]),
            contact_count=counts, assumed_signed_weight_us=signed_weights,
            placed_contact_count=runnable_counts,
            placed_assumed_signed_weight_us=np.where(runnable_counts > 0, signed_weights, 0.))
        print(topology["counts"], flush=True)
        if not args.build and not args.init_only and args.duration_ms == 0:
            return
        components = json.loads(args.components.read_text(encoding="utf-8")) if args.include_isolated else None
        options = dict(control=args.control, include_isolated=args.include_isolated, cells=args.cells, components=components)
        plan = plan_h01_cells(topology, **options)
        started = time.perf_counter()
        emit = lambda message: print(f"[{time.perf_counter()-started:.1f}s] {message}", flush=True)
        network, evidence = make_h01_network(topology, H01Archive(args.cache/"proofread104.zip"),
            H01Annotations(args.cache), max_cv_length_um=args.max_cv_um, solver=args.solver, **options,
            currents_na={identity: float(overrides.get(identity, args.current_na)) for identity in plan["simulated_cell_ids"]},
            pulse_delay_ms=args.pulse_delay_ms, pulse_duration_ms=args.pulse_duration_ms, progress=emit)
        print("Built", len(evidence["cells"]), "cells and", len(network.projections), "projections", flush=True)
        evidence["execution"] = "constructed; not simulated"
        evidence["construction_seconds"] = time.perf_counter()-started
        evidence["compartments_by_cell"] = {identity: record["n_compartments"] for identity, record in evidence["cells"].items()}
        evidence["n_compartments"] = sum(evidence["compartments_by_cell"].values())
        evidence["timing_separation"] = "init_state separable; compile measured together with stepping"
        build_path = args.output.parent/(args.output.name+"-build.json")
        build_path.write_text(json.dumps(evidence, indent=2)+"\n")
        if args.init_only or args.duration_ms:
            emit("Initializing cell states"+(f" and running {args.duration_ms} ms" if args.duration_ms else " only"))
            run_started = time.perf_counter()
            init = init_h01_network_states(network, progress=emit, heartbeat_seconds=args.heartbeat_s)
            init_seconds = time.perf_counter()-run_started
            evidence.update(init_state_seconds=init_seconds, init_seconds_by_cell=init["init_seconds_by_population"],
                            init_peak_rss_mb=init["peak_rss_mb"],
                            rss_convention="psutil resident set of this interpreter; peak_wset where the platform reports it")
            emit(f"Cell states initialized in {init_seconds:.1f} s, peak RSS {init['peak_rss_mb']:.0f} MB")
            evidence["execution"] = "constructed and initialized; not compiled or simulated"
            build_path.write_text(json.dumps(evidence, indent=2)+"\n")
        if args.init_only:
            emit("Init-only mode: exiting before compile and stepping")
            return
        if args.duration_ms:
            emit("Compiling and stepping")
            with heartbeat("compile and run", emit, seconds=args.heartbeat_s):
                result = network.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms, spike_recording="population")
            arrays = {"time_ms": np.asarray(result.time.to_decimal(u.ms))+args.dt_ms}
            for population, traces in result.traces.items():
                for name, trace in traces.items():
                    values = np.asarray(trace.to_decimal(u.uS if name.endswith("_g") else u.mV))
                    arrays[population+"_"+name] = values
                arrays[population+"_events"] = np.asarray(result.spikes[population])
            nonfinite = {}
            for name, values in arrays.items():
                mask = ~np.isfinite(values)
                if mask.any():
                    index = tuple(int(i) for i in np.argwhere(mask)[0])
                    timestamp = arrays["time_ms"][index[0]]
                    nonfinite[name] = dict(count=int(mask.sum()), first_index=list(index),
                        first_time_ms=float(timestamp) if np.isfinite(timestamp) else None)
            np.savez_compressed(args.output.parent/(args.output.name+"-traces.npz"), **arrays)
            evidence.update(execution="nonfinite compiled run" if nonfinite else "finite compiled smoke run",
                            nonfinite_arrays=nonfinite, dt_ms=args.dt_ms, duration_ms=args.duration_ms,
                            run_peak_rss_mb=process_rss_mb(peak=True),
                            initialization_and_run_seconds=time.perf_counter()-run_started,
                            compile_and_run_seconds=time.perf_counter()-run_started-init_seconds,
                            sample_convention="end of step; Network start times plus dt")
            emit("Nonfinite simulation traces saved for diagnosis" if nonfinite else "Finite simulation traces saved")
        build_path.write_text(json.dumps(evidence, indent=2)+"\n")
        if args.duration_ms and nonfinite:
            raise RuntimeError("Nonfinite traces: "+", ".join(nonfinite))


if __name__ == "__main__":
    main()
