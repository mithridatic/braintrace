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
from braintrace.datasets.h01_network import make_h01_network


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
    parser.add_argument("--disconnected", action="store_true")
    parser.add_argument("--duration-ms", type=float, default=0.)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--current-na", type=float, default=0., help="Same assumed soma pulse for each incident cell.")
    parser.add_argument("--solver", default="h01_staggered_scan", choices=["h01_staggered_scan", "staggered"])
    args = parser.parse_args()
    if not np.isfinite([args.duration_ms, args.dt_ms, args.current_na]).all() or args.duration_ms < 0 or args.dt_ms <= 0:
        parser.error("duration must be nonnegative, dt positive, and all inputs finite")
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
        if not args.build and args.duration_ms == 0:
            return
        ids = {e[s+"_cell"] for e in topology["contacts"] if e["construction_ready"] for s in ("pre", "post")}
        started = time.perf_counter()
        network, evidence = make_h01_network(topology, H01Archive(args.cache/"proofread104.zip"),
            H01Annotations(args.cache), disconnected=args.disconnected, max_cv_length_um=args.max_cv_um,
            currents_na={identity: args.current_na for identity in ids}, solver=args.solver,
            progress=lambda message: print(f"[{time.perf_counter()-started:.1f}s] {message}", flush=True))
        print("Built", len(evidence["cells"]), "cells and", len(network.projections), "projections", flush=True)
        evidence["execution"] = "constructed; not simulated"
        evidence["construction_seconds"] = time.perf_counter()-started
        evidence["compartments_by_cell"] = {identity: record["n_compartments"] for identity, record in evidence["cells"].items()}
        build_path = args.output.parent/(args.output.name+"-build.json")
        build_path.write_text(json.dumps(evidence, indent=2)+"\n")
        if args.duration_ms:
            print(f"[{time.perf_counter()-started:.1f}s] Initializing and running {args.duration_ms} ms", flush=True)
            run_started = time.perf_counter()
            result = network.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms, spike_recording="population")
            arrays = {"time_ms": np.asarray(result.time.to_decimal(u.ms))+args.dt_ms}
            for population, traces in result.traces.items():
                for name, trace in traces.items():
                    values = np.asarray(trace.to_decimal(u.uS if name.endswith("_g") else u.mV))
                    if not np.isfinite(values).all():
                        raise RuntimeError("Nonfinite trace: "+population+"/"+name)
                    arrays[population+"_"+name] = values
                arrays[population+"_events"] = np.asarray(result.spikes[population])
            np.savez_compressed(args.output.parent/(args.output.name+"-traces.npz"), **arrays)
            evidence.update(execution="finite compiled smoke run", dt_ms=args.dt_ms, duration_ms=args.duration_ms,
                            initialization_and_run_seconds=time.perf_counter()-run_started,
                            sample_convention="end of step; Network start times plus dt")
            print(f"[{time.perf_counter()-started:.1f}s] Finite simulation traces saved", flush=True)
        build_path.write_text(json.dumps(evidence, indent=2)+"\n")


if __name__ == "__main__":
    main()
