"""Measure real H01 construction, cable execution and the pp-prop compile gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

import brainstate
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import psutil

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_network import make_h01_network
from braintrace.datasets.h01_network_init import init_h01_network_states, process_rss_mb
from examples.pp_prop.h01_arc_model import H01ArcModel


def main():
    """Run one bounded population probe and retain each completed phase.

    Returns
    -------
    None
        The explicit output file records measured phases and any failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cells", type=int, choices=(4, 12, 40, 104), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-learning", action="store_true")
    parser.add_argument("--wall-limit-seconds", type=float, default=900.)
    parser.add_argument("--rss-limit-gib", type=float, default=16.)
    args = parser.parse_args()
    if not np.isfinite([args.wall_limit_seconds, args.rss_limit_gib]).all() or min(
            args.wall_limit_seconds, args.rss_limit_gib) <= 0:
        parser.error("Resource limits must be finite and positive")
    report = {"status": "running", "cells": args.cells, "phases": {},
              "event_ms": .1, "dt_ms": .005, "substeps": 20,
              "solver": "h01_staggered_calcium_implicit", "physiology": "unqualified"}
    report["feature_probe"] = "all-ones synthetic 441-feature vector; not an encoded ARC episode"
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ("examples/pp_prop/h01_arc_model.py", "braintrace/datasets/h01_network_step.py",
                     "braintrace/datasets/h01_calcium_solver.py", "braintrace/datasets/h01_construction.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["limits"] = {"wall_seconds": args.wall_limit_seconds, "rss_gib": args.rss_limit_gib}
    stopped = threading.Event()
    started = time.monotonic()

    def monitor():
        process = psutil.Process()
        while not stopped.wait(1.):
            elapsed, rss = time.monotonic()-started, process.memory_info().rss/2**30
            if elapsed >= args.wall_limit_seconds or rss >= args.rss_limit_gib:
                args.output.with_suffix(".limit.json").write_text(json.dumps({
                    "status": "untested_resource_limit", "elapsed_seconds": elapsed,
                    "rss_gib": rss, "phase": report.get("active_phase"),
                    "limits": report["limits"],
                }, indent=2) + "\n")
                os._exit(124)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()

    def save():
        report["peak_rss_mb"] = process_rss_mb(peak=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(name, call):
        report["active_phase"] = name
        save()
        start = time.perf_counter()
        status = "failed"
        try:
            result = call()
            status = "pass"
            return result
        finally:
            report["phases"][name] = {"seconds": time.perf_counter()-start, "status": status}
            save()
            print(name, report["phases"][name], flush=True)

    try:
        with brainstate.environ.context(precision=64):
            topology = json.loads(Path("docs/evidence/h01-verified-network.json").read_text())
            components = json.loads(Path("docs/evidence/h01-population-components-soma.json").read_text())
            network, evidence = phase("construction", lambda: make_h01_network(
                topology, H01Archive(args.cache/"proofread104.zip"), H01Annotations(args.cache),
                include_isolated=True, cells=args.cells, components=components,
                solver=report["solver"], progress=lambda text: print(text, flush=True)))
            report["evidence"] = evidence
            report["compartments"] = sum(row["n_compartments"] for row in evidence["cells"].values())
            phase("initialization", lambda: init_h01_network_states(network, progress=print))
            native = phase("native_compile_and_event", lambda: network.run(
                dt=.005*u.ms, duration=.1*u.ms, spike_recording="population"))
            native_soma = np.asarray([native.traces[name]["voltage"][-1].to_decimal(u.mV).reshape(())
                                      for name in network.populations])
            network.reset_state()
            model = phase("model_setup", lambda: H01ArcModel(network, evidence["simulated_cell_ids"]))
            step = brainstate.transform.jit(model.step)
            soma = phase("adapter_compile_and_event", lambda: np.asarray(step(jnp.zeros(441))))
            report["zero_input_native_max_error_mv"] = float(np.max(np.abs(soma-native_soma)))
            report["finite_zero_input"] = bool(np.isfinite(soma).all())
            model.reset_episode()
            driven = phase("encoded_event_warm", lambda: np.asarray(step(jnp.ones(441))))
            report["finite_driven"] = bool(np.isfinite(driven).all())
            report["encoder_effect_mv"] = float(np.max(np.abs(driven-soma)))
            states = brainstate.graph.states(model)
            report["nonfinite_states"] = [str(path) for path, state in states.items()
                if any(not np.isfinite(np.asarray(u.get_mantissa(value))).all()
                       for value in jax.tree_util.tree_leaves(state.value))]
            report["state_count"] = len(states)
            model.reset_episode()
            if args.compile_learning:
                learner = braintrace.pp_prop(model, decay_or_rank=.99, vjp_method="single-step")
                phase("pp_prop_compile", lambda: learner.compile_graph(jnp.zeros(441)))
            report["status"] = "forward_pass" if (not report["nonfinite_states"] and report["finite_driven"] and
                report["finite_zero_input"] and report["zero_input_native_max_error_mv"] < 1e-8) else "fail"
    except Exception as exc:
        report["status"] = "blocked"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:16000]
        report["traceback"] = "".join(traceback.format_tb(exc.__traceback__))
        print(report["error_type"], report["error"][:2000], flush=True)
    finally:
        save()
        stopped.set()
        watcher.join()


if __name__ == "__main__":
    main()
