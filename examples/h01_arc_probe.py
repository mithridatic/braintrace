"""Measure real H01 construction, cable execution and the pp-prop compile gate."""

import argparse
import hashlib
from importlib.metadata import version
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


def _sparse_learning_probe(model, report, phase):
    from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter
    module = Example21ArcAdapter(Path('.'))._model()
    learner = braintrace.pp_prop.sparse(model, .99, max_bytes=512*1024**2)
    phase('sparse_pp_prop_compile', lambda: learner.compile_graph(jnp.zeros(441)))
    layout = learner.graph.layout
    report['eligibility'] = {'elements': layout.elements, 'bytes': layout.nbytes,
        'colors': layout.color_count, 'max_support_width': max(map(len, layout.outputs), default=0),
        'state_blocks': len(layout.shapes), 'factor_limit_bytes': 512*1024**2}
    parameters = {'input': model.input_weight.value, 'recurrent': model.recurrent_weight.value,
                  'readout_weight': model.readout_weight.value, 'readout_bias': model.readout_bias.value}
    before = {name: np.array(value) for name, value in parameters.items()}
    trainer = module.PPPropEpisodeTrainer(learner, parameters)
    def loss(event):
        learner(event)
        return jnp.mean(jnp.square(model.readout()))
    train = brainstate.transform.jit(lambda: trainer.update_episode(jnp.ones((1, 441)), step_fn=loss))
    cold = phase('sparse_muon_compile_and_update', lambda: jax.block_until_ready(train()))
    report['first_learning_update'] = {'loss': float(cold[0]), 'gradient_norm': float(cold[1]),
        'finite': bool(np.isfinite(np.asarray(cold)).all()), 'updates': int(trainer.updates)}
    warm = phase('sparse_muon_warm_update', lambda: jax.block_until_ready(train()))
    report['learning_probe'] = {'objective': 'synthetic event squared-logit loss; not ARC training',
        'cold_loss': float(cold[0]), 'warm_loss': float(warm[0]),
        'cold_gradient_norm': float(cold[1]), 'warm_gradient_norm': float(warm[1]),
        'updates': int(trainer.updates), 'optimizer_finite': bool(trainer.optimizer_is_finite()),
        'changed_parameters': [name for name, value in trainer.parameters.items()
                               if not np.array_equal(before[name], value)],
        'contact_note': 'Contact activity is not forced; parameter changes can include Muon weight decay.'}
    report['learning_probe']['finite'] = bool(np.isfinite(np.asarray(cold)).all()
        and np.isfinite(np.asarray(warm)).all() and trainer.optimizer_is_finite()
        and all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(learner.factors.value)))


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
    parser.add_argument("--sparse-learning", action="store_true")
    parser.add_argument("--wall-limit-seconds", type=float, default=900.)
    parser.add_argument("--rss-limit-gib", type=float, default=16.)
    args = parser.parse_args()
    if not np.isfinite([args.wall_limit_seconds, args.rss_limit_gib]).all() or min(
            args.wall_limit_seconds, args.rss_limit_gib) <= 0:
        parser.error("Resource limits must be finite and positive")
    report = {"status": "running", "cells": args.cells, "phases": {},
              "event_ms": .1, "dt_ms": .005, "substeps": 20,
              "solver": "h01_staggered_calcium_implicit", "physiology": "unqualified"}
    report['devices'] = [str(device) for device in jax.devices()]
    report['dependencies'] = {name: version(name) for name in
        ('jax', 'jaxlib', 'brainstate', 'brainunit', 'braincell', 'brainevent', 'optax', 'numpy')}
    report["feature_probe"] = "all-ones synthetic 441-feature vector; not an encoded ARC episode"
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ("examples/pp_prop/h01_arc_model.py", "braintrace/datasets/h01_network_step.py",
                     "braintrace/datasets/h01_calcium_solver.py", "braintrace/datasets/h01_construction.py",
                     "braintrace/datasets/h01_dhs_scan.py")}
    if args.sparse_learning:
        report['source_sha256'].update({name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('braintrace/_algorithm/sparse_pp_prop.py', 'braintrace/_algorithm/sparse_io.py',
                         'braintrace/_compiler/sparse_io_graph.py', 'braintrace/_compiler/sparse_support.py',
                         'braintrace/_compiler/sparse_influence.py')})
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
        report['device_memory_bytes'] = {str(device): {key: int(value) for key, value in
            (device.memory_stats() or {}).items() if isinstance(value, (int, np.integer))}
            for device in jax.devices()}
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
            if args.sparse_learning:
                _sparse_learning_probe(model, report, phase)
            report["status"] = "forward_pass" if (not report["nonfinite_states"] and report["finite_driven"] and
                report["finite_zero_input"] and report["zero_input_native_max_error_mv"] < 1e-8) else "fail"
            if args.sparse_learning:
                report['status'] = ('learning_probe_pass' if report['status'] == 'forward_pass'
                                    and report['learning_probe']['finite'] else 'fail')
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
