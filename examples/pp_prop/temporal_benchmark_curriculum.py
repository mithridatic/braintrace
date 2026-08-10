"""Checkpointed short-to-long curriculum with persistent optimizer state."""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import brainstate
import brainunit as u
import jax
import numpy as np

from temporal_benchmark_config import (
    FEEDFORWARD_SYNAPSE_TAU_MS,
    HORIZONS,
    MEMBRANE_TAU_MS,
    READOUT_TAU_MS,
    RECURRENT_SYNAPSE_TAU_MS,
    HorizonName,
    TemporalBenchmarkConfig,
    half_life_decay,
)
from temporal_benchmark_data import response_is_label_independent
from temporal_benchmark_manifest import SeedBundle
from temporal_benchmark_metrics import stability_passes
from temporal_benchmark_supervision import algorithm_label, policy_for_arm
from temporal_benchmark_topology import topology_metrics
from temporal_benchmark_training import (
    _build_runtime,
    _compile_learner,
    _evaluate,
    _format_telemetry,
    _make_train_many,
    _training_batches,
)

CURRICULUM_PHASES: tuple[tuple[HorizonName, int], ...] = (
    ("short", 200),
    ("medium", 400),
    ("long", 800),
)
PROMOTION_ACCURACY = 0.80
PROMOTION_CHECKPOINTS = 2


def _merge_telemetry(
    destination: dict[str, list[jax.Array]], source: dict[str, jax.Array]
) -> None:
    for group, values in source.items():
        destination.setdefault(group, []).append(values)


def _concatenate_telemetry(
    chunks: dict[str, list[jax.Array]],
) -> dict[str, jax.Array]:
    return {
        group: jax.numpy.concatenate(values, axis=0) for group, values in chunks.items()
    }


def _recurrent_ratios(telemetry: dict[str, jax.Array]) -> np.ndarray:
    recurrent = telemetry.get("recurrent")
    return np.zeros(1) if recurrent is None else np.asarray(recurrent)[:, 5]


def _run_phase(runtime, config, bundle):
    spikes, labels = _training_batches(config, bundle)
    trainer = _make_train_many(runtime, config)
    telemetry_chunks: dict[str, list[jax.Array]] = {}
    loss_chunks: list[jax.Array] = []
    history: list[dict[str, object]] = []
    consecutive = 0
    completed = 0
    while completed < config.updates:
        boundary = min(completed + config.evaluation_interval, config.updates)
        losses, telemetry = trainer(
            spikes[completed:boundary], labels[completed:boundary]
        )
        jax.block_until_ready(losses)
        loss_chunks.append(losses)
        _merge_telemetry(telemetry_chunks, telemetry)
        completed = boundary
        combined = _concatenate_telemetry(telemetry_chunks)
        metrics, dynamics = _evaluate(runtime, config, bundle)
        stable = stability_passes(dynamics, _recurrent_ratios(combined))
        consecutive = (
            consecutive + 1
            if metrics["ensemble_accuracy"] >= PROMOTION_ACCURACY and stable
            else 0
        )
        history.append(
            {"update": completed, "metrics": metrics, "stability_passed": stable}
        )
        if consecutive >= PROMOTION_CHECKPOINTS:
            break
    return {
        "promoted": consecutive >= PROMOTION_CHECKPOINTS,
        "updates_completed": completed,
        "history": history,
        "losses": np.asarray(jax.numpy.concatenate(loss_chunks)).tolist(),
        "telemetry": _concatenate_telemetry(telemetry_chunks),
    }


def _execute_curriculum(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> dict[str, object]:
    started = time.perf_counter()
    total_budget = sum(budget for _, budget in CURRICULUM_PHASES)
    short_config = replace(
        config,
        horizon="short",
        updates=CURRICULUM_PHASES[0][1],
        trace_half_life_x_steps=config.curriculum_trace_half_lives.short.x,
        trace_half_life_f_steps=config.curriculum_trace_half_lives.short.f,
    )
    runtime = _build_runtime(short_config, bundle, total_budget)
    phase_results: dict[str, object] = {}
    all_telemetry: dict[str, list[jax.Array]] = {}
    failed_phase = None
    for horizon, budget in CURRICULUM_PHASES:
        half_lives = config.curriculum_trace_half_lives.for_horizon(horizon)
        phase_config = replace(
            config,
            horizon=horizon,
            updates=budget,
            trace_half_life_x_steps=half_lives.x,
            trace_half_life_f_steps=half_lives.f,
        )
        if horizon != "short":
            runtime = replace(
                runtime, learner=_compile_learner(runtime.model, phase_config)
            )
        phase = _run_phase(runtime, phase_config, bundle)
        phase_results[horizon] = {
            key: value for key, value in phase.items() if key != "telemetry"
        }
        _merge_telemetry(all_telemetry, phase["telemetry"])
        if not phase["promoted"]:
            failed_phase = horizon
            break
    long_config = replace(config, horizon="long")
    final_metrics, dynamics = _evaluate(runtime, long_config, bundle)
    combined = _concatenate_telemetry(all_telemetry)
    sealed_metrics = (
        _evaluate(runtime, long_config, bundle, "test")[0]
        if config.sealed_test
        else None
    )
    return {
        "status": "completed" if failed_phase is None else "stopped",
        "algorithm": algorithm_label(policy_for_arm(config.arm)),
        "bundle_id": bundle.bundle_id,
        "arm": config.arm,
        "horizon": "long",
        "curriculum": True,
        "failed_phase": failed_phase,
        "phases": phase_results,
        "final_validation": final_metrics,
        "sealed_test_metrics": sealed_metrics,
        "optimizer_telemetry": _format_telemetry(combined),
        "dynamics": dynamics,
        "topology": topology_metrics(runtime.topology),
        "trace_half_lives": {
            horizon: asdict(config.curriculum_trace_half_lives.for_horizon(horizon))
            for horizon in HORIZONS
        },
        "trace_decays": {
            horizon: {
                "x": half_life_decay(
                    config.curriculum_trace_half_lives.for_horizon(horizon).x
                ),
                "f": half_life_decay(
                    config.curriculum_trace_half_lives.for_horizon(horizon).f
                ),
            }
            for horizon in HORIZONS
        },
        "fixed_dynamics_time_constants_ms": {
            "feedforward_synapse": FEEDFORWARD_SYNAPSE_TAU_MS,
            "recurrent_synapse": RECURRENT_SYNAPSE_TAU_MS,
            "membrane": MEMBRANE_TAU_MS,
            "readout": READOUT_TAU_MS,
        },
        "response_label_independent": response_is_label_independent(
            HORIZONS["long"], config.cue_rate_hz, config.go_rate_hz
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "config": asdict(config),
    }


def run_curriculum(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> dict[str, object]:
    """Train short, medium, and long phases while carrying weights and Adam state."""
    with brainstate.environ.context(dt=config.dt_seconds * u.second):
        return _execute_curriculum(config, bundle)
