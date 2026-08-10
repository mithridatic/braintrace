"""Paired curriculum and sample-tick-matched direct-long development run."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping

import brainstate
import brainunit as u
import jax
import numpy as np

from temporal_benchmark_config import HORIZONS, TemporalBenchmarkConfig
from temporal_benchmark_curriculum import CURRICULUM_PHASES, run_curriculum
from temporal_benchmark_data import response_is_label_independent
from temporal_benchmark_manifest import SeedBundle
from temporal_benchmark_metrics import stability_passes
from temporal_benchmark_supervision import algorithm_label, policy_for_arm
from temporal_benchmark_topology import topology_metrics
from temporal_benchmark_training import (
    _build_runtime,
    _evaluate,
    _format_telemetry,
    _make_train_many,
    _training_batches,
)

THRESHOLD_ACCURACY = 0.80


def sample_tick_count(updates: int, batch_size: int, horizon: str) -> int:
    """Return trial samples advanced through physical timesteps."""
    if updates <= 0 or batch_size <= 0 or horizon not in HORIZONS:
        raise ValueError("updates, batch_size, and horizon must be valid")
    return updates * batch_size * HORIZONS[horizon].total_steps


def matched_sample_tick_budget(
    curriculum_result: Mapping[str, object], batch_size: int
) -> dict[str, object]:
    """Derive an exact integer direct-long budget from curriculum evidence."""
    raw_phases = curriculum_result.get("phases")
    if not isinstance(raw_phases, Mapping):
        raise ValueError("curriculum result lacks phase evidence")
    phases: dict[str, object] = {}
    total = 0
    saw_gap = False
    for horizon, ceiling in CURRICULUM_PHASES:
        raw_phase = raw_phases.get(horizon)
        if raw_phase is None:
            saw_gap = True
            continue
        if saw_gap or not isinstance(raw_phase, Mapping):
            raise ValueError("curriculum phases must form a contiguous prefix")
        updates = raw_phase.get("updates_completed")
        if isinstance(updates, bool) or not isinstance(updates, int):
            raise ValueError("phase updates must be integers")
        if updates <= 0 or updates > ceiling:
            raise ValueError("phase updates fall outside their declared ceiling")
        ticks = sample_tick_count(updates, batch_size, horizon)
        phases[horizon] = {
            "updates_completed": updates,
            "horizon_steps": HORIZONS[horizon].total_steps,
            "batch_size": batch_size,
            "sample_ticks": ticks,
        }
        total += ticks
    direct_unit = batch_size * HORIZONS["long"].total_steps
    quotient, remainder = divmod(total, direct_unit)
    if total <= 0 or remainder:
        raise ValueError(
            "curriculum sample-ticks are not an integer number of long updates"
        )
    return {
        "definition": "updates * batch_size * horizon_steps",
        "curriculum_phases": phases,
        "curriculum_total_sample_ticks": total,
        "direct_long": {
            "updates": quotient,
            "horizon_steps": HORIZONS["long"].total_steps,
            "batch_size": batch_size,
            "sample_ticks_per_update": direct_unit,
            "total_sample_ticks": quotient * direct_unit,
        },
        "exact_match": quotient * direct_unit == total,
    }


def _merge_telemetry(
    destination: dict[str, list[jax.Array]], source: Mapping[str, jax.Array]
) -> None:
    for group, values in source.items():
        destination.setdefault(group, []).append(values)


def _concatenate_telemetry(
    chunks: Mapping[str, list[jax.Array]],
) -> dict[str, jax.Array]:
    if not chunks:
        raise ValueError("direct-long telemetry must be nonempty")
    return {
        group: jax.numpy.concatenate(values, axis=0) for group, values in chunks.items()
    }


def _recurrent_ratios(telemetry: Mapping[str, jax.Array]) -> np.ndarray:
    recurrent = telemetry.get("recurrent")
    return np.zeros(1) if recurrent is None else np.asarray(recurrent)[:, 5]


def _run_direct_long(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> dict[str, object]:
    runtime = _build_runtime(config, bundle)
    initial_metrics, _ = _evaluate(runtime, config, bundle)
    spikes, labels = _training_batches(config, bundle)
    trainer = _make_train_many(runtime, config)
    telemetry_chunks: dict[str, list[jax.Array]] = {}
    loss_chunks: list[jax.Array] = []
    history: list[dict[str, object]] = []
    completed = 0
    final_dynamics: dict[str, object] | None = None
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
        metrics, final_dynamics = _evaluate(runtime, config, bundle)
        stable = stability_passes(final_dynamics, _recurrent_ratios(combined))
        history.append(
            {"update": completed, "metrics": metrics, "stability_passed": stable}
        )
    if final_dynamics is None:
        raise RuntimeError("direct-long run produced no validation checkpoint")
    combined = _concatenate_telemetry(telemetry_chunks)
    return {
        "status": "completed",
        "algorithm": algorithm_label(policy_for_arm(config.arm)),
        "bundle_id": bundle.bundle_id,
        "horizon": "long",
        "initial_validation": initial_metrics,
        "history": history,
        "final_validation": history[-1]["metrics"],
        "sealed_test_metrics": None,
        "losses": np.asarray(jax.numpy.concatenate(loss_chunks)).tolist(),
        "optimizer_telemetry": _format_telemetry(combined),
        "dynamics": final_dynamics,
        "topology": topology_metrics(runtime.topology),
        "response_label_independent": response_is_label_independent(
            HORIZONS["long"], config.cue_rate_hz, config.go_rate_hz
        ),
        "config": asdict(config),
    }


def _threshold_time(
    history: object, preceding_ticks: int, sample_ticks_per_update: int
) -> int | None:
    if not isinstance(history, list):
        return None
    for checkpoint in history:
        if not isinstance(checkpoint, Mapping):
            continue
        metrics = checkpoint.get("metrics")
        update = checkpoint.get("update")
        if not isinstance(metrics, Mapping) or isinstance(update, bool):
            continue
        accuracy = metrics.get("ensemble_accuracy")
        if (
            isinstance(update, int)
            and isinstance(accuracy, (int, float))
            and checkpoint.get("stability_passed") is True
            and float(accuracy) >= THRESHOLD_ACCURACY
        ):
            return preceding_ticks + update * sample_ticks_per_update
    return None


def threshold_times(
    curriculum_result: Mapping[str, object],
    direct_result: Mapping[str, object],
    budget: Mapping[str, object],
) -> dict[str, int | None]:
    """Return first stable long-validation threshold times for both methods."""
    phases = curriculum_result.get("phases")
    phase_ticks = budget.get("curriculum_phases")
    if not isinstance(phases, Mapping) or not isinstance(phase_ticks, Mapping):
        raise ValueError("threshold evidence lacks curriculum phases")
    preceding = 0
    for horizon in ("short", "medium"):
        phase = phase_ticks.get(horizon)
        if isinstance(phase, Mapping):
            preceding += int(phase["sample_ticks"])
    long_phase = phases.get("long")
    curriculum_history = (
        long_phase.get("history") if isinstance(long_phase, Mapping) else None
    )
    direct_budget = budget.get("direct_long")
    if not isinstance(direct_budget, Mapping):
        raise ValueError("threshold evidence lacks direct-long budget")
    direct_unit = int(direct_budget["sample_ticks_per_update"])
    return {
        "curriculum_sample_ticks": _threshold_time(
            curriculum_history, preceding, direct_unit
        ),
        "direct_long_sample_ticks": _threshold_time(
            direct_result.get("history"), 0, direct_unit
        ),
    }


def _formatted_recurrent_ratios(result: Mapping[str, object]) -> np.ndarray:
    telemetry = result.get("optimizer_telemetry")
    if not isinstance(telemetry, Mapping):
        return np.asarray([])
    recurrent = telemetry.get("recurrent")
    if not isinstance(recurrent, Mapping):
        return np.asarray([])
    return np.asarray(recurrent.get("update_to_weight_ratio", []), dtype=np.float64)


def _result_stable(result: Mapping[str, object]) -> bool:
    dynamics = result.get("dynamics")
    return isinstance(dynamics, dict) and stability_passes(
        dynamics, _formatted_recurrent_ratios(result)
    )


def run_paired_curriculum_experiment(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> dict[str, object]:
    """Run curriculum and a fresh exactly sample-tick-matched direct control."""
    if config.sealed_test or not config.curriculum or config.arm != "all_pp_prop":
        raise ValueError(
            "paired adoption execution requires unsealed pp-prop curriculum"
        )
    curriculum = run_curriculum(config, bundle)
    budget = matched_sample_tick_budget(curriculum, config.batch_size)
    direct_budget = budget.get("direct_long")
    if not isinstance(direct_budget, Mapping):
        raise ValueError("matched budget lacks direct-long evidence")
    direct_updates = int(direct_budget["updates"])
    long_trace = config.curriculum_trace_half_lives.long
    direct_config = replace(
        config,
        curriculum=False,
        updates=direct_updates,
        trace_half_life_x_steps=long_trace.x,
        trace_half_life_f_steps=long_trace.f,
    )
    with brainstate.environ.context(dt=config.dt_seconds * u.second):
        direct = _run_direct_long(direct_config, bundle)
    return {
        "status": "completed",
        "bundle_id": bundle.bundle_id,
        "sealed_test_metrics": None,
        "curriculum": curriculum,
        "direct_long": direct,
        "sample_tick_budget": budget,
        "time_to_0_80": threshold_times(curriculum, direct, budget),
        "stability": {
            "curriculum": _result_stable(curriculum),
            "direct_long": _result_stable(direct),
        },
        "base_config": asdict(config),
    }
