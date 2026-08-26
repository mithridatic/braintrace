"""Phase profiler for the reduced Gate A and Gate B test fixtures."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from examples.pp_prop import latent_workspace_ablation_gate as gate_c
from examples.pp_prop import latent_workspace_ablation_gate_test as gate_test
from examples.pp_prop import latent_workspace_depth_gate as gate_b


_EVENTS: Counter[str] = Counter()


def _compiler_events() -> dict[str, int]:
    return {
        name: count
        for name, count in sorted(_EVENTS.items())
        if "compil" in name or "cache" in name
    }


def _timed(label: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    before = _compiler_events()
    started = time.perf_counter()
    result = function()
    result = jax.device_get(jax.block_until_ready(result))
    after = _compiler_events()
    event_delta = {
        name: after.get(name, 0) - before.get(name, 0)
        for name in after.keys() | before.keys()
        if after.get(name, 0) != before.get(name, 0)
    }
    return result, {
        "phase": label,
        "seconds": time.perf_counter() - started,
        "compiler_event_delta": dict(sorted(event_delta.items())),
    }


def _train_and_evaluate(
    arm: str,
    model: Any,
    trainer: Any,
    events: Any,
    targets: Any,
    weights: Any,
    advances: Any,
    validation: Any,
    config: Any,
    gate: str,
) -> list[dict[str, Any]]:
    _, train_timing = _timed(
        f"{arm}.cold_train",
        lambda: trainer.train_chunk(events, targets, weights, advances),
    )
    _, evaluate_timing = _timed(
        f"{arm}.cold_evaluate",
        lambda: gate_c._evaluate_arm(model, validation, config, gate, arm),
    )
    return [train_timing, evaluate_timing]


def _profile_gate_a() -> list[dict[str, Any]]:
    timings: list[dict[str, Any]] = []
    config = gate_test._reduced_gate_c_config()
    data, timing = _timed("data", lambda: gate_c._regenerate_gate_a_data(config))
    timings.append(timing)

    models, timing = _timed(
        "models",
        lambda: {
            arm: gate_c._new_model_for_arm(
                config, "gate_a", arm, batch_size=config.gate_a_config.batch_size
            )
            for arm in ("full", "legacy")
        },
    )
    timings.append(timing)
    gate_c._copy_shared_initialization(models["full"], models["legacy"])
    trainers, timing = _timed(
        "trainers",
        lambda: {
            arm: gate_c._make_arm_trainer(models[arm], config, "gate_a", arm)
            for arm in ("full", "legacy")
        },
    )
    timings.append(timing)
    advances = np.ones(data.training_events.shape[:3], dtype=np.bool_)
    full_weights = gate_c._loss_weights(
        "gate_a", "full", efforts=np.asarray([1], dtype=np.int32)
    )
    legacy_weights = gate_c._loss_weights(
        "gate_a", "legacy", efforts=np.asarray([1], dtype=np.int32)
    )
    timings.extend(
        _train_and_evaluate(
            "full", models["full"], trainers["full"], data.training_events,
            data.training_targets, full_weights, advances, data, config, "gate_a"
        )
    )
    timings.extend(
        _train_and_evaluate(
            "legacy", models["legacy"], trainers["legacy"], data.training_events,
            data.training_targets, legacy_weights, advances, data, config, "gate_a"
        )
    )
    return timings


def _profile_gate_b() -> list[dict[str, Any]]:
    timings: list[dict[str, Any]] = []
    config = gate_test._reduced_gate_c_config()
    regenerated, timing = _timed("data", lambda: gate_c._regenerate_gate_b_data(config))
    timings.append(timing)
    schedule, validation = regenerated
    schedule_chunk = next(gate_b._iter_schedule_chunks(schedule, config.gate_b_config))
    chunk = gate_b._encode_training_chunk(schedule_chunk, config.gate_b_config)
    arms = gate_test._ARMS
    models, timing = _timed(
        "models",
        lambda: {
            arm: gate_c._new_model_for_arm(
                config, "gate_b", arm, batch_size=config.gate_b_config.batch_size
            )
            for arm in arms
        },
    )
    timings.append(timing)
    gate_c._copy_shared_initialization(models["full"], models["legacy"])
    trainers, timing = _timed(
        "trainers",
        lambda: {
            arm: gate_c._make_arm_trainer(models[arm], config, "gate_b", arm)
            for arm in arms
        },
    )
    timings.append(timing)
    def profile_arm(arm: str) -> None:
        timings.extend(
            _train_and_evaluate(
                arm,
                models[arm],
                trainers[arm],
                chunk.events,
                chunk.targets,
                gate_c._loss_weights("gate_b", arm, efforts=chunk.efforts),
                chunk.advance_masks,
                validation,
                config,
                "gate_b",
            )
        )

    profile_arm("full")
    profile_arm("query_only")
    profile_arm("terminal_only")
    profile_arm("legacy")
    profile_arm("frozen_write")
    return timings


def main() -> None:
    """Run one reduced fixture profile and emit one JSON record."""

    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("a", "b"))
    args = parser.parse_args()
    def count_event(name: str, **metadata: Any) -> None:
        del metadata
        _EVENTS[name] += 1

    jax.monitoring.register_event_listener(count_event)
    started = time.perf_counter()
    timings = _profile_gate_a() if args.gate == "a" else _profile_gate_b()
    print(
        json.dumps(
            {
                "gate": args.gate,
                "total_seconds": time.perf_counter() - started,
                "timings": timings,
                "compile_events": _compiler_events(),
            }
        )
    )


if __name__ == "__main__":
    main()
