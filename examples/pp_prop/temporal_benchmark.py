"""Command-line orchestration for the delayed-cue temporal-credit benchmark."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Sequence, cast

from sparse_benchmark_device import apply_device_selection
from temporal_benchmark_config import (
    ARMS,
    HORIZONS,
    ArmName,
    CurriculumTraceHalfLives,
    GradientClipNorms,
    HorizonName,
    LearningRates,
    TemporalBenchmarkConfig,
    TraceHalfLives,
)
from temporal_benchmark_manifest import find_bundle, load_manifest

DEFAULT_MANIFEST = pathlib.Path(__file__).with_name("temporal_benchmark_manifest.json")


def _optional_clip_norm(value: str) -> float | None:
    if value.casefold() == "disabled":
        return None
    try:
        clip_norm = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "clip norm must be a positive number or 'disabled'"
        ) from error
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise argparse.ArgumentTypeError(
            "clip norm must be a positive number or 'disabled'"
        )
    return clip_norm


def _parser() -> argparse.ArgumentParser:
    defaults = TemporalBenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bundle-id", default=defaults.bundle_id)
    parser.add_argument("--arm", choices=ARMS, default=defaults.arm)
    parser.add_argument("--horizon", choices=tuple(HORIZONS), default=defaults.horizon)
    parser.add_argument("--neurons", type=int, default=defaults.neurons)
    parser.add_argument("--degree", type=int, default=defaults.degree)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--gain", type=float, default=defaults.gain)
    parser.add_argument(
        "--trace-half-life-x-steps",
        type=float,
        default=defaults.trace_half_life_x_steps,
    )
    parser.add_argument(
        "--trace-half-life-f-steps",
        type=float,
        default=defaults.trace_half_life_f_steps,
    )
    for horizon in HORIZONS:
        half_lives = defaults.curriculum_trace_half_lives.for_horizon(horizon)
        parser.add_argument(
            f"--{horizon}-trace-half-life-x-steps",
            type=float,
            default=half_lives.x,
        )
        parser.add_argument(
            f"--{horizon}-trace-half-life-f-steps",
            type=float,
            default=half_lives.f,
        )
    parser.add_argument("--readout-learning-rate", type=float, default=3e-3)
    parser.add_argument("--feedforward-learning-rate", type=float, default=1e-3)
    parser.add_argument("--recurrent-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--clip-norm",
        type=_optional_clip_norm,
        default=1.0,
        help="baseline for every parameter group; use 'disabled' to turn off",
    )
    for group_name in ("readout", "feedforward", "recurrent"):
        parser.add_argument(
            f"--{group_name}-clip-norm",
            type=_optional_clip_norm,
            default=argparse.SUPPRESS,
            help="override the group clip norm; accepts 'disabled'",
        )
    parser.add_argument(
        "--recurrent-weight-decay", type=float, default=defaults.recurrent_weight_decay
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default=defaults.device
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=defaults.curriculum,
    )
    parser.add_argument(
        "--gradient-evidence",
        action=argparse.BooleanOptionalAction,
        default=defaults.gradient_evidence,
    )
    parser.add_argument(
        "--sealed-test",
        action=argparse.BooleanOptionalAction,
        default=defaults.sealed_test,
    )
    parser.add_argument(
        "--allow-dirty",
        action=argparse.BooleanOptionalAction,
        default=defaults.allow_dirty,
    )
    parser.add_argument("--json-output", type=pathlib.Path)
    return parser


def _config(values: argparse.Namespace) -> TemporalBenchmarkConfig:
    clip_norms = GradientClipNorms(
        readout=getattr(values, "readout_clip_norm", values.clip_norm),
        feedforward=getattr(values, "feedforward_clip_norm", values.clip_norm),
        recurrent=getattr(values, "recurrent_clip_norm", values.clip_norm),
    )
    return TemporalBenchmarkConfig(
        bundle_id=values.bundle_id,
        arm=cast(ArmName, values.arm),
        horizon=cast(HorizonName, values.horizon),
        neurons=values.neurons,
        degree=values.degree,
        batch_size=values.batch_size,
        updates=values.updates,
        gain=values.gain,
        trace_half_life_x_steps=values.trace_half_life_x_steps,
        trace_half_life_f_steps=values.trace_half_life_f_steps,
        curriculum_trace_half_lives=CurriculumTraceHalfLives(
            short=TraceHalfLives(
                values.short_trace_half_life_x_steps,
                values.short_trace_half_life_f_steps,
            ),
            medium=TraceHalfLives(
                values.medium_trace_half_life_x_steps,
                values.medium_trace_half_life_f_steps,
            ),
            long=TraceHalfLives(
                values.long_trace_half_life_x_steps,
                values.long_trace_half_life_f_steps,
            ),
        ),
        gradient_clip_norms=clip_norms,
        recurrent_weight_decay=values.recurrent_weight_decay,
        learning_rates=LearningRates(
            values.readout_learning_rate,
            values.feedforward_learning_rate,
            values.recurrent_learning_rate,
        ),
        curriculum=values.curriculum,
        gradient_evidence=values.gradient_evidence,
        sealed_test=values.sealed_test,
        allow_dirty=values.allow_dirty,
        device=values.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one paired benchmark arm and emit strict schema-versioned JSON."""
    values = _parser().parse_args(argv)
    config = _config(values)
    apply_device_selection(config.device)

    import jax

    from sparse_benchmark_device import verify_device_selection
    from temporal_benchmark_curriculum import run_curriculum
    from temporal_benchmark_gradient_evidence import run_gradient_evidence
    from temporal_benchmark_reporting import environment_fingerprint, write_result
    from temporal_benchmark_training import run_training

    verify_device_selection(config.device, jax.devices()[0].platform)
    source_root = pathlib.Path(__file__).resolve().parents[2]
    environment = environment_fingerprint(source_root)
    dirty = environment.get("source_dirty")
    if config.sealed_test and dirty is not False:
        raise RuntimeError(
            "sealed test execution requires a confirmed clean source tree"
        )
    if dirty and not config.allow_dirty:
        raise RuntimeError(
            "benchmark source is dirty; pass --allow-dirty for development only"
        )
    document = load_manifest(values.manifest)
    bundle = find_bundle(document, config.bundle_id)
    result = (
        run_curriculum(config, bundle)
        if config.curriculum
        else run_training(config, bundle)
    )
    if config.gradient_evidence:
        result["gradient_evidence"] = run_gradient_evidence(config, bundle)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": result["status"],
        "sealed_test": config.sealed_test,
        "environment": environment,
        "result": result,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if values.json_output is not None:
        write_result(values.json_output, payload)
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
