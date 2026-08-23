"""Run one paired development bundle for curriculum adoption evidence."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections.abc import Sequence

from sparse_benchmark_device import apply_device_selection
from temporal_benchmark_config import (
    CurriculumTraceHalfLives,
    GradientClipNorms,
    LearningRates,
    TemporalBenchmarkConfig,
    TraceHalfLives,
)
from temporal_benchmark_curriculum_adoption_config import (
    CURRICULUM_ADOPTION_SCHEMA_VERSION,
    CURRICULUM_BUNDLE_KIND,
    selected_config_from_benchmark,
)


def _optional_clip_norm(value: str) -> float | None:
    if value.casefold() == "disabled":
        return None
    try:
        clip_norm = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Clip norm must be positive or disabled. Set Clip norm to a positive value."
        ) from error
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise argparse.ArgumentTypeError("Clip norm must be positive or disabled. Set Clip norm to a positive value.")
    return clip_norm


def _parser() -> argparse.ArgumentParser:
    defaults = TemporalBenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("temporal_benchmark_manifest.json"),
    )
    parser.add_argument("--bundle-id", default=defaults.bundle_id)
    parser.add_argument("--neurons", type=int, default=defaults.neurons)
    parser.add_argument("--degree", type=int, default=defaults.degree)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--evaluation-interval", type=int, default=defaults.evaluation_interval
    )
    parser.add_argument("--gain", type=float, default=defaults.gain)
    parser.add_argument("--readout-learning-rate", type=float, default=0.01)
    parser.add_argument("--feedforward-learning-rate", type=float, default=0.001)
    parser.add_argument("--recurrent-learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--recurrent-weight-decay", type=float, default=defaults.recurrent_weight_decay
    )
    for horizon in ("short", "medium", "long"):
        pair = defaults.curriculum_trace_half_lives.for_horizon(horizon)
        parser.add_argument(
            f"--{horizon}-trace-half-life-x-steps", type=float, default=pair.x
        )
        parser.add_argument(
            f"--{horizon}-trace-half-life-f-steps", type=float, default=pair.f
        )
    for group in ("readout", "feedforward", "recurrent"):
        parser.add_argument(
            f"--{group}-clip-norm", type=_optional_clip_norm, default=1.0
        )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default=defaults.device
    )
    parser.add_argument("--json-output", type=pathlib.Path, required=True)
    return parser


def _config(values: argparse.Namespace) -> TemporalBenchmarkConfig:
    traces = CurriculumTraceHalfLives(
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
    )
    return TemporalBenchmarkConfig(
        bundle_id=values.bundle_id,
        arm="all_pp_prop",
        horizon="long",
        neurons=values.neurons,
        degree=values.degree,
        batch_size=values.batch_size,
        updates=800,
        evaluation_interval=values.evaluation_interval,
        gain=values.gain,
        trace_half_life_x_steps=traces.long.x,
        trace_half_life_f_steps=traces.long.f,
        curriculum_trace_half_lives=traces,
        gradient_clip_norms=GradientClipNorms(
            values.readout_clip_norm,
            values.feedforward_clip_norm,
            values.recurrent_clip_norm,
        ),
        recurrent_weight_decay=values.recurrent_weight_decay,
        learning_rates=LearningRates(
            values.readout_learning_rate,
            values.feedforward_learning_rate,
            values.recurrent_learning_rate,
        ),
        curriculum=True,
        gradient_evidence=False,
        sealed_test=False,
        allow_dirty=True,
        device=values.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one unsealed paired bundle and write a strict raw document."""
    values = _parser().parse_args(argv)
    config = _config(values)
    apply_device_selection(config.device)

    import jax

    from sparse_benchmark_device import verify_device_selection
    from temporal_benchmark_curriculum_adoption_experiment import (
        run_paired_curriculum_experiment,
    )
    from temporal_benchmark_manifest import find_bundle, load_manifest
    from temporal_benchmark_reporting import environment_fingerprint, write_result

    verify_device_selection(config.device, jax.devices()[0].platform)
    source_root = pathlib.Path(__file__).resolve().parents[2]
    environment = environment_fingerprint(source_root)
    bundle = find_bundle(load_manifest(values.manifest), config.bundle_id)
    result = run_paired_curriculum_experiment(config, bundle)
    payload: dict[str, object] = {
        "schema_version": CURRICULUM_ADOPTION_SCHEMA_VERSION,
        "kind": CURRICULUM_BUNDLE_KIND,
        "development_only": True,
        "sealed_test": False,
        "environment": environment,
        "selected_config": selected_config_from_benchmark(config),
        "result": result,
    }
    write_result(values.json_output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
