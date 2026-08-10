"""Run the fixed development curriculum adoption comparison."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Sequence

from temporal_benchmark_config import (
    CurriculumTraceHalfLives,
    GradientClipNorms,
    LearningRates,
    TraceHalfLives,
)
from temporal_benchmark_curriculum_adoption_config import CurriculumAdoptionSettings
from temporal_benchmark_curriculum_adoption_runner import (
    run_development_curriculum_adoption,
)
from temporal_benchmark_example15_control_schema import validated_accuracy_change


def _optional_clip_norm(value: str) -> float | None:
    if value.casefold() == "disabled":
        return None
    try:
        clip_norm = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "clip norm must be positive or disabled"
        ) from error
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise argparse.ArgumentTypeError("clip norm must be positive or disabled")
    return clip_norm


def _parser() -> argparse.ArgumentParser:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=pathlib.Path,
        default=source_root / "temp" / "temporal-credit-curriculum-adoption",
    )
    parser.add_argument(
        "--experiment-script",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name(
            "temporal_benchmark_curriculum_adoption_bundle.py"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("temporal_benchmark_manifest.json"),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-dirty", choices=("true", "false"), required=True)
    static_control = parser.add_mutually_exclusive_group(required=True)
    static_control.add_argument(
        "--example15-static-control-result",
        type=pathlib.Path,
        help="preferred validated temporal_credit_example15_static_control artifact",
    )
    static_control.add_argument(
        "--example15-accuracy-change",
        type=float,
        help="development-only raw diagnostic override",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--neurons", type=int, default=96)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--readout-learning-rate", type=float, default=0.01)
    parser.add_argument("--feedforward-learning-rate", type=float, default=0.001)
    parser.add_argument("--recurrent-learning-rate", type=float, default=0.001)
    parser.add_argument("--recurrent-weight-decay", type=float, default=0.0)
    defaults = CurriculumTraceHalfLives()
    for horizon in ("short", "medium", "long"):
        pair = defaults.for_horizon(horizon)
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
    return parser


def _settings(values: argparse.Namespace) -> CurriculumAdoptionSettings:
    source_root = pathlib.Path(__file__).resolve().parents[2]
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
    accuracy_change = values.example15_accuracy_change
    if values.example15_static_control_result is not None:
        control_path = values.example15_static_control_result.resolve()
        document = json.loads(control_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Example 15 static-control artifact must be an object")
        accuracy_change = validated_accuracy_change(document)
    assert accuracy_change is not None
    return CurriculumAdoptionSettings(
        source_root=source_root,
        output_directory=values.output_directory.resolve(),
        experiment_script=values.experiment_script.resolve(),
        manifest_path=values.manifest.resolve(),
        python_executable=values.python_executable,
        container_image_digest=values.container_image_digest,
        source_commit=values.source_commit,
        source_dirty=values.source_dirty == "true",
        example15_accuracy_change=accuracy_change,
        device=values.device,
        neurons=values.neurons,
        degree=values.degree,
        batch_size=values.batch_size,
        evaluation_interval=values.evaluation_interval,
        gain=values.gain,
        learning_rates=LearningRates(
            values.readout_learning_rate,
            values.feedforward_learning_rate,
            values.recurrent_learning_rate,
        ),
        recurrent_weight_decay=values.recurrent_weight_decay,
        gradient_clip_norms=GradientClipNorms(
            values.readout_clip_norm,
            values.feedforward_clip_norm,
            values.recurrent_clip_norm,
        ),
        trace_half_lives=traces,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run all fixed development bundles and print the adoption document."""
    settings = _settings(_parser().parse_args(argv))
    if not settings.experiment_script.is_file():
        raise FileNotFoundError(settings.experiment_script)
    if not settings.manifest_path.is_file():
        raise FileNotFoundError(settings.manifest_path)
    decision = run_development_curriculum_adoption(settings)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
