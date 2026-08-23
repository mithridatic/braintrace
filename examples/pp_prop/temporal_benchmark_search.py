"""CLI for the resumable development-only optimizer search."""

from __future__ import annotations

import argparse
import msgspec_json
import os
import pathlib
import sys
from collections.abc import Sequence

from temporal_benchmark_config import GradientClipNorms
from temporal_benchmark_search_config import SearchSettings
from temporal_benchmark_search_runner import run_development_optimizer_search


def _parser() -> argparse.ArgumentParser:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=pathlib.Path,
        default=source_root / "temp" / "temporal-credit-optimizer-search",
    )
    parser.add_argument(
        "--benchmark-script",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("17-temporal-credit-benchmark.py"),
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("temporal_benchmark_manifest.json"),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--neurons", type=int, default=96)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--trace-half-life-x-steps", type=float, default=60.0)
    parser.add_argument("--trace-half-life-f-steps", type=float, default=60.0)
    parser.add_argument("--recurrent-weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    return parser


def _settings(values: argparse.Namespace) -> SearchSettings:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    image_digest = os.environ.get("BRAINTRACE_IMAGE_DIGEST")
    source_commit = os.environ.get("BRAINTRACE_SOURCE_COMMIT")
    if not image_digest or not source_commit:
        raise RuntimeError(
            "BRAINTRACE_IMAGE_DIGEST and BRAINTRACE_SOURCE_COMMIT are required"
        )
    clip_norms = GradientClipNorms(values.clip_norm, values.clip_norm, values.clip_norm)
    return SearchSettings(
        source_root=source_root,
        output_directory=values.output_directory.resolve(),
        benchmark_script=values.benchmark_script.resolve(),
        manifest_path=values.manifest.resolve(),
        python_executable=values.python_executable,
        container_image_digest=image_digest,
        source_commit=source_commit,
        device=values.device,
        neurons=values.neurons,
        degree=values.degree,
        batch_size=values.batch_size,
        gain=values.gain,
        trace_half_life_x_steps=values.trace_half_life_x_steps,
        trace_half_life_f_steps=values.trace_half_life_f_steps,
        recurrent_weight_decay=values.recurrent_weight_decay,
        gradient_clip_norms=clip_norms,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the development-only search and print the winning configuration."""
    settings = _settings(_parser().parse_args(argv))
    if not settings.benchmark_script.is_file():
        raise FileNotFoundError(settings.benchmark_script)
    if not settings.manifest_path.is_file():
        raise FileNotFoundError(settings.manifest_path)
    winner = run_development_optimizer_search(settings)
    print(msgspec_json.dumps(winner, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
