"""Thin CLI for the fixed development recurrent-weight-decay search."""

from __future__ import annotations

import argparse
import msgspec_json
import pathlib
import sys
from collections.abc import Sequence

from temporal_benchmark_weight_decay_search_config import WeightDecaySearchSettings
from temporal_benchmark_weight_decay_search_runner import (
    run_development_weight_decay_search,
)


def _parser() -> argparse.ArgumentParser:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=pathlib.Path,
        default=source_root / "temp" / "temporal-credit-weight-decay-search",
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
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--neurons", type=int, default=96)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def _settings(values: argparse.Namespace) -> WeightDecaySearchSettings:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    return WeightDecaySearchSettings(
        source_root=source_root,
        output_directory=values.output_directory.resolve(),
        benchmark_script=values.benchmark_script.resolve(),
        manifest_path=values.manifest.resolve(),
        python_executable=values.python_executable,
        container_image_digest=values.container_image_digest,
        source_commit=values.source_commit,
        device=values.device,
        neurons=values.neurons,
        degree=values.degree,
        batch_size=values.batch_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed development decay search and print its winner document."""
    settings = _settings(_parser().parse_args(argv))
    if not settings.benchmark_script.is_file():
        raise FileNotFoundError(settings.benchmark_script)
    if not settings.manifest_path.is_file():
        raise FileNotFoundError(settings.manifest_path)
    winner = run_development_weight_decay_search(settings)
    print(msgspec_json.dumps(winner, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
