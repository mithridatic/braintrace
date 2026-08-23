"""Freeze exact development selections before any sealed Example 17 run."""

from __future__ import annotations

import argparse
import msgspec_json
import pathlib
import sys
from collections.abc import Sequence

from temporal_benchmark_freeze_builder import build_frozen_selection
from temporal_benchmark_freeze_io import write_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain-winner", type=pathlib.Path, required=True)
    parser.add_argument("--optimizer-winner", type=pathlib.Path, required=True)
    parser.add_argument("--weight-decay-winner", type=pathlib.Path, required=True)
    parser.add_argument("--trace-selection", type=pathlib.Path, required=True)
    parser.add_argument("--clip-selection", type=pathlib.Path, required=True)
    parser.add_argument("--curriculum-adoption", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name(
            "temporal_benchmark_frozen_selection.json"
        ),
    )
    return parser


def _paths(values: argparse.Namespace) -> dict[str, pathlib.Path]:
    return {
        "gain": values.gain_winner.resolve(),
        "optimizer": values.optimizer_winner.resolve(),
        "weight_decay": values.weight_decay_winner.resolve(),
        "trace": values.trace_selection.resolve(),
        "clip": values.clip_selection.resolve(),
        "curriculum": values.curriculum_adoption.resolve(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs, atomically write the freeze, and print the document."""
    values = _parser().parse_args(argv)
    document = build_frozen_selection(_paths(values))
    write_artifact(values.output.resolve(), document)
    print(msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
