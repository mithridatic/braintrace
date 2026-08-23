"""Derive Example 17's clip-selection artifact from weight-decay evidence."""

from __future__ import annotations

import argparse
import msgspec_json
import pathlib
import sys
from collections.abc import Sequence

from temporal_benchmark_clip_selection_builder import build_clip_selection
from temporal_benchmark_freeze_io import write_artifact


def _parser() -> argparse.ArgumentParser:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-decay-winner", type=pathlib.Path, required=True)
    parser.add_argument("--clip-search-evidence", type=pathlib.Path)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            source_root / "temp" / "temporal-credit-clip-selection" / "selection.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate source evidence and atomically write the clip decision."""
    values = _parser().parse_args(argv)
    document = build_clip_selection(
        values.weight_decay_winner, values.clip_search_evidence
    )
    write_artifact(values.output.resolve(), document)
    print(msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
