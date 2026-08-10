"""Create and compare provenance-bound Example 15 static-control evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import string
import sys
from collections.abc import Mapping, Sequence

from temporal_benchmark_example15_control_run import (
    run_fixed_example15,
    sha256_file,
)
from temporal_benchmark_example15_control_schema import (
    accept_example15_baseline,
    compare_example15_runs,
)


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: pathlib.Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in string.hexdigits for character in normalized
    ):
        raise argparse.ArgumentTypeError("SHA-256 must contain exactly 64 hex digits")
    return normalized


def _run_parser(subparsers) -> None:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser = subparsers.add_parser("run", help="execute the unchanged fixed profile")
    parser.add_argument("--source-root", type=pathlib.Path, default=source_root)
    parser.add_argument(
        "--example-script",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("15-sparse-temporal-learning.py"),
    )
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=pathlib.Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _run_parser(subparsers)
    accept = subparsers.add_parser(
        "accept-baseline", help="explicitly promote a clean passing candidate"
    )
    accept.add_argument("--candidate", type=pathlib.Path, required=True)
    accept.add_argument("--output", type=pathlib.Path, required=True)
    compare = subparsers.add_parser(
        "compare", help="compare a current artifact to a pinned accepted baseline"
    )
    compare.add_argument("--baseline", type=pathlib.Path, required=True)
    compare.add_argument("--baseline-sha256", type=_sha256, required=True)
    compare.add_argument("--current", type=pathlib.Path, required=True)
    compare.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def _run(values: argparse.Namespace) -> dict[str, object]:
    if not values.example_script.is_file():
        raise FileNotFoundError(values.example_script)
    return run_fixed_example15(
        values.example_script.resolve(),
        values.source_root.resolve(),
        values.container_image_digest,
        values.device,
    )


def _accept(values: argparse.Namespace) -> dict[str, object]:
    candidate_path = values.candidate.resolve()
    return accept_example15_baseline(
        _load_json(candidate_path), sha256_file(candidate_path)
    )


def _compare(values: argparse.Namespace) -> dict[str, object]:
    baseline_path = values.baseline.resolve()
    current_path = values.current.resolve()
    actual_baseline_hash = sha256_file(baseline_path)
    if actual_baseline_hash != values.baseline_sha256:
        raise ValueError("baseline artifact does not match the pinned SHA-256")
    return compare_example15_runs(
        _load_json(baseline_path),
        _load_json(current_path),
        actual_baseline_hash,
        sha256_file(current_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one evidence operation and atomically serialize its output."""
    values = _parser().parse_args(argv)
    operations = {"run": _run, "accept-baseline": _accept, "compare": _compare}
    document = operations[values.operation](values)
    output = values.output.resolve()
    _write_json(output, document)
    response: dict[str, object] = {"output": str(output), "sha256": sha256_file(output)}
    if values.operation == "compare":
        response["example15_accuracy_change"] = document["example15_accuracy_change"]
    print(json.dumps(response, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
