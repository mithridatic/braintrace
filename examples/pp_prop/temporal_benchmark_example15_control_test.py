"""Tests for Example 15 static-control artifact operations."""

import json
from pathlib import Path

import pytest

import temporal_benchmark_example15_control as control
from temporal_benchmark_example15_control_run import sha256_file
from temporal_benchmark_example15_control_schema_test import _run_document


def _write(path: Path, document) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def test_accept_and_compare_cli_require_the_pinned_baseline_hash(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    comparison = tmp_path / "comparison.json"
    _write(candidate, _run_document())
    _write(current, _run_document(dirty=True))
    assert (
        control.main(
            [
                "accept-baseline",
                "--candidate",
                str(candidate),
                "--output",
                str(baseline),
            ]
        )
        == 0
    )

    with pytest.raises(ValueError, match="pinned"):
        control.main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--baseline-sha256",
                "0" * 64,
                "--current",
                str(current),
                "--output",
                str(comparison),
            ]
        )

    assert (
        control.main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--baseline-sha256",
                sha256_file(baseline),
                "--current",
                str(current),
                "--output",
                str(comparison),
            ]
        )
        == 0
    )
    result = json.loads(comparison.read_text(encoding="utf-8"))
    assert result["kind"] == "temporal_credit_example15_static_control"


def test_run_operation_delegates_without_changing_fixed_profile(
    monkeypatch, tmp_path: Path
) -> None:
    script = tmp_path / "15.py"
    script.write_text("fixed", encoding="utf-8")
    observed = {}

    def run(example_script, source_root, digest, device):
        observed.update(
            script=example_script,
            root=source_root,
            digest=digest,
            device=device,
        )
        return _run_document(dirty=True)

    monkeypatch.setattr(control, "run_fixed_example15", run)
    output = tmp_path / "current.json"

    assert (
        control.main(
            [
                "run",
                "--source-root",
                str(tmp_path),
                "--example-script",
                str(script),
                "--container-image-digest",
                "sha256:image",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed["digest"] == "sha256:image"
    assert observed["device"] == "cpu"
    assert output.is_file()
