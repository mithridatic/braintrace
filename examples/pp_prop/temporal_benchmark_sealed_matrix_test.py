"""Tests for fail-closed sealed Example 17 matrix orchestration."""

from pathlib import Path

import pytest

from temporal_benchmark_config import ARMS
from temporal_benchmark_manifest import load_manifest
from temporal_benchmark_sealed_matrix import (
    SealedMatrixSettings,
    expected_result_paths,
    run_sealed_matrix,
)


def _settings(tmp_path: Path) -> SealedMatrixSettings:
    return SealedMatrixSettings(
        output_directory=tmp_path / "sealed",
        benchmark_script=tmp_path / "benchmark.py",
        manifest=Path(__file__).with_name("temporal_benchmark_manifest.json"),
        frozen_selection=tmp_path / "freeze.json",
        python_executable="python",
    )


def test_matrix_has_every_arm_and_committed_bundle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bundles = load_manifest(settings.manifest)["bundles"]
    paths = expected_result_paths(settings)

    assert len(paths) == len(ARMS) * len(bundles)
    assert len(set(paths)) == len(paths)


def test_matrix_stops_on_failed_child_without_promoting_partial(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def fail(_command, _root):
        raise RuntimeError("gpu failed")

    with pytest.raises(RuntimeError, match="gpu failed"):
        run_sealed_matrix(settings, runner=fail, analyzer=lambda _paths: {})
    assert not any(settings.output_directory.rglob("*.json"))
