"""Tests for curriculum-adoption configuration and child commands."""

from pathlib import Path

import pytest

from temporal_benchmark_config import (
    CurriculumTraceHalfLives,
    GradientClipNorms,
    TraceHalfLives,
)
from temporal_benchmark_curriculum_adoption_config import (
    CurriculumAdoptionSettings,
    curriculum_bundle_command,
    expected_curriculum_config,
    selected_config_document,
)


def _settings(tmp_path: Path, **overrides) -> CurriculumAdoptionSettings:
    values = {
        "source_root": tmp_path,
        "output_directory": tmp_path / "out",
        "experiment_script": tmp_path / "bundle.py",
        "manifest_path": tmp_path / "manifest.json",
        "python_executable": "python-test",
        "container_image_digest": "sha256:test",
        "source_commit": "0123456789abcdef",
        "source_dirty": True,
        "example15_accuracy_change": -0.005,
    }
    values.update(overrides)
    return CurriculumAdoptionSettings(**values)


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def test_selected_configuration_and_child_command_preserve_independent_knobs(
    tmp_path: Path,
) -> None:
    traces = CurriculumTraceHalfLives(
        short=TraceHalfLives(5.0, 10.0),
        medium=TraceHalfLives(30.0, 10.0),
        long=TraceHalfLives(100.0, 30.0),
    )
    settings = _settings(
        tmp_path,
        trace_half_lives=traces,
        gradient_clip_norms=GradientClipNorms(0.5, 2.0, None),
    )
    config = expected_curriculum_config(settings, "split1-topology0-weight0")
    command = curriculum_bundle_command(
        settings, config.bundle_id, tmp_path / "raw.json"
    )

    assert config.curriculum is True
    assert config.sealed_test is False
    assert _value(command, "--medium-trace-half-life-x-steps") == "30.0"
    assert _value(command, "--long-trace-half-life-f-steps") == "30.0"
    assert _value(command, "--recurrent-clip-norm") == "disabled"
    assert selected_config_document(settings)["trace_half_lives"] == {
        "short": {"x": 5.0, "f": 10.0},
        "medium": {"x": 30.0, "f": 10.0},
        "long": {"x": 100.0, "f": 30.0},
    }


def test_adoption_settings_reject_nonfinite_static_control(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="accuracy_change"):
        _settings(tmp_path, example15_accuracy_change=float("nan"))
