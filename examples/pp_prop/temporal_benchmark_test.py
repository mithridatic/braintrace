"""Tests for Example 17 command-line configuration."""

import pathlib
from unittest.mock import patch

import pytest

from temporal_benchmark import _config, _parser, _sealed_overrides, _startup_device
from temporal_benchmark_freeze_io import FreezeArtifactError


def test_cli_exposes_only_half_life_trace_knobs() -> None:
    parser = _parser()
    destinations = {action.dest for action in parser._actions}

    assert "trace_half_life_x_steps" in destinations
    assert "trace_half_life_f_steps" in destinations
    for horizon in ("short", "medium", "long"):
        assert f"{horizon}_trace_half_life_x_steps" in destinations
        assert f"{horizon}_trace_half_life_f_steps" in destinations
    assert "rank" not in destinations
    assert "decay" not in destinations


def test_cli_builds_independent_curriculum_trace_pairs() -> None:
    values = _parser().parse_args(
        [
            "--short-trace-half-life-x-steps",
            "5",
            "--short-trace-half-life-f-steps",
            "10",
            "--medium-trace-half-life-x-steps",
            "30",
            "--medium-trace-half-life-f-steps",
            "10",
            "--long-trace-half-life-x-steps",
            "100",
            "--long-trace-half-life-f-steps",
            "30",
        ]
    )

    half_lives = _config(values).curriculum_trace_half_lives

    assert (half_lives.short.x, half_lives.short.f) == (5.0, 10.0)
    assert (half_lives.medium.x, half_lives.medium.f) == (30.0, 10.0)
    assert (half_lives.long.x, half_lives.long.f) == (100.0, 30.0)


def test_direct_trace_flags_keep_their_non_curriculum_semantics() -> None:
    config = _config(
        _parser().parse_args(
            [
                "--trace-half-life-x-steps",
                "7",
                "--trace-half-life-f-steps",
                "11",
            ]
        )
    )

    assert config.trace_half_life_x_steps == 7.0
    assert config.trace_half_life_f_steps == 11.0
    assert (
        config.curriculum_trace_half_lives.long.x,
        config.curriculum_trace_half_lives.long.f,
    ) == (60.0, 60.0)


def test_cli_builds_requested_paired_arm() -> None:
    values = _parser().parse_args(
        [
            "--arm",
            "all_bptt",
            "--horizon",
            "short",
            "--device",
            "cpu",
            "--updates",
            "2",
            "--curriculum",
        ]
    )

    config = _config(values)

    assert config.arm == "all_bptt"
    assert config.horizon == "short"
    assert config.device == "cpu"
    assert config.updates == 2
    assert config.curriculum is True


def test_cli_defaults_each_parameter_group_clip_norm_to_one() -> None:
    config = _config(_parser().parse_args([]))

    assert config.gradient_clip_norms.readout == 1.0
    assert config.gradient_clip_norms.feedforward == 1.0
    assert config.gradient_clip_norms.recurrent == 1.0


def test_cli_parses_disabled_group_clip_norm() -> None:
    values = _parser().parse_args(
        [
            "--clip-norm",
            "2.0",
            "--feedforward-clip-norm",
            "disabled",
            "--recurrent-clip-norm",
            "0.25",
        ]
    )

    clip_norms = _config(values).gradient_clip_norms

    assert clip_norms.readout == 2.0
    assert clip_norms.feedforward is None
    assert clip_norms.recurrent == 0.25


def test_legacy_cli_clip_norm_applies_to_every_parameter_group() -> None:
    config = _config(_parser().parse_args(["--clip-norm", "disabled"]))

    assert config.gradient_clip_norms.readout is None
    assert config.gradient_clip_norms.feedforward is None
    assert config.gradient_clip_norms.recurrent is None


def test_sealed_cli_requires_frozen_selection() -> None:
    values = _parser().parse_args(["--sealed-test"])
    with pytest.raises(FreezeArtifactError, match="frozen selection"):
        _sealed_overrides(values, {"source_commit": "a" * 40})


def test_sealed_cli_rejects_device_before_backend_initialization() -> None:
    values = _parser().parse_args(["--sealed-test", "--device", "cpu"])
    with pytest.raises(FreezeArtifactError, match="device gpu"):
        _startup_device(values)


def test_sealed_cli_rejects_freeze_provenance_mismatch(tmp_path: pathlib.Path) -> None:
    frozen = tmp_path / "freeze.json"
    frozen.write_text("{}", encoding="utf-8")
    values = _parser().parse_args(
        ["--sealed-test", "--frozen-selection", str(frozen)]
    )
    document = {
        "selection_provenance": {
            "source_commit": "b" * 40,
            "selection_source_dirty": False,
            "construction": {
                "device": "gpu", "neurons": 96, "degree": 8, "batch_size": 32
            },
            "input_artifacts": {},
        }
    }
    with (
        patch("temporal_benchmark.load_artifact", return_value=document),
        patch("temporal_benchmark.validate_frozen_selection", return_value={}),
    ):
        with pytest.raises(FreezeArtifactError, match="source commit"):
            _sealed_overrides(values, {"source_commit": "a" * 40})
