"""Tests for fixed development gain-search configuration."""

from pathlib import Path

from temporal_benchmark_gain_search_config import (
    DEVELOPMENT_GAIN_VALUES,
    GAIN_SEARCH_STAGE,
    GainSearchSettings,
    candidate_search_settings,
    expected_gain_benchmark_config,
    gain_run_candidate,
    ordered_gain_candidates,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES


def _settings(tmp_path: Path) -> GainSearchSettings:
    return GainSearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "gain-search",
        benchmark_script=tmp_path / "17-temporal-credit-benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
        device="cpu",
    )


def test_gain_candidates_follow_declared_coordinate_search_order() -> None:
    assert DEVELOPMENT_GAIN_VALUES == (0.5, 0.8, 1.0, 1.2)
    assert [item.gain for item in ordered_gain_candidates()] == [
        0.5,
        0.8,
        1.0,
        1.2,
    ]


def test_gain_run_fixes_every_non_gain_scientific_knob(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    candidate = ordered_gain_candidates()[2]
    runner_settings = candidate_search_settings(settings, candidate)
    run_candidate = gain_run_candidate(candidate)
    config = expected_gain_benchmark_config(
        settings, candidate, DEVELOPMENT_BUNDLES[0]
    )

    assert runner_settings.search_kind == "gain"
    assert run_candidate.learning_rates.readout == 0.003
    assert run_candidate.learning_rates.feedforward == 0.001
    assert run_candidate.learning_rates.recurrent == 0.0003
    assert config.arm == "all_pp_prop"
    assert config.horizon == "long"
    assert config.updates == GAIN_SEARCH_STAGE.updates == 800
    assert config.gain == 1.0
    assert config.trace_half_life_x_steps == 60.0
    assert config.trace_half_life_f_steps == 60.0
    assert config.gradient_clip_norms.readout == 1.0
    assert config.gradient_clip_norms.feedforward == 1.0
    assert config.gradient_clip_norms.recurrent == 1.0
    assert config.recurrent_weight_decay == 0.0
    assert config.curriculum is False
    assert config.gradient_evidence is False
    assert config.sealed_test is False
