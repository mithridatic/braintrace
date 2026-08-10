"""Tests for fixed recurrent-weight-decay search configuration."""

from pathlib import Path

from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_weight_decay_search_config import (
    DEVELOPMENT_WEIGHT_DECAYS,
    WEIGHT_DECAY_SEARCH_STAGE,
    WeightDecaySearchSettings,
    candidate_search_settings,
    expected_weight_decay_benchmark_config,
    ordered_weight_decay_candidates,
    weight_decay_run_candidate,
)


def _settings(tmp_path: Path) -> WeightDecaySearchSettings:
    return WeightDecaySearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "weight-decay-search",
        benchmark_script=tmp_path / "17-temporal-credit-benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
        device="cpu",
    )


def test_weight_decay_candidates_follow_declared_order() -> None:
    assert DEVELOPMENT_WEIGHT_DECAYS == (0.0, 1e-5, 1e-4)
    assert [item.weight_decay for item in ordered_weight_decay_candidates()] == [
        0.0,
        1e-5,
        1e-4,
    ]


def test_weight_decay_run_fixes_other_scientific_knobs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    candidate = ordered_weight_decay_candidates()[2]
    runner_settings = candidate_search_settings(settings, candidate)
    run_candidate = weight_decay_run_candidate(candidate)
    config = expected_weight_decay_benchmark_config(
        settings, candidate, DEVELOPMENT_BUNDLES[0]
    )

    assert runner_settings.search_kind == "weight_decay"
    assert run_candidate.learning_rates.readout == 0.01
    assert run_candidate.learning_rates.feedforward == 0.001
    assert run_candidate.learning_rates.recurrent == 0.001
    assert config.arm == "all_pp_prop"
    assert config.horizon == "long"
    assert config.updates == WEIGHT_DECAY_SEARCH_STAGE.updates == 800
    assert config.gain == 0.8
    assert config.trace_half_life_x_steps == 60.0
    assert config.trace_half_life_f_steps == 60.0
    assert config.gradient_clip_norms.readout == 1.0
    assert config.gradient_clip_norms.feedforward == 1.0
    assert config.gradient_clip_norms.recurrent == 1.0
    assert config.recurrent_weight_decay == 1e-4
    assert config.curriculum is False
    assert config.gradient_evidence is False
    assert config.sealed_test is False
