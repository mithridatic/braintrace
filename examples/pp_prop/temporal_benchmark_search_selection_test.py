"""Tests for development evidence validation and candidate ranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    SearchSettings,
    expected_benchmark_config,
    ordered_candidates,
)
from temporal_benchmark_search_selection import (
    CandidateScore,
    RunEvidenceError,
    rank_candidate_scores,
    score_raw_document,
)


def _settings(tmp_path: Path) -> SearchSettings:
    return SearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "search",
        benchmark_script=tmp_path / "benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
    )


def _document(settings: SearchSettings, mean_firing: float = 0.1):
    candidate = ordered_candidates()[0]
    config = expected_benchmark_config(settings, candidate, DEVELOPMENT_BUNDLES[0], 100)
    return config, {
        "schema_version": 1,
        "sealed_test": False,
        "environment": {
            "container_image_digest": settings.container_image_digest,
            "source_commit": settings.source_commit,
        },
        "result": {
            "status": "completed",
            "config": config_to_dict(config),
            "final_validation": {
                "ensemble_nll": 0.4,
                "ensemble_accuracy": 0.8,
            },
            "dynamics": {
                "mean_firing_spikes_per_neuron_step": mean_firing,
                "silent_neuron_fraction": 0.1,
                "saturated_neuron_fraction": 0.0,
            },
            "optimizer_telemetry": {
                "recurrent": {"update_to_weight_ratio": [0.001, 0.002]}
            },
            "sealed_test_metrics": None,
        },
    }


def _score(index, nll, accuracy, recurrent_p99, rejected=False):
    return CandidateScore(
        candidate=ordered_candidates()[index],
        bundle_scores=(),
        rejection_reasons=("invalid",) if rejected else (),
        mean_validation_nll=None if rejected else nll,
        mean_validation_accuracy=None if rejected else accuracy,
        mean_recurrent_update_ratio_p99=None if rejected else recurrent_p99,
    )


def test_ranking_uses_nll_accuracy_ratio_then_grid_order() -> None:
    scores = (
        _score(5, 0.5, 0.9, 0.01),
        _score(1, 0.5, 0.9, 0.01),
        _score(3, 0.5, 0.9, 0.02),
        _score(4, 0.5, 0.8, 0.001),
        _score(6, 0.4, 0.1, 0.04),
        _score(0, 0.1, 1.0, 0.0, rejected=True),
    )

    ranking = rank_candidate_scores(scores)

    assert [score.candidate.grid_index for score in ranking] == [6, 1, 5, 3, 4]


def test_scoring_rejects_stability_invalid_bundle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    config, document = _document(settings, mean_firing=0.4)

    with pytest.raises(RunEvidenceError, match="stability"):
        score_raw_document(
            document,
            config,
            "raw.json",
            settings.container_image_digest,
            settings.source_commit,
            reused=False,
        )


def test_scoring_rejects_nonfinite_and_sealed_metrics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    config, document = _document(settings)
    result = document["result"]
    result["final_validation"]["ensemble_nll"] = float("nan")

    with pytest.raises(RunEvidenceError, match="(?i)non-finite"):
        score_raw_document(
            document,
            config,
            "raw.json",
            settings.container_image_digest,
            settings.source_commit,
            reused=False,
        )

    _, sealed_document = _document(settings)
    sealed_document["result"]["sealed_test_metrics"] = {"ensemble_accuracy": 1.0}
    with pytest.raises(RunEvidenceError, match="sealed test metrics"):
        score_raw_document(
            sealed_document,
            config,
            "raw.json",
            settings.container_image_digest,
            settings.source_commit,
            reused=False,
        )
