"""Tests for resumable development gain selection."""

from __future__ import annotations

import msgspec_json
from pathlib import Path

import pytest

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_gain_search_config import (
    GAIN_SEARCH_STAGE,
    GainSearchSettings,
    candidate_search_settings,
    gain_run_candidate,
    ordered_gain_candidates,
)
from temporal_benchmark_gain_search_runner import (
    GainCandidateScore,
    rank_gain_scores,
    run_development_gain_search,
)
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    expected_benchmark_config,
)
from temporal_benchmark_search_runner import raw_result_path
from temporal_benchmark_search_selection import ResumeConfigurationError


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


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def _raw_document(
    settings: GainSearchSettings,
    gain: float,
    bundle_id: str,
    *,
    accuracy: float,
    nll: float,
    mean_firing: float = 0.1,
) -> dict[str, object]:
    candidate = next(
        item for item in ordered_gain_candidates() if item.gain == gain
    )
    runner_settings = candidate_search_settings(settings, candidate)
    config = expected_benchmark_config(
        runner_settings,
        gain_run_candidate(candidate),
        bundle_id,
        GAIN_SEARCH_STAGE.updates,
    )
    return {
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
                "ensemble_accuracy": accuracy,
                "ensemble_nll": nll,
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


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(msgspec_json.dumps(document), encoding="utf-8")


def _score(index: int, accuracy: float, nll: float, rejected: bool = False):
    return GainCandidateScore(
        candidate=ordered_gain_candidates()[index],
        bundle_scores=(),
        rejection_reasons=("invalid",) if rejected else (),
        mean_validation_accuracy=None if rejected else accuracy,
        mean_validation_nll=None if rejected else nll,
    )


def test_gain_ranking_uses_accuracy_nll_then_gain() -> None:
    ranking = rank_gain_scores(
        (
            _score(0, 0.8, 0.1),
            _score(1, 0.9, 0.4),
            _score(2, 0.9, 0.3),
            _score(3, 0.9, 0.3),
            _score(0, 1.0, 0.0, rejected=True),
        )
    )

    assert [score.candidate.gain for score in ranking] == [1.0, 1.2, 0.8, 0.5]


def test_gain_search_is_resumable_and_emits_development_only_winner(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    calls: list[tuple[float, str]] = []
    metrics = {
        0.5: (0.75, 0.4),
        0.8: (0.90, 0.2),
        1.0: (0.90, 0.2),
        1.2: (0.85, 0.3),
    }

    def runner(command, _source_root) -> None:
        command = tuple(command)
        gain = float(_value(command, "--gain"))
        bundle_id = _value(command, "--bundle-id")
        output_path = Path(_value(command, "--json-output"))
        calls.append((gain, bundle_id))
        accuracy, nll = metrics[gain]
        _write(
            output_path,
            _raw_document(
                settings,
                gain,
                bundle_id,
                accuracy=accuracy,
                nll=nll,
            ),
        )

    winner = run_development_gain_search(
        settings, runner=runner, progress=lambda _: None
    )

    assert len(calls) == 4 * len(DEVELOPMENT_BUNDLES)
    assert winner["development_only"] is True
    assert winner["sealed_test"] is False
    assert winner["winner"]["gain"] == 0.8
    summary = msgspec_json.loads(
        (settings.output_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["candidate_gains"] == [0.5, 0.8, 1.0, 1.2]
    assert summary["valid_candidate_count"] == 4
    assert summary["sealed_test"] is False

    resumed = run_development_gain_search(
        settings,
        runner=lambda *_: (_ for _ in ()).throw(
            AssertionError("Valid raw results must be reused. Set Valid raw results to reused.")
        ),
        progress=lambda _: None,
    )
    assert all(item["reused"] for item in resumed["winner"]["bundle_scores"])


def test_gain_search_rejects_stability_invalid_bundle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def runner(command, _source_root) -> None:
        command = tuple(command)
        gain = float(_value(command, "--gain"))
        bundle_id = _value(command, "--bundle-id")
        mean_firing = (
            0.4
            if gain == 0.8 and bundle_id == DEVELOPMENT_BUNDLES[1]
            else 0.1
        )
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(
                settings,
                gain,
                bundle_id,
                accuracy=0.8,
                nll=0.3,
                mean_firing=mean_firing,
            ),
        )

    run_development_gain_search(settings, runner=runner, progress=lambda _: None)
    summary = msgspec_json.loads(
        (settings.output_directory / "summary.json").read_text(encoding="utf-8")
    )
    rejected = next(item for item in summary["candidates"] if item["gain"] == 0.8)

    assert rejected["status"] == "rejected"
    assert len(rejected["rejection_reasons"]) == 1
    assert "stability" in rejected["rejection_reasons"][0]


def test_gain_resume_refuses_config_or_provenance_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    candidate = ordered_gain_candidates()[0]
    runner_settings = candidate_search_settings(settings, candidate)
    path = raw_result_path(
        runner_settings,
        GAIN_SEARCH_STAGE,
        gain_run_candidate(candidate),
        DEVELOPMENT_BUNDLES[0],
    )
    document = _raw_document(
        settings,
        candidate.gain,
        DEVELOPMENT_BUNDLES[0],
        accuracy=0.8,
        nll=0.3,
    )
    document["environment"]["source_commit"] = "different"
    _write(path, document)

    with pytest.raises(ResumeConfigurationError, match="provenance"):
        run_development_gain_search(
            settings,
            runner=lambda *_: (_ for _ in ()).throw(
                AssertionError("Mismatched raw result must not run. Ensure Mismatched raw result does not run.")
            ),
            progress=lambda _: None,
        )
