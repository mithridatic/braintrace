"""Tests for optimizer-search subprocess isolation, resume, and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    SEARCH_STAGES,
    LearningRateCandidate,
    SearchSettings,
    expected_benchmark_config,
    ordered_candidates,
)
from temporal_benchmark_search_runner import (
    evaluate_candidate,
    failure_result_path,
    load_raw_document,
    obtain_bundle_score,
    partial_result_path,
    raw_result_path,
    run_development_optimizer_search,
)
from temporal_benchmark_search_selection import (
    ResumeConfigurationError,
    RunEvidenceError,
)


def _settings(tmp_path: Path) -> SearchSettings:
    return SearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "search",
        benchmark_script=tmp_path / "17-temporal-credit-benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
    )


def _raw_document(
    settings: SearchSettings,
    candidate: LearningRateCandidate,
    bundle_id: str,
    updates: int,
    *,
    nll: float = 0.4,
    accuracy: float = 0.8,
    mean_firing: float = 0.1,
) -> dict[str, object]:
    config = expected_benchmark_config(settings, candidate, bundle_id, updates)
    return {
        "schema_version": 1,
        "sealed_test": False,
        "environment": {
            "source_dirty": True,
            "container_image_digest": settings.container_image_digest,
            "source_commit": settings.source_commit,
        },
        "result": {
            "status": "completed",
            "config": config_to_dict(config),
            "final_validation": {
                "ensemble_nll": nll,
                "ensemble_accuracy": accuracy,
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
    path.write_text(json.dumps(document), encoding="utf-8")


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def test_candidate_rejects_one_stability_invalid_bundle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    candidate = ordered_candidates()[0]
    stage = SEARCH_STAGES[0]

    def runner(command, _source_root) -> None:
        command = tuple(command)
        bundle_id = _value(command, "--bundle-id")
        path = Path(_value(command, "--json-output"))
        mean_firing = 0.4 if bundle_id == DEVELOPMENT_BUNDLES[1] else 0.1
        _write(
            path,
            _raw_document(
                settings,
                candidate,
                bundle_id,
                stage.updates,
                mean_firing=mean_firing,
            ),
        )

    score = evaluate_candidate(settings, stage, candidate, runner, lambda _: None)

    assert score.accepted is False
    assert len(score.bundle_scores) == 2
    assert len(score.rejection_reasons) == 1
    assert "stability" in score.rejection_reasons[0]
    invalid_final = raw_result_path(settings, stage, candidate, DEVELOPMENT_BUNDLES[1])
    assert invalid_final.is_file()
    assert not partial_result_path(
        settings, stage, candidate, DEVELOPMENT_BUNDLES[1]
    ).exists()


def test_strict_loader_rejects_nonfinite_json_constant(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"metric": NaN}', encoding="utf-8")

    with pytest.raises(RunEvidenceError, match="non-finite"):
        load_raw_document(path)


def test_stale_partial_is_overwritten_then_atomically_promoted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stage = SEARCH_STAGES[0]
    candidate = ordered_candidates()[0]
    bundle_id = DEVELOPMENT_BUNDLES[0]
    partial = partial_result_path(settings, stage, candidate, bundle_id)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text('{"interrupted":', encoding="utf-8")

    def runner(command, _source_root) -> None:
        command = tuple(command)
        output = Path(_value(command, "--json-output"))
        assert output == partial
        assert not output.exists()
        _write(
            output,
            _raw_document(settings, candidate, bundle_id, stage.updates),
        )

    score = obtain_bundle_score(settings, stage, candidate, bundle_id, runner)

    assert score.reused is False
    assert raw_result_path(settings, stage, candidate, bundle_id).is_file()
    assert not partial.exists()


def test_resume_refuses_any_complete_config_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stage = SEARCH_STAGES[0]
    candidate = ordered_candidates()[0]
    bundle_id = DEVELOPMENT_BUNDLES[0]
    path = raw_result_path(settings, stage, candidate, bundle_id)
    document = _raw_document(settings, candidate, bundle_id, stage.updates)
    result = document["result"]
    assert isinstance(result, dict)
    config = result["config"]
    assert isinstance(config, dict)
    config["gain"] = 1.2
    _write(path, document)

    def forbidden_runner(_command, _source_root) -> None:
        raise AssertionError("an existing file must be validated, not rerun")

    with pytest.raises(ResumeConfigurationError, match="exactly match"):
        obtain_bundle_score(settings, stage, candidate, bundle_id, forbidden_runner)


def test_resume_refuses_image_or_source_provenance_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stage = SEARCH_STAGES[0]
    candidate = ordered_candidates()[0]
    bundle_id = DEVELOPMENT_BUNDLES[0]
    path = raw_result_path(settings, stage, candidate, bundle_id)
    document = _raw_document(settings, candidate, bundle_id, stage.updates)
    environment = document["environment"]
    assert isinstance(environment, dict)
    environment["container_image_digest"] = "sha256:different-image"
    _write(path, document)

    with pytest.raises(ResumeConfigurationError, match="provenance"):
        obtain_bundle_score(
            settings,
            stage,
            candidate,
            bundle_id,
            lambda *_: (_ for _ in ()).throw(AssertionError("must reuse")),
        )


def test_failed_child_is_persisted_rejected_and_other_bundles_continue(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    stage = SEARCH_STAGES[0]
    candidate = ordered_candidates()[0]
    calls: list[str] = []

    def runner(command, _source_root) -> None:
        command = tuple(command)
        bundle_id = _value(command, "--bundle-id")
        calls.append(bundle_id)
        if bundle_id == DEVELOPMENT_BUNDLES[0]:
            raise RuntimeError("non-finite child result")
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(settings, candidate, bundle_id, stage.updates),
        )

    score = evaluate_candidate(settings, stage, candidate, runner, lambda _: None)

    assert calls == list(DEVELOPMENT_BUNDLES)
    assert score.accepted is False
    assert "child run failed" in score.rejection_reasons[0]
    failure_path = failure_result_path(
        settings, stage, candidate, DEVELOPMENT_BUNDLES[0]
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["kind"] == "temporal_credit_optimizer_search_child_failure"
    assert failure["provenance"] == {
        "container_image_digest": settings.container_image_digest,
        "source_commit": settings.source_commit,
    }


def test_successive_halving_writes_summaries_and_resumes_raw_runs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    candidates_by_rates = {
        (candidate.readout, candidate.feedforward, candidate.recurrent): candidate
        for candidate in ordered_candidates()
    }
    calls: list[tuple[int, str, int]] = []

    def runner(command, _source_root) -> None:
        command = tuple(command)
        rates = (
            float(_value(command, "--readout-learning-rate")),
            float(_value(command, "--feedforward-learning-rate")),
            float(_value(command, "--recurrent-learning-rate")),
        )
        candidate = candidates_by_rates[rates]
        bundle_id = _value(command, "--bundle-id")
        updates = int(_value(command, "--updates"))
        output = Path(_value(command, "--json-output"))
        calls.append((candidate.grid_index, bundle_id, updates))
        _write(
            output,
            _raw_document(
                settings,
                candidate,
                bundle_id,
                updates,
                nll=0.1 + candidate.grid_index / 100.0,
                accuracy=0.9 - candidate.grid_index / 1000.0,
            ),
        )

    winner = run_development_optimizer_search(
        settings, runner=runner, progress=lambda _: None
    )

    assert len(calls) == (27 + 9 + 3) * len(DEVELOPMENT_BUNDLES)
    assert [call[2] for call in calls if call[0] == 0] == [
        100,
        100,
        100,
        300,
        300,
        300,
        800,
        800,
        800,
    ]
    winner_record = winner["winner"]
    assert isinstance(winner_record, dict)
    assert winner_record["grid_index"] == 0
    for stage, expected_count, expected_promotions in zip(
        SEARCH_STAGES, (27, 9, 3), (9, 3, 1)
    ):
        summary_path = settings.output_directory / (
            f"stage-{stage.number:02d}-summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["candidate_count"] == expected_count
        assert len(summary["promoted_grid_indices"]) == expected_promotions
        assert summary["development_only"] is True
        assert summary["sealed_test"] is False

    def forbidden_runner(_command, _source_root) -> None:
        raise AssertionError("valid resumed raw files must not launch subprocesses")

    resumed = run_development_optimizer_search(
        settings, runner=forbidden_runner, progress=lambda _: None
    )
    resumed_winner = resumed["winner"]
    assert isinstance(resumed_winner, dict)
    assert resumed_winner["grid_index"] == 0
    resumed_bundles = resumed_winner["bundle_scores"]
    assert isinstance(resumed_bundles, list)
    assert all(bundle["reused"] is True for bundle in resumed_bundles)
