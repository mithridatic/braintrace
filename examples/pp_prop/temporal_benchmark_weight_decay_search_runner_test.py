"""Tests for resumable recurrent-weight-decay selection."""

from __future__ import annotations

import msgspec_json
from pathlib import Path

import pytest

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    expected_benchmark_config,
)
from temporal_benchmark_search_runner import raw_result_path
from temporal_benchmark_search_selection import ResumeConfigurationError
from temporal_benchmark_weight_decay_search_config import (
    WEIGHT_DECAY_SEARCH_STAGE,
    WeightDecaySearchSettings,
    candidate_search_settings,
    ordered_weight_decay_candidates,
    weight_decay_run_candidate,
)
from temporal_benchmark_weight_decay_search_runner import (
    WeightDecayCandidateScore,
    rank_weight_decay_scores,
    run_development_weight_decay_search,
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


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def _raw_document(
    settings: WeightDecaySearchSettings,
    weight_decay: float,
    bundle_id: str,
    *,
    nll: float,
    accuracy: float,
    recurrent_ratio: float,
    mean_firing: float = 0.1,
) -> dict[str, object]:
    candidate = next(
        item
        for item in ordered_weight_decay_candidates()
        if item.weight_decay == weight_decay
    )
    runner_settings = candidate_search_settings(settings, candidate)
    config = expected_benchmark_config(
        runner_settings,
        weight_decay_run_candidate(candidate),
        bundle_id,
        WEIGHT_DECAY_SEARCH_STAGE.updates,
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
                "ensemble_nll": nll,
                "ensemble_accuracy": accuracy,
            },
            "dynamics": {
                "mean_firing_spikes_per_neuron_step": mean_firing,
                "silent_neuron_fraction": 0.1,
                "saturated_neuron_fraction": 0.0,
            },
            "optimizer_telemetry": {
                "recurrent": {
                    "update_to_weight_ratio": [recurrent_ratio, recurrent_ratio]
                }
            },
            "sealed_test_metrics": None,
        },
    }


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(msgspec_json.dumps(document), encoding="utf-8")


def _score(index: int, nll: float, accuracy: float, ratio: float):
    return WeightDecayCandidateScore(
        candidate=ordered_weight_decay_candidates()[index],
        bundle_scores=(),
        rejection_reasons=(),
        mean_validation_nll=nll,
        mean_validation_accuracy=accuracy,
        mean_recurrent_update_ratio_p99=ratio,
    )


def test_weight_decay_ranking_uses_all_declared_tie_breaks() -> None:
    lowest_nll = _score(0, 0.2, 0.1, 0.04)
    highest_accuracy = _score(1, 0.3, 0.9, 0.03)
    lower_ratio = _score(2, 0.3, 0.9, 0.02)
    smaller_decay = _score(0, 0.3, 0.9, 0.02)
    rejected = WeightDecayCandidateScore(
        ordered_weight_decay_candidates()[0],
        (),
        ("invalid",),
        None,
        None,
        None,
    )

    ranking = rank_weight_decay_scores(
        (highest_accuracy, rejected, lower_ratio, lowest_nll, smaller_decay)
    )

    assert ranking == (lowest_nll, smaller_decay, lower_ratio, highest_accuracy)


def test_weight_decay_search_resumes_and_emits_unsealed_winner(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    calls: list[tuple[float, str]] = []
    metrics = {
        0.0: (0.30, 0.80, 0.010),
        1e-5: (0.20, 0.90, 0.008),
        1e-4: (0.25, 0.95, 0.006),
    }

    def runner(command, _source_root) -> None:
        command = tuple(command)
        decay = float(_value(command, "--recurrent-weight-decay"))
        bundle_id = _value(command, "--bundle-id")
        calls.append((decay, bundle_id))
        nll, accuracy, ratio = metrics[decay]
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(
                settings,
                decay,
                bundle_id,
                nll=nll,
                accuracy=accuracy,
                recurrent_ratio=ratio,
            ),
        )

    winner = run_development_weight_decay_search(
        settings, runner=runner, progress=lambda _: None
    )

    assert len(calls) == 3 * len(DEVELOPMENT_BUNDLES)
    assert winner["development_only"] is True
    assert winner["sealed_test"] is False
    assert winner["winner"]["recurrent_weight_decay"] == 1e-5
    summary = msgspec_json.loads(
        (settings.output_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["candidate_weight_decays"] == [0.0, 1e-5, 1e-4]
    assert summary["valid_candidate_count"] == 3

    resumed = run_development_weight_decay_search(
        settings,
        runner=lambda *_: (_ for _ in ()).throw(
            AssertionError("Valid raw results must be reused. Set Valid raw results to reused.")
        ),
        progress=lambda _: None,
    )
    assert all(item["reused"] for item in resumed["winner"]["bundle_scores"])


def test_weight_decay_search_rejects_unstable_and_nonfinite_bundles(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    def runner(command, _source_root) -> None:
        command = tuple(command)
        decay = float(_value(command, "--recurrent-weight-decay"))
        bundle_id = _value(command, "--bundle-id")
        mean_firing = (
            0.4
            if decay == 1e-5 and bundle_id == DEVELOPMENT_BUNDLES[1]
            else 0.1
        )
        nll = (
            float("nan")
            if decay == 1e-4 and bundle_id == DEVELOPMENT_BUNDLES[2]
            else 0.3
        )
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(
                settings,
                decay,
                bundle_id,
                nll=nll,
                accuracy=0.8,
                recurrent_ratio=0.01,
                mean_firing=mean_firing,
            ),
        )

    run_development_weight_decay_search(
        settings, runner=runner, progress=lambda _: None
    )
    summary = msgspec_json.loads(
        (settings.output_directory / "summary.json").read_text(encoding="utf-8")
    )
    rejected = next(
        item
        for item in summary["candidates"]
        if item["recurrent_weight_decay"] == 1e-5
    )

    assert rejected["status"] == "rejected"
    assert "stability" in rejected["rejection_reasons"][0]
    nonfinite = next(
        item
        for item in summary["candidates"]
        if item["recurrent_weight_decay"] == 1e-4
    )
    assert nonfinite["status"] == "rejected"
    assert "non-finite" in nonfinite["rejection_reasons"][0]


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("config", "exactly match"),
        ("image", "provenance"),
        ("commit", "provenance"),
    ),
)
def test_weight_decay_resume_refuses_config_or_provenance_drift(
    tmp_path: Path, drift: str, message: str
) -> None:
    settings = _settings(tmp_path)
    candidate = ordered_weight_decay_candidates()[0]
    runner_settings = candidate_search_settings(settings, candidate)
    path = raw_result_path(
        runner_settings,
        WEIGHT_DECAY_SEARCH_STAGE,
        weight_decay_run_candidate(candidate),
        DEVELOPMENT_BUNDLES[0],
    )
    document = _raw_document(
        settings,
        candidate.weight_decay,
        DEVELOPMENT_BUNDLES[0],
        nll=0.3,
        accuracy=0.8,
        recurrent_ratio=0.01,
    )
    if drift == "config":
        document["result"]["config"]["gain"] = 1.2
    elif drift == "image":
        document["environment"]["container_image_digest"] = "sha256:different"
    else:
        document["environment"]["source_commit"] = "different"
    _write(path, document)

    with pytest.raises(ResumeConfigurationError, match=message):
        run_development_weight_decay_search(
            settings,
            runner=lambda *_: (_ for _ in ()).throw(
                AssertionError("Mismatched final must not launch a child. Ensure Mismatched final does not launch a child.")
            ),
            progress=lambda _: None,
        )
