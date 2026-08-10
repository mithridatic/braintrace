"""Tests for trace coordinate ranking, resume, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_benchmark_config import (
    GradientClipNorms,
    LearningRates,
    TemporalBenchmarkConfig,
    config_to_dict,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_search_selection import BundleScore, ResumeConfigurationError
from temporal_benchmark_trace_search_config import (
    HORIZON_TRACE_GRIDS,
    TraceCandidate,
    TraceSearchSettings,
    coordinate_candidates,
    expected_trace_benchmark_config,
)
from temporal_benchmark_trace_search_runner import (
    TraceCandidateScore,
    obtain_trace_bundle_score,
    rank_trace_scores,
    raw_trace_result_path,
    run_development_trace_search,
)


def _settings(tmp_path: Path) -> TraceSearchSettings:
    return TraceSearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "trace-search",
        benchmark_script=tmp_path / "17-temporal-credit-benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
        device="cpu",
    )


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def _config_from_command(command: tuple[str, ...]) -> TemporalBenchmarkConfig:
    return TemporalBenchmarkConfig(
        bundle_id=_value(command, "--bundle-id"),
        arm="all_pp_prop",
        horizon=_value(command, "--horizon"),
        neurons=int(_value(command, "--neurons")),
        degree=int(_value(command, "--degree")),
        batch_size=int(_value(command, "--batch-size")),
        updates=int(_value(command, "--updates")),
        gain=float(_value(command, "--gain")),
        trace_half_life_x_steps=float(
            _value(command, "--trace-half-life-x-steps")
        ),
        trace_half_life_f_steps=float(
            _value(command, "--trace-half-life-f-steps")
        ),
        gradient_clip_norms=GradientClipNorms(1.0, 1.0, 1.0),
        recurrent_weight_decay=float(_value(command, "--recurrent-weight-decay")),
        learning_rates=LearningRates(
            float(_value(command, "--readout-learning-rate")),
            float(_value(command, "--feedforward-learning-rate")),
            float(_value(command, "--recurrent-learning-rate")),
        ),
        allow_dirty=True,
        device="cpu",
    )


def _raw_document(
    settings: TraceSearchSettings,
    config: TemporalBenchmarkConfig,
    nll: float,
    accuracy: float = 0.8,
    mean_firing: float = 0.1,
) -> dict[str, object]:
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
                "recurrent": {"update_to_weight_ratio": [0.001, 0.002]}
            },
            "sealed_test_metrics": None,
        },
    }


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _score(
    half_life: float,
    nll: float,
    accuracy: float,
    ratio: float,
    rejected: bool = False,
) -> TraceCandidateScore:
    bundle = BundleScore("bundle", nll, accuracy, ratio, "raw.json", False)
    return TraceCandidateScore(
        TraceCandidate("x", 0, half_life, 10.0),
        () if rejected else (bundle,),
        ("invalid",) if rejected else (),
        None if rejected else nll,
        None if rejected else accuracy,
        None if rejected else ratio,
    )


def test_ranking_uses_nll_accuracy_ratio_then_lower_half_life() -> None:
    ranking = rank_trace_scores(
        (
            _score(5, 0.2, 0.9, 0.01),
            _score(10, 0.1, 0.8, 0.02),
            _score(20, 0.1, 0.9, 0.02),
            _score(30, 0.1, 0.9, 0.01),
            _score(40, 0.1, 0.9, 0.01),
            _score(1, 0.0, 1.0, 0.0, rejected=True),
        )
    )

    assert [item.candidate.half_life for item in ranking] == [30, 40, 20, 10, 5]


def test_full_coordinate_search_selects_pairs_and_resumes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    targets = {"short": (10.0, 5.0), "medium": (20.0, 30.0), "long": (60.0, 100.0)}
    calls: list[tuple[str, float, float, str]] = []

    def runner(command, _source_root) -> None:
        command = tuple(command)
        config = _config_from_command(command)
        target_x, target_f = targets[config.horizon]
        nll = abs(config.trace_half_life_x_steps - target_x) + abs(
            config.trace_half_life_f_steps - target_f
        )
        output = Path(_value(command, "--json-output"))
        calls.append(
            (
                config.horizon,
                config.trace_half_life_x_steps,
                config.trace_half_life_f_steps,
                config.bundle_id,
            )
        )
        _write(output, _raw_document(settings, config, nll))

    selection = run_development_trace_search(
        settings, runner=runner, progress=lambda _: None
    )

    # The X/F intersection is the same exact configuration and is safely reused.
    assert len(calls) == 39
    assert selection["selections"] == {
        "short": {
            "updates": 200,
            "trace_half_life_x_steps": 10.0,
            "trace_half_life_f_steps": 5.0,
            "x_summary": "short-x-summary.json",
            "f_summary": "short-f-summary.json",
        },
        "medium": {
            "updates": 400,
            "trace_half_life_x_steps": 20.0,
            "trace_half_life_f_steps": 30.0,
            "x_summary": "medium-x-summary.json",
            "f_summary": "medium-f-summary.json",
        },
        "long": {
            "updates": 800,
            "trace_half_life_x_steps": 60.0,
            "trace_half_life_f_steps": 100.0,
            "x_summary": "long-x-summary.json",
            "f_summary": "long-f-summary.json",
        },
    }
    assert selection["development_only"] is True
    assert selection["sealed_test"] is False

    resumed = run_development_trace_search(
        settings,
        runner=lambda *_: (_ for _ in ()).throw(
            AssertionError("exact raw results must be reused")
        ),
        progress=lambda _: None,
    )
    assert resumed["selections"] == selection["selections"]
    summary = json.loads(
        (settings.output_directory / "long-f-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        bundle["reused"]
        for candidate in summary["candidates"]
        for bundle in candidate["bundle_scores"]
    )


def test_stability_invalid_candidate_is_rejected_not_selected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def runner(command, _source_root) -> None:
        command = tuple(command)
        config = _config_from_command(command)
        mean_firing = (
            0.4
            if config.horizon == "short"
            and config.trace_half_life_x_steps == 5.0
            else 0.1
        )
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(settings, config, 0.1, mean_firing=mean_firing),
        )

    run_development_trace_search(settings, runner=runner, progress=lambda _: None)
    summary = json.loads(
        (settings.output_directory / "short-x-summary.json").read_text(
            encoding="utf-8"
        )
    )
    rejected = next(item for item in summary["candidates"] if item["half_life"] == 5)

    assert rejected["status"] == "rejected"
    assert summary["selected_half_life"] == 10
    assert all("stability" in reason for reason in rejected["rejection_reasons"])


@pytest.mark.parametrize("drift", ["configuration", "provenance"])
def test_resume_refuses_configuration_or_provenance_drift(
    tmp_path: Path, drift: str
) -> None:
    settings = _settings(tmp_path)
    grid = HORIZON_TRACE_GRIDS[0]
    candidate = coordinate_candidates(grid, "x", grid.provisional_f)[0]
    bundle_id = DEVELOPMENT_BUNDLES[0]
    path = raw_trace_result_path(settings, grid, candidate, bundle_id)
    config = expected_trace_benchmark_config(
        settings, grid, candidate, bundle_id
    )
    document = _raw_document(settings, config, 0.1)
    if drift == "configuration":
        document["result"]["config"]["trace_half_life_x_steps"] = 10.0
    else:
        document["environment"]["source_commit"] = "different"
    _write(path, document)

    with pytest.raises(ResumeConfigurationError, match=drift):
        obtain_trace_bundle_score(
            settings,
            grid,
            candidate,
            bundle_id,
            lambda *_: (_ for _ in ()).throw(AssertionError("must reuse")),
        )
