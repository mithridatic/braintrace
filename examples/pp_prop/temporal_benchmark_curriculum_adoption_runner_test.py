"""Tests for resumable paired evidence and the curriculum adoption decision."""

from __future__ import annotations

import msgspec_json
from dataclasses import replace
from pathlib import Path

import pytest

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_curriculum_adoption_config import (
    CURRICULUM_ADOPTION_KIND,
    CURRICULUM_ADOPTION_SCHEMA_VERSION,
    CURRICULUM_BUNDLE_KIND,
    CurriculumAdoptionSettings,
    expected_curriculum_config,
    selected_config_document,
)
from temporal_benchmark_curriculum_adoption_runner import (
    run_development_curriculum_adoption,
    validate_curriculum_bundle_document,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_search_selection import (
    ResumeConfigurationError,
    RunEvidenceError,
)


def _settings(tmp_path: Path, **overrides) -> CurriculumAdoptionSettings:
    values = {
        "source_root": tmp_path,
        "output_directory": tmp_path / "comparison",
        "experiment_script": tmp_path / "paired.py",
        "manifest_path": tmp_path / "manifest.json",
        "python_executable": "python-test",
        "container_image_digest": "sha256:test-image",
        "source_commit": "0123456789abcdef",
        "source_dirty": True,
        "example15_accuracy_change": -0.005,
        "device": "cpu",
    }
    values.update(overrides)
    return CurriculumAdoptionSettings(**values)


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def _raw_document(
    settings: CurriculumAdoptionSettings,
    bundle_id: str,
    *,
    curriculum_accuracy: float = 0.9,
    direct_accuracy: float = 0.7,
    curriculum_time: int | None = None,
    direct_time: int | None = None,
    stable: bool = True,
) -> dict[str, object]:
    config = expected_curriculum_config(settings, bundle_id)
    updates = {"short": 100, "medium": 100, "long": 100}
    steps = {"short": 10, "medium": 30, "long": 100}
    phases = {
        horizon: {
            "updates_completed": count,
            "horizon_steps": steps[horizon],
            "batch_size": settings.batch_size,
            "sample_ticks": count * settings.batch_size * steps[horizon],
        }
        for horizon, count in updates.items()
    }
    total = sum(int(phase["sample_ticks"]) for phase in phases.values())
    direct_updates = total // (settings.batch_size * steps["long"])
    direct_config = replace(config, curriculum=False, updates=direct_updates)
    curriculum = {
        "status": "completed",
        "config": config_to_dict(config),
        "final_validation": {"ensemble_accuracy": curriculum_accuracy},
        "sealed_test_metrics": None,
    }
    direct = {
        "status": "completed",
        "config": config_to_dict(direct_config),
        "final_validation": {"ensemble_accuracy": direct_accuracy},
        "sealed_test_metrics": None,
    }
    return {
        "schema_version": CURRICULUM_ADOPTION_SCHEMA_VERSION,
        "kind": CURRICULUM_BUNDLE_KIND,
        "development_only": True,
        "sealed_test": False,
        "environment": {
            "container_image_digest": settings.container_image_digest,
            "source_commit": settings.source_commit,
            "source_dirty": settings.source_dirty,
        },
        "selected_config": selected_config_document(settings),
        "result": {
            "status": "completed",
            "bundle_id": bundle_id,
            "sealed_test_metrics": None,
            "base_config": config_to_dict(config),
            "curriculum": curriculum,
            "direct_long": direct,
            "sample_tick_budget": {
                "definition": "updates * batch_size * horizon_steps",
                "curriculum_phases": phases,
                "curriculum_total_sample_ticks": total,
                "direct_long": {
                    "updates": direct_updates,
                    "horizon_steps": 100,
                    "batch_size": settings.batch_size,
                    "sample_ticks_per_update": settings.batch_size * 100,
                    "total_sample_ticks": total,
                },
                "exact_match": True,
            },
            "time_to_0_80": {
                "curriculum_sample_ticks": curriculum_time,
                "direct_long_sample_ticks": direct_time,
            },
            "stability": {"curriculum": stable, "direct_long": stable},
        },
    }


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(msgspec_json.dumps(document), encoding="utf-8")


def test_accuracy_interval_adopts_and_resume_reuses_all_raw_bundles(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    calls: list[str] = []

    def runner(command, _source_root) -> None:
        command = tuple(command)
        bundle_id = _value(command, "--bundle-id")
        calls.append(bundle_id)
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(settings, bundle_id),
        )

    decision = run_development_curriculum_adoption(
        settings, runner=runner, progress=lambda _: None
    )

    assert calls == list(DEVELOPMENT_BUNDLES)
    assert decision["kind"] == CURRICULUM_ADOPTION_KIND
    assert decision["schema_version"] == 1
    assert decision["development_only"] is True
    assert decision["sealed_test"] is False
    assert decision["adoption"] is True
    assert decision["decision_evidence"]["time_to_0_80_complete"] is False
    assert decision["decision_evidence"]["time_to_0_80_reduction_fraction"] is None
    assert decision["decision_evidence"]["paired_long_accuracy_interval"]["lower"] > 0
    assert decision["provenance"]["source_dirty"] is True
    assert decision["device"] == "cpu"
    assert decision["neurons"] == 96
    assert decision["degree"] == 8
    assert decision["batch_size"] == 32

    resumed = run_development_curriculum_adoption(
        settings,
        runner=lambda *_: (_ for _ in ()).throw(
            AssertionError("valid raw comparisons must be reused")
        ),
        progress=lambda _: None,
    )
    assert all(item["reused"] for item in resumed["bundle_evidence"])


def test_complete_time_reduction_can_adopt_without_accuracy_advantage(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    def runner(command, _source_root) -> None:
        command = tuple(command)
        bundle_id = _value(command, "--bundle-id")
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(
                settings,
                bundle_id,
                curriculum_accuracy=0.8,
                direct_accuracy=0.8,
                curriculum_time=320_000,
                direct_time=400_000,
            ),
        )

    decision = run_development_curriculum_adoption(
        settings, runner=runner, progress=lambda _: None
    )

    assert decision["decision_evidence"]["time_to_0_80_complete"] is True
    assert decision["decision_evidence"][
        "time_to_0_80_reduction_fraction"
    ] == pytest.approx(0.2)
    assert decision["decision_evidence"]["time_gate_passed"] is True
    assert decision["adoption"] is True


def test_example15_regression_or_unstable_pair_blocks_adoption(tmp_path: Path) -> None:
    settings = _settings(tmp_path, example15_accuracy_change=-0.011)

    def runner(command, _source_root) -> None:
        command = tuple(command)
        bundle_id = _value(command, "--bundle-id")
        _write(
            Path(_value(command, "--json-output")),
            _raw_document(
                settings, bundle_id, stable=bundle_id != DEVELOPMENT_BUNDLES[1]
            ),
        )

    decision = run_development_curriculum_adoption(
        settings, runner=runner, progress=lambda _: None
    )

    assert decision["decision_evidence"]["static_control_gate_passed"] is False
    assert decision["decision_evidence"]["all_paired_runs_stable"] is False
    assert decision["adoption"] is False


@pytest.mark.parametrize("drift", ("config", "commit", "dirty", "image"))
def test_resume_refuses_configuration_or_provenance_drift(
    tmp_path: Path, drift: str
) -> None:
    settings = _settings(tmp_path)
    document = _raw_document(settings, DEVELOPMENT_BUNDLES[0])
    if drift == "config":
        document["result"]["base_config"]["gain"] = 1.2
    elif drift == "commit":
        document["environment"]["source_commit"] = "different"
    elif drift == "dirty":
        document["environment"]["source_dirty"] = False
    else:
        document["environment"]["container_image_digest"] = "sha256:different"

    with pytest.raises(ResumeConfigurationError):
        validate_curriculum_bundle_document(
            document, settings, DEVELOPMENT_BUNDLES[0], "raw.json", reused=True
        )


def test_raw_validation_rejects_inexact_sample_tick_arithmetic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    document = _raw_document(settings, DEVELOPMENT_BUNDLES[0])
    document["result"]["sample_tick_budget"]["direct_long"]["total_sample_ticks"] += 1

    with pytest.raises(RunEvidenceError, match="do not match"):
        validate_curriculum_bundle_document(
            document, settings, DEVELOPMENT_BUNDLES[0], "raw.json", reused=False
        )
