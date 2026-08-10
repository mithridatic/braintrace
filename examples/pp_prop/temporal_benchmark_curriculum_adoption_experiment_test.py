"""Tests for exact sample-tick matching and threshold evidence."""

from types import SimpleNamespace

import pytest

import temporal_benchmark_curriculum_adoption_experiment as experiment
from temporal_benchmark_config import TemporalBenchmarkConfig


def _curriculum_phases(updates=(100, 100, 100)) -> dict[str, object]:
    return {
        "phases": {
            horizon: {"updates_completed": count, "history": []}
            for (horizon, _), count in zip(experiment.CURRICULUM_PHASES, updates)
        }
    }


def test_matched_budget_records_exact_phase_and_direct_arithmetic() -> None:
    budget = experiment.matched_sample_tick_budget(_curriculum_phases(), 32)

    assert budget["curriculum_total_sample_ticks"] == 448_000
    assert budget["direct_long"] == {
        "updates": 140,
        "horizon_steps": 100,
        "batch_size": 32,
        "sample_ticks_per_update": 3200,
        "total_sample_ticks": 448_000,
    }
    assert budget["exact_match"] is True


def test_matched_budget_fails_closed_when_custom_updates_do_not_divide() -> None:
    with pytest.raises(ValueError, match="integer number"):
        experiment.matched_sample_tick_budget(_curriculum_phases((1, 1, 1)), 32)


def test_threshold_time_counts_prior_phases_and_requires_stability() -> None:
    curriculum = _curriculum_phases()
    curriculum["phases"]["long"]["history"] = [
        {
            "update": 50,
            "metrics": {"ensemble_accuracy": 0.90},
            "stability_passed": False,
        },
        {
            "update": 100,
            "metrics": {"ensemble_accuracy": 0.80},
            "stability_passed": True,
        },
    ]
    direct = {
        "history": [
            {
                "update": 50,
                "metrics": {"ensemble_accuracy": 0.80},
                "stability_passed": True,
            }
        ]
    }
    budget = experiment.matched_sample_tick_budget(curriculum, 32)

    times = experiment.threshold_times(curriculum, direct, budget)

    assert times == {
        "curriculum_sample_ticks": 448_000,
        "direct_long_sample_ticks": 160_000,
    }


def test_pair_uses_fresh_direct_config_and_actual_curriculum_budget(
    monkeypatch,
) -> None:
    config = TemporalBenchmarkConfig(curriculum=True, device="cpu")
    curriculum = _curriculum_phases()
    curriculum.update(
        {
            "final_validation": {"ensemble_accuracy": 0.8},
            "optimizer_telemetry": {},
            "dynamics": {},
        }
    )
    observed = {}
    monkeypatch.setattr(experiment, "run_curriculum", lambda *_: curriculum)
    monkeypatch.setattr(experiment, "_result_stable", lambda *_: True)

    def direct(direct_config, bundle):
        observed["config"] = direct_config
        observed["bundle"] = bundle
        return {
            "history": [],
            "final_validation": {"ensemble_accuracy": 0.8},
            "optimizer_telemetry": {},
            "dynamics": {},
        }

    monkeypatch.setattr(experiment, "_run_direct_long", direct)
    bundle = SimpleNamespace(bundle_id="split0-topology0-weight0")

    result = experiment.run_paired_curriculum_experiment(config, bundle)

    assert observed["bundle"] is bundle
    assert observed["config"].curriculum is False
    assert observed["config"].updates == 140
    assert result["sample_tick_budget"]["exact_match"] is True
