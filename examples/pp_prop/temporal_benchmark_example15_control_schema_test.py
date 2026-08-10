"""Tests for fail-closed Example 15 evidence schemas."""

import copy
import statistics

import pytest

from temporal_benchmark_example15_control_run import fixed_config_document
from temporal_benchmark_example15_control_schema import (
    accept_example15_baseline,
    compare_example15_runs,
    validate_example15_run,
    validated_accuracy_change,
)


def _run_document(accuracies=(0.95, 0.96, 0.97), *, dirty=False):
    seed_results = [
        {
            "seed": seed,
            "final_accuracy": accuracy,
            "losses": [1.0, 0.8, 0.6, 0.4, 0.2],
            "recurrent_values_changed": 100,
        }
        for seed, accuracy in enumerate(accuracies)
    ]
    return {
        "schema_version": 1,
        "kind": "braintrace_example15_static_control_run",
        "development_only": True,
        "sealed_test": False,
        "accepted_baseline": False,
        "environment": {
            "source_commit": "0123456789abcdef",
            "source_dirty": dirty,
            "container_image_digest": "sha256:image",
        },
        "example_source_sha256": "a" * 64,
        "fixed_config": fixed_config_document(),
        "result": {
            "status": "completed",
            "seed_results": seed_results,
            "mean_accuracy": sum(accuracies) / 3,
            "minimum_accuracy": min(accuracies),
            "std_accuracy": statistics.pstdev(accuracies),
            "recurrent_nnz": 768,
        },
    }


def test_clean_passing_candidate_requires_explicit_baseline_acceptance() -> None:
    candidate = _run_document()

    with pytest.raises(ValueError, match="explicitly accepted"):
        validate_example15_run(candidate, require_accepted_baseline=True)

    accepted = accept_example15_baseline(candidate, "b" * 64)
    evidence = validate_example15_run(accepted, require_accepted_baseline=True)

    assert evidence.accepted_baseline is True
    assert accepted["baseline_acceptance"]["candidate_sha256"] == "b" * 64


def test_dirty_or_failing_candidate_cannot_become_baseline() -> None:
    with pytest.raises(ValueError, match="clean"):
        accept_example15_baseline(_run_document(dirty=True), "b" * 64)
    with pytest.raises(ValueError, match="pass"):
        accept_example15_baseline(_run_document((0.5, 0.5, 0.5)), "b" * 64)


def test_comparison_reports_current_minus_pinned_baseline_for_adoption() -> None:
    baseline = accept_example15_baseline(_run_document(), "b" * 64)
    current = _run_document((0.94, 0.96, 0.97), dirty=True)

    comparison = compare_example15_runs(baseline, current, "c" * 64, "d" * 64)

    expected = (0.94 + 0.96 + 0.97) / 3 - 0.96
    assert comparison["example15_accuracy_change"] == pytest.approx(expected)
    assert comparison["static_control_gate_passed"] is True
    assert validated_accuracy_change(comparison) == pytest.approx(expected)


@pytest.mark.parametrize("drift", ("config", "mean", "seed", "nonfinite"))
def test_run_validation_refuses_incomplete_or_inconsistent_evidence(drift) -> None:
    document = _run_document()
    if drift == "config":
        document["fixed_config"]["n_rec"] = 97
    elif drift == "mean":
        document["result"]["mean_accuracy"] = 0.5
    elif drift == "seed":
        document["result"]["seed_results"][1]["seed"] = 9
    else:
        document["result"]["seed_results"][1]["final_accuracy"] = float("nan")

    with pytest.raises(ValueError):
        validate_example15_run(document)


def test_static_control_consumer_refuses_tampered_delta_or_gate() -> None:
    baseline = accept_example15_baseline(_run_document(), "b" * 64)
    comparison = compare_example15_runs(
        baseline, _run_document(dirty=True), "c" * 64, "d" * 64
    )
    tampered = copy.deepcopy(comparison)
    tampered["example15_accuracy_change"] = -0.5
    with pytest.raises(ValueError, match="inconsistent"):
        validated_accuracy_change(tampered)
    tampered = copy.deepcopy(comparison)
    tampered["static_control_gate_passed"] = False
    with pytest.raises(ValueError, match="gate"):
        validated_accuracy_change(tampered)
