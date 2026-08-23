"""Fail-closed schemas for Example 15 static-control evidence."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from temporal_benchmark_example15_control_run import (
    EXAMPLE15_RUN_KIND,
    EXAMPLE15_RUN_SCHEMA_VERSION,
    fixed_config_document,
)

EXAMPLE15_CONTROL_SCHEMA_VERSION = 1
EXAMPLE15_CONTROL_KIND = "temporal_credit_example15_static_control"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class Example15RunEvidence:
    """Store validated fixed-profile accuracy and provenance."""

    mean_accuracy: float
    minimum_accuracy: float
    seed_accuracies: tuple[float, ...]
    acceptance_passed: bool
    environment: Mapping[str, object]
    example_source_sha256: str
    accepted_baseline: bool


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object. Set {location} to an object.")
    return value


def _finite_tree(value: object, location: str = "root") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {location}. Use finite values.")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{location}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _finite_tree(child, f"{location}[{index}]")


def _number(mapping: Mapping[str, object], key: str, location: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be numeric. Set {location}.{key} to numeric.")
    return float(value)


def _validate_environment(document: Mapping[str, object]) -> Mapping[str, object]:
    environment = _mapping(document.get("environment"), "environment")
    for key in ("source_commit", "container_image_digest"):
        value = environment.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Environment.{key} is required. Fix the input condition named in the error, then rerun the operation.")
    if not isinstance(environment.get("source_dirty"), bool):
        raise ValueError("Environment.source_dirty must be boolean. Set Environment.source_dirty to boolean.")
    return environment


def _validate_seed_results(
    result: Mapping[str, object],
) -> tuple[tuple[float, ...], bool]:
    raw = result.get("seed_results")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("Example 15 must contain exactly three seed results. Add exactly three seed results to Example 15.")
    accuracies: list[float] = []
    losses_decrease = True
    recurrent_changes = True
    for expected_seed, item in enumerate(raw):
        seed_result = _mapping(item, f"seed_results[{expected_seed}]")
        if seed_result.get("seed") != expected_seed:
            raise ValueError("Example 15 seed results must be ordered 0, 1, 2. Set Example 15 seed results to ordered 0, 1, 2.")
        accuracy = _number(seed_result, "final_accuracy", "seed_result")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("Example 15 accuracy must lie in [0, 1]. Set Example 15 accuracy to a value in [0, 1].")
        losses = seed_result.get("losses")
        if not isinstance(losses, list) or len(losses) != 5:
            raise ValueError("Each Example 15 seed must contain five epoch losses. Add five epoch losses to Each Example 15 seed.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in losses
        ):
            raise ValueError("Example 15 epoch losses must be numeric. Set Example 15 epoch losses to numeric.")
        losses_decrease &= float(losses[-1]) < float(losses[0])
        changed = seed_result.get("recurrent_values_changed")
        recurrent_changes &= (
            not isinstance(changed, bool) and isinstance(changed, int) and changed > 0
        )
        accuracies.append(accuracy)
    return tuple(accuracies), losses_decrease and recurrent_changes


def validate_example15_run(
    document: Mapping[str, object], *, require_accepted_baseline: bool = False
) -> Example15RunEvidence:
    """Validate an exact fixed-profile run and its required provenance."""
    if (
        document.get("schema_version") != EXAMPLE15_RUN_SCHEMA_VERSION
        or document.get("kind") != EXAMPLE15_RUN_KIND
        or document.get("development_only") is not True
        or document.get("sealed_test") is not False
    ):
        raise ValueError("Unsupported Example 15 run schema. Use a supported option or change the configuration.")
    _finite_tree(document)
    environment = _validate_environment(document)
    if document.get("fixed_config") != fixed_config_document():
        raise ValueError("Example 15 run does not match the fixed numerical profile. Use matching values and structures.")
    source_hash = document.get("example_source_sha256")
    if not _is_sha256(source_hash):
        raise ValueError("Example 15 source SHA-256 is required. Fix the input condition named in the error, then rerun the operation.")
    assert isinstance(source_hash, str)
    result = _mapping(document.get("result"), "result")
    if result.get("status") != "completed":
        raise ValueError("Example 15 run did not complete. Fix the input condition named in the error, then rerun the operation.")
    accuracies, per_seed_pass = _validate_seed_results(result)
    mean = statistics.fmean(accuracies)
    minimum = min(accuracies)
    recorded_mean = _number(result, "mean_accuracy", "result")
    recorded_minimum = _number(result, "minimum_accuracy", "result")
    recorded_std = _number(result, "std_accuracy", "result")
    if not math.isclose(mean, recorded_mean, abs_tol=1e-12):
        raise ValueError("Example 15 mean accuracy does not match its seed results. Use matching values and structures.")
    if not math.isclose(minimum, recorded_minimum, abs_tol=1e-12):
        raise ValueError("Example 15 minimum accuracy does not match its seed results. Use matching values and structures.")
    if not math.isclose(statistics.pstdev(accuracies), recorded_std, abs_tol=1e-12):
        raise ValueError(
            "Example 15 accuracy deviation does not match its seed results. Use matching values and structures."
        )
    acceptance = (
        per_seed_pass
        and result.get("recurrent_nnz") == 768
        and minimum >= 0.90
        and mean >= 0.95
    )
    accepted = document.get("accepted_baseline") is True
    if require_accepted_baseline:
        if not accepted or environment.get("source_dirty") is not False:
            raise ValueError("Baseline must be explicitly accepted from clean source. Set Baseline to explicitly accepted from clean source.")
        baseline = _mapping(document.get("baseline_acceptance"), "baseline_acceptance")
        if (
            baseline.get("acceptance_checks_passed") is not True
            or baseline.get("operation")
            != "explicit_static_control_baseline_acceptance"
            or not _is_sha256(baseline.get("candidate_sha256"))
            or not acceptance
        ):
            raise ValueError("Accepted baseline must pass Example 15 acceptance")
    return Example15RunEvidence(
        mean,
        minimum,
        accuracies,
        acceptance,
        environment,
        source_hash,
        accepted,
    )


def accept_example15_baseline(
    candidate: Mapping[str, object], candidate_sha256: str
) -> dict[str, object]:
    """Explicitly promote one clean passing run to baseline status."""
    evidence = validate_example15_run(candidate)
    if evidence.environment.get("source_dirty") is not False:
        raise ValueError("Baseline candidate must come from a clean source tree. Set Baseline candidate to come from a clean source tree.")
    if not evidence.acceptance_passed:
        raise ValueError("Baseline candidate must pass Example 15 acceptance")
    if not _is_sha256(candidate_sha256):
        raise ValueError("Candidate SHA-256 is required. Fix the input condition named in the error, then rerun the operation.")
    accepted = copy.deepcopy(dict(candidate))
    accepted["accepted_baseline"] = True
    accepted["baseline_acceptance"] = {
        "candidate_sha256": candidate_sha256,
        "acceptance_checks_passed": True,
        "operation": "explicit_static_control_baseline_acceptance",
    }
    return accepted


def compare_example15_runs(
    baseline: Mapping[str, object],
    current: Mapping[str, object],
    baseline_sha256: str,
    current_sha256: str,
) -> dict[str, object]:
    """Compare fixed-profile mean accuracy against a pinned accepted baseline."""
    if not _is_sha256(baseline_sha256) or not _is_sha256(current_sha256):
        raise ValueError("Baseline and current artifact SHA-256 values are required. Fix the input condition named in the error, then rerun the operation.")
    baseline_evidence = validate_example15_run(baseline, require_accepted_baseline=True)
    current_evidence = validate_example15_run(current)
    change = current_evidence.mean_accuracy - baseline_evidence.mean_accuracy
    return {
        "schema_version": EXAMPLE15_CONTROL_SCHEMA_VERSION,
        "kind": EXAMPLE15_CONTROL_KIND,
        "status": "completed",
        "development_only": True,
        "sealed_test": False,
        "fixed_config": fixed_config_document(),
        "baseline": {
            "artifact_sha256": baseline_sha256,
            "provenance": dict(baseline_evidence.environment),
            "example_source_sha256": baseline_evidence.example_source_sha256,
            "mean_accuracy": baseline_evidence.mean_accuracy,
            "minimum_accuracy": baseline_evidence.minimum_accuracy,
            "acceptance_passed": baseline_evidence.acceptance_passed,
            "accepted_baseline": baseline_evidence.accepted_baseline,
        },
        "current": {
            "artifact_sha256": current_sha256,
            "provenance": dict(current_evidence.environment),
            "example_source_sha256": current_evidence.example_source_sha256,
            "mean_accuracy": current_evidence.mean_accuracy,
            "minimum_accuracy": current_evidence.minimum_accuracy,
            "acceptance_passed": current_evidence.acceptance_passed,
        },
        "example15_accuracy_change": change,
        "static_control_gate_passed": change >= -0.01,
    }


def validated_accuracy_change(document: Mapping[str, object]) -> float:
    """Return the adoption input only from a self-consistent comparison schema."""
    if (
        document.get("schema_version") != EXAMPLE15_CONTROL_SCHEMA_VERSION
        or document.get("kind") != EXAMPLE15_CONTROL_KIND
        or document.get("status") != "completed"
        or document.get("development_only") is not True
        or document.get("sealed_test") is not False
        or document.get("fixed_config") != fixed_config_document()
    ):
        raise ValueError("Unsupported Example 15 static-control schema. Use a supported option or change the configuration.")
    _finite_tree(document)
    baseline = _mapping(document.get("baseline"), "baseline")
    current = _mapping(document.get("current"), "current")
    if (
        baseline.get("accepted_baseline") is not True
        or baseline.get("acceptance_passed") is not True
        or _mapping(baseline.get("provenance"), "baseline.provenance").get(
            "source_dirty"
        )
        is not False
    ):
        raise ValueError("Static control lacks an accepted passing baseline. Provide the missing item named in the message.")
    baseline_mean = _number(baseline, "mean_accuracy", "baseline")
    current_mean = _number(current, "mean_accuracy", "current")
    change = _number(document, "example15_accuracy_change", "root")
    if not math.isclose(change, current_mean - baseline_mean, abs_tol=1e-12):
        raise ValueError("Static-control accuracy change is inconsistent")
    if document.get("static_control_gate_passed") is not (change >= -0.01):
        raise ValueError("Static-control gate is inconsistent. Use matching values and structures.")
    for side in (baseline, current):
        artifact_hash = side.get("artifact_sha256")
        provenance = _mapping(side.get("provenance"), "provenance")
        if not _is_sha256(artifact_hash):
            raise ValueError("Static-control artifact SHA-256 is required. Fix the input condition named in the error, then rerun the operation.")
        if not isinstance(provenance.get("source_dirty"), bool):
            raise ValueError("Static-control provenance is incomplete. Fix the input condition named in the error, then rerun the operation.")
        for key in ("source_commit", "container_image_digest"):
            value = provenance.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Static-control provenance is incomplete. Fix the input condition named in the error, then rerun the operation.")
    return change
