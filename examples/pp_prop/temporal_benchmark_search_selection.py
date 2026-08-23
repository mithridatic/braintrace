"""Pure raw-evidence validation and optimizer-candidate selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from temporal_benchmark_config import TemporalBenchmarkConfig, config_to_dict
from temporal_benchmark_search_config import LearningRateCandidate


@dataclass(frozen=True)
class BundleScore:
    """Store validation and stability values for one development bundle."""

    bundle_id: str
    validation_nll: float
    validation_accuracy: float
    recurrent_update_ratio_p99: float
    raw_path: str
    reused: bool


@dataclass(frozen=True)
class CandidateScore:
    """Store one candidate's aggregate result at one search stage."""

    candidate: LearningRateCandidate
    bundle_scores: tuple[BundleScore, ...]
    rejection_reasons: tuple[str, ...]
    mean_validation_nll: float | None
    mean_validation_accuracy: float | None
    mean_recurrent_update_ratio_p99: float | None

    @property
    def accepted(self) -> bool:
        """Return whether every development bundle passed validation."""
        return not self.rejection_reasons


class ResumeConfigurationError(ValueError):
    """show that an existing raw run does not match the requested run."""


class RunEvidenceError(ValueError):
    """show that a raw run is incomplete, non-finite, or stability-invalid."""


class StabilityEvidenceError(RunEvidenceError):
    """show complete evidence that fails the scientific stability limits."""


def ensure_finite(value: object, location: str = "root") -> None:
    """Reject non-finite numeric leaves in a JSON-compatible value tree."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RunEvidenceError(f"Non-finite numeric value at {location}. Use finite values.")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            ensure_finite(child, f"{location}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            ensure_finite(child, f"{location}[{index}]")


def _number(mapping: Mapping[str, object], key: str, location: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunEvidenceError(f"{location}.{key} must be numeric. Set {location}.{key} to numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RunEvidenceError(f"{location}.{key} must be finite. Use finite values for {location}.{key}.")
    return numeric


def percentile_99(values: Sequence[float]) -> float:
    """Return NumPy-compatible linear interpolation for the 99th percentile."""
    if not values:
        raise RunEvidenceError("Recurrent update ratios must be nonempty. Set Recurrent update ratios to nonempty.")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise RunEvidenceError("Recurrent update ratios must be finite. Use finite values for Recurrent update ratios.")
    position = 0.99 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_complete_config(
    result: Mapping[str, object], expected: TemporalBenchmarkConfig
) -> None:
    if result.get("config") != config_to_dict(expected):
        raise ResumeConfigurationError(
            "Existing raw result configuration does not exactly match the requested "
            "benchmark configuration. Fix the input condition named in the error, then rerun the operation."
        )


def _validate_provenance(
    document: Mapping[str, object],
    container_image_digest: str,
    source_commit: str,
) -> None:
    environment = document.get("environment")
    if not isinstance(environment, Mapping):
        raise ResumeConfigurationError(
            "Raw result lacks required development-search provenance. Provide the missing item named in the message."
        )
    if (
        environment.get("container_image_digest") != container_image_digest
        or environment.get("source_commit") != source_commit
    ):
        raise ResumeConfigurationError(
            "Raw result provenance does not match the requested image and source. Use matching values and structures."
        )


def score_raw_document(
    document: Mapping[str, object],
    expected: TemporalBenchmarkConfig,
    raw_path: str,
    container_image_digest: str,
    source_commit: str,
    *,
    reused: bool,
) -> BundleScore:
    """Validate one raw development result and return its ranking values."""
    if document.get("schema_version") != 1:
        raise RunEvidenceError("Raw result has an unsupported schema version. Use a supported option or change the configuration.")
    if document.get("sealed_test") is not False:
        raise RunEvidenceError("Development search cannot consume sealed results. Fix the input condition named in the error, then rerun the operation.")
    result = document.get("result")
    if not isinstance(result, Mapping):
        raise RunEvidenceError("Raw result lacks its result object. Provide the missing item named in the message.")
    _validate_complete_config(result, expected)
    _validate_provenance(document, container_image_digest, source_commit)
    if result.get("status") != "completed":
        raise RunEvidenceError("Development run did not complete. Fix the input condition named in the error, then rerun the operation.")
    if result.get("sealed_test_metrics") is not None:
        raise RunEvidenceError("Development run materialized sealed test metrics. Fix the input condition named in the error, then rerun the operation.")
    ensure_finite(document)

    validation = result.get("final_validation")
    dynamics = result.get("dynamics")
    telemetry = result.get("optimizer_telemetry")
    if not isinstance(validation, Mapping) or not isinstance(dynamics, Mapping):
        raise RunEvidenceError("Raw result lacks validation or dynamics metrics. Provide the missing item named in the message.")
    if not isinstance(telemetry, Mapping):
        raise RunEvidenceError("Raw result lacks optimizer telemetry. Provide the missing item named in the message.")
    recurrent = telemetry.get("recurrent")
    if not isinstance(recurrent, Mapping):
        raise RunEvidenceError("Raw result lacks recurrent optimizer telemetry. Provide the missing item named in the message.")
    ratios_value = recurrent.get("update_to_weight_ratio")
    if not isinstance(ratios_value, list):
        raise RunEvidenceError("Raw result lacks recurrent update ratios. Provide the missing item named in the message.")
    ratio_p99 = percentile_99([float(value) for value in ratios_value])

    nll = _number(validation, "ensemble_nll", "final_validation")
    accuracy = _number(validation, "ensemble_accuracy", "final_validation")
    mean_firing = _number(dynamics, "mean_firing_spikes_per_neuron_step", "dynamics")
    silent_fraction = _number(dynamics, "silent_neuron_fraction", "dynamics")
    saturated_fraction = _number(dynamics, "saturated_neuron_fraction", "dynamics")
    if nll < 0.0 or not 0.0 <= accuracy <= 1.0:
        raise RunEvidenceError("Validation metrics are outside their valid ranges. Set the named field to a value in the stated range, then rerun the operation.")
    stable = (
        0.001 <= mean_firing <= 0.30
        and silent_fraction <= 0.95
        and saturated_fraction < 0.05
        and ratio_p99 <= 0.05
    )
    if not stable:
        raise StabilityEvidenceError(
            "Development run failed the primary stability gates. Correct the reported inputs, then retry the operation."
        )
    return BundleScore(
        bundle_id=expected.bundle_id,
        validation_nll=nll,
        validation_accuracy=accuracy,
        recurrent_update_ratio_p99=ratio_p99,
        raw_path=raw_path,
        reused=reused,
    )


def rank_candidate_scores(
    scores: Sequence[CandidateScore],
) -> tuple[CandidateScore, ...]:
    """Rank valid candidates by the specification's deterministic criteria."""
    accepted = [score for score in scores if score.accepted]
    return tuple(
        sorted(
            accepted,
            key=lambda score: (
                float(score.mean_validation_nll),
                -float(score.mean_validation_accuracy),
                float(score.mean_recurrent_update_ratio_p99),
                score.candidate.grid_index,
            ),
        )
    )
