"""Validate clip-trigger and curriculum-adoption development decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_freeze_validation import (
    GROUPS,
    ensure_finite_tree,
    require_mapping,
    require_number,
    validate_decision_provenance,
    validate_header,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES

CLIP_CANDIDATES = [0.5, 1.0, 2.0, None]
CLIP_TRIGGER_THRESHOLD = 0.25


def _require_selected_config(
    document: Mapping[str, Any], expected: Mapping[str, object], location: str
) -> None:
    selected = require_mapping(
        document.get("selected_config"), f"{location}.selected_config"
    )
    if selected != expected:
        raise FreezeArtifactError(
            f"{location}.selected_config does not match prior selections"
        )


def _clip_group(value: object, group: str) -> float | None:
    evidence = require_mapping(value, f"clip.groups.{group}")
    fractions = require_mapping(
        evidence.get("observed_clip_event_fractions"), f"clip.groups.{group}.fractions"
    )
    if set(fractions) != set(DEVELOPMENT_BUNDLES):
        raise FreezeArtifactError(
            f"clip {group} fractions do not cover development bundles"
        )
    values = [
        require_number(fractions[bundle], f"clip.groups.{group}.{bundle}")
        for bundle in DEVELOPMENT_BUNDLES
    ]
    if not all(0.0 <= item <= 1.0 for item in values):
        raise FreezeArtifactError(f"clip {group} fractions must be probabilities")
    triggered = evidence.get("triggered")
    expected_trigger = any(item > CLIP_TRIGGER_THRESHOLD for item in values)
    if not isinstance(triggered, bool) or triggered != expected_trigger:
        raise FreezeArtifactError(f"clip {group} trigger decision is inconsistent")
    candidates = evidence.get("candidates")
    expected_candidates: list[float | None] = CLIP_CANDIDATES if triggered else []
    if candidates != expected_candidates:
        raise FreezeArtifactError(f"clip {group} candidates do not match its trigger")
    selected = evidence.get("selected_clip_norm")
    allowed = CLIP_CANDIDATES if triggered else [1.0]
    if isinstance(selected, bool) or selected not in allowed:
        raise FreezeArtifactError(f"clip {group} selected norm is invalid")
    return None if selected is None else float(selected)


def validate_clip_selection(
    document: Mapping[str, Any], prior: Mapping[str, object]
) -> tuple[dict[str, float | None], dict[str, object]]:
    """Validate trigger arithmetic and return selected per-group clip norms."""
    ensure_finite_tree(document, "clip")
    validate_header(document, "temporal_credit_clip_selection")
    provenance = validate_decision_provenance(document, "clip")
    if (
        require_number(document.get("trigger_threshold"), "clip.trigger_threshold")
        != CLIP_TRIGGER_THRESHOLD
    ):
        raise FreezeArtifactError("clip trigger threshold must be 0.25")
    expected = {
        key: prior[key] for key in ("gain", "learning_rates", "recurrent_weight_decay")
    }
    _require_selected_config(document, expected, "clip")
    groups = require_mapping(document.get("groups"), "clip.groups")
    if set(groups) != set(GROUPS):
        raise FreezeArtifactError("clip selection must contain exactly three groups")
    return ({group: _clip_group(groups[group], group) for group in GROUPS}, provenance)


def _adoption_evidence(document: Mapping[str, Any]) -> bool:
    evidence = require_mapping(
        document.get("decision_evidence"), "curriculum.decision_evidence"
    )
    complete = evidence.get("time_to_0_80_complete")
    if not isinstance(complete, bool):
        raise FreezeArtifactError(
            "curriculum time-to-threshold completeness must be boolean"
        )
    reduction_value = evidence.get("time_to_0_80_reduction_fraction")
    if complete:
        reduction = require_number(reduction_value, "curriculum.time_reduction")
    elif reduction_value is not None:
        raise FreezeArtifactError("censored time-to-threshold reduction must be null")
    else:
        reduction = None
    interval = require_mapping(
        evidence.get("paired_long_accuracy_interval"), "curriculum.accuracy_interval"
    )
    lower = require_number(interval.get("lower"), "curriculum.accuracy_interval.lower")
    upper = require_number(interval.get("upper"), "curriculum.accuracy_interval.upper")
    if not -1.0 <= lower <= upper <= 1.0:
        raise FreezeArtifactError("curriculum accuracy interval is invalid")
    if interval.get("resamples") != 10_000 or interval.get("seed") != 20_260_811:
        raise FreezeArtifactError(
            "curriculum accuracy interval must use the fixed bootstrap"
        )
    static_change = require_number(
        evidence.get("example15_accuracy_change"), "curriculum.example15_change"
    )
    if not -1.0 <= static_change <= 1.0:
        raise FreezeArtifactError("Example 15 accuracy change is outside [-1, 1]")
    stable = evidence.get("all_paired_runs_stable")
    if not isinstance(stable, bool):
        raise FreezeArtifactError("curriculum paired stability decision is missing")
    time_gate = complete and reduction is not None and reduction >= 0.20
    accuracy_gate = lower > 0.0
    static_gate = static_change >= -0.01
    expected_gates = {
        "time_gate_passed": time_gate,
        "accuracy_gate_passed": accuracy_gate,
        "static_control_gate_passed": static_gate,
    }
    if any(evidence.get(name) is not result for name, result in expected_gates.items()):
        raise FreezeArtifactError("curriculum recorded gate decisions are inconsistent")
    return stable and static_gate and (time_gate or accuracy_gate)


def validate_curriculum_adoption(
    document: Mapping[str, Any], expected: Mapping[str, object]
) -> tuple[bool, dict[str, object]]:
    """Require the declared adoption to equal the precommitted decision rule."""
    ensure_finite_tree(document, "curriculum")
    validate_header(document, "temporal_credit_curriculum_adoption")
    if document.get("status") != "completed":
        raise FreezeArtifactError("curriculum adoption decision did not complete")
    provenance = validate_decision_provenance(document, "curriculum")
    _require_selected_config(document, expected, "curriculum")
    adoption = document.get("adoption")
    if not isinstance(adoption, bool) or adoption != _adoption_evidence(document):
        raise FreezeArtifactError(
            "curriculum adoption is inconsistent with decision evidence"
        )
    return adoption, provenance
