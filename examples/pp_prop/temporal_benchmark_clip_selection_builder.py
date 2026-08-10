"""Build the clip-trigger decision without launching any benchmark runs."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from temporal_benchmark_clip_selection_evidence import (
    validate_clip_search_evidence,
)
from temporal_benchmark_clip_selection_source import derive_weight_decay_clip_source
from temporal_benchmark_freeze_decisions import (
    CLIP_CANDIDATES,
    CLIP_TRIGGER_THRESHOLD,
    validate_clip_selection,
)
from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_freeze_validation import GROUPS, require_mapping
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES


def _triggered_groups(fractions: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        group
        for group in GROUPS
        if any(
            float(require_mapping(fractions[group], f"fractions.{group}")[bundle])
            > CLIP_TRIGGER_THRESHOLD
            for bundle in DEVELOPMENT_BUNDLES
        )
    )


def build_clip_selection(
    weight_decay_winner: pathlib.Path,
    clip_search_evidence: pathlib.Path | None = None,
) -> dict[str, object]:
    """Derive the trigger decision and fail closed if triggered evidence is absent."""
    source = derive_weight_decay_clip_source(weight_decay_winner.resolve())
    fractions = require_mapping(source["fractions"], "source.fractions")
    triggered = _triggered_groups(fractions)
    if triggered and clip_search_evidence is None:
        raise FreezeArtifactError(
            "clip search triggered but explicit candidate-search evidence is missing"
        )
    if not triggered and clip_search_evidence is not None:
        raise FreezeArtifactError(
            "clip search evidence is forbidden when no parameter group triggered"
        )
    selected: dict[str, float | None] = {group: 1.0 for group in GROUPS}
    search_reference: dict[str, object] | None = None
    if clip_search_evidence is not None:
        winners, search_reference = validate_clip_search_evidence(
            clip_search_evidence.resolve(), source, triggered
        )
        selected.update(winners)
    groups = {
        group: {
            "observed_clip_event_fractions": dict(
                require_mapping(fractions[group], f"fractions.{group}")
            ),
            "triggered": group in triggered,
            "candidates": list(CLIP_CANDIDATES) if group in triggered else [],
            "selected_clip_norm": selected[group],
        }
        for group in GROUPS
    }
    provenance = dict(require_mapping(source["provenance"], "source.provenance"))
    construction = require_mapping(source["construction"], "source.construction")
    document: dict[str, object] = {
        "schema_version": 1,
        "kind": "temporal_credit_clip_selection",
        "status": "completed",
        "development_only": True,
        "sealed_test": False,
        "provenance": provenance,
        "development_bundles": list(DEVELOPMENT_BUNDLES),
        **{
            key: construction[key]
            for key in ("device", "neurons", "degree", "batch_size")
        },
        "trigger_threshold": CLIP_TRIGGER_THRESHOLD,
        "selected_config": source["selected_config"],
        "groups": groups,
        "input_artifacts": {
            **dict(require_mapping(source["input_artifacts"], "source.artifacts")),
            "clip_search_selection": search_reference,
        },
    }
    validate_clip_selection(
        document,
        require_mapping(source["selected_config"], "source.selected_config"),
    )
    return document
