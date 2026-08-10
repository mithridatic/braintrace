"""Validate explicit candidate-search evidence for triggered clip groups."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from temporal_benchmark_freeze_decisions import CLIP_CANDIDATES
from temporal_benchmark_freeze_io import (
    FreezeArtifactError,
    artifact_reference,
    load_artifact,
)
from temporal_benchmark_freeze_validation import (
    ensure_finite_tree,
    require_mapping,
    validate_decision_provenance,
    validate_header,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES


def _winner(value: object, group: str) -> float | None:
    winner = require_mapping(value, f"clip_search.groups.{group}.winner")
    if (
        winner.get("status") != "accepted"
        or winner.get("rank") != 1
        or winner.get("rejection_reasons") != []
    ):
        raise FreezeArtifactError(f"clip-search {group} winner is not accepted")
    scores = winner.get("bundle_scores")
    if not isinstance(scores, list):
        raise FreezeArtifactError(f"clip-search {group} bundle scores are missing")
    bundle_ids = [
        require_mapping(item, f"clip_search.{group}.score").get("bundle_id")
        for item in scores
    ]
    if bundle_ids != list(DEVELOPMENT_BUNDLES):
        raise FreezeArtifactError(f"clip-search {group} bundle scores drifted")
    selected = winner.get("selected_clip_norm")
    if isinstance(selected, bool) or selected not in CLIP_CANDIDATES:
        raise FreezeArtifactError(f"clip-search {group} winner norm is invalid")
    return None if selected is None else float(selected)


def validate_clip_search_evidence(
    path: pathlib.Path,
    source: Mapping[str, Any],
    triggered_groups: tuple[str, ...],
) -> tuple[dict[str, float | None], dict[str, object]]:
    """Validate candidate evidence and return each triggered group's winner."""
    document = load_artifact(path)
    ensure_finite_tree(document, "clip_search")
    validate_header(document, "temporal_credit_clip_search_selection")
    if document.get("status") != "completed":
        raise FreezeArtifactError("clip-search selection did not complete")
    provenance = validate_decision_provenance(document, "clip_search")
    source_provenance = require_mapping(source["provenance"], "source.provenance")
    for name in ("source_commit", "source_dirty"):
        if provenance[name] != source_provenance[name]:
            raise FreezeArtifactError(f"clip-search {name} drifted")
    construction = require_mapping(source["construction"], "source.construction")
    for name in ("device", "neurons", "degree", "batch_size"):
        if provenance[name] != construction[name]:
            raise FreezeArtifactError(f"clip-search construction {name} drifted")
    if document.get("selected_config") != source["selected_config"]:
        raise FreezeArtifactError("clip-search selected configuration drifted")
    groups = require_mapping(document.get("groups"), "clip_search.groups")
    if set(groups) != set(triggered_groups):
        raise FreezeArtifactError("clip-search groups do not match triggered groups")
    selected: dict[str, float | None] = {}
    for group in triggered_groups:
        group_evidence = require_mapping(groups[group], f"clip_search.groups.{group}")
        if group_evidence.get("candidates") != CLIP_CANDIDATES:
            raise FreezeArtifactError(f"clip-search {group} candidates are incomplete")
        selected[group] = _winner(group_evidence.get("winner"), group)
    reference = artifact_reference(path, document)
    reference["container_image_digest"] = provenance["container_image_digest"]
    return selected, reference
