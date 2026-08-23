"""Compose a validated, immutable development-selection freeze document."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from temporal_benchmark_freeze_decisions import (
    validate_clip_selection,
    validate_curriculum_adoption,
)
from temporal_benchmark_freeze_io import (
    FreezeArtifactError,
    artifact_reference,
    load_artifact,
)
from temporal_benchmark_freeze_schema import (
    FREEZE_KIND,
    FREEZE_SCHEMA_VERSION,
    INPUT_KINDS,
    validate_frozen_selection,
)
from temporal_benchmark_freeze_search import validate_search_selections
from temporal_benchmark_freeze_validation import require_mapping

INPUT_ROLES = tuple(INPUT_KINDS)


def _validate_declared_dirty_states(
    documents: Mapping[str, Mapping[str, Any]], expected: bool
) -> None:
    for role in INPUT_ROLES[:4]:
        settings = require_mapping(documents[role].get("settings"), f"{role}.settings")
        if "source_dirty" not in settings:
            continue
        dirty = settings["source_dirty"]
        if not isinstance(dirty, bool) or dirty != expected:
            raise FreezeArtifactError(
                f"{role} source dirty state conflicts with decision evidence. Fix the input condition named in the error, then rerun the operation."
            )


def _consistent_provenance(
    search: Mapping[str, Mapping[str, object]],
    clip: Mapping[str, object],
    curriculum: Mapping[str, object],
) -> dict[str, object]:
    all_items = [*search.values(), clip, curriculum]
    commit = all_items[0]["source_commit"]
    construction_keys = ("device", "neurons", "degree", "batch_size")
    construction = {key: all_items[0][key] for key in construction_keys}
    if any(item["source_commit"] != commit for item in all_items[1:]):
        raise FreezeArtifactError("Selection artifacts use different source commits")
    if any(
        any(item[key] != construction[key] for key in construction_keys)
        for item in all_items[1:]
    ):
        raise FreezeArtifactError("Selection artifacts use different constructions")
    dirty = clip["source_dirty"]
    if curriculum["source_dirty"] != dirty:
        raise FreezeArtifactError("Decision artifacts disagree on source dirty state. Use matching values and structures.")
    return {
        "source_commit": commit,
        "selection_source_dirty": dirty,
        "construction": construction,
    }


def _references(
    paths: Mapping[str, pathlib.Path],
    documents: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    references: dict[str, object] = {}
    for role in INPUT_ROLES:
        reference = artifact_reference(paths[role], documents[role])
        reference["container_image_digest"] = provenance[role]["container_image_digest"]
        references[role] = reference
    return references


def build_frozen_selection(paths: Mapping[str, pathlib.Path]) -> dict[str, object]:
    """Load exact development artifacts and return the frozen configuration."""
    if set(paths) != set(INPUT_ROLES):
        raise FreezeArtifactError("All six named selection artifact paths are required. Fix the input condition named in the error, then rerun the operation.")
    documents = {role: load_artifact(paths[role]) for role in INPUT_ROLES}
    search_result = validate_search_selections(
        {role: documents[role] for role in INPUT_ROLES[:4]}
    )
    selected = dict(require_mapping(search_result["selected_config"], "selected"))
    search_provenance = require_mapping(search_result["provenance"], "provenance")
    clips, clip_provenance = validate_clip_selection(documents["clip"], selected)
    selected["gradient_clip_norms"] = clips
    adoption, curriculum_provenance = validate_curriculum_adoption(
        documents["curriculum"], selected
    )
    selected["curriculum"] = adoption
    decision_provenance = {
        "clip": clip_provenance,
        "curriculum": curriculum_provenance,
    }
    provenance_by_role = {**search_provenance, **decision_provenance}
    common = _consistent_provenance(
        search_provenance, clip_provenance, curriculum_provenance
    )
    _validate_declared_dirty_states(documents, bool(common["selection_source_dirty"]))
    common["input_artifacts"] = _references(paths, documents, provenance_by_role)
    result: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "kind": FREEZE_KIND,
        "development_only": True,
        "sealed_test": False,
        "frozen_for_sealed_evaluation": True,
        "selected_config": selected,
        "selection_provenance": common,
    }
    validate_frozen_selection(result)
    return result
