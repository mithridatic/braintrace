"""Shared structural and provenance checks for development selections."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES

IMAGE_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
GROUPS = ("readout", "feedforward", "recurrent")
HORIZONS = ("short", "medium", "long")


def ensure_finite_tree(value: object, location: str = "root") -> None:
    """Reject non-finite numeric leaves anywhere in an input artifact."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require_number(value, location)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            ensure_finite_tree(child, f"{location}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            ensure_finite_tree(child, f"{location}[{index}]")


def require_mapping(value: object, location: str) -> Mapping[str, Any]:
    """Return a mapping or fail with its artifact location."""
    if not isinstance(value, Mapping):
        raise FreezeArtifactError(f"{location} must be an object. Set {location} to an object.")
    return value


def require_number(value: object, location: str) -> float:
    """Return a finite non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeArtifactError(f"{location} must be numeric. Set {location} to numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise FreezeArtifactError(f"{location} must be finite. Use finite values for {location}.")
    return result


def validate_header(document: Mapping[str, Any], kind: str) -> None:
    """Require one unsealed schema-1 development artifact of ``kind``."""
    if document.get("schema_version") != 1 or document.get("kind") != kind:
        raise FreezeArtifactError(f"Expected {kind} schema version 1. Fix the input condition named in the error, then rerun the operation.")
    if document.get("development_only") is not True:
        raise FreezeArtifactError(f"{kind} must be development-only. Set {kind} to development-only.")
    if document.get("sealed_test") is not False:
        raise FreezeArtifactError(f"{kind} cannot contain sealed-test evidence. Fix the input condition named in the error, then rerun the operation.")
    if contains_sealed_metrics(document):
        raise FreezeArtifactError(f"{kind} materialized sealed test metrics. Fix the input condition named in the error, then rerun the operation.")


def contains_sealed_metrics(value: object) -> bool:
    """Return whether any nested sealed-test metric payload was materialized."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "sealed_test_metrics" and child is not None:
                return True
            if contains_sealed_metrics(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_sealed_metrics(child) for child in value)
    return False


def validate_common_settings(
    settings: Mapping[str, Any], location: str
) -> dict[str, object]:
    """Extract the construction and provenance shared across search stages."""
    commit = settings.get("source_commit")
    digest = settings.get("container_image_digest")
    bundles = settings.get("development_bundles")
    if not isinstance(commit, str) or not SOURCE_COMMIT_PATTERN.fullmatch(commit):
        raise FreezeArtifactError(f"{location}.source_commit must be a full hash. Set {location}.source_commit to a full hash.")
    if not isinstance(digest, str) or not IMAGE_DIGEST_PATTERN.fullmatch(digest):
        raise FreezeArtifactError(f"{location}.container_image_digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    if bundles != list(DEVELOPMENT_BUNDLES):
        raise FreezeArtifactError(f"{location}.development_bundles do not match. Fix the input condition named in the error, then rerun the operation.")
    shared = {
        name: settings.get(name)
        for name in ("device", "neurons", "degree", "batch_size")
    }
    if shared["device"] != "gpu":
        raise FreezeArtifactError(f"{location}.device must be gpu. Set {location}.device to gpu.")
    for name in ("neurons", "degree", "batch_size"):
        if isinstance(shared[name], bool) or not isinstance(shared[name], int):
            raise FreezeArtifactError(f"{location}.{name} must be an integer. Set {location}.{name} to an integer.")
    return {"source_commit": commit, "container_image_digest": digest, **shared}


def validate_decision_provenance(
    document: Mapping[str, Any], location: str
) -> dict[str, object]:
    """Validate provenance required from post-search decision artifacts."""
    provenance = require_mapping(document.get("provenance"), f"{location}.provenance")
    settings = dict(provenance)
    settings["development_bundles"] = document.get("development_bundles")
    for name in ("device", "neurons", "degree", "batch_size"):
        settings[name] = document.get(name)
    common = validate_common_settings(settings, f"{location}.provenance")
    dirty = provenance.get("source_dirty")
    if not isinstance(dirty, bool):
        raise FreezeArtifactError(f"{location}.provenance.source_dirty must be boolean. Set {location}.provenance.source_dirty to boolean.")
    common["source_dirty"] = dirty
    return common
