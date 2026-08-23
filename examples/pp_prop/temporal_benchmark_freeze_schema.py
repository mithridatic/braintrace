"""Validate frozen-selection schema before applying it to sealed runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from temporal_benchmark_config import (
    CurriculumTraceHalfLives,
    GradientClipNorms,
    HorizonName,
    LearningRates,
    TraceHalfLives,
)
from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_freeze_validation import (
    GROUPS,
    HORIZONS,
    IMAGE_DIGEST_PATTERN,
    SOURCE_COMMIT_PATTERN,
    ensure_finite_tree,
    require_mapping,
    require_number,
    validate_header,
)
from temporal_benchmark_gain_search_config import DEVELOPMENT_GAIN_VALUES
from temporal_benchmark_search_config import ORDERED_LEARNING_RATE_GRID
from temporal_benchmark_trace_search_config import HORIZON_TRACE_GRIDS
from temporal_benchmark_weight_decay_search_config import DEVELOPMENT_WEIGHT_DECAYS

FREEZE_SCHEMA_VERSION = 1
FREEZE_KIND = "temporal_credit_frozen_development_selection"
INPUT_KINDS = {
    "gain": "temporal_credit_gain_search_winner",
    "optimizer": "temporal_credit_optimizer_search_winner",
    "weight_decay": "temporal_credit_weight_decay_search_winner",
    "trace": "temporal_credit_trace_half_life_selection",
    "clip": "temporal_credit_clip_selection",
    "curriculum": "temporal_credit_curriculum_adoption",
}


def _selected_config(value: object) -> dict[str, object]:
    selected = require_mapping(value, "freeze.selected_config")
    if set(selected) != {
        "gain",
        "learning_rates",
        "recurrent_weight_decay",
        "trace_half_lives",
        "gradient_clip_norms",
        "curriculum",
    }:
        raise FreezeArtifactError("Frozen selected_config has an invalid shape. Set the named field to a value in the stated range, then rerun the operation.")
    gain = require_number(selected["gain"], "freeze.gain")
    decay = require_number(selected["recurrent_weight_decay"], "freeze.weight_decay")
    rates = require_mapping(selected["learning_rates"], "freeze.learning_rates")
    if set(rates) != set(GROUPS):
        raise FreezeArtifactError("Frozen learning-rate groups are incomplete. Fix the input condition named in the error, then rerun the operation.")
    rate_tuple = tuple(
        require_number(rates.get(group), f"freeze.rate.{group}") for group in GROUPS
    )
    if gain not in DEVELOPMENT_GAIN_VALUES or decay not in DEVELOPMENT_WEIGHT_DECAYS:
        raise FreezeArtifactError("Frozen gain or weight decay is outside its grid. Set the named field to a value in the stated range, then rerun the operation.")
    if rate_tuple not in ORDERED_LEARNING_RATE_GRID:
        raise FreezeArtifactError("Frozen learning rates are outside their grid. Set the named field to a value in the stated range, then rerun the operation.")
    clips = require_mapping(selected["gradient_clip_norms"], "freeze.clips")
    if set(clips) != set(GROUPS) or any(
        isinstance(item, bool) or item not in (0.5, 1.0, 2.0, None)
        for item in clips.values()
    ):
        raise FreezeArtifactError("Frozen clip norms are invalid. Set the named field to a value in the stated range, then rerun the operation.")
    traces = require_mapping(selected["trace_half_lives"], "freeze.traces")
    grids = {grid.horizon: grid.half_lives for grid in HORIZON_TRACE_GRIDS}
    if set(traces) != set(HORIZONS):
        raise FreezeArtifactError("Frozen trace horizons are incomplete. Fix the input condition named in the error, then rerun the operation.")
    for horizon in HORIZONS:
        pair = require_mapping(traces[horizon], f"freeze.traces.{horizon}")
        if set(pair) != {"x", "f"} or any(
            require_number(pair.get(name), f"freeze.{horizon}.{name}")
            not in grids[horizon]
            for name in ("x", "f")
        ):
            raise FreezeArtifactError(f"Frozen {horizon} trace pair is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    if not isinstance(selected.get("curriculum"), bool):
        raise FreezeArtifactError("Frozen curriculum decision must be boolean. Set Frozen curriculum decision to boolean.")
    return dict(selected)


def _selection_provenance(value: object) -> None:
    provenance = require_mapping(value, "freeze.selection_provenance")
    commit = provenance.get("source_commit")
    if not isinstance(commit, str) or not SOURCE_COMMIT_PATTERN.fullmatch(commit):
        raise FreezeArtifactError("Frozen provenance source commit is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    if not isinstance(provenance.get("selection_source_dirty"), bool):
        raise FreezeArtifactError("Frozen provenance dirty state is missing. Provide the missing value or resource, then rerun the operation.")
    construction = require_mapping(
        provenance.get("construction"), "freeze.construction"
    )
    if construction.get("device") != "gpu":
        raise FreezeArtifactError("Frozen construction device must be gpu. Set Frozen construction device to gpu.")
    for name in ("neurons", "degree", "batch_size"):
        if isinstance(construction.get(name), bool) or not isinstance(
            construction.get(name), int
        ):
            raise FreezeArtifactError(f"Frozen construction {name} must be an integer. Set Frozen construction {name} to an integer.")
    neurons = int(construction["neurons"])
    degree = int(construction["degree"])
    batch_size = int(construction["batch_size"])
    if neurons <= 0 or not 0 < degree < neurons or batch_size <= 0:
        raise FreezeArtifactError("Frozen construction dimensions are invalid. Set the named field to a value in the stated range, then rerun the operation.")
    references = require_mapping(
        provenance.get("input_artifacts"), "freeze.input_artifacts"
    )
    if set(references) != set(INPUT_KINDS):
        raise FreezeArtifactError("Frozen input artifact references are incomplete. Fix the input condition named in the error, then rerun the operation.")
    for role, kind in INPUT_KINDS.items():
        reference = require_mapping(references[role], f"freeze.input_artifacts.{role}")
        digest = reference.get("container_image_digest")
        sha256 = reference.get("sha256")
        if reference.get("kind") != kind or reference.get("schema_version") != 1:
            raise FreezeArtifactError(f"Frozen {role} reference has a wrong header. Fix the input condition named in the error, then rerun the operation.")
        if not isinstance(reference.get("filename"), str) or not reference["filename"]:
            raise FreezeArtifactError(f"Frozen {role} filename is invalid. Set the named field to a value in the stated range, then rerun the operation.")
        if not isinstance(digest, str) or not IMAGE_DIGEST_PATTERN.fullmatch(digest):
            raise FreezeArtifactError(f"Frozen {role} image digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise FreezeArtifactError(f"Frozen {role} artifact digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")


def validate_frozen_selection(document: Mapping[str, Any]) -> dict[str, object]:
    """Return selected overrides only after complete schema validation."""
    ensure_finite_tree(document, "freeze")
    validate_header(document, FREEZE_KIND)
    if document.get("frozen_for_sealed_evaluation") is not True:
        raise FreezeArtifactError("Selection is not frozen for sealed evaluation. Fix the input condition named in the error, then rerun the operation.")
    selected = _selected_config(document.get("selected_config"))
    _selection_provenance(document.get("selection_provenance"))
    return selected


def frozen_config_overrides(
    document: Mapping[str, Any], horizon: HorizonName
) -> dict[str, object]:
    """Translate a valid freeze into typed ``TemporalBenchmarkConfig`` overrides."""
    selected = validate_frozen_selection(document)
    rates = require_mapping(selected["learning_rates"], "freeze.learning_rates")
    clips = require_mapping(selected["gradient_clip_norms"], "freeze.clips")
    traces = require_mapping(selected["trace_half_lives"], "freeze.traces")
    pairs = {
        name: require_mapping(traces[name], f"freeze.traces.{name}")
        for name in HORIZONS
    }
    curriculum_traces = CurriculumTraceHalfLives(
        **{
            name: TraceHalfLives(float(pair["x"]), float(pair["f"]))
            for name, pair in pairs.items()
        }
    )
    selected_pair = pairs[horizon]
    return {
        "gain": require_number(selected["gain"], "freeze.gain"),
        "learning_rates": LearningRates(*(float(rates[name]) for name in GROUPS)),
        "recurrent_weight_decay": require_number(
            selected["recurrent_weight_decay"], "freeze.weight_decay"
        ),
        "gradient_clip_norms": GradientClipNorms(*(clips[name] for name in GROUPS)),
        "trace_half_life_x_steps": float(selected_pair["x"]),
        "trace_half_life_f_steps": float(selected_pair["f"]),
        "curriculum_trace_half_lives": curriculum_traces,
        "curriculum": selected["curriculum"],
    }
