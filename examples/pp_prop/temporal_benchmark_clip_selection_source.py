"""Validate the selected weight-decay raws and derive clip-event fractions."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from temporal_benchmark_config import config_to_dict
from temporal_benchmark_freeze_io import (
    FreezeArtifactError,
    artifact_reference,
    load_artifact,
)
from temporal_benchmark_freeze_validation import (
    GROUPS,
    contains_sealed_metrics,
    ensure_finite_tree,
    require_mapping,
    validate_common_settings,
    validate_header,
)
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_weight_decay_search_config import (
    WeightDecaySearchSettings,
    expected_weight_decay_benchmark_config,
    ordered_weight_decay_candidates,
)


def _search_settings(
    path: pathlib.Path, settings: Mapping[str, Any]
) -> WeightDecaySearchSettings:
    return WeightDecaySearchSettings(
        source_root=path.parent,
        output_directory=path.parent,
        benchmark_script=pathlib.Path(str(settings.get("benchmark_script"))),
        manifest_path=pathlib.Path(str(settings.get("manifest_path"))),
        python_executable="unused-by-clip-selection",
        container_image_digest=str(settings["container_image_digest"]),
        source_commit=str(settings["source_commit"]),
        device=str(settings["device"]),
        neurons=int(settings["neurons"]),
        degree=int(settings["degree"]),
        batch_size=int(settings["batch_size"]),
    )


def _winner(document: Mapping[str, Any]) -> Mapping[str, Any]:
    winner = require_mapping(document.get("winner"), "weight_decay.winner")
    if (
        winner.get("status") != "accepted"
        or winner.get("rank") != 1
        or winner.get("rejection_reasons") != []
    ):
        raise FreezeArtifactError("weight-decay winner is not uniquely accepted")
    return winner


def _validate_fixed_settings(
    settings: Mapping[str, Any], expected: Mapping[str, object]
) -> None:
    fixed = require_mapping(
        settings.get("fixed_configuration"),
        "weight_decay.settings.fixed_configuration",
    )
    expected_fixed = {
        "arm": expected["arm"],
        "horizon": expected["horizon"],
        "updates": expected["updates"],
        "gain": expected["gain"],
        "learning_rates": expected["learning_rates"],
        "trace_half_life_x_steps": expected["trace_half_life_x_steps"],
        "trace_half_life_f_steps": expected["trace_half_life_f_steps"],
        "gradient_clip_norms": expected["gradient_clip_norms"],
        "curriculum": False,
        "gradient_evidence": False,
        "sealed_test": False,
    }
    if fixed != expected_fixed:
        raise FreezeArtifactError("weight-decay winner fixed configuration drifted")


def _raw_path(root: pathlib.Path, relative: object) -> pathlib.Path:
    if not isinstance(relative, str) or not relative:
        raise FreezeArtifactError("weight-decay winner raw path is invalid")
    path = (root / pathlib.PurePosixPath(relative)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise FreezeArtifactError(
            "weight-decay raw path escapes its artifact root"
        ) from error
    return path


def _clip_fraction(telemetry: Mapping[str, Any], group: str, updates: int) -> float:
    group_values = require_mapping(telemetry.get(group), f"telemetry.{group}")
    events = group_values.get("clip_event")
    if not isinstance(events, list) or len(events) != updates:
        raise FreezeArtifactError(f"{group} clip events must cover every update")
    if any(
        isinstance(value, str) or value not in (0, 1, False, True) for value in events
    ):
        raise FreezeArtifactError(f"{group} clip events must be binary")
    return sum(float(value) for value in events) / updates


def _validate_raw(
    path: pathlib.Path,
    expected_config: Mapping[str, object],
    bundle_id: str,
    provenance: Mapping[str, object],
) -> tuple[dict[str, float], bool, dict[str, object]]:
    document = load_artifact(path)
    ensure_finite_tree(document, f"raw.{bundle_id}")
    if document.get("schema_version") != 1 or document.get("sealed_test") is not False:
        raise FreezeArtifactError(f"raw {bundle_id} is not unsealed schema 1")
    if contains_sealed_metrics(document):
        raise FreezeArtifactError(f"raw {bundle_id} materialized sealed metrics")
    result = require_mapping(document.get("result"), f"raw.{bundle_id}.result")
    if document.get("status") != "completed" or result.get("status") != "completed":
        raise FreezeArtifactError(f"raw {bundle_id} did not complete")
    if result.get("bundle_id") != bundle_id or result.get("config") != expected_config:
        raise FreezeArtifactError(f"raw {bundle_id} configuration drifted")
    environment = require_mapping(
        document.get("environment"), f"raw.{bundle_id}.environment"
    )
    actual = (
        environment.get("container_image_digest"),
        environment.get("source_commit"),
    )
    expected = (provenance["container_image_digest"], provenance["source_commit"])
    dirty = environment.get("source_dirty")
    if (
        actual != expected
        or environment.get("backend") != "gpu"
        or not isinstance(dirty, bool)
    ):
        raise FreezeArtifactError(f"raw {bundle_id} provenance drifted")
    telemetry = require_mapping(
        result.get("optimizer_telemetry"), f"raw.{bundle_id}.telemetry"
    )
    updates_value = expected_config["updates"]
    if isinstance(updates_value, bool) or not isinstance(updates_value, int):
        raise FreezeArtifactError("expected update count is invalid")
    updates = updates_value
    fractions = {group: _clip_fraction(telemetry, group, updates) for group in GROUPS}
    return fractions, dirty, artifact_reference(path, document)


def derive_weight_decay_clip_source(winner_path: pathlib.Path) -> dict[str, object]:
    """Return exact selected config, provenance, fractions, and input hashes."""
    document = load_artifact(winner_path)
    ensure_finite_tree(document, "weight_decay")
    validate_header(document, "temporal_credit_weight_decay_search_winner")
    settings = require_mapping(document.get("settings"), "weight_decay.settings")
    common = validate_common_settings(settings, "weight_decay.settings")
    winner = _winner(document)
    index = winner.get("index")
    candidates = ordered_weight_decay_candidates()
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(candidates)
    ):
        raise FreezeArtifactError("weight-decay winner index is invalid")
    candidate = candidates[index]
    if winner.get("recurrent_weight_decay") != candidate.weight_decay:
        raise FreezeArtifactError("weight-decay winner index and value disagree")
    search_settings = _search_settings(winner_path, settings)
    first_expected = config_to_dict(
        expected_weight_decay_benchmark_config(
            search_settings, candidate, DEVELOPMENT_BUNDLES[0]
        )
    )
    _validate_fixed_settings(settings, first_expected)
    scores = winner.get("bundle_scores")
    if not isinstance(scores, list) or len(scores) != len(DEVELOPMENT_BUNDLES):
        raise FreezeArtifactError("weight-decay winner bundle evidence is incomplete")
    fractions: dict[str, dict[str, float]] = {group: {} for group in GROUPS}
    raw_references: dict[str, object] = {}
    dirty_states: list[bool] = []
    for score, bundle_id in zip(scores, DEVELOPMENT_BUNDLES, strict=True):
        score_mapping = require_mapping(score, f"weight_decay.score.{bundle_id}")
        if score_mapping.get("bundle_id") != bundle_id:
            raise FreezeArtifactError("weight-decay winner bundle order drifted")
        path = _raw_path(winner_path.parent, score_mapping.get("raw_path"))
        expected = config_to_dict(
            expected_weight_decay_benchmark_config(
                search_settings, candidate, bundle_id
            )
        )
        bundle_fractions, dirty, reference = _validate_raw(
            path, expected, bundle_id, common
        )
        for group in GROUPS:
            fractions[group][bundle_id] = bundle_fractions[group]
        dirty_states.append(dirty)
        raw_references[bundle_id] = reference
    if len(set(dirty_states)) != 1:
        raise FreezeArtifactError("weight-decay raws disagree on source dirty state")
    return {
        "selected_config": {
            "gain": expected["gain"],
            "learning_rates": expected["learning_rates"],
            "recurrent_weight_decay": candidate.weight_decay,
        },
        "provenance": {
            "source_commit": common["source_commit"],
            "source_dirty": dirty_states[0],
            "container_image_digest": common["container_image_digest"],
        },
        "construction": {
            key: common[key] for key in ("device", "neurons", "degree", "batch_size")
        },
        "fractions": fractions,
        "input_artifacts": {
            "weight_decay_winner": artifact_reference(winner_path, document),
            "raw_results": raw_references,
        },
    }
