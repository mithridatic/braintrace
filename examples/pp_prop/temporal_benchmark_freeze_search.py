"""Extract and cross-check exact gain, optimizer, decay, and trace winners."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_freeze_validation import (
    HORIZONS,
    ensure_finite_tree,
    require_mapping,
    require_number,
    validate_common_settings,
    validate_header,
)
from temporal_benchmark_gain_search_config import DEVELOPMENT_GAIN_VALUES
from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    ORDERED_LEARNING_RATE_GRID,
)
from temporal_benchmark_trace_search_config import HORIZON_TRACE_GRIDS
from temporal_benchmark_weight_decay_search_config import DEVELOPMENT_WEIGHT_DECAYS

SEARCH_KINDS = {
    "gain": "temporal_credit_gain_search_winner",
    "optimizer": "temporal_credit_optimizer_search_winner",
    "weight_decay": "temporal_credit_weight_decay_search_winner",
    "trace": "temporal_credit_trace_half_life_selection",
}
INITIAL_CLIPS = {"readout": 1.0, "feedforward": 1.0, "recurrent": 1.0}
PROVISIONAL_RATES = {
    "readout": 0.003,
    "feedforward": 0.001,
    "recurrent": 0.0003,
}


def _winner(document: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    winner = require_mapping(document.get("winner"), f"{role}.winner")
    if winner.get("status") != "accepted" or winner.get("rank") != 1:
        raise FreezeArtifactError(f"{role} winner must be accepted at rank one")
    if winner.get("rejection_reasons") not in (None, []):
        raise FreezeArtifactError(f"{role} winner retains rejection reasons")
    scores = winner.get("bundle_scores")
    if not isinstance(scores, list):
        raise FreezeArtifactError(f"{role} winner lacks bundle scores")
    bundle_ids = [
        require_mapping(item, f"{role}.bundle_score").get("bundle_id")
        for item in scores
    ]
    if bundle_ids != list(DEVELOPMENT_BUNDLES):
        raise FreezeArtifactError(f"{role} winner bundle scores do not match")
    return winner


def _learning_rates(value: object, location: str) -> dict[str, float]:
    rates = require_mapping(value, location)
    result = {
        name: require_number(rates.get(name), f"{location}.{name}")
        for name in ("readout", "feedforward", "recurrent")
    }
    if tuple(result.values()) not in ORDERED_LEARNING_RATE_GRID:
        raise FreezeArtifactError(f"{location} is outside the fixed search grid")
    return result


def _common(document: Mapping[str, Any], role: str) -> dict[str, object]:
    ensure_finite_tree(document, role)
    validate_header(document, SEARCH_KINDS[role])
    settings = require_mapping(document.get("settings"), f"{role}.settings")
    return validate_common_settings(settings, f"{role}.settings")


def _trace_pairs(document: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    selections = require_mapping(document.get("selections"), "trace.selections")
    if set(selections) != set(HORIZONS):
        raise FreezeArtifactError("trace selections must contain every horizon")
    pairs: dict[str, dict[str, float]] = {}
    grids = {grid.horizon: grid for grid in HORIZON_TRACE_GRIDS}
    for horizon in HORIZONS:
        selection = require_mapping(selections[horizon], f"trace.selections.{horizon}")
        x_value = require_number(
            selection.get("trace_half_life_x_steps"), f"trace.{horizon}.x"
        )
        f_value = require_number(
            selection.get("trace_half_life_f_steps"), f"trace.{horizon}.f"
        )
        if selection.get("updates") != grids[horizon].updates:
            raise FreezeArtifactError(f"trace {horizon} update budget is invalid")
        if (
            x_value not in grids[horizon].half_lives
            or f_value not in grids[horizon].half_lives
        ):
            raise FreezeArtifactError(f"trace {horizon} selection is outside its grid")
        pairs[horizon] = {"x": x_value, "f": f_value}
    return pairs


def validate_search_selections(
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Return selected knobs after strict cross-stage consistency checks."""
    if set(documents) != set(SEARCH_KINDS):
        raise FreezeArtifactError(
            "gain, optimizer, weight_decay, and trace artifacts are required"
        )
    provenance = {role: _common(document, role) for role, document in documents.items()}
    commits = {item["source_commit"] for item in provenance.values()}
    construction = [
        {key: item[key] for key in ("device", "neurons", "degree", "batch_size")}
        for item in provenance.values()
    ]
    if len(commits) != 1 or any(item != construction[0] for item in construction[1:]):
        raise FreezeArtifactError(
            "search artifacts have mismatched source or construction provenance"
        )
    gain_winner = _winner(documents["gain"], "gain")
    gain = require_number(gain_winner.get("gain"), "gain.winner.gain")
    if gain not in DEVELOPMENT_GAIN_VALUES:
        raise FreezeArtifactError("selected gain is outside the fixed grid")
    if gain_winner.get("index") != DEVELOPMENT_GAIN_VALUES.index(gain):
        raise FreezeArtifactError("selected gain index does not match its value")
    gain_settings = require_mapping(documents["gain"]["settings"], "gain.settings")
    gain_fixed = require_mapping(
        gain_settings.get("fixed_configuration"),
        "gain.settings.fixed_configuration",
    )
    if (
        _learning_rates(gain_fixed.get("learning_rates"), "gain.settings.rates")
        != PROVISIONAL_RATES
        or gain_fixed.get("trace_half_life_x_steps") != 60.0
        or gain_fixed.get("trace_half_life_f_steps") != 60.0
        or gain_fixed.get("gradient_clip_norms") != INITIAL_CLIPS
        or gain_fixed.get("recurrent_weight_decay") != 0.0
    ):
        raise FreezeArtifactError("gain search did not use its provisional settings")
    optimizer = documents["optimizer"]
    optimizer_settings = require_mapping(optimizer["settings"], "optimizer.settings")
    if (
        require_number(
            optimizer_settings.get("gain"),
            "optimizer.settings.gain",
        )
        != gain
    ):
        raise FreezeArtifactError("optimizer search did not use the selected gain")
    if (
        optimizer_settings.get("search_kind") != "optimizer"
        or optimizer_settings.get("trace_half_life_x_steps") != 60.0
        or optimizer_settings.get("trace_half_life_f_steps") != 60.0
        or optimizer_settings.get("gradient_clip_norms") != INITIAL_CLIPS
        or optimizer_settings.get("recurrent_weight_decay") != 0.0
    ):
        raise FreezeArtifactError("optimizer search fixed settings are invalid")
    optimizer_winner = _winner(optimizer, "optimizer")
    rates = _learning_rates(
        optimizer_winner.get("learning_rates"),
        "optimizer.winner.learning_rates",
    )
    if optimizer_winner.get("grid_index") != ORDERED_LEARNING_RATE_GRID.index(
        tuple(rates.values())
    ):
        raise FreezeArtifactError("optimizer grid index does not match selected rates")
    decay_document = documents["weight_decay"]
    decay_settings = require_mapping(
        decay_document["settings"], "weight_decay.settings"
    )
    fixed = require_mapping(
        decay_settings.get("fixed_configuration"),
        "weight_decay.settings.fixed_configuration",
    )
    if (
        require_number(fixed.get("gain"), "weight_decay.settings.gain") != gain
        or _learning_rates(
            fixed.get("learning_rates"), "weight_decay.settings.learning_rates"
        )
        != rates
    ):
        raise FreezeArtifactError(
            "weight-decay search did not use selected gain and rates"
        )
    if (
        fixed.get("trace_half_life_x_steps") != 60.0
        or fixed.get("trace_half_life_f_steps") != 60.0
        or fixed.get("gradient_clip_norms") != INITIAL_CLIPS
    ):
        raise FreezeArtifactError("weight-decay fixed settings are invalid")
    decay_winner = _winner(decay_document, "weight_decay")
    decay = require_number(
        decay_winner.get("recurrent_weight_decay"),
        "weight_decay.winner.recurrent_weight_decay",
    )
    if decay not in DEVELOPMENT_WEIGHT_DECAYS:
        raise FreezeArtifactError("selected weight decay is outside the fixed grid")
    if decay_winner.get("index") != DEVELOPMENT_WEIGHT_DECAYS.index(decay):
        raise FreezeArtifactError("weight-decay index does not match selected value")
    trace_settings = require_mapping(documents["trace"]["settings"], "trace.settings")
    if (
        require_number(trace_settings.get("fixed_gain"), "trace.settings.fixed_gain")
        != gain
        or _learning_rates(
            trace_settings.get("fixed_learning_rates"),
            "trace.settings.fixed_learning_rates",
        )
        != rates
        or require_number(
            trace_settings.get("recurrent_weight_decay"),
            "trace.settings.recurrent_weight_decay",
        )
        != decay
        or trace_settings.get("fixed_gradient_clip_norms") != INITIAL_CLIPS
    ):
        raise FreezeArtifactError("trace search did not use prior selected knobs")
    return {
        "selected_config": {
            "gain": gain,
            "learning_rates": rates,
            "recurrent_weight_decay": decay,
            "trace_half_lives": _trace_pairs(documents["trace"]),
        },
        "provenance": provenance,
    }
