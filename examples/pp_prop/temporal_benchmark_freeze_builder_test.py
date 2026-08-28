"""Focused tests for strict development-selection freezing."""

from __future__ import annotations

import copy
import msgspec_json
import pathlib

import pytest

from temporal_benchmark_freeze_builder import build_frozen_selection
from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_freeze_schema import (
    frozen_config_overrides,
    validate_frozen_selection,
)
from temporal_benchmark_freeze import main
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES

COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
RATES = {"readout": 0.01, "feedforward": 0.001, "recurrent": 0.001}
TRACES = {
    "short": {"x": 5.0, "f": 10.0},
    "medium": {"x": 20.0, "f": 30.0},
    "long": {"x": 60.0, "f": 100.0},
}


def _settings() -> dict[str, object]:
    return {
        "source_commit": COMMIT,
        "container_image_digest": DIGEST,
        "development_bundles": list(DEVELOPMENT_BUNDLES),
        "device": "gpu",
        "neurons": 96,
        "degree": 8,
        "batch_size": 32,
    }


def _winner(kind: str, selection: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": kind,
        "development_only": True,
        "sealed_test": False,
        "settings": _settings(),
        "winner": {
            **selection,
            "status": "accepted",
            "rank": 1,
            "bundle_scores": [
                {"bundle_id": bundle, "validation_nll": 0.1}
                for bundle in DEVELOPMENT_BUNDLES
            ],
        },
    }


def _decision_base(kind: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": kind,
        "development_only": True,
        "sealed_test": False,
        "development_bundles": list(DEVELOPMENT_BUNDLES),
        "device": "gpu",
        "neurons": 96,
        "degree": 8,
        "batch_size": 32,
        "provenance": {
            "source_commit": COMMIT,
            "source_dirty": True,
            "container_image_digest": DIGEST,
        },
    }


def _documents() -> dict[str, dict[str, object]]:
    gain = _winner("temporal_credit_gain_search_winner", {"gain": 0.8, "index": 1})
    gain["settings"]["fixed_configuration"] = {
        "learning_rates": {
            "readout": 0.003,
            "feedforward": 0.001,
            "recurrent": 0.0003,
        },
        "trace_half_life_x_steps": 60.0,
        "trace_half_life_f_steps": 60.0,
        "gradient_clip_norms": {
            "readout": 1.0,
            "feedforward": 1.0,
            "recurrent": 1.0,
        },
        "recurrent_weight_decay": 0.0,
    }
    optimizer = _winner(
        "temporal_credit_optimizer_search_winner",
        {"learning_rates": RATES, "grid_index": 23},
    )
    optimizer["settings"].update(
        {
            "gain": 0.8,
            "search_kind": "optimizer",
            "trace_half_life_x_steps": 60.0,
            "trace_half_life_f_steps": 60.0,
            "gradient_clip_norms": {
                "readout": 1.0,
                "feedforward": 1.0,
                "recurrent": 1.0,
            },
            "recurrent_weight_decay": 0.0,
        }
    )
    decay = _winner(
        "temporal_credit_weight_decay_search_winner",
        {"recurrent_weight_decay": 1e-5, "index": 1},
    )
    decay["settings"]["fixed_configuration"] = {
        "gain": 0.8,
        "learning_rates": RATES,
        "trace_half_life_x_steps": 60.0,
        "trace_half_life_f_steps": 60.0,
        "gradient_clip_norms": {
            "readout": 1.0,
            "feedforward": 1.0,
            "recurrent": 1.0,
        },
    }
    trace = {
        "schema_version": 1,
        "kind": "temporal_credit_trace_half_life_selection",
        "development_only": True,
        "sealed_test": False,
        "settings": {
            **_settings(),
            "fixed_gain": 0.8,
            "fixed_learning_rates": RATES,
            "fixed_gradient_clip_norms": {
                "readout": 1.0,
                "feedforward": 1.0,
                "recurrent": 1.0,
            },
            "recurrent_weight_decay": 1e-5,
        },
        "selections": {
            horizon: {
                "updates": {"short": 200, "medium": 400, "long": 800}[horizon],
                "trace_half_life_x_steps": pair["x"],
                "trace_half_life_f_steps": pair["f"],
            }
            for horizon, pair in TRACES.items()
        },
    }
    prior = {"gain": 0.8, "learning_rates": RATES, "recurrent_weight_decay": 1e-5}
    clip = _decision_base("temporal_credit_clip_selection")
    clip.update(
        {
            "trigger_threshold": 0.25,
            "selected_config": prior,
            "groups": {
                group: {
                    "observed_clip_event_fractions": {
                        bundle: 0.1 for bundle in DEVELOPMENT_BUNDLES
                    },
                    "triggered": False,
                    "candidates": [],
                    "selected_clip_norm": 1.0,
                }
                for group in ("readout", "feedforward", "recurrent")
            },
        }
    )
    curriculum = _decision_base("temporal_credit_curriculum_adoption")
    curriculum.update(
        {
            "status": "completed",
            "selected_config": {
                **prior,
                "trace_half_lives": TRACES,
                "gradient_clip_norms": {
                    "readout": 1.0,
                    "feedforward": 1.0,
                    "recurrent": 1.0,
                },
            },
            "adoption": False,
            "decision_evidence": {
                "time_to_0_80_complete": True,
                "time_to_0_80_reduction_fraction": 0.1,
                "paired_long_accuracy_interval": {
                    "estimate": 0.0,
                    "lower": -0.02,
                    "upper": 0.02,
                    "resamples": 10_000,
                    "seed": 20_260_811,
                },
                "example15_accuracy_change": 0.0,
                "all_paired_runs_stable": True,
                "time_gate_passed": False,
                "accuracy_gate_passed": False,
                "static_control_gate_passed": True,
            },
        }
    )
    return {
        "gain": gain,
        "optimizer": optimizer,
        "weight_decay": decay,
        "trace": trace,
        "clip": clip,
        "curriculum": curriculum,
    }


def _write_documents(
    directory: pathlib.Path, documents: dict[str, dict[str, object]]
) -> dict[str, pathlib.Path]:
    paths = {}
    for role, document in documents.items():
        path = directory / f"{role}.json"
        path.write_text(msgspec_json.dumps(document) + "\n", encoding="utf-8")
        paths[role] = path
    return paths


def test_builds_hash_pinned_frozen_selection(tmp_path: pathlib.Path) -> None:
    frozen = build_frozen_selection(_write_documents(tmp_path, _documents()))
    selected = validate_frozen_selection(frozen)
    assert selected["gain"] == 0.8
    assert selected["curriculum"] is False
    assert selected["trace_half_lives"] == TRACES
    provenance = frozen["selection_provenance"]
    assert provenance["selection_source_dirty"] is True
    assert len(provenance["input_artifacts"]) == 6


def test_translates_selection_to_typed_benchmark_overrides(
    tmp_path: pathlib.Path,
) -> None:
    frozen = build_frozen_selection(_write_documents(tmp_path, _documents()))
    overrides = frozen_config_overrides(frozen, "medium")
    assert overrides["trace_half_life_x_steps"] == 20.0
    assert overrides["trace_half_life_f_steps"] == 30.0
    assert overrides["learning_rates"].readout == 0.01
    assert overrides["curriculum_trace_half_lives"].long.f == 100.0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_rejects_nonfinite_input(tmp_path: pathlib.Path, bad_value: float) -> None:
    documents = _documents()
    documents["gain"]["unexpected"] = bad_value
    with pytest.raises(FreezeArtifactError, match="non-finite"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_rejects_materialized_sealed_metrics(tmp_path: pathlib.Path) -> None:
    documents = _documents()
    documents["optimizer"]["sealed_test_metrics"] = {"accuracy": 1.0}
    with pytest.raises(FreezeArtifactError, match="sealed test metrics"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_rejects_source_commit_mismatch(tmp_path: pathlib.Path) -> None:
    documents = _documents()
    documents["trace"]["settings"]["source_commit"] = "c" * 40
    with pytest.raises(FreezeArtifactError, match="mismatched source"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_rejects_dirty_state_mismatch(tmp_path: pathlib.Path) -> None:
    documents = _documents()
    documents["curriculum"]["provenance"]["source_dirty"] = False
    with pytest.raises(FreezeArtifactError, match="dirty state"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_rejects_search_declared_dirty_state_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    documents = _documents()
    documents["optimizer"]["settings"]["source_dirty"] = False
    with pytest.raises(FreezeArtifactError, match="conflicts"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_rejects_inconsistent_clip_trigger(tmp_path: pathlib.Path) -> None:
    documents = _documents()
    recurrent = documents["clip"]["groups"]["recurrent"]
    recurrent["observed_clip_event_fractions"][DEVELOPMENT_BUNDLES[0]] = 0.3
    with pytest.raises(FreezeArtifactError, match="trigger decision"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_accepts_triggered_disabled_clip(tmp_path: pathlib.Path) -> None:
    documents = _documents()
    recurrent = documents["clip"]["groups"]["recurrent"]
    recurrent["observed_clip_event_fractions"][DEVELOPMENT_BUNDLES[0]] = 0.3
    recurrent.update(
        {
            "triggered": True,
            "candidates": [0.5, 1.0, 2.0, None],
            "selected_clip_norm": None,
        }
    )
    documents["curriculum"]["selected_config"]["gradient_clip_norms"][
        "recurrent"
    ] = None
    frozen = build_frozen_selection(_write_documents(tmp_path, documents))
    assert frozen["selected_config"]["gradient_clip_norms"]["recurrent"] is None


def test_rejects_curriculum_adoption_not_supported_by_evidence(
    tmp_path: pathlib.Path,
) -> None:
    documents = _documents()
    documents["curriculum"]["adoption"] = True
    with pytest.raises(FreezeArtifactError, match="inconsistent"):
        build_frozen_selection(_write_documents(tmp_path, documents))


def test_accepts_curriculum_adoption_gate(tmp_path: pathlib.Path) -> None:
    documents = _documents()
    documents["curriculum"]["adoption"] = True
    evidence = documents["curriculum"]["decision_evidence"]
    evidence["time_to_0_80_reduction_fraction"] = 0.2
    evidence["time_gate_passed"] = True
    frozen = build_frozen_selection(_write_documents(tmp_path, documents))
    assert frozen["selected_config"]["curriculum"] is True


def test_censored_time_reduction_does_not_pass(tmp_path: pathlib.Path) -> None:
    documents = copy.deepcopy(_documents())
    evidence = documents["curriculum"]["decision_evidence"]
    evidence["time_to_0_80_complete"] = False
    evidence["time_to_0_80_reduction_fraction"] = None
    frozen = build_frozen_selection(_write_documents(tmp_path, documents))
    assert frozen["selected_config"]["curriculum"] is False


def test_unstable_curriculum_evidence_blocks_adoption(
    tmp_path: pathlib.Path,
) -> None:
    documents = _documents()
    evidence = documents["curriculum"]["decision_evidence"]
    interval = evidence["paired_long_accuracy_interval"]
    interval["lower"] = 0.01
    evidence["accuracy_gate_passed"] = True
    evidence["all_paired_runs_stable"] = False
    frozen = build_frozen_selection(_write_documents(tmp_path, documents))
    assert frozen["selected_config"]["curriculum"] is False


def test_cli_writes_valid_frozen_artifact(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_documents(tmp_path, _documents())
    output = tmp_path / "frozen.json"
    arguments = []
    flags = {
        "gain": "--gain-winner",
        "optimizer": "--optimizer-winner",
        "weight_decay": "--weight-decay-winner",
        "trace": "--trace-selection",
        "clip": "--clip-selection",
        "curriculum": "--curriculum-adoption",
    }
    for role, flag in flags.items():
        arguments.extend((flag, str(paths[role])))
    assert main([*arguments, "--output", str(output)]) == 0
    validate_frozen_selection(msgspec_json.loads(output.read_text(encoding="utf-8")))
    assert '"frozen_for_sealed_evaluation": true' in capsys.readouterr().out


def test_frozen_consumer_rejects_tampered_config(tmp_path: pathlib.Path) -> None:
    frozen = build_frozen_selection(_write_documents(tmp_path, _documents()))
    frozen["selected_config"]["gain"] = 0.7
    with pytest.raises(FreezeArtifactError, match="outside its grid"):
        validate_frozen_selection(frozen)


def test_frozen_consumer_rejects_tampered_input_digest(
    tmp_path: pathlib.Path,
) -> None:
    frozen = build_frozen_selection(_write_documents(tmp_path, _documents()))
    references = frozen["selection_provenance"]["input_artifacts"]
    references["gain"]["sha256"] = "0" * 63
    with pytest.raises(FreezeArtifactError, match="artifact digest"):
        validate_frozen_selection(frozen)
