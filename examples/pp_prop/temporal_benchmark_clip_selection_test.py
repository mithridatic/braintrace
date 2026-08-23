"""Tests for evidence-only clip selection and its fail-closed boundaries."""

from __future__ import annotations

import msgspec_json
import pathlib

import pytest

from temporal_benchmark_clip_selection import main
from temporal_benchmark_clip_selection_builder import build_clip_selection
from temporal_benchmark_config import config_to_dict
from temporal_benchmark_freeze_decisions import CLIP_CANDIDATES
from temporal_benchmark_freeze_io import FreezeArtifactError
from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_weight_decay_search_config import (
    WEIGHT_DECAY_SEARCH_STAGE,
    WeightDecaySearchSettings,
    expected_weight_decay_benchmark_config,
    ordered_weight_decay_candidates,
    weight_decay_search_settings_document,
)

COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64


def _write(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(msgspec_json.dumps(value) + "\n", encoding="utf-8")


def _settings(root: pathlib.Path) -> WeightDecaySearchSettings:
    return WeightDecaySearchSettings(
        source_root=root,
        output_directory=root,
        benchmark_script=root / "benchmark.py",
        manifest_path=root / "manifest.json",
        python_executable="python",
        container_image_digest=DIGEST,
        source_commit=COMMIT,
    )


def _events(group: str, bundle_index: int, triggered: bool) -> list[int]:
    counts = {
        "readout": (0, 0, 0),
        "feedforward": (0, 0, 0),
        "recurrent": ((201 if triggered else 72), 0, 80),
    }
    count = counts[group][bundle_index]
    return [1] * count + [0] * (WEIGHT_DECAY_SEARCH_STAGE.updates - count)


def _fixture(
    root: pathlib.Path, *, triggered: bool = False
) -> tuple[pathlib.Path, dict[str, object], dict[str, pathlib.Path]]:
    settings = _settings(root)
    candidate = ordered_weight_decay_candidates()[0]
    raw_paths: dict[str, pathlib.Path] = {}
    scores = []
    selected_config: dict[str, object] | None = None
    for bundle_index, bundle_id in enumerate(DEVELOPMENT_BUNDLES):
        relative = pathlib.Path("raw") / f"{bundle_id}.json"
        path = root / relative
        config = config_to_dict(
            expected_weight_decay_benchmark_config(settings, candidate, bundle_id)
        )
        selected_config = {
            "gain": config["gain"],
            "learning_rates": config["learning_rates"],
            "recurrent_weight_decay": candidate.weight_decay,
        }
        telemetry = {
            group: {"clip_event": _events(group, bundle_index, triggered)}
            for group in ("readout", "feedforward", "recurrent")
        }
        raw = {
            "schema_version": 1,
            "sealed_test": False,
            "status": "completed",
            "environment": {
                "backend": "gpu",
                "container_image_digest": DIGEST,
                "source_commit": COMMIT,
                "source_dirty": True,
            },
            "result": {
                "status": "completed",
                "bundle_id": bundle_id,
                "config": config,
                "sealed_test_metrics": None,
                "optimizer_telemetry": telemetry,
            },
        }
        _write(path, raw)
        raw_paths[bundle_id] = path
        scores.append({"bundle_id": bundle_id, "raw_path": relative.as_posix()})
    winner = {
        "schema_version": 1,
        "kind": "temporal_credit_weight_decay_search_winner",
        "development_only": True,
        "sealed_test": False,
        "settings": weight_decay_search_settings_document(settings),
        "winner": {
            "index": 0,
            "recurrent_weight_decay": 0.0,
            "status": "accepted",
            "rank": 1,
            "rejection_reasons": [],
            "bundle_scores": scores,
        },
    }
    winner_path = root / "winner.json"
    _write(winner_path, winner)
    assert selected_config is not None
    return winner_path, selected_config, raw_paths


def _search_evidence(
    path: pathlib.Path,
    selected_config: dict[str, object],
    selected_clip_norm: float | None = None,
) -> pathlib.Path:
    evidence = {
        "schema_version": 1,
        "kind": "temporal_credit_clip_search_selection",
        "status": "completed",
        "development_only": True,
        "sealed_test": False,
        "provenance": {
            "source_commit": COMMIT,
            "source_dirty": True,
            "container_image_digest": DIGEST,
        },
        "development_bundles": list(DEVELOPMENT_BUNDLES),
        "device": "gpu",
        "neurons": 96,
        "degree": 8,
        "batch_size": 32,
        "selected_config": selected_config,
        "groups": {
            "recurrent": {
                "candidates": CLIP_CANDIDATES,
                "winner": {
                    "selected_clip_norm": selected_clip_norm,
                    "status": "accepted",
                    "rank": 1,
                    "rejection_reasons": [],
                    "bundle_scores": [
                        {"bundle_id": bundle_id, "validation_nll": 0.1}
                        for bundle_id in DEVELOPMENT_BUNDLES
                    ],
                },
            }
        },
    }
    _write(path, evidence)
    return path


def test_derives_no_search_decision_from_winner_raws(tmp_path: pathlib.Path) -> None:
    winner, _, _ = _fixture(tmp_path)
    decision = build_clip_selection(winner)
    recurrent = decision["groups"]["recurrent"]
    assert list(recurrent["observed_clip_event_fractions"].values()) == [0.09, 0.0, 0.1]
    assert recurrent["triggered"] is False
    assert recurrent["candidates"] == []
    assert recurrent["selected_clip_norm"] == 1.0
    assert decision["provenance"]["source_dirty"] is True
    assert len(decision["input_artifacts"]["raw_results"]) == 3


def test_cli_atomically_writes_decision(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    winner, _, _ = _fixture(tmp_path)
    output = tmp_path / "selection.json"
    assert main(["--weight-decay-winner", str(winner), "--output", str(output)]) == 0
    assert (
        msgspec_json.loads(output.read_text(encoding="utf-8"))["kind"]
        == "temporal_credit_clip_selection"
    )
    assert '"triggered": false' in capsys.readouterr().out


def test_triggered_group_requires_explicit_search_evidence(
    tmp_path: pathlib.Path,
) -> None:
    winner, _, _ = _fixture(tmp_path, triggered=True)
    with pytest.raises(FreezeArtifactError, match="explicit candidate-search"):
        build_clip_selection(winner)


def test_triggered_group_consumes_explicit_search_winner(
    tmp_path: pathlib.Path,
) -> None:
    winner, selected_config, _ = _fixture(tmp_path, triggered=True)
    evidence = _search_evidence(tmp_path / "clip-search.json", selected_config)
    decision = build_clip_selection(winner, evidence)
    recurrent = decision["groups"]["recurrent"]
    assert recurrent["triggered"] is True
    assert recurrent["candidates"] == CLIP_CANDIDATES
    assert recurrent["selected_clip_norm"] is None


def test_untriggered_group_rejects_unnecessary_search_evidence(
    tmp_path: pathlib.Path,
) -> None:
    winner, selected_config, _ = _fixture(tmp_path)
    evidence = _search_evidence(tmp_path / "clip-search.json", selected_config)
    with pytest.raises(FreezeArtifactError, match="forbidden"):
        build_clip_selection(winner, evidence)


@pytest.mark.parametrize("drift", ("sealed", "nonfinite", "config", "commit", "dirty"))
def test_rejects_raw_evidence_drift(tmp_path: pathlib.Path, drift: str) -> None:
    winner, _, raw_paths = _fixture(tmp_path)
    path = raw_paths[DEVELOPMENT_BUNDLES[0]]
    raw = msgspec_json.loads(path.read_text(encoding="utf-8"))
    if drift == "sealed":
        raw["result"]["sealed_test_metrics"] = {"accuracy": 1.0}
    elif drift == "nonfinite":
        raw["unexpected"] = float("inf")
    elif drift == "config":
        raw["result"]["config"]["gain"] = 1.2
    elif drift == "commit":
        raw["environment"]["source_commit"] = "c" * 40
    else:
        raw["environment"]["source_dirty"] = "true"
    _write(path, raw)
    with pytest.raises(FreezeArtifactError):
        build_clip_selection(winner)


def test_rejects_raw_path_escape(tmp_path: pathlib.Path) -> None:
    winner, _, _ = _fixture(tmp_path)
    document = msgspec_json.loads(winner.read_text(encoding="utf-8"))
    document["winner"]["bundle_scores"][0]["raw_path"] = "../outside.json"
    _write(winner, document)
    with pytest.raises(FreezeArtifactError, match="escapes"):
        build_clip_selection(winner)


def test_rejects_invalid_clip_event_stream(tmp_path: pathlib.Path) -> None:
    winner, _, raw_paths = _fixture(tmp_path)
    path = raw_paths[DEVELOPMENT_BUNDLES[1]]
    raw = msgspec_json.loads(path.read_text(encoding="utf-8"))
    raw["result"]["optimizer_telemetry"]["readout"]["clip_event"][0] = 2
    _write(path, raw)
    with pytest.raises(FreezeArtifactError, match="binary"):
        build_clip_selection(winner)


def test_rejects_triggered_search_configuration_drift(tmp_path: pathlib.Path) -> None:
    winner, selected_config, _ = _fixture(tmp_path, triggered=True)
    evidence_path = _search_evidence(tmp_path / "clip-search.json", selected_config)
    evidence = msgspec_json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["selected_config"]["gain"] = 1.2
    _write(evidence_path, evidence)
    with pytest.raises(FreezeArtifactError, match="configuration drifted"):
        build_clip_selection(winner, evidence_path)
