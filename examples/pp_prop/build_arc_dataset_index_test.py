"""Tests for the build-time ARC dataset indexer."""

from __future__ import annotations

import msgspec_json
from pathlib import Path
import sys

from examples.pp_prop import build_arc_dataset_index as index_module
from examples.pp_prop.build_arc_dataset_index import build_arc_dataset_indexes
from examples.pp_prop.latent_workspace_task import DatasetSource, load_dataset_source


def _payload(color: int) -> dict[str, object]:
    return {
        "train": [{"input": [[color]], "output": [[color]]}],
        "test": [{"input": [[color]], "output": [[color]]}],
    }


def test_build_arc_dataset_indexes_writes_runtime_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "arc"
    training = dataset / "data" / "training"
    evaluation = dataset / "data" / "evaluation"
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    (training / "train.json").write_text(msgspec_json.dumps(_payload(1)), encoding="utf-8")
    (evaluation / "eval.json").write_text(msgspec_json.dumps(_payload(2)), encoding="utf-8")
    output = tmp_path / "index"
    manifest_path = tmp_path / "example21-sources.json"

    report = build_arc_dataset_indexes(
        dataset,
        output,
        manifest_path,
        version="test-revision",
        runtime_index_root="/datasets/arc/index",
        expected_training_tasks=1,
        expected_evaluation_tasks=1,
    )

    assert report == {"training_tasks": 1, "evaluation_tasks": 1}
    manifest = msgspec_json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["format"] for item in manifest["sources"]] == [
        "indexed_json",
        "indexed_json",
    ]
    assert [item["path"] for item in manifest["sources"]] == [
        "/datasets/arc/index/training.index.json",
        "/datasets/arc/index/evaluation.index.json",
    ]
    for item, filename in zip(
        manifest["sources"],
        ("training.index.json", "evaluation.index.json"),
        strict=True,
    ):
        item["path"] = str(output / filename)
        loaded = load_dataset_source(DatasetSource(**item))
        assert len(loaded.tasks) == 1
        assert loaded.manifest.files[0].sha256


def test_build_arc_dataset_indexes_enforces_expected_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "arc"
    (dataset / "data" / "training").mkdir(parents=True)
    (dataset / "data" / "evaluation").mkdir(parents=True)
    (dataset / "data" / "training" / "train.json").write_text(
        msgspec_json.dumps(_payload(1)), encoding="utf-8"
    )
    (dataset / "data" / "evaluation" / "eval.json").write_text(
        msgspec_json.dumps(_payload(2)), encoding="utf-8"
    )

    try:
        build_arc_dataset_indexes(
            dataset,
            tmp_path / "index",
            tmp_path / "manifest.json",
            version="test-revision",
            expected_training_tasks=399,
            expected_evaluation_tasks=400,
        )
    except ValueError as error:
        assert "expected 399 training tasks and 400 evaluation tasks" in str(error)
    else:
        raise AssertionError("Unexpected task counts must fail the image build. Set Unexpected task counts to fail the image build.")


def test_index_builder_cli_emits_count_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dataset = tmp_path / "arc"
    training = dataset / "data" / "training"
    evaluation = dataset / "data" / "evaluation"
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    (training / "train.json").write_text(msgspec_json.dumps(_payload(1)), encoding="utf-8")
    (evaluation / "eval.json").write_text(msgspec_json.dumps(_payload(2)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_arc_dataset_index.py",
            "--dataset-root",
            str(dataset),
            "--output-root",
            str(tmp_path / "index"),
            "--manifest-path",
            str(tmp_path / "manifest.json"),
            "--version",
            "test-revision",
            "--expected-training-tasks",
            "1",
            "--expected-evaluation-tasks",
            "1",
        ],
    )

    index_module._main()

    assert msgspec_json.loads(capsys.readouterr().out) == {
        "evaluation_tasks": 1,
        "training_tasks": 1,
    }
