"""Tests for the V47 surface-diversified curriculum experiment runner."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_task import ArcGrid, ArcPair, ArcTask


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_diverse_experiment"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _arc_task(pairs, task_id: str) -> ArcTask:
    built = tuple(
        ArcPair(ArcGrid(input_rows), ArcGrid(output_rows))
        for input_rows, output_rows in pairs
    )
    return ArcTask(train=built[:-1], test=built[-1:], task_id=task_id)


def _recolor_task(index: int, *, role: str = "train") -> ArcTask:
    first = index % 9 + 1
    second = (index + 1) % 9 + 1
    return _arc_task(
        [
            ([[first, 0]], [[second, 0]]),
            ([[second, 0]], [[first, 0]]),
            ([[first, 0]], [[second, 0]]),
        ],
        f"{role}-{index:03d}",
    )


def test_config_validation(tmp_path) -> None:
    subject = _subject()
    base = dict(
        output_dir=tmp_path / "out",
        source_manifest=tmp_path / "sources.json",
    )
    with pytest.raises(ValueError, match="different"):
        subject.DiverseExperimentConfig(
            **base, device="cpu", synthetic_seed=44108, holdout_seed=44108
        )
    with pytest.raises(ValueError, match="divide"):
        subject.DiverseExperimentConfig(
            **base, device="cpu", training_updates=7, training_chunk_size=4
        )
    with pytest.raises(ValueError, match="device"):
        subject.DiverseExperimentConfig(**base, device="tpu")
    config = subject.DiverseExperimentConfig(**base, device="cpu", synthetic_task_count=12)
    payload = config.to_dict()
    assert payload["synthetic_task_count"] == 12
    assert payload["max_grid_size"] == 30
    assert payload["min_demonstrations"] == 2
    assert payload["max_demonstrations"] == 6
    assert payload["trace_decay"] == pytest.approx(2.0 ** (-1.0 / 40.0))


def test_in_library_classifier_covers_every_family() -> None:
    subject = _subject()
    copy_task = _arc_task(
        [
            ([[1, 2], [0, 3]], [[1, 2], [0, 3]]),
            ([[4, 0], [0, 0]], [[4, 0], [0, 0]]),
            ([[0, 5]], [[0, 5]]),
        ],
        "copy-000",
    )
    dihedral_task = _arc_task(
        [
            ([[1, 1], [0, 3]], [[1, 3], [1, 0]]),
            ([[2, 2], [0, 4]], [[2, 4], [2, 0]]),
            ([[5, 5], [0, 6]], [[5, 6], [5, 0]]),
        ],
        "dihedral-000",
    )
    crop_task = _arc_task(
        [
            ([[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[2]]),
            ([[0, 0, 0, 0], [0, 3, 3, 0], [0, 0, 0, 0]], [[3, 3]]),
            ([[0, 0, 0], [0, 4, 0], [0, 5, 0]], [[4], [5]]),
        ],
        "crop-000",
    )
    upscale_task = _arc_task(
        [
            ([[1]], [[1, 1], [1, 1]]),
            ([[2]], [[2, 2], [2, 2]]),
            ([[3]], [[3, 3], [3, 3]]),
        ],
        "upscale-000",
    )
    other_task = _arc_task(
        [
            ([[1, 2], [3, 4]], [[4, 1], [2, 3]]),
            ([[1, 0], [0, 1]], [[1, 1], [0, 0]]),
            ([[2, 2], [1, 1]], [[1, 2], [2, 1]]),
        ],
        "other-000",
    )

    assert subject.in_library_family(copy_task) == "copy"
    assert subject.in_library_family(_recolor_task(0)) == "recolor"
    assert subject.in_library_family(dihedral_task) == "dihedral"
    assert subject.in_library_family(crop_task) == "crop"
    assert subject.in_library_family(upscale_task) == "upscale"
    assert subject.in_library_family(other_task) is None
    with pytest.raises(TypeError):
        subject.in_library_family("copy-000")
    unlabeled = ArcTask(
        train=(ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),))),),
        test=(ArcPair(ArcGrid(((1,),)), None),),
        task_id="unlabeled-000",
    )
    with pytest.raises(ValueError, match="labelled"):
        subject.in_library_family(unlabeled)


def test_select_in_library_groups_and_preserves_corpus_order() -> None:
    subject = _subject()
    other_task = _arc_task(
        [
            ([[1, 2], [3, 4]], [[4, 1], [2, 3]]),
            ([[1, 0], [0, 1]], [[1, 1], [0, 0]]),
            ([[2, 2], [1, 1]], [[1, 2], [2, 1]]),
        ],
        "train-other",
    )
    training = (
        _recolor_task(1),
        other_task,
        _recolor_task(2),
        _recolor_task(3),
    )
    grouped = subject.select_in_library_tasks(training)

    assert sorted(grouped) == ["recolor"]
    assert [task.task_id for task in grouped["recolor"]] == [
        "train-001",
        "train-002",
        "train-003",
    ]
    with pytest.raises(ValueError):
        subject.select_in_library_tasks(())


def test_v4_family_summary() -> None:
    subject = _subject()
    summary = subject._v4_family_summary(
        ["synthetic-v4:crop:000001", "synthetic-v4:copy:000002"]
    )
    assert summary["family_counts"] == {"copy": 1, "crop": 1}
    with pytest.raises(ValueError, match="synthetic-v4"):
        subject._v4_family_summary(["synthetic-v3:crop:000001"])


def test_tiny_v47_run_binds_scopes_leaves_gates_and_artifact(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    training = tuple(_recolor_task(index) for index in range(4))
    evaluation = tuple(
        _recolor_task(index, role="evaluation") for index in range(2)
    )

    class Manifest:
        def __init__(self, role: str):
            self.role = role

        def to_dict(self):
            return {"source": {"role": self.role}, "valid_task_count": 4}

    corpora = SimpleNamespace(
        training=training,
        evaluation=evaluation,
        loaded=(
            SimpleNamespace(manifest=Manifest("train")),
            SimpleNamespace(manifest=Manifest("evaluation")),
        ),
    )
    monkeypatch.setattr(subject, "load_corpora", lambda _: corpora)
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "7" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.DiverseExperimentConfig(
        output_dir=tmp_path / "artifact",
        source_manifest=tmp_path / "sources.json",
        device="cpu",
        seed=41,
        synthetic_seed=51,
        holdout_seed=61,
        synthetic_task_count=12,
        holdout_task_count=12,
        max_grid_size=6,
        min_demonstrations=2,
        max_demonstrations=2,
        training_updates=2,
        training_chunk_size=2,
        training_batch_size=2,
        memory_width=8,
        expert_count=12,
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
    )

    result = subject.run_diverse_experiment(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "7" * 40
    assert result["source_dirty"] is False
    assert result["data"]["training_schema_version"] == "direct_synthetic_curriculum_v4"
    assert result["scopes"]["in_library"]["task_count"] == 4
    assert result["scopes"]["in_library"]["family_counts"] == {"recolor": 4}
    assert result["scopes"]["fold_zero"]["task_count"] == 2
    assert result["evaluation_in_library"]["task_count"] == 4
    assert result["evaluation_fold_zero"]["task_count"] == 2
    assert not any(
        task_id.startswith("evaluation-")
        for scope in result["scopes"].values()
        for task_id in scope.get("task_ids", scope.get("families", {}))
    )
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    assert all(result["model"]["parameter_leaves_moved"].values())
    assert len(result["model"]["parameter_leaves_moved"]) == 18
    assert result["learner"]["compiler"]["recurrent_excluded_paths"] == []
    assert result["training"]["finite"] is True
    for gate in (
        "mechanism_gate_passed",
        "anti_collapse_gate_passed",
        "synthetic_learning_gate_passed",
        "l4_recognition_gate_passed",
    ):
        assert isinstance(result[gate], bool)
    assert (config.output_dir / "checkpoint.npz").exists()
    assert stored["configuration"] == config.to_dict()
    assert stored["checkpoint"]["parameter_sha256"] == result["checkpoint"][
        "parameter_sha256"
    ]
