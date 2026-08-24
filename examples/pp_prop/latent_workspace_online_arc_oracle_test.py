"""Tests for public-ARC-aligned online PP-prop training."""

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
        "examples.pp_prop.latent_workspace_online_arc_oracle"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task(index: int, *, role: str = "train") -> ArcTask:
    first = index % 9 + 1
    second = (index + 1) % 9 + 1
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((first, 0),)), ArcGrid(((second, 0),))),
            ArcPair(ArcGrid(((second, 0),)), ArcGrid(((first, 0),))),
        ),
        test=(ArcPair(ArcGrid(((first, 0),)), ArcGrid(((second, 0),))),),
        task_id=f"{role}-{index:03d}",
    )


def test_config_validates_schedule_counts_and_serializes_paths(tmp_path) -> None:
    subject = _subject()
    config = subject.OnlineARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        output_dir=tmp_path / "output",
        device="cpu",
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        encoder_width=8,
        hidden_width=12,
    )

    payload = config.to_dict()

    assert payload["source_manifest"] == str(tmp_path / "sources.json")
    assert payload["output_dir"] == str(tmp_path / "output")
    assert payload["sampling_seed"] == 12108
    assert payload["validation_fold_index"] == 0
    with pytest.raises(ValueError, match="divide"):
        subject.OnlineARCOracleConfig(
            source_manifest=tmp_path / "sources.json",
            output_dir=tmp_path,
            training_updates=3,
            training_chunk_size=2,
        )
    with pytest.raises(ValueError, match="smaller"):
        subject.OnlineARCOracleConfig(
            source_manifest=tmp_path / "sources.json",
            output_dir=tmp_path,
            validation_task_count=4,
            expected_training_task_count=4,
        )
    with pytest.raises(ValueError, match="fold"):
        subject.OnlineARCOracleConfig(
            source_manifest=tmp_path / "sources.json",
            output_dir=tmp_path,
            validation_task_count=2,
            validation_fold_index=2,
            expected_training_task_count=4,
        )


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ({"device": "tpu"}, ValueError, "device"),
        ({"training_updates": True}, TypeError, "positive integer"),
        ({"training_updates": 0}, ValueError, "positive integer"),
        ({"seed": 1.5}, TypeError, "nonnegative integer"),
        ({"sampling_seed": -1}, ValueError, "nonnegative integer"),
        ({"learning_rate": "bad"}, TypeError, "positive finite real"),
        ({"learning_rate": 0.0}, ValueError, "positive finite real"),
        ({"trace_decay": 1.1}, ValueError, "at most"),
        ({"augment": 1}, TypeError, "boolean"),
    ],
)
def test_config_rejects_invalid_scalar_contracts(
    tmp_path, override: dict[str, object], error: type[Exception], message: str
) -> None:
    subject = _subject()
    values = {
        "source_manifest": tmp_path / "sources.json",
        "output_dir": tmp_path / "output",
    }
    values.update(override)

    with pytest.raises(error, match=message):
        subject.OnlineARCOracleConfig(**values)


def test_pilot_scope_is_deterministic_disjoint_and_never_scores_evaluation() -> None:
    subject = _subject()
    training = tuple(_task(index) for index in range(6))
    evaluation = tuple(_task(index, role="evaluation") for index in range(3))

    first = subject.select_arc_scope(
        training,
        evaluation,
        validation_task_count=2,
        validation_fold_index=0,
        complete=False,
    )
    second = subject.select_arc_scope(
        training,
        evaluation,
        validation_task_count=2,
        validation_fold_index=0,
        complete=False,
    )

    assert [task.task_id for task in first.fit_tasks] == [
        task.task_id for task in second.fit_tasks
    ]
    assert [task.task_id for task in first.score_tasks] == [
        task.task_id for task in second.score_tasks
    ]
    assert len(first.fit_tasks) == 4
    assert len(first.score_tasks) == 2
    assert set(first.fit_task_ids).isdisjoint(first.score_task_ids)
    assert set(first.score_task_ids).isdisjoint(task.task_id for task in evaluation)
    assert first.scope == "held_out_public_training"

    complete = subject.select_arc_scope(
        training,
        evaluation,
        validation_task_count=2,
        validation_fold_index=0,
        complete=True,
    )
    assert complete.fit_task_ids == tuple(task.task_id for task in training)
    assert complete.score_task_ids == tuple(task.task_id for task in evaluation)
    assert complete.scope == "complete_arc_evaluation"

    next_fold = subject.select_arc_scope(
        training,
        evaluation,
        validation_task_count=2,
        validation_fold_index=1,
        complete=False,
    )
    assert set(first.score_task_ids).isdisjoint(next_fold.score_task_ids)


def test_tiny_arc_run_binds_fit_scope_compiler_and_artifact(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    training = tuple(_task(index) for index in range(4))
    evaluation = tuple(_task(index, role="evaluation") for index in range(2))

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
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.OnlineARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        output_dir=tmp_path / "artifact",
        device="cpu",
        seed=41,
        sampling_seed=51,
        validation_task_count=2,
        validation_fold_index=0,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
        training_updates=1,
        training_chunk_size=1,
        training_batch_size=1,
        learning_rate=0.001,
        encoder_width=4,
        hidden_width=6,
        recurrent_layers=2,
        augment=False,
    )

    result = subject.run_arc_oracle(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "b" * 40
    assert result["source_dirty"] is False
    assert result["scope"]["name"] == "held_out_public_training"
    assert result["scope"]["fit_task_count"] == 2
    assert result["evaluation"]["task_count"] == 2
    assert result["evaluation"]["query_count"] == 2
    assert set(result["scope"]["fit_task_ids"]).isdisjoint(
        result["scope"]["score_task_ids"]
    )
    assert not any(
        task_id.startswith("evaluation-")
        for task_id in result["scope"]["score_task_ids"]
    )
    assert result["training"]["finite"] is True
    assert np.isfinite(result["training"]["losses"]).all()
    assert result["model"]["parameters_moved"] is True
    assert result["learner"]["compiler"]["recurrent_excluded_paths"] == []
    assert stored["configuration"] == config.to_dict()
    assert stored["evaluation"]["candidate_sha256"] == result["evaluation"][
        "candidate_sha256"
    ]


def test_run_rejects_wrong_type_missing_device_and_corpus_count(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    with pytest.raises(TypeError, match="OnlineARCOracleConfig"):
        subject.run_arc_oracle(object())

    config = subject.OnlineARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        output_dir=tmp_path / "output",
    )
    cpu_device = jax.devices()[0]
    monkeypatch.setattr(subject.jax, "devices", lambda _: [])
    with pytest.raises(RuntimeError, match="unavailable"):
        subject.run_arc_oracle(config)

    monkeypatch.setattr(subject.jax, "devices", lambda _: [cpu_device])
    monkeypatch.setattr(
        subject,
        "load_corpora",
        lambda _: SimpleNamespace(training=(_task(1),), evaluation=(_task(2),)),
    )
    with pytest.raises(ValueError, match="Expected 399 training"):
        subject.run_arc_oracle(config)


def test_main_binds_cli_and_prints_single_strict_count(
    tmp_path, monkeypatch, capsys
) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {
            "evaluation": {"strict_task_pass_at_1_count": 2},
            "pilot_gate_passed": True,
            "acceptance_threshold_passed": False,
        }

    monkeypatch.setattr(subject, "run_arc_oracle", fake_run)
    exit_code = subject.main(
        [
            "--source-manifest",
            str(tmp_path / "sources.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-augment",
        ]
    )

    assert exit_code == 0
    assert observed["config"].augment is False
    output = capsys.readouterr().out
    assert '"strict_task_pass_at_1_count":2' in output
    assert '"pilot_gate_passed":true' in output
