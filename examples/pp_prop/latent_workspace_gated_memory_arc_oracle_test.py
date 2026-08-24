"""Tests for V45 public-ARC gated-memory transfer."""

from __future__ import annotations

import hashlib
import importlib
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_gated_memory_model import (
    MODEL_INPUT_WIDTH,
    GatedMemoryConfig,
    PhaseSeparatedGatedMemoryRNN,
)
from examples.pp_prop.latent_workspace_online_training import (
    parameter_digest,
    save_online_checkpoint,
)
from examples.pp_prop.latent_workspace_task import ArcGrid, ArcPair, ArcTask


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_gated_memory_arc_oracle"
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


def _checkpoint(tmp_path, *, seed: int = 41) -> tuple[object, str, str]:
    model = PhaseSeparatedGatedMemoryRNN(
        GatedMemoryConfig(
            input_width=MODEL_INPUT_WIDTH,
            memory_width=8,
            expert_count=12,
            seed=seed,
        )
    )
    path = tmp_path / "pretrained.npz"
    digest = save_online_checkpoint(model, path)
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == parameter_digest(model)
    return path, digest, file_digest


def test_v45_config_binds_checkpoint_scope_schedule_and_gate(tmp_path) -> None:
    subject = _subject()
    checkpoint, digest, file_digest = _checkpoint(tmp_path)
    config = subject.GatedMemoryARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        pretrained_checkpoint=checkpoint,
        expected_pretrained_parameter_sha256=digest,
        expected_pretrained_file_sha256=file_digest,
        output_dir=tmp_path / "output",
        device="cpu",
        validation_start_index=0,
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        memory_width=8,
        expert_count=12,
    )

    payload = config.to_dict()

    assert payload["pretrained_checkpoint"] == str(checkpoint)
    assert payload["expected_pretrained_parameter_sha256"] == digest
    assert payload["expected_pretrained_file_sha256"] == file_digest
    assert payload["sampling_seed"] == 26108
    assert payload["minimum_strict_task_count"] == 4
    with pytest.raises(ValueError, match="divide"):
        subject.GatedMemoryARCOracleConfig(
            source_manifest=tmp_path / "sources.json",
            pretrained_checkpoint=checkpoint,
            expected_pretrained_parameter_sha256=digest,
            expected_pretrained_file_sha256=file_digest,
            output_dir=tmp_path,
            training_updates=3,
            training_chunk_size=2,
        )
    with pytest.raises(ValueError, match="fit inside"):
        subject.GatedMemoryARCOracleConfig(
            source_manifest=tmp_path / "sources.json",
            pretrained_checkpoint=checkpoint,
            expected_pretrained_parameter_sha256=digest,
            expected_pretrained_file_sha256=file_digest,
            output_dir=tmp_path,
            validation_start_index=3,
            validation_task_count=2,
            expected_training_task_count=4,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        subject.GatedMemoryARCOracleConfig(
            source_manifest=tmp_path / "sources.json",
            pretrained_checkpoint=checkpoint,
            expected_pretrained_parameter_sha256="bad",
            expected_pretrained_file_sha256=file_digest,
            output_dir=tmp_path,
        )


def test_v45_scope_is_exact_disjoint_and_never_scores_evaluation() -> None:
    subject = _subject()
    training = tuple(_task(index) for index in range(6))
    evaluation = tuple(_task(index, role="evaluation") for index in range(3))

    scope = subject.select_transfer_scope(
        training,
        evaluation,
        validation_start_index=0,
        validation_task_count=2,
    )
    ordered = tuple(
        sorted(training, key=subject.canonical_task_fingerprint)
    )

    assert scope.score_task_ids == tuple(task.task_id for task in ordered[:2])
    assert len(scope.fit_tasks) == 4
    assert set(scope.fit_task_ids).isdisjoint(scope.score_task_ids)
    assert set(scope.score_task_ids).isdisjoint(
        task.task_id for task in evaluation
    )


def test_tiny_v45_run_binds_pretraining_scope_leaves_and_artifact(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    checkpoint, digest, file_digest = _checkpoint(tmp_path)
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
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "7" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.GatedMemoryARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        pretrained_checkpoint=checkpoint,
        expected_pretrained_parameter_sha256=digest,
        expected_pretrained_file_sha256=file_digest,
        output_dir=tmp_path / "artifact",
        device="cpu",
        seed=41,
        sampling_seed=51,
        validation_start_index=0,
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        learning_rate=0.001,
        memory_width=8,
        expert_count=12,
        augment=False,
        minimum_strict_task_count=1,
    )

    result = subject.run_gated_memory_arc_oracle(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "7" * 40
    assert result["source_dirty"] is False
    assert result["scope"]["fit_task_count"] == 2
    assert result["scope"]["score_task_count"] == 2
    assert set(result["scope"]["fit_task_ids"]).isdisjoint(
        result["scope"]["score_task_ids"]
    )
    assert not any(
        task_id.startswith("evaluation-")
        for task_id in result["scope"]["score_task_ids"]
    )
    assert result["pretraining_checkpoint"]["parameter_sha256"] == digest
    assert result["pretraining_checkpoint"]["file_sha256"] == file_digest
    assert result["model"]["parameter_sha256_before"] == digest
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    unmoved_leaves = {
        name
        for name, moved in result["model"]["parameter_leaves_moved"].items()
        if not moved
    }
    assert not unmoved_leaves, unmoved_leaves
    assert len(result["model"]["parameter_leaves_moved"]) == 18
    assert result["learner"]["compiler"]["recurrent_excluded_paths"] == []
    assert result["training"]["finite"] is True
    assert np.isfinite(result["training"]["losses"]).all()
    assert result["evaluation"]["task_count"] == 2
    assert result["evaluation"]["query_count"] == 2
    assert stored["configuration"] == config.to_dict()
    assert stored["evaluation"]["candidate_sha256"] == result["evaluation"][
        "candidate_sha256"
    ]


@pytest.mark.parametrize("mismatch", ["parameter", "file"])
def test_v45_rejects_pretrained_checkpoint_mismatch(
    tmp_path, monkeypatch, mismatch: str
) -> None:
    subject = _subject()
    checkpoint, digest, file_digest = _checkpoint(tmp_path)
    config = subject.GatedMemoryARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        pretrained_checkpoint=checkpoint,
        expected_pretrained_parameter_sha256=("f" * 64 if mismatch == "parameter" else digest),
        expected_pretrained_file_sha256=("e" * 64 if mismatch == "file" else file_digest),
        output_dir=tmp_path / "output",
        device="cpu",
        seed=41,
        validation_start_index=0,
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
        training_updates=1,
        training_chunk_size=1,
        training_batch_size=1,
        memory_width=8,
        expert_count=12,
    )
    monkeypatch.setattr(
        subject,
        "load_corpora",
        lambda _: SimpleNamespace(
            training=tuple(_task(index) for index in range(4)),
            evaluation=tuple(_task(index, role="evaluation") for index in range(2)),
            loaded=(),
        ),
    )

    with pytest.raises(ValueError, match=mismatch):
        subject.run_gated_memory_arc_oracle(config)


def test_v45_rejects_wrong_type_and_corpus_count(tmp_path, monkeypatch) -> None:
    subject = _subject()
    with pytest.raises(TypeError, match="GatedMemoryARCOracleConfig"):
        subject.run_gated_memory_arc_oracle(object())

    checkpoint, digest, file_digest = _checkpoint(tmp_path)
    config = subject.GatedMemoryARCOracleConfig(
        source_manifest=tmp_path / "sources.json",
        pretrained_checkpoint=checkpoint,
        expected_pretrained_parameter_sha256=digest,
        expected_pretrained_file_sha256=file_digest,
        output_dir=tmp_path / "output",
        device="cpu",
        validation_start_index=0,
        validation_task_count=2,
        expected_training_task_count=4,
        expected_evaluation_task_count=2,
        training_updates=1,
        training_chunk_size=1,
        training_batch_size=1,
        memory_width=8,
        expert_count=12,
    )
    monkeypatch.setattr(
        subject,
        "load_corpora",
        lambda _: SimpleNamespace(training=(_task(1),), evaluation=(_task(2),)),
    )
    with pytest.raises(ValueError, match="Expected 4 training"):
        subject.run_gated_memory_arc_oracle(config)


def test_v45_main_binds_cli_and_prints_single_strict_count(
    tmp_path, monkeypatch, capsys
) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {
            "evaluation": {"strict_task_pass_at_1_count": 4},
            "pilot_gate_passed": True,
        }

    monkeypatch.setattr(subject, "run_gated_memory_arc_oracle", fake_run)
    exit_code = subject.main(
        [
            "--source-manifest",
            str(tmp_path / "sources.json"),
            "--pretrained-checkpoint",
            str(tmp_path / "checkpoint.npz"),
            "--expected-pretrained-parameter-sha256",
            "a" * 64,
            "--expected-pretrained-file-sha256",
            "b" * 64,
            "--output-dir",
            str(tmp_path / "output"),
            "--no-augment",
        ]
    )

    assert exit_code == 0
    assert observed["config"].augment is False
    assert observed["config"].validation_start_index == 0
    assert observed["config"].minimum_strict_task_count == 4
    output = capsys.readouterr().out
    assert '"strict_task_pass_at_1_count":4' in output
    assert '"pilot_gate_passed":true' in output
