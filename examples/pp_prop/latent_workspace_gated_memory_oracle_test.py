"""Tests for the sealed V44 phase-separated memory screen."""

from __future__ import annotations

import importlib
import json

import jax
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_gated_memory_oracle"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_v44_oracle_config_validates_and_serializes(tmp_path) -> None:
    subject = _subject()
    config = subject.GatedMemoryOracleConfig(
        output_dir=tmp_path,
        device="cpu",
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        synthetic_task_count=12,
        oracle_task_count=4,
        memory_width=8,
        expert_count=12,
    )

    payload = config.to_dict()

    assert payload["output_dir"] == str(tmp_path)
    assert payload["synthetic_seed"] == 24108
    assert payload["oracle_seed"] == 112108
    assert payload["minimum_strict_task_count"] == 4
    with pytest.raises(ValueError, match="divide"):
        subject.GatedMemoryOracleConfig(
            output_dir=tmp_path, training_updates=3, training_chunk_size=2
        )
    with pytest.raises(ValueError, match="different"):
        subject.GatedMemoryOracleConfig(
            output_dir=tmp_path, synthetic_seed=41, oracle_seed=41
        )
    with pytest.raises(ValueError, match="device"):
        subject.GatedMemoryOracleConfig(output_dir=tmp_path, device="tpu")
    with pytest.raises(ValueError, match="minimum_strict_task_count"):
        subject.GatedMemoryOracleConfig(
            output_dir=tmp_path, minimum_strict_task_count=0
        )


def test_v44_promotion_gate_binds_minimum_and_family_evidence() -> None:
    subject = _subject()
    families = {
        "non_label_families": ["copy", "count"],
        "non_copy_non_label_ids": ["synthetic-v3:count:000001"],
    }

    assert subject.gated_memory_promotion_gate(12, families, True, minimum=12)
    assert not subject.gated_memory_promotion_gate(11, families, True, minimum=12)
    assert not subject.gated_memory_promotion_gate(12, families, False, minimum=12)
    assert not subject.gated_memory_promotion_gate(
        12,
        {"non_label_families": ["copy"], "non_copy_non_label_ids": []},
        True,
        minimum=12,
    )


def test_tiny_v44_oracle_writes_bound_finite_artifact(tmp_path, monkeypatch) -> None:
    subject = _subject()
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "f" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.GatedMemoryOracleConfig(
        output_dir=tmp_path / "oracle",
        device="cpu",
        seed=51,
        training_updates=1,
        training_chunk_size=1,
        training_batch_size=1,
        synthetic_task_count=4,
        synthetic_demonstrations=2,
        synthetic_max_grid_size=6,
        synthetic_seed=61,
        oracle_task_count=4,
        oracle_seed=71,
        learning_rate=0.001,
        memory_width=8,
        expert_count=12,
    )

    result = subject.run_gated_memory_oracle(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "f" * 40
    assert result["source_dirty"] is False
    assert result["training"]["finite"] is True
    assert np.isfinite(result["training"]["losses"]).all()
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    assert all(result["model"]["parameter_leaves_moved"].values())
    assert len(result["model"]["parameter_leaves_moved"]) == 18
    assert result["model"]["architecture"]["architecture_version"] == (
        "phase_separated_gated_memory_v44"
    )
    assert result["evaluation"]["task_count"] == 4
    assert result["evaluation"]["query_count"] == 4
    assert result["checkpoint"]["parameter_sha256"] == (
        result["model"]["parameter_sha256_after"]
    )
    assert result["learner"]["compiler"]["recurrent_excluded_paths"] == []
    assert stored["configuration"] == config.to_dict()
    assert stored["evaluation"]["candidate_sha256"] == (
        result["evaluation"]["candidate_sha256"]
    )


def test_run_v44_oracle_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="GatedMemoryOracleConfig"):
        _subject().run_gated_memory_oracle(object())


def test_v44_oracle_main_binds_cli_and_prints_gate(
    tmp_path, monkeypatch, capsys
) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {
            "evaluation": {"strict_task_pass_at_1_count": 4},
            "promotion_gate_passed": True,
        }

    monkeypatch.setattr(subject, "run_gated_memory_oracle", fake_run)
    exit_code = subject.main(
        [
            "--output-dir",
            str(tmp_path / "cli"),
            "--device",
            "cpu",
            "--training-updates",
            "2",
            "--training-chunk-size",
            "1",
            "--training-batch-size",
            "1",
            "--synthetic-task-count",
            "4",
            "--synthetic-demonstrations",
            "2",
            "--synthetic-max-grid-size",
            "6",
            "--synthetic-seed",
            "31",
            "--oracle-task-count",
            "4",
            "--oracle-seed",
            "41",
            "--memory-width",
            "8",
            "--expert-count",
            "12",
        ]
    )

    assert exit_code == 0
    assert observed["config"].training_updates == 2
    assert observed["config"].memory_width == 8
    assert '"promotion_gate_passed":true' in capsys.readouterr().out
