"""Tests for the sealed V46 paired spatial capability screen."""

from __future__ import annotations

import importlib
import json

import jax
import numpy as np
import pytest


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_paired_spatial_oracle"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_v46_oracle_config_validates_and_serializes(tmp_path) -> None:
    subject = _subject()
    config = subject.PairedSpatialOracleConfig(
        output_dir=tmp_path,
        device="cpu",
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        synthetic_task_count=12,
        oracle_task_count=4,
        spatial_channels=2,
        refinement_steps=2,
    )

    payload = config.to_dict()

    assert payload["output_dir"] == str(tmp_path)
    assert payload["synthetic_seed"] == 27108
    assert payload["oracle_seed"] == 132108
    assert payload["minimum_strict_task_count"] == 5
    assert payload["retention"] == 0.8
    with pytest.raises(ValueError, match="divide"):
        subject.PairedSpatialOracleConfig(
            output_dir=tmp_path, training_updates=3, training_chunk_size=2
        )
    with pytest.raises(ValueError, match="different"):
        subject.PairedSpatialOracleConfig(
            output_dir=tmp_path, synthetic_seed=41, oracle_seed=41
        )
    with pytest.raises(ValueError, match="device"):
        subject.PairedSpatialOracleConfig(output_dir=tmp_path, device="tpu")


def test_v46_promotion_gate_requires_structural_family() -> None:
    subject = _subject()
    structural = {
        "family_counts": {"copy": 4, "complete_corner": 1},
        "non_label_families": ["complete_corner", "copy"],
        "non_copy_non_label_ids": ["synthetic-v3:complete_corner:000001"],
    }
    nonstructural = {
        "family_counts": {"copy": 4, "recolor": 1},
        "non_label_families": ["copy", "recolor"],
        "non_copy_non_label_ids": ["synthetic-v3:recolor:000001"],
    }

    assert subject.paired_spatial_promotion_gate(
        5, structural, True, True, minimum=5
    )
    assert not subject.paired_spatial_promotion_gate(
        4, structural, True, True, minimum=5
    )
    assert not subject.paired_spatial_promotion_gate(
        5, nonstructural, True, True, minimum=5
    )
    assert not subject.paired_spatial_promotion_gate(
        5, structural, False, True, minimum=5
    )
    assert not subject.paired_spatial_promotion_gate(
        5, structural, True, False, minimum=5
    )


def test_tiny_v46_oracle_writes_bound_finite_artifact(tmp_path, monkeypatch) -> None:
    subject = _subject()
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "e" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.PairedSpatialOracleConfig(
        output_dir=tmp_path / "oracle",
        device="cpu",
        seed=51,
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        synthetic_task_count=12,
        synthetic_demonstrations=2,
        synthetic_max_grid_size=6,
        synthetic_seed=61,
        oracle_task_count=4,
        oracle_seed=71,
        learning_rate=0.001,
        spatial_channels=2,
        refinement_steps=2,
        retention=0.8,
    )

    result = subject.run_paired_spatial_oracle(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "e" * 40
    assert result["source_dirty"] is False
    assert result["training"]["finite"] is True
    assert np.isfinite(result["training"]["losses"]).all()
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    assert all(result["model"]["parameter_leaves_moved"].values())
    assert result["model"]["architecture"]["architecture_version"] == (
        "paired_spatial_conv_tanh_v46"
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


def test_run_v46_oracle_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="PairedSpatialOracleConfig"):
        _subject().run_paired_spatial_oracle(object())


def test_v46_oracle_main_binds_cli_and_prints_gate(
    tmp_path, monkeypatch, capsys
) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {
            "evaluation": {"strict_task_pass_at_1_count": 5},
            "promotion_gate_passed": True,
        }

    monkeypatch.setattr(subject, "run_paired_spatial_oracle", fake_run)
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
            "12",
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
            "--spatial-channels",
            "2",
            "--refinement-steps",
            "2",
        ]
    )

    assert exit_code == 0
    assert observed["config"].training_updates == 2
    assert observed["config"].spatial_channels == 2
    assert '"promotion_gate_passed":true' in capsys.readouterr().out
