"""Tests for the sealed V22 spatial Conv-LIF PP-prop oracle."""

from __future__ import annotations

import importlib
import json

import jax
import numpy as np
import pytest


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_spatial_oracle")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_spatial_oracle_config_validates_and_serializes(tmp_path) -> None:
    subject = _subject()
    config = subject.SpatialOracleConfig(
        output_dir=tmp_path,
        device="cpu",
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        synthetic_task_count=12,
        oracle_task_count=4,
        spatial_channels=2,
    )

    payload = config.to_dict()

    assert payload["output_dir"] == str(tmp_path)
    assert payload["oracle_seed"] == 62108
    with pytest.raises(ValueError, match="divide"):
        subject.SpatialOracleConfig(
            output_dir=tmp_path, training_updates=3, training_chunk_size=2
        )
    with pytest.raises(ValueError, match="different"):
        subject.SpatialOracleConfig(
            output_dir=tmp_path, synthetic_seed=41, oracle_seed=41
        )
    with pytest.raises(ValueError, match="device"):
        subject.SpatialOracleConfig(output_dir=tmp_path, device="tpu")


def test_tiny_spatial_oracle_writes_bound_finite_artifact(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.SpatialOracleConfig(
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
        spatial_channels=1,
    )

    result = subject.run_spatial_oracle(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "b" * 40
    assert result["source_dirty"] is False
    assert result["training"]["finite"] is True
    assert np.isfinite(result["training"]["losses"]).all()
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    assert "recurrent_conv.weight#1" in result["model"]["parameter_leaves_moved"]
    assert result["mechanism_gate_passed"] is False
    assert result["model"]["architecture"]["architecture_version"] == (
        "spatial_conv_lif_v26"
    )
    assert result["evaluation"]["task_count"] == 4
    assert result["evaluation"]["query_count"] == 4
    assert result["checkpoint"]["parameter_sha256"] == (
        result["model"]["parameter_sha256_after"]
    )
    assert stored["configuration"] == config.to_dict()
    assert stored["evaluation"]["candidate_sha256"] == (
        result["evaluation"]["candidate_sha256"]
    )


def test_run_spatial_oracle_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="SpatialOracleConfig"):
        _subject().run_spatial_oracle(object())


def test_spatial_oracle_main_binds_cli_and_prints_gate(
    tmp_path, monkeypatch, capsys
) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {
            "evaluation": {"strict_task_pass_at_1_count": 9},
            "promotion_gate_passed": True,
        }

    monkeypatch.setattr(subject, "run_spatial_oracle", fake_run)
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
            "--spatial-channels",
            "2",
        ]
    )

    assert exit_code == 0
    assert observed["config"].training_updates == 2
    assert observed["config"].spatial_channels == 2
    assert '"promotion_gate_passed":true' in capsys.readouterr().out
