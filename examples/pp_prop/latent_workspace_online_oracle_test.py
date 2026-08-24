"""Tests for the staged V20 synthetic PP-prop oracle."""

from __future__ import annotations

import importlib
import json

import jax
import numpy as np
import pytest


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_oracle")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def test_config_validates_schedule_and_serializes_paths(tmp_path) -> None:
    subject = _subject()
    config = subject.OnlineOracleConfig(
        output_dir=tmp_path,
        device="cpu",
        training_updates=4,
        training_chunk_size=2,
        training_batch_size=2,
        synthetic_task_count=12,
        oracle_task_count=4,
        encoder_width=8,
        hidden_width=12,
    )

    payload = config.to_dict()

    assert payload["output_dir"] == str(tmp_path)
    assert payload["trace_decay"] == pytest.approx(2.0 ** (-1.0 / 40.0))
    with pytest.raises(ValueError, match="divide"):
        subject.OnlineOracleConfig(
            output_dir=tmp_path, training_updates=3, training_chunk_size=2
        )
    with pytest.raises(ValueError, match="different"):
        subject.OnlineOracleConfig(
            output_dir=tmp_path, synthetic_seed=41, oracle_seed=41
        )
    with pytest.raises(ValueError, match="device"):
        subject.OnlineOracleConfig(output_dir=tmp_path, device="tpu")


def test_exact_family_summary_excludes_label_and_copy() -> None:
    subject = _subject()
    summary = subject.exact_family_summary(
        [
            "synthetic-v3:pattern_label:000001",
            "synthetic-v3:copy:000002",
            "synthetic-v3:crop:000003",
            "synthetic-v3:crop:000004",
            "synthetic-v3:recolor:000005",
        ]
    )

    assert summary["family_counts"] == {
        "copy": 1,
        "crop": 2,
        "pattern_label": 1,
        "recolor": 1,
    }
    assert summary["non_label_families"] == ["copy", "crop", "recolor"]
    assert summary["non_copy_non_label_ids"] == [
        "synthetic-v3:crop:000003",
        "synthetic-v3:crop:000004",
        "synthetic-v3:recolor:000005",
    ]
    with pytest.raises(ValueError, match="synthetic-v3"):
        subject.exact_family_summary(["bad-id"])


def test_tiny_oracle_run_writes_bound_finite_artifact(tmp_path, monkeypatch) -> None:
    subject = _subject()
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.OnlineOracleConfig(
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
        encoder_width=4,
        hidden_width=6,
        recurrent_layers=2,
    )

    result = subject.run_oracle(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "a" * 40
    assert result["source_dirty"] is False
    assert result["training"]["finite"] is True
    assert np.isfinite(result["training"]["losses"]).all()
    assert result["model"]["parameters_moved"] is True
    assert result["model"]["architecture"]["architecture_version"] == (
        "online_task_patch_decoder_v39"
    )
    assert result["learner"]["compiler"]["recurrent_excluded_paths"] == []
    assert "relation_excluded_weight_to_weight" not in result["learner"][
        "compiler"
    ]["diagnostic_kinds"]
    assert result["evaluation"]["task_count"] == 4
    assert result["evaluation"]["query_count"] == 4
    assert result["checkpoint"]["parameter_sha256"] == (
        result["model"]["parameter_sha256_after"]
    )
    assert stored["configuration"] == config.to_dict()
    assert stored["evaluation"]["candidate_sha256"] == (
        result["evaluation"]["candidate_sha256"]
    )


def test_run_oracle_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="OnlineOracleConfig"):
        _subject().run_oracle(object())


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"BRAINTRACE_SOURCE_REVISION": "a" * 40}, "set together"),
        (
            {
                "BRAINTRACE_SOURCE_REVISION": "bad",
                "BRAINTRACE_SOURCE_DIRTY": "false",
            },
            "40 hex",
        ),
        (
            {
                "BRAINTRACE_SOURCE_REVISION": "a" * 40,
                "BRAINTRACE_SOURCE_DIRTY": "maybe",
            },
            "true or false",
        ),
    ],
)
def test_source_revision_environment_fails_closed(
    monkeypatch, environment: dict[str, str], message: str
) -> None:
    monkeypatch.delenv("BRAINTRACE_SOURCE_REVISION", raising=False)
    monkeypatch.delenv("BRAINTRACE_SOURCE_DIRTY", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        _subject()._source_revision()


def test_main_binds_cli_configuration_and_prints_gate(tmp_path, monkeypatch, capsys) -> None:
    subject = _subject()
    observed = {}

    def fake_run(config):
        observed["config"] = config
        return {
            "evaluation": {"strict_task_pass_at_1_count": 9},
            "promotion_gate_passed": True,
        }

    monkeypatch.setattr(subject, "run_oracle", fake_run)

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
            "--encoder-width",
            "4",
            "--hidden-width",
            "6",
            "--recurrent-layers",
            "2",
        ]
    )

    assert exit_code == 0
    assert observed["config"].training_updates == 2
    assert observed["config"].oracle_seed == 41
    assert '"promotion_gate_passed":true' in capsys.readouterr().out
