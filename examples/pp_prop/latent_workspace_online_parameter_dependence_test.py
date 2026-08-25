"""Tests for the online-line checkpoint-dependence qualification."""

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
    load_online_checkpoint,
    parameter_digest,
    save_online_checkpoint,
)
from examples.pp_prop.latent_workspace_task import ArcGrid, ArcPair, ArcTask


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_online_parameter_dependence"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _checkpoint(tmp_path, name: str = "baseline.npz", *, seed: int = 41):
    model = PhaseSeparatedGatedMemoryRNN(
        GatedMemoryConfig(
            input_width=MODEL_INPUT_WIDTH, memory_width=8, expert_count=12, seed=seed
        )
    )
    path = tmp_path / name
    digest = save_online_checkpoint(model, path)
    assert digest == parameter_digest(model)
    return path, digest


def _task(index: int, *, role: str = "evaluation") -> ArcTask:
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


def test_scale_perturbation_roundtrips_with_halved_leaves(tmp_path) -> None:
    subject = _subject()
    source, source_digest = _checkpoint(tmp_path)
    destination = tmp_path / "scaled.npz"

    report = subject.write_online_checkpoint_perturbation(
        source, destination, kind="scale", factor=0.5
    )

    assert report["kind"] == "scale"
    assert report["factor"] == 0.5
    assert report["source_parameter_sha256"] == source_digest
    assert report["parameter_sha256"] != source_digest
    assert report["checkpoint_sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    with np.load(source, allow_pickle=False) as original, np.load(
        destination, allow_pickle=False
    ) as scaled:
        assert set(original.files) == set(scaled.files)
        for name in original.files:
            if name == "__metadata__":
                continue
            assert scaled[name].dtype == original[name].dtype
            assert scaled[name].shape == original[name].shape
            assert np.allclose(
                scaled[name], original[name] * np.float32(0.5), rtol=1e-6
            )
    _, metadata = load_online_checkpoint(destination)
    assert metadata["parameter_sha256"] == report["parameter_sha256"]


def test_reseed_perturbation_is_deterministic_and_rms_matched(tmp_path) -> None:
    subject = _subject()
    source, source_digest = _checkpoint(tmp_path)
    first = tmp_path / "reseed-a.npz"
    second = tmp_path / "reseed-b.npz"
    third = tmp_path / "reseed-c.npz"

    report_a = subject.write_online_checkpoint_perturbation(
        source, first, kind="reseed", seed=73
    )
    report_b = subject.write_online_checkpoint_perturbation(
        source, second, kind="reseed", seed=73
    )
    report_c = subject.write_online_checkpoint_perturbation(
        source, third, kind="reseed", seed=74
    )

    assert report_a["parameter_sha256"] == report_b["parameter_sha256"]
    assert report_a["checkpoint_sha256"] == report_b["checkpoint_sha256"]
    assert report_a["parameter_sha256"] != report_c["parameter_sha256"]
    assert report_a["source_parameter_sha256"] == source_digest
    with np.load(source, allow_pickle=False) as original, np.load(
        first, allow_pickle=False
    ) as reseeded:
        for name in original.files:
            if name == "__metadata__":
                continue
            source_rms = float(np.sqrt(np.mean(np.square(original[name]))))
            draw_rms = float(np.sqrt(np.mean(np.square(reseeded[name]))))
            if source_rms > 1e-6:
                assert draw_rms == pytest.approx(source_rms, rel=1e-3)
            else:
                assert draw_rms < 1e-5
    _, metadata = load_online_checkpoint(first)
    assert metadata["parameter_sha256"] == report_a["parameter_sha256"]


def test_perturbation_rejects_invalid_interventions(tmp_path) -> None:
    subject = _subject()
    source, _ = _checkpoint(tmp_path)
    occupied = tmp_path / "occupied.npz"
    occupied.write_bytes(b"x")

    with pytest.raises(ValueError, match="'scale' or 'reseed'"):
        subject.write_online_checkpoint_perturbation(
            source, tmp_path / "a.npz", kind="shift"
        )
    with pytest.raises(ValueError, match="non-unit"):
        subject.write_online_checkpoint_perturbation(
            source, tmp_path / "b.npz", kind="scale", factor=1.0
        )
    with pytest.raises(ValueError, match="positive finite"):
        subject.write_online_checkpoint_perturbation(
            source, tmp_path / "c.npz", kind="scale", factor=-0.5
        )
    with pytest.raises(ValueError, match="nonnegative"):
        subject.write_online_checkpoint_perturbation(
            source, tmp_path / "d.npz", kind="reseed", seed=-1
        )
    with pytest.raises(ValueError, match="differ"):
        subject.write_online_checkpoint_perturbation(
            source, source, kind="scale", factor=0.5
        )
    with pytest.raises(FileExistsError):
        subject.write_online_checkpoint_perturbation(
            source, occupied, kind="scale", factor=0.5
        )


def _arm(tag: str, count: int):
    return {
        "checkpoint_file_sha256": tag * 64,
        "parameter_sha256": tag * 64,
        "candidate_sha256": tag * 64,
        "strict_task_pass_at_1_count": count,
        "strict_task_ids": [f"task-{index}" for index in range(count)],
        "task_membership": {f"task-{index}": True for index in range(count)},
    }


def _arms(baseline_count: int = 17):
    return {
        "baseline": _arm("a", baseline_count),
        "repeat_intact": _arm("a", baseline_count),
        "scale": _arm("b", baseline_count - 1),
        "reseed": _arm("c", baseline_count - 2),
        "swap": _arm("d", baseline_count + 1),
    }


def test_assess_online_dependence_passes_only_when_every_gate_moves() -> None:
    subject = _subject()

    assert subject.assess_online_dependence(_arms())["passed"] is True

    flat_scale = _arms()
    flat_scale["scale"] = dict(flat_scale["scale"])
    flat_scale["scale"]["strict_task_pass_at_1_count"] = 17
    report = subject.assess_online_dependence(flat_scale)
    assert report["passed"] is False
    assert report["checks"]["scale_score_moved"] is False
    assert "scale_score_moved failed" in report["reasons"]

    flat_candidate = _arms()
    flat_candidate["reseed"] = dict(flat_candidate["reseed"])
    flat_candidate["reseed"]["candidate_sha256"] = "a" * 64
    report = subject.assess_online_dependence(flat_candidate)
    assert report["checks"]["reseed_candidate_moved"] is False
    assert report["passed"] is False

    broken_repeat = _arms()
    broken_repeat["repeat_intact"] = dict(broken_repeat["repeat_intact"])
    broken_repeat["repeat_intact"]["candidate_sha256"] = "z" * 64
    report = subject.assess_online_dependence(broken_repeat)
    assert report["checks"]["repeat_candidate_identical"] is False
    assert report["passed"] is False

    with pytest.raises(ValueError, match="missing"):
        subject.assess_online_dependence({"baseline": _arm("a", 17)})


def test_tiny_matrix_run_binds_arms_interventions_and_artifact(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    baseline, _ = _checkpoint(tmp_path, "baseline.npz", seed=41)
    swap, _ = _checkpoint(tmp_path, "swap.npz", seed=42)
    evaluation = tuple(_task(index) for index in range(2))

    class Manifest:
        def to_dict(self):
            return {"source": {"role": "evaluation"}, "valid_task_count": 2}

    corpora = SimpleNamespace(
        training=(_task(9, role="train"),),
        evaluation=evaluation,
        loaded=(SimpleNamespace(manifest=Manifest()),),
    )
    monkeypatch.setattr(subject, "load_corpora", lambda _: corpora)
    monkeypatch.setattr(subject, "EXPECTED_EVALUATION_TASK_COUNT", 2)
    monkeypatch.setattr(subject, "EXPECTED_EVALUATION_QUERY_COUNT", 2)
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "7" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")

    result = subject.run_online_dependence_matrix(
        baseline_checkpoint=baseline,
        swap_checkpoint=swap,
        source_manifest=tmp_path / "sources.json",
        output_dir=tmp_path / "matrix",
    )
    stored = json.loads((tmp_path / "matrix" / "matrix-result.json").read_text())

    assert result["authority"] == "online_parameter_dependence_matrix"
    assert result["source_revision"] == "7" * 40
    assert result["manifest"]["task_count"] == 2
    assert sorted(result["arms"]) == [
        "baseline",
        "repeat_intact",
        "reseed",
        "scale",
        "swap",
    ]
    baseline_arm = result["arms"]["baseline"]
    assert baseline_arm["task_count"] == 2
    assert baseline_arm["query_count"] == 2
    repeat_arm = result["arms"]["repeat_intact"]
    assert repeat_arm["candidate_sha256"] == baseline_arm["candidate_sha256"]
    for name in ("scale", "reseed", "swap"):
        arm = result["arms"][name]
        assert arm["parameter_sha256"] != baseline_arm["parameter_sha256"]
    interventions = result["interventions"]
    assert interventions["scale"]["factor"] == 0.5
    assert interventions["reseed"]["seed"] == 73
    assert isinstance(result["qualification"]["passed"], bool)
    assert stored["manifest"]["manifest_sha256"] == result["manifest"][
        "manifest_sha256"
    ]
