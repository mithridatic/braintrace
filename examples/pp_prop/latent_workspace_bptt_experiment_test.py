"""Tests for the matched reverse-mode (BPTT) trainer and experiment."""

from __future__ import annotations

import importlib
import json

import brainstate
import jax
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_expert_training import TaskGatedPPPropTrainer
from examples.pp_prop.latent_workspace_gated_memory_model import (
    MODEL_INPUT_WIDTH,
    GatedMemoryConfig,
    PhaseSeparatedGatedMemoryRNN,
)
from examples.pp_prop.latent_workspace_online_training import (
    parameter_digest,
    sample_online_training_chunk,
)
from examples.pp_prop.latent_workspace_task import RowEventConfig


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_bptt_experiment")


def _trainer_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_bptt_training")


def _curriculum_subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_diverse_curriculum"
    )


def _experiment_subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_experiment"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _model(seed: int = 41) -> PhaseSeparatedGatedMemoryRNN:
    return PhaseSeparatedGatedMemoryRNN(
        GatedMemoryConfig(
            input_width=MODEL_INPUT_WIDTH, memory_width=8, expert_count=12, seed=seed
        )
    )


def _chunk(seed: int = 51):
    curriculum_module = _curriculum_subject()
    experiment_module = _experiment_subject()
    curriculum = curriculum_module.generate_diverse_curriculum(
        curriculum_module.DiverseCurriculumConfig(
            task_count=12, max_grid_size=6, min_demonstrations=2, max_demonstrations=2
        ),
        brainstate.random.RandomState(seed),
    )
    catalog = experiment_module.training_episode_catalog(curriculum.tasks)
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    chunk = sample_online_training_chunk(
        catalog,
        row_config,
        brainstate.random.RandomState(seed + 1),
        updates=2,
        batch_size=2,
        augment=False,
    )
    return chunk


def test_trainer_validation() -> None:
    subject = _trainer_subject()
    with pytest.raises(TypeError):
        subject.TaskGatedBPTTTrainer(object(), batch_size=2)
    with pytest.raises(TypeError):
        subject.TaskGatedBPTTTrainer(_model(), batch_size=2.5)
    with pytest.raises(ValueError):
        subject.TaskGatedBPTTTrainer(_model(), batch_size=0)
    with pytest.raises(ValueError):
        subject.TaskGatedBPTTTrainer(_model(), batch_size=2, learning_rate=-1.0)
    trainer = subject.TaskGatedBPTTTrainer(_model(), batch_size=2)
    assert trainer.algorithm == "bptt"
    assert trainer.loss_version == "task_gated_fourth_root_v42"
    assert sorted(trainer.groups) == ["height", "recurrent", "row_color", "width"]


def test_bptt_objective_matches_ppprop_objective() -> None:
    subject = _trainer_subject()
    chunk = _chunk()
    model_ppprop = _model()
    model_bptt = _model()
    assert parameter_digest(model_ppprop) == parameter_digest(model_bptt)

    ppprop = TaskGatedPPPropTrainer(model_ppprop, batch_size=2)
    bptt = subject.TaskGatedBPTTTrainer(model_bptt, batch_size=2)
    ppprop_losses, ppprop_norms = ppprop.train_chunk(chunk)
    bptt_losses, bptt_norms = bptt.train_chunk(chunk)

    first_ppprop = float(np.asarray(ppprop_losses)[0])
    first_bptt = float(np.asarray(bptt_losses)[0])
    assert first_bptt == pytest.approx(first_ppprop, rel=1e-3)
    for norms in (ppprop_norms, bptt_norms):
        assert set(norms) == {"recurrent", "row_color", "height", "width"}
        assert all(np.isfinite(value) and value > 0.0 for value in norms.values())
    assert parameter_digest(model_ppprop) != parameter_digest(model_bptt)


def test_bptt_gradients_flow_through_every_leaf() -> None:
    subject = _trainer_subject()
    chunk = _chunk(seed=61)
    model = _model()
    before = {
        path: np.asarray(state.value).copy()
        for path, state in model.states(brainstate.ParamState).items()
    }
    trainer = subject.TaskGatedBPTTTrainer(model, batch_size=2)
    losses, _ = trainer.train_chunk(chunk)

    assert np.isfinite(np.asarray(losses)).all()
    moved = {
        ".".join(map(str, path)): (
            np.asarray(before[path]).tobytes() != np.asarray(state.value).tobytes()
        )
        for path, state in model.states(brainstate.ParamState).items()
    }
    assert all(moved.values()), moved


def test_tiny_bptt_experiment_run_binds_matched_arm_and_artifact(tmp_path, monkeypatch) -> None:
    subject = _subject()
    monkeypatch.setenv("BRAINTRACE_SOURCE_REVISION", "7" * 40)
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")
    config = subject.BPTTExperimentConfig(
        output_dir=tmp_path / "artifact",
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
    )

    result = subject.run_bptt_experiment(config)
    stored = json.loads((config.output_dir / "result.json").read_text())

    assert result["source_revision"] == "7" * 40
    assert result["learner"]["algorithm"] == "bptt"
    assert result["learner"]["vjp_method"] == "reverse-mode"
    assert result["matched_ppprop_arm"]["artifact"] == (
        "var/ex21-online-v47-diverse-curriculum-v1"
    )
    assert result["data"]["training_schema_version"] == "direct_synthetic_curriculum_v4"
    assert result["model"]["parameters_moved"] is True
    assert all(result["model"]["parameter_groups_moved"].values())
    assert all(result["model"]["parameter_leaves_moved"].values())
    assert len(result["model"]["parameter_leaves_moved"]) == 18
    assert result["training"]["finite"] is True
    assert isinstance(result["mechanism_gate_passed"], bool)
    assert (config.output_dir / "checkpoint.npz").exists()
    assert stored["configuration"] == config.to_dict()


def test_config_validation(tmp_path) -> None:
    subject = _subject()
    base = dict(output_dir=tmp_path / "out", device="cpu")
    with pytest.raises(ValueError, match="different"):
        subject.BPTTExperimentConfig(
            **base, synthetic_seed=44108, holdout_seed=44108
        )
    with pytest.raises(ValueError, match="divide"):
        subject.BPTTExperimentConfig(
            **base, training_updates=7, training_chunk_size=4
        )
    with pytest.raises(ValueError, match="device"):
        subject.BPTTExperimentConfig(output_dir=tmp_path / "o2", device="tpu")
    config = subject.BPTTExperimentConfig(**base, synthetic_task_count=12)
    assert config.to_dict()["training_updates"] == 800
