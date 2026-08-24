"""Tests for V42 task-gated PP-prop training and persistence."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
)


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_expert_training")


def _model_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_expert_model")


def _online_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_training")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task() -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((1, 0),)), ArcGrid(((2, 0),))),
            ArcPair(ArcGrid(((3, 0),)), ArcGrid(((4, 0),))),
        ),
        test=(ArcPair(ArcGrid(((5, 0),)), ArcGrid(((6, 0),))),),
        task_id="expert-tiny",
    )


def _model(seed: int = 41):
    model_subject = _model_subject()
    return model_subject.TaskGatedOnlineRNN(
        model_subject.ExpertModelConfig(
            input_width=model_subject.MODEL_INPUT_WIDTH,
            encoder_width=5,
            hidden_width=16,
            expert_count=12,
            recurrent_layers=2,
            seed=seed,
        )
    )


def test_v42_pp_prop_moves_every_parameter_group_and_leaf() -> None:
    subject = _subject()
    online = _online_subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episode = online.encode_online_episode(_task(), 0, row_config)
    batch = online.stack_online_episodes((episode, episode))
    model = _model(seed=41)
    trainer = subject.TaskGatedPPPropTrainer(
        model, batch_size=2, learning_rate=0.001
    )
    before = subject.parameter_leaf_arrays(model)

    losses, norms = trainer.train_chunk(online.repeat_online_batch(batch, updates=2))
    after = subject.parameter_leaf_arrays(model)

    assert np.asarray(losses).shape == (2,)
    assert np.all(np.isfinite(np.asarray(losses)))
    assert set(norms) == {"recurrent", "row_color", "height", "width"}
    assert all(np.isfinite(value) and value > 0.0 for value in norms.values()), norms
    assert before.keys() == after.keys()
    assert all(before[name].tobytes() != after[name].tobytes() for name in before)
    assert trainer.loss_version == "task_gated_fourth_root_v42"


def test_v42_training_entrypoints_reject_wrong_model_types() -> None:
    subject = _subject()

    with pytest.raises(TypeError, match="model"):
        subject.parameter_leaf_arrays(object())
    with pytest.raises(TypeError, match="model"):
        subject.TaskGatedPPPropTrainer(object(), batch_size=1)
    with pytest.raises(TypeError, match="model"):
        subject.evaluate_task_gated_model(object(), ())


def test_v42_shared_checkpoint_loader_restores_exact_outputs(tmp_path) -> None:
    online = _online_subject()
    model_subject = _model_subject()
    model = _model(seed=43)
    path = tmp_path / "expert.npz"

    digest = online.save_online_checkpoint(model, path)
    loaded, metadata = online.load_online_checkpoint(path)

    assert isinstance(loaded, model_subject.TaskGatedOnlineRNN)
    assert metadata["parameter_sha256"] == digest
    assert online.parameter_digest(loaded) == digest
    event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    brainstate.nn.init_all_states(model, batch_size=1)
    original = np.asarray(model(event))
    brainstate.nn.init_all_states(loaded, batch_size=1)
    restored = np.asarray(loaded(event))
    assert original.tobytes() == restored.tobytes()


def test_v42_target_free_evaluation_is_deterministic_and_registered() -> None:
    subject = _subject()
    online = _online_subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episodes = (online.encode_online_episode(_task(), 0, row_config),)
    model = _model(seed=47)

    first = subject.evaluate_task_gated_model(model, episodes, batch_size=1)
    second = subject.evaluate_task_gated_model(model, episodes, batch_size=1)

    assert first["candidate_sha256"] == second["candidate_sha256"]
    candidate = first["candidates"][0]
    assert candidate["answer_head_version"] == "task_gated_operator_bank_v42"
    assert candidate["proposal_source"] == "task_gated_model_logits"
    assert candidate["dependency_class"] == "model_checkpoint"
    assert candidate["ranking_source"] == "none_single_greedy_candidate"
