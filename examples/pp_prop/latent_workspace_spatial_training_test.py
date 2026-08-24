"""Tests for compiled PP-prop training of the V22 spatial Conv-LIF model."""

from __future__ import annotations

import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

from examples.pp_prop.latent_workspace_task import ArcGrid, ArcPair, ArcTask, RowEventConfig


def _subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_spatial_training")


def _model_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_spatial_model")


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
        task_id="spatial-tiny",
    )


def test_spatial_pp_prop_pilot_moves_every_parameter_group() -> None:
    subject = _subject()
    model_subject = _model_subject()
    online = _online_subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episode = online.encode_online_episode(_task(), 0, row_config)
    batch = online.stack_online_episodes((episode, episode))
    model = model_subject.SpatialARCConvLIF(
        model_subject.SpatialModelConfig(
            input_width=model_subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            seed=29,
        )
    )
    trainer = subject.SpatialPPPropTrainer(
        model, batch_size=2, learning_rate=0.001
    )
    relation_paths = {
        str(relation.path[0]) for relation in trainer.learner.graph.hidden_param_op_relations
    }
    assert relation_paths == {
        "input_conv",
        "recurrent_conv",
        "color_head",
        "height_head",
        "width_head",
    }
    before = subject.spatial_parameter_arrays(model)

    losses, norms = trainer.train_chunk(online.repeat_online_batch(batch, updates=2))
    after = subject.spatial_parameter_arrays(model)

    assert np.asarray(losses).shape == (2,)
    assert np.all(np.isfinite(np.asarray(losses)))
    assert set(norms) == {"input", "recurrent", "color", "height", "width"}
    assert all(
        np.isfinite(value) and value > 0.0 for value in norms.values()
    ), norms
    assert all(before[name].tobytes() != after[name].tobytes() for name in before)
    assert trainer.algorithm == "pp_prop"
    assert trainer.vjp_method == "single-step"
    assert trainer.loss_version == "target_balanced_color_v21"


def test_spatial_checkpoint_roundtrip_is_output_exact(tmp_path) -> None:
    subject = _subject()
    model_subject = _model_subject()
    model = model_subject.SpatialARCConvLIF(
        model_subject.SpatialModelConfig(
            input_width=model_subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            seed=31,
        )
    )
    path = tmp_path / "spatial.npz"

    digest = subject.save_spatial_checkpoint(model, path)
    loaded, metadata = subject.load_spatial_checkpoint(path)

    assert metadata["parameter_sha256"] == digest
    assert subject.spatial_parameter_digest(loaded) == digest
    event = jnp.zeros((1, model_subject.MODEL_INPUT_WIDTH), dtype=jnp.float32)
    with brainstate.environ.context(dt=1.0 * u.ms):
        brainstate.nn.init_all_states(model, batch_size=1)
        original = np.asarray(model(event))
        brainstate.nn.init_all_states(loaded, batch_size=1)
        restored = np.asarray(loaded(event))
    assert original.tobytes() == restored.tobytes()


def test_spatial_target_free_evaluation_is_deterministic() -> None:
    subject = _subject()
    model_subject = _model_subject()
    online = _online_subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episodes = (online.encode_online_episode(_task(), 0, row_config),)
    model = model_subject.SpatialARCConvLIF(
        model_subject.SpatialModelConfig(
            input_width=model_subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            seed=37,
        )
    )

    first = subject.evaluate_spatial_model(model, episodes, batch_size=1)
    second = subject.evaluate_spatial_model(model, episodes, batch_size=1)

    assert first["candidate_sha256"] == second["candidate_sha256"]
    assert first["query_count"] == 1
    assert first["candidates"][0]["answer_head_version"] == (
        "spatial_conv_lif_row_decoder_v22"
    )
