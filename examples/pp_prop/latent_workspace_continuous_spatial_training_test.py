"""Tests for V41 continuous spatial PP-prop training and persistence."""

from __future__ import annotations

import importlib
import pathlib

import brainstate
import jax
import jax.numpy as jnp
import msgspec
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
)


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_continuous_spatial_training"
    )


def _model_subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_continuous_spatial_model"
    )


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
        task_id="continuous-spatial-tiny",
    )


def _model(seed: int = 37):
    model_subject = _model_subject()
    return model_subject.ContinuousSpatialARC(
        model_subject.ContinuousSpatialConfig(
            input_width=model_subject.MODEL_INPUT_WIDTH,
            spatial_channels=2,
            retention=0.8,
            seed=seed,
        )
    )


def test_v41_pp_prop_moves_every_leaf_without_recurrent_exclusion() -> None:
    subject = _subject()
    online = _online_subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episode = online.encode_online_episode(_task(), 0, row_config)
    batch = online.stack_online_episodes((episode, episode))
    model = _model(seed=37)
    trainer = subject.ContinuousSpatialPPPropTrainer(
        model, batch_size=2, learning_rate=0.001
    )
    recurrent_exclusions = [
        path
        for path, _ in trainer.learner.report.excluded_weights
        if path[0] in {"input_conv", "recurrent_conv"}
    ]
    before = subject.continuous_spatial_parameter_leaf_arrays(model)

    losses, norms = trainer.train_chunk(online.repeat_online_batch(batch, updates=2))
    after = subject.continuous_spatial_parameter_leaf_arrays(model)

    assert recurrent_exclusions == []
    assert np.asarray(losses).shape == (2,)
    assert np.all(np.isfinite(np.asarray(losses)))
    assert set(norms) == {"input", "recurrent", "color", "height", "width"}
    assert all(np.isfinite(value) and value > 0.0 for value in norms.values()), norms
    assert before.keys() == after.keys()
    assert all(before[name].tobytes() != after[name].tobytes() for name in before)
    assert trainer.algorithm == "pp_prop"
    assert trainer.vjp_method == "single-step"
    assert trainer.loss_version == "fourth_root_hierarchical_v41"


def test_v41_training_entrypoints_fail_closed() -> None:
    subject = _subject()
    online = _online_subject()

    for operation in (
        subject.continuous_spatial_parameter_leaf_arrays,
        subject.continuous_spatial_parameter_arrays,
        subject.continuous_spatial_parameter_digest,
    ):
        with pytest.raises(TypeError, match="model"):
            operation(object())
    with pytest.raises(TypeError, match="model"):
        subject.ContinuousSpatialPPPropTrainer(object(), batch_size=1)

    model = _model(seed=39)
    with pytest.raises(TypeError, match="batch_size"):
        subject.ContinuousSpatialPPPropTrainer(model, batch_size=True)
    with pytest.raises(ValueError, match="batch_size"):
        subject.ContinuousSpatialPPPropTrainer(model, batch_size=0)
    with pytest.raises(TypeError, match="learning_rate"):
        subject.ContinuousSpatialPPPropTrainer(
            model, batch_size=1, learning_rate="bad"
        )
    with pytest.raises(ValueError, match="learning_rate"):
        subject.ContinuousSpatialPPPropTrainer(
            model, batch_size=1, learning_rate=0.0
        )
    with pytest.raises(ValueError, match="trace_decay"):
        subject.ContinuousSpatialPPPropTrainer(
            model, batch_size=1, trace_decay=1.1
        )

    trainer = subject.ContinuousSpatialPPPropTrainer(model, batch_size=2)
    with pytest.raises(TypeError, match="chunk"):
        trainer.train_chunk(object())
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episode = online.encode_online_episode(_task(), 0, row_config)
    one = online.stack_online_episodes((episode,))
    with pytest.raises(ValueError, match="batch axis"):
        trainer.train_chunk(online.repeat_online_batch(one, updates=1))

    with pytest.raises(TypeError, match="model"):
        subject.evaluate_continuous_spatial_model(object(), (episode,))
    with pytest.raises(ValueError, match="episodes"):
        subject.evaluate_continuous_spatial_model(model, ())
    with pytest.raises(TypeError, match="batch_size"):
        subject.evaluate_continuous_spatial_model(model, (episode,), batch_size=True)
    with pytest.raises(TypeError, match="model"):
        subject.save_continuous_spatial_checkpoint(object(), pathlib.Path("unused"))


def test_v41_checkpoint_roundtrip_preserves_digest_and_outputs(tmp_path) -> None:
    subject = _subject()
    model_subject = _model_subject()
    model = _model(seed=41)
    path = tmp_path / "continuous-spatial.npz"

    digest = subject.save_continuous_spatial_checkpoint(model, path)
    loaded, metadata = subject.load_continuous_spatial_checkpoint(path)

    assert metadata["parameter_sha256"] == digest
    assert subject.continuous_spatial_parameter_digest(loaded) == digest
    event = jnp.zeros((1, model_subject.MODEL_INPUT_WIDTH), dtype=jnp.float32)
    brainstate.nn.init_all_states(model, batch_size=1)
    original = np.asarray(model(event))
    brainstate.nn.init_all_states(loaded, batch_size=1)
    restored = np.asarray(loaded(event))
    assert original.tobytes() == restored.tobytes()


def test_v41_checkpoint_loader_rejects_missing_metadata_and_digest_drift(
    tmp_path,
) -> None:
    subject = _subject()
    missing = tmp_path / "missing.npz"
    np.savez(missing, leaf_0000=np.zeros((1,), dtype=np.float32))
    with pytest.raises(ValueError, match="metadata"):
        subject.load_continuous_spatial_checkpoint(missing)

    valid = tmp_path / "valid.npz"
    subject.save_continuous_spatial_checkpoint(_model(seed=42), valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = msgspec.json.decode(bytes(arrays["__metadata__"]))
    metadata["parameter_sha256"] = "0" * 64
    arrays["__metadata__"] = np.frombuffer(
        msgspec.json.encode(metadata, order="sorted"), dtype=np.uint8
    )
    drift = tmp_path / "drift.npz"
    np.savez(drift, **arrays)
    with pytest.raises(ValueError, match="digest"):
        subject.load_continuous_spatial_checkpoint(drift)


def test_v41_target_free_evaluation_is_deterministic_and_neural() -> None:
    subject = _subject()
    online = _online_subject()
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    episodes = (online.encode_online_episode(_task(), 0, row_config),)
    model = _model(seed=43)

    first = subject.evaluate_continuous_spatial_model(model, episodes, batch_size=1)
    second = subject.evaluate_continuous_spatial_model(model, episodes, batch_size=1)

    assert first["candidate_sha256"] == second["candidate_sha256"]
    assert first["query_count"] == 1
    candidate = first["candidates"][0]
    assert candidate["answer_head_version"] == "continuous_spatial_row_decoder_v41"
    assert candidate["proposal_source"] == "continuous_spatial_model_logits"
    assert candidate["dependency_class"] == "model_checkpoint"
    assert candidate["ranking_source"] == "none_single_greedy_candidate"
