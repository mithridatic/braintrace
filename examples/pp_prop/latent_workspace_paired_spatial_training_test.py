"""Tests for V46 paired spatial packing, PP-prop, and checkpoints."""

from __future__ import annotations

from dataclasses import replace
import importlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_paired_spatial_model import (
    PairedSpatialARC,
    PairedSpatialConfig,
)
from examples.pp_prop.latent_workspace_task import ArcGrid, ArcPair, ArcTask


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_paired_spatial_training"
    )


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task(
    target: tuple[tuple[int, ...], ...] = ((8, 0), (0, 8)),
) -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(
                ArcGrid(((1, 0), (0, 1))),
                ArcGrid(((2, 0), (0, 2))),
            ),
            ArcPair(
                ArcGrid(((3, 3), (0, 0))),
                ArcGrid(((4, 4), (0, 0))),
            ),
        ),
        test=(
            ArcPair(
                ArcGrid(((7, 0), (0, 7))),
                ArcGrid(target),
            ),
        ),
        task_id="paired-spatial-tiny",
    )


def test_v46_episode_is_lossless_fixed_length_and_target_isolated() -> None:
    subject = _subject()
    config = PairedSpatialConfig(spatial_channels=2, refinement_steps=3)
    first = subject.encode_paired_spatial_episode(_task(), 0, config)
    second = subject.encode_paired_spatial_episode(
        _task(((6, 6, 6),)), 0, config
    )

    assert first.events.shape == (13, 30, 30, 6)
    assert first.events.tobytes() == second.events.tobytes()
    assert first.target_colors.tobytes() != second.target_colors.tobytes()
    assert np.all(first.events[:2, ..., subject.DEMO_PHASE_CHANNEL] == 1.0)
    assert np.count_nonzero(first.events[2:10]) == 0
    assert np.all(first.events[10:, ..., subject.QUERY_PHASE_CHANNEL] == 1.0)
    assert np.count_nonzero(
        first.events[10:, ..., subject.OUTPUT_MASK_CHANNEL]
    ) == 0
    assert np.all(first.loss_step_mask[:-1] == 0)
    assert first.loss_step_mask[-1]
    assert first.task_id == "paired-spatial-tiny"


def test_v46_stack_and_sampling_shapes_are_time_major() -> None:
    subject = _subject()
    config = PairedSpatialConfig(spatial_channels=2, refinement_steps=2)
    episodes = (
        subject.encode_paired_spatial_episode(_task(), 0, config),
        subject.encode_paired_spatial_episode(_task(((5, 0), (0, 5))), 0, config),
    )
    batch = subject.stack_paired_spatial_episodes(episodes)
    chunk = subject.sample_paired_spatial_training_chunk(
        (_task(),),
        config,
        brainstate.random.RandomState(19),
        updates=2,
        batch_size=2,
        augment=False,
    )

    assert batch.events.shape == (12, 2, 30, 30, 6)
    assert batch.target_colors.shape == (12, 2, 30, 30)
    assert chunk.events.shape == (2, 12, 2, 30, 30, 6)
    assert chunk.target_heights.shape == (2, 12, 2)
    assert chunk.loss_step_mask.shape == (12,)
    with pytest.raises(TypeError, match="RandomState"):
        subject.sample_paired_spatial_training_chunk(
            (_task(),), config, object(), updates=1, batch_size=1, augment=False
        )


def test_v46_packing_and_loss_boundaries_fail_closed() -> None:
    subject = _subject()
    config = PairedSpatialConfig(spatial_channels=2, refinement_steps=2)
    episode = subject.encode_paired_spatial_episode(_task(), 0, config)
    no_target = ArcTask(
        train=_task().train,
        test=(ArcPair(ArcGrid(((1,),)), None),),
        task_id="no-target",
    )
    too_many = ArcTask(
        train=_task().train * 6,
        test=_task().test,
        task_id="too-many",
    )

    with pytest.raises(TypeError, match="ArcTask"):
        subject.encode_paired_spatial_episode(object(), 0, config)
    with pytest.raises(TypeError, match="query_index"):
        subject.encode_paired_spatial_episode(_task(), True, config)
    with pytest.raises(ValueError, match="outside"):
        subject.encode_paired_spatial_episode(_task(), -1, config)
    with pytest.raises(TypeError, match="config"):
        subject.encode_paired_spatial_episode(_task(), 0, object())
    with pytest.raises(ValueError, match="at most"):
        subject.encode_paired_spatial_episode(too_many, 0, config)
    with pytest.raises(ValueError, match="output"):
        subject.encode_paired_spatial_episode(no_target, 0, config)
    with pytest.raises(ValueError, match="nonempty"):
        subject.stack_paired_spatial_episodes(())
    with pytest.raises(TypeError, match="PairedSpatialEpisode"):
        subject.stack_paired_spatial_episodes((object(),))
    mismatched = replace(
        episode, loss_step_mask=np.zeros_like(episode.loss_step_mask)
    )
    with pytest.raises(ValueError, match="masks"):
        subject.stack_paired_spatial_episodes((episode, mismatched))
    with pytest.raises(TypeError, match="PairedSpatialBatch"):
        subject.repeat_paired_spatial_batch(object(), 1)
    with pytest.raises(TypeError, match="positive integer"):
        subject.repeat_paired_spatial_batch(
            subject.stack_paired_spatial_episodes((episode,)), True
        )
    with pytest.raises(ValueError, match="nonempty"):
        subject.sample_paired_spatial_training_chunk(
            (), config, brainstate.random.RandomState(1),
            updates=1, batch_size=1, augment=False
        )
    with pytest.raises(TypeError, match="config"):
        subject.sample_paired_spatial_training_chunk(
            (_task(),), object(), brainstate.random.RandomState(1),
            updates=1, batch_size=1, augment=False
        )
    with pytest.raises(TypeError, match="boolean"):
        subject.sample_paired_spatial_training_chunk(
            (_task(),), config, brainstate.random.RandomState(1),
            updates=1, batch_size=1, augment=1
        )
    augmented = subject.sample_paired_spatial_training_chunk(
        (_task(),), config, brainstate.random.RandomState(2),
        updates=1, batch_size=1, augment=True
    )
    assert augmented.events.shape[:3] == (1, 12, 1)

    with pytest.raises(ValueError, match="match in 3-D"):
        subject.paired_spatial_hierarchical_mass(
            jnp.zeros((1, 2)), jnp.zeros((1, 2))
        )
    target = jnp.zeros((1, 30, 30), dtype=jnp.int32)
    mask = jnp.ones_like(target, dtype=jnp.float32)
    gate_mass, color_mass = subject.paired_spatial_hierarchical_mass(target, mask)
    output = jnp.zeros((1, subject.OUTPUT_WIDTH))
    with pytest.raises(ValueError, match="last dimension must be 2"):
        subject.paired_spatial_hierarchical_loss(
            output, target, mask, jnp.asarray([0]), jnp.asarray([0]),
            gate_mass[:, :1], color_mass
        )
    with pytest.raises(ValueError, match="last dimension must be 9"):
        subject.paired_spatial_hierarchical_loss(
            output, target, mask, jnp.asarray([0]), jnp.asarray([0]),
            gate_mass, color_mass[:, :8]
        )
    with pytest.raises(ValueError, match="output last dimension"):
        subject.paired_spatial_hierarchical_loss(
            output[:, :-1], target, mask, jnp.asarray([0]), jnp.asarray([0]),
            gate_mass, color_mass
        )


def test_v46_parameter_trainer_evaluator_boundaries_fail_closed(tmp_path) -> None:
    subject = _subject()
    config = PairedSpatialConfig(spatial_channels=2, refinement_steps=2, seed=39)
    model = PairedSpatialARC(config)

    for function in (
        subject.paired_spatial_parameter_leaf_arrays,
        subject.paired_spatial_parameter_arrays,
        subject.paired_spatial_parameter_digest,
    ):
        with pytest.raises(TypeError, match="PairedSpatialARC"):
            function(object())
    with pytest.raises(TypeError, match="PairedSpatialARC"):
        subject.PairedSpatialPPPropTrainer(object(), batch_size=1)
    with pytest.raises(ValueError, match="positive integer"):
        subject.PairedSpatialPPPropTrainer(model, batch_size=0)
    with pytest.raises(TypeError, match="learning_rate"):
        subject.PairedSpatialPPPropTrainer(
            PairedSpatialARC(config), batch_size=1, learning_rate=None
        )
    with pytest.raises(ValueError, match="learning_rate"):
        subject.PairedSpatialPPPropTrainer(
            PairedSpatialARC(config), batch_size=1, learning_rate=0.0
        )
    with pytest.raises(ValueError, match="trace_decay"):
        subject.PairedSpatialPPPropTrainer(
            PairedSpatialARC(config), batch_size=1, trace_decay=1.1
        )
    with pytest.raises(TypeError, match="PairedSpatialARC"):
        subject.evaluate_paired_spatial_model(object(), (), batch_size=1)
    with pytest.raises(ValueError, match="nonempty"):
        subject.evaluate_paired_spatial_model(model, (), batch_size=1)
    with pytest.raises(ValueError, match="finite fixed-width"):
        subject._decode_paired_spatial_output(np.zeros(4), ())
    with pytest.raises(TypeError, match="PairedSpatialARC"):
        subject.save_paired_spatial_checkpoint(object(), tmp_path / "bad.npz")


def test_v46_hierarchical_loss_rewards_exact_grid_and_shape() -> None:
    subject = _subject()
    targets = jnp.asarray([[[1, 0], [0, 2]]], dtype=jnp.int32)
    mask = jnp.ones_like(targets, dtype=jnp.float32)
    gate_mass, color_mass = subject.paired_spatial_hierarchical_mass(targets, mask)
    exact = jnp.full((1, subject.OUTPUT_WIDTH), -8.0)
    wrong = jnp.full((1, subject.OUTPUT_WIDTH), -8.0)
    exact_grid = exact[:, :9000].reshape(1, 30, 30, 10)
    wrong_grid = wrong[:, :9000].reshape(1, 30, 30, 10)
    padded_targets = jnp.zeros((1, 30, 30), dtype=jnp.int32)
    padded_targets = padded_targets.at[:, :2, :2].set(targets)
    padded_mask = jnp.zeros((1, 30, 30), dtype=jnp.float32)
    padded_mask = padded_mask.at[:, :2, :2].set(mask)
    foreground = padded_targets != 0
    exact_grid = exact_grid.at[..., 0].set(
        jnp.where(foreground, 8.0, -8.0)
    )
    exact_grid = jnp.where(
        (jax.nn.one_hot(padded_targets, 10) > 0) & foreground[..., None],
        8.0,
        exact_grid,
    )
    exact = exact.at[:, :9000].set(exact_grid.reshape(1, -1))
    wrong = wrong.at[:, :9000].set(wrong_grid.reshape(1, -1))
    exact = exact.at[:, 9000 + 1].set(8.0)
    exact = exact.at[:, 9030 + 1].set(8.0)
    wrong = wrong.at[:, 9000].set(8.0)
    wrong = wrong.at[:, 9030].set(8.0)

    exact_loss = subject.paired_spatial_hierarchical_loss(
        exact,
        padded_targets,
        padded_mask,
        jnp.asarray([1]),
        jnp.asarray([1]),
        gate_mass,
        color_mass,
    )
    wrong_loss = subject.paired_spatial_hierarchical_loss(
        wrong,
        padded_targets,
        padded_mask,
        jnp.asarray([1]),
        jnp.asarray([1]),
        gate_mass,
        color_mass,
    )

    assert float(exact_loss) < 1e-3
    assert float(wrong_loss) > float(exact_loss) + 1.0


def test_v46_compiler_and_training_move_every_group_and_leaf() -> None:
    subject = _subject()
    config = PairedSpatialConfig(
        spatial_channels=2, retention=0.8, refinement_steps=2, seed=41
    )
    model = PairedSpatialARC(config)
    before = subject.paired_spatial_parameter_leaf_arrays(model)
    trainer = subject.PairedSpatialPPPropTrainer(
        model, batch_size=2, learning_rate=0.001
    )
    batch = subject.stack_paired_spatial_episodes(
        (
            subject.encode_paired_spatial_episode(_task(), 0, config),
            subject.encode_paired_spatial_episode(
                _task(((5, 5), (0, 5))), 0, config
            ),
        )
    )
    chunk = subject.repeat_paired_spatial_batch(batch, updates=4)

    with pytest.raises(TypeError, match="PairedSpatialTrainingChunk"):
        trainer.train_chunk(object())
    wrong_batch = replace(chunk, events=chunk.events[:, :, :1])
    with pytest.raises(ValueError, match="batch axis"):
        trainer.train_chunk(wrong_batch)
    losses, norms = trainer.train_chunk(chunk)
    after = subject.paired_spatial_parameter_leaf_arrays(model)
    recurrent_roots = {
        "demo_input_conv",
        "demo_recurrent_conv",
        "query_input_conv",
        "query_recurrent_conv",
    }
    recurrent_exclusions = [
        path
        for path, _ in trainer.learner.report.excluded_weights
        if path[0] in recurrent_roots
    ]

    assert np.isfinite(np.asarray(losses)).all()
    assert all(np.isfinite(value) and value > 0.0 for value in norms.values())
    assert recurrent_exclusions == []
    assert before.keys() == after.keys()
    assert all(
        before[name].tobytes() != after[name].tobytes() for name in before
    )


def test_v46_evaluation_is_target_free_deterministic_and_direct() -> None:
    subject = _subject()
    config = PairedSpatialConfig(spatial_channels=2, refinement_steps=2, seed=43)
    model = PairedSpatialARC(config)
    first_episode = subject.encode_paired_spatial_episode(_task(), 0, config)
    second_episode = subject.encode_paired_spatial_episode(
        _task(((9, 9, 9),)), 0, config
    )

    first = subject.evaluate_paired_spatial_model(
        model, (first_episode,), batch_size=1
    )
    second = subject.evaluate_paired_spatial_model(
        model, (second_episode,), batch_size=1
    )

    assert first["candidate_sha256"] == second["candidate_sha256"]
    candidate = first["candidates"][0]
    assert candidate["provenance"] == "model"
    assert candidate["dependency_class"] == "model_checkpoint"
    assert candidate["proposal_source"] == "paired_spatial_model_logits"
    assert candidate["ranking_source"] == "none_single_greedy_candidate"
    assert first["query_count"] == 1
    assert first["task_count"] == 1


def test_v46_checkpoint_roundtrip_binds_exact_schema(tmp_path) -> None:
    subject = _subject()
    model = PairedSpatialARC(
        PairedSpatialConfig(spatial_channels=2, refinement_steps=2, seed=47)
    )
    path = tmp_path / "paired-spatial.npz"

    digest = subject.save_paired_spatial_checkpoint(model, path)
    restored, metadata = subject.load_paired_spatial_checkpoint(path)

    assert digest == subject.paired_spatial_parameter_digest(model)
    assert digest == subject.paired_spatial_parameter_digest(restored)
    assert metadata["architecture"] == {
        "architecture_version": "paired_spatial_conv_tanh_v46",
        "refinement_steps": 2,
        "retention": 0.8,
        "seed": 47,
        "spatial_channels": 2,
    }
    assert subject.paired_spatial_parameter_leaf_arrays(model).keys() == (
        subject.paired_spatial_parameter_leaf_arrays(restored).keys()
    )
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="decoded safely"):
        subject.load_paired_spatial_checkpoint(broken)
