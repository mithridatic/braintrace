"""Tests for target-free direct ARC episode training."""

from __future__ import annotations

import importlib

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
    return importlib.import_module("examples.pp_prop.latent_workspace_direct_training")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task(second_output: int = 2) -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((0,),)), ArcGrid(((0,),))),
            ArcPair(ArcGrid(((1,),)), ArcGrid(((second_output,),))),
        ),
        test=(ArcPair(ArcGrid(((3,),)), ArcGrid(((3,),))),),
        task_id="tiny",
    )


def test_leave_one_out_episode_keeps_held_out_output_out_of_events() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    first = subject.leave_one_out_tasks(_task(second_output=2))[1]
    changed = subject.leave_one_out_tasks(_task(second_output=9))[1]

    first_encoded = subject.encode_direct_episode(first, 0, row_config)
    changed_encoded = subject.encode_direct_episode(changed, 0, row_config)

    assert first_encoded.events.tobytes() == changed_encoded.events.tobytes()
    assert first_encoded.query_features.tobytes() == changed_encoded.query_features.tobytes()
    assert first_encoded.shape_features.tobytes() == changed_encoded.shape_features.tobytes()
    assert first_encoded.target_colors[0, 0] == 2
    assert changed_encoded.target_colors[0, 0] == 9


def test_query_features_are_lossless_color_plus_validity() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)

    encoded = subject.encode_direct_episode(_task(), 0, row_config)

    assert encoded.query_features.shape == (30, 30, 11)
    assert encoded.query_features[0, 0, 3] == 1.0
    assert encoded.query_features[0, 0, 10] == 1.0
    assert np.count_nonzero(encoded.query_features) == 2


def test_shape_features_encode_demos_but_exclude_held_out_dimensions() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    first_task = ArcTask(
        train=(
            ArcPair(ArcGrid(((0,),)), ArcGrid(((0, 0),))),
            ArcPair(ArcGrid(((1,),)), ArcGrid(((1, 1),))),
        ),
        test=(ArcPair(ArcGrid(((2,),)), ArcGrid(((2,),))),),
        task_id="shape-firewall",
    )
    changed_task = ArcTask(
        train=(
            first_task.train[0],
            ArcPair(ArcGrid(((1,),)), ArcGrid(((1,), (1,)))),
        ),
        test=first_task.test,
        task_id="shape-firewall",
    )

    first = subject.encode_direct_episode(
        subject.leave_one_out_tasks(first_task)[1], 0, row_config
    )
    changed = subject.encode_direct_episode(
        subject.leave_one_out_tasks(changed_task)[1], 0, row_config
    )

    assert first.shape_features.shape == (subject.SHAPE_FEATURE_WIDTH,)
    assert first.shape_features.tobytes() == changed.shape_features.tobytes()
    assert (first.target_height, first.target_width) == (0, 1)
    assert (changed.target_height, changed.target_width) == (1, 0)
    assert np.count_nonzero(first.shape_features) == 7


def test_stack_direct_episodes_is_time_major_and_target_separated() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    first, second = subject.leave_one_out_tasks(_task())
    batch = subject.stack_direct_episodes(
        (
            subject.encode_direct_episode(first, 0, row_config),
            subject.encode_direct_episode(second, 0, row_config),
        )
    )

    assert batch.events.shape == (9, 2, row_config.input_width)
    assert batch.query_features.shape == (2, 30, 30, 11)
    assert batch.shape_features.shape == (2, subject.SHAPE_FEATURE_WIDTH)
    assert batch.target_colors.shape == (2, 30, 30)
    assert batch.target_mask.shape == (2, 30, 30)
    assert batch.target_heights.tolist() == [0, 0]
    assert batch.target_widths.tolist() == [0, 0]


def test_compiled_training_chunk_moves_parameters_and_reduces_tiny_loss() -> None:
    subject = _subject()
    model_subject = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_model"
    )
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episode = subject.encode_direct_episode(
        subject.leave_one_out_tasks(_task())[0], 0, row_config
    )
    batch = subject.stack_direct_episodes((episode, episode))
    model = model_subject.DirectARCGRU(
        model_subject.DirectModelConfig(
            input_width=row_config.input_width,
            encoder_width=8,
            hidden_width=16,
            decoder_width=12,
            recurrent_layers=1,
            seed=3,
        )
    )
    trainer = subject.DirectBPTTTrainer(model, batch_size=2, learning_rate=0.01)
    before = subject.parameter_digest(model)
    chunk = subject.repeat_batch(batch, updates=12)

    losses = np.asarray(trainer.train_chunk(chunk))

    assert losses.shape == (12,)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert subject.parameter_digest(model) != before


def test_direct_loss_penalizes_rare_color_error_more_than_background() -> None:
    subject = _subject()
    height = jnp.asarray([[20.0, -20.0]])
    width = jnp.asarray([[20.0, -20.0]])
    target_colors = jnp.asarray([[[0, 0], [0, 7]]])
    mask = jnp.ones_like(target_colors, dtype=jnp.float32)
    correct = jnp.full((1, 2, 2, 10), -10.0)
    correct = correct.at[0, 0, 0, 0].set(10.0)
    correct = correct.at[0, 0, 1, 0].set(10.0)
    correct = correct.at[0, 1, 0, 0].set(10.0)
    correct = correct.at[0, 1, 1, 7].set(10.0)
    background_error = correct.at[0, 0, 0, 0].set(-10.0)
    background_error = background_error.at[0, 0, 0, 1].set(10.0)
    rare_error = correct.at[0, 1, 1, 7].set(-10.0)
    rare_error = rare_error.at[0, 1, 1, 1].set(10.0)
    dimensions = jnp.asarray([0])

    background_loss = subject.direct_prediction_loss(
        (height, width, background_error),
        dimensions,
        dimensions,
        target_colors,
        mask,
    )
    rare_loss = subject.direct_prediction_loss(
        (height, width, rare_error),
        dimensions,
        dimensions,
        target_colors,
        mask,
    )

    assert float(rare_loss) > float(background_loss)


def test_hard_cell_error_is_not_diluted_by_large_grid() -> None:
    subject = _subject()
    shape_logits = jnp.full((1, 30), -20.0)
    shape_logits = shape_logits.at[0, 29].set(20.0)
    colors = jnp.full((1, 30, 30, 10), -10.0)
    colors = colors.at[..., 0].set(10.0)
    colors = colors.at[0, 0, 0, 0].set(-10.0)
    colors = colors.at[0, 0, 0, 1].set(10.0)
    targets = jnp.zeros((1, 30, 30), dtype=jnp.int32)
    small_mask = jnp.zeros((1, 30, 30), dtype=jnp.float32)
    small_mask = small_mask.at[0, 0, 0].set(1.0)
    large_mask = jnp.ones((1, 30, 30), dtype=jnp.float32)
    dimensions = jnp.asarray([29])

    small_loss = subject.direct_prediction_loss(
        (shape_logits, shape_logits, colors),
        dimensions,
        dimensions,
        targets,
        small_mask,
    )
    large_loss = subject.direct_prediction_loss(
        (shape_logits, shape_logits, colors),
        dimensions,
        dimensions,
        targets,
        large_mask,
    )

    assert float(large_loss) > 0.2 * float(small_loss)


def test_training_interfaces_reject_invalid_inputs() -> None:
    subject = _subject()
    row_config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    with pytest.raises(ValueError, match="at least two"):
        subject.leave_one_out_tasks(
            ArcTask(
                train=(ArcPair(ArcGrid(((0,),)), ArcGrid(((0,),))),),
                test=(ArcPair(ArcGrid(((0,),)), ArcGrid(((0,),))),),
            )
        )
    with pytest.raises(ValueError, match="nonempty"):
        subject.stack_direct_episodes(())
    batch = subject.stack_direct_episodes(
        (subject.encode_direct_episode(subject.leave_one_out_tasks(_task())[0], 0, row_config),)
    )
    with pytest.raises(ValueError, match="positive"):
        subject.repeat_batch(batch, updates=0)
    with pytest.raises(TypeError, match="DirectARCGRU"):
        subject.parameter_digest(object())


def test_checkpoint_round_trip_is_byte_exact(tmp_path) -> None:
    subject = _subject()
    model_subject = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_model"
    )
    config = model_subject.DirectModelConfig(
        input_width=6,
        encoder_width=4,
        hidden_width=8,
        decoder_width=6,
        seed=17,
    )
    model = model_subject.DirectARCGRU(config)
    path = tmp_path / "checkpoint.npz"

    saved = subject.save_direct_checkpoint(model, path)
    loaded, metadata = subject.load_direct_checkpoint(path)

    assert saved == subject.parameter_digest(model)
    assert subject.parameter_digest(loaded) == saved
    assert metadata["parameter_sha256"] == saved
    assert loaded.config == config
    events = jnp.ones((3, 1, 6), dtype=jnp.float32)
    query = jnp.zeros((1, 30, 30, 11), dtype=jnp.float32)
    brainstate.nn.init_all_states(model, batch_size=1)
    original = tuple(np.asarray(value) for value in model.run(events, query))
    brainstate.nn.init_all_states(loaded, batch_size=1)
    reloaded = tuple(np.asarray(value) for value in loaded.run(events, query))
    assert all(
        left.tobytes() == right.tobytes()
        for left, right in zip(original, reloaded, strict=True)
    )


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "schema"])
def test_checkpoint_load_rejects_schema_corruption(tmp_path, corruption: str) -> None:
    subject = _subject()
    model_subject = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_model"
    )
    model = model_subject.DirectARCGRU(
        model_subject.DirectModelConfig(
            input_width=6,
            encoder_width=4,
            hidden_width=8,
            decoder_width=6,
            recurrent_layers=1,
            seed=19,
        )
    )
    path = tmp_path / "checkpoint.npz"
    subject.save_direct_checkpoint(model, path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = msgspec.json.decode(bytes(payload.pop("__metadata__")))
    if corruption == "missing":
        payload.pop("leaf_0000")
    elif corruption == "shape":
        original_shape = payload["leaf_0000"].shape
        payload["leaf_0000"] = payload["leaf_0000"].reshape(1, -1)
        assert payload["leaf_0000"].shape != original_shape
    elif corruption == "dtype":
        payload["leaf_0000"] = payload["leaf_0000"].astype(np.float64)
    else:
        metadata["schema_version"] = 999
    payload["__metadata__"] = np.frombuffer(msgspec.json.encode(metadata), dtype=np.uint8)
    with path.open("wb") as stream:
        np.savez(stream, **payload)

    with pytest.raises(ValueError, match="checkpoint"):
        subject.load_direct_checkpoint(path)
