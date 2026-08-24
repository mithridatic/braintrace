"""Tests for target-isolated row-decoded PP-prop training."""

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
    return importlib.import_module("examples.pp_prop.latent_workspace_online_training")


def _model_subject():
    return importlib.import_module("examples.pp_prop.latent_workspace_online_model")


@pytest.fixture(autouse=True)
def _clear_compilation_caches():
    yield
    jax.clear_caches()


def _task(output_color: int = 3) -> ArcTask:
    return ArcTask(
        train=(
            ArcPair(ArcGrid(((1, 0),)), ArcGrid(((2, 0),))),
            ArcPair(ArcGrid(((4, 0),)), ArcGrid(((5, 0),))),
        ),
        test=(ArcPair(ArcGrid(((6, 0),)), ArcGrid(((output_color, 0),))),),
        task_id="online-tiny",
    )


def test_packing_is_lossless_ordered_and_target_independent() -> None:
    subject = _subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    first = subject.encode_online_episode(_task(3), 0, config)
    changed = subject.encode_online_episode(_task(9), 0, config)
    direct = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_training"
    ).encode_direct_episode(_task(3), 0, config)
    valid_rows = direct.events[direct.events[:, config.valid_slice.start] == 1.0]
    input_start = config.max_events - len(valid_rows)

    assert first.events.tobytes() == changed.events.tobytes()
    assert first.decode_mask.tobytes() == changed.decode_mask.tobytes()
    assert first.target_rows.tobytes() != changed.target_rows.tobytes()
    assert first.events.shape == (config.max_events + 30, config.input_width + 31)
    assert (
        first.events[input_start : config.max_events, : config.input_width].tobytes()
        == valid_rows.tobytes()
    )
    assert np.count_nonzero(first.events[:input_start]) == 0
    assert np.count_nonzero(
        first.events[input_start : config.max_events, config.input_width :]
    ) == 0
    decode = first.events[config.max_events :]
    assert np.all(decode[:, config.input_width] == 1.0)
    assert np.count_nonzero(decode[:, : config.input_width]) == 0
    assert np.array_equal(decode[:, config.input_width + 1 :], np.eye(30, dtype=np.float32))
    assert first.decode_mask.sum() == 30


def test_stack_and_repeat_keep_time_major_targets_out_of_events() -> None:
    subject = _subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episode = subject.encode_online_episode(_task(), 0, config)

    batch = subject.stack_online_episodes((episode, episode))
    chunk = subject.repeat_online_batch(batch, updates=4)

    assert batch.events.shape == (config.max_events + 30, 2, config.input_width + 31)
    assert batch.target_rows.shape == (config.max_events + 30, 2, 30)
    assert batch.target_cell_mask.shape == batch.target_rows.shape
    assert batch.decode_mask.shape == (config.max_events + 30,)
    assert chunk.events.shape[0] == 4
    assert chunk.target_heights.shape == (4, config.max_events + 30, 2)


def test_online_step_loss_masks_cells_and_scores_shape() -> None:
    subject = _subject()
    model_subject = _model_subject()
    row_logits = jnp.full((1, 30, 10), -10.0).at[0, :, 0].set(10.0)
    height = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    width = jnp.full((1, 30), -10.0).at[0, 1].set(10.0)
    output = jnp.concatenate((row_logits.reshape(1, -1), height, width), axis=-1)
    target_row = jnp.zeros((1, 30), dtype=jnp.int32)
    target_mask = jnp.zeros((1, 30), dtype=jnp.float32).at[0, :2].set(1.0)

    correct = subject.online_step_loss(
        output, target_row, target_mask, jnp.asarray([0]), jnp.asarray([1])
    )
    wrong_output = output.at[0, model_subject.ROW_COLOR_WIDTH :].set(0.0)
    wrong = subject.online_step_loss(
        wrong_output, target_row, target_mask, jnp.asarray([0]), jnp.asarray([1])
    )

    assert float(correct) < float(wrong)


def test_pp_prop_compiler_descent_pilot_moves_all_parameter_groups() -> None:
    subject = _subject()
    model_subject = _model_subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episode = subject.encode_online_episode(_task(), 0, config)
    batch = subject.stack_online_episodes((episode, episode))
    model = model_subject.OnlineARCGRU(
        model_subject.OnlineModelConfig(
            input_width=config.input_width + 31,
            encoder_width=8,
            hidden_width=12,
            recurrent_layers=2,
            seed=13,
        )
    )
    trainer = subject.OnlinePPPropTrainer(
        model,
        batch_size=2,
        learning_rate=0.01,
        trace_decay=2.0 ** (-1.0 / 40.0),
    )
    before = subject.parameter_arrays(model)

    @brainstate.transform.jit
    def forward(events):
        brainstate.nn.reset_all_states(model, batch_size=2)
        return brainstate.transform.for_loop(model, events)

    candidate_before = subject.decode_online_outputs(
        np.asarray(forward(jnp.asarray(batch.events)))[:, 0],
        episode.decode_mask,
        (),
    )

    losses, gradient_norms = trainer.train_chunk(
        subject.repeat_online_batch(batch, updates=5)
    )
    after = subject.parameter_arrays(model)
    candidate_after = subject.decode_online_outputs(
        np.asarray(forward(jnp.asarray(batch.events)))[:, 0],
        episode.decode_mask,
        (),
    )

    assert np.asarray(losses).shape == (5,)
    assert np.all(np.isfinite(np.asarray(losses)))
    assert set(gradient_norms) == {
        "recurrent",
        "row_color",
        "height",
        "width",
    }
    assert all(np.isfinite(float(value)) and float(value) > 0.0 for value in gradient_norms.values())
    assert all(before[name].tobytes() != after[name].tobytes() for name in before)
    assert msgspec.json.encode(candidate_before["grid"]) != msgspec.json.encode(
        candidate_after["grid"]
    )
    assert trainer.algorithm == "pp_prop"
    assert trainer.vjp_method == "single-step"


def test_sampling_is_brainstate_deterministic_and_target_isolated() -> None:
    subject = _subject()
    direct = importlib.import_module(
        "examples.pp_prop.latent_workspace_direct_training"
    )
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    catalog = direct.leave_one_out_tasks(_task())

    first = subject.sample_online_training_chunk(
        catalog,
        config,
        brainstate.random.RandomState(31),
        updates=2,
        batch_size=2,
        augment=False,
    )
    second = subject.sample_online_training_chunk(
        catalog,
        config,
        brainstate.random.RandomState(31),
        updates=2,
        batch_size=2,
        augment=False,
    )

    assert first.events.tobytes() == second.events.tobytes()
    assert first.target_rows.tobytes() == second.target_rows.tobytes()
    with pytest.raises(TypeError, match="RandomState"):
        subject.sample_online_training_chunk(
            catalog, config, object(), updates=1, batch_size=1, augment=False
        )


def test_target_free_evaluation_is_deterministic() -> None:
    subject = _subject()
    model_subject = _model_subject()
    config = RowEventConfig(max_demonstrations=2, max_grid_size=3)
    episodes = (
        subject.encode_online_episode(_task(3), 0, config),
        subject.encode_online_episode(_task(9), 0, config),
    )
    model = model_subject.OnlineARCGRU(
        model_subject.OnlineModelConfig(
            input_width=config.input_width + 31,
            encoder_width=6,
            hidden_width=8,
            recurrent_layers=2,
            seed=37,
        )
    )

    first = subject.evaluate_online_model(model, episodes, batch_size=2)
    second = subject.evaluate_online_model(model, episodes, batch_size=2)

    assert first["candidate_sha256"] == second["candidate_sha256"]
    assert first["candidates"][0]["grid"] == first["candidates"][1]["grid"]
    assert first["query_count"] == 2
    assert first["task_count"] == 1
    assert first["candidates"][0]["parameter_dependencies"]


def test_checkpoint_roundtrip_preserves_outputs_and_digest(tmp_path) -> None:
    subject = _subject()
    model_subject = _model_subject()
    config = model_subject.OnlineModelConfig(
        input_width=7,
        encoder_width=5,
        hidden_width=9,
        recurrent_layers=2,
        seed=17,
    )
    model = model_subject.OnlineARCGRU(config)
    path = tmp_path / "online.npz"

    digest = subject.save_online_checkpoint(model, path)
    loaded, metadata = subject.load_online_checkpoint(path)

    assert metadata["parameter_sha256"] == digest
    assert subject.parameter_digest(loaded) == digest
    inputs = jnp.ones((2, 7), dtype=jnp.float32)
    brainstate.nn.init_all_states(model, batch_size=2)
    original = np.asarray(model(inputs))
    brainstate.nn.init_all_states(loaded, batch_size=2)
    restored = np.asarray(loaded(inputs))
    assert original.tobytes() == restored.tobytes()


@pytest.mark.parametrize("corruption", ["missing", "shape", "nonfinite", "schema"])
def test_checkpoint_rejects_exact_schema_corruption(tmp_path, corruption: str) -> None:
    subject = _subject()
    model_subject = _model_subject()
    model = model_subject.OnlineARCGRU(
        model_subject.OnlineModelConfig(
            input_width=7,
            encoder_width=5,
            hidden_width=9,
            recurrent_layers=2,
            seed=19,
        )
    )
    path = tmp_path / "online.npz"
    subject.save_online_checkpoint(model, path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = msgspec.json.decode(bytes(payload.pop("__metadata__")))
    if corruption == "missing":
        payload.pop("leaf_0000")
    elif corruption == "shape":
        payload["leaf_0000"] = payload["leaf_0000"].reshape(1, -1)
    elif corruption == "nonfinite":
        payload["leaf_0000"] = payload["leaf_0000"].copy()
        payload["leaf_0000"].flat[0] = np.nan
    else:
        metadata["schema_version"] = 999
    payload["__metadata__"] = np.frombuffer(
        msgspec.json.encode(metadata), dtype=np.uint8
    )
    with path.open("wb") as stream:
        np.savez(stream, **payload)

    with pytest.raises(ValueError, match="checkpoint"):
        subject.load_online_checkpoint(path)


def test_decode_online_outputs_uses_only_fixed_decode_steps() -> None:
    subject = _subject()
    model_subject = _model_subject()
    time = 40
    outputs = np.zeros((time, model_subject.OUTPUT_WIDTH), dtype=np.float32)
    decode_mask = np.zeros((time,), dtype=np.bool_)
    decode_mask[5:35] = True
    outputs[5:35, model_subject.ROW_COLOR_WIDTH] = 10.0
    outputs[5:35, model_subject.ROW_COLOR_WIDTH + 30 + 1] = 10.0
    for row in range(30):
        outputs[5 + row, : model_subject.ROW_COLOR_WIDTH : 10] = 10.0

    candidate = subject.decode_online_outputs(outputs, decode_mask, ())

    assert candidate["height"] == 1
    assert candidate["width"] == 2
    assert candidate["grid"] == [[0, 0]]
    assert candidate["parameter_dependencies"] == []
