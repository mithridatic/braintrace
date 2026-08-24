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


def _confident_output(color: int) -> jnp.ndarray:
    row = jnp.full((1, 30, 10), -10.0).at[0, :, color].set(10.0)
    height = jnp.full((1, 30), -10.0).at[0, 0].set(10.0)
    width = jnp.full((1, 30), -10.0).at[0, 29].set(10.0)
    return jnp.concatenate((row.reshape(1, -1), height, width), axis=-1)


def test_v23_loss_equalizes_foreground_and_background_partition_mass() -> None:
    subject = _subject()
    online = _online_subject()
    target = jnp.asarray([[7] + [0] * 29])
    mask = jnp.ones((1, 30), dtype=jnp.float32)
    weights = jnp.asarray(
        [[np.sqrt(30.0 / 29.0)] + [0.0] * 6 + [4.0] + [0.0] * 2]
    )
    zero_prediction = _confident_output(0)
    foreground_prediction = _confident_output(7)
    dimensions = jnp.asarray([0])

    v22_zero = online.online_step_loss(
        zero_prediction, target, mask, dimensions, jnp.asarray([29]), weights
    )
    v22_foreground = online.online_step_loss(
        foreground_prediction, target, mask, dimensions, jnp.asarray([29]), weights
    )
    v23_zero = subject.foreground_background_step_loss(
        zero_prediction, target, mask, dimensions, jnp.asarray([29]), weights
    )
    v23_foreground = subject.foreground_background_step_loss(
        foreground_prediction, target, mask, dimensions, jnp.asarray([29]), weights
    )

    assert float(v22_zero) < float(v22_foreground)
    assert float(v23_zero) == pytest.approx(float(v23_foreground), rel=1e-5)


def test_v23_loss_handles_all_background_and_ignores_masked_cells() -> None:
    subject = _subject()
    target = jnp.zeros((1, 30), dtype=jnp.int32)
    mask = jnp.zeros((1, 30), dtype=jnp.float32).at[0, :3].set(1.0)
    weights = jnp.asarray([[1.0] + [0.0] * 9])
    baseline = _confident_output(0)
    outside_error = baseline.at[0, 29 * 10].set(-20.0)
    dimensions = jnp.asarray([0])

    baseline_loss = subject.foreground_background_step_loss(
        baseline, target, mask, dimensions, jnp.asarray([29]), weights
    )
    outside_loss = subject.foreground_background_step_loss(
        outside_error, target, mask, dimensions, jnp.asarray([29]), weights
    )

    assert np.isfinite(float(baseline_loss))
    assert float(outside_loss) == pytest.approx(float(baseline_loss))


def test_v24_loss_equalizes_every_present_color_not_just_foreground() -> None:
    subject = _subject()
    target = jnp.asarray([[7, 8] + [0] * 28])
    mask = jnp.ones((1, 30), dtype=jnp.float32)
    weights = jnp.asarray(
        [[np.sqrt(30.0 / 28.0)] + [0.0] * 6 + [4.0, 4.0, 0.0]]
    )
    zero_prediction = _confident_output(0)
    seven_prediction = _confident_output(7)
    eight_prediction = _confident_output(8)
    dimensions = jnp.asarray([0])
    widths = jnp.asarray([29])

    v23_zero = subject.foreground_background_step_loss(
        zero_prediction, target, mask, dimensions, widths, weights
    )
    v23_seven = subject.foreground_background_step_loss(
        seven_prediction, target, mask, dimensions, widths, weights
    )
    v24_zero = subject.present_color_step_loss(
        zero_prediction, target, mask, dimensions, widths, weights
    )
    v24_seven = subject.present_color_step_loss(
        seven_prediction, target, mask, dimensions, widths, weights
    )
    v24_eight = subject.present_color_step_loss(
        eight_prediction, target, mask, dimensions, widths, weights
    )

    assert float(v23_zero) < float(v23_seven)
    assert float(v24_zero) == pytest.approx(float(v24_seven), rel=1e-5)
    assert float(v24_zero) == pytest.approx(float(v24_eight), rel=1e-5)


def test_v25_loss_balances_present_colors_across_the_whole_grid() -> None:
    subject = _subject()
    steps = 10
    rows = jnp.zeros((steps, 1, 30), dtype=jnp.int32)
    rows = rows.at[-1, 0, 0].set(7)
    masks = jnp.zeros((steps, 1, 30), dtype=jnp.float32).at[:, 0, :10].set(1.0)
    dimensions = jnp.zeros((steps, 1), dtype=jnp.int32)
    widths = jnp.full((steps, 1), 9, dtype=jnp.int32)
    v21_weights = jnp.broadcast_to(
        jnp.asarray([[np.sqrt(100.0 / 99.0)] + [0.0] * 6 + [4.0] + [0.0] * 2]),
        (steps, 1, 10),
    )
    zero_outputs = jnp.broadcast_to(_confident_output(0), (steps, 1, 360))
    seven_outputs = jnp.broadcast_to(_confident_output(7), (steps, 1, 360))

    v24 = jax.vmap(subject.present_color_step_loss)
    v24_zero = jnp.mean(
        v24(zero_outputs, rows, masks, dimensions, widths, v21_weights)
    )
    v24_seven = jnp.mean(
        v24(seven_outputs, rows, masks, dimensions, widths, v21_weights)
    )
    color_mass = subject.whole_grid_color_mass(rows, masks)
    v25 = jax.vmap(subject.whole_grid_present_color_step_loss)
    v25_zero = jnp.mean(
        v25(zero_outputs, rows, masks, dimensions, widths, color_mass)
    )
    v25_seven = jnp.mean(
        v25(seven_outputs, rows, masks, dimensions, widths, color_mass)
    )

    assert float(v24_zero) < float(v24_seven)
    assert float(v25_zero) == pytest.approx(float(v25_seven), rel=1e-5)
    assert float(color_mass[0, 0, 0]) == pytest.approx(30.0 / (2.0 * 99.0))
    assert float(color_mass[-1, 0, 7]) == pytest.approx(15.0)
    assert np.count_nonzero(np.asarray(color_mass[..., 1:7])) == 0
    assert np.count_nonzero(np.asarray(color_mass[..., 8:])) == 0


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
    assert trainer.loss_version == "whole_grid_present_color_balanced_v25"


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
    assert first["diagnostics"]["shape_exact_count"] >= 0
    assert sum(first["diagnostics"]["predicted_color_counts"].values()) == sum(
        candidate["height"] * candidate["width"] for candidate in first["candidates"]
    )
    assert first["candidates"][0]["answer_head_version"] == (
        "spatial_conv_lif_row_decoder_v22"
    )
