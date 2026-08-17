"""Tests for the legacy Example 21 minimal-binding capability control."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop import latent_workspace_binding_control as control


def test_config_preregisters_k4_finite_window_and_unique_capacity():
    config = control.BindingControlConfig.smoke_config()

    assert config.symbol_count == 4
    assert config.sequence_length == 6
    assert config.configuration_scale == "reduced_smoke"
    assert control.BindingControlConfig().configuration_scale == "production_topology"
    assert config.gradient_chunk_size < config.sequence_length
    assert config.training_episode_count + config.validation_episodes <= 1_058_400

    with pytest.raises(ValueError, match="shorter than the sequence"):
        dataclasses.replace(config, gradient_chunk_size=config.sequence_length)
    with pytest.raises(ValueError, match="unique K=4 mapping catalog"):
        dataclasses.replace(
            config,
            training_updates=264_599,
            batch_size=4,
            validation_episodes=8,
        )
    with pytest.raises(ValueError, match="divisible by 64"):
        dataclasses.replace(config, neuron_count=65)


def test_binding_data_is_deterministic_fresh_and_split_disjoint():
    config = control.BindingControlConfig.smoke_config()
    first = control.build_binding_data(config)
    second = control.build_binding_data(config)
    train_ids = first.training_mapping_ids.reshape(-1)

    assert np.array_equal(first.training_events, second.training_events)
    assert np.array_equal(first.training_targets, second.training_targets)
    assert np.array_equal(first.validation_intact, second.validation_intact)
    assert np.unique(train_ids).size == train_ids.size
    assert np.unique(first.validation_mapping_ids).size == config.validation_episodes
    assert not np.intersect1d(train_ids, first.validation_mapping_ids).size
    assert first.training_events.shape == (
        config.training_updates,
        config.sequence_length,
        config.batch_size,
        config.row_config.input_width,
    )
    assert not first.training_events.flags.writeable

    changed = control.build_binding_data(
        dataclasses.replace(config, split_seed=config.split_seed + 1)
    )
    assert not np.array_equal(first.training_mapping_ids, changed.training_mapping_ids)


def test_controls_preserve_marginals_and_timing_but_change_pairing():
    config = control.BindingControlConfig.smoke_config()
    data = control.build_binding_data(config)
    rows = config.row_config
    intact = data.validation_intact
    shuffled = data.validation_shuffled
    no_context = data.validation_no_context

    assert intact.shape == shuffled.shape == no_context.shape
    assert np.array_equal(
        intact[control.SYMBOL_COUNT :], shuffled[control.SYMBOL_COUNT :]
    )
    assert np.array_equal(
        intact[control.SYMBOL_COUNT :], no_context[control.SYMBOL_COUNT :]
    )
    assert not np.count_nonzero(no_context[: control.SYMBOL_COUNT])
    assert not np.count_nonzero(intact[-config.gap_steps :])
    assert not np.count_nonzero(shuffled[-config.gap_steps :])

    input_slice = rows.input_color_slice
    output_slice = rows.output_color_slice
    assert np.array_equal(
        intact[: control.SYMBOL_COUNT, :, input_slice],
        shuffled[: control.SYMBOL_COUNT, :, input_slice],
    )
    assert np.array_equal(
        intact[: control.SYMBOL_COUNT, :, output_slice].sum(axis=0),
        shuffled[: control.SYMBOL_COUNT, :, output_slice].sum(axis=0),
    )
    assert not np.array_equal(
        intact[: control.SYMBOL_COUNT, :, output_slice],
        shuffled[: control.SYMBOL_COUNT, :, output_slice],
    )


def test_every_encoded_episode_is_a_four_way_one_cell_bijection():
    config = control.BindingControlConfig.smoke_config()
    data = control.build_binding_data(config)
    rows = config.row_config
    demos = data.validation_intact[: control.SYMBOL_COUNT]

    for episode in range(config.validation_episodes):
        input_colors = []
        output_colors = []
        for demonstration in range(control.SYMBOL_COUNT):
            input_features = demos[demonstration, episode, rows.input_color_slice]
            output_features = demos[demonstration, episode, rows.output_color_slice]
            input_colors.append(int(np.flatnonzero(input_features)[0]))
            output_colors.append(int(np.flatnonzero(output_features)[0]))
        assert len(set(input_colors)) == control.SYMBOL_COUNT
        assert len(set(output_colors)) == control.SYMBOL_COUNT


def test_seen_training_probe_uses_the_shared_schedule_and_rotates_only_outputs():
    config = control.BindingControlConfig.smoke_config()
    data = control.build_binding_data(config)
    intact, shuffled, no_context, targets = control._training_probe(data, config)
    rows = config.row_config

    expected = data.training_events.transpose(0, 2, 1, 3).reshape(
        config.training_episode_count,
        config.sequence_length,
        rows.input_width,
    )[: config.validation_episodes]
    assert np.array_equal(intact, expected.transpose(1, 0, 2))
    assert np.array_equal(targets, data.training_targets.reshape(-1))
    assert np.array_equal(
        intact[: control.SYMBOL_COUNT, :, rows.input_color_slice],
        shuffled[: control.SYMBOL_COUNT, :, rows.input_color_slice],
    )
    assert np.array_equal(
        intact[: control.SYMBOL_COUNT, :, rows.output_color_slice].sum(axis=0),
        shuffled[: control.SYMBOL_COUNT, :, rows.output_color_slice].sum(axis=0),
    )
    assert not np.count_nonzero(no_context[: control.SYMBOL_COUNT])


def test_terminal_residual_wrapper_gates_every_unsupervised_tick():
    config = control.BindingControlConfig.smoke_config()
    data = control.build_binding_data(config)
    packed = control._oracle_inputs(data, config)
    model = control._TerminalResidualModel(config)

    outputs = brainstate.transform.for_loop(model.update, packed)

    assert outputs.shape == (config.sequence_length, 1, control.COLOR_COUNT)
    assert np.count_nonzero(np.asarray(outputs[:-1])) == 0
    assert bool(jnp.all(jnp.isfinite(outputs[-1])))
    assert float(jnp.linalg.norm(outputs[-1])) > 0.0


def test_model_drivers_use_compiled_brainstate_loops_not_python_loops():
    tree = ast.parse(Path(control.__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("train_all", "train_one", "predict"):
        matches = [node for key, node in functions.items() if key == name]
        assert matches
        for node in matches:
            assert not any(
                isinstance(child, (ast.For, ast.While)) for child in ast.walk(node)
            )
    assert "brainstate.transform.for_loop" in Path(control.__file__).read_text(
        encoding="utf-8"
    )
    assert "jax.random" not in Path(control.__file__).read_text(encoding="utf-8")


def test_interpretation_is_scale_gated_and_covers_preregistered_outcomes():
    production = control.BindingControlConfig(
        training_updates=1,
        batch_size=4,
        validation_episodes=4,
    )
    reduced = control.BindingControlConfig.smoke_config()

    def result(intact: float, gap: float) -> dict[str, object]:
        return {
            "intact": {"accuracy": intact},
            "intact_minus_shuffled": gap,
        }

    binds = result(0.9, 0.4)
    fails = result(0.3, 0.05)
    assert (
        control._interpretation(fails, fails, reduced)
        == "reduced_smoke_only_no_architecture_conclusion"
    )
    assert (
        control._interpretation(fails, fails, production)
        == "legacy_architecture_necessary_bptt_also_fails_binding"
    )
    assert (
        control._interpretation(binds, fails, production)
        == "pp_prop_truncation_blocker_bptt_binds"
    )
    assert (
        control._interpretation(binds, binds, production)
        == "both_bind_increase_only_the_preregistered_gap"
    )
    assert (
        control._interpretation(fails, binds, production)
        == "invalid_control_pp_prop_binds_while_bptt_fails"
    )


def test_smoke_control_runs_both_arms_and_finite_window_oracle():
    config = control.BindingControlConfig.smoke_config()
    result = control.run_binding_control(config)

    assert result["initialization"]["byte_identical"] is True
    assert result["config"]["configuration_scale"] == "reduced_smoke"
    assert (
        result["training"]["bptt"]["parameter_sha256_before"]
        == result["training"]["pp_prop"]["parameter_sha256_before"]
    )
    assert len(result["training"]["bptt"]["losses"]) == config.training_updates
    assert len(result["training"]["pp_prop"]["losses"]) == config.training_updates
    assert np.isfinite(result["training"]["bptt"]["losses"]).all()
    assert np.isfinite(result["training"]["pp_prop"]["losses"]).all()
    for algorithm in ("bptt", "pp_prop"):
        evaluation = result["evaluation"][algorithm]
        for arm in ("intact", "shuffled", "no_context"):
            assert 0.0 <= evaluation[arm]["accuracy"] <= 1.0
            assert evaluation[arm]["count"] == config.validation_episodes
            assert (
                evaluation["training_probe"][arm]["count"] == config.validation_episodes
            )
    oracle = result["gradient_oracle"]
    assert oracle["finite_window"] is True
    assert oracle["chunk_size"] < oracle["sequence_length"]
    assert oracle["bptt_norm"] > 0.0
    assert oracle["pp_prop_norm"] > 0.0
    assert oracle["relative_deviation"] > 1e-7
    assert oracle["by_parameter_group"]["feedforward"]["bptt_norm"] > 0.0
    assert result["interpretation"] in {
        "reduced_smoke_only_no_architecture_conclusion",
    }


def test_artifact_writer_emits_strict_json(tmp_path):
    payload = {
        "finite": 1.0,
        "nonfinite": float("nan"),
        "array": np.asarray([1, 2], dtype=np.int32),
    }

    destination = control.write_artifact(payload, tmp_path / "control.json")
    parsed = json.loads(destination.read_text(encoding="utf-8"))

    assert parsed == {"array": [1, 2], "finite": 1.0, "nonfinite": None}
    assert not (tmp_path / "control.json.tmp").exists()
