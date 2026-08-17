"""Tests for the in-context latent-reasoning entry point."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import pathlib
import sys
from types import SimpleNamespace

import brainstate
import numpy as np
import pytest

try:
    from examples.pp_prop.latent_workspace_model import shuffled_memory_factors
except ModuleNotFoundError:
    from latent_workspace_model import shuffled_memory_factors


EXAMPLE = (
    pathlib.Path(__file__).resolve().with_name("21-latent-reasoning-in-context.py")
)


def _load():
    name = "_pp_prop_latent_reasoning_entry"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example 21 from {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fake_train_depth(calls):
    def train(config, depth, corpus):
        calls.append(
            (
                depth,
                tuple(corpus.binding_counts.tolist()),
                tuple(corpus.targets.tolist()),
            )
        )
        model = SimpleNamespace(depth=depth, config=SimpleNamespace(task=None))
        return model, {
            "depth": depth,
            "losses": [1.0 / (depth + 1)],
            "parameter_l2_deltas": {
                "Wk": 0.0,
                "Wv": 0.0,
                "Wf": 0.01,
                "Wo": 0.01,
            },
            "terminal_only_supervision": True,
            "write_mode": "fixed_random",
            "write_projections_updated": False,
            "compiler": {"warnings": [], "diagnostic_counts": {}},
            "initial_parameter_fingerprints": {
                "Wf": "same-wf",
                "Wk": "same-wk",
                "Wo": "same-wo",
                "Wv": "same-wv",
            },
            "canonical_prefix_fingerprint": "same-demo-query",
        }

    return train


def _fake_evaluate(example, calls=None):
    def evaluate(model, task, episode_cells, shuffled_flags):
        if calls is not None:
            calls.append(len(episode_cells))
        batches = []
        for episodes, shuffled in zip(episode_cells, shuffled_flags, strict=True):
            count = len(episodes)
            width = 4
            depth = model.depth
            answers = np.asarray([episode.target for episode in episodes])
            states = np.zeros((count, depth + 1, width), dtype=np.float32)
            states[:, :, 0] = answers[:, None]
            states[:, :, 1] = np.arange(depth + 1, dtype=np.float32)[None, :]
            memory_read = states[:, 0].copy()
            values = np.zeros((count, task.slot_capacity, width), dtype=np.float32)
            keys = np.zeros_like(values)
            values[:, 0] = memory_read
            keys[:, 0, 0] = 1.0
            supported = episodes[0].condition == "supported"
            correct = np.full(count, supported and not shuffled, dtype=bool)
            batches.append(
                example.EvaluationBatch(
                    correct=correct,
                    workspace=states,
                    memory_read=memory_read,
                    memory_values=values,
                    memory_keys=keys,
                    parameters_unchanged=True,
                )
            )
        return tuple(batches)

    return evaluate


def _smoke_config(example, figure_path, seed=2108):
    return example.ExperimentConfig.smoke(
        seed=seed,
        figure_path=figure_path,
        device="cpu",
    )


def test_default_cli_requests_gpu_and_cpu_is_explicit(tmp_path):
    example = _load()
    default = example._parse_args([])
    cpu = example._parse_args(
        [
            "--device",
            "cpu",
            "--smoke",
            "--codebook-seed",
            "41",
            "--projection-seed",
            "42",
            "--figure",
            str(tmp_path / "smoke.png"),
        ]
    )

    assert default.device == "gpu"
    default_config = example._config_from_args(default)
    assert default_config.batch_size == 4
    assert default_config.codebook_seed == 313320
    assert default_config.projection_seed == 210848
    assert cpu.device == "cpu"
    assert cpu.smoke is True
    cpu_config = example._config_from_args(cpu)
    assert cpu_config.codebook_seed == 41
    assert cpu_config.projection_seed == 42


@pytest.mark.parametrize("field", ["codebook_seed", "projection_seed"])
@pytest.mark.parametrize("value", [True, np.bool_(False), -1, 1.5])
def test_experiment_code_and_projection_seeds_are_nonnegative_non_boolean_integers(
    field, value
):
    example = _load()

    with pytest.raises(ValueError, match=field):
        example.ExperimentConfig(device="cpu", **{field: value})


def test_measured_seeds_reach_every_task_and_model_configuration(tmp_path, monkeypatch):
    example = _load()
    config = example.ExperimentConfig.smoke(
        device="cpu",
        figure_path=tmp_path / "seeds.png",
        codebook_seed=71,
        projection_seed=72,
    )

    assert example._model_task(config, 0).codebook_seed == 71
    assert example._episode_task(config, 2).codebook_seed == 71

    captured = {}
    sentinel = object()

    def capture_model_config(**kwargs):
        captured.update(kwargs)
        return sentinel

    def stop_after_configuration(model_config):
        assert model_config is sentinel
        raise RuntimeError("configuration captured")

    monkeypatch.setattr(example, "ModelConfig", capture_model_config)
    monkeypatch.setattr(example, "LatentWorkspaceModel", stop_after_configuration)
    with pytest.raises(RuntimeError, match="configuration captured"):
        example._train_depth(config, 0, None)

    assert captured["projection_seed"] == 72


def test_gpu_request_fails_closed_when_no_gpu_is_visible(monkeypatch):
    example = _load()

    def unavailable(platform):
        if platform == "gpu":
            raise RuntimeError("backend is unavailable")
        return [SimpleNamespace(platform="cpu", id=0)]

    monkeypatch.setattr(example, "_devices_for_platform", unavailable)
    with pytest.raises(RuntimeError, match="GPU.*--device cpu"):
        example._resolve_device("gpu")


def test_experiment_configuration_rejects_contradictions(tmp_path):
    example = _load()
    base = dict(
        seed=1,
        device="cpu",
        depths=(0, 1),
        binding_counts=(2, 8),
        batch_size=2,
        training_updates=1,
        latent_width=4,
        code_width=12,
        symbol_ticks=1,
        figure_path=tmp_path / "x.png",
        learning_rate=1e-5,
    )
    with pytest.raises(ValueError, match="depth"):
        example.ExperimentConfig(**(base | {"depths": (0, -1)}))
    with pytest.raises(ValueError, match="binding count.*capacity"):
        example.ExperimentConfig(**(base | {"binding_counts": (2, 9)}))
    with pytest.raises(ValueError, match="batch_size"):
        example.ExperimentConfig(**(base | {"batch_size": 1}))
    with pytest.raises(ValueError, match="batch_size 8.*170459136.*67108864"):
        example.ExperimentConfig(**(base | {"batch_size": 8, "latent_width": 32}))
    default = example.ExperimentConfig(device="cpu")
    assert default.coupled_jacobian_elements == 42_614_784
    assert default.jacobian_budget_elements == 67_108_864


def test_shared_mixed_training_corpus_is_seeded_and_depth_independent(tmp_path):
    example = _load()
    config = _smoke_config(example, tmp_path / "a.png", seed=77)

    first = example._build_training_corpus(config)
    second = example._build_training_corpus(config)

    np.testing.assert_array_equal(first.binding_counts, second.binding_counts)
    np.testing.assert_array_equal(first.targets, second.targets)
    np.testing.assert_array_equal(first.rules, second.rules)
    assert len(set(first.binding_counts.tolist())) >= 2


def test_canonical_inputs_pad_capacity_without_changing_query(tmp_path):
    example = _load()
    config = _smoke_config(example, tmp_path / "a.png")
    source_task = example.TaskConfig(
        symbol_count=10,
        binding_count=2,
        slot_capacity=8,
        latent_steps=0,
        code_width=12,
        spike_rate=0.25,
        symbol_ticks=config.symbol_ticks,
    )
    episode = example.generate_episode(
        source_task,
        brainstate.random.RandomState(19),
        condition="supported",
    )
    destination = example._model_task(config, depth=4)

    packed = example._canonical_inputs(episode, destination)

    assert packed.shape == (destination.total_steps, destination.input_width)
    phase = packed[:, destination.phase_slice]
    assert np.count_nonzero(phase[:, 0]) == destination.demonstration_steps
    assert np.count_nonzero(phase[:, 1]) == destination.symbol_ticks
    assert np.count_nonzero(phase[:, 2]) == 1
    assert np.count_nonzero(phase[:, 3]) == 3
    assert phase[destination.latent_slice.start, 2] == 1.0
    np.testing.assert_array_equal(phase.sum(axis=1), 1.0)
    np.testing.assert_array_equal(packed[destination.query_slice], episode.query_inputs)

    zero_depth = example._canonical_inputs(episode, example._model_task(config, 0))
    eight_depth = example._canonical_inputs(episode, example._model_task(config, 8))
    np.testing.assert_array_equal(
        zero_depth,
        eight_depth[: zero_depth.shape[0]],
    )


def test_demonstration_prefix_uses_brainstate_loop_not_python():
    example = _load()
    tree = ast.parse(inspect.getsource(example._run_demonstration_prefix))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "for_loop"
        for node in ast.walk(tree)
    )

    train_tree = ast.parse(inspect.getsource(example._train_depth))
    train_calls = [node for node in ast.walk(train_tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "etrace_grad"
        for node in train_calls
    )
    assert any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "for_loop"
        for node in train_calls
    )


def test_depth_grid_has_one_compiled_model_call_outside_host_label_loops():
    example = _load()
    source = inspect.getsource(example._depth_interventions)

    assert source.count("_evaluate_grid(") == 1
    assert "_evaluate_batch(" not in source


@pytest.mark.parametrize("occupied_count", range(2, 9))
def test_shuffled_arm_deranges_only_occupied_slots_and_preserves_norm(
    occupied_count,
):
    example = _load()
    slot_count = 8
    permutation = example.occupied_slot_derangement(slot_count, occupied_count)
    values = np.arange(slot_count * 3, dtype=np.float32).reshape(1, slot_count, 3)
    keys = values + 100.0

    shuffled, unchanged_keys = shuffled_memory_factors(values, keys, permutation)

    permutation = np.asarray(permutation)
    occupied = np.arange(occupied_count)
    unused = np.arange(occupied_count, slot_count)
    assert np.all(permutation[:occupied_count] != occupied)
    np.testing.assert_array_equal(permutation[occupied_count:], unused)
    np.testing.assert_array_equal(
        shuffled[:, occupied_count:], values[:, occupied_count:]
    )
    np.testing.assert_array_equal(unchanged_keys, keys)
    assert np.linalg.norm(shuffled) == pytest.approx(np.linalg.norm(values))


def test_complete_smoke_grid_is_reproducible_and_json_friendly(
    tmp_path, monkeypatch, capsys
):
    example = _load()
    calls = []
    grid_calls = []
    monkeypatch.setattr(example, "_train_depth", _fake_train_depth(calls))
    monkeypatch.setattr(example, "_evaluate_grid", _fake_evaluate(example, grid_calls))
    first_config = _smoke_config(example, tmp_path / "first.png", seed=91)
    second_config = _smoke_config(example, tmp_path / "second.png", seed=91)

    first = example._run_experiment(first_config)
    second = example._run_experiment(second_config)

    assert first_config.depths == (0, 1, 2, 4, 8)
    assert first_config.binding_counts == tuple(range(2, 9))
    assert set(first["training"]["depths"]) == {"0", "1", "2", "4", "8"}
    assert set(first["interventions"]["depths"]) == {"0", "1", "2", "4", "8"}
    assert len(calls) == 10
    assert all(call[1:] == calls[0][1:] for call in calls)
    assert grid_calls == [28] * 10

    for depth in first_config.depths:
        depth_result = first["interventions"]["depths"][str(depth)]
        assert set(depth_result["per_binding_count"]) == {
            str(value) for value in range(2, 9)
        }
        for cell in depth_result["per_binding_count"].values():
            assert set(cell) == {"supported", "short"}
            assert set(cell["supported"]) == {"intact", "shuffled"}
            assert set(cell["short"]) == {"intact", "shuffled"}
        geometry = first["geometry"]["depths"][str(depth)]
        assert len(geometry["participation_ratio"]) == depth + 1
        assert len(geometry["trajectory_step_norm"]) == depth + 1
        assert len(geometry["answer_decodability"]["workspace_per_iteration"]) == (
            depth + 1
        )
        assert "raw_memory_factor_decodability" in geometry

    def stable(result):
        copy = json.loads(json.dumps(result))
        copy["figure_path"] = "<figure>"
        copy["config"]["figure_path"] = "<figure>"
        return copy

    assert stable(first) == stable(second)
    assert first["reproducibility"]["initial_parameters_identical_across_depths"]
    assert first["reproducibility"]["demo_query_inputs_identical_across_depths"]
    assert pathlib.Path(first["figure_path"]).is_file()
    assert pathlib.Path(second["figure_path"]).is_file()
    json.dumps(first, allow_nan=False)
    metadata = json.loads(first["canonical_metadata_json"])
    assert "figure_path" not in metadata["config"]
    assert metadata["config"]["depths"] == [0, 1, 2, 4, 8]
    assert metadata["config"]["binding_counts"] == list(range(2, 9))
    assert metadata["config"]["symbol_count"] == 10
    assert metadata["config"]["slot_capacity"] == 8
    assert metadata["config"]["code_width"] == 12
    assert metadata["config"]["spike_rate"] == 0.25
    assert metadata["config"]["symbol_ticks"] == 4
    assert metadata["config"]["codebook_seed"] == 313320
    assert metadata["config"]["projection_seed"] == 210848
    assert metadata["config"]["learning_rate"] == 1e-5
    assert metadata["device"]["platform"] == "cpu"
    assert first["reproducibility"]["codebook_seed"] == 313320
    assert first["reproducibility"]["projection_seed"] == 210848
    assert metadata["reproducibility"]["codebook_seed"] == 313320
    assert metadata["reproducibility"]["projection_seed"] == 210848
    assert metadata["reproducibility"]["numerical_tolerance"] == 1e-6

    series = example._figure_series(first)
    assert set(series) == {
        "accuracy_vs_depth",
        "accuracy_vs_binding_count",
        "decodability_per_iteration",
    }
    assert len(series["accuracy_vs_depth"]["x"]) == 5
    assert set(series["accuracy_vs_binding_count"]) >= {"supported", "short"}
    assert set(series["decodability_per_iteration"]) >= {
        "workspace",
        "memory_read",
    }

    report = example._render_report(first)
    assert "R=0" in report and "R=8" in report
    assert "K=2" in report and "K=8" in report
    assert "supported-short" in report
    assert "intact-shuffled" in report
    assert "fixed_random" in report
    assert "primary exact_query_memory_read" in report
    assert "secondary raw_memory_factors" in report
    assert first["canonical_metadata_json"] in report
    assert first["claim_boundary"] in report
    assert capsys.readouterr().out == ""


def test_main_prints_plain_report_and_returns_mapping(tmp_path, monkeypatch, capsys):
    example = _load()
    monkeypatch.setattr(example, "_train_depth", _fake_train_depth([]))
    monkeypatch.setattr(example, "_evaluate_grid", _fake_evaluate(example))

    result = example.main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--figure",
            str(tmp_path / "main.png"),
        ]
    )

    output = capsys.readouterr().out
    assert result["config"]["seed"] == 123
    assert result["device"]["requested"] == "cpu"
    assert "Example 21" in output
    assert "probe split" in output
    assert "interface only" in output
    assert "codebook_seed=313320" in output
    assert "projection_seed=210848" in output
    assert pathlib.Path(result["figure_path"]).is_file()

    import matplotlib

    assert str(matplotlib.get_backend()).lower() == "agg"


def test_device_resolution_accepts_explicit_cpu(monkeypatch):
    example = _load()
    cpu = SimpleNamespace(platform="cpu", id=3, device_kind="test cpu")
    monkeypatch.setattr(example, "_devices_for_platform", lambda platform: [cpu])

    resolved, report = example._resolve_device("cpu")

    assert resolved is cpu
    assert report == {
        "requested": "cpu",
        "platform": "cpu",
        "id": 3,
        "kind": "test cpu",
    }


def test_real_tiny_cpu_training_and_frozen_evaluation(tmp_path):
    example = _load()
    config = example.ExperimentConfig(
        seed=321,
        device="cpu",
        depths=(2,),
        binding_counts=(2, 8),
        batch_size=2,
        training_updates=1,
        latent_width=4,
        code_width=12,
        symbol_ticks=4,
        figure_path=tmp_path / "real-tiny.png",
        learning_rate=1e-5,
    )
    corpus = example._build_training_corpus(config)

    model, training = example._train_depth(config, 2, corpus)

    assert set(training) == {
        "depth",
        "losses",
        "parameter_l2_deltas",
        "terminal_only_supervision",
        "write_mode",
        "write_projections_updated",
        "mixed_binding_counts",
        "trainable_parameters",
        "initial_parameter_fingerprints",
        "canonical_prefix_fingerprint",
        "compiler",
    }
    assert training["terminal_only_supervision"] is True
    assert training["write_mode"] == "fixed_random"
    assert training["write_projections_updated"] is False
    assert np.all(np.isfinite(training["losses"]))
    assert training["parameter_l2_deltas"]["Wk"] == 0.0
    assert training["parameter_l2_deltas"]["Wv"] == 0.0
    assert (
        max(
            training["parameter_l2_deltas"]["Wf"],
            training["parameter_l2_deltas"]["Wo"],
        )
        > 0.0
    )
    assert set(training["trainable_parameters"]) == {"Wf", "Wc", "Wo"}

    held_out = example._build_held_out_corpus(config)
    episodes = tuple(pair.supported for pair in held_out[2])
    task = example._model_task(config, 2)
    intact = example._evaluate_batch(model, task, episodes, shuffled=False)
    shuffled = example._evaluate_batch(model, task, episodes, shuffled=True)

    assert intact.workspace.shape == (2, 3, 4)
    assert shuffled.workspace.shape == (2, 3, 4)
    assert intact.memory_read.shape == (2, 4)
    assert intact.parameters_unchanged is True
    assert shuffled.parameters_unchanged is True
    assert set(np.unique(intact.workspace)).issubset({0.0, 1.0})
    assert np.linalg.norm(shuffled.memory_values) == pytest.approx(
        np.linalg.norm(intact.memory_values)
    )


def test_real_cpu_smoke_runs_every_depth_arm_and_measurement(tmp_path, capsys):
    example = _load()

    result = example.main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--seed",
            "654",
            "--figure",
            str(tmp_path / "real-smoke.png"),
        ]
    )

    assert set(result["training"]["depths"]) == {"0", "1", "2", "4", "8"}
    assert result["interventions"]["frozen_no_retraining"] is True
    assert all(
        depth["all_frozen_parameter_audits_passed"]
        for depth in result["interventions"]["depths"].values()
    )
    assert all(
        set(depth["per_binding_count"]) == {str(value) for value in range(2, 9)}
        for depth in result["interventions"]["depths"].values()
    )
    assert all(
        "raw_memory_factor_decodability" in depth
        for depth in result["geometry"]["depths"].values()
    )
    assert result["device"]["requested"] == "cpu"
    assert result["device"]["platform"] == "cpu"
    assert pathlib.Path(result["figure_path"]).is_file()
    json.dumps(result, allow_nan=False)
    output = capsys.readouterr().out
    assert "R=0" in output and "R=8" in output
    assert "K=2" in output and "K=8" in output
    assert "primary exact_query_memory_read" in output
    assert "secondary raw_memory_factors" in output


def test_real_same_seed_cpu_metrics_are_reproducible(tmp_path):
    example = _load()
    common = dict(
        seed=777,
        device="cpu",
        depths=(0,),
        binding_counts=(2, 8),
        batch_size=2,
        training_updates=1,
        latent_width=4,
        code_width=12,
        symbol_ticks=4,
        learning_rate=1e-5,
    )
    first_config = example.ExperimentConfig(
        **common, figure_path=tmp_path / "repro-first.png"
    )
    second_config = example.ExperimentConfig(
        **common, figure_path=tmp_path / "repro-second.png"
    )
    cpu, device_report = example._resolve_device("cpu")

    with example.jax.default_device(cpu):
        first = example._run_experiment(first_config, device_report)
        second = example._run_experiment(second_config, device_report)

    for key in ("training", "interventions", "geometry", "reproducibility"):
        assert first[key] == second[key]
    assert first["canonical_metadata_json"] == second["canonical_metadata_json"]
