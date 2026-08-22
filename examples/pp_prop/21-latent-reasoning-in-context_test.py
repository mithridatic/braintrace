"""Tests for the standard-ARC latent-reasoning entry point."""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import inspect
import json
import pathlib
import sys
import threading
import time
import warnings
from enum import Enum
from types import SimpleNamespace

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle import (
    assert_gradients_differ,
    bptt_param_gradients,
    chunked_online_param_gradients,
    gradient_norm,
    relative_deviation,
)
from examples.pp_prop.latent_workspace_model import (
    MAX_GRID_SIZE,
    LatentWorkspaceModel,
    ModelConfig,
    run_packed_stream,
    run_selected_packed_stream,
)
from examples.pp_prop.latent_workspace_refinement import RowRefinementLayout
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    DatasetSource,
    LoadedDataset,
    RowEventConfig,
    SourceManifest,
    encode_arc_query_episode,
    encode_query_episode,
    leave_one_demonstration_out_episodes,
    smoke_loaded_dataset,
)

EXAMPLE = pathlib.Path(__file__).with_name("21-latent-reasoning-in-context.py")
TEST_CHECKPOINTS = (0, 30, 60)
TEST_TRAINING_EFFORTS = (30, 60)
TEST_DEPTH_COUNT = 61


_PADDING_PROBE_EVENT_WIDTH = 6


class _PaddingTraceProbe(brainstate.nn.Module):
    """Expose a terminal residual around the real Example 21 reservoir."""

    def __init__(self) -> None:
        super().__init__()
        config = ModelConfig(
            input_width=_PADDING_PROBE_EVENT_WIDTH,
            neuron_count=64,
            recurrent_edges=64,
            max_latent_steps=4,
            readout_width=8,
            color_rank=1,
            input_gain=40.0,
            seed=41,
            sparse_backend="jax_raw",
        )
        self.reservoir = LatentWorkspaceModel(config)
        self.target = (
            jnp.zeros((1, config.compact_output_width), dtype=jnp.float32)
            .at[:, 0]
            .set(1.0)
        )

    def update(self, packed: jax.Array) -> jax.Array:
        event = packed[:, :_PADDING_PROBE_EVENT_WIDTH]
        advance = packed[:, _PADDING_PROBE_EVENT_WIDTH] > 0.5
        loss_gate = packed[:, _PADDING_PROBE_EVENT_WIDTH + 1 :]
        return loss_gate * (self.reservoir(event, advance) - self.target)


class _TrainingRowTraceProbe(brainstate.nn.Module):
    """Expose masked residuals around a tiny production reservoir."""

    def __init__(self, config: ModelConfig, target: jax.Array) -> None:
        super().__init__()
        self.reservoir = LatentWorkspaceModel(config)
        self.event_width = config.input_width
        self.target = jnp.asarray(target, dtype=jnp.float32)

    def update(self, packed: jax.Array) -> jax.Array:
        event = packed[:, : self.event_width]
        advance = packed[:, self.event_width] > 0.5
        loss_scale = packed[:, self.event_width + 1 :]
        return loss_scale * (self.reservoir(event, advance) - self.target)


def _padding_trace_probe_inputs() -> tuple[jax.Array, jax.Array]:
    padded = jnp.asarray(
        [
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    return padded, padded[jnp.asarray([0, 4, 5])]


def _padding_probe_pp_prop(model: brainstate.nn.Module):
    trace_config = braintrace.ETraceConfig(
        trace_factorization="io_factorized",
        recurrence_scope="diagonal",
        decay=0.9,
    )
    return braintrace.pp_prop(
        model,
        decay_or_rank=0.9,
        vjp_method="multi-step",
        config=trace_config,
    )


def _load_example():
    name = "_pp_prop_arc_latent_reasoning_entry"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


def _encoded_fixture(example, query_index: int = 0):
    data = smoke_loaded_dataset()
    rows = example._row_config(example.ExperimentConfig.smoke_config())
    return encode_query_episode(data.tasks[0], query_index, rows), rows


def _tiny_compaction_fixture(example):
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),))),
            ArcPair(ArcGrid(((3,),)), ArcGrid(((4,),))),
        ),
        test=(ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),))),),
        task_id="compact-trace",
    )
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(seed=41, decoder_mode="legacy_cp"),
        max_demonstrations=2,
    )
    rows = RowEventConfig(max_demonstrations=2, max_grid_size=4)
    encoded = encode_query_episode(task, 0, rows)
    return task, config, rows, encoded


def _tiny_trace_model_config(rows: RowEventConfig) -> ModelConfig:
    return ModelConfig(
        input_width=rows.input_width,
        neuron_count=64,
        recurrent_edges=64,
        max_latent_steps=32,
        readout_width=8,
        color_rank=1,
        input_gain=40.0,
        seed=41,
        sparse_backend="jax_raw",
    )


_TRAINING_ARRAY_FIELDS = (
    "events",
    "advances",
    "heights",
    "widths",
    "colors",
    "masks",
    "efforts",
)
_TRAINING_METADATA_FIELDS = (
    "task_fingerprints",
    "base_task_fingerprints",
    "source_names",
    "held_out_demonstration_indices",
)


def _materialize_training(example, data, config, rows):
    """Assemble streamed chunks for tests that inspect a full schedule."""
    chunks = list(example._training_chunks(data, config, rows))
    assert chunks
    if len(chunks) == 1:
        return chunks[0]
    arrays = {
        name: np.concatenate([getattr(chunk, name) for chunk in chunks])
        for name in _TRAINING_ARRAY_FIELDS
    }
    metadata = {
        name: tuple(value for chunk in chunks for value in getattr(chunk, name))
        for name in _TRAINING_METADATA_FIELDS
    }
    return example._TrainingTensors(**arrays, **metadata)


def _metric(value: float = 0.0) -> dict[str, float | int]:
    return {
        "query_count": 1,
        "task_count": 1,
        "query_pass_at_1": value,
        "query_pass_at_2": value,
        "strict_task_pass_at_1": value,
        "strict_task_pass_at_2": value,
        "shape_accuracy_diagnostic": value,
        "valid_cell_pixel_accuracy_diagnostic": value,
    }


def _checkpoint_candidate_payloads(effort: int) -> list[dict[str, object]]:
    if effort == 0:
        roles = (
            "diagnostic_checkpoint_joint_argmax",
            "diagnostic_checkpoint_logit_runner_up",
        )
    else:
        roles = (
            "latest_sweep_joint_argmax",
            "latest_sweep_logit_runner_up",
        )
    return [
        {
            "rank": rank,
            "height": 1,
            "width": 1,
            "grid": [[rank - 1]],
            "changed_decision": None if rank == 1 else "cell:0,0",
            "log_probability": -float(rank),
            "provenance": "model",
            "source_checkpoint": effort,
            "selection_role": roles[rank - 1],
        }
        for rank in (1, 2)
    ]


def _numeric_evidence(steps: list[int]) -> dict[str, object]:
    names = (
        "compact_logits",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    )
    return {
        "evaluated_steps": steps,
        "query_count": 1,
        "state_byte_identical": True,
        "compact_logits_byte_identical": True,
        "within_declared_tolerance": True,
        "within_tolerance_by_query": [True],
        "within_tolerance_query_count": 1,
        "declared_per_query_axis_rms_tolerance": 1e-6,
        "spike_hamming_count": 0,
        "spike_hamming_count_by_query": [0],
        "per_step_query_rms": {name: [[0.0] for _ in steps] for name in names},
        "per_query_maximum_rms": {name: [0.0] for name in names},
        "maximum_rms": {name: 0.0 for name in names},
        "intact_dtype_by_state": {name: "float32" for name in names},
        "candidate_dtype_by_state": {name: "float32" for name in names},
        "required_float32_dtypes": True,
    }


def _evaluation_payload() -> dict[str, object]:
    metrics = {str(effort): _metric(float(effort == 60)) for effort in TEST_CHECKPOINTS}
    comparison = {
        "causally_null_at_measured_precision": False,
        "state_byte_identical_by_step": [False] * TEST_DEPTH_COUNT,
        "spike_hamming_by_step": [1] * TEST_DEPTH_COUNT,
        "spike_hamming_fraction_by_step": [0.125] * TEST_DEPTH_COUNT,
        "voltage_l2_by_step": [1.0] * TEST_DEPTH_COUNT,
        "synaptic_current_l2_by_step": {
            "feedforward": [1.0] * TEST_DEPTH_COUNT,
            "recurrent": [1.0] * TEST_DEPTH_COUNT,
        },
        "score_deltas_control_minus_intact": {
            "60.query_pass_at_2": 0.0,
            "60.valid_cell_pixel_accuracy_diagnostic": 0.0,
        },
    }
    control = {
        "metrics_by_effort": metrics,
        "trajectory_comparison": comparison,
        "causally_null_query_count": 0,
        "query_count": 1,
        "applicable_query_count": 1,
        "available_query_count": 1,
        "unavailable_query_count": 0,
        "timing_matched_applicable_query_count": 1,
    }
    trajectory = [
        {
            "step": step,
            "mean_firing_rate": step / 64.0,
            "mean_spike_count": float(step),
            "mean_voltage_l2": float(step),
            "mean_feedforward_current_l2": float(step),
            "mean_recurrent_current_l2": float(step),
            "mean_predictive_entropy": 1.0,
            "mean_changed_cell_fraction": None if step == 0 else 0.0,
            "converged_fraction": 0.0,
            "near_silence_fraction": 0.0,
            "near_saturation_fraction": 0.0,
            "unique_state_hashes": 1,
            "pair_sample_count": 0,
            "pairwise_spike_hamming_fraction": None,
            "pairwise_voltage_rms_distance": None,
            "pairwise_feedforward_current_rms_distance": None,
            "pairwise_recurrent_current_rms_distance": None,
        }
        for step in range(TEST_DEPTH_COUNT)
    ]
    query_steps = [
        {
            "step": step,
            "candidates": [{"grid": [[0]]}],
            "changed_cell_count": None if step == 0 else 0,
            "changed_cell_fraction": None if step == 0 else 0.0,
            "predictive_entropy": 1.0,
            "top_two_logit_margin": 0.0,
            "spike_count": 0,
            "firing_rate": 0.0,
            "raster_active_indices": [],
            "voltage_mean": 0.0,
            "voltage_std": 0.0,
            "voltage_mean_absolute": 0.0,
            "voltage_l2": 0.0,
            "spike_hamming_displacement": None if step == 0 else 0,
            "spike_hamming_fraction": None if step == 0 else 0.0,
            "voltage_l2_displacement": None if step == 0 else 0.0,
            "feedforward_current_mean_absolute": 0.0,
            "feedforward_current_l2": 0.0,
            "feedforward_current_l2_displacement": None if step == 0 else 0.0,
            "recurrent_current_mean_absolute": 0.0,
            "recurrent_current_l2": 0.0,
            "recurrent_current_l2_displacement": None if step == 0 else 0.0,
            "converged": False,
            "near_silence": True,
            "near_saturation": False,
            "state_sha256": "0" * 64,
            "score": {},
        }
        for step in range(TEST_DEPTH_COUNT)
    ]
    checkpoint_queries = {
        str(effort): [
            {
                "task_id": "task",
                "query_index": 0,
                "primary_candidate_mode": "model_only",
                "submission_role": (
                    "primary_submission" if effort == 60 else "diagnostic_only"
                ),
                "candidates": _checkpoint_candidate_payloads(effort),
                "score": {},
            }
        ]
        for effort in TEST_CHECKPOINTS
    }
    control["decoded_candidates_match_intact"] = True
    control["decoded_candidates_match_intact_by_effort"] = {
        str(effort): True for effort in TEST_CHECKPOINTS
    }
    control["decoded_candidate_match_query_count_by_effort"] = {
        str(effort): 1 for effort in TEST_CHECKPOINTS
    }
    control["byte_identical_query_count"] = 0
    return {
        "primary_candidate_mode": "model_only",
        "query_count": 1,
        "task_count": 1,
        "same_frozen_parameter_bytes": True,
        "submission_policy": {
            "name": "latest_sweep_plus_newest_distinct_earlier",
            "submission_checkpoint": 60,
            "completed_sweep_checkpoints": [30, 60],
            "candidate_budget": 2,
            "fallback": "latest_sweep_deterministic_logit_runner_up",
            "target_free_selection": True,
            "rule_channel_enabled": False,
        },
        "model_only_completion": {
            "primary_candidate_mode": "model_only",
            "eligible_for_completion": False,
            "eligibility_reason": "completion requires a complete 400-task run",
            "submission_checkpoint": 60,
            "submission_policy": "latest_sweep_plus_newest_distinct_earlier",
            "required_task_count": 400,
            "evaluated_task_count": 1,
            "evaluated_query_count": 1,
            "required_exact_task_count": 160,
            "exact_task_count": 1,
            "strict_task_pass_at_2": 1.0,
            "passed": False,
            "tasks": {"task": {"query_count": 1, "pass_at_2": True}},
        },
        "checkpoint_queries": checkpoint_queries,
        "query_trajectories": [
            {
                "step_count": TEST_DEPTH_COUNT,
                "neuron_count": 4096,
                "steps": query_steps,
            }
        ],
        "metrics_by_effort": metrics,
        "task_local_adaptation": {
            "performed": False,
            "mode": "disabled",
            "reason": "decoder_mode_is_not_row_refinement",
            "target_free_query_bank": True,
        },
        "frozen_no_adaptation": {
            "role": "diagnostic_control_not_primary_submission",
            "metrics_by_effort": copy.deepcopy(metrics),
            "checkpoint_queries": copy.deepcopy(checkpoint_queries),
        },
        "aggregate_trajectory": trajectory,
        "determinism": {
            "same_control_capable_execution_path": True,
            "state_rms_tolerance": 1e-6,
            "spike_tolerance": "exact identity",
            "metric_absolute_tolerance": 0.0,
            "repeat_intact_state_byte_identical": True,
            "repeat_intact_compact_logits_byte_identical": True,
            "repeat_intact_within_tolerance": True,
            "repeat_intact_metrics_exact": True,
            "repeat_intact_decoded_candidates_exact": True,
            "repeat_intact_numeric_evidence": _numeric_evidence(
                list(range(TEST_DEPTH_COUNT))
            ),
            "slot_ablation_checkpoint_zero_byte_identical": True,
            "slot_ablation_checkpoint_zero_state_within_tolerance": True,
            "slot_ablation_checkpoint_zero_decoded_candidates_exact": True,
            "slot_ablation_checkpoint_zero_metrics_exact": True,
            "slot_ablation_checkpoint_zero_within_tolerance": True,
            "slot_ablation_checkpoint_zero_numeric_evidence": _numeric_evidence([0]),
        },
        "controls": {
            "repeat_intact": dict(
                copy.deepcopy(control),
                trajectory_comparison={
                    **comparison,
                    "causally_null_at_measured_precision": True,
                    "state_byte_identical_by_step": [True] * TEST_DEPTH_COUNT,
                    "spike_hamming_by_step": [0] * TEST_DEPTH_COUNT,
                    "spike_hamming_fraction_by_step": [0.0] * TEST_DEPTH_COUNT,
                    "voltage_l2_by_step": [0.0] * TEST_DEPTH_COUNT,
                    "synaptic_current_l2_by_step": {
                        "feedforward": [0.0] * TEST_DEPTH_COUNT,
                        "recurrent": [0.0] * TEST_DEPTH_COUNT,
                    },
                },
                causally_null_query_count=1,
                byte_identical_query_count=1,
            ),
            "no_context": copy.deepcopy(control),
            "shuffled_demonstrations": copy.deepcopy(control),
            "slot_ablation": copy.deepcopy(control),
            "truncation": {
                "checkpoints": list(TEST_CHECKPOINTS),
                "uses_one_continuous_intact_trajectory": True,
            },
        },
    }


def test_full_and_smoke_configs_preserve_declared_physical_scales(example, tmp_path):
    full = example.ExperimentConfig(output_dir=tmp_path, structural_only=True)
    smoke = example.ExperimentConfig.smoke_config(output_dir=tmp_path)

    assert (full.neuron_count, full.recurrent_edges) == (4096, 4_194_304)
    assert (smoke.neuron_count, smoke.recurrent_edges) == (128, 1024)
    assert full.max_demonstrations == 10
    assert full.to_dict()["checkpoints"] == [0, 30, 60]
    assert smoke.smoke is True
    assert full.context_memory_width == 32
    assert smoke.context_memory_width == 2
    assert full.decoder_mode == "latent_row_decode"
    assert smoke.decoder_mode == "latent_row_decode"
    assert full.memory_coding == smoke.memory_coding == "learned_update"
    assert full.latent_steps == smoke.latent_steps == 60
    assert full.memory_decay == 1.0
    assert full.balanced_color_loss is False
    assert full.to_dict()["balanced_color_loss"] is False


def test_smoke_config_defaults_to_protocol_v2_controls(example):
    config = example.ExperimentConfig.smoke_config()

    assert config.task_local_adaptation is False
    assert config.evaluation_controls is True
    assert config.to_dict()["primary_evaluation_mode"] == "shared_model_frozen"


def test_cli_can_select_the_single_shared_gpu_run(example):
    args = example._parser().parse_args(
        [
            "--device",
            "gpu",
            "--neurons",
            "1024",
            "--recurrent-edges",
            "1024",
            "--max-demonstrations",
            "10",
            "--latent-steps",
            "300",
            "--training-updates",
            "13",
            "--training-batch-size",
            "32",
        ]
    )
    config = example._config_from_args(args)

    assert (config.neuron_count, config.recurrent_edges) == (1024, 1024)
    assert (config.max_demonstrations, config.max_grid_size) == (10, 30)
    assert (config.latent_steps, config.training_updates) == (300, 13)
    assert config.training_batch_size == 32
    assert config.task_local_adaptation is False
    assert config.evaluation_controls is True


def test_cli_can_explicitly_disable_protocol_v2_controls(example):
    args = example._parser().parse_args(["--no-evaluation-controls"])

    config = example._config_from_args(args)

    assert config.evaluation_controls is False


def test_git_provenance_reports_declared_revision_mismatch(
    example, monkeypatch, tmp_path
):
    calls = iter(("actual-revision\n", " M changed.py\n"))

    def run(*_args, **_kwargs):
        return SimpleNamespace(stdout=next(calls))

    monkeypatch.setattr(example.subprocess, "run", run)
    monkeypatch.setenv("EXAMPLE21_SOURCE_REVISION", "declared-revision")

    report = example._git_source_provenance(tmp_path)

    assert report["source_revision"] == "actual-revision"
    assert report["source_dirty"] is True
    assert report["declared_revision_mismatch"] is True


def test_artifact_manifest_hashes_materialized_files(example, tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abc")
    second.write_bytes(b"defg")

    report = example._artifact_manifest({"second": second, "first": first})

    assert list(report["artifacts"]) == ["first", "second"]
    assert report["artifacts"]["first"]["size_bytes"] == 3
    assert report["artifacts"]["first"]["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_latent_steps_300_extends_scoring_and_training_through_300(example):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
        latent_steps=300,
        training_updates=11,
    )

    assert config.checkpoints == tuple(range(0, 301, 30))
    assert config.training_efforts == tuple(range(0, 301, 30))
    assert config.submission_checkpoint == 300
    assert config.to_dict()["checkpoints"] == list(range(0, 301, 30))
    assert config.to_dict()["training_efforts"] == list(range(0, 301, 30))
    assert config.to_dict()["submission_checkpoint"] == 300
    score_parameters = inspect.signature(example._score_windows).parameters
    assert "checkpoints" in score_parameters
    assert "submission_checkpoint" in score_parameters
    evaluate_tree = ast.parse(inspect.getsource(example._evaluate))
    score_calls = [
        node
        for node in ast.walk(evaluate_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_windows"
    ]
    assert score_calls
    assert all(len(call.args) >= 6 for call in score_calls)


def test_progress_evidence_reports_stage_chunk_elapsed_and_eta(example):
    evidence = example._progress_evidence(
        stage="training",
        completed=3,
        total=12,
        started_at=100.0,
        now=112.0,
    )

    assert evidence == {
        "stage": "training",
        "completed": 3,
        "total": 12,
        "elapsed_seconds": 12.0,
        "eta_seconds": 36.0,
    }


def test_lean_390_tick_evaluation_gathers_only_scoring_checkpoints(example):
    lean = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
        latent_steps=390,
        training_updates=14,
        evaluation_controls=False,
    )
    diagnostic = dataclasses.replace(lean, evaluation_controls=True)

    np.testing.assert_array_equal(
        example._evaluation_offsets(lean), np.arange(0, 391, 30)
    )
    np.testing.assert_array_equal(
        example._evaluation_offsets(diagnostic), np.arange(0, 391)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_memory_width": True}, "context_memory_width"),
        ({"context_memory_width": -1}, "context_memory_width"),
        ({"context_memory_width": 513}, "at most 512"),
        ({"memory_decay": True}, "memory_decay"),
        ({"memory_decay": float("nan")}, "memory_decay"),
        ({"memory_decay": -0.01}, "memory_decay"),
        ({"memory_decay": 1.01}, "memory_decay"),
    ],
)
def test_associative_memory_config_rejects_invalid_values(example, kwargs, message):
    with pytest.raises(ValueError, match=message):
        example.ExperimentConfig(structural_only=True, **kwargs)


def test_associative_memory_config_accepts_width_512(example):
    config = example.ExperimentConfig(structural_only=True, context_memory_width=512)
    assert config.context_memory_width == 512


def test_optimizer_defaults_to_muon_and_resolves_decoupled_decay(example):
    default = example.ExperimentConfig(structural_only=True)
    smoke = example.ExperimentConfig.smoke_config()
    cli = example._config_from_args(
        example._parser().parse_args(["--structural-only"])
    )
    adam = example.ExperimentConfig(structural_only=True, optimizer="adam")
    adamw = example.ExperimentConfig(structural_only=True, optimizer="adamw")
    muon = example.ExperimentConfig(structural_only=True, optimizer="muon")

    assert (default.optimizer, default.weight_decay) == ("muon", 0.1)
    assert (smoke.optimizer, smoke.weight_decay) == ("muon", 0.1)
    assert (cli.optimizer, cli.weight_decay) == ("muon", 0.1)
    assert (adam.optimizer, adam.weight_decay) == ("adam", 0.0)
    assert (adamw.optimizer, adamw.weight_decay) == ("adamw", 0.01)
    assert (muon.optimizer, muon.weight_decay) == ("muon", 0.1)
    assert default.to_dict()["optimizer"] == "muon"
    assert muon.to_dict()["weight_decay"] == 0.1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"optimizer": "sgd"}, "optimizer"),
        ({"weight_decay": True}, "weight_decay"),
        ({"weight_decay": -1e-3}, "weight_decay"),
        ({"weight_decay": float("inf")}, "weight_decay"),
    ],
)
def test_optimizer_config_rejects_invalid_values(example, kwargs, message):
    with pytest.raises(ValueError, match=message):
        example.ExperimentConfig(structural_only=True, **kwargs)


def test_optimizer_cli_round_trips_resolved_policy(example):
    args = example._parser().parse_args(
        ["--structural-only", "--optimizer", "muon", "--weight-decay", "0.02"]
    )
    config = example._config_from_args(args)

    assert config.optimizer == "muon"
    assert config.weight_decay == 0.02


def test_adam_rejects_explicit_nonzero_weight_decay(example):
    with pytest.raises(ValueError, match="weight_decay"):
        example.ExperimentConfig(
            structural_only=True, optimizer="adam", weight_decay=0.01
        )
    with pytest.raises(ValueError, match="weight_decay"):
        example._config_from_args(
            example._parser().parse_args(
                ["--structural-only", "--optimizer", "adam", "--weight-decay", "0.01"]
            )
        )
    explicit_zero = example.ExperimentConfig(
        structural_only=True, optimizer="adam", weight_decay=0.0
    )
    assert explicit_zero.weight_decay == 0.0


@pytest.mark.parametrize("name", ["adam", "adamw", "muon"])
def test_zero_gradient_update_applies_decoupled_weight_decay(example, name):
    config = example.ExperimentConfig.smoke_config(optimizer=name)
    states = {
        ("matrix",): brainstate.ParamState(jnp.asarray([[1.0, -0.5], [0.2, 0.7]])),
        ("vector",): brainstate.ParamState(jnp.asarray([0.4, -0.3])),
    }
    optimizer = example._make_training_optimizer(config, states)
    before = {path: np.asarray(state.value).copy() for path, state in states.items()}
    gradients = {path: jnp.zeros_like(state.value) for path, state in states.items()}

    @brainstate.transform.jit
    def update():
        optimizer.update(gradients)

    update()

    shrink = 1.0 - config.learning_rate * config.weight_decay
    assert (shrink == 1.0) == (name == "adam")
    for path, state in states.items():
        np.testing.assert_allclose(
            np.asarray(state.value), before[path] * shrink, rtol=1e-6
        )


def test_experiment_defaults_pin_training_regime(example):
    config = example.ExperimentConfig(structural_only=True)
    parsed = example._parser().parse_args([])
    smoke = example.ExperimentConfig.smoke_config()

    assert config.seed == parsed.seed == 9999
    assert config.recurrent_edges == parsed.recurrent_edges == 4_194_304
    assert config.training_updates == parsed.training_updates == 260
    assert config.training_batch_size == parsed.training_batch_size == 32
    assert config.training_workers == parsed.training_workers == 8
    assert config.adaptation_epochs == parsed.adaptation_epochs == 1
    assert smoke.training_batch_size == 1


def test_lr_schedule_defaults_to_cosine_at_raised_base_rate(example):
    default = example.ExperimentConfig(structural_only=True)
    smoke = example.ExperimentConfig.smoke_config()
    cli = example._config_from_args(
        example._parser().parse_args(["--structural-only"])
    )

    assert (default.lr_schedule, default.learning_rate) == ("cosine", 1e-3)
    assert default.lr_warmup_fraction == 0.0
    assert (cli.lr_schedule, cli.learning_rate) == ("cosine", 1e-3)
    assert cli.lr_warmup_fraction == 0.0
    assert smoke.lr_schedule == "cosine"
    constant = example.ExperimentConfig(structural_only=True, lr_schedule="constant")
    assert constant.lr_schedule == "constant"
    with pytest.raises(ValueError, match="lr_schedule"):
        example.ExperimentConfig(structural_only=True, lr_schedule="linear")


def test_cosine_training_schedule_decays_base_rate_to_zero(example):
    config = example.ExperimentConfig(structural_only=True)
    schedule = example._training_learning_rate(config)
    horizon = config.training_updates

    assert float(schedule(0)) == pytest.approx(config.learning_rate)
    assert float(schedule(horizon // 2)) == pytest.approx(
        config.learning_rate / 2.0, rel=1e-6
    )
    assert float(schedule(horizon)) == pytest.approx(0.0, abs=1e-9)
    constant = example.ExperimentConfig(structural_only=True, lr_schedule="constant")
    assert example._training_learning_rate(constant) == constant.learning_rate


def test_cosine_training_schedule_supports_linear_warmup(example):
    config = example.ExperimentConfig(
        structural_only=True,
        training_updates=100,
        lr_warmup_fraction=0.01,
    )
    schedule = example._training_learning_rate(config)

    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(1)) == pytest.approx(config.learning_rate)
    assert float(schedule(config.training_updates)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "value",
    [True, -0.01, 1.0, float("nan"), float("inf")],
)
def test_lr_warmup_fraction_rejects_invalid_values(example, value):
    with pytest.raises(ValueError, match="lr_warmup_fraction"):
        example.ExperimentConfig(
            structural_only=True,
            lr_warmup_fraction=value,
        )


def test_constant_schedule_rejects_nonzero_warmup(example):
    with pytest.raises(ValueError, match="lr_warmup_fraction"):
        example.ExperimentConfig(
            structural_only=True,
            lr_schedule="constant",
            lr_warmup_fraction=0.01,
        )


def test_lr_schedule_round_trips_policy_dict_and_cli(example):
    cosine = example.ExperimentConfig(structural_only=True)
    warm = example._config_from_args(
        example._parser().parse_args(
            ["--structural-only", "--lr-warmup-fraction", "0.01"]
        )
    )
    constant = example.ExperimentConfig(structural_only=True, lr_schedule="constant")

    assert example._optimizer_policy(cosine)["lr_schedule"] == "cosine"
    assert example._optimizer_policy(cosine)["lr_warmup_fraction"] == 0.0
    assert example._optimizer_policy(warm)["lr_warmup_fraction"] == 0.01
    assert example._optimizer_policy(constant)["lr_schedule"] == "constant"
    assert cosine.to_dict()["lr_schedule"] == "cosine"
    assert warm.to_dict()["lr_warmup_fraction"] == 0.01
    assert constant.to_dict()["lr_schedule"] == "constant"
    assert constant.learning_rate == 1e-3


def test_parameter_travel_budget_halves_under_cosine_schedule(example):
    constant = example.ExperimentConfig(
        structural_only=True, optimizer="adam", lr_schedule="constant"
    )
    cosine = example.ExperimentConfig(structural_only=True, optimizer="adam")

    flat = example._parameter_travel_budget(constant)
    halved = example._parameter_travel_budget(cosine)

    assert flat["schedule_integral_factor"] == 1.0
    assert halved["schedule_integral_factor"] == 0.5
    assert halved["displacement_bound"] == pytest.approx(
        0.5 * flat["displacement_bound"]
    )


def test_softcap_betas_default_and_plumb_into_model_config(example):
    config = example.ExperimentConfig(structural_only=True)
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    model_config = example._model_config(config, rows, batch_size=1)
    cli = example._config_from_args(
        example._parser().parse_args(
            [
                "--structural-only",
                "--memory-value-softcap-beta",
                "1.0",
                "--reasoning-query-softcap-beta",
                "2.5",
            ]
        )
    )

    assert config.memory_value_softcap_beta == 4.0
    assert config.reasoning_query_softcap_beta == 25.0
    assert model_config.memory_value_softcap_beta == 4.0
    assert model_config.reasoning_query_softcap_beta == 25.0
    assert cli.memory_value_softcap_beta == 1.0
    assert cli.reasoning_query_softcap_beta == 2.5
    assert config.to_dict()["memory_value_softcap_beta"] == 4.0
    with pytest.raises(ValueError, match="memory_value_softcap_beta"):
        example.ExperimentConfig(structural_only=True, memory_value_softcap_beta=0.0)
    with pytest.raises(ValueError, match="reasoning_query_softcap_beta"):
        example.ExperimentConfig(
            structural_only=True, reasoning_query_softcap_beta=-1.0
        )


def test_memory_read_transform_round_trips_cli_model_and_reports(example):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    default = example.ExperimentConfig(structural_only=True)
    gated = example._config_from_args(
        example._parser().parse_args(
            ["--structural-only", "--memory-read-transform", "gated_rms"]
        )
    )
    model_config = example._model_config(gated, rows, batch_size=1)
    architecture = example._memory_architecture_report(
        gated,
        rows,
        training_batch_size=1,
        evaluation_batch_size=2,
    )

    assert default.memory_read_transform == "linear"
    assert gated.memory_read_transform == "gated_rms"
    assert gated.to_dict()["memory_read_transform"] == "gated_rms"
    assert model_config.memory_read_transform == "gated_rms"
    assert architecture["memory_read_transform"] == "gated_rms"
    with pytest.raises(ValueError, match="memory_read_transform"):
        example.ExperimentConfig(
            structural_only=True, memory_read_transform="normalized"
        )
    with pytest.raises(TypeError, match="memory_read_transform"):
        example.ExperimentConfig(
            structural_only=True, memory_read_transform=3
        )
    with pytest.raises(ValueError, match="positive context_memory_width"):
        example.ExperimentConfig(
            structural_only=True,
            context_memory_width=0,
            memory_coding="frozen",
            decoder_mode="legacy_cp",
            memory_read_transform="gated",
        )


def test_memory_read_interval_round_trips_cli_model_and_reports(example):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    default = example.ExperimentConfig(structural_only=True)
    interval = example._config_from_args(
        example._parser().parse_args(
            ["--structural-only", "--memory-read-interval", "4"]
        )
    )
    smoke = example.ExperimentConfig.smoke_config(memory_read_interval=8)
    model_config = example._model_config(interval, rows, batch_size=1)
    architecture = example._memory_architecture_report(
        interval,
        rows,
        training_batch_size=1,
        evaluation_batch_size=2,
    )

    assert default.memory_read_interval == 1
    assert interval.memory_read_interval == 4
    assert smoke.memory_read_interval == 8
    assert interval.to_dict()["memory_read_interval"] == 4
    assert model_config.memory_read_interval == 4
    assert architecture["memory_read_interval"] == 4
    for value in (True, 1.5, "4"):
        with pytest.raises(ValueError, match="memory_read_interval"):
            example.ExperimentConfig(
                structural_only=True, memory_read_interval=value
            )
    with pytest.raises(ValueError, match="memory_read_interval"):
        example.ExperimentConfig(structural_only=True, memory_read_interval=0)


@pytest.mark.parametrize("name", ["adam", "adamw", "muon"])
def test_training_optimizer_compiled_update_moves_matrix_and_vector_leaves(
    example, name
):
    config = example.ExperimentConfig.smoke_config(
        optimizer=name,
        weight_decay=0.01 if name != "adam" else 0.0,
    )
    states = {
        ("matrix",): brainstate.ParamState(jnp.asarray([[1.0, -0.5], [0.2, 0.7]])),
        ("vector",): brainstate.ParamState(jnp.asarray([0.4, -0.3])),
    }
    optimizer = example._make_training_optimizer(config, states)
    before = {path: np.asarray(state.value).copy() for path, state in states.items()}
    gradients = {
        ("matrix",): jnp.asarray([[0.3, -0.2], [0.1, 0.4]]),
        ("vector",): jnp.asarray([0.2, -0.1]),
    }

    @brainstate.transform.jit
    def update():
        optimizer.update(gradients)

    update()

    for path, state in states.items():
        after = np.asarray(state.value)
        assert np.all(np.isfinite(after))
        assert not np.array_equal(after, before[path])
    assert int(optimizer.step_count.value) == 1
    policy = example._optimizer_policy(config)
    if name == "muon":
        assert policy["matrix_optimizer"] == "muon"
        assert policy["nonmatrix_optimizer"] == "adamw"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"neuron_count": 127}, "divisible by 64"),
        ({"recurrent_edges": 4096 * 4096}, "no-self capacity"),
        ({"max_grid_size": 29}, "standard ARC"),
        ({"latent_steps": 59}, "latent_steps"),
        ({"ablation_slot": 64}, "ablation_slot"),
        ({"training_updates": 1}, "effort zero"),
        ({"decoder_mode": "width_inferred"}, "decoder_mode"),
        ({"device": "tpu"}, "device"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"sparse_backend": "invalid"}, "sparse_backend"),
    ],
)
def test_config_rejects_invalid_or_scientifically_incomplete_values(
    example, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        example.ExperimentConfig(**kwargs)


def test_config_fails_closed_above_1024_recurrent_edges_per_neuron(example):
    with pytest.raises(RuntimeError, match="recurrent edge budget"):
        example.ExperimentConfig(
            structural_only=True,
            neuron_count=2048,
            recurrent_edges=2_097_153,
        )


def test_structural_config_allows_zero_updates(example):
    config = example.ExperimentConfig(structural_only=True, training_updates=0)
    assert config.training_updates == 0


def test_row_refinement_requires_continuous_workspace_memory(example):
    with pytest.raises(ValueError, match="positive context_memory_width"):
        example.ExperimentConfig(
            structural_only=True,
            context_memory_width=0,
            decoder_mode="row_refinement",
        )

    legacy = example.ExperimentConfig(
        structural_only=True,
        context_memory_width=0,
        memory_coding="frozen",
        decoder_mode="legacy_cp",
    )
    assert legacy.context_memory_width == 0


def test_cli_defaults_fail_closed_to_full_gpu_and_smoke_owns_scale(example, tmp_path):
    parsed = example._parser().parse_args([])
    assert parsed.device == "gpu"
    assert parsed.neurons == 4096
    assert parsed.recurrent_edges == 4_194_304
    assert parsed.context_memory_width == 32
    assert parsed.decoder_mode == "latent_row_decode"
    assert parsed.memory_decay == 1.0
    assert parsed.balanced_color_loss is False
    assert parsed.profile is False
    assert parsed.sparse_backend == "default"

    smoke_args = example._parser().parse_args(
        [
            "--smoke",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--decoder-mode",
            "legacy_cp",
            "--balanced-color-loss",
            "--profile",
            "--sparse-backend",
            "jax_raw",
        ]
    )
    smoke = example._config_from_args(smoke_args)
    assert smoke.smoke and smoke.device == "cpu"
    assert smoke.decoder_mode == "legacy_cp"
    assert smoke.balanced_color_loss is True
    assert smoke.runtime_profile is True
    assert smoke.sparse_backend == "jax_raw"

    bad = example._parser().parse_args(["--smoke", "--neurons", "128"])
    with pytest.raises(ValueError, match="owns its reduced"):
        example._config_from_args(bad)

    full_args = example._parser().parse_args(
        ["--structural-only", "--device", "cpu", "--training-updates", "0"]
    )
    full = example._config_from_args(full_args)
    assert full.structural_only and full.training_updates == 0

    memory_args = example._parser().parse_args(
        [
            "--structural-only",
            "--training-updates",
            "0",
            "--context-memory-width",
            "32",
            "--memory-decay",
            "0.95",
            "--decoder-mode",
            "legacy_cp",
            "--balanced-color-loss",
        ]
    )
    memory = example._config_from_args(memory_args)
    assert memory.context_memory_width == 32
    assert memory.memory_decay == 0.95
    assert memory.balanced_color_loss is True


def test_device_resolution_reports_backend_and_fails_closed(example, monkeypatch):
    device = SimpleNamespace(platform="gpu", id=2, device_kind="test accelerator")
    monkeypatch.setattr(example, "_devices_for", lambda platform: [device])
    selected, report = example._resolve_device("gpu")
    assert selected is device
    assert report == {
        "requested": "gpu",
        "platform": "gpu",
        "id": 2,
        "kind": "test accelerator",
        "memory_stats": {},
    }

    monkeypatch.setattr(example, "_devices_for", lambda platform: [])
    with pytest.raises(RuntimeError, match="backend is unavailable"):
        example._resolve_device("gpu")

    def unavailable(platform):
        raise RuntimeError("driver missing")

    monkeypatch.setattr(example, "_devices_for", unavailable)
    with pytest.raises(RuntimeError, match="driver missing"):
        example._resolve_device("gpu")


def test_nvidia_smi_sampler_and_monitor_record_current_process_peak(
    example, monkeypatch
):
    calls: list[list[str]] = []
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout="2048, 24576\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="41, 1024\n7, 512\n", stderr=""),
        )
    )

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {
            "capture_output": True,
            "check": False,
            "text": True,
            "timeout": 5.0,
        }
        return next(responses)

    monkeypatch.setattr(example.subprocess, "run", run)
    sample = example._sample_nvidia_smi(device_index=2, process_id=41)

    assert sample == {
        "physical_device_bytes": 24_576 * 1024 * 1024,
        "current_device_bytes": 2048 * 1024 * 1024,
        "current_process_bytes": 1024 * 1024 * 1024,
        "error": None,
    }
    assert all("--id=2" in command for command in calls)

    monitor = example._NvidiaSmiGpuMonitor(device_index=2, process_id=41)
    monitor._record(sample)
    monitor._record(
        {
            "physical_device_bytes": 24_576 * 1024 * 1024,
            "current_device_bytes": 3072 * 1024 * 1024,
            "current_process_bytes": 1536 * 1024 * 1024,
            "error": None,
        }
    )
    report = monitor.report()
    assert report["sample_count"] == 2
    assert report["physical_device_bytes"] == 24_576 * 1024 * 1024
    assert report["peak_device_bytes"] == 3072 * 1024 * 1024
    assert report["peak_process_bytes"] == 1536 * 1024 * 1024
    assert report["errors"] == []


def test_nvidia_smi_wddm_na_process_uses_conservative_device_wide_peak(
    example, monkeypatch
):
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout="4096, 24576\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="41, [N/A]\n", stderr=""),
        )
    )
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    sample = example._sample_nvidia_smi(device_index=0, process_id=41)
    assert sample["current_process_bytes"] is None
    assert sample["current_device_bytes"] == 4096 * 1024 * 1024
    assert sample["error"] is None

    monitor = example._NvidiaSmiGpuMonitor(device_index=0, process_id=41)
    monitor._record(sample)
    report = monitor.report()
    assert report["peak_process_bytes"] is None
    assert report["peak_device_bytes"] == 4096 * 1024 * 1024
    assert report["evidence_complete"] is True


def test_gpu_monitor_retains_later_valid_bound_after_transient_error(example) -> None:
    monitor = example._NvidiaSmiGpuMonitor(device_index=0, process_id=41)
    monitor._record(
        {
            "physical_device_bytes": None,
            "current_device_bytes": None,
            "current_process_bytes": None,
            "error": "temporary nvidia-smi timeout",
        }
    )
    monitor._record(
        {
            "physical_device_bytes": 100,
            "current_device_bytes": 30,
            "current_process_bytes": 20,
            "error": None,
        }
    )

    report = monitor.report()
    assert report["peak_device_bytes"] == 30
    assert report["evidence_complete"] is True
    assert report["errors"] == ["temporary nvidia-smi timeout"]
    safety = example._gpu_runtime_safety_report(
        example.ExperimentConfig(structural_only=True),
        {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80"},
        {"peak_bytes_in_use": 20, "bytes_limit": 80},
        report,
    )
    assert safety["full_qualification_safe"] is True


@pytest.mark.parametrize(
    ("config", "memory_stats", "monitor", "status", "full_safe"),
    [
        (
            None,
            {"peak_bytes_in_use": 20, "bytes_limit": 80},
            {
                "physical_device_bytes": 100,
                "peak_device_bytes": 30,
                "peak_process_bytes": 20,
            },
            "safe",
            True,
        ),
        (
            None,
            {"peak_bytes_in_use": 20},
            {
                "physical_device_bytes": 100,
                "peak_device_bytes": 30,
                "peak_process_bytes": 20,
            },
            "insufficient_evidence",
            False,
        ),
        (
            None,
            {"peak_bytes_in_use": 20, "bytes_limit": 80},
            {
                "physical_device_bytes": 100,
                "peak_device_bytes": 86,
                "peak_process_bytes": 20,
            },
            "unsafe",
            False,
        ),
        (
            "smoke",
            {"peak_bytes_in_use": 20, "bytes_limit": 80},
            {
                "physical_device_bytes": 100,
                "peak_device_bytes": 30,
                "peak_process_bytes": None,
            },
            "smoke_within_limits",
            False,
        ),
        (
            None,
            {"peak_bytes_in_use": 20, "bytes_limit": 80},
            {"physical_device_bytes": 100, "peak_process_bytes": 20},
            "insufficient_evidence",
            False,
        ),
    ],
)
def test_gpu_runtime_report_normalizes_complete_missing_and_over_limit_evidence(
    example, config, memory_stats, monitor, status, full_safe
):
    selected = (
        example.ExperimentConfig.smoke_config(device="gpu")
        if config == "smoke"
        else example.ExperimentConfig(structural_only=True)
    )
    report = example._gpu_runtime_safety_report(
        selected,
        {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80"},
        memory_stats,
        monitor,
    )

    assert report["applicable"] is True
    assert report["status"] == status
    assert report["full_qualification_safe"] is full_safe


def test_gpu_environment_gate_precedes_device_resolution_and_monitoring(
    example, monkeypatch
):
    config = example.ExperimentConfig.smoke_config(device="gpu")
    order: list[str] = []

    class StopAfterOrdering(RuntimeError):
        pass

    class Monitor:
        def start(self):
            order.append("monitor-start")

        def stop(self):
            order.append("monitor-stop")
            return {
                "physical_device_bytes": None,
                "peak_process_bytes": None,
            }

    def require(environment):
        assert environment is example.os.environ
        order.append("environment-gate")
        return SimpleNamespace(to_dict=lambda: {"safe": True})

    def resolve(name):
        order.append("resolve-device")
        return SimpleNamespace(id=0), {"platform": "gpu", "id": 0}

    def load(_config):
        order.append("load-data")
        raise StopAfterOrdering

    monkeypatch.setattr(
        example, "require_pre_device_gpu_environment", require, raising=False
    )
    monkeypatch.setattr(example, "_resolve_device", resolve)
    monkeypatch.setattr(
        example, "_make_gpu_monitor", lambda device: Monitor(), raising=False
    )
    monkeypatch.setattr(example, "_load_data", load)

    with pytest.raises(StopAfterOrdering):
        example.run_experiment(config)

    assert order == [
        "environment-gate",
        "resolve-device",
        "monitor-start",
        "load-data",
        "monitor-stop",
    ]


def test_source_manifest_resolves_paths_and_exclusions(example, tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "ARC training",
                        "role": "train",
                        "version": "v1",
                        "path": "arc/training",
                        "license_reference": "https://example.test/license",
                        "format": "task_json",
                        "exclude_fingerprints": ["a" * 64],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    (source,) = example._source_declarations(manifest)

    assert source.path == str(tmp_path / "arc/training")
    assert source.exclude_fingerprints == ("a" * 64,)


@pytest.mark.parametrize(
    "payload",
    [{}, {"sources": []}, {"sources": [1]}, {"sources": [{"name": "missing"}]}],
)
def test_source_manifest_rejects_incomplete_declarations(example, tmp_path, payload):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        example._source_declarations(manifest)


def test_smoke_data_is_explicitly_plumbing_only(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)

    assert data.plumbing_only is True
    assert data.training == data.evaluation
    assert all(origin.role == "fixture" for origin in data.training)
    assert data.loaded[0].manifest.plumbing_only is True


def test_full_data_uses_baked_manifest_environment_default(
    example, monkeypatch, tmp_path
):
    manifest = tmp_path / "baked-sources.json"
    manifest.write_text("{}", encoding="utf-8")

    class StopAfterManifest(Exception):
        pass

    def capture(path):
        assert path == manifest
        raise StopAfterManifest

    monkeypatch.setenv("EXAMPLE21_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setattr(example, "_source_declarations", capture)

    with pytest.raises(StopAfterManifest):
        example._load_data(example.ExperimentConfig(device="cpu"))


def test_full_data_requires_manifest_and_both_roles(example, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="source-manifest"):
        example._load_data(example.ExperimentConfig(device="cpu"))

    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "eval",
                        "role": "evaluation",
                        "version": "1",
                        "path": ".",
                        "license_reference": "license",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fixture = smoke_loaded_dataset()
    evaluation_source = DatasetSource(
        "eval", "evaluation", "1", str(tmp_path), "license"
    )
    evaluation_manifest = SourceManifest(
        source=evaluation_source,
        resolved_path=str(tmp_path),
        files=fixture.manifest.files,
        parsed_task_count=len(fixture.tasks),
        valid_task_count=len(fixture.tasks),
        rejected=(),
        duplicate_fingerprints=(),
        task_fingerprints=fixture.manifest.task_fingerprints,
        plumbing_only=False,
    )
    monkeypatch.setattr(
        example,
        "load_dataset_source",
        lambda source: LoadedDataset(fixture.tasks, evaluation_manifest),
    )
    monkeypatch.setattr(example, "assert_no_evaluation_leakage", lambda manifests: None)

    with pytest.raises(ValueError, match="training requires"):
        example._load_data(
            example.ExperimentConfig(source_manifest=manifest, device="cpu")
        )


def test_row_layout_and_packed_latent_input_are_exactly_fixed(example):
    encoded, rows = _encoded_fixture(example)
    config = example.ExperimentConfig.smoke_config(decoder_mode="row_refinement")

    packed = example._packed_events(encoded, config)
    advances = example._packed_advances(encoded, config, rows)

    assert rows.max_events == 150
    assert packed.shape == (rows.max_events + config.latent_steps, rows.input_width)
    assert np.count_nonzero(packed[encoded.events.shape[0] :]) == 0
    assert (
        np.count_nonzero(
            packed[encoded.query_stop : encoded.query_stop + config.latent_steps]
        )
        == 0
    )
    assert advances.dtype == np.bool_
    assert advances[encoded.query_stop : encoded.query_stop + config.latent_steps].all()


def test_advance_schedule_skips_demonstration_padding_rows(example):
    """Padding inside a demo block must not burn leak steps.

    A demo block is a fixed 30 rows regardless of grid height, while the query
    block advances only over its true height.  Advancing the whole demo block
    spends ``30 - height`` all-zero steps of pure membrane leak per
    demonstration, pushing the demonstrations further from the readout than the
    data requires and making the two blocks asymmetric.
    """
    encoded, rows = _encoded_fixture(example)
    config = example.ExperimentConfig.smoke_config()
    advances = example._packed_advances(encoded, config, rows)

    valid = encoded.events[:, rows.valid_slice.start] > 0.0
    occupied = max(
        int(valid[start:stop].sum()) for start, stop in encoded.demonstration_spans
    )
    assert occupied < rows.max_grid_size, "fixture must exercise padded blocks"
    for start, stop in encoded.demonstration_spans:
        assert advances[start : start + occupied].all()
        assert not advances[start + occupied : stop].any()
    occupied_stop = encoded.demonstration_spans[-1][1]
    assert not advances[occupied_stop : encoded.query_start].any()
    assert advances[encoded.query_start : encoded.query_stop].all()
    assert not advances[encoded.query_stop + config.latent_steps :].any()


def test_advance_schedule_never_drops_a_valid_demonstration_row(example):
    """Every encoded demo row must still advance, in intact and deranged arms.

    The demo advance width is one number shared by all blocks so that the
    ``shuffled_demonstrations`` control keeps a byte-identical schedule.  It is
    the per-episode maximum, and because ``_derange_task`` rotates the outputs
    the multiset of grids is preserved, so intact and deranged agree exactly.
    """
    data = smoke_loaded_dataset()
    rows = example._row_config(example.ExperimentConfig.smoke_config())
    config = example.ExperimentConfig.smoke_config()
    task = data.tasks[0]
    deranged = example._derange_task(task)
    assert deranged is not None

    intact = encode_query_episode(task, 0, rows)
    other = encode_query_episode(deranged, 0, rows)
    intact_advances = example._packed_advances(intact, config, rows)
    assert np.array_equal(
        intact_advances, example._packed_advances(other, config, rows)
    )
    for encoded in (intact, other):
        valid = encoded.events[:, rows.valid_slice.start] > 0.0
        assert not (valid & ~intact_advances[: len(valid)]).any()


def test_padding_changes_finite_window_pp_prop_credit_not_bptt_objective():
    """Frozen rows preserve exact dynamics but still age pp-prop's trace."""
    padded, compact = _padding_trace_probe_inputs()
    factory = _PaddingTraceProbe

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        padded_bptt = bptt_param_gradients(factory, padded)
        compact_bptt = bptt_param_gradients(factory, compact)
        padded_pp_prop = chunked_online_param_gradients(
            factory,
            padded,
            algo_factory=_padding_probe_pp_prop,
            chunk_size=1,
        )
        compact_pp_prop = chunked_online_param_gradients(
            factory,
            compact,
            algo_factory=_padding_probe_pp_prop,
            chunk_size=1,
        )

    assert gradient_norm(compact_bptt) > 1.0
    assert relative_deviation(padded_bptt, compact_bptt) == pytest.approx(0.0, abs=1e-7)
    assert_gradients_differ(padded_pp_prop, compact_pp_prop, min_rel=1e-4)


def test_effort_schedule_is_balanced_reproducible_and_mixed(example):
    first = example._effort_schedule(11, brainstate.random.RandomState(7))
    second = example._effort_schedule(11, brainstate.random.RandomState(7))
    counts = {effort: int(np.sum(first == effort)) for effort in (30, 60)}

    assert np.array_equal(first, second)
    assert set(first.tolist()) == {30, 60}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_training_stream_compacts_learning_rule_timeline(example):
    """No frozen layout position may age the trace before supervised depths."""
    config = example.ExperimentConfig.smoke_config(decoder_mode="row_refinement")
    rows = example._row_config(config)
    tensors = _materialize_training(example, example._load_data(config), config, rows)

    for advances, events in zip(tensors.advances, tensors.events, strict=True):
        advancing = advances[:, 0]
        prefix_length = int(np.count_nonzero(advancing))
        assert advancing[:prefix_length].all()
        assert not advancing[prefix_length:].any()
        assert np.count_nonzero(events[prefix_length:]) == 0


def test_training_row_uses_one_held_out_demonstration_as_the_query(example):
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1,),)), ArcGrid(((4,),))),
            ArcPair(ArcGrid(((2,),)), ArcGrid(((5,),))),
            ArcPair(ArcGrid(((3,),)), ArcGrid(((6,),))),
        ),
        test=(ArcPair(ArcGrid(((9,),)), ArcGrid(((8,),))),),
        task_id="loo-row",
    )
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(seed=41, decoder_mode="row_refinement"),
        max_demonstrations=3,
    )
    rows = RowEventConfig(max_demonstrations=3, max_grid_size=4)

    produced = example._training_row(
        example._OriginTask("fixture", "fixture", task),
        config,
        rows,
        brainstate.random.RandomState(7),
        effort=8,
        plumbing_only=True,
    )

    held_out_index = produced["held_out_demonstration_index"]
    episode = leave_one_demonstration_out_episodes(task)[held_out_index]
    assert episode.query_input == task.train[held_out_index].input
    assert episode.demonstrations == (
        task.train[:held_out_index] + task.train[held_out_index + 1 :]
    )
    encoded = encode_arc_query_episode(episode, rows)
    expected_events, expected_advances, _checkpoint = example._compact_training_stream(
        encoded, config, rows
    )
    np.testing.assert_array_equal(produced["events"][:, 0], expected_events)
    np.testing.assert_array_equal(produced["advances"][:, 0], expected_advances)
    assert int(produced["heights"]) == task.train[held_out_index].output.height - 1
    assert int(produced["widths"]) == task.train[held_out_index].output.width - 1
    np.testing.assert_array_equal(
        produced["colors"][0, :1, :1],
        task.train[held_out_index].output.as_array(),
    )


def test_training_row_is_byte_identical_when_official_test_outputs_change(example):
    train = (
        ArcPair(ArcGrid(((1, 0),)), ArcGrid(((2, 0),))),
        ArcPair(ArcGrid(((3, 0),)), ArcGrid(((4, 0),))),
        ArcPair(ArcGrid(((5, 0),)), ArcGrid(((6, 0),))),
    )
    official_input = ArcGrid(((7, 0),))
    with_output = ArcTask(
        train=train,
        test=(ArcPair(official_input, ArcGrid(((8, 0),))),),
        task_id="no-official-target-leakage",
    )
    without_output = ArcTask(
        train=train,
        test=(ArcPair(official_input, None),),
        task_id="no-official-target-leakage",
    )
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(seed=43, decoder_mode="row_refinement"),
        max_demonstrations=3,
    )
    rows = RowEventConfig(max_demonstrations=3, max_grid_size=4)

    def produce(task):
        return example._training_row(
            example._OriginTask("arc-agi-1 training", "train", task),
            config,
            rows,
            brainstate.random.RandomState(97),
            effort=16,
            plumbing_only=False,
        )

    left = produce(with_output)
    right = produce(without_output)
    assert left.keys() == right.keys()
    for name in left:
        left_value = left[name]
        right_value = right[name]
        if isinstance(left_value, np.ndarray):
            assert left_value.dtype == right_value.dtype, name
            assert left_value.shape == right_value.shape, name
            assert left_value.tobytes() == right_value.tobytes(), name
        else:
            assert left_value == right_value, name


def test_primary_candidate_mode_is_fail_closed_to_model_only(example):
    config = example.ExperimentConfig.smoke_config()
    assert config.primary_candidate_mode == "model_only"
    assert config.to_dict()["primary_candidate_mode"] == "model_only"

    with pytest.raises(ValueError, match="primary_candidate_mode must be 'model_only'"):
        dataclasses.replace(config, primary_candidate_mode="verified_rules")


def test_compaction_preserves_every_semantic_checkpoint(example):
    """Compaction changes trace timing, never the ordered physical trajectory."""
    _task, config, rows, encoded = _tiny_compaction_fixture(example)
    padded = example._packed_events(encoded, config)
    padded_advances = example._packed_advances(encoded, config, rows)
    compact, compact_advances, query_checkpoint = example._compact_training_stream(
        encoded, config, rows
    )
    active_indices = np.flatnonzero(padded_advances)
    compact_indices = np.arange(active_indices.size)

    np.testing.assert_array_equal(
        padded[active_indices], compact[: active_indices.size]
    )
    assert active_indices[query_checkpoint] == encoded.query_stop - 1
    assert compact_advances[: active_indices.size].all()
    assert not compact_advances[active_indices.size :].any()
    np.testing.assert_array_equal(
        np.flatnonzero(~compact_advances),
        np.arange(active_indices.size, compact_advances.size),
    )
    np.testing.assert_array_equal(compact[active_indices.size :], 0.0)

    model_config = _tiny_trace_model_config(rows)
    padded_trajectory = run_packed_stream(
        LatentWorkspaceModel(model_config),
        jnp.asarray(padded[:, None, :]),
        advance_gates=jnp.asarray(padded_advances[:, None]),
    )
    compact_trajectory = run_packed_stream(
        LatentWorkspaceModel(model_config),
        jnp.asarray(compact[:, None, :]),
        advance_gates=jnp.asarray(compact_advances[:, None]),
    )

    for field in (
        "compact_logits",
        "spikes",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    ):
        padded_values = np.asarray(getattr(padded_trajectory, field))[active_indices]
        compact_values = np.asarray(getattr(compact_trajectory, field))[compact_indices]
        np.testing.assert_array_equal(padded_values, compact_values, err_msg=field)


def test_production_training_row_matches_explicit_compact_pp_prop_gradient(example):
    """The emitted static row has the compact finite-window pp-prop gradient."""
    task, config, rows, _official_encoded = _tiny_compaction_fixture(example)
    effort = 8
    row = example._training_row(
        example._OriginTask("fixture", "fixture", task),
        config,
        rows,
        brainstate.random.RandomState(7),
        effort=effort,
        plumbing_only=True,
    )
    encoded = encode_arc_query_episode(
        leave_one_demonstration_out_episodes(task)[row["held_out_demonstration_index"]],
        rows,
    )

    padded = example._packed_events(encoded, config)
    active_indices = np.flatnonzero(example._packed_advances(encoded, config, rows))
    query_checkpoint = int(np.flatnonzero(active_indices == encoded.query_stop - 1)[0])
    reference_mask = np.zeros((active_indices.size,), dtype=np.float32)
    reference_mask[query_checkpoint : query_checkpoint + effort + 1] = np.float32(
        1.0 / (effort + 1)
    )

    produced_events = row["events"]
    produced_advances = row["advances"]
    produced_mask = row["masks"]
    np.testing.assert_array_equal(
        produced_events[: active_indices.size, 0], padded[active_indices]
    )
    np.testing.assert_array_equal(produced_mask[: active_indices.size], reference_mask)

    def pack(events, advances, mask):
        return jnp.concatenate(
            (
                jnp.asarray(events, dtype=jnp.float32),
                jnp.asarray(advances, dtype=jnp.float32)[..., None],
                jnp.sqrt(jnp.asarray(mask, dtype=jnp.float32))[..., None],
            ),
            axis=-1,
        )

    produced_inputs = pack(
        produced_events,
        produced_advances,
        produced_mask[:, None],
    )
    reference_inputs = pack(
        padded[active_indices, None, :],
        np.ones((active_indices.size, 1), dtype=np.bool_),
        reference_mask[:, None],
    )

    model_config = _tiny_trace_model_config(rows)
    target = np.zeros((1, model_config.compact_output_width), dtype=np.float32)
    target[0, int(row["heights"])] = 1.0
    target[0, MAX_GRID_SIZE + int(row["widths"])] = 1.0

    def factory():
        return _TrainingRowTraceProbe(model_config, jnp.asarray(target))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        produced_gradient = chunked_online_param_gradients(
            factory,
            produced_inputs,
            algo_factory=_padding_probe_pp_prop,
            chunk_size=4,
        )
        reference_gradient = chunked_online_param_gradients(
            factory,
            reference_inputs,
            algo_factory=_padding_probe_pp_prop,
            chunk_size=4,
        )

    assert gradient_norm(reference_gradient) > 1e-3
    assert relative_deviation(produced_gradient, reference_gradient) == pytest.approx(
        0.0, abs=1e-7
    )


def test_training_mask_supervises_only_latent_row_ticks_with_unit_weight(example):
    """An effort-R update weights only its R generated row ticks."""
    config = example.ExperimentConfig.smoke_config(decoder_mode="row_refinement")
    rows = example._row_config(config)
    tensors = _materialize_training(example, example._load_data(config), config, rows)

    for effort, mask in zip(tensors.efforts, tensors.masks, strict=True):
        supervised = np.flatnonzero(mask)
        assert supervised.size == int(effort)
        np.testing.assert_array_equal(
            supervised,
            np.arange(supervised[-1] - int(effort) + 1, supervised[-1] + 1),
        )
        np.testing.assert_allclose(
            mask[supervised], np.full(int(effort), 1.0 / int(effort))
        )
        assert config.latent_steps == 60
        assert not mask[supervised[0] - 1]
        assert float(np.sum(mask)) == pytest.approx(1.0)


def test_protocol_v2_training_uniformly_supervises_complete_decoder_sweeps(example):
    config = example.ExperimentConfig.smoke_config(
        decoder_mode="latent_row_decode",
        memory_coding="learned_update",
    )
    rows = example._row_config(config)
    tensors = _materialize_training(example, example._load_data(config), config, rows)

    assert sorted(tensors.efforts.tolist()) == [0, 30, 60]
    for mask, events, advances in zip(
        tensors.masks, tensors.events, tensors.advances, strict=True
    ):
        supervised = np.flatnonzero(mask)
        assert supervised.size == 30
        np.testing.assert_allclose(mask[supervised], np.full(30, 1.0 / 30.0))
        assert not advances[supervised].any()
        assert np.all(events[supervised, :, rows.phase_slice.start + 1] == 1.0)
        assert float(mask.sum()) == pytest.approx(1.0)


def test_training_tensor_terminals_follow_each_sample_effort(example):
    config = example.ExperimentConfig.smoke_config(decoder_mode="row_refinement")
    data = example._load_data(config)
    rows = example._row_config(config)

    tensors = _materialize_training(example, data, config, rows)
    horizon = example._training_sequence_length(data, config)

    assert tensors.events.shape == (3, horizon, 1, rows.input_width)
    assert tensors.advances.shape == (3, horizon, 1)
    assert tensors.colors.shape == (3, 1, 30, 30)
    assert np.all((0 <= tensors.heights) & (tensors.heights < 30))
    assert np.all((0 <= tensors.widths) & (tensors.widths < 30))
    assert np.all(np.sum(tensors.masks, axis=1) == 1.0)
    for index, effort in enumerate(tensors.efforts):
        supervised = np.flatnonzero(tensors.masks[index])
        first_row_tick = int(supervised[0])
        terminal = int(supervised[-1])
        assert terminal - first_row_tick + 1 == int(effort)
        assert (
            tensors.events[index, first_row_tick - 1, 0, rows.valid_slice.start] == 1.0
        )
        assert (
            np.count_nonzero(tensors.events[index, first_row_tick : terminal + 1]) == 0
        )
        assert tensors.advances[index, first_row_tick : terminal + 1].all()


def test_model_configuration_parameter_copy_and_digest_are_explicit(example):
    config = example.ExperimentConfig.smoke_config(decoder_mode="row_refinement")
    rows = example._row_config(config)
    model_config = example._model_config(config, rows, batch_size=3)
    assert model_config.batch_size == 3
    assert model_config.input_width == rows.input_width
    assert model_config.recurrent_edges == 1024
    assert model_config.decoder_mode == "row_refinement"
    assert model_config.refinement_steps == 60
    assert isinstance(model_config.refinement_layout, RowRefinementLayout)
    assert model_config.refinement_layout.input_width == rows.input_width
    assert model_config.training_output_width == 360
    assert model_config.checkpoint_output_width == 9060

    source_state = SimpleNamespace(value={"weight": np.array([1.0, 2.0])})
    target_state = SimpleNamespace(value={"weight": np.array([0.0, 0.0])})
    source = SimpleNamespace(states=lambda kind: {"p": source_state})
    target = SimpleNamespace(states=lambda kind: {"p": target_state})
    example._copy_parameters(source, target)
    assert np.array_equal(target_state.value["weight"], np.array([1.0, 2.0]))
    source_state.value["weight"][0] = 9.0
    assert target_state.value["weight"][0] == 1.0

    digest = example._tree_digest({"p": target_state.value})
    changed = example._tree_digest({"p": {"weight": np.array([1.0, 3.0])}})
    assert digest != changed
    changes = example._parameter_change_evidence(
        {"p": {"weight": np.array([1.0, 2.0])}},
        {"p": {"weight": np.array([1.0, 3.0])}},
    )
    assert changes["p"]["changed"] is True
    assert changes["p"]["l2_delta"] == 1.0

    mismatch = SimpleNamespace(states=lambda kind: {"other": target_state})
    with pytest.raises(ValueError, match="parameter paths differ"):
        example._copy_parameters(source, mismatch)


def test_model_configuration_keeps_legacy_decoder_dispatch_explicit(example):
    config = example.ExperimentConfig.smoke_config(decoder_mode="legacy_cp")
    rows = example._row_config(config)

    model_config = example._model_config(config, rows, batch_size=1)

    assert model_config.decoder_mode == "legacy_cp"
    assert model_config.refinement_layout is None
    assert model_config.training_output_width == model_config.compact_output_width


def test_model_configuration_wires_opt_in_associative_memory_features(example):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    legacy = example._model_config(
        example.ExperimentConfig.smoke_config(decoder_mode="legacy_cp"),
        rows,
        batch_size=3,
    )
    assert legacy.context_memory_width == 0
    assert legacy.memory_key_indices == ()
    assert legacy.memory_value_indices == ()
    assert legacy.demonstration_phase_index is None
    assert legacy.query_phase_index is None
    assert legacy.input_side_valid_index is None
    assert legacy.output_side_valid_index is None

    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(decoder_mode="row_refinement"),
        context_memory_width=32,
        memory_decay=0.95,
    )
    memory = example._model_config(config, rows, batch_size=3)
    expected = example.associative_memory_feature_indices(rows)
    assert memory.context_memory_width == 32
    assert memory.memory_decay == 0.95
    assert memory.memory_key_indices == expected.key_indices
    assert memory.memory_value_indices == expected.value_indices
    assert len(memory.memory_key_indices) == 424
    assert len(memory.memory_value_indices) == 424
    assert memory.demonstration_phase_index == rows.phase_slice.start
    assert memory.query_phase_index == rows.phase_slice.start + 1
    assert memory.input_side_valid_index == rows.side_valid_slice.start
    assert memory.output_side_valid_index == rows.side_valid_slice.start + 1


def test_memory_architecture_report_records_raw_width_and_dense_state_cost(example):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    legacy = example._memory_architecture_report(
        example.ExperimentConfig(
            structural_only=True,
            context_memory_width=0,
            memory_coding="frozen",
            decoder_mode="legacy_cp",
        ),
        rows,
        training_batch_size=1,
        evaluation_batch_size=419,
    )
    assert legacy == {
        "reasoning_mode": "legacy_reservoir",
        "context_memory_width": 0,
        "memory_decay": 1.0,
        "raw_key_feature_width": 0,
        "raw_value_feature_width": 0,
        "context_memory_bytes_per_example": 0,
        "context_memory_bytes_training_batch": 0,
        "context_memory_bytes_evaluation_batch": 0,
    }

    memory = example._memory_architecture_report(
        example.ExperimentConfig(
            structural_only=True,
            context_memory_width=32,
            memory_decay=1.0,
        ),
        rows,
        training_batch_size=1,
        evaluation_batch_size=419,
    )
    assert memory["reasoning_mode"] == "associative_workspace"
    assert memory["raw_key_feature_width"] == 424
    assert memory["raw_value_feature_width"] == 424
    assert memory["context_memory_bytes_per_example"] == 4096
    assert memory["context_memory_bytes_training_batch"] == 4096
    assert memory["context_memory_bytes_evaluation_batch"] == 1_716_224


def test_model_memory_report_adds_carrier_metadata_without_changing_legacy_json(
    example,
):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    legacy_model = LatentWorkspaceModel(
        example._model_config(
            example.ExperimentConfig.smoke_config(decoder_mode="legacy_cp"),
            rows,
            batch_size=1,
        )
    )
    legacy = example._model_memory_report(legacy_model)
    expected_legacy = {
        "mode": "legacy_reservoir",
        "memory_width": 0,
        "key_feature_width": 0,
        "value_feature_width": 0,
        "key_map": None,
        "value_map": None,
        "rff_gamma": None,
        "key_basis_seed": None,
        "key_bias_seed": None,
        "value_basis_seed": None,
        "key_basis_sha256": None,
        "key_bias_sha256": None,
        "value_basis_sha256": None,
        "write_component_type": None,
        "query_component_type": None,
        "read_component_type": None,
    }
    assert json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    ) == json.dumps(expected_legacy, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    memory_config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
        context_memory_width=2,
        memory_decay=1.0,
    )
    memory_model = LatentWorkspaceModel(
        example._model_config(memory_config, rows, batch_size=1)
    )
    base_memory = dataclasses.asdict(memory_model.associative_memory_report())
    memory = example._model_memory_report(memory_model)

    assert {key: memory[key] for key in base_memory} == base_memory
    assert set(memory) - set(base_memory) == {
        "carrier_stabilizer",
        "carrier_radius",
        "carrier_consumers",
        "carrier_normalization_by_consumer",
        "memory_read_rms",
        "memory_drive_rms",
            "gate_saturation_fraction",
            "gate_channel_activation",
            "memory_read_interval",
            "memory_read_count",
            "memory_read_active",
        }
    assert memory["gate_saturation_fraction"] is None
    assert memory["gate_channel_activation"] is None
    assert memory["carrier_stabilizer"] == "per_example_stopped_unit_l2_cap"
    assert memory["carrier_radius"] == 1.0
    assert memory["carrier_consumers"] == (
        "answer_row_head",
        "answer_shape_head",
        "workspace_query_projection",
    )
    assert memory["carrier_normalization_by_consumer"] == {
        "answer_row_head": "per_example_unit_rms",
        "answer_shape_head": "per_example_unit_rms",
        "workspace_query_projection": "per_example_stopped_unit_l2_cap",
    }


def test_compiler_evidence_retains_warnings_and_parameter_classification(example):
    class Level(Enum):
        INFO = "info"
        WARNING = "warning"

    class Kind(Enum):
        INCLUDED = "relation_included"
        EXCLUDED = "relation_excluded_non_temporal"

    diagnostics = (
        SimpleNamespace(
            kind=Kind.INCLUDED,
            level=Level.INFO,
            message="included",
            weight_path=("rec_syn", "weight"),
            hidden_paths=(("neu", "V"),),
            context={
                "path_classification": {("neu", "V"): "all_direct"},
            },
        ),
        SimpleNamespace(
            kind=Kind.EXCLUDED,
            level=Level.WARNING,
            message="head is non-temporal",
            weight_path=("height_head", "weight"),
        ),
    )
    report = SimpleNamespace(
        diagnostics=diagnostics,
        hidden_groups=[object()],
        etrace_weights=[(("rec_syn", "weight"), [0])],
        excluded_weights=[(("height_head", "weight"), "non_temporal")],
    )

    evidence = example._compiler_evidence(SimpleNamespace(report=report))

    assert evidence["counts"] == {
        "hidden_groups": 1,
        "etrace_weights": 1,
        "excluded_weights": 1,
        "warnings": 1,
        "errors": 0,
    }
    assert evidence["etrace_weights"][0]["parameter"] == "rec_syn.weight"
    assert evidence["excluded_weights"][0]["parameter"] == "height_head.weight"
    assert evidence["diagnostics"][1]["level"] == "warning"
    assert evidence["diagnostics"][1]["message"] == "head is non-temporal"
    assert evidence["diagnostics"][0]["path_classification_by_hidden_state"] == {
        "neu.V": "all_direct"
    }


def test_structural_training_tensors_are_empty(example):
    config = example.ExperimentConfig(structural_only=True, training_updates=0)
    fixture = example._load_data(config)
    tensors = _materialize_training(
        example, fixture, config, example._row_config(config)
    )
    assert tensors.events.size == 0
    assert tensors.task_fingerprints == ()


def test_derangement_changes_only_demonstration_output_associations(example):
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1,),)), ArcGrid(((2,), (2,)))),
            ArcPair(ArcGrid(((3,), (3,), (3,))), ArcGrid(((4,),))),
        ),
        test=(ArcPair(ArcGrid(((5,),)), ArcGrid(((6,),))),),
        task_id="unequal",
    )
    changed = example._derange_task(task)

    assert changed is not None
    assert tuple(pair.input for pair in changed.train) == tuple(
        pair.input for pair in task.train
    )
    assert tuple(pair.output for pair in changed.train) == tuple(
        pair.output for pair in task.train[1:] + task.train[:1]
    )
    assert changed.test == task.test
    rows = RowEventConfig(max_demonstrations=2)
    intact = encode_query_episode(task, 0, rows)
    shuffled = encode_query_episode(changed, 0, rows)
    assert (intact.query_start, intact.query_stop) == (
        shuffled.query_start,
        shuffled.query_stop,
    )


def test_derangement_is_unavailable_for_one_demo(example):
    pair = ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),)))
    assert example._derange_task(ArcTask((pair,), (pair,), task_id="one")) is None


def test_control_sequences_preserve_query_boundaries_and_zero_only_context(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    rows = example._row_config(config)
    records = example._evaluation_records(data, config, rows)

    intact, advances, stops, intact_meta = example._arm_sequences(
        records,
        config,
        rows,
        arm="intact",
        source_tasks=data.evaluation,
    )
    no_context, no_advances, no_stops, _ = example._arm_sequences(
        records,
        config,
        rows,
        arm="no_context",
        source_tasks=data.evaluation,
    )
    shuffled, shuffled_advances, shuffled_stops, shuffled_meta = example._arm_sequences(
        records,
        config,
        rows,
        arm="shuffled",
        source_tasks=data.evaluation,
    )

    assert np.array_equal(stops, no_stops)
    assert np.array_equal(stops, shuffled_stops)
    assert np.array_equal(advances, no_advances)
    assert np.array_equal(advances, shuffled_advances)
    for index, record in enumerate(records):
        assert np.count_nonzero(no_context[: record.encoded.query_start, index]) == 0
        assert np.array_equal(
            no_context[record.encoded.query_start :, index],
            intact[record.encoded.query_start :, index],
        )
    assert all(item["timing_matched"] for item in intact_meta + shuffled_meta)
    assert not np.array_equal(intact, shuffled)


def test_duplicate_task_ids_keep_distinct_strict_scoring_keys(example):
    pair_a = ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),)))
    pair_b = ArcPair(ArcGrid(((3,),)), ArcGrid(((4,),)))
    task_a = ArcTask((pair_a, pair_b), (pair_a,), task_id="duplicate-name")
    task_b = ArcTask((pair_b, pair_a), (pair_b,), task_id="duplicate-name")
    fixture = smoke_loaded_dataset()
    origins = (
        example._OriginTask("same-source", "fixture", task_a),
        example._OriginTask("same-source", "fixture", task_b),
    )
    data = example._ExperimentData(origins, origins, (fixture,), True)
    config = example.ExperimentConfig.smoke_config()

    records = example._evaluation_records(data, config, example._row_config(config))

    assert len(records) == 2
    assert records[0].task_key != records[1].task_key
    assert all("duplicate-name" in record.task_key for record in records)


def test_one_demo_shuffle_is_reported_unavailable_without_timing_change(example):
    pair = ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),)))
    task = ArcTask((pair,), (pair,), task_id="one-demo")
    fixture = smoke_loaded_dataset()
    origin = example._OriginTask("fixture", "fixture", task)
    data = example._ExperimentData((origin,), (origin,), (fixture,), True)
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    records = example._evaluation_records(data, config, rows)

    _, _, stops, metadata = example._arm_sequences(
        records,
        config,
        rows,
        arm="shuffled",
        source_tasks=data.evaluation,
    )

    assert stops.tolist() == [records[0].encoded.query_stop]
    assert metadata == [
        {
            "available": False,
            "reason": "fewer than two demonstrations",
            "timing_matched": True,
        }
    ]


def test_equal_output_rotation_is_excluded_from_pairing_applicability(example):
    output = ArcGrid(((7,),))
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1,),)), output),
            ArcPair(ArcGrid(((2,),)), output),
        ),
        test=(ArcPair(ArcGrid(((1,),)), output),),
        task_id="equal-output-rotation",
    )
    fixture = smoke_loaded_dataset()
    origin = example._OriginTask("fixture", "fixture", task)
    data = example._ExperimentData((origin,), (origin,), (fixture,), True)
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    records = example._evaluation_records(data, config, rows)

    intact, _, _, _ = example._arm_sequences(
        records,
        config,
        rows,
        arm="intact",
        source_tasks=data.evaluation,
    )
    shuffled, _, _, metadata = example._arm_sequences(
        records,
        config,
        rows,
        arm="shuffled",
        source_tasks=data.evaluation,
    )

    np.testing.assert_array_equal(shuffled, intact)
    assert metadata == [
        {
            "available": False,
            "reason": "rotation leaves demonstration associations unchanged",
            "timing_matched": True,
        }
    ]


@pytest.mark.parametrize(
    ("decoder_mode", "output_width"),
    [("legacy_cp", 340), ("row_refinement", 9060)],
)
def test_scoring_trajectory_and_null_control_share_the_same_frozen_windows(
    example, decoder_mode, output_width
):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))
    batch = len(records)
    compact = np.zeros((TEST_DEPTH_COUNT, batch, output_width), dtype=np.float32)
    spikes = np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32)
    voltage = np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32)
    feedforward = np.zeros_like(voltage)
    recurrent = np.zeros_like(voltage)

    metrics, details = example._score_windows(
        compact, records, color_rank=4, decoder_mode=decoder_mode
    )
    reports, aggregate = example._trajectory_reports(
        compact,
        spikes,
        voltage,
        feedforward,
        recurrent,
        records,
        color_rank=4,
        decoder_mode=decoder_mode,
    )
    control = example._control_summary(
        "identity",
        (compact, spikes, voltage, feedforward, recurrent),
        (
            compact.copy(),
            spikes.copy(),
            voltage.copy(),
            feedforward.copy(),
            recurrent.copy(),
        ),
        records,
        4,
        decoder_mode,
        metrics,
        [{"available": True, "timing_matched": True} for _ in records],
    )

    assert set(metrics) == {"0", "30", "60"}
    assert all(len(details[str(effort)]) == batch for effort in TEST_CHECKPOINTS)
    assert all(
        candidate["provenance"] == "model"
        for rows_at_effort in details.values()
        for row in rows_at_effort
        for candidate in row["candidates"]
    )
    assert all(
        len(row["candidates"]) == 2
        for rows_at_effort in details.values()
        for row in rows_at_effort
    )
    assert all(
        row["primary_candidate_mode"] == "model_only"
        for rows_at_effort in details.values()
        for row in rows_at_effort
    )
    assert len(reports) == batch
    assert len(aggregate) == TEST_DEPTH_COUNT
    assert aggregate[0]["unique_state_hashes"] == 1
    assert control["causally_null_query_count"] == batch
    assert control["available_query_count"] == batch
    assert control["timing_matched_query_count"] == batch
    assert control["trajectory_comparison"]["causally_null_at_measured_precision"]
    assert "checkpoint_queries" not in control


def test_primary_scoring_uses_latest_checkpoint_global_top_two(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))
    compact = np.zeros((TEST_DEPTH_COUNT, len(records), 9060), dtype=np.float32)
    compact[30, :, 0] = 20.0
    compact[30, :, 30] = 20.0
    compact[30, :, 60 + 3] = 20.0
    compact[60, :, 0] = 20.0
    compact[60, :, 30] = 20.0
    compact[60, :, 60 + 6] = 20.0

    _metrics, details = example._score_windows(
        compact, records, color_rank=4, decoder_mode="row_refinement"
    )

    final = details["60"][0]
    assert final["submission_role"] == "primary_submission"
    assert final["candidates"][0]["grid"] == [[6]]
    assert final["candidates"][0]["source_checkpoint"] == 60
    assert final["candidates"][0]["selection_role"] == ("latest_sweep_joint_argmax")
    assert final["candidates"][1]["grid"] == [[0]]
    assert final["candidates"][1]["source_checkpoint"] == 60
    assert final["candidates"][1]["selection_role"] == (
        "latest_sweep_logit_runner_up"
    )
    assert details["0"][0]["submission_role"] == "diagnostic_only"


def test_primary_candidate_bytes_do_not_depend_on_official_targets(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))
    changed_records = tuple(
        dataclasses.replace(
            record,
            encoded=dataclasses.replace(record.encoded, target=ArcGrid(((9,),))),
        )
        for record in records
    )
    compact = np.zeros((TEST_DEPTH_COUNT, len(records), 9060), dtype=np.float32)

    _left_metrics, left = example._score_windows(
        compact, records, color_rank=4, decoder_mode="row_refinement"
    )
    _right_metrics, right = example._score_windows(
        compact, changed_records, color_rank=4, decoder_mode="row_refinement"
    )

    assert json.dumps(left["60"][0]["candidates"], sort_keys=True).encode() == (
        json.dumps(right["60"][0]["candidates"], sort_keys=True).encode()
    )


def test_evaluation_identity_and_encoded_record_ignore_official_targets(example):
    fixture = smoke_loaded_dataset()
    original = fixture.tasks[0]
    changed = dataclasses.replace(
        original,
        test=tuple(
            dataclasses.replace(pair, output=ArcGrid(((9,),))) for pair in original.test
        ),
    )
    original_origin = example._OriginTask("evaluation", "evaluation", original)
    changed_origin = example._OriginTask("evaluation", "evaluation", changed)
    original_data = example._ExperimentData((), (original_origin,), (), True)
    changed_data = example._ExperimentData((), (changed_origin,), (), True)
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)

    original_record = example._evaluation_records(original_data, config, rows)[0]
    changed_record = example._evaluation_records(changed_data, config, rows)[0]

    assert original_record.task_key == changed_record.task_key
    np.testing.assert_array_equal(
        original_record.encoded.events, changed_record.encoded.events
    )
    assert original_record.encoded.query_stop == changed_record.encoded.query_stop
    assert original_record.encoded.target != changed_record.encoded.target


def test_completion_report_uses_all_400_expected_task_identities(example, monkeypatch):
    fixture = smoke_loaded_dataset()
    evaluation = tuple(
        example._OriginTask(
            "ARC-AGI-1 evaluation",
            "evaluation",
            dataclasses.replace(fixture.tasks[0], task_id=f"task-{index:03d}"),
        )
        for index in range(400)
    )
    data = example._ExperimentData((), evaluation, (), False)
    captured = {}

    def assess(rows, expected):
        captured["rows"] = rows
        captured["expected"] = expected
        return {
            "primary_candidate_mode": "model_only",
            "required_task_count": 400,
            "evaluated_task_count": 400,
            "evaluated_query_count": 400,
            "required_exact_task_count": 160,
            "exact_task_count": 160,
            "strict_task_pass_at_2": 0.4,
            "passed": True,
            "tasks": {},
        }

    monkeypatch.setattr(example, "assess_model_only_completion", assess)
    rows = [{"sentinel": True}]

    report = example._model_only_completion_report(
        rows, {"tasks": {}}, data, example.ExperimentConfig()
    )

    assert report["eligible_for_completion"] is True
    assert report["passed"] is True
    assert captured["rows"] is rows
    assert len(captured["expected"]) == 400
    assert set(captured["expected"].values()) == {len(fixture.tasks[0].test)}


def test_capped_and_smoke_completion_reports_cannot_pass(example, monkeypatch):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    monkeypatch.setattr(
        example,
        "assess_model_only_completion",
        lambda *args: pytest.fail("incomplete evaluations must not call the full gate"),
    )

    report = example._model_only_completion_report(
        [],
        {
            "query_count": 1,
            "task_count": 1,
            "strict_task_pass_at_2": 1.0,
            "tasks": {"fixture": {"query_count": 1, "pass_at_2": True}},
        },
        data,
        config,
    )

    assert report["eligible_for_completion"] is False
    assert report["passed"] is False
    assert report["exact_task_count"] == 1
    assert report["required_task_count"] == 400
    assert "complete 400-task" in report["eligibility_reason"]


def test_scoring_rejects_non_model_candidate_provenance(example, monkeypatch):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))
    compact = np.zeros((TEST_DEPTH_COUNT, len(records), 340), dtype=np.float32)

    class Candidate:
        grid = np.zeros((1, 1), dtype=np.int32)

        def to_dict(self):
            return {"grid": [[0]], "provenance": "verified_rule"}

    monkeypatch.setattr(example, "decode_candidates", lambda logits: [Candidate()])

    with pytest.raises(ValueError, match="non-model candidate provenance"):
        example._score_windows(compact, records, color_rank=4, decoder_mode="legacy_cp")


def test_primary_evaluation_has_no_rule_proposal_path(example):
    evaluate_source = ast.get_source_segment(
        EXAMPLE.read_text(encoding="utf-8"),
        next(
            node
            for node in ast.parse(EXAMPLE.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_evaluate"
        ),
    )
    assert evaluate_source is not None
    assert "_rule_proposals" not in evaluate_source
    assert "verified_rule_candidates" not in evaluate_source


def test_unavailable_shuffle_queries_are_excluded_from_control_statistics(
    example, monkeypatch
):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))[:2]
    compact = np.zeros((TEST_DEPTH_COUNT, 2, 340), dtype=np.float32)
    spikes = np.zeros((TEST_DEPTH_COUNT, 2, 8), dtype=np.float32)
    voltage = np.zeros((TEST_DEPTH_COUNT, 2, 8), dtype=np.float32)
    feedforward = np.zeros_like(voltage)
    recurrent = np.zeros_like(voltage)
    captured: dict[str, object] = {}

    def score(subset_compact, subset_records, color_rank, decoder_mode):
        captured["scored_records"] = len(subset_records)
        captured["scored_batch"] = subset_compact.shape[1]
        captured["decoder_mode"] = decoder_mode
        details = {
            str(effort): [{"candidates": [{"grid": [[0]]}]} for _ in subset_records]
            for effort in TEST_CHECKPOINTS
        }
        return {str(effort): _metric() for effort in TEST_CHECKPOINTS}, details

    def compare(
        intact_spikes, intact_voltage, control_spikes, control_voltage, **kwargs
    ):
        captured["comparison_width"] = intact_spikes.shape[1]
        return {"causally_null_at_measured_precision": True}

    monkeypatch.setattr(example, "_score_windows", score)
    monkeypatch.setattr(example, "compare_control_trajectories", compare)
    metadata = [
        {"available": False, "timing_matched": True, "reason": "one demo"},
        {"available": True, "timing_matched": True},
    ]

    result = example._control_summary(
        "shuffle",
        (compact, spikes, voltage, feedforward, recurrent),
        (
            compact.copy(),
            spikes.copy(),
            voltage.copy(),
            feedforward.copy(),
            recurrent.copy(),
        ),
        records,
        4,
        "legacy_cp",
        {str(effort): _metric() for effort in TEST_CHECKPOINTS},
        metadata,
    )

    assert captured == {
        "scored_records": 1,
        "scored_batch": 1,
        "decoder_mode": "legacy_cp",
        "comparison_width": 8,
    }
    assert result["query_count"] == 2
    assert result["applicable_query_count"] == 1
    assert result["unavailable_query_count"] == 1


def test_all_unavailable_shuffle_renders_as_unavailable(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))[:1]
    compact = np.zeros((TEST_DEPTH_COUNT, 1, 340), dtype=np.float32)
    state = np.zeros((TEST_DEPTH_COUNT, 1, 8), dtype=np.float32)
    window = (compact, state, state.copy(), state.copy(), state.copy())

    unavailable = example._control_summary(
        "shuffle",
        window,
        tuple(value.copy() for value in window),
        records,
        4,
        "legacy_cp",
        {str(effort): _metric() for effort in TEST_CHECKPOINTS},
        [{"available": False, "timing_matched": True, "reason": "one demo"}],
    )
    payload = _evaluation_payload()
    payload["controls"] = dict(payload["controls"], shuffled_demonstrations=unavailable)
    result = {
        "evaluation": payload,
        "model": {},
        "training": {},
        "qualification": {},
    }
    path = tmp_path / "unavailable.png"

    report = example._render_report(result)
    example._plot(result, path)

    assert unavailable["metrics_by_effort"] == {}
    assert unavailable["trajectory_comparison"]["available"] is False
    assert "shuffled_demonstrations: applicable=0/1, unavailable=1" in report
    assert path.read_bytes().startswith(b"\x89PNG")


def test_state_repeat_tolerance_keeps_byte_identity_separate(example):
    compact = np.zeros((2, 1, 3), dtype=np.float32)
    spikes = np.zeros((2, 1, 4), dtype=np.float32)
    voltage = np.ones((2, 1, 4), dtype=np.float32)
    current = np.ones((2, 1, 4), dtype=np.float32)
    intact = (compact, spikes, voltage, current, current.copy())
    roundoff = (
        compact.copy(),
        spikes.copy(),
        voltage + np.float32(5e-7),
        current.copy(),
        current + np.float32(5e-7),
    )

    tolerated = example._state_tolerance_summary(intact, roundoff)
    excessive = list(roundoff)
    excessive[2] = voltage + np.float32(2e-6)
    rejected = example._state_tolerance_summary(intact, tuple(excessive))

    assert tolerated["state_byte_identical"] is False
    assert tolerated["within_declared_tolerance"] is True
    assert tolerated["spike_hamming_count"] == 0
    assert rejected["within_declared_tolerance"] is False


def test_state_tolerance_rejects_one_bad_query_hidden_by_batch_average(example):
    compact = np.zeros((2, 2, 3), dtype=np.float32)
    spikes = np.zeros((2, 2, 4), dtype=np.float32)
    voltage = np.zeros((2, 2, 4), dtype=np.float32)
    current = np.zeros_like(voltage)
    intact = (compact, spikes, voltage, current, current.copy())
    candidate_voltage = voltage.copy()
    candidate_voltage[:, 0, :] = np.float32(1.1e-6)
    candidate = (
        compact.copy(),
        spikes.copy(),
        candidate_voltage,
        current.copy(),
        current.copy(),
    )

    summary = example._state_tolerance_summary(intact, candidate)

    assert summary["within_declared_tolerance"] is False
    assert summary["within_tolerance_by_query"] == [False, True]
    assert summary["within_tolerance_query_count"] == 1
    assert summary["query_count"] == 2
    assert summary["per_query_maximum_rms"]["voltage"][0] > 1e-6


def test_state_tolerance_rejects_logit_drift_without_argmax_drift(example):
    compact = np.zeros((2, 1, 3), dtype=np.float32)
    state = np.zeros((2, 1, 4), dtype=np.float32)
    intact = (compact, state, state.copy(), state.copy(), state.copy())
    shifted_compact = compact + np.float32(2e-6)
    candidate = (
        shifted_compact,
        state.copy(),
        state.copy(),
        state.copy(),
        state.copy(),
    )

    summary = example._state_tolerance_summary(intact, candidate)

    np.testing.assert_array_equal(
        np.argmax(compact, axis=2), np.argmax(shifted_compact, axis=2)
    )
    assert summary["state_byte_identical"] is True
    assert summary["compact_logits_byte_identical"] is False
    assert summary["maximum_rms"]["compact_logits"] > 1e-6
    assert summary["within_declared_tolerance"] is False


def test_byte_identity_distinguishes_signed_zero_raw_bytes(example):
    compact = np.zeros((1, 1, 2), dtype=np.float32)
    spikes = np.zeros((1, 1, 2), dtype=np.float32)
    positive_zero = np.zeros((1, 1, 2), dtype=np.float32)
    negative_zero = np.full((1, 1, 2), np.float32(-0.0), dtype=np.float32)
    intact = (
        compact,
        spikes,
        positive_zero,
        positive_zero.copy(),
        positive_zero.copy(),
    )
    candidate = (
        compact.copy(),
        spikes.copy(),
        negative_zero,
        positive_zero.copy(),
        positive_zero.copy(),
    )

    summary = example._state_tolerance_summary(intact, candidate)

    assert np.array_equal(positive_zero, negative_zero)
    assert summary["state_byte_identical"] is False
    assert summary["state_byte_identical_by_query"] == [False]
    assert summary["within_declared_tolerance"] is True
    assert summary["intact_dtype_by_state"]["voltage"] == "float32"
    assert summary["candidate_dtype_by_state"]["voltage"] == "float32"


def test_state_tolerance_rejects_non_float32_physical_evidence(example):
    compact = np.zeros((1, 1, 2), dtype=np.float32)
    spikes = np.zeros((1, 1, 2), dtype=np.float32)
    physical = np.zeros((1, 1, 2), dtype=np.float32)
    intact = (compact, spikes, physical, physical.copy(), physical.copy())
    candidate = (
        compact.copy(),
        spikes.copy(),
        physical.astype(np.float64),
        physical.copy(),
        physical.copy(),
    )

    summary = example._state_tolerance_summary(intact, candidate)

    assert summary["maximum_rms"]["voltage"] == 0.0
    assert summary["required_float32_dtypes"] is False
    assert summary["within_declared_tolerance"] is False


@pytest.mark.parametrize(
    ("candidate_exact", "metric_value", "failed_field"),
    [
        (False, 0.0, "decoded_candidates_exact"),
        (True, 1.0, "metrics_exact"),
    ],
)
def test_checkpoint_zero_gate_rejects_candidate_or_metric_mismatch(
    example, candidate_exact, metric_value, failed_field
):
    intact_metrics = {"0": _metric()}
    control = {
        "metrics_by_effort": {"0": _metric(metric_value)},
        "decoded_candidates_match_intact_by_effort": {"0": candidate_exact},
    }

    summary = example._checkpoint_zero_gate_summary(
        intact_metrics, control, {"within_declared_tolerance": True}
    )

    assert summary["state_within_tolerance"] is True
    assert summary[failed_field] is False
    assert summary["matched"] is False


def test_structural_path_compiles_pp_prop_before_qualification(example, monkeypatch):
    calls: list[object] = []

    class Learner:
        def reset_state(self, *, batch_size):
            calls.append(("learner_reset", batch_size))

    class Model:
        config = SimpleNamespace(batch_size=1)

        def reset_state(self):
            calls.append("model_reset")

    monkeypatch.setattr(
        example, "compile_pp_prop", lambda model: calls.append(model) or Learner()
    )
    config = example.ExperimentConfig(structural_only=True, training_updates=0)
    empty = np.zeros((0,), dtype=np.float32)
    tensors = example._TrainingTensors(
        empty, empty, empty, empty, empty, empty, empty, ()
    )

    result = example._train_model(Model(), tensors, config)

    assert result["pp_prop_compiled"] is True
    assert result["performed"] is False
    assert calls[-2:] == ["model_reset", ("learner_reset", 1)]


@pytest.mark.parametrize(
    ("decoder_mode", "output_width"),
    [("row_refinement", 360), ("legacy_cp", 340)],
)
def test_training_uses_explicit_decoder_loss_and_all_sweep_efforts(
    example, monkeypatch, decoder_mode, output_width
):
    updates: list[object] = []
    row_loss_calls: list[tuple[object, ...]] = []
    legacy_loss_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Learner:
        param_states = {}

        def reset_state(self, *, batch_size):
            assert batch_size == 1

        def __call__(self, event, advance):
            assert event.shape == (1, 4)
            assert advance.shape == (1,)
            return jnp.zeros((1, output_width), dtype=jnp.float32)

        def etrace_grad(self, sequence, advance, *, step_fn, **kwargs):
            assert kwargs["loss_output"] == "scalar"
            objective = step_fn(sequence[0], advance[0])
            return {}, objective

    class Optimizer:
        def __init__(self, *, lr):
            assert lr > 0

        def register_trainable_weights(self, states):
            assert states == {}

        def update(self, gradients):
            updates.append(gradients)

    class Model:
        config = SimpleNamespace(
            batch_size=1,
            color_rank=4,
            decoder_mode=decoder_mode,
            event_valid_index=0,
        )
        reasoning_index = SimpleNamespace(value=jnp.asarray([1], dtype=jnp.int32))

        def reset_state(self):
            return None

        def project_recurrent_dale_weights(self):
            return None

    def host_for_loop(function, xs):
        return jnp.stack(
            [
                function(tuple(value[index] for value in xs))
                for index in range(xs[0].shape[0])
            ]
        )

    monkeypatch.setattr(example, "compile_pp_prop", lambda model: Learner())
    monkeypatch.setattr(
        example.braintools.optim,
        "OptaxOptimizer",
        lambda *, tx, lr: Optimizer(lr=lr),
    )
    monkeypatch.setattr(example.brainstate.transform, "jit", lambda function: function)
    monkeypatch.setattr(example.brainstate.transform, "for_loop", host_for_loop)

    def row_loss(*args):
        row_loss_calls.append(args)
        return jnp.ones((1,))

    def legacy_loss(*args, **kwargs):
        legacy_loss_calls.append((args, kwargs))
        return jnp.ones((1,))

    monkeypatch.setattr(example, "row_refinement_loss_per_example", row_loss)
    monkeypatch.setattr(example, "arc_loss_per_example", legacy_loss)
    monkeypatch.setattr(example, "parameter_snapshot", lambda model: {})
    monkeypatch.setattr(
        example.brainstate.nn, "clip_grad_norm", lambda value, limit: value
    )
    events = np.zeros((2, 1, 1, 4), dtype=np.float32)
    advances = np.ones((2, 1, 1), dtype=np.bool_)
    heights = np.asarray([[0], [1]], dtype=np.int32)
    widths = np.asarray([[1], [2]], dtype=np.int32)
    colors = np.zeros((2, 1, 30, 30), dtype=np.int32)
    masks = np.ones((2, 1), dtype=np.float32)
    tensors = example._TrainingTensors(
        events,
        advances,
        heights,
        widths,
        colors,
        masks,
        np.array([30, 60]),
        ("a", "b"),
        ("base-a", "base-b"),
        ("source", "source"),
        (0, 0),
    )

    config = example.ExperimentConfig.smoke_config(
        optimizer="adam",
        decoder_mode=decoder_mode,
        balanced_color_loss=decoder_mode == "legacy_cp",
    )
    result = example._train_model(Model(), [tensors], config)

    assert len(updates) == 2
    assert result["one_shared_model"] is True
    assert result["one_shared_optimizer_state"] is True
    assert result["optimizer_updates_by_effort"] == {"30": 1, "60": 1}
    assert result["losses"] == [1.0, 1.0]
    if decoder_mode == "row_refinement":
        assert result["supervised_depths"] == "latent_row_ticks_1..effort"
        assert len(row_loss_calls) == 2
        assert not legacy_loss_calls
        assert all(np.asarray(call[0]).shape == (1, 360) for call in row_loss_calls)
        assert all(np.asarray(call[1]).min() >= 0 for call in row_loss_calls)
        assert all(np.asarray(call[2]).min() >= 0 for call in row_loss_calls)
        assert all(np.asarray(call[4]).tolist() == [0] for call in row_loss_calls)
    else:
        assert result["supervised_depths"] == "0..effort"
        assert not row_loss_calls
        assert len(legacy_loss_calls) == 2
        assert all(
            call[1]["class_balanced_colors"] is True for call in legacy_loss_calls
        )
        assert [np.asarray(call[0][1]).tolist() for call in legacy_loss_calls] == [
            [1],
            [2],
        ]


def test_default_evaluation_runs_only_intact_without_adaptation_or_repeat(
    example, monkeypatch
):
    config = example.ExperimentConfig.smoke_config(
        decoder_mode="row_refinement"
    )
    config = dataclasses.replace(config, evaluation_controls=False)
    data = example._load_data(config)
    rows = example._row_config(config)
    records = example._evaluation_records(data, config, rows)
    batch = len(records)
    time_steps = rows.max_events + config.latent_steps
    run_markers: list[float] = []
    fake_model = SimpleNamespace(
        config=SimpleNamespace(color_rank=4, decoder_mode="row_refinement")
    )

    def sequences(records, config, rows, *, arm, source_tasks):
        assert arm == "intact"
        events = np.ones(
            (time_steps, batch, rows.input_width), dtype=np.float32
        )
        advances = np.ones((time_steps, batch), dtype=np.bool_)
        stops = np.asarray(
            [record.encoded.query_stop for record in records], dtype=np.int32
        )
        metadata = [{"available": True, "timing_matched": True} for _ in records]
        return events, advances, stops, metadata

    def packed(model, events, selected_indices, **kwargs):
        run_markers.append(float(np.asarray(events)[0, 0, 0]))
        return SimpleNamespace(
            compact_logits=np.zeros((TEST_DEPTH_COUNT, batch, 9060), dtype=np.float32),
            spikes=np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32),
            voltage=np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32),
            feedforward_current=np.zeros(
                (TEST_DEPTH_COUNT, batch, 8), dtype=np.float32
            ),
            recurrent_current=np.zeros(
                (TEST_DEPTH_COUNT, batch, 8), dtype=np.float32
            ),
            memory_read=np.zeros((TEST_DEPTH_COUNT, batch, 2), dtype=np.float32),
            final_context_memory=np.zeros((batch, 2, 2), dtype=np.float32),
        )

    metrics = {str(effort): _metric() for effort in TEST_CHECKPOINTS}
    checkpoint_queries = {
        str(effort): [
            {
                "submission_role": (
                    "primary_submission" if effort == 60 else "diagnostic_only"
                ),
                "candidates": _checkpoint_candidate_payloads(effort),
                "score": {"pass_at_2": False},
            }
            for _ in range(batch)
        ]
        for effort in TEST_CHECKPOINTS
    }
    monkeypatch.setattr(example, "_make_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(example, "_copy_parameters", lambda source, target: None)
    monkeypatch.setattr(
        example, "parameter_snapshot", lambda model: {"p": np.array([1])}
    )
    monkeypatch.setattr(example, "_arm_sequences", sequences)
    monkeypatch.setattr(example, "run_selected_packed_stream", packed)
    monkeypatch.setattr(
        example.brainstate.transform,
        "jit",
        lambda **kwargs: lambda function: function,
    )
    monkeypatch.setattr(
        example,
        "_task_local_adaptation_evaluation",
        lambda *args: pytest.fail("default evaluation invoked task-local adaptation"),
    )
    monkeypatch.setattr(
        example, "_score_windows", lambda *args: (metrics, checkpoint_queries)
    )
    monkeypatch.setattr(example, "_trajectory_reports", lambda *args: ([], []))

    result = example._evaluate(SimpleNamespace(), data, config, rows, SimpleNamespace())

    assert run_markers == [1.0]
    assert result["task_local_adaptation"]["performed"] is False
    assert result["execution"]["arm_order"] == ["intact"]
    assert "repeat_intact" not in result["execution"]["wall_seconds_by_arm"]


def test_evaluation_runs_four_frozen_arms_and_ablation_at_latent_step_one(
    example, monkeypatch
):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(decoder_mode="row_refinement"),
        task_local_adaptation=True,
        evaluation_controls=True,
    )
    data = example._load_data(config)
    rows = example._row_config(config)
    records = example._evaluation_records(data, config, rows)
    batch = len(records)
    time = rows.max_events + config.latent_steps
    stops = np.asarray(
        [record.encoded.query_stop for record in records], dtype=np.int32
    )
    run_calls: list[dict[str, object]] = []
    event_markers: list[float] = []
    jit_options: list[dict[str, object]] = []
    fake_model = SimpleNamespace(
        config=SimpleNamespace(color_rank=4, decoder_mode="row_refinement")
    )

    def sequences(records, config, rows, *, arm, source_tasks):
        marker = {"intact": 1.0, "no_context": 2.0, "shuffled": 3.0}[arm]
        events = np.full((time, batch, rows.input_width), marker, dtype=np.float32)
        advances = np.ones((time, batch), dtype=np.bool_)
        metadata = [{"available": True, "timing_matched": True} for _ in records]
        return events, advances, stops, metadata

    def packed(model, events, selected_indices, **kwargs):
        run_calls.append(kwargs)
        event_markers.append(float(np.asarray(events)[0, 0, 0]))
        assert selected_indices.shape == (TEST_DEPTH_COUNT, batch)
        return SimpleNamespace(
            compact_logits=np.zeros((TEST_DEPTH_COUNT, batch, 9060), dtype=np.float32),
            spikes=np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32),
            voltage=np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32),
            feedforward_current=np.zeros(
                (TEST_DEPTH_COUNT, batch, 8), dtype=np.float32
            ),
            recurrent_current=np.zeros((TEST_DEPTH_COUNT, batch, 8), dtype=np.float32),
            memory_read=np.zeros((TEST_DEPTH_COUNT, batch, 2), dtype=np.float32),
            final_context_memory=np.zeros((batch, 2, 2), dtype=np.float32),
        )

    def identity_jit(*, inline, name):
        jit_options.append({"inline": inline, "name": name})

        def decorate(function):
            return function

        return decorate

    metrics = {str(effort): _metric() for effort in TEST_CHECKPOINTS}
    adapted_metrics = {
        str(effort): _metric(float(effort == 60)) for effort in TEST_CHECKPOINTS
    }
    monkeypatch.setattr(example, "_make_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(example, "_copy_parameters", lambda source, target: None)
    monkeypatch.setattr(
        example, "parameter_snapshot", lambda model: {"p": np.array([1])}
    )
    monkeypatch.setattr(example, "_arm_sequences", sequences)
    monkeypatch.setattr(example, "run_selected_packed_stream", packed)
    monkeypatch.setattr(example.brainstate.transform, "jit", identity_jit)
    checkpoint_queries = {
        str(effort): [
            {
                "submission_role": (
                    "primary_submission" if effort == 60 else "diagnostic_only"
                ),
                "candidates": _checkpoint_candidate_payloads(effort),
                "score": {"pass_at_2": False},
            }
            for _ in range(batch)
        ]
        for effort in TEST_CHECKPOINTS
    }
    adapted_checkpoint_queries = copy.deepcopy(checkpoint_queries)
    adaptation_evidence = {
        "performed": True,
        "mode": "compiled_task_local_pp_prop_leave_one_out",
        "target_free_query_bank": True,
    }
    monkeypatch.setattr(
        example,
        "_task_local_adaptation_evaluation",
        lambda *args: (
            adapted_metrics,
            adapted_checkpoint_queries,
            adaptation_evidence,
        ),
    )
    monkeypatch.setattr(
        example, "_score_windows", lambda *args: (metrics, checkpoint_queries)
    )
    monkeypatch.setattr(example, "_trajectory_reports", lambda *args: ([], []))
    monkeypatch.setattr(
        example,
        "_control_summary",
        lambda name, *args, **kwargs: {
            "name": name,
            "metrics_by_effort": metrics,
            "decoded_candidates_match_intact": True,
            "decoded_candidates_match_intact_by_effort": {
                str(effort): True for effort in TEST_CHECKPOINTS
            },
            "decoded_candidate_match_query_count_by_effort": {
                str(effort): batch for effort in TEST_CHECKPOINTS
            },
            "trajectory_comparison": {"causally_null_at_measured_precision": True},
        },
    )

    result = example._evaluate(SimpleNamespace(), data, config, rows, SimpleNamespace())

    assert len(run_calls) == 5
    assert event_markers == [1.0, 1.0, 2.0, 3.0, 1.0]
    assert jit_options == [{"inline": False, "name": "example21_evaluation_arm"}]
    assert result["same_frozen_parameter_bytes"] is True
    assert result["primary_candidate_mode"] == "model_only"
    assert result["metrics_by_effort"] is adapted_metrics
    assert result["checkpoint_queries"] is adapted_checkpoint_queries
    assert result["task_local_adaptation"] is adaptation_evidence
    assert result["frozen_no_adaptation"]["role"] == (
        "diagnostic_control_not_primary_submission"
    )
    assert result["frozen_no_adaptation"]["metrics_by_effort"] is metrics
    assert result["frozen_no_adaptation"]["checkpoint_queries"] is checkpoint_queries
    assert result["frozen_no_adaptation"]["trajectory_role"] == (
        "frozen_no_adaptation_diagnostic"
    )
    assert (
        result["frozen_no_adaptation"]["aggregate_trajectory"]
        is result["aggregate_trajectory"]
    )
    assert (
        result["frozen_no_adaptation"]["query_trajectories"]
        is result["query_trajectories"]
    )
    assert result["physical_diagnostic_role"] == ("frozen_no_adaptation_diagnostic")
    assert all("ablation_slots" in call for call in run_calls)
    assert all("ablation_gates" in call for call in run_calls)
    assert not np.any(np.asarray(run_calls[0]["ablation_gates"]))
    assert not np.any(np.asarray(run_calls[1]["ablation_gates"]))
    ablation = run_calls[-1]
    gates = np.asarray(ablation["ablation_gates"])
    assert gates.shape == (time, batch)
    assert np.all(gates[stops, np.arange(batch)])
    assert result["determinism"]["same_control_capable_execution_path"] is True
    assert result["determinism"]["state_rms_tolerance"] == 1e-6
    assert result["determinism"]["metric_absolute_tolerance"] == 0.0
    assert result["determinism"]["repeat_intact_state_byte_identical"] is True
    assert result["determinism"]["repeat_intact_compact_logits_byte_identical"] is True
    assert result["determinism"]["repeat_intact_within_tolerance"] is True
    assert result["determinism"]["repeat_intact_metrics_exact"] is True
    assert result["determinism"]["slot_ablation_checkpoint_zero_byte_identical"] is True
    assert (
        result["determinism"]["slot_ablation_checkpoint_zero_decoded_candidates_exact"]
        is True
    )
    assert result["determinism"]["slot_ablation_checkpoint_zero_metrics_exact"] is True
    assert (
        result["determinism"]["slot_ablation_checkpoint_zero_within_tolerance"] is True
    )
    assert "repeat_intact" in result["controls"]
    assert result["controls"]["truncation"]["checkpoints"] == [0, 30, 60]
    execution = result["execution"]
    assert execution["arm_order"] == [
        "intact",
        "repeat_intact",
        "no_context",
        "shuffled_demonstrations",
        "slot_ablation",
    ]
    assert execution["selected_arm_driver"] == "brainstate.transform.jit"
    assert execution["jit_name"] == "example21_evaluation_arm"
    assert execution["jit_inline"] is False
    assert execution["sequential_separate_arms"] is True
    assert execution["repeat_intact_cached"] is False
    assert list(execution["wall_seconds_by_arm"]) == execution["arm_order"]
    assert all(value >= 0.0 for value in execution["wall_seconds_by_arm"].values())
    memory_diagnostics = result["associative_memory_diagnostics"]
    assert memory_diagnostics["available"] is True
    assert memory_diagnostics["depth_count"] == TEST_DEPTH_COUNT
    assert memory_diagnostics["query_count"] == batch


def test_evaluation_arm_jit_traces_once_and_matches_direct_dynamic_outputs(
    example, monkeypatch
):
    traces = 0
    selected = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
    slots = jnp.asarray([0, 1], dtype=jnp.int32)

    def packed(
        model,
        events,
        selected_indices,
        *,
        reset,
        advance_gates,
        ablation_slots,
        ablation_gates,
    ):
        nonlocal traces
        traces += 1
        assert reset is True
        assert ablation_slots.shape == (2,)
        base = jnp.take_along_axis(events[:, :, 0], selected_indices, axis=0)
        advance = jnp.take_along_axis(advance_gates, selected_indices, axis=0)
        gate = jnp.take_along_axis(ablation_gates, selected_indices, axis=0)
        return SimpleNamespace(
            compact_logits=base[..., None],
            spikes=advance[..., None].astype(jnp.float32),
            voltage=gate[..., None].astype(jnp.float32),
            feedforward_current=(base + advance)[..., None],
            recurrent_current=(base + gate)[..., None],
            memory_read=jnp.zeros((*base.shape, 0), dtype=jnp.float32),
            final_context_memory=jnp.zeros((base.shape[1], 0, 0), dtype=jnp.float32),
        )

    def expected(events, advances, gates):
        base = jnp.take_along_axis(events[:, :, 0], selected, axis=0)
        advance = jnp.take_along_axis(advances, selected, axis=0)
        gate = jnp.take_along_axis(gates, selected, axis=0)
        return (
            base[..., None],
            advance[..., None].astype(jnp.float32),
            gate[..., None].astype(jnp.float32),
            (base + advance)[..., None],
            (base + gate)[..., None],
            jnp.zeros((*base.shape, 0), dtype=jnp.float32),
            jnp.zeros((base.shape[1], 0, 0), dtype=jnp.float32),
            jnp.zeros((*base.shape, 0, 0), dtype=jnp.float32),
        )

    monkeypatch.setattr(example, "run_selected_packed_stream", packed)
    run = example._compile_evaluation_arm(SimpleNamespace(), selected, slots)
    advances = jnp.ones((4, 2), dtype=jnp.bool_)
    first_events = jnp.arange(8, dtype=jnp.float32).reshape(4, 2, 1)
    first_gates = jnp.zeros((4, 2), dtype=jnp.bool_)
    second_events = first_events + 10.0
    second_gates = jnp.ones((4, 2), dtype=jnp.bool_)

    first = run(first_events, advances, first_gates)
    second = run(second_events, advances, second_gates)
    jax.block_until_ready(second)

    assert traces == 1
    for actual, reference in zip(
        first, expected(first_events, advances, first_gates), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    for actual, reference in zip(
        second, expected(second_events, advances, second_gates), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_evaluation_arm_outer_jit_matches_real_selected_stream_exactly(example):
    config = ModelConfig(
        input_width=6,
        batch_size=1,
        neuron_count=64,
        recurrent_edges=64,
        max_latent_steps=4,
        readout_width=8,
        color_rank=1,
        seed=41,
        sparse_backend="jax_raw",
    )
    model = LatentWorkspaceModel(config)
    events = jnp.asarray(
        [
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    advances = jnp.ones((4, 1), dtype=jnp.bool_)
    selected = jnp.asarray([[0], [1], [2], [3]], dtype=jnp.int32)
    slots = jnp.asarray([0], dtype=jnp.int32)
    gates = jnp.zeros((4, 1), dtype=jnp.bool_)

    direct = run_selected_packed_stream(
        model,
        events,
        selected,
        reset=True,
        advance_gates=advances,
        ablation_slots=slots,
        ablation_gates=gates,
    )
    compiled = example._compile_evaluation_arm(model, selected, slots)
    actual = compiled(events, advances, gates)
    jax.block_until_ready(actual)
    expected = (
        direct.compact_logits,
        direct.spikes,
        direct.voltage,
        direct.feedforward_current,
        direct.recurrent_current,
        direct.memory_read,
        direct.final_context_memory,
        direct.context_memory,
    )

    for compiled_value, direct_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(compiled_value), np.asarray(direct_value)
        )


def test_evaluation_arm_outer_jit_matches_nonzero_associative_state_exactly(
    example, monkeypatch
):
    rows = RowEventConfig(max_demonstrations=2, max_grid_size=1)
    features = example.associative_memory_feature_indices(rows)
    config = ModelConfig(
        input_width=rows.input_width,
        batch_size=1,
        neuron_count=64,
        recurrent_edges=64,
        max_latent_steps=4,
        readout_width=8,
        color_rank=1,
        context_memory_width=2,
        memory_decay=1.0,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        memory_key_indices=features.key_indices,
        memory_value_indices=features.value_indices,
        seed=41,
        sparse_backend="jax_raw",
    )
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),))),
            ArcPair(ArcGrid(((3,),)), ArcGrid(((4,),))),
        ),
        test=(ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),))),),
        task_id="jit-memory",
    )
    encoded = encode_query_episode(task, 0, rows)
    events = jnp.asarray(
        np.concatenate(
            (
                encoded.events,
                np.zeros((1, rows.input_width), dtype=np.float32),
            ),
            axis=0,
        )[:, None, :]
    )
    advances = jnp.ones((events.shape[0], 1), dtype=jnp.bool_)
    selected = jnp.arange(events.shape[0], dtype=jnp.int32)[:, None]
    slots = jnp.asarray([0], dtype=jnp.int32)
    gates = jnp.zeros((events.shape[0], 1), dtype=jnp.bool_)
    model = LatentWorkspaceModel(config)

    direct = run_selected_packed_stream(
        model,
        events,
        selected,
        reset=True,
        advance_gates=advances,
        ablation_slots=slots,
        ablation_gates=gates,
    )
    original = example.run_selected_packed_stream
    traces = 0

    def counted(*args, **kwargs):
        nonlocal traces
        traces += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(example, "run_selected_packed_stream", counted)
    compiled = example._compile_evaluation_arm(model, selected, slots)
    cold_started = time.perf_counter()
    actual = compiled(events, advances, gates)
    jax.block_until_ready(actual)
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    warm_actual = compiled(events, advances, gates)
    jax.block_until_ready(warm_actual)
    warm_seconds = time.perf_counter() - warm_started
    expected = (
        direct.compact_logits,
        direct.spikes,
        direct.voltage,
        direct.feedforward_current,
        direct.recurrent_current,
        direct.memory_read,
        direct.final_context_memory,
        direct.context_memory,
    )

    assert np.count_nonzero(np.asarray(direct.final_context_memory)) > 0
    assert np.count_nonzero(np.asarray(direct.memory_read)) > 0
    assert traces == 1
    assert cold_seconds >= 1.5 * warm_seconds
    for compiled_value, direct_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(compiled_value), np.asarray(direct_value)
        )
    for warm_value, direct_value in zip(warm_actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(warm_value), np.asarray(direct_value))
    print(
        "example21_evaluation_arm_benchmark "
        f"cold_seconds={cold_seconds:.9f} warm_seconds={warm_seconds:.9f} "
        f"speedup={cold_seconds / warm_seconds:.3f} traces={traces}"
    )


def test_associative_diagnostics_require_per_query_shuffled_binding_evidence(example):
    workspace = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    memory_read = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    context_memory = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    intact = (workspace, memory_read, context_memory)
    metadata = ({"available": True}, {"available": True})
    no_context = (
        np.zeros_like(workspace),
        np.zeros_like(memory_read),
        np.zeros_like(context_memory),
    )
    shuffled_read = memory_read + 1.0
    shuffled = (
        workspace.copy(),
        shuffled_read,
        context_memory[:, :, ::-1].copy(),
    )
    controls = {
        "repeat_intact": (tuple(array.copy() for array in intact), metadata),
        "no_context": (no_context, metadata),
        "shuffled_demonstrations": (shuffled, metadata),
        "slot_ablation": (tuple(array.copy() for array in intact), metadata),
    }

    report = example._associative_evaluation_diagnostics(True, intact, controls)

    assert report["complete"] is True
    assert report["repeat_intact_exact"] is True
    assert report["no_context_memory_exactly_zero"] is True
    assert report["shuffled_pairing_sensitive_for_every_applicable_query"] is True
    shuffled_report = report["controls"]["shuffled_demonstrations"]
    assert shuffled_report["applicable_query_count"] == 2
    assert shuffled_report["context_memory_changed_applicable_query_count"] == 2
    assert (
        shuffled_report["memory_read_changed_at_any_depth_applicable_query_count"] == 2
    )
    assert len(report["intact_context_memory_sha256_by_query"]) == 2

    disconnected = dict(
        controls,
        shuffled_demonstrations=(
            tuple(array.copy() for array in intact),
            metadata,
        ),
    )
    failed = example._associative_evaluation_diagnostics(True, intact, disconnected)
    assert failed["complete"] is False
    assert failed["shuffled_pairing_sensitive_for_every_applicable_query"] is False


def test_associative_diagnostics_exclude_unavailable_queries_and_validate_arrays(
    example,
):
    workspace = np.zeros((2, 2, 3), dtype=np.float32)
    memory_read = np.zeros((2, 2, 2), dtype=np.float32)
    context_memory = np.ones((2, 2, 2), dtype=np.float32)
    intact = (workspace, memory_read, context_memory)
    available = ({"available": True}, {"available": False})
    shuffled_memory = context_memory.copy()
    shuffled_memory[0, 0, 0] = 2.0
    shuffled_read = memory_read.copy()
    shuffled_read[0, 0, 0] = 1.0
    controls = {
        "repeat_intact": (
            tuple(array.copy() for array in intact),
            ({"available": True}, {"available": True}),
        ),
        "no_context": (
            tuple(np.zeros_like(array) for array in intact),
            ({"available": True}, {"available": True}),
        ),
        "shuffled_demonstrations": (
            (workspace.copy(), shuffled_read, shuffled_memory),
            available,
        ),
        "slot_ablation": (
            tuple(array.copy() for array in intact),
            ({"available": True}, {"available": True}),
        ),
    }

    report = example._associative_evaluation_diagnostics(True, intact, controls)
    shuffled = report["controls"]["shuffled_demonstrations"]
    assert report["complete"] is True
    assert shuffled["applicable_query_count"] == 1
    assert shuffled["context_memory_changed_applicable_query_count"] == 1
    assert shuffled["memory_read_changed_at_any_depth_applicable_query_count"] == 1

    legacy = example._associative_evaluation_diagnostics(False, intact, {})
    assert legacy == {
        "available": False,
        "complete": True,
        "reason": "legacy_reservoir_has_no_associative_state",
    }

    malformed = dict(controls)
    malformed["slot_ablation"] = (
        (
            workspace,
            np.full_like(memory_read, np.nan),
            context_memory,
        ),
        ({"available": True}, {"available": True}),
    )
    with pytest.raises(ValueError, match="must be finite"):
        example._associative_evaluation_diagnostics(True, intact, malformed)


def test_qualification_separates_plumbing_structural_and_scientific_claims(example):
    fixture = smoke_loaded_dataset()
    fixture_origin = example._OriginTask("fixture", "fixture", fixture.tasks[0])
    fixture_data = example._ExperimentData(
        (fixture_origin,), (fixture_origin,), (fixture,), True
    )
    training = {
        "performed": True,
        "one_shared_model": True,
        "one_shared_optimizer_state": True,
        "terminal_supervision_only": True,
        "pp_prop_compiled": True,
        "compiler_report": {
            "available": True,
            "counts": {
                "hidden_groups": 1,
                "etrace_weights": 2,
                "excluded_weights": 4,
                "warnings": 4,
                "errors": 0,
            },
            "etrace_weights": [
                {"parameter": "ff_syn.comm.weight"},
                {"parameter": "rec_syn.comm.weight"},
            ],
            "excluded_weights": [
                {"parameter": "color_factor_head.weight"},
                {"parameter": "height_head.weight"},
                {"parameter": "readout_projection.weight"},
                {"parameter": "width_head.weight"},
            ],
        },
        "optimizer_updates_by_effort": {"30": 2, "60": 1},
        "losses": [1.0, 0.9, 0.8],
        "parameters_moved": True,
        "parameter_changes": {
            "color_factor_head.weight": {"changed": True, "l2_delta": 1.0},
            "ff_syn.comm.weight": {"changed": True, "l2_delta": 1.0},
            "height_head.weight": {"changed": True, "l2_delta": 1.0},
            "readout_projection.weight": {"changed": True, "l2_delta": 1.0},
            "rec_syn.comm.weight": {"changed": True, "l2_delta": 1.0},
            "width_head.weight": {"changed": True, "l2_delta": 1.0},
        },
    }
    evaluation = _evaluation_payload()
    model_report = {
        "neuron_count": 4096,
        "recurrent_edge_count": 4_194_304,
        "slot_count": 64,
        "parameter_count": 1,
        "component_types": {
            "neuron": "LIF",
            "feedforward_projection_wrapper": "AlignPostProj",
            "feedforward_projection": "Linear",
            "feedforward_synapse": "Expon",
            "feedforward_output": "CUBA",
            "recurrent_projection_wrapper": "AlignPostProj",
            "recurrent_projection": "SparseLinear",
            "recurrent_synapse": "Expon",
            "recurrent_output": "CUBA",
        },
    }
    gpu = {
        "platform": "gpu",
        "kind": "test",
        "gpu_runtime_safety": {
            "applicable": True,
            "status": "safe",
            "full_qualification_safe": True,
        },
    }
    legacy_config = example.ExperimentConfig(
        training_updates=3,
        context_memory_width=0,
        memory_coding="frozen",
        decoder_mode="legacy_cp",
    )
    legacy_memory_config = dataclasses.replace(
        legacy_config,
        context_memory_width=32,
    )

    plumbing = example._qualification(
        example.ExperimentConfig.smoke_config(decoder_mode="legacy_cp"),
        fixture_data,
        training,
        evaluation,
        {"platform": "cpu", "kind": "test"},
        dict(model_report, neuron_count=128, recurrent_edge_count=1024, slot_count=2),
    )
    assert plumbing["full_structural_qualification"] is False
    assert plumbing["full_scientific_qualification"] is False
    assert "plumbing-only" in " ".join(plumbing["reasons_not_scientific"])

    train_source = SimpleNamespace(role="train", name="ARC-AGI-1 training")
    eval_source = SimpleNamespace(role="evaluation", name="ARC-AGI-1 evaluation")
    loaded = (
        SimpleNamespace(
            manifest=SimpleNamespace(source=train_source, plumbing_only=False)
        ),
        SimpleNamespace(
            manifest=SimpleNamespace(source=eval_source, plumbing_only=False)
        ),
    )
    public_data = example._ExperimentData(
        (example._OriginTask("ARC-AGI-1 training", "train", fixture.tasks[0]),),
        (example._OriginTask("ARC-AGI-1 evaluation", "evaluation", fixture.tasks[1]),),
        loaded,
        False,
    )
    scientific = example._qualification(
        legacy_config,
        public_data,
        training,
        evaluation,
        gpu,
        model_report,
    )
    assert scientific["full_structural_qualification"] is False
    assert scientific["full_scientific_qualification"] is False
    assert scientific["reasons_not_structural"]

    missing_gpu_safety = example._qualification(
        legacy_config,
        public_data,
        training,
        evaluation,
        {"platform": "gpu", "kind": "test"},
        model_report,
    )
    assert missing_gpu_safety["structural_checks"]["gpu_runtime_resource_safe"] is False
    assert missing_gpu_safety["full_structural_qualification"] is False

    over_limit_gpu_safety = example._qualification(
        legacy_config,
        public_data,
        training,
        evaluation,
        {
            "platform": "gpu",
            "kind": "test",
            "gpu_runtime_safety": {
                "applicable": True,
                "status": "unsafe",
                "full_qualification_safe": False,
            },
        },
        model_report,
    )
    assert (
        over_limit_gpu_safety["structural_checks"]["gpu_runtime_resource_safe"] is False
    )
    assert over_limit_gpu_safety["full_structural_qualification"] is False

    memory_paths = {
        "memory_write_scale",
        "workspace_query_projection.weight",
        "memory_read_projection.weight",
    }
    memory_training = copy.deepcopy(training)
    memory_training["compiler_report"]["counts"]["etrace_weights"] = 5
    memory_training["compiler_report"]["etrace_weights"].extend(
        {"parameter": path} for path in sorted(memory_paths)
    )
    memory_training["compiler_report"]["diagnostics"] = [
        {
            "kind": "relation_included",
            "level": "info",
            "weight_path": path,
            "path_classification_by_hidden_state": {"associative_state": "all_direct"},
        }
        for path in sorted(memory_paths)
    ]
    memory_training["parameter_changes"].update(
        {path: {"changed": True, "l2_delta": 1.0} for path in memory_paths}
    )
    memory_evaluation = copy.deepcopy(evaluation)
    memory_evaluation["associative_memory_diagnostics"] = {
        "available": True,
        "complete": True,
        "query_count": 1,
        "depth_count": TEST_DEPTH_COUNT,
        "repeat_intact_exact": True,
        "no_context_memory_exactly_zero": True,
        "shuffled_pairing_sensitive_for_every_applicable_query": True,
    }
    memory_qualification = example._qualification(
        legacy_memory_config,
        public_data,
        memory_training,
        memory_evaluation,
        gpu,
        model_report,
    )
    assert memory_qualification["full_structural_qualification"] is False
    assert memory_qualification["full_scientific_qualification"] is False
    assert (
        memory_qualification["associative_capability_status"]
        == "associative_capability_gates_pending"
    )
    assert (
        "associative_capability_gates_pending"
        in (memory_qualification["reasons_not_scientific"])
    )

    row_training = copy.deepcopy(memory_training)
    row_training["compiler_report"]["etrace_weights"].extend(
        (
            {"parameter": "answer_row_head.weight"},
            {"parameter": "answer_shape_head.weight"},
        )
    )
    row_training["compiler_report"]["excluded_weights"] = []
    row_training["compiler_report"]["diagnostics"].extend(
        (
            {
                "kind": "relation_included",
                "level": "info",
                "weight_path": "answer_row_head.weight",
                "path_classification_by_hidden_state": {"answer_row": "all_direct"},
            },
            {
                "kind": "relation_included",
                "level": "info",
                "weight_path": "answer_shape_head.weight",
                "path_classification_by_hidden_state": {"answer_shape": "all_direct"},
            },
        )
    )
    for path in (
        "color_factor_head.weight",
        "height_head.weight",
        "readout_projection.weight",
        "width_head.weight",
    ):
        row_training["parameter_changes"][path] = {
            "changed": False,
            "l2_delta": 0.0,
        }
    row_training["parameter_changes"].update(
        {
            "answer_row_head.weight": {"changed": True, "l2_delta": 1.0},
            "answer_shape_head.weight": {"changed": True, "l2_delta": 1.0},
        }
    )
    row_qualification = example._qualification(
        dataclasses.replace(
            legacy_memory_config,
            training_updates=3,
            decoder_mode="row_refinement",
        ),
        public_data,
        row_training,
        memory_evaluation,
        gpu,
        model_report,
    )
    assert row_qualification["structural_checks"]["pp_prop_compiler_routes"] is True
    assert row_qualification["structural_checks"]["row_routes_all_direct"] is True
    assert (
        row_qualification["structural_checks"]["complete_primary_evaluation"] is False
    )
    assert row_qualification["structural_checks"]["complete_frozen_diagnostics"] is True
    assert (
        row_qualification["scientific_checks"][
            "all_active_parameter_groups_moved_with_finite_delta"
        ]
        is True
    )

    gated_paths = {
        "answer_row_event_head.weight",
        "answer_row_carrier_head.weight",
        "row_carrier_gate_head.weight",
        "answer_shape_head.weight",
    }
    gated_training = copy.deepcopy(memory_training)
    gated_training["compiler_report"]["etrace_weights"].extend(
        {"parameter": path} for path in sorted(gated_paths)
    )
    gated_training["compiler_report"]["excluded_weights"] = []
    gated_training["compiler_report"]["diagnostics"].extend(
        {
            "kind": "relation_included",
            "level": "info",
            "weight_path": path,
            "path_classification_by_hidden_state": {
                "answer_row" if "row" in path else "answer_shape": "all_direct"
            },
        }
        for path in sorted(gated_paths)
    )
    gated_training["parameter_changes"].update(
        {path: {"changed": True, "l2_delta": 1.0} for path in gated_paths}
    )
    for path in (
        "color_factor_head.weight",
        "height_head.weight",
        "readout_projection.weight",
        "width_head.weight",
    ):
        gated_training["parameter_changes"][path] = {
            "changed": False,
            "l2_delta": 0.0,
        }
    gated_qualification = example._qualification(
        dataclasses.replace(
            legacy_memory_config,
            training_updates=3,
            decoder_mode="row_refinement",
            row_head_carrier_gate=True,
        ),
        public_data,
        gated_training,
        memory_evaluation,
        gpu,
        model_report,
    )
    assert (
        gated_qualification["structural_checks"]["pp_prop_compiler_routes"] is True
    )
    assert gated_qualification["structural_checks"]["row_routes_all_direct"] is True

    mixed_memory_training = copy.deepcopy(memory_training)
    mixed_memory_training["compiler_report"]["diagnostics"][0][
        "path_classification_by_hidden_state"
    ] = {"associative_state": "mixed"}
    mixed_memory = example._qualification(
        legacy_memory_config,
        public_data,
        mixed_memory_training,
        memory_evaluation,
        gpu,
        model_report,
    )
    assert mixed_memory["full_structural_qualification"] is False
    assert mixed_memory["structural_checks"]["associative_routes_all_direct"] is False
    assert "all_direct" in " ".join(mixed_memory["reasons_not_structural"])

    incomplete_memory_evaluation = copy.deepcopy(memory_evaluation)
    incomplete_memory_evaluation["associative_memory_diagnostics"]["complete"] = False
    incomplete_memory = example._qualification(
        legacy_memory_config,
        public_data,
        memory_training,
        incomplete_memory_evaluation,
        gpu,
        model_report,
    )
    assert incomplete_memory["full_structural_qualification"] is False
    assert (
        incomplete_memory["structural_checks"]["associative_diagnostics_complete"]
        is False
    )

    no_compile = dict(training, pp_prop_compiled=False)
    failed = example._qualification(
        legacy_config,
        public_data,
        no_compile,
        evaluation,
        gpu,
        model_report,
    )
    assert failed["full_structural_qualification"] is False
    assert "compilation" in " ".join(failed["reasons_not_scientific"])

    cpu = example._qualification(
        legacy_config,
        public_data,
        training,
        evaluation,
        {"platform": "cpu", "kind": "test"},
        model_report,
    )
    assert cpu["full_structural_qualification"] is False

    bad_compiler = dict(
        training,
        compiler_report={
            "available": True,
            "counts": {"hidden_groups": 1, "etrace_weights": 2, "errors": 1},
        },
    )
    compiler_failure = example._qualification(
        legacy_config,
        public_data,
        bad_compiler,
        evaluation,
        gpu,
        model_report,
    )
    assert compiler_failure["full_structural_qualification"] is False

    nondeterministic = _evaluation_payload()
    nondeterministic["determinism"] = dict(
        nondeterministic["determinism"],
        repeat_intact_within_tolerance=False,
    )
    repeat_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        nondeterministic,
        gpu,
        model_report,
    )
    assert repeat_failure["full_structural_qualification"] is False
    assert "repeat" in " ".join(repeat_failure["reasons_not_structural"])

    missing_numeric = _evaluation_payload()
    missing_numeric["determinism"].pop("repeat_intact_numeric_evidence")
    missing_numeric_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        missing_numeric,
        gpu,
        model_report,
    )
    assert missing_numeric_failure["full_structural_qualification"] is False

    corrupted_numeric = _evaluation_payload()
    corrupted_numeric["determinism"]["repeat_intact_numeric_evidence"]["maximum_rms"][
        "compact_logits"
    ] = 2e-6
    corrupted_numeric_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        corrupted_numeric,
        gpu,
        model_report,
    )
    assert corrupted_numeric_failure["full_structural_qualification"] is False

    wrong_declared_tolerance = _evaluation_payload()
    wrong_declared_tolerance["determinism"]["metric_absolute_tolerance"] = 1e-6
    tolerance_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        wrong_declared_tolerance,
        gpu,
        model_report,
    )
    assert tolerance_failure["full_structural_qualification"] is False

    ablation_candidate_mismatch = _evaluation_payload()
    ablation_candidate_mismatch["controls"]["slot_ablation"][
        "decoded_candidates_match_intact_by_effort"
    ]["0"] = False
    ablation_candidate_mismatch["controls"]["slot_ablation"][
        "decoded_candidate_match_query_count_by_effort"
    ]["0"] = 0
    candidate_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        ablation_candidate_mismatch,
        gpu,
        model_report,
    )
    assert candidate_failure["full_structural_qualification"] is False

    ablation_metric_mismatch = _evaluation_payload()
    slot_control = ablation_metric_mismatch["controls"]["slot_ablation"]
    ablation_metric_mismatch["controls"]["slot_ablation"] = dict(
        slot_control,
        metrics_by_effort={
            **slot_control["metrics_by_effort"],
            "0": _metric(1.0),
        },
    )
    metric_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        ablation_metric_mismatch,
        gpu,
        model_report,
    )
    assert metric_failure["full_structural_qualification"] is False

    missing_readout = dict(
        training,
        parameter_changes={
            path: change
            for path, change in training["parameter_changes"].items()
            if path != "readout_projection.weight"
        },
    )
    movement_failure = example._qualification(
        legacy_config,
        public_data,
        missing_readout,
        evaluation,
        gpu,
        model_report,
    )
    assert movement_failure["full_structural_qualification"] is False
    assert movement_failure["full_scientific_qualification"] is False
    assert "every active parameter group" in " ".join(
        movement_failure["reasons_not_scientific"]
    )

    wrong_components = dict(
        model_report,
        component_types=dict(model_report["component_types"], neuron="NotLIF"),
    )
    component_failure = example._qualification(
        legacy_config,
        public_data,
        training,
        evaluation,
        gpu,
        wrong_components,
    )
    assert component_failure["full_structural_qualification"] is False
    assert "component types" in " ".join(component_failure["reasons_not_structural"])


def test_report_and_agg_plot_expose_exact_metrics_controls_and_claim_boundary(
    example, tmp_path
):
    result = {
        "claim_boundary": example.CLAIM_BOUNDARY,
        "configuration": {"seed": 7},
        "device": {"platform": "cpu", "kind": "test"},
        "model": {"neuron_count": 128, "recurrent_edge_count": 1024, "slot_count": 2},
        "training": {"optimizer_updates_by_effort": {"30": 1, "60": 1}},
        "evaluation": _evaluation_payload(),
        "data_summary": {
            "manifest_sha256": "abc",
            "task_counts_by_role": {"fixture": 2},
            "query_counts_by_role": {"fixture": 3},
            "excluded_task_count": 0,
            "rejected_task_count": 0,
        },
        "runtime_seconds": 1.25,
        "qualification": {
            "full_structural_qualification": False,
            "full_scientific_qualification": False,
            "reasons_not_scientific": ["fixture"],
        },
    }

    report = example._render_report(result)
    plot_path = tmp_path / "plot.png"
    example._plot(result, plot_path)

    assert "query pass@1" in report
    assert "shuffled_demonstrations" in report
    assert "not a reproduction" in report
    assert "Seed: 7" in report
    assert "Runtime: 1.250" in report
    assert "Voltage L2" in report
    assert "feature axis for logits; neuron axis for physical state" in report
    assert "Repeat numeric noise: queries=1; steps=61" in report
    assert "step-60 state deltas" in report
    assert "step-32 state deltas" not in report
    assert "maximum RMS=" in report
    assert "per_step_query_rms" not in report
    assert "no routes were trained in this run" in report
    assert "plain routes are trained" not in report

    result["training"]["performed"] = True
    trained_report = example._render_report(result)
    assert "plain routes received exact current-window gradients in this run" in (
        trained_report
    )
    assert plot_path.read_bytes().startswith(b"\x89PNG")

    memory_result = copy.deepcopy(result)
    memory_result["model"].update(
        {
            "reasoning_mode": "associative_workspace",
            "context_memory_width": 32,
            "memory_decay": 1.0,
            "raw_key_feature_width": 424,
            "raw_value_feature_width": 424,
            "context_memory_bytes_per_example": 4096,
            "context_memory_bytes_training_batch": 4096,
            "context_memory_bytes_evaluation_batch": 1_716_224,
            "associative_memory_implementation": {
                "mode": "associative_workspace",
                "key_basis_sha256": "key-sha",
                "value_basis_sha256": "value-sha",
            },
        }
    )
    memory_report = example._render_report(memory_result)
    assert "associative_workspace" in memory_report
    assert "raw key/value widths=424/424" in memory_report
    assert "4096/4096/1716224 bytes" in memory_report
    assert "key-sha" in memory_report


def test_run_experiment_writes_complete_artifact_set(example, monkeypatch, tmp_path):
    config = example.ExperimentConfig.smoke_config(
        output_dir=tmp_path,
        decoder_mode="legacy_cp",
    )
    fixture = smoke_loaded_dataset()
    origin = example._OriginTask("fixture", "fixture", fixture.tasks[0])
    data = example._ExperimentData((origin,), (origin,), (fixture,), True)
    model = SimpleNamespace(
        neuron_count=128,
        recurrent_edge_count=1024,
        slot_count=2,
        neuron_typing_report=lambda: {"mode": "none"},
        config=SimpleNamespace(
            batch_size=1,
            decoder_mode="legacy_cp",
            refinement_steps=60,
            training_output_width=340,
            checkpoint_output_width=340,
            compact_output_width=340,
            color_rank=4,
        ),
        neu=SimpleNamespace(),
        ff_syn=SimpleNamespace(
            comm=SimpleNamespace(), syn=SimpleNamespace(), out=SimpleNamespace()
        ),
        rec_syn=SimpleNamespace(
            comm=SimpleNamespace(), syn=SimpleNamespace(), out=SimpleNamespace()
        ),
    )
    training = {
        "performed": True,
        "pp_prop_compiled": True,
        "optimizer_updates_by_effort": {"30": 1, "60": 1},
    }
    evaluation = _evaluation_payload()
    monkeypatch.setattr(
        example,
        "require_pre_device_gpu_environment",
        lambda environment: pytest.fail(
            "CPU runs must not apply the GPU environment gate"
        ),
    )
    monkeypatch.setattr(
        example,
        "_resolve_device",
        lambda name: (SimpleNamespace(), {"platform": "cpu", "kind": "test", "id": 0}),
    )
    monkeypatch.setattr(example, "_load_data", lambda config: data)
    monkeypatch.setattr(example, "_make_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(example, "_train_model", lambda *args: training)
    monkeypatch.setattr(example, "_evaluate", lambda *args: evaluation)
    monkeypatch.setattr(
        example,
        "_device_memory_stats",
        lambda device: {"peak_bytes_in_use": 123456},
    )
    monkeypatch.setattr(
        example, "parameter_snapshot", lambda model: {"p": np.ones((3,))}
    )
    monkeypatch.setattr(
        example,
        "_model_memory_report",
        lambda model: {
            "mode": "legacy_reservoir",
            "memory_width": 0,
            "key_feature_width": 0,
            "value_feature_width": 0,
        },
    )
    monkeypatch.setattr(
        example,
        "_qualification",
        lambda *args: {
            "full_structural_qualification": False,
            "full_scientific_qualification": False,
            "reasons_not_scientific": ["fixture"],
        },
    )
    monkeypatch.setattr(example, "_plot", lambda result, path: path.write_bytes(b"png"))

    result = example.run_experiment(config)

    assert json.loads((tmp_path / "result.json").read_text())["schema_version"] == 2
    artifact_manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert artifact_manifest["schema_version"] == 2
    assert set(artifact_manifest["artifacts"]) == {
        "data_manifest",
        "result",
        "report",
        "figure",
    }
    assert json.loads((tmp_path / "data_manifest.json").read_text())[0]["plumbing_only"]
    assert "Claim boundary" in (tmp_path / "report.txt").read_text()
    assert (tmp_path / "latent_reasoning.png").read_bytes() == b"png"
    assert set(result["artifacts"]) == {
        "data_manifest",
        "result",
        "report",
        "figure",
        "manifest",
    }
    assert result["device"]["memory_stats"] == {"peak_bytes_in_use": 123456}
    assert result["device"]["memory_stats_capture"] == "after training and evaluation"
    assert result["device"]["pre_device_gpu_environment"] == {
        "applicable": False,
        "status": "not_applicable_cpu",
    }
    assert result["device"]["gpu_runtime_safety"] == {
        "applicable": False,
        "status": "not_applicable_cpu",
        "full_qualification_safe": False,
    }
    assert result["software"]["xla_python_client_mem_fraction"] is None
    assert result["software"]["pre_device_gpu_environment"] == {
        "applicable": False,
        "status": "not_applicable_cpu",
    }
    assert result["model"]["reasoning_mode"] == "legacy_reservoir"
    assert result["model"]["context_memory_bytes_per_example"] == 0
    assert result["model"]["associative_memory_implementation"] == {
        "mode": "legacy_reservoir",
        "memory_width": 0,
        "key_feature_width": 0,
        "value_feature_width": 0,
    }


def test_main_prints_report_and_returns_result(example, monkeypatch, capsys, tmp_path):
    result = {
        "device": {"platform": "cpu", "kind": "test"},
        "model": {"neuron_count": 128, "recurrent_edge_count": 1024, "slot_count": 2},
        "training": {"optimizer_updates_by_effort": {"30": 1, "60": 1}},
        "evaluation": _evaluation_payload(),
        "qualification": {
            "full_structural_qualification": False,
            "full_scientific_qualification": False,
            "reasons_not_scientific": ["fixture"],
        },
    }
    monkeypatch.setattr(example, "run_experiment", lambda config: result)

    returned = example.main(
        ["--smoke", "--device", "cpu", "--output-dir", str(tmp_path)]
    )

    assert returned is result
    assert "Example 21" in capsys.readouterr().out


def test_repeated_model_execution_is_lowered_through_brainstate_transforms(example):
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    training_loops = [
        node
        for node in ast.walk(functions["_train_model"])
        if isinstance(node, (ast.For, ast.While))
    ]
    assert training_loops == []
    # The chunk driver may loop over data-staging steps, but the model itself
    # must stay inside the compiled scan: no gradient call may appear there.
    staging = ast.dump(functions["_train_chunks"])
    assert "etrace_grad" not in staging
    assert "for_loop" not in staging
    calls = [
        node
        for node in ast.walk(functions["_train_model"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "for_loop"
    ]
    assert calls
    assert "run_selected_packed_stream" in EXAMPLE.read_text(encoding="utf-8")


def test_chunk_size_must_divide_the_update_budget(example):
    with pytest.raises(ValueError, match="training_chunk_size must divide"):
        example.ExperimentConfig(training_updates=96, training_chunk_size=7)
    assert (
        example.ExperimentConfig(
            training_updates=96, training_chunk_size=0
        ).training_chunk_size
        == 0
    )
    assert (
        example.ExperimentConfig(
            training_updates=96, training_chunk_size=32
        ).training_chunk_size
        == 32
    )
    example.ExperimentConfig(
        structural_only=True, training_updates=0, training_chunk_size=32
    )


def test_automatic_chunk_size_is_bounded_and_keeps_the_schedule_complete(example):
    full = example.ExperimentConfig(training_updates=260, training_chunk_size=0)
    smoke = example.ExperimentConfig(training_updates=13, training_chunk_size=0)

    assert example._resolved_training_chunk_size(full) == 5
    assert example._resolved_training_chunk_size(smoke) == 1
    assert full.training_updates % example._resolved_training_chunk_size(full) == 0
    assert smoke.training_updates % example._resolved_training_chunk_size(smoke) == 0


def test_chunking_does_not_change_the_prepared_schedule(example):
    whole = example.ExperimentConfig.smoke_config()
    split = dataclasses.replace(whole, training_chunk_size=1)
    data = example._load_data(whole)
    rows = example._row_config(whole)

    reference = _materialize_training(example, data, whole, rows)
    chunked = _materialize_training(example, data, split, rows)

    assert len(list(example._training_chunks(data, split, rows))) == 3
    for field in _TRAINING_ARRAY_FIELDS:
        assert np.array_equal(getattr(reference, field), getattr(chunked, field)), field
    for field in _TRAINING_METADATA_FIELDS:
        assert getattr(reference, field) == getattr(chunked, field), field


def test_training_sequence_length_covers_every_enabled_orientation(example):
    """The dataset bound covers every fold before static compilation."""

    def grid(height, width, color):
        return ArcGrid(tuple(tuple(color for _ in range(width)) for _ in range(height)))

    task = ArcTask(
        train=(
            ArcPair(grid(1, 7, 1), grid(2, 6, 2)),
            ArcPair(grid(3, 5, 3), grid(1, 4, 4)),
            ArcPair(grid(2, 8, 5), grid(3, 7, 6)),
        ),
        test=(ArcPair(grid(1, 1, 7), None),),
        task_id="rectangular-horizon",
    )
    origin = example._OriginTask("ARC-AGI-1 training", "train", task)
    data = example._ExperimentData((origin,), (origin,), (), False)
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(), max_demonstrations=10
    )
    rows = example._row_config(config)

    def transpose_grid(value):
        return ArcGrid(tuple(zip(*value.cells)))

    def transpose_task(value):
        def pair(item):
            return ArcPair(
                transpose_grid(item.input),
                None if item.output is None else transpose_grid(item.output),
            )

        return ArcTask(
            tuple(pair(item) for item in value.train),
            tuple(pair(item) for item in value.test),
            task_id=value.task_id,
        )

    required = []
    for oriented in (task, transpose_task(task)):
        for episode in leave_one_demonstration_out_episodes(oriented):
            encoded = encode_arc_query_episode(episode, rows)
            required.append(
                int(np.count_nonzero(example._packed_advances(encoded, config, rows)))
            )

    horizon = example._training_sequence_length(data, config)

    assert horizon == max(required) + 30
    assert (
        horizon
        < (config.max_demonstrations + 1) * config.max_grid_size + config.latent_steps
    )
    assert horizon < 390


def test_compact_training_horizon_preserves_the_complete_semantic_prefix(example):
    """Short static rows discard only the guaranteed all-zero suffix."""
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(decoder_mode="row_refinement"),
        max_demonstrations=10,
    )
    data = example._load_data(config)
    rows = example._row_config(config)
    horizon = example._training_sequence_length(data, config)
    origin = data.training[0]

    legacy = example._training_row(
        origin,
        config,
        rows,
        brainstate.random.RandomState(17),
        effort=60,
        plumbing_only=data.plumbing_only,
    )
    compact = example._training_row(
        origin,
        config,
        rows,
        brainstate.random.RandomState(17),
        effort=60,
        plumbing_only=data.plumbing_only,
        sequence_length=horizon,
    )

    assert compact["events"].shape[0] == horizon < legacy["events"].shape[0]
    np.testing.assert_array_equal(compact["events"], legacy["events"][:horizon])
    np.testing.assert_array_equal(compact["advances"], legacy["advances"][:horizon])
    np.testing.assert_array_equal(compact["masks"], legacy["masks"][:horizon])
    assert not np.any(legacy["advances"][horizon:])
    assert not np.any(legacy["events"][horizon:])
    assert not np.any(legacy["masks"][horizon:])

    chunks = list(example._training_chunks(data, config, rows))
    assert chunks[0].events.shape[1] == horizon


def test_compact_horizon_reproduces_legacy_cpu_training_numerically(
    example, monkeypatch
):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(), training_chunk_size=1
    )
    data = example._load_data(config)
    rows = example._row_config(config)
    compact_length = example._training_sequence_length(data, config)
    legacy_length = rows.max_events + config.latent_steps

    def train(sequence_length):
        monkeypatch.setattr(
            example,
            "_training_sequence_length",
            lambda _data, _config: sequence_length,
        )
        model = example._make_model(
            config, rows, batch_size=1, device=jax.devices("cpu")[0]
        )
        report = example._train_model(
            model, example._training_chunks(data, config, rows), config
        )
        return report, example.parameter_snapshot(model)

    legacy, legacy_parameters = train(legacy_length)
    compact, compact_parameters = train(compact_length)

    np.testing.assert_allclose(
        compact["losses"], legacy["losses"], rtol=1e-6, atol=1e-6
    )
    assert compact["effort_schedule"] == legacy["effort_schedule"]
    assert compact["training_samples"] == legacy["training_samples"]
    for compact_leaf, legacy_leaf in zip(
        jax.tree.leaves(compact_parameters),
        jax.tree.leaves(legacy_parameters),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(compact_leaf),
            np.asarray(legacy_leaf),
            rtol=1e-6,
            atol=1e-6,
        )


def test_training_chunk_prefetch_is_ordered_and_at_most_one_ahead(example):
    produced: list[int] = []

    def source():
        for value in range(4):
            produced.append(value)
            yield value

    chunks = example._prefetched_training_chunks(source())
    assert next(chunks) == 0
    deadline = time.monotonic() + 1.0
    while len(produced) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    time.sleep(0.02)
    assert produced == [0, 1]

    assert next(chunks) == 1
    deadline = time.monotonic() + 1.0
    while len(produced) < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    time.sleep(0.02)
    assert produced == [0, 1, 2]
    assert list(chunks) == [2, 3]


def test_training_chunk_prefetch_propagates_the_producer_exception(example):
    def source():
        yield "ready"
        raise RuntimeError("episode encoding failed")

    chunks = example._prefetched_training_chunks(source())
    assert next(chunks) == "ready"
    with pytest.raises(RuntimeError, match="episode encoding failed"):
        next(chunks)


def test_profiled_chunk_execution_reports_pipeline_timing(example):
    profile = example._TrainingProfile()
    chunk = example._TrainingTensors(
        events=np.zeros((1, 2, 1, 2), dtype=np.float32),
        advances=np.zeros((1, 2, 1), dtype=np.bool_),
        heights=np.zeros((1, 1), dtype=np.int32),
        widths=np.zeros((1, 1), dtype=np.int32),
        colors=np.zeros((1, 1, 30, 30), dtype=np.int32),
        masks=np.zeros((1, 2), dtype=np.float32),
        efforts=np.asarray([30], dtype=np.int32),
        task_fingerprints=("task",),
        base_task_fingerprints=("base",),
        source_names=("source",),
        held_out_demonstration_indices=(0,),
    )

    losses, schedule = example._train_chunks(
        example._prefetched_training_chunks([chunk], profile),
        lambda *_args: np.asarray([3.0], dtype=np.float32),
        profile=profile,
    )

    assert losses == [3.0]
    assert schedule.efforts == (30,)
    report = profile.to_dict()
    assert report["chunk_count"] == 1
    assert report["producer_encoding_seconds"] >= 0.0
    assert report["consumer_wait_seconds"] >= 0.0
    assert report["host_to_device_staging_seconds"] >= 0.0
    assert report["first_call_compilation_seconds"] >= 0.0
    assert report["first_call_device_compute_seconds"] >= 0.0
    assert report["host_result_copy_seconds"] >= 0.0


def test_training_episode_workers_restore_order_and_join_on_failure(
    example, monkeypatch
):
    started = threading.Event()
    finished = threading.Event()

    def materialize(job):
        started.set()
        if job.ordinal == 1:
            raise RuntimeError("episode worker failed")
        time.sleep(0.01)
        finished.set()
        return job.ordinal

    monkeypatch.setattr(example, "_materialize_training_episode", materialize)
    jobs = [type("Job", (), {"ordinal": index})() for index in range(8)]
    with pytest.raises(RuntimeError, match="episode worker failed"):
        list(example._ordered_training_episodes(jobs, 2))
    assert started.is_set()
    assert finished.is_set()
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("example21-training-row")
    ]


def test_cached_base_training_task_is_target_free_and_identity_stable(example):
    config = example.ExperimentConfig.smoke_config()
    origin = example._load_data(config).training[0]
    example._cached_base_training_task.cache_clear()

    first, first_fingerprint = example._cached_base_training_task(origin.task)
    second, second_fingerprint = example._cached_base_training_task(origin.task)

    assert first is second
    assert first_fingerprint == second_fingerprint
    assert all(pair.output is not None for pair in first.train)
    assert all(pair.output is None for pair in first.test)
    assert example._cached_base_training_task.cache_info().hits == 1


def test_training_workers_preserve_rows_and_metadata(example):
    serial = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
        training_workers=1,
        training_chunk_size=1,
    )
    parallel = dataclasses.replace(serial, training_workers=4)
    data = example._load_data(serial)
    rows = example._row_config(serial)

    serial_chunks = list(example._training_chunks(data, serial, rows))
    parallel_chunks = list(example._training_chunks(data, parallel, rows))
    assert len(serial_chunks) == len(parallel_chunks)
    for left, right in zip(serial_chunks, parallel_chunks, strict=True):
        for field in _TRAINING_ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        for field in _TRAINING_METADATA_FIELDS:
            assert getattr(left, field) == getattr(right, field), field


def test_batched_training_chunks_match_scalar_oracle_and_losses(
    example, monkeypatch
):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
        training_updates=10,
        training_batch_size=2,
        training_chunk_size=5,
        training_workers=2,
    )
    loaded = example._load_data(config)
    data = dataclasses.replace(loaded, plumbing_only=False)
    rows = example._row_config(config)

    optimized = list(example._training_chunks(data, config, rows))
    monkeypatch.setattr(
        example,
        "_encode_arc_query_episodes_batched",
        lambda episodes, row_config: tuple(
            example.encode_arc_query_episode(episode, row_config)
            for episode in episodes
        ),
    )
    oracle = list(example._training_chunks(data, config, rows))

    assert len(optimized) == len(oracle) == 2
    for left, right in zip(optimized, oracle, strict=True):
        for field in _TRAINING_ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        for field in _TRAINING_METADATA_FIELDS:
            assert getattr(left, field) == getattr(right, field), field

    def fake_train(
        events, advances, heights, widths, colors, masks
    ) -> np.ndarray:
        arrays = (events, advances, heights, widths, colors, masks)
        return np.asarray(
            [
                sum(float(np.asarray(array[index]).sum()) for array in arrays)
                for index in range(int(np.asarray(events).shape[0]))
            ],
            dtype=np.float32,
        )

    optimized_report = example._train_chunks(optimized, fake_train)[0]
    oracle_report = example._train_chunks(oracle, fake_train)[0]
    np.testing.assert_allclose(optimized_report, oracle_report, rtol=0.0, atol=1e-6)


def _adaptation_bank_fixture(example, tasks=4):
    from examples.pp_prop.latent_workspace_arc_adaptation import (
        build_arc_target_free_task_bank,
    )

    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    data = example._load_data(config)
    pool = [example._without_official_test_targets(item.task) for item in data.training]
    while len(pool) < tasks:
        pool = pool + pool
    return build_arc_target_free_task_bank(tuple(pool[:tasks]), rows), rows


def test_a_task_slice_keeps_every_trailing_bank_shape(example):
    bank, _ = _adaptation_bank_fixture(example, tasks=4)

    sliced = example._bank_task_slice(bank, 1, 3)

    for whole, part in zip(jax.tree.leaves(bank), jax.tree.leaves(sliced), strict=True):
        assert np.shape(part)[0] == 2
        assert np.shape(part)[1:] == np.shape(whole)[1:]
        np.testing.assert_array_equal(np.asarray(part), np.asarray(whole)[1:3])


def test_task_groups_reproduce_the_whole_bank_result(example):
    bank, _ = _adaptation_bank_fixture(example, tasks=4)
    calls: list[int] = []

    def runner(sub_bank):
        count = int(np.asarray(sub_bank.query_valid).shape[0])
        calls.append(count)
        ordinals = jnp.asarray(sub_bank.task_ordinals, dtype=jnp.float32)
        return example.ArcTaskBankAdaptationResult(
            fold_losses=jnp.broadcast_to(ordinals[:, None], (count, 2)),
            fold_applied=jnp.ones((count, 2), dtype=jnp.bool_),
            checkpoint_outputs=jnp.broadcast_to(
                ordinals[:, None, None, None], (count, 1, 3, 4)
            ),
            checkpoint_recorded=jnp.ones((count, 1, 3), dtype=jnp.bool_),
            query_valid=jnp.asarray(sub_bank.query_valid),
        )

    whole = runner(bank)
    calls.clear()
    grouped = example._run_adaptation_in_task_groups(runner, bank, 2)

    assert calls == [2, 2]
    for left, right in zip(
        jax.tree.leaves(whole), jax.tree.leaves(grouped), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_group_covering_the_bank_makes_one_call(example):
    bank, _ = _adaptation_bank_fixture(example, tasks=3)
    calls: list[int] = []

    def runner(sub_bank):
        calls.append(int(np.asarray(sub_bank.query_valid).shape[0]))
        return "whole"

    assert example._run_adaptation_in_task_groups(runner, bank, 0) == "whole"
    assert example._run_adaptation_in_task_groups(runner, bank, 99) == "whole"
    assert calls == [3, 3]


def test_a_negative_task_group_is_rejected(example):
    with pytest.raises(ValueError, match="adaptation_task_group"):
        example.ExperimentConfig(adaptation_task_group=-1)


def test_a_parameter_checkpoint_round_trips_every_leaf(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    source = example._make_model(config, rows, batch_size=1, device=device)
    target = example._make_model(config, rows, batch_size=1, device=device)
    for state in target.states(example.brainstate.ParamState).values():
        state.value = jax.tree.map(lambda leaf: leaf + 1.0, state.value)
    path = tmp_path / "parameters.npz"

    written = example._write_parameter_checkpoint(source, path)
    restored = example._read_parameter_checkpoint(target, path)

    assert written == restored
    for left, right in zip(
        jax.tree.leaves(example.parameter_snapshot(source)),
        jax.tree.leaves(example.parameter_snapshot(target)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_checkpoint_from_another_scale_is_rejected(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    wider = dataclasses.replace(config, neuron_count=config.neuron_count * 2)
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    narrow = example._make_model(config, rows, batch_size=1, device=device)
    broad = example._make_model(wider, rows, batch_size=1, device=device)
    path = tmp_path / "parameters.npz"
    example._write_parameter_checkpoint(narrow, path)

    with pytest.raises(ValueError, match="parameter checkpoint"):
        example._read_parameter_checkpoint(broad, path)


def test_memory_read_transform_participates_in_checkpoint_compatibility(
    example, tmp_path
):
    linear_config = example.ExperimentConfig.smoke_config(
        memory_read_transform="linear"
    )
    gated_config = example.ExperimentConfig.smoke_config(
        memory_read_transform="gated"
    )
    rows = example._row_config(linear_config)
    device = jax.devices("cpu")[0]
    linear = example._make_model(linear_config, rows, batch_size=1, device=device)
    gated = example._make_model(gated_config, rows, batch_size=1, device=device)
    path = tmp_path / "linear-parameters.npz"
    example._write_parameter_checkpoint(linear, path)

    with pytest.raises(ValueError, match="parameter checkpoint"):
        example._read_parameter_checkpoint(gated, path)


def test_memory_read_interval_participates_in_checkpoint_compatibility(
    example, tmp_path
):
    every_tick_config = example.ExperimentConfig.smoke_config(memory_read_interval=1)
    periodic_config = example.ExperimentConfig.smoke_config(memory_read_interval=4)
    rows = example._row_config(every_tick_config)
    device = jax.devices("cpu")[0]
    every_tick = example._make_model(
        every_tick_config, rows, batch_size=1, device=device
    )
    periodic = example._make_model(periodic_config, rows, batch_size=1, device=device)
    path = tmp_path / "every-tick-parameters.npz"
    example._write_parameter_checkpoint(every_tick, path)

    with pytest.raises(ValueError, match="model architecture"):
        example._read_parameter_checkpoint(periodic, path)


def test_restoring_a_checkpoint_permits_a_zero_update_budget(example, tmp_path):
    path = tmp_path / "parameters.npz"
    path.write_bytes(b"")
    restored = example.ExperimentConfig(training_updates=0, parameter_checkpoint=path)
    assert restored.training_updates == 0
    assert restored.to_dict()["parameter_checkpoint"] == str(path)

    with pytest.raises(ValueError, match="training_updates"):
        example.ExperimentConfig(
            training_updates=0, parameter_checkpoint=tmp_path / "absent.npz"
        )


def test_a_restored_report_records_no_optimizer_update(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    model = example._make_model(config, rows, batch_size=1, device=device)
    path = tmp_path / "parameters.npz"
    digest = example._write_parameter_checkpoint(model, path)
    restored = dataclasses.replace(config, parameter_checkpoint=path)

    report = example._restored_training_report(model, restored, digest)

    assert report["performed"] is False
    assert report["reason"] == "restored_parameter_checkpoint"
    assert report["parameter_checkpoint_sha256"] == digest
    assert report["losses"] == []
    assert set(report["optimizer_updates_by_effort"].values()) == {0}
    assert report["parameter_sha256_before"] == report["parameter_sha256_after"]


def test_restore_then_continue_optimizes_from_the_restored_parameters(
    example, tmp_path
):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    data = example._load_data(config)
    device = jax.devices("cpu")[0]
    seed_model = example._make_model(config, rows, batch_size=1, device=device)
    for state in seed_model.states(example.brainstate.ParamState).values():
        state.value = jax.tree.map(lambda leaf: leaf * 0.5, state.value)
    path = tmp_path / "segment.npz"
    digest = example._write_parameter_checkpoint(seed_model, path)
    seeded = example._tree_digest(example.parameter_snapshot(seed_model))

    chained = dataclasses.replace(config, initial_checkpoint=path)
    model = example._make_model(config, rows, batch_size=1, device=device)
    assert example._restore_initial_parameters(model, chained) == digest
    assert example._tree_digest(example.parameter_snapshot(model)) == seeded

    report = example._train_model(
        model, example._training_chunks(data, chained, rows), chained
    )

    assert report["performed"] is True
    assert report["parameter_sha256_before"] == seeded
    assert report["parameters_moved"] is True


def test_a_training_holdout_reserves_the_tail_of_the_split(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    reserved = dataclasses.replace(config, training_holdout_tasks=1)

    whole = example._training_pool(data, config)
    trimmed = example._training_pool(data, reserved)

    assert len(trimmed) == len(whole) - 1
    assert list(trimmed) == list(whole)[:-1]


def test_no_holdout_admits_every_training_task(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)

    assert example._training_pool(data, config) is data.training


def test_a_holdout_that_consumes_the_split_is_rejected(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    empty = dataclasses.replace(
        config, training_holdout_tasks=len(data.training)
    )

    with pytest.raises(ValueError, match="training_holdout_tasks"):
        example._training_pool(data, empty)


def test_a_reserved_task_never_enters_the_training_schedule(example):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(), training_updates=6
    )
    data = example._load_data(config)
    reserved = dataclasses.replace(config, training_holdout_tasks=1)
    withheld = example.canonical_task_fingerprint(
        example._without_official_test_targets(list(data.training)[-1].task)
    )

    chunks = list(example._training_chunks(data, reserved, example._row_config(config)))
    sampled = {
        fingerprint
        for chunk in chunks
        for fingerprint in chunk.base_task_fingerprints
    }

    assert withheld not in sampled


def test_an_initial_checkpoint_seeds_a_further_training_segment(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    source = example._make_model(config, rows, batch_size=1, device=device)
    target = example._make_model(config, rows, batch_size=1, device=device)
    for state in target.states(example.brainstate.ParamState).values():
        state.value = jax.tree.map(lambda leaf: leaf + 1.0, state.value)
    path = tmp_path / "segment.npz"
    written = example._write_parameter_checkpoint(source, path)
    chained = dataclasses.replace(config, initial_checkpoint=path)

    digest = example._restore_initial_parameters(target, chained)

    assert digest == written
    for left, right in zip(
        jax.tree.leaves(example.parameter_snapshot(source)),
        jax.tree.leaves(example.parameter_snapshot(target)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_an_absent_initial_checkpoint_leaves_the_initialization_alone(
    example, tmp_path
):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    model = example._make_model(config, rows, batch_size=1, device=device)
    before = example._tree_digest(example.parameter_snapshot(model))
    absent = dataclasses.replace(config, initial_checkpoint=tmp_path / "absent.npz")

    assert example._restore_initial_parameters(model, absent) is None
    assert example._tree_digest(example.parameter_snapshot(model)) == before


def test_an_initial_checkpoint_from_another_scale_is_rejected(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    narrow = example._make_model(config, rows, batch_size=1, device=device)
    wider = dataclasses.replace(config, neuron_count=config.neuron_count * 2)
    broad = example._make_model(wider, rows, batch_size=1, device=device)
    path = tmp_path / "segment.npz"
    example._write_parameter_checkpoint(narrow, path)

    with pytest.raises(ValueError, match="parameter checkpoint"):
        example._restore_initial_parameters(
            broad, dataclasses.replace(wider, initial_checkpoint=path)
        )


def test_periodic_checkpoints_land_on_the_configured_chunk_boundary(
    example, tmp_path
):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    model = example._make_model(config, rows, batch_size=1, device=device)
    path = tmp_path / "periodic.npz"
    periodic = dataclasses.replace(
        config, parameter_checkpoint=path, checkpoint_every=2
    )

    write = example._checkpoint_writer(model, periodic)
    write(0)
    assert not path.exists()
    write(1)
    assert path.exists()

    restored = example._make_model(config, rows, batch_size=1, device=device)
    example._read_parameter_checkpoint(restored, path)


def test_no_periodic_writer_exists_without_an_interval(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    model = example._make_model(config, rows, batch_size=1, device=jax.devices("cpu")[0])
    single = dataclasses.replace(
        config, parameter_checkpoint=tmp_path / "once.npz", checkpoint_every=0
    )

    assert example._checkpoint_writer(model, single) is None


def test_a_checkpoint_interval_without_a_destination_is_rejected(example):
    with pytest.raises(ValueError, match="checkpoint_every requires"):
        example.ExperimentConfig(checkpoint_every=1)


def test_a_failed_checkpoint_write_leaves_the_previous_file_intact(
    example, tmp_path, monkeypatch
):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    device = jax.devices("cpu")[0]
    model = example._make_model(config, rows, batch_size=1, device=device)
    path = tmp_path / "segment.npz"
    example._write_parameter_checkpoint(model, path)
    survivor = path.read_bytes()

    def explode(*_args, **_kwargs):
        raise OSError("device full")

    monkeypatch.setattr(example.np, "savez", explode)
    with pytest.raises(OSError):
        example._write_parameter_checkpoint(model, path)

    assert path.read_bytes() == survivor


def test_the_configuration_reports_its_initial_checkpoint(example, tmp_path):
    path = tmp_path / "segment.npz"
    config = example.ExperimentConfig.smoke_config()

    assert config.to_dict()["initial_checkpoint"] is None
    chained = dataclasses.replace(config, initial_checkpoint=path)
    assert chained.to_dict()["initial_checkpoint"] == str(path)


def test_every_chunk_reaches_the_periodic_checkpoint_callback(example):
    seen = []
    chunks = [
        example._TrainingTensors(
            events=np.zeros((1, 4, 1, 2), dtype=np.float32),
            advances=np.zeros((1, 4, 1), dtype=np.bool_),
            heights=np.zeros((1, 1), dtype=np.int32),
            widths=np.zeros((1, 1), dtype=np.int32),
            colors=np.zeros((1, 1, 30, 30), dtype=np.int32),
            masks=np.zeros((1, 4), dtype=np.float32),
            efforts=np.asarray([30], dtype=np.int32),
            task_fingerprints=("a",),
            base_task_fingerprints=("a",),
            source_names=("train",),
            held_out_demonstration_indices=(0,),
        )
        for _ in range(3)
    ]

    example._train_chunks(chunks, lambda *_args: np.zeros((1,)), seen.append)

    assert seen == [0, 1, 2]


def test_single_episode_batches_reproduce_the_unbatched_schedule(example):
    reference = example.ExperimentConfig.smoke_config()
    explicit = dataclasses.replace(reference, training_batch_size=1)
    data = example._load_data(reference)
    rows = example._row_config(reference)

    first = _materialize_training(example, data, reference, rows)
    second = _materialize_training(example, data, explicit, rows)

    for field in _TRAINING_ARRAY_FIELDS:
        assert np.array_equal(getattr(first, field), getattr(second, field)), field
    for field in _TRAINING_METADATA_FIELDS:
        assert getattr(first, field) == getattr(second, field), field


def test_batched_updates_carry_one_episode_per_batch_slot(example):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(), training_batch_size=3
    )
    data = example._load_data(config)
    rows = example._row_config(config)

    tensors = _materialize_training(example, data, config, rows)

    updates = config.training_updates
    assert tensors.events.shape[0] == updates
    assert tensors.events.shape[2] == 3
    assert tensors.advances.shape[1:] == (tensors.events.shape[1], 3)
    assert tensors.heights.shape == (updates, 3)
    assert tensors.widths.shape == (updates, 3)
    assert tensors.colors.shape == (updates, 3, 30, 30)
    assert tensors.masks.shape == (updates, tensors.events.shape[1])
    assert len(tensors.task_fingerprints) == updates * 3
    assert len(tensors.held_out_demonstration_indices) == updates * 3


def test_batched_tick_mask_selects_every_latent_tick_of_the_batch(example):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(), training_batch_size=2
    )
    data = example._load_data(config)
    rows = example._row_config(config)
    rng = example.brainstate.random.RandomState(config.seed)
    first = example._training_row(
        data.training[0], config, rows, rng, effort=30, plumbing_only=data.plumbing_only
    )
    second = example._training_row(
        data.training[-1],
        config,
        rows,
        rng,
        effort=30,
        plumbing_only=data.plumbing_only,
    )

    merged = example._merge_training_rows([first, second])

    union = (first["masks"] > 0.0) | (second["masks"] > 0.0)
    assert np.array_equal(merged["masks"] > 0.0, union)
    assert np.isclose(float(merged["masks"].sum()), 1.0)
    assert merged["events"].shape[1] == 2
    assert np.array_equal(merged["events"][:, :1], first["events"])
    assert np.array_equal(merged["events"][:, 1:], second["events"])


def test_merging_one_episode_leaves_its_tensors_unchanged(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    rows = example._row_config(config)
    rng = example.brainstate.random.RandomState(config.seed)
    row = example._training_row(
        data.training[0], config, rows, rng, effort=60, plumbing_only=data.plumbing_only
    )

    merged = example._merge_training_rows([row])

    assert np.array_equal(merged["events"], row["events"])
    assert np.array_equal(merged["advances"], row["advances"])
    assert np.array_equal(merged["masks"], row["masks"])
    assert np.array_equal(merged["colors"], row["colors"])


def test_episode_bank_reuses_encoded_folds_for_every_effort(example):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(), training_bank_size=2
    )
    data = example._load_data(config)
    rows = example._row_config(config)
    rng = example.brainstate.random.RandomState(config.seed)

    bank = example._training_bank(data, config, rows, rng)

    assert set(bank) == {int(effort) for effort in config.training_efforts}
    assert all(len(episodes) == 2 for episodes in bank.values())
    for effort, episodes in bank.items():
        for episode in episodes:
            assert np.isclose(float(episode["masks"].sum()), 1.0)
            assert int(np.count_nonzero(episode["masks"])) == 30


def test_an_empty_bank_encodes_each_episode_slot_directly(example):
    config = example.ExperimentConfig.smoke_config(decoder_mode="row_refinement")
    data = example._load_data(config)
    rows = example._row_config(config)
    rng = example.brainstate.random.RandomState(config.seed)

    assert example._training_bank(data, config, rows, rng) == {}

    fresh = example._banked_training_row(
        {},
        data.training[0],
        config,
        rows,
        rng,
        effort=30,
        plumbing_only=data.plumbing_only,
    )
    assert int(np.count_nonzero(fresh["masks"])) == 30


def test_banked_schedules_are_reproducible_and_chunk_independent(example):
    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
        training_batch_size=2,
        training_bank_size=4,
    )
    split = dataclasses.replace(config, training_chunk_size=1)
    data = example._load_data(config)
    rows = example._row_config(config)

    reference = _materialize_training(example, data, config, rows)
    repeated = _materialize_training(example, data, config, rows)
    chunked = _materialize_training(example, data, split, rows)

    for field in _TRAINING_ARRAY_FIELDS:
        assert np.array_equal(getattr(reference, field), getattr(repeated, field)), (
            field
        )
        assert np.array_equal(getattr(reference, field), getattr(chunked, field)), field
    for field in _TRAINING_METADATA_FIELDS:
        assert getattr(reference, field) == getattr(chunked, field), field


def test_batched_training_expands_one_effort_per_episode(example):
    assert example._per_episode_efforts((30, 60), 1) == (30, 60)
    assert example._per_episode_efforts((30, 60), 3) == (30, 30, 30, 60, 60, 60)


def test_a_bank_smaller_than_one_batch_is_rejected(example):
    with pytest.raises(ValueError, match="training_bank_size"):
        example.ExperimentConfig(training_batch_size=8, training_bank_size=4)
    with pytest.raises(ValueError, match="training_batch_size"):
        example.ExperimentConfig(training_batch_size=0)


def test_chunked_training_reproduces_unchunked_losses_bitwise(example):
    whole = example.ExperimentConfig.smoke_config()
    split = dataclasses.replace(whole, training_chunk_size=1)
    data = example._load_data(whole)
    rows = example._row_config(whole)

    def train(config):
        model = example._make_model(
            config, rows, batch_size=1, device=jax.devices("cpu")[0]
        )
        return example._train_model(
            model, example._training_chunks(data, config, rows), config
        )

    reference = train(whole)
    chunked = train(split)

    assert chunked["losses"] == reference["losses"]
    assert chunked["effort_schedule"] == reference["effort_schedule"]
    assert chunked["training_samples"] == reference["training_samples"]
    assert chunked["parameter_sha256_after"] == reference["parameter_sha256_after"]


def test_adapted_checkpoint_tensor_scoring_preserves_frozen_window_semantics(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))
    compact = np.zeros((TEST_DEPTH_COUNT, len(records), 9060), dtype=np.float32)
    compact[0, :, 60 + 1] = 3.0
    compact[30, :, 60 + 2] = 4.0
    compact[60, :, 60 + 3] = 5.0

    frozen = example._score_windows(
        compact,
        records,
        color_rank=config.color_rank,
        decoder_mode="row_refinement",
    )
    adapted = example._score_checkpoint_logits(
        compact[np.asarray(TEST_CHECKPOINTS)],
        records,
        color_rank=config.color_rank,
        decoder_mode="row_refinement",
    )

    assert adapted == frozen


def test_task_local_entry_uses_post_pretraining_snapshot_and_target_free_bank(
    example, monkeypatch
):
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1,),)), ArcGrid(((2,),))),
            ArcPair(ArcGrid(((3,),)), ArcGrid(((4,),))),
        ),
        test=(ArcPair(ArcGrid(((5,),)), ArcGrid(((6,),))),),
        task_id="entry-adaptation",
    )
    changed_task = dataclasses.replace(
        task,
        test=(ArcPair(task.test[0].input, ArcGrid(((9,),))),),
    )
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    trained_model = SimpleNamespace(pretraining_marker=73)
    real_builder = example.build_arc_target_free_task_bank
    call_order: list[str] = []
    banks = []
    scored_logits = []

    class FakeOptimizer:
        def __init__(self, *, lr):
            self.lr = lr
            self.step_count = SimpleNamespace(value=np.int32(0))

        def register_trainable_weights(self, states):
            assert states == {"learned": "paths"}

    def make_model(*args, **kwargs):
        return SimpleNamespace(pretraining_marker=None)

    def copy_parameters(source, target):
        call_order.append("copy_pretrained_parameters")
        target.pretraining_marker = source.pretraining_marker

    def snapshot_parameters(model):
        call_order.append("snapshot_adaptation_base")
        assert model.pretraining_marker == 73
        return "post-pretraining-snapshot"

    def parameter_snapshot(model):
        return {"marker": np.asarray([model.pretraining_marker], dtype=np.int32)}

    def build_bank(tasks, row_config, *, latent_steps):
        bank = real_builder(tasks, row_config, latent_steps=latent_steps)
        banks.append(bank)
        return bank

    def compile_runner(model, learner, optimizer, **kwargs):
        assert kwargs["base_parameters"] == "post-pretraining-snapshot"
        # The real runner repeats each task's fold schedule `epochs` times, so
        # the fake must too: the caller counts applied folds against
        # valid-folds-times-epochs and would otherwise see a phantom shortfall.
        epochs = int(kwargs.get("epochs", 1))

        def run(bank):
            task_count, query_count = bank.query_valid.shape
            outputs = np.zeros((task_count, query_count, 3, 9060), dtype=np.float32)
            outputs[..., 0] = np.arange(3, dtype=np.float32)
            applied = np.tile(np.asarray(bank.fold_inputs.fold_valid), (1, epochs))
            return SimpleNamespace(
                fold_losses=np.ones(applied.shape, dtype=np.float32),
                fold_applied=applied,
                checkpoint_outputs=outputs,
                checkpoint_recorded=np.broadcast_to(
                    np.asarray(bank.query_valid)[..., None],
                    (task_count, query_count, 3),
                ),
                query_valid=np.asarray(bank.query_valid),
            )

        return run

    def score(checkpoint_logits, records, color_rank, decoder_mode):
        scored_logits.append(np.asarray(checkpoint_logits).copy())
        assert decoder_mode == "latent_row_decode"
        metrics = {str(step): _metric() for step in TEST_CHECKPOINTS}
        details = {str(step): [] for step in TEST_CHECKPOINTS}
        return metrics, details

    monkeypatch.setattr(example, "_make_model", make_model)
    monkeypatch.setattr(example, "_copy_parameters", copy_parameters)
    monkeypatch.setattr(example, "snapshot_parameters", snapshot_parameters)
    monkeypatch.setattr(example, "parameter_snapshot", parameter_snapshot)
    monkeypatch.setattr(
        example,
        "compile_pp_prop",
        lambda model: SimpleNamespace(param_states={"learned": "paths"}, report=None),
    )
    monkeypatch.setattr(example.braintools.optim, "Adam", FakeOptimizer)
    monkeypatch.setattr(example, "build_arc_target_free_task_bank", build_bank)
    monkeypatch.setattr(
        example, "compile_arc_task_local_adaptation_runner", compile_runner
    )
    monkeypatch.setattr(example, "_score_checkpoint_logits", score)

    evidences = []
    for current_task in (task, changed_task):
        origin = example._OriginTask("fixture", "evaluation", current_task)
        data = example._ExperimentData((), (origin,), (), True)
        records = example._evaluation_records(data, config, rows)
        _metrics, _details, evidence = example._task_local_adaptation_evaluation(
            trained_model,
            data,
            config,
            rows,
            SimpleNamespace(),
            records,
        )
        evidences.append(evidence)

    assert call_order == [
        "copy_pretrained_parameters",
        "snapshot_adaptation_base",
        "copy_pretrained_parameters",
        "snapshot_adaptation_base",
    ]
    assert len(scored_logits) == 2
    assert scored_logits[0].shape == (3, 1, 9060)
    assert np.array_equal(scored_logits[0], scored_logits[1])
    for left, right in zip(
        jax.tree.leaves(banks[0]), jax.tree.leaves(banks[1]), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert not any("target" in field for field in banks[0]._fields)
    assert not any("output" in field for field in banks[0]._fields)
    for evidence in evidences:
        assert evidence["performed"] is True
        assert evidence["mode"] == "compiled_task_local_pp_prop_leave_one_out"
        assert evidence["target_free_query_bank"] is True
        # Two leave-one-out folds, replayed once per configured epoch. Pinning
        # the product rather than the constant keeps this honest if the default
        # epoch count moves again.
        assert evidence["fold_count"] == 2 * config.adaptation_epochs
        assert evidence["applied_fold_count"] == 2 * config.adaptation_epochs
        assert evidence["query_count"] == 1
        assert evidence["bank_bytes"] == banks[0].projected_bytes
        assert (
            evidence["base_parameter_sha256"] == evidence["restored_parameter_sha256"]
        )
        assert evidence["base_parameters_restored"] is True


def test_report_labels_adapted_primary_and_frozen_no_adaptation_separately(example):
    evaluation = _evaluation_payload()
    evaluation["task_local_adaptation"] = {
        "performed": True,
        "mode": "compiled_task_local_pp_prop_leave_one_out",
        "fold_count": 2,
        "applied_fold_count": 2,
        "query_count": 1,
        "bank_bytes": 4096,
        "base_parameter_sha256": "abc",
        "restored_parameter_sha256": "abc",
        "base_parameters_restored": True,
    }
    evaluation["frozen_no_adaptation"] = {
        "role": "diagnostic_control_not_primary_submission",
        "metrics_by_effort": copy.deepcopy(evaluation["metrics_by_effort"]),
    }
    result = {
        "evaluation": evaluation,
        "model": {},
        "training": {},
        "qualification": {},
    }

    report = example._render_report(result)

    assert "Task-local adaptation:" in report
    assert "folds applied=2/2" in report
    assert "target-free query bank" in report
    assert "Frozen no-adaptation diagnostic:" in report
    assert "diagnostic control, not primary submission" in report
    assert "Frozen no-adaptation aggregate latent trajectory:" in report


def test_report_names_the_resolved_shared_training_optimizer(example):
    result = {
        "evaluation": _evaluation_payload(),
        "model": {},
        "training": {
            "performed": True,
            "optimizer": {"name": "muon"},
            "optimizer_updates_by_effort": {"30": 1, "60": 1},
        },
        "qualification": {},
    }

    report = example._render_report(result)

    assert "one Muon state" in report
    assert "one Adam state" not in report


def test_memory_coding_flag_wires_into_model_config(example):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    default_args = example._parser().parse_args([])
    assert default_args.memory_coding is None
    args = example._parser().parse_args(["--memory-coding", "learned_keys"])
    config = example._config_from_args(args)
    assert config.memory_coding == "learned_keys"
    model_config = example._model_config(config, rows, batch_size=2)
    assert model_config.memory_coding == "learned_keys"
    default_config = example._config_from_args(example._parser().parse_args([]))
    default_model = example._model_config(default_config, rows, batch_size=2)
    assert default_model.memory_coding == "learned_update"
    smoke_args = example._parser().parse_args(
        ["--smoke", "--memory-coding", "learned_keys"]
    )
    smoke_config = example._config_from_args(smoke_args)
    assert smoke_config.memory_coding == "learned_keys"
    with pytest.raises(SystemExit):
        example._parser().parse_args(["--memory-coding", "bogus"])


def test_learned_write_coding_flag_wires_into_model_config(example):
    """The fused-write arm reaches the model config through the same path."""
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    args = example._parser().parse_args(["--memory-coding", "learned_write"])
    config = example._config_from_args(args)
    assert config.memory_coding == "learned_write"
    model_config = example._model_config(config, rows, batch_size=2)
    assert model_config.memory_coding == "learned_write"
    smoke_config = example._config_from_args(
        example._parser().parse_args(["--smoke", "--memory-coding", "learned_write"])
    )
    assert smoke_config.memory_coding == "learned_write"


def test_trace_engine_cli_threads_into_both_config_layers(example):
    parsed = example._parser().parse_args([])
    assert parsed.trace_engine == "pp_prop"

    args = example._parser().parse_args(["--trace-engine", "d_rtrl"])
    config = example._config_from_args(args)
    assert config.trace_engine == "d_rtrl"
    assert config.to_dict()["trace_engine"] == "d_rtrl"

    with pytest.raises(ValueError, match="trace_engine"):
        dataclasses.replace(config, trace_engine="rtrl")

    smoke_args = example._parser().parse_args(
        ["--smoke", "--trace-engine", "d_rtrl"])
    smoke = example._config_from_args(smoke_args)
    assert smoke.trace_engine == "d_rtrl"

    rows = example._row_config(smoke)
    model_config = example._model_config(smoke, rows, batch_size=1)
    assert model_config.trace_engine == "d_rtrl"
