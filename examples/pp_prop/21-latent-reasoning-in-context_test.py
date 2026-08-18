"""Tests for the standard-ARC latent-reasoning entry point."""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import json
import pathlib
import sys
import time
import warnings
from enum import Enum
from types import SimpleNamespace

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    DatasetSource,
    LoadedDataset,
    RowEventConfig,
    SourceManifest,
    encode_query_episode,
    smoke_loaded_dataset,
)


EXAMPLE = pathlib.Path(__file__).with_name("21-latent-reasoning-in-context.py")


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
        self.target = jnp.zeros(
            (1, config.compact_output_width), dtype=jnp.float32
        ).at[:, 0].set(1.0)

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
        example.ExperimentConfig.smoke_config(seed=41),
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
    metrics = {str(effort): _metric(float(effort == 32)) for effort in (0, 8, 16, 32)}
    comparison = {
        "causally_null_at_measured_precision": False,
        "state_byte_identical_by_step": [False] * 33,
        "spike_hamming_by_step": [1] * 33,
        "spike_hamming_fraction_by_step": [0.125] * 33,
        "voltage_l2_by_step": [1.0] * 33,
        "synaptic_current_l2_by_step": {
            "feedforward": [1.0] * 33,
            "recurrent": [1.0] * 33,
        },
        "score_deltas_control_minus_intact": {
            "32.query_pass_at_2": 0.0,
            "32.valid_cell_pixel_accuracy_diagnostic": 0.0,
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
        for step in range(33)
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
        for step in range(33)
    ]
    checkpoint_queries = {
        str(effort): [
            {
                "task_id": "task",
                "query_index": 0,
                "candidates": [{"grid": [[0]]}],
                "score": {},
            }
        ]
        for effort in (0, 8, 16, 32)
    }
    control["decoded_candidates_match_intact"] = True
    control["decoded_candidates_match_intact_by_effort"] = {
        str(effort): True for effort in (0, 8, 16, 32)
    }
    control["decoded_candidate_match_query_count_by_effort"] = {
        str(effort): 1 for effort in (0, 8, 16, 32)
    }
    control["byte_identical_query_count"] = 0
    return {
        "query_count": 1,
        "task_count": 1,
        "same_frozen_parameter_bytes": True,
        "checkpoint_queries": checkpoint_queries,
        "query_trajectories": [
            {"step_count": 33, "neuron_count": 2048, "steps": query_steps}
        ],
        "metrics_by_effort": metrics,
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
            "repeat_intact_numeric_evidence": _numeric_evidence(list(range(33))),
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
                    "state_byte_identical_by_step": [True] * 33,
                    "spike_hamming_by_step": [0] * 33,
                    "spike_hamming_fraction_by_step": [0.0] * 33,
                    "voltage_l2_by_step": [0.0] * 33,
                    "synaptic_current_l2_by_step": {
                        "feedforward": [0.0] * 33,
                        "recurrent": [0.0] * 33,
                    },
                },
                causally_null_query_count=1,
                byte_identical_query_count=1,
            ),
            "no_context": copy.deepcopy(control),
            "shuffled_demonstrations": copy.deepcopy(control),
            "slot_ablation": copy.deepcopy(control),
            "truncation": {
                "checkpoints": [0, 8, 16, 32],
                "uses_one_continuous_intact_trajectory": True,
            },
        },
    }


def test_full_and_smoke_configs_preserve_declared_physical_scales(example, tmp_path):
    full = example.ExperimentConfig(output_dir=tmp_path, structural_only=True)
    smoke = example.ExperimentConfig.smoke_config(output_dir=tmp_path)

    assert (full.neuron_count, full.recurrent_edges) == (2048, 16384)
    assert (smoke.neuron_count, smoke.recurrent_edges) == (128, 1024)
    assert full.max_demonstrations == 10
    assert full.to_dict()["checkpoints"] == [0, 8, 16, 32]
    assert smoke.smoke is True
    assert full.context_memory_width == 0
    assert full.memory_decay == 1.0
    assert full.balanced_color_loss is False
    assert full.to_dict()["balanced_color_loss"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_memory_width": True}, "context_memory_width"),
        ({"context_memory_width": -1}, "context_memory_width"),
        ({"context_memory_width": 129}, "at most 128"),
        ({"memory_decay": True}, "memory_decay"),
        ({"memory_decay": float("nan")}, "memory_decay"),
        ({"memory_decay": -0.01}, "memory_decay"),
        ({"memory_decay": 1.01}, "memory_decay"),
    ],
)
def test_associative_memory_config_rejects_invalid_values(example, kwargs, message):
    with pytest.raises(ValueError, match=message):
        example.ExperimentConfig(structural_only=True, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"neuron_count": 127}, "divisible by 64"),
        ({"recurrent_edges": 2048 * 2048}, "no-self capacity"),
        ({"max_grid_size": 29}, "standard ARC"),
        ({"latent_steps": 31}, "latent_steps"),
        ({"ablation_slot": 32}, "ablation_slot"),
        ({"training_updates": 2}, "8, 16, and 32"),
        ({"device": "tpu"}, "device"),
        ({"learning_rate": float("nan")}, "learning_rate"),
    ],
)
def test_config_rejects_invalid_or_scientifically_incomplete_values(
    example, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        example.ExperimentConfig(**kwargs)


def test_structural_config_allows_zero_updates(example):
    config = example.ExperimentConfig(structural_only=True, training_updates=0)
    assert config.training_updates == 0


def test_cli_defaults_fail_closed_to_full_gpu_and_smoke_owns_scale(example, tmp_path):
    parsed = example._parser().parse_args([])
    assert parsed.device == "gpu"
    assert parsed.neurons == 2048
    assert parsed.recurrent_edges == 16384
    assert parsed.context_memory_width == 0
    assert parsed.memory_decay == 1.0
    assert parsed.balanced_color_loss is False

    smoke_args = example._parser().parse_args(
        [
            "--smoke",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--balanced-color-loss",
        ]
    )
    smoke = example._config_from_args(smoke_args)
    assert smoke.smoke and smoke.device == "cpu"
    assert smoke.balanced_color_loss is True

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
    config = example.ExperimentConfig.smoke_config()

    packed = example._packed_events(encoded, config)
    advances = example._packed_advances(encoded, config, rows)

    assert rows.max_events == 150
    assert packed.shape == (182, rows.input_width)
    assert np.count_nonzero(packed[encoded.events.shape[0] :]) == 0
    assert np.count_nonzero(packed[encoded.query_stop : encoded.query_stop + 32]) == 0
    assert advances.dtype == np.bool_
    assert advances[encoded.query_stop : encoded.query_stop + 32].all()


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
    assert not advances[encoded.query_stop + 32 :].any()


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
    assert relative_deviation(padded_bptt, compact_bptt) == pytest.approx(
        0.0, abs=1e-7
    )
    assert_gradients_differ(padded_pp_prop, compact_pp_prop, min_rel=1e-4)


def test_effort_schedule_is_balanced_reproducible_and_mixed(example):
    first = example._effort_schedule(11, brainstate.random.RandomState(7))
    second = example._effort_schedule(11, brainstate.random.RandomState(7))
    counts = {effort: int(np.sum(first == effort)) for effort in (8, 16, 32)}

    assert np.array_equal(first, second)
    assert set(first.tolist()) == {8, 16, 32}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_training_stream_compacts_learning_rule_timeline(example):
    """No frozen layout position may age the trace before supervised depths."""
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    tensors = example._prepare_training(example._load_data(config), config, rows)

    for advances, events in zip(tensors.advances, tensors.events, strict=True):
        advancing = advances[:, 0]
        prefix_length = int(np.count_nonzero(advancing))
        assert advancing[:prefix_length].all()
        assert not advancing[prefix_length:].any()
        assert np.count_nonzero(events[prefix_length:]) == 0


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
    task, config, rows, encoded = _tiny_compaction_fixture(example)
    effort = 8
    row = example._training_row(
        example._OriginTask("fixture", "fixture", task),
        config,
        rows,
        brainstate.random.RandomState(7),
        effort=effort,
        plumbing_only=True,
    )

    padded = example._packed_events(encoded, config)
    active_indices = np.flatnonzero(
        example._packed_advances(encoded, config, rows)
    )
    query_checkpoint = int(
        np.flatnonzero(active_indices == encoded.query_stop - 1)[0]
    )
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
    np.testing.assert_array_equal(
        produced_mask[: active_indices.size], reference_mask
    )

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
    target[0, int(row["heights"]) - 1] = 1.0
    target[0, MAX_GRID_SIZE + int(row["widths"]) - 1] = 1.0

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


def test_training_mask_supervises_every_depth_with_unit_update_weight(example):
    """An effort-R update weights every depth 0..R and totals one update."""
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    tensors = example._prepare_training(example._load_data(config), config, rows)

    for effort, mask in zip(tensors.efforts, tensors.masks, strict=True):
        depth_count = int(effort) + 1
        supervised = np.flatnonzero(mask)
        assert supervised.size == depth_count
        np.testing.assert_array_equal(
            supervised, np.arange(supervised[-1] - int(effort), supervised[-1] + 1)
        )
        np.testing.assert_allclose(
            mask[supervised], np.full(depth_count, 1.0 / depth_count)
        )
        assert float(np.sum(mask)) == pytest.approx(1.0)


def test_training_tensor_terminals_follow_each_sample_effort(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    rows = example._row_config(config)

    tensors = example._prepare_training(data, config, rows)

    assert tensors.events.shape == (3, rows.max_events + 32, 1, rows.input_width)
    assert tensors.advances.shape == (3, rows.max_events + 32, 1)
    assert tensors.colors.shape == (3, 1, 30, 30)
    assert np.all(np.sum(tensors.masks, axis=1) == 1.0)
    for index, effort in enumerate(tensors.efforts):
        supervised = np.flatnonzero(tensors.masks[index])
        checkpoint = int(supervised[0])
        terminal = int(supervised[-1])
        assert terminal - checkpoint == int(effort)
        assert tensors.events[index, checkpoint, 0, rows.valid_slice.start] == 1.0
        assert (
            np.count_nonzero(tensors.events[index, checkpoint + 1 : terminal + 1]) == 0
        )
        assert tensors.advances[index, checkpoint + 1 : terminal + 1].all()


def test_model_configuration_parameter_copy_and_digest_are_explicit(example):
    config = example.ExperimentConfig.smoke_config()
    rows = example._row_config(config)
    model_config = example._model_config(config, rows, batch_size=3)
    assert model_config.batch_size == 3
    assert model_config.input_width == rows.input_width
    assert model_config.recurrent_edges == 1024

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


def test_model_configuration_wires_opt_in_associative_memory_features(example):
    rows = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    legacy = example._model_config(
        example.ExperimentConfig.smoke_config(), rows, batch_size=3
    )
    assert legacy.context_memory_width == 0
    assert legacy.memory_key_indices == ()
    assert legacy.memory_value_indices == ()
    assert legacy.demonstration_phase_index is None
    assert legacy.query_phase_index is None
    assert legacy.input_side_valid_index is None
    assert legacy.output_side_valid_index is None

    config = dataclasses.replace(
        example.ExperimentConfig.smoke_config(),
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
        example.ExperimentConfig(structural_only=True),
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
            example.ExperimentConfig.smoke_config(), rows, batch_size=1
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
    assert json.dumps(
        legacy, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") == json.dumps(
        expected_legacy, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

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
    }
    assert memory["carrier_stabilizer"] == "per_example_stopped_unit_l2_cap"
    assert memory["carrier_radius"] == 1.0
    assert memory["carrier_consumers"] == (
        "readout_projection",
        "workspace_query_projection",
    )


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
    tensors = example._prepare_training(fixture, config, example._row_config(config))
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


def test_gather_window_uses_each_query_terminal(example):
    time, batch = 40, 2
    packed = SimpleNamespace(
        compact_logits=np.arange(time * batch * 3).reshape(time, batch, 3),
        spikes=np.arange(time * batch * 4).reshape(time, batch, 4),
        voltage=np.arange(time * batch * 5).reshape(time, batch, 5),
        feedforward_current=np.arange(time * batch * 6).reshape(time, batch, 6),
        recurrent_current=np.arange(time * batch * 7).reshape(time, batch, 7),
    )
    compact, spikes, voltage, feedforward, recurrent = example._gather_window(
        packed, np.array([2, 5])
    )
    assert compact.shape == (33, 2, 3)
    assert spikes[0, 0, 0] == packed.spikes[1, 0, 0]
    assert voltage[32, 1, 0] == packed.voltage[36, 1, 0]
    assert feedforward.shape == (33, 2, 6)
    assert recurrent.shape == (33, 2, 7)


def test_scoring_trajectory_and_null_control_share_the_same_frozen_windows(example):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))
    batch = len(records)
    compact = np.zeros((33, batch, 340), dtype=np.float32)
    spikes = np.zeros((33, batch, 8), dtype=np.float32)
    voltage = np.zeros((33, batch, 8), dtype=np.float32)
    feedforward = np.zeros_like(voltage)
    recurrent = np.zeros_like(voltage)

    metrics, details = example._score_windows(compact, records, color_rank=4)
    reports, aggregate = example._trajectory_reports(
        compact,
        spikes,
        voltage,
        feedforward,
        recurrent,
        records,
        color_rank=4,
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
        metrics,
        [{"available": True, "timing_matched": True} for _ in records],
    )

    assert set(metrics) == {"0", "8", "16", "32"}
    assert all(len(details[str(effort)]) == batch for effort in (0, 8, 16, 32))
    assert len(reports) == batch
    assert len(aggregate) == 33
    assert aggregate[0]["unique_state_hashes"] == 1
    assert control["causally_null_query_count"] == batch
    assert control["available_query_count"] == batch
    assert control["timing_matched_query_count"] == batch
    assert control["trajectory_comparison"]["causally_null_at_measured_precision"]
    assert "checkpoint_queries" not in control


def test_unavailable_shuffle_queries_are_excluded_from_control_statistics(
    example, monkeypatch
):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))[:2]
    compact = np.zeros((33, 2, 340), dtype=np.float32)
    spikes = np.zeros((33, 2, 8), dtype=np.float32)
    voltage = np.zeros((33, 2, 8), dtype=np.float32)
    feedforward = np.zeros_like(voltage)
    recurrent = np.zeros_like(voltage)
    captured: dict[str, object] = {}

    def score(subset_compact, subset_records, color_rank, subset_rules=None):
        captured["scored_records"] = len(subset_records)
        captured["scored_batch"] = subset_compact.shape[1]
        captured["scored_rules"] = None if subset_rules is None else len(subset_rules)
        details = {
            str(effort): [{"candidates": [{"grid": [[0]]}]} for _ in subset_records]
            for effort in (0, 8, 16, 32)
        }
        return {str(effort): _metric() for effort in (0, 8, 16, 32)}, details

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
        {str(effort): _metric() for effort in (0, 8, 16, 32)},
        metadata,
    )

    assert captured == {
        "scored_records": 1,
        "scored_batch": 1,
        "scored_rules": 1,
        "comparison_width": 8,
    }
    assert result["query_count"] == 2
    assert result["applicable_query_count"] == 1
    assert result["unavailable_query_count"] == 1


def test_all_unavailable_shuffle_renders_as_unavailable(example, tmp_path):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    records = example._evaluation_records(data, config, example._row_config(config))[:1]
    compact = np.zeros((33, 1, 340), dtype=np.float32)
    state = np.zeros((33, 1, 8), dtype=np.float32)
    window = (compact, state, state.copy(), state.copy(), state.copy())

    unavailable = example._control_summary(
        "shuffle",
        window,
        tuple(value.copy() for value in window),
        records,
        4,
        {str(effort): _metric() for effort in (0, 8, 16, 32)},
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


def test_training_uses_one_learner_optimizer_and_all_efforts(example, monkeypatch):
    updates: list[object] = []
    color_loss_kwargs: list[dict[str, object]] = []

    class Learner:
        param_states = {}

        def reset_state(self, *, batch_size):
            assert batch_size == 1

        def __call__(self, event, advance):
            assert event.shape == (1, 4)
            assert advance.shape == (1,)
            return jnp.zeros((1, 340), dtype=jnp.float32)

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
        config = SimpleNamespace(batch_size=1, color_rank=4)

        def reset_state(self):
            return None

    def host_for_loop(function, xs):
        return jnp.stack(
            [
                function(tuple(value[index] for value in xs))
                for index in range(xs[0].shape[0])
            ]
        )

    monkeypatch.setattr(example, "compile_pp_prop", lambda model: Learner())
    monkeypatch.setattr(example.braintools.optim, "Adam", Optimizer)
    monkeypatch.setattr(example.brainstate.transform, "jit", lambda function: function)
    monkeypatch.setattr(example.brainstate.transform, "for_loop", host_for_loop)
    def color_loss(*args, **kwargs):
        color_loss_kwargs.append(kwargs)
        return jnp.ones((1,))

    monkeypatch.setattr(example, "arc_loss_per_example", color_loss)
    monkeypatch.setattr(example, "parameter_snapshot", lambda model: {})
    monkeypatch.setattr(
        example.brainstate.nn, "clip_grad_norm", lambda value, limit: value
    )
    events = np.zeros((3, 1, 1, 4), dtype=np.float32)
    advances = np.ones((3, 1, 1), dtype=np.bool_)
    targets = np.ones((3, 1), dtype=np.int32)
    colors = np.zeros((3, 1, 30, 30), dtype=np.int32)
    masks = np.ones((3, 1), dtype=np.float32)
    tensors = example._TrainingTensors(
        events,
        advances,
        targets,
        targets,
        colors,
        masks,
        np.array([8, 16, 32]),
        ("a", "b", "c"),
        ("base-a", "base-b", "base-c"),
        ("source", "source", "source"),
        (0, 0, 0),
    )

    config = example.ExperimentConfig.smoke_config(balanced_color_loss=True)
    result = example._train_model(Model(), [tensors], config)

    assert len(updates) == 3
    assert result["one_shared_model"] is True
    assert result["one_shared_optimizer_state"] is True
    assert result["optimizer_updates_by_effort"] == {"8": 1, "16": 1, "32": 1}
    assert result["losses"] == [1.0, 1.0, 1.0]
    assert result["balanced_color_loss"] is True
    assert [kwargs["class_balanced_colors"] for kwargs in color_loss_kwargs] == [
        True,
        True,
        True,
    ]


def test_evaluation_runs_four_frozen_arms_and_ablation_at_latent_step_one(
    example, monkeypatch
):
    config = example.ExperimentConfig.smoke_config()
    data = example._load_data(config)
    rows = example._row_config(config)
    records = example._evaluation_records(data, config, rows)
    batch = len(records)
    time = rows.max_events + 32
    stops = np.asarray(
        [record.encoded.query_stop for record in records], dtype=np.int32
    )
    run_calls: list[dict[str, object]] = []
    event_markers: list[float] = []
    jit_options: list[dict[str, object]] = []
    fake_model = SimpleNamespace(config=SimpleNamespace(color_rank=4))

    def sequences(records, config, rows, *, arm, source_tasks):
        marker = {"intact": 1.0, "no_context": 2.0, "shuffled": 3.0}[arm]
        events = np.full(
            (time, batch, rows.input_width), marker, dtype=np.float32
        )
        advances = np.ones((time, batch), dtype=np.bool_)
        metadata = [{"available": True, "timing_matched": True} for _ in records]
        return events, advances, stops, metadata

    def packed(model, events, selected_indices, **kwargs):
        run_calls.append(kwargs)
        event_markers.append(float(np.asarray(events)[0, 0, 0]))
        assert selected_indices.shape == (33, batch)
        return SimpleNamespace(
            compact_logits=np.zeros((33, batch, 340), dtype=np.float32),
            spikes=np.zeros((33, batch, 8), dtype=np.float32),
            voltage=np.zeros((33, batch, 8), dtype=np.float32),
            feedforward_current=np.zeros((33, batch, 8), dtype=np.float32),
            recurrent_current=np.zeros((33, batch, 8), dtype=np.float32),
            memory_read=np.zeros((33, batch, 0), dtype=np.float32),
            final_context_memory=np.zeros((batch, 0, 0), dtype=np.float32),
        )

    def identity_jit(*, inline, name):
        jit_options.append({"inline": inline, "name": name})

        def decorate(function):
            return function

        return decorate

    metrics = {str(effort): _metric() for effort in (0, 8, 16, 32)}
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
                "candidates": [{"grid": [[0]]}],
                "rule_name": None,
                "rule_solved": False,
                "score": {"pass_at_2": False},
            }
            for _ in range(batch)
        ]
        for effort in (0, 8, 16, 32)
    }
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
                str(effort): True for effort in (0, 8, 16, 32)
            },
            "decoded_candidate_match_query_count_by_effort": {
                str(effort): batch for effort in (0, 8, 16, 32)
            },
            "trajectory_comparison": {"causally_null_at_measured_precision": True},
        },
    )

    result = example._evaluate(SimpleNamespace(), data, config, rows, SimpleNamespace())

    assert len(run_calls) == 5
    assert event_markers == [1.0, 1.0, 2.0, 3.0, 1.0]
    assert jit_options == [
        {"inline": False, "name": "example21_evaluation_arm"}
    ]
    assert result["same_frozen_parameter_bytes"] is True
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
    assert result["controls"]["truncation"]["checkpoints"] == [0, 8, 16, 32]
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
    assert result["associative_memory_diagnostics"] == {
        "available": False,
        "complete": True,
        "reason": "legacy_reservoir_has_no_associative_state",
    }


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
        shuffled_report[
            "memory_read_changed_at_any_depth_applicable_query_count"
        ]
        == 2
    )
    assert len(report["intact_context_memory_sha256_by_query"]) == 2

    disconnected = dict(
        controls,
        shuffled_demonstrations=(
            tuple(array.copy() for array in intact),
            metadata,
        ),
    )
    failed = example._associative_evaluation_diagnostics(
        True, intact, disconnected
    )
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
        "optimizer_updates_by_effort": {"8": 1, "16": 1, "32": 1},
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
        "neuron_count": 2048,
        "recurrent_edge_count": 16384,
        "slot_count": 32,
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
    gpu = {"platform": "gpu", "kind": "test"}

    plumbing = example._qualification(
        example.ExperimentConfig.smoke_config(),
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
        example.ExperimentConfig(training_updates=3),
        public_data,
        training,
        evaluation,
        gpu,
        model_report,
    )
    assert scientific["full_structural_qualification"] is True
    assert scientific["full_scientific_qualification"] is True

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
            "path_classification_by_hidden_state": {
                "associative_state": "all_direct"
            },
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
        "depth_count": 33,
        "repeat_intact_exact": True,
        "no_context_memory_exactly_zero": True,
        "shuffled_pairing_sensitive_for_every_applicable_query": True,
    }
    memory_qualification = example._qualification(
        example.ExperimentConfig(training_updates=3, context_memory_width=32),
        public_data,
        memory_training,
        memory_evaluation,
        gpu,
        model_report,
    )
    assert memory_qualification["full_structural_qualification"] is True
    assert memory_qualification["full_scientific_qualification"] is False
    assert (
        memory_qualification["associative_capability_status"]
        == "associative_capability_gates_pending"
    )
    assert "associative_capability_gates_pending" in (
        memory_qualification["reasons_not_scientific"]
    )

    mixed_memory_training = copy.deepcopy(memory_training)
    mixed_memory_training["compiler_report"]["diagnostics"][0][
        "path_classification_by_hidden_state"
    ] = {"associative_state": "mixed"}
    mixed_memory = example._qualification(
        example.ExperimentConfig(training_updates=3, context_memory_width=32),
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
    incomplete_memory_evaluation["associative_memory_diagnostics"]["complete"] = (
        False
    )
    incomplete_memory = example._qualification(
        example.ExperimentConfig(training_updates=3, context_memory_width=32),
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
        example.ExperimentConfig(training_updates=3),
        public_data,
        no_compile,
        evaluation,
        gpu,
        model_report,
    )
    assert failed["full_structural_qualification"] is False
    assert "compilation" in " ".join(failed["reasons_not_scientific"])

    cpu = example._qualification(
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
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
        example.ExperimentConfig(training_updates=3),
        public_data,
        missing_readout,
        evaluation,
        gpu,
        model_report,
    )
    assert movement_failure["full_structural_qualification"] is True
    assert movement_failure["full_scientific_qualification"] is False
    assert "every parameter group" in " ".join(
        movement_failure["reasons_not_scientific"]
    )

    wrong_components = dict(
        model_report,
        component_types=dict(model_report["component_types"], neuron="NotLIF"),
    )
    component_failure = example._qualification(
        example.ExperimentConfig(training_updates=3),
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
        "training": {"optimizer_updates_by_effort": {"8": 1, "16": 1, "32": 1}},
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
    assert "Repeat numeric noise: queries=1; steps=33" in report
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
    config = example.ExperimentConfig.smoke_config(output_dir=tmp_path)
    fixture = smoke_loaded_dataset()
    origin = example._OriginTask("fixture", "fixture", fixture.tasks[0])
    data = example._ExperimentData((origin,), (origin,), (fixture,), True)
    model = SimpleNamespace(
        neuron_count=128,
        recurrent_edge_count=1024,
        slot_count=2,
        config=SimpleNamespace(
            batch_size=1,
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
        "optimizer_updates_by_effort": {"8": 1, "16": 1, "32": 1},
    }
    evaluation = _evaluation_payload()
    monkeypatch.setattr(
        example,
        "_resolve_device",
        lambda name: (SimpleNamespace(), {"platform": "cpu", "kind": "test", "id": 0}),
    )
    monkeypatch.setattr(example, "_load_data", lambda config: data)
    monkeypatch.setattr(example, "_prepare_training", lambda *args: SimpleNamespace())
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

    assert json.loads((tmp_path / "result.json").read_text())["schema_version"] == 1
    assert json.loads((tmp_path / "data_manifest.json").read_text())[0]["plumbing_only"]
    assert "Claim boundary" in (tmp_path / "report.txt").read_text()
    assert (tmp_path / "latent_reasoning.png").read_bytes() == b"png"
    assert set(result["artifacts"]) == {"data_manifest", "result", "report", "figure"}
    assert result["device"]["memory_stats"] == {"peak_bytes_in_use": 123456}
    assert result["device"]["memory_stats_capture"] == "after training and evaluation"
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
        "training": {"optimizer_updates_by_effort": {"8": 1, "16": 1, "32": 1}},
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


def test_chunking_does_not_change_the_prepared_schedule(example):
    whole = example.ExperimentConfig.smoke_config()
    split = dataclasses.replace(whole, training_chunk_size=1)
    data = example._load_data(whole)
    rows = example._row_config(whole)

    reference = example._prepare_training(data, whole, rows)
    chunked = example._prepare_training(data, split, rows)

    assert len(list(example._training_chunks(data, split, rows))) == 3
    for field in example._CHUNK_ARRAY_FIELDS:
        assert np.array_equal(getattr(reference, field), getattr(chunked, field)), field
    for field in example._CHUNK_METADATA_FIELDS:
        assert getattr(reference, field) == getattr(chunked, field), field


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
