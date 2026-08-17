"""Tests for the preregistered Example 21 demonstrated-depth Gate B."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import math
import warnings
from collections.abc import Mapping
from typing import Any

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._testing.oracle import (
    chunked_online_param_gradients,
    flat_gradient_leaves,
    gradient_norm,
    relative_deviation,
)
from examples.pp_prop import latent_workspace_depth_gate as depth
from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop.latent_workspace_model import LatentWorkspaceModel
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
    associative_memory_feature_indices,
    encode_query_episode,
)


def _iterate(mapping: np.ndarray, color: int, applications: int) -> int:
    value = color
    for _ in range(applications):
        value = int(mapping[value])
    return value


def _is_single_ten_cycle(mapping: np.ndarray) -> bool:
    visited: list[int] = []
    value = 0
    for _ in range(10):
        visited.append(value)
        value = int(mapping[value])
    return value == 0 and sorted(visited) == list(range(10))


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _one_cell(color: int) -> ArcGrid:
    return ArcGrid(((int(color),),))


def _encoded_cycle_episode(
    mapping: np.ndarray,
    query_color: int,
    presentation_order: np.ndarray,
    row_config: RowEventConfig,
) -> np.ndarray:
    demonstrations = tuple(
        ArcPair(_one_cell(color), _one_cell(int(mapping[color])))
        for color in np.asarray(presentation_order, dtype=np.int32)
    )
    task = ArcTask(
        train=demonstrations,
        test=(
            ArcPair(
                _one_cell(query_color),
                _one_cell(int(mapping[query_color])),
            ),
        ),
    )
    return np.asarray(encode_query_episode(task, 0, row_config).events)


def _reduced_depth_config() -> depth.DepthGateConfig:
    return depth.DepthGateConfig(
        training_updates=1,
        batch_size=2,
        validation_episodes=2,
        neuron_count=64,
        recurrent_edges=64,
        readout_width=8,
        color_rank=1,
        staging_chunk_updates=1,
    )


class _DepthObjectiveProbe(brainstate.nn.Module):
    def __init__(self, config: depth.DepthGateConfig) -> None:
        super().__init__()
        self.reservoir = LatentWorkspaceModel(depth._model_config(config, batch_size=1))
        self.event_width = config.row_config.input_width
        self.color_rank = config.color_rank

    def update(self, packed: jax.Array) -> jax.Array:
        event = packed[:, : self.event_width]
        advance = packed[:, self.event_width] > 0.5
        target = packed[:, self.event_width + 1].astype(jnp.int32)
        loss_scale = packed[:, self.event_width + 2]
        compact = self.reservoir(event, advance)
        loss = legacy._classification_loss(compact, target, self.color_rank)
        return loss_scale * jnp.sqrt(loss)


def _depth_probe_pp_prop(model: brainstate.nn.Module):
    return braintrace.pp_prop(
        model,
        decay_or_rank=0.9,
        vjp_method="multi-step",
        config=braintrace.ETraceConfig(
            trace_factorization="io_factorized",
            recurrence_scope="diagonal",
            decay=0.9,
        ),
    )


def _depth_objective_probe_inputs(
    effort: int,
) -> tuple[depth.DepthGateConfig, jax.Array, jax.Array]:
    config = depth.DepthGateConfig(
        training_updates=1,
        batch_size=1,
        validation_episodes=1,
        neuron_count=64,
        recurrent_edges=1,
        readout_width=2,
        color_rank=1,
        context_memory_width=2,
        staging_chunk_updates=1,
    )
    mapping = np.asarray(depth.unrank_ten_cycle(12_345))
    query = 7
    events = np.zeros((19, 1, config.row_config.input_width), dtype=np.float32)
    events[:11, 0] = _encoded_cycle_episode(
        mapping,
        query,
        np.arange(10, dtype=np.int32),
        config.row_config,
    )
    contract = depth._checkpoint_contract(mapping, query, effort)
    packed = np.concatenate(
        (
            events,
            np.asarray(contract.advance_mask, dtype=np.float32)[:, None, None],
            np.asarray(contract.targets, dtype=np.float32)[:, None, None],
            np.sqrt(np.asarray(contract.loss_weights, dtype=np.float32))[:, None, None],
        ),
        axis=-1,
    )
    padded = jnp.asarray(packed)
    return config, padded, padded[: contract.active_length]


def _probe_objective(
    config: depth.DepthGateConfig,
    inputs: jax.Array,
) -> float:
    model = _DepthObjectiveProbe(config)
    brainstate.nn.init_all_states(model, batch_size=1)
    outputs = brainstate.transform.for_loop(model, inputs)
    return float(jnp.square(outputs).sum())


def test_depth_gate_config_is_exact_preregistered_contract() -> None:
    config = depth.DepthGateConfig()

    assert depth.QUALIFYING_EFFORTS == (1, 2, 4, 8)
    assert depth.TEN_CYCLE_CATALOG_SIZE == math.factorial(9) == 362_880
    assert depth.STAGING_CHUNK_UPDATES == 128
    assert depth.STAGING_CHUNK_COUNT == 32
    assert depth.GATE_B_SCHEMA_VERSION == 1
    assert depth.GATE_B_CONTROL == "example21_demonstrated_depth_gate_b"
    assert config.training_updates == 4_096
    assert config.batch_size == 64
    assert config.validation_episodes == 512
    assert config.gap_steps == 8
    assert config.neuron_count == 2_048
    assert config.recurrent_edges == 16_384
    assert config.readout_width == 128
    assert config.color_rank == 16
    assert config.context_memory_width == 32
    assert config.memory_decay == 1.0
    assert config.trace_decay == 0.9
    assert config.learning_rate == 0.003
    assert config.clip_norm == 1.0
    assert config.input_gain == 4.0
    assert config.recurrent_gain == 0.8
    assert config.model_seed == 2_108
    assert config.catalog_seed == 20_260_818
    assert config.train_episode_seed == 32_021
    assert config.validation_episode_seed == 92_021
    assert config.staging_chunk_updates == 128
    assert config.row_config.max_demonstrations == 10
    assert config.row_config.input_width == 47
    assert config.sequence_length == 19
    assert config.training_episode_count == 262_144
    assert config.staging_chunk_count == 32
    assert config.qualification_regime == "preregistered_full"
    assert config.gate_a_result_sha256 == (
        "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632"
    )
    assert config.gate_a_manifest_sha256 == (
        "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf"
    )
    assert config.gate_a_source_commit == "4737e9172b1c6ca99347af5b2c83fc795a294a16"
    assert config.model_source_sha256 == (
        "467022c79123b976dd5cebc8d5ae5da37d1373bc46477133003b0b263abd8216"
    )
    assert config.task_source_sha256 == (
        "cfaec054bd42f6dccf9fb24c5fbec0cd703fdef17ba8d3b6dd68bf78366de18b"
    )
    assert dataclasses.replace(config, training_updates=4).qualification_regime == (
        "nonqualifying_abbreviated"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_updates", 4_095),
        ("batch_size", 32),
        ("validation_episodes", 256),
        ("neuron_count", 1_024),
        ("recurrent_edges", 8_192),
        ("readout_width", 64),
        ("color_rank", 8),
        ("context_memory_width", 16),
        ("memory_decay", 0.95),
        ("trace_decay", 0.8),
        ("learning_rate", 0.001),
        ("clip_norm", 0.5),
        ("input_gain", 3.0),
        ("recurrent_gain", 0.7),
        ("model_seed", 2_109),
        ("catalog_seed", 20_260_819),
        ("train_episode_seed", 32_022),
        ("validation_episode_seed", 92_022),
        ("staging_chunk_updates", 64),
    ],
)
def test_any_mutable_frozen_config_coordinate_change_is_nonqualifying(
    field: str,
    value: object,
) -> None:
    changed = dataclasses.replace(depth.DepthGateConfig(), **{field: value})

    assert changed.qualification_regime == "nonqualifying_abbreviated"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gap_steps", 4),
        (
            "row_config",
            RowEventConfig(max_demonstrations=9, max_grid_size=1),
        ),
        (
            "row_config",
            RowEventConfig(max_demonstrations=10, max_grid_size=2),
        ),
    ],
)
def test_incompatible_frozen_layout_change_fails_closed(
    field: str,
    value: object,
) -> None:
    try:
        changed = dataclasses.replace(depth.DepthGateConfig(), **{field: value})
    except ValueError:
        return

    assert changed.qualification_regime == "nonqualifying_abbreviated"


@pytest.mark.parametrize("mapping_id", [0, 1, 9, 10, 12_345, math.factorial(9) - 1])
def test_ten_cycle_catalog_unranking_is_bijective(mapping_id: int) -> None:
    mapping = np.asarray(depth.unrank_ten_cycle(mapping_id))

    assert mapping.shape == (10,)
    assert np.issubdtype(mapping.dtype, np.integer)
    assert sorted(mapping.tolist()) == list(range(10))
    assert _is_single_ten_cycle(mapping)
    assert depth.rank_ten_cycle(mapping) == mapping_id


@pytest.mark.parametrize(
    ("mapping_id", "expected"),
    [
        (0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        (1, [1, 2, 3, 4, 5, 6, 7, 9, 0, 8]),
        (math.factorial(9) - 1, [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_ten_cycle_unranking_uses_anchored_lexicographic_lehmer_order(
    mapping_id: int, expected: list[int]
) -> None:
    assert np.asarray(depth.unrank_ten_cycle(mapping_id)).tolist() == expected


@pytest.mark.parametrize("mapping_id", [-1, math.factorial(9)])
def test_ten_cycle_unranking_rejects_out_of_catalog_ids(mapping_id: int) -> None:
    with pytest.raises(ValueError, match="catalog|mapping"):
        depth.unrank_ten_cycle(mapping_id)


@pytest.fixture(scope="module")
def production_schedule() -> depth.DepthSchedule:
    return depth._build_schedule(depth.DepthGateConfig())


def test_frozen_schedule_is_balanced_unique_and_disjoint(
    production_schedule: depth.DepthSchedule,
) -> None:
    schedule = production_schedule
    training_ids = np.asarray(schedule.training_mapping_ids)
    validation_ids = np.asarray(schedule.validation_mapping_ids)
    efforts = np.asarray(schedule.training_efforts)
    training_queries = np.asarray(schedule.training_query_colors)
    validation_queries = np.asarray(schedule.validation_query_colors)
    training_orders = np.asarray(schedule.training_presentation_orders)
    validation_orders = np.asarray(schedule.validation_presentation_orders)

    assert training_ids.shape == (4_096, 64)
    assert validation_ids.shape == (512,)
    assert efforts.shape == (4_096,)
    assert training_queries.shape == (4_096, 64)
    assert validation_queries.shape == (512,)
    assert training_orders.shape == (4_096, 64, 10)
    assert validation_orders.shape == (512, 10)
    assert np.array_equal(efforts, np.resize([1, 2, 4, 8], 4_096))
    assert np.array_equal(
        np.bincount(efforts, minlength=9)[[1, 2, 4, 8]],
        np.full((4,), 1_024),
    )
    flat_training_ids = training_ids.reshape(-1)
    assert np.unique(flat_training_ids).size == 262_144
    assert np.unique(validation_ids).size == 512
    assert not np.intersect1d(flat_training_ids, validation_ids).size
    assert flat_training_ids.min() >= 0
    assert validation_ids.min() >= 0
    assert flat_training_ids.max() < math.factorial(9)
    assert validation_ids.max() < math.factorial(9)
    assert flat_training_ids[:8].tolist() == [
        42_599,
        59_110,
        75_621,
        92_132,
        108_643,
        125_154,
        141_665,
        158_176,
    ]
    assert validation_ids[:4].tolist() == [232_423, 248_934, 265_445, 281_956]
    assert _array_digest(flat_training_ids) == (
        "b604a27206a0f64d222cb06530586622522b4f0951579f3aa0132a52e541381d"
    )
    assert _array_digest(validation_ids) == (
        "b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8"
    )
    assert np.array_equal(training_queries.reshape(-1), np.arange(262_144) % 10)
    assert np.array_equal(validation_queries, (262_144 + np.arange(512)) % 10)
    for queries in (training_queries.reshape(-1), validation_queries):
        counts = np.bincount(queries, minlength=10)
        assert int(counts.max() - counts.min()) <= 1
    assert training_orders.reshape(-1, 10)[:3].tolist() == [
        [3, 5, 0, 1, 4, 6, 2, 8, 7, 9],
        [3, 9, 2, 8, 0, 7, 4, 1, 6, 5],
        [6, 4, 7, 1, 0, 3, 5, 8, 2, 9],
    ]
    assert validation_orders[:3].tolist() == [
        [6, 2, 5, 3, 8, 7, 4, 9, 0, 1],
        [3, 4, 2, 7, 5, 9, 0, 1, 6, 8],
        [3, 7, 8, 9, 4, 0, 5, 2, 6, 1],
    ]
    assert _array_digest(training_orders.reshape(-1, 10)) == (
        "79f8c7385699f29595e6fd99ff2e2e0feb56e6f69469ae3b16b4d7f4b8ae588d"
    )
    assert _array_digest(validation_orders) == (
        "0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0"
    )


def test_frozen_schedule_is_sensitive_to_each_declared_seed(
    production_schedule: depth.DepthSchedule,
) -> None:
    config = depth.DepthGateConfig()
    catalog_changed = depth._build_schedule(
        dataclasses.replace(config, catalog_seed=config.catalog_seed + 1)
    )
    train_changed = depth._build_schedule(
        dataclasses.replace(config, train_episode_seed=config.train_episode_seed + 1)
    )
    validation_changed = depth._build_schedule(
        dataclasses.replace(
            config,
            validation_episode_seed=config.validation_episode_seed + 1,
        )
    )

    assert not np.array_equal(
        catalog_changed.training_mapping_ids,
        production_schedule.training_mapping_ids,
    )
    assert not np.array_equal(
        train_changed.training_presentation_orders,
        production_schedule.training_presentation_orders,
    )
    assert not np.array_equal(
        validation_changed.validation_presentation_orders,
        production_schedule.validation_presentation_orders,
    )
    assert np.array_equal(
        train_changed.training_mapping_ids,
        production_schedule.training_mapping_ids,
    )
    assert np.array_equal(
        validation_changed.validation_mapping_ids,
        production_schedule.validation_mapping_ids,
    )


def test_schedule_is_deterministic_and_chunking_is_identity(
    production_schedule: depth.DepthSchedule,
) -> None:
    config = depth.DepthGateConfig()
    repeated = depth._build_schedule(config)
    for field in dataclasses.fields(depth.DepthSchedule):
        assert np.array_equal(
            np.asarray(getattr(production_schedule, field.name)),
            np.asarray(getattr(repeated, field.name)),
        )

    chunks = tuple(depth._iter_schedule_chunks(production_schedule, config))
    assert len(chunks) == 32
    assert all(
        np.asarray(chunk.training_mapping_ids).shape == (128, 64) for chunk in chunks
    )
    assert all(np.asarray(chunk.training_efforts).shape == (128,) for chunk in chunks)
    assert all(
        np.asarray(chunk.training_presentation_orders).shape == (128, 64, 10)
        for chunk in chunks
    )
    assert np.array_equal(
        np.concatenate([np.asarray(chunk.training_mapping_ids) for chunk in chunks]),
        np.asarray(production_schedule.training_mapping_ids),
    )
    assert np.array_equal(
        np.concatenate([np.asarray(chunk.training_efforts) for chunk in chunks]),
        np.asarray(production_schedule.training_efforts),
    )
    assert np.array_equal(
        np.concatenate([np.asarray(chunk.training_query_colors) for chunk in chunks]),
        np.asarray(production_schedule.training_query_colors),
    )
    assert np.array_equal(
        np.concatenate(
            [np.asarray(chunk.training_presentation_orders) for chunk in chunks]
        ),
        np.asarray(production_schedule.training_presentation_orders),
    )


def test_training_chunk_encoding_matches_row_events_and_checkpoint_contracts() -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)
    schedule_chunk = next(depth._iter_schedule_chunks(schedule, config))

    encoded = depth._encode_training_chunk(schedule_chunk, config)

    assert isinstance(encoded, depth.DepthTrainingChunk)
    assert np.asarray(encoded.events).shape == (1, 19, 2, 47)
    assert np.asarray(encoded.targets).shape == (1, 19, 2)
    assert np.asarray(encoded.loss_weights).shape == (1, 19)
    assert np.asarray(encoded.advance_masks).shape == (1, 19, 2)
    assert np.array_equal(encoded.mapping_ids, schedule_chunk.training_mapping_ids)
    assert np.array_equal(encoded.efforts, schedule_chunk.training_efforts)
    assert np.array_equal(encoded.query_colors, schedule_chunk.training_query_colors)
    assert np.array_equal(
        encoded.presentation_orders,
        schedule_chunk.training_presentation_orders,
    )

    effort = int(schedule_chunk.training_efforts[0])
    for batch_index in range(config.batch_size):
        mapping = np.asarray(
            depth.unrank_ten_cycle(
                int(schedule_chunk.training_mapping_ids[0, batch_index])
            )
        )
        query = int(schedule_chunk.training_query_colors[0, batch_index])
        expected_events = np.zeros((19, 47), dtype=np.float32)
        expected_events[:11] = _encoded_cycle_episode(
            mapping,
            query,
            schedule_chunk.training_presentation_orders[0, batch_index],
            config.row_config,
        )
        contract = depth._checkpoint_contract(mapping, query, effort)

        assert np.array_equal(encoded.events[0, :, batch_index], expected_events)
        assert np.array_equal(encoded.targets[0, :, batch_index], contract.targets)
        assert np.array_equal(
            encoded.advance_masks[0, :, batch_index],
            contract.advance_mask,
        )
        assert np.array_equal(encoded.loss_weights[0], contract.loss_weights)


def test_validation_encoding_contains_exact_intact_and_control_episodes() -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)

    data = depth._encode_validation_data(schedule, config)

    assert isinstance(data, depth.DepthValidationData)
    assert np.asarray(data.intact).shape == (19, 2, 47)
    assert np.asarray(data.shuffled).shape == (19, 2, 47)
    assert np.asarray(data.no_context).shape == (19, 2, 47)
    assert np.asarray(data.targets_by_depth).shape == (9, 2)
    assert np.asarray(data.advance_masks).shape == (19, 2)
    assert np.all(data.advance_masks)
    assert np.array_equal(data.mapping_ids, schedule.validation_mapping_ids)
    assert np.array_equal(data.query_colors, schedule.validation_query_colors)
    assert np.array_equal(
        data.presentation_orders,
        schedule.validation_presentation_orders,
    )
    assert np.asarray(data.shuffled_shifts).shape == (2,)

    for episode_index in range(config.validation_episodes):
        mapping = np.asarray(
            depth.unrank_ten_cycle(int(schedule.validation_mapping_ids[episode_index]))
        )
        query = int(schedule.validation_query_colors[episode_index])
        order = schedule.validation_presentation_orders[episode_index]
        shift, shuffled_mapping = depth._select_shuffled_rotation(mapping, query)
        expected_intact = np.zeros((19, 47), dtype=np.float32)
        expected_shuffled = np.zeros_like(expected_intact)
        expected_no_context = np.zeros_like(expected_intact)
        expected_intact[:11] = _encoded_cycle_episode(
            mapping,
            query,
            order,
            config.row_config,
        )
        expected_shuffled[:11] = _encoded_cycle_episode(
            shuffled_mapping,
            query,
            order,
            config.row_config,
        )
        expected_no_context[10] = expected_intact[10]
        expected_targets = np.asarray(
            [_iterate(mapping, query, depth_index + 1) for depth_index in range(9)],
            dtype=np.int32,
        )

        assert int(data.shuffled_shifts[episode_index]) == shift
        assert np.array_equal(data.intact[:, episode_index], expected_intact)
        assert np.array_equal(data.shuffled[:, episode_index], expected_shuffled)
        assert np.array_equal(data.no_context[:, episode_index], expected_no_context)
        assert np.array_equal(data.targets_by_depth[:, episode_index], expected_targets)
        assert np.array_equal(
            data.intact[10, episode_index], data.shuffled[10, episode_index]
        )
        assert np.array_equal(
            data.intact[10, episode_index], data.no_context[10, episode_index]
        )


def test_model_config_preserves_gate_b_semantic_indices_and_topology() -> None:
    config = _reduced_depth_config()
    model_config = depth._model_config(config, batch_size=3)
    indices = associative_memory_feature_indices(config.row_config)

    assert model_config.input_width == 47
    assert model_config.batch_size == 3
    assert model_config.neuron_count == config.neuron_count
    assert model_config.recurrent_edges == config.recurrent_edges
    assert model_config.max_latent_steps == 8
    assert model_config.readout_width == config.readout_width
    assert model_config.color_rank == config.color_rank
    assert model_config.context_memory_width == config.context_memory_width
    assert model_config.memory_decay == config.memory_decay
    assert model_config.trace_decay == config.trace_decay
    assert model_config.event_valid_index == config.row_config.valid_slice.start
    assert model_config.demonstration_phase_index == config.row_config.phase_slice.start
    assert model_config.query_phase_index == config.row_config.phase_slice.start + 1
    assert (
        model_config.input_side_valid_index == config.row_config.side_valid_slice.start
    )
    assert (
        model_config.output_side_valid_index
        == config.row_config.side_valid_slice.start + 1
    )
    assert model_config.memory_key_indices == indices.key_indices
    assert model_config.memory_value_indices == indices.value_indices
    assert model_config.seed == config.model_seed


def test_training_driver_is_one_jit_with_internal_brainstate_for_loop() -> None:
    source = inspect.getsource(depth._make_pp_prop_trainer)
    function = ast.parse(source).body[0]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    nested = {
        node.name: node
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    train_chunk = nested["train_chunk"]

    assert len(train_chunk.decorator_list) == 1
    assert ast.unparse(train_chunk.decorator_list[0]) == "brainstate.transform.jit"
    assert not any(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(train_chunk)
    )
    for_loop_calls = [
        node
        for node in ast.walk(train_chunk)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "brainstate.transform.for_loop"
    ]
    assert len(for_loop_calls) == 1
    assert ast.unparse(for_loop_calls[0].args[0]) == "train_one"
    assert ast.unparse(for_loop_calls[0].args[1]) == (
        "(events, targets, loss_weights, advance_masks)"
    )


@pytest.mark.parametrize("effort", [1, 2, 4, 8])
def test_padded_t19_matches_compact_prefix_finite_window_pp_prop(
    effort: int,
) -> None:
    config, padded, compact = _depth_objective_probe_inputs(effort)

    padded_objective = _probe_objective(config, padded)
    compact_objective = _probe_objective(config, compact)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        padded_gradients = chunked_online_param_gradients(
            lambda: _DepthObjectiveProbe(config),
            padded,
            algo_factory=_depth_probe_pp_prop,
            chunk_size=1,
        )
        compact_gradients = chunked_online_param_gradients(
            lambda: _DepthObjectiveProbe(config),
            compact,
            algo_factory=_depth_probe_pp_prop,
            chunk_size=1,
        )

    assert padded_objective == pytest.approx(compact_objective, rel=0.0, abs=1e-7)
    assert gradient_norm(compact_gradients) > 1e-8
    assert relative_deviation(padded_gradients, compact_gradients) == pytest.approx(
        0.0,
        abs=1e-7,
    )
    padded_leaves = flat_gradient_leaves(padded_gradients)
    compact_leaves = flat_gradient_leaves(compact_gradients)
    assert padded_leaves.keys() == compact_leaves.keys()
    for path in padded_leaves:
        np.testing.assert_allclose(
            padded_leaves[path],
            compact_leaves[path],
            rtol=0.0,
            atol=1e-7,
            err_msg=path,
        )


@pytest.fixture(scope="module")
def reduced_gate_run() -> dict[str, Any]:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)
    chunk = depth._encode_training_chunk(
        next(depth._iter_schedule_chunks(schedule, config)),
        config,
    )
    validation = depth._encode_validation_data(schedule, config)
    model = LatentWorkspaceModel(
        depth._model_config(config, batch_size=config.batch_size)
    )

    initialization = depth._initialization_report(model, config)
    trainer = depth._make_pp_prop_trainer(model, config)
    telemetry = trainer.train_chunk(
        chunk.events,
        chunk.targets,
        chunk.loss_weights,
        chunk.advance_masks,
    )
    evaluation = depth._evaluate_model(model, validation, config)
    return {
        "config": config,
        "initialization": initialization,
        "trainer": trainer,
        "telemetry": telemetry,
        "evaluation": evaluation,
    }


def test_reduced_gate_b_runs_real_pp_prop_train_and_evaluation_smoke(
    reduced_gate_run: dict[str, Any],
) -> None:
    config = reduced_gate_run["config"]
    initialization = reduced_gate_run["initialization"]
    trainer = reduced_gate_run["trainer"]
    telemetry = reduced_gate_run["telemetry"]
    evaluation = reduced_gate_run["evaluation"]

    assert initialization["fresh_model"] is True
    assert initialization["model_seed"] == config.model_seed
    assert initialization["configuration"] == dataclasses.asdict(config)
    assert initialization["parameter_count"] > 0
    assert len(initialization["parameter_sha256"]) == 64
    assert initialization["compiler"]["available"] is True
    assert trainer.algorithm == "production_pp_prop"
    assert trainer.compiler["available"] is True
    assert np.asarray(telemetry["loss"]).shape == (1,)
    assert np.isfinite(np.asarray(telemetry["loss"])).all()
    assert isinstance(evaluation, Mapping)
    assert evaluation["finite"] is True
    assert set(evaluation) == {"finite", "h0_proper", "depths", "efforts"}
    assert set(evaluation["efforts"]) == {"1", "2", "4", "8"}
    h0_hash = evaluation["h0_proper"]["prediction_sha256"]
    assert evaluation["h0_proper"]["checkpoint"] == 0
    assert len(h0_hash) == 64
    for effort in depth.QUALIFYING_EFFORTS:
        evidence = evaluation["efforts"][str(effort)]
        assert set(evidence) == {
            "intact",
            "shuffled",
            "no_context",
            "h0_final_target",
            "intact_minus_h0",
            "intact_minus_shuffled",
        }
        assert evidence["intact"]["checkpoint"] == effort
        assert evidence["shuffled"]["checkpoint"] == effort
        assert evidence["no_context"]["checkpoint"] == effort
        assert evidence["h0_final_target"]["checkpoint"] == 0
        assert evidence["h0_final_target"]["prediction_sha256"] == h0_hash
        for metric_name in ("intact", "shuffled", "no_context", "h0_final_target"):
            metric = evidence[metric_name]
            assert metric["count"] == config.validation_episodes
            assert len(metric["prediction_sha256"]) == 64
            assert np.isfinite(metric["accuracy"])
        assert np.isfinite(evidence["intact_minus_h0"])
        assert np.isfinite(evidence["intact_minus_shuffled"])


def test_train_chunk_retains_complete_finite_telemetry(
    reduced_gate_run: dict[str, Any],
) -> None:
    telemetry = reduced_gate_run["telemetry"]
    categories = {
        "logits",
        "model_states",
        "gradients",
        "pp_prop_traces",
        "adam",
        "parameters",
    }

    assert set(telemetry) == {"loss", "finite", "max_abs", "value_count"}
    for section in ("finite", "max_abs", "value_count"):
        assert set(telemetry[section]) == categories
    for category in categories:
        finite = np.asarray(telemetry["finite"][category])
        maximum = np.asarray(telemetry["max_abs"][category])
        count = np.asarray(telemetry["value_count"][category])
        assert finite.shape == maximum.shape == count.shape == (1,)
        assert np.issubdtype(finite.dtype, np.bool_)
        assert np.all(finite)
        assert np.all(np.isfinite(maximum))
        assert np.all(maximum >= 0.0)
        assert np.issubdtype(count.dtype, np.integer)
        assert np.all(count > 0)


def test_evaluation_retains_all_checkpoint_metrics_from_shared_predictions(
    reduced_gate_run: dict[str, Any],
) -> None:
    config = reduced_gate_run["config"]
    evaluation = reduced_gate_run["evaluation"]
    metric_keys = {
        "correct",
        "count",
        "accuracy",
        "wilson_95_lower",
        "wilson_95_upper",
        "prediction_histogram",
        "prediction_sha256",
        "checkpoint",
    }

    assert set(evaluation["depths"]) == {str(index) for index in range(9)}
    assert evaluation["h0_proper"] == evaluation["depths"]["0"]["intact"]
    for depth_index in range(9):
        evidence = evaluation["depths"][str(depth_index)]
        assert set(evidence) == {"intact", "shuffled", "no_context"}
        for arm in ("intact", "shuffled", "no_context"):
            metric = evidence[arm]
            assert set(metric) == metric_keys
            assert metric["checkpoint"] == depth_index
            assert metric["count"] == config.validation_episodes
            assert 0 <= metric["correct"] <= metric["count"]
            assert metric["accuracy"] == pytest.approx(
                metric["correct"] / metric["count"],
                rel=0.0,
                abs=0.0,
            )
            assert len(metric["prediction_histogram"]) == 10
            assert sum(metric["prediction_histogram"]) == metric["count"]
            assert len(metric["prediction_sha256"]) == 64
            assert 0.0 <= metric["wilson_95_lower"] <= metric["accuracy"]
            assert metric["accuracy"] <= math.nextafter(
                metric["wilson_95_upper"], math.inf
            )
            assert metric["wilson_95_upper"] <= 1.0

    h0_hash = evaluation["h0_proper"]["prediction_sha256"]
    for effort in depth.QUALIFYING_EFFORTS:
        effort_evidence = evaluation["efforts"][str(effort)]
        depth_evidence = evaluation["depths"][str(effort)]
        for arm in ("intact", "shuffled", "no_context"):
            assert effort_evidence[arm] == depth_evidence[arm]
        assert effort_evidence["h0_final_target"]["prediction_sha256"] == h0_hash


def test_first_valid_rotation_preserves_marginal_and_breaks_every_depth(
    production_schedule: depth.DepthSchedule,
) -> None:
    for mapping_id, query_color in zip(
        np.asarray(production_schedule.validation_mapping_ids),
        np.asarray(production_schedule.validation_query_colors),
        strict=True,
    ):
        mapping = np.asarray(depth.unrank_ten_cycle(int(mapping_id)))
        shift, shuffled = depth._select_shuffled_rotation(mapping, int(query_color))
        shuffled = np.asarray(shuffled)
        valid_shifts = [
            candidate
            for candidate in range(1, 10)
            if np.all((mapping + candidate) % 10 != mapping)
            and all(
                _iterate((mapping + candidate) % 10, int(query_color), effort + 1)
                != _iterate(mapping, int(query_color), effort + 1)
                for effort in depth.QUALIFYING_EFFORTS
            )
        ]

        assert valid_shifts
        assert shift == valid_shifts[0]
        assert np.array_equal(shuffled, (mapping + shift) % 10)
        assert sorted(shuffled.tolist()) == list(range(10))
        assert np.all(shuffled != mapping)
        assert all(
            _iterate(shuffled, int(query_color), effort + 1)
            != _iterate(mapping, int(query_color), effort + 1)
            for effort in depth.QUALIFYING_EFFORTS
        )


@pytest.mark.parametrize("effort", [1, 2, 4, 8])
def test_checkpoint_contract_has_exact_targets_masks_and_compact_prefix(
    effort: int,
) -> None:
    mapping = np.asarray(depth.unrank_ten_cycle(12_345))
    query_color = 7
    contract = depth._checkpoint_contract(mapping, query_color, effort)
    targets = np.asarray(contract.targets)
    weights = np.asarray(contract.loss_weights)
    advances = np.asarray(contract.advance_mask)
    active_length = 11 + effort

    assert isinstance(contract, depth.CheckpointContract)
    assert targets.shape == weights.shape == advances.shape == (19,)
    assert np.issubdtype(targets.dtype, np.integer)
    assert np.issubdtype(weights.dtype, np.floating)
    assert np.issubdtype(advances.dtype, np.bool_)
    assert contract.active_length == active_length
    assert np.all(advances[:active_length])
    assert not np.any(advances[active_length:])
    assert np.all(weights[:10] == 0.0)
    assert np.all(weights[active_length:] == 0.0)
    assert np.allclose(
        weights[10:active_length],
        np.full((effort + 1,), 1.0 / (effort + 1)),
        rtol=0.0,
        atol=0.0,
    )
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-7)
    expected = np.asarray(
        [
            _iterate(mapping, query_color, applications)
            for applications in range(1, effort + 2)
        ]
    )
    assert np.array_equal(targets[10:active_length], expected)
    assert targets[10] == mapping[query_color]
    assert targets[10 + effort] != targets[10]


@pytest.mark.parametrize("effort", [0, 3, 9])
def test_checkpoint_contract_rejects_undeclared_effort(effort: int) -> None:
    with pytest.raises(ValueError, match="effort"):
        depth._checkpoint_contract(np.asarray(depth.unrank_ten_cycle(0)), 0, effort)


def test_missing_gate_b_evidence_fails_closed() -> None:
    qualification = depth._qualification_report({}, config=depth.DepthGateConfig())

    assert qualification["passed"] is False
    assert qualification["interpretation"] == (
        "gate_b_failed_stop_no_capability_conclusion"
    )
    assert qualification["criteria"]
    assert not any(qualification["criteria"].values())
