"""Tests for Example 21's recurrent spiking ARC workspace."""

from __future__ import annotations

import ast
import inspect
import math
import warnings

import brainpy.state as bpstate
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from examples.pp_prop import latent_workspace_model as latent_workspace_module
    from examples.pp_prop.latent_workspace_model import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        MEMORY_KEY_RFF_GAMMA,
        NEURONS_PER_SLOT,
        ArcLogits,
        AssociativeMemoryReport,
        LatentWorkspaceModel,
        ModelConfig,
        ModelStateSnapshot,
        SelectedPackedTrajectory,
        arc_loss_components,
        arc_loss_per_example,
        build_sparse_topology,
        compact_output_width,
        compile_pp_prop,
        expand_compact_logits,
        parameter_snapshot,
        run_context,
        run_latent_trajectory,
        run_packed_stream,
        run_selected_packed_stream,
        run_sequence,
        terminal_arc_loss,
    )
except ImportError:
    import latent_workspace_model as latent_workspace_module
    from latent_workspace_model import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        MEMORY_KEY_RFF_GAMMA,
        NEURONS_PER_SLOT,
        ArcLogits,
        AssociativeMemoryReport,
        LatentWorkspaceModel,
        ModelConfig,
        ModelStateSnapshot,
        SelectedPackedTrajectory,
        arc_loss_components,
        arc_loss_per_example,
        build_sparse_topology,
        compact_output_width,
        compile_pp_prop,
        expand_compact_logits,
        parameter_snapshot,
        run_context,
        run_latent_trajectory,
        run_packed_stream,
        run_selected_packed_stream,
        run_sequence,
        terminal_arc_loss,
    )

try:
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        associative_memory_feature_indices,
        encode_query_episode,
    )
except ImportError:
    from latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        associative_memory_feature_indices,
        encode_query_episode,
    )


def _config(**changes: object) -> ModelConfig:
    values: dict[str, object] = {
        "input_width": 6,
        "neuron_count": 64,
        "recurrent_edges": 96,
        "max_latent_steps": 4,
        "readout_width": 8,
        "color_rank": 2,
        "seed": 41,
    }
    values.update(changes)
    return ModelConfig(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def model() -> LatentWorkspaceModel:
    return LatentWorkspaceModel(_config())


def _tree_arrays(value: object) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(u.get_mantissa(leaf)) for leaf in jax.tree.leaves(value))


def _assert_parameter_snapshots_equal(
    left: dict[str, object], right: dict[str, object]
) -> None:
    assert left.keys() == right.keys()
    for name, value in left.items():
        left_arrays = _tree_arrays(value)
        right_arrays = _tree_arrays(right[name])
        assert len(left_arrays) == len(right_arrays)
        for first, second in zip(left_arrays, right_arrays, strict=True):
            np.testing.assert_array_equal(first, second, err_msg=name)


def _valid_events(time: int, width: int) -> jax.Array:
    events = jnp.zeros((time, width), dtype=jnp.float32)
    return events.at[:, 0].set(1.0).at[:, 1].set(1.0)


def _memory_config(**changes: object) -> ModelConfig:
    values: dict[str, object] = {
        "input_width": 9,
        "neuron_count": 64,
        "recurrent_edges": 96,
        "max_latent_steps": 4,
        "readout_width": 8,
        "color_rank": 2,
        "seed": 41,
        "context_memory_width": 2,
        "memory_decay": 1.0,
        "demonstration_phase_index": 1,
        "query_phase_index": 2,
        "input_side_valid_index": 3,
        "output_side_valid_index": 4,
        "memory_key_indices": (5, 6),
        "memory_value_indices": (7, 8),
    }
    values.update(changes)
    return ModelConfig(**values)  # type: ignore[arg-type]


def _phase_events(
    config: ModelConfig,
    keys: jax.Array,
    values: jax.Array | None = None,
    *,
    phase: str,
) -> jax.Array:
    keys = jnp.asarray(keys, dtype=jnp.float32)
    if keys.ndim == 2:
        keys = keys[:, None, :]
    time_steps, batch_size, _ = keys.shape
    events = jnp.zeros((time_steps, batch_size, config.input_width), dtype=jnp.float32)
    events = events.at[..., config.event_valid_index].set(1.0)
    if phase == "demonstration":
        phase_index = config.demonstration_phase_index
    elif phase == "query":
        phase_index = config.query_phase_index
    else:
        raise ValueError(f"unknown phase {phase!r}")
    assert phase_index is not None
    events = events.at[..., phase_index].set(1.0)
    assert config.input_side_valid_index is not None
    events = events.at[..., config.input_side_valid_index].set(1.0)
    events = events.at[..., jnp.asarray(config.memory_key_indices)].set(keys)
    if values is not None:
        values = jnp.asarray(values, dtype=jnp.float32)
        if values.ndim == 2:
            values = values[:, None, :]
        events = events.at[..., jnp.asarray(config.memory_value_indices)].set(values)
        if phase == "demonstration":
            assert config.output_side_valid_index is not None
            events = events.at[..., config.output_side_valid_index].set(1.0)
    return events


def _state_array(state: object) -> np.ndarray:
    return np.asarray(getattr(state, "value"))


def _production_memory_config(
    memory_width: int, *, batch_size: int = 4, seed: int = 2108
) -> ModelConfig:
    rows = RowEventConfig()
    features = associative_memory_feature_indices(rows)
    return ModelConfig(
        input_width=rows.input_width,
        batch_size=batch_size,
        neuron_count=64,
        recurrent_edges=96,
        max_latent_steps=4,
        readout_width=8,
        color_rank=2,
        seed=seed,
        context_memory_width=memory_width,
        memory_decay=1.0,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        memory_key_indices=features.key_indices,
        memory_value_indices=features.value_indices,
    )


def _production_k4_events() -> tuple[jax.Array, jax.Array]:
    rows = RowEventConfig()
    demonstration_events = []
    query_events = []
    for color in range(4):
        task = ArcTask(
            train=(
                ArcPair(
                    ArcGrid(((color,),)),
                    ArcGrid(((color + 4,),)),
                ),
            ),
            test=(ArcPair(ArcGrid(((color,),)), None),),
            task_id=f"k4-{color}",
        )
        encoded = encode_query_episode(task, 0, rows)
        demonstration_events.append(encoded.events[0])
        query_events.append(
            encoded.events[rows.max_demonstrations * rows.max_grid_size]
        )
    return (
        jnp.asarray(np.stack(demonstration_events)),
        jnp.asarray(np.stack(query_events)),
    )


def test_full_configuration_is_2048_neurons_16384_edges_and_32_slots() -> None:
    config = ModelConfig(input_width=830)

    assert config.neuron_count == 2048
    assert config.recurrent_edges == 16384
    assert config.max_latent_steps == 32
    assert config.slot_count == 32
    assert config.slot_count * NEURONS_PER_SLOT == config.neuron_count


def test_full_topology_has_exact_unique_no_self_edge_count() -> None:
    topology = build_sparse_topology(2048, 16384, seed=2108)
    flat = topology.rows.astype(np.int64) * 2048 + topology.columns

    assert topology.edge_count == 16384
    assert np.unique(flat).size == 16384
    assert not np.any(topology.rows == topology.columns)
    assert topology.rows.flags.writeable is False
    assert topology.columns.flags.writeable is False
    assert topology.values.flags.writeable is False


def test_full_model_instantiates_physical_components() -> None:
    full = LatentWorkspaceModel(ModelConfig(input_width=4))

    assert full.neuron_count == 2048
    assert full.recurrent_edge_count == 16384
    assert full.slot_count == 32
    assert full.neu.varshape == (2048,)
    assert isinstance(full.neu, bpstate.LIF)
    assert isinstance(full.ff_syn, bpstate.AlignPostProj)
    assert isinstance(full.ff_syn.syn, bpstate.Expon)
    assert isinstance(full.ff_syn.out, bpstate.CUBA)
    assert isinstance(full.rec_syn, bpstate.AlignPostProj)
    assert isinstance(full.rec_syn.syn, bpstate.Expon)
    assert isinstance(full.rec_syn.out, bpstate.CUBA)


def test_model_uses_braintrace_dense_and_sparse_operators(
    model: LatentWorkspaceModel,
) -> None:
    import braintrace

    assert isinstance(model.ff_syn.comm, braintrace.nn.Linear)
    assert isinstance(model.rec_syn.comm, braintrace.nn.SparseLinear)
    assert isinstance(model.readout_projection, braintrace.nn.Linear)
    assert isinstance(model.height_head, braintrace.nn.Linear)
    assert isinstance(model.width_head, braintrace.nn.Linear)
    assert isinstance(model.color_factor_head, braintrace.nn.Linear)


def test_topology_is_deterministic_by_seed_and_seed_sensitive() -> None:
    first = build_sparse_topology(65, 257, seed=9)
    repeated = build_sparse_topology(65, 257, seed=9)
    changed = build_sparse_topology(65, 257, seed=10)

    np.testing.assert_array_equal(first.rows, repeated.rows)
    np.testing.assert_array_equal(first.columns, repeated.columns)
    np.testing.assert_array_equal(first.values, repeated.values)
    assert not np.array_equal(first.columns, changed.columns)
    assert not np.array_equal(first.values, changed.values)


def test_two_neuron_topology_exercises_single_legal_offset() -> None:
    topology = build_sparse_topology(2, 2, seed=1)

    np.testing.assert_array_equal(topology.rows, np.asarray([0, 1]))
    np.testing.assert_array_equal(topology.columns, np.asarray([1, 0]))


@pytest.mark.parametrize(
    ("changes", "exception", "message"),
    [
        ({"input_width": 0}, ValueError, "input_width"),
        ({"batch_size": True}, TypeError, "batch_size"),
        ({"neuron_count": 65}, ValueError, "divisible"),
        ({"neuron_count": 64, "recurrent_edges": 4033}, ValueError, "capacity"),
        ({"max_latent_steps": 0}, ValueError, "max_latent_steps"),
        ({"readout_width": 0}, ValueError, "readout_width"),
        ({"color_rank": 0}, ValueError, "color_rank"),
        ({"trace_decay": 1.0}, ValueError, "trace_decay"),
        ({"trace_decay": math.nan}, ValueError, "trace_decay"),
        ({"input_gain": 0.0}, ValueError, "input_gain"),
        ({"event_valid_index": 6}, ValueError, "event_valid_index"),
        ({"sparse_backend": 3}, TypeError, "sparse_backend"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_invalid_model_configuration_fails_closed(
    changes: dict[str, object], exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        _config(**changes)


def test_context_memory_is_opt_in_and_legacy_mode_is_the_default() -> None:
    config = _config()

    assert config.context_memory_width == 0
    assert config.memory_decay == 1.0
    assert config.demonstration_phase_index is None
    assert config.query_phase_index is None
    assert config.input_side_valid_index is None
    assert config.output_side_valid_index is None
    assert config.memory_key_indices == ()
    assert config.memory_value_indices == ()


def test_associative_memory_report_is_stable_and_legacy_safe() -> None:
    legacy_model = LatentWorkspaceModel(_config())
    legacy = legacy_model.associative_memory_report()

    assert isinstance(legacy, AssociativeMemoryReport)
    assert legacy.mode == "legacy_reservoir"
    assert legacy.memory_width == 0
    assert legacy.key_basis_sha256 is None
    assert legacy.write_component_type is None

    memory_model = LatentWorkspaceModel(_production_memory_config(32))
    report = memory_model.associative_memory_report()
    repeated = LatentWorkspaceModel(
        _production_memory_config(32)
    ).associative_memory_report()

    assert report == repeated
    assert report.mode == "associative_workspace"
    assert report.memory_width == 32
    assert report.key_feature_width == 424
    assert report.value_feature_width == 424
    assert report.key_map == "fixed_rff_cosine"
    assert report.value_map == "fixed_tanh_projection"
    assert report.rff_gamma == MEMORY_KEY_RFF_GAMMA == 2.0
    assert (
        report.key_basis_seed,
        report.key_bias_seed,
        report.value_basis_seed,
    ) == (2209, 2210, 2211)
    for digest in (
        report.key_basis_sha256,
        report.key_bias_sha256,
        report.value_basis_sha256,
    ):
        assert digest is not None
        assert len(digest) == 64
    assert report.write_component_type == "braintrace.element_wise"
    assert report.query_component_type == "braintrace.nn.Linear"
    assert report.read_component_type == "braintrace.nn.Linear"


def test_width32_production_k4_rff_keys_clear_preregistered_margin() -> None:
    _, query_events = _production_k4_events()
    memory_model = LatentWorkspaceModel(_production_memory_config(32))

    keys = memory_model.encode_memory_key(query_events)
    gram = np.asarray(keys @ keys.T)
    expected = np.asarray(
        [
            [0.74782699, 0.07341479, 0.21628249, -0.18112631],
            [0.07341479, 0.82878375, -0.04830217, 0.24480711],
            [0.21628249, -0.04830217, 0.91903085, -0.28928122],
            [-0.18112631, 0.24480711, -0.28928122, 0.85196108],
        ],
        dtype=np.float32,
    )
    diagonal_min = float(np.diag(gram).min())
    off_diagonal_max = float(gram[~np.eye(4, dtype=np.bool_)].max())
    margin = diagonal_min - off_diagonal_max
    backend_transcendental_tolerance = 7e-4

    np.testing.assert_allclose(
        gram, expected, rtol=0.0, atol=backend_transcendental_tolerance
    )
    assert diagonal_min == pytest.approx(
        0.74782699, abs=backend_transcendental_tolerance
    )
    assert off_diagonal_max == pytest.approx(
        0.24480711, abs=backend_transcendental_tolerance
    )
    assert margin == pytest.approx(0.50301987, abs=1e-3)
    assert margin > 0.25
    zero_codes = memory_model.encode_memory_key(jnp.zeros_like(query_events))
    np.testing.assert_array_equal(zero_codes, 0.0)


def test_untrained_k4_outer_memory_read_is_pairing_sensitive() -> None:
    demonstration_events, query_events = _production_k4_events()
    memory_model = LatentWorkspaceModel(_production_memory_config(32))
    demonstration_keys = memory_model.encode_memory_key(demonstration_events)
    query_keys = memory_model.encode_memory_key(query_events)
    values = memory_model.encode_memory_value(demonstration_events)
    rotated_values = values[jnp.asarray([1, 2, 3, 0])]
    intact_memory = jnp.einsum("bi,bj->ij", demonstration_keys, values)
    shuffled_memory = jnp.einsum(
        "bi,bj->ij", demonstration_keys, rotated_values
    )
    intact_reads = jnp.einsum("ik,kv->iv", query_keys, intact_memory)
    shuffled_reads = jnp.einsum("ik,kv->iv", query_keys, shuffled_memory)

    np.testing.assert_allclose(
        demonstration_keys, query_keys, rtol=0.0, atol=0.0
    )
    assert not np.allclose(intact_memory, shuffled_memory)
    assert np.all(
        np.linalg.norm(np.asarray(intact_reads - shuffled_reads), axis=1) > 1e-3
    )


@pytest.mark.parametrize(
    ("changes", "exception", "message"),
    [
        ({"context_memory_width": True}, TypeError, "context_memory_width"),
        ({"context_memory_width": -1}, ValueError, "context_memory_width"),
        ({"memory_decay": math.nan}, ValueError, "memory_decay"),
        ({"memory_decay": -0.01}, ValueError, "memory_decay"),
        ({"memory_decay": 1.01}, ValueError, "memory_decay"),
        ({"demonstration_phase_index": None}, ValueError, "demonstration_phase_index"),
        ({"query_phase_index": None}, ValueError, "query_phase_index"),
        ({"input_side_valid_index": None}, ValueError, "input_side_valid_index"),
        ({"output_side_valid_index": None}, ValueError, "output_side_valid_index"),
        ({"memory_key_indices": ()}, ValueError, "memory_key_indices"),
        ({"memory_value_indices": ()}, ValueError, "memory_value_indices"),
        ({"memory_key_indices": (5, 9)}, ValueError, "memory_key_indices"),
    ],
)
def test_invalid_context_memory_configuration_fails_closed(
    changes: dict[str, object], exception: type[Exception], message: str
) -> None:
    assert "context_memory_width" in ModelConfig.__dataclass_fields__
    with pytest.raises(exception, match=message):
        _memory_config(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"demonstration_phase_index": 1},
        {"query_phase_index": 2},
        {"input_side_valid_index": 3},
        {"output_side_valid_index": 4},
        {"memory_key_indices": (3,)},
        {"memory_value_indices": (4,)},
    ],
)
def test_legacy_mode_rejects_partial_memory_configuration(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="context_memory_width"):
        _config(**changes)


def test_fast_weight_update_matches_decay_outer_product_and_batch_gate() -> None:
    update_context_memory = getattr(latent_workspace_module, "update_context_memory")
    memory = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=jnp.float32,
    )
    key = jnp.asarray([[2.0, -1.0], [3.0, 4.0]], dtype=jnp.float32)
    value = jnp.asarray([[0.5, 3.0], [-2.0, 1.0]], dtype=jnp.float32)
    write_gate = jnp.asarray([True, False])

    actual = update_context_memory(
        memory, key, value, write_gate=write_gate, decay=0.25
    )
    written = 0.25 * memory[0] + jnp.outer(key[0], value[0])
    expected = jnp.stack((written, memory[1]))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual.dtype == jnp.float32


def test_fast_weight_self_jacobian_is_exactly_decay_times_identity() -> None:
    update_context_memory = getattr(latent_workspace_module, "update_context_memory")
    decay = 0.625
    key = jnp.asarray([[0.25, -0.5]], dtype=jnp.float32)
    value = jnp.asarray([[2.0, 3.0]], dtype=jnp.float32)
    gate = jnp.asarray([True])

    def flattened_update(flat_memory: jax.Array) -> jax.Array:
        memory = flat_memory.reshape(1, 2, 2)
        updated = update_context_memory(
            memory, key, value, write_gate=gate, decay=decay
        )
        return updated.reshape(-1)

    jacobian = jax.jacrev(flattened_update)(jnp.arange(4, dtype=jnp.float32))

    np.testing.assert_allclose(
        jacobian, decay * np.eye(4, dtype=np.float32), rtol=0.0, atol=0.0
    )


def test_memory_states_reset_snapshot_and_restore_exactly() -> None:
    memory_model = LatentWorkspaceModel(_memory_config())
    for name, shape in (
        ("context_memory", (1, 2, 2)),
        ("query_encoding", (1, 2)),
        ("reasoning_query", (1, 2)),
        ("memory_read", (1, 2)),
        ("workspace_carrier", (1, 64)),
    ):
        state = getattr(memory_model, name)
        assert isinstance(state, brainstate.HiddenState)
        assert _state_array(state).shape == shape
        np.testing.assert_array_equal(_state_array(state), 0.0)

    demonstrations = _phase_events(
        memory_model.config,
        jnp.asarray([[1.0, 0.5]]),
        jnp.asarray([[0.25, 1.0]]),
        phase="demonstration",
    )
    query = _phase_events(memory_model.config, jnp.asarray([[1.0, 0.5]]), phase="query")
    run_context(memory_model, demonstrations)
    run_context(memory_model, query, reset=False)
    snapshot = memory_model.snapshot_state()
    expected = {
        name: _state_array(getattr(memory_model, name)).copy()
        for name in (
            "context_memory",
            "query_encoding",
            "reasoning_query",
            "memory_read",
            "workspace_carrier",
        )
    }

    memory_model.reset_state()
    for name in expected:
        np.testing.assert_array_equal(_state_array(getattr(memory_model, name)), 0.0)
    memory_model.restore_state(snapshot)
    for name, value in expected.items():
        np.testing.assert_array_equal(_state_array(getattr(memory_model, name)), value)


def test_memory_snapshot_rejects_a_different_memory_shape() -> None:
    source = LatentWorkspaceModel(_memory_config(context_memory_width=2))
    incompatible = LatentWorkspaceModel(_memory_config(context_memory_width=3))

    with pytest.raises(ValueError, match="snapshot|shape|configuration"):
        incompatible.restore_state(source.snapshot_state())


def test_model_writes_exact_projected_binding_then_freezes_it() -> None:
    config = _memory_config(memory_decay=0.5)
    memory_model = LatentWorkspaceModel(config)
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    first_key = memory_model.encode_memory_key(demonstrations[0])
    first_value = memory_model.encode_memory_value(demonstrations[0])
    second_key = memory_model.encode_memory_key(demonstrations[1])
    second_value = memory_model.encode_memory_value(demonstrations[1])
    expected = 0.5 * jnp.einsum("bi,bj->bij", first_key, first_value) + jnp.einsum(
        "bi,bj->bij", second_key, second_value
    )

    run_context(memory_model, demonstrations)

    np.testing.assert_allclose(
        _state_array(memory_model.context_memory), expected, rtol=1e-6, atol=1e-6
    )
    frozen_memory = _state_array(memory_model.context_memory).copy()
    query = _phase_events(
        config,
        jnp.asarray([[0.5, 1.0]]),
        jnp.asarray([[99.0, -99.0]]),
        phase="query",
    )
    run_context(memory_model, query, reset=False)
    frozen_query = _state_array(memory_model.query_encoding).copy()
    np.testing.assert_array_equal(memory_model.context_memory.value, frozen_memory)

    memory_model.cell_step(
        jnp.zeros((1, config.input_width), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(memory_model.context_memory.value, frozen_memory)
    np.testing.assert_array_equal(memory_model.query_encoding.value, frozen_query)


def test_each_valid_unequal_side_demo_row_advances_memory_decay_once() -> None:
    config = _memory_config(memory_decay=0.5)
    memory_model = LatentWorkspaceModel(config)
    memory_model.context_memory.value = jnp.ones((1, 2, 2), dtype=jnp.float32)
    input_only = _phase_events(
        config,
        jnp.asarray([[1.0, -0.5]]),
        phase="demonstration",
    )
    output_only = _phase_events(
        config,
        jnp.asarray([[0.25, 0.75]]),
        jnp.asarray([[1.5, -1.0]]),
        phase="demonstration",
    )
    assert config.input_side_valid_index is not None
    output_only = output_only.at[..., config.input_side_valid_index].set(0.0)

    memory_model.cell_step(input_only[0])
    np.testing.assert_allclose(
        memory_model.context_memory.value, 0.5, rtol=0.0, atol=0.0
    )
    memory_model.cell_step(output_only[0])
    np.testing.assert_allclose(
        memory_model.context_memory.value, 0.25, rtol=0.0, atol=0.0
    )


def test_multi_row_query_encoding_accumulates_then_freezes() -> None:
    config = _memory_config()
    memory_model = LatentWorkspaceModel(config)
    queries = _phase_events(
        config,
        jnp.asarray([[1.0, -0.5], [0.25, 0.75]]),
        phase="query",
    )
    expected = (
        memory_model.encode_memory_key(queries[0])
        + memory_model.encode_memory_key(queries[1])
    )

    run_context(memory_model, queries)

    np.testing.assert_allclose(
        memory_model.query_encoding.value, expected, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        memory_model.reasoning_query.value, expected, rtol=0.0, atol=0.0
    )
    frozen = _state_array(memory_model.query_encoding).copy()
    memory_model.cell_step(
        jnp.zeros((1, config.input_width), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(memory_model.query_encoding.value, frozen)


def test_equal_marginals_with_different_pairings_change_memory_and_read() -> None:
    config = _memory_config(memory_decay=1.0)
    keys = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    values = jnp.asarray([[1.0, 0.25], [-0.5, 2.0]])
    rotated_values = values[jnp.asarray([1, 0])]
    np.testing.assert_array_equal(
        jnp.sum(values, axis=0), jnp.sum(rotated_values, axis=0)
    )

    intact_model = LatentWorkspaceModel(config)
    shuffled_model = LatentWorkspaceModel(config)
    run_context(
        intact_model,
        _phase_events(config, keys, values, phase="demonstration"),
    )
    run_context(
        shuffled_model,
        _phase_events(config, keys, rotated_values, phase="demonstration"),
    )
    intact_memory = _state_array(intact_model.context_memory)
    shuffled_memory = _state_array(shuffled_model.context_memory)

    assert not np.allclose(intact_memory, shuffled_memory)
    query = _phase_events(config, keys[:1], phase="query")
    run_context(intact_model, query, reset=False)
    run_context(shuffled_model, query, reset=False)
    np.testing.assert_allclose(
        intact_model.query_encoding.value,
        shuffled_model.query_encoding.value,
        rtol=0.0,
        atol=0.0,
    )
    intact_read = intact_model.read_context_memory()
    shuffled_read = shuffled_model.read_context_memory()
    assert not np.allclose(intact_read, shuffled_read)


def test_second_latent_tick_rereads_current_memory() -> None:
    config = _memory_config(memory_decay=1.0)
    memory_model = LatentWorkspaceModel(config)
    run_context(
        memory_model,
        _phase_events(
            config,
            jnp.asarray([[1.0, 0.5]]),
            jnp.asarray([[0.25, 2.0]]),
            phase="demonstration",
        ),
    )
    run_context(
        memory_model,
        _phase_events(config, jnp.asarray([[1.0, 0.5]]), phase="query"),
        reset=False,
    )
    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)
    memory_model.cell_step(zero_event, advance)
    before_second_tick = memory_model.snapshot_state()

    memory_model.cell_step(zero_event, advance)
    baseline_read = _state_array(memory_model.memory_read).copy()
    baseline_workspace = _state_array(memory_model.workspace_carrier).copy()

    memory_model.restore_state(before_second_tick)
    query_encoding = jnp.asarray(memory_model.query_encoding.value)
    aligned_delta = jnp.einsum(
        "bi,bj->bij", query_encoding, jnp.ones_like(query_encoding)
    )
    memory_model.context_memory.value = (
        memory_model.context_memory.value + aligned_delta
    )
    memory_model.cell_step(zero_event, advance)

    assert not np.allclose(_state_array(memory_model.memory_read), baseline_read)
    assert not np.allclose(
        _state_array(memory_model.workspace_carrier), baseline_workspace
    )


def test_latent_query_depends_on_previous_continuous_workspace() -> None:
    config = _memory_config(memory_decay=1.0)
    memory_model = LatentWorkspaceModel(config)
    run_context(
        memory_model,
        _phase_events(
            config,
            jnp.asarray([[1.0, 0.5]]),
            jnp.asarray([[0.25, 2.0]]),
            phase="demonstration",
        ),
    )
    run_context(
        memory_model,
        _phase_events(config, jnp.asarray([[1.0, 0.5]]), phase="query"),
        reset=False,
    )
    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)
    memory_model.cell_step(zero_event, advance)
    before_second_tick = memory_model.snapshot_state()
    frozen_memory = _state_array(memory_model.context_memory).copy()
    frozen_query = _state_array(memory_model.query_encoding).copy()

    memory_model.cell_step(zero_event, advance)
    baseline_read = _state_array(memory_model.memory_read).copy()

    memory_model.restore_state(before_second_tick)
    perturbation = jnp.linspace(
        -1.0, 1.0, config.neuron_count, dtype=jnp.float32
    )[None, :]
    memory_model.workspace_carrier.value = (
        memory_model.workspace_carrier.value + perturbation
    )
    np.testing.assert_array_equal(memory_model.context_memory.value, frozen_memory)
    np.testing.assert_array_equal(memory_model.query_encoding.value, frozen_query)
    memory_model.cell_step(zero_event, advance)

    assert not np.allclose(_state_array(memory_model.memory_read), baseline_read)
    np.testing.assert_array_equal(memory_model.context_memory.value, frozen_memory)
    np.testing.assert_array_equal(memory_model.query_encoding.value, frozen_query)


def test_memory_etp_paths_are_direct_with_finite_window_pp_prop_gradients() -> None:
    import braintrace
    from braintrace._testing.oracle import chunked_online_param_gradients

    config = _memory_config(memory_decay=0.9, input_gain=8.0)
    model = LatentWorkspaceModel(config)
    learner = compile_pp_prop(model)
    write_path = ("memory_write_scale",)
    query_path = ("workspace_query_projection", "weight")
    read_path = ("memory_read_projection", "weight")
    expected_paths = {write_path, query_path, read_path}
    etrace_paths = {path for path, _ in learner.report.etrace_weights}

    assert expected_paths <= etrace_paths
    for path in expected_paths:
        relations = [
            relation
            for relation in learner.graph.hidden_param_op_relations
            if path in relation.trainable_paths.values()
        ]
        assert len(relations) == 1
        assert set(relations[0].path_classification.values()) == {"all_direct"}
    group_paths = [
        {".".join(map(str, path)) for path in group.hidden_paths}
        for group in learner.graph.hidden_groups
    ]
    assert not any(
        {"context_memory", "workspace_carrier"}.issubset(paths)
        for paths in group_paths
    )

    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    latent = jnp.zeros((2, 1, config.input_width), dtype=jnp.float32)
    inputs = jnp.concatenate((demonstrations, query, latent), axis=0)

    class _AlwaysAdvanceMemoryModel(LatentWorkspaceModel):
        def update(self, event: jax.Array) -> jax.Array:
            advance = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)
            return super().update(event, advance)

    gradients = chunked_online_param_gradients(
        lambda: _AlwaysAdvanceMemoryModel(config),
        inputs,
        algo_factory=lambda candidate: braintrace.pp_prop(
            candidate,
            decay_or_rank=candidate.config.trace_decay,
            vjp_method="multi-step",
        ),
        chunk_size=1,
    )
    for path in expected_paths:
        gradient_leaves = jax.tree.leaves(gradients[path])
        gradient_norm = sum(
            float(jnp.sum(jnp.abs(jnp.asarray(u.get_mantissa(leaf)))))
            for leaf in gradient_leaves
        )
        assert gradient_leaves
        assert math.isfinite(gradient_norm)
        assert gradient_norm > 0.0


def test_memory_mode_uses_one_shared_decoder_on_continuous_workspace() -> None:
    config = _memory_config()
    memory_model = LatentWorkspaceModel(config)
    run_context(
        memory_model,
        _phase_events(
            config,
            jnp.asarray([[1.0, -0.5]]),
            jnp.asarray([[0.75, 1.5]]),
            phase="demonstration",
        ),
    )
    run_context(
        memory_model,
        _phase_events(config, jnp.asarray([[1.0, -0.5]]), phase="query"),
        reset=False,
    )
    decoder = memory_model.readout_projection
    carrier = jnp.asarray(memory_model.workspace_carrier.value)

    assert carrier.shape == (1, config.neuron_count)
    assert jnp.issubdtype(carrier.dtype, jnp.floating)
    assert np.any((np.asarray(carrier) != 0.0) & (np.asarray(carrier) != 1.0))
    np.testing.assert_array_equal(
        memory_model.compact_readout(), memory_model.compact_readout(carrier)
    )

    memory_model.cell_step(
        jnp.zeros((1, config.input_width), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.bool_),
    )
    assert memory_model.readout_projection is decoder
    np.testing.assert_array_equal(
        memory_model.compact_readout(),
        memory_model.compact_readout(memory_model.workspace_carrier.value),
    )


@pytest.mark.parametrize(
    "driver",
    [
        "update",
        "run_context",
        "run_packed_stream",
        "run_selected_packed_stream",
        "run_latent_trajectory",
        "run_sequence",
    ],
)
def test_memory_mode_drivers_use_the_implicit_workspace_decoder(
    monkeypatch: pytest.MonkeyPatch,
    driver: str,
) -> None:
    config = _memory_config()
    memory_model = LatentWorkspaceModel(config)
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    context = jnp.concatenate((demonstrations, query), axis=0)
    if driver == "run_latent_trajectory":
        run_context(memory_model, context)

    original = LatentWorkspaceModel.compact_readout

    def guarded_compact_readout(
        self: LatentWorkspaceModel,
        carrier: jax.Array | None = None,
    ) -> jax.Array:
        assert carrier is None, f"{driver} bypassed the memory-mode workspace"
        return original(self)

    monkeypatch.setattr(
        LatentWorkspaceModel,
        "compact_readout",
        guarded_compact_readout,
    )

    if driver == "update":
        memory_model(context[-1], jnp.ones((1,), dtype=jnp.bool_))
    elif driver == "run_context":
        run_context(memory_model, context)
    elif driver == "run_packed_stream":
        run_packed_stream(memory_model, context)
    elif driver == "run_selected_packed_stream":
        selected = jnp.asarray([[context.shape[0] - 1]], dtype=jnp.int32)
        run_selected_packed_stream(memory_model, context, selected)
    elif driver == "run_latent_trajectory":
        run_latent_trajectory(memory_model, steps=1)
    else:
        run_sequence(memory_model, context, latent_steps=1)


def test_zero_width_mode_is_byte_identical_to_implicit_legacy_mode() -> None:
    implicit_model = LatentWorkspaceModel(_config())
    explicit_model = LatentWorkspaceModel(_config(context_memory_width=0))
    _assert_parameter_snapshots_equal(
        parameter_snapshot(implicit_model), parameter_snapshot(explicit_model)
    )
    events = _valid_events(4, implicit_model.config.input_width)

    implicit = run_sequence(implicit_model, events, latent_steps=2)
    explicit = run_sequence(explicit_model, events, latent_steps=2)

    for name in (
        "compact_logits",
        "spikes",
        "voltage",
        "feedforward_current",
        "recurrent_current",
        "zero_inputs",
    ):
        np.testing.assert_array_equal(
            getattr(implicit.trajectory, name), getattr(explicit.trajectory, name)
        )
    implicit_entries = implicit_model.snapshot_state().entries
    explicit_entries = explicit_model.snapshot_state().entries
    assert tuple(path for path, _ in implicit_entries) == tuple(
        path for path, _ in explicit_entries
    )
    for (_, left), (_, right) in zip(implicit_entries, explicit_entries, strict=True):
        for left_array, right_array in zip(
            _tree_arrays(left), _tree_arrays(right), strict=True
        ):
            np.testing.assert_array_equal(left_array, right_array)


def test_context_memory_writes_are_isolated_per_batch_example() -> None:
    config = _memory_config(batch_size=2)
    memory_model = LatentWorkspaceModel(config)
    demonstrations = _phase_events(
        config,
        jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        jnp.asarray([[[0.5, 1.0], [2.0, -1.0]]]),
        phase="demonstration",
    )
    assert config.demonstration_phase_index is not None
    assert config.query_phase_index is not None
    demonstrations = demonstrations.at[:, 1, config.demonstration_phase_index].set(0.0)
    demonstrations = demonstrations.at[:, 1, config.query_phase_index].set(1.0)
    assert config.output_side_valid_index is not None
    demonstrations = demonstrations.at[:, 1, config.output_side_valid_index].set(0.0)

    run_context(memory_model, demonstrations)

    memory = _state_array(memory_model.context_memory)
    assert memory.dtype == np.float32
    assert np.any(memory[0] != 0.0)
    np.testing.assert_array_equal(memory[1], 0.0)
    queries = _phase_events(
        config,
        jnp.asarray([[[1.0, 0.25], [-0.5, 1.0]]]),
        phase="query",
    )
    run_context(memory_model, queries, reset=False)
    assert not np.allclose(
        _state_array(memory_model.query_encoding)[0],
        _state_array(memory_model.query_encoding)[1],
    )
    np.testing.assert_array_equal(_state_array(memory_model.memory_read)[1], 0.0)


@pytest.mark.parametrize(
    ("neuron_count", "edge_count", "message"),
    [(1, 1, "at least 2"), (4, 13, "capacity"), (4, 0, "edge_count")],
)
def test_invalid_topology_request_fails_closed(
    neuron_count: int, edge_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_sparse_topology(neuron_count, edge_count, seed=0)


def test_compact_width_and_expansion_have_exact_arc_shapes() -> None:
    rank = 2
    compact = jnp.zeros((3, compact_output_width(rank)))
    expanded = expand_compact_logits(compact, rank)

    assert isinstance(expanded, ArcLogits)
    assert expanded.height.shape == (3, MAX_GRID_SIZE)
    assert expanded.width.shape == (3, MAX_GRID_SIZE)
    assert expanded.colors.shape == (
        3,
        MAX_GRID_SIZE,
        MAX_GRID_SIZE,
        COLOR_COUNT,
    )


def test_cp_expansion_matches_constructed_rank_one_product() -> None:
    rank = 1
    compact = jnp.zeros((1, compact_output_width(rank)))
    row_start = 2 * MAX_GRID_SIZE
    column_start = row_start + MAX_GRID_SIZE
    color_start = column_start + MAX_GRID_SIZE
    compact = compact.at[:, row_start : row_start + MAX_GRID_SIZE].set(2.0)
    compact = compact.at[:, column_start : column_start + MAX_GRID_SIZE].set(3.0)
    compact = compact.at[:, color_start:].set(4.0)

    expanded = expand_compact_logits(compact, rank)

    np.testing.assert_array_equal(expanded.colors, 24.0)


def test_expansion_rejects_wrong_width_or_rank() -> None:
    with pytest.raises(ValueError, match="final width"):
        expand_compact_logits(jnp.zeros((1, 10)), 2)
    with pytest.raises(ValueError, match="color_rank"):
        compact_output_width(0)


def test_uniform_terminal_loss_matches_three_cross_entropies() -> None:
    compact = jnp.zeros((2, compact_output_width(2)))
    colors = jnp.zeros((2, 30, 30), dtype=jnp.int32)
    components = arc_loss_components(
        compact,
        jnp.asarray([1, 30]),
        jnp.asarray([1, 30]),
        colors,
        color_rank=2,
    )

    assert float(components.height) == pytest.approx(math.log(30), rel=1e-6)
    assert float(components.width) == pytest.approx(math.log(30), rel=1e-6)
    assert float(components.colors) == pytest.approx(math.log(10), rel=1e-6)
    assert float(components.total) == pytest.approx(
        2 * math.log(30) + math.log(10), rel=1e-6
    )


def test_loss_is_per_example_and_ignores_padded_target_cells() -> None:
    compact = jnp.zeros((2, compact_output_width(2)))
    baseline = jnp.zeros((2, 30, 30), dtype=jnp.int32)
    changed_padding = baseline.at[:, 1:, 1:].set(9)
    heights = jnp.asarray([1, 1])
    widths = jnp.asarray([1, 1])

    first = arc_loss_per_example(compact, heights, widths, baseline, color_rank=2)
    second = arc_loss_per_example(
        compact, heights, widths, changed_padding, color_rank=2
    )

    assert first.shape == (2,)
    np.testing.assert_array_equal(first, second)


def test_terminal_loss_has_finite_gradient_through_compact_factors() -> None:
    compact = jnp.ones((1, compact_output_width(2))) * 0.1
    colors = jnp.zeros((1, 30, 30), dtype=jnp.int32)

    gradient = jax.grad(
        lambda value: terminal_arc_loss(
            value,
            jnp.asarray([2]),
            jnp.asarray([3]),
            colors,
            color_rank=2,
        )
    )(compact)

    assert gradient.shape == compact.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)


@pytest.mark.parametrize(
    ("compact", "height", "width", "colors", "message"),
    [
        (
            jnp.zeros((200,)),
            jnp.asarray([1]),
            jnp.asarray([1]),
            jnp.zeros((1, 30, 30)),
            "compact_logits",
        ),
        (
            jnp.zeros((1, 200)),
            jnp.asarray([1, 2]),
            jnp.asarray([1]),
            jnp.zeros((1, 30, 30)),
            "target_height",
        ),
        (
            jnp.zeros((1, 200)),
            jnp.asarray([1]),
            jnp.asarray([1]),
            jnp.zeros((1, 2, 2)),
            "target_colors",
        ),
    ],
)
def test_terminal_loss_rejects_malformed_shapes(
    compact: jax.Array,
    height: jax.Array,
    width: jax.Array,
    colors: jax.Array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        terminal_arc_loss(compact, height, width, colors, color_rank=2)


@pytest.mark.parametrize(
    ("name", "value"), [("shape_weight", -1.0), ("color_weight", math.inf)]
)
def test_terminal_loss_rejects_invalid_weights(name: str, value: float) -> None:
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        terminal_arc_loss(
            jnp.zeros((1, 200)),
            jnp.asarray([1]),
            jnp.asarray([1]),
            jnp.zeros((1, 30, 30), dtype=jnp.int32),
            color_rank=2,
            **kwargs,
        )


def test_all_zero_prefix_from_reset_is_exactly_inert(
    model: LatentWorkspaceModel,
) -> None:
    model.reset_state()
    before = model.snapshot_state()

    result = run_context(model, jnp.zeros((7, model.config.input_width)))
    after = model.snapshot_state()

    np.testing.assert_array_equal(result.spikes, 0.0)
    np.testing.assert_array_equal(result.voltage, 0.0)
    for (_, first), (_, second) in zip(before.entries, after.entries, strict=True):
        for left, right in zip(_tree_arrays(first), _tree_arrays(second), strict=True):
            np.testing.assert_array_equal(left, right)


def test_invalid_padding_freezes_nonzero_voltage_and_both_synapses(
    model: LatentWorkspaceModel,
) -> None:
    run_context(model, _valid_events(4, model.config.input_width))
    before = model.snapshot_state()

    model.cell_step(
        jnp.zeros((1, model.config.input_width)), jnp.zeros((1,), dtype=jnp.bool_)
    )
    after = model.snapshot_state()

    for (_, first), (_, second) in zip(before.entries, after.entries, strict=True):
        for left, right in zip(_tree_arrays(first), _tree_arrays(second), strict=True):
            np.testing.assert_array_equal(left, right)


def test_zero_external_latent_input_advances_post_query_state(
    model: LatentWorkspaceModel,
) -> None:
    run_context(model, _valid_events(8, model.config.input_width))
    before = np.asarray(model.voltage).copy()

    trajectory = run_latent_trajectory(model, steps=2)

    np.testing.assert_array_equal(trajectory.zero_inputs, 0.0)
    assert trajectory.zero_inputs.shape == (2, 1, model.config.input_width)
    assert not np.array_equal(before, np.asarray(trajectory.voltage[-1]))


def test_sequence_exposes_checkpoint_zero_through_requested_effort(
    model: LatentWorkspaceModel,
) -> None:
    result = run_sequence(
        model, _valid_events(5, model.config.input_width), latent_steps=4
    )

    assert result.context.context_steps == 5
    assert result.trajectory.compact_logits.shape == (5, 1, 200)
    assert result.trajectory.spikes.shape == (5, 1, 64)
    assert result.trajectory.voltage.shape == (5, 1, 64)
    assert result.trajectory.feedforward_current.shape == (5, 1, 64)
    assert result.trajectory.recurrent_current.shape == (5, 1, 64)
    assert result.context.feedforward_current.shape == (1, 64)
    assert result.context.recurrent_current.shape == (1, 64)
    np.testing.assert_array_equal(
        result.feedforward_current, result.trajectory.feedforward_current
    )
    np.testing.assert_array_equal(
        result.recurrent_current, result.trajectory.recurrent_current
    )
    assert result.trajectory.expanded.colors.shape == (5, 1, 30, 30, 10)
    assert result.trajectory.at_effort(0).height.shape == (1, 30)
    assert result.trajectory.at_effort(4).colors.shape == (1, 30, 30, 10)
    with pytest.raises(ValueError, match="exceeds"):
        result.trajectory.at_effort(5)


def test_zero_length_latent_trajectory_retains_only_checkpoint_zero(
    model: LatentWorkspaceModel,
) -> None:
    run_context(model, _valid_events(2, model.config.input_width))

    result = run_latent_trajectory(model, steps=0)

    assert result.compact_logits.shape == (1, 1, 200)
    assert result.spikes.shape == (1, 1, 64)
    assert result.feedforward_current.shape == (1, 1, 64)
    assert result.recurrent_current.shape == (1, 1, 64)
    assert result.zero_inputs.shape == (0, 1, model.config.input_width)


def test_context_and_latent_length_validation(model: LatentWorkspaceModel) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_context(model, jnp.zeros((0, model.config.input_width)))
    with pytest.raises(ValueError, match="rank-two"):
        other = LatentWorkspaceModel(_config(batch_size=2))
        run_context(other, jnp.zeros((2, other.config.input_width)))
    with pytest.raises(ValueError, match="events must have shape"):
        run_context(model, jnp.zeros((2, 1, model.config.input_width + 1)))
    with pytest.raises(ValueError, match="exceeds configured"):
        run_latent_trajectory(model, steps=5)


def test_packed_stream_records_every_tick_with_explicit_advance_gates(
    model: LatentWorkspaceModel,
) -> None:
    events = jnp.zeros((6, 1, model.config.input_width))
    events = events.at[1, :, 0].set(1.0).at[1, :, 1].set(1.0)
    advance = jnp.asarray([[0], [1], [0], [1], [1], [1]], dtype=jnp.bool_)

    result = run_packed_stream(model, events, advance_gates=advance)

    assert result.compact_logits.shape == (6, 1, 200)
    assert result.spikes.shape == (6, 1, 64)
    assert result.voltage.shape == (6, 1, 64)
    assert result.feedforward_current.shape == (6, 1, 64)
    assert result.recurrent_current.shape == (6, 1, 64)
    assert result.expanded.colors.shape == (6, 1, 30, 30, 10)
    np.testing.assert_array_equal(result.voltage[0], 0.0)
    np.testing.assert_array_equal(result.voltage[2], result.voltage[1])
    np.testing.assert_array_equal(
        result.feedforward_current[2], result.feedforward_current[1]
    )
    np.testing.assert_array_equal(
        result.recurrent_current[2], result.recurrent_current[1]
    )


def test_synaptic_currents_are_unit_safe_numeric_arrays(
    model: LatentWorkspaceModel,
) -> None:
    result = run_sequence(
        model, _valid_events(5, model.config.input_width), latent_steps=2
    )

    assert isinstance(model.feedforward_current, jax.Array)
    assert isinstance(model.recurrent_current, jax.Array)
    assert result.context.feedforward_current.dtype == jnp.float32
    assert result.context.recurrent_current.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(result.feedforward_current)))
    assert np.all(np.isfinite(np.asarray(result.recurrent_current)))
    assert np.any(np.asarray(result.context.feedforward_current) != 0.0)


def test_equal_workspace_outputs_can_hide_synaptic_control_divergence(
    model: LatentWorkspaceModel,
) -> None:
    events = jnp.zeros((1, 1, model.config.input_width), dtype=jnp.float32)
    frozen = jnp.zeros((1, 1), dtype=jnp.bool_)
    intact = run_packed_stream(model, events, reset=True, advance_gates=frozen)

    model.reset_state()
    model.ff_syn.syn.g.value = (
        jnp.full((model.config.batch_size, model.config.neuron_count), 0.25) * u.mA
    )
    control = run_packed_stream(model, events, reset=False, advance_gates=frozen)

    np.testing.assert_array_equal(control.compact_logits, intact.compact_logits)
    np.testing.assert_array_equal(control.spikes, intact.spikes)
    np.testing.assert_array_equal(control.voltage, intact.voltage)
    np.testing.assert_array_equal(control.recurrent_current, intact.recurrent_current)
    assert not np.array_equal(
        np.asarray(control.feedforward_current),
        np.asarray(intact.feedforward_current),
    )


def test_packed_stream_supports_boundary_slot_control() -> None:
    controlled = LatentWorkspaceModel(_config(batch_size=2, neuron_count=128))
    events = jnp.zeros((4, 2, controlled.config.input_width))
    events = events.at[0, :, 0].set(1.0).at[0, :, 1].set(1.0)
    advance = jnp.ones((4, 2), dtype=jnp.bool_)
    gates = jnp.zeros((4, 2), dtype=jnp.bool_)
    gates = gates.at[1, 0].set(True).at[2, 1].set(True)

    result = run_packed_stream(
        controlled,
        events,
        advance_gates=advance,
        ablation_slots=jnp.asarray([0, 1]),
        ablation_gates=gates,
    )

    assert result.spikes.shape == (4, 2, 128)
    assert np.all(np.isfinite(np.asarray(result.voltage)))


@pytest.mark.parametrize("controlled", [False, True])
def test_selected_packed_scan_equals_full_gather_for_variable_terminals(
    controlled: bool,
) -> None:
    selected_model = LatentWorkspaceModel(_config(batch_size=2, neuron_count=128))
    time_steps = 9
    events = jnp.zeros((time_steps, 2, selected_model.config.input_width))
    events = events.at[0, :, 0].set(1.0).at[0, :, 1].set(1.0)
    events = events.at[1, 0, 0].set(1.0).at[1, 0, 2].set(1.0)
    events = events.at[2, 1, 0].set(1.0).at[2, 1, 3].set(1.0)
    advances = jnp.ones((time_steps, 2), dtype=jnp.bool_)
    indices = jnp.asarray([[1, 3], [2, 4], [5, 6], [8, 8]], dtype=jnp.int32)
    kwargs: dict[str, object] = {"advance_gates": advances}
    if controlled:
        ablation_gates = jnp.zeros((time_steps, 2), dtype=jnp.bool_)
        ablation_gates = ablation_gates.at[2, 0].set(True).at[4, 1].set(True)
        kwargs.update(
            ablation_slots=jnp.asarray([0, 1]),
            ablation_gates=ablation_gates,
        )

    full = run_packed_stream(selected_model, events, **kwargs)
    selected = run_selected_packed_stream(selected_model, events, indices, **kwargs)

    assert isinstance(selected, SelectedPackedTrajectory)
    batch = np.arange(2, dtype=np.int32)[None, :]
    raw_indices = np.asarray(indices)
    for name in (
        "compact_logits",
        "spikes",
        "voltage",
        "feedforward_current",
        "recurrent_current",
        "workspace_carrier",
        "memory_read",
    ):
        expected = np.asarray(getattr(full, name))[raw_indices, batch]
        np.testing.assert_array_equal(getattr(selected, name), expected)
    np.testing.assert_array_equal(selected.selected_indices, indices)
    assert selected.workspace_carrier is selected.voltage
    assert selected.memory_read.shape == (4, 2, 0)
    assert selected.final_context_memory.shape == (2, 0, 0)
    np.testing.assert_array_equal(
        selected.final_context_memory, full.final_context_memory
    )
    assert selected.expanded.colors.shape == (4, 2, 30, 30, 10)


def test_selected_memory_diagnostics_are_bounded_and_pairing_sensitive() -> None:
    config = _memory_config()
    full_model = LatentWorkspaceModel(config)
    selected_model = LatentWorkspaceModel(config)
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    latent = jnp.zeros((2, 1, config.input_width), dtype=jnp.float32)
    events = jnp.concatenate((demonstrations, query, latent), axis=0)
    advances = jnp.ones((events.shape[0], 1), dtype=jnp.bool_)
    indices = jnp.asarray([[2], [3], [4]], dtype=jnp.int32)

    full = run_packed_stream(
        full_model,
        events,
        advance_gates=advances,
    )
    selected = run_selected_packed_stream(
        selected_model,
        events,
        indices,
        advance_gates=advances,
    )

    assert selected.workspace_carrier is selected.voltage
    assert selected.workspace_carrier.shape == (3, 1, config.neuron_count)
    assert selected.memory_read.shape == (3, 1, config.context_memory_width)
    assert selected.final_context_memory.shape == (
        1,
        config.context_memory_width,
        config.context_memory_width,
    )
    raw_indices = np.asarray(indices)
    batch = np.asarray([[0]], dtype=np.int32)
    np.testing.assert_array_equal(
        selected.workspace_carrier,
        np.asarray(full.workspace_carrier)[raw_indices, batch],
    )
    np.testing.assert_array_equal(
        selected.memory_read,
        np.asarray(full.memory_read)[raw_indices, batch],
    )
    np.testing.assert_array_equal(
        selected.final_context_memory, selected_model.context_memory.value
    )
    np.testing.assert_array_equal(
        selected.final_context_memory, full.final_context_memory
    )
    assert np.any(np.asarray(selected.memory_read) != 0.0)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (jnp.zeros((2,), dtype=jnp.int32), "shape"),
        (jnp.zeros((0, 1), dtype=jnp.int32), "at least one"),
        (jnp.asarray([[0.0], [1.0]]), "integers"),
        (jnp.asarray([[-1], [1]]), "lie in"),
        (jnp.asarray([[0], [3]]), "lie in"),
        (jnp.asarray([[1], [1]]), "strictly increasing"),
    ],
)
def test_selected_packed_scan_rejects_invalid_indices(
    model: LatentWorkspaceModel, indices: jax.Array, message: str
) -> None:
    events = jnp.zeros((3, 1, model.config.input_width))
    with pytest.raises(ValueError, match=message):
        run_selected_packed_stream(model, events, indices)


def test_selected_packed_scan_rejects_malformed_gates(
    model: LatentWorkspaceModel,
) -> None:
    events = jnp.zeros((3, 1, model.config.input_width))
    indices = jnp.asarray([[0], [2]])
    with pytest.raises(ValueError, match="advance_gates"):
        run_selected_packed_stream(
            model, events, indices, advance_gates=jnp.ones((2, 1))
        )
    with pytest.raises(ValueError, match="supplied together"):
        run_selected_packed_stream(
            model, events, indices, ablation_slots=jnp.asarray([0])
        )
    with pytest.raises(ValueError, match="ablation_slots"):
        run_selected_packed_stream(
            model,
            events,
            indices,
            ablation_slots=jnp.asarray([0.0]),
            ablation_gates=jnp.zeros((3, 1)),
        )
    with pytest.raises(ValueError, match="ablation_gates"):
        run_selected_packed_stream(
            model,
            events,
            indices,
            ablation_slots=jnp.asarray([0]),
            ablation_gates=jnp.zeros((2, 1)),
        )


def test_packed_stream_rejects_malformed_control_arrays(
    model: LatentWorkspaceModel,
) -> None:
    events = jnp.zeros((3, 1, model.config.input_width))
    with pytest.raises(ValueError, match="advance_gates"):
        run_packed_stream(model, events, advance_gates=jnp.ones((2, 1)))
    with pytest.raises(ValueError, match="supplied together"):
        run_packed_stream(model, events, ablation_slots=jnp.asarray([0]))
    with pytest.raises(ValueError, match="contain integers"):
        run_packed_stream(
            model,
            events,
            ablation_slots=jnp.asarray([0.0]),
            ablation_gates=jnp.zeros((3, 1)),
        )
    with pytest.raises(ValueError, match="must lie"):
        run_packed_stream(
            model,
            events,
            ablation_slots=jnp.asarray([1]),
            ablation_gates=jnp.zeros((3, 1)),
        )
    with pytest.raises(ValueError, match="ablation_gates"):
        run_packed_stream(
            model,
            events,
            ablation_slots=jnp.asarray([0]),
            ablation_gates=jnp.zeros((2, 1)),
        )


def test_snapshot_restore_is_exact_and_does_not_alias(
    model: LatentWorkspaceModel,
) -> None:
    run_context(model, _valid_events(4, model.config.input_width))
    snapshot = model.snapshot_state()
    expected_voltage = np.asarray(model.voltage).copy()
    model.cell_step(jnp.zeros((1, model.config.input_width)), jnp.ones((1,), bool))

    model.restore_state(snapshot)

    np.testing.assert_array_equal(model.voltage, expected_voltage)
    model.reset_state()
    assert np.any(expected_voltage != 0.0)


def test_restore_rejects_wrong_type_configuration_or_paths(
    model: LatentWorkspaceModel,
) -> None:
    with pytest.raises(TypeError, match="ModelStateSnapshot"):
        model.restore_state(object())  # type: ignore[arg-type]
    snapshot = model.snapshot_state()
    incompatible = ModelStateSnapshot(snapshot.entries, 2, snapshot.neuron_count)
    with pytest.raises(ValueError, match="configuration"):
        model.restore_state(incompatible)
    wrong_paths = ModelStateSnapshot(
        ((("unknown",), snapshot.entries[0][1]),) + snapshot.entries[1:],
        snapshot.batch_size,
        snapshot.neuron_count,
    )
    with pytest.raises(ValueError, match="paths"):
        model.restore_state(wrong_paths)


def test_reset_clears_hidden_state_and_preserves_parameters(
    model: LatentWorkspaceModel,
) -> None:
    before_parameters = parameter_snapshot(model)
    run_context(model, _valid_events(5, model.config.input_width))
    assert np.any(np.asarray(model.voltage) != 0.0)

    model.reset_state()
    after_parameters = parameter_snapshot(model)

    np.testing.assert_array_equal(model.voltage, 0.0)
    np.testing.assert_array_equal(u.get_mantissa(model.ff_syn.syn.g.value), 0.0)
    np.testing.assert_array_equal(u.get_mantissa(model.rec_syn.syn.g.value), 0.0)
    _assert_parameter_snapshots_equal(before_parameters, after_parameters)
    with pytest.raises(ValueError, match="batch_size"):
        model.reset_state(batch_size=2)


def test_complete_frozen_sequence_leaves_every_parameter_bitwise_identical(
    model: LatentWorkspaceModel,
) -> None:
    before = parameter_snapshot(model)

    run_sequence(model, _valid_events(4, model.config.input_width), latent_steps=4)

    _assert_parameter_snapshots_equal(before, parameter_snapshot(model))


def test_first_and_last_slot_ablation_zero_exact_voltage_and_derived_spikes() -> None:
    ablation_model = LatentWorkspaceModel(_config(neuron_count=128))
    ablation_model.neu.V.value = jnp.full((1, 128), 1.5) * u.mV
    before_parameters = parameter_snapshot(ablation_model)

    voltage, spikes = ablation_model.ablate_slot(0)
    np.testing.assert_array_equal(voltage[:, :64], 0.0)
    np.testing.assert_array_equal(spikes[:, :64], 0.0)
    assert np.all(np.asarray(spikes[:, 64:]) == 1.0)
    ablation_model.neu.V.value = jnp.full((1, 128), 1.5) * u.mV
    voltage, spikes = ablation_model.ablate_slot(1)
    np.testing.assert_array_equal(voltage[:, 64:], 0.0)
    np.testing.assert_array_equal(spikes[:, 64:], 0.0)
    _assert_parameter_snapshots_equal(
        before_parameters, parameter_snapshot(ablation_model)
    )
    with pytest.raises(ValueError, match="slot_index"):
        ablation_model.ablate_slot(2)


def test_jitted_per_batch_slot_mask_changes_only_enabled_example() -> None:
    ablation_model = LatentWorkspaceModel(_config(batch_size=2, neuron_count=128))
    ablation_model.neu.V.value = jnp.full((2, 128), 1.5) * u.mV

    masked = brainstate.transform.jit(ablation_model.mask_slots)(
        jnp.asarray([0, 1]), jnp.asarray([True, False])
    )

    voltage, spikes = masked
    np.testing.assert_array_equal(voltage[0, :64], 0.0)
    np.testing.assert_array_equal(spikes[0, :64], 0.0)
    assert np.all(np.asarray(voltage[0, 64:]) == 1.5)
    assert np.all(np.asarray(voltage[1]) == 1.5)
    with pytest.raises(ValueError, match="shape"):
        ablation_model.mask_slots(jnp.asarray([0]), jnp.asarray([True]))


def test_etrace_coordinate_is_explicit_pp_prop() -> None:
    config = _config(trace_decay=0.75)
    model = LatentWorkspaceModel(config)
    trace = model.etrace_config()

    assert trace.trace_factorization == "io_factorized"
    assert trace.temporal_recursion == ("scalar_leak", "jacobian")
    assert trace.recurrence_scope == "diagonal"
    assert trace.decay == (0.75, 0.75)


def test_small_pp_prop_compile_and_terminal_gradient_are_finite() -> None:
    training_model = LatentWorkspaceModel(_config(recurrent_edges=64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        learner = compile_pp_prop(training_model)
    events = jnp.zeros((2, 1, training_model.config.input_width))
    events = events.at[0, :, 0].set(1.0).at[0, :, 1].set(1.0)
    advance = jnp.ones((2, 1), dtype=jnp.bool_)
    colors = jnp.zeros((1, 30, 30), dtype=jnp.int32)

    def step_loss(event: jax.Array, gate: jax.Array) -> jax.Array:
        return terminal_arc_loss(
            learner(event, gate),
            jnp.asarray([1]),
            jnp.asarray([1]),
            colors,
            color_rank=training_model.config.color_rank,
        )

    gradients, loss = learner.etrace_grad(
        events,
        advance,
        step_fn=step_loss,
        mask=jnp.asarray([0.0, 1.0]),
        reduction="mean",
        loss_output="scalar",
        return_value=True,
    )

    assert np.isfinite(float(loss))
    assert gradients
    assert all(
        np.all(np.isfinite(np.asarray(u.get_mantissa(leaf))))
        for value in gradients.values()
        for leaf in jax.tree.leaves(value)
    )


@pytest.mark.parametrize(
    "function",
    [run_context, run_latent_trajectory, run_packed_stream, run_sequence],
)
def test_repeated_model_execution_uses_brainstate_loops_without_python_driver(
    function: object,
) -> None:
    tree = ast.parse(inspect.getsource(function))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert (
        any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "for_loop"
            for call in calls
        )
        or function is run_sequence
    )


def test_selected_packed_runner_uses_compiled_scan_without_python_driver() -> None:
    tree = ast.parse(inspect.getsource(run_selected_packed_stream))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "scan"
        for call in calls
    )
