"""Tests for Example 21's recurrent spiking ARC workspace."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import msgspec_json
import math
import warnings

import brainpy.state as bpstate
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_protocol import (
    StepGates,
    build_batched_protocol_v2_arm,
)

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
        RowRefinementLayout,
        SelectedPackedTrajectory,
        arc_loss_components,
        arc_loss_per_example,
        build_sparse_topology,
        compact_output_width,
        compile_etrace,
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
        RowRefinementLayout,
        SelectedPackedTrajectory,
        arc_loss_components,
        arc_loss_per_example,
        build_sparse_topology,
        compact_output_width,
        compile_etrace,
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


def _assert_state_snapshots_equal(
    left: ModelStateSnapshot, right: ModelStateSnapshot
) -> None:
    assert left.batch_size == right.batch_size
    assert left.neuron_count == right.neuron_count
    assert tuple(path for path, _ in left.entries) == tuple(
        path for path, _ in right.entries
    )
    for (path, left_value), (_, right_value) in zip(
        left.entries, right.entries, strict=True
    ):
        left_arrays = _tree_arrays(left_value)
        right_arrays = _tree_arrays(right_value)
        assert len(left_arrays) == len(right_arrays)
        for first, second in zip(left_arrays, right_arrays, strict=True):
            np.testing.assert_array_equal(first, second, err_msg=str(path))


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


def test_protocol_v2_decoder_validates_and_rejects_attention_bypass() -> None:
    config = _row_refinement_config(decoder_mode="latent_row_decode")

    assert config.decoder_mode == "latent_row_decode"
    assert config.row_refinement_enabled
    with pytest.raises(ValueError, match="bypasses the latent-binding"):
        dataclasses.replace(config, refinement_mixer="attention_residual")


def test_latent_row_decoder_preserves_physical_and_memory_state() -> None:
    model = LatentWorkspaceModel(
        _row_refinement_config(decoder_mode="latent_row_decode")
    )
    _, events = _row_refinement_episode()
    run_context(model, events)
    voltage = np.asarray(u.get_mantissa(model.neu.V.value)).copy()
    feedforward = np.asarray(u.get_mantissa(model.ff_syn.syn.g.value)).copy()
    recurrent = np.asarray(u.get_mantissa(model.rec_syn.syn.g.value)).copy()
    memory = np.asarray(model.context_memory.value).copy()
    gates = StepGates(
        advance_physics=jnp.zeros((1,), dtype=jnp.bool_),
        latent_update=jnp.zeros((1,), dtype=jnp.bool_),
        decode_row=jnp.ones((1,), dtype=jnp.bool_),
        answer_feedback=jnp.zeros((1,), dtype=jnp.bool_),
        recurrent_enabled=jnp.zeros((1,), dtype=jnp.bool_),
    )

    model.cell_step(jnp.zeros((1, model.config.input_width)), gates)

    np.testing.assert_array_equal(u.get_mantissa(model.neu.V.value), voltage)
    np.testing.assert_array_equal(u.get_mantissa(model.ff_syn.syn.g.value), feedforward)
    np.testing.assert_array_equal(u.get_mantissa(model.rec_syn.syn.g.value), recurrent)
    np.testing.assert_array_equal(model.context_memory.value, memory)
    np.testing.assert_array_equal(model.reasoning_index.value, 1)


def test_protocol_v2_stream_runs_all_three_complete_decoder_checkpoints() -> None:
    model = LatentWorkspaceModel(
        _row_refinement_config(decoder_mode="latent_row_decode")
    )
    rows, events = _row_refinement_episode()
    valid = np.asarray(events[:, rows.valid_slice.start] > 0.5)
    query_stop = int(np.flatnonzero(valid)[-1]) + 1
    arm = build_batched_protocol_v2_arm(
        np.asarray(events)[:, None, :],
        valid[:, None],
        np.asarray([query_stop], dtype=np.int32),
    )
    checkpoints = np.asarray(arm.metadata["checkpoint_indices"], dtype=np.int32)

    trajectory = run_selected_packed_stream(
        model,
        arm.events,
        checkpoints,
        advance_gates=arm.gates,
    )

    assert trajectory.compact_logits.shape == (3, 1, 9060)
    assert checkpoints[:, 0].tolist() == [query_stop + 29, query_stop + 89, query_stop + 149]


def test_compile_pp_prop_is_the_compatibility_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(latent_workspace_module.braintrace, "compile", lambda *a, **k: sentinel)
    model = LatentWorkspaceModel(_config())

    assert compile_etrace(model) is sentinel
    assert compile_pp_prop(model) is sentinel


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
    return np.asarray(state.value)


def _snapshot_entry_map(
    snapshot: ModelStateSnapshot,
) -> dict[tuple[object, ...], object]:
    return {path: value for path, value in snapshot.entries}


def _max_per_example_rms(left: object, right: object) -> float:
    left_array = np.asarray(u.get_mantissa(left), dtype=np.float64)
    right_array = np.asarray(u.get_mantissa(right), dtype=np.float64)
    assert left_array.shape == right_array.shape
    assert left_array.size > 0
    assert np.all(np.isfinite(left_array))
    assert np.all(np.isfinite(right_array))
    difference = right_array - left_array
    if difference.ndim == 0:
        per_example = np.abs(difference).reshape(1)
    else:
        flattened = difference.reshape(difference.shape[0], -1)
        per_example = np.sqrt(np.mean(np.square(flattened), axis=1))
    assert np.all(np.isfinite(per_example))
    return float(np.max(per_example))


def _assert_max_per_example_rms(
    left: object,
    right: object,
    *,
    limit: float,
    label: object,
) -> None:
    observed = _max_per_example_rms(left, right)
    assert observed <= limit, f"{label}: max per-example RMS {observed} > {limit}"


def _assert_non_context_snapshot_close(
    left: ModelStateSnapshot,
    right: ModelStateSnapshot,
    *,
    limit: float,
) -> None:
    assert left.batch_size == right.batch_size
    assert left.neuron_count == right.neuron_count
    left_entries = _snapshot_entry_map(left)
    right_entries = _snapshot_entry_map(right)
    assert left_entries.keys() == right_entries.keys()
    for path, left_value in left_entries.items():
        if path == ("context_memory",):
            continue
        left_arrays = _tree_arrays(left_value)
        right_arrays = _tree_arrays(right_entries[path])
        assert len(left_arrays) == len(right_arrays)
        for left_array, right_array in zip(left_arrays, right_arrays, strict=True):
            _assert_max_per_example_rms(
                left_array,
                right_array,
                limit=limit,
                label=path,
            )


def _decoded_argmax_prediction(
    compact: jax.Array,
    color_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expanded = expand_compact_logits(compact, color_rank)
    return (
        np.argmax(np.asarray(expanded.height), axis=-1) + 1,
        np.argmax(np.asarray(expanded.width), axis=-1) + 1,
        np.argmax(np.asarray(expanded.colors), axis=-1),
    )


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


def _row_refinement_layout(
    rows: RowEventConfig | None = None,
) -> RowRefinementLayout:
    if rows is None:
        rows = RowEventConfig()
    return RowRefinementLayout(
        input_width=rows.input_width,
        event_valid_index=rows.valid_slice.start,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        normalized_start=rows.normalized_slice.start,
        row_index_start=rows.row_index_slice.start,
        input_height_start=rows.input_height_slice.start,
        input_width_start=rows.input_width_slice.start,
        output_height_start=rows.output_height_slice.start,
        output_width_start=rows.output_width_slice.start,
        input_mask_start=rows.input_mask_slice.start,
        output_mask_start=rows.output_mask_slice.start,
        input_color_start=rows.input_color_slice.start,
        output_color_start=rows.output_color_slice.start,
    )


def _row_refinement_config(**changes: object) -> ModelConfig:
    rows = RowEventConfig()
    features = associative_memory_feature_indices(rows)
    values: dict[str, object] = {
        "input_width": rows.input_width,
        "batch_size": 1,
        "neuron_count": 64,
        "recurrent_edges": 64,
        "max_latent_steps": 30,
        "readout_width": 8,
        "color_rank": 2,
        "seed": 2171,
        "context_memory_width": 2,
        "memory_decay": 1.0,
        "demonstration_phase_index": rows.phase_slice.start,
        "query_phase_index": rows.phase_slice.start + 1,
        "input_side_valid_index": rows.side_valid_slice.start,
        "output_side_valid_index": rows.side_valid_slice.start + 1,
        "memory_key_indices": features.key_indices,
        "memory_value_indices": features.value_indices,
        "decoder_mode": "row_refinement",
        "refinement_steps": 30,
        "refinement_layout": _row_refinement_layout(rows),
    }
    values.update(changes)
    return ModelConfig(**values)  # type: ignore[arg-type]


def _row_refinement_episode() -> tuple[RowEventConfig, jax.Array]:
    rows = RowEventConfig()
    task = ArcTask(
        train=(
            ArcPair(ArcGrid(((1, 2),)), ArcGrid(((2, 1),))),
            ArcPair(ArcGrid(((3,), (4,))), ArcGrid(((4,), (3,)))),
        ),
        test=(ArcPair(ArcGrid(((5, 6), (7, 8))), None),),
        task_id="row-refinement-contract",
    )
    encoded = encode_query_episode(task, 0, rows)
    return rows, jnp.asarray(encoded.events)


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


def test_full_configuration_is_4096_neurons_4194304_edges_and_64_slots() -> None:
    config = ModelConfig(input_width=830)

    assert config.neuron_count == 4096
    assert config.recurrent_edges == 4194304
    assert config.max_latent_steps == 32
    assert config.slot_count == 64
    assert config.slot_count * NEURONS_PER_SLOT == config.neuron_count


def test_full_topology_has_exact_unique_no_self_edge_count() -> None:
    topology = build_sparse_topology(4096, 4194304, seed=2108)
    flat = topology.rows.astype(np.int64) * 4096 + topology.columns

    assert topology.edge_count == 4194304
    assert np.unique(flat).size == 4194304
    assert not np.any(topology.rows == topology.columns)
    assert topology.rows.flags.writeable is False
    assert topology.columns.flags.writeable is False
    assert topology.values.flags.writeable is False


def test_full_model_instantiates_physical_components() -> None:
    full = LatentWorkspaceModel(ModelConfig(input_width=4))

    assert full.neuron_count == 4096
    assert full.recurrent_edge_count == 4194304
    assert full.slot_count == 64
    assert full.neu.varshape == (4096,)
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


def test_associative_memory_report_declares_fixed_carrier_stabilization() -> None:
    legacy = LatentWorkspaceModel(_config()).associative_memory_report()
    memory = LatentWorkspaceModel(_memory_config()).associative_memory_report()
    serialized_legacy = legacy.to_dict()
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

    assert legacy.carrier_stabilizer is None
    assert legacy.carrier_radius is None
    assert legacy.carrier_consumers is None
    assert legacy.carrier_normalization_by_consumer is None
    assert memory.carrier_stabilizer == "per_example_stopped_unit_l2_cap"
    assert memory.carrier_radius == 1.0
    assert memory.carrier_consumers == (
        "readout_projection",
        "workspace_query_projection",
    )
    assert memory.carrier_normalization_by_consumer is None
    assert serialized_legacy == expected_legacy
    assert memory.to_dict()["read_transform"] == "linear"
    assert memory.to_dict()["read_interval"] == 1
    assert memory.to_dict()["latent_residual_mixer"] == "none"
    assert memory.to_dict()["latent_residual_block_size"] == 10
    assert msgspec_json.dumps(serialized_legacy, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    ) == msgspec_json.dumps(expected_legacy, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert legacy.to_dict() == expected_legacy
    serialized_memory = memory.to_dict()
    assert serialized_memory["carrier_stabilizer"] == (
        "per_example_stopped_unit_l2_cap"
    )
    assert serialized_memory["carrier_radius"] == 1.0
    assert serialized_memory["carrier_consumers"] == (
        "readout_projection",
        "workspace_query_projection",
    )
    assert set(serialized_memory) == set(expected_legacy) | {
        "carrier_stabilizer",
        "carrier_radius",
        "carrier_consumers",
        "read_transform",
        "read_interval",
        "latent_residual_mixer",
        "latent_residual_block_size",
    }


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_unit_l2_cap_is_per_example_and_preserves_dtype(dtype: jnp.dtype) -> None:
    unit_l2_cap = latent_workspace_module._unit_l2_cap
    carrier = jnp.asarray(
        [[0.0, 0.0], [0.25, -0.5], [3.0, 4.0], [-6.0, 8.0]],
        dtype=dtype,
    )

    capped = unit_l2_cap(carrier)

    assert capped.shape == carrier.shape
    assert capped.dtype == carrier.dtype
    assert np.isfinite(np.asarray(capped)).all()
    np.testing.assert_allclose(capped[:2], carrier[:2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        capped[2:],
        jnp.asarray([[0.6, 0.8], [-0.6, 0.8]], dtype=dtype),
        rtol=2e-3 if dtype == jnp.float16 else 1e-6,
        atol=2e-3 if dtype == jnp.float16 else 1e-6,
    )
    norms = np.linalg.norm(np.asarray(capped, dtype=np.float32), axis=-1)
    assert np.max(norms) <= 1.0 + (2e-3 if dtype == jnp.float16 else 1e-6)


def test_unit_l2_cap_has_batch_block_scalar_diagonal_jacobian() -> None:
    unit_l2_cap = latent_workspace_module._unit_l2_cap
    carrier = jnp.asarray([[0.3, 0.4], [3.0, 4.0]], dtype=jnp.float32)

    def flattened_cap(flat_carrier: jax.Array) -> jax.Array:
        return unit_l2_cap(flat_carrier.reshape(carrier.shape)).reshape(-1)

    jacobian = jax.jacrev(flattened_cap)(carrier.reshape(-1))

    np.testing.assert_allclose(
        jacobian,
        np.diag(np.asarray([1.0, 1.0, 0.2, 0.2], dtype=np.float32)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_unit_l2_cap_accumulates_low_precision_norm_without_overflow() -> None:
    unit_l2_cap = latent_workspace_module._unit_l2_cap
    carrier = jnp.full((2, 2048), 100.0, dtype=jnp.float16)

    capped = unit_l2_cap(carrier)

    assert capped.dtype == jnp.float16
    assert np.isfinite(np.asarray(capped)).all()
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(capped, dtype=np.float32), axis=-1),
        np.ones((2,), dtype=np.float32),
        rtol=2e-3,
        atol=2e-3,
    )


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
    shuffled_memory = jnp.einsum("bi,bj->ij", demonstration_keys, rotated_values)
    intact_reads = jnp.einsum("ik,kv->iv", query_keys, intact_memory)
    shuffled_reads = jnp.einsum("ik,kv->iv", query_keys, shuffled_memory)

    np.testing.assert_allclose(demonstration_keys, query_keys, rtol=0.0, atol=0.0)
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
    update_context_memory = latent_workspace_module.update_context_memory
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
    update_context_memory = latent_workspace_module.update_context_memory
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


@pytest.mark.parametrize(
    ("policy", "exception"),
    [(None, TypeError), (1, TypeError), ("cached", ValueError)],
)
def test_memory_read_policy_rejects_invalid_constructor_values(
    policy: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception, match="memory_read_policy"):
        LatentWorkspaceModel(_memory_config(), memory_read_policy=policy)  # type: ignore[arg-type]


def test_memory_read_policy_is_constructor_only_and_config_neutral() -> None:
    config = _memory_config()
    serialized_config = dataclasses.asdict(config)
    implicit_full = LatentWorkspaceModel(config)
    explicit_full = LatentWorkspaceModel(config, memory_read_policy="full")
    query_only = LatentWorkspaceModel(config, memory_read_policy="query_only")

    assert "memory_read_policy" not in serialized_config
    assert dataclasses.asdict(config) == serialized_config
    assert implicit_full.memory_read_policy == "full"
    assert explicit_full.memory_read_policy == "full"
    assert query_only.memory_read_policy == "query_only"
    with pytest.raises(AttributeError):
        query_only.memory_read_policy = "full"  # type: ignore[misc]

    expected_parameters = parameter_snapshot(implicit_full)
    _assert_parameter_snapshots_equal(
        expected_parameters, parameter_snapshot(explicit_full)
    )
    _assert_parameter_snapshots_equal(
        expected_parameters, parameter_snapshot(query_only)
    )
    _assert_state_snapshots_equal(
        implicit_full.snapshot_state(), explicit_full.snapshot_state()
    )
    _assert_state_snapshots_equal(
        implicit_full.snapshot_state(), query_only.snapshot_state()
    )


def test_implicit_and_explicit_full_memory_policy_are_byte_identical() -> None:
    config = _memory_config(memory_decay=1.0)
    implicit = LatentWorkspaceModel(config)
    explicit = LatentWorkspaceModel(config, memory_read_policy="full")
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    context = jnp.concatenate((demonstrations, query), axis=0)

    implicit_result = run_sequence(implicit, context, latent_steps=2)
    explicit_result = run_sequence(explicit, context, latent_steps=2)

    for name in (
        "compact_logits",
        "spikes",
        "voltage",
        "feedforward_current",
        "recurrent_current",
        "zero_inputs",
    ):
        np.testing.assert_array_equal(
            getattr(implicit_result.trajectory, name),
            getattr(explicit_result.trajectory, name),
        )
    _assert_state_snapshots_equal(implicit.snapshot_state(), explicit.snapshot_state())


def test_query_only_matches_h0_then_removes_latent_memory_read_and_drive() -> None:
    config = _memory_config(memory_decay=1.0, input_gain=8.0)
    full = LatentWorkspaceModel(config, memory_read_policy="full")
    query_only = LatentWorkspaceModel(config, memory_read_policy="query_only")
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    context = jnp.concatenate((demonstrations, query), axis=0)

    full_h0 = run_context(full, context)
    query_only_h0 = run_context(query_only, context)

    for name in (
        "compact_logits",
        "spikes",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    ):
        np.testing.assert_array_equal(
            getattr(full_h0, name), getattr(query_only_h0, name)
        )
    _assert_state_snapshots_equal(full_h0.snapshot, query_only_h0.snapshot)
    assert np.any(_state_array(query_only.memory_read) != 0.0)

    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)
    query_only_snapshot = query_only.snapshot_state()
    before_physical = (
        np.asarray(query_only.voltage).copy(),
        np.asarray(query_only.feedforward_current).copy(),
        np.asarray(query_only.recurrent_current).copy(),
    )
    full.cell_step(zero_event, advance)
    query_only.cell_step(zero_event, advance)
    baseline_physical = (
        np.asarray(query_only.voltage).copy(),
        np.asarray(query_only.feedforward_current).copy(),
        np.asarray(query_only.recurrent_current).copy(),
    )
    baseline_reasoning = _state_array(query_only.reasoning_query).copy()
    baseline_workspace = _state_array(query_only.workspace_carrier).copy()

    assert np.any(_state_array(full.memory_read) != 0.0)
    np.testing.assert_array_equal(_state_array(query_only.memory_read), 0.0)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(before_physical, baseline_physical, strict=True)
    )

    query_only.restore_state(query_only_snapshot)
    query_only.context_memory.value = query_only.context_memory.value + 7.0
    query_only.cell_step(zero_event, advance)

    np.testing.assert_array_equal(_state_array(query_only.memory_read), 0.0)
    np.testing.assert_array_equal(
        _state_array(query_only.reasoning_query), baseline_reasoning
    )
    np.testing.assert_array_equal(
        _state_array(query_only.workspace_carrier), baseline_workspace
    )
    for expected, actual in zip(
        baseline_physical,
        (
            np.asarray(query_only.voltage),
            np.asarray(query_only.feedforward_current),
            np.asarray(query_only.recurrent_current),
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fill_value", [7.0, -7.0], ids=["plus_7", "minus_7"])
def test_query_only_latent_read_drive_and_memory_perturbation_contract(
    fill_value: float,
) -> None:
    config = _memory_config(memory_decay=1.0, input_gain=8.0)
    full = LatentWorkspaceModel(config, memory_read_policy="full")
    query_only = LatentWorkspaceModel(config, memory_read_policy="query_only")
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    context = jnp.concatenate((demonstrations, query), axis=0)

    full_h0 = run_context(full, context)
    query_only_h0 = run_context(query_only, context)

    for name in (
        "compact_logits",
        "spikes",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    ):
        np.testing.assert_array_equal(
            getattr(full_h0, name), getattr(query_only_h0, name)
        )
    _assert_state_snapshots_equal(full_h0.snapshot, query_only_h0.snapshot)
    h0_read = _state_array(query_only.memory_read).copy()
    assert np.any(h0_read != 0.0)

    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)
    query_only_h0_snapshot = query_only.snapshot_state()
    full_h0_snapshot = full.snapshot_state()
    query_only_parameters = parameter_snapshot(query_only)
    full_parameters = parameter_snapshot(full)
    before_physical = (
        np.asarray(query_only.voltage).copy(),
        np.asarray(query_only.feedforward_current).copy(),
        np.asarray(query_only.recurrent_current).copy(),
    )

    @brainstate.transform.jit
    def query_only_latent_step(
        event: jax.Array, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        compact = query_only.update(event, gate)
        selected_read = jnp.asarray(query_only.memory_read.value)
        selected_drive = query_only.memory_read_projection(selected_read)
        return compact, selected_read, selected_drive

    @brainstate.transform.jit
    def full_latent_step(
        event: jax.Array, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        compact = full.update(event, gate)
        selected_read = jnp.asarray(full.memory_read.value)
        selected_drive = full.memory_read_projection(selected_read)
        return compact, selected_read, selected_drive

    baseline_query_output, baseline_query_read, baseline_query_drive = (
        jax.block_until_ready(query_only_latent_step(zero_event, advance))
    )
    baseline_query_snapshot = query_only.snapshot_state()
    baseline_query_prediction = _decoded_argmax_prediction(
        baseline_query_output, config.color_rank
    )
    baseline_physical = (
        np.asarray(query_only.voltage).copy(),
        np.asarray(query_only.feedforward_current).copy(),
        np.asarray(query_only.recurrent_current).copy(),
    )

    baseline_full_output, baseline_full_read, baseline_full_drive = (
        jax.block_until_ready(full_latent_step(zero_event, advance))
    )
    assert np.any(np.asarray(baseline_full_read) != 0.0)
    assert np.any(np.asarray(baseline_full_drive) != 0.0)
    np.testing.assert_array_equal(baseline_query_read, 0.0)
    np.testing.assert_array_equal(baseline_query_drive, 0.0)
    assert not np.array_equal(np.asarray(baseline_query_read), h0_read)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(before_physical, baseline_physical, strict=True)
    )

    query_only.restore_state(query_only_h0_snapshot)
    source_memory = np.asarray(query_only.context_memory.value).copy()
    query_only.context_memory.value = jnp.full_like(
        query_only.context_memory.value, fill_value
    )
    assert not np.array_equal(
        np.asarray(query_only.context_memory.value), source_memory
    )
    _assert_non_context_snapshot_close(
        query_only_h0_snapshot,
        query_only.snapshot_state(),
        limit=0.0,
    )
    output, selected_read, selected_drive = jax.block_until_ready(
        query_only_latent_step(zero_event, advance)
    )

    np.testing.assert_array_equal(selected_read, 0.0)
    np.testing.assert_array_equal(selected_drive, 0.0)
    assert not np.array_equal(np.asarray(selected_read), h0_read)
    _assert_max_per_example_rms(
        baseline_query_output,
        output,
        limit=1e-6,
        label="compact logits",
    )
    _assert_non_context_snapshot_close(
        baseline_query_snapshot,
        query_only.snapshot_state(),
        limit=1e-6,
    )
    actual_prediction = _decoded_argmax_prediction(output, config.color_rank)
    for expected, actual in zip(
        baseline_query_prediction, actual_prediction, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    _assert_parameter_snapshots_equal(
        query_only_parameters, parameter_snapshot(query_only)
    )

    full.restore_state(full_h0_snapshot)
    full.context_memory.value = jnp.full_like(full.context_memory.value, fill_value)
    full_output, full_read, full_drive = jax.block_until_ready(
        full_latent_step(zero_event, advance)
    )
    assert np.all(np.isfinite(np.asarray(full_output)))
    read_difference = np.max(
        np.abs(np.asarray(full_read) - np.asarray(baseline_full_read))
    )
    drive_difference = np.max(
        np.abs(np.asarray(full_drive) - np.asarray(baseline_full_drive))
    )
    assert read_difference > 0.0 or drive_difference > 0.0
    _assert_parameter_snapshots_equal(full_parameters, parameter_snapshot(full))


def test_max_per_example_rms_rejects_a_concentrated_outlier() -> None:
    limit = 1e-6
    baseline = np.zeros((2, 100), dtype=np.float64)
    inside_boundary = baseline.copy()
    inside_boundary[0, 0] = 0.999 * limit * math.sqrt(100.0)
    _assert_max_per_example_rms(
        baseline,
        inside_boundary,
        limit=limit,
        label="inside boundary",
    )

    concentrated_outlier = baseline.copy()
    concentrated_outlier[0, 0] = 1.001 * limit * math.sqrt(100.0)
    global_rms = float(np.sqrt(np.mean(np.square(concentrated_outlier - baseline))))
    assert global_rms < limit
    assert _max_per_example_rms(baseline, concentrated_outlier) > limit
    with pytest.raises(AssertionError, match="max per-example RMS"):
        _assert_max_per_example_rms(
            baseline,
            concentrated_outlier,
            limit=limit,
            label="concentrated outlier",
        )


@pytest.mark.parametrize("cached_read_fill", [11.0, -11.0], ids=["plus_11", "minus_11"])
def test_query_only_latent_step_discards_perturbed_nonzero_cached_h0_read(
    cached_read_fill: float,
) -> None:
    config = _memory_config(memory_decay=1.0, input_gain=8.0)
    query_only = LatentWorkspaceModel(config, memory_read_policy="query_only")
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    context = jnp.concatenate((demonstrations, query), axis=0)

    query_only_h0 = run_context(query_only, context)
    h0_cached_read = _state_array(query_only.memory_read).copy()
    assert np.any(h0_cached_read != 0.0)

    query_only_h0_snapshot = query_only_h0.snapshot
    query_only_parameters = parameter_snapshot(query_only)
    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)
    cached_read_sentinel = np.float32(cached_read_fill)

    @brainstate.transform.jit
    def query_only_latent_step(
        event: jax.Array, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        compact = query_only.update(event, gate)
        selected_read = jnp.asarray(query_only.memory_read.value)
        selected_drive = query_only.memory_read_projection(selected_read)
        return compact, selected_read, selected_drive

    query_only.restore_state(query_only_h0_snapshot)
    baseline_output, baseline_read, baseline_drive = jax.block_until_ready(
        query_only_latent_step(zero_event, advance)
    )
    baseline_snapshot = query_only.snapshot_state()
    baseline_prediction = _decoded_argmax_prediction(baseline_output, config.color_rank)
    np.testing.assert_array_equal(baseline_read, 0.0)
    np.testing.assert_array_equal(baseline_drive, 0.0)

    query_only.restore_state(query_only_h0_snapshot)
    source_context = np.asarray(query_only.context_memory.value).copy()
    before_replacement = query_only.snapshot_state()
    query_only.memory_read.value = jnp.full_like(
        query_only.memory_read.value, cached_read_sentinel
    )
    after_replacement = query_only.snapshot_state()
    np.testing.assert_array_equal(query_only.context_memory.value, source_context)
    np.testing.assert_array_equal(
        np.asarray(query_only.memory_read.value), cached_read_sentinel
    )
    assert not np.array_equal(np.asarray(query_only.memory_read.value), h0_cached_read)
    before_entries = _snapshot_entry_map(before_replacement)
    after_entries = _snapshot_entry_map(after_replacement)
    assert before_entries.keys() == after_entries.keys()
    changed_paths = []
    for path, before_value in before_entries.items():
        before_arrays = _tree_arrays(before_value)
        after_arrays = _tree_arrays(after_entries[path])
        assert len(before_arrays) == len(after_arrays)
        if any(
            not np.array_equal(before_array, after_array)
            for before_array, after_array in zip(
                before_arrays, after_arrays, strict=True
            )
        ):
            changed_paths.append(path)
    assert changed_paths == [("memory_read",)]
    _assert_parameter_snapshots_equal(
        query_only_parameters, parameter_snapshot(query_only)
    )

    output, selected_read, selected_drive = jax.block_until_ready(
        query_only_latent_step(zero_event, advance)
    )
    perturbed_snapshot = query_only.snapshot_state()
    np.testing.assert_array_equal(selected_read, 0.0)
    np.testing.assert_array_equal(selected_drive, 0.0)
    assert not np.array_equal(np.asarray(selected_read), h0_cached_read)
    assert not np.array_equal(
        np.asarray(selected_read),
        np.full_like(h0_cached_read, cached_read_sentinel),
    )
    _assert_max_per_example_rms(
        baseline_output,
        output,
        limit=1e-6,
        label="compact logits",
    )
    _assert_non_context_snapshot_close(
        baseline_snapshot,
        perturbed_snapshot,
        limit=1e-6,
    )
    baseline_context = _snapshot_entry_map(baseline_snapshot)[("context_memory",)]
    perturbed_context = _snapshot_entry_map(perturbed_snapshot)[("context_memory",)]
    for expected, actual in zip(
        _tree_arrays(baseline_context),
        _tree_arrays(perturbed_context),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, source_context)
    actual_prediction = _decoded_argmax_prediction(output, config.color_rank)
    for expected, actual in zip(baseline_prediction, actual_prediction, strict=True):
        np.testing.assert_array_equal(actual, expected)
    _assert_parameter_snapshots_equal(
        query_only_parameters, parameter_snapshot(query_only)
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
    expected = memory_model.encode_memory_key(
        queries[0]
    ) + memory_model.encode_memory_key(queries[1])

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
    perturbation = jnp.linspace(-1.0, 1.0, config.neuron_count, dtype=jnp.float32)[
        None, :
    ]
    memory_model.workspace_carrier.value = (
        memory_model.workspace_carrier.value + perturbation
    )
    np.testing.assert_array_equal(memory_model.context_memory.value, frozen_memory)
    np.testing.assert_array_equal(memory_model.query_encoding.value, frozen_query)
    memory_model.cell_step(zero_event, advance)

    assert not np.allclose(_state_array(memory_model.memory_read), baseline_read)
    np.testing.assert_array_equal(memory_model.context_memory.value, frozen_memory)
    np.testing.assert_array_equal(memory_model.query_encoding.value, frozen_query)


@pytest.mark.parametrize("memory_read_policy", ["full", "query_only"])
def test_memory_etp_paths_are_direct_with_finite_window_pp_prop_gradients(
    memory_read_policy: str,
) -> None:
    import braintrace
    from braintrace._testing.oracle import chunked_online_param_gradients

    config = _memory_config(memory_decay=0.9, input_gain=8.0)
    model = LatentWorkspaceModel(config, memory_read_policy=memory_read_policy)
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
        {"context_memory", "workspace_carrier"}.issubset(paths) for paths in group_paths
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
        lambda: _AlwaysAdvanceMemoryModel(
            config, memory_read_policy=memory_read_policy
        ),
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
        if memory_read_policy == "query_only" and path == query_path:
            assert gradient_norm == 0.0
        else:
            assert gradient_norm > 0.0, path


def test_query_only_latent_window_has_zero_read_path_gradients_and_live_control() -> (
    None
):
    import braintrace
    from braintrace._testing.oracle import chunked_online_param_gradients
    from examples.pp_prop import latent_workspace_binding_control as legacy

    config = _memory_config(memory_decay=1.0, input_gain=8.0)
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    h0_prefix = jnp.concatenate((demonstrations, query), axis=0)

    class _MaterializedLatentObjective(LatentWorkspaceModel):
        def __init__(self, memory_read_policy: str):
            super().__init__(
                config,
                memory_read_policy=memory_read_policy,  # type: ignore[arg-type]
            )
            run_context(self, h0_prefix)
            self._materialized_h0_snapshot = self.snapshot_state()

        @brainstate.nn.call_order(9)
        def init_state(self, *, batch_size: int | None = None, **_: object) -> None:
            if batch_size is not None and batch_size != self.config.batch_size:
                raise ValueError("batch size differs from the materialized objective")
            snapshot = getattr(self, "_materialized_h0_snapshot", None)
            if snapshot is not None:
                self.restore_state(snapshot)

        def update(self, packed: jax.Array) -> jax.Array:
            width = self.config.input_width
            if packed.shape != (self.config.batch_size, width + 3):
                raise ValueError("packed latent objective has the wrong shape")
            event = packed[:, :width]
            advance = packed[:, width] > 0.5
            target = packed[:, width + 1].astype(jnp.int32)
            weight = packed[:, width + 2]
            raw_loss = legacy._classification_loss(
                super().update(event, advance),
                target,
                self.config.color_rank,
            )
            weighted = jnp.sqrt(weight) * jnp.sqrt(jnp.maximum(raw_loss, 0.0))
            return jnp.where(weight == 0.0, jnp.zeros_like(weight), weighted)

    packed = np.zeros((1, 1, config.input_width + 3), dtype=np.float32)
    packed[0, 0, config.input_width] = 1.0
    packed[0, 0, config.input_width + 1] = 8.0
    packed[0, 0, config.input_width + 2] = np.float32(0.5)
    inputs = jnp.asarray(packed)
    assert float(np.asarray(inputs[0, 0, -1])) == 0.5

    def algorithm_factory(
        candidate: brainstate.nn.Module,
    ) -> braintrace.ETraceAlgorithm:
        assert isinstance(candidate, LatentWorkspaceModel)
        return braintrace.pp_prop(
            candidate,
            decay_or_rank=candidate.config.trace_decay,
            vjp_method="multi-step",
        )

    query_only_gradients = chunked_online_param_gradients(
        lambda: _MaterializedLatentObjective("query_only"),
        inputs,
        algo_factory=algorithm_factory,
        chunk_size=1,
    )
    full_gradients = chunked_online_param_gradients(
        lambda: _MaterializedLatentObjective("full"),
        inputs,
        algo_factory=algorithm_factory,
        chunk_size=1,
    )

    def gradient_l2(value: object) -> float:
        squared = 0.0
        for leaf in jax.tree.leaves(value):
            array = np.asarray(u.get_mantissa(leaf))
            assert np.all(np.isfinite(array))
            squared += float(np.sum(array.astype(np.float64) ** 2))
        return math.sqrt(squared)

    removed_paths = (
        ("memory_read_projection", "weight"),
        ("workspace_query_projection", "weight"),
    )
    live_paths = (
        ("color_factor_head", "weight"),
        ("readout_projection", "weight"),
        ("rec_syn", "comm", "weight"),
    )
    assert tuple(query_only_gradients) == tuple(full_gradients)
    for path in removed_paths:
        query_leaves = jax.tree.leaves(query_only_gradients[path])
        assert query_leaves
        for leaf in query_leaves:
            array = np.asarray(u.get_mantissa(leaf))
            np.testing.assert_array_equal(array, np.zeros_like(array))
        assert gradient_l2(query_only_gradients[path]) == 0.0
        assert gradient_l2(full_gradients[path]) > 0.0

    for path in live_paths:
        assert gradient_l2(query_only_gradients[path]) > 0.0

    assert gradient_l2(query_only_gradients) > 0.0
    assert gradient_l2(full_gradients) > 0.0


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


def test_memory_carrier_cap_is_confined_to_both_dense_consumer_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_l2_cap = latent_workspace_module._unit_l2_cap
    observed: list[np.ndarray] = []

    def recording_cap(carrier: jax.Array) -> jax.Array:
        observed.append(np.asarray(carrier).copy())
        return unit_l2_cap(carrier)

    monkeypatch.setattr(latent_workspace_module, "_unit_l2_cap", recording_cap)
    config = _memory_config()
    memory_model = LatentWorkspaceModel(config)
    raw_workspace = jnp.linspace(-8.0, 8.0, config.neuron_count, dtype=jnp.float32)[
        None, :
    ]
    memory_model.workspace_carrier.value = raw_workspace
    capped_workspace = unit_l2_cap(raw_workspace)

    def compact_formula(carrier: jax.Array) -> jax.Array:
        hidden = jax.nn.gelu(memory_model.readout_projection(carrier))
        return jnp.concatenate(
            (
                memory_model.height_head(hidden),
                memory_model.width_head(hidden),
                memory_model.color_factor_head(hidden),
            ),
            axis=-1,
        )

    actual_logits = memory_model.compact_readout(raw_workspace)
    expected_logits = compact_formula(capped_workspace)
    uncapped_logits = compact_formula(raw_workspace)

    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-6, atol=1e-6)
    assert not np.allclose(actual_logits, uncapped_logits)
    np.testing.assert_array_equal(memory_model.workspace_carrier.value, raw_workspace)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], raw_workspace)

    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)
    memory_model.query_encoding.value = jnp.zeros(
        (1, config.context_memory_width), dtype=jnp.float32
    )
    expected_query = jnp.tanh(memory_model.workspace_query_projection(capped_workspace))
    uncapped_query = jnp.tanh(memory_model.workspace_query_projection(raw_workspace))

    memory_model.cell_step(zero_event, advance)

    np.testing.assert_allclose(
        memory_model.reasoning_query.value,
        expected_query,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(memory_model.reasoning_query.value, uncapped_query)
    assert len(observed) == 2
    np.testing.assert_array_equal(observed[1], raw_workspace)

    memory_model.workspace_carrier.value = raw_workspace
    memory_model.cell_step(zero_event, jnp.zeros((1,), dtype=jnp.bool_))
    np.testing.assert_array_equal(memory_model.workspace_carrier.value, raw_workspace)
    assert len(observed) == 3
    np.testing.assert_array_equal(observed[2], raw_workspace)

    legacy_model = LatentWorkspaceModel(_config())
    legacy_model.compact_readout(raw_workspace)
    legacy_model.cell_step(
        jnp.zeros((1, legacy_model.config.input_width), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.bool_),
    )
    assert len(observed) == 3


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
    explicit_model = LatentWorkspaceModel(
        _config(context_memory_width=0), memory_read_policy="full"
    )
    query_only_model = LatentWorkspaceModel(
        _config(context_memory_width=0), memory_read_policy="query_only"
    )
    for candidate in (explicit_model, query_only_model):
        _assert_parameter_snapshots_equal(
            parameter_snapshot(implicit_model), parameter_snapshot(candidate)
        )
        _assert_state_snapshots_equal(
            implicit_model.snapshot_state(), candidate.snapshot_state()
        )
    implicit_report = msgspec_json.dumps(
        dataclasses.asdict(implicit_model.associative_memory_report()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    explicit_report = msgspec_json.dumps(
        dataclasses.asdict(explicit_model.associative_memory_report()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert implicit_report == explicit_report
    events = _valid_events(4, implicit_model.config.input_width)

    implicit = run_sequence(implicit_model, events, latent_steps=2)
    explicit = run_sequence(explicit_model, events, latent_steps=2)
    query_only = run_sequence(query_only_model, events, latent_steps=2)

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
        np.testing.assert_array_equal(
            getattr(implicit.trajectory, name),
            getattr(query_only.trajectory, name),
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
    _assert_state_snapshots_equal(
        implicit_model.snapshot_state(), query_only_model.snapshot_state()
    )


def test_memory_read_transform_defaults_to_linear_and_validates() -> None:
    assert _memory_config().memory_read_transform == "linear"
    assert _config().memory_read_transform == "linear"
    for transform in ("gated", "gated_rms"):
        assert _memory_config(memory_read_transform=transform).memory_read_transform == transform
    with pytest.raises(TypeError, match="memory_read_transform"):
        _memory_config(memory_read_transform=7)
    with pytest.raises(ValueError, match="memory_read_transform"):
        _memory_config(memory_read_transform="rms")
    with pytest.raises(ValueError, match="positive context_memory_width"):
        _config(memory_read_transform="gated")


def test_memory_read_interval_defaults_and_validates() -> None:
    assert _config().memory_read_interval == 1
    assert _memory_config().memory_read_interval == 1
    assert _memory_config(memory_read_interval=8).memory_read_interval == 8
    for value in (True, 1.5, "4"):
        with pytest.raises(TypeError, match="memory_read_interval"):
            _memory_config(memory_read_interval=value)
    with pytest.raises(ValueError, match="memory_read_interval"):
        _memory_config(memory_read_interval=0)


@pytest.mark.parametrize(
    ("interval", "expected_mask"),
    [
        (1, [False, False, True, True, True, True, True, True, True, True, True]),
        (4, [False, False, True, False, False, False, True, False, False, False, True]),
        (8, [False, False, True, False, False, False, False, False, False, False, True]),
    ],
)
def test_memory_read_interval_records_exact_compiled_mask_and_count(
    interval: int,
    expected_mask: list[bool],
) -> None:
    config = _memory_config(memory_read_interval=interval, max_latent_steps=8)
    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    latent = jnp.zeros((8, 1, config.input_width), dtype=jnp.float32)
    events = jnp.concatenate((demonstrations, query, latent), axis=0)
    advances = jnp.ones((events.shape[0], 1), dtype=jnp.bool_)

    full = run_packed_stream(
        LatentWorkspaceModel(config), events, advance_gates=advances
    )
    selected = run_selected_packed_stream(
        LatentWorkspaceModel(config),
        events,
        jnp.asarray([[2], [6], [10]], dtype=jnp.int32),
        advance_gates=advances,
    )

    expected = np.asarray(expected_mask, dtype=np.bool_)[:, None]
    np.testing.assert_array_equal(full.memory_read_mask, expected)
    np.testing.assert_array_equal(selected.memory_read_mask, expected)
    np.testing.assert_array_equal(full.memory_read_count, [sum(expected_mask)])
    np.testing.assert_array_equal(selected.memory_read_count, [sum(expected_mask)])


def test_memory_read_interval_retains_diagnostics_on_local_ticks_and_resets() -> None:
    config = _memory_config(memory_read_interval=4, max_latent_steps=4)
    model = LatentWorkspaceModel(config)
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")[0]
    model.cell_step(query, jnp.ones((1,), dtype=jnp.bool_))
    retained_read = np.asarray(model.memory_read.value).copy()
    retained_drive = np.asarray(model.memory_drive.value).copy()
    snapshot = model.snapshot_state()

    run_packed_stream(
        model,
        jnp.zeros((3, 1, config.input_width), dtype=jnp.float32),
        reset=False,
        advance_gates=jnp.ones((3, 1), dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(model.memory_read.value, retained_read)
    np.testing.assert_array_equal(model.memory_drive.value, retained_drive)
    np.testing.assert_array_equal(model.memory_read_active.value, [False])
    np.testing.assert_array_equal(model.memory_read_count.value, [1])
    model.restore_state(snapshot)
    np.testing.assert_array_equal(model.memory_read_active.value, [True])
    np.testing.assert_array_equal(model.memory_read_count.value, [1])
    model.reset_state()
    np.testing.assert_array_equal(model.memory_read_step.value, 0)
    np.testing.assert_array_equal(model.memory_read_count.value, 0)
    np.testing.assert_array_equal(model.memory_read_active.value, False)


def test_query_only_policy_suppresses_interval_reads() -> None:
    config = _memory_config(memory_read_interval=4, max_latent_steps=8)
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    latent = jnp.zeros((8, 1, config.input_width), dtype=jnp.float32)
    events = jnp.concatenate((query, latent), axis=0)
    trajectory = run_packed_stream(
        LatentWorkspaceModel(config, memory_read_policy="query_only"),
        events,
        advance_gates=jnp.ones((9, 1), dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(
        trajectory.memory_read_mask[:, 0],
        [True, False, False, False, False, False, False, False, False],
    )
    np.testing.assert_array_equal(trajectory.memory_read_count, [1])


def test_latent_residual_configuration_is_opt_in_and_validates() -> None:
    default = _memory_config()
    enabled = _memory_config(
        latent_residual_mixer="attention_residual",
        latent_residual_block_size=2,
    )
    assert default.latent_residual_mixer == "none"
    assert default.latent_residual_block_size == 10
    assert enabled.latent_residual_mixer == "attention_residual"
    assert enabled.latent_residual_block_size == 2
    assert not hasattr(LatentWorkspaceModel(default), "latent_attention_residual")
    for value in (True, 1.5, "2"):
        with pytest.raises(TypeError, match="latent_residual_block_size"):
            _memory_config(latent_residual_block_size=value)
    with pytest.raises(ValueError, match="latent_residual_block_size"):
        _memory_config(latent_residual_block_size=0)
    with pytest.raises(TypeError, match="latent_residual_mixer"):
        _memory_config(latent_residual_mixer=2)
    with pytest.raises(ValueError, match="latent_residual_mixer"):
        _memory_config(latent_residual_mixer="linear")
    with pytest.raises(ValueError, match="positive context_memory_width"):
        _config(latent_residual_mixer="attention_residual")


def test_latent_attention_residual_tracks_full_and_partial_blocks() -> None:
    config = _memory_config(
        max_latent_steps=5,
        latent_residual_mixer="attention_residual",
        latent_residual_block_size=2,
    )
    model = LatentWorkspaceModel(config)
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    model.cell_step(query[0], jnp.ones((1,), dtype=jnp.bool_))
    source_zero = np.asarray(model.reasoning_query.value).copy()
    np.testing.assert_array_equal(model.latent_residual_source_zero.value, source_zero)
    np.testing.assert_array_equal(model.latent_residual_completed_count.value, 0)

    one_tick = jnp.zeros((1, 1, config.input_width), dtype=jnp.float32)
    one_advance = jnp.ones((1, 1), dtype=jnp.bool_)
    run_packed_stream(model, one_tick, reset=False, advance_gates=one_advance)
    np.testing.assert_array_equal(model.latent_residual_block_count.value, 1)
    np.testing.assert_array_equal(model.latent_residual_completed_count.value, 0)
    run_packed_stream(model, one_tick, reset=False, advance_gates=one_advance)
    np.testing.assert_array_equal(model.latent_residual_block_count.value, 0)
    np.testing.assert_array_equal(model.latent_residual_completed_count.value, 1)
    assert np.any(np.asarray(model.latent_residual_history.value[:, 0]) != 0.0)

    two_ticks = jnp.zeros((2, 1, config.input_width), dtype=jnp.float32)
    two_advances = jnp.ones((2, 1), dtype=jnp.bool_)
    run_packed_stream(model, two_ticks, reset=False, advance_gates=two_advances)
    np.testing.assert_array_equal(model.latent_residual_completed_count.value, 2)
    run_packed_stream(model, one_tick, reset=False, advance_gates=one_advance)
    np.testing.assert_array_equal(model.latent_residual_completed_count.value, 3)
    np.testing.assert_array_equal(model.latent_residual_block_count.value, 0)
    assert isinstance(model.latent_residual_candidate, brainstate.HiddenState)


def test_latent_attention_residual_state_resets_snapshots_and_restores() -> None:
    config = _memory_config(
        max_latent_steps=3,
        latent_residual_mixer="attention_residual",
        latent_residual_block_size=2,
    )
    model = LatentWorkspaceModel(config)
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    events = jnp.concatenate(
        (query, jnp.zeros((2, 1, config.input_width), dtype=jnp.float32)), axis=0
    )
    run_packed_stream(
        model,
        events,
        advance_gates=jnp.ones((3, 1), dtype=jnp.bool_),
    )
    snapshot = model.snapshot_state()
    history = np.asarray(model.latent_residual_history.value).copy()
    model.reset_state()
    np.testing.assert_array_equal(model.latent_residual_history.value, 0.0)
    np.testing.assert_array_equal(model.latent_residual_candidate.value, 0.0)
    model.restore_state(snapshot)
    np.testing.assert_array_equal(model.latent_residual_history.value, history)


def test_latent_attention_residual_parameter_is_trace_compiled() -> None:
    model = LatentWorkspaceModel(
        _memory_config(
            latent_residual_mixer="attention_residual",
            latent_residual_block_size=2,
        )
    )

    learner = compile_etrace(model)

    paths = {path for path, _ in learner.report.etrace_weights}
    assert ("latent_attention_residual", "query") in paths


def test_gated_memory_read_is_linear_equivalent_at_initialization() -> None:
    linear = LatentWorkspaceModel(_memory_config(memory_read_transform="linear"))
    gated = LatentWorkspaceModel(_memory_config(memory_read_transform="gated"))
    raw_read = jnp.asarray([[0.4, -1.2]], dtype=jnp.float32)
    workspace = jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32)[None, :]

    linear_drive = linear.memory_read_projection(raw_read)
    gated_drive = gated.memory_read_projection(
        raw_read, latent_workspace_module._unit_l2_cap(workspace)
    )

    np.testing.assert_array_equal(gated.memory_read_projection.gate_weight.value, 0.0)
    np.testing.assert_array_equal(gated.memory_read_projection.gate_bias.value, 0.0)
    np.testing.assert_array_equal(
        gated.memory_read_projection.output_weight.value,
        2.0 * linear.memory_read_projection.weight.value["weight"],
    )
    np.testing.assert_allclose(gated_drive, linear_drive, rtol=0.0, atol=0.0)


def test_gated_rms_memory_read_is_finite_and_reported() -> None:
    model = LatentWorkspaceModel(
        _memory_config(memory_read_transform="gated_rms")
    )
    raw_read = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)
    workspace = jnp.ones((1, 64), dtype=jnp.float32)

    drive = model.memory_read_projection(
        raw_read, latent_workspace_module._unit_l2_cap(workspace)
    )
    report = model.associative_memory_report()

    assert np.all(np.isfinite(np.asarray(drive)))
    np.testing.assert_array_equal(drive, 0.0)
    assert report.read_transform == "gated_rms"
    assert report.read_component_type == "braintrace.nn.GatedProjection"
    model.memory_read.value = jnp.asarray([[3.0, 4.0]], dtype=jnp.float32)
    model.memory_drive.value = jnp.ones((1, 64), dtype=jnp.float32)
    model.memory_read_gate.value = jnp.asarray([[0.5, 0.999]], dtype=jnp.float32)
    diagnostics = model.memory_read_diagnostics()
    assert diagnostics["memory_read_rms"] == pytest.approx(math.sqrt(12.5))
    assert diagnostics["memory_drive_rms"] == pytest.approx(1.0)
    assert diagnostics["gate_saturation_fraction"] == pytest.approx(0.5)
    assert diagnostics["gate_channel_activation"] == pytest.approx([0.5, 0.999])
    model.reset_state()
    assert model.memory_read_diagnostics()["memory_read_rms"] == 0.0


def test_gated_memory_read_parameters_are_trace_compiled() -> None:
    model = LatentWorkspaceModel(_memory_config(memory_read_transform="gated"))

    learner = compile_etrace(model)

    paths = {path for path, _ in learner.report.etrace_weights}
    assert ("memory_read_projection", "gate_weight") in paths
    assert ("memory_read_projection", "gate_bias") in paths
    assert ("memory_read_projection", "output_weight") in paths


def test_zero_width_compact_readout_is_byte_identical_to_raw_legacy_formula() -> None:
    legacy_model = LatentWorkspaceModel(_config(context_memory_width=0))
    carrier = jnp.linspace(
        -8.0, 8.0, legacy_model.config.neuron_count, dtype=jnp.float32
    )[None, :]
    hidden = jax.nn.gelu(legacy_model.readout_projection(carrier))
    expected = jnp.concatenate(
        (
            legacy_model.height_head(hidden),
            legacy_model.width_head(hidden),
            legacy_model.color_factor_head(hidden),
        ),
        axis=-1,
    )

    actual = legacy_model.compact_readout(carrier)

    np.testing.assert_array_equal(actual, expected)


def test_zero_width_compiler_paths_and_finite_window_gradients_are_byte_identical() -> (
    None
):
    import braintrace
    from braintrace._testing.oracle import chunked_online_param_gradients

    implicit_config = _config(recurrent_edges=64)
    explicit_config = _config(recurrent_edges=64, context_memory_width=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        implicit_learner = compile_pp_prop(LatentWorkspaceModel(implicit_config))
        explicit_learner = compile_pp_prop(
            LatentWorkspaceModel(explicit_config, memory_read_policy="full")
        )
        query_only_learner = compile_pp_prop(
            LatentWorkspaceModel(explicit_config, memory_read_policy="query_only")
        )
    implicit_report = implicit_learner.report
    for candidate_report in (explicit_learner.report, query_only_learner.report):
        assert implicit_report.counts == candidate_report.counts
        assert implicit_report.hidden_groups == candidate_report.hidden_groups
        assert implicit_report.etrace_weights == candidate_report.etrace_weights
        assert implicit_report.excluded_weights == candidate_report.excluded_weights
        assert implicit_report.dynamic_states == candidate_report.dynamic_states
        assert implicit_report.to_str(2) == candidate_report.to_str(2)

    events = _valid_events(3, implicit_config.input_width)[:, None, :]

    def pp_prop_factory(
        candidate: LatentWorkspaceModel,
    ) -> braintrace.ETraceAlgorithm:
        return braintrace.pp_prop(
            candidate,
            decay_or_rank=candidate.config.trace_decay,
            vjp_method="multi-step",
        )

    implicit_gradients = chunked_online_param_gradients(
        lambda: LatentWorkspaceModel(implicit_config),
        events,
        algo_factory=pp_prop_factory,
        chunk_size=1,
    )
    explicit_gradients = chunked_online_param_gradients(
        lambda: LatentWorkspaceModel(explicit_config, memory_read_policy="full"),
        events,
        algo_factory=pp_prop_factory,
        chunk_size=1,
    )
    query_only_gradients = chunked_online_param_gradients(
        lambda: LatentWorkspaceModel(explicit_config, memory_read_policy="query_only"),
        events,
        algo_factory=pp_prop_factory,
        chunk_size=1,
    )

    for candidate_gradients in (explicit_gradients, query_only_gradients):
        assert tuple(implicit_gradients) == tuple(candidate_gradients)
        for path in implicit_gradients:
            implicit_arrays = _tree_arrays(implicit_gradients[path])
            candidate_arrays = _tree_arrays(candidate_gradients[path])
            assert len(implicit_arrays) == len(candidate_arrays), path
            for implicit, candidate in zip(
                implicit_arrays, candidate_arrays, strict=True
            ):
                assert implicit.shape == candidate.shape, path
                assert implicit.dtype == candidate.dtype, path
                assert implicit.tobytes() == candidate.tobytes(), path


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


def _uniform_spatial_color_logits(color_logits: jax.Array) -> jax.Array:
    rank = 1
    compact = jnp.zeros((1, compact_output_width(rank)))
    row_start = 2 * MAX_GRID_SIZE
    column_start = row_start + MAX_GRID_SIZE
    color_start = column_start + MAX_GRID_SIZE
    compact = compact.at[:, row_start:column_start].set(1.0)
    compact = compact.at[:, column_start:color_start].set(1.0)
    return compact.at[:, color_start:].set(color_logits)


def test_class_balancing_is_opt_in_and_matches_legacy_for_one_color() -> None:
    compact = _uniform_spatial_color_logits(jnp.arange(COLOR_COUNT) / 3.0)
    colors = jnp.zeros((1, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=jnp.int32)
    colors = colors.at[:, :2, :3].set(4)
    arguments = (compact, jnp.asarray([2]), jnp.asarray([3]), colors)

    default = arc_loss_components(*arguments, color_rank=1)
    explicit_legacy = arc_loss_components(
        *arguments, color_rank=1, class_balanced_colors=False
    )
    balanced = arc_loss_components(*arguments, color_rank=1, class_balanced_colors=True)

    np.testing.assert_allclose(default, explicit_legacy, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(balanced, explicit_legacy, rtol=1e-6, atol=1e-6)


def test_class_balancing_gives_rare_color_equal_total_weight() -> None:
    logits = jnp.asarray([3.0, -3.0] + [-3.0] * (COLOR_COUNT - 2))
    compact = _uniform_spatial_color_logits(logits)
    colors = jnp.zeros((1, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=jnp.int32)
    colors = colors.at[0, 1, 1].set(1)
    arguments = (compact, jnp.asarray([2]), jnp.asarray([2]), colors)

    legacy = arc_loss_components(*arguments, color_rank=1, class_balanced_colors=False)
    balanced = arc_loss_components(*arguments, color_rank=1, class_balanced_colors=True)
    nll = -jax.nn.log_softmax(logits)

    assert float(legacy.colors) == pytest.approx(
        float(0.75 * nll[0] + 0.25 * nll[1]), rel=1e-6
    )
    assert float(balanced.colors) == pytest.approx(
        float(0.5 * nll[0] + 0.5 * nll[1]), rel=1e-6
    )
    assert float(balanced.colors) > float(legacy.colors)


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


def test_row_refinement_is_opt_in_and_legacy_output_contract_is_unchanged() -> None:
    legacy = _config()
    refinement = _row_refinement_config(max_latent_steps=60, refinement_steps=60)

    assert legacy.decoder_mode == "legacy_cp"
    assert legacy.row_refinement_enabled is False
    assert legacy.compact_output_width == compact_output_width(legacy.color_rank) == 200
    assert legacy.training_output_width == 200
    assert legacy.checkpoint_output_width == 200
    assert not hasattr(LatentWorkspaceModel(legacy), "answer_row_head")

    assert refinement.decoder_mode == "row_refinement"
    assert refinement.row_refinement_enabled is True
    assert refinement.refinement_steps == 60
    assert refinement.compact_output_width == 200
    assert refinement.training_output_width == 360
    assert refinement.checkpoint_output_width == 9060


def test_row_refinement_configuration_requires_layout_and_complete_sweeps() -> None:
    rows = RowEventConfig()
    with pytest.raises(ValueError, match="refinement_layout"):
        ModelConfig(
            input_width=rows.input_width,
            decoder_mode="row_refinement",
            refinement_steps=30,
        )

    for steps, maximum, message in (
        (0, 30, "refinement_steps"),
        (31, 60, "multiple of 30"),
        (60, 30, "max_latent_steps"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            _row_refinement_config(
                refinement_steps=steps,
                max_latent_steps=maximum,
            )


def test_model_configuration_enforces_1024_edges_per_neuron_policy_cap() -> None:
    neuron_count = 1088
    with pytest.raises(ValueError, match="1024 edges per neuron policy cap"):
        ModelConfig(
            input_width=1,
            neuron_count=neuron_count,
            recurrent_edges=1024 * neuron_count + 1,
        )


def test_row_refinement_model_instantiates_direct_heads_and_typed_answer_state() -> (
    None
):
    import braintrace

    model = LatentWorkspaceModel(_row_refinement_config(batch_size=2))

    assert isinstance(model.answer_row_head, braintrace.nn.Linear)
    assert isinstance(model.answer_shape_head, braintrace.nn.Linear)
    assert isinstance(model.answer_row, brainstate.HiddenState)
    assert isinstance(model.answer_shape, brainstate.HiddenState)
    assert isinstance(model.query_grid, brainstate.ShortTermState)
    assert isinstance(model.query_shape, brainstate.ShortTermState)
    assert isinstance(model.answer_grid, brainstate.ShortTermState)
    assert isinstance(model.reasoning_index, brainstate.ShortTermState)
    assert model.answer_row.value.shape == (2, 300)
    assert model.answer_shape.value.shape == (2, 60)
    assert model.query_grid.value.shape == (2, 30, 30, 10)
    assert model.query_shape.value.shape == (2, 60)
    assert model.answer_grid.value.shape == (2, 30, 30, 10)
    assert model.reasoning_index.value.shape == (2,)


def test_run_context_captures_target_free_query_grid_and_shape() -> None:
    _, events = _row_refinement_episode()
    model = LatentWorkspaceModel(_row_refinement_config())

    run_context(model, events)

    expected_grid = np.zeros((1, 30, 30, 10), dtype=np.float32)
    expected_grid[0, 0, 0, 5] = 1.0
    expected_grid[0, 0, 1, 6] = 1.0
    expected_grid[0, 1, 0, 7] = 1.0
    expected_grid[0, 1, 1, 8] = 1.0
    expected_shape = np.zeros((1, 60), dtype=np.float32)
    expected_shape[0, 1] = 1.0
    expected_shape[0, 30 + 1] = 1.0
    np.testing.assert_array_equal(model.query_grid.value, expected_grid)
    np.testing.assert_array_equal(model.query_shape.value, expected_shape)
    np.testing.assert_array_equal(model.answer_grid.value, 0.0)
    np.testing.assert_array_equal(model.reasoning_index.value, 0)


def test_one_latent_tick_replaces_only_row_zero_and_advances_index() -> None:
    model = LatentWorkspaceModel(_row_refinement_config())
    model.answer_grid.value = jnp.full((1, 30, 30, 10), 7.0, dtype=jnp.float32)
    model.neu.V.value = jnp.full((1, model.neuron_count), 1.5) * u.mV
    zero_event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)

    model.cell_step(zero_event, jnp.ones((1,), dtype=jnp.bool_))

    updated = np.asarray(model.answer_grid.value)
    assert not np.array_equal(updated[:, 0], np.full((1, 30, 10), 7.0))
    np.testing.assert_array_equal(updated[:, 1:], 7.0)
    np.testing.assert_array_equal(model.answer_row.value, updated[:, 0].reshape(1, 300))
    np.testing.assert_array_equal(model.reasoning_index.value, 1)


def test_compiled_thirty_tick_refinement_sweep_wraps_reasoning_index() -> None:
    model = LatentWorkspaceModel(_row_refinement_config())
    zero_event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)

    def latent_step(_: jax.Array) -> jax.Array:
        model.cell_step(zero_event, advance)
        return model.reasoning_index.value

    indices = brainstate.transform.for_loop(latent_step, jnp.arange(30))

    assert indices.shape == (30, 1)
    np.testing.assert_array_equal(indices[-1], 0)
    np.testing.assert_array_equal(model.reasoning_index.value, 0)
    assert np.count_nonzero(np.asarray(model.answer_grid.value)) > 0


def test_refinement_reset_snapshot_and_restore_include_short_term_accumulators() -> (
    None
):
    model = LatentWorkspaceModel(_row_refinement_config())
    model.answer_row.value = jnp.arange(300, dtype=jnp.float32)[None]
    model.answer_shape.value = jnp.arange(60, dtype=jnp.float32)[None]
    model.query_grid.value = model.query_grid.value.at[0, 2, 3, 4].set(1.0)
    model.query_shape.value = model.query_shape.value.at[0, 7].set(1.0)
    model.answer_grid.value = model.answer_grid.value.at[0, 5, 6, 7].set(2.0)
    model.reasoning_index.value = jnp.asarray([11], dtype=jnp.int32)
    snapshot = model.snapshot_state()
    snapshot_paths = {path for path, _ in snapshot.entries}
    expected_paths = {
        ("answer_row",),
        ("answer_shape",),
        ("query_grid",),
        ("query_shape",),
        ("answer_grid",),
        ("reasoning_index",),
    }

    assert expected_paths <= snapshot_paths
    model.reset_state()
    for state in (
        model.answer_row,
        model.answer_shape,
        model.query_grid,
        model.query_shape,
        model.answer_grid,
        model.reasoning_index,
    ):
        np.testing.assert_array_equal(state.value, 0)
    model.restore_state(snapshot)
    _assert_state_snapshots_equal(snapshot, model.snapshot_state())


def test_advance_false_preserves_every_refinement_and_physical_state_exactly() -> None:
    model = LatentWorkspaceModel(_row_refinement_config())
    _, events = _row_refinement_episode()
    run_context(model, events)
    before = model.snapshot_state()
    event = events[-1][None]

    model.cell_step(event, jnp.zeros((1,), dtype=jnp.bool_))

    _assert_state_snapshots_equal(before, model.snapshot_state())


def test_row_refinement_uses_compact_training_and_explicit_checkpoint_outputs() -> None:
    model = LatentWorkspaceModel(_row_refinement_config())
    zero_event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)

    training_output = model.update(
        zero_event,
        jnp.ones((1,), dtype=jnp.bool_),
    )
    checkpoint_output = model.compact_readout()

    assert training_output.shape == (1, 360)
    assert checkpoint_output.shape == (1, 9060)
    np.testing.assert_array_equal(training_output[:, :60], model.answer_shape.value)
    np.testing.assert_array_equal(training_output[:, 60:], model.answer_row.value)
    np.testing.assert_array_equal(checkpoint_output[:, :60], model.answer_shape.value)
    np.testing.assert_array_equal(
        checkpoint_output[:, 60:], model.answer_grid.value.reshape(1, 9000)
    )


def test_selected_row_refinement_checkpoints_equal_full_trajectory_gather() -> None:
    config = _row_refinement_config()
    rows, encoded_events = _row_refinement_episode()
    valid = encoded_events[:, rows.valid_slice.start] > 0.5
    context = encoded_events[valid]
    latent = jnp.zeros((30, config.input_width), dtype=jnp.float32)
    events = jnp.concatenate((context, latent), axis=0)[:, None, :]
    advances = jnp.ones(events.shape[:2], dtype=jnp.bool_)
    selected_indices = jnp.asarray(
        [[context.shape[0] - 1], [context.shape[0] + 14], [context.shape[0] + 29]],
        dtype=jnp.int32,
    )

    full = run_packed_stream(
        LatentWorkspaceModel(config), events, advance_gates=advances
    )
    selected = run_selected_packed_stream(
        LatentWorkspaceModel(config),
        events,
        selected_indices,
        advance_gates=advances,
    )

    np.testing.assert_array_equal(
        selected.compact_logits,
        np.take_along_axis(
            np.asarray(full.compact_logits),
            np.asarray(selected_indices)[:, :, None],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        selected.expanded.colors,
        np.take_along_axis(
            np.asarray(full.expanded.colors),
            np.asarray(selected_indices)[:, :, None, None, None],
            axis=0,
        ),
    )


def test_row_refinement_heads_are_direct_etraces_without_accumulator_groups() -> None:
    model = LatentWorkspaceModel(_row_refinement_config(recurrent_edges=32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        learner = compile_pp_prop(model)
    expected_paths = {
        ("answer_row_head", "weight"),
        ("answer_shape_head", "weight"),
    }
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
    grouped_paths = {
        tuple(path)
        for group in learner.graph.hidden_groups
        for path in group.hidden_paths
    }
    assert {("answer_row",), ("answer_shape",)} <= grouped_paths
    for accumulator in ("query_grid", "query_shape", "answer_grid", "reasoning_index"):
        assert not any(path and path[0] == accumulator for path in grouped_paths)


def test_memory_free_row_refinement_compiles_from_continuous_voltage() -> None:
    rows = RowEventConfig()
    config = ModelConfig(
        input_width=rows.input_width,
        batch_size=1,
        neuron_count=64,
        recurrent_edges=32,
        max_latent_steps=30,
        readout_width=8,
        color_rank=2,
        seed=2173,
        event_valid_index=rows.valid_slice.start,
        decoder_mode="row_refinement",
        refinement_steps=30,
        refinement_layout=_row_refinement_layout(rows),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        learner = compile_pp_prop(LatentWorkspaceModel(config))

    etrace_paths = {path for path, _ in learner.report.etrace_weights}
    assert {
        ("answer_row_head", "weight"),
        ("answer_shape_head", "weight"),
    } <= etrace_paths

    low_voltage = LatentWorkspaceModel(config)
    high_voltage = LatentWorkspaceModel(config)
    low_voltage.neu.V.value = jnp.full((1, 64), 0.5, dtype=jnp.float32) * u.mV
    high_voltage.neu.V.value = jnp.full((1, 64), 1.5, dtype=jnp.float32) * u.mV
    event = jnp.zeros((1, config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)

    low_voltage.cell_step(event, advance)
    high_voltage.cell_step(event, advance)

    np.testing.assert_array_equal(low_voltage.spikes, high_voltage.spikes)
    assert not np.allclose(low_voltage.answer_row.value, high_voltage.answer_row.value)


def test_feedback_uses_prior_answer_row_without_writing_context_memory() -> None:
    model = LatentWorkspaceModel(_row_refinement_config(input_gain=12.0))
    _, events = _row_refinement_episode()
    run_context(model, events)
    before_tick = model.snapshot_state()
    frozen_memory = np.asarray(model.context_memory.value).copy()
    zero_event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)

    model.cell_step(zero_event, advance)
    baseline_current = np.asarray(model.feedforward_current).copy()
    np.testing.assert_array_equal(model.context_memory.value, frozen_memory)

    model.restore_state(before_tick)
    model.answer_row.value = jnp.linspace(-20.0, 20.0, 300)[None]
    model.cell_step(zero_event, advance)

    np.testing.assert_array_equal(model.context_memory.value, frozen_memory)
    assert not np.allclose(model.feedforward_current, baseline_current)


def _synthetic_grid(side: int, offset: int) -> ArcGrid:
    return ArcGrid(
        tuple(
            tuple((r * side + c + offset) % 9 + 1 for c in range(side))
            for r in range(side)
        )
    )


def _row_refinement_full_size_episode() -> jax.Array:
    """Return a production-shaped 10x10 query episode.

    The 2x2 fixture used elsewhere in this file drives the membrane far less
    hard than a real ARC query and understates the defect (sweep cosine 0.88
    against 0.95-0.99 on real evaluation tasks). Row blindness is a property of
    the saturated integrator, so it must be reproduced at production grid size.
    """
    rows = RowEventConfig()
    task = ArcTask(
        train=tuple(
            ArcPair(_synthetic_grid(10, i), _synthetic_grid(10, i + 3))
            for i in range(2)
        ),
        test=(ArcPair(_synthetic_grid(10, 7), None),),
        task_id="row-refinement-full-size",
    )
    return jnp.asarray(encode_query_episode(task, 0, rows).events)


def _swept_answer_grid(model: LatentWorkspaceModel) -> np.ndarray:
    """Return the decoded answer grid after one full 30-row refinement sweep."""
    zero_event = jnp.zeros((1, model.config.input_width), dtype=jnp.float32)
    advance = jnp.ones((1,), dtype=jnp.bool_)

    def latent_step(_: jax.Array) -> jax.Array:
        model.cell_step(zero_event, advance)
        return model.reasoning_index.value

    brainstate.transform.for_loop(latent_step, jnp.arange(MAX_GRID_SIZE))
    return np.asarray(model.answer_grid.value)[0].argmax(axis=-1)


def test_refinement_sweep_writes_a_distinct_pattern_for_each_row() -> None:
    """A refinement sweep must write 30 rows, not one row 30 times.

    The answer heads read the LIF membrane potential, a slow integrator whose
    state is ~99% identical between consecutive refinement rows, so the head
    emits a per-query constant replicated across the sweep. Training then sees
    30 mutually inconsistent row targets through one input and can only
    converge to the row-independent colour marginal, whose argmax is colour
    zero in every cell -- the all-black grid every evaluation run reports.

    Distinct decoded row patterns is the scale-robust form of this check. The
    mean pairwise cosine between row logit vectors is not: a shared per-query
    component inflates it in proportion to ``neuron_count``, so a threshold
    tuned at 64 neurons says nothing at 1024.
    """
    model = LatentWorkspaceModel(_row_refinement_config())
    run_context(model, _row_refinement_full_size_episode())

    decoded = _swept_answer_grid(model)

    assert decoded.shape == (MAX_GRID_SIZE, MAX_GRID_SIZE)
    distinct = {tuple(row.tolist()) for row in decoded}
    assert len(distinct) >= 25


def test_refinement_head_input_separates_the_row_being_written() -> None:
    """Two ticks differing only in row index must reach the head differently.

    The head is a bias-free linear map, so anything absent from its input is
    unreachable by any amount of training. The row index and the query colours
    of the row being written are both present in the refinement feedback event
    but currently enter only through ``ff_syn``.
    """
    build_head_input = latent_workspace_module._refinement_head_input
    layout = _row_refinement_layout()
    carrier = jnp.asarray(
        np.linspace(-1.0, 1.0, 64, dtype=np.float32)[None, :]
    )
    event = jnp.zeros((1, layout.input_width), dtype=jnp.float32)
    other_row = event.at[:, layout.row_index_start + 7].set(1.0)
    other_colours = event.at[:, layout.input_color_start + 3].set(1.0)

    baseline = np.asarray(build_head_input(carrier, event, layout))
    row_changed = np.asarray(build_head_input(carrier, other_row, layout))
    colour_changed = np.asarray(build_head_input(carrier, other_colours, layout))

    assert baseline.shape == (1, latent_workspace_module.refinement_head_width(64))
    assert not np.array_equal(baseline, row_changed)
    assert not np.array_equal(baseline, colour_changed)


def test_refinement_head_carrier_is_scaled_to_unit_root_mean_square() -> None:
    """The head carrier must be O(1) per coordinate, not O(1/sqrt(n)).

    ``_unit_l2_cap`` normalises to unit *total* L2 across all neurons, so each
    coordinate carries 1/sqrt(n). Against bias-free heads initialised at
    1/sqrt(n) that yields initialisation logits of standard deviation 1/sqrt(n)
    -- 0.031 at 1024 neurons -- and a colour softmax within 0.0004 nats of
    uniform, so the optimizer spends its whole budget merely reaching O(1).
    """
    scale_carrier = latent_workspace_module._unit_rms_carrier
    neuron_count = 1024
    carrier = jnp.asarray(
        np.linspace(-40.0, 40.0, neuron_count, dtype=np.float32)[None, :]
    )

    scaled = np.asarray(scale_carrier(carrier), dtype=np.float64)

    assert scaled.shape == (1, neuron_count)
    np.testing.assert_allclose(np.sqrt((scaled**2).mean()), 1.0, rtol=1e-5)


def test_refinement_heads_stay_compiled_all_direct_with_row_conditioning() -> None:
    """Widening the head input must not drop it from the compiled model.

    A head whose input the compiler cannot trace would silently stop training
    while still looking correct at the model boundary.
    """
    model = LatentWorkspaceModel(_row_refinement_config(batch_size=2))
    learner = compile_pp_prop(model)

    def text(path: object) -> str:
        if isinstance(path, (tuple, list)):
            return ".".join(str(part) for part in path)
        return str(path)

    compiled = {text(key) for key in learner.param_states}
    assert {"answer_row_head.weight", "answer_shape_head.weight"} <= compiled

    classifications: dict[str, set[str]] = {}
    for record in learner.report.diagnostics:
        context = getattr(record, "context", None)
        if not isinstance(context, dict):
            continue
        by_hidden_state = context.get("path_classification")
        if not isinstance(by_hidden_state, dict):
            continue
        classifications[text(getattr(record, "weight_path", ""))] = {
            str(getattr(value, "value", value)) for value in by_hidden_state.values()
        }

    assert classifications["answer_row_head.weight"] == {"all_direct"}
    assert classifications["answer_shape_head.weight"] == {"all_direct"}


def test_row_refinement_owns_only_active_decoder_states() -> None:
    """Inactive legacy heads must not appear as compiler state mismatches."""
    model = LatentWorkspaceModel(_row_refinement_config(batch_size=2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner = compile_pp_prop(model)

    mismatch_messages = [
        str(item.message)
        for item in caught
        if "not found in the compiled model" in str(item.message)
    ]
    assert mismatch_messages == []

    parameters = set(parameter_snapshot(model))
    legacy_prefixes = {
        "readout_projection.",
        "height_head.",
        "width_head.",
        "color_factor_head.",
    }
    assert not any(
        path.startswith(prefix)
        for path in parameters
        for prefix in legacy_prefixes
    )

    diagnostics = [
        record
        for record in learner.report.diagnostics
        if str(getattr(getattr(record, "kind", None), "value", ""))
        == "state_mismatch"
    ]
    assert diagnostics == []


def test_legacy_decoder_retains_its_active_states() -> None:
    """Conditional ownership must preserve the legacy decoder parameter tree."""
    model = LatentWorkspaceModel(_config(batch_size=2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner = compile_pp_prop(model)

    assert not any(
        "not found in the compiled model" in str(item.message) for item in caught
    )
    parameters = set(parameter_snapshot(model))
    assert {
        "readout_projection.weight",
        "height_head.weight",
        "width_head.weight",
        "color_factor_head.weight",
    } <= parameters
    assert not any(
        str(getattr(getattr(record, "kind", None), "value", ""))
        == "state_mismatch"
        for record in learner.report.diagnostics
    )


def test_copy_residual_raises_the_query_colour_logit_by_the_gain() -> None:
    """The residual must add exactly ``gain`` at each occupied colour logit.

    The colour block is indexed ``column * 10 + colour``
    (``latent_workspace_task.py``) and the row logits reshape ``(30, 10)`` on
    the same index, so the identity map is a direct vector addition. Every
    coordinate the query does not occupy — including the whole block beyond the
    input's height, which ``input_row_valid`` zeroes — must be bit-unchanged,
    so the learned head keeps sole custody of out-of-range cells.
    """
    apply_residual = latent_workspace_module._copy_residual_logits
    layout = _row_refinement_layout()
    row_logits = jnp.asarray(
        np.linspace(-1.0, 1.0, MAX_GRID_SIZE * COLOR_COUNT, dtype=np.float32)[None, :]
    )
    event = jnp.zeros((1, layout.input_width), dtype=jnp.float32)
    event = event.at[:, layout.input_color_start + 0 * COLOR_COUNT + 3].set(1.0)
    event = event.at[:, layout.input_color_start + 7 * COLOR_COUNT + 9].set(1.0)

    boosted = np.asarray(apply_residual(row_logits, event, layout, 2.0))

    expected = np.asarray(row_logits).copy()
    expected[0, 0 * COLOR_COUNT + 3] += 2.0
    expected[0, 7 * COLOR_COUNT + 9] += 2.0
    np.testing.assert_array_equal(boosted, expected)


def test_copy_residual_gain_zero_reproduces_the_bare_head() -> None:
    """``gain = 0`` must be bit-exact backward compatibility."""
    apply_residual = latent_workspace_module._copy_residual_logits
    layout = _row_refinement_layout()
    row_logits = jnp.asarray(
        np.linspace(-1.0, 1.0, MAX_GRID_SIZE * COLOR_COUNT, dtype=np.float32)[None, :]
    )
    event = jnp.zeros((1, layout.input_width), dtype=jnp.float32)
    event = event.at[:, layout.input_color_start + 5].set(1.0)

    unchanged = np.asarray(apply_residual(row_logits, event, layout, 0.0))

    np.testing.assert_array_equal(unchanged, np.asarray(row_logits))


def test_model_config_rejects_a_negative_copy_residual_gain() -> None:
    """The gain is a logit magnitude; a negative copy prior is a config error."""
    with pytest.raises(ValueError, match="copy_residual_gain"):
        _row_refinement_config(copy_residual_gain=-1.0)


def test_model_config_rejects_a_negative_row_head_carrier_scale() -> None:
    """The carrier scale is a magnitude; a sign flip is a config error."""
    with pytest.raises(ValueError, match="row_head_carrier_scale"):
        _row_refinement_config(row_head_carrier_scale=-0.5)


def test_row_head_carrier_scale_zero_makes_the_row_answer_carrier_free() -> None:
    """At scale zero the decoded rows must not depend on the membrane carrier.

    Two models with the same seed but different recurrent gains develop
    different membrane trajectories on the same episode. With the row head's
    carrier block scaled to zero the row logits read only the event blocks,
    which are identical across the two runs, so the scattered answer grids
    must agree bit for bit while the shape head (which keeps its carrier)
    remains free to differ.
    """
    grids = []
    for recurrent_gain in (0.8, 1.6):
        model = LatentWorkspaceModel(
            _row_refinement_config(
                row_head_carrier_scale=0.0,
                copy_residual_gain=2.0,
                recurrent_gain=recurrent_gain,
            )
        )
        run_context(model, _row_refinement_full_size_episode())
        grids.append(_swept_answer_grid(model))
    np.testing.assert_array_equal(grids[0], grids[1])


def test_model_config_rejects_a_negative_shape_head_carrier_scale() -> None:
    """The shape carrier scale is a magnitude; a sign flip is a config error."""
    with pytest.raises(ValueError, match="shape_head_carrier_scale"):
        _row_refinement_config(shape_head_carrier_scale=-0.5)


def test_shape_head_carrier_scale_zero_makes_the_shape_answer_carrier_free() -> None:
    """At scale zero the decoded shape must not depend on the membrane carrier.

    Two models with the same seed but different recurrent gains develop
    different membrane trajectories on the same episode. With the shape head's
    carrier block scaled to zero the shape logits read only the event blocks,
    which are identical across the two runs, so the final shape logits must
    agree bit for bit while the row head (which keeps its carrier) remains
    free to differ.
    """
    shapes = []
    for recurrent_gain in (0.8, 1.6):
        model = LatentWorkspaceModel(
            _row_refinement_config(
                shape_head_carrier_scale=0.0,
                recurrent_gain=recurrent_gain,
            )
        )
        run_context(model, _row_refinement_full_size_episode())
        shapes.append(np.asarray(model.answer_shape.value))
    np.testing.assert_array_equal(shapes[0], shapes[1])


def test_model_config_rejects_the_gate_with_a_carrier_scale() -> None:
    """The gate replaces the scale mechanism; combining them is an error."""
    with pytest.raises(ValueError, match="row_head_carrier_gate"):
        _row_refinement_config(
            row_head_carrier_gate=True, row_head_carrier_scale=0.5
        )


def _gated_swept_grid(recurrent_gain: float, gate_shift: float) -> np.ndarray:
    """Run one gated episode, optionally forcing the gate weight open."""
    model = LatentWorkspaceModel(
        _row_refinement_config(
            row_head_carrier_gate=True,
            copy_residual_gain=2.0,
            recurrent_gain=recurrent_gain,
        )
    )
    if gate_shift:
        for path, state in model.states(brainstate.ParamState).items():
            if "row_carrier_gate_head" in ".".join(map(str, path)):
                state.value = jax.tree.map(lambda a: a + gate_shift, state.value)
    run_context(model, _row_refinement_full_size_episode())
    return _swept_answer_grid(model)


def test_row_head_carrier_gate_starts_exactly_carrier_free() -> None:
    """At the zero-initialised gate the row answer must ignore the carrier.

    Two models differing only in recurrent gain develop different membrane
    trajectories; with ``tanh(0) == 0`` the carrier head contributes nothing,
    so the swept answer grids must agree bit for bit.
    """
    np.testing.assert_array_equal(
        _gated_swept_grid(0.8, 0.0), _gated_swept_grid(1.6, 0.0)
    )


def test_gated_heads_are_literal_slices_of_same_seed_linear_initialization() -> None:
    """The gate ablation must not change the event/carrier initialization scale."""
    linear = LatentWorkspaceModel(_row_refinement_config())
    gated = LatentWorkspaceModel(
        _row_refinement_config(row_head_carrier_gate=True)
    )

    combined = np.asarray(linear.answer_row_head.weight.value["weight"])
    carrier = np.asarray(gated.answer_row_carrier_head.weight.value["weight"])
    event = np.asarray(gated.answer_row_event_head.weight.value["weight"])
    split = linear.config.neuron_count

    np.testing.assert_array_equal(carrier, combined[:split])
    np.testing.assert_array_equal(event, combined[split:])


def test_zero_gate_matches_carrier_free_linear_head_exactly() -> None:
    linear = LatentWorkspaceModel(_row_refinement_config())
    gated = LatentWorkspaceModel(
        _row_refinement_config(refinement_mixer="carrier_gate")
    )
    layout = _row_refinement_layout()
    carrier = jnp.zeros((1, linear.config.neuron_count), dtype=jnp.float32)
    event = jnp.linspace(0.0, 1.0, layout.input_width, dtype=jnp.float32)[None]
    head_input = latent_workspace_module._refinement_head_input(
        carrier, event, layout
    )

    expected = linear.answer_row_head(head_input)
    actual = gated._row_head_logits(carrier, event, head_input)
    np.testing.assert_array_equal(actual, expected)


def test_refinement_routing_manifest_tracks_each_architecture() -> None:
    manifest = latent_workspace_module.refinement_parameter_paths

    assert set(manifest(_row_refinement_config())) == {
        "answer_row_head.weight",
        "answer_shape_head.weight",
    }
    assert set(
        manifest(_row_refinement_config(refinement_mixer="carrier_gate"))
    ) == {
        "answer_row_event_head.weight",
        "answer_row_carrier_head.weight",
        "row_carrier_gate_head.weight",
        "answer_shape_head.weight",
    }
    assert set(
        manifest(_row_refinement_config(refinement_mixer="attention_residual"))
    ) == {
        "answer_row_proposal_head.weight",
        "answer_shape_proposal_head.weight",
        "row_attention_residual.query",
        "shape_attention_residual.query",
    }


def test_attention_residual_configuration_rejects_conflicting_legacy_ablations() -> None:
    for conflict in (
        {"row_head_carrier_gate": True},
        {"copy_residual_gain": 1.0},
        {"row_head_carrier_scale": 0.0},
        {"shape_head_carrier_scale": 0.0},
    ):
        with pytest.raises(ValueError, match="attention_residual"):
            _row_refinement_config(
                refinement_mixer="attention_residual", **conflict
            )


def test_attention_residual_refinement_uses_uniform_sources_and_tracks_history() -> None:
    config = _row_refinement_config(
        max_latent_steps=60,
        refinement_steps=60,
        refinement_mixer="attention_residual",
    )
    model = LatentWorkspaceModel(config)
    model.query_grid.value = model.query_grid.value.at[0, 0, 0, 3].set(1.0)
    model.query_shape.value = model.query_shape.value.at[0, 4].set(1.0)
    zero_event = jnp.zeros((1, config.input_width), dtype=jnp.float32)

    model.cell_step(zero_event, jnp.ones((1,), dtype=jnp.bool_))

    balance = latent_workspace_module._rms_balanced_identity_source
    row_identity = model.query_grid.value[:, 0].reshape(1, -1)
    expected_row = (
        balance(row_identity, model.row_proposal.value) + model.row_proposal.value
    ) / 2.0
    expected_shape = (
        balance(model.query_shape.value, model.shape_proposal.value)
        + model.shape_proposal.value
    ) / 2.0
    np.testing.assert_allclose(model.answer_row.value, expected_row, atol=2e-6)
    np.testing.assert_allclose(model.answer_shape.value, expected_shape, atol=2e-6)
    np.testing.assert_array_equal(
        model.row_proposal_history.value[0, 0, 0], model.row_proposal.value[0]
    )
    np.testing.assert_array_equal(model.reasoning_sweep.value, 0)

    def latent_step(_: jax.Array) -> jax.Array:
        model.cell_step(zero_event, jnp.ones((1,), dtype=jnp.bool_))
        return model.reasoning_sweep.value

    brainstate.transform.for_loop(latent_step, jnp.arange(29))
    np.testing.assert_array_equal(model.reasoning_index.value, 0)
    np.testing.assert_array_equal(model.reasoning_sweep.value, 1)
    assert np.count_nonzero(np.asarray(model.row_proposal_history.value[:, 0])) > 0
    np.testing.assert_array_equal(
        model.shape_proposal_history.value[:, 0], model.shape_proposal.value
    )


def test_attention_residual_refinement_snapshot_reset_and_restore() -> None:
    model = LatentWorkspaceModel(
        _row_refinement_config(
            max_latent_steps=60,
            refinement_steps=60,
            refinement_mixer="attention_residual",
        )
    )
    model.row_proposal.value = jnp.ones_like(model.row_proposal.value)
    model.shape_proposal.value = jnp.ones_like(model.shape_proposal.value) * 2
    model.row_proposal_history.value = model.row_proposal_history.value.at[
        0, 0, 3, 4
    ].set(5.0)
    model.shape_proposal_history.value = model.shape_proposal_history.value.at[
        0, 0, 7
    ].set(6.0)
    model.reasoning_sweep.value = jnp.asarray([1], dtype=jnp.int32)
    snapshot = model.snapshot_state()
    paths = {path for path, _ in snapshot.entries}

    assert {
        ("row_proposal",),
        ("shape_proposal",),
        ("row_proposal_history",),
        ("shape_proposal_history",),
        ("reasoning_sweep",),
    } <= paths
    model.reset_state()
    np.testing.assert_array_equal(model.row_proposal.value, 0.0)
    np.testing.assert_array_equal(model.shape_proposal.value, 0.0)
    np.testing.assert_array_equal(model.row_proposal_history.value, 0.0)
    np.testing.assert_array_equal(model.shape_proposal_history.value, 0.0)
    np.testing.assert_array_equal(model.reasoning_sweep.value, 0)
    model.restore_state(snapshot)
    _assert_state_snapshots_equal(snapshot, model.snapshot_state())


def test_attention_residual_proposal_and_query_paths_compile_all_direct() -> None:
    model = LatentWorkspaceModel(
        _row_refinement_config(
            batch_size=2,
            max_latent_steps=60,
            refinement_steps=60,
            refinement_mixer="attention_residual",
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner = compile_pp_prop(model)

    def text(path: object) -> str:
        return ".".join(map(str, path)) if isinstance(path, (tuple, list)) else str(path)

    expected = set(latent_workspace_module.refinement_parameter_paths(model.config))
    assert expected <= {text(path) for path in learner.param_states}
    classifications: dict[str, set[str]] = {}
    for record in learner.report.diagnostics:
        context = getattr(record, "context", None)
        if not isinstance(context, dict):
            continue
        values = context.get("path_classification")
        if isinstance(values, dict):
            classifications[text(getattr(record, "weight_path", ""))] = {
                str(getattr(value, "value", value)) for value in values.values()
            }
    assert all(classifications[path] == {"all_direct"} for path in expected)


def test_row_head_carrier_gate_opens_a_carrier_path() -> None:
    """Forcing the gate weight open must make the row answer carrier-bound.

    With the gate weight shifted to 3.0 (``tanh`` ≈ 0.995) the carrier head's
    random projection reaches the logits, so the two recurrent gains must no
    longer produce identical grids — the gate is a real dial, not a dead end.
    """
    assert not np.array_equal(
        _gated_swept_grid(0.8, 3.0), _gated_swept_grid(1.6, 3.0)
    )


def test_gated_row_heads_stay_compiled_all_direct() -> None:
    """Event head, carrier head, and gate must all remain trainable.

    §9.1: a parameter trains only as a trainable invar of an ETP primitive.
    If the gate multiply confused the compiler, any of the three would drop
    from ``param_states`` and silently freeze.
    """
    model = LatentWorkspaceModel(
        _row_refinement_config(batch_size=2, row_head_carrier_gate=True)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner = compile_pp_prop(model)

    def text(path: object) -> str:
        if isinstance(path, (tuple, list)):
            return ".".join(str(part) for part in path)
        return str(path)

    compiled = {text(key) for key in learner.param_states}
    assert {
        "answer_row_event_head.weight",
        "answer_row_carrier_head.weight",
        "row_carrier_gate_head.weight",
        "answer_shape_head.weight",
    } <= compiled

    classifications: dict[str, set[str]] = {}
    for record in learner.report.diagnostics:
        context = getattr(record, "context", None)
        if not isinstance(context, dict):
            continue
        by_hidden_state = context.get("path_classification")
        if not isinstance(by_hidden_state, dict):
            continue
        classifications[text(getattr(record, "weight_path", ""))] = {
            str(getattr(value, "value", value)) for value in by_hidden_state.values()
        }
    assert classifications["answer_row_event_head.weight"] == {"all_direct"}
    assert classifications["answer_row_carrier_head.weight"] == {"all_direct"}
    assert classifications["row_carrier_gate_head.weight"] == {"all_direct"}


def test_refinement_heads_stay_compiled_all_direct_with_copy_residual() -> None:
    """The residual must not disturb ETP tracking of either answer head.

    The residual is a constant-scaled slice of the event added after the
    tracked ``answer_row_head`` op. If the addition confused the compiler the
    head would silently drop from ``param_states`` and train with a zero
    gradient while still looking correct.
    """
    model = LatentWorkspaceModel(
        _row_refinement_config(batch_size=2, copy_residual_gain=2.0)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner = compile_pp_prop(model)

    def text(path: object) -> str:
        if isinstance(path, (tuple, list)):
            return ".".join(str(part) for part in path)
        return str(path)

    compiled = {text(key) for key in learner.param_states}
    assert {"answer_row_head.weight", "answer_shape_head.weight"} <= compiled

    classifications: dict[str, set[str]] = {}
    for record in learner.report.diagnostics:
        context = getattr(record, "context", None)
        if not isinstance(context, dict):
            continue
        by_hidden_state = context.get("path_classification")
        if not isinstance(by_hidden_state, dict):
            continue
        classifications[text(getattr(record, "weight_path", ""))] = {
            str(getattr(value, "value", value)) for value in by_hidden_state.values()
        }

    assert classifications["answer_row_head.weight"] == {"all_direct"}
    assert classifications["answer_shape_head.weight"] == {"all_direct"}


def test_refinement_head_input_separates_the_query_grid_shape() -> None:
    """Two ticks differing only in the query grid shape must reach the head.

    The shape loss fires on one tick of a 30-row sweep, the completed sweep at
    row 29 (``latent_workspace_refinement.py:694``). At that tick the query
    colour block the head receives holds the query's *row 29*, which is padding
    for every query shorter than 30 rows. The query's input height and width
    one-hots are written into the same event but are not read by the head, so
    ``output_shape == input_shape`` -- true for 66.1% of ARC-AGI-1 evaluation
    queries and a 30x30 identity block once the one-hots are present -- is not
    representable by a bias-free linear head at all.
    """
    build_head_input = latent_workspace_module._refinement_head_input
    layout = _row_refinement_layout()
    carrier = jnp.asarray(np.linspace(-1.0, 1.0, 64, dtype=np.float32)[None, :])
    event = jnp.zeros((1, layout.input_width), dtype=jnp.float32)
    other_height = event.at[:, layout.input_height_start + 9].set(1.0)
    other_width = event.at[:, layout.input_width_start + 4].set(1.0)

    baseline = np.asarray(build_head_input(carrier, event, layout))
    height_changed = np.asarray(build_head_input(carrier, other_height, layout))
    width_changed = np.asarray(build_head_input(carrier, other_width, layout))

    assert not np.array_equal(baseline, height_changed)
    assert not np.array_equal(baseline, width_changed)
    assert not np.array_equal(height_changed, width_changed)


def test_refinement_head_width_counts_the_query_shape_blocks() -> None:
    """The head input carries the row code and both query dimension one-hots."""
    head_width = latent_workspace_module.refinement_head_width
    blocks = MAX_GRID_SIZE + MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE

    assert head_width(1024) == 1024 + blocks
    assert head_width(64) == 64 + blocks


def test_refinement_heads_stay_compiled_all_direct_with_shape_conditioning() -> None:
    """Widening the head input again must not drop it from the compiled model.

    A head whose input the compiler cannot trace would silently stop training
    while still looking correct at the model boundary.
    """
    model = LatentWorkspaceModel(_row_refinement_config(batch_size=2))
    learner = compile_pp_prop(model)

    def text(path: object) -> str:
        if isinstance(path, (tuple, list)):
            return ".".join(str(part) for part in path)
        return str(path)

    compiled = {text(key) for key in learner.param_states}
    assert {"answer_row_head.weight", "answer_shape_head.weight"} <= compiled

    classifications: dict[str, set[str]] = {}
    for record in learner.report.diagnostics:
        context = getattr(record, "context", None)
        if not isinstance(context, dict):
            continue
        by_hidden_state = context.get("path_classification")
        if not isinstance(by_hidden_state, dict):
            continue
        classifications[text(getattr(record, "weight_path", ""))] = {
            str(getattr(value, "value", value)) for value in by_hidden_state.values()
        }

    assert classifications["answer_row_head.weight"] == {"all_direct"}
    assert classifications["answer_shape_head.weight"] == {"all_direct"}


def test_memory_coding_defaults_to_frozen_and_validates() -> None:
    assert _memory_config().memory_coding == "frozen"
    assert _config().memory_coding == "frozen"
    with pytest.raises(TypeError, match="memory_coding"):
        _memory_config(memory_coding=7)
    with pytest.raises(ValueError, match="memory_coding"):
        _memory_config(memory_coding="learned")
    with pytest.raises(ValueError, match="memory_coding"):
        _memory_config(memory_coding="learned_keys_values")
    with pytest.raises(ValueError, match="memory_coding"):
        _config(memory_coding="learned_keys")


def test_learned_update_writes_nonzero_distinct_one_sided_rows() -> None:
    config = _memory_config(memory_coding="learned_update")
    model = LatentWorkspaceModel(config)
    input_only = _phase_events(
        config, jnp.asarray([[1.0, -0.5]]), phase="demonstration"
    )[0]
    output_only = jnp.zeros_like(input_only)
    output_only = output_only.at[:, config.event_valid_index].set(1.0)
    assert config.demonstration_phase_index is not None
    assert config.output_side_valid_index is not None
    output_only = output_only.at[:, config.demonstration_phase_index].set(1.0)
    output_only = output_only.at[:, config.output_side_valid_index].set(1.0)
    output_only = output_only.at[:, jnp.asarray(config.memory_value_indices)].set(
        jnp.asarray([[0.25, 0.75]])
    )

    input_write = np.asarray(model.encode_memory_update(input_only))
    output_write = np.asarray(model.encode_memory_update(output_only))
    assert np.linalg.norm(input_write) > 0.0
    assert np.linalg.norm(output_write) > 0.0
    assert not np.array_equal(input_write, output_write)

    model.cell_step(input_only, jnp.ones((1,), dtype=jnp.bool_))
    input_memory = np.asarray(model.context_memory.value).copy()
    assert np.linalg.norm(input_memory) > 0.0
    model.reset_state()
    model.cell_step(output_only, jnp.ones((1,), dtype=jnp.bool_))
    output_memory = np.asarray(model.context_memory.value)
    assert np.linalg.norm(output_memory) > 0.0
    assert not np.array_equal(input_memory, output_memory)


def test_learned_update_reports_feature_order_projection_hash_and_routing() -> None:
    config = _memory_config(memory_coding="learned_update")
    report = LatentWorkspaceModel(config).associative_memory_report()

    assert report.update_feature_width == len(config.memory_update_indices)
    assert report.update_feature_order == config.memory_update_feature_order
    assert report.update_projection_seed == config.seed + 106
    assert len(report.update_projection_sha256 or "") == 64
    assert report.update_routing == (
        "memory_update_projection->context_memory;shared_retrieval_key"
    )


def test_learned_update_projection_is_trace_compiled() -> None:
    model = LatentWorkspaceModel(
        _row_refinement_config(
            decoder_mode="latent_row_decode", memory_coding="learned_update"
        )
    )

    learner = compile_etrace(model)

    paths = {path for path, _ in learner.report.etrace_weights}
    assert ("memory_update_projection", "weight") in paths
    assert ("memory_key_projection", "weight") in paths


@pytest.mark.parametrize("coding", ["delta_write", "situ_glu_update"])
def test_stage5_memory_codings_validate_and_require_memory(coding: str) -> None:
    assert _memory_config(memory_coding=coding).memory_coding == coding
    with pytest.raises(ValueError, match="positive context_memory_width"):
        _config(memory_coding=coding)


def test_delta_write_one_sided_rows_are_finite_nonzero_and_distinct() -> None:
    config = _memory_config(memory_coding="delta_write")
    input_model = LatentWorkspaceModel(config)
    output_model = LatentWorkspaceModel(config)
    input_only = _phase_events(
        config, jnp.asarray([[1.0, -0.5]]), phase="demonstration"
    )[0]
    output_only = jnp.zeros_like(input_only)
    output_only = output_only.at[:, config.event_valid_index].set(1.0)
    assert config.demonstration_phase_index is not None
    assert config.output_side_valid_index is not None
    output_only = output_only.at[:, config.demonstration_phase_index].set(1.0)
    output_only = output_only.at[:, config.output_side_valid_index].set(1.0)
    output_only = output_only.at[:, jnp.asarray(config.memory_value_indices)].set(
        jnp.asarray([[0.25, 0.75]])
    )

    input_model.cell_step(input_only, jnp.ones((1,), dtype=jnp.bool_))
    output_model.cell_step(output_only, jnp.ones((1,), dtype=jnp.bool_))
    input_memory = np.asarray(input_model.context_memory.value)
    output_memory = np.asarray(output_model.context_memory.value)

    assert np.all(np.isfinite(input_memory))
    assert np.all(np.isfinite(output_memory))
    assert np.linalg.norm(input_memory) > 0.0
    assert np.linalg.norm(output_memory) > 0.0
    assert not np.array_equal(input_memory, output_memory)
    report = input_model.associative_memory_report()
    assert report.update_projection_seed == config.seed + 107
    assert len(report.update_projection_sha256 or "") == 64


def test_delta_write_false_lane_and_latent_ticks_do_not_mutate_memory() -> None:
    config = _memory_config(memory_coding="delta_write")
    model = LatentWorkspaceModel(config)
    demonstration = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[0.25, 1.0]]),
        phase="demonstration",
    )[0]
    before = np.asarray(model.context_memory.value).copy()

    model.cell_step(demonstration, jnp.zeros((1,), dtype=jnp.bool_))
    np.testing.assert_array_equal(model.context_memory.value, before)
    model.cell_step(demonstration, jnp.ones((1,), dtype=jnp.bool_))
    written = np.asarray(model.context_memory.value).copy()
    assert not np.array_equal(written, before)
    model.cell_step(
        jnp.zeros((1, config.input_width), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(model.context_memory.value, written)


def test_situ_glu_update_owns_caps_and_resets_flat_memory() -> None:
    config = _memory_config(memory_coding="situ_glu_update")
    model = LatentWorkspaceModel(config)
    event = _phase_events(
        config, jnp.asarray([[1.0, -0.5]]), phase="demonstration"
    )[0]

    update = model.encode_memory_update(event)
    report = model.associative_memory_report()

    assert update.shape == (1, config.context_memory_width**2)
    assert np.all(np.isfinite(np.asarray(update)))
    assert model.memory_situ_glu.hidden_size == 4 * config.context_memory_width
    assert report.write_component_type == "braintrace.nn.SiTUGLU"
    assert report.value_map == "situ_glu_softcapped_update"
    assert report.update_projection_seed == config.seed + 110
    assert len(report.update_projection_sha256 or "") == 64
    model.memory_situ_glu.gate_weight.value = jnp.zeros_like(
        model.memory_situ_glu.gate_weight.value
    )
    model.memory_situ_glu.gate_bias.value = jnp.ones_like(
        model.memory_situ_glu.gate_bias.value
    )
    model.memory_situ_glu.up_weight.value = jnp.zeros_like(
        model.memory_situ_glu.up_weight.value
    )
    model.memory_situ_glu.up_bias.value = jnp.ones_like(
        model.memory_situ_glu.up_bias.value
    )
    model.memory_situ_glu.output_weight.value = jnp.full_like(
        model.memory_situ_glu.output_weight.value, 10.0
    )
    assert float(jnp.max(jnp.abs(model.encode_memory_update(event)))) > (
        config.memory_value_softcap_beta
    )
    model.context_memory.value = jnp.ones_like(model.context_memory.value)
    model.reset_state()
    assert model.context_memory.value.shape == update.shape
    np.testing.assert_array_equal(model.context_memory.value, 0.0)


@pytest.mark.parametrize(
    ("coding", "required_paths"),
    [
        (
            "delta_write",
            {
                ("delta_key_weight",),
                ("delta_key_bias",),
                ("delta_value_weight",),
                ("delta_beta_weight",),
                ("delta_beta_bias",),
                ("delta_retention_weight",),
                ("delta_retention_bias",),
                ("memory_key_projection", "weight"),
            },
        ),
        (
            "situ_glu_update",
            {
                ("memory_situ_glu", "gate_weight"),
                ("memory_situ_glu", "gate_bias"),
                ("memory_situ_glu", "up_weight"),
                ("memory_situ_glu", "up_bias"),
                ("memory_situ_glu", "output_weight"),
            },
        ),
    ],
)
def test_stage5_memory_parameters_are_trace_compiled(
    coding: str,
    required_paths: set[tuple[str, ...]],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        learner = compile_etrace(LatentWorkspaceModel(_memory_config(memory_coding=coding)))

    paths = {path for path, _ in learner.report.etrace_weights}
    assert required_paths <= paths


def test_learned_memory_coding_matches_frozen_codes_at_initialization() -> None:
    config_frozen = _memory_config(batch_size=3)
    config_learned = _memory_config(batch_size=3, memory_coding="learned_keys")
    frozen = LatentWorkspaceModel(config_frozen)
    learned = LatentWorkspaceModel(config_learned)
    events = jnp.zeros((3, config_frozen.input_width), dtype=jnp.float32)
    events = events.at[:, jnp.asarray(config_frozen.memory_key_indices)].set(
        jnp.asarray([[1.0, -0.5], [0.25, 2.0], [0.0, 0.0]])
    )
    events = events.at[:, jnp.asarray(config_frozen.memory_value_indices)].set(
        jnp.asarray([[0.5, 1.0], [-1.0, 0.75], [2.0, -2.0]])
    )
    assert config_frozen.input_side_valid_index is not None
    assert config_frozen.output_side_valid_index is not None
    events = events.at[:, config_frozen.input_side_valid_index].set(
        jnp.asarray([1.0, 0.0, 1.0])
    )
    events = events.at[:, config_frozen.output_side_valid_index].set(
        jnp.asarray([1.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(
        np.asarray(learned.encode_memory_key(events)),
        np.asarray(frozen.encode_memory_key(events)),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(learned.encode_memory_value(events)),
        np.asarray(frozen.encode_memory_value(events)),
    )
    invalid_key_rows = np.asarray(learned.encode_memory_key(events))[1]
    np.testing.assert_array_equal(invalid_key_rows, np.zeros_like(invalid_key_rows))


def test_learned_memory_coding_report_names_components_and_keeps_hashes() -> None:
    frozen = LatentWorkspaceModel(_memory_config()).associative_memory_report()
    learned = LatentWorkspaceModel(
        _memory_config(memory_coding="learned_keys")
    ).associative_memory_report()
    assert frozen.key_map == "fixed_rff_cosine"
    assert frozen.value_map == "fixed_tanh_projection"
    assert learned.key_map == "learned_rff_cosine_retrieval_path"
    assert learned.value_map == "fixed_tanh_projection"
    assert learned.key_basis_sha256 == frozen.key_basis_sha256
    assert learned.key_bias_sha256 == frozen.key_bias_sha256
    assert learned.value_basis_sha256 == frozen.value_basis_sha256


def test_learned_key_coding_trains_through_retrieval_path_only() -> None:
    import braintrace
    from braintrace._testing.oracle import chunked_online_param_gradients

    config = _memory_config(
        memory_decay=0.9, input_gain=8.0, memory_coding="learned_keys"
    )
    model = LatentWorkspaceModel(config)
    learner = compile_pp_prop(model)
    key_path = ("memory_key_projection", "weight")
    etrace_paths = {path for path, _ in learner.report.etrace_weights}
    assert key_path in etrace_paths
    frozen_learner = compile_pp_prop(LatentWorkspaceModel(_memory_config()))
    frozen_paths = {path for path, _ in frozen_learner.report.etrace_weights}
    assert key_path not in frozen_paths
    relations = [
        relation
        for relation in learner.graph.hidden_param_op_relations
        if key_path in relation.trainable_paths.values()
    ]
    assert len(relations) == 1
    assert set(relations[0].path_classification.values()) == {"all_direct"}
    relation_hidden_paths = {
        path for group in relations[0].hidden_groups for path in group.hidden_paths
    }
    assert ("context_memory",) not in relation_hidden_paths
    assert ("query_encoding",) in relation_hidden_paths

    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    latent = jnp.zeros((2, 1, config.input_width), dtype=jnp.float32)
    inputs = jnp.concatenate((demonstrations, query, latent), axis=0)

    class _AlwaysAdvanceCodingModel(LatentWorkspaceModel):
        def update(self, event: jax.Array) -> jax.Array:
            advance = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)
            return super().update(event, advance)

    gradients = chunked_online_param_gradients(
        lambda: _AlwaysAdvanceCodingModel(config),
        inputs,
        algo_factory=lambda candidate: braintrace.pp_prop(
            candidate,
            decay_or_rank=candidate.config.trace_decay,
            vjp_method="multi-step",
        ),
        chunk_size=1,
    )
    gradient_leaves = jax.tree.leaves(gradients[key_path])
    gradient_norm = sum(
        float(jnp.sum(jnp.abs(jnp.asarray(u.get_mantissa(leaf)))))
        for leaf in gradient_leaves
    )
    assert gradient_leaves
    assert math.isfinite(gradient_norm)
    assert gradient_norm > 0.0


def test_learned_write_coding_validates_and_requires_memory() -> None:
    assert _memory_config(memory_coding="learned_write").memory_coding == (
        "learned_write"
    )
    with pytest.raises(ValueError, match="learned_write"):
        _memory_config(memory_coding="learned_writes")
    with pytest.raises(ValueError, match="context_memory_width must be positive"):
        _memory_config(memory_coding="learned_write", context_memory_width=0)


def test_learned_write_reproduces_the_frozen_memory_at_initialization() -> None:
    """The fused write starts from the same bases, so step 0 must agree.

    This is what makes a later divergence attributable to *learning* rather
    than to a different write being computed.
    """
    frozen_config = _memory_config(memory_decay=0.9, input_gain=8.0)
    learned_config = _memory_config(
        memory_decay=0.9, input_gain=8.0, memory_coding="learned_write"
    )
    events = _phase_events(
        frozen_config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )

    def _memory_after(config: ModelConfig) -> jax.Array:
        model = LatentWorkspaceModel(config)
        for event in events:
            model.update(event)
        return jnp.asarray(model.context_memory.value)

    np.testing.assert_allclose(
        _memory_after(learned_config), _memory_after(frozen_config),
        rtol=1e-5, atol=1e-6,
    )


def test_learned_write_report_names_the_fused_component() -> None:
    report = LatentWorkspaceModel(
        _memory_config(memory_coding="learned_write")
    ).associative_memory_report()
    assert report.key_map == "learned_rff_cosine_write_and_retrieval"
    assert report.value_map == "learned_tanh_projection"
    assert report.write_component_type == "braintrace.outer_write"
    # Provenance still hashes the frozen bases the write was initialized from.
    frozen = LatentWorkspaceModel(_memory_config()).associative_memory_report()
    assert report.key_basis_sha256 == frozen.key_basis_sha256
    assert report.value_basis_sha256 == frozen.value_basis_sha256


def test_encode_memory_write_rejects_other_codings() -> None:
    model = LatentWorkspaceModel(_memory_config(memory_coding="learned_keys"))
    event = _phase_events(
        model.config, jnp.asarray([[1.0, 0.0]]), jnp.asarray([[0.5, 0.5]]),
        phase="demonstration",
    )[0]
    with pytest.raises(RuntimeError, match="learned_write"):
        model.encode_memory_write(event)


def test_learned_write_zeroes_rows_missing_a_side() -> None:
    """A one-sided row must contribute nothing, exactly as the frozen path's
    zeroed key or value annihilates the outer product."""
    config = _memory_config(memory_coding="learned_write")
    model = LatentWorkspaceModel(config)
    event = _phase_events(
        config, jnp.asarray([[1.0, 0.0]]), phase="demonstration"
    )[0]
    assert config.output_side_valid_index is not None
    one_sided = event.at[:, config.output_side_valid_index].set(0.0)
    assert jnp.all(model.encode_memory_write(one_sided) == 0.0)


def test_learned_write_trains_the_write_projections_through_the_memory() -> None:
    """The point of the whole primitive: gradient now reaches ``context_memory``.

    Under ``learned_keys`` the key projection's relation deliberately excludes
    the memory (the write was stop-gradient detached and unfused). Here the
    write projections must own a relation that *includes* it, and a
    finite-window pp-prop gradient must be finite and nonzero for all three.
    """
    import braintrace
    from braintrace._testing.oracle import chunked_online_param_gradients

    config = _memory_config(
        memory_decay=0.9, input_gain=8.0, memory_coding="learned_write"
    )
    model = LatentWorkspaceModel(config)
    learner = compile_pp_prop(model)
    write_paths = {
        ("write_key_weight",), ("write_key_bias",), ("write_value_weight",),
    }
    etrace_paths = {path for path, _ in learner.report.etrace_weights}
    assert write_paths <= etrace_paths
    # Two trainable ETP primitives now sit on the write path: the fused coding
    # and the elementwise write scale. Adding the first must not displace the
    # second, or the write scale silently stops learning.
    assert ("memory_write_scale",) in etrace_paths

    relations = [
        relation
        for relation in learner.graph.hidden_param_op_relations
        if write_paths & set(relation.trainable_paths.values())
    ]
    assert len(relations) == 1
    assert set(relations[0].path_classification.values()) == {"all_direct"}
    relation_hidden_paths = {
        path for group in relations[0].hidden_groups for path in group.hidden_paths
    }
    assert ("context_memory",) in relation_hidden_paths

    demonstrations = _phase_events(
        config,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.25, 1.0], [1.5, -0.5]]),
        phase="demonstration",
    )
    query = _phase_events(config, jnp.asarray([[1.0, 0.0]]), phase="query")
    latent = jnp.zeros((2, 1, config.input_width), dtype=jnp.float32)
    inputs = jnp.concatenate((demonstrations, query, latent), axis=0)

    class _AlwaysAdvanceWriteModel(LatentWorkspaceModel):
        def update(self, event: jax.Array) -> jax.Array:
            advance = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)
            return super().update(event, advance)

    gradients = chunked_online_param_gradients(
        lambda: _AlwaysAdvanceWriteModel(config),
        inputs,
        algo_factory=lambda candidate: braintrace.pp_prop(
            candidate,
            decay_or_rank=candidate.config.trace_decay,
            vjp_method="multi-step",
        ),
        chunk_size=1,
    )
    for path in sorted(write_paths):
        leaves = jax.tree.leaves(gradients[path])
        assert leaves, path
        norm = sum(
            float(jnp.sum(jnp.abs(jnp.asarray(u.get_mantissa(leaf)))))
            for leaf in leaves
        )
        assert math.isfinite(norm), path
        assert norm > 0.0, path


def test_memory_coding_divergence_is_empty_without_a_learned_write() -> None:
    for coding in ("frozen", "learned_keys"):
        model = LatentWorkspaceModel(_memory_config(memory_coding=coding))
        assert model.memory_coding_divergence() == {}


def test_memory_coding_divergence_starts_at_zero_and_tracks_drift() -> None:
    """Attribution for a pinned-at-zero pairing result.

    ``learned_write`` gives the write and the retrieval paths *separate* key
    encoders, initialized identically and then trained independently. If they
    drift apart the memory is written in one code and queried in another, and
    retrieval degrades for reasons that have nothing to do with binding. This
    diagnostic is what separates that failure from a genuine null, so it must
    read zero at initialization and grow when the write encoder moves.
    """
    model = LatentWorkspaceModel(_memory_config(memory_coding="learned_write"))
    at_init = model.memory_coding_divergence()
    assert set(at_init) == {
        "write_retrieval_key_cosine",
        "write_retrieval_key_relative_l2",
        "write_retrieval_key_bias_relative_l2",
        "write_key_row_norm_mean",
    }
    assert at_init["write_retrieval_key_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert at_init["write_retrieval_key_relative_l2"] == pytest.approx(0.0, abs=1e-6)
    assert at_init["write_retrieval_key_bias_relative_l2"] == pytest.approx(
        0.0, abs=1e-6)
    assert at_init["write_key_row_norm_mean"] > 0.0

    model.write_key_weight.value = model.write_key_weight.value * 2.0
    drifted = model.memory_coding_divergence()
    assert drifted["write_retrieval_key_relative_l2"] == pytest.approx(1.0, abs=1e-5)
    assert drifted["write_retrieval_key_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert drifted["write_key_row_norm_mean"] == pytest.approx(
        2.0 * at_init["write_key_row_norm_mean"], rel=1e-5)


def test_trace_engine_defaults_to_pp_prop_and_validates() -> None:
    assert _config().trace_engine == "pp_prop"
    with pytest.raises(TypeError, match="trace_engine"):
        _config(trace_engine=7)
    with pytest.raises(ValueError, match="trace_engine"):
        _config(trace_engine="rtrl")


def test_etrace_coordinate_for_d_rtrl_engine_is_per_param() -> None:
    """The exact-trace arm: per-parameter traces, diagonal scope, true
    hidden Jacobian, no decay knob (the trace follows the recurrence)."""
    model = LatentWorkspaceModel(_config(trace_engine="d_rtrl"))
    trace = model.etrace_config()

    assert trace.trace_factorization == "per_param"
    assert trace.recurrence_scope == "diagonal"


def test_small_d_rtrl_compile_and_terminal_gradient_are_finite() -> None:
    """Mirror of the pp-prop compile gate under the exact-trace engine."""
    training_model = LatentWorkspaceModel(
        _config(recurrent_edges=64, trace_engine="d_rtrl"))
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


def test_softcap_beta_one_reproduces_tanh_bitwise():
    values = jnp.asarray(
        [-50.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 50.0], dtype=jnp.float32
    )

    capped = latent_workspace_module.softcap(values, 1.0)

    assert np.array_equal(np.asarray(capped), np.asarray(jnp.tanh(values)))


def test_softcap_matches_scaled_tanh_and_stays_bounded():
    values = jnp.linspace(-100.0, 100.0, 201, dtype=jnp.float32)

    capped = np.asarray(latent_workspace_module.softcap(values, 4.0))

    assert np.all(np.abs(capped) <= 4.0)
    moderate = np.abs(np.asarray(values)) <= 8.0
    assert np.all(np.abs(capped[moderate]) < 4.0)
    np.testing.assert_allclose(
        capped, 4.0 * np.tanh(np.asarray(values) / 4.0), rtol=1e-6
    )
    small = jnp.asarray([1e-3, -1e-3], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(latent_workspace_module.softcap(small, 25.0)),
        np.asarray(small),
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "name", ["memory_value_softcap_beta", "reasoning_query_softcap_beta"]
)
def test_model_config_softcap_betas_default_legacy_and_validate(name):
    config = ModelConfig(input_width=8)
    assert getattr(config, name) == 1.0

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match=name):
            ModelConfig(input_width=8, **{name: bad})
