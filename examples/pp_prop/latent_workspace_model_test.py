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
    from examples.pp_prop.latent_workspace_model import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        NEURONS_PER_SLOT,
        ArcLogits,
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
    from latent_workspace_model import (
        COLOR_COUNT,
        MAX_GRID_SIZE,
        NEURONS_PER_SLOT,
        ArcLogits,
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
    ):
        expected = np.asarray(getattr(full, name))[raw_indices, batch]
        np.testing.assert_array_equal(getattr(selected, name), expected)
    np.testing.assert_array_equal(selected.selected_indices, indices)
    assert selected.expanded.colors.shape == (4, 2, 30, 30, 10)


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
