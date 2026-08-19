"""Compact target-free ARC tensors for task-local pp-prop adaptation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any
from typing import NamedTuple

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

try:
    from examples.pp_prop.latent_workspace_adaptation import (
        ParameterSnapshot,
        restore_parameters,
    )
    from examples.pp_prop.latent_workspace_refinement import (
        row_refinement_loss_per_example,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcTask,
        RowEventConfig,
        decode_row_events,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
    )
except ImportError:
    from latent_workspace_adaptation import ParameterSnapshot, restore_parameters
    from latent_workspace_refinement import row_refinement_loss_per_example
    from latent_workspace_task import (
        ArcGrid,
        ArcTask,
        RowEventConfig,
        decode_row_events,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
    )


ARC_ADAPTATION_CHECKPOINTS = (0, 30, 60)


class ArcAdaptationFoldInputs(NamedTuple):
    """Store compact demonstrations and a padded leave-one-out schedule.

    Parameters
    ----------
    demonstration_inputs, demonstration_outputs : jax.Array
        Padded color grids shaped ``(tasks, demonstrations, rows, columns)``.
        Outputs are demonstration supervision, never official query targets.
    demonstration_input_shapes, demonstration_output_shapes : jax.Array
        Grid ``(height, width)`` pairs shaped ``(tasks, demonstrations, 2)``.
    demonstration_valid : jax.Array
        Boolean mask for padded demonstration slots.
    held_out_demonstration_index : jax.Array
        Ordered LOO query index shaped ``(tasks, folds)``; padding is ``-1``.
    fold_valid : jax.Array
        Boolean mask for padded fold slots.

    Notes
    -----
    Demonstration grids are stored once per task instead of once per fold. A
    compiled ARC callback can synthesize the selected LOO row-event stream from
    one held index without materializing a multi-gigabyte stream bank.
    """

    demonstration_inputs: jax.Array
    demonstration_input_shapes: jax.Array
    demonstration_outputs: jax.Array
    demonstration_output_shapes: jax.Array
    demonstration_valid: jax.Array
    held_out_demonstration_index: jax.Array
    fold_valid: jax.Array


class ArcTargetFreeTaskBank(NamedTuple):
    """Store compact task-local adaptation and official-query model inputs.

    Parameters
    ----------
    fold_inputs : ArcAdaptationFoldInputs
        Demonstration-only adaptation inputs and ordered LOO schedule.
    query_inputs : jax.Array
        Padded official query input grids shaped ``(tasks, queries, rows, columns)``.
    query_shapes : jax.Array
        Query input ``(height, width)`` pairs shaped ``(tasks, queries, 2)``.
    query_valid : jax.Array
        Boolean mask selecting real query slots.
    checkpoint_indices : jax.Array
        Semantic refinement checkpoints ``(0, 30, 60)`` per query slot.
    task_ordinals, query_ordinals : jax.Array
        Stable collection ordinals. Padded query ordinals are ``-1``.

    Notes
    -----
    Official query outputs, output shapes, task identifiers, fingerprints, and
    target-derived seeds are deliberately absent. The compact grids must be
    expanded into row events inside a compiled ARC runner, then joined with
    official targets only after candidate generation.
    """

    fold_inputs: ArcAdaptationFoldInputs
    query_inputs: jax.Array
    query_shapes: jax.Array
    query_valid: jax.Array
    checkpoint_indices: jax.Array
    task_ordinals: jax.Array
    query_ordinals: jax.Array

    @property
    def projected_bytes(self) -> int:
        """Return exact bytes occupied by all array leaves.

        Returns
        -------
        int
            Sum of leaf ``nbytes`` values for fail-closed resource reporting.
        """

        return sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(self))


class ArcContextStream(NamedTuple):
    """Hold one fixed-shape context stream synthesized from compact ARC grids.

    Parameters
    ----------
    events : jax.Array
        Row events shaped ``(row_config.max_events, row_config.input_width)``.
    advances : jax.Array
        Boolean gates selecting real context rows and freezing all padding.
    query_start, query_stop : jax.Array
        Fixed query-block start and exclusive visible-query stop indices.
    valid : jax.Array
        Scalar validity gate. Padded leave-one-out folds are completely inert.
    """

    events: jax.Array
    advances: jax.Array
    query_start: jax.Array
    query_stop: jax.Array
    valid: jax.Array


class ArcTaskBankAdaptationResult(NamedTuple):
    """Hold bounded outputs from compiled ARC task-local adaptation.

    Parameters
    ----------
    fold_losses : jax.Array
        Scalar pp-prop objectives shaped ``(tasks, folds)``.
    fold_applied : jax.Array
        Boolean mask identifying real leave-one-out optimizer updates.
    checkpoint_outputs : jax.Array
        Model logits shaped ``(tasks, queries, 3, 9060)`` for semantic
        checkpoints ``(0, 30, 60)``.
    checkpoint_recorded : jax.Array
        Boolean checkpoint validity shaped ``(tasks, queries, 3)``.
    query_valid : jax.Array
        Original padded-query validity mask.
    """

    fold_losses: jax.Array
    fold_applied: jax.Array
    checkpoint_outputs: jax.Array
    checkpoint_recorded: jax.Array
    query_valid: jax.Array


@dataclass(frozen=True)
class _ArcParameterBinding:
    state: Any
    base_value: Any


@dataclass(frozen=True)
class _ArcGatedParameter:
    state: Any
    previous_value: Any


def _copy_tree(value: Any) -> Any:
    return jax.tree.map(lambda leaf: jnp.array(leaf, copy=True), value)


def _validate_compact_task_inputs(
    fold_inputs: ArcAdaptationFoldInputs,
    row_config: RowEventConfig,
) -> int:
    if not isinstance(fold_inputs, ArcAdaptationFoldInputs):
        raise TypeError("fold_inputs must be ArcAdaptationFoldInputs")
    if not isinstance(row_config, RowEventConfig):
        raise TypeError("row_config must be a RowEventConfig")
    grid_size = row_config.max_grid_size
    inputs = jnp.asarray(fold_inputs.demonstration_inputs)
    input_shapes = jnp.asarray(fold_inputs.demonstration_input_shapes)
    outputs = jnp.asarray(fold_inputs.demonstration_outputs)
    output_shapes = jnp.asarray(fold_inputs.demonstration_output_shapes)
    valid = jnp.asarray(fold_inputs.demonstration_valid)
    held = jnp.asarray(fold_inputs.held_out_demonstration_index)
    fold_valid = jnp.asarray(fold_inputs.fold_valid)
    if inputs.ndim != 3 or inputs.shape[1:] != (grid_size, grid_size):
        raise ValueError(
            "task demonstration_inputs must have shape (demonstrations, rows, columns)"
        )
    demonstration_count = int(inputs.shape[0])
    if not 1 <= demonstration_count <= row_config.max_demonstrations:
        raise ValueError("task demonstration capacity is outside row_config")
    expected_grids = (demonstration_count, grid_size, grid_size)
    expected_shapes = (demonstration_count, 2)
    expected_masks = (demonstration_count,)
    if outputs.shape != expected_grids:
        raise ValueError("task demonstration_outputs shape is inconsistent")
    if input_shapes.shape != expected_shapes or output_shapes.shape != expected_shapes:
        raise ValueError("task demonstration shape tensors are inconsistent")
    if valid.shape != expected_masks:
        raise ValueError("task demonstration_valid shape is inconsistent")
    if held.shape != expected_masks or fold_valid.shape != expected_masks:
        raise ValueError("task leave-one-out schedule shape is inconsistent")
    for name, value in (
        ("demonstration_inputs", inputs),
        ("demonstration_outputs", outputs),
        ("demonstration_input_shapes", input_shapes),
        ("demonstration_output_shapes", output_shapes),
        ("held_out_demonstration_index", held),
    ):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must have integer dtype")
    if not np.issubdtype(valid.dtype, np.bool_) or not np.issubdtype(
        fold_valid.dtype, np.bool_
    ):
        raise ValueError("demonstration_valid and fold_valid must have boolean dtype")
    return demonstration_count


def _validate_query_arrays(
    query_input: jax.Array,
    query_shape: jax.Array,
    row_config: RowEventConfig,
) -> tuple[jax.Array, jax.Array]:
    grid = jnp.asarray(query_input)
    shape = jnp.asarray(query_shape)
    expected = (row_config.max_grid_size, row_config.max_grid_size)
    if grid.shape != expected:
        raise ValueError(f"query_input must have shape {expected}")
    if shape.shape != (2,):
        raise ValueError("query_shape must have shape (2,)")
    if not np.issubdtype(grid.dtype, np.integer) or not np.issubdtype(
        shape.dtype, np.integer
    ):
        raise ValueError("query_input and query_shape must have integer dtype")
    return grid, shape


def _encode_compact_side_rows(
    rows: jax.Array,
    grid: jax.Array,
    shape: jax.Array,
    *,
    is_output: bool,
    row_config: RowEventConfig,
) -> jax.Array:
    grid_size = row_config.max_grid_size
    height = jnp.asarray(shape[0], dtype=jnp.int32)
    width = jnp.asarray(shape[1], dtype=jnp.int32)
    row_indices = jnp.arange(grid_size, dtype=jnp.int32)
    column_indices = jnp.arange(grid_size, dtype=jnp.int32)
    row_valid = row_indices < height
    column_valid = column_indices < width
    side_offset = 1 if is_output else 0
    normal_offset = 3 if is_output else 1
    height_slice = (
        row_config.output_height_slice if is_output else row_config.input_height_slice
    )
    width_slice = (
        row_config.output_width_slice if is_output else row_config.input_width_slice
    )
    mask_slice = (
        row_config.output_mask_slice if is_output else row_config.input_mask_slice
    )
    color_slice = (
        row_config.output_color_slice if is_output else row_config.input_color_slice
    )
    row_gate = row_valid.astype(jnp.float32)
    normalized_dimensions = jnp.asarray(
        np.asarray(
            np.arange(grid_size + 1, dtype=np.float64) / grid_size,
            dtype=np.float32,
        )
    )
    rows = rows.at[:, row_config.side_valid_slice.start + side_offset].set(row_gate)
    rows = rows.at[:, row_config.normalized_slice.start + normal_offset].set(
        row_gate * normalized_dimensions[jnp.clip(height, 0, grid_size)]
    )
    rows = rows.at[:, row_config.normalized_slice.start + normal_offset + 1].set(
        row_gate * normalized_dimensions[jnp.clip(width, 0, grid_size)]
    )
    height_one_hot = jax.nn.one_hot(
        jnp.maximum(height - 1, 0), grid_size, dtype=jnp.float32
    )
    width_one_hot = jax.nn.one_hot(
        jnp.maximum(width - 1, 0), grid_size, dtype=jnp.float32
    )
    rows = rows.at[:, height_slice].set(row_gate[:, None] * height_one_hot[None])
    rows = rows.at[:, width_slice].set(row_gate[:, None] * width_one_hot[None])
    cell_valid = row_valid[:, None] & column_valid[None, :]
    rows = rows.at[:, mask_slice].set(cell_valid.astype(jnp.float32))
    colors = jax.nn.one_hot(
        jnp.asarray(grid, dtype=jnp.int32),
        row_config.color_count,
        dtype=jnp.float32,
    )
    colors = colors * cell_valid[:, :, None]
    return rows.at[:, color_slice].set(
        colors.reshape(grid_size, grid_size * row_config.color_count)
    )


def _encode_compact_pair_rows(
    input_grid: jax.Array,
    input_shape: jax.Array,
    output_grid: jax.Array,
    output_shape: jax.Array,
    *,
    demonstration_slot: jax.Array,
    is_query: bool,
    row_config: RowEventConfig,
) -> jax.Array:
    grid_size = row_config.max_grid_size
    row_indices = jnp.arange(grid_size, dtype=jnp.int32)
    input_valid = row_indices < jnp.asarray(input_shape[0], dtype=jnp.int32)
    output_valid = row_indices < jnp.asarray(output_shape[0], dtype=jnp.int32)
    event_valid = input_valid | output_valid
    gate = event_valid.astype(jnp.float32)
    rows = jnp.zeros((grid_size, row_config.input_width), dtype=jnp.float32)
    rows = rows.at[:, row_config.valid_slice.start].set(gate)
    phase_index = row_config.phase_slice.start + int(is_query)
    rows = rows.at[:, phase_index].set(gate)
    if not is_query:
        demonstration = jax.nn.one_hot(
            demonstration_slot,
            row_config.max_demonstrations,
            dtype=jnp.float32,
        )
        rows = rows.at[:, row_config.demonstration_slice].set(
            gate[:, None] * demonstration[None]
        )
    rows = rows.at[:, row_config.normalized_slice.start].set(
        gate
        * jnp.asarray(
            np.asarray(
                np.arange(1, grid_size + 1, dtype=np.float64) / grid_size,
                dtype=np.float32,
            )
        )
    )
    rows = rows.at[:, row_config.row_index_slice].set(
        gate[:, None] * jnp.eye(grid_size, dtype=jnp.float32)
    )
    rows = _encode_compact_side_rows(
        rows,
        input_grid,
        input_shape,
        is_output=False,
        row_config=row_config,
    )
    return _encode_compact_side_rows(
        rows,
        output_grid,
        output_shape,
        is_output=True,
        row_config=row_config,
    )


def _synthesize_arc_context(
    fold_inputs: ArcAdaptationFoldInputs,
    query_input: jax.Array,
    query_shape: jax.Array,
    held_out_demonstration_index: jax.Array,
    *,
    use_held_demonstration_as_query: bool,
    row_config: RowEventConfig,
) -> ArcContextStream:
    demonstration_count = _validate_compact_task_inputs(fold_inputs, row_config)
    query_grid, query_grid_shape = _validate_query_arrays(
        query_input, query_shape, row_config
    )
    inputs = jnp.asarray(fold_inputs.demonstration_inputs)
    input_shapes = jnp.asarray(fold_inputs.demonstration_input_shapes)
    outputs = jnp.asarray(fold_inputs.demonstration_outputs)
    output_shapes = jnp.asarray(fold_inputs.demonstration_output_shapes)
    demonstration_valid = jnp.asarray(fold_inputs.demonstration_valid)
    held = jnp.asarray(held_out_demonstration_index, dtype=jnp.int32)
    if held.shape != ():
        raise ValueError("held_out_demonstration_index must be scalar")
    safe_held = jnp.clip(held, 0, demonstration_count - 1)
    held_valid = (
        (held >= 0) & (held < demonstration_count) & demonstration_valid[safe_held]
    )
    stream_valid = held_valid if use_held_demonstration_as_query else jnp.bool_(True)
    if use_held_demonstration_as_query:
        query_grid = inputs[safe_held]
        query_grid_shape = input_shapes[safe_held]

    initial_events = jnp.zeros(
        (row_config.max_events, row_config.input_width), dtype=jnp.float32
    )
    initial_slot = jnp.asarray(0, dtype=jnp.int32)

    def add_demonstration(
        carry: tuple[jax.Array, jax.Array], source_index: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        events, slot = carry
        include = demonstration_valid[source_index] & stream_valid
        if use_held_demonstration_as_query:
            include = include & (source_index != safe_held)
        rows = _encode_compact_pair_rows(
            inputs[source_index],
            input_shapes[source_index],
            outputs[source_index],
            output_shapes[source_index],
            demonstration_slot=slot,
            is_query=False,
            row_config=row_config,
        )
        indices = slot * row_config.max_grid_size + jnp.arange(
            row_config.max_grid_size, dtype=jnp.int32
        )
        events = jax.lax.cond(
            include,
            lambda current: current.at[indices].set(rows),
            lambda current: current,
            events,
        )
        return (events, slot + include.astype(jnp.int32)), None

    (events, included_count), _ = jax.lax.scan(
        add_demonstration,
        (initial_events, initial_slot),
        jnp.arange(demonstration_count, dtype=jnp.int32),
    )
    zero_grid = jnp.zeros_like(query_grid)
    zero_shape = jnp.zeros_like(query_grid_shape)
    query_rows = _encode_compact_pair_rows(
        query_grid,
        query_grid_shape,
        zero_grid,
        zero_shape,
        demonstration_slot=jnp.asarray(0, dtype=jnp.int32),
        is_query=True,
        row_config=row_config,
    )
    query_rows = query_rows * stream_valid.astype(jnp.float32)
    query_start = row_config.max_demonstrations * row_config.max_grid_size
    query_indices = query_start + jnp.arange(row_config.max_grid_size, dtype=jnp.int32)
    events = events.at[query_indices].set(query_rows)
    source_indices = jnp.arange(demonstration_count, dtype=jnp.int32)
    included_sources = demonstration_valid & stream_valid
    if use_held_demonstration_as_query:
        included_sources = included_sources & (source_indices != safe_held)
    occupied_heights = jnp.maximum(input_shapes[:, 0], output_shapes[:, 0]).astype(
        jnp.int32
    )
    shared_height = jnp.max(
        jnp.where(included_sources, occupied_heights, jnp.asarray(0, jnp.int32))
    )
    demonstration_slots = jnp.arange(row_config.max_demonstrations, dtype=jnp.int32)[
        :, None
    ]
    demonstration_rows = jnp.arange(row_config.max_grid_size, dtype=jnp.int32)[None, :]
    demonstration_advances = (demonstration_slots < included_count) & (
        demonstration_rows < shared_height
    )
    advances = jnp.zeros((row_config.max_events,), dtype=jnp.bool_)
    advances = advances.at[:query_start].set(demonstration_advances.reshape(-1))
    advances = advances.at[query_indices].set(
        (demonstration_rows[0] < query_grid_shape[0]) & stream_valid
    )
    query_stop = query_start + jnp.where(
        stream_valid,
        jnp.asarray(query_grid_shape[0], dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return ArcContextStream(
        events=events,
        advances=advances,
        query_start=jnp.asarray(query_start, dtype=jnp.int32),
        query_stop=query_stop,
        valid=stream_valid,
    )


def synthesize_arc_evaluation_context(
    fold_inputs: ArcAdaptationFoldInputs,
    query_input: jax.Array,
    query_shape: jax.Array,
    row_config: RowEventConfig = RowEventConfig(),
) -> ArcContextStream:
    """Materialize one target-free official-query context inside JAX.

    Parameters
    ----------
    fold_inputs : ArcAdaptationFoldInputs
        One task slice from :class:`ArcTargetFreeTaskBank.fold_inputs`.
    query_input, query_shape : jax.Array
        One compact official query grid and its ``(height, width)`` shape.
    row_config : RowEventConfig, default=RowEventConfig()
        Static row-event layout shared with the model.

    Returns
    -------
    ArcContextStream
        Exact fixed-shape equivalent of :func:`encode_query_episode`, without
        any official output or target-derived value.
    """

    return _synthesize_arc_context(
        fold_inputs,
        query_input,
        query_shape,
        jnp.asarray(-1, dtype=jnp.int32),
        use_held_demonstration_as_query=False,
        row_config=row_config,
    )


def synthesize_arc_loo_context(
    fold_inputs: ArcAdaptationFoldInputs,
    held_out_demonstration_index: jax.Array,
    row_config: RowEventConfig = RowEventConfig(),
) -> ArcContextStream:
    """Materialize one demonstration-only leave-one-out context inside JAX.

    Parameters
    ----------
    fold_inputs : ArcAdaptationFoldInputs
        One compact task slice containing demonstrations once.
    held_out_demonstration_index : jax.Array
        Scalar original demonstration index. Negative padded indices produce
        an entirely inert stream with ``valid=False``.
    row_config : RowEventConfig, default=RowEventConfig()
        Static row-event layout shared with the model.

    Returns
    -------
    ArcContextStream
        Exact target-free equivalent of encoding the selected LOO episode.
        The held output remains only in ``fold_inputs.demonstration_outputs``
        for the caller to use as supervision.
    """

    demonstration_inputs = jnp.asarray(fold_inputs.demonstration_inputs)
    demonstration_shapes = jnp.asarray(fold_inputs.demonstration_input_shapes)
    return _synthesize_arc_context(
        fold_inputs,
        demonstration_inputs[0],
        demonstration_shapes[0],
        held_out_demonstration_index,
        use_held_demonstration_as_query=True,
        row_config=row_config,
    )


def _write_grid(target: np.ndarray, grid: ArcGrid) -> None:
    target[: grid.height, : grid.width] = grid.as_array()


def build_arc_target_free_task_bank(
    tasks: Sequence[ArcTask],
    row_config: RowEventConfig = RowEventConfig(),
    *,
    latent_steps: int = 60,
) -> ArcTargetFreeTaskBank:
    """Build compact fixed-shape ARC adaptation and prediction inputs.

    Public row-event encoders and decoders establish the prediction boundary:
    each official query is encoded without its output and decoded back to the
    exact visible demonstrations/query input before compact storage. Public
    leave-one-demonstration-out episodes establish fold count and order, while
    their targets remain represented only by the demonstration outputs.

    Parameters
    ----------
    tasks : sequence of ArcTask
        Evaluation tasks. Every task must contain at least two demonstrations.
    row_config : RowEventConfig, default=RowEventConfig()
        Static demonstration and grid capacities used by the model encoder.
    latent_steps : int, default=60
        Available row-refinement ticks. Must cover checkpoint 60.

    Returns
    -------
    ArcTargetFreeTaskBank
        Compact JAX arrays with static task/demo/query capacities and no
        official-query targets or fingerprints.

    Raises
    ------
    TypeError
        If inputs have the wrong types or ``latent_steps`` is not an integer.
    ValueError
        If the collection is empty, a task has fewer than two demonstrations,
        a capacity is exceeded, or fewer than 60 latent ticks are available.
    """

    if not isinstance(row_config, RowEventConfig):
        raise TypeError("row_config must be a RowEventConfig")
    if isinstance(latent_steps, bool) or not isinstance(latent_steps, int):
        raise TypeError("latent_steps must be an integer")
    if latent_steps < ARC_ADAPTATION_CHECKPOINTS[-1]:
        raise ValueError("latent_steps must be at least 60")
    task_items = tuple(tasks)
    if not task_items:
        raise ValueError("tasks must be a non-empty sequence")
    if any(not isinstance(task, ArcTask) for task in task_items):
        raise TypeError("every task must be an ArcTask")
    if any(len(task.train) < 2 for task in task_items):
        raise ValueError("every task needs at least two demonstrations")
    if any(len(task.train) > row_config.max_demonstrations for task in task_items):
        raise ValueError("task demonstration count exceeds row_config capacity")

    task_count = len(task_items)
    demo_capacity = max(len(task.train) for task in task_items)
    grid_capacity = row_config.max_grid_size
    query_capacity = max(len(task.test) for task in task_items)
    if query_capacity == 0:
        raise ValueError("every task collection must expose at least one query")

    demo_inputs = np.zeros(
        (task_count, demo_capacity, grid_capacity, grid_capacity), dtype=np.uint8
    )
    demo_outputs = np.zeros_like(demo_inputs)
    demo_input_shapes = np.zeros((task_count, demo_capacity, 2), dtype=np.uint8)
    demo_output_shapes = np.zeros_like(demo_input_shapes)
    demo_valid = np.zeros((task_count, demo_capacity), dtype=np.bool_)
    held_indices = np.full((task_count, demo_capacity), -1, dtype=np.int32)
    fold_valid = np.zeros_like(demo_valid)
    query_inputs = np.zeros(
        (task_count, query_capacity, grid_capacity, grid_capacity), dtype=np.uint8
    )
    query_shapes = np.zeros((task_count, query_capacity, 2), dtype=np.uint8)
    query_valid = np.zeros((task_count, query_capacity), dtype=np.bool_)
    query_ordinals = np.full((task_count, query_capacity), -1, dtype=np.int32)

    for task_index, task in enumerate(task_items):
        folds = leave_one_demonstration_out_episodes(task, task_index=task_index)
        held_indices[task_index, : len(folds)] = np.asarray(
            [fold.query_index for fold in folds], dtype=np.int32
        )
        fold_valid[task_index, : len(folds)] = True
        for query_index in range(len(task.test)):
            encoded = encode_query_episode(
                task,
                query_index,
                row_config,
                task_index=task_index,
            )
            visible = decode_row_events(encoded.events, row_config)
            if query_index == 0:
                for demonstration_index, pair in enumerate(visible.demonstrations):
                    assert pair.output is not None
                    _write_grid(
                        demo_inputs[task_index, demonstration_index], pair.input
                    )
                    _write_grid(
                        demo_outputs[task_index, demonstration_index], pair.output
                    )
                    demo_input_shapes[task_index, demonstration_index] = (
                        pair.input.height,
                        pair.input.width,
                    )
                    demo_output_shapes[task_index, demonstration_index] = (
                        pair.output.height,
                        pair.output.width,
                    )
                    demo_valid[task_index, demonstration_index] = True
            _write_grid(query_inputs[task_index, query_index], visible.query_input)
            query_shapes[task_index, query_index] = (
                visible.query_input.height,
                visible.query_input.width,
            )
            query_valid[task_index, query_index] = True
            query_ordinals[task_index, query_index] = query_index

    checkpoints = np.broadcast_to(
        np.asarray(ARC_ADAPTATION_CHECKPOINTS, dtype=np.int32),
        (task_count, query_capacity, len(ARC_ADAPTATION_CHECKPOINTS)),
    ).copy()
    return ArcTargetFreeTaskBank(
        fold_inputs=ArcAdaptationFoldInputs(
            demonstration_inputs=jnp.asarray(demo_inputs),
            demonstration_input_shapes=jnp.asarray(demo_input_shapes),
            demonstration_outputs=jnp.asarray(demo_outputs),
            demonstration_output_shapes=jnp.asarray(demo_output_shapes),
            demonstration_valid=jnp.asarray(demo_valid),
            held_out_demonstration_index=jnp.asarray(held_indices),
            fold_valid=jnp.asarray(fold_valid),
        ),
        query_inputs=jnp.asarray(query_inputs),
        query_shapes=jnp.asarray(query_shapes),
        query_valid=jnp.asarray(query_valid),
        checkpoint_indices=jnp.asarray(checkpoints),
        task_ordinals=jnp.arange(task_count, dtype=jnp.int32),
        query_ordinals=jnp.asarray(query_ordinals),
    )


def _validate_arc_runner_bank(
    bank: ArcTargetFreeTaskBank,
    row_config: RowEventConfig,
) -> None:
    if not isinstance(bank, ArcTargetFreeTaskBank):
        raise TypeError("bank must be an ArcTargetFreeTaskBank")
    task_count = int(bank.query_inputs.shape[0])
    if task_count < 1:
        raise ValueError("bank must contain at least one task")
    query_count = int(bank.query_inputs.shape[1])
    grid_size = row_config.max_grid_size
    if bank.query_inputs.shape != (task_count, query_count, grid_size, grid_size):
        raise ValueError("bank query_inputs shape is inconsistent")
    if bank.query_shapes.shape != (task_count, query_count, 2):
        raise ValueError("bank query_shapes shape is inconsistent")
    if bank.query_valid.shape != (task_count, query_count):
        raise ValueError("bank query_valid shape is inconsistent")
    if not np.issubdtype(bank.query_valid.dtype, np.bool_):
        raise ValueError("bank query_valid must have boolean dtype")
    if bank.checkpoint_indices.shape != (task_count, query_count, 3):
        raise ValueError("bank checkpoint_indices must have shape (tasks, queries, 3)")
    expected = np.broadcast_to(
        np.asarray(ARC_ADAPTATION_CHECKPOINTS, dtype=np.int32),
        bank.checkpoint_indices.shape,
    )
    if not np.array_equal(np.asarray(bank.checkpoint_indices), expected):
        raise ValueError("bank checkpoints must be exactly (0, 30, 60)")
    fold_leaves = jax.tree.leaves(bank.fold_inputs)
    if not fold_leaves or any(int(leaf.shape[0]) != task_count for leaf in fold_leaves):
        raise ValueError("bank fold inputs must share the query task axis")


def _validate_arc_runner_configuration(
    model: Any,
    learner: Any,
    optimizer: Any,
    base_parameters: ParameterSnapshot,
    row_config: RowEventConfig,
    latent_steps: int,
    clip_norm: float,
) -> None:
    if not isinstance(base_parameters, ParameterSnapshot):
        raise TypeError("base_parameters must be a ParameterSnapshot")
    if not isinstance(row_config, RowEventConfig):
        raise TypeError("row_config must be a RowEventConfig")
    if isinstance(latent_steps, bool) or not isinstance(latent_steps, int):
        raise TypeError("latent_steps must be an integer")
    if latent_steps != ARC_ADAPTATION_CHECKPOINTS[-1]:
        raise ValueError("ARC task-local adaptation requires exactly 60 latent steps")
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, Real):
        raise TypeError("clip_norm must be a positive finite real")
    if not np.isfinite(float(clip_norm)) or float(clip_norm) <= 0.0:
        raise ValueError("clip_norm must be a positive finite real")
    config = getattr(model, "config", None)
    if getattr(config, "batch_size", None) != 1:
        raise ValueError("ARC task-local adaptation requires model batch_size=1")
    if getattr(config, "decoder_mode", None) != "row_refinement":
        raise ValueError("ARC task-local adaptation requires row_refinement mode")
    if getattr(config, "input_width", None) != row_config.input_width:
        raise ValueError("model input width does not match row_config")
    if getattr(config, "refinement_steps", 0) < latent_steps:
        raise ValueError("model refinement_steps must cover 60 latent steps")
    if getattr(config, "checkpoint_output_width", None) != 9060:
        raise ValueError("row-refinement checkpoint output width must be 9060")
    if not hasattr(learner, "etrace_grad") or not hasattr(learner, "reset_state"):
        raise TypeError("learner must expose etrace_grad and reset_state")
    if type(optimizer).__name__ != "Adam":
        raise ValueError("ARC task-local adaptation requires a plain Adam optimizer")
    if float(getattr(optimizer, "weight_decay", 0.0)) != 0.0:
        raise ValueError("ARC task-local adaptation requires Adam weight_decay=0")
    if len(getattr(optimizer, "param_groups", ())) > 1:
        raise ValueError("ARC task-local adaptation does not support Adam groups")
    if getattr(optimizer, "_schedulers", ()):  # noqa: SLF001
        raise ValueError("ARC task-local adaptation does not support schedulers")
    for name in ("opt_state", "step_count", "_current_lr"):
        if not hasattr(optimizer, name):
            raise ValueError(f"Adam optimizer lacks required state {name}")


def compile_arc_task_local_adaptation_runner(
    model: Any,
    learner: Any,
    optimizer: braintools.optim.Adam,
    *,
    base_parameters: ParameterSnapshot,
    row_config: RowEventConfig,
    latent_steps: int = 60,
    clip_norm: float = 1.0,
) -> Any:
    """Compile target-free ARC leave-one-out adaptation and query inference.

    The runner reuses one batch-size-one row-refinement model. It restores the
    shared pretrained parameters and fresh Adam state before every task,
    synthesizes one compact leave-one-out stream at a time, performs pp-prop
    updates only for valid folds, and records only semantic checkpoints 0, 30,
    and 60 for official target-free queries. Parameters, optimizer state,
    model dynamics, and eligibility traces are restored on every exit.

    Parameters
    ----------
    model
        Batch-size-one row-refinement model.
    learner
        Compiled pp-prop learner for ``model``.
    optimizer
        Fresh plain Adam registered against ``learner.param_states``.
    base_parameters
        Immutable shared pretrained parameter snapshot.
    row_config
        Static row-event layout shared by the compact bank and model.
    latent_steps
        Fixed refinement depth. The current protocol requires exactly 60.
    clip_norm
        Positive finite gradient clipping norm.

    Returns
    -------
    callable
        A compiled function from :class:`ArcTargetFreeTaskBank` to
        :class:`ArcTaskBankAdaptationResult`.
    """

    _validate_arc_runner_configuration(
        model,
        learner,
        optimizer,
        base_parameters,
        row_config,
        latent_steps,
        clip_norm,
    )
    restore_parameters(model, base_parameters)
    parameter_bindings = tuple(
        map(
            lambda state: _ArcParameterBinding(state, _copy_tree(state.value)),
            model.states(brainstate.ParamState).values(),
        )
    )
    optimizer_base_state = _copy_tree(optimizer.opt_state.value)
    optimizer_base_step = jnp.array(optimizer.step_count.value, copy=True)
    optimizer_base_lr = jnp.array(optimizer._current_lr.value, copy=True)  # noqa: SLF001
    zero_latent_events = jnp.zeros(
        (latent_steps, 1, row_config.input_width), dtype=jnp.float32
    )
    latent_advances = jnp.ones((latent_steps, 1), dtype=jnp.bool_)
    loss_mask = jnp.concatenate(
        (
            jnp.zeros((row_config.max_events,), dtype=jnp.float32),
            jnp.full((latent_steps,), 1.0 / latent_steps, dtype=jnp.float32),
        )
    )

    def restore_binding(binding: _ArcParameterBinding) -> None:
        binding.state.value = _copy_tree(binding.base_value)

    def restore_optimizer() -> None:
        optimizer.opt_state.value = _copy_tree(optimizer_base_state)
        optimizer.step_count.value = jnp.array(optimizer_base_step, copy=True)
        optimizer._current_lr.value = jnp.array(  # noqa: SLF001
            optimizer_base_lr, copy=True
        )

    def reset_dynamics() -> None:
        model.reset_state()
        learner.reset_state(batch_size=1)

    def reset_task() -> None:
        jax.tree.map(
            restore_binding,
            parameter_bindings,
            is_leaf=lambda value: isinstance(value, _ArcParameterBinding),
        )
        restore_optimizer()
        reset_dynamics()

    def gate_parameter(binding: _ArcGatedParameter, apply_update: jax.Array) -> None:
        binding.state.value = jax.tree.map(
            lambda current, previous: jnp.where(apply_update, current, previous),
            binding.state.value,
            binding.previous_value,
        )

    def run_compiled(
        fold_inputs: ArcAdaptationFoldInputs,
        query_inputs: jax.Array,
        query_shapes: jax.Array,
        query_valid: jax.Array,
    ) -> ArcTaskBankAdaptationResult:
        def run_task(
            task_inputs: tuple[
                ArcAdaptationFoldInputs, jax.Array, jax.Array, jax.Array
            ],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            task_folds, task_queries, task_query_shapes, task_query_valid = task_inputs
            reset_task()

            def adapt_one(
                fold_schedule: tuple[jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array]:
                held_index, scheduled_valid = fold_schedule
                reset_dynamics()
                context = synthesize_arc_loo_context(task_folds, held_index, row_config)
                apply_update = scheduled_valid & context.valid
                safe_held = jnp.clip(
                    jnp.asarray(held_index, dtype=jnp.int32),
                    0,
                    task_folds.demonstration_outputs.shape[0] - 1,
                )
                target_shape = task_folds.demonstration_output_shapes[safe_held]
                target_height = jnp.asarray(
                    [target_shape[0].astype(jnp.int32) - 1], dtype=jnp.int32
                )
                target_width = jnp.asarray(
                    [target_shape[1].astype(jnp.int32) - 1], dtype=jnp.int32
                )
                target_colors = jnp.asarray(
                    task_folds.demonstration_outputs[safe_held][None],
                    dtype=jnp.int32,
                )
                events = jnp.concatenate((context.events[:, None], zero_latent_events))
                advances = jnp.concatenate((context.advances[:, None], latent_advances))
                gated_parameters = tuple(
                    map(
                        lambda binding: _ArcGatedParameter(
                            binding.state, _copy_tree(binding.state.value)
                        ),
                        parameter_bindings,
                    )
                )
                previous_optimizer_state = _copy_tree(optimizer.opt_state.value)
                previous_optimizer_step = jnp.array(
                    optimizer.step_count.value, copy=True
                )
                previous_optimizer_lr = jnp.array(
                    optimizer._current_lr.value,
                    copy=True,  # noqa: SLF001
                )

                def step_loss(event: jax.Array, advance: jax.Array) -> jax.Array:
                    logits = learner(event, advance)
                    row_indices = jnp.mod(
                        jnp.asarray(model.reasoning_index.value, dtype=jnp.int32) - 1,
                        row_config.max_grid_size,
                    )
                    return jnp.mean(
                        row_refinement_loss_per_example(
                            logits,
                            target_height,
                            target_width,
                            target_colors,
                            row_indices,
                        )
                    )

                gradients, objective = learner.etrace_grad(
                    events,
                    advances,
                    step_fn=step_loss,
                    mask=loss_mask,
                    reduction="mean",
                    loss_output="scalar",
                    return_value=True,
                )
                optimizer.update(
                    brainstate.nn.clip_grad_norm(gradients, float(clip_norm))
                )
                jax.tree.map(
                    lambda binding: gate_parameter(binding, apply_update),
                    gated_parameters,
                    is_leaf=lambda value: isinstance(value, _ArcGatedParameter),
                )
                optimizer.opt_state.value = jax.tree.map(
                    lambda current, previous: jnp.where(
                        apply_update, current, previous
                    ),
                    optimizer.opt_state.value,
                    previous_optimizer_state,
                )
                optimizer.step_count.value = jnp.where(
                    apply_update,
                    optimizer.step_count.value,
                    previous_optimizer_step,
                )
                optimizer._current_lr.value = jnp.where(  # noqa: SLF001
                    apply_update,
                    optimizer._current_lr.value,  # noqa: SLF001
                    previous_optimizer_lr,
                )
                return jnp.where(apply_update, objective, 0.0), apply_update

            fold_losses, fold_applied = brainstate.transform.for_loop(
                adapt_one,
                (
                    task_folds.held_out_demonstration_index,
                    task_folds.fold_valid,
                ),
            )

            def run_query(
                query: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array]:
                query_input, query_shape, is_valid = query
                reset_dynamics()
                context = synthesize_arc_evaluation_context(
                    task_folds, query_input, query_shape, row_config
                )

                def context_step(
                    step: tuple[jax.Array, jax.Array],
                ) -> jax.Array:
                    event, advance = step
                    model.cell_step(event[None], (advance & is_valid)[None])
                    return jnp.asarray(0, dtype=jnp.uint8)

                brainstate.transform.for_loop(
                    context_step, (context.events, context.advances)
                )
                checkpoint_zero = model.compact_readout()[0]

                def latent_step(_index: jax.Array) -> jax.Array:
                    model.cell_step(
                        zero_latent_events[0],
                        jnp.asarray([is_valid], dtype=jnp.bool_),
                    )
                    return jnp.asarray(0, dtype=jnp.uint8)

                brainstate.transform.for_loop(
                    latent_step,
                    jnp.arange(ARC_ADAPTATION_CHECKPOINTS[1], dtype=jnp.int32),
                )
                checkpoint_thirty = model.compact_readout()[0]
                brainstate.transform.for_loop(
                    latent_step,
                    jnp.arange(
                        ARC_ADAPTATION_CHECKPOINTS[2] - ARC_ADAPTATION_CHECKPOINTS[1],
                        dtype=jnp.int32,
                    ),
                )
                checkpoint_sixty = model.compact_readout()[0]
                checkpoints = jnp.stack(
                    (checkpoint_zero, checkpoint_thirty, checkpoint_sixty)
                )
                checkpoints = jnp.where(is_valid, checkpoints, 0.0)
                return checkpoints, jnp.full((3,), is_valid, dtype=jnp.bool_)

            checkpoint_outputs, checkpoint_recorded = brainstate.transform.for_loop(
                run_query,
                (task_queries, task_query_shapes, task_query_valid),
            )
            reset_task()
            return (
                fold_losses,
                fold_applied,
                checkpoint_outputs,
                checkpoint_recorded,
            )

        fold_losses, fold_applied, checkpoint_outputs, checkpoint_recorded = (
            brainstate.transform.for_loop(
                run_task,
                (fold_inputs, query_inputs, query_shapes, query_valid),
            )
        )
        reset_task()
        return ArcTaskBankAdaptationResult(
            fold_losses=fold_losses,
            fold_applied=fold_applied,
            checkpoint_outputs=checkpoint_outputs,
            checkpoint_recorded=checkpoint_recorded,
            query_valid=query_valid,
        )

    compiled = brainstate.transform.jit(run_compiled)

    def run(bank: ArcTargetFreeTaskBank) -> ArcTaskBankAdaptationResult:
        _validate_arc_runner_bank(bank, row_config)
        try:
            return compiled(
                bank.fold_inputs,
                bank.query_inputs,
                bank.query_shapes,
                bank.query_valid,
            )
        finally:
            restore_parameters(model, base_parameters)
            restore_optimizer()
            reset_dynamics()

    return run
