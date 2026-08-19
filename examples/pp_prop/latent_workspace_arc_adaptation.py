"""Compact target-free ARC tensors for task-local pp-prop adaptation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcTask,
        RowEventConfig,
        decode_row_events,
        encode_query_episode,
        leave_one_demonstration_out_episodes,
    )
except ImportError:
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
