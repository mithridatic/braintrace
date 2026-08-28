"""Task-local parameter isolation for Example 21 ARC adaptation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ParameterRecord:
    """Store one immutable path-keyed ``ParamState`` value.

    Parameters
    ----------
    path : tuple
        Exact BrainState graph path of the parameter.
    tree_structure : object
        JAX pytree definition for the parameter value.
    leaves : tuple
        Copied immutable array leaves in tree traversal order.
    """

    path: tuple[Any, ...]
    tree_structure: Any
    leaves: tuple[jax.Array, ...]


@dataclass(frozen=True)
class ParameterSnapshot:
    """Store an exact immutable snapshot of all model parameters.

    Parameters
    ----------
    records : tuple of ParameterRecord
        Parameter records in deterministic BrainState graph order.
    """

    records: tuple[ParameterRecord, ...]


@dataclass(frozen=True)
class TargetFreeQuery:
    """Hold an official query stream without any output target.

    Parameters
    ----------
    events : jax.Array
        Query and latent events with time on the leading axis.
    advances : jax.Array
        Boolean advance gates shaped ``(time,)``.
    """

    events: jax.Array
    advances: jax.Array

    def __post_init__(self) -> None:
        events = jnp.asarray(self.events)
        advances = jnp.asarray(self.advances)
        if events.ndim < 1:
            raise ValueError("Events must have a leading step axis. Ensure Events has a leading step axis.")
        if advances.ndim != 1:
            raise ValueError("Advances must be one-dimensional. Set Advances to one-dimensional.")
        if events.shape[0] == 0:
            raise ValueError("Query must contain at least one step. Add at least one step to Query.")
        if events.shape[0] != advances.shape[0]:
            raise ValueError("Events and advances must have the same step count. Ensure Events and advances has the same step count.")
        if not np.issubdtype(advances.dtype, np.bool_):
            raise ValueError("Advances must have boolean dtype. Ensure Advances has boolean dtype.")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "advances", advances)


@dataclass(frozen=True)
class TaskLocalAdaptationResult:
    """Hold outputs from one isolated task-local adaptation run.

    Parameters
    ----------
    fold_outputs : object
        Outputs stacked by the compiled adaptation-fold loop.
    prediction : object
        Copied target-free query prediction returned before cleanup.
    adapted_parameters : ParameterSnapshot or None
        Optional task-local snapshot captured before query inference. The live
        model has already been restored to the shared base parameters.
    """

    fold_outputs: Any
    prediction: Any
    adapted_parameters: ParameterSnapshot | None


class TargetFreeTaskBank(NamedTuple):
    """Store fixed-shape task-local adaptation and query tensors.

    Parameters
    ----------
    fold_inputs : object
        Array pytree shaped ``(tasks, updates, ...)``. Demonstration targets
        may occur here because they supervise leave-one-out adaptation.
    query_events : jax.Array
        Official target-free query events shaped ``(tasks, queries, time, ...)``.
    query_advances : jax.Array
        Boolean query advance gates shaped ``(tasks, queries, time)``.
    query_valid : jax.Array
        Boolean mask shaped ``(tasks, queries)`` for padded query slots.
    checkpoint_indices : jax.Array
        Selected query steps shaped ``(tasks, queries, checkpoints)``.

    Notes
    -----
    This type deliberately has no official-query target field. Targets remain
    outside the compiled prediction boundary and may be joined only for scoring.
    """

    fold_inputs: Any
    query_events: jax.Array
    query_advances: jax.Array
    query_valid: jax.Array
    checkpoint_indices: jax.Array


class TaskBankAdaptationResult(NamedTuple):
    """Hold bounded outputs from a compiled task-bank adaptation run.

    Parameters
    ----------
    fold_outputs : object
        Adaptation diagnostics stacked by task and update.
    checkpoint_outputs : jax.Array
        Outputs recorded only at selected checkpoints.
    checkpoint_recorded : jax.Array
        Boolean mask identifying recorded valid-query checkpoints.
    query_valid : jax.Array
        Original padded-query validity mask.
    """

    fold_outputs: Any
    checkpoint_outputs: jax.Array
    checkpoint_recorded: jax.Array
    query_valid: jax.Array


@dataclass(frozen=True)
class _ParameterBinding:
    state: Any
    value: Any


@dataclass(frozen=True)
class _OptimizerSnapshot:
    opt_state: Any
    step_count: jax.Array
    current_lr: jax.Array


def _copy_leaf(leaf: Any) -> jax.Array:
    return jnp.array(leaf, copy=True)


def _copy_tree(value: Any) -> Any:
    return jax.tree.map(_copy_leaf, value)


def _parameter_items(model: Any) -> tuple[tuple[tuple[Any, ...], Any], ...]:
    try:
        states = model.states(brainstate.ParamState)
    except AttributeError as error:
        raise TypeError("Model must expose BrainState ParamState objects. Set Model to expose BrainState ParamState objects.") from error
    return tuple((tuple(path), state) for path, state in states.items())


def snapshot_parameters(model: Any) -> ParameterSnapshot:
    """Copy every ``ParamState`` into an immutable path-keyed snapshot.

    Parameters
    ----------
    model : brainstate.nn.Module
        Model that owns the parameters, including ``LatentWorkspaceModel``.

    Returns
    -------
    ParameterSnapshot
        Exact parameter paths, pytree structures, and copied array leaves.
    """
    records = []
    for path, state in _parameter_items(model):
        leaves, tree_structure = jax.tree.flatten(state.value)
        records.append(
            ParameterRecord(
                path=path,
                tree_structure=tree_structure,
                leaves=tuple(_copy_leaf(leaf) for leaf in leaves),
            )
        )
    return ParameterSnapshot(records=tuple(records))


def restore_parameters(model: Any, snapshot: ParameterSnapshot) -> None:
    """Transactionally restore an exactly compatible parameter snapshot.

    Every path, pytree structure, leaf count, shape, and dtype is validated
    before any live parameter is modified.

    Parameters
    ----------
    model : brainstate.nn.Module
        Model receiving the parameter values.
    snapshot : ParameterSnapshot
        Snapshot previously captured from the same model configuration.

    Raises
    ------
    TypeError
        If ``snapshot`` is not a :class:`ParameterSnapshot`.
    ValueError
        If parameter paths, structures, shapes, or dtypes differ.
    """
    if not isinstance(snapshot, ParameterSnapshot):
        raise TypeError("Snapshot must be a ParameterSnapshot. Set Snapshot to a ParameterSnapshot.")
    parameter_items = _parameter_items(model)
    expected_paths = tuple(path for path, _ in parameter_items)
    actual_paths = tuple(record.path for record in snapshot.records)
    if actual_paths != expected_paths:
        raise ValueError("Parameter paths do not match this model. Fix the input condition named in the error, then rerun the operation.")
    validated: list[tuple[Any, Any]] = []
    for record, (_, state) in zip(snapshot.records, parameter_items, strict=True):
        state_leaves, state_structure = jax.tree.flatten(state.value)
        if record.tree_structure != state_structure:
            raise ValueError(f"Parameter structure does not match at {record.path}. Use matching values and structures.")
        if len(record.leaves) != len(state_leaves):
            raise ValueError(f"Parameter leaf count does not match at {record.path}. Use matching values and structures.")
        for snapshot_leaf, state_leaf in zip(record.leaves, state_leaves, strict=True):
            if np.shape(snapshot_leaf) != np.shape(state_leaf):
                raise ValueError(f"Parameter shape does not match at {record.path}. Use matching values and structures.")
            if np.dtype(snapshot_leaf.dtype) != np.dtype(state_leaf.dtype):
                raise ValueError(f"Parameter dtype does not match at {record.path}. Use matching values and structures.")
        value = jax.tree.unflatten(
            record.tree_structure,
            tuple(_copy_leaf(leaf) for leaf in record.leaves),
        )
        validated.append((state, value))
    for state, value in validated:
        state.value = value


def _fold_count(fold_inputs: Any) -> int:
    leaves = jax.tree.leaves(fold_inputs)
    if not leaves:
        raise ValueError("Fold inputs must contain at least one array. Add at least one array to Fold inputs.")
    counts: list[int] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if array.ndim < 1:
            raise ValueError("Every fold input must have a leading fold axis. Ensure Every fold input has a leading fold axis.")
        counts.append(int(array.shape[0]))
    if not counts or counts[0] == 0:
        raise ValueError("Fold inputs must contain at least one fold. Add at least one fold to Fold inputs.")
    if any(count != counts[0] for count in counts[1:]):
        raise ValueError("All fold inputs must have the same fold count. Ensure All fold inputs has the same fold count.")
    return counts[0]


def build_target_free_task_bank(
    *,
    fold_inputs: Any,
    query_events: Any,
    query_advances: Any,
    query_valid: Any,
    checkpoint_indices: Any,
) -> TargetFreeTaskBank:
    """Validate and build a fixed-shape target-free query task bank.

    Parameters
    ----------
    fold_inputs : object
        Array pytree with leading ``(tasks, updates)`` axes.
    query_events : array-like
        Target-free events with leading ``(tasks, queries, time)`` axes.
    query_advances : array-like
        Boolean advance gates shaped ``(tasks, queries, time)``.
    query_valid : array-like
        Boolean padded-query mask shaped ``(tasks, queries)``.
    checkpoint_indices : array-like
        Integer selected steps shaped ``(tasks, queries, checkpoints)``.

    Returns
    -------
    TargetFreeTaskBank
        JAX-array pytree accepted by the compiled task-local runner.

    Raises
    ------
    ValueError
        If axes, dtypes, counts, or checkpoint bounds are inconsistent.
    """
    events = jnp.asarray(query_events)
    advances = jnp.asarray(query_advances)
    valid = jnp.asarray(query_valid)
    indices = jnp.asarray(checkpoint_indices)
    if events.ndim < 3:
        raise ValueError("query_events must have task, query, and time axes. Ensure query_events has task, query, and time axes.")
    task_count, query_count, time_count = events.shape[:3]
    if task_count == 0 or query_count == 0 or time_count == 0:
        raise ValueError("Query task, query, and time counts must be positive. Set Query task, query, and time counts to a positive value.")
    if advances.shape != (task_count, query_count, time_count):
        raise ValueError("query_advances shape must match query task/query/time axes. Make query_advances shape match query task/query/time axes.")
    if not np.issubdtype(advances.dtype, np.bool_):
        raise ValueError("query_advances must have boolean dtype. Ensure query_advances has boolean dtype.")
    if valid.ndim != 2:
        raise ValueError("query_valid must be two-dimensional. Set query_valid to two-dimensional.")
    if valid.shape != (task_count, query_count):
        raise ValueError("query_valid shape must match query task/query axes. Make query_valid shape match query task/query axes.")
    if not np.issubdtype(valid.dtype, np.bool_):
        raise ValueError("query_valid must have boolean dtype. Ensure query_valid has boolean dtype.")
    if indices.ndim != 3:
        raise ValueError("checkpoint_indices must be three-dimensional. Set checkpoint_indices to three-dimensional.")
    if indices.shape[:2] != (task_count, query_count) or indices.shape[2] == 0:
        raise ValueError("checkpoint_indices must match task/query axes. Make checkpoint_indices match task/query axes.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("checkpoint_indices must have integer dtype. Ensure checkpoint_indices has integer dtype.")
    host_indices = np.asarray(indices)
    if np.any(host_indices < 0) or np.any(host_indices >= time_count):
        raise ValueError("Checkpoint index is outside query time. Set the named field to a value in the stated range, then rerun the operation.")
    if np.any(np.diff(host_indices, axis=-1) <= 0):
        raise ValueError("Checkpoint indices must be strictly increasing. Set Checkpoint indices to strictly increasing.")

    fold_leaves = jax.tree.leaves(fold_inputs)
    if not fold_leaves:
        raise ValueError("Fold inputs must contain at least one array. Add at least one array to Fold inputs.")
    fold_shapes = tuple(np.shape(leaf) for leaf in fold_leaves)
    if any(len(shape) < 2 for shape in fold_shapes):
        raise ValueError("Every fold input must have task and update axes. Ensure Every fold input has task and update axes.")
    if any(shape[0] != task_count for shape in fold_shapes):
        raise ValueError("Fold input task count must match query task count. Make Fold input task count match query task count.")
    update_count = fold_shapes[0][1]
    if update_count == 0:
        raise ValueError("Fold inputs must contain at least one update. Add at least one update to Fold inputs.")
    if any(shape[1] != update_count for shape in fold_shapes[1:]):
        raise ValueError("All fold inputs must have the same update count. Ensure All fold inputs has the same update count.")
    return TargetFreeTaskBank(
        fold_inputs=jax.tree.map(jnp.asarray, fold_inputs),
        query_events=events,
        query_advances=advances,
        query_valid=valid,
        checkpoint_indices=indices.astype(jnp.int32),
    )


def run_task_local_adaptation(
    model: Any,
    *,
    base_parameters: ParameterSnapshot,
    fold_inputs: Any,
    query: TargetFreeQuery,
    make_optimizer: Callable[[], Any],
    adapt_fold: Callable[[Any, Any], Any],
    query_step: Callable[[jax.Array, jax.Array], None],
    finish_query: Callable[[], Any],
    reset_dynamic_state: Callable[[], None] | None = None,
    capture_adapted_parameters: bool = False,
) -> TaskLocalAdaptationResult:
    """Adapt and infer one task without leaking state into another task.

    The shared base parameters are restored at entry and again on every exit,
    including exceptions. A fresh optimizer is created for the task. Dynamic
    model and learner state is reset before every adaptation fold, before query
    inference, and after cleanup. Adaptation folds and query steps are each
    executed by :func:`brainstate.transform.for_loop`.

    Parameters
    ----------
    model : brainstate.nn.Module
        Model whose parameters and dynamics are isolated.
    base_parameters : ParameterSnapshot
        Immutable shared pretrained parameter snapshot.
    fold_inputs : object
        Array pytree with a common nonempty leading fold axis. Supervised
        demonstration targets may appear here.
    query : TargetFreeQuery
        Official target-free inference sequence.
    make_optimizer : callable
        Zero-argument factory returning a newly initialized optimizer already
        registered against the model's task-local parameters.
    adapt_fold : callable
        ``adapt_fold(optimizer, fold)`` performs one bounded fold update and
        returns a small diagnostic value.
    query_step : callable
        ``query_step(event, advance)`` advances the model once. Its return is
        discarded so full checkpoint logits are not stacked per step.
    finish_query : callable
        Zero-argument function reading the final model state into a prediction.
    reset_dynamic_state : callable, optional
        Reset both model and learner dynamics. Defaults to ``model.reset_state``.
    capture_adapted_parameters : bool, default=False
        Capture task-local parameters for diagnostics before query inference.

    Returns
    -------
    TaskLocalAdaptationResult
        Fold diagnostics, copied prediction, and optional adapted parameters.

    Raises
    ------
    TypeError
        If the query does not use the target-free query type.
    ValueError
        If the fold batch is empty or has inconsistent leading dimensions.
    """
    if not isinstance(query, TargetFreeQuery):
        raise TypeError("Query must be a TargetFreeQuery. Set Query to a TargetFreeQuery.")
    _fold_count(fold_inputs)
    reset = model.reset_state if reset_dynamic_state is None else reset_dynamic_state
    restore_parameters(model, base_parameters)
    reset()
    try:
        optimizer = make_optimizer()

        def adapt_one(fold: Any) -> Any:
            reset()
            return adapt_fold(optimizer, fold)

        fold_outputs = brainstate.transform.for_loop(adapt_one, fold_inputs)
        adapted_parameters = (
            snapshot_parameters(model) if capture_adapted_parameters else None
        )
        reset()

        def advance_query(inputs: tuple[jax.Array, jax.Array]) -> None:
            event, advance = inputs
            query_step(event, advance)

        brainstate.transform.for_loop(
            advance_query,
            (query.events, query.advances),
        )
        prediction = _copy_tree(finish_query())
        return TaskLocalAdaptationResult(
            fold_outputs=_copy_tree(fold_outputs),
            prediction=prediction,
            adapted_parameters=adapted_parameters,
        )
    finally:
        restore_parameters(model, base_parameters)
        reset()


def compile_task_local_adaptation_runner(
    model: Any,
    learner: Any,
    optimizer: Any,
    *,
    base_parameters: ParameterSnapshot,
    adapt_fold: Callable[[Any, Any, Any], Any],
    query_step: Callable[[jax.Array, jax.Array], None],
    checkpoint_output: Callable[[], jax.Array],
    checkpoint_output_shape: tuple[int, ...],
    checkpoint_output_dtype: Any,
) -> Callable[[TargetFreeTaskBank], TaskBankAdaptationResult]:
    """Compile isolated adaptation and bounded query checkpoints for a task bank.

    One batch-size-one model, pp-prop learner, and Adam optimizer are reused by
    the compiled driver. Before every task it restores the shared parameter
    bytes, zero/fresh optimizer state, model dynamics, and eligibility traces.
    Model dynamics and traces reset again before every fold and query, while
    Adam state persists only across the folds belonging to the current task.

    Repeated work is lowered through nested :mod:`brainstate.transform` loops:
    A task loop, an adaptation-fold loop, a padded-query loop, and a query-time
    scan. Only named checkpoint outputs are carried; per-step outputs are not
    stacked.

    Parameters
    ----------
    model : brainstate.nn.Module
        Batch-size-one model whose parameters and dynamics are isolated.
    learner : object
        Compiled pp-prop learner exposing ``reset_state(batch_size=1)``.
    optimizer : braintools.optim.Adam
        Registered plain Adam optimizer with no weight decay or schedulers.
    base_parameters : ParameterSnapshot
        Immutable shared pretrained parameter values.
    adapt_fold : callable
        ``adapt_fold(learner, optimizer, fold)`` performs one update and returns
        a bounded diagnostic pytree.
    query_step : callable
        ``query_step(event, advance)`` advances the model once.
    checkpoint_output : callable
        Reads one fixed-shape selected output from current model state.
    checkpoint_output_shape : tuple of int
        Static shape returned by ``checkpoint_output``.
    checkpoint_output_dtype : dtype-like
        Static JAX dtype returned by ``checkpoint_output``.

    Returns
    -------
    callable
        Function accepting :class:`TargetFreeTaskBank` and returning
        :class:`TaskBankAdaptationResult`. The live model, optimizer, and
        learner are restored even when tracing or execution fails.

    Raises
    ------
    TypeError
        If the base snapshot, callbacks, or checkpoint shape are invalid.
    ValueError
        If the model is not batch size one or the optimizer is not a fresh-state
        compatible plain Adam configuration.
    """
    if not isinstance(base_parameters, ParameterSnapshot):
        raise TypeError("base_parameters must be a ParameterSnapshot. Set base_parameters to a ParameterSnapshot.")
    if any(
        not callable(value) for value in (adapt_fold, query_step, checkpoint_output)
    ):
        raise TypeError(
            "adapt_fold, query_step, and checkpoint_output must be callable. Pass a callable value for adapt_fold, query_step, and checkpoint_output."
        )
    if not isinstance(checkpoint_output_shape, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in checkpoint_output_shape
    ):
        raise TypeError("checkpoint_output_shape must be a tuple of non-negative ints. Set checkpoint_output_shape to a tuple of non-negative ints.")
    batch_size = getattr(getattr(model, "config", None), "batch_size", None)
    if batch_size != 1:
        raise ValueError("Compiled task-local adaptation requires model batch_size=1. Provide the required value for Compiled task-local adaptation.")
    if type(optimizer).__name__ != "Adam":
        raise ValueError(
            "Compiled task-local adaptation requires a plain Adam optimizer. Provide the required value for Compiled task-local adaptation."
        )
    if float(getattr(optimizer, "weight_decay", 0.0)) != 0.0:
        raise ValueError("Compiled task-local adaptation requires Adam weight_decay=0. Provide the required value for Compiled task-local adaptation.")
    if len(getattr(optimizer, "param_groups", ())) > 1:
        raise ValueError("Compiled task-local adaptation does not support Adam groups. Fix the input condition named in the error, then rerun the operation.")
    if getattr(optimizer, "_schedulers", ()):
        raise ValueError("Compiled task-local adaptation does not support schedulers. Fix the input condition named in the error, then rerun the operation.")
    missing_optimizer_state = next(
        (
            name
            for name in ("opt_state", "step_count", "_current_lr")
            if not hasattr(optimizer, name)
        ),
        None,
    )
    if missing_optimizer_state is not None:
        raise ValueError(
            f"Adam optimizer lacks required state {missing_optimizer_state}. Provide the missing item named in the message."
        )

    restore_parameters(model, base_parameters)
    parameter_bindings = tuple(
        map(
            lambda item: _ParameterBinding(item[1], _copy_tree(item[1].value)),
            _parameter_items(model),
        )
    )
    optimizer_base = _OptimizerSnapshot(
        opt_state=_copy_tree(optimizer.opt_state.value),
        step_count=_copy_leaf(optimizer.step_count.value),
        current_lr=_copy_leaf(optimizer._current_lr.value),
    )
    output_dtype = jnp.dtype(checkpoint_output_dtype)

    def restore_parameter_binding(binding: _ParameterBinding) -> None:
        binding.state.value = _copy_tree(binding.value)

    def restore_optimizer() -> None:
        optimizer.opt_state.value = _copy_tree(optimizer_base.opt_state)
        optimizer.step_count.value = _copy_leaf(optimizer_base.step_count)
        optimizer._current_lr.value = _copy_leaf(
            optimizer_base.current_lr
        )

    def reset_dynamics() -> None:
        model.reset_state()
        learner.reset_state(batch_size=1)

    def reset_task() -> None:
        jax.tree.map(
            restore_parameter_binding,
            parameter_bindings,
            is_leaf=lambda value: isinstance(value, _ParameterBinding),
        )
        restore_optimizer()
        reset_dynamics()

    def run_compiled(
        fold_inputs: Any,
        query_events: jax.Array,
        query_advances: jax.Array,
        query_valid: jax.Array,
        checkpoint_indices: jax.Array,
    ) -> TaskBankAdaptationResult:
        checkpoint_count = checkpoint_indices.shape[2]
        query_time = query_events.shape[2]
        time_indices = jnp.arange(query_time, dtype=jnp.int32)

        def run_task(inputs: tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array]):
            folds, events, advances, valid, selected = inputs
            reset_task()

            def adapt_one(fold: Any) -> Any:
                reset_dynamics()
                return adapt_fold(learner, optimizer, fold)

            fold_outputs = brainstate.transform.for_loop(adapt_one, folds)
            reset_dynamics()

            def run_query(
                query_inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array]:
                query_event, query_advance, is_valid, query_indices = query_inputs
                reset_dynamics()
                initial_buffer = jnp.zeros(
                    (checkpoint_count, *checkpoint_output_shape), dtype=output_dtype
                )
                initial_recorded = jnp.zeros((checkpoint_count,), dtype=jnp.bool_)
                initial_carry = (
                    initial_buffer,
                    initial_recorded,
                    jnp.asarray(0, dtype=jnp.int32),
                )

                def query_time_step(
                    carry: tuple[jax.Array, jax.Array, jax.Array],
                    step_inputs: tuple[jax.Array, jax.Array, jax.Array],
                ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
                    buffer, recorded, cursor = carry
                    event, advance, time_index = step_inputs
                    query_step(event, advance & is_valid)
                    value = jnp.asarray(checkpoint_output(), dtype=output_dtype)
                    if value.shape != checkpoint_output_shape:
                        raise ValueError(
                            "checkpoint_output returned a value with the wrong shape. Fix the input condition named in the error, then rerun the operation."
                        )
                    safe_cursor = jnp.minimum(cursor, checkpoint_count - 1)
                    matched = (
                        is_valid
                        & (cursor < checkpoint_count)
                        & (time_index == query_indices[safe_cursor])
                    )
                    previous = buffer[safe_cursor]
                    buffer = buffer.at[safe_cursor].set(
                        jnp.where(matched, value, previous)
                    )
                    recorded = recorded.at[safe_cursor].set(
                        recorded[safe_cursor] | matched
                    )
                    cursor = cursor + matched.astype(jnp.int32)
                    return (buffer, recorded, cursor), jnp.asarray(0, dtype=jnp.uint8)

                final_carry, _ = brainstate.transform.scan(
                    query_time_step,
                    initial_carry,
                    (query_event, query_advance, time_indices),
                )
                return final_carry[0], final_carry[1]

            checkpoint_outputs, checkpoint_recorded = brainstate.transform.for_loop(
                run_query,
                (events, advances, valid, selected),
            )
            reset_task()
            return fold_outputs, checkpoint_outputs, checkpoint_recorded

        fold_outputs, checkpoint_outputs, checkpoint_recorded = (
            brainstate.transform.for_loop(
                run_task,
                (
                    fold_inputs,
                    query_events,
                    query_advances,
                    query_valid,
                    checkpoint_indices,
                ),
            )
        )
        reset_task()
        return TaskBankAdaptationResult(
            fold_outputs=fold_outputs,
            checkpoint_outputs=checkpoint_outputs,
            checkpoint_recorded=checkpoint_recorded,
            query_valid=query_valid,
        )

    compiled_task_bank = brainstate.transform.jit(run_compiled)

    def run(bank: TargetFreeTaskBank) -> TaskBankAdaptationResult:
        if not isinstance(bank, TargetFreeTaskBank):
            raise TypeError("Bank must be a TargetFreeTaskBank. Set Bank to a TargetFreeTaskBank.")
        try:
            return compiled_task_bank(
                bank.fold_inputs,
                bank.query_events,
                bank.query_advances,
                bank.query_valid,
                bank.checkpoint_indices,
            )
        finally:
            restore_parameters(model, base_parameters)
            restore_optimizer()
            reset_dynamics()

    return run
