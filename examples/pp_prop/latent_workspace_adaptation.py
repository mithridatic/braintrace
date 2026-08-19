"""Task-local parameter isolation for Example 21 ARC adaptation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
            raise ValueError("events must have a leading step axis")
        if advances.ndim != 1:
            raise ValueError("advances must be one-dimensional")
        if events.shape[0] == 0:
            raise ValueError("query must contain at least one step")
        if events.shape[0] != advances.shape[0]:
            raise ValueError("events and advances must have the same step count")
        if not np.issubdtype(advances.dtype, np.bool_):
            raise ValueError("advances must have boolean dtype")
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


def _copy_leaf(leaf: Any) -> jax.Array:
    return jnp.array(leaf, copy=True)


def _copy_tree(value: Any) -> Any:
    return jax.tree.map(_copy_leaf, value)


def _parameter_items(model: Any) -> tuple[tuple[tuple[Any, ...], Any], ...]:
    try:
        states = model.states(brainstate.ParamState)
    except AttributeError as error:
        raise TypeError("model must expose BrainState ParamState objects") from error
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
        raise TypeError("snapshot must be a ParameterSnapshot")
    parameter_items = _parameter_items(model)
    expected_paths = tuple(path for path, _ in parameter_items)
    actual_paths = tuple(record.path for record in snapshot.records)
    if actual_paths != expected_paths:
        raise ValueError("parameter paths do not match this model")
    validated: list[tuple[Any, Any]] = []
    for record, (_, state) in zip(snapshot.records, parameter_items, strict=True):
        state_leaves, state_structure = jax.tree.flatten(state.value)
        if record.tree_structure != state_structure:
            raise ValueError(f"parameter structure does not match at {record.path}")
        if len(record.leaves) != len(state_leaves):
            raise ValueError(f"parameter leaf count does not match at {record.path}")
        for snapshot_leaf, state_leaf in zip(record.leaves, state_leaves, strict=True):
            if np.shape(snapshot_leaf) != np.shape(state_leaf):
                raise ValueError(f"parameter shape does not match at {record.path}")
            if np.dtype(snapshot_leaf.dtype) != np.dtype(state_leaf.dtype):
                raise ValueError(f"parameter dtype does not match at {record.path}")
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
        raise ValueError("fold inputs must contain at least one array")
    counts: list[int] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if array.ndim < 1:
            raise ValueError("every fold input must have a leading fold axis")
        counts.append(int(array.shape[0]))
    if not counts or counts[0] == 0:
        raise ValueError("fold inputs must contain at least one fold")
    if any(count != counts[0] for count in counts[1:]):
        raise ValueError("all fold inputs must have the same fold count")
    return counts[0]


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
        raise TypeError("query must be a TargetFreeQuery")
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
