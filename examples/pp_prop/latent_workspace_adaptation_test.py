"""Tests for task-local ARC parameter adaptation and isolation."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from typing import Any
from types import SimpleNamespace

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from examples.pp_prop.latent_workspace_adaptation import (
        ParameterRecord,
        ParameterSnapshot,
        TargetFreeTaskBank,
        TargetFreeQuery,
        build_target_free_task_bank,
        compile_task_local_adaptation_runner,
        restore_parameters,
        run_task_local_adaptation,
        snapshot_parameters,
    )
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
    )
except ImportError:
    from latent_workspace_adaptation import (
        ParameterRecord,
        ParameterSnapshot,
        TargetFreeTaskBank,
        TargetFreeQuery,
        build_target_free_task_bank,
        compile_task_local_adaptation_runner,
        restore_parameters,
        run_task_local_adaptation,
        snapshot_parameters,
    )
    from latent_workspace_model import LatentWorkspaceModel, ModelConfig


class _ToyModel(brainstate.nn.Module):
    def __init__(self) -> None:
        self.weight = brainstate.ParamState(jnp.asarray(2.0, dtype=jnp.float32))
        self.nested = brainstate.ParamState(
            {
                "left": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
                "right": (jnp.asarray(3.0, dtype=jnp.float32),),
            }
        )
        self.hidden = brainstate.HiddenState(jnp.asarray(0.0, dtype=jnp.float32))

    def reset_state(self, **_: object) -> None:
        self.hidden.value = jnp.asarray(0.0, dtype=jnp.float32)


def _record(snapshot: ParameterSnapshot, path: tuple[Any, ...]) -> ParameterRecord:
    return next(record for record in snapshot.records if record.path == path)


def _snapshot_arrays(
    snapshot: ParameterSnapshot,
) -> dict[tuple[Any, ...], tuple[np.ndarray, ...]]:
    return {
        record.path: tuple(np.asarray(leaf) for leaf in record.leaves)
        for record in snapshot.records
    }


def _assert_snapshots_equal(left: ParameterSnapshot, right: ParameterSnapshot) -> None:
    left_arrays = _snapshot_arrays(left)
    right_arrays = _snapshot_arrays(right)
    assert left_arrays.keys() == right_arrays.keys()
    for path, leaves in left_arrays.items():
        for left_leaf, right_leaf in zip(leaves, right_arrays[path], strict=True):
            np.testing.assert_array_equal(left_leaf, right_leaf)


def test_parameter_snapshot_is_immutable_and_does_not_alias_model() -> None:
    model = _ToyModel()

    snapshot = snapshot_parameters(model)
    model.weight.value = jnp.asarray(91.0, dtype=jnp.float32)
    model.nested.value["left"] = jnp.asarray([-1.0, -2.0], dtype=jnp.float32)

    assert dataclasses.is_dataclass(snapshot)
    assert snapshot.__dataclass_params__.frozen is True
    assert snapshot.records == tuple(snapshot.records)
    np.testing.assert_array_equal(
        np.asarray(_record(snapshot, ("weight",)).leaves[0]),
        np.asarray(2.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(_record(snapshot, ("nested",)).leaves[0]),
        np.asarray([1.0, 2.0], dtype=np.float32),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.records = ()  # type: ignore[misc]


def test_restore_parameters_round_trips_real_latent_workspace_model() -> None:
    with brainstate.random.seed_context(7):
        model = LatentWorkspaceModel(
            ModelConfig(
                input_width=6,
                neuron_count=64,
                recurrent_edges=96,
                max_latent_steps=4,
                readout_width=8,
                color_rank=2,
                seed=7,
            )
        )
    baseline = snapshot_parameters(model)
    first_path, first_state = next(iter(model.states(brainstate.ParamState).items()))
    first_state.value = jax.tree.map(
        lambda leaf: jnp.asarray(leaf) + 1,
        first_state.value,
    )
    assert first_path == baseline.records[0].path

    restore_parameters(model, baseline)

    _assert_snapshots_equal(snapshot_parameters(model), baseline)


@pytest.mark.parametrize("fault", ["path", "shape", "structure", "dtype"])
def test_restore_parameters_rejects_incompatible_snapshot_transactionally(
    fault: str,
) -> None:
    model = _ToyModel()
    baseline = snapshot_parameters(model)
    model.weight.value = jnp.asarray(11.0, dtype=jnp.float32)
    before_failure = snapshot_parameters(model)
    records = list(baseline.records)
    record_index = next(
        index for index, item in enumerate(records) if item.path == ("weight",)
    )
    record = records[record_index]
    if fault == "path":
        records[record_index] = dataclasses.replace(record, path=("unknown",))
    elif fault == "shape":
        records[record_index] = dataclasses.replace(
            record,
            leaves=(jnp.zeros((2,), dtype=record.leaves[0].dtype),),
        )
    elif fault == "structure":
        records[record_index] = dataclasses.replace(
            record,
            tree_structure=jax.tree.structure({"unexpected": record.leaves[0]}),
        )
    else:
        records[record_index] = dataclasses.replace(
            record,
            leaves=(record.leaves[0].astype(jnp.int32),),
        )
    incompatible = ParameterSnapshot(records=tuple(records))

    with pytest.raises(ValueError, match="parameter"):
        restore_parameters(model, incompatible)

    _assert_snapshots_equal(snapshot_parameters(model), before_failure)


@pytest.mark.parametrize(
    ("events", "advances", "message"),
    [
        (jnp.zeros((0, 2)), jnp.zeros((0,), dtype=jnp.bool_), "at least one"),
        (jnp.asarray(1.0), jnp.asarray([True]), "leading step axis"),
        (jnp.zeros((2, 2)), jnp.zeros((2, 1), dtype=jnp.bool_), "one-dimensional"),
        (jnp.zeros((2, 2)), jnp.zeros((1,), dtype=jnp.bool_), "same step count"),
        (jnp.zeros((1, 2)), jnp.ones((1,), dtype=jnp.int32), "boolean dtype"),
    ],
)
def test_target_free_query_fails_closed_on_invalid_sequence(
    events: jax.Array, advances: jax.Array, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TargetFreeQuery(events=events, advances=advances)


def test_target_free_query_has_no_target_boundary() -> None:
    fields = tuple(field.name for field in dataclasses.fields(TargetFreeQuery))
    assert fields == ("events", "advances")
    with pytest.raises(TypeError):
        TargetFreeQuery(  # type: ignore[call-arg]
            events=jnp.ones((1, 1)),
            advances=jnp.ones((1,), dtype=jnp.bool_),
            target=jnp.asarray(9),
        )


def test_restore_parameters_rejects_wrong_snapshot_type() -> None:
    with pytest.raises(TypeError, match="ParameterSnapshot"):
        restore_parameters(_ToyModel(), object())  # type: ignore[arg-type]


def test_task_local_runner_resets_folds_and_discards_adapted_parameters() -> None:
    model = _ToyModel()
    base = snapshot_parameters(model)
    optimizer_ids: list[int] = []

    def make_optimizer() -> braintools.optim.Adam:
        optimizer = braintools.optim.Adam(lr=0.1)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        optimizer_ids.append(id(optimizer))
        return optimizer

    def adapt_fold(optimizer: braintools.optim.Adam, fold: jax.Array) -> jax.Array:
        hidden_at_start = model.hidden.value
        model.hidden.value = model.hidden.value + fold
        optimizer.update(
            {
                ("nested",): {
                    "left": jnp.zeros((2,), dtype=jnp.float32),
                    "right": (jnp.asarray(0.0, dtype=jnp.float32),),
                },
                ("weight",): jnp.asarray(-1.0, dtype=jnp.float32),
            }
        )
        optimizer_count = optimizer.opt_state.value[0].count
        return jnp.stack((hidden_at_start, optimizer_count.astype(jnp.float32)))

    def query_step(event: jax.Array, advance: jax.Array) -> None:
        next_hidden = model.hidden.value + event * model.weight.value
        model.hidden.value = jnp.where(advance, next_hidden, model.hidden.value)

    def finish_query() -> jax.Array:
        return jnp.stack((model.hidden.value, model.weight.value))

    query = TargetFreeQuery(
        events=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        advances=jnp.asarray([True, True]),
    )
    first = run_task_local_adaptation(
        model,
        base_parameters=base,
        fold_inputs=jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float32),
        query=query,
        make_optimizer=make_optimizer,
        adapt_fold=adapt_fold,
        query_step=query_step,
        finish_query=finish_query,
        capture_adapted_parameters=True,
    )
    second = run_task_local_adaptation(
        model,
        base_parameters=base,
        fold_inputs=jnp.asarray([20.0, 40.0, 80.0], dtype=jnp.float32),
        query=query,
        make_optimizer=make_optimizer,
        adapt_fold=adapt_fold,
        query_step=query_step,
        finish_query=finish_query,
    )

    np.testing.assert_array_equal(
        np.asarray(first.fold_outputs[:, 0]), np.zeros((3,), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(second.fold_outputs[:, 0]), np.zeros((3,), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(first.fold_outputs[:, 1]), np.asarray([1.0, 2.0, 3.0])
    )
    np.testing.assert_array_equal(
        np.asarray(second.fold_outputs[:, 1]), np.asarray([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(
        np.asarray(first.prediction), np.asarray([6.9, 2.3]), atol=1e-5
    )
    np.testing.assert_array_equal(first.prediction, second.prediction)
    assert first.adapted_parameters is not None
    assert second.adapted_parameters is None
    np.testing.assert_allclose(
        np.asarray(_record(first.adapted_parameters, ("weight",)).leaves[0]),
        np.asarray(2.3),
        atol=1e-5,
    )
    assert len(set(optimizer_ids)) == 2
    assert float(model.hidden.value) == 0.0
    _assert_snapshots_equal(snapshot_parameters(model), base)


def test_task_local_runner_prediction_does_not_accept_out_of_band_target() -> None:
    model = _ToyModel()
    base = snapshot_parameters(model)

    def make_optimizer() -> braintools.optim.Adam:
        optimizer = braintools.optim.Adam(lr=0.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        return optimizer

    def adapt_fold(_: braintools.optim.Adam, fold: jax.Array) -> jax.Array:
        return fold

    def query_step(event: jax.Array, advance: jax.Array) -> None:
        model.hidden.value = jnp.where(
            advance, model.hidden.value + event, model.hidden.value
        )

    query = TargetFreeQuery(
        events=jnp.asarray([1.0, 3.0]),
        advances=jnp.asarray([True, True]),
    )

    def predict_with_unseen_target(_target: jax.Array) -> jax.Array:
        result = run_task_local_adaptation(
            model,
            base_parameters=base,
            fold_inputs=jnp.asarray([0.0]),
            query=query,
            make_optimizer=make_optimizer,
            adapt_fold=adapt_fold,
            query_step=query_step,
            finish_query=lambda: jnp.asarray(model.hidden.value),
        )
        return result.prediction

    first = predict_with_unseen_target(jnp.asarray(1))
    second = predict_with_unseen_target(jnp.asarray(9))

    np.testing.assert_array_equal(first, second)


def test_task_local_runner_restores_shared_state_after_failure() -> None:
    model = _ToyModel()
    base = snapshot_parameters(model)

    def make_optimizer() -> object:
        return object()

    def fail_after_mutation(_: object, fold: jax.Array) -> jax.Array:
        model.weight.value = model.weight.value + fold
        model.hidden.value = model.hidden.value + fold
        raise RuntimeError("synthetic adaptation failure")

    with pytest.raises(RuntimeError, match="synthetic adaptation failure"):
        run_task_local_adaptation(
            model,
            base_parameters=base,
            fold_inputs=jnp.asarray([1.0]),
            query=TargetFreeQuery(
                events=jnp.asarray([1.0]),
                advances=jnp.asarray([True]),
            ),
            make_optimizer=make_optimizer,
            adapt_fold=fail_after_mutation,
            query_step=lambda _event, _advance: None,
            finish_query=lambda: jnp.asarray(0.0),
        )

    assert float(model.hidden.value) == 0.0
    _assert_snapshots_equal(snapshot_parameters(model), base)


@pytest.mark.parametrize(
    "fold_inputs",
    [(), jnp.asarray(1.0), jnp.asarray([]), (jnp.ones((2,)), jnp.ones((3,)))],
)
def test_task_local_runner_rejects_invalid_fold_batches(fold_inputs: object) -> None:
    model = _ToyModel()
    with pytest.raises(ValueError, match="fold"):
        run_task_local_adaptation(
            model,
            base_parameters=snapshot_parameters(model),
            fold_inputs=fold_inputs,
            query=TargetFreeQuery(
                events=jnp.asarray([1.0]),
                advances=jnp.asarray([True]),
            ),
            make_optimizer=object,
            adapt_fold=lambda _optimizer, fold: fold,
            query_step=lambda _event, _advance: None,
            finish_query=lambda: jnp.asarray(0.0),
        )


def test_task_local_runner_rejects_non_target_free_query() -> None:
    model = _ToyModel()
    with pytest.raises(TypeError, match="TargetFreeQuery"):
        run_task_local_adaptation(
            model,
            base_parameters=snapshot_parameters(model),
            fold_inputs=jnp.asarray([1.0]),
            query=object(),  # type: ignore[arg-type]
            make_optimizer=object,
            adapt_fold=lambda _optimizer, fold: fold,
            query_step=lambda _event, _advance: None,
            finish_query=lambda: jnp.asarray(0.0),
        )


def test_runner_source_uses_compiled_loops_without_python_model_loops() -> None:
    tree = ast.parse(inspect.getsource(run_task_local_adaptation))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "for_loop"
    ]
    assert len(calls) == 2


class _ToyLearner:
    def __init__(self) -> None:
        self.trace = brainstate.ShortTermState(jnp.asarray(99.0, dtype=jnp.float32))

    def reset_state(self, batch_size: int | None = None, **_: object) -> None:
        assert batch_size == 1
        self.trace.value = jnp.asarray(0.0, dtype=jnp.float32)


def _compiled_toy_runner():
    model = _ToyModel()
    model.config = SimpleNamespace(batch_size=1)
    learner = _ToyLearner()
    optimizer = braintools.optim.Adam(lr=0.1)
    optimizer.register_trainable_weights({("weight",): model.weight})
    base = snapshot_parameters(model)

    def adapt_fold(
        active_learner: _ToyLearner,
        active_optimizer: braintools.optim.Adam,
        fold: jax.Array,
    ) -> jax.Array:
        starts = jnp.stack(
            (
                model.hidden.value,
                active_learner.trace.value,
                active_optimizer.step_count.value.astype(jnp.float32),
            )
        )
        model.hidden.value = model.hidden.value + fold
        active_learner.trace.value = active_learner.trace.value + 1.0
        active_optimizer.update({("weight",): -jnp.asarray(fold)})
        return starts

    def query_step(event: jax.Array, advance: jax.Array) -> None:
        updated = model.hidden.value + event * model.weight.value
        model.hidden.value = jnp.where(advance, updated, model.hidden.value)

    runner = compile_task_local_adaptation_runner(
        model,
        learner,
        optimizer,
        base_parameters=base,
        adapt_fold=adapt_fold,
        query_step=query_step,
        checkpoint_output=lambda: jnp.stack((model.hidden.value, model.weight.value)),
        checkpoint_output_shape=(2,),
        checkpoint_output_dtype=jnp.float32,
    )
    return model, learner, optimizer, base, runner


def _toy_task_bank() -> TargetFreeTaskBank:
    return build_target_free_task_bank(
        fold_inputs=jnp.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float32),
        query_events=jnp.asarray(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 1.0, 0.0], [9.0, 9.0, 9.0]],
            ],
            dtype=jnp.float32,
        ),
        query_advances=jnp.ones((2, 2, 3), dtype=jnp.bool_),
        query_valid=jnp.asarray([[True, True], [True, False]]),
        checkpoint_indices=jnp.asarray(
            [
                [[0, 2], [0, 2]],
                [[0, 2], [0, 2]],
            ],
            dtype=jnp.int32,
        ),
    )


def test_target_free_task_bank_has_no_official_target_boundary() -> None:
    fields = TargetFreeTaskBank._fields
    assert fields == (
        "fold_inputs",
        "query_events",
        "query_advances",
        "query_valid",
        "checkpoint_indices",
    )
    bank = _toy_task_bank()
    assert bank.query_events.shape == (2, 2, 3)
    with pytest.raises(TypeError):
        TargetFreeTaskBank(  # type: ignore[call-arg]
            fold_inputs=bank.fold_inputs,
            query_events=bank.query_events,
            query_advances=bank.query_advances,
            query_valid=bank.query_valid,
            checkpoint_indices=bank.checkpoint_indices,
            target=jnp.asarray(1),
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"query_events": jnp.ones((2, 2))}, "task, query, and time axes"),
        ({"query_events": jnp.ones((0, 2, 3))}, "counts must be positive"),
        ({"fold_inputs": jnp.ones((3, 2))}, "task count"),
        ({"query_advances": jnp.ones((2, 2, 2), dtype=jnp.bool_)}, "shape"),
        ({"query_advances": jnp.ones((2, 2, 3), dtype=jnp.int32)}, "boolean"),
        ({"query_valid": jnp.ones((2, 2, 1), dtype=jnp.bool_)}, "two-dimensional"),
        ({"query_valid": jnp.ones((2, 1), dtype=jnp.bool_)}, "shape"),
        ({"query_valid": jnp.ones((2, 2), dtype=jnp.int32)}, "boolean"),
        (
            {"checkpoint_indices": jnp.zeros((2, 2), dtype=jnp.int32)},
            "three-dimensional",
        ),
        (
            {"checkpoint_indices": jnp.zeros((2, 1, 2), dtype=jnp.int32)},
            "match task/query axes",
        ),
        (
            {"checkpoint_indices": jnp.zeros((2, 2, 2), dtype=jnp.float32)},
            "integer dtype",
        ),
        (
            {"checkpoint_indices": jnp.asarray([[[0, 3], [0, 2]]] * 2)},
            "outside query time",
        ),
        (
            {"checkpoint_indices": jnp.asarray([[[0, 0], [0, 2]]] * 2)},
            "strictly increasing",
        ),
        ({"fold_inputs": ()}, "at least one array"),
        ({"fold_inputs": jnp.asarray([1.0, 2.0])}, "task and update axes"),
        ({"fold_inputs": jnp.ones((2, 0))}, "at least one update"),
        (
            {"fold_inputs": (jnp.ones((2, 2)), jnp.ones((2, 3)))},
            "same update count",
        ),
    ],
)
def test_target_free_task_bank_rejects_inconsistent_static_shapes(
    replacement: dict[str, Any], message: str
) -> None:
    bank = _toy_task_bank()._asdict()
    bank.update(replacement)
    with pytest.raises(ValueError, match=message):
        build_target_free_task_bank(**bank)


def test_compiled_task_runner_resets_every_state_and_bounds_outputs() -> None:
    model, learner, optimizer, base, runner = _compiled_toy_runner()

    first = runner(_toy_task_bank())
    second = runner(_toy_task_bank())

    assert first.fold_outputs.shape == (2, 2, 3)
    np.testing.assert_array_equal(first.fold_outputs[..., 0], 0.0)
    np.testing.assert_array_equal(first.fold_outputs[..., 1], 0.0)
    np.testing.assert_array_equal(
        first.fold_outputs[..., 2], np.asarray([[0.0, 1.0], [0.0, 1.0]])
    )
    assert first.checkpoint_outputs.shape == (2, 2, 2, 2)
    assert first.checkpoint_recorded.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        first.checkpoint_recorded,
        np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ]
        ),
    )
    np.testing.assert_array_equal(
        first.checkpoint_outputs[1, 1], np.zeros((2, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(first.fold_outputs, second.fold_outputs)
    np.testing.assert_array_equal(first.checkpoint_outputs, second.checkpoint_outputs)
    np.testing.assert_array_equal(first.query_valid, _toy_task_bank().query_valid)
    assert float(model.hidden.value) == 0.0
    assert float(learner.trace.value) == 0.0
    assert int(optimizer.step_count.value) == 0
    _assert_snapshots_equal(snapshot_parameters(model), base)


def test_compiled_task_runner_restores_everything_after_callback_failure() -> None:
    model = _ToyModel()
    model.config = SimpleNamespace(batch_size=1)
    learner = _ToyLearner()
    optimizer = braintools.optim.Adam(lr=0.1)
    optimizer.register_trainable_weights({("weight",): model.weight})
    base = snapshot_parameters(model)

    def fail(
        _learner: _ToyLearner,
        active_optimizer: braintools.optim.Adam,
        fold: jax.Array,
    ) -> jax.Array:
        model.hidden.value = fold
        active_optimizer.update({("weight",): -fold})
        raise RuntimeError("compiled synthetic failure")

    runner = compile_task_local_adaptation_runner(
        model,
        learner,
        optimizer,
        base_parameters=base,
        adapt_fold=fail,
        query_step=lambda _event, _advance: None,
        checkpoint_output=lambda: jnp.zeros((1,), dtype=jnp.float32),
        checkpoint_output_shape=(1,),
        checkpoint_output_dtype=jnp.float32,
    )

    with pytest.raises(RuntimeError, match="compiled synthetic failure"):
        runner(_toy_task_bank())

    assert float(model.hidden.value) == 0.0
    assert int(optimizer.step_count.value) == 0
    _assert_snapshots_equal(snapshot_parameters(model), base)


def test_compiled_runner_requires_batch_one_and_plain_adam() -> None:
    model = _ToyModel()
    model.config = SimpleNamespace(batch_size=2)
    learner = _ToyLearner()
    optimizer = braintools.optim.Adam(lr=0.1, weight_decay=0.1)
    optimizer.register_trainable_weights({("weight",): model.weight})
    arguments = dict(
        base_parameters=snapshot_parameters(model),
        adapt_fold=lambda _learner, _optimizer, fold: fold,
        query_step=lambda _event, _advance: None,
        checkpoint_output=lambda: jnp.zeros((1,), dtype=jnp.float32),
        checkpoint_output_shape=(1,),
        checkpoint_output_dtype=jnp.float32,
    )
    with pytest.raises(ValueError, match="batch_size=1"):
        compile_task_local_adaptation_runner(model, learner, optimizer, **arguments)
    model.config = SimpleNamespace(batch_size=1)
    with pytest.raises(ValueError, match="weight_decay=0"):
        compile_task_local_adaptation_runner(model, learner, optimizer, **arguments)


def test_compiled_runner_source_uses_nested_brainstate_transforms() -> None:
    source = inspect.getsource(compile_task_local_adaptation_runner)
    tree = ast.parse(source)
    repeated_model_loops = [
        node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))
    ]
    transform_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"for_loop", "scan", "jit"}
    ]

    assert repeated_model_loops == []
    assert transform_calls.count("for_loop") >= 3
    assert "scan" in transform_calls
    assert "jit" in transform_calls
