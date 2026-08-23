"""Synthetic end-to-end qualification for learned ARC row binding.

The qualification deliberately uses ordinary ARC targets only as loss and
scoring labels.  Every reported candidate is decoded from the row-refinement
model's explicit logits; no symbolic or task-specific proposal path exists in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

try:
    from examples.pp_prop.latent_workspace_analysis import (
        OutputLogits,
        decode_candidates,
    )
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        compile_pp_prop,
        parameter_snapshot,
        run_selected_packed_stream,
    )
    from examples.pp_prop.latent_workspace_refinement import (
        RowRefinementLayout,
        row_refinement_loss_per_example,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        associative_memory_feature_indices,
        encode_query_episode,
    )
except ImportError:
    from latent_workspace_analysis import OutputLogits, decode_candidates
    from latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        compile_pp_prop,
        parameter_snapshot,
        run_selected_packed_stream,
    )
    from latent_workspace_refinement import (
        RowRefinementLayout,
        row_refinement_loss_per_example,
    )
    from latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        associative_memory_feature_indices,
        encode_query_episode,
    )


_REFINEMENT_STEPS = 30


class BindingQualificationError(RuntimeError):
    """Raised when the strict synthetic row-binding gate does not qualify."""


@dataclass(frozen=True, slots=True)
class SyntheticBindingCase:
    """Name one deterministic ARC task used by the qualification.

    Parameters
    ----------
    name
        Stable family name used in the report.
    Task
        Two-demonstration ARC task with one scored query.
    """

    name: str
    task: ArcTask

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SyntheticBindingCase.name must be non-empty. Provide at least one value for SyntheticBindingCase.name.")
        if not isinstance(self.task, ArcTask):
            raise TypeError("SyntheticBindingCase.task must be an ArcTask. Set SyntheticBindingCase.task to an ArcTask.")


@dataclass(frozen=True, slots=True)
class BindingQualificationConfig:
    """Configure a bounded real-model row-binding diagnostic.

    Parameters
    ----------
    updates
        Number of compiled pp-prop optimizer updates.
    learning_rate
        Positive Adam learning rate.
    clip_norm
        Positive gradient clipping norm.
    neuron_count
        Physical LIF neuron count, divisible by 64.
    recurrent_edges
        Positive count of real sparse recurrent edges.
    readout_width
        Positive learned readout bottleneck width.
    context_memory_width
        Positive associative workspace width.
    Seed
        Non-negative deterministic initialization seed.
    """

    updates: int = 8
    learning_rate: float = 0.01
    clip_norm: float = 1.0
    neuron_count: int = 64
    recurrent_edges: int = 64
    readout_width: int = 8
    context_memory_width: int = 2
    seed: int = 2108

    def __post_init__(self) -> None:
        for name in (
            "updates",
            "neuron_count",
            "recurrent_edges",
            "readout_width",
            "context_memory_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a positive integer. Set {name} to a positive integer.")
            value = int(value)
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer. Set {name} to a positive integer.")
            object.__setattr__(self, name, value)
        if self.neuron_count % 64:
            raise ValueError("neuron_count must be divisible by 64. Set neuron_count to a value divisible by 64.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("Seed must be a non-negative integer. Set Seed to a non-negative integer.")
        if int(self.seed) < 0:
            raise ValueError("Seed must be a non-negative integer. Set Seed to a non-negative integer.")
        object.__setattr__(self, "seed", int(self.seed))
        for name in ("learning_rate", "clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a positive finite real. Set {name} to a positive finite real.")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite real. Set {name} to a positive finite real.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class _BindingBatch:
    events: np.ndarray
    advances: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    target_colors: np.ndarray
    loss_mask: np.ndarray
    selected_indices: np.ndarray


def _grid(*rows: tuple[int, ...]) -> ArcGrid:
    return ArcGrid(tuple(rows))


def _pair(input_grid: ArcGrid, output_grid: ArcGrid) -> ArcPair:
    return ArcPair(input_grid, output_grid)


def synthetic_binding_tasks() -> tuple[SyntheticBindingCase, ...]:
    """Return tiny deterministic ARC families with distinct binding demands.

    Returns
    -------
    tuple of SyntheticBindingCase
        Identity, recoloring, non-square transpose, and context-defined row
        reversal tasks.  Each has two demonstrations and one scored query.
    """

    identity = ArcTask(
        train=(
            _pair(_grid((1,)), _grid((1,))),
            _pair(_grid((2, 0), (0, 2)), _grid((2, 0), (0, 2))),
        ),
        test=(_pair(_grid((3, 3), (0, 3)), _grid((3, 3), (0, 3))),),
        task_id="synthetic-identity",
    )
    recolor = ArcTask(
        train=(
            _pair(_grid((1,)), _grid((4,))),
            _pair(_grid((1, 0), (0, 1)), _grid((4, 0), (0, 4))),
        ),
        test=(_pair(_grid((1, 0), (0, 1)), _grid((4, 0), (0, 4))),),
        task_id="synthetic-recolor",
    )
    transpose = ArcTask(
        train=(
            _pair(_grid((1, 2)), _grid((1,), (2,))),
            _pair(
                _grid((3, 0, 4), (5, 6, 0)),
                _grid((3, 5), (0, 6), (4, 0)),
            ),
        ),
        test=(
            _pair(
                _grid((1, 2, 3), (4, 5, 6)),
                _grid((1, 4), (2, 5), (3, 6)),
            ),
        ),
        task_id="synthetic-non-square-transpose",
    )
    row_reverse = ArcTask(
        train=(
            _pair(_grid((1, 2), (3, 4)), _grid((3, 4), (1, 2))),
            _pair(
                _grid((5, 0), (6, 7), (8, 9)),
                _grid((8, 9), (6, 7), (5, 0)),
            ),
        ),
        test=(
            _pair(
                _grid((2, 3), (6, 7)),
                _grid((6, 7), (2, 3)),
            ),
        ),
        task_id="synthetic-demo-dependent-row-reverse",
    )
    return (
        SyntheticBindingCase("identity", identity),
        SyntheticBindingCase("recolor", recolor),
        SyntheticBindingCase("non_square_transpose", transpose),
        SyntheticBindingCase("demo_dependent_row_reverse", row_reverse),
    )


def _row_layout(rows: RowEventConfig) -> RowRefinementLayout:
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


def _model_config(
    config: BindingQualificationConfig,
    rows: RowEventConfig,
    *,
    batch_size: int,
) -> ModelConfig:
    memory = associative_memory_feature_indices(rows)
    return ModelConfig(
        input_width=rows.input_width,
        batch_size=batch_size,
        neuron_count=config.neuron_count,
        recurrent_edges=config.recurrent_edges,
        max_latent_steps=_REFINEMENT_STEPS,
        readout_width=config.readout_width,
        color_rank=2,
        context_memory_width=config.context_memory_width,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        memory_key_indices=memory.key_indices,
        memory_value_indices=memory.value_indices,
        decoder_mode="row_refinement",
        refinement_steps=_REFINEMENT_STEPS,
        refinement_layout=_row_layout(rows),
        event_valid_index=rows.valid_slice.start,
        seed=config.seed,
    )


def _binding_batch(
    cases: tuple[SyntheticBindingCase, ...], rows: RowEventConfig
) -> _BindingBatch:
    batch_size = len(cases)
    time_count = rows.max_events + _REFINEMENT_STEPS
    events = np.zeros((time_count, batch_size, rows.input_width), dtype=np.float32)
    advances = np.zeros((time_count, batch_size), dtype=np.bool_)
    target_heights = np.zeros((batch_size,), dtype=np.int32)
    target_widths = np.zeros((batch_size,), dtype=np.int32)
    target_colors = np.zeros((batch_size, 30, 30), dtype=np.int32)
    loss_mask = np.zeros((time_count,), dtype=np.float32)
    per_example_mask = np.zeros((time_count, batch_size), dtype=np.float32)
    selected_indices = np.zeros((1, batch_size), dtype=np.int32)

    for case_index, case in enumerate(cases):
        encoded = encode_query_episode(case.task, 0, rows, task_index=case_index)
        target = encoded.target
        if target is None:
            raise ValueError(f"Synthetic case {case.name} has no target. Provide the missing item named in the message.")
        events[: rows.max_events, case_index] = encoded.events
        advances[: rows.max_events, case_index] = (
            encoded.events[:, rows.valid_slice.start] > 0.5
        )
        latent_start = encoded.query_stop
        latent_stop = latent_start + _REFINEMENT_STEPS
        advances[latent_start:latent_stop, case_index] = True
        per_example_mask[latent_start:latent_stop, case_index] = 1.0
        selected_indices[0, case_index] = latent_stop - 1
        target_heights[case_index] = target.height - 1
        target_widths[case_index] = target.width - 1
        target_colors[case_index, : target.height, : target.width] = target.as_array()

    active_loss_ticks = np.any(per_example_mask > 0.0, axis=1)
    loss_mask[active_loss_ticks] = 1.0 / np.count_nonzero(active_loss_ticks)
    return _BindingBatch(
        events=events,
        advances=advances,
        target_heights=target_heights,
        target_widths=target_widths,
        target_colors=target_colors,
        loss_mask=loss_mask,
        selected_indices=selected_indices,
    )


def _evaluation_loss(model: LatentWorkspaceModel, batch: _BindingBatch) -> float:
    events = jnp.asarray(batch.events)
    advances = jnp.asarray(batch.advances)
    heights = jnp.asarray(batch.target_heights)
    widths = jnp.asarray(batch.target_widths)
    colors = jnp.asarray(batch.target_colors)
    mask = jnp.asarray(batch.loss_mask)

    @brainstate.transform.jit
    def evaluate() -> jax.Array:
        model.reset_state()

        def step(inputs: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            event, advance, weight = inputs
            logits = model.update(event, advance)
            row_indices = jnp.mod(
                jnp.asarray(model.reasoning_index.value, dtype=jnp.int32) - 1,
                _REFINEMENT_STEPS,
            )
            losses = row_refinement_loss_per_example(
                logits, heights, widths, colors, row_indices
            )
            latent = advance & ~(event[:, model.config.event_valid_index] > 0.5)
            active_count = jnp.maximum(jnp.sum(latent), 1)
            return jnp.sum(jnp.where(latent, losses, 0.0)) / active_count * weight

        losses = brainstate.transform.for_loop(step, (events, advances, mask))
        return jnp.sum(losses)

    return float(np.asarray(evaluate()))


def _compiled_training_losses(
    model: LatentWorkspaceModel,
    learner: Any,
    batch: _BindingBatch,
    config: BindingQualificationConfig,
) -> np.ndarray:
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(learner.param_states)
    events = jnp.asarray(batch.events)
    advances = jnp.asarray(batch.advances)
    heights = jnp.asarray(batch.target_heights)
    widths = jnp.asarray(batch.target_widths)
    colors = jnp.asarray(batch.target_colors)
    mask = jnp.asarray(batch.loss_mask)

    @brainstate.transform.jit
    def train_all(update_indices: jax.Array) -> jax.Array:
        def train_one(_update_index: jax.Array) -> jax.Array:
            model.reset_state()
            learner.reset_state(batch_size=model.config.batch_size)

            def step_loss(event: jax.Array, advance: jax.Array) -> jax.Array:
                logits = learner(event, advance)
                row_indices = jnp.mod(
                    jnp.asarray(model.reasoning_index.value, dtype=jnp.int32) - 1,
                    _REFINEMENT_STEPS,
                )
                losses = row_refinement_loss_per_example(
                    logits, heights, widths, colors, row_indices
                )
                latent = advance & ~(event[:, model.config.event_valid_index] > 0.5)
                active_count = jnp.maximum(jnp.sum(latent), 1)
                return jnp.sum(jnp.where(latent, losses, 0.0)) / active_count

            gradients, objective = learner.etrace_grad(
                events,
                advances,
                step_fn=step_loss,
                mask=mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            return objective

        return brainstate.transform.for_loop(train_one, update_indices)

    return np.asarray(train_all(jnp.arange(config.updates, dtype=jnp.int32)))


def _decoded_model_candidates(
    model: LatentWorkspaceModel,
    batch: _BindingBatch,
) -> list[list[dict[str, object]]]:
    trajectory = run_selected_packed_stream(
        model,
        jnp.asarray(batch.events),
        batch.selected_indices,
        advance_gates=jnp.asarray(batch.advances),
    )
    expanded = trajectory.expanded
    result: list[list[dict[str, object]]] = []
    for batch_index in range(model.config.batch_size):
        logits = OutputLogits(
            height=np.asarray(expanded.height[0, batch_index]),
            width=np.asarray(expanded.width[0, batch_index]),
            colors=np.asarray(expanded.colors[0, batch_index]),
        )
        result.append(
            [candidate.to_dict() for candidate in decode_candidates(logits, 2)]
        )
    return result


def _exact_count(
    candidates: list[list[dict[str, object]]],
    cases: tuple[SyntheticBindingCase, ...],
) -> tuple[int, list[bool]]:
    exact: list[bool] = []
    for case, proposed in zip(cases, candidates, strict=True):
        target = case.task.test[0].output
        assert target is not None
        target_array = target.as_array()
        exact.append(
            any(
                np.array_equal(candidate["grid"], target_array)
                for candidate in proposed
            )
        )
    return sum(exact), exact


def _parameter_movement(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, object]:
    if before.keys() != after.keys():
        raise ValueError("Parameter paths changed during qualification. Fix the input condition named in the error, then rerun the operation.")
    changed_paths: list[str] = []
    squared_delta = 0.0
    for path in before:
        before_leaves = jax.tree.leaves(before[path])
        after_leaves = jax.tree.leaves(after[path])
        if len(before_leaves) != len(after_leaves):
            raise ValueError(
                f"Parameter structure changed during qualification: {path}. Fix the input condition named in the error, then rerun the operation."
            )
        changed = False
        for before_leaf, after_leaf in zip(before_leaves, after_leaves, strict=True):
            delta = np.asarray(after_leaf, dtype=np.float64) - np.asarray(
                before_leaf, dtype=np.float64
            )
            squared_delta += float(np.sum(delta * delta))
            changed = changed or bool(np.any(delta != 0.0))
        if changed:
            changed_paths.append(path)
    return {
        "parameters_moved": bool(changed_paths),
        "changed_parameter_count": len(changed_paths),
        "changed_parameter_paths": changed_paths,
        "l2_delta": math.sqrt(squared_delta),
    }


def _qualification_gate(
    *,
    exact_before: int,
    exact_after: int,
    family_count: int,
    loss_before: float,
    loss_after: float,
    parameters_moved: bool,
) -> dict[str, object]:
    requirements = {
        "parameters_moved": parameters_moved,
        "loss_improved": (
            math.isfinite(loss_before)
            and math.isfinite(loss_after)
            and loss_after < loss_before
        ),
        "new_exact_grid_learned": exact_after > exact_before,
        "all_families_exact": exact_after == family_count,
    }
    violation_names = {
        "parameters_moved": "parameters_did_not_move",
        "loss_improved": "loss_not_finite_or_improved",
        "new_exact_grid_learned": "no_new_exact_grid",
        "all_families_exact": "not_all_families_exact",
    }
    violations = [
        violation_names[name]
        for name, satisfied in requirements.items()
        if not satisfied
    ]
    return {
        "qualified": all(requirements.values()),
        "requirements": requirements,
        "violations": violations,
    }


def run_binding_qualification(
    config: BindingQualificationConfig | None = None,
) -> dict[str, object]:
    """Run the real row-refinement/pp-prop synthetic diagnostic.

    Parameters
    ----------
    config
        Bounded physical and optimizer configuration.  Defaults to a small CPU
        diagnostic rather than the production-scale ARC experiment.

    Returns
    -------
    dict
        JSON-safe before/after candidates, losses, parameter movement,
        compiler evidence, and strict qualification decision.  Failed learning
        is reported honestly instead of raising or relaxing the gate.
    """

    config = config or BindingQualificationConfig()
    if not isinstance(config, BindingQualificationConfig):
        raise TypeError("Config must be a BindingQualificationConfig. Set Config to a BindingQualificationConfig.")
    cases = synthetic_binding_tasks()
    rows = RowEventConfig(max_demonstrations=2)
    batch = _binding_batch(cases, rows)
    with brainstate.random.seed_context(config.seed):
        model = LatentWorkspaceModel(_model_config(config, rows, batch_size=len(cases)))
    learner = compile_pp_prop(model)

    before_snapshot = parameter_snapshot(model)
    loss_before = _evaluation_loss(model, batch)
    before_candidates = _decoded_model_candidates(model, batch)
    update_losses = _compiled_training_losses(model, learner, batch, config)
    loss_after = _evaluation_loss(model, batch)
    after_candidates = _decoded_model_candidates(model, batch)
    after_snapshot = parameter_snapshot(model)

    movement = _parameter_movement(before_snapshot, after_snapshot)
    exact_before, before_flags = _exact_count(before_candidates, cases)
    exact_after, after_flags = _exact_count(after_candidates, cases)
    gate = _qualification_gate(
        exact_before=exact_before,
        exact_after=exact_after,
        family_count=len(cases),
        loss_before=loss_before,
        loss_after=loss_after,
        parameters_moved=bool(movement["parameters_moved"]),
    )
    compiler_report = getattr(learner, "report", None)
    family_reports: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        target = case.task.test[0].output
        assert target is not None
        family_reports.append(
            {
                "name": case.name,
                "target": target.to_list(),
                "candidate_provenance": "model",
                "before_candidates": before_candidates[index],
                "after_candidates": after_candidates[index],
                "exact_before": before_flags[index],
                "exact_after": after_flags[index],
            }
        )
    return {
        "mode": "synthetic_row_binding_diagnostic",
        "family_count": len(cases),
        "model": {
            "decoder_mode": model.config.decoder_mode,
            "neuron_count": model.config.neuron_count,
            "recurrent_edges": model.config.recurrent_edges,
            "context_memory_width": model.config.context_memory_width,
            "training_output_width": model.config.training_output_width,
            "checkpoint_output_width": model.config.checkpoint_output_width,
        },
        "compiler": {
            "pp_prop_compiled": True,
            "learner_type": type(learner).__name__,
            "compiled_parameter_count": len(learner.param_states),
            "report_available": compiler_report is not None,
        },
        "losses": {
            "before": loss_before,
            "updates": [float(value) for value in update_losses],
            "after": loss_after,
        },
        "movement": movement,
        "exact_grids": {"before": exact_before, "after": exact_after},
        "families": family_reports,
        **gate,
    }


def require_binding_qualification(
    config: BindingQualificationConfig | None = None,
) -> dict[str, object]:
    """Run the diagnostic and raise unless every strict gate is satisfied.

    Parameters
    ----------
    config
        Optional bounded qualification configuration.

    Returns
    -------
    dict
        Successful qualification report.

    Raises
    ------
    BindingQualificationError
        If parameter movement, loss improvement, newly learned exact grids, or
        complete exact-grid performance is absent.
    """

    report = run_binding_qualification(config or BindingQualificationConfig())
    if not report["qualified"]:
        violations = ", ".join(str(item) for item in report["violations"])
        raise BindingQualificationError(
            f"Synthetic row-binding qualification failed: {violations}. Correct the reported inputs, then retry the operation."
        )
    return report
