"""Post-architecture associative-binding gate for Example 21.

This runner retains the deterministic K=4 binding curriculum from the legacy
reservoir control, enables the explicit contextual fast-weight memory, and
trains one shared model with production pp-prop.  Query checkpoint ``H_0`` and
every declared latent checkpoint receive equal, normalized supervision.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop.latent_workspace_binding_control import (
    BindingData,
    SYMBOL_COUNT,
    build_binding_data,
)
from examples.pp_prop.latent_workspace_model import (
    LatentWorkspaceModel,
    ModelConfig,
    compile_pp_prop,
)
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
    associative_memory_feature_indices,
    encode_query_episode,
)


PREREGISTERED_MEMORY_WIDTH = 32
PREREGISTERED_MEMORY_DECAY = 1.0
_REQUIRED_DIRECT_PATHS = (
    ("memory_write_scale",),
    ("workspace_query_projection", "weight"),
    ("memory_read_projection", "weight"),
)


@dataclass(frozen=True)
class BindingGateConfig(legacy.BindingControlConfig):
    """Configure the preregistered post-architecture binding gate.

    Parameters
    ----------
    context_memory_width : int, default=32
        Square fast-weight memory width in ``[1, 128]``.  Width 32 is the
        preregistered Gate A candidate; other widths are nonqualifying
        ablations.
    memory_decay : float, default=1.0
        Demonstration-write memory self-dependence.  Gate A fixes the
        non-fading endpoint so early bindings survive the complete context.

    Notes
    -----
    All inherited defaults retain the legacy control's production topology,
    schedule, optimizer, and deterministic data split.  This runner uses only
    pp-prop; the retained legacy module owns the BPTT diagnostic.
    """

    context_memory_width: int = PREREGISTERED_MEMORY_WIDTH
    memory_decay: float = PREREGISTERED_MEMORY_DECAY

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size < SYMBOL_COUNT:
            raise ValueError("batch_size must be at least four for the K=4 key gate")
        width = self.context_memory_width
        if isinstance(width, (bool, np.bool_)) or not isinstance(width, int):
            raise TypeError("context_memory_width must be a positive integer")
        if width <= 0 or width > 128:
            raise ValueError("context_memory_width must be in [1, 128]")
        object.__setattr__(self, "context_memory_width", int(width))
        decay = self.memory_decay
        if isinstance(decay, (bool, np.bool_)) or not isinstance(decay, (int, float)):
            raise TypeError("memory_decay must be a finite scalar in [0, 1]")
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("memory_decay must be a finite scalar in [0, 1]")
        object.__setattr__(self, "memory_decay", decay)

    @property
    def qualification_regime(self) -> str:
        """Return whether every Gate A evidence setting is preregistered."""

        complete = (
            super().qualification_regime == "preregistered_full"
            and self.context_memory_width == PREREGISTERED_MEMORY_WIDTH
            and self.memory_decay == PREREGISTERED_MEMORY_DECAY
        )
        return "preregistered_full" if complete else "nonqualifying_abbreviated"

    @classmethod
    def smoke_config(cls) -> "BindingGateConfig":
        """Return a deterministic nonqualifying runner configuration.

        Returns
        -------
        BindingGateConfig
            Two updates with the real 64-neuron topology and width-32 memory.
        """

        return cls(
            training_updates=2,
            batch_size=4,
            validation_episodes=8,
            neuron_count=64,
            recurrent_edges=64,
            readout_width=8,
            color_rank=1,
            sparse_backend="jax_raw",
        )


def _model_config(config: BindingGateConfig, *, batch_size: int) -> ModelConfig:
    indices = associative_memory_feature_indices(config.row_config)
    rows = config.row_config
    return ModelConfig(
        input_width=rows.input_width,
        batch_size=batch_size,
        neuron_count=config.neuron_count,
        recurrent_edges=config.recurrent_edges,
        max_latent_steps=config.gap_steps,
        readout_width=config.readout_width,
        color_rank=config.color_rank,
        input_gain=config.input_gain,
        recurrent_gain=config.recurrent_gain,
        trace_decay=config.trace_decay,
        event_valid_index=rows.valid_slice.start,
        context_memory_width=config.context_memory_width,
        memory_decay=config.memory_decay,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        memory_key_indices=indices.key_indices,
        memory_value_indices=indices.value_indices,
        seed=config.model_seed,
        sparse_backend=config.sparse_backend,
    )


def _deep_supervision_mask(config: BindingGateConfig) -> jax.Array:
    depth_count = config.gap_steps + 1
    weight = 1.0 / depth_count
    return (
        jnp.zeros((config.sequence_length,), dtype=jnp.float32)
        .at[SYMBOL_COUNT:]
        .set(weight)
    )


def _path(path: Sequence[object]) -> str:
    return "/".join(map(str, path))


def _configuration_digest(config: BindingGateConfig, model: ModelConfig) -> str:
    payload = json.dumps(
        {
            "experiment": dataclasses.asdict(config),
            "model": dataclasses.asdict(model),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _marginal_identity_report(
    data: BindingData, config: BindingGateConfig
) -> dict[str, Any]:
    rows = config.row_config
    features = associative_memory_feature_indices(rows)
    demo_slice = slice(0, SYMBOL_COUNT)
    intact = data.validation_intact[demo_slice]
    shuffled = data.validation_shuffled[demo_slice]
    key_indices = np.asarray(features.key_indices, dtype=np.int32)
    value_indices = np.asarray(features.value_indices, dtype=np.int32)
    intact_keys = intact[..., key_indices]
    shuffled_keys = shuffled[..., key_indices]
    intact_values = intact[..., value_indices]
    shuffled_values = shuffled[..., value_indices]
    input_equal = np.array_equal(intact_keys, shuffled_keys)
    output_equal = np.array_equal(
        intact_values.sum(axis=0), shuffled_values.sum(axis=0)
    )
    pairing_different = np.any(intact_values != shuffled_values, axis=(0, 2))
    return {
        "exact_input_marginal_equality": bool(input_equal),
        "exact_output_marginal_equality": bool(output_equal),
        "exact_marginal_equality": bool(input_equal and output_equal),
        "different_pairing_count": int(np.count_nonzero(pairing_different)),
        "applicable_count": int(pairing_different.size),
        "every_pairing_differs": bool(np.all(pairing_different)),
        "intact_key_marginal_sha256": legacy._digest_arrays(intact_keys.sum(axis=0)),
        "shuffled_key_marginal_sha256": legacy._digest_arrays(
            shuffled_keys.sum(axis=0)
        ),
        "intact_value_marginal_sha256": legacy._digest_arrays(
            intact_values.sum(axis=0)
        ),
        "shuffled_value_marginal_sha256": legacy._digest_arrays(
            shuffled_values.sum(axis=0)
        ),
    }


def _k4_query_events(rows: RowEventConfig) -> np.ndarray:
    query_index = rows.max_demonstrations * rows.max_grid_size
    episodes = tuple(
        ArcTask(
            train=(ArcPair(ArcGrid(((color,),)), ArcGrid(((color,),))),),
            test=(ArcPair(ArcGrid(((color,),)), None),),
        )
        for color in range(SYMBOL_COUNT)
    )
    return np.stack(
        [encode_query_episode(task, 0, rows).events[query_index] for task in episodes]
    )


def _key_separation_metrics(
    keys: np.ndarray,
    zero_keys: np.ndarray,
    *,
    protocol: str,
    rows: RowEventConfig,
    memory_width: int,
    model_seed: int,
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    gram = keys @ keys.T
    diagonal = np.diag(gram)
    off_diagonal = gram[~np.eye(SYMBOL_COUNT, dtype=bool)]
    diagonal_minimum = float(diagonal.min())
    off_diagonal_maximum = float(off_diagonal.max())
    margin = diagonal_minimum - off_diagonal_maximum
    zero_maximum = float(np.max(np.abs(zero_keys)))
    features = associative_memory_feature_indices(rows)
    return {
        "protocol": protocol,
        "row_event_input_width": rows.input_width,
        "raw_key_feature_width": len(features.key_indices),
        "candidate_colors": list(range(SYMBOL_COUNT)),
        "memory_width": memory_width,
        "model_seed": model_seed,
        "diagonal": diagonal.tolist(),
        "diagonal_minimum": diagonal_minimum,
        "off_diagonal_maximum": off_diagonal_maximum,
        "separation_margin": margin,
        "required_margin": 0.25,
        "margin_passed": margin > 0.25,
        "zero_event_key_max_abs": zero_maximum,
        "zero_event_key_exact_zero": zero_maximum == 0.0,
        "key_sha256": legacy._digest_arrays(keys),
        "architecture": dict(architecture),
    }


def _standard_arc_key_separation_report(
    *,
    memory_width: int = PREREGISTERED_MEMORY_WIDTH,
    model_seed: int = 2108,
) -> dict[str, Any]:
    rows = RowEventConfig()
    features = associative_memory_feature_indices(rows)
    events = jnp.asarray(_k4_query_events(rows), dtype=jnp.float32)
    model = LatentWorkspaceModel(
        ModelConfig(
            input_width=rows.input_width,
            batch_size=SYMBOL_COUNT,
            neuron_count=64,
            recurrent_edges=96,
            max_latent_steps=1,
            readout_width=8,
            color_rank=2,
            context_memory_width=memory_width,
            memory_decay=PREREGISTERED_MEMORY_DECAY,
            demonstration_phase_index=rows.phase_slice.start,
            query_phase_index=rows.phase_slice.start + 1,
            input_side_valid_index=rows.side_valid_slice.start,
            output_side_valid_index=rows.side_valid_slice.start + 1,
            memory_key_indices=features.key_indices,
            memory_value_indices=features.value_indices,
            seed=model_seed,
        )
    )

    @brainstate.transform.jit
    def encode(batch: jax.Array) -> jax.Array:
        return model.encode_memory_key(batch)

    keys = np.asarray(jax.block_until_ready(encode(events)))
    zero_keys = np.asarray(jax.block_until_ready(encode(jnp.zeros_like(events))))
    return _key_separation_metrics(
        keys,
        zero_keys,
        protocol=(
            f"standard_arc_k4_query_key_gram_{len(features.key_indices)}_features"
        ),
        rows=rows,
        memory_width=memory_width,
        model_seed=model_seed,
        architecture=dataclasses.asdict(model.associative_memory_report()),
    )


def _gate_native_key_separation_report(
    model: LatentWorkspaceModel, config: BindingGateConfig
) -> dict[str, Any]:
    if model.config.batch_size < SYMBOL_COUNT:
        raise ValueError("binding gate batch_size must be at least four")
    rows = config.row_config
    events = np.zeros((model.config.batch_size, rows.input_width), dtype=np.float32)
    events[:SYMBOL_COUNT] = _k4_query_events(rows)
    packed = jnp.asarray(events)

    @brainstate.transform.jit
    def encode(batch: jax.Array) -> jax.Array:
        return model.encode_memory_key(batch)

    encoded = np.asarray(jax.block_until_ready(encode(packed)))
    zero_keys = np.asarray(jax.block_until_ready(encode(jnp.zeros_like(packed))))
    return _key_separation_metrics(
        encoded[:SYMBOL_COUNT],
        zero_keys,
        protocol=(
            "gate_native_k4_query_key_gram_"
            f"{len(associative_memory_feature_indices(rows).key_indices)}_features"
        ),
        rows=rows,
        memory_width=config.context_memory_width,
        model_seed=config.model_seed,
        architecture=dataclasses.asdict(model.associative_memory_report()),
    )


def _paired_array_report(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("paired diagnostic arrays must have the same batched shape")
    different = np.any(left != right, axis=tuple(range(1, left.ndim)))
    flat_difference = (left - right).reshape(left.shape[0], -1)
    return {
        "applicable_count": int(left.shape[0]),
        "different_count": int(np.count_nonzero(different)),
        "every_pair_differs": bool(np.all(different)),
        "mean_l2_difference": float(np.linalg.norm(flat_difference, axis=1).mean()),
        "left_sha256": legacy._digest_arrays(left),
        "right_sha256": legacy._digest_arrays(right),
    }


def _diagnostic_report(
    trajectories: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    required = {"intact", "shuffled", "no_context"}
    if set(trajectories) != required:
        raise ValueError("diagnostic trajectories must contain all three arms")
    intact_memory, intact_read, intact_workspace = trajectories["intact"]
    shuffled_memory, shuffled_read, shuffled_workspace = trajectories["shuffled"]
    no_context_memory, no_context_read, no_context_workspace = trajectories[
        "no_context"
    ]
    if no_context_memory.shape != intact_memory.shape:
        raise ValueError("all memory snapshots must share one batched shape")
    memory = _paired_array_report(intact_memory, shuffled_memory)
    memory.update(
        {
            "intact_shuffled_different_count": memory["different_count"],
            "every_intact_shuffled_pair_differs": memory["every_pair_differs"],
            "no_context_exact_zero": bool(np.count_nonzero(no_context_memory) == 0),
            "no_context_sha256": legacy._digest_arrays(no_context_memory),
            "intact_l2_norm": float(np.linalg.norm(intact_memory)),
            "shuffled_l2_norm": float(np.linalg.norm(shuffled_memory)),
            "no_context_l2_norm": float(np.linalg.norm(no_context_memory)),
            "storage_contract": "one final S_K snapshot per arm; S_K is not stacked",
        }
    )
    if (
        intact_read.shape != shuffled_read.shape
        or intact_read.shape != no_context_read.shape
        or intact_workspace.shape != shuffled_workspace.shape
        or intact_workspace.shape != no_context_workspace.shape
        or intact_read.ndim < 2
        or intact_workspace.ndim < 2
        or intact_read.shape[:2] != intact_workspace.shape[:2]
    ):
        raise ValueError("read and workspace trajectories must share depth and batch")
    read_by_depth = {
        str(depth): {
            **_paired_array_report(intact_read[depth], shuffled_read[depth]),
            "no_context_l2_norm": float(np.linalg.norm(no_context_read[depth])),
        }
        for depth in range(intact_read.shape[0])
    }
    workspace_by_depth = {
        str(depth): {
            **_paired_array_report(intact_workspace[depth], shuffled_workspace[depth]),
            "no_context_l2_norm": float(np.linalg.norm(no_context_workspace[depth])),
        }
        for depth in range(intact_workspace.shape[0])
    }
    return {
        "memory": memory,
        "read_by_depth": read_by_depth,
        "workspace_by_depth": workspace_by_depth,
    }


def _compiler_report(learner: Any) -> dict[str, Any]:
    report = legacy._compiler_summary(learner)
    relations = getattr(
        getattr(learner, "graph", None), "hidden_param_op_relations", ()
    )
    direct: dict[tuple[str, ...], bool] = {
        path: False for path in _REQUIRED_DIRECT_PATHS
    }
    evidence: dict[str, list[dict[str, str]]] = {
        _path(path): [] for path in _REQUIRED_DIRECT_PATHS
    }
    for relation in relations:
        trainable_paths = getattr(relation, "trainable_paths", {})
        classifications = getattr(relation, "path_classification", {})
        classification_items = [
            (
                str(key),
                str(getattr(classification, "value", classification)),
            )
            for key, classification in classifications.items()
        ]
        for label, raw_path in trainable_paths.items():
            normalized = tuple(map(str, raw_path))
            if normalized not in direct:
                continue
            if classification_items:
                evidence[_path(normalized)].extend(
                    {
                        "relation_key": relation_key,
                        "classification": classification,
                    }
                    for relation_key, classification in classification_items
                )
            else:
                evidence[_path(normalized)].append(
                    {"relation_key": str(label), "classification": "missing"}
                )
    direct = {
        path: bool(evidence[_path(path)])
        and all(
            item["classification"] == "all_direct" for item in evidence[_path(path)]
        )
        for path in _REQUIRED_DIRECT_PATHS
    }
    report.update(
        {
            "required_direct_paths": [_path(path) for path in _REQUIRED_DIRECT_PATHS],
            "direct_path_status": {
                _path(path): value for path, value in direct.items()
            },
            "direct_path_evidence": evidence,
            "all_required_direct": bool(all(direct.values())),
        }
    )
    return report


def _required_parameter_movement(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, dict[str, float | int | bool]]:
    movement = {}
    for path in _REQUIRED_DIRECT_PATHS:
        label = _path(path)
        if label not in before or label not in after:
            raise ValueError(f"required parameter path is absent: {label}")
        squared_delta = 0.0
        parameter_count = 0
        for old, new in zip(
            jax.tree.leaves(before[label]),
            jax.tree.leaves(after[label]),
            strict=True,
        ):
            delta = np.asarray(new, dtype=np.float64) - np.asarray(
                old, dtype=np.float64
            )
            squared_delta += float(np.sum(delta * delta))
            parameter_count += int(delta.size)
        norm = math.sqrt(squared_delta)
        movement[label] = {
            "l2_delta": norm,
            "parameter_count": parameter_count,
            "changed": norm > 0.0,
        }
    return movement


def _shared_encoder_identity_matches(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    shared_fields = (
        "mode",
        "memory_width",
        "key_map",
        "value_map",
        "rff_gamma",
        "key_basis_seed",
        "key_bias_seed",
        "value_basis_seed",
        "write_component_type",
        "query_component_type",
        "read_component_type",
    )
    return all(left[field] == right[field] for field in shared_fields)


def _train_pp_prop(
    model: LatentWorkspaceModel,
    data: BindingData,
    config: BindingGateConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    compiler = _compiler_report(learner)
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(learner.param_states)
    advances = jnp.ones((config.sequence_length, config.batch_size), dtype=jnp.bool_)
    mask = _deep_supervision_mask(config)
    rank = config.color_rank

    @brainstate.transform.jit
    def train_all(events: jax.Array, targets: jax.Array) -> jax.Array:
        def train_one(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
            sequence, target = inputs
            model.reset_state()
            learner.reset_state(batch_size=config.batch_size)

            def step_loss(event: jax.Array, advance: jax.Array) -> jax.Array:
                return legacy._classification_loss(
                    learner(event, advance), target, rank
                )

            gradients, loss = learner.etrace_grad(
                sequence,
                advances,
                step_fn=step_loss,
                mask=mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            return loss

        return brainstate.transform.for_loop(train_one, (events, targets))

    before = legacy._parameter_values(model)
    start = time.perf_counter()
    losses = train_all(
        jnp.asarray(data.training_events), jnp.asarray(data.training_targets)
    )
    losses = np.asarray(jax.block_until_ready(losses), dtype=np.float64)
    elapsed = time.perf_counter() - start
    after = legacy._parameter_values(model)
    return (
        {
            "algorithm": "production_pp_prop",
            "etrace_config": dataclasses.asdict(model.etrace_config()),
            "loss_definition": (
                "equal normalized one-cell color cross_entropy at query H_0 "
                "and every latent checkpoint"
            ),
            "supervised_depths": list(range(config.gap_steps + 1)),
            "supervision_mask": np.asarray(mask).tolist(),
            "supervision_weight_sum": float(np.asarray(mask).sum()),
            "losses": losses.tolist(),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "tail_64_mean_loss": float(losses[-min(64, losses.size) :].mean()),
            "cold_compile_and_train_seconds": elapsed,
            "parameter_sha256_before": legacy._array_digest(before),
            "parameter_sha256_after": legacy._array_digest(after),
            "parameter_movement": legacy._parameter_movement(before, after),
            "required_direct_parameter_movement": _required_parameter_movement(
                before, after
            ),
            "compiler": compiler,
            "compile_warnings": [str(item.message) for item in caught],
        },
        after,
        compiler,
    )


def _evaluate_model(
    trained_model: LatentWorkspaceModel,
    data: BindingData,
    config: BindingGateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = LatentWorkspaceModel(
        _model_config(config, batch_size=config.validation_episodes)
    )
    legacy._copy_parameters(trained_model, model)
    advances = jnp.ones(
        (config.sequence_length, config.validation_episodes), dtype=jnp.bool_
    )
    rank = config.color_rank

    @brainstate.transform.jit
    def evaluate(
        events: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        model.reset_state()

        def step(inputs: tuple[jax.Array, jax.Array]):
            event, advance = inputs
            compact = model.update(event, advance)
            return (
                compact,
                jnp.asarray(model.memory_read.value),
                jnp.asarray(model.workspace_carrier.value),
            )

        compact, reads, workspaces = brainstate.transform.for_loop(
            step, (events, advances)
        )
        return (
            compact[SYMBOL_COUNT:],
            reads[SYMBOL_COUNT:],
            workspaces[SYMBOL_COUNT:],
            jnp.asarray(model.context_memory.value),
        )

    start = time.perf_counter()
    intact_values = jax.block_until_ready(evaluate(jnp.asarray(data.validation_intact)))
    shuffled_values = jax.block_until_ready(
        evaluate(jnp.asarray(data.validation_shuffled))
    )
    no_context_values = jax.block_until_ready(
        evaluate(jnp.asarray(data.validation_no_context))
    )
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        "intact": tuple(  # type: ignore[dict-item]
            np.asarray(value) for value in intact_values
        ),
        "shuffled": tuple(  # type: ignore[dict-item]
            np.asarray(value) for value in shuffled_values
        ),
        "no_context": tuple(  # type: ignore[dict-item]
            np.asarray(value) for value in no_context_values
        ),
    }
    elapsed = time.perf_counter() - start

    depth_reports: dict[str, dict[str, Any]] = {}
    targets = data.validation_targets
    for depth in range(config.gap_steps + 1):
        arm_metrics = {}
        for arm in ("intact", "shuffled", "no_context"):
            compact = raw[arm][0][depth]
            predictions = np.argmax(
                np.asarray(legacy._color_logits(jnp.asarray(compact), rank)), axis=-1
            )
            arm_metrics[arm] = legacy._accuracy(predictions, targets)
        depth_reports[str(depth)] = arm_metrics

    final = depth_reports[str(config.gap_steps)]
    evaluation = {
        "intact": final["intact"],
        "shuffled": final["shuffled"],
        "no_context": final["no_context"],
        "intact_minus_shuffled": (
            final["intact"]["accuracy"] - final["shuffled"]["accuracy"]
        ),
        "intact_minus_no_context": (
            final["intact"]["accuracy"] - final["no_context"]["accuracy"]
        ),
        "pairing_chance": 1.0 / SYMBOL_COUNT,
        "unconditional_color_chance": 1.0 / legacy.COLOR_COUNT,
        "reported_checkpoint": config.gap_steps,
        "supervised_depths": list(range(config.gap_steps + 1)),
        "depths": depth_reports,
        "cold_compile_and_three_arm_eval_seconds": elapsed,
    }
    diagnostics = _diagnostic_report(
        {arm: (values[3], values[1], values[2]) for arm, values in raw.items()}
    )
    return evaluation, diagnostics


def _qualification_report(
    *,
    evaluation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    compiler: Mapping[str, Any],
    training: Mapping[str, Any],
    architecture: Mapping[str, Any],
    gate_native_separation: Mapping[str, Any],
    standard_arc_separation: Mapping[str, Any],
    marginals: Mapping[str, Any],
    source: Mapping[str, Any],
    config: BindingGateConfig,
) -> dict[str, Any]:
    try:
        intact = evaluation["intact"]
        shuffled = evaluation["shuffled"]
        pairing_chance = float(evaluation["pairing_chance"])
        memory = diagnostics["memory"]
        required_paths = {_path(path) for path in _REQUIRED_DIRECT_PATHS}
        compiler_errors = [
            diagnostic
            for diagnostic in compiler["diagnostics"]
            if str(diagnostic.get("level", "")).lower() == "error"
        ]
        direct_evidence = compiler["direct_path_evidence"]
        movements = training["required_direct_parameter_movement"]
        criteria = {
            "held_out_count_at_least_256": (
                int(intact["count"]) >= 256
                and int(intact["count"]) == config.validation_episodes
            ),
            "intact_accuracy_at_least_0_80": float(intact["accuracy"]) >= 0.80,
            "intact_wilson_lower_above_pairing_chance": (
                float(intact["wilson_95_lower"]) > pairing_chance
            ),
            "intact_minus_shuffled_at_least_0_25": (
                float(evaluation["intact_minus_shuffled"]) >= 0.25
            ),
            "shuffled_not_demonstrably_above_pairing_chance": (
                float(shuffled["wilson_95_lower"]) <= pairing_chance
            ),
            "exact_marginal_equality": marginals["exact_marginal_equality"] is True,
            "every_intact_shuffled_memory_differs": (
                memory["every_intact_shuffled_pair_differs"] is True
                and int(memory["intact_shuffled_different_count"])
                == int(memory["applicable_count"])
                and int(memory["applicable_count"]) == int(intact["count"])
            ),
            "no_context_memory_exact_zero": (memory["no_context_exact_zero"] is True),
            "compiler_required_paths_all_direct": (
                compiler["available"] is True
                and not compiler_errors
                and set(compiler["required_direct_paths"]) == required_paths
                and required_paths.issubset(compiler["compiled_parameter_paths"])
                and set(compiler["direct_path_status"]) == required_paths
                and all(
                    compiler["direct_path_status"][path] is True
                    for path in required_paths
                )
                and set(direct_evidence) == required_paths
                and all(
                    direct_evidence[path]
                    and all(
                        item.get("classification") == "all_direct"
                        for item in direct_evidence[path]
                    )
                    for path in required_paths
                )
                and compiler["all_required_direct"] is True
            ),
            "required_direct_parameters_moved": (
                set(movements) == required_paths
                and all(
                    movements[path]["changed"] is True
                    and math.isfinite(float(movements[path]["l2_delta"]))
                    and float(movements[path]["l2_delta"]) > 0.0
                    and int(movements[path]["parameter_count"]) > 0
                    for path in required_paths
                )
            ),
            "architecture_matches_preregistered_components": (
                architecture["mode"] == "associative_workspace"
                and int(architecture["memory_width"]) == PREREGISTERED_MEMORY_WIDTH
                and architecture["key_map"] == "fixed_rff_cosine"
                and architecture["value_map"] == "fixed_tanh_projection"
                and float(architecture["rff_gamma"]) == 2.0
                and int(architecture["key_basis_seed"]) == config.model_seed + 101
                and int(architecture["key_bias_seed"]) == config.model_seed + 102
                and int(architecture["value_basis_seed"]) == config.model_seed + 103
                and architecture["write_component_type"] == "braintrace.element_wise"
                and architecture["query_component_type"] == "braintrace.nn.Linear"
                and architecture["read_component_type"] == "braintrace.nn.Linear"
                and all(
                    isinstance(architecture[name], str)
                    and len(architecture[name]) == 64
                    for name in (
                        "key_basis_sha256",
                        "key_bias_sha256",
                        "value_basis_sha256",
                    )
                )
            ),
            "gate_native_key_separation_margin_passed": (
                gate_native_separation["margin_passed"] is True
                and int(gate_native_separation["memory_width"])
                == config.context_memory_width
                and int(gate_native_separation["model_seed"]) == config.model_seed
                and float(gate_native_separation["separation_margin"])
                > float(gate_native_separation["required_margin"])
            ),
            "gate_native_zero_event_key_exact_zero": (
                gate_native_separation["zero_event_key_exact_zero"] is True
                and float(gate_native_separation["zero_event_key_max_abs"]) == 0.0
            ),
            "gate_native_basis_matches_training_model": (
                gate_native_separation["architecture"] == architecture
            ),
            "standard_arc_key_separation_margin_passed": (
                standard_arc_separation["margin_passed"] is True
                and int(standard_arc_separation["memory_width"])
                == config.context_memory_width
                and int(standard_arc_separation["model_seed"]) == config.model_seed
                and float(standard_arc_separation["separation_margin"])
                > float(standard_arc_separation["required_margin"])
            ),
            "standard_arc_zero_event_key_exact_zero": (
                standard_arc_separation["zero_event_key_exact_zero"] is True
                and float(standard_arc_separation["zero_event_key_max_abs"]) == 0.0
            ),
            "standard_arc_shared_encoder_invariants_match": (
                _shared_encoder_identity_matches(
                    standard_arc_separation["architecture"], architecture
                )
            ),
            "source_clean": source["dirty"] is False,
            "source_commit_available": str(source["commit"]) not in {"", "unavailable"},
            "preregistered_full_configuration": (
                config.qualification_regime == "preregistered_full"
            ),
        }
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        criteria = {
            "held_out_count_at_least_256": False,
            "intact_accuracy_at_least_0_80": False,
            "intact_wilson_lower_above_pairing_chance": False,
            "intact_minus_shuffled_at_least_0_25": False,
            "shuffled_not_demonstrably_above_pairing_chance": False,
            "exact_marginal_equality": False,
            "every_intact_shuffled_memory_differs": False,
            "no_context_memory_exact_zero": False,
            "compiler_required_paths_all_direct": False,
            "required_direct_parameters_moved": False,
            "architecture_matches_preregistered_components": False,
            "gate_native_key_separation_margin_passed": False,
            "gate_native_zero_event_key_exact_zero": False,
            "gate_native_basis_matches_training_model": False,
            "standard_arc_key_separation_margin_passed": False,
            "standard_arc_zero_event_key_exact_zero": False,
            "standard_arc_shared_encoder_invariants_match": False,
            "source_clean": False,
            "source_commit_available": False,
            "preregistered_full_configuration": False,
        }
    passed = bool(all(criteria.values()))
    if passed:
        interpretation = "gate_a_passed_associative_binding"
    elif config.qualification_regime != "preregistered_full":
        interpretation = "nonqualifying_abbreviated_no_capability_conclusion"
    else:
        interpretation = "gate_a_failed_stop_no_capability_conclusion"
    return {
        "passed": passed,
        "criteria": criteria,
        "interpretation": interpretation,
    }


def _source_report() -> dict[str, Any]:
    commit = os.environ.get("BRAINTRACE_SOURCE_COMMIT", "").strip()
    dirty_env = os.environ.get("BRAINTRACE_SOURCE_DIRTY", "").strip()
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        normalized_dirty = dirty_env.lower()
        dirty = (
            normalized_dirty not in {"0", "false", "no"} if normalized_dirty else True
        )
    return {"commit": commit, "dirty": dirty}


def run_binding_gate(config: BindingGateConfig) -> dict[str, Any]:
    """Train and evaluate the post-architecture pp-prop binding gate.

    Parameters
    ----------
    config : BindingGateConfig
        Complete deterministic Gate A configuration.

    Returns
    -------
    dict
        JSON-compatible artifact with training, interventions, state
        diagnostics, compiler evidence, timing, and fail-closed qualification.
    """

    if not isinstance(config, BindingGateConfig):
        raise TypeError("config must be a BindingGateConfig")
    total_start = time.perf_counter()
    source = _source_report()
    data_start = time.perf_counter()
    data = build_binding_data(config)
    data_seconds = time.perf_counter() - data_start
    training_model_config = _model_config(config, batch_size=config.batch_size)
    model = LatentWorkspaceModel(training_model_config)
    architecture = dataclasses.asdict(model.associative_memory_report())
    gate_native_separation = _gate_native_key_separation_report(model, config)
    standard_arc_separation = _standard_arc_key_separation_report(
        memory_width=config.context_memory_width,
        model_seed=config.model_seed,
    )
    initial = legacy._parameter_values(model)
    initial_digest = legacy._array_digest(initial)
    training, _, compiler = _train_pp_prop(model, data, config)
    evaluation, diagnostics = _evaluate_model(model, data, config)
    marginals = _marginal_identity_report(data, config)
    qualification = _qualification_report(
        evaluation=evaluation,
        diagnostics=diagnostics,
        compiler=compiler,
        training=training,
        architecture=architecture,
        gate_native_separation=gate_native_separation,
        standard_arc_separation=standard_arc_separation,
        marginals=marginals,
        source=source,
        config=config,
    )
    result = {
        "schema_version": 1,
        "control": "example21_associative_workspace_binding_gate_a",
        "learner": "pp_prop_only",
        "source": source,
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_memory_after_run": legacy._device_memory_report(),
        },
        "config": {
            **dataclasses.asdict(config),
            "configuration_scale": config.configuration_scale,
            "qualification_regime": config.qualification_regime,
            "configuration_sha256": _configuration_digest(
                config, training_model_config
            ),
            "model": dataclasses.asdict(training_model_config),
        },
        "data": {
            **legacy._data_report(data, config),
            "generation_seconds": data_seconds,
            "marginals": marginals,
        },
        "initialization": {
            "parameter_sha256": initial_digest,
            "parameter_count": int(
                sum(
                    np.asarray(leaf).size
                    for value in initial.values()
                    for leaf in jax.tree.leaves(value)
                )
            ),
        },
        "architecture": architecture,
        "gate_native_key_separation": gate_native_separation,
        "standard_arc_key_separation": standard_arc_separation,
        "training": training,
        "evaluation": evaluation,
        "diagnostics": diagnostics,
        "qualification": qualification,
        "interpretation": qualification["interpretation"],
    }
    result["total_wall_seconds"] = time.perf_counter() - total_start
    return result


def write_artifact(result: dict[str, Any], path: str | Path) -> Path:
    """Write one standards-compliant JSON artifact atomically.

    Parameters
    ----------
    result : dict
        Result returned by :func:`run_binding_gate`.
    path : str or pathlib.Path
        Destination JSON path.

    Returns
    -------
    pathlib.Path
        Resolved artifact path.
    """

    return legacy.write_artifact(result, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/example21-binding-gate/gate-a.json"),
    )
    parser.add_argument("--training-updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=512)
    parser.add_argument("--gap-steps", type=int, default=1)
    parser.add_argument("--neuron-count", type=int, default=2048)
    parser.add_argument("--recurrent-edges", type=int, default=16_384)
    parser.add_argument("--readout-width", type=int, default=128)
    parser.add_argument("--color-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--context-memory-width", type=int, default=PREREGISTERED_MEMORY_WIDTH
    )
    parser.add_argument(
        "--memory-decay", type=float, default=PREREGISTERED_MEMORY_DECAY
    )
    parser.add_argument("--sparse-backend", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Gate A and write its JSON artifact.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a complete artifact is written.
    """

    args = _parser().parse_args(argv)
    if args.smoke:
        config = BindingGateConfig.smoke_config()
    else:
        config = BindingGateConfig(
            training_updates=args.training_updates,
            batch_size=args.batch_size,
            validation_episodes=args.validation_episodes,
            gap_steps=args.gap_steps,
            neuron_count=args.neuron_count,
            recurrent_edges=args.recurrent_edges,
            readout_width=args.readout_width,
            color_rank=args.color_rank,
            learning_rate=args.learning_rate,
            context_memory_width=args.context_memory_width,
            memory_decay=args.memory_decay,
            sparse_backend=args.sparse_backend,
        )
    result = run_binding_gate(config)
    destination = write_artifact(result, args.output)
    print(destination)
    print(json.dumps(legacy._json_ready(result["qualification"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
