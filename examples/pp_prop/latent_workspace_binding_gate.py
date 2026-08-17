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
import re
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
STRUCTURAL_COLOR_COUNT = legacy.COLOR_COUNT
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
        if self.batch_size < STRUCTURAL_COLOR_COUNT:
            raise ValueError(
                "batch_size must be at least ten for the all-color key gate"
            )
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
            batch_size=16,
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


def _catalog_query_events(rows: RowEventConfig) -> np.ndarray:
    query_index = rows.max_demonstrations * rows.max_grid_size
    episodes = tuple(
        ArcTask(
            train=(ArcPair(ArcGrid(((color,),)), ArcGrid(((color,),))),),
            test=(ArcPair(ArcGrid(((color,),)), None),),
        )
        for color in range(STRUCTURAL_COLOR_COUNT)
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
    if keys.ndim != 2 or keys.shape[0] != STRUCTURAL_COLOR_COUNT:
        raise ValueError("key separation requires one key for every catalog color")
    if zero_keys.ndim != 2 or zero_keys.shape != keys.shape:
        raise ValueError("zero-event keys must match the catalog key matrix")
    gram = keys @ keys.T
    diagonal = np.diag(gram)
    off_diagonal = gram[~np.eye(STRUCTURAL_COLOR_COUNT, dtype=bool)]
    diagonal_minimum = float(diagonal.min())
    off_diagonal_maximum = float(off_diagonal.max())
    margin = diagonal_minimum - off_diagonal_maximum
    zero_maximum = float(np.max(np.abs(zero_keys)))
    features = associative_memory_feature_indices(rows)
    return {
        "protocol": protocol,
        "row_event_input_width": rows.input_width,
        "raw_key_feature_width": len(features.key_indices),
        "candidate_colors": list(range(STRUCTURAL_COLOR_COUNT)),
        "color_count": STRUCTURAL_COLOR_COUNT,
        "gram_shape": list(gram.shape),
        "gram": gram.tolist(),
        "memory_width": memory_width,
        "model_seed": model_seed,
        "diagonal": diagonal.tolist(),
        "diagonal_minimum": diagonal_minimum,
        "off_diagonal_maximum": off_diagonal_maximum,
        "separation_margin": margin,
        "worst_global_margin": margin,
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
    events = jnp.asarray(_catalog_query_events(rows), dtype=jnp.float32)
    model = LatentWorkspaceModel(
        ModelConfig(
            input_width=rows.input_width,
            batch_size=STRUCTURAL_COLOR_COUNT,
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
            f"standard_arc_k10_query_key_gram_{len(features.key_indices)}_features"
        ),
        rows=rows,
        memory_width=memory_width,
        model_seed=model_seed,
        architecture=dataclasses.asdict(model.associative_memory_report()),
    )


def _gate_native_key_separation_report(
    model: LatentWorkspaceModel, config: BindingGateConfig
) -> dict[str, Any]:
    if model.config.batch_size < STRUCTURAL_COLOR_COUNT:
        raise ValueError("binding gate batch_size must be at least ten")
    rows = config.row_config
    events = np.zeros((model.config.batch_size, rows.input_width), dtype=np.float32)
    events[:STRUCTURAL_COLOR_COUNT] = _catalog_query_events(rows)
    packed = jnp.asarray(events)

    @brainstate.transform.jit
    def encode(batch: jax.Array) -> jax.Array:
        return model.encode_memory_key(batch)

    encoded = np.asarray(jax.block_until_ready(encode(packed)))
    zero_keys = np.asarray(jax.block_until_ready(encode(jnp.zeros_like(packed))))
    return _key_separation_metrics(
        encoded[:STRUCTURAL_COLOR_COUNT],
        zero_keys[:STRUCTURAL_COLOR_COUNT],
        protocol=(
            "gate_native_k10_query_key_gram_"
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
    all_state_tensors_finite = all(
        bool(np.isfinite(array).all())
        for trajectory in trajectories.values()
        for array in trajectory
    )
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
        "all_state_tensors_finite": all_state_tensors_finite,
        "memory": memory,
        "read_by_depth": read_by_depth,
        "workspace_by_depth": workspace_by_depth,
    }


def _hidden_group_report(group: Any) -> dict[str, Any]:
    return {
        "index": int(getattr(group, "index")),
        "hidden_paths": [
            _path(path) for path in getattr(group, "hidden_paths", ())
        ],
    }


def _context_memory_isolated(hidden_groups: Sequence[Mapping[str, Any]]) -> bool:
    context_groups = [
        group
        for group in hidden_groups
        if "context_memory" in group.get("hidden_paths", ())
    ]
    if len(context_groups) != 1:
        return False
    forbidden_roots = {"ff_syn", "rec_syn", "neu", "workspace_carrier"}
    for hidden_path in context_groups[0].get("hidden_paths", ()):
        normalized = str(hidden_path).lower()
        root = normalized.split("/", maxsplit=1)[0]
        if (
            root in forbidden_roots
            or root.startswith("workspace")
            or "lif" in normalized
        ):
            return False
    return True


def _compiler_report(learner: Any) -> dict[str, Any]:
    report = legacy._compiler_summary(learner)
    graph = getattr(learner, "graph", None)
    relations = getattr(graph, "hidden_param_op_relations", ())
    hidden_groups = [
        _hidden_group_report(group)
        for group in getattr(graph, "hidden_groups", ())
    ]
    direct: dict[tuple[str, ...], bool] = {
        path: False for path in _REQUIRED_DIRECT_PATHS
    }
    evidence: dict[str, list[dict[str, Any]]] = {
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
        relation_groups = [
            _hidden_group_report(group)
            for group in getattr(relation, "hidden_groups", ())
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
                        "hidden_groups": relation_groups,
                    }
                    for relation_key, classification in classification_items
                )
            else:
                evidence[_path(normalized)].append(
                    {
                        "relation_key": str(label),
                        "classification": "missing",
                        "hidden_groups": relation_groups,
                    }
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
            "hidden_groups": hidden_groups,
            "context_memory_isolated_from_workspace_lif": (
                _context_memory_isolated(hidden_groups)
            ),
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
    all_compact_logits_finite = all(
        bool(np.isfinite(values[0]).all()) for values in raw.values()
    )
    for depth in range(config.gap_steps + 1):
        arm_metrics = {}
        for arm in ("intact", "shuffled", "no_context"):
            compact = raw[arm][0][depth]
            color_logits = np.asarray(
                legacy._color_logits(jnp.asarray(compact), rank)
            )
            all_compact_logits_finite = (
                all_compact_logits_finite and bool(np.isfinite(color_logits).all())
            )
            predictions = np.argmax(color_logits, axis=-1)
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
        "all_compact_logits_finite": all_compact_logits_finite,
        "depths": depth_reports,
        "cold_compile_and_three_arm_eval_seconds": elapsed,
    }
    diagnostics = _diagnostic_report(
        {arm: (values[3], values[1], values[2]) for arm, values in raw.items()}
    )
    return evaluation, diagnostics


def _numeric_tree_is_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_tree_is_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return bool(np.isfinite(value).all())
        return True
    if isinstance(value, (int, float, np.number)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return math.isfinite(float(value))
    return True


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _training_evidence_complete(
    training: Mapping[str, Any], config: BindingGateConfig
) -> bool:
    losses = np.asarray(training["losses"], dtype=np.float64)
    if (
        losses.ndim != 1
        or losses.size != config.training_updates
        or not np.isfinite(losses).all()
    ):
        return False
    initial = float(training["initial_loss"])
    final = float(training["final_loss"])
    tail = float(training["tail_64_mean_loss"])
    expected_tail = float(losses[-min(64, losses.size) :].mean())
    supervision_mask = np.asarray(training["supervision_mask"], dtype=np.float32)
    expected_mask = np.asarray(_deep_supervision_mask(config), dtype=np.float32)
    return bool(
        training["algorithm"] == "production_pp_prop"
        and training["supervised_depths"]
        == list(range(config.gap_steps + 1))
        and supervision_mask.shape == expected_mask.shape
        and np.isfinite(supervision_mask).all()
        and np.array_equal(supervision_mask, expected_mask)
        and math.isclose(
            float(training["supervision_weight_sum"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(initial, float(losses[0]), rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(final, float(losses[-1]), rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(tail, expected_tail, rel_tol=0.0, abs_tol=1e-12)
        and _numeric_tree_is_finite(training)
    )


def _accuracy_evidence_complete(metric: Mapping[str, Any], count: int) -> bool:
    reported_count = metric["count"]
    correct = metric["correct"]
    if (
        not _is_integer(reported_count)
        or int(reported_count) != count
        or not _is_integer(correct)
        or not 0 <= int(correct) <= count
    ):
        return False
    accuracy = float(metric["accuracy"])
    lower = float(metric["wilson_95_lower"])
    upper = float(metric["wilson_95_upper"])
    expected_lower, expected_upper = legacy._wilson_interval(int(correct), count)
    histogram = metric["prediction_histogram"]
    if (
        not isinstance(histogram, (list, tuple))
        or len(histogram) != legacy.COLOR_COUNT
        or not all(_is_integer(item) and int(item) >= 0 for item in histogram)
        or sum(map(int, histogram)) != count
    ):
        return False
    return bool(
        all(math.isfinite(value) for value in (accuracy, lower, upper))
        and math.isclose(
            accuracy,
            int(correct) / count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and 0.0 <= lower <= accuracy <= upper <= 1.0
        and math.isclose(lower, expected_lower, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(upper, expected_upper, rel_tol=0.0, abs_tol=1e-12)
        and re.fullmatch(r"[0-9a-f]{64}", str(metric["prediction_sha256"]))
    )


def _evaluation_evidence_complete(
    evaluation: Mapping[str, Any], config: BindingGateConfig
) -> bool:
    expected_depths = list(range(config.gap_steps + 1))
    depth_reports = evaluation["depths"]
    if (
        not isinstance(depth_reports, Mapping)
        or set(depth_reports) != {str(depth) for depth in expected_depths}
        or evaluation["supervised_depths"] != expected_depths
        or int(evaluation["reported_checkpoint"]) != config.gap_steps
        or evaluation["all_compact_logits_finite"] is not True
        or not _numeric_tree_is_finite(evaluation)
    ):
        return False
    for depth in expected_depths:
        report = depth_reports[str(depth)]
        if not isinstance(report, Mapping) or set(report) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        for arm in ("intact", "shuffled", "no_context"):
            if not _accuracy_evidence_complete(
                report[arm], config.validation_episodes
            ):
                return False
    final = depth_reports[str(config.gap_steps)]
    if any(evaluation[arm] != final[arm] for arm in final):
        return False
    intact_accuracy = float(final["intact"]["accuracy"])
    shuffled_accuracy = float(final["shuffled"]["accuracy"])
    no_context_accuracy = float(final["no_context"]["accuracy"])
    return bool(
        math.isclose(
            float(evaluation["intact_minus_shuffled"]),
            intact_accuracy - shuffled_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(evaluation["intact_minus_no_context"]),
            intact_accuracy - no_context_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(evaluation["pairing_chance"]),
            1.0 / SYMBOL_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(evaluation["unconditional_color_chance"]),
            1.0 / legacy.COLOR_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )


def _paired_diagnostic_complete(report: Mapping[str, Any], count: int) -> bool:
    applicable = report["applicable_count"]
    different = report["different_count"]
    if (
        not _is_integer(applicable)
        or int(applicable) != count
        or not _is_integer(different)
        or not 0 <= int(different) <= count
    ):
        return False
    return bool(
        int(different) == count
        and report["every_pair_differs"] is True
        and math.isfinite(float(report["mean_l2_difference"]))
        and float(report["mean_l2_difference"]) > 0.0
        and math.isfinite(float(report["no_context_l2_norm"]))
    )


def _diagnostic_evidence_complete(
    diagnostics: Mapping[str, Any], config: BindingGateConfig
) -> bool:
    expected_depths = {
        str(depth) for depth in range(config.gap_steps + 1)
    }
    memory = diagnostics["memory"]
    read_by_depth = diagnostics["read_by_depth"]
    workspace_by_depth = diagnostics["workspace_by_depth"]
    if (
        diagnostics["all_state_tensors_finite"] is not True
        or not _numeric_tree_is_finite(diagnostics)
        or set(read_by_depth) != expected_depths
        or set(workspace_by_depth) != expected_depths
        or int(memory["applicable_count"]) != config.validation_episodes
        or int(memory["different_count"])
        != int(memory["intact_shuffled_different_count"])
        or memory["every_pair_differs"]
        is not memory["every_intact_shuffled_pair_differs"]
        or memory["no_context_exact_zero"] is not True
        or not all(
            math.isfinite(float(memory[name]))
            for name in (
                "intact_l2_norm",
                "shuffled_l2_norm",
                "no_context_l2_norm",
            )
        )
    ):
        return False
    for depth in expected_depths:
        if not _paired_diagnostic_complete(
            read_by_depth[depth], config.validation_episodes
        ) or not _paired_diagnostic_complete(
            workspace_by_depth[depth], config.validation_episodes
        ):
            return False
    return True


def _compiler_topology_complete(compiler: Mapping[str, Any]) -> bool:
    hidden_groups = compiler["hidden_groups"]
    if not isinstance(hidden_groups, list) or not hidden_groups:
        return False
    known_groups: dict[int, tuple[str, ...]] = {}
    for group in hidden_groups:
        if (
            not _is_integer(group.get("index"))
            or not isinstance(group.get("hidden_paths"), list)
            or not all(isinstance(path, str) and path for path in group["hidden_paths"])
            or int(group["index"]) in known_groups
        ):
            return False
        known_groups[int(group["index"])] = tuple(group["hidden_paths"])
    evidence = compiler["direct_path_evidence"]
    topology_is_consistent = all(
        item.get("hidden_groups")
        and all(
            _is_integer(group.get("index"))
            and isinstance(group.get("hidden_paths"), list)
            and bool(group["hidden_paths"])
            and known_groups.get(int(group["index"]))
            == tuple(group["hidden_paths"])
            for group in item["hidden_groups"]
        )
        for path in compiler["required_direct_paths"]
        for item in evidence[path]
    )
    if not topology_is_consistent:
        return False

    def targets(path: str) -> list[set[str]]:
        return [
            set(item["hidden_groups"][0]["hidden_paths"])
            for item in evidence[path]
            if len(item["hidden_groups"]) == 1
        ]

    write_targets = targets("memory_write_scale")
    query_targets = targets("workspace_query_projection/weight")
    read_targets = targets("memory_read_projection/weight")
    read_has_workspace_lif = all(
        "context_memory" not in paths
        and "workspace_carrier" in paths
        and any(
            path.split("/", maxsplit=1)[0] in {"ff_syn", "rec_syn", "neu"}
            or "lif" in path.lower()
            for path in paths
        )
        for paths in read_targets
    )
    return bool(
        len(write_targets) == len(evidence["memory_write_scale"])
        and all(paths == {"context_memory"} for paths in write_targets)
        and len(query_targets)
        == len(evidence["workspace_query_projection/weight"])
        and all(paths == {"reasoning_query"} for paths in query_targets)
        and len(read_targets) == len(evidence["memory_read_projection/weight"])
        and read_has_workspace_lif
    )


def _architecture_digests_valid(architecture: Mapping[str, Any]) -> bool:
    return all(
        isinstance(architecture.get(name), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", architecture[name]))
        for name in (
            "key_basis_sha256",
            "key_bias_sha256",
            "value_basis_sha256",
        )
    )


def _all_color_separation_complete(
    report: Mapping[str, Any],
    *,
    protocol: str,
    row_event_input_width: int,
    raw_key_feature_width: int,
) -> bool:
    gram = np.asarray(report["gram"], dtype=np.float64)
    if (
        report["protocol"] != protocol
        or int(report["row_event_input_width"]) != row_event_input_width
        or int(report["raw_key_feature_width"]) != raw_key_feature_width
        or not _architecture_digests_valid(report["architecture"])
        or report["candidate_colors"] != list(range(STRUCTURAL_COLOR_COUNT))
        or int(report["color_count"]) != STRUCTURAL_COLOR_COUNT
        or report["gram_shape"]
        != [STRUCTURAL_COLOR_COUNT, STRUCTURAL_COLOR_COUNT]
        or gram.shape != (STRUCTURAL_COLOR_COUNT, STRUCTURAL_COLOR_COUNT)
        or not np.isfinite(gram).all()
        or not np.allclose(gram, gram.T, rtol=0.0, atol=1e-12)
    ):
        return False
    diagonal = np.diag(gram)
    off_diagonal = gram[~np.eye(STRUCTURAL_COLOR_COUNT, dtype=bool)]
    diagonal_minimum = float(diagonal.min())
    off_diagonal_maximum = float(off_diagonal.max())
    margin = diagonal_minimum - off_diagonal_maximum
    return bool(
        np.allclose(
            np.asarray(report["diagonal"], dtype=np.float64),
            diagonal,
            rtol=0.0,
            atol=1e-12,
        )
        and math.isclose(
            float(report["diagonal_minimum"]),
            diagonal_minimum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(report["off_diagonal_maximum"]),
            off_diagonal_maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(report["separation_margin"]),
            margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(report["worst_global_margin"]),
            margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _gpu_environment_verified(environment: Mapping[str, Any]) -> bool:
    devices = environment["devices"]
    return bool(
        environment["backend"] == "gpu"
        and isinstance(devices, list)
        and devices
        and all(isinstance(device, Mapping) for device in devices)
        and any(device.get("platform") == "gpu" for device in devices)
        and bool(
            re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(environment["image_digest"])
            )
        )
    )


def _source_evidence_clean(source: Mapping[str, Any]) -> bool:
    commit = str(source["commit"])
    asserted_commit = source["asserted_commit"]
    return bool(
        re.fullmatch(r"[0-9a-f]{40}", commit)
        and isinstance(asserted_commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", asserted_commit.lower()))
        and asserted_commit.lower() == commit
        and source["commit_is_valid_40_hex"] is True
        and source["asserted_commit_matches_head"] is True
        and source["asserted_dirty"] is False
        and source["asserted_dirty_matches_worktree"] is True
        and source["verified"] is True
        and source["dirty"] is False
    )


_QUALIFICATION_CRITERIA = (
    "held_out_count_at_least_256",
    "intact_accuracy_at_least_0_80",
    "intact_wilson_lower_above_pairing_chance",
    "intact_minus_shuffled_at_least_0_25",
    "shuffled_not_demonstrably_above_pairing_chance",
    "exact_marginal_equality",
    "every_intact_shuffled_memory_differs",
    "no_context_memory_exact_zero",
    "compiler_required_paths_all_direct",
    "context_memory_hidden_group_isolated",
    "required_direct_parameters_moved",
    "training_losses_complete_and_finite",
    "evaluation_complete_and_finite",
    "diagnostic_state_tensors_complete_and_finite",
    "gpu_backend_verified",
    "architecture_matches_preregistered_components",
    "gate_native_all_colors_covered",
    "gate_native_key_separation_margin_passed",
    "gate_native_zero_event_key_exact_zero",
    "gate_native_basis_matches_training_model",
    "standard_arc_all_colors_covered",
    "standard_arc_key_separation_margin_passed",
    "standard_arc_zero_event_key_exact_zero",
    "standard_arc_shared_encoder_invariants_match",
    "source_start_verified_clean",
    "source_end_verified_clean",
    "source_head_stable",
    "preregistered_full_configuration",
)


def _qualification_report(
    *,
    evaluation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    compiler: Mapping[str, Any],
    training: Mapping[str, Any],
    environment: Mapping[str, Any],
    architecture: Mapping[str, Any],
    gate_native_separation: Mapping[str, Any],
    standard_arc_separation: Mapping[str, Any],
    marginals: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end: Mapping[str, Any],
    config: BindingGateConfig,
) -> dict[str, Any]:
    criteria = {name: False for name in _QUALIFICATION_CRITERIA}
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
        compiler_direct = bool(
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
            and _compiler_topology_complete(compiler)
        )
        gate_native_all_colors = _all_color_separation_complete(
            gate_native_separation,
            protocol="gate_native_k10_query_key_gram_18_features",
            row_event_input_width=41,
            raw_key_feature_width=18,
        )
        standard_arc_all_colors = _all_color_separation_complete(
            standard_arc_separation,
            protocol="standard_arc_k10_query_key_gram_424_features",
            row_event_input_width=830,
            raw_key_feature_width=424,
        )
        criteria.update({
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
            "compiler_required_paths_all_direct": compiler_direct,
            "context_memory_hidden_group_isolated": (
                compiler["context_memory_isolated_from_workspace_lif"] is True
                and _context_memory_isolated(compiler["hidden_groups"])
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
            "training_losses_complete_and_finite": _training_evidence_complete(
                training, config
            ),
            "evaluation_complete_and_finite": _evaluation_evidence_complete(
                evaluation, config
            ),
            "diagnostic_state_tensors_complete_and_finite": (
                _diagnostic_evidence_complete(diagnostics, config)
            ),
            "gpu_backend_verified": _gpu_environment_verified(environment),
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
                and _architecture_digests_valid(architecture)
            ),
            "gate_native_all_colors_covered": gate_native_all_colors,
            "gate_native_key_separation_margin_passed": (
                gate_native_all_colors
                and gate_native_separation["margin_passed"] is True
                and int(gate_native_separation["memory_width"])
                == config.context_memory_width
                and int(gate_native_separation["model_seed"]) == config.model_seed
                and float(gate_native_separation["separation_margin"])
                > float(gate_native_separation["required_margin"])
                and float(gate_native_separation["required_margin"]) == 0.25
            ),
            "gate_native_zero_event_key_exact_zero": (
                gate_native_separation["zero_event_key_exact_zero"] is True
                and float(gate_native_separation["zero_event_key_max_abs"]) == 0.0
            ),
            "gate_native_basis_matches_training_model": (
                gate_native_separation["architecture"] == architecture
            ),
            "standard_arc_all_colors_covered": standard_arc_all_colors,
            "standard_arc_key_separation_margin_passed": (
                standard_arc_all_colors
                and standard_arc_separation["margin_passed"] is True
                and int(standard_arc_separation["memory_width"])
                == config.context_memory_width
                and int(standard_arc_separation["model_seed"]) == config.model_seed
                and float(standard_arc_separation["separation_margin"])
                > float(standard_arc_separation["required_margin"])
                and float(standard_arc_separation["required_margin"]) == 0.25
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
            "source_start_verified_clean": _source_evidence_clean(source_start),
            "source_end_verified_clean": _source_evidence_clean(source_end),
            "source_head_stable": (
                str(source_start["commit"]) == str(source_end["commit"])
                and bool(re.fullmatch(r"[0-9a-f]{40}", str(source_start["commit"])))
            ),
            "preregistered_full_configuration": (
                config.qualification_regime == "preregistered_full"
            ),
        })
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        pass
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
    asserted_commit = os.environ.get("BRAINTRACE_SOURCE_COMMIT", "").strip()
    asserted_dirty_raw = os.environ.get("BRAINTRACE_SOURCE_DIRTY", "").strip()
    head_command_succeeded = False
    status_command_succeeded = False
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        head_command_succeeded = True
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
        status_command_succeeded = True
    except (OSError, subprocess.CalledProcessError):
        dirty = True
    commit_is_valid = bool(re.fullmatch(r"[0-9a-f]{40}", commit))
    asserted_commit_matches = (
        not asserted_commit or asserted_commit.lower() == commit
    )
    asserted_dirty: bool | None
    normalized_dirty = asserted_dirty_raw.lower()
    if not asserted_dirty_raw:
        asserted_dirty = None
        asserted_dirty_matches = True
    elif normalized_dirty in {"1", "true", "yes"}:
        asserted_dirty = True
        asserted_dirty_matches = dirty is True
    elif normalized_dirty in {"0", "false", "no"}:
        asserted_dirty = False
        asserted_dirty_matches = dirty is False
    else:
        asserted_dirty = None
        asserted_dirty_matches = False
    verified = bool(
        head_command_succeeded
        and status_command_succeeded
        and commit_is_valid
        and bool(asserted_commit)
        and asserted_commit_matches
        and asserted_dirty is not None
        and asserted_dirty_matches
    )
    return {
        "commit": commit,
        "asserted_commit": asserted_commit or None,
        "asserted_commit_matches_head": asserted_commit_matches,
        "commit_is_valid_40_hex": commit_is_valid,
        "head_command_succeeded": head_command_succeeded,
        "dirty": dirty,
        "asserted_dirty": asserted_dirty,
        "asserted_dirty_matches_worktree": asserted_dirty_matches,
        "status_command_succeeded": status_command_succeeded,
        "verified": verified,
    }


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
    source_start = _source_report()
    environment = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in jax.devices()
        ],
        "image_digest": os.environ.get("BRAINTRACE_IMAGE_DIGEST", "").strip(),
    }
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
    environment["device_memory_after_run"] = legacy._device_memory_report()
    source_end = _source_report()
    qualification = _qualification_report(
        evaluation=evaluation,
        diagnostics=diagnostics,
        compiler=compiler,
        training=training,
        environment=environment,
        architecture=architecture,
        gate_native_separation=gate_native_separation,
        standard_arc_separation=standard_arc_separation,
        marginals=marginals,
        source_start=source_start,
        source_end=source_end,
        config=config,
    )
    result = {
        "schema_version": 2,
        "control": "example21_associative_workspace_binding_gate_a",
        "learner": "pp_prop_only",
        "source": source_start,
        "source_end": source_end,
        "environment": environment,
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
