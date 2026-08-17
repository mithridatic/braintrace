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
    _unit_l2_cap,
    compile_pp_prop,
)
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    MAX_GRID_SIZE,
    RowEventConfig,
    associative_memory_feature_indices,
    encode_query_episode,
)


PREREGISTERED_MEMORY_WIDTH = 32
PREREGISTERED_MEMORY_DECAY = 1.0
STRUCTURAL_COLOR_COUNT = legacy.COLOR_COUNT
STAGE21_STABILITY_UPDATES = 256
STAGE21_ARTIFACT_SCHEMA_VERSION = 3
STAGE21_ADMISSION_SCHEMA_VERSION = 1
STAGE21_CARRIER_NORM_TOLERANCE = 2e-6
STAGE21_DECODER_REPLAY_ATOL = 3e-5
PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256 = (
    "b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8"
)
PREREGISTERED_PARAMETER_COUNT = 646_940
PREREGISTERED_TRAINING_SCHEDULE_SHA256 = (
    "25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b"
)
PREREGISTERED_UPDATE_ZERO_DIGESTS = {
    "update_zero_event_sha256": (
        "25d451acf3acf89713cd1fc053568769b73bde04f4e465835b982a4b412d168f"
    ),
    "update_zero_target_sha256": (
        "0e2d0ce71541888e8ca160ca5177cd3ccad900afb822ebb14cb3b469aa743031"
    ),
    "update_zero_mapping_sha256": (
        "dd42d9393e148f6e39a4fa2eb35ca77cb28b81e38fc94a15bba3dc0965143175"
    ),
    "update_zero_query_sha256": (
        "bad858fa344a6d7eb1cdfd5128c1ccd25cd8a6de5967a84ae93632beb336f5ee"
    ),
}
PREREGISTERED_STABILITY_DIGESTS = {
    "training_schedule_sha256": (
        "2b5ab5dadd6e67117ee3aa52d08864e936b7f3ba9346f84125636415f02c5daa"
    ),
    "training_query_indices_sha256": (
        "965feae6ae670f19b859d851cef5814f8a313e44cf743fa3803fd91cc1164a6e"
    ),
    "validation_schedule_sha256": (
        "80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd"
    ),
    "training_mapping_ids_sha256": (
        "bd66b59e5a79fcadd21d0fc4a7166baefae5ec489acf76959530a13690239b8b"
    ),
    "validation_mapping_ids_sha256": (
        "a75b3b2ab05110e21fef1ea44ae3fb701d557f45351e5a17cf89a80e80f689f3"
    ),
}
_REQUIRED_DIRECT_PATHS = (
    ("memory_write_scale",),
    ("workspace_query_projection", "weight"),
    ("memory_read_projection", "weight"),
)
_STAGE21_OPTIMIZATION_PATHS = (
    "memory_write_scale",
    "workspace_query_projection/weight",
    "memory_read_projection/weight",
    "readout_projection/weight",
    "color_factor_head/weight",
)
_STAGE21_TRACE_PATHS = _STAGE21_OPTIMIZATION_PATHS[:3]
_STAGE21_PARAMETER_COUNTS = {
    "memory_write_scale": 1_024,
    "workspace_query_projection/weight": 65_536,
    "memory_read_projection/weight": 65_536,
    "readout_projection/weight": 262_272,
    "color_factor_head/weight": 144_480,
}
_STAGE21_TRACE_FACTOR_COUNTS = {
    "memory_write_scale": 65_536,
    "workspace_query_projection/weight": 133_120,
    "memory_read_projection/weight": 526_336,
}
_STAGE21_FINITE_ONE_UPDATE = {
    "cross_entropies",
    "color_logits",
    "raw_carriers",
    "capped_carriers",
    "gradients",
    "pp_prop_factors",
    "adam_factors",
    "parameter_updates",
    "decoder_factors",
}
_STAGE21_FINITE_STABILITY = {
    "losses",
    "states",
    "logits",
    "gradients",
    "pp_prop_factors",
    "adam_factors",
    "parameters",
}


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

    @classmethod
    def stage21_one_update_config(cls) -> "BindingGateConfig":
        """Return the exact full-production schedule used for update zero.

        Returns
        -------
        BindingGateConfig
            Preregistered 10,000-update configuration whose update-zero batch
            is consumed by the one-update admission.
        """

        return cls()

    @classmethod
    def stage21_stability_config(cls) -> "BindingGateConfig":
        """Return the fixed nonqualifying 256-update stabilization run.

        Returns
        -------
        BindingGateConfig
            Production topology and seeds with only the update count reduced
            to the preregistered 256-update admission budget.
        """

        return cls(training_updates=STAGE21_STABILITY_UPDATES)


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
        architecture=_architecture_report(model),
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
        architecture=_architecture_report(model),
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
    try:
        return _json_exact(
            {field: left[field] for field in shared_fields},
            {field: right[field] for field in shared_fields},
        )
    except KeyError:
        return False


@dataclass
class _PPPropTrainer:
    learner: Any
    optimizer: Any
    compiler: dict[str, Any]
    compile_warnings: list[str]
    train: Any
    parameter_counts: dict[str, int]
    trace_factor_counts: dict[str, int]
    parameter_keys: dict[str, Any]
    supervision_mask: jax.Array


def _jax_tree_stats(value: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(value))
    if not leaves:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero, jnp.asarray(False)
    squared = sum(
        jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves
    )
    maximum = jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(leaf.astype(jnp.float32)), initial=0.0)
                for leaf in leaves
            ]
        )
    )
    finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
    )
    return jnp.sqrt(squared), maximum, finite


def _mapping_value_by_path(values: Mapping[Any, Any], label: str) -> Any:
    matches = [value for path, value in values.items() if _path(path) == label]
    if len(matches) != 1:
        raise KeyError(f"expected one value for parameter path {label!r}")
    return matches[0]


def _make_pp_prop_trainer(
    model: LatentWorkspaceModel, config: BindingGateConfig
) -> _PPPropTrainer:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    compiler = _compiler_report(learner)
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(learner.param_states)
    advances = jnp.ones((config.sequence_length, config.batch_size), dtype=jnp.bool_)
    mask = _deep_supervision_mask(config)
    rank = config.color_rank
    parameter_keys = {
        _path(path): path for path in learner.param_states.keys()
    }
    missing = set(_STAGE21_OPTIMIZATION_PATHS) - set(parameter_keys)
    if missing:
        raise RuntimeError(f"trainer is missing required parameters: {sorted(missing)}")
    parameter_counts = {
        label: int(
            sum(
                np.asarray(leaf).size
                for leaf in jax.tree.leaves(learner.param_states[key].value)
            )
        )
        for label, key in parameter_keys.items()
        if label in _STAGE21_OPTIMIZATION_PATHS
    }
    model_states = tuple(model.states().values())

    @brainstate.transform.jit
    def train_all(events: jax.Array, targets: jax.Array) -> jax.Array:
        def train_one(inputs: tuple[jax.Array, jax.Array]) -> dict[str, jax.Array]:
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
            gradient_stats = tuple(
                _jax_tree_stats(_mapping_value_by_path(gradients, label))
                for label in _STAGE21_OPTIMIZATION_PATHS
            )
            trace_factors = tuple(
                learner.get_etrace_of(parameter_keys[label])
                for label in _STAGE21_TRACE_PATHS
            )
            trace_stats = tuple(
                _jax_tree_stats(factors) for factors in trace_factors
            )
            compact = model.compact_readout()
            logits = legacy._color_logits(compact, rank)
            _, logit_maximum, logits_finite = _jax_tree_stats(logits)
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            _, state_maximum, states_finite = _jax_tree_stats(
                tuple(state.value for state in model_states)
            )
            _, adam_maximum, adam_finite = _jax_tree_stats(optimizer.opt_state.value)
            _, parameter_maximum, parameters_finite = _jax_tree_stats(
                tuple(state.value for state in learner.param_states.values())
            )
            return {
                "loss": loss,
                "gradient_norms": jnp.stack([item[0] for item in gradient_stats]),
                "gradient_max_abs": jnp.stack([item[1] for item in gradient_stats]),
                "gradients_finite": jnp.all(
                    jnp.stack([item[2] for item in gradient_stats])
                ),
                "trace_norms": jnp.stack([item[0] for item in trace_stats]),
                "trace_max_abs": jnp.stack([item[1] for item in trace_stats]),
                "traces_finite": jnp.all(
                    jnp.stack([item[2] for item in trace_stats])
                ),
                "logit_max_abs": logit_maximum,
                "logits_finite": logits_finite,
                "state_max_abs": state_maximum,
                "states_finite": states_finite,
                "adam_max_abs": adam_maximum,
                "adam_finite": adam_finite,
                "parameter_max_abs": parameter_maximum,
                "parameters_finite": parameters_finite,
            }

        return brainstate.transform.for_loop(train_one, (events, targets))

    learner.reset_state(batch_size=config.batch_size)
    trace_factor_counts: dict[str, int] = {}
    for label in _STAGE21_TRACE_PATHS:
        xs, dfs = learner.get_etrace_of(parameter_keys[label])
        trace_factor_counts[label] = int(
            sum(np.asarray(leaf).size for leaf in jax.tree.leaves((xs, dfs)))
        )
    return _PPPropTrainer(
        learner=learner,
        optimizer=optimizer,
        compiler=compiler,
        compile_warnings=[str(item.message) for item in caught],
        train=train_all,
        parameter_counts=parameter_counts,
        trace_factor_counts=trace_factor_counts,
        parameter_keys={
            label: parameter_keys[label] for label in _STAGE21_OPTIMIZATION_PATHS
        },
        supervision_mask=mask,
    )


def _train_pp_prop(
    model: LatentWorkspaceModel,
    data: BindingData,
    config: BindingGateConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trainer = _make_pp_prop_trainer(model, config)

    before = legacy._parameter_values(model)
    start = time.perf_counter()
    telemetry = trainer.train(
        jnp.asarray(data.training_events), jnp.asarray(data.training_targets)
    )
    telemetry = jax.block_until_ready(telemetry)
    losses = np.asarray(telemetry["loss"], dtype=np.float64)
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
            "supervision_mask": np.asarray(trainer.supervision_mask).tolist(),
            "supervision_weight_sum": float(
                np.asarray(trainer.supervision_mask).sum()
            ),
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
            "finite_telemetry": {
                "losses": bool(np.isfinite(losses).all()),
                "states": bool(np.asarray(telemetry["states_finite"]).all()),
                "logits": bool(np.asarray(telemetry["logits_finite"]).all()),
                "gradients": bool(
                    np.asarray(telemetry["gradients_finite"]).all()
                ),
                "pp_prop_factors": bool(
                    np.asarray(telemetry["traces_finite"]).all()
                ),
                "adam_factors": bool(np.asarray(telemetry["adam_finite"]).all()),
                "parameters": bool(
                    np.asarray(telemetry["parameters_finite"]).all()
                ),
            },
            "telemetry_max_abs": {
                name: float(np.asarray(telemetry[field]).max(initial=0.0))
                for name, field in {
                    "states": "state_max_abs",
                    "logits": "logit_max_abs",
                    "gradients": "gradient_max_abs",
                    "pp_prop_factors": "trace_max_abs",
                    "adam_factors": "adam_max_abs",
                    "parameters": "parameter_max_abs",
                }.items()
            },
            "compiler": trainer.compiler,
            "compile_warnings": trainer.compile_warnings,
        },
        after,
        trainer.compiler,
    )


def _architecture_report(model: LatentWorkspaceModel) -> dict[str, Any]:
    return dict(model.associative_memory_report().to_dict())


def _make_batch_measurement(model: LatentWorkspaceModel, config: BindingGateConfig):
    advances = jnp.ones((config.sequence_length, config.batch_size), dtype=jnp.bool_)

    @brainstate.transform.jit
    def measure(
        events: jax.Array, targets: jax.Array
    ) -> dict[str, jax.Array]:
        model.reset_state()

        def step(inputs: tuple[jax.Array, jax.Array]):
            event, advance = inputs
            compact = model.update(event, advance)
            return (
                compact,
                jnp.asarray(model.workspace_carrier.value),
                jnp.asarray(model.reasoning_query.value),
                jnp.asarray(model.query_encoding.value),
            )

        compact, raw_carrier, reasoning_query, query_encoding = (
            brainstate.transform.for_loop(
            step, (events, advances)
            )
        )
        compact = compact[SYMBOL_COUNT:]
        raw_carrier = raw_carrier[SYMBOL_COUNT:]
        reasoning_query = reasoning_query[SYMBOL_COUNT:]
        query_encoding = query_encoding[SYMBOL_COUNT:]
        capped_carrier = _unit_l2_cap(raw_carrier)
        depth_count, batch_size, neuron_count = raw_carrier.shape
        capped_flat = capped_carrier.reshape(depth_count * batch_size, neuron_count)
        raw_flat = raw_carrier.reshape(depth_count * batch_size, neuron_count)
        readout_preactivation = model.readout_projection(capped_flat).reshape(
            depth_count, batch_size, config.readout_width
        )
        readout_post_gelu = jax.nn.gelu(readout_preactivation)
        hidden_flat = readout_post_gelu.reshape(
            depth_count * batch_size, config.readout_width
        )
        height = model.height_head(hidden_flat).reshape(depth_count, batch_size, -1)
        width = model.width_head(hidden_flat).reshape(depth_count, batch_size, -1)
        factors = model.color_factor_head(hidden_flat).reshape(
            depth_count, batch_size, -1
        )
        factor_width = config.color_rank * MAX_GRID_SIZE
        row_factors = factors[..., :factor_width].reshape(
            depth_count, batch_size, config.color_rank, MAX_GRID_SIZE
        )
        column_factors = factors[..., factor_width : 2 * factor_width].reshape(
            depth_count, batch_size, config.color_rank, MAX_GRID_SIZE
        )
        color_factors = factors[..., 2 * factor_width :].reshape(
            depth_count, batch_size, config.color_rank, legacy.COLOR_COUNT
        )
        reconstructed_compact = jnp.concatenate((height, width, factors), axis=-1)
        uncapped_hidden = jax.nn.gelu(model.readout_projection(raw_flat))
        uncapped_compact = jnp.concatenate(
            (
                model.height_head(uncapped_hidden),
                model.width_head(uncapped_hidden),
                model.color_factor_head(uncapped_hidden),
            ),
            axis=-1,
        ).reshape(compact.shape)
        h0 = raw_carrier[0]
        capped_reasoning = jnp.tanh(
            query_encoding[1] + model.workspace_query_projection(_unit_l2_cap(h0))
        )
        uncapped_reasoning = jnp.tanh(
            query_encoding[1] + model.workspace_query_projection(h0)
        )
        logits = jax.vmap(
            lambda value: legacy._color_logits(value, config.color_rank)
        )(compact)
        cross_entropy = jax.vmap(
            lambda value: braintools.metric.softmax_cross_entropy_with_integer_labels(
                value, targets
            ).mean()
        )(logits)
        return {
            "cross_entropy": cross_entropy,
            "logits": logits,
            "raw_carrier": raw_carrier,
            "capped_carrier": capped_carrier,
            "readout_preactivation": readout_preactivation,
            "readout_post_gelu": readout_post_gelu,
            "row_factors": row_factors,
            "column_factors": column_factors,
            "color_factors": color_factors,
            "compact_reconciliation_max_abs": jnp.max(
                jnp.abs(reconstructed_compact - compact), axis=(1, 2)
            ),
            "readout_uncapped_delta_l2": jnp.linalg.vector_norm(
                compact - uncapped_compact, axis=-1
            ),
            "query_capped_residual_l2": jnp.linalg.vector_norm(
                reasoning_query[1] - capped_reasoning, axis=-1
            ),
            "query_uncapped_delta_l2": jnp.linalg.vector_norm(
                reasoning_query[1] - uncapped_reasoning, axis=-1
            ),
        }

    return measure


def _measurement_report(values: Mapping[str, Any]) -> dict[str, Any]:
    cross_entropy = np.asarray(values["cross_entropy"], dtype=np.float64)
    logits = np.asarray(values["logits"])
    raw = np.asarray(values["raw_carrier"])
    capped = np.asarray(values["capped_carrier"])
    raw_norms = np.linalg.norm(raw.astype(np.float64), axis=-1)
    capped_norms = np.linalg.norm(capped.astype(np.float64), axis=-1)
    capped_count = int(np.count_nonzero(raw_norms > 1.0))
    sample_count = int(raw_norms.size)
    projection_telemetry: dict[str, list[dict[str, float]]] = {}
    for name in (
        "readout_preactivation",
        "readout_post_gelu",
        "row_factors",
        "column_factors",
        "color_factors",
    ):
        array = np.asarray(values[name], dtype=np.float64)
        projection_telemetry[name] = [
            {
                "rms": float(np.sqrt(np.mean(np.square(array[depth])))),
                "max_abs": float(np.max(np.abs(array[depth]), initial=0.0)),
                "nonzero_fraction": float(np.count_nonzero(array[depth]) / array[depth].size),
            }
            for depth in range(array.shape[0])
        ]
    reconciliation = np.asarray(
        values["compact_reconciliation_max_abs"], dtype=np.float64
    )
    readout_uncapped = np.asarray(
        values["readout_uncapped_delta_l2"], dtype=np.float64
    )
    query_residual = np.asarray(
        values["query_capped_residual_l2"], dtype=np.float64
    )
    query_uncapped = np.asarray(
        values["query_uncapped_delta_l2"], dtype=np.float64
    )
    return {
        "cross_entropy": cross_entropy.tolist(),
        "max_abs_color_logit": np.max(
            np.abs(logits), axis=tuple(range(1, logits.ndim))
        ).astype(np.float64).tolist(),
        "carrier": {
            "sample_count": sample_count,
            "raw_max_l2_norm": float(raw_norms.max(initial=0.0)),
            "capped_max_l2_norm": float(capped_norms.max(initial=0.0)),
            "capped_count": capped_count,
            "capped_fraction": capped_count / sample_count,
        },
        "finite": {
            "cross_entropies": bool(np.isfinite(cross_entropy).all()),
            "color_logits": bool(np.isfinite(logits).all()),
            "raw_carriers": bool(np.isfinite(raw).all()),
            "capped_carriers": bool(np.isfinite(capped).all()),
            "decoder_factors": bool(
                all(
                    np.isfinite(np.asarray(values[name])).all()
                    for name in projection_telemetry
                )
                and np.isfinite(reconciliation).all()
            ),
        },
        "projection_telemetry": projection_telemetry,
        "compact_reconciliation_max_abs": reconciliation.tolist(),
        "consumer_witnesses": {
            "readout_capped_residual_max_abs": float(
                reconciliation.max(initial=0.0)
            ),
            "readout_uncapped_delta_min_l2": float(
                readout_uncapped.min(initial=np.inf)
            ),
            "query_capped_residual_max_l2": float(
                query_residual.max(initial=0.0)
            ),
            "query_uncapped_delta_min_l2": float(
                query_uncapped.min(initial=np.inf)
            ),
            "sample_count": int(readout_uncapped.size + query_residual.size),
        },
        "prediction_histograms": [
            np.bincount(
                np.argmax(logits[depth], axis=-1), minlength=legacy.COLOR_COUNT
            ).tolist()
            for depth in range(logits.shape[0])
        ],
    }


def _reports_from_vector(
    values: Any,
    paths: Sequence[str],
    counts: Mapping[str, int],
    *,
    count_name: str,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape[-1:] != (len(paths),):
        raise ValueError("telemetry vector does not match required paths")
    vector = vector.reshape(-1, len(paths))[-1]
    return {
        path: {
            "l2_norm": float(vector[index]),
            count_name: int(counts[path]),
        }
        for index, path in enumerate(paths)
    }


def _parameter_delta_reports(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for path in _STAGE21_OPTIMIZATION_PATHS:
        squared = 0.0
        count = 0
        for left, right in zip(
            jax.tree.leaves(before[path]), jax.tree.leaves(after[path]), strict=True
        ):
            delta = np.asarray(right, dtype=np.float64) - np.asarray(
                left, dtype=np.float64
            )
            squared += float(np.sum(delta * delta))
            count += int(delta.size)
        reports[path] = {
            "l2_norm": math.sqrt(squared),
            "parameter_count": count,
        }
    return reports


def _initialization_report(
    parameters: Mapping[str, Any], config: BindingGateConfig
) -> dict[str, Any]:
    return {
        "fresh_model": True,
        "model_seed": config.model_seed,
        "parameter_sha256": legacy._array_digest(parameters),
        "parameter_count": int(
            sum(
                np.asarray(leaf).size
                for value in parameters.values()
                for leaf in jax.tree.leaves(value)
            )
        ),
    }


def _scalar_step(value: Any, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise RuntimeError(f"{label} is not one scalar integer step")
    return int(array)


def _adam_factor_reports(trainer: _PPPropTrainer) -> dict[str, Any]:
    optimizer = trainer.optimizer
    state = optimizer.opt_state.value
    if (
        not isinstance(state, tuple)
        or len(state) != 2
        or getattr(state[0], "_fields", ()) != ("count", "mu", "nu")
        or getattr(state[1], "_fields", ()) != ("count",)
        or not hasattr(state[0], "mu")
        or not hasattr(state[0], "nu")
        or not hasattr(state[0], "count")
        or not hasattr(state[1], "count")
    ):
        raise RuntimeError("Adam optimizer state does not expose first/second moments")
    adam = state[0]
    if not isinstance(adam.mu, Mapping) or not isinstance(adam.nu, Mapping):
        raise RuntimeError("Adam moments are not keyed by learner parameter paths")
    learner_keys = set(trainer.learner.param_states.keys())
    if (
        set(adam.mu) != learner_keys
        or set(adam.nu) != learner_keys
    ):
        raise RuntimeError("Adam moments do not exactly match learner parameter keys")
    adam_step = _scalar_step(adam.count, "Adam moment count")
    schedule_step = _scalar_step(state[1].count, "Adam schedule count")
    optimizer_step = _scalar_step(optimizer.step_count.value, "optimizer step count")
    reports = {}
    for path in _STAGE21_OPTIMIZATION_PATHS:
        learner_key = trainer.parameter_keys[path]
        if learner_key not in adam.mu or learner_key not in adam.nu:
            raise RuntimeError(f"Adam moments omit exact learner key for {path!r}")
        first_value = adam.mu[learner_key]
        second_value = adam.nu[learner_key]
        parameter_value = trainer.learner.param_states[learner_key].value
        parameter_structure = jax.tree.structure(parameter_value)
        if (
            jax.tree.structure(first_value) != parameter_structure
            or jax.tree.structure(second_value) != parameter_structure
        ):
            raise RuntimeError(f"Adam moment structure differs for {path!r}")
        first_leaves = [
            np.asarray(jax.device_get(leaf), dtype=np.float64)
            for leaf in jax.tree.leaves(first_value)
        ]
        second_leaves = [
            np.asarray(jax.device_get(leaf), dtype=np.float64)
            for leaf in jax.tree.leaves(second_value)
        ]
        first_squared = math.fsum(
            float(np.sum(leaf * leaf)) for leaf in first_leaves
        )
        second_squared = math.fsum(
            float(np.sum(leaf * leaf)) for leaf in second_leaves
        )
        reports[path] = {
            "first_moment_l2_norm": math.sqrt(first_squared),
            "second_moment_l2_norm": math.sqrt(second_squared),
            "first_moment_count": int(sum(leaf.size for leaf in first_leaves)),
            "second_moment_count": int(sum(leaf.size for leaf in second_leaves)),
            "adam_step": adam_step,
            "schedule_step": schedule_step,
            "optimizer_step": optimizer_step,
        }
    return reports


def _production_prefix_data(
    full_data: BindingData, updates: int
) -> BindingData:
    return BindingData(
        training_events=full_data.training_events[:updates],
        training_targets=full_data.training_targets[:updates],
        training_mapping_ids=full_data.training_mapping_ids[:updates],
        training_query_indices=full_data.training_query_indices[:updates],
        validation_intact=full_data.validation_intact,
        validation_shuffled=full_data.validation_shuffled,
        validation_no_context=full_data.validation_no_context,
        validation_targets=full_data.validation_targets,
        validation_mapping_ids=full_data.validation_mapping_ids,
        validation_query_indices=full_data.validation_query_indices,
    )


def _encode_admission_episodes(
    mapping_ids: np.ndarray,
    orders: np.ndarray,
    queries: np.ndarray,
    *,
    config: BindingGateConfig,
    controls: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    count = int(mapping_ids.size)
    rows = config.row_config
    intact = np.zeros(
        (count, config.sequence_length, rows.input_width), dtype=np.float32
    )
    shuffled = np.zeros_like(intact) if controls else None
    targets = np.zeros((count,), dtype=np.int32)
    for episode_index, mapping_id in enumerate(mapping_ids):
        input_colors, output_colors = legacy._decode_mapping(int(mapping_id))
        order = orders[episode_index]
        demonstrations = tuple(
            ArcPair(
                legacy._one_cell(input_colors[index]),
                legacy._one_cell(output_colors[index]),
            )
            for index in order
        )
        query_index = int(queries[episode_index])
        query = ArcPair(
            legacy._one_cell(input_colors[query_index]),
            legacy._one_cell(output_colors[query_index]),
        )
        task = ArcTask(train=demonstrations, test=(query,))
        intact[episode_index, : rows.max_events] = encode_query_episode(
            task, 0, rows
        ).events
        targets[episode_index] = output_colors[query_index]
        if shuffled is not None:
            rotated = tuple(
                ArcPair(pair.input, demonstrations[(index + 1) % SYMBOL_COUNT].output)
                for index, pair in enumerate(demonstrations)
            )
            shuffled[episode_index, : rows.max_events] = encode_query_episode(
                ArcTask(train=rotated, test=(query,)), 0, rows
            ).events
    no_context = None
    if shuffled is not None:
        no_context = np.array(intact, copy=True)
        no_context[:, :SYMBOL_COUNT] = 0.0
    return intact, targets, shuffled, no_context


def _build_stage21_admission_data() -> BindingData:
    full = BindingGateConfig.stage21_one_update_config()
    stability = BindingGateConfig.stage21_stability_config()
    full_training_count = full.training_episode_count
    prefix_count = stability.training_episode_count
    all_ids = legacy._affine_mapping_ids(
        full_training_count + full.validation_episodes, seed=full.split_seed
    )
    training_ids = all_ids[:prefix_count]
    validation_ids = all_ids[
        full_training_count : full_training_count + full.validation_episodes
    ]
    all_orders, all_queries = legacy._episode_choices(
        full_training_count, full.train_episode_seed
    )
    training_orders = all_orders[:prefix_count]
    training_queries = all_queries[:prefix_count]
    train, train_targets, _, _ = _encode_admission_episodes(
        training_ids,
        training_orders,
        training_queries,
        config=stability,
        controls=False,
    )
    validation_orders, validation_queries = legacy._episode_choices(
        full.validation_episodes, full.validation_episode_seed
    )
    intact, validation_targets, shuffled, no_context = _encode_admission_episodes(
        validation_ids,
        validation_orders,
        validation_queries,
        config=stability,
        controls=True,
    )
    assert shuffled is not None and no_context is not None
    return BindingData(
        training_events=legacy._readonly(
            train.reshape(
                stability.training_updates,
                stability.batch_size,
                stability.sequence_length,
                stability.row_config.input_width,
            ).transpose(0, 2, 1, 3)
        ),
        training_targets=legacy._readonly(
            train_targets.reshape(stability.training_updates, stability.batch_size)
        ),
        training_mapping_ids=legacy._readonly(
            training_ids.reshape(stability.training_updates, stability.batch_size)
        ),
        training_query_indices=legacy._readonly(
            training_queries.reshape(stability.training_updates, stability.batch_size)
        ),
        validation_intact=legacy._readonly(intact.transpose(1, 0, 2)),
        validation_shuffled=legacy._readonly(shuffled.transpose(1, 0, 2)),
        validation_no_context=legacy._readonly(no_context.transpose(1, 0, 2)),
        validation_targets=legacy._readonly(validation_targets),
        validation_mapping_ids=legacy._readonly(validation_ids),
        validation_query_indices=legacy._readonly(validation_queries),
    )


def _stability_schedule_report(data: BindingData) -> dict[str, str]:
    return {
        "training_schedule_sha256": legacy._digest_arrays(
            data.training_events,
            data.training_targets,
            data.training_mapping_ids,
        ),
        "training_query_indices_sha256": legacy._digest_arrays(
            data.training_query_indices
        ),
        "validation_schedule_sha256": legacy._digest_arrays(
            data.validation_intact,
            data.validation_targets,
            data.validation_mapping_ids,
        ),
        "training_mapping_ids_sha256": legacy._digest_arrays(
            data.training_mapping_ids.reshape(-1)
        ),
        "validation_mapping_ids_sha256": legacy._digest_arrays(
            data.validation_mapping_ids
        ),
    }


def _run_one_update_admission(
    full_data: BindingData,
    config: BindingGateConfig,
) -> dict[str, Any]:
    model = LatentWorkspaceModel(_model_config(config, batch_size=config.batch_size))
    architecture = _architecture_report(model)
    measure = _make_batch_measurement(model, config)
    events = jnp.asarray(full_data.training_events[0])
    targets = jnp.asarray(full_data.training_targets[0])
    pre = _measurement_report(jax.block_until_ready(measure(events, targets)))
    trainer = _make_pp_prop_trainer(model, config)
    before = legacy._parameter_values(model)
    telemetry = jax.block_until_ready(
        trainer.train(events[None, ...], targets[None, ...])
    )
    after = legacy._parameter_values(model)
    post = _measurement_report(jax.block_until_ready(measure(events, targets)))
    gradient_reports = _reports_from_vector(
        telemetry["gradient_norms"],
        _STAGE21_OPTIMIZATION_PATHS,
        trainer.parameter_counts,
        count_name="parameter_count",
    )
    trace_reports = _reports_from_vector(
        telemetry["trace_norms"],
        _STAGE21_TRACE_PATHS,
        trainer.trace_factor_counts,
        count_name="factor_count",
    )
    report = {
        "schema_version": STAGE21_ADMISSION_SCHEMA_VERSION,
        "control": "example21_stage21_one_update_admission",
        "target": "one_update",
        "executed_updates": 1,
        "source_training_updates": config.training_updates,
        "batch_size": config.batch_size,
        "configuration_scale": config.configuration_scale,
        "learner": "pp_prop_only",
        "optimizer": "Adam",
        "config": {
            **dataclasses.asdict(config),
            "configuration_scale": config.configuration_scale,
        },
        "data": {
            "training_schedule_sha256": PREREGISTERED_TRAINING_SCHEDULE_SHA256,
            "update_zero_event_sha256": legacy._digest_arrays(
                full_data.training_events[0]
            ),
            "update_zero_target_sha256": legacy._digest_arrays(
                full_data.training_targets[0]
            ),
            "update_zero_mapping_sha256": legacy._digest_arrays(
                full_data.training_mapping_ids[0]
            ),
            "update_zero_query_sha256": legacy._digest_arrays(
                full_data.training_query_indices[0]
            ),
            "update_zero_episode_count": config.batch_size,
            "rng_source_episode_count": config.training_episode_count,
        },
        "initialization": _initialization_report(before, config),
        "architecture": architecture,
        "depths": {
            str(depth): {
                "pre_cross_entropy": pre["cross_entropy"][depth],
                "post_cross_entropy": post["cross_entropy"][depth],
                "pre_max_abs_color_logit": pre["max_abs_color_logit"][depth],
                "post_max_abs_color_logit": post["max_abs_color_logit"][depth],
            }
            for depth in range(config.gap_steps + 1)
        },
        "carrier": {"pre": pre["carrier"], "post": post["carrier"]},
        "pre_measurement": {
            "projection_telemetry": pre["projection_telemetry"],
            "compact_reconciliation_max_abs": pre[
                "compact_reconciliation_max_abs"
            ],
            "consumer_witnesses": pre["consumer_witnesses"],
        },
        "post_measurement": {
            "projection_telemetry": post["projection_telemetry"],
            "compact_reconciliation_max_abs": post[
                "compact_reconciliation_max_abs"
            ],
            "consumer_witnesses": post["consumer_witnesses"],
        },
        "gradient_group_norms": gradient_reports,
        "pp_prop_factor_group_norms": trace_reports,
        "adam_factor_group_norms": _adam_factor_reports(trainer),
        "parameter_update_group_norms": _parameter_delta_reports(before, after),
        "finite_telemetry": {
            **{
                name: bool(pre["finite"][name] and post["finite"][name])
                for name in (
                    "cross_entropies",
                    "color_logits",
                    "raw_carriers",
                    "capped_carriers",
                    "decoder_factors",
                )
            },
            "gradients": bool(np.asarray(telemetry["gradients_finite"]).all()),
            "pp_prop_factors": bool(np.asarray(telemetry["traces_finite"]).all()),
            "adam_factors": bool(np.asarray(telemetry["adam_finite"]).all()),
            "parameter_updates": _numeric_tree_is_finite(
                _parameter_delta_reports(before, after)
            ),
        },
    }
    report["qualification"] = _one_update_admission_qualification(report)
    return report


def _telemetry_summary(
    telemetry: Mapping[str, Any], losses: np.ndarray
) -> dict[str, Any]:
    definitions = {
        "losses": (losses, np.isfinite(losses)),
        "states": (telemetry["state_max_abs"], telemetry["states_finite"]),
        "logits": (telemetry["logit_max_abs"], telemetry["logits_finite"]),
        "gradients": (
            telemetry["gradient_max_abs"],
            telemetry["gradients_finite"],
        ),
        "pp_prop_factors": (
            telemetry["trace_max_abs"],
            telemetry["traces_finite"],
        ),
        "adam_factors": (telemetry["adam_max_abs"], telemetry["adam_finite"]),
        "parameters": (
            telemetry["parameter_max_abs"],
            telemetry["parameters_finite"],
        ),
    }
    return {
        name: {
            "observed_count": int(np.asarray(values).size),
            "max_abs": float(np.max(np.abs(np.asarray(values)), initial=0.0)),
            "finite": bool(np.asarray(finite).all()),
        }
        for name, (values, finite) in definitions.items()
    }


def _run_stability_admission(
    full_data: BindingData,
    config: BindingGateConfig,
) -> dict[str, Any]:
    data = _production_prefix_data(full_data, STAGE21_STABILITY_UPDATES)
    model = LatentWorkspaceModel(_model_config(config, batch_size=config.batch_size))
    architecture = _architecture_report(model)
    initial_parameters = legacy._parameter_values(model)
    measure = _make_batch_measurement(model, config)
    initial = _measurement_report(
        jax.block_until_ready(
            measure(
                jnp.asarray(data.training_events[0]),
                jnp.asarray(data.training_targets[0]),
            )
        )
    )
    trainer = _make_pp_prop_trainer(model, config)
    telemetry = jax.block_until_ready(
        trainer.train(
            jnp.asarray(data.training_events), jnp.asarray(data.training_targets)
        )
    )
    losses = np.asarray(telemetry["loss"], dtype=np.float64)
    evaluation, diagnostics = _evaluate_model(model, data, config)
    summaries = _telemetry_summary(telemetry, losses)
    finite = {name: value["finite"] for name, value in summaries.items()}
    report = {
        "schema_version": STAGE21_ADMISSION_SCHEMA_VERSION,
        "control": "example21_stage21_stability_256_admission",
        "target": "stability_256",
        "training_updates": config.training_updates,
        "batch_size": config.batch_size,
        "validation_episodes": config.validation_episodes,
        "configuration_scale": config.configuration_scale,
        "qualification_regime": config.qualification_regime,
        "learner": "pp_prop_only",
        "optimizer": "Adam",
        "config": {
            **dataclasses.asdict(config),
            "configuration_scale": config.configuration_scale,
            "qualification_regime": config.qualification_regime,
        },
        "data": _stability_schedule_report(data),
        "architecture": architecture,
        "initialization": {
            **_initialization_report(initial_parameters, config),
            "update_zero_event_sha256": legacy._digest_arrays(
                data.training_events[0]
            ),
            "update_zero_target_sha256": legacy._digest_arrays(
                data.training_targets[0]
            ),
        },
        "losses": losses.tolist(),
        "initial_depth_cross_entropy": {
            str(depth): initial["cross_entropy"][depth]
            for depth in range(config.gap_steps + 1)
        },
        "tail_64_mean_loss": float(losses[-64:].mean()),
        "held_out_intact_by_depth": {
            str(depth): {
                "count": config.validation_episodes,
                "prediction_histogram": evaluation["depths"][str(depth)]["intact"][
                    "prediction_histogram"
                ],
                "unique_predicted_colors": int(
                    np.count_nonzero(
                        evaluation["depths"][str(depth)]["intact"][
                            "prediction_histogram"
                        ]
                    )
                ),
            }
            for depth in range(config.gap_steps + 1)
        },
        "finite_telemetry": finite,
        "telemetry_summaries": summaries,
        "evaluation_all_compact_logits_finite": evaluation[
            "all_compact_logits_finite"
        ],
        "evaluation_all_state_tensors_finite": diagnostics[
            "all_state_tensors_finite"
        ],
    }
    report["qualification"] = _stability_admission_qualification(report)
    return report


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


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and math.isfinite(float(value))


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_boolean(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or (
            value.dtype.kind == "O" and _contains_boolean(value.tolist())
        )
    return False


def _real_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if _contains_boolean(value):
        raise TypeError("numeric evidence cannot contain boolean values")
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("numeric evidence must contain non-boolean real values")
    return array.astype(dtype, copy=False)


def _json_exact(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            left, allow_nan=False, sort_keys=True, separators=(",", ":")
        ) == json.dumps(
            right, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _training_evidence_complete(
    training: Mapping[str, Any], config: BindingGateConfig
) -> bool:
    losses = _real_array(training["losses"])
    if (
        losses.ndim != 1
        or losses.size != config.training_updates
        or not np.isfinite(losses).all()
    ):
        return False
    scalar_names = (
        "initial_loss",
        "final_loss",
        "tail_64_mean_loss",
        "supervision_weight_sum",
    )
    if not all(_is_finite_real(training[name]) for name in scalar_names):
        return False
    initial = float(training["initial_loss"])
    final = float(training["final_loss"])
    tail = float(training["tail_64_mean_loss"])
    expected_tail = float(losses[-min(64, losses.size) :].mean())
    supervision_mask = _real_array(training["supervision_mask"], dtype=np.float32)
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
    if not all(
        _is_finite_real(metric[name])
        for name in ("accuracy", "wilson_95_lower", "wilson_95_upper")
    ):
        return False
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
        or not _is_integer(evaluation["reported_checkpoint"])
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
    if any(not _json_exact(evaluation[arm], final[arm]) for arm in final):
        return False
    intact_accuracy = float(final["intact"]["accuracy"])
    shuffled_accuracy = float(final["shuffled"]["accuracy"])
    no_context_accuracy = float(final["no_context"]["accuracy"])
    direct_scalars = (
        "intact_minus_shuffled",
        "intact_minus_no_context",
        "pairing_chance",
        "unconditional_color_chance",
    )
    if not all(_is_finite_real(evaluation[name]) for name in direct_scalars):
        return False
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
        and _is_finite_real(report["mean_l2_difference"])
        and float(report["mean_l2_difference"]) > 0.0
        and _is_finite_real(report["no_context_l2_norm"])
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
        or not _is_integer(memory["applicable_count"])
        or int(memory["applicable_count"]) != config.validation_episodes
        or not _is_integer(memory["different_count"])
        or not _is_integer(memory["intact_shuffled_different_count"])
        or int(memory["different_count"])
        != int(memory["intact_shuffled_different_count"])
        or memory["every_pair_differs"]
        is not memory["every_intact_shuffled_pair_differs"]
        or memory["no_context_exact_zero"] is not True
        or not all(
            _is_finite_real(memory[name])
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
    gram = _real_array(report["gram"])
    scalar_names = (
        "diagonal_minimum",
        "off_diagonal_maximum",
        "separation_margin",
        "worst_global_margin",
        "required_margin",
        "zero_event_key_max_abs",
    )
    if (
        report["protocol"] != protocol
        or not _is_integer(report["row_event_input_width"])
        or int(report["row_event_input_width"]) != row_event_input_width
        or not _is_integer(report["raw_key_feature_width"])
        or int(report["raw_key_feature_width"]) != raw_key_feature_width
        or not _is_integer(report["memory_width"])
        or not _is_integer(report["model_seed"])
        or not _architecture_digests_valid(report["architecture"])
        or not _json_exact(
            report["candidate_colors"], list(range(STRUCTURAL_COLOR_COUNT))
        )
        or not _is_integer(report["color_count"])
        or int(report["color_count"]) != STRUCTURAL_COLOR_COUNT
        or report["gram_shape"]
        != [STRUCTURAL_COLOR_COUNT, STRUCTURAL_COLOR_COUNT]
        or gram.shape != (STRUCTURAL_COLOR_COUNT, STRUCTURAL_COLOR_COUNT)
        or not np.isfinite(gram).all()
        or not all(_is_finite_real(report[name]) for name in scalar_names)
        or not np.allclose(gram, gram.T, rtol=0.0, atol=1e-12)
    ):
        return False
    diagonal = np.diag(gram)
    off_diagonal = gram[~np.eye(STRUCTURAL_COLOR_COUNT, dtype=bool)]
    diagonal_minimum = float(diagonal.min())
    off_diagonal_maximum = float(off_diagonal.max())
    margin = diagonal_minimum - off_diagonal_maximum
    reported_diagonal = _real_array(report["diagonal"])
    return bool(
        reported_diagonal.shape == (STRUCTURAL_COLOR_COUNT,)
        and np.allclose(
            reported_diagonal,
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
        and source["head_command_succeeded"] is True
        and source["asserted_dirty"] is False
        and source["asserted_dirty_matches_worktree"] is True
        and source["status_command_succeeded"] is True
        and source["verified"] is True
        and source["dirty"] is False
    )


def _carrier_architecture_identity(architecture: Mapping[str, Any]) -> bool:
    radius = architecture.get("carrier_radius")
    return bool(
        architecture.get("carrier_stabilizer")
        == "per_example_stopped_unit_l2_cap"
        and _is_finite_real(radius)
        and float(radius) == 1.0
        and tuple(architecture.get("carrier_consumers", ()))
        == ("readout_projection", "workspace_query_projection")
    )


def _preregistered_gpu_initialization(
    initialization: Mapping[str, Any], *, extra_keys: Sequence[str] = ()
) -> bool:
    expected_keys = {
        "fresh_model",
        "model_seed",
        "parameter_sha256",
        "parameter_count",
        *extra_keys,
    }
    return bool(
        set(initialization) == expected_keys
        and initialization["fresh_model"] is True
        and _is_integer(initialization["model_seed"])
        and int(initialization["model_seed"]) == 2108
        and initialization["parameter_sha256"]
        == PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256
        and _is_integer(initialization["parameter_count"])
        and int(initialization["parameter_count"])
        == PREREGISTERED_PARAMETER_COUNT
    )


def _exact_admission_config(
    value: Mapping[str, Any], config: BindingGateConfig, *, include_regime: bool
) -> bool:
    expected: dict[str, Any] = {
        **dataclasses.asdict(config),
        "configuration_scale": "production_topology",
    }
    if include_regime:
        expected["qualification_regime"] = "nonqualifying_abbreviated"
    return _json_exact(value, expected)


def _complete_positive_norm_reports(
    reports: Mapping[str, Any],
    expected_counts: Mapping[str, int],
    *,
    count_name: str,
) -> bool:
    if set(reports) != set(expected_counts):
        return False
    for path, expected_count in expected_counts.items():
        report = reports[path]
        if not isinstance(report, Mapping):
            return False
        norm_value = report["l2_norm"]
        count = report[count_name]
        if (
            not _is_finite_real(norm_value)
            or float(norm_value) <= 0.0
            or not _is_integer(count)
            or int(count) != expected_count
        ):
            return False
    return True


def _complete_adam_reports(
    reports: Mapping[str, Any], gradients: Mapping[str, Any]
) -> bool:
    if set(reports) != set(_STAGE21_OPTIMIZATION_PATHS):
        return False
    expected = {
        "first_moment_l2_norm",
        "second_moment_l2_norm",
        "first_moment_count",
        "second_moment_count",
        "adam_step",
        "schedule_step",
        "optimizer_step",
    }
    for path in _STAGE21_OPTIMIZATION_PATHS:
        report = reports[path]
        if not isinstance(report, Mapping) or set(report) != expected:
            return False
        first_value = report["first_moment_l2_norm"]
        second_value = report["second_moment_l2_norm"]
        parameter_count = _STAGE21_PARAMETER_COUNTS[path]
        if (
            not _is_finite_real(first_value)
            or not _is_finite_real(second_value)
            or float(first_value) <= 0.0
            or float(second_value) <= 0.0
            or not _is_integer(report["first_moment_count"])
            or int(report["first_moment_count"]) != parameter_count
            or not _is_integer(report["second_moment_count"])
            or int(report["second_moment_count"]) != parameter_count
            or not _is_integer(gradients[path]["parameter_count"])
            or int(gradients[path]["parameter_count"]) != parameter_count
            or not _is_integer(report["adam_step"])
            or int(report["adam_step"]) != 1
            or not _is_integer(report["schedule_step"])
            or int(report["schedule_step"]) != 1
            or not _is_integer(report["optimizer_step"])
            or int(report["optimizer_step"]) != 1
        ):
            return False
    return True


def _carrier_measurement_complete(report: Mapping[str, Any]) -> bool:
    sample_count = report["sample_count"]
    capped_count = report["capped_count"]
    if (
        not _is_integer(sample_count)
        or int(sample_count) != 2 * 64
        or not _is_integer(capped_count)
        or not 0 <= int(capped_count) <= int(sample_count)
    ):
        return False
    scalar_names = (
        "capped_fraction",
        "raw_max_l2_norm",
        "capped_max_l2_norm",
    )
    if not all(_is_finite_real(report[name]) for name in scalar_names):
        return False
    fraction = float(report["capped_fraction"])
    expected_fraction = int(capped_count) / int(sample_count)
    return bool(
        _numeric_tree_is_finite(report)
        and math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-12)
        and float(report["raw_max_l2_norm"]) >= 0.0
        and float(report["capped_max_l2_norm"]) >= 0.0
    )


def _decoder_measurement_complete(report: Mapping[str, Any]) -> bool:
    expected = {
        "readout_preactivation",
        "readout_post_gelu",
        "row_factors",
        "column_factors",
        "color_factors",
    }
    telemetry = report["projection_telemetry"]
    reconciliation = _real_array(report["compact_reconciliation_max_abs"])
    witnesses = report["consumer_witnesses"]
    if set(telemetry) != expected or reconciliation.shape != (2,):
        return False
    if not np.isfinite(reconciliation).all() or np.any(
        reconciliation > STAGE21_DECODER_REPLAY_ATOL
    ):
        return False
    if set(witnesses) != {
        "readout_capped_residual_max_abs",
        "readout_uncapped_delta_min_l2",
        "query_capped_residual_max_l2",
        "query_uncapped_delta_min_l2",
        "sample_count",
    }:
        return False
    if (
        not _is_integer(witnesses["sample_count"])
        or int(witnesses["sample_count"]) != 3 * 64
        or not _numeric_tree_is_finite(witnesses)
        or not all(
            _is_finite_real(witnesses[name])
            for name in (
                "readout_capped_residual_max_abs",
                "readout_uncapped_delta_min_l2",
                "query_capped_residual_max_l2",
                "query_uncapped_delta_min_l2",
            )
        )
        or float(witnesses["readout_capped_residual_max_abs"])
        > STAGE21_DECODER_REPLAY_ATOL
        or float(witnesses["query_capped_residual_max_l2"]) > 1e-6
        or float(witnesses["readout_uncapped_delta_min_l2"]) <= 0.0
        or float(witnesses["query_uncapped_delta_min_l2"]) <= 0.0
    ):
        return False
    for values in telemetry.values():
        if not isinstance(values, list) or len(values) != 2:
            return False
        for value in values:
            if set(value) != {"rms", "max_abs", "nonzero_fraction"}:
                return False
            if not all(
                _is_finite_real(value[name])
                for name in ("rms", "max_abs", "nonzero_fraction")
            ):
                return False
            rms = float(value["rms"])
            maximum = float(value["max_abs"])
            fraction = float(value["nonzero_fraction"])
            if (
                not all(math.isfinite(item) for item in (rms, maximum, fraction))
                or rms <= 0.0
                or maximum <= 0.0
                or not 0.0 < fraction <= 1.0
            ):
                return False
    return True


def _one_update_admission_qualification(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "schema_and_target": False,
        "fixed_production_configuration": False,
        "exact_production_rng_prefix": False,
        "carrier_architecture_identity": False,
        "finite_telemetry": False,
        "post_cross_entropy_envelope": False,
        "post_logit_envelope": False,
        "carrier_cap_nonvacuous": False,
        "carrier_norm_envelope": False,
        "required_gradient_groups_nonzero": False,
        "required_pp_prop_factors_nonzero": False,
        "required_adam_factors_nonzero": False,
        "required_parameter_updates_nonzero": False,
        "decoder_carrier_signal_finite_nonzero": False,
    }
    try:
        data = report["data"]
        initialization = report["initialization"]
        architecture = report["architecture"]
        depths = report["depths"]
        carrier = report["carrier"]
        finite = report["finite_telemetry"]
        hashes = (
            data["update_zero_event_sha256"],
            data["update_zero_target_sha256"],
            data["update_zero_mapping_sha256"],
            data["update_zero_query_sha256"],
        )
        depth_complete = set(depths) == {"0", "1"} and all(
            set(depths[depth])
            == {
                "pre_cross_entropy",
                "post_cross_entropy",
                "pre_max_abs_color_logit",
                "post_max_abs_color_logit",
            }
            for depth in ("0", "1")
        )
        depth_scalars_complete = depth_complete and all(
            _is_finite_real(depths[depth][name])
            for depth in ("0", "1")
            for name in (
                "pre_cross_entropy",
                "post_cross_entropy",
                "pre_max_abs_color_logit",
                "post_max_abs_color_logit",
            )
        )
        carrier_complete = set(carrier) == {"pre", "post"} and all(
            _carrier_measurement_complete(carrier[phase])
            for phase in ("pre", "post")
        )
        criteria.update(
            {
                "schema_and_target": (
                    _is_integer(report["schema_version"])
                    and int(report["schema_version"])
                    == STAGE21_ADMISSION_SCHEMA_VERSION
                    and report["control"]
                    == "example21_stage21_one_update_admission"
                    and report["target"] == "one_update"
                ),
                "fixed_production_configuration": (
                    _is_integer(report["executed_updates"])
                    and int(report["executed_updates"]) == 1
                    and _is_integer(report["source_training_updates"])
                    and int(report["source_training_updates"]) == 10_000
                    and _is_integer(report["batch_size"])
                    and int(report["batch_size"]) == 64
                    and report["configuration_scale"] == "production_topology"
                    and report["learner"] == "pp_prop_only"
                    and report["optimizer"] == "Adam"
                    and _exact_admission_config(
                        report["config"],
                        BindingGateConfig.stage21_one_update_config(),
                        include_regime=False,
                    )
                    and _preregistered_gpu_initialization(initialization)
                ),
                "exact_production_rng_prefix": (
                    data["training_schedule_sha256"]
                    == PREREGISTERED_TRAINING_SCHEDULE_SHA256
                    and _is_integer(data["update_zero_episode_count"])
                    and int(data["update_zero_episode_count"]) == 64
                    and _is_integer(data["rng_source_episode_count"])
                    and int(data["rng_source_episode_count"]) == 640_000
                    and _json_exact(
                        dict(
                            zip(
                                PREREGISTERED_UPDATE_ZERO_DIGESTS,
                                hashes,
                                strict=True,
                            )
                        ),
                        PREREGISTERED_UPDATE_ZERO_DIGESTS,
                    )
                ),
                "carrier_architecture_identity": _carrier_architecture_identity(
                    architecture
                ),
                "finite_telemetry": (
                    depth_scalars_complete
                    and carrier_complete
                    and set(finite) == _STAGE21_FINITE_ONE_UPDATE
                    and all(value is True for value in finite.values())
                    and _numeric_tree_is_finite(report)
                ),
                "post_cross_entropy_envelope": depth_scalars_complete
                and all(
                    float(depths[depth]["post_cross_entropy"])
                    <= float(depths[depth]["pre_cross_entropy"]) + 1.0
                    for depth in ("0", "1")
                ),
                "post_logit_envelope": depth_scalars_complete
                and all(
                    float(depths[depth]["post_max_abs_color_logit"]) < 10.0
                    for depth in ("0", "1")
                ),
                "carrier_cap_nonvacuous": carrier_complete
                and max(
                    float(carrier[phase]["raw_max_l2_norm"])
                    for phase in ("pre", "post")
                )
                > 1.0
                and sum(
                    int(carrier[phase]["capped_count"])
                    for phase in ("pre", "post")
                )
                > 0,
                "carrier_norm_envelope": carrier_complete
                and all(
                    float(carrier[phase]["capped_max_l2_norm"])
                    <= 1.0 + STAGE21_CARRIER_NORM_TOLERANCE
                    for phase in ("pre", "post")
                ),
                "required_gradient_groups_nonzero": (
                    _complete_positive_norm_reports(
                        report["gradient_group_norms"],
                        _STAGE21_PARAMETER_COUNTS,
                        count_name="parameter_count",
                    )
                ),
                "required_pp_prop_factors_nonzero": (
                    _complete_positive_norm_reports(
                        report["pp_prop_factor_group_norms"],
                        _STAGE21_TRACE_FACTOR_COUNTS,
                        count_name="factor_count",
                    )
                ),
                "required_adam_factors_nonzero": (
                    _complete_adam_reports(
                        report["adam_factor_group_norms"],
                        report["gradient_group_norms"],
                    )
                ),
                "required_parameter_updates_nonzero": (
                    _complete_positive_norm_reports(
                        report["parameter_update_group_norms"],
                        _STAGE21_PARAMETER_COUNTS,
                        count_name="parameter_count",
                    )
                ),
                "decoder_carrier_signal_finite_nonzero": all(
                    _decoder_measurement_complete(report[phase])
                    for phase in ("pre_measurement", "post_measurement")
                ),
            }
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    passed = bool(all(criteria.values()))
    return {
        "passed": passed,
        "criteria": criteria,
        "interpretation": (
            "stage21_one_update_passed"
            if passed
            else "stage21_one_update_failed_stop_no_gate_a"
        ),
    }


def _stability_admission_qualification(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "schema_and_target": False,
        "fixed_production_configuration": False,
        "explicitly_nonqualifying": False,
        "carrier_architecture_identity": False,
        "schedule_digests_complete": False,
        "complete_finite_telemetry": False,
        "tail_64_descends_from_initial_depth_mean": False,
        "held_out_predictions_do_not_collapse": False,
    }
    try:
        losses = _real_array(report["losses"])
        initial = report["initial_depth_cross_entropy"]
        finite = report["finite_telemetry"]
        summaries = report["telemetry_summaries"]
        held_out = report["held_out_intact_by_depth"]
        data = report["data"]
        initialization = report["initialization"]
        initial_complete = (
            isinstance(initial, Mapping)
            and set(initial) == {"0", "1"}
            and all(_is_finite_real(initial[depth]) for depth in ("0", "1"))
        )
        expected_tail = float(losses[-64:].mean()) if losses.size >= 64 else float("nan")
        initial_mean = (
            (float(initial["0"]) + float(initial["1"])) / 2.0
            if initial_complete
            else float("nan")
        )
        prediction_complete = set(held_out) == {"0", "1"}
        if prediction_complete:
            for depth in ("0", "1"):
                evidence = held_out[depth]
                histogram_value = evidence["prediction_histogram"]
                if _contains_boolean(histogram_value):
                    prediction_complete = False
                    continue
                histogram = np.asarray(histogram_value)
                unique = int(np.count_nonzero(histogram))
                prediction_complete = bool(
                    prediction_complete
                    and _is_integer(evidence["count"])
                    and int(evidence["count"]) == 512
                    and histogram.shape == (legacy.COLOR_COUNT,)
                    and np.issubdtype(histogram.dtype, np.integer)
                    and np.all(histogram >= 0)
                    and int(histogram.sum()) == 512
                    and _is_integer(evidence["unique_predicted_colors"])
                    and int(evidence["unique_predicted_colors"]) == unique
                    and unique >= 2
                )
        expected_counts = {
            "losses": 256,
            "states": 256,
            "logits": 256,
            "gradients": 256 * len(_STAGE21_OPTIMIZATION_PATHS),
            "pp_prop_factors": 256 * len(_STAGE21_TRACE_PATHS),
            "adam_factors": 256,
            "parameters": 256,
        }
        summaries_complete = set(summaries) == set(expected_counts)
        if summaries_complete:
            summaries_complete = all(
                set(summaries[name]) == {"observed_count", "max_abs", "finite"}
                and _is_integer(summaries[name]["observed_count"])
                and int(summaries[name]["observed_count"])
                == expected_counts[name]
                and _is_finite_real(summaries[name]["max_abs"])
                and float(summaries[name]["max_abs"]) >= 0.0
                and summaries[name]["finite"] is True
                for name in expected_counts
            )
        if summaries_complete:
            summaries_complete = all(
                float(summaries[name]["max_abs"]) > 0.0
                for name in (
                    "gradients",
                    "pp_prop_factors",
                    "adam_factors",
                    "parameters",
                )
            )
        initialization_complete = bool(
            _preregistered_gpu_initialization(
                initialization,
                extra_keys=(
                    "update_zero_event_sha256",
                    "update_zero_target_sha256",
                ),
            )
            and initialization["update_zero_event_sha256"]
            == PREREGISTERED_UPDATE_ZERO_DIGESTS["update_zero_event_sha256"]
            and initialization["update_zero_target_sha256"]
            == PREREGISTERED_UPDATE_ZERO_DIGESTS["update_zero_target_sha256"]
        )
        criteria.update(
            {
                "schema_and_target": (
                    _is_integer(report["schema_version"])
                    and int(report["schema_version"])
                    == STAGE21_ADMISSION_SCHEMA_VERSION
                    and report["control"]
                    == "example21_stage21_stability_256_admission"
                    and report["target"] == "stability_256"
                ),
                "fixed_production_configuration": (
                    _is_integer(report["training_updates"])
                    and int(report["training_updates"])
                    == STAGE21_STABILITY_UPDATES
                    and _is_integer(report["batch_size"])
                    and int(report["batch_size"]) == 64
                    and _is_integer(report["validation_episodes"])
                    and int(report["validation_episodes"]) == 512
                    and report["configuration_scale"] == "production_topology"
                    and report["learner"] == "pp_prop_only"
                    and report["optimizer"] == "Adam"
                    and _exact_admission_config(
                        report["config"],
                        BindingGateConfig.stage21_stability_config(),
                        include_regime=True,
                    )
                ),
                "explicitly_nonqualifying": (
                    report["qualification_regime"] == "nonqualifying_abbreviated"
                ),
                "carrier_architecture_identity": _carrier_architecture_identity(
                    report["architecture"]
                ),
                "schedule_digests_complete": _json_exact(
                    data, PREREGISTERED_STABILITY_DIGESTS
                ),
                "complete_finite_telemetry": (
                    losses.shape == (STAGE21_STABILITY_UPDATES,)
                    and np.isfinite(losses).all()
                    and initial_complete
                    and set(finite) == _STAGE21_FINITE_STABILITY
                    and all(value is True for value in finite.values())
                    and summaries_complete
                    and initialization_complete
                    and report["evaluation_all_compact_logits_finite"] is True
                    and report["evaluation_all_state_tensors_finite"] is True
                    and _numeric_tree_is_finite(report)
                ),
                "tail_64_descends_from_initial_depth_mean": (
                    math.isfinite(expected_tail)
                    and _is_finite_real(report["tail_64_mean_loss"])
                    and math.isclose(
                        float(report["tail_64_mean_loss"]),
                        expected_tail,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        float(losses[0]),
                        initial_mean,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                    and expected_tail < initial_mean
                ),
                "held_out_predictions_do_not_collapse": prediction_complete,
            }
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    passed = bool(all(criteria.values()))
    return {
        "passed": passed,
        "criteria": criteria,
        "interpretation": (
            "stage21_stability_256_passed_nonqualifying"
            if passed
            else "stage21_stability_256_failed_stop_no_gate_a"
        ),
    }


_QUALIFICATION_CRITERIA = (
    "stage21_one_update_admitted",
    "stage21_stability_256_admitted",
    "formal_initialization_matches_admissions",
    "formal_schedule_matches_admissions",
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
    one_update_admission: Mapping[str, Any],
    stability_admission: Mapping[str, Any],
    initialization: Mapping[str, Any],
    data: Mapping[str, Any],
    config: BindingGateConfig,
) -> dict[str, Any]:
    criteria = {name: False for name in _QUALIFICATION_CRITERIA}
    try:
        intact = evaluation["intact"]
        shuffled = evaluation["shuffled"]
        pairing_value = evaluation["pairing_chance"]
        pairing_chance = (
            float(pairing_value) if _is_finite_real(pairing_value) else float("nan")
        )
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
            "stage21_one_update_admitted": (
                _one_update_admission_qualification(one_update_admission)["passed"]
                is True
            ),
            "stage21_stability_256_admitted": (
                _stability_admission_qualification(stability_admission)["passed"]
                is True
            ),
            "formal_initialization_matches_admissions": (
                _preregistered_gpu_initialization(initialization)
            ),
            "formal_schedule_matches_admissions": (
                data["training_schedule_sha256"]
                == PREREGISTERED_TRAINING_SCHEDULE_SHA256
                and data["validation_schedule_sha256"]
                == PREREGISTERED_STABILITY_DIGESTS["validation_schedule_sha256"]
            ),
            "held_out_count_at_least_256": (
                _is_integer(intact["count"])
                and int(intact["count"]) >= 256
                and int(intact["count"]) == config.validation_episodes
            ),
            "intact_accuracy_at_least_0_80": (
                _is_finite_real(intact["accuracy"])
                and float(intact["accuracy"]) >= 0.80
            ),
            "intact_wilson_lower_above_pairing_chance": (
                _is_finite_real(intact["wilson_95_lower"])
                and _is_finite_real(pairing_value)
                and float(intact["wilson_95_lower"]) > pairing_chance
            ),
            "intact_minus_shuffled_at_least_0_25": (
                _is_finite_real(evaluation["intact_minus_shuffled"])
                and float(evaluation["intact_minus_shuffled"]) >= 0.25
            ),
            "shuffled_not_demonstrably_above_pairing_chance": (
                _is_finite_real(shuffled["wilson_95_lower"])
                and _is_finite_real(pairing_value)
                and float(shuffled["wilson_95_lower"]) <= pairing_chance
            ),
            "exact_marginal_equality": marginals["exact_marginal_equality"] is True,
            "every_intact_shuffled_memory_differs": (
                memory["every_intact_shuffled_pair_differs"] is True
                and _is_integer(memory["intact_shuffled_different_count"])
                and _is_integer(memory["applicable_count"])
                and _is_integer(intact["count"])
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
                    and _is_finite_real(movements[path]["l2_delta"])
                    and float(movements[path]["l2_delta"]) > 0.0
                    and _is_integer(movements[path]["parameter_count"])
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
                and _is_integer(architecture["memory_width"])
                and int(architecture["memory_width"])
                == PREREGISTERED_MEMORY_WIDTH
                and architecture["key_map"] == "fixed_rff_cosine"
                and architecture["value_map"] == "fixed_tanh_projection"
                and _is_finite_real(architecture["rff_gamma"])
                and float(architecture["rff_gamma"]) == 2.0
                and _is_integer(architecture["key_basis_seed"])
                and int(architecture["key_basis_seed"]) == config.model_seed + 101
                and _is_integer(architecture["key_bias_seed"])
                and int(architecture["key_bias_seed"]) == config.model_seed + 102
                and _is_integer(architecture["value_basis_seed"])
                and int(architecture["value_basis_seed"]) == config.model_seed + 103
                and architecture["write_component_type"] == "braintrace.element_wise"
                and architecture["query_component_type"] == "braintrace.nn.Linear"
                and architecture["read_component_type"] == "braintrace.nn.Linear"
                and _carrier_architecture_identity(architecture)
                and _architecture_digests_valid(architecture)
            ),
            "gate_native_all_colors_covered": gate_native_all_colors,
            "gate_native_key_separation_margin_passed": (
                gate_native_all_colors
                and gate_native_separation["margin_passed"] is True
                and _is_integer(gate_native_separation["memory_width"])
                and int(gate_native_separation["memory_width"])
                == config.context_memory_width
                and _is_integer(gate_native_separation["model_seed"])
                and int(gate_native_separation["model_seed"]) == config.model_seed
                and _is_finite_real(gate_native_separation["separation_margin"])
                and _is_finite_real(gate_native_separation["required_margin"])
                and float(gate_native_separation["separation_margin"])
                > float(gate_native_separation["required_margin"])
                and float(gate_native_separation["required_margin"]) == 0.25
            ),
            "gate_native_zero_event_key_exact_zero": (
                gate_native_separation["zero_event_key_exact_zero"] is True
                and _is_finite_real(gate_native_separation["zero_event_key_max_abs"])
                and float(gate_native_separation["zero_event_key_max_abs"]) == 0.0
            ),
            "gate_native_basis_matches_training_model": (
                _json_exact(gate_native_separation["architecture"], architecture)
            ),
            "standard_arc_all_colors_covered": standard_arc_all_colors,
            "standard_arc_key_separation_margin_passed": (
                standard_arc_all_colors
                and standard_arc_separation["margin_passed"] is True
                and _is_integer(standard_arc_separation["memory_width"])
                and int(standard_arc_separation["memory_width"])
                == config.context_memory_width
                and _is_integer(standard_arc_separation["model_seed"])
                and int(standard_arc_separation["model_seed"]) == config.model_seed
                and _is_finite_real(standard_arc_separation["separation_margin"])
                and _is_finite_real(standard_arc_separation["required_margin"])
                and float(standard_arc_separation["separation_margin"])
                > float(standard_arc_separation["required_margin"])
                and float(standard_arc_separation["required_margin"]) == 0.25
            ),
            "standard_arc_zero_event_key_exact_zero": (
                standard_arc_separation["zero_event_key_exact_zero"] is True
                and _is_finite_real(
                    standard_arc_separation["zero_event_key_max_abs"]
                )
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


def _environment_report() -> dict[str, Any]:
    return {
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


def _formal_admission_evidence(
    manifest_paths: Mapping[str, str | Path],
    *,
    source: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    if set(manifest_paths) != {"one_update", "stability_256"}:
        raise ValueError("formal Gate A requires both fixed admission manifests")
    from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher

    root = Path(__file__).resolve().parents[2]
    evidence: dict[str, Any] = {}
    qualifiers = {
        "one_update": _one_update_admission_qualification,
        "stability_256": _stability_admission_qualification,
    }
    for target in ("one_update", "stability_256"):
        bundle = launcher.load_authenticated_admission(
            manifest_paths[target],
            target=target,
            head=str(source["commit"]),
            image_id=str(environment["image_digest"]),
            repo_root=root,
        )
        admission = dict(bundle["admission"])
        if qualifiers[target](admission)["passed"] is not True:
            raise ValueError(f"authenticated {target} admission fails recomputation")
        evidence[target] = {
            "target": target,
            "source_head": source["commit"],
            "image_digest": environment["image_digest"],
            "bundle_sha256": bundle["manifest"]["bundle_sha256"],
            "manifest_sha256": bundle["manifest_sha256"],
            "preflight_sha256": bundle["preflight_sha256"],
            "result_sha256": bundle["result_sha256"],
            "admission": admission,
        }
    return evidence


def _require_authenticated_gpu_launch(
    source: Mapping[str, Any], environment: Mapping[str, Any]
) -> None:
    if not _source_evidence_clean(source):
        raise RuntimeError("launch source is not a verified clean full revision")
    if not _gpu_environment_verified(environment):
        raise RuntimeError("launch did not select an authenticated GPU image/device")


def run_stage21_admission(
    target: str,
) -> dict[str, Any]:
    """Execute one fixed Stage 2.1 admission target.

    Parameters
    ----------
    target : {"one_update", "stability_256"}
        Preregistered admission target. No topology or budget override is
        accepted by this entry point.

    Returns
    -------
    dict
        Schema-3 provenance envelope containing the inner admission report and
        its fail-closed scientific qualification.
    """

    configurations = {
        "one_update": BindingGateConfig.stage21_one_update_config,
        "stability_256": BindingGateConfig.stage21_stability_config,
    }
    if target not in configurations:
        raise ValueError("target must be 'one_update' or 'stability_256'")
    source_start = _source_report()
    environment = _environment_report()
    _require_authenticated_gpu_launch(source_start, environment)
    config = configurations[target]()
    data = _build_stage21_admission_data()
    if target == "one_update":
        admission = _run_one_update_admission(data, config)
    else:
        admission = _run_stability_admission(data, config)
    environment["device_memory_after_run"] = legacy._device_memory_report()
    source_end = _source_report()
    return {
        "schema_version": STAGE21_ARTIFACT_SCHEMA_VERSION,
        "control": admission["control"],
        "target": target,
        "learner": "pp_prop_only",
        "source": source_start,
        "source_end": source_end,
        "environment": environment,
        "config": dict(admission["config"]),
        "admission": admission,
        "qualification": admission["qualification"],
        "interpretation": admission["qualification"]["interpretation"],
    }


def run_binding_gate(
    config: BindingGateConfig,
    *,
    admission_manifests: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Train and evaluate the post-architecture pp-prop binding gate.

    Parameters
    ----------
    config : BindingGateConfig
        Complete deterministic Gate A configuration.
    admission_manifests : mapping, optional
        Fixed one-update and 256-update authenticated manifest paths. They are
        mandatory before a preregistered full Gate A run and are revalidated
        before any Gate A data generation or training begins.

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
    environment = _environment_report()
    authenticated_admissions: dict[str, Any] = {}
    if config.qualification_regime == "preregistered_full":
        _require_authenticated_gpu_launch(source_start, environment)
        if admission_manifests is None:
            raise ValueError(
                "preregistered full Gate A requires authenticated Stage 2.1 manifests"
            )
        authenticated_admissions = _formal_admission_evidence(
            admission_manifests,
            source=source_start,
            environment=environment,
        )
    data_start = time.perf_counter()
    data = build_binding_data(config)
    data_seconds = time.perf_counter() - data_start
    data_report = legacy._data_report(data, config)
    training_model_config = _model_config(config, batch_size=config.batch_size)
    model = LatentWorkspaceModel(training_model_config)
    architecture = _architecture_report(model)
    gate_native_separation = _gate_native_key_separation_report(model, config)
    standard_arc_separation = _standard_arc_key_separation_report(
        memory_width=config.context_memory_width,
        model_seed=config.model_seed,
    )
    initial = legacy._parameter_values(model)
    initialization = _initialization_report(initial, config)
    if config.qualification_regime == "preregistered_full":
        if not _preregistered_gpu_initialization(initialization):
            raise RuntimeError("formal Gate initialization differs from admissions")
        if (
            data_report["training_schedule_sha256"]
            != PREREGISTERED_TRAINING_SCHEDULE_SHA256
            or data_report["validation_schedule_sha256"]
            != PREREGISTERED_STABILITY_DIGESTS["validation_schedule_sha256"]
        ):
            raise RuntimeError("formal Gate schedule differs from admissions")
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
        one_update_admission=authenticated_admissions.get("one_update", {}).get(
            "admission", {}
        ),
        stability_admission=authenticated_admissions.get("stability_256", {}).get(
            "admission", {}
        ),
        initialization=initialization,
        data=data_report,
        config=config,
    )
    result = {
        "schema_version": STAGE21_ARTIFACT_SCHEMA_VERSION,
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
            **data_report,
            "generation_seconds": data_seconds,
            "marginals": marginals,
        },
        "initialization": initialization,
        "architecture": architecture,
        "gate_native_key_separation": gate_native_separation,
        "standard_arc_key_separation": standard_arc_separation,
        "stage21_admissions": authenticated_admissions,
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
    parser.add_argument(
        "--target",
        choices=("one_update", "stability_256"),
        help="run one fixed Stage 2.1 admission instead of Gate A",
    )
    parser.add_argument("--one-update-manifest", type=Path)
    parser.add_argument("--stability-manifest", type=Path)
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
    if args.target is not None:
        if args.smoke or args.one_update_manifest or args.stability_manifest:
            raise ValueError(
                "fixed admission targets do not accept smoke or manifest options"
            )
        admission_defaults = {
            "training_updates": 10_000,
            "batch_size": 64,
            "validation_episodes": 512,
            "gap_steps": 1,
            "neuron_count": 2048,
            "recurrent_edges": 16_384,
            "readout_width": 128,
            "color_rank": 16,
            "learning_rate": 3e-3,
            "context_memory_width": PREREGISTERED_MEMORY_WIDTH,
            "memory_decay": PREREGISTERED_MEMORY_DECAY,
            "sparse_backend": None,
        }
        if any(
            getattr(args, name) != expected
            for name, expected in admission_defaults.items()
        ):
            raise ValueError("fixed admission targets reject topology and budget overrides")
        result = run_stage21_admission(args.target)
        destination = write_artifact(result, args.output)
        print(destination)
        print(json.dumps(legacy._json_ready(result["qualification"]), sort_keys=True))
        return 0
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
    manifests = None
    if args.one_update_manifest is not None or args.stability_manifest is not None:
        if args.smoke:
            raise ValueError("smoke runs cannot consume formal admission manifests")
        if args.one_update_manifest is None or args.stability_manifest is None:
            raise ValueError("formal Gate A requires both admission manifests")
        manifests = {
            "one_update": args.one_update_manifest,
            "stability_256": args.stability_manifest,
        }
        if config.qualification_regime != "preregistered_full":
            raise ValueError(
                "admission manifests are accepted only by preregistered full Gate A"
            )
    if manifests is None:
        result = run_binding_gate(config)
    else:
        result = run_binding_gate(config, admission_manifests=manifests)
    destination = write_artifact(result, args.output)
    print(destination)
    print(json.dumps(legacy._json_ready(result["qualification"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
