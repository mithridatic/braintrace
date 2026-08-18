"""Preregistered pp-prop mechanism ablations for Example 21 Gate C."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._testing.oracle import chunked_online_param_gradients

from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop import latent_workspace_binding_gate as gate_a
from examples.pp_prop import latent_workspace_depth_gate as gate_b
from examples.pp_prop.latent_workspace_model import (
    LatentWorkspaceModel,
    ModelConfig,
    compile_pp_prop,
)


GATE_C_SCHEMA_VERSION = 1
GATE_C_INITIALIZATION_CONTROL = "example21_gate_c_initialization_admission"
GATE_C_CONTROL = "example21_pp_prop_learnability_gate_c"

_GATE_C_SOURCE_FILES = (
    "examples/pp_prop/latent_workspace_model.py",
    "examples/pp_prop/latent_workspace_task.py",
    "examples/pp_prop/latent_workspace_binding_control.py",
    "examples/pp_prop/latent_workspace_binding_gate.py",
    "examples/pp_prop/latent_workspace_depth_gate.py",
    "examples/pp_prop/latent_workspace_ablation_gate.py",
)
_GATE_A_REFERENCE = {
    "qualification_passed": True,
    "result_sha256": "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632",
    "manifest_sha256": "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf",
    "source_commit": "4737e9172b1c6ca99347af5b2c83fc795a294a16",
    "bundle_sha256": "ba850a205c4691d573facef7b8e90cabd4824905c73fcd4f6add29293cd95875",
    "preflight_sha256": "d1d54406d0972d52ac10cddec7e6d1ed38c55481d51e21989e444fe7c3f03d08",
    "result_path": (
        "var/example21-binding-gate/"
        "4737e9172b1c6ca99347af5b2c83fc795a294a16-formal-gate-a.json"
    ),
    "manifest_path": (
        "var/example21-binding-gate/"
        "4737e9172b1c6ca99347af5b2c83fc795a294a16-formal-gate-a.manifest.json"
    ),
}
_GATE_B_REFERENCE = {
    "qualification_passed": True,
    "result_sha256": "6456537ea108cea8892d00c8a71c1f647217e074b525bc9ed01b64aef9001766",
    "manifest_sha256": "99c42985e203413eb0600a5dabe321188776eff8058500dc86f4a1618b413eab",
    "source_commit": "dafa64a8b4c3848241baa117affa55b632518a8e",
    "bundle_sha256": "be07e8c92d8deaa94508f34dcee45f5feb09740cb2804778d6280a2fa3c64851",
    "preflight_sha256": "91e86d92670cd33d3f4206ff3d5096e3721104996a9506223a9e34c082dd052f",
    "result_path": (
        "var/example21-depth-gate/"
        "dafa64a8b4c3848241baa117affa55b632518a8e-formal-gate-b.json"
    ),
    "manifest_path": (
        "var/example21-depth-gate/"
        "dafa64a8b4c3848241baa117affa55b632518a8e-formal-gate-b.manifest.json"
    ),
}

ARM_ORDER = ("full", "query_only", "terminal_only", "legacy", "frozen_write")
REGIME_ORDER = ("gate_a", "gate_b")

SHARED_PARAMETER_PATHS = (
    "color_factor_head/weight",
    "ff_syn/comm/weight",
    "height_head/weight",
    "readout_projection/weight",
    "rec_syn/comm/weight",
    "width_head/weight",
)
MEMORY_PARAMETER_PATHS = (
    "memory_read_projection/weight",
    "memory_write_scale",
    "workspace_query_projection/weight",
)
FULL_PARAMETER_PATHS = tuple(sorted((*SHARED_PARAMETER_PATHS, *MEMORY_PARAMETER_PATHS)))

_GATE_A_SCHEDULE_SHA256 = {
    "training_schedule_sha256": (
        "25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b"
    ),
    "validation_schedule_sha256": (
        "80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd"
    ),
    "training_mapping_ids_sha256": (
        "fbd48ad9a8d3ecb0dd0812abbbda35953def52862785ce048e17b2eb9fdd3499"
    ),
    "validation_mapping_ids_sha256": (
        "a75b3b2ab05110e21fef1ea44ae3fb701d557f45351e5a17cf89a80e80f689f3"
    ),
}

QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "exact_configuration",
    "prerequisites_authenticated",
    "initialization_authenticated",
    "canonical_schedules_complete",
    "fresh_isolated_optimizers",
    "compiler_and_training_complete",
    "full_gate_a_passed",
    "full_gate_b_passed",
    "blocking_behavioral_margins",
    "paired_h0_identity",
    "frozen_write_complete",
    "mechanism_oracle_complete",
    "source_and_gpu_authenticated",
)

GATE_C_INITIALIZATION_QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "preregistered_regimes",
    "gate_a_prerequisite_authenticated",
    "gate_b_prerequisite_authenticated",
    "source_and_gpu_authenticated",
    "source_files_exact",
    "canonical_full_initializations_exact",
    "legacy_initializations_complete",
    "shared_paths_byte_identical",
    "arm_initialization_refs_exact",
    "optimizer_paths_exact",
    "fresh_optimizer_states_zero_and_finite",
    "compiler_topologies_complete",
    "no_behavioral_updates",
)


@dataclass(frozen=True, slots=True)
class GateCArmSpec:
    """Describe one fixed Gate C intervention.

    Parameters
    ----------
    name
        Stable arm identifier.
    memory_mode
        Full, query-only, or legacy contextual-memory policy.
    supervision
        Per-checkpoint or terminal-only loss policy.
    context_memory_width
        Fast-weight width; zero selects the legacy reservoir.
    optimizer_excluded_paths
        Exact parameter paths withheld from optimizer updates.
    """

    name: str
    memory_mode: str
    supervision: str
    context_memory_width: int
    optimizer_excluded_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateCRegimeSpec:
    """Describe one canonical Gate C data and initialization regime.

    Parameters
    ----------
    name
        Stable regime identifier.
    sequence_length, input_width
        Static encoded event geometry.
    training_updates, batch_size, validation_episodes
        Exact production data budget.
    full_parameter_count, legacy_parameter_count
        Backend-independent parameter counts.
    full_parameter_sha256
        Authenticated GPU full-memory initialization digest.
    """

    name: str
    sequence_length: int
    input_width: int
    training_updates: int
    batch_size: int
    validation_episodes: int
    full_parameter_count: int
    full_parameter_sha256: str
    legacy_parameter_count: int


ARM_SPECS: dict[str, GateCArmSpec] = {
    "full": GateCArmSpec("full", "full", "per_checkpoint", 32),
    "query_only": GateCArmSpec(
        "query_only", "query_only", "per_checkpoint", 32
    ),
    "terminal_only": GateCArmSpec(
        "terminal_only", "full", "terminal_only", 32
    ),
    "legacy": GateCArmSpec("legacy", "legacy", "per_checkpoint", 0),
    "frozen_write": GateCArmSpec(
        "frozen_write",
        "full",
        "per_checkpoint",
        32,
        ("memory_write_scale",),
    ),
}

REGIME_SPECS: dict[str, GateCRegimeSpec] = {
    "gate_a": GateCRegimeSpec(
        name="gate_a",
        sequence_length=6,
        input_width=41,
        training_updates=10_000,
        batch_size=64,
        validation_episodes=512,
        full_parameter_count=646_940,
        full_parameter_sha256=(
            "b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8"
        ),
        legacy_parameter_count=514_844,
    ),
    "gate_b": GateCRegimeSpec(
        name="gate_b",
        sequence_length=19,
        input_width=47,
        training_updates=4_096,
        batch_size=64,
        validation_episodes=512,
        full_parameter_count=659_228,
        full_parameter_sha256=(
            "aa463549a8c3c1dbc24c9f727944eada035b776df666a83139214078d0f83d6d"
        ),
        legacy_parameter_count=527_132,
    ),
}


@dataclass(frozen=True, slots=True)
class GateCConfig:
    """Configure the fixed Gate C paired mechanism experiment.

    Parameters
    ----------
    gate_a_config, gate_b_config
        Canonical binding and demonstrated-depth regimes.
    oracle_validation_index
        Fixed Gate B held-out episode used by the mechanism oracle.
    oracle_effort
        Fixed demonstrated depth used by the mechanism oracle.
    gradient_chunk_size
        Strictly finite pp-prop gradient window.
    """

    gate_a_config: gate_a.BindingGateConfig = field(
        default_factory=gate_a.BindingGateConfig
    )
    gate_b_config: gate_b.DepthGateConfig = field(default_factory=gate_b.DepthGateConfig)
    oracle_validation_index: int = 0
    oracle_effort: int = 8
    gradient_chunk_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.gate_a_config, gate_a.BindingGateConfig):
            raise TypeError("gate_a_config must be a BindingGateConfig")
        if not isinstance(self.gate_b_config, gate_b.DepthGateConfig):
            raise TypeError("gate_b_config must be a DepthGateConfig")
        for name in (
            "oracle_validation_index",
            "oracle_effort",
            "gradient_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.gradient_chunk_size == 0:
            raise ValueError("gradient_chunk_size must be positive")

    @property
    def qualification_regime(self) -> str:
        """Return whether every preregistered production coordinate is exact."""

        exact = (
            self.gate_a_config == gate_a.BindingGateConfig()
            and self.gate_b_config == gate_b.DepthGateConfig()
            and self.oracle_validation_index == 0
            and self.oracle_effort == 8
            and self.gradient_chunk_size == 1
        )
        return "preregistered_full" if exact else "nonqualifying_abbreviated"


def _arm_spec(arm: str) -> GateCArmSpec:
    try:
        return ARM_SPECS[arm]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown Gate C arm: {arm!r}") from error


def _regime_spec(regime: str) -> GateCRegimeSpec:
    try:
        return REGIME_SPECS[regime]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown Gate C regime: {regime!r}") from error


def _loss_weights(
    regime: str,
    arm: str,
    *,
    efforts: np.ndarray,
) -> np.ndarray:
    """Return the exact normalized temporal loss mask for an arm."""

    _regime_spec(regime)
    arm_spec = _arm_spec(arm)
    raw_efforts = np.asarray(efforts)
    if raw_efforts.ndim != 1 or raw_efforts.dtype == np.bool_:
        raise ValueError("efforts must be a one-dimensional integer array")
    if not np.issubdtype(raw_efforts.dtype, np.integer):
        raise TypeError("efforts must contain integers")

    if regime == "gate_a":
        weights = np.zeros((REGIME_SPECS[regime].sequence_length,), dtype=np.float32)
        if arm_spec.supervision == "terminal_only":
            weights[-1] = 1.0
        else:
            weights[-2:] = 0.5
        return weights

    if raw_efforts.size == 0 or not np.isin(raw_efforts, gate_b.QUALIFYING_EFFORTS).all():
        raise ValueError("Gate B efforts must lie in {1, 2, 4, 8}")
    weights = np.zeros(
        (raw_efforts.size, REGIME_SPECS[regime].sequence_length), dtype=np.float32
    )
    for row, effort_value in enumerate(raw_efforts):
        effort = int(effort_value)
        if arm_spec.supervision == "terminal_only":
            weights[row, 10 + effort] = 1.0
        else:
            weights[row, 10 : 11 + effort] = np.float32(1.0 / (effort + 1))
    return weights


def _model_config_for_arm(
    config: GateCConfig,
    regime: str,
    arm: str,
    *,
    batch_size: int,
) -> ModelConfig:
    """Return one fixed model configuration for a regime and arm."""

    _regime_spec(regime)
    arm_spec = _arm_spec(arm)
    if regime == "gate_a":
        result = gate_a._model_config(config.gate_a_config, batch_size=batch_size)
    else:
        result = gate_b._model_config(config.gate_b_config, batch_size=batch_size)
    if arm_spec.context_memory_width:
        return result
    return dataclasses.replace(
        result,
        context_memory_width=0,
        demonstration_phase_index=None,
        query_phase_index=None,
        input_side_valid_index=None,
        output_side_valid_index=None,
        memory_key_indices=(),
        memory_value_indices=(),
    )


def _optimizer_parameter_paths(
    available_paths: tuple[str, ...],
    arm: str,
) -> tuple[str, ...]:
    """Return the exact optimizer path set after the declared intervention."""

    spec = _arm_spec(arm)
    paths = tuple(available_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("optimizer parameter paths must be unique")
    missing = set(spec.optimizer_excluded_paths) - set(paths)
    if missing:
        raise ValueError(
            "memory_write_scale is required by the frozen-write intervention"
        )
    return tuple(path for path in paths if path not in spec.optimizer_excluded_paths)


def _new_model_for_arm(
    config: GateCConfig,
    regime: str,
    arm: str,
    *,
    batch_size: int,
) -> LatentWorkspaceModel:
    """Construct one fresh, statically intervened Gate C model."""

    arm_spec = _arm_spec(arm)
    model_config = _model_config_for_arm(
        config,
        regime,
        arm,
        batch_size=batch_size,
    )
    policy = "query_only" if arm_spec.memory_mode == "query_only" else "full"
    return LatentWorkspaceModel(model_config, memory_read_policy=policy)


def _parameter_states_by_path(
    model: LatentWorkspaceModel,
) -> dict[str, brainstate.ParamState]:
    return {
        gate_a._path(path): state
        for path, state in model.states(brainstate.ParamState).items()
    }


def _copy_shared_initialization(
    canonical: LatentWorkspaceModel,
    legacy_model: LatentWorkspaceModel,
) -> dict[str, Any]:
    """Copy and verify the six same-shaped initialization paths."""

    if not isinstance(canonical, LatentWorkspaceModel) or not isinstance(
        legacy_model, LatentWorkspaceModel
    ):
        raise TypeError("shared initialization subjects must be workspace models")
    canonical_states = _parameter_states_by_path(canonical)
    legacy_states = _parameter_states_by_path(legacy_model)
    if not set(SHARED_PARAMETER_PATHS).issubset(canonical_states):
        raise ValueError("canonical model is missing a shared parameter path")
    if tuple(sorted(legacy_states)) != SHARED_PARAMETER_PATHS:
        raise ValueError("legacy model must contain exactly the shared paths")
    for path in SHARED_PARAMETER_PATHS:
        source = canonical_states[path].value
        target = legacy_states[path].value
        if jax.tree.structure(source) != jax.tree.structure(target):
            raise ValueError(f"shared parameter structure differs: {path}")
        for source_leaf, target_leaf in zip(
            jax.tree.leaves(source),
            jax.tree.leaves(target),
            strict=True,
        ):
            if source_leaf.shape != target_leaf.shape or source_leaf.dtype != target_leaf.dtype:
                raise ValueError(f"shared parameter geometry differs: {path}")
        legacy_states[path].value = jax.tree.map(
            lambda leaf: jnp.array(leaf, copy=True), source
        )

    canonical_values = legacy._parameter_values(canonical)
    copied_values = legacy._parameter_values(legacy_model)
    all_equal = all(
        jax.tree.structure(canonical_values[path])
        == jax.tree.structure(copied_values[path])
        and all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree.leaves(canonical_values[path]),
                jax.tree.leaves(copied_values[path]),
                strict=True,
            )
        )
        for path in SHARED_PARAMETER_PATHS
    )
    shared_values = {path: copied_values[path] for path in SHARED_PARAMETER_PATHS}
    return {
        "paths": list(SHARED_PARAMETER_PATHS),
        "all_equal": bool(all_equal),
        "sha256": legacy._array_digest(shared_values),
    }


def _regenerate_gate_a_data(config: GateCConfig) -> legacy.BindingData:
    """Regenerate the canonical Gate A schedule from its frozen config."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    return legacy.build_binding_data(config.gate_a_config)


def _regenerate_gate_b_data(
    config: GateCConfig,
) -> tuple[gate_b.DepthSchedule, gate_b.DepthValidationData]:
    """Regenerate the canonical Gate B schedule and held-out controls."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    schedule = gate_b._build_schedule(config.gate_b_config)
    return schedule, gate_b._encode_validation_data(schedule, config.gate_b_config)


@dataclass(slots=True)
class GateCTrainer:
    """Hold one isolated production pp-prop arm trainer.

    Parameters
    ----------
    learner
        Compiled pp-prop sequence learner.
    optimizer
        Fresh Adam optimizer registered only on the arm's update paths.
    compiler, compile_warnings
        Retained compiler topology and warning evidence.
    train_chunk
        One JIT-compiled chunk driver with an internal BrainState loop.
    algorithm
        Stable learning-rule identifier.
    optimizer_parameter_paths, excluded_optimizer_paths
        Exact updated and deliberately frozen parameter paths.
    """

    learner: Any
    optimizer: Any
    compiler: dict[str, Any]
    compile_warnings: list[str]
    train_chunk: Any
    algorithm: str
    optimizer_parameter_paths: tuple[str, ...]
    excluded_optimizer_paths: tuple[str, ...]


def _tree_telemetry(value: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(value))
    if not leaves:
        raise RuntimeError("telemetry subject has no array leaves")
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    maximum = jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(leaf.astype(jnp.float32)), initial=0.0)
                for leaf in leaves
            ]
        )
    )
    count = jnp.asarray(sum(leaf.size for leaf in leaves), dtype=jnp.int32)
    return finite, maximum, count


def _make_arm_trainer(
    model: LatentWorkspaceModel,
    config: GateCConfig,
    regime: str,
    arm: str,
) -> GateCTrainer:
    """Compile one fresh pp-prop trainer for a fixed Gate C arm."""

    if not isinstance(model, LatentWorkspaceModel):
        raise TypeError("model must be a LatentWorkspaceModel")
    _regime_spec(regime)
    arm_spec = _arm_spec(arm)
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    compiler = gate_a._compiler_report(learner)
    parameter_keys = {
        gate_a._path(path): path for path in learner.param_states.keys()
    }
    available_paths = tuple(sorted(parameter_keys))
    optimizer_parameter_paths = _optimizer_parameter_paths(available_paths, arm)
    optimizer_keys = tuple(parameter_keys[path] for path in optimizer_parameter_paths)
    optimizer_states = {
        key: learner.param_states[key] for key in optimizer_keys
    }
    optimizer = braintools.optim.Adam(lr=regime_config.learning_rate)
    optimizer.register_trainable_weights(optimizer_states)
    trace_labels = tuple(
        path
        for path in (
            "ff_syn/comm/weight",
            "rec_syn/comm/weight",
            "memory_write_scale",
            "workspace_query_projection/weight",
            "memory_read_projection/weight",
        )
        if path in parameter_keys
    )
    if not trace_labels:
        raise RuntimeError("trainer has no pp-prop trace parameter")
    model_states = tuple(
        state
        for state in model.states().values()
        if not isinstance(state, brainstate.ParamState)
    )
    rank = regime_config.color_rank
    batch_size = model.config.batch_size

    @brainstate.transform.jit
    def train_chunk(
        events: jax.Array,
        targets: jax.Array,
        loss_weights: jax.Array,
        advance_masks: jax.Array,
    ) -> dict[str, jax.Array]:
        if targets.ndim == 2:
            target_sequences = jnp.broadcast_to(
                targets[:, None, :], events.shape[:3]
            )
        elif targets.ndim == 3:
            target_sequences = targets
        else:
            raise ValueError("targets must have shape (updates,batch) or (updates,time,batch)")
        if loss_weights.ndim == 1:
            weight_sequences = jnp.broadcast_to(
                loss_weights[None, :], events.shape[:2]
            )
        elif loss_weights.ndim == 2:
            weight_sequences = loss_weights
        else:
            raise ValueError("loss weights must have shape (time,) or (updates,time)")

        def train_one(
            inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> dict[str, jax.Array]:
            sequence, target_sequence, weights, advances = inputs
            model.reset_state()
            learner.reset_state(batch_size=batch_size)

            def step_loss(
                event: jax.Array,
                advance: jax.Array,
                target: jax.Array,
            ) -> jax.Array:
                return legacy._classification_loss(
                    learner(event, advance), target, rank
                )

            gradients, loss = learner.etrace_grad(
                sequence,
                advances,
                target_sequence,
                step_fn=step_loss,
                mask=weights.astype(jnp.float32),
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            compact = model.compact_readout()
            trace_factors = tuple(
                learner.get_etrace_of(parameter_keys[label])
                for label in trace_labels
            )
            clipped_gradients = brainstate.nn.clip_grad_norm(
                gradients, regime_config.clip_norm
            )
            optimizer.update(
                {key: clipped_gradients[key] for key in optimizer_keys}
            )
            measurements = {
                "logits": _tree_telemetry(legacy._color_logits(compact, rank)),
                "model_states": _tree_telemetry(
                    tuple(state.value for state in model_states)
                ),
                "gradients": _tree_telemetry(gradients),
                "pp_prop_traces": _tree_telemetry(trace_factors),
                "adam": _tree_telemetry(optimizer.opt_state.value),
                "parameters": _tree_telemetry(
                    tuple(state.value for state in learner.param_states.values())
                ),
            }
            return {
                "loss": loss,
                "finite": {key: value[0] for key, value in measurements.items()},
                "max_abs": {key: value[1] for key, value in measurements.items()},
                "value_count": {
                    key: value[2] for key, value in measurements.items()
                },
            }

        return brainstate.transform.for_loop(
            train_one,
            (events, target_sequences, weight_sequences, advance_masks),
        )

    return GateCTrainer(
        learner=learner,
        optimizer=optimizer,
        compiler=compiler,
        compile_warnings=[str(item.message) for item in caught],
        train_chunk=train_chunk,
        algorithm="production_pp_prop",
        optimizer_parameter_paths=optimizer_parameter_paths,
        excluded_optimizer_paths=arm_spec.optimizer_excluded_paths,
    )


def _optimizer_initial_state_report(
    trainer: GateCTrainer,
    *,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    """Report one fresh arm optimizer before any update."""

    _regime_spec(regime)
    _arm_spec(arm)
    if not isinstance(trainer, GateCTrainer):
        raise TypeError("trainer must be a GateCTrainer")
    leaves = [
        np.ascontiguousarray(np.asarray(leaf))
        for leaf in jax.tree.leaves(trainer.optimizer.opt_state.value)
    ]
    if not leaves:
        raise RuntimeError("optimizer state has no array leaves")
    if any(
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        for array in leaves
    ):
        raise TypeError("optimizer state leaves must be numeric")
    finite = all(np.isfinite(array).all() for array in leaves)
    all_zero = all(np.count_nonzero(array) == 0 for array in leaves)
    fields: list[bytes] = [
        b"example21-gate-c-optimizer-state-v1",
        regime.encode("utf-8"),
        arm.encode("utf-8"),
        *(
            path.encode("utf-8")
            for path in sorted(trainer.optimizer_parameter_paths)
        ),
    ]
    for index, array in enumerate(leaves):
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return {
        "included": list(trainer.optimizer_parameter_paths),
        "excluded": list(trainer.excluded_optimizer_paths),
        "fresh_state_finite": bool(finite),
        "fresh_state_all_zero": bool(all_zero),
        "state_leaf_count": len(leaves),
        "value_count": int(sum(array.size for array in leaves)),
        "state_sha256": hashlib.sha256(b"\0".join(fields)).hexdigest(),
        "executed_updates": int(
            np.asarray(trainer.optimizer.step_count.value).item()
        ),
    }


def _initialization_topology_report(
    model: LatentWorkspaceModel,
    trainer: GateCTrainer,
    *,
    regime: str,
    tree: str,
) -> dict[str, Any]:
    """Bind one fresh model tree to its pp-prop compiler evidence."""

    _regime_spec(regime)
    if tree not in ("canonical_full", "legacy"):
        raise ValueError("initialization tree must be canonical_full or legacy")
    if not isinstance(model, LatentWorkspaceModel):
        raise TypeError("model must be a LatentWorkspaceModel")
    if not isinstance(trainer, GateCTrainer):
        raise TypeError("trainer must be a GateCTrainer")
    values = legacy._parameter_values(model)
    leaves = [
        np.asarray(leaf)
        for value in values.values()
        for leaf in jax.tree.leaves(value)
    ]
    expected_paths = (
        FULL_PARAMETER_PATHS if tree == "canonical_full" else SHARED_PARAMETER_PATHS
    )
    if tuple(sorted(values)) != expected_paths:
        raise ValueError("initialization parameter paths differ from the tree")
    model_states = _parameter_states_by_path(model)
    learner_states = {
        gate_a._path(path): state
        for path, state in trainer.learner.param_states.items()
    }
    if tuple(sorted(learner_states)) != expected_paths or any(
        learner_states[path] is not model_states[path] for path in expected_paths
    ):
        raise ValueError("trainer must be compiled from the same model")
    return {
        "fresh_model": True,
        "model_seed": model.config.seed,
        "memory_read_policy": model.memory_read_policy,
        "model_config": dataclasses.asdict(model.config),
        "parameter_paths": list(expected_paths),
        "parameter_count": int(sum(array.size for array in leaves)),
        "parameter_sha256": legacy._array_digest(values),
        "parameters_finite": bool(
            leaves and all(np.isfinite(array).all() for array in leaves)
        ),
        "compiler": trainer.compiler,
    }


def _normalized_prerequisites(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy the two frozen authenticated prerequisite refs."""

    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "gate_a",
        "gate_b",
    }:
        raise ValueError("Gate C initialization requires Gate A and Gate B")
    if not all(isinstance(prerequisites[name], Mapping) for name in prerequisites):
        raise TypeError("Gate C prerequisite references must be mappings")
    expected = {"gate_a": _GATE_A_REFERENCE, "gate_b": _GATE_B_REFERENCE}
    for name in ("gate_a", "gate_b"):
        if not gate_a._json_exact(prerequisites[name], expected[name]):
            raise ValueError(f"Gate C {name} prerequisite is not authenticated")
    return {
        name: dict(prerequisites[name]) for name in ("gate_a", "gate_b")
    }


def _sha256_complete(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _strict_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _source_files_complete(source_files: Any) -> bool:
    if not isinstance(source_files, Mapping) or set(source_files) != set(
        _GATE_C_SOURCE_FILES
    ):
        return False
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in _GATE_C_SOURCE_FILES
    }
    return gate_a._json_exact(source_files, expected)


def _source_and_gpu_complete(report: Mapping[str, Any]) -> bool:
    start = report["source_start"]
    end = report["source_end"]
    environment = report["environment"]
    source_keys = {
        "asserted_commit",
        "asserted_commit_matches_head",
        "asserted_dirty",
        "asserted_dirty_matches_worktree",
        "commit",
        "commit_is_valid_40_hex",
        "dirty",
        "head_command_succeeded",
        "status_command_succeeded",
        "verified",
    }
    environment_keys = {"backend", "devices", "image_digest", "jax", "python"}
    if (
        not isinstance(start, Mapping)
        or not isinstance(end, Mapping)
        or set(start) != source_keys
        or set(end) != source_keys
        or not isinstance(environment, Mapping)
        or set(environment) != environment_keys
        or not isinstance(environment["jax"], str)
        or not environment["jax"]
        or not isinstance(environment["python"], str)
        or not environment["python"]
        or not isinstance(environment["devices"], list)
        or not environment["devices"]
    ):
        return False
    device_keys = {"device_kind", "id", "platform", "process_index"}
    for device in environment["devices"]:
        if (
            not isinstance(device, Mapping)
            or set(device) != device_keys
            or not isinstance(device["device_kind"], str)
            or not device["device_kind"]
            or not isinstance(device["platform"], str)
            or device["platform"] != "gpu"
            or not _strict_integer(device["id"])
            or not _strict_integer(device["process_index"])
        ):
            return False
    return bool(
        gate_a._source_evidence_clean(start)
        and gate_a._source_evidence_clean(end)
        and start.get("commit") == end.get("commit")
        and gate_a._gpu_environment_verified(environment)
    )


def _compiler_common_complete(
    compiler: Any,
    *,
    expected_paths: tuple[str, ...],
) -> bool:
    if not isinstance(compiler, Mapping):
        return False
    diagnostics = compiler.get("diagnostics")
    diagnostic_keys = {"kind", "level", "message"}
    diagnostics_complete = isinstance(diagnostics, list) and all(
        isinstance(item, Mapping)
        and set(item) in (diagnostic_keys, diagnostic_keys | {"weight_path"})
        and all(isinstance(item[key], str) for key in diagnostic_keys)
        and (
            "weight_path" not in item
            or isinstance(item["weight_path"], str)
        )
        and item["level"].lower() != "error"
        for item in diagnostics
    )
    return bool(
        compiler.get("available") is True
        and diagnostics_complete
        and compiler.get("compiled_parameter_paths") == list(expected_paths)
    )


def _full_compiler_complete(compiler: Any) -> bool:
    try:
        return bool(
            _compiler_common_complete(
                compiler, expected_paths=FULL_PARAMETER_PATHS
            )
            and gate_b._compiler_evidence_complete(compiler)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _legacy_compiler_complete(compiler: Any) -> bool:
    required = {
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    }
    try:
        hidden_groups = compiler["hidden_groups"]
        groups_complete = bool(hidden_groups) and all(
            isinstance(group, Mapping)
            and set(group) == {"index", "hidden_paths"}
            and _strict_integer(group["index"])
            and isinstance(group["hidden_paths"], list)
            and bool(group["hidden_paths"])
            and all(isinstance(path, str) and path for path in group["hidden_paths"])
            and not {
                "context_memory",
                "reasoning_query",
                "memory_read",
            }.intersection(group["hidden_paths"])
            for group in hidden_groups
        )
        return bool(
            _compiler_common_complete(
                compiler, expected_paths=SHARED_PARAMETER_PATHS
            )
            and set(compiler["required_direct_paths"]) == required
            and set(compiler["direct_path_status"]) == required
            and set(compiler["direct_path_evidence"]) == required
            and all(compiler["direct_path_status"][path] is False for path in required)
            and all(compiler["direct_path_evidence"][path] == [] for path in required)
            and compiler["all_required_direct"] is False
            and compiler["context_memory_isolated_from_workspace_lif"] is False
            and isinstance(hidden_groups, list)
            and groups_complete
        )
    except (KeyError, TypeError, ValueError):
        return False


def _regimes_complete(report: Mapping[str, Any], config: GateCConfig) -> bool:
    expected = {
        regime: {
            "spec": dataclasses.asdict(REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
        }
        for regime in REGIME_ORDER
    }
    return bool(
        config.qualification_regime == "preregistered_full"
        and report.get("qualification_regime") == "preregistered_full"
        and gate_a._json_exact(report.get("regimes"), expected)
    )


def _topology_report_complete(
    topology: Any,
    *,
    config: GateCConfig,
    regime: str,
    tree: str,
) -> bool:
    if not isinstance(topology, Mapping) or set(topology) != {
        "fresh_model",
        "model_seed",
        "memory_read_policy",
        "model_config",
        "parameter_paths",
        "parameter_count",
        "parameter_sha256",
        "parameters_finite",
        "compiler",
    }:
        return False
    spec = REGIME_SPECS[regime]
    arm = "legacy" if tree == "legacy" else "full"
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    expected_model = dataclasses.asdict(
        _model_config_for_arm(
            config,
            regime,
            arm,
            batch_size=regime_config.batch_size,
        )
    )
    expected_paths = SHARED_PARAMETER_PATHS if tree == "legacy" else FULL_PARAMETER_PATHS
    expected_count = (
        spec.legacy_parameter_count if tree == "legacy" else spec.full_parameter_count
    )
    sha = topology["parameter_sha256"]
    sha_valid = _sha256_complete(sha) and (
        tree == "legacy" or sha == spec.full_parameter_sha256
    )
    return bool(
        topology["fresh_model"] is True
        and _strict_integer(topology["model_seed"])
        and int(topology["model_seed"]) == 2108
        and topology["memory_read_policy"] == "full"
        and gate_a._json_exact(topology["model_config"], expected_model)
        and topology["parameter_paths"] == list(expected_paths)
        and _strict_integer(topology["parameter_count"])
        and int(topology["parameter_count"]) == expected_count
        and sha_valid
        and topology["parameters_finite"] is True
    )


def _shared_digest_from_path_digests(path_digests: Mapping[str, str]) -> str:
    fields: list[bytes] = [b"example21-gate-c-shared-global-v1"]
    for path in sorted(path_digests):
        fields.extend(
            (path.encode("utf-8"), path_digests[path].encode("ascii"))
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _shared_report_complete(shared: Any) -> bool:
    if not isinstance(shared, Mapping) or set(shared) != {
        "paths",
        "framing",
        "canonical_path_sha256",
        "legacy_path_sha256",
        "canonical_sha256",
        "legacy_sha256",
        "all_equal",
    }:
        return False
    expected_framing = {
        "path": "example21-gate-c-shared-path-v1",
        "global": "example21-gate-c-shared-global-v1",
    }
    canonical = shared["canonical_path_sha256"]
    legacy_paths = shared["legacy_path_sha256"]
    if not (
        shared["paths"] == list(SHARED_PARAMETER_PATHS)
        and gate_a._json_exact(shared["framing"], expected_framing)
        and isinstance(canonical, Mapping)
        and isinstance(legacy_paths, Mapping)
        and set(canonical) == set(legacy_paths) == set(SHARED_PARAMETER_PATHS)
        and all(_sha256_complete(canonical[path]) for path in SHARED_PARAMETER_PATHS)
        and gate_a._json_exact(canonical, legacy_paths)
    ):
        return False
    expected_global = _shared_digest_from_path_digests(canonical)
    return bool(
        shared["canonical_sha256"] == expected_global
        and shared["legacy_sha256"] == expected_global
        and shared["all_equal"] is True
    )


def _arm_refs_complete(initialization: Mapping[str, Any]) -> bool:
    refs = initialization["arm_initialization_refs"]
    if not isinstance(refs, Mapping) or set(refs) != set(ARM_ORDER):
        return False
    canonical_sha = initialization["canonical_full"]["parameter_sha256"]
    legacy_sha = initialization["legacy"]["parameter_sha256"]
    for arm in ARM_ORDER:
        expected = {
            "tree": "legacy" if arm == "legacy" else "canonical_full",
            "parameter_sha256": legacy_sha if arm == "legacy" else canonical_sha,
        }
        if not gate_a._json_exact(refs[arm], expected):
            return False
    return True


def _optimizer_report_complete(
    value: Any,
    *,
    arm: str,
    expected_paths: tuple[str, ...],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "included",
        "excluded",
        "fresh_state_finite",
        "fresh_state_all_zero",
        "state_leaf_count",
        "value_count",
        "state_sha256",
        "executed_updates",
    }:
        return False
    return bool(
        value["included"] == list(_optimizer_parameter_paths(expected_paths, arm))
        and value["excluded"] == list(ARM_SPECS[arm].optimizer_excluded_paths)
        and _strict_integer(value["state_leaf_count"])
        and int(value["state_leaf_count"]) > 0
        and _strict_integer(value["value_count"])
        and int(value["value_count"]) > 0
        and _sha256_complete(value["state_sha256"])
    )


def _optimizer_paths_complete(initialization: Mapping[str, Any]) -> bool:
    reports = initialization["optimizer_paths"]
    if not isinstance(reports, Mapping) or set(reports) != set(ARM_ORDER):
        return False
    return all(
        _optimizer_report_complete(
            reports[arm],
            arm=arm,
            expected_paths=(
                SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
            ),
        )
        for arm in ARM_ORDER
    )


def _optimizer_states_complete(initialization: Mapping[str, Any]) -> bool:
    return all(
        initialization["optimizer_paths"][arm]["fresh_state_finite"] is True
        and initialization["optimizer_paths"][arm]["fresh_state_all_zero"] is True
        for arm in ARM_ORDER
    )


def _no_behavioral_updates(report: Mapping[str, Any]) -> bool:
    allowed = {
        "schema_version",
        "control",
        "qualification_regime",
        "prerequisites",
        "regimes",
        "initialization",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
    }
    expected_initialization_keys = {
        "canonical_full",
        "legacy",
        "shared_paths",
        "arm_initialization_refs",
        "optimizer_paths",
    }
    return bool(
        set(report).issubset(allowed)
        and isinstance(report.get("initialization"), Mapping)
        and set(report["initialization"]) == set(REGIME_ORDER)
        and all(
            isinstance(report["initialization"][regime], Mapping)
            and set(report["initialization"][regime])
            == expected_initialization_keys
            for regime in REGIME_ORDER
        )
        and all(
            report["initialization"][regime]["optimizer_paths"][arm][
                "executed_updates"
            ]
            == 0
            and _strict_integer(
                report["initialization"][regime]["optimizer_paths"][arm][
                    "executed_updates"
                ]
            )
            for regime in REGIME_ORDER
            for arm in ARM_ORDER
        )
    )


def _gate_c_initialization_qualification(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Recompute every Gate C initialization admission criterion."""

    criteria = {
        name: False for name in GATE_C_INITIALIZATION_QUALIFICATION_CRITERIA
    }
    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        return {
            "criteria": criteria,
            "passed": False,
            "interpretation": "gate_c_initialization_admission_failed_stop",
        }
    if config.qualification_regime != "preregistered_full":
        return {
            "criteria": criteria,
            "passed": False,
            "interpretation": "gate_c_initialization_admission_failed_stop",
        }
    try:
        base_keys = {
            "schema_version",
            "control",
            "qualification_regime",
            "prerequisites",
            "regimes",
            "initialization",
            "source_start",
            "source_end",
            "source_files",
            "environment",
        }
        criteria["schema_and_control"] = bool(
            set(report) in (base_keys, base_keys | {"qualification"})
            and _strict_integer(report["schema_version"])
            and int(report["schema_version"]) == GATE_C_SCHEMA_VERSION
            and report["control"] == GATE_C_INITIALIZATION_CONTROL
        )
        criteria["preregistered_regimes"] = _regimes_complete(report, config)
        prerequisites = report["prerequisites"]
        exact_prerequisites = isinstance(prerequisites, Mapping) and set(
            prerequisites
        ) == {"gate_a", "gate_b"}
        criteria["gate_a_prerequisite_authenticated"] = bool(
            exact_prerequisites
            and "gate_a" in prerequisites
            and gate_a._json_exact(prerequisites["gate_a"], _GATE_A_REFERENCE)
        )
        criteria["gate_b_prerequisite_authenticated"] = bool(
            exact_prerequisites
            and "gate_b" in prerequisites
            and gate_a._json_exact(prerequisites["gate_b"], _GATE_B_REFERENCE)
        )
        criteria["source_and_gpu_authenticated"] = _source_and_gpu_complete(report)
        criteria["source_files_exact"] = _source_files_complete(
            report["source_files"]
        )
        initialization = report["initialization"]
        exact_regimes = isinstance(initialization, Mapping) and set(
            initialization
        ) == set(REGIME_ORDER) and all(
            isinstance(initialization[regime], Mapping)
            and set(initialization[regime])
            == {
                "canonical_full",
                "legacy",
                "shared_paths",
                "arm_initialization_refs",
                "optimizer_paths",
            }
            for regime in REGIME_ORDER
        )
        criteria["schema_and_control"] = bool(
            criteria["schema_and_control"] and exact_regimes
        )
        if exact_regimes:
            criteria["canonical_full_initializations_exact"] = all(
                _topology_report_complete(
                    initialization[regime]["canonical_full"],
                    config=config,
                    regime=regime,
                    tree="canonical_full",
                )
                for regime in REGIME_ORDER
            )
            criteria["legacy_initializations_complete"] = all(
                _topology_report_complete(
                    initialization[regime]["legacy"],
                    config=config,
                    regime=regime,
                    tree="legacy",
                )
                for regime in REGIME_ORDER
            )
            criteria["shared_paths_byte_identical"] = all(
                _shared_report_complete(initialization[regime]["shared_paths"])
                for regime in REGIME_ORDER
            )
            criteria["arm_initialization_refs_exact"] = all(
                _arm_refs_complete(initialization[regime])
                for regime in REGIME_ORDER
            )
            criteria["optimizer_paths_exact"] = all(
                _optimizer_paths_complete(initialization[regime])
                for regime in REGIME_ORDER
            )
            criteria["fresh_optimizer_states_zero_and_finite"] = all(
                _optimizer_states_complete(initialization[regime])
                for regime in REGIME_ORDER
            )
            criteria["compiler_topologies_complete"] = all(
                _full_compiler_complete(
                    initialization[regime]["canonical_full"]["compiler"]
                )
                and _legacy_compiler_complete(
                    initialization[regime]["legacy"]["compiler"]
                )
                for regime in REGIME_ORDER
            )
            criteria["no_behavioral_updates"] = _no_behavioral_updates(report)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        pass
    passed = bool(criteria and all(criteria.values()))
    return {
        "criteria": criteria,
        "passed": passed,
        "interpretation": (
            "gate_c_initialization_admission_passed"
            if passed
            else "gate_c_initialization_admission_failed_stop"
        ),
    }


def _validated_gate_c_initialization_admission(
    prerequisite: Mapping[str, Any],
    config: GateCConfig,
    *,
    source_start: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_files: Mapping[str, str],
    require_pass: bool,
) -> Mapping[str, Any]:
    """Validate the complete authenticated Gate C initialization wrapper."""

    expected_keys = {
        "target",
        "source_head",
        "image_digest",
        "bundle_sha256",
        "manifest_sha256",
        "preflight_sha256",
        "result_sha256",
        "admission",
    }
    if not isinstance(prerequisite, Mapping) or set(prerequisite) != expected_keys:
        raise ValueError("Gate C initialization prerequisite is not authenticated")
    source_head = prerequisite["source_head"]
    image_digest = prerequisite["image_digest"]
    if (
        prerequisite["target"] != "gate_c_init"
        or not isinstance(source_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
        or not all(
            _sha256_complete(prerequisite[name])
            for name in (
                "bundle_sha256",
                "manifest_sha256",
                "preflight_sha256",
                "result_sha256",
            )
        )
    ):
        raise ValueError("Gate C initialization provenance fields are invalid")
    admission = prerequisite["admission"]
    if not isinstance(admission, Mapping):
        raise ValueError("Gate C initialization admission is missing")
    admission_keys = {
        "schema_version",
        "control",
        "qualification_regime",
        "prerequisites",
        "regimes",
        "initialization",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
    }
    if set(admission) != admission_keys:
        raise ValueError("Gate C initialization admission schema is invalid")
    if gate_b._strict_json_sha256(admission) != prerequisite["result_sha256"]:
        raise ValueError("Gate C initialization result digest is invalid")
    expected_bundle = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_c_init\0"
            f"{source_head}\0{prerequisite['preflight_sha256']}\0"
            f"{prerequisite['result_sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    if prerequisite["bundle_sha256"] != expected_bundle:
        raise ValueError("Gate C initialization bundle digest is invalid")
    qualification = _gate_c_initialization_qualification(admission, config=config)
    if not gate_a._json_exact(admission.get("qualification"), qualification):
        raise ValueError("Gate C initialization qualification is stale")
    if require_pass and qualification["passed"] is not True:
        raise ValueError("Gate C initialization admission did not pass")
    try:
        source_matches = bool(
            admission["source_start"]["commit"] == source_head
            and admission["source_end"]["commit"] == source_head
            and admission["environment"]["image_digest"] == image_digest
            and source_start["commit"] == source_head
            and environment["image_digest"] == image_digest
        )
    except (KeyError, TypeError):
        source_matches = False
    if not source_matches:
        raise ValueError("Gate C initialization source or image differs")
    if not _source_and_gpu_complete(
        {
            "source_start": source_start,
            "source_end": source_start,
            "environment": environment,
        }
    ):
        raise ValueError("Gate C formal source or GPU evidence is invalid")
    if not (
        gate_a._json_exact(admission.get("source_files"), source_files)
        and _source_files_complete(source_files)
    ):
        raise ValueError("Gate C initialization source files differ")
    return admission


def _arm_initialization_reproduced(
    report: Mapping[str, Any],
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> bool:
    """Check one formal arm against its authenticated initialization evidence."""

    try:
        _regime_spec(regime)
        _arm_spec(arm)
        if not isinstance(report, Mapping) or set(report) != {
            "initialization",
            "optimizer",
            "compiler",
        }:
            return False
        initialization = report["initialization"]
        if not isinstance(initialization, Mapping) or set(initialization) != {
            "tree",
            "parameter_sha256",
            "parameter_count",
            "parameter_paths",
            "shared_paths",
        }:
            return False
        regime_admission = admission["initialization"][regime]
        reference = regime_admission["arm_initialization_refs"][arm]
        topology = regime_admission[reference["tree"]]
        expected_paths = (
            SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
        )
        return bool(
            initialization["tree"] == reference["tree"]
            and initialization["parameter_sha256"]
            == reference["parameter_sha256"]
            and _strict_integer(initialization["parameter_count"])
            and initialization["parameter_count"] == topology["parameter_count"]
            and initialization["parameter_paths"] == list(expected_paths)
            and gate_a._json_exact(
                initialization["shared_paths"],
                regime_admission["shared_paths"],
            )
            and gate_a._json_exact(
                report["optimizer"],
                regime_admission["optimizer_paths"][arm],
            )
            and gate_a._json_exact(report["compiler"], topology["compiler"])
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _formal_arm_initialization_report(
    model: LatentWorkspaceModel,
    trainer: GateCTrainer,
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> dict[str, Any]:
    """Rebuild and bind one arm's complete pre-update initialization report."""

    reference = admission["initialization"][regime]["arm_initialization_refs"][arm]
    tree = str(reference["tree"])
    topology = _initialization_topology_report(
        model,
        trainer,
        regime=regime,
        tree=tree,
    )
    values = legacy._parameter_values(model)
    shared_values = {path: values[path] for path in SHARED_PARAMETER_PATHS}
    shared = admission["initialization"][regime]["shared_paths"]
    digest_key = "legacy" if arm == "legacy" else "canonical"
    actual_path_digests = {
        path: _shared_path_sha256(path, shared_values[path])
        for path in SHARED_PARAMETER_PATHS
    }
    actual_shared = {
        **shared,
        f"{digest_key}_path_sha256": actual_path_digests,
        f"{digest_key}_sha256": _shared_global_sha256(shared_values),
    }
    report = {
        "initialization": {
            "tree": tree,
            "parameter_sha256": topology["parameter_sha256"],
            "parameter_count": topology["parameter_count"],
            "parameter_paths": topology["parameter_paths"],
            "shared_paths": actual_shared,
        },
        "optimizer": _optimizer_initial_state_report(
            trainer,
            regime=regime,
            arm=arm,
        ),
        "compiler": topology["compiler"],
    }
    if not _arm_initialization_reproduced(
        report,
        admission,
        regime,
        arm,
    ):
        raise RuntimeError("formal arm initialization was not reproduced")
    return report


def _fresh_formal_arm(
    config: GateCConfig,
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> tuple[LatentWorkspaceModel, GateCTrainer, dict[str, Any]]:
    """Construct and authenticate one fresh formal arm before any update."""

    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    if arm == "legacy":
        canonical = _new_model_for_arm(
            config,
            regime,
            "full",
            batch_size=regime_config.batch_size,
        )
        model = _new_model_for_arm(
            config,
            regime,
            arm,
            batch_size=regime_config.batch_size,
        )
        _copy_shared_initialization(canonical, model)
        del canonical
    else:
        model = _new_model_for_arm(
            config,
            regime,
            arm,
            batch_size=regime_config.batch_size,
        )
    trainer = _make_arm_trainer(model, config, regime, arm)
    report = _formal_arm_initialization_report(
        model,
        trainer,
        admission,
        regime,
        arm,
    )
    return model, trainer, report


def _source_files_report() -> dict[str, str]:
    """Hash the exact six scientific source files for Gate C."""

    repo_root = Path(__file__).resolve().parents[2]
    return {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in _GATE_C_SOURCE_FILES
    }


def write_artifact(value: Mapping[str, Any], path: str | Path) -> Path:
    """Write one deterministic, strict Gate C artifact atomically.

    Parameters
    ----------
    value
        JSON-compatible top-level mapping. NaN and infinity are rejected.
    path
        Final artifact path.

    Returns
    -------
    pathlib.Path
        Final artifact path after atomic replacement.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def run_gate_c_initialization(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and inspect all Gate C initial states without behavior.

    Parameters
    ----------
    config
        Fixed paired Gate C configuration.
    prerequisites
        Launcher-authenticated compact Gate A and Gate B references.
    source_start
        Live clean-source evidence captured before construction.
    source_end_reporter
        Zero-argument callback that captures live source evidence after every
        topology and optimizer report is complete.
    source_files
        Exact six-file scientific source digest mapping.
    environment
        Authenticated GPU image and device evidence.

    Returns
    -------
    dict
        Strict initialization-only artifact payload.
    """

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    normalized_prerequisites = _normalized_prerequisites(prerequisites)
    source_keys = {
        "asserted_commit",
        "asserted_commit_matches_head",
        "asserted_dirty",
        "asserted_dirty_matches_worktree",
        "commit",
        "commit_is_valid_40_hex",
        "dirty",
        "head_command_succeeded",
        "status_command_succeeded",
        "verified",
    }
    authenticate_inputs = bool(
        config.qualification_regime == "preregistered_full"
        or (isinstance(source_start, Mapping) and set(source_start) == source_keys)
    )
    if authenticate_inputs:
        if not isinstance(source_start, Mapping) or not gate_a._source_evidence_clean(
            source_start
        ):
            raise RuntimeError("Gate C initialization source is not authenticated")
        if not isinstance(environment, Mapping) or not gate_a._gpu_environment_verified(
            environment
        ):
            raise RuntimeError("Gate C initialization GPU is not authenticated")
        if not _source_files_complete(source_files):
            raise RuntimeError("Gate C initialization source files are not exact")
    initialization: dict[str, Any] = {}
    regime_reports = {
        regime: {
            "spec": dataclasses.asdict(REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
        }
        for regime in REGIME_ORDER
    }

    for regime in REGIME_ORDER:
        regime_config = (
            config.gate_a_config if regime == "gate_a" else config.gate_b_config
        )
        models = {
            arm: _new_model_for_arm(
                config,
                regime,
                arm,
                batch_size=regime_config.batch_size,
            )
            for arm in ARM_ORDER
        }
        _copy_shared_initialization(models["full"], models["legacy"])
        trainers = {
            arm: _make_arm_trainer(models[arm], config, regime, arm)
            for arm in ARM_ORDER
        }
        canonical = _initialization_topology_report(
            models["full"],
            trainers["full"],
            regime=regime,
            tree="canonical_full",
        )
        legacy_report = _initialization_topology_report(
            models["legacy"],
            trainers["legacy"],
            regime=regime,
            tree="legacy",
        )
        canonical_values = legacy._parameter_values(models["full"])
        legacy_values = legacy._parameter_values(models["legacy"])
        canonical_shared = {
            path: canonical_values[path] for path in SHARED_PARAMETER_PATHS
        }
        legacy_shared = {
            path: legacy_values[path] for path in SHARED_PARAMETER_PATHS
        }
        canonical_path_sha256 = {
            path: _shared_path_sha256(path, canonical_shared[path])
            for path in SHARED_PARAMETER_PATHS
        }
        legacy_path_sha256 = {
            path: _shared_path_sha256(path, legacy_shared[path])
            for path in SHARED_PARAMETER_PATHS
        }
        canonical_shared_sha256 = _shared_global_sha256(canonical_shared)
        legacy_shared_sha256 = _shared_global_sha256(legacy_shared)
        arm_refs = {
            arm: {
                "tree": "legacy" if arm == "legacy" else "canonical_full",
                "parameter_sha256": legacy._array_digest(
                    legacy._parameter_values(models[arm])
                ),
            }
            for arm in ARM_ORDER
        }
        optimizer_paths = {
            arm: _optimizer_initial_state_report(
                trainers[arm], regime=regime, arm=arm
            )
            for arm in ARM_ORDER
        }
        initialization[regime] = {
            "canonical_full": canonical,
            "legacy": legacy_report,
            "shared_paths": {
                "paths": list(SHARED_PARAMETER_PATHS),
                "framing": {
                    "path": "example21-gate-c-shared-path-v1",
                    "global": "example21-gate-c-shared-global-v1",
                },
                "canonical_path_sha256": canonical_path_sha256,
                "legacy_path_sha256": legacy_path_sha256,
                "canonical_sha256": canonical_shared_sha256,
                "legacy_sha256": legacy_shared_sha256,
                "all_equal": bool(
                    canonical_path_sha256 == legacy_path_sha256
                    and canonical_shared_sha256 == legacy_shared_sha256
                ),
            },
            "arm_initialization_refs": arm_refs,
            "optimizer_paths": optimizer_paths,
        }

    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    report: dict[str, Any] = {
        "schema_version": GATE_C_SCHEMA_VERSION,
        "control": GATE_C_INITIALIZATION_CONTROL,
        "qualification_regime": config.qualification_regime,
        "prerequisites": normalized_prerequisites,
        "regimes": regime_reports,
        "initialization": initialization,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": dict(environment),
    }
    report["qualification"] = _gate_c_initialization_qualification(
        report, config=config
    )
    return report


def _evaluate_arm(
    trained_model: LatentWorkspaceModel,
    data: legacy.BindingData | gate_b.DepthValidationData,
    config: GateCConfig,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    """Evaluate one trained arm on its three canonical held-out streams."""

    _regime_spec(regime)
    _arm_spec(arm)
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    validation_episodes = regime_config.validation_episodes
    model = _new_model_for_arm(
        config,
        regime,
        arm,
        batch_size=validation_episodes,
    )
    legacy._copy_parameters(trained_model, model)
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A evaluation requires BindingData")
        event_streams = {
            "intact": data.validation_intact,
            "shuffled": data.validation_shuffled,
            "no_context": data.validation_no_context,
        }
        advances = jnp.ones(
            (regime_config.sequence_length, validation_episodes), dtype=jnp.bool_
        )
        targets_by_depth = np.broadcast_to(
            np.asarray(data.validation_targets)[None, :],
            (regime_config.gap_steps + 1, validation_episodes),
        )
    else:
        if not isinstance(data, gate_b.DepthValidationData):
            raise TypeError("Gate B evaluation requires DepthValidationData")
        event_streams = {
            "intact": data.intact,
            "shuffled": data.shuffled,
            "no_context": data.no_context,
        }
        advances = jnp.asarray(data.advance_masks)
        targets_by_depth = np.asarray(data.targets_by_depth)
    model_states = tuple(model.states().values())
    checkpoint_start = regime_config.sequence_length - regime_config.gap_steps - 1

    @brainstate.transform.jit
    def evaluate_stream(
        events: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        model.reset_state()

        def step(
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            event, advance = inputs
            compact = model.update(event, advance)
            state_finite = jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
                        for state in model_states
                        for leaf in jax.tree.leaves(state.value)
                    ]
                )
            )
            if model.config.memory_enabled:
                memory_read = jnp.asarray(model.memory_read.value)
                workspace = jnp.asarray(model.workspace_carrier.value)
            else:
                memory_read = jnp.zeros(
                    (validation_episodes, 0), dtype=jnp.float32
                )
                workspace = jnp.zeros(
                    (validation_episodes, 0), dtype=jnp.float32
                )
            return compact, state_finite, memory_read, workspace

        compact, state_finite, reads, workspaces = brainstate.transform.for_loop(
            step, (events, advances)
        )
        final_memory = (
            jnp.asarray(model.context_memory.value)
            if model.config.memory_enabled
            else jnp.zeros((validation_episodes, 0, 0), dtype=jnp.float32)
        )
        return (
            compact[checkpoint_start:],
            state_finite,
            reads[checkpoint_start:],
            workspaces[checkpoint_start:],
            final_memory,
        )

    raw: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for name, events in event_streams.items():
        values = jax.block_until_ready(evaluate_stream(jnp.asarray(events)))
        raw[name] = tuple(np.asarray(value) for value in values)  # type: ignore[assignment]
    finite = all(
        bool(np.isfinite(value).all())
        for values in raw.values()
        for value in values
    ) and all(bool(values[1].all()) for values in raw.values())
    predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in event_streams
    }
    depth_reports: dict[str, dict[str, Any]] = {}
    for depth in range(regime_config.gap_steps + 1):
        depth_metrics: dict[str, Any] = {}
        for name in event_streams:
            color_logits = np.asarray(
                legacy._color_logits(jnp.asarray(raw[name][0][depth]), regime_config.color_rank)
            )
            finite = finite and bool(np.isfinite(color_logits).all())
            prediction = np.argmax(color_logits, axis=-1)
            predictions[name].append(prediction)
            depth_metrics[name] = {
                **legacy._accuracy(prediction, targets_by_depth[depth]),
                "checkpoint": depth,
            }
        depth_reports[str(depth)] = depth_metrics

    if regime == "gate_a":
        final = depth_reports[str(regime_config.gap_steps)]
        binding_diagnostic = gate_a._diagnostic_report(
            {
                name: (values[4], values[2], values[3])
                for name, values in raw.items()
            }
        )
        memory = binding_diagnostic["memory"]
        binding_state = {
            "applicable_count": memory["applicable_count"],
            "intact_shuffled_different_count": memory[
                "intact_shuffled_different_count"
            ],
            "every_intact_shuffled_pair_differs": memory[
                "every_intact_shuffled_pair_differs"
            ],
            "no_context_exact_zero": memory["no_context_exact_zero"],
            "intact_sha256": memory["left_sha256"],
            "shuffled_sha256": memory["right_sha256"],
            "no_context_sha256": memory["no_context_sha256"],
            "all_finite": binding_diagnostic["all_state_tensors_finite"],
        }
        return {
            "finite": finite,
            "all_compact_logits_finite": finite,
            "all_state_tensors_finite": bool(
                all(values[1].all() for values in raw.values())
            ),
            "depths": depth_reports,
            "intact": final["intact"],
            "shuffled": final["shuffled"],
            "no_context": final["no_context"],
            "intact_minus_shuffled": (
                final["intact"]["accuracy"] - final["shuffled"]["accuracy"]
            ),
            "binding_state": binding_state,
            "binding_diagnostic": binding_diagnostic,
        }

    h0_predictions = predictions["intact"][0]
    efforts: dict[str, dict[str, Any]] = {}
    for effort in gate_b.QUALIFYING_EFFORTS:
        intact = depth_reports[str(effort)]["intact"]
        shuffled = depth_reports[str(effort)]["shuffled"]
        no_context = depth_reports[str(effort)]["no_context"]
        h0_final = {
            **legacy._accuracy(h0_predictions, targets_by_depth[effort]),
            "checkpoint": 0,
        }
        efforts[str(effort)] = {
            "intact": intact,
            "shuffled": shuffled,
            "no_context": no_context,
            "h0_final_target": h0_final,
            "intact_minus_h0": intact["accuracy"] - h0_final["accuracy"],
            "intact_minus_shuffled": (
                intact["accuracy"] - shuffled["accuracy"]
            ),
        }
    return {
        "finite": finite,
        "h0_proper": depth_reports["0"]["intact"],
        "depths": depth_reports,
        "efforts": efforts,
    }


def _parameter_movement_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    path_reports: dict[str, dict[str, float | int]] = {}
    total_squared = 0.0
    total_count = 0
    if set(before) != set(after):
        raise RuntimeError("formal arm parameter paths changed during training")
    for path in sorted(before):
        squared = 0.0
        count = 0
        for old, new in zip(
            jax.tree.leaves(before[path]),
            jax.tree.leaves(after[path]),
            strict=True,
        ):
            delta = np.asarray(new, dtype=np.float64) - np.asarray(
                old, dtype=np.float64
            )
            squared += float(np.sum(delta * delta, dtype=np.float64))
            count += int(delta.size)
        path_reports[path] = {
            "l2_delta": math.sqrt(squared),
            "parameter_count": count,
        }
        total_squared += squared
        total_count += count
    return {
        "l2_delta": math.sqrt(total_squared),
        "parameter_count": total_count,
        "paths": path_reports,
    }


def _aggregate_training_telemetry(
    chunks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("formal arm executed no training chunks")
    losses = np.concatenate(
        [np.asarray(chunk["loss"], dtype=np.float64).reshape(-1) for chunk in chunks]
    )
    categories = tuple(chunks[0]["finite"])
    finite = {
        category: bool(
            all(
                np.asarray(chunk["finite"][category], dtype=np.bool_).all()
                for chunk in chunks
            )
        )
        for category in categories
    }
    maxima = {
        category: float(
            max(
                np.max(
                    np.asarray(chunk["max_abs"][category], dtype=np.float64),
                    initial=0.0,
                )
                for chunk in chunks
            )
        )
        for category in categories
    }
    value_counts = {
        category: int(
            sum(
                int(
                    np.sum(
                        np.asarray(
                            chunk["value_count"][category],
                            dtype=np.int64,
                        )
                    )
                )
                for chunk in chunks
            )
        )
        for category in categories
    }
    if not np.isfinite(losses).all():
        raise RuntimeError("formal arm produced a non-finite training loss")
    return {
        "losses": losses.tolist(),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "tail_64_mean_loss": float(losses[-min(64, losses.size) :].mean()),
        "finite": finite,
        "max_abs": maxima,
        "value_count": value_counts,
    }


def _run_gate_c_arm(
    model: LatentWorkspaceModel,
    trainer: GateCTrainer,
    data: legacy.BindingData | tuple[gate_b.DepthSchedule, gate_b.DepthValidationData],
    config: GateCConfig,
    regime: str,
    arm: str,
    *,
    initialization_report: Mapping[str, Any],
    execution_index: int | None = None,
    data_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate one isolated formal Gate C arm."""

    _regime_spec(regime)
    _arm_spec(arm)
    expected_execution_index = (
        REGIME_ORDER.index(regime) * len(ARM_ORDER) + ARM_ORDER.index(arm)
    )
    if execution_index is None:
        execution_index = expected_execution_index
    if (
        not _strict_integer(execution_index)
        or int(execution_index) != expected_execution_index
    ):
        raise ValueError("formal arm execution index differs from the fixed order")
    before = legacy._parameter_values(model)
    before_sha256 = legacy._array_digest(before)
    telemetry: list[Mapping[str, Any]] = []
    consumed_weight_chunks: list[np.ndarray] = []
    start = time.perf_counter()
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A arm requires BindingData")
        updates = config.gate_a_config.training_updates
        efforts = np.ones((updates,), dtype=np.int32)
        advances = np.ones(
            (
                updates,
                config.gate_a_config.sequence_length,
                config.gate_a_config.batch_size,
            ),
            dtype=np.bool_,
        )
        weights = _loss_weights("gate_a", arm, efforts=efforts)
        consumed_weight_chunks.append(np.asarray(weights))
        telemetry.append(
            jax.device_get(
                jax.block_until_ready(
                    trainer.train_chunk(
                        data.training_events,
                        data.training_targets,
                        weights,
                        advances,
                    )
                )
            )
        )
        evaluation_data: legacy.BindingData | gate_b.DepthValidationData = data
        chunk_count = 1
        actual_data_identity = {
            "training_schedule_sha256": legacy._digest_arrays(
                data.training_events,
                data.training_targets,
                data.training_mapping_ids,
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
    else:
        if not isinstance(data, tuple) or len(data) != 2:
            raise TypeError("Gate B arm requires its schedule and validation data")
        schedule, validation = data
        if not isinstance(schedule, gate_b.DepthSchedule) or not isinstance(
            validation, gate_b.DepthValidationData
        ):
            raise TypeError("Gate B arm data has the wrong type")
        chunk_count = 0
        hash_state = gate_b._new_encoded_schedule_hash_state()
        for schedule_chunk in gate_b._iter_schedule_chunks(
            schedule,
            config.gate_b_config,
        ):
            encoded = gate_b._encode_training_chunk(
                schedule_chunk,
                config.gate_b_config,
            )
            gate_b._update_encoded_schedule_hash_state(
                hash_state,
                encoded,
                config.gate_b_config,
            )
            weights = _loss_weights(
                "gate_b",
                arm,
                efforts=np.asarray(encoded.efforts),
            )
            consumed_weight_chunks.append(np.asarray(weights))
            telemetry.append(
                jax.device_get(
                    jax.block_until_ready(
                        trainer.train_chunk(
                            encoded.events,
                            encoded.targets,
                            weights,
                            encoded.advance_masks,
                        )
                    )
                )
            )
            chunk_count += 1
        evaluation_data = validation
        encoded_identity = gate_b._finish_encoded_schedule_report(
            hash_state,
            config.gate_b_config,
        )
        actual_data_identity = {
            "training_global_sha256": dict(encoded_identity["global_sha256"]),
            "validation_sha256": dict(
                gate_b._validation_data_report(validation)["sha256"]
            ),
        }
    if data_identity is not None and not gate_a._json_exact(
        data_identity, actual_data_identity
    ):
        raise ValueError("formal arm data identity differs from consumed bytes")
    training_seconds = time.perf_counter() - start
    after = legacy._parameter_values(model)
    aggregate = _aggregate_training_telemetry(telemetry)
    consumed_weights = (
        consumed_weight_chunks[0]
        if regime == "gate_a"
        else np.concatenate(consumed_weight_chunks, axis=0)
    )
    executed_updates = len(aggregate["losses"])
    training = {
        "algorithm": trainer.algorithm,
        "execution_index": int(execution_index),
        "intervention": dataclasses.asdict(ARM_SPECS[arm]),
        "data_identity": actual_data_identity,
        "executed_updates": executed_updates,
        "batch_size": (
            config.gate_a_config.batch_size
            if regime == "gate_a"
            else config.gate_b_config.batch_size
        ),
        "chunk_count": chunk_count,
        "cold_compile_and_train_seconds": training_seconds,
        "initial_parameter_sha256": before_sha256,
        "final_parameter_sha256": legacy._array_digest(after),
        "optimizer_final_step": int(
            np.asarray(jax.device_get(trainer.optimizer.step_count.value)).item()
        ),
        "loss_weights": {
            "dtype": consumed_weights.dtype.str,
            "shape": list(consumed_weights.shape),
            "sha256": legacy._digest_arrays(consumed_weights),
        },
        "compile_warnings": list(trainer.compile_warnings),
        **aggregate,
    }
    if training["optimizer_final_step"] != executed_updates:
        raise RuntimeError("formal arm optimizer step count differs from updates")
    if not all(training["finite"].values()):
        raise RuntimeError("formal arm telemetry is not finite")
    movement = _parameter_movement_report(before, after)
    write_before = before.get("memory_write_scale")
    write_after = after.get("memory_write_scale")
    training["frozen_write"] = {
        "applicable": arm == "frozen_write",
        "all_ones_before": bool(
            write_before is not None
            and all(
                np.equal(np.asarray(leaf), 1.0).all()
                for leaf in jax.tree.leaves(write_before)
            )
        ),
        "all_ones_after": bool(
            write_after is not None
            and all(
                np.equal(np.asarray(leaf), 1.0).all()
                for leaf in jax.tree.leaves(write_after)
            )
        ),
        "excluded_from_optimizer": (
            "memory_write_scale" in trainer.excluded_optimizer_paths
        ),
    }
    evaluation = _evaluate_arm(
        model,
        evaluation_data,
        config,
        regime,
        arm,
    )
    return {
        "initialization": dict(initialization_report["initialization"]),
        "optimizer": dict(initialization_report["optimizer"]),
        "compiler": dict(initialization_report["compiler"]),
        "training": training,
        "parameter_movement": movement,
        "evaluation": evaluation,
        "metrics": {},
    }


def _schedule_identity_report(config: GateCConfig) -> dict[str, Any]:
    """Return the preregistered schedule identities for both regimes."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    return {
        "gate_a": dict(_GATE_A_SCHEDULE_SHA256),
        "gate_b": {
            "training_global_sha256": dict(
                gate_b._PRODUCTION_ENCODED_GLOBAL_SHA256
            ),
            "validation_sha256": dict(gate_b._PRODUCTION_VALIDATION_SHA256),
        },
    }


def _actual_schedule_identity_report(
    config: GateCConfig,
    gate_a_data: legacy.BindingData,
    gate_b_data: tuple[gate_b.DepthSchedule, gate_b.DepthValidationData],
) -> dict[str, Any]:
    """Hash the generated Gate A and Gate B schedule bytes."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if not isinstance(gate_a_data, legacy.BindingData):
        raise TypeError("Gate C schedule evidence requires BindingData")
    if (
        not isinstance(gate_b_data, tuple)
        or len(gate_b_data) != 2
        or not isinstance(gate_b_data[0], gate_b.DepthSchedule)
        or not isinstance(gate_b_data[1], gate_b.DepthValidationData)
    ):
        raise TypeError("Gate C schedule evidence requires Gate B data")
    schedule, validation = gate_b_data
    gate_b_training = gate_b._encoded_schedule_report(
        schedule,
        config.gate_b_config,
    )
    return {
        "gate_a": {
            "training_schedule_sha256": legacy._digest_arrays(
                gate_a_data.training_events,
                gate_a_data.training_targets,
                gate_a_data.training_mapping_ids,
            ),
            "validation_schedule_sha256": legacy._digest_arrays(
                gate_a_data.validation_intact,
                gate_a_data.validation_targets,
                gate_a_data.validation_mapping_ids,
            ),
            "training_mapping_ids_sha256": legacy._digest_arrays(
                gate_a_data.training_mapping_ids.reshape(-1)
            ),
            "validation_mapping_ids_sha256": legacy._digest_arrays(
                gate_a_data.validation_mapping_ids
            ),
        },
        "gate_b": {
            "training_global_sha256": dict(gate_b_training["global_sha256"]),
            "validation_sha256": dict(
                gate_b._validation_data_report(validation)["sha256"]
            ),
        },
    }


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _metric_summary(
    gate_a_evaluation: Mapping[str, Any],
    gate_b_evaluation: Mapping[str, Any],
) -> dict[str, float]:
    """Compute Gate C binding and demonstrated-depth metrics."""

    try:
        binding = gate_a_evaluation["depths"]["1"]
        intact = _finite_real(binding["intact"]["accuracy"], "Gate A intact accuracy")
        shuffled = _finite_real(
            binding["shuffled"]["accuracy"], "Gate A shuffled accuracy"
        )
        depth_values = [
            _finite_real(
                gate_b_evaluation["efforts"][str(effort)]["intact"]["accuracy"],
                f"Gate B effort {effort} intact accuracy",
            )
            for effort in gate_b.QUALIFYING_EFFORTS
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("evaluation evidence is incomplete") from error
    for value in (intact, shuffled, *depth_values):
        if not 0.0 <= value <= 1.0:
            raise ValueError("accuracy evidence must lie in [0, 1]")
    return {
        "binding_gap": intact - shuffled,
        "depth_accuracy": math.fsum(depth_values) / len(depth_values),
    }


def _decimal_margin(full: float, arm: float) -> Decimal:
    return Decimal(str(full)) - Decimal(str(arm))


def _blocking_margin_report(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute all blocking and characterization-only Gate C margins."""

    if set(metrics) != set(ARM_ORDER):
        raise ValueError("metrics must contain exactly the five Gate C arms")
    values: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        item = metrics[arm]
        values[arm] = {
            name: _finite_real(item[name], f"{arm} {name}")
            for name in ("binding_gap", "depth_accuracy")
        }
    full = values["full"]

    def comparison(
        arm: str,
        *,
        depth_threshold: float,
        binding_threshold: float,
        blocking: bool = True,
    ) -> dict[str, Any]:
        depth_margin = full["depth_accuracy"] - values[arm]["depth_accuracy"]
        binding_margin = full["binding_gap"] - values[arm]["binding_gap"]
        depth_passed = _decimal_margin(
            full["depth_accuracy"], values[arm]["depth_accuracy"]
        ) >= Decimal(str(depth_threshold))
        binding_passed = _decimal_margin(
            full["binding_gap"], values[arm]["binding_gap"]
        ) >= Decimal(str(binding_threshold))
        return {
            "binding_gap_difference": binding_margin,
            "depth_accuracy_difference": depth_margin,
            "binding_threshold": binding_threshold,
            "depth_threshold": depth_threshold,
            "binding_passed": bool(binding_passed),
            "depth_passed": bool(depth_passed),
            "blocking": blocking,
            "passed": bool(binding_passed and depth_passed),
        }

    query = comparison("query_only", depth_threshold=0.15, binding_threshold=-0.02)
    terminal = comparison(
        "terminal_only", depth_threshold=0.10, binding_threshold=-0.02
    )
    legacy = comparison("legacy", depth_threshold=0.15, binding_threshold=0.25)
    frozen = comparison(
        "frozen_write", depth_threshold=0.05, binding_threshold=0.05, blocking=False
    )
    frozen["write_modulation_necessary"] = frozen["passed"]
    frozen["interpretation"] = (
        "learned_memory_write_modulation_necessary"
        if frozen["passed"]
        else "learned_memory_write_modulation_not_shown_necessary"
    )
    return {
        "query_only": query,
        "terminal_only": terminal,
        "legacy": legacy,
        "frozen_write": frozen,
        "blocking_passed": bool(
            query["passed"] and terminal["passed"] and legacy["passed"]
        ),
    }


def _hidden_state_sha256(model: LatentWorkspaceModel) -> str:
    fields: list[bytes] = [b"example21-gate-c-hidden-state-v1"]
    states = sorted(
        model.states(brainstate.HiddenState).items(),
        key=lambda item: gate_a._path(item[0]),
    )
    for path, state in states:
        fields.append(gate_a._path(path).encode("utf-8"))
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            array = np.ascontiguousarray(np.asarray(u.get_mantissa(leaf)))
            fields.extend(
                (
                    str(index).encode("ascii"),
                    array.dtype.str.encode("ascii"),
                    ",".join(map(str, array.shape)).encode("ascii"),
                    array.tobytes(),
                )
            )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _paired_h0_identity_report(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    regime: str,
    data: Any,
) -> dict[str, Any]:
    """Compare fresh full and query-only models at checkpoint H0."""

    _regime_spec(regime)
    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    try:
        canonical = initialization["initialization"][regime]["canonical_full"]
    except (KeyError, TypeError) as error:
        raise ValueError("Gate C H0 initialization evidence is incomplete") from error
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    count = regime_config.validation_episodes
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A H0 identity requires BindingData")
        streams = {
            "intact": data.validation_intact,
            "shuffled": data.validation_shuffled,
            "no_context": data.validation_no_context,
        }
        advances = np.ones(
            (regime_config.sequence_length, count), dtype=np.bool_
        )
    else:
        if (
            not isinstance(data, tuple)
            or len(data) != 2
            or not isinstance(data[1], gate_b.DepthValidationData)
        ):
            raise TypeError("Gate B H0 identity requires DepthValidationData")
        validation = data[1]
        streams = {
            "intact": validation.intact,
            "shuffled": validation.shuffled,
            "no_context": validation.no_context,
        }
        advances = np.asarray(validation.advance_masks)
    checkpoint_index = regime_config.sequence_length - regime_config.gap_steps - 1
    models = {
        arm: _new_model_for_arm(
            config,
            regime,
            arm,
            batch_size=count,
        )
        for arm in ("full", "query_only")
    }
    legacy._copy_parameters(models["full"], models["query_only"])
    parameter_sha256 = {
        arm: legacy._array_digest(legacy._parameter_values(model))
        for arm, model in models.items()
    }
    if (
        not isinstance(canonical, Mapping)
        or any(
            digest != canonical.get("parameter_sha256")
            for digest in parameter_sha256.values()
        )
    ):
        raise ValueError("Gate C H0 models did not reproduce initialization")
    initial_snapshot = models["full"].snapshot_state()
    models["query_only"].restore_state(initial_snapshot)

    def driver(model: LatentWorkspaceModel) -> Any:
        @brainstate.transform.jit
        def run_h0(events: jax.Array, advance_values: jax.Array) -> jax.Array:
            def step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
                event, advance = inputs
                return model.update(event, advance)

            compact = brainstate.transform.for_loop(
                step,
                (events, advance_values),
            )
            return compact[-1]

        return run_h0

    drivers = {arm: driver(model) for arm, model in models.items()}
    stream_reports: dict[str, Any] = {}
    for name, event_values in streams.items():
        compact_sha256: dict[str, str] = {}
        state_sha256: dict[str, str] = {}
        for arm in ("full", "query_only"):
            models[arm].restore_state(initial_snapshot)
            compact = jax.block_until_ready(
                drivers[arm](
                    jnp.asarray(event_values[: checkpoint_index + 1]),
                    jnp.asarray(advances[: checkpoint_index + 1]),
                )
            )
            compact_sha256[arm] = legacy._digest_arrays(np.asarray(compact))
            state_sha256[arm] = _hidden_state_sha256(models[arm])
        stream_reports[name] = {
            "full_compact_sha256": compact_sha256["full"],
            "query_only_compact_sha256": compact_sha256["query_only"],
            "full_state_sha256": state_sha256["full"],
            "query_only_state_sha256": state_sha256["query_only"],
            "compact_byte_identical": (
                compact_sha256["full"] == compact_sha256["query_only"]
            ),
            "state_byte_identical": (
                state_sha256["full"] == state_sha256["query_only"]
            ),
        }
    passed = all(
        evidence["compact_byte_identical"] is True
        and evidence["state_byte_identical"] is True
        for evidence in stream_reports.values()
    )
    return {
        "checkpoint": 0,
        "initialization_parameter_sha256": parameter_sha256,
        "streams": stream_reports,
        "passed": bool(passed),
    }


def _numeric_gradient_leaves(value: Any) -> list[np.ndarray]:
    leaves: list[np.ndarray] = []
    for leaf in jax.tree.leaves(value):
        array = np.ascontiguousarray(np.asarray(u.get_mantissa(leaf)))
        if np.issubdtype(array.dtype, np.bool_):
            raise TypeError("gradient leaves must be numeric, not boolean")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("gradient leaves must be numeric")
        if not np.isfinite(array).all():
            raise ValueError("gradient leaves must be finite")
        leaves.append(array)
    if not leaves:
        raise ValueError("gradient tree has no leaves")
    return leaves


def _shared_path_sha256(path: str, value: Any) -> str:
    """Hash one shared initialization subtree with Gate C framing."""

    if not isinstance(path, str) or not path:
        raise TypeError("shared parameter path must be a nonempty string")
    fields: list[bytes] = [
        b"example21-gate-c-shared-path-v1",
        path.encode("utf-8"),
    ]
    for index, array in enumerate(_numeric_gradient_leaves(value)):
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _shared_global_sha256(values: Mapping[str, Any]) -> str:
    """Hash all shared initialization paths in canonical name order."""

    if not isinstance(values, Mapping) or not values:
        raise TypeError("shared values must be a nonempty mapping")
    fields: list[bytes] = [b"example21-gate-c-shared-global-v1"]
    for path in sorted(values):
        fields.extend(
            (
                path.encode("utf-8"),
                _shared_path_sha256(path, values[path]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_path_sha256(path: str, value: Any) -> str:
    """Hash one gradient subtree with the frozen Gate C framing."""

    if not isinstance(path, str) or not path:
        raise TypeError("gradient path must be a nonempty string")
    fields: list[bytes] = [
        b"example21-gate-c-gradient-path-v1",
        path.encode("utf-8"),
    ]
    for index, array in enumerate(_numeric_gradient_leaves(value)):
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_global_sha256(gradients: Mapping[str, Any]) -> str:
    """Hash all gradient paths in canonical name order."""

    if not isinstance(gradients, Mapping) or not gradients:
        raise TypeError("gradients must be a nonempty mapping")
    fields: list[bytes] = [b"example21-gate-c-gradient-global-v1"]
    for path in sorted(gradients):
        fields.extend(
            (
                path.encode("utf-8"),
                _gradient_path_sha256(path, gradients[path]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_comparison(full: Any, arm: Any) -> dict[str, Any]:
    """Compare two flattened gradients using the full norm denominator."""

    if jax.tree.structure(full) != jax.tree.structure(arm):
        raise ValueError("gradient trees must have the same structure")
    full_leaves = _numeric_gradient_leaves(full)
    arm_leaves = _numeric_gradient_leaves(arm)
    if any(
        full_leaf.shape != arm_leaf.shape or full_leaf.dtype != arm_leaf.dtype
        for full_leaf, arm_leaf in zip(full_leaves, arm_leaves, strict=True)
    ):
        raise ValueError("gradient leaves must have the same shape and dtype")
    full_vector = np.concatenate([leaf.astype(np.float64).reshape(-1) for leaf in full_leaves])
    arm_vector = np.concatenate([leaf.astype(np.float64).reshape(-1) for leaf in arm_leaves])
    if full_vector.shape != arm_vector.shape:
        raise ValueError("gradient vectors must have the same shape")
    full_norm = float(np.linalg.norm(full_vector))
    arm_norm = float(np.linalg.norm(arm_vector))
    difference = float(np.linalg.norm(arm_vector - full_vector))
    relative = difference / full_norm if full_norm > 0.0 else None
    cosine = (
        float(np.dot(arm_vector, full_vector) / (arm_norm * full_norm))
        if full_norm > 0.0 and arm_norm > 0.0
        else None
    )
    return {
        "full_norm": full_norm,
        "arm_norm": arm_norm,
        "l2_difference": difference,
        "relative_deviation": relative,
        "relative_deviation_defined": relative is not None,
        "cosine": cosine,
        "cosine_defined": cosine is not None,
    }


def _oracle_contract(config: GateCConfig) -> dict[str, Any]:
    """Return the exact preregistered finite-window mechanism episode."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    return {
        "regime": "gate_b",
        "validation_episode_index": config.oracle_validation_index,
        "arm": "intact",
        "effort": config.oracle_effort,
        "batch_size": 1,
        "mapping_id": 232_423,
        "mapping": [6, 7, 5, 2, 0, 4, 8, 9, 1, 3],
        "query_color": 4,
        "presentation_order": [6, 2, 5, 3, 8, 7, 4, 9, 0, 1],
        "shuffled_shift": 1,
        "targets": [0, 6, 8, 1, 7, 9, 3, 2, 5],
        "advance_mask": [True] * 19,
        "events_shape": [19, 47],
        "events_dtype": "<f4",
        "events_sha256": (
            "36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec"
        ),
        "gradient_chunk_size": config.gradient_chunk_size,
    }


def _mechanism_oracle(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    gate_b_data: Any,
) -> dict[str, Any]:
    """Measure the preregistered Gate C finite-window pp-prop mechanism."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if (
        config.oracle_validation_index != 0
        or config.oracle_effort != 8
        or config.gradient_chunk_size != 1
        or config.gradient_chunk_size >= REGIME_SPECS["gate_b"].sequence_length
    ):
        raise ValueError("Gate C oracle coordinates differ from preregistration")
    contract = _oracle_contract(config)
    if (
        not isinstance(gate_b_data, tuple)
        or len(gate_b_data) != 2
        or not isinstance(gate_b_data[0], gate_b.DepthSchedule)
        or not isinstance(gate_b_data[1], gate_b.DepthValidationData)
    ):
        raise TypeError("Gate C oracle requires canonical Gate B data")
    schedule, validation = gate_b_data
    index = contract["validation_episode_index"]
    if not (
        np.array_equal(
            np.asarray(validation.mapping_ids[index]),
            np.asarray(schedule.validation_mapping_ids[index]),
        )
        and np.array_equal(
            np.asarray(validation.query_colors[index]),
            np.asarray(schedule.validation_query_colors[index]),
        )
        and np.array_equal(
            np.asarray(validation.presentation_orders[index]),
            np.asarray(schedule.validation_presentation_orders[index]),
        )
    ):
        raise ValueError("Gate C oracle validation metadata differs from schedule")
    events = np.ascontiguousarray(np.asarray(validation.intact[:, index, :]))
    mapping_id = int(np.asarray(schedule.validation_mapping_ids[index]).item())
    actual_contract = {
        "mapping_id": mapping_id,
        "mapping": gate_b.unrank_ten_cycle(mapping_id).tolist(),
        "query_color": int(np.asarray(schedule.validation_query_colors[index]).item()),
        "presentation_order": np.asarray(
            schedule.validation_presentation_orders[index]
        ).tolist(),
        "shuffled_shift": int(np.asarray(validation.shuffled_shifts[index]).item()),
        "targets": np.asarray(validation.targets_by_depth[:, index]).tolist(),
        "advance_mask": np.asarray(validation.advance_masks[:, index]).tolist(),
        "events_shape": list(events.shape),
        "events_dtype": events.dtype.str,
        "events_sha256": legacy._digest_arrays(events),
    }
    if any(
        not gate_a._json_exact(actual_contract[name], contract[name])
        for name in actual_contract
    ):
        raise ValueError("Gate C oracle event contract or digest differs")

    try:
        gate_b_initialization = initialization["initialization"]["gate_b"]
        canonical = gate_b_initialization["canonical_full"]
        arm_refs = gate_b_initialization["arm_initialization_refs"]
    except (KeyError, TypeError) as error:
        raise ValueError("Gate C oracle initialization evidence is incomplete") from error
    if (
        not isinstance(canonical, Mapping)
        or canonical.get("parameter_paths") != list(FULL_PARAMETER_PATHS)
        or not _strict_integer(canonical.get("parameter_count"))
        or not _sha256_complete(canonical.get("parameter_sha256"))
        or not isinstance(arm_refs, Mapping)
        or any(
            not isinstance(arm_refs.get(arm), Mapping)
            or arm_refs[arm].get("tree") != "canonical_full"
            or arm_refs[arm].get("parameter_sha256")
            != canonical["parameter_sha256"]
            for arm in ("full", "query_only", "terminal_only")
        )
    ):
        raise ValueError("Gate C oracle initialization identity differs")

    targets = np.zeros((events.shape[0],), dtype=np.int32)
    targets[10:] = np.asarray(validation.targets_by_depth[:, index], dtype=np.int32)
    advances = np.asarray(validation.advance_masks[:, index], dtype=np.float32)
    full_weights = _loss_weights(
        "gate_b",
        "full",
        efforts=np.asarray([contract["effort"]], dtype=np.int32),
    )[0]
    terminal_weights = _loss_weights(
        "gate_b",
        "terminal_only",
        efforts=np.asarray([contract["effort"]], dtype=np.int32),
    )[0]

    class _OracleObjective(LatentWorkspaceModel):
        def update(self, packed: jax.Array) -> jax.Array:
            expected_width = self.config.input_width + 3
            if packed.ndim != 2 or packed.shape[-1] != expected_width:
                raise ValueError(
                    "Gate C oracle input must contain event, advance, target, and weight"
                )
            event = packed[:, : self.config.input_width]
            advance = packed[:, self.config.input_width] > 0.5
            target = packed[:, self.config.input_width + 1].astype(jnp.int32)
            weight = packed[:, self.config.input_width + 2]
            loss = legacy._classification_loss(
                super().update(event, advance),
                target,
                self.config.color_rank,
            )
            weighted = jnp.sqrt(weight) * jnp.sqrt(jnp.maximum(loss, 0.0))
            return jnp.where(weight == 0.0, jnp.zeros_like(weight), weighted)

    expected_sha256 = canonical["parameter_sha256"]
    expected_count = canonical["parameter_count"]

    def model_factory(arm: str) -> Any:
        def factory() -> _OracleObjective:
            model_config = _model_config_for_arm(
                config,
                "gate_b",
                arm,
                batch_size=1,
            )
            policy = "query_only" if arm == "query_only" else "full"
            model = _OracleObjective(model_config, memory_read_policy=policy)
            values = legacy._parameter_values(model)
            count = sum(
                np.asarray(u.get_mantissa(leaf)).size
                for value in values.values()
                for leaf in jax.tree.leaves(value)
            )
            if (
                tuple(sorted(values)) != FULL_PARAMETER_PATHS
                or count != expected_count
                or legacy._array_digest(values) != expected_sha256
            ):
                raise ValueError("Gate C oracle did not reproduce initialization")
            return model

        return factory

    def packed_inputs(weights: np.ndarray) -> jax.Array:
        return jnp.asarray(
            np.concatenate(
                (
                    events[:, None, :],
                    advances[:, None, None],
                    targets[:, None, None].astype(np.float32),
                    np.asarray(weights, dtype=np.float32)[:, None, None],
                ),
                axis=-1,
            ),
            dtype=jnp.float32,
        )

    def algorithm_factory(model: brainstate.nn.Module) -> braintrace.ETraceAlgorithm:
        return braintrace.pp_prop(
            model,
            decay_or_rank=model.config.trace_decay,
            vjp_method="multi-step",
        )

    reference_models = {
        arm: model_factory(arm)()
        for arm in ("full", "query_only", "terminal_only")
    }
    for model in reference_models.values():
        brainstate.nn.init_all_states(model, batch_size=1)
    reference_parameters = {
        arm: legacy._parameter_values(model)
        for arm, model in reference_models.items()
    }

    def snapshots_equal(left: Any, right: Any) -> bool:
        if (
            left.batch_size != right.batch_size
            or left.neuron_count != right.neuron_count
            or tuple(path for path, _ in left.entries)
            != tuple(path for path, _ in right.entries)
        ):
            return False
        for (_, left_value), (_, right_value) in zip(
            left.entries, right.entries, strict=True
        ):
            if jax.tree.structure(left_value) != jax.tree.structure(right_value):
                return False
            left_leaves = jax.tree.leaves(left_value)
            right_leaves = jax.tree.leaves(right_value)
            for left_leaf, right_leaf in zip(
                left_leaves, right_leaves, strict=True
            ):
                left_array = np.ascontiguousarray(
                    np.asarray(u.get_mantissa(left_leaf))
                )
                right_array = np.ascontiguousarray(
                    np.asarray(u.get_mantissa(right_leaf))
                )
                if (
                    left_array.shape != right_array.shape
                    or left_array.dtype != right_array.dtype
                    or left_array.tobytes() != right_array.tobytes()
                ):
                    return False
        return True

    full_snapshot = reference_models["full"].snapshot_state()
    if any(
        not snapshots_equal(full_snapshot, reference_models[arm].snapshot_state())
        for arm in ("query_only", "terminal_only")
    ):
        raise ValueError("Gate C oracle hidden-state snapshots differ")

    raw_gradients = {
        "full": chunked_online_param_gradients(
            model_factory("full"),
            packed_inputs(full_weights),
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        ),
        "query_only": chunked_online_param_gradients(
            model_factory("query_only"),
            packed_inputs(full_weights),
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        ),
        "terminal_only": chunked_online_param_gradients(
            model_factory("terminal_only"),
            packed_inputs(terminal_weights),
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        ),
    }

    gradients: dict[str, dict[str, Any]] = {}
    for arm, raw in raw_gradients.items():
        if not isinstance(raw, Mapping):
            raise TypeError("Gate C oracle gradients must be a path mapping")
        normalized = {
            key if isinstance(key, str) else gate_a._path(key): value
            for key, value in raw.items()
        }
        if (
            len(normalized) != len(raw)
            or tuple(sorted(normalized)) != FULL_PARAMETER_PATHS
        ):
            raise ValueError("Gate C oracle gradient paths differ")
        for path in FULL_PARAMETER_PATHS:
            gradient = normalized[path]
            parameter = reference_parameters[arm][path]
            if jax.tree.structure(gradient) != jax.tree.structure(parameter):
                raise ValueError("Gate C oracle gradient tree differs from parameter")
            gradient_leaves = _numeric_gradient_leaves(gradient)
            parameter_leaves = _numeric_gradient_leaves(parameter)
            if any(
                gradient_leaf.shape != parameter_leaf.shape
                or gradient_leaf.dtype != parameter_leaf.dtype
                for gradient_leaf, parameter_leaf in zip(
                    gradient_leaves, parameter_leaves, strict=True
                )
            ):
                raise ValueError("Gate C oracle gradient geometry differs")
        gradients[arm] = normalized

    def numeric_record(full: Any, arm: Any, *, path: str | None) -> dict[str, Any]:
        record = _gradient_comparison(full, arm)
        if path is None:
            record["full_sha256"] = _gradient_global_sha256(full)
            record["arm_sha256"] = _gradient_global_sha256(arm)
        else:
            record["full_sha256"] = _gradient_path_sha256(path, full)
            record["arm_sha256"] = _gradient_path_sha256(path, arm)
        return record

    def threshold_passed(record: Mapping[str, Any]) -> bool:
        return bool(
            record["full_norm"] > 0.0
            and record["relative_deviation_defined"] is True
            and record["relative_deviation"] >= 1e-3
            and record["l2_difference"]
            > max(1e-8, 1e-4 * record["full_norm"])
        )

    comparisons: dict[str, Any] = {}
    for arm in ("query_only", "terminal_only"):
        global_record = numeric_record(
            gradients["full"], gradients[arm], path=None
        )
        path_records = {
            path: numeric_record(
                gradients["full"][path],
                gradients[arm][path],
                path=path,
            )
            for path in FULL_PARAMETER_PATHS
        }
        required_paths = (
            [
                "memory_read_projection/weight",
                "workspace_query_projection/weight",
            ]
            if arm == "query_only"
            else []
        )
        required_paths_passed = all(
            threshold_passed(path_records[path]) for path in required_paths
        )
        global_passed = bool(
            global_record["arm_norm"] > 0.0
            and threshold_passed(global_record)
        )
        comparisons[arm] = {
            "global": global_record,
            "paths": path_records,
            "required_paths": required_paths,
            "required_paths_passed": bool(required_paths_passed),
            "passed": bool(global_passed and required_paths_passed),
        }
    return {
        "contract": contract,
        "objective": {
            "wrapper": "sqrt_weight_times_sqrt_cross_entropy",
            "unsupervised_output_exact_zero": True,
        },
        "gradient_chunk_size": config.gradient_chunk_size,
        "comparisons": comparisons,
        "complete": all(
            comparison["passed"] is True for comparison in comparisons.values()
        ),
    }


def run_gate_c(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all ten isolated formal Gate C pp-prop trainings.

    Parameters
    ----------
    config
        Fixed paired Gate C configuration.
    prerequisites
        Exact Gate A, Gate B, and authenticated Gate C initialization evidence.
    source_start, source_end_reporter, source_files, environment
        Live provenance and exact GPU evidence.

    Returns
    -------
    dict
        Strict formal Gate C artifact payload.
    """

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }:
        raise ValueError("formal Gate C requires exact authenticated prerequisites")
    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    start = time.perf_counter()
    initialization = _validated_gate_c_initialization_admission(
        prerequisites["gate_c_initialization"],
        config,
        source_start=source_start,
        environment=environment,
        source_files=source_files,
        require_pass=True,
    )
    normalized_prerequisites = _normalized_prerequisites(
        {"gate_a": prerequisites["gate_a"], "gate_b": prerequisites["gate_b"]}
    )
    gate_a_data = _regenerate_gate_a_data(config)
    gate_b_data = _regenerate_gate_b_data(config)
    schedule_reports = _actual_schedule_identity_report(
        config,
        gate_a_data,
        gate_b_data,
    )
    if (
        config.qualification_regime == "preregistered_full"
        and not gate_a._json_exact(schedule_reports, _schedule_identity_report(config))
    ):
        raise RuntimeError("generated Gate C schedules differ from preregistration")

    initialization_reports: dict[str, dict[str, dict[str, Any]]] = {
        regime: {} for regime in REGIME_ORDER
    }
    for regime in REGIME_ORDER:
        for arm in ARM_ORDER:
            model, trainer, initialization_report = _fresh_formal_arm(
                config,
                initialization,
                regime,
                arm,
            )
            initialization_reports[regime][arm] = initialization_report
            del model, trainer
            gc.collect()

    paired_h0_reports = {
        "gate_a": _paired_h0_identity_report(
            config,
            initialization=initialization,
            regime="gate_a",
            data=gate_a_data,
        ),
        "gate_b": _paired_h0_identity_report(
            config,
            initialization=initialization,
            regime="gate_b",
            data=gate_b_data,
        ),
    }
    gc.collect()

    arms: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in ARM_ORDER
    }
    for regime in REGIME_ORDER:
        data = gate_a_data if regime == "gate_a" else gate_b_data
        for arm in ARM_ORDER:
            execution_index = (
                REGIME_ORDER.index(regime) * len(ARM_ORDER)
                + ARM_ORDER.index(arm)
            )
            model, trainer, reproduced = _fresh_formal_arm(
                config,
                initialization,
                regime,
                arm,
            )
            audit = initialization_reports[regime][arm]
            if not gate_a._json_exact(reproduced, audit):
                raise RuntimeError(
                    "formal arm initialization changed between audit and training"
                )
            arms[arm][regime] = _run_gate_c_arm(
                model,
                trainer,
                data,
                config,
                regime,
                arm,
                initialization_report=audit,
                execution_index=execution_index,
                data_identity=schedule_reports[regime],
            )
            del model, trainer
            gc.collect()
    metrics: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        gate_a_metrics = arms[arm]["gate_a"].get("metrics")
        gate_b_metrics = arms[arm]["gate_b"].get("metrics")
        if (
            isinstance(gate_a_metrics, Mapping)
            and set(gate_a_metrics) == {"binding_gap", "depth_accuracy"}
            and gate_a._json_exact(gate_a_metrics, gate_b_metrics)
        ):
            metrics[arm] = {
                name: _finite_real(gate_a_metrics[name], f"{arm} {name}")
                for name in ("binding_gap", "depth_accuracy")
            }
        else:
            metrics[arm] = _metric_summary(
                arms[arm]["gate_a"]["evaluation"],
                arms[arm]["gate_b"]["evaluation"],
            )
    for arm in ARM_ORDER:
        for regime in REGIME_ORDER:
            arms[arm][regime]["metrics"] = dict(metrics[arm])
    margins = _blocking_margin_report(metrics)
    mechanism_oracle = _mechanism_oracle(
        config,
        initialization=initialization,
        gate_b_data=gate_b_data,
    )
    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    regimes = {
        regime: {
            "spec": dataclasses.asdict(REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
            "schedule": dict(schedule_reports[regime]),
            "metrics": {
                arm: dict(metrics[arm]) for arm in ARM_ORDER
            },
            "margins": dict(margins),
            "paired_h0_identity": paired_h0_reports[regime],
        }
        for regime in REGIME_ORDER
    }
    report: dict[str, Any] = {
        "schema_version": GATE_C_SCHEMA_VERSION,
        "control": GATE_C_CONTROL,
        "qualification_regime": config.qualification_regime,
        "learner": "pp_prop_only",
        "prerequisites": {
            **normalized_prerequisites,
            "gate_c_initialization": dict(prerequisites["gate_c_initialization"]),
        },
        "regimes": regimes,
        "arms": arms,
        "mechanism_oracle": mechanism_oracle,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": dict(environment),
        "total_wall_seconds": time.perf_counter() - start,
    }
    report["qualification"] = _qualification_report(report, config=config)
    return report


_ACCURACY_KEYS = {
    "correct",
    "count",
    "accuracy",
    "wilson_95_lower",
    "wilson_95_upper",
    "prediction_histogram",
    "prediction_sha256",
    "checkpoint",
}
_TELEMETRY_CATEGORIES = {
    "logits",
    "model_states",
    "gradients",
    "pp_prop_traces",
    "adam",
    "parameters",
}


def _accuracy_record_complete(
    metric: Any,
    *,
    count: int,
    checkpoint: int,
) -> bool:
    if not isinstance(metric, Mapping) or set(metric) != _ACCURACY_KEYS:
        return False
    return bool(
        _strict_integer(metric["checkpoint"])
        and int(metric["checkpoint"]) == checkpoint
        and _sha256_complete(metric["prediction_sha256"])
        and gate_a._accuracy_evidence_complete(metric, count)
    )


def _paired_diagnostic_record_complete(value: Any, *, count: int) -> bool:
    expected = {
        "applicable_count",
        "different_count",
        "every_pair_differs",
        "mean_l2_difference",
        "left_sha256",
        "right_sha256",
        "no_context_l2_norm",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    applicable = value["applicable_count"]
    different = value["different_count"]
    if (
        not _strict_integer(applicable)
        or int(applicable) != count
        or not _strict_integer(different)
        or not 0 <= int(different) <= count
    ):
        return False
    mean_difference = _finite_real(
        value["mean_l2_difference"], "mean diagnostic difference"
    )
    no_context_norm = _finite_real(
        value["no_context_l2_norm"], "no-context diagnostic norm"
    )
    return bool(
        value["every_pair_differs"] is (int(different) == count)
        and mean_difference >= 0.0
        and no_context_norm >= 0.0
        and _sha256_complete(value["left_sha256"])
        and _sha256_complete(value["right_sha256"])
        and (
            (
                int(different) == 0
                and mean_difference == 0.0
                and value["left_sha256"] == value["right_sha256"]
            )
            or (
                int(different) > 0
                and mean_difference > 0.0
                and value["left_sha256"] != value["right_sha256"]
            )
        )
    )


def _binding_diagnostic_complete(value: Any, *, count: int, depths: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "all_state_tensors_finite",
        "memory",
        "read_by_depth",
        "workspace_by_depth",
    }:
        return False
    memory = value["memory"]
    memory_keys = {
        "applicable_count",
        "different_count",
        "every_pair_differs",
        "mean_l2_difference",
        "left_sha256",
        "right_sha256",
        "intact_shuffled_different_count",
        "every_intact_shuffled_pair_differs",
        "no_context_exact_zero",
        "no_context_sha256",
        "intact_l2_norm",
        "shuffled_l2_norm",
        "no_context_l2_norm",
        "storage_contract",
    }
    if not isinstance(memory, Mapping) or set(memory) != memory_keys:
        return False
    memory_pair = {
        key: memory[key]
        for key in (
            "applicable_count",
            "different_count",
            "every_pair_differs",
            "mean_l2_difference",
            "left_sha256",
            "right_sha256",
            "no_context_l2_norm",
        )
    }
    if not _paired_diagnostic_record_complete(memory_pair, count=count):
        return False
    expected_depths = {str(index) for index in range(depths)}
    read_by_depth = value["read_by_depth"]
    workspace_by_depth = value["workspace_by_depth"]
    if (
        value["all_state_tensors_finite"] is not True
        or not isinstance(read_by_depth, Mapping)
        or not isinstance(workspace_by_depth, Mapping)
        or set(read_by_depth) != expected_depths
        or set(workspace_by_depth) != expected_depths
        or not all(
            _paired_diagnostic_record_complete(records[depth], count=count)
            for records in (read_by_depth, workspace_by_depth)
            for depth in expected_depths
        )
    ):
        return False
    different = int(memory["different_count"])
    return bool(
        _strict_integer(memory["intact_shuffled_different_count"])
        and int(memory["intact_shuffled_different_count"]) == different
        and memory["every_intact_shuffled_pair_differs"]
        is memory["every_pair_differs"]
        and memory["no_context_exact_zero"] is True
        and _finite_real(
            memory["no_context_l2_norm"], "no-context memory norm"
        )
        == 0.0
        and _sha256_complete(memory["no_context_sha256"])
        and all(
            _finite_real(memory[name], name) >= 0.0
            for name in (
                "intact_l2_norm",
                "shuffled_l2_norm",
                "no_context_l2_norm",
            )
        )
        and memory["storage_contract"]
        == "one final S_K snapshot per arm; S_K is not stacked"
    )


def _gate_a_evaluation_complete(
    evaluation: Any,
    config: GateCConfig,
) -> bool:
    expected_keys = {
        "finite",
        "all_compact_logits_finite",
        "all_state_tensors_finite",
        "depths",
        "intact",
        "shuffled",
        "no_context",
        "intact_minus_shuffled",
        "binding_state",
        "binding_diagnostic",
    }
    if not isinstance(evaluation, Mapping) or set(evaluation) != expected_keys:
        return False
    count = config.gate_a_config.validation_episodes
    depth_count = config.gate_a_config.gap_steps + 1
    depths = evaluation["depths"]
    if (
        evaluation["finite"] is not True
        or evaluation["all_compact_logits_finite"] is not True
        or evaluation["all_state_tensors_finite"] is not True
        or not isinstance(depths, Mapping)
        or set(depths) != {str(index) for index in range(depth_count)}
    ):
        return False
    for depth in range(depth_count):
        streams = depths[str(depth)]
        if not isinstance(streams, Mapping) or set(streams) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        if not all(
            _accuracy_record_complete(
                streams[name],
                count=count,
                checkpoint=depth,
            )
            for name in ("intact", "shuffled", "no_context")
        ):
            return False
    final = depths[str(config.gate_a_config.gap_steps)]
    if not all(
        gate_a._json_exact(evaluation[name], final[name])
        for name in ("intact", "shuffled", "no_context")
    ):
        return False
    gap = _finite_real(evaluation["intact_minus_shuffled"], "Gate A gap")
    if not math.isclose(
        gap,
        float(final["intact"]["accuracy"])
        - float(final["shuffled"]["accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    diagnostic = evaluation["binding_diagnostic"]
    if not _binding_diagnostic_complete(
        diagnostic,
        count=count,
        depths=depth_count,
    ):
        return False
    memory = diagnostic["memory"]
    binding_state = evaluation["binding_state"]
    expected_binding_state = {
        "applicable_count": memory["applicable_count"],
        "intact_shuffled_different_count": memory[
            "intact_shuffled_different_count"
        ],
        "every_intact_shuffled_pair_differs": memory[
            "every_intact_shuffled_pair_differs"
        ],
        "no_context_exact_zero": memory["no_context_exact_zero"],
        "intact_sha256": memory["left_sha256"],
        "shuffled_sha256": memory["right_sha256"],
        "no_context_sha256": memory["no_context_sha256"],
        "all_finite": diagnostic["all_state_tensors_finite"],
    }
    return gate_a._json_exact(binding_state, expected_binding_state)


def _gate_a_intervention_diagnostic_complete(
    evaluation: Mapping[str, Any],
    *,
    arm: str,
) -> bool:
    diagnostic = evaluation["binding_diagnostic"]

    def exact_zero_pair(value: Mapping[str, Any]) -> bool:
        return bool(
            _strict_integer(value["different_count"])
            and int(value["different_count"]) == 0
            and value["every_pair_differs"] is False
            and _finite_real(value["mean_l2_difference"], "diagnostic difference")
            == 0.0
            and value["left_sha256"] == value["right_sha256"]
            and _finite_real(
                value["no_context_l2_norm"], "no-context diagnostic norm"
            )
            == 0.0
        )

    if arm == "query_only":
        return exact_zero_pair(diagnostic["read_by_depth"]["1"])
    if arm != "legacy":
        return True
    memory = diagnostic["memory"]
    if not (
        exact_zero_pair(memory)
        and _finite_real(memory["intact_l2_norm"], "legacy intact memory norm")
        == 0.0
        and _finite_real(memory["shuffled_l2_norm"], "legacy shuffled memory norm")
        == 0.0
    ):
        return False
    return all(
        exact_zero_pair(value)
        for section in ("read_by_depth", "workspace_by_depth")
        for value in diagnostic[section].values()
    )


def _gate_b_evaluation_complete(
    evaluation: Any,
    config: GateCConfig,
    *,
    require_no_collapse: bool,
) -> bool:
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "finite",
        "h0_proper",
        "depths",
        "efforts",
    }:
        return False
    count = config.gate_b_config.validation_episodes
    depths = evaluation["depths"]
    expected_depths = {
        str(index) for index in range(config.gate_b_config.gap_steps + 1)
    }
    if (
        evaluation["finite"] is not True
        or not isinstance(depths, Mapping)
        or set(depths) != expected_depths
    ):
        return False
    for depth in range(config.gate_b_config.gap_steps + 1):
        streams = depths[str(depth)]
        if not isinstance(streams, Mapping) or set(streams) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        for name in ("intact", "shuffled", "no_context"):
            metric = streams[name]
            if not _accuracy_record_complete(
                metric,
                count=count,
                checkpoint=depth,
            ):
                return False
            if require_no_collapse and max(map(int, metric["prediction_histogram"])) >= count:
                return False
    if not gate_a._json_exact(evaluation["h0_proper"], depths["0"]["intact"]):
        return False
    efforts = evaluation["efforts"]
    if not isinstance(efforts, Mapping) or set(efforts) != {
        str(effort) for effort in gate_b.QUALIFYING_EFFORTS
    }:
        return False
    h0 = evaluation["h0_proper"]
    for effort in gate_b.QUALIFYING_EFFORTS:
        evidence = efforts[str(effort)]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "intact",
            "shuffled",
            "no_context",
            "h0_final_target",
            "intact_minus_h0",
            "intact_minus_shuffled",
        }:
            return False
        matching = depths[str(effort)]
        if not all(
            gate_a._json_exact(evidence[name], matching[name])
            for name in ("intact", "shuffled", "no_context")
        ):
            return False
        h0_final = evidence["h0_final_target"]
        if (
            not _accuracy_record_complete(h0_final, count=count, checkpoint=0)
            or h0_final["prediction_sha256"] != h0["prediction_sha256"]
            or h0_final["prediction_histogram"] != h0["prediction_histogram"]
        ):
            return False
        expected_h0_gap = (
            float(evidence["intact"]["accuracy"])
            - float(h0_final["accuracy"])
        )
        expected_shuffled_gap = (
            float(evidence["intact"]["accuracy"])
            - float(evidence["shuffled"]["accuracy"])
        )
        if not (
            math.isclose(
                _finite_real(evidence["intact_minus_h0"], "Gate B H0 gap"),
                expected_h0_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _finite_real(
                    evidence["intact_minus_shuffled"], "Gate B shuffled gap"
                ),
                expected_shuffled_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def _formal_initialization_complete(
    arm_report: Any,
    admission: Mapping[str, Any],
    *,
    regime: str,
    arm: str,
) -> bool:
    if not isinstance(arm_report, Mapping):
        return False
    initialization = arm_report.get("initialization")
    if not isinstance(initialization, Mapping) or set(initialization) != {
        "tree",
        "parameter_sha256",
        "parameter_count",
        "parameter_paths",
        "shared_paths",
    }:
        return False
    regime_admission = admission["initialization"][regime]
    reference = regime_admission["arm_initialization_refs"][arm]
    topology = regime_admission[reference["tree"]]
    expected_paths = (
        SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
    )
    return bool(
        initialization["tree"] == reference["tree"]
        and initialization["parameter_sha256"] == reference["parameter_sha256"]
        and _strict_integer(initialization["parameter_count"])
        and int(initialization["parameter_count"]) == topology["parameter_count"]
        and initialization["parameter_paths"] == list(expected_paths)
        and gate_a._json_exact(
            initialization["shared_paths"],
            regime_admission["shared_paths"],
        )
    )


def _formal_optimizer_complete(
    arm_report: Any,
    admission: Mapping[str, Any],
    *,
    regime: str,
    arm: str,
    updates: int,
) -> bool:
    optimizer = arm_report["optimizer"]
    expected = admission["initialization"][regime]["optimizer_paths"][arm]
    training = arm_report["training"]
    return bool(
        gate_a._json_exact(optimizer, expected)
        and optimizer["fresh_state_finite"] is True
        and optimizer["fresh_state_all_zero"] is True
        and _strict_integer(optimizer["executed_updates"])
        and int(optimizer["executed_updates"]) == 0
        and _strict_integer(training["optimizer_final_step"])
        and int(training["optimizer_final_step"]) == updates
    )


def _expected_parameter_counts(regime: str, arm: str) -> dict[str, int]:
    counts = {
        "color_factor_head/weight": 144_480,
        "ff_syn/comm/weight": 83_968 if regime == "gate_a" else 96_256,
        "height_head/weight": 3_870,
        "memory_read_projection/weight": 65_536,
        "memory_write_scale": 1_024,
        "readout_projection/weight": 262_272,
        "rec_syn/comm/weight": 16_384,
        "width_head/weight": 3_870,
        "workspace_query_projection/weight": 65_536,
    }
    paths = SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
    return {path: counts[path] for path in paths}


def _parameter_movement_complete(
    movement: Any,
    *,
    regime: str,
    arm: str,
) -> bool:
    if not isinstance(movement, Mapping) or set(movement) != {
        "l2_delta",
        "parameter_count",
        "paths",
    }:
        return False
    expected_counts = _expected_parameter_counts(regime, arm)
    paths = movement["paths"]
    if not isinstance(paths, Mapping) or set(paths) != set(expected_counts):
        return False
    squared = 0.0
    count = 0
    for path, expected_count in expected_counts.items():
        value = paths[path]
        if not isinstance(value, Mapping) or set(value) != {
            "l2_delta",
            "parameter_count",
        }:
            return False
        delta = _finite_real(value["l2_delta"], f"{path} movement")
        if (
            delta < 0.0
            or not _strict_integer(value["parameter_count"])
            or int(value["parameter_count"]) != expected_count
        ):
            return False
        expected_zero = path in {
            "height_head/weight",
            "width_head/weight",
        } or (arm == "frozen_write" and path == "memory_write_scale") or (
            arm == "query_only"
            and path == "workspace_query_projection/weight"
        )
        if expected_zero:
            if delta != 0.0:
                return False
        squared += delta * delta
        count += expected_count
    total = _finite_real(movement["l2_delta"], "total parameter movement")
    return bool(
        total > 0.0
        and math.isclose(total, math.sqrt(squared), rel_tol=1e-12, abs_tol=1e-12)
        and _strict_integer(movement["parameter_count"])
        and int(movement["parameter_count"]) == count
    )


def _expected_loss_weight_report(
    config: GateCConfig,
    *,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    if regime == "gate_a":
        efforts = np.ones((1,), dtype=np.int32)
    else:
        efforts = np.resize(
            np.asarray(gate_b.QUALIFYING_EFFORTS, dtype=np.int32),
            config.gate_b_config.training_updates,
        )
    weights = _loss_weights(regime, arm, efforts=efforts)
    return {
        "dtype": weights.dtype.str,
        "shape": list(weights.shape),
        "sha256": legacy._digest_arrays(weights),
    }


def _training_report_complete(
    arm_report: Any,
    admission: Mapping[str, Any],
    config: GateCConfig,
    *,
    regime: str,
    arm: str,
) -> bool:
    expected_arm_keys = {
        "initialization",
        "optimizer",
        "compiler",
        "training",
        "parameter_movement",
        "evaluation",
        "metrics",
    }
    if not isinstance(arm_report, Mapping) or set(arm_report) != expected_arm_keys:
        return False
    training = arm_report["training"]
    expected_training_keys = {
        "algorithm",
        "execution_index",
        "intervention",
        "data_identity",
        "executed_updates",
        "batch_size",
        "chunk_count",
        "cold_compile_and_train_seconds",
        "initial_parameter_sha256",
        "final_parameter_sha256",
        "optimizer_final_step",
        "loss_weights",
        "compile_warnings",
        "losses",
        "initial_loss",
        "final_loss",
        "tail_64_mean_loss",
        "finite",
        "max_abs",
        "value_count",
        "frozen_write",
    }
    if not isinstance(training, Mapping) or set(training) != expected_training_keys:
        return False
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    updates = regime_config.training_updates
    losses = training["losses"]
    if not isinstance(losses, list) or len(losses) != updates:
        return False
    try:
        loss_values = np.asarray(
            [_finite_real(value, "training loss") for value in losses],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return False
    if not np.isfinite(loss_values).all():
        return False
    finite = training["finite"]
    maxima = training["max_abs"]
    value_counts = training["value_count"]
    if not all(
        isinstance(section, Mapping) and set(section) == _TELEMETRY_CATEGORIES
        for section in (finite, maxima, value_counts)
    ):
        return False
    if not all(finite[name] is True for name in _TELEMETRY_CATEGORIES):
        return False
    if not all(
        _finite_real(maxima[name], f"{name} maximum") >= 0.0
        for name in _TELEMETRY_CATEGORIES
    ):
        return False
    if not all(
        _strict_integer(value_counts[name]) and int(value_counts[name]) > 0
        for name in _TELEMETRY_CATEGORIES
    ):
        return False
    reference = admission["initialization"][regime]["arm_initialization_refs"][arm]
    chunk_count = 1 if regime == "gate_a" else config.gate_b_config.staging_chunk_count
    tail = float(loss_values[-min(64, updates) :].mean())
    if not (
        training["algorithm"] == "production_pp_prop"
        and _strict_integer(training["executed_updates"])
        and int(training["executed_updates"]) == updates
        and _strict_integer(training["batch_size"])
        and int(training["batch_size"]) == regime_config.batch_size
        and _strict_integer(training["chunk_count"])
        and int(training["chunk_count"]) == chunk_count
        and _finite_real(
            training["cold_compile_and_train_seconds"], "training time"
        )
        >= 0.0
        and training["initial_parameter_sha256"] == reference["parameter_sha256"]
        and _sha256_complete(training["final_parameter_sha256"])
        and training["final_parameter_sha256"]
        != training["initial_parameter_sha256"]
        and gate_a._json_exact(
            training["loss_weights"],
            _expected_loss_weight_report(config, regime=regime, arm=arm),
        )
        and isinstance(training["compile_warnings"], list)
        and all(isinstance(item, str) for item in training["compile_warnings"])
        and math.isclose(
            _finite_real(training["initial_loss"], "initial loss"),
            float(loss_values[0]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            _finite_real(training["final_loss"], "final loss"),
            float(loss_values[-1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            _finite_real(training["tail_64_mean_loss"], "tail loss"),
            tail,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False
    frozen = training["frozen_write"]
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "applicable",
        "all_ones_before",
        "all_ones_after",
        "excluded_from_optimizer",
    }:
        return False
    if not all(isinstance(frozen[name], bool) for name in frozen):
        return False
    if frozen["applicable"] is not (arm == "frozen_write"):
        return False
    if frozen["excluded_from_optimizer"] is not (arm == "frozen_write"):
        return False
    if arm == "legacy":
        if frozen["all_ones_before"] is not False or frozen["all_ones_after"] is not False:
            return False
    elif frozen["all_ones_before"] is not True:
        return False
    if arm == "frozen_write" and frozen["all_ones_after"] is not True:
        return False
    compiler = arm_report["compiler"]
    reference_tree = reference["tree"]
    expected_compiler = admission["initialization"][regime][reference_tree]["compiler"]
    return bool(
        gate_a._json_exact(compiler, expected_compiler)
        and (
            _legacy_compiler_complete(compiler)
            if arm == "legacy"
            else _full_compiler_complete(compiler)
        )
        and _parameter_movement_complete(
            arm_report["parameter_movement"],
            regime=regime,
            arm=arm,
        )
    )


def _exact_formal_configuration_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    if config != GateCConfig() or config.qualification_regime != "preregistered_full":
        return False
    regimes = report["regimes"]
    arms = report["arms"]
    if (
        report["qualification_regime"] != "preregistered_full"
        or not isinstance(regimes, Mapping)
        or set(regimes) != set(REGIME_ORDER)
        or not isinstance(arms, Mapping)
        or set(arms) != set(ARM_ORDER)
    ):
        return False
    for regime in REGIME_ORDER:
        value = regimes[regime]
        if not isinstance(value, Mapping) or set(value) != {
            "spec",
            "config",
            "schedule",
            "metrics",
            "margins",
            "paired_h0_identity",
        }:
            return False
        regime_config = (
            config.gate_a_config if regime == "gate_a" else config.gate_b_config
        )
        if not (
            gate_a._json_exact(value["spec"], dataclasses.asdict(REGIME_SPECS[regime]))
            and gate_a._json_exact(value["config"], dataclasses.asdict(regime_config))
        ):
            return False
    for regime_index, regime in enumerate(REGIME_ORDER):
        for arm_index, arm in enumerate(ARM_ORDER):
            if not isinstance(arms[arm], Mapping) or set(arms[arm]) != set(
                REGIME_ORDER
            ):
                return False
            arm_report = arms[arm][regime]
            if not isinstance(arm_report, Mapping) or set(arm_report) != {
                "initialization",
                "optimizer",
                "compiler",
                "training",
                "parameter_movement",
                "evaluation",
                "metrics",
            }:
                return False
            training = arm_report["training"]
            expected_index = regime_index * len(ARM_ORDER) + arm_index
            if not (
                isinstance(training, Mapping)
                and _strict_integer(training.get("execution_index"))
                and int(training["execution_index"]) == expected_index
                and gate_a._json_exact(
                    training.get("intervention"),
                    dataclasses.asdict(ARM_SPECS[arm]),
                )
            ):
                return False
    return True


def _canonical_schedules_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    expected = _schedule_identity_report(config)
    for regime in REGIME_ORDER:
        if not gate_a._json_exact(report["regimes"][regime]["schedule"], expected[regime]):
            return False
        for arm in ARM_ORDER:
            training = report["arms"][arm][regime]["training"]
            if not (
                gate_a._json_exact(training["data_identity"], expected[regime])
                and gate_a._json_exact(
                    training["loss_weights"],
                    _expected_loss_weight_report(config, regime=regime, arm=arm),
                )
            ):
                return False
    return True


def _paired_h0_identity_complete(
    value: Any,
    admission: Mapping[str, Any],
    *,
    regime: str,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "checkpoint",
        "initialization_parameter_sha256",
        "streams",
        "passed",
    }:
        return False
    canonical_sha = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]
    if not (
        _strict_integer(value["checkpoint"])
        and int(value["checkpoint"]) == 0
        and gate_a._json_exact(
            value["initialization_parameter_sha256"],
            {"full": canonical_sha, "query_only": canonical_sha},
        )
        and isinstance(value["streams"], Mapping)
        and set(value["streams"]) == {"intact", "shuffled", "no_context"}
    ):
        return False
    for stream in value["streams"].values():
        if not isinstance(stream, Mapping) or set(stream) != {
            "full_compact_sha256",
            "query_only_compact_sha256",
            "full_state_sha256",
            "query_only_state_sha256",
            "compact_byte_identical",
            "state_byte_identical",
        }:
            return False
        if not (
            all(
                _sha256_complete(stream[name])
                for name in (
                    "full_compact_sha256",
                    "query_only_compact_sha256",
                    "full_state_sha256",
                    "query_only_state_sha256",
                )
            )
            and stream["full_compact_sha256"]
            == stream["query_only_compact_sha256"]
            and stream["full_state_sha256"] == stream["query_only_state_sha256"]
            and stream["compact_byte_identical"] is True
            and stream["state_byte_identical"] is True
        ):
            return False
    return value["passed"] is True


def _full_gate_a_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    arm_report = report["arms"]["full"]["gate_a"]
    evaluation = arm_report["evaluation"]
    if not _gate_a_evaluation_complete(evaluation, config):
        return False
    final = evaluation["depths"][str(config.gate_a_config.gap_steps)]
    intact = final["intact"]
    shuffled = final["shuffled"]
    memory = evaluation["binding_diagnostic"]["memory"]
    pairing_chance = 1.0 / legacy.SYMBOL_COUNT
    return bool(
        float(intact["accuracy"]) >= 0.80
        and float(intact["wilson_95_lower"]) > pairing_chance
        and float(evaluation["intact_minus_shuffled"]) >= 0.25
        and float(shuffled["wilson_95_lower"]) <= pairing_chance
        and gate_a._diagnostic_evidence_complete(
            evaluation["binding_diagnostic"],
            config.gate_a_config,
        )
        and _strict_integer(memory["applicable_count"])
        and int(memory["applicable_count"])
        == config.gate_a_config.validation_episodes
        and _strict_integer(memory["intact_shuffled_different_count"])
        and int(memory["intact_shuffled_different_count"])
        == int(memory["applicable_count"])
        and memory["every_intact_shuffled_pair_differs"] is True
        and memory["no_context_exact_zero"] is True
        and _full_compiler_complete(arm_report["compiler"])
        and arm_report["compiler"].get(
            "context_memory_isolated_from_workspace_lif"
        )
        is True
    )


def _full_gate_b_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    evaluation = report["arms"]["full"]["gate_b"]["evaluation"]
    if not _gate_b_evaluation_complete(
        evaluation,
        config,
        require_no_collapse=True,
    ):
        return False
    efforts = evaluation["efforts"]
    improvements = [
        float(efforts[str(effort)]["intact_minus_h0"])
        for effort in gate_b.QUALIFYING_EFFORTS
    ]
    return bool(
        all(
            float(efforts[str(effort)]["intact"]["wilson_95_lower"])
            > 1.0 / 8.0
            for effort in gate_b.QUALIFYING_EFFORTS
        )
        and sum(value >= 0.15 for value in improvements) >= 2
        and all(
            float(efforts[str(effort)]["intact_minus_shuffled"]) >= 0.15
            for effort in gate_b.QUALIFYING_EFFORTS
        )
        and all(
            float(efforts[str(effort)][stream]["wilson_95_lower"])
            <= 1.0 / 8.0
            for effort in gate_b.QUALIFYING_EFFORTS
            for stream in ("shuffled", "no_context")
        )
        and float(evaluation["h0_proper"]["wilson_95_lower"]) > 1.0 / 8.0
    )


def _behavioral_margins_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    metrics: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        gate_a_evaluation = report["arms"][arm]["gate_a"]["evaluation"]
        gate_b_evaluation = report["arms"][arm]["gate_b"]["evaluation"]
        if not (
            _gate_a_evaluation_complete(gate_a_evaluation, config)
            and _gate_b_evaluation_complete(
                gate_b_evaluation,
                config,
                require_no_collapse=arm == "full",
            )
        ):
            return False
        metrics[arm] = _metric_summary(gate_a_evaluation, gate_b_evaluation)
        for regime in REGIME_ORDER:
            if not gate_a._json_exact(
                report["arms"][arm][regime]["metrics"],
                metrics[arm],
            ):
                return False
    expected_margins = _blocking_margin_report(metrics)
    for regime in REGIME_ORDER:
        regime_report = report["regimes"][regime]
        if not (
            gate_a._json_exact(regime_report["metrics"], metrics)
            and gate_a._json_exact(regime_report["margins"], expected_margins)
        ):
            return False
    return expected_margins["blocking_passed"] is True


def _frozen_write_complete(report: Mapping[str, Any]) -> bool:
    for regime in REGIME_ORDER:
        arm_report = report["arms"]["frozen_write"][regime]
        frozen = arm_report["training"]["frozen_write"]
        movement = arm_report["parameter_movement"]["paths"][
            "memory_write_scale"
        ]
        if not (
            gate_a._json_exact(
                frozen,
                {
                    "applicable": True,
                    "all_ones_before": True,
                    "all_ones_after": True,
                    "excluded_from_optimizer": True,
                },
            )
            and _strict_integer(movement["parameter_count"])
            and int(movement["parameter_count"]) == 1_024
            and _finite_real(movement["l2_delta"], "frozen write movement") == 0.0
        ):
            return False
    metrics = {
        arm: report["regimes"]["gate_a"]["metrics"][arm]
        for arm in ARM_ORDER
    }
    expected = _blocking_margin_report(metrics)["frozen_write"]
    return all(
        gate_a._json_exact(
            report["regimes"][regime]["margins"]["frozen_write"],
            expected,
        )
        for regime in REGIME_ORDER
    )


def _source_and_gpu_formal_complete(
    report: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> bool:
    return bool(
        _source_and_gpu_complete(report)
        and _source_files_complete(report["source_files"])
        and gate_a._json_exact(report["source_files"], admission["source_files"])
        and report["source_start"]["commit"]
        == admission["source_start"]["commit"]
        and report["environment"]["image_digest"]
        == admission["environment"]["image_digest"]
    )


def _gradient_record_complete(value: Any) -> bool:
    expected = {
        "full_norm",
        "arm_norm",
        "l2_difference",
        "relative_deviation",
        "relative_deviation_defined",
        "cosine",
        "cosine_defined",
        "full_sha256",
        "arm_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    try:
        full_norm = _finite_real(value["full_norm"], "full gradient norm")
        arm_norm = _finite_real(value["arm_norm"], "arm gradient norm")
        difference = _finite_real(value["l2_difference"], "gradient difference")
    except (TypeError, ValueError):
        return False
    if min(full_norm, arm_norm, difference) < 0.0:
        return False
    if full_norm == 0.0 or arm_norm == 0.0:
        if not math.isclose(
            difference,
            max(full_norm, arm_norm),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return False
    else:
        tolerance = max(1e-12, 1e-9 * (full_norm + arm_norm))
        if not (
            abs(full_norm - arm_norm) - tolerance
            <= difference
            <= full_norm + arm_norm + tolerance
        ):
            return False
    relative_defined = full_norm > 0.0
    cosine_defined = full_norm > 0.0 and arm_norm > 0.0
    if (
        value["relative_deviation_defined"] is not relative_defined
        or value["cosine_defined"] is not cosine_defined
        or not _sha256_complete(value["full_sha256"])
        or not _sha256_complete(value["arm_sha256"])
    ):
        return False
    if relative_defined:
        try:
            relative = _finite_real(
                value["relative_deviation"], "relative gradient deviation"
            )
        except (TypeError, ValueError):
            return False
        if not math.isclose(
            relative,
            difference / full_norm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return False
    elif value["relative_deviation"] is not None:
        return False
    if cosine_defined:
        try:
            cosine = _finite_real(value["cosine"], "gradient cosine")
        except (TypeError, ValueError):
            return False
        expected_cosine = (
            full_norm * full_norm + arm_norm * arm_norm - difference * difference
        ) / (2.0 * full_norm * arm_norm)
        if not (
            -1.0 - 1e-9 <= cosine <= 1.0 + 1e-9
            and math.isclose(
                cosine,
                expected_cosine,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            return False
    elif value["cosine"] is not None:
        return False
    return True


def _gradient_digest_from_records(
    paths: Mapping[str, Mapping[str, Any]],
    *,
    side: str,
) -> str:
    fields: list[bytes] = [b"example21-gate-c-gradient-global-v1"]
    for path in sorted(paths):
        fields.extend(
            (
                path.encode("utf-8"),
                str(paths[path][f"{side}_sha256"]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _mechanism_oracle_complete(
    value: Any,
    config: GateCConfig,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "contract",
        "objective",
        "gradient_chunk_size",
        "comparisons",
        "complete",
    }:
        return False
    if not (
        gate_a._json_exact(value["contract"], _oracle_contract(config))
        and gate_a._json_exact(
            value["objective"],
            {
                "wrapper": "sqrt_weight_times_sqrt_cross_entropy",
                "unsupervised_output_exact_zero": True,
            },
        )
        and _strict_integer(value["gradient_chunk_size"])
        and int(value["gradient_chunk_size"]) == 1
    ):
        return False
    comparisons = value["comparisons"]
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "query_only",
        "terminal_only",
    }:
        return False
    expected_required = {
        "query_only": [
            "memory_read_projection/weight",
            "workspace_query_projection/weight",
        ],
        "terminal_only": [],
    }
    recomputed_pass: dict[str, bool] = {}
    full_records: dict[str, dict[str, Any]] = {}
    for arm in ("query_only", "terminal_only"):
        comparison = comparisons[arm]
        if not isinstance(comparison, Mapping) or set(comparison) != {
            "global",
            "paths",
            "required_paths",
            "required_paths_passed",
            "passed",
        }:
            return False
        paths = comparison["paths"]
        if not isinstance(paths, Mapping) or tuple(sorted(paths)) != FULL_PARAMETER_PATHS:
            return False
        if not all(_gradient_record_complete(paths[path]) for path in FULL_PARAMETER_PATHS):
            return False
        global_record = comparison["global"]
        if not _gradient_record_complete(global_record):
            return False
        expected_full_norm = math.sqrt(
            math.fsum(float(paths[path]["full_norm"]) ** 2 for path in paths)
        )
        expected_arm_norm = math.sqrt(
            math.fsum(float(paths[path]["arm_norm"]) ** 2 for path in paths)
        )
        expected_difference = math.sqrt(
            math.fsum(float(paths[path]["l2_difference"]) ** 2 for path in paths)
        )
        expected_dot = math.fsum(
            0.0
            if paths[path]["cosine"] is None
            else float(paths[path]["cosine"])
            * float(paths[path]["full_norm"])
            * float(paths[path]["arm_norm"])
            for path in paths
        )
        expected_cosine = (
            expected_dot / (expected_full_norm * expected_arm_norm)
            if expected_full_norm > 0.0 and expected_arm_norm > 0.0
            else None
        )
        if not (
            math.isclose(
                float(global_record["full_norm"]),
                expected_full_norm,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(global_record["arm_norm"]),
                expected_arm_norm,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(global_record["l2_difference"]),
                expected_difference,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and (
                expected_cosine is not None
                and global_record["cosine"] is not None
                and math.isclose(
                    float(global_record["cosine"]),
                    expected_cosine,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            )
            and global_record["full_sha256"]
            == _gradient_digest_from_records(paths, side="full")
            and global_record["arm_sha256"]
            == _gradient_digest_from_records(paths, side="arm")
        ):
            return False
        if comparison["required_paths"] != expected_required[arm]:
            return False

        def threshold(record: Mapping[str, Any]) -> bool:
            return bool(
                float(record["full_norm"]) > 0.0
                and record["relative_deviation_defined"] is True
                and float(record["relative_deviation"]) >= 1e-3
                and float(record["l2_difference"])
                > max(1e-8, 1e-4 * float(record["full_norm"]))
            )

        required_passed = all(threshold(paths[path]) for path in expected_required[arm])
        global_passed = bool(float(global_record["arm_norm"]) > 0.0 and threshold(global_record))
        passed = bool(global_passed and required_passed)
        if not (
            comparison["required_paths_passed"] is required_passed
            and comparison["passed"] is passed
        ):
            return False
        recomputed_pass[arm] = passed
        full_records[arm] = {
            path: {
                "norm": paths[path]["full_norm"],
                "sha256": paths[path]["full_sha256"],
            }
            for path in FULL_PARAMETER_PATHS
        }
    if not gate_a._json_exact(
        full_records["query_only"],
        full_records["terminal_only"],
    ):
        return False
    complete = all(recomputed_pass.values())
    return value["complete"] is complete and complete


def _qualification_report(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Recompute every formal Gate C criterion from retained raw evidence."""

    criteria = {name: False for name in QUALIFICATION_CRITERIA}
    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        passed = False
        return {
            "criteria": criteria,
            "passed": passed,
            "interpretation": "gate_c_failed_stop_no_causal_mechanism_conclusion",
        }
    base_keys = {
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "arms",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "total_wall_seconds",
    }
    try:
        keys = set(report)
        qualification_shape = True
        if "qualification" in report:
            qualification_shape = bool(
                isinstance(report["qualification"], Mapping)
                and set(report["qualification"])
                == {"criteria", "passed", "interpretation"}
            )
        criteria["schema_and_control"] = bool(
            keys in (base_keys, base_keys | {"qualification"})
            and qualification_shape
            and _strict_integer(report["schema_version"])
            and int(report["schema_version"]) == GATE_C_SCHEMA_VERSION
            and report["control"] == GATE_C_CONTROL
            and report["learner"] == "pp_prop_only"
            and _finite_real(report["total_wall_seconds"], "total wall time") >= 0.0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    try:
        criteria["exact_configuration"] = _exact_formal_configuration_complete(
            report,
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    prerequisites: Any = report.get("prerequisites")
    try:
        criteria["prerequisites_authenticated"] = bool(
            isinstance(prerequisites, Mapping)
            and set(prerequisites)
            == {"gate_a", "gate_b", "gate_c_initialization"}
            and gate_a._json_exact(
                _normalized_prerequisites(
                    {
                        "gate_a": prerequisites["gate_a"],
                        "gate_b": prerequisites["gate_b"],
                    }
                ),
                {"gate_a": _GATE_A_REFERENCE, "gate_b": _GATE_B_REFERENCE},
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    admission: Mapping[str, Any] | None = None
    try:
        admission = _validated_gate_c_initialization_admission(
            prerequisites["gate_c_initialization"],
            config,
            source_start=report["source_start"],
            environment=report["environment"],
            source_files=report["source_files"],
            require_pass=True,
        )
        criteria["initialization_authenticated"] = all(
            _formal_initialization_complete(
                report["arms"][arm][regime],
                admission,
                regime=regime,
                arm=arm,
            )
            for regime in REGIME_ORDER
            for arm in ARM_ORDER
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        admission = None

    try:
        criteria["canonical_schedules_complete"] = (
            _canonical_schedules_complete(report, config)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    if admission is not None:
        try:
            optimizer_digests: set[str] = set()
            optimizers_complete = True
            for regime in REGIME_ORDER:
                regime_config = (
                    config.gate_a_config
                    if regime == "gate_a"
                    else config.gate_b_config
                )
                for arm in ARM_ORDER:
                    arm_report = report["arms"][arm][regime]
                    optimizers_complete = bool(
                        optimizers_complete
                        and _formal_optimizer_complete(
                            arm_report,
                            admission,
                            regime=regime,
                            arm=arm,
                            updates=regime_config.training_updates,
                        )
                    )
                    optimizer_digests.add(arm_report["optimizer"]["state_sha256"])
            criteria["fresh_isolated_optimizers"] = bool(
                optimizers_complete and len(optimizer_digests) == 10
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass

        try:
            criteria["compiler_and_training_complete"] = all(
                _training_report_complete(
                    report["arms"][arm][regime],
                    admission,
                    config,
                    regime=regime,
                    arm=arm,
                )
                and (
                    _gate_a_evaluation_complete(
                        report["arms"][arm][regime]["evaluation"],
                        config,
                    )
                    and _gate_a_intervention_diagnostic_complete(
                        report["arms"][arm][regime]["evaluation"],
                        arm=arm,
                    )
                    if regime == "gate_a"
                    else _gate_b_evaluation_complete(
                        report["arms"][arm][regime]["evaluation"],
                        config,
                        require_no_collapse=arm == "full",
                    )
                )
                for regime in REGIME_ORDER
                for arm in ARM_ORDER
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass

    try:
        criteria["full_gate_a_passed"] = _full_gate_a_complete(report, config)
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    try:
        criteria["full_gate_b_passed"] = _full_gate_b_complete(report, config)
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    try:
        criteria["blocking_behavioral_margins"] = _behavioral_margins_complete(
            report,
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    if admission is not None:
        try:
            criteria["paired_h0_identity"] = all(
                _paired_h0_identity_complete(
                    report["regimes"][regime]["paired_h0_identity"],
                    admission,
                    regime=regime,
                )
                for regime in REGIME_ORDER
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    try:
        criteria["frozen_write_complete"] = _frozen_write_complete(report)
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    try:
        criteria["mechanism_oracle_complete"] = _mechanism_oracle_complete(
            report["mechanism_oracle"],
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    if admission is not None:
        try:
            criteria["source_and_gpu_authenticated"] = (
                _source_and_gpu_formal_complete(report, admission)
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass
    passed = bool(criteria and all(criteria.values()))
    return {
        "criteria": criteria,
        "passed": passed,
        "interpretation": (
            "gate_c_passed_pp_prop_learnability_mechanism"
            if passed
            else "gate_c_failed_stop_no_causal_mechanism_conclusion"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("gate_c_init", "formal_gate_c"),
        required=True,
    )
    parser.add_argument("--gate-a-result", type=Path, required=True)
    parser.add_argument("--gate-a-manifest", type=Path, required=True)
    parser.add_argument("--gate-b-manifest", type=Path, required=True)
    parser.add_argument("--gate-c-init-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a fixed authenticated Gate C target.

    Parameters
    ----------
    argv
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a complete artifact is written. Scientific failure remains
        encoded in the artifact for the authenticated launcher to sign.
    """

    from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher

    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    launch_config = launcher.LaunchConfig(
        target=args.target,
        repo_root=repo_root,
        output_dir=args.output.resolve().parent,
    )
    gate_a_paths = launcher._gate_a_artifact_paths(launch_config)
    gate_b_paths = launcher._formal_gate_b_artifact_paths(launch_config)
    if (
        args.gate_a_result.resolve() != gate_a_paths.result.resolve()
        or args.gate_a_manifest.resolve() != gate_a_paths.manifest.resolve()
    ):
        raise ValueError("Gate C target requires the fixed Gate A artifact paths")
    if args.gate_b_manifest.resolve() != gate_b_paths.manifest.resolve():
        raise ValueError("Gate C target requires the fixed Gate B manifest path")

    if args.target == "gate_c_init":
        if args.gate_c_init_manifest is not None:
            raise ValueError(
                "gate_c_init does not accept a Gate C initialization manifest"
            )
    elif args.gate_c_init_manifest is None:
        raise ValueError("formal_gate_c requires the fixed initialization manifest")

    source_start = gate_a._source_report()
    environment = gate_a._environment_report()
    gate_a._require_authenticated_gpu_launch(source_start, environment)
    head = str(source_start["commit"])
    expected_paths = launcher.target_paths(
        launch_config,
        head,
        args.target,
    )
    if args.output.resolve() != expected_paths.result.resolve():
        raise ValueError("Gate C target requires the fixed output path")

    if args.target == "formal_gate_c":
        initialization_paths = launcher.target_paths(
            launch_config,
            head,
            "gate_c_init",
        )
        if (
            args.gate_c_init_manifest is None
            or args.gate_c_init_manifest.resolve()
            != initialization_paths.manifest.resolve()
        ):
            raise ValueError(
                "formal_gate_c requires the fixed initialization manifest path"
            )
        prerequisites = launcher._load_formal_gate_c_prerequisites(
            launch_config,
            head=head,
            image_id=str(environment["image_digest"]),
        )
    else:
        prerequisites = launcher._load_gate_c_prerequisites(launch_config)
    source_files = _source_files_report()
    if args.target == "formal_gate_c":
        result = run_gate_c(
            GateCConfig(),
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=gate_a._source_report,
            source_files=source_files,
            environment=environment,
        )
    else:
        result = run_gate_c_initialization(
            GateCConfig(),
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=gate_a._source_report,
            source_files=source_files,
            environment=environment,
        )
    destination = write_artifact(result, args.output)
    print(destination)
    print(json.dumps(result["qualification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
