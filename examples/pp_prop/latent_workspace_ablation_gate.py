"""Preregistered pp-prop mechanism ablations for Example 21 Gate C."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

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
    def evaluate_stream(events: jax.Array) -> tuple[jax.Array, jax.Array]:
        model.reset_state()

        def step(inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
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
            return compact, state_finite

        compact, state_finite = brainstate.transform.for_loop(
            step, (events, advances)
        )
        return compact[checkpoint_start:], state_finite

    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
    return {
        "query_only": query,
        "terminal_only": terminal,
        "legacy": legacy,
        "frozen_write": frozen,
        "blocking_passed": bool(
            query["passed"] and terminal["passed"] and legacy["passed"]
        ),
    }


def _numeric_gradient_leaves(value: Any) -> list[np.ndarray]:
    leaves: list[np.ndarray] = []
    for leaf in jax.tree.leaves(value):
        array = np.ascontiguousarray(np.asarray(leaf))
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

    full_leaves = _numeric_gradient_leaves(full)
    arm_leaves = _numeric_gradient_leaves(arm)
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


def _qualification_report(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Return a fail-closed Gate C qualification skeleton.

    The complete evidence recomputation is added with the authenticated runner;
    no abbreviated or incomplete report can pass this initial boundary.
    """

    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        criteria = {name: False for name in QUALIFICATION_CRITERIA}
    else:
        criteria = {name: False for name in QUALIFICATION_CRITERIA}
        criteria["schema_and_control"] = bool(
            report.get("schema_version") == GATE_C_SCHEMA_VERSION
            and not isinstance(report.get("schema_version"), (bool, np.bool_))
            and report.get("control") == GATE_C_CONTROL
        )
        criteria["exact_configuration"] = bool(
            config.qualification_regime == "preregistered_full"
            and report.get("qualification_regime") == "preregistered_full"
        )
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
    parser.add_argument("--target", choices=("gate_c_init",), required=True)
    parser.add_argument("--gate-a-result", type=Path, required=True)
    parser.add_argument("--gate-a-manifest", type=Path, required=True)
    parser.add_argument("--gate-b-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fixed authenticated Gate C initialization target.

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

    source_start = gate_a._source_report()
    environment = gate_a._environment_report()
    gate_a._require_authenticated_gpu_launch(source_start, environment)
    head = str(source_start["commit"])
    expected_paths = launcher.target_paths(
        launch_config,
        head,
        "gate_c_init",
    )
    if args.output.resolve() != expected_paths.result.resolve():
        raise ValueError("Gate C target requires the fixed output path")

    prerequisites = launcher._load_gate_c_prerequisites(launch_config)
    source_files = _source_files_report()
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
