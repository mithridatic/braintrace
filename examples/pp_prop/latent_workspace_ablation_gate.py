"""Preregistered pp-prop mechanism ablations for Example 21 Gate C."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real
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
