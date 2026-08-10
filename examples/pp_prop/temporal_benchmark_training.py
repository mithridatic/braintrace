"""BrainTrace pp-prop and matched BPTT execution for Example 17."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import brainevent
import brainpy.state
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintrace
from temporal_benchmark_config import (
    HORIZONS,
    FEEDFORWARD_SYNAPSE_TAU_MS,
    MEMBRANE_TAU_MS,
    READOUT_TAU_MS,
    RECURRENT_SYNAPSE_TAU_MS,
    TemporalBenchmarkConfig,
    half_life_decay,
)
from temporal_benchmark_data import (
    encode_trials,
    response_is_label_independent,
    response_mask,
)
from temporal_benchmark_manifest import (
    SeedBundle,
    materialize_sealed_test_specs,
    split_specs,
)
from temporal_benchmark_metrics import classification_metrics, dynamics_metrics
from temporal_benchmark_optimizer import SealedLearningRateSchedule
from temporal_benchmark_supervision import (
    algorithm_label,
    parameter_group,
    policy_for_arm,
)
from temporal_benchmark_topology import (
    FixedDegreeTopology,
    fixed_degree_topology,
    topology_metrics,
)


def _derived_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def _initial_projection(shape: tuple[int, int], seed: int, scale: float) -> jax.Array:
    random = np.random.default_rng(seed)
    return jnp.asarray(random.normal(0.0, scale, shape), dtype=jnp.float32)


class _TemporalCell(brainstate.nn.Module):
    """LIF cell with fixed dense input and optional native sparse recurrence."""

    def __init__(
        self,
        config: TemporalBenchmarkConfig,
        topology: FixedDegreeTopology,
        feedforward: jax.Array,
        recurrence_module: bool,
        recurrence_active: bool,
    ):
        super().__init__()
        self.recurrence_active = recurrence_active
        self.neu = brainpy.state.LIF(
            config.neurons,
            R=1.0 * u.ohm,
            tau=MEMBRANE_TAU_MS * u.ms,
            V_th=1.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_rest=0.0 * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        self.ff_syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                17, config.neurons, w_init=feedforward * u.mA, b_init=None
            ),
            syn=brainpy.state.Expon(
                config.neurons,
                tau=FEEDFORWARD_SYNAPSE_TAU_MS * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.0),
            post=self.neu,
        )
        self.rec_syn = (
            self._recurrent_projection(config, topology) if recurrence_module else None
        )

    def _recurrent_projection(
        self, config: TemporalBenchmarkConfig, topology: FixedDegreeTopology
    ) -> brainpy.state.AlignPostProj:
        sparse = brainevent.CSR(
            jnp.asarray(topology.values),
            jnp.asarray(topology.indices, dtype=jnp.int32),
            jnp.asarray(topology.indptr, dtype=jnp.int32),
            shape=(config.neurons, config.neurons),
            backend="jax_raw",
        )
        linear = braintrace.nn.SparseLinear(sparse, b_init=None)
        parameters = dict(linear.weight.value)
        parameters["weight"] = parameters["weight"] * u.mA
        linear.weight.value = parameters
        return brainpy.state.AlignPostProj(
            comm=linear,
            syn=brainpy.state.Expon(
                config.neurons,
                tau=RECURRENT_SYNAPSE_TAU_MS * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.0),
            post=self.neu,
        )

    def update(self, inputs: jax.Array) -> jax.Array:
        """Advance feed-forward, optional recurrent, and LIF state one step."""
        self.ff_syn(inputs)
        if self.rec_syn is not None:
            recurrent_input = self.neu.get_spike()
            if not self.recurrence_active:
                recurrent_input = jnp.zeros_like(recurrent_input)
            self.rec_syn(recurrent_input)
        self.neu(0.0 * u.mA)
        return self.neu.get_spike()


class _TemporalNet(brainstate.nn.Module):
    """Delayed-cue classifier with a frozen-threshold LIF substrate."""

    def __init__(
        self,
        config: TemporalBenchmarkConfig,
        topology: FixedDegreeTopology,
        feedforward: jax.Array,
        readout: jax.Array,
    ):
        super().__init__()
        policy = policy_for_arm(config.arm)
        self.cell = _TemporalCell(
            config,
            topology,
            feedforward,
            policy.recurrence_module,
            policy.recurrence_active,
        )
        self.readout = braintrace.nn.LeakyRateReadout(
            in_size=config.neurons,
            out_size=2,
            tau=READOUT_TAU_MS * u.ms,
            w_init=readout,
        )

    def update(self, inputs: jax.Array) -> jax.Array:
        """Return current classification logits after one LIF step."""
        return self.readout(self.cell(inputs))


@dataclass(frozen=True)
class _Runtime:
    model: _TemporalNet
    learner: Any | None
    groups: dict[str, Any]
    optimizers: dict[str, Any]
    schedules: dict[str, SealedLearningRateSchedule]
    topology: FixedDegreeTopology
    batch_size: int


def _build_model(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> tuple[_TemporalNet, FixedDegreeTopology]:
    topology = fixed_degree_topology(
        config.neurons,
        config.degree,
        bundle.topology_seed,
        bundle.weight_seed,
        config.gain,
    )
    feedforward = _initial_projection(
        (17, config.neurons),
        _derived_seed(bundle.weight_seed, "feedforward"),
        6.0 / np.sqrt(17.0),
    )
    readout = _initial_projection(
        (config.neurons, 2),
        _derived_seed(bundle.weight_seed, "readout"),
        1.0 / np.sqrt(config.neurons),
    )
    return _TemporalNet(config, topology, feedforward, readout), topology


def _parameter_groups(model: _TemporalNet) -> dict[str, Any]:
    weights = model.states(brainstate.ParamState)
    return {
        name: brainstate.util.FlattedDict(
            {
                path: state
                for path, state in weights.items()
                if parameter_group(path) == name
            }
        )
        for name in ("readout", "feedforward", "recurrent")
    }


def _group_enabled(config: TemporalBenchmarkConfig, name: str) -> bool:
    policy = policy_for_arm(config.arm)
    return bool(getattr(policy, f"train_{name}"))


def _build_optimizers(
    config: TemporalBenchmarkConfig,
    groups: dict[str, Any],
    total_updates: int | None = None,
) -> tuple[dict[str, Any], dict[str, SealedLearningRateSchedule]]:
    optimizers: dict[str, Any] = {}
    schedules: dict[str, SealedLearningRateSchedule] = {}
    for name, group in groups.items():
        if not group or not _group_enabled(config, name):
            continue
        peak = float(getattr(config.learning_rates, name))
        schedule = SealedLearningRateSchedule(peak, total_updates or config.updates)
        decay = config.recurrent_weight_decay if name == "recurrent" else 0.0
        optimizer = braintools.optim.Adam(lr=schedule, weight_decay=decay)
        optimizer.register_trainable_weights(group)
        optimizers[name] = optimizer
        schedules[name] = schedule
    return optimizers, schedules


def _build_runtime(
    config: TemporalBenchmarkConfig,
    bundle: SeedBundle,
    optimizer_updates: int | None = None,
) -> _Runtime:
    model, topology = _build_model(config, bundle)
    brainstate.nn.init_all_states(model, batch_size=config.batch_size)
    groups = _parameter_groups(model)
    optimizers, schedules = _build_optimizers(config, groups, optimizer_updates)
    learner = _compile_learner(model, config)
    return _Runtime(
        model, learner, groups, optimizers, schedules, topology, config.batch_size
    )


def _compile_learner(model: _TemporalNet, config: TemporalBenchmarkConfig):
    """Compile phase-local eligibility traces without replacing model weights."""
    policy = policy_for_arm(config.arm)
    if policy.algorithm != "pp_prop":
        return None
    decays = (
        half_life_decay(config.trace_half_life_x_steps),
        half_life_decay(config.trace_half_life_f_steps),
    )
    return braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((config.batch_size, 17), dtype=jnp.float32),
        batch_size=config.batch_size,
        vmap=False,
        decay_or_rank=decays,
        vjp_method="single-step",
    )


def _reset(runtime: _Runtime) -> None:
    brainstate.nn.reset_all_states(runtime.model, batch_size=runtime.batch_size)
    if runtime.learner is not None:
        runtime.learner.reset_state(batch_size=runtime.batch_size)


def _tree_norm(tree: Any) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    sums = [jnp.sum(jnp.square(u.get_mantissa(leaf))) for leaf in leaves]
    return jnp.sqrt(sum(sums, jnp.asarray(0.0)))


def _clip_tree(tree: Any, clip_norm: float | None) -> tuple[Any, jax.Array, jax.Array]:
    raw_norm = _tree_norm(tree)
    if clip_norm is None:
        return tree, raw_norm, raw_norm
    scale = jnp.minimum(1.0, clip_norm / jnp.maximum(raw_norm, 1e-12))
    clipped = jax.tree.map(lambda leaf: leaf * scale, tree)
    return clipped, raw_norm, _tree_norm(clipped)


def _group_gradients(gradients: Any, group: Any) -> Any:
    return brainstate.util.FlattedDict(
        {path: gradients[path] for path in group if path in gradients}
    )


def _moment_trees(value: Any) -> tuple[Any, Any]:
    if hasattr(value, "mu") and hasattr(value, "nu"):
        return value.mu, value.nu
    if isinstance(value, (tuple, list)):
        for child in value:
            first, second = _moment_trees(child)
            if first is not None:
                return first, second
    return None, None


def _group_update(
    runtime: _Runtime, name: str, gradients: Any, clip_norm: float | None
) -> jax.Array:
    group = runtime.groups[name]
    optimizer = runtime.optimizers[name]
    before = group.to_dict_values()
    group_gradients = _group_gradients(gradients, group)
    clipped, raw_norm, clipped_norm = _clip_tree(group_gradients, clip_norm)
    weight_norm = _tree_norm(before)
    applied_learning_rate = runtime.schedules[name].current_lrs.value[0]
    optimizer.update(clipped)
    runtime.schedules[name].step()
    after = group.to_dict_values()
    updates = jax.tree.map(lambda new, old: new - old, after, before)
    first, second = _moment_trees(optimizer.opt_state.value)
    first_norm = _tree_norm(first) if first is not None else jnp.asarray(0.0)
    second_norm = _tree_norm(second) if second is not None else jnp.asarray(0.0)
    update_norm = _tree_norm(updates)
    ratio = update_norm / jnp.maximum(weight_norm, 1e-12)
    return jnp.stack(
        (
            raw_norm,
            clipped_norm,
            raw_norm > clipped_norm + 1e-12,
            weight_norm,
            update_norm,
            ratio,
            first_norm,
            second_norm,
            applied_learning_rate,
        )
    )


def _pp_prop_gradients(
    runtime: _Runtime, spikes: jax.Array, labels: jax.Array, mask: jax.Array
) -> tuple[Any, jax.Array]:
    assert runtime.learner is not None

    def step_loss(step_spikes: jax.Array) -> jax.Array:
        logits = runtime.learner(step_spikes)
        return braintools.metric.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()

    return runtime.learner.etrace_grad(
        spikes,
        step_fn=step_loss,
        mask=mask,
        reduction="mean",
        loss_output="scalar",
        return_value=True,
    )


def _bptt_gradients(
    runtime: _Runtime, spikes: jax.Array, labels: jax.Array, mask: jax.Array
) -> tuple[Any, jax.Array]:
    weights = runtime.model.states(brainstate.ParamState)

    def step_loss(step_spikes: jax.Array) -> jax.Array:
        logits = runtime.model(step_spikes)
        return braintools.metric.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()

    def objective() -> jax.Array:
        losses = brainstate.transform.for_loop(step_loss, spikes)
        return jnp.sum(losses * mask) / jnp.sum(mask)

    return brainstate.transform.grad(objective, weights, return_value=True)()


def _make_train_many(runtime: _Runtime, config: TemporalBenchmarkConfig):
    mask = jnp.asarray(response_mask(HORIZONS[config.horizon]))

    def update(batch_spikes: jax.Array, labels: jax.Array):
        _reset(runtime)
        if runtime.learner is None:
            gradients, objective = _bptt_gradients(runtime, batch_spikes, labels, mask)
        else:
            gradients, objective = _pp_prop_gradients(
                runtime, batch_spikes, labels, mask
            )
        telemetry = {
            name: _group_update(
                runtime,
                name,
                gradients,
                getattr(config.gradient_clip_norms, name),
            )
            for name in runtime.optimizers
        }
        return objective, telemetry

    @brainstate.transform.jit
    def train_many(spikes: jax.Array, labels: jax.Array):
        return brainstate.transform.for_loop(update, spikes, labels)

    return train_many


def _training_batches(
    config: TemporalBenchmarkConfig,
    bundle: SeedBundle,
) -> tuple[jax.Array, jax.Array]:
    horizon = HORIZONS[config.horizon]
    specs = split_specs(bundle, "train", config.split_sizes)
    encoded = encode_trials(
        specs,
        horizon,
        bundle.training_encoding_seed,
        cue_rate_hz=config.cue_rate_hz,
        go_rate_hz=config.go_rate_hz,
        dt_seconds=config.dt_seconds,
    )
    labels = np.asarray([spec.label for spec in specs], dtype=np.int32)
    random = np.random.default_rng(bundle.training_order_seed)
    batch_spikes = np.empty(
        (config.updates, horizon.total_steps, config.batch_size, 17), dtype=np.float32
    )
    batch_labels = np.empty((config.updates, config.batch_size), dtype=np.int32)
    batches_per_epoch = len(specs) // config.batch_size
    order = np.arange(len(specs))
    for update in range(config.updates):
        batch_index = update % batches_per_epoch
        if batch_index == 0:
            order = random.permutation(len(specs))
        start = batch_index * config.batch_size
        indices = order[start : start + config.batch_size]
        batch_spikes[update] = encoded[:, indices]
        batch_labels[update] = labels[indices]
    return jnp.asarray(batch_spikes), jnp.asarray(batch_labels)


def _copy_parameters(source: _TemporalNet, target: _TemporalNet) -> None:
    source_states = source.states(brainstate.ParamState)
    target_states = target.states(brainstate.ParamState)
    for path, state in target_states.items():
        state.value = jax.tree.map(lambda value: value, source_states[path].value)


def _evaluate(
    runtime: _Runtime,
    config: TemporalBenchmarkConfig,
    bundle: SeedBundle,
    split: str = "validation",
) -> tuple[dict[str, float], dict[str, object]]:
    specs = (
        materialize_sealed_test_specs(
            bundle, config.split_sizes, sealed=config.sealed_test
        )
        if split == "test"
        else split_specs(bundle, split, config.split_sizes)
    )
    horizon = HORIZONS[config.horizon]
    encoded = [
        encode_trials(
            specs,
            horizon,
            seed,
            cue_rate_hz=config.cue_rate_hz,
            go_rate_hz=config.go_rate_hz,
            dt_seconds=config.dt_seconds,
        )
        for seed in bundle.evaluation_encoding_seeds
    ]
    stacked = jnp.asarray(np.concatenate(encoded, axis=1))
    evaluator, _ = _build_model(config, bundle)
    evaluation_batch = len(specs) * len(bundle.evaluation_encoding_seeds)
    brainstate.nn.init_all_states(evaluator, batch_size=evaluation_batch)
    _copy_parameters(runtime.model, evaluator)

    @brainstate.transform.jit
    def evolve(inputs: jax.Array):
        brainstate.nn.reset_all_states(evaluator, batch_size=evaluation_batch)

        def step(step_inputs: jax.Array):
            logits = evaluator(step_inputs)
            spikes = evaluator.cell.neu.get_spike()
            voltage = u.get_mantissa(evaluator.cell.neu.V.value)
            return logits, spikes, voltage

        return brainstate.transform.for_loop(step, inputs)

    outputs, spikes, voltages = evolve(stacked)
    jax.block_until_ready(outputs)
    response_logits = np.asarray(outputs[-horizon.response_steps :].mean(axis=0))
    logits = response_logits.reshape(
        len(bundle.evaluation_encoding_seeds), len(specs), 2
    )
    labels = np.asarray([spec.label for spec in specs], dtype=np.int64)
    return classification_metrics(logits, labels), dynamics_metrics(
        np.asarray(spikes), np.asarray(voltages)
    )


_TELEMETRY_FIELDS = (
    "raw_gradient_norm",
    "clipped_gradient_norm",
    "clip_event",
    "weight_norm",
    "update_norm",
    "update_to_weight_ratio",
    "adam_first_moment_norm",
    "adam_second_moment_norm",
    "effective_learning_rate",
)


def _format_telemetry(values: dict[str, jax.Array]) -> dict[str, object]:
    formatted: dict[str, object] = {}
    for group, group_values in values.items():
        array = np.asarray(group_values)
        formatted[group] = {
            field: array[:, index].tolist()
            for index, field in enumerate(_TELEMETRY_FIELDS)
        }
    return formatted


def run_training(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> dict[str, object]:
    """Run one arm and return raw metrics without claiming a scientific gate."""
    started = time.perf_counter()
    with brainstate.environ.context(dt=config.dt_seconds * u.second):
        runtime = _build_runtime(config, bundle)
        initial_metrics, _ = _evaluate(runtime, config, bundle)
        batch_spikes, batch_labels = _training_batches(config, bundle)
        losses, telemetry = _make_train_many(runtime, config)(
            batch_spikes, batch_labels
        )
        jax.block_until_ready(losses)
        final_metrics, dynamics = _evaluate(runtime, config, bundle)
        sealed_test_metrics = (
            _evaluate(runtime, config, bundle, "test")[0]
            if config.sealed_test
            else None
        )
    return {
        "status": "completed",
        "algorithm": algorithm_label(policy_for_arm(config.arm)),
        "bundle_id": bundle.bundle_id,
        "arm": config.arm,
        "horizon": config.horizon,
        "initial_validation": initial_metrics,
        "final_validation": final_metrics,
        "sealed_test_metrics": sealed_test_metrics,
        "losses": np.asarray(losses).tolist(),
        "optimizer_telemetry": _format_telemetry(telemetry),
        "dynamics": dynamics,
        "topology": topology_metrics(runtime.topology),
        "trace_decays": {
            "x": half_life_decay(config.trace_half_life_x_steps),
            "f": half_life_decay(config.trace_half_life_f_steps),
        },
        "fixed_dynamics_time_constants_ms": {
            "feedforward_synapse": FEEDFORWARD_SYNAPSE_TAU_MS,
            "recurrent_synapse": RECURRENT_SYNAPSE_TAU_MS,
            "membrane": MEMBRANE_TAU_MS,
            "readout": READOUT_TAU_MS,
        },
        "response_label_independent": response_is_label_independent(
            HORIZONS[config.horizon], config.cue_rate_hz, config.go_rate_hz
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "config": asdict(config),
    }
