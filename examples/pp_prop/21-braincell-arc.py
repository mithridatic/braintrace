"""BrainCell compatibility fixtures for the Example 21 replacement."""

from __future__ import annotations

import braincell
import brainstate
import braintrace
import brainunit as u
import braintools
import brainevent
import jax
import jax.numpy as jnp


N_INPUTS = 441
N_NEURONS = 2048
N_READOUT = 360
INPUT_FANOUT = 32
RECURRENT_FANOUT = 8
TRACE_DECAY = 0.95
GRADIENT_CLIP_NORM = 1.0
PROOF_UPDATES = 8
ORDINARY_UPDATES = 64
LEARNING_RATES = {"input": 0.001, "recurrent": 0.0003, "readout": 0.003}


def _normal(shape, seed, scale):
    return brainstate.random.normal(
        size=shape, key=brainstate.random.RandomState(seed).value
    ).astype(jnp.float32) * scale


def input_topology():
    """Build the deterministic feature-to-neuron CSR topology."""

    targets = jnp.asarray(
        [(131 * feature + 61 * k) % N_NEURONS
         for feature in range(N_INPUTS)
         for k in range(INPUT_FANOUT)],
        dtype=jnp.int32,
    )
    return brainevent.CSR(
        _normal((N_INPUTS * INPUT_FANOUT,), 21, 1.0 / jnp.sqrt(32.0)),
        targets,
        jnp.arange(0, N_INPUTS * INPUT_FANOUT + 1, INPUT_FANOUT, dtype=jnp.int32),
        shape=(N_INPUTS, N_NEURONS),
    )


def recurrent_topology():
    """Build the deterministic source-row recurrent CSR topology."""

    offsets = jnp.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=jnp.int32)
    sources = jnp.arange(N_NEURONS, dtype=jnp.int32)[:, None]
    targets = ((sources + offsets) % N_NEURONS).reshape(-1)
    return brainevent.CSR(
        _normal((N_NEURONS * RECURRENT_FANOUT,), 22, 1.0 / jnp.sqrt(8.0)),
        targets,
        jnp.arange(0, N_NEURONS * RECURRENT_FANOUT + 1, RECURRENT_FANOUT,
                   dtype=jnp.int32),
        shape=(N_NEURONS, N_NEURONS),
    )


def bounded_population_current(input_drive, recurrent_drive):
    """Return bounded BrainCell current density from dimensionless drives."""

    return (
        0.02 * jnp.tanh(input_drive) + 0.01 * jnp.tanh(recurrent_drive)
    ) * u.mA / u.cm**2


def clip_gradient(gradient, max_norm=GRADIENT_CLIP_NORM):
    """Clip a pytree gradient by its global Euclidean norm."""

    leaves = jax.tree_util.tree_leaves(gradient)
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny))
    return jax.tree_util.tree_map(lambda leaf: leaf * scale, gradient), norm


def accumulate_masked_loss(losses, valid_rows):
    """Sum only losses belonging to valid shape or row requests."""

    values = jnp.asarray(losses)
    mask = jnp.asarray(valid_rows, dtype=values.dtype)
    return jnp.sum(values * mask)


class AdamState:
    """Adam first and second moments for one episode schedule."""

    def __init__(self, first, second, step=0):
        self.first = first
        self.second = second
        self.step = step


def adam_update(parameters, gradient, state, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply one bias-corrected Adam update and return new state."""

    first = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g, state.first, gradient
    )
    second = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g), state.second, gradient
    )
    step = state.step + 1
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    updated = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * (m / correction1) /
        (jnp.sqrt(v / correction2) + eps), parameters, first, second
    )
    return updated, AdamState(first, second, step)


def grouped_adam_update(parameters, gradients, states=None):
    """Apply the declared input, recurrent, and readout Adam rates."""

    states = states or {
        name: AdamState(jnp.zeros_like(value), jnp.zeros_like(value))
        for name, value in parameters.items()
    }
    updated = dict(parameters)
    next_states = dict(states)
    for name, rate in LEARNING_RATES.items():
        if name not in parameters or name not in gradients:
            continue
        updated[name], next_states[name] = adam_update(
            parameters[name], gradients[name], states[name], rate
        )
    return updated, next_states


def update_schedule(steps, proof=False):
    """Return the fixed update count for proof or ordinary training."""

    expected = PROOF_UPDATES if proof else ORDINARY_UPDATES
    if steps != expected:
        raise ValueError(f"expected exactly {expected} updates, got {steps}")
    return tuple(range(expected))


def compile_pp_prop_model(model):
    """Compile state-changing sparse parameters with PP-Prop single-step."""

    return braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((N_INPUTS,), dtype=jnp.float32),
        batch_size=None,
        decay_or_rank=TRACE_DECAY,
        vjp_method="single-step",
    )


class BrainCellArcModel(brainstate.nn.Module):
    """The 2,048-neuron sparse BrainCell baseline."""

    def __init__(self):
        super().__init__()
        self.input_csr = input_topology()
        self.recurrent_csr = recurrent_topology()
        self.input_weight = brainstate.ParamState(self.input_csr.data)
        self.recurrent_weight = brainstate.ParamState(self.recurrent_csr.data)
        self.readout_weight = brainstate.ParamState(
            _normal((N_NEURONS, N_READOUT), 23, 1.0 / jnp.sqrt(float(N_NEURONS)))
        )
        self.readout_bias = brainstate.ParamState(jnp.zeros((N_READOUT,), dtype=jnp.float32))
        self.previous_spikes = brainstate.HiddenState(jnp.zeros((N_NEURONS,), dtype=jnp.float32))
        self.cell = CompatibilityHodgkinHuxley(N_NEURONS)
        self.cell.init_state()
        self.reset_episode()

    def reset_episode(self, learner=None):
        """Reset biological and eligibility state while retaining parameters."""

        self.cell.reset_state()
        self.previous_spikes.value = jnp.zeros((N_NEURONS,), dtype=jnp.float32)
        if learner is not None and hasattr(learner, "reset_state"):
            learner.reset_state()

    def _advance(self, event, dt_ms=0.1):
        input_drive = braintrace.sparse_matmul(event, self.input_weight.value, sparse_mat=self.input_csr)
        recurrent_drive = braintrace.sparse_matmul(
            self.previous_spikes.value, self.recurrent_weight.value, sparse_mat=self.recurrent_csr
        )
        current = bounded_population_current(input_drive, recurrent_drive)
        with brainstate.environ.context(dt=dt_ms * u.ms):
            self.cell.update(current)
        self.previous_spikes.value = jax.lax.stop_gradient(
            self.cell.spike.value.astype(jnp.float32)
        )
        return self.cell.V.value.to_decimal(u.mV)

    def update(self, event):
        """Advance one event for the BrainTrace compiler."""

        return self._advance(event)

    def step(self, event, advance=True):
        """Run one event, preserving all state for a false advance."""

        return brainstate.transform.cond(
            advance, lambda: self._advance(event), lambda: jnp.zeros((N_NEURONS,))
        )

    def interval(self, event, advance=True, *, substeps=1):
        """Advance one biological interval with one or two compiled substeps."""

        def run(_):
            return self._advance(event, 0.1 / substeps)

        def advancing():
            return brainstate.transform.for_loop(
                run, jnp.arange(substeps, dtype=jnp.int32)
            )[-1]

        return brainstate.transform.cond(
            advance, advancing, lambda: jnp.zeros((N_NEURONS,))
        )

    def readout(self):
        """Return direct voltage readout logits."""

        feature = jnp.tanh((self.cell.V.value.to_decimal(u.mV) + 65.0) / 20.0)
        return feature @ self.readout_weight.value + self.readout_bias.value


def run_event_sequence(model, events, advances=None):
    """Run a compiled event sequence and return one voltage vector per event.

    Parameters
    ----------
    model : BrainCellArcModel
        Model whose state is advanced.
    events : array-like
        Event vectors with shape ``(time, 441)``.
    advances : array-like, optional
        Boolean event mask. Missing values mean that every event advances.

    Returns
    -------
    jax.Array
        Voltage values with shape ``(time, 2048)``.
    """

    events = jnp.asarray(events, dtype=jnp.float32)
    if advances is None:
        advances = jnp.ones((events.shape[0],), dtype=bool)
    advances = jnp.asarray(advances, dtype=bool)
    if events.ndim != 2 or events.shape[1] != N_INPUTS:
        raise ValueError(f"events must have shape (time, {N_INPUTS})")
    if advances.shape != (events.shape[0],):
        raise ValueError("advances must have one boolean per event")

    def drive(xs, mask):
        return brainstate.transform.for_loop(
            lambda event, advance: model.step(event, advance), xs, mask
        )

    return brainstate.transform.jit(drive)(events, advances)


def run_pp_prop_sequence(learner, events, advances=None):
    """Evolve a PP-Prop learner only on advancing events."""

    events = jnp.asarray(events, dtype=jnp.float32)
    if advances is None:
        advances = jnp.ones((events.shape[0],), dtype=bool)
    advances = jnp.asarray(advances, dtype=bool)

    def drive(event, advance):
        def evolve():
            return learner.etrace_evolve(
                event[None, :], return_outputs=True
            )[0]

        return brainstate.transform.cond(
            advance, evolve, lambda: jnp.zeros((N_NEURONS,))
        )

    return brainstate.transform.jit(
        lambda xs, mask: brainstate.transform.for_loop(drive, xs, mask)
    )(events, advances)


def matched_integration_check(events):
    """Compare the default interval with two compiled half-steps."""

    events = jnp.asarray(events, dtype=jnp.float32)
    one_step = BrainCellArcModel()
    half_step = BrainCellArcModel()
    default = brainstate.transform.jit(
        lambda xs: brainstate.transform.for_loop(
            lambda event: one_step.interval(event), xs
        )
    )(events)
    matched = brainstate.transform.jit(
        lambda xs: brainstate.transform.for_loop(
            lambda event: half_step.interval(event, substeps=2), xs
        )
    )(events)
    voltage_difference = jnp.max(jnp.abs(default - matched))
    spike_equal = jnp.array_equal(
        one_step.previous_spikes.value, half_step.previous_spikes.value
    )
    prediction_equal = jnp.array_equal(one_step.readout(), half_step.readout())
    strict_equal = prediction_equal
    return {
        "finite": bool(jnp.all(jnp.isfinite(default)) and jnp.all(jnp.isfinite(matched))),
        "max_voltage_difference": float(voltage_difference),
        "spike_equal": bool(spike_equal),
        "prediction_equal": bool(prediction_equal),
        "strict_equal": bool(strict_equal),
        "default_selected": bool(
            jnp.all(jnp.isfinite(default))
            and jnp.all(jnp.isfinite(matched))
            and voltage_difference <= 1.0
            and spike_equal
            and prediction_equal
            and strict_equal
        ),
    }


def decoder_boundary_intervention(model):
    """Return direct predictions before and after a state-only intervention."""

    before = model.readout()
    model.cell.V.value = model.cell.V.value + 1.0 * u.mV
    after = model.readout()
    return before, after


class PPPropEpisodeTrainer:
    """Accumulate one PP-Prop episode and apply one clipped Adam update."""

    def __init__(self, learner, parameters, learning_rates=None):
        self.learner = learner
        self.parameters = parameters
        self.learning_rates = learning_rates or LEARNING_RATES
        zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
        self.adam = AdamState(zeros, zeros)
        self.updates = 0
        self.adam_groups = {
            name: AdamState(jnp.zeros_like(value), jnp.zeros_like(value))
            for name, value in parameters.items()
        } if isinstance(parameters, dict) else None

    def reset_episode(self, model):
        """Reset model and eligibility state before the next query episode."""

        model.reset_episode(self.learner)

    def optimizer_is_finite(self):
        """Return whether parameters, moments, and step state are finite."""

        leaves = jax.tree_util.tree_leaves((self.parameters, self.adam.first, self.adam.second))
        return bool(all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves))

    def update_episode(self, events, step_fn, valid_rows=None):
        """Apply one update from one masked PP-Prop query episode."""

        gradients, losses = self.learner.etrace_grad(
            events, step_fn=step_fn, reduction="sum", return_value=True
        )
        if valid_rows is not None:
            losses = accumulate_masked_loss(losses, valid_rows)
        gradients, norm = clip_gradient(gradients)
        if self.adam_groups is not None:
            self.parameters, self.adam_groups = grouped_adam_update(
                self.parameters, gradients, self.adam_groups
            )
            self.adam = self.adam_groups.get("input", self.adam)
        else:
            rate = self.learning_rates.get("input", 0.001)
            self.parameters, self.adam = adam_update(
                self.parameters, gradients, self.adam, rate
            )
        self.updates += 1
        return losses, norm


def run_fixed_schedule(trainer, episodes, *, proof=False):
    """Run the exact proof or ordinary number of ordered episodes."""

    update_schedule(len(episodes), proof=proof)
    if proof and any(
        "task_id" in episode and episode["task_id"] != "d631b094"
        for episode in episodes
    ):
        raise ValueError("proof schedule accepts only d631b094")
    if any(episode.get("validation", False) for episode in episodes):
        raise ValueError("validation episodes are forward-only")
    return brainstate.transform.for_loop(
        lambda episode: trainer.update_episode(**episode), episodes
    )
import importlib.util
import sys
from pathlib import Path


_ARC_CONTRACTS_SPEC = importlib.util.spec_from_file_location(
    "example21_arc_contracts", Path(__file__).with_name("arc_contracts.py")
)
assert _ARC_CONTRACTS_SPEC is not None and _ARC_CONTRACTS_SPEC.loader is not None
_ARC_CONTRACTS = importlib.util.module_from_spec(_ARC_CONTRACTS_SPEC)
sys.modules[_ARC_CONTRACTS_SPEC.name] = _ARC_CONTRACTS
_ARC_CONTRACTS_SPEC.loader.exec_module(_ARC_CONTRACTS)
ARCTask = _ARC_CONTRACTS.ARCTask
load_task = _ARC_CONTRACTS.load_task
encode_episode = _ARC_CONTRACTS.encode_episode
decode_episode = _ARC_CONTRACTS.decode_episode
request_loss = _ARC_CONTRACTS.request_loss
decode_prediction = _ARC_CONTRACTS.decode_prediction
query_exact = _ARC_CONTRACTS.query_exact
strict_task_pass_at_1 = _ARC_CONTRACTS.strict_task_pass_at_1
write_result = _ARC_CONTRACTS.write_result
write_checkpoint = _ARC_CONTRACTS.write_checkpoint
load_checkpoint = _ARC_CONTRACTS.load_checkpoint


N_FEATURES = 1
N_CELLS = 4
N_READOUT = 360
FINITE_DIFFERENCE_EPSILON = 1e-3
OBJECTIVE_VOLTAGE_OFFSET = 65.0
OBJECTIVE_VOLTAGE_SCALE = 20.0
SPIKE_DRIVE = 20.0


class CompatibilityHodgkinHuxley(braincell.SingleCompartment):
    """Construct the four-cell BrainCell 0.1.0 Hodgkin–Huxley fixture."""

    def __init__(self, size: int = N_CELLS):
        super().__init__(
            size,
            length=10.0 * u.um,
            radius=5.0 * u.um,
            C=1.0 * u.uF / u.cm**2,
            V_th=0.0 * u.mV,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            spk_fun=braintools.surrogate.ReluGrad(alpha=0.3, width=1.0),
            solver="ind_exp_euler",
        )
        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(
            INa=braincell.channel.Na_HH1952(
                size,
                g_max=120.0 * u.mS / u.cm**2,
                temp=309.15 * u.kelvin,
                temp_ref=309.15 * u.kelvin,
                V_sh=-45.0 * u.mV,
                q10=3.0,
            )
        )
        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(
            IK=braincell.channel.K_HH1952(
                size,
                g_max=10.0 * u.mS / u.cm**2,
                temp=309.15 * u.kelvin,
                temp_ref=309.15 * u.kelvin,
                V_sh=-45.0 * u.mV,
                q10=3.0,
            )
        )
        self.IL = braincell.channel.IL(
            size,
            g_max=0.03 * u.mS / u.cm**2,
            E=-54.387 * u.mV,
        )


def input_csr() -> brainevent.CSR:
    """Return the declared one-feature by four-cell CSR relation."""

    return brainevent.CSR(
        jnp.asarray([0.1, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        jnp.asarray([0, 4], dtype=jnp.int32),
        shape=(1, 4),
    )


def recurrent_csr() -> brainevent.CSR:
    """Return the four-cell recurrent CSR relation."""

    return brainevent.CSR(
        jnp.ones((N_CELLS,), dtype=jnp.float32),
        jnp.arange(N_CELLS, dtype=jnp.int32),
        jnp.arange(N_CELLS + 1, dtype=jnp.int32),
        shape=(N_CELLS, N_CELLS),
    )


def bounded_current_density(input_drive, recurrent_drive=0.0):
    """Convert bounded dimensionless drives to current density."""

    return (
        0.02 * jnp.tanh(input_drive / 20.0) * u.mA / u.cm**2
        + 0.01 * jnp.tanh(recurrent_drive / 20.0) * u.mA / u.cm**2
    )


class PPPropRelationFixture(brainstate.nn.Module):
    """Expose input and recurrent sparse weights to the PP-Prop compiler."""

    def __init__(self, input_weight=0.1):
        super().__init__()
        self.hidden = brainstate.HiddenState(jnp.zeros((1, N_CELLS)))
        self.cell = CompatibilityHodgkinHuxley()
        self.cell.init_state()
        self.cell.reset_state()
        self.input_weight = brainstate.ParamState(
            jnp.asarray([input_weight, 0.0, 0.0, 0.0])
        )
        self.recurrent_weight = brainstate.ParamState(jnp.ones((N_CELLS,)))
        self._input_csr = input_csr()
        self._recurrent_csr = recurrent_csr()

    def update(self, x):
        hidden = braintrace.sparse_matmul(
            x,
            self.input_weight.value,
            sparse_mat=self._input_csr,
        )
        recurrent = braintrace.sparse_matmul(
            self.hidden.value,
            self.recurrent_weight.value,
            sparse_mat=self._recurrent_csr,
        )
        drive = hidden + recurrent
        with brainstate.environ.context(dt=0.1 * u.ms):
            self.cell.update(bounded_current_density(drive))
        self.hidden.value = self.cell.V.value.to_decimal(u.mV)
        return self.hidden.value


def pp_prop_relation_fixture():
    """Compile and return the two sparse hidden-state relations."""

    model = PPPropRelationFixture()
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, N_FEATURES), dtype=jnp.float32),
        batch_size=1,
        decay_or_rank=0.95,
        vjp_method="single-step",
    )
    return learner.graph.hidden_param_op_relations


def _pp_prop_gradient(input_weight):
    model = PPPropRelationFixture(input_weight)
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, N_FEATURES), dtype=jnp.float32),
        batch_size=1,
        decay_or_rank=0.95,
        vjp_method="single-step",
    )

    def step_loss(x):
        voltage = learner(x)
        return jnp.mean(
            jnp.tanh((voltage + OBJECTIVE_VOLTAGE_OFFSET) / OBJECTIVE_VOLTAGE_SCALE)
        )

    gradients = learner.etrace_grad(
        jnp.ones((1, 1, N_FEATURES), dtype=jnp.float32),
        step_fn=step_loss,
        reduction="sum",
    )
    return gradients[("input_weight",)][0]


def _pp_prop_objective(input_weight):
    return _braincell_objective(input_weight)["objective"]


def _csr_input_drive(input_weight):
    weights = jnp.asarray([input_weight, 0.0, 0.0, 0.0])
    return braintrace.sparse_matmul(
        jnp.ones((1, N_FEATURES)), weights, sparse_mat=input_csr()
    ).reshape((N_CELLS,))


def advance_one_step(cell: CompatibilityHodgkinHuxley, current):
    """Advance a fixture cell by one compiled 0.1 ms interval."""

    with brainstate.environ.context(dt=0.1 * u.ms):
        cell.update(current)
    return cell.V.value, cell.spike.value


def compiled_one_step(cell: CompatibilityHodgkinHuxley):
    """Return a JIT-compiled one-step driver for a fixture cell."""

    return brainstate.transform.jit(advance_one_step, static_argnums=0)(
        cell, bounded_current_density(0.0)
    )


def finite_difference_fixture() -> dict[str, float]:
    """Compare one PP-Prop BrainCell step with a central difference."""

    weight = jnp.asarray(0.1, dtype=jnp.float32)
    epsilon = FINITE_DIFFERENCE_EPSILON

    relations = pp_prop_relation_fixture()
    direct = float(_pp_prop_gradient(weight))
    plus = _braincell_objective(weight + epsilon)
    minus = _braincell_objective(weight - epsilon)
    centered = float((plus["objective"] - minus["objective"]) / (2.0 * epsilon))
    tolerance = 1e-5 + 1e-2 * max(abs(direct), abs(centered))
    return {
        "pp_prop": direct,
        "finite_difference": centered,
        "absolute_error": abs(direct - centered),
        "tolerance": tolerance,
        "relations": float(len(relations)),
        "finite_voltage": float(
            bool(jnp.isfinite(plus["voltage"]) and jnp.isfinite(minus["voltage"]))
        ),
        "finite_gates": float(plus["finite_gates"] and minus["finite_gates"]),
        "zero_spikes": float(plus["zero_spikes"] and minus["zero_spikes"]),
        "reset_isolated": float(plus["reset_isolated"] and minus["reset_isolated"]),
    }


def _braincell_objective(input_weight):
    cell = CompatibilityHodgkinHuxley()
    cell.init_state()
    cell.reset_state()
    reset_copy = CompatibilityHodgkinHuxley()
    reset_copy.init_state()
    reset_copy.reset_state()
    reset_voltage = reset_copy.V.value.to_decimal(u.mV).copy()
    reset_gates = tuple(
        gate.value.copy()
        for gate in (reset_copy.na.INa.p, reset_copy.na.INa.q, reset_copy.k.IK.p)
    )
    current = bounded_current_density(_csr_input_drive(input_weight))
    voltage, spikes = advance_one_step(cell, current)
    objective = jnp.mean(
        jnp.tanh((voltage.to_decimal(u.mV) + OBJECTIVE_VOLTAGE_OFFSET) / OBJECTIVE_VOLTAGE_SCALE)
    )
    finite_gates = all(
        jnp.all(jnp.isfinite(gate))
        for gate in (cell.na.INa.p.value, cell.na.INa.q.value, cell.k.IK.p.value)
    )
    reset_isolated = all(
        jnp.allclose(gate.value, initial)
        for gate, initial in zip(
            (reset_copy.na.INa.p, reset_copy.na.INa.q, reset_copy.k.IK.p),
            reset_gates,
        )
    ) and jnp.allclose(reset_copy.V.value.to_decimal(u.mV), reset_voltage)
    return {
        "objective": objective,
        "voltage": jnp.mean(voltage.to_decimal(u.mV)),
        "finite_gates": bool(finite_gates),
        "zero_spikes": bool(jnp.all(spikes == 0)),
        "reset_isolated": reset_isolated,
    }


def spike_path_fixture() -> dict[str, bool]:
    """Check deterministic threshold crossing and finite surrogate activity."""

    cell = CompatibilityHodgkinHuxley()
    cell.init_state()
    cell.V.value = jnp.full((N_CELLS,), -0.001) * u.mV
    def spike_objective(input_drive):
        trial = CompatibilityHodgkinHuxley()
        trial.init_state()
        trial.reset_state()
        trial.V.value = jnp.full((N_CELLS,), -0.001) * u.mV
        _, spikes = advance_one_step(trial, bounded_current_density(input_drive))
        return jnp.sum(spikes)

    voltage, spikes = advance_one_step(cell, bounded_current_density(SPIKE_DRIVE))
    spike_gradient = jax.grad(spike_objective)(jnp.asarray(SPIKE_DRIVE))
    return {
        "threshold_crossed": bool(jnp.any(voltage >= 0.0 * u.mV)),
        "finite_voltage": bool(jnp.all(jnp.isfinite(voltage.mantissa))),
        "finite_spikes": bool(jnp.all(jnp.isfinite(spikes))),
        "finite_gradient": bool(jnp.all(jnp.isfinite(spike_gradient))),
        "nonzero_gradient": bool(jnp.any(spike_gradient != 0.0)),
    }


def direct_readout_gradient_fixture() -> dict[str, bool]:
    """Check finite direct gradients for all 360 voltage-readout values."""

    features = jnp.asarray([[0.1, -0.2, 0.3, -0.4]])
    weight = jnp.ones((N_CELLS, N_READOUT)) * 0.1
    bias = jnp.zeros((N_READOUT,))

    def objective(readout_weight, readout_bias):
        return jnp.sum(jnp.tanh(features @ readout_weight + readout_bias))

    grad_weight, grad_bias = jax.grad(objective, argnums=(0, 1))(weight, bias)
    return {
        "shape": grad_bias.shape == (N_READOUT,),
        "finite": bool(
            jnp.all(jnp.isfinite(grad_weight))
            and jnp.all(jnp.isfinite(grad_bias))
        ),
        "height_nonzero": bool(jnp.any(grad_weight[:, :30] != 0.0)),
        "width_nonzero": bool(jnp.any(grad_weight[:, 30:60] != 0.0)),
        "color_nonzero": bool(jnp.any(grad_weight[:, 60:] != 0.0)),
    }


def main() -> None:
    """Run the local compatibility checks."""

    print(finite_difference_fixture())
    print(spike_path_fixture())


if __name__ == "__main__":
    main()
