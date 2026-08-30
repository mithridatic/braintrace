"""BrainCell compatibility fixtures for the Example 21 replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import braincell
import brainevent
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import optax

import braintrace
from examples.pp_prop.arc_contracts import (
    decode_prediction,
    encode_episode,
    load_task,
    query_exact,
)
from examples.pp_prop.dale_candidates import (
    deferred_biology_defaults,
    make_dale_weight_fn,
    project_dale_raw_weights,
    sparse_dale_matmul,
    validate_deferred_biology,
)

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
TRAINING_TASK_IDS = (
    "d631b094", "dc433765", "b782dc8a", "d06dbe63",
    "aedd82e4", "0b148d64", "b2862040", "150deff5",
)
VALIDATION_TASK_IDS = ("46f33fce", "3428a4f5", "d8c310e9", "09629e4f")


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


def grouped_adam_update(parameters, gradients, states=None, learning_rates=None):
    """Apply the declared input, recurrent, and readout Adam rates."""

    learning_rates = learning_rates or LEARNING_RATES
    states = states or {
        name: AdamState(jnp.zeros_like(value), jnp.zeros_like(value))
        for name, value in parameters.items()
    }
    updated = dict(parameters)
    next_states = dict(states)
    for name, parameter in parameters.items():
        group = next(
            (candidate for candidate in learning_rates if candidate in name),
            None,
        )
        gradient = gradients.get(name)
        if gradient is None and group is not None:
            gradient = gradients.get(group)
        if group is None or gradient is None:
            continue
        state = states.get(name)
        if state is None:
            state = AdamState(jnp.zeros_like(parameter), jnp.zeros_like(parameter))
        updated[name], next_states[name] = adam_update(
            parameter, gradient, state, learning_rates[group]
        )
    return updated, next_states


def grouped_muon_update(parameters, gradients, states=None, learning_rates=None):
    """Apply Muon with AdamW fallback at the declared group rates."""

    learning_rates = learning_rates or LEARNING_RATES
    states = states or {}
    updated = dict(parameters)
    next_states = dict(states)
    for name, parameter in parameters.items():
        group = next(
            (candidate for candidate in learning_rates if candidate in name),
            None,
        )
        gradient = gradients.get(name)
        if gradient is None and group is not None:
            gradient = gradients.get(group)
        if group is None or gradient is None:
            continue
        rate = learning_rates[group]
        transform = optax.contrib.muon(
            learning_rate=rate,
            weight_decay=0.1,
            adam_learning_rate=rate,
            adam_weight_decay=0.1,
        )
        state = states.get(name, transform.init(parameter))
        updates, next_states[name] = transform.update(gradient, state, parameter)
        updated[name] = optax.apply_updates(parameter, updates)
    return updated, next_states


def update_schedule(steps, proof=False):
    """Return the fixed update count for proof or ordinary training."""

    expected = PROOF_UPDATES if proof else ORDINARY_UPDATES
    if steps != expected:
        raise ValueError(f"expected exactly {expected} updates, got {steps}")
    return tuple(range(expected))


def select_integration_substeps(check):
    """Select one event step or the matched two-half-step fallback."""

    return 1 if check["default_selected"] else 2


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


def _csr_indptr(source, rows):
    counts = jnp.bincount(jnp.asarray(source, dtype=jnp.int32), length=rows)
    return jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(counts)))


class BrainCellArcModel(brainstate.nn.Module):
    """The 2,048-neuron sparse BrainCell baseline."""

    def __init__(self, topology=None, biology_options=None):
        super().__init__()
        validate_deferred_biology(**(biology_options or {}))
        if topology is None:
            self.input_csr = input_topology()
            self.recurrent_csr = recurrent_topology()
            input_values = self.input_csr.data
            recurrent_values = self.recurrent_csr.data
            neuron_count = N_NEURONS
            readout = _normal((N_NEURONS, N_READOUT), 23, 1.0 / jnp.sqrt(float(N_NEURONS)))
            dale = jnp.zeros((neuron_count,), dtype=jnp.int8)
            mechanisms = ((),) * neuron_count
        else:
            neuron_count = topology.neuron_count
            recurrent_order = np.argsort(topology.recurrent_source, kind="stable")
            self.input_csr = brainevent.CSR(
                jnp.asarray(topology.input_value),
                jnp.asarray(topology.input_target, dtype=jnp.int32),
                _csr_indptr(topology.input_source, N_INPUTS),
                shape=(N_INPUTS, neuron_count),
            )
            self.recurrent_csr = brainevent.CSR(
                jnp.asarray(topology.recurrent_value)[recurrent_order],
                jnp.asarray(topology.recurrent_target, dtype=jnp.int32)[recurrent_order],
                _csr_indptr(topology.recurrent_source, neuron_count),
                shape=(neuron_count, neuron_count),
            )
            input_values = self.input_csr.data
            recurrent_values = self.recurrent_csr.data
            readout = jnp.asarray(topology.readout)
            dale = jnp.asarray(getattr(topology, "dale", jnp.zeros((neuron_count,), dtype=jnp.int8)))
            mechanisms = tuple(getattr(topology, "mechanisms", ((),) * neuron_count))
        self.input_weight = brainstate.ParamState(input_values)
        self.recurrent_weight = brainstate.ParamState(recurrent_values)
        self.dale = dale
        self.mechanisms = mechanisms
        self.biology_options = deferred_biology_defaults()
        self.readout_weight = brainstate.ParamState(
            readout
        )
        self.readout_bias = brainstate.ParamState(jnp.zeros((N_READOUT,), dtype=jnp.float32))
        self.previous_spikes = brainstate.HiddenState(jnp.zeros((neuron_count,), dtype=jnp.float32))
        self.cell = CompatibilityHodgkinHuxley(neuron_count)
        self.cell.init_state()
        self.recurrent_sources = jnp.repeat(
            jnp.arange(neuron_count), jnp.diff(self.recurrent_csr.indptr)
        )
        self.recurrent_type_signs = jnp.repeat(
            self.dale, jnp.diff(self.recurrent_csr.indptr)
        )
        self.reset_episode()

    def reset_episode(self, learner=None):
        """Reset biological and eligibility state while retaining parameters."""

        self.cell.reset_state()
        self.previous_spikes.value = jnp.zeros((self.cell.V.value.shape[0],), dtype=jnp.float32)
        if learner is not None and hasattr(learner, "reset_state"):
            learner.reset_state()

    def _advance(self, event, dt_ms=0.1, blocked_source=None):
        input_drive = braintrace.sparse_matmul(event, self.input_weight.value, sparse_mat=self.input_csr)
        if blocked_source is None:
            recurrent_drive = sparse_dale_matmul(
                self.previous_spikes.value,
                self.recurrent_weight.value,
                self.recurrent_csr,
                self.recurrent_type_signs,
            )
        else:
            dale_weight_fn = make_dale_weight_fn(self.recurrent_type_signs)

            def recurrent_weight_fn(raw):
                return jnp.where(
                    self.recurrent_sources == blocked_source,
                    0.0,
                    dale_weight_fn(raw),
                )

            recurrent_drive = braintrace.sparse_matmul(
                self.previous_spikes.value,
                self.recurrent_weight.value,
                sparse_mat=self.recurrent_csr,
                weight_fn=recurrent_weight_fn,
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

    def step(self, event, advance=True, blocked_source=None):
        """Run one event, preserving all state for a false advance."""

        return brainstate.transform.cond(
            advance, lambda: self._advance(event, blocked_source=blocked_source),
            lambda: jnp.zeros_like(self.previous_spikes.value),
        )

    def interval(self, event, advance=True, *, substeps=1):
        """Advance one biological interval with one or two compiled substeps."""

        substep_events = integration_substep_events(event, substeps)

        def advancing():
            return brainstate.transform.for_loop(
                lambda subevent: self._advance(subevent, 0.1 / substeps),
                substep_events,
            )[-1]

        return brainstate.transform.cond(
            advance, advancing, lambda: jnp.zeros_like(self.previous_spikes.value)
        )

    def readout(self):
        """Return direct voltage readout logits."""

        feature = jnp.tanh((self.cell.V.value.to_decimal(u.mV) + 65.0) / 20.0)
        return feature @ self.readout_weight.value + self.readout_bias.value


def integration_substep_events(event, substeps):
    """Return one external event followed by zero half-step events."""

    if substeps < 1:
        raise ValueError("substeps must be positive")
    event = jnp.asarray(event)
    return jnp.concatenate(
        (event[None, :], jnp.zeros((substeps - 1,) + event.shape, dtype=event.dtype))
    )


def run_event_sequence(
    model, events, advances=None, *, return_spikes=False, block_source=None
):
    """Run a compiled event sequence and return direct biological state.

    Parameters
    ----------
    model : BrainCellArcModel
        Model whose state is advanced.
    events : array-like
        Event vectors with shape ``(time, 441)``.
    advances : array-like, optional
        Boolean event mask. Missing values mean that every event advances.
    return_spikes : bool, optional
        Return the direct ``model.previous_spikes`` value after each advancing
        event in addition to voltage.
    block_source : int, optional
        Block all recurrent coordinates emitted by one source neuron.

    Returns
    -------
    jax.Array or tuple of jax.Array
        Voltage values with shape ``(time, neurons)``. When ``return_spikes``
        is true, the second array contains direct spike states with the same
        shape and exact zeros for non-advancing padding events.
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
        if not return_spikes:
            return brainstate.transform.for_loop(
                lambda event, advance: model.step(
                    event, advance, blocked_source=block_source
                ), xs, mask
            )

        def step_with_spikes(event, advance):
            voltage = model.step(
                event, advance, blocked_source=block_source
            )
            spikes = jnp.where(
                advance,
                model.previous_spikes.value,
                jnp.zeros_like(model.previous_spikes.value),
            )
            return voltage, spikes

        return brainstate.transform.for_loop(step_with_spikes, xs, mask)

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
        "selected_substeps": select_integration_substeps({
            "default_selected": bool(
                jnp.all(jnp.isfinite(default))
                and jnp.all(jnp.isfinite(matched))
                and voltage_difference <= 1.0
                and spike_equal
                and prediction_equal
                and strict_equal
            )
        }),
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
    """Accumulate one PP-Prop episode and apply one clipped Muon update."""

    def __init__(self, learner, parameters, learning_rates=None):
        self.learner = learner
        parameter_values = dict(parameters)
        model = getattr(learner, "model4compile", None)
        for name in ("readout_weight", "readout_bias"):
            state = getattr(model, name, None)
            if name not in parameter_values and state is not None:
                parameter_values[name] = state.value
        self._parameters_state = brainstate.State(parameter_values)
        self.learning_rates = learning_rates or LEARNING_RATES
        zeros = jax.tree_util.tree_map(jnp.zeros_like, self.parameters)
        self.adam = AdamState(zeros, zeros)
        self._updates_state = brainstate.State(jnp.asarray(0, dtype=jnp.int32))
        self.adam_groups = {
            name: AdamState(jnp.zeros_like(value), jnp.zeros_like(value))
            for name, value in self.parameters.items()
        } if isinstance(self.parameters, dict) else None
        initial_muon_groups = {}
        for name, parameter in self.parameters.items():
            group = next(
                (candidate for candidate in self.learning_rates if candidate in name),
                None,
            )
            if group is None:
                continue
            rate = self.learning_rates[group]
            initial_muon_groups[name] = optax.contrib.muon(
                learning_rate=rate,
                weight_decay=0.1,
                adam_learning_rate=rate,
                adam_weight_decay=0.1,
            ).init(parameter)
        self._muon_groups_state = brainstate.State(initial_muon_groups)

    @property
    def parameters(self):
        """Return optimizer-managed parameter values."""

        return self._parameters_state.value

    @parameters.setter
    def parameters(self, value):
        self._parameters_state.value = value

    @property
    def muon_groups(self):
        """Return optimizer state carried by BrainState transforms."""

        return self._muon_groups_state.value

    @muon_groups.setter
    def muon_groups(self, value):
        self._muon_groups_state.value = value

    @property
    def updates(self):
        """Return the transformed optimizer update count."""

        return self._updates_state.value

    @updates.setter
    def updates(self, value):
        self._updates_state.value = value

    def reset_episode(self, model):
        """Reset model and eligibility state before the next query episode."""

        model.reset_episode(self.learner)

    def _group_gradients(self, gradients):
        """Map compiled parameter paths to the declared optimizer groups."""

        if not self.adam_groups or set(gradients).issubset(self.adam_groups):
            return gradients
        grouped = {}
        for path, gradient in gradients.items():
            parts = path if isinstance(path, tuple) else (path,)
            name = next(
                (group for group in self.adam_groups if any(group in str(part) for part in parts)),
                None,
            )
            if name is not None:
                grouped[name] = gradient
        return grouped

    def _sync_compiled_parameters(self):
        """Write grouped optimizer values back to matching compiled states."""

        states = getattr(self.learner, "param_states", {})
        for path, state in states.items():
            parts = path if isinstance(path, tuple) else (path,)
            name = next(
                (group for group in self.adam_groups or {} if any(group in str(part) for part in parts)),
                None,
            )
            if name in self.parameters:
                state.value = self.parameters[name]
        model = getattr(self.learner, "model4compile", None)
        for name in ("readout_weight", "readout_bias"):
            state = getattr(model, name, None)
            if state is not None and name in self.parameters:
                state.value = self.parameters[name]

    def _project_dale_parameters(self):
        """Project typed recurrent raw values to the declared effective floor."""
        model = getattr(self.learner, "model4compile", None)
        signs = getattr(model, "recurrent_type_signs", None)
        if signs is not None and "recurrent" in self.parameters:
            self.parameters["recurrent"] = project_dale_raw_weights(
                self.parameters["recurrent"], signs
            )

    def optimizer_is_finite(self):
        """Return whether parameters, moments, and step state are finite."""

        if self.adam_groups is None:
            moments = (self.adam.first, self.adam.second)
        else:
            moments = tuple(
                (state.first, state.second) for state in self.adam_groups.values()
            )
        leaves = jax.tree_util.tree_leaves((self.parameters, moments, self.muon_groups))
        return bool(all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves))

    def update_episode(
        self, events, step_fn, valid_rows=None, loss_mask=None, direct_grad_fn=None
    ):
        """Apply one update from one masked PP-Prop query episode."""

        if loss_mask is not None and valid_rows is not None:
            raise ValueError("provide only one episode loss mask")
        mask = loss_mask if loss_mask is not None else valid_rows
        gradients, losses = self.learner.etrace_grad(
            events,
            step_fn=step_fn,
            mask=mask,
            reduction="sum",
            loss_output="masked",
            return_value=True,
        )
        if direct_grad_fn is not None:
            direct_gradients = direct_grad_fn(
                events=events, step_fn=step_fn, mask=mask
            )
            gradients = {**gradients, **direct_gradients}
        losses = jnp.sum(losses)
        gradients, norm = clip_gradient(gradients)
        gradients = self._group_gradients(gradients)
        if self.adam_groups is not None:
            self.parameters, self.muon_groups = grouped_muon_update(
                self.parameters, gradients, self.muon_groups, self.learning_rates
            )
            self._project_dale_parameters()
            self._sync_compiled_parameters()
        else:
            rate = self.learning_rates.get("input", 0.001)
            self.parameters, self.adam = adam_update(
                self.parameters, gradients, self.adam, rate
            )
            self._project_dale_parameters()
        self.updates += 1
        return losses, norm

    def evaluate_forward(self, forward_fn, *args, **kwargs):
        """Evaluate an episode without changing parameters or learner state."""

        before = jax.tree_util.tree_map(jnp.array, self.parameters)
        state_values = self._non_parameter_state_values()
        result = forward_fn(*args, **kwargs)
        after = jax.tree_util.tree_map(jnp.array, self.parameters)
        if not bool(jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, before, after)
        )):
            raise RuntimeError("forward validation changed trainable parameters")
        current_states = self._non_parameter_state_values()
        if len(state_values) != len(current_states) or any(
            not bool(jnp.array_equal(initial, current))
            for initial, current in zip(state_values, current_states)
        ):
            raise RuntimeError("forward validation changed biological or eligibility state")
        return result

    def _non_parameter_state_values(self):
        """Snapshot learner states that validation must preserve."""

        containers = []
        states = getattr(self.learner, "states", None)
        if callable(states):
            containers.append(states())
        for name in ("hidden_states", "other_states"):
            value = getattr(self.learner, name, None)
            if value is not None:
                containers.append(value)
        executor = getattr(self.learner, "graph_executor", None)
        if executor is not None:
            containers.append(executor.states)
        values = []
        for container in containers:
            for state in container.values():
                if not isinstance(state, brainstate.ParamState):
                    values.append(jnp.array(state.value))
        running_index = getattr(self.learner, "running_index", None)
        if running_index is not None:
            values.append(jnp.array(running_index.value))
        return values


def run_fixed_schedule(trainer, episodes, *, proof=False):
    """Run the exact proof or ordinary number of ordered episodes."""

    update_schedule(len(episodes), proof=proof)
    task_ids = [episode.get("task_id") for episode in episodes]
    if any(task_id is None for task_id in task_ids):
        raise ValueError("every counted episode must declare task_id")
    if proof:
        if any(task_id != "d631b094" for task_id in task_ids):
            raise ValueError("proof schedule accepts only d631b094")
    else:
        expected = tuple(
            TRAINING_TASK_IDS[index % len(TRAINING_TASK_IDS)]
            for index in range(len(episodes))
        )
        if tuple(task_ids) != expected:
            raise ValueError("ordinary schedule task order is not fixed")
    if any(
        episode.get("validation", False) or task_id in VALIDATION_TASK_IDS
        for episode, task_id in zip(episodes, task_ids)
    ):
        raise ValueError("validation episodes are forward-only")
    static_payload = {
        key: episodes[0][key]
        for key in ("step_fn", "direct_grad_fn")
        if key in episodes[0]
    }
    payloads = [{
            key: value for key, value in episode.items()
            if key not in {"task_id", "validation", "target", "query_index", "step_fn", "direct_grad_fn"}
    } for episode in episodes]
    stacked = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values), *payloads
    )

    def update(episode):
        payload = {**episode, **static_payload}
        return trainer.update_episode(**payload)

    return brainstate.transform.for_loop(update, stacked)


def _supervised_episodes(data_root, task_ids):
    """Load one supervised direct ARC episode for each declared task."""

    episodes = []
    for task_id in task_ids:
        task = load_task(data_root, task_id, "practice")
        query_index = next(
            (index for index, target in enumerate(task.targets) if target is not None),
            None,
        )
        if query_index is None:
            raise ValueError(f"task {task_id} has no supervised query")
        events, advances = encode_episode(task, query_index)
        episodes.append({
            "task_id": task_id,
            "events": jnp.asarray(events, dtype=jnp.float32),
            "loss_mask": jnp.asarray(advances, dtype=bool),
            "target": task.targets[query_index],
            "query_index": query_index,
        })
    return episodes


def _screen_predictions(model, learner, episodes):
    """Run direct request decoding for loaded ARC episodes."""

    event_values = jnp.stack([episode["events"] for episode in episodes])
    advance_values = jnp.stack([episode["loss_mask"] for episode in episodes])

    def evaluate(events, advances):
        model.reset_episode(learner)
        voltages = run_event_sequence(model, events, advances)
        features = jnp.tanh((voltages[-31:] + 65.0) / 20.0)
        return features @ model.readout_weight.value + model.readout_bias.value

    logits = brainstate.transform.for_loop(evaluate, event_values, advance_values)
    records = []
    for episode, output in zip(episodes, np.asarray(logits)):
        prediction = decode_prediction(np.asarray(output))
        target = episode["target"]
        records.append({
            "task_id": episode["task_id"],
            "query_index": episode["query_index"],
            "prediction": prediction.tolist(),
            "target": np.asarray(target).tolist(),
            "exact": query_exact(prediction, target),
        })
    return records


def _real_workflow_report(data_root, *, proof):
    """Execute a real-data BrainCell PP-Prop proof or ordinary run."""

    if data_root is None:
        raise ValueError("real-data proof and run require --arc-root")
    task_ids = ("d631b094",) if proof else TRAINING_TASK_IDS
    validation_task_ids = ("46f33fce",) if proof else VALIDATION_TASK_IDS
    training_episodes = _supervised_episodes(data_root, task_ids)
    validation_episodes = _supervised_episodes(data_root, validation_task_ids)
    model = BrainCellArcModel()
    learner = compile_pp_prop_model(model)
    trainer = PPPropEpisodeTrainer(
        learner,
        {"input": model.input_weight.value, "recurrent": model.recurrent_weight.value},
    )
    if proof:
        scheduled = [training_episodes[0]] * PROOF_UPDATES
    else:
        scheduled = [training_episodes[index % len(training_episodes)] for index in range(ORDINARY_UPDATES)]
    step_fn = lambda event: jnp.sum(
        learner.etrace_evolve(event[None, :], return_outputs=True)[0]
    )
    scheduled = [{**episode, "step_fn": step_fn} for episode in scheduled]

    screened_training = training_episodes[:1] if proof else training_episodes
    before = _screen_predictions(model, learner, screened_training)
    recurrent_before = np.asarray(model.recurrent_weight.value).copy()
    run_fixed_schedule(trainer, scheduled, proof=proof)
    recurrent_after = np.asarray(model.recurrent_weight.value)
    after = _screen_predictions(model, learner, screened_training)

    parameter_snapshot = jax.tree_util.tree_map(jnp.array, trainer.parameters)
    validation = _screen_predictions(model, learner, validation_episodes)
    validation_isolated = bool(jax.tree_util.tree_all(jax.tree_util.tree_map(
        jnp.array_equal, parameter_snapshot, trainer.parameters
    )))
    prediction_changed = any(
        not np.array_equal(old["prediction"], new["prediction"])
        for old, new in zip(before, after)
    )
    recurrent_changed = not np.array_equal(recurrent_before, recurrent_after)
    optimizer_finite = trainer.optimizer_is_finite()
    training_strict_count = sum(record["exact"] for record in after)
    validation_strict_count = sum(record["exact"] for record in validation)
    passed = bool(
        trainer.updates == (PROOF_UPDATES if proof else ORDINARY_UPDATES)
        and optimizer_finite
        and recurrent_changed
        and prediction_changed
        and validation_isolated
    ) if proof else bool(
        trainer.updates == ORDINARY_UPDATES
        and optimizer_finite
        and validation_isolated
    )
    return {
        "mode": "proof" if proof else "run",
        "passed": passed,
        "training_task_ids": list(task_ids),
        "validation_task_ids": list(validation_task_ids),
        "updates": int(trainer.updates),
        "optimizer_finite": optimizer_finite,
        "recurrent_weight_changed": recurrent_changed,
        "prediction_changed": prediction_changed,
        "validation_parameter_state_unchanged": validation_isolated,
        "training_strict_pass_at_1_count": int(training_strict_count),
        "validation_strict_pass_at_1_count": int(validation_strict_count),
        "training_before": before,
        "training_after": after,
        "validation": validation,
    }
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


def _parser():
    parser = argparse.ArgumentParser(
        description="Run the BrainCell Example 21 compatibility checks."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("proof", "run"),
        help="run the real-data proof or fixed ordinary schedule",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the bounded CPU/GPU compatibility smoke checks",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="device for model execution (default: cpu)",
    )
    parser.add_argument(
        "--arc-root",
        type=Path,
        default=Path("/datasets/arc/raw"),
        help="raw ARC root for proof and run (default: /datasets/arc/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write the JSON report into this directory",
    )
    return parser


def _smoke_report():
    finite_difference = finite_difference_fixture()
    spike_path = spike_path_fixture()
    readout = direct_readout_gradient_fixture()
    passed = (
        finite_difference["absolute_error"] <= finite_difference["tolerance"]
        and all(spike_path.values())
        and all(readout.values())
    )
    return {
        "mode": "smoke",
        "passed": bool(passed),
        "finite_difference": finite_difference,
        "spike_path": spike_path,
        "direct_readout": readout,
    }


def _write_report(output_dir, report):
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"example21-{report['mode']}.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_command(args):
    if args.command is not None and args.smoke:
        raise ValueError("choose one of proof, run, or --smoke")
    if args.command is None and not args.smoke:
        _parser().print_help()
        return 0
    try:
        device = jax.devices(args.device)[0]
    except RuntimeError as error:
        raise ValueError(f"requested device {args.device!r} is unavailable") from error
    with jax.default_device(device):
        if args.command == "proof":
            report = _real_workflow_report(args.arc_root, proof=True)
        elif args.command == "run":
            report = _real_workflow_report(args.arc_root, proof=False)
        else:
            report = _smoke_report()
    _write_report(args.output_dir, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


def main(argv=None) -> int:
    """Run the BrainCell Example 21 command line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. ``None`` reads the process command line.

    Returns
    -------
    int
        Zero when the selected check passes.
    """

    args = _parser().parse_args(argv)
    try:
        return _run_command(args)
    except ValueError as error:
        _parser().error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
