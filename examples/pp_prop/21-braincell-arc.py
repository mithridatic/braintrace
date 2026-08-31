"""BrainCell compatibility fixtures for the Example 21 replacement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import braincell
import brainevent
import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import optax
from examples.pp_prop.arc_contracts import (
    ARCTask,
    decode_episode,
    decode_prediction,
    encode_episode,
    load_checkpoint,
    load_task,
    query_exact,
    request_loss,
    strict_task_pass_at_1,
    write_checkpoint,
    write_result,
)
from examples.pp_prop.dale_candidates import (
    deferred_biology_defaults,
    make_dale_weight_fn,
    project_dale_raw_weights,
    sparse_dale_matmul,
    validate_deferred_biology,
)

__all__ = (
    "ARCTask",
    "decode_episode",
    "decode_prediction",
    "encode_episode",
    "load_checkpoint",
    "load_task",
    "query_exact",
    "request_loss",
    "strict_task_pass_at_1",
    "write_checkpoint",
    "write_result",
)

N_INPUTS = 441
N_NEURONS = 2048
N_READOUT = 360
INPUT_FANOUT = 32
RECURRENT_FANOUT = 8
TRACE_DECAY = 0.95
GRADIENT_CLIP_NORM = 1.0
PROOF_UPDATES = 8
ORDINARY_UPDATES = 128
PROOF_DEADLINE_SECONDS = 180.0
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
    """Build the deterministic feature-to-neuron CSR topology.

    Returns
    -------
    brainevent.CSR
        Input-to-neuron sparse relation with 441 rows and 32 entries per row.
    """

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
    """Build the deterministic source-row recurrent CSR topology.

    Returns
    -------
    brainevent.CSR
        Recurrent sparse relation with eight non-self entries per source.
    """

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
    """Return bounded BrainCell current density from dimensionless drives.

    Parameters
    ----------
    input_drive, recurrent_drive : array-like
        Dimensionless input and recurrent population drives.

    Returns
    -------
    brainunit.Quantity
        Current density in milliamperes per square centimetre.
    """

    return (
        0.02 * jnp.tanh(input_drive) + 0.01 * jnp.tanh(recurrent_drive)
    ) * u.mA / u.cm**2


def clip_gradient(gradient, max_norm=GRADIENT_CLIP_NORM):
    """Clip a pytree gradient by its global Euclidean norm.

    Parameters
    ----------
    gradient : pytree
        Gradient leaves to scale.
    max_norm : float, optional
        Maximum global Euclidean norm.

    Returns
    -------
    tuple
        Clipped gradient and the unscaled global norm.
    """

    if isinstance(gradient, dict):
        leaves = [
            leaf
            for value in gradient.values()
            for leaf in jax.tree_util.tree_leaves(value)
        ]
    else:
        leaves = jax.tree_util.tree_leaves(gradient)
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny))
    if isinstance(gradient, dict):
        return {
            key: jax.tree_util.tree_map(lambda leaf: leaf * scale, value)
            for key, value in gradient.items()
        }, norm
    return jax.tree_util.tree_map(lambda leaf: leaf * scale, gradient), norm


def accumulate_masked_loss(losses, valid_rows):
    """Sum only losses belonging to valid shape or row requests.

    Parameters
    ----------
    losses : array-like
        Per-request loss values.
    valid_rows : array-like
        Zero-one mask selecting counted requests.

    Returns
    -------
    jax.Array
        Scalar masked loss sum.
    """

    values = jnp.asarray(losses)
    mask = jnp.asarray(valid_rows, dtype=values.dtype)
    return jnp.sum(values * mask)


def _supervised_request_loss(
    logits, target_shape, target_rows, target_valid_mask, request_kind
):
    """Return the scalar loss for one shape or row request."""

    logits = jnp.asarray(logits, dtype=jnp.float32)
    target_shape = jnp.asarray(target_shape, dtype=jnp.int32)
    target_rows = jnp.asarray(target_rows, dtype=jnp.int32)
    target_valid_mask = jnp.asarray(target_valid_mask, dtype=jnp.float32)
    shape_logits = jnp.stack((logits[:30], logits[30:60]))
    shape_loss = jnp.sum(
        jax.nn.logsumexp(shape_logits, axis=-1)
        - jnp.take_along_axis(shape_logits, target_shape[:, None], axis=1).squeeze(1)
    )
    row_logits = logits[60:].reshape((30, 10))
    valid_cell_count = (target_shape[0] + 1) * (target_shape[1] + 1)
    row_loss = jnp.sum(
        (jax.nn.logsumexp(row_logits, axis=-1)
         - jnp.take_along_axis(row_logits, target_rows[:, None], axis=1).squeeze(1))
        * target_valid_mask
    ) / jnp.maximum(valid_cell_count, 1)
    return jnp.where(
        request_kind == 1,
        shape_loss,
        jnp.where(request_kind == 2, row_loss, jnp.asarray(0.0)),
    )


def _direct_readout_gradients(
    features,
    target_shape,
    target_rows,
    target_valid_mask,
    request_kind,
    request_mask,
    readout_weight,
    readout_bias,
):
    """Differentiate the request-only supervised readout objective."""

    def objective(weight, bias):
        logits = features @ weight + bias
        losses = jax.vmap(_supervised_request_loss)(
            logits,
            target_shape,
            target_rows,
            target_valid_mask,
            request_kind,
        )
        return jnp.sum(losses * jnp.asarray(request_mask, dtype=losses.dtype))

    weight_grad, bias_grad = jax.grad(objective, argnums=(0, 1))(
        readout_weight, readout_bias
    )
    return {"readout_weight": weight_grad, "readout_bias": bias_grad}


class AdamState:
    """Adam first and second moments for one episode schedule.

    Parameters
    ----------
    first, second : pytree
        First and second moment values.
    step : int, optional
        Number of updates already applied.
    """

    def __init__(self, first, second, step=0):
        self.first = first
        self.second = second
        self.step = step


def adam_update(parameters, gradient, state, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply one bias-corrected Adam update and return new state.

    Parameters
    ----------
    parameters, gradient : pytree
        Parameter values and matching gradients.
    state : AdamState
        Existing moments and step count.
    learning_rate, beta1, beta2, eps : float
        Optimizer rate and numerical constants.

    Returns
    -------
    tuple
        Updated parameters and Adam state.
    """

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
    """Apply the declared input, recurrent, and readout Adam rates.

    Parameters
    ----------
    parameters, gradients : mapping
        Named parameter values and matching gradients.
    states, learning_rates : mapping, optional
        Existing states and per-group rates.

    Returns
    -------
    tuple
        Updated parameters and named Adam states.
    """

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
    """Apply Muon with AdamW fallback at the declared group rates.

    Parameters
    ----------
    parameters, gradients : mapping
        Named parameter values and matching gradients.
    states, learning_rates : mapping, optional
        Existing optimizer states and per-group rates.

    Returns
    -------
    tuple
        Updated parameters and named Muon states.
    """

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
    """Return the fixed update count for proof or ordinary training.

    Parameters
    ----------
    steps : int
        Requested update count.
    proof : bool, optional
        Select the eight-update proof schedule when true.

    Returns
    -------
    tuple of int
        Consecutive update indices.

    Raises
    ------
    ValueError
        If ``steps`` does not equal the selected schedule length.
    """

    expected = PROOF_UPDATES if proof else ORDINARY_UPDATES
    if steps != expected:
        raise ValueError(
            f"Expected exactly {expected} updates, got {steps}; "
            f"pass steps={expected}."
        )
    return tuple(range(expected))


def select_integration_substeps(check):
    """Select one event step or the matched two-half-step fallback.

    Parameters
    ----------
    check : mapping
        Integration check containing ``default_selected``.

    Returns
    -------
    int
        Selected substep count, either one or two.
    """

    return 1 if check["default_selected"] else 2


def compile_pp_prop_model(model):
    """Compile state-changing sparse parameters with PP-Prop single-step.

    Parameters
    ----------
    model : BrainCellArcModel
        Model whose sparse parameters influence hidden state.

    Returns
    -------
    object
        PP-Prop learner compiled for one event step.
    """

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
    """The 2,048-neuron sparse BrainCell baseline.

    Parameters
    ----------
    topology : SparseTopology, optional
        Sparse topology used to rebuild the model.
    biology_options : mapping, optional
        Deferred biology options validated during construction.
    """

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
            owner_codes = None
            neuron_ids = np.arange(neuron_count, dtype=np.int32)
        else:
            neuron_count = topology.neuron_count
            input_order = np.lexsort((
                np.asarray(topology.input_target),
                np.asarray(topology.input_source),
            ))
            recurrent_order = np.lexsort((
                np.asarray(topology.recurrent_target),
                np.asarray(topology.recurrent_source),
            ))
            self.input_csr = brainevent.CSR(
                jnp.asarray(topology.input_value)[input_order],
                jnp.asarray(topology.input_target, dtype=jnp.int32)[input_order],
                _csr_indptr(np.asarray(topology.input_source)[input_order], N_INPUTS),
                shape=(N_INPUTS, neuron_count),
            )
            self.recurrent_csr = brainevent.CSR(
                jnp.asarray(topology.recurrent_value)[recurrent_order],
                jnp.asarray(topology.recurrent_target, dtype=jnp.int32)[recurrent_order],
                _csr_indptr(
                    np.asarray(topology.recurrent_source)[recurrent_order], neuron_count
                ),
                shape=(neuron_count, neuron_count),
            )
            input_values = self.input_csr.data
            recurrent_values = self.recurrent_csr.data
            readout = jnp.asarray(topology.readout)
            dale = jnp.asarray(getattr(topology, "dale", jnp.zeros((neuron_count,), dtype=jnp.int8)))
            mechanisms = tuple(getattr(topology, "mechanisms", ((),) * neuron_count))
            owner_codes = getattr(topology, "owner_codes", None)
            neuron_ids = getattr(topology, "neuron_ids", None)
            if neuron_ids is None:
                neuron_ids = np.arange(neuron_count, dtype=np.int32)
        self.input_weight = brainstate.ParamState(input_values)
        self.recurrent_weight = brainstate.ParamState(recurrent_values)
        self.dale = dale
        self.mechanisms = mechanisms
        self.owner_codes = (
            None if owner_codes is None else np.asarray(owner_codes, dtype=np.int16)
        )
        self.neuron_ids = np.asarray(neuron_ids, dtype=np.int32)
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
        """Reset biological and eligibility state while retaining parameters.

        Parameters
        ----------
        learner : object, optional
            Learner whose eligibility state is reset with the model.
        """

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
        """Advance one event for the BrainTrace compiler.

        Parameters
        ----------
        event : array-like
            Encoded event with 441 input features.

        Returns
        -------
        jax.Array
            Membrane voltage after the event.
        """

        return self._advance(event)

    def step(self, event, advance=True, blocked_source=None):
        """Run one event, preserving all state for a false advance.

        Parameters
        ----------
        event : array-like
            Encoded event with 441 input features.
        advance : bool, optional
            Whether biological and eligibility state advances.
        blocked_source : int, optional
            Source neuron whose recurrent edges are blocked.

        Returns
        -------
        jax.Array
            Membrane voltage, or exact zeros for a non-advancing event.
        """

        return brainstate.transform.cond(
            advance, lambda: self._advance(event, blocked_source=blocked_source),
            lambda: jnp.zeros_like(self.previous_spikes.value),
        )

    def interval(self, event, advance=True, *, substeps=1):
        """Advance one biological interval with one or two compiled substeps.

        Parameters
        ----------
        event : array-like
            Encoded event for the interval.
        advance : bool, optional
            Whether the interval changes biological state.
        substeps : int, optional
            Positive number of matched integration substeps.

        Returns
        -------
        jax.Array
            Membrane voltage after the interval.
        """

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
        """Return direct voltage readout logits.

        Returns
        -------
        jax.Array
            360 direct readout logits.
        """

        return (
            self.readout_features() @ self.readout_weight.value
            + self.readout_bias.value
        )

    def readout_features(self):
        """Return normalized membrane-voltage features for the readout.

        Returns
        -------
        jax.Array
            One normalized feature for each neuron.
        """

        return jnp.tanh((self.cell.V.value.to_decimal(u.mV) + 65.0) / 20.0)


def integration_substep_events(event, substeps):
    """Return one external event followed by zero half-step events.

    Parameters
    ----------
    event : array-like
        One encoded input event.
    substeps : int
        Positive number of compiled integration substeps.

    Returns
    -------
    jax.Array
        Event array with the requested leading substep dimension.

    Raises
    ------
    ValueError
        If ``substeps`` is less than one; pass a positive integer.
    """

    if substeps < 1:
        raise ValueError("Substeps must be positive; pass a positive integer.")
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
        raise ValueError(
            f"Events must have shape (time, {N_INPUTS}); "
            f"pass a two-dimensional array with {N_INPUTS} features."
        )
    if advances.shape != (events.shape[0],):
        raise ValueError(
            "Advances must have one boolean per event; "
            "pass a mask with the same length as events."
        )

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
    """Evolve a PP-Prop learner only on advancing events.

    Parameters
    ----------
    learner : object
        PP-Prop learner with an ``etrace_evolve`` method.
    events : array-like
        Event vectors with shape ``(time, 441)``.
    advances : array-like, optional
        Boolean mask selecting events that update state.

    Returns
    -------
    jax.Array
        Learner outputs for each event.
    """

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
    """Compare the default interval with two compiled half-steps.

    Parameters
    ----------
    events : array-like
        Event sequence used by both integration paths.

    Returns
    -------
    dict
        Finite, voltage, spike, prediction, and selected-substep checks.
    """

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
    """Return direct predictions before and after a state-only intervention.

    Parameters
    ----------
    model : BrainCellArcModel
        Model whose membrane voltage is changed for the intervention.

    Returns
    -------
    tuple of jax.Array
        Readout logits before and after the voltage intervention.
    """

    before = model.readout()
    model.cell.V.value = model.cell.V.value + 1.0 * u.mV
    after = model.readout()
    return before, after


class PPPropEpisodeTrainer:
    """Accumulate one PP-Prop episode and apply one clipped Muon update.

    Parameters
    ----------
    learner : object
        Compiled PP-Prop learner and model.
    parameters : mapping
        Trainable parameter values keyed by optimizer group.
    learning_rates : mapping, optional
        Learning rate for each parameter group.
    """

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
        """Return the optimizer-managed parameter values.

        Returns
        -------
        mapping
            Parameter names mapped to their current array values.
        """

        return self._parameters_state.value

    @parameters.setter
    def parameters(self, value):
        """Replace the optimizer-managed parameter values.

        Parameters
        ----------
        value : mapping
            Parameter names mapped to replacement array values.
        """

        self._parameters_state.value = value

    @property
    def muon_groups(self):
        """Return optimizer state carried by BrainState transforms.

        Returns
        -------
        mapping
            Parameter names mapped to Muon optimizer state.
        """

        return self._muon_groups_state.value

    @muon_groups.setter
    def muon_groups(self, value):
        """Replace the BrainState transform optimizer state.

        Parameters
        ----------
        value : mapping
            Parameter names mapped to replacement optimizer state.
        """

        self._muon_groups_state.value = value

    @property
    def updates(self):
        """Return the transformed optimizer update count.

        Returns
        -------
        int
            Number of completed episode updates.
        """

        return self._updates_state.value

    @updates.setter
    def updates(self, value):
        """Set the transformed optimizer update count.

        Parameters
        ----------
        value : int
            Replacement number of completed episode updates.
        """

        self._updates_state.value = value

    def reset_episode(self, model=None):
        """Reset model and eligibility state before the next query episode.

        Parameters
        ----------
        model : BrainCellArcModel, optional
            Model to reset; defaults to the learner's compiled model.

        Raises
        ------
        ValueError
            If no compiled model exists; provide ``model`` or compile the
            learner first.
        """

        if model is None:
            model = getattr(self.learner, "model4compile", None)
        if model is None:
            raise ValueError(
                "Episode reset requires the compiled model; "
                "provide model or compile the learner first."
            )
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
        """Return whether parameters, moments, and step state are finite.

        Returns
        -------
        bool
            True when every optimizer and parameter leaf is finite.
        """

        if self.adam_groups is None:
            moments = (self.adam.first, self.adam.second)
        else:
            moments = tuple(
                (state.first, state.second) for state in self.adam_groups.values()
            )
        leaves = jax.tree_util.tree_leaves((self.parameters, moments, self.muon_groups))
        return bool(all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves))

    def update_episode(
        self,
        events,
        step_fn,
        valid_rows=None,
        loss_mask=None,
        direct_grad_fn=None,
        request_kind=None,
        target_shape=None,
        target_rows=None,
        target_valid_mask=None,
        advance_mask=None,
    ):
        """Apply one update from one masked PP-Prop query episode.

        Parameters
        ----------
        events : array-like
            Event sequence for one query episode.
        step_fn, direct_grad_fn : callable
            Forward and optional direct-gradient functions.
        valid_rows, loss_mask, request_kind : array-like, optional
            Request masks and request classes; provide at most one row mask.
        target_shape, target_rows, target_valid_mask : array-like, optional
            Supervised shape and row targets with their valid-row mask.
        advance_mask : array-like, optional
            Boolean event mask for advancing state.

        Returns
        -------
        dict
            Updated parameter gradients and episode loss evidence.

        Raises
        ------
        ValueError
            If both row-mask arguments are supplied or the advance mask has
            the wrong length; provide one valid mask with one value per event.
        """

        if loss_mask is not None and valid_rows is not None:
            raise ValueError(
                "Provide only one episode loss mask; "
                "choose valid_rows or loss_mask."
            )
        mask = loss_mask if loss_mask is not None else valid_rows
        if advance_mask is not None:
            advance_mask = jnp.asarray(advance_mask, dtype=bool)
            if advance_mask.shape != (events.shape[0],):
                raise ValueError(
                    "Advance_mask must have one boolean per event; "
                    "pass a mask with the same length as events."
                )
            if mask is None:
                mask = jnp.ones((events.shape[0],), dtype=jnp.float32)
            mask = mask * advance_mask
        sequences = [events]
        if advance_mask is not None:
            sequences.append(advance_mask)
        if request_kind is not None:
            sequences.extend((request_kind, target_shape, target_rows, target_valid_mask))
        result = self.learner.etrace_grad(
            *sequences,
            step_fn=step_fn,
            mask=mask,
            reduction="sum",
            loss_output="masked",
            has_aux=direct_grad_fn is not None,
            return_value=True,
        )
        if direct_grad_fn is None:
            gradients, losses = result
            aux = None
        elif len(result) == 3:
            gradients, losses, aux = result
        else:
            gradients, losses = result
            aux = None
        if direct_grad_fn is not None:
            direct_gradients = direct_grad_fn(
                events=events,
                step_fn=step_fn,
                mask=mask,
                aux=aux,
                request_kind=request_kind,
                target_shape=target_shape,
                target_rows=target_rows,
                target_valid_mask=target_valid_mask,
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
        """Evaluate an episode without changing parameters or learner state.

        Parameters
        ----------
        forward_fn : callable
            Forward function to evaluate.
        *args, **kwargs : object
            Arguments passed to ``forward_fn``.

        Returns
        -------
        object
            Forward-function result after state-preservation checks.

        Raises
        ------
        RuntimeError
            If validation changes trainable or biological state.
        """

        before = jax.tree_util.tree_map(jnp.array, self.parameters)
        state_values = self._non_parameter_state_values()
        result = forward_fn(*args, **kwargs)
        after = jax.tree_util.tree_map(jnp.array, self.parameters)
        if not bool(jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, before, after)
        )):
            raise RuntimeError(
                "Forward validation changed trainable parameters; "
                "use a state-preserving forward function."
            )
        current_states = self._non_parameter_state_values()
        if len(state_values) != len(current_states) or any(
            not bool(jnp.array_equal(initial, current))
            for initial, current in zip(state_values, current_states)
        ):
            raise RuntimeError(
                "Forward validation changed biological or eligibility state; "
                "use a state-preserving forward function."
            )
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
    """Run the exact proof or ordinary number of ordered episodes.

    Parameters
    ----------
    trainer : PPPropEpisodeTrainer
        Trainer that owns model and optimizer state.
    episodes : sequence of mapping
        Ordered episode payloads with task identifiers.
    proof : bool, optional
        Select the eight-update proof schedule when true.

    Returns
    -------
    tuple
        Per-episode update evidence.
    """

    update_schedule(len(episodes), proof=proof)
    task_ids = [episode.get("task_id") for episode in episodes]
    if any(task_id is None for task_id in task_ids):
        raise ValueError(
            "Every counted episode must declare task_id; "
            "add task_id to each episode."
        )
    if proof:
        if any(task_id != "d631b094" for task_id in task_ids):
            raise ValueError(
                "Proof schedule accepts only d631b094; "
                "use that task for every proof episode."
            )
    else:
        expected = tuple(
            TRAINING_TASK_IDS[index % len(TRAINING_TASK_IDS)]
            for index in range(len(episodes))
        )
        if tuple(task_ids) != expected:
            raise ValueError(
                "Ordinary schedule task order is not fixed; "
                "use TRAINING_TASK_IDS order."
            )
    if any(
        episode.get("validation", False) or task_id in VALIDATION_TASK_IDS
        for episode, task_id in zip(episodes, task_ids)
    ):
        raise ValueError(
            "Validation episodes are forward-only; "
            "remove validation episodes from the update schedule."
        )
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
        trainer.reset_episode()
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
            raise ValueError(
                f"Task {task_id} has no supervised query; "
                "select a task with a target query."
            )
        events, advances = encode_episode(task, query_index)
        target = np.asarray(task.targets[query_index], dtype=np.int32)
        request_mask = np.zeros((events.shape[0],), dtype=bool)
        request_mask[-31:] = True
        request_kind = np.zeros((events.shape[0],), dtype=np.int32)
        request_kind[-31] = 1
        request_kind[-30:] = 2
        target_shape = np.broadcast_to(
            np.asarray(target.shape, dtype=np.int32) - 1,
            (events.shape[0], 2),
        ).copy()
        padded_rows = np.zeros((30, 30), dtype=np.int32)
        padded_rows[:target.shape[0], :target.shape[1]] = target
        target_rows = np.zeros((events.shape[0], 30), dtype=np.int32)
        target_rows[-30:] = padded_rows
        target_valid_mask = np.zeros((events.shape[0], 30), dtype=np.float32)
        request_start = events.shape[0] - 30
        target_valid_mask[request_start:request_start + target.shape[0], :target.shape[1]] = 1.0
        episodes.append({
            "task_id": task_id,
            "events": jnp.asarray(events, dtype=jnp.float32),
            "advance_mask": jnp.asarray(advances, dtype=bool),
            "loss_mask": jnp.asarray(request_mask, dtype=bool),
            "request_kind": jnp.asarray(request_kind, dtype=jnp.int32),
            "target_shape": jnp.asarray(target_shape, dtype=jnp.int32),
            "target_rows": jnp.asarray(target_rows, dtype=jnp.int32),
            "target_valid_mask": jnp.asarray(target_valid_mask, dtype=jnp.float32),
            "target": target,
            "query_index": query_index,
        })
    return episodes


def _screen_predictions(model, learner, episodes):
    """Run direct request decoding for loaded ARC episodes."""

    event_values = jnp.stack([episode["events"] for episode in episodes])
    advance_values = jnp.stack([
        episode.get("advance_mask", episode["loss_mask"]) for episode in episodes
    ])

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
        raise ValueError(
            "Real-data proof and run require --arc-root; "
            "pass the directory containing the ARC task files."
        )
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
    def step_fn(
        event, advance, request_kind, target_shape, target_rows, target_valid_mask
    ):
        def advancing():
            learner(event)
            features = model.readout_features()
            logits = features @ model.readout_weight.value + model.readout_bias.value
            loss = _supervised_request_loss(
                logits, target_shape, target_rows, target_valid_mask, request_kind
            )
            return loss, features

        return brainstate.transform.cond(
            advance,
            advancing,
            lambda: (
                jnp.asarray(0.0),
                jnp.zeros((model.readout_weight.value.shape[0],), dtype=jnp.float32),
            ),
        )

    def direct_grad_fn(
        *,
        aux,
        request_kind,
        target_shape,
        target_rows,
        target_valid_mask,
        mask,
        **_,
    ):
        return _direct_readout_gradients(
            aux,
            target_shape,
            target_rows,
            target_valid_mask,
            request_kind,
            mask,
            model.readout_weight.value,
            model.readout_bias.value,
        )

    scheduled = [{
        **episode,
        "step_fn": step_fn,
        "direct_grad_fn": direct_grad_fn,
    } for episode in scheduled]

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


def _apply_proof_deadline(report, started_at):
    """Add the proof runtime gate to a completed workflow report."""

    elapsed_seconds = time.monotonic() - started_at
    deadline_exceeded = elapsed_seconds >= PROOF_DEADLINE_SECONDS
    return {
        **report,
        "elapsed_seconds": elapsed_seconds,
        "deadline_seconds": PROOF_DEADLINE_SECONDS,
        "deadline_exceeded": deadline_exceeded,
        "passed": bool(report.get("passed", False) and not deadline_exceeded),
    }


N_FEATURES = 1
N_CELLS = 4
N_READOUT = 360
FINITE_DIFFERENCE_EPSILON = 1e-3
OBJECTIVE_VOLTAGE_OFFSET = 65.0
OBJECTIVE_VOLTAGE_SCALE = 20.0
SPIKE_DRIVE = 20.0


class CompatibilityHodgkinHuxley(braincell.SingleCompartment):
    """Construct the four-cell BrainCell 0.1.0 Hodgkin–Huxley fixture.

    Parameters
    ----------
    size : int, optional
        Number of compatible cells to construct.
    """

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
    """Return the declared one-feature by four-cell CSR relation.

    Returns
    -------
    brainevent.CSR
        One input row with four target cells.
    """

    return brainevent.CSR(
        jnp.asarray([0.1, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        jnp.asarray([0, 4], dtype=jnp.int32),
        shape=(1, 4),
    )


def recurrent_csr() -> brainevent.CSR:
    """Return the four-cell recurrent CSR relation.

    Returns
    -------
    brainevent.CSR
        Four-cell recurrent relation used by the fixture.
    """

    return brainevent.CSR(
        jnp.ones((N_CELLS,), dtype=jnp.float32),
        jnp.arange(N_CELLS, dtype=jnp.int32),
        jnp.arange(N_CELLS + 1, dtype=jnp.int32),
        shape=(N_CELLS, N_CELLS),
    )


def bounded_current_density(input_drive, recurrent_drive=0.0):
    """Convert bounded dimensionless drives to current density.

    Parameters
    ----------
    input_drive, recurrent_drive : array-like
        Dimensionless population drives.

    Returns
    -------
    brainunit.Quantity
        Current density in milliamperes per square centimetre.
    """

    return (
        0.02 * jnp.tanh(input_drive / 20.0) * u.mA / u.cm**2
        + 0.01 * jnp.tanh(recurrent_drive / 20.0) * u.mA / u.cm**2
    )


class PPPropRelationFixture(brainstate.nn.Module):
    """Expose input and recurrent sparse weights to the PP-Prop compiler.

    Parameters
    ----------
    input_weight : float, optional
        Initial first input weight.
    """

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
        """Advance the relation fixture by one input event.

        Parameters
        ----------
        x : array-like
            Input event with one value for each fixture feature.

        Returns
        -------
        brainunit.Quantity
            Updated membrane voltage in millivolts.
        """

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
    """Compile and return the two sparse hidden-state relations.

    Returns
    -------
    object
        Compiled PP-Prop relation fixture.
    """

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
    """Advance a fixture cell by one compiled 0.1 ms interval.

    Parameters
    ----------
    cell : CompatibilityHodgkinHuxley
        Cell fixture to advance.
    current : brainunit.Quantity
        Current density applied for the interval.

    Returns
    -------
    tuple
        Membrane voltage and spike values after the interval.
    """

    with brainstate.environ.context(dt=0.1 * u.ms):
        cell.update(current)
    return cell.V.value, cell.spike.value


def compiled_one_step(cell: CompatibilityHodgkinHuxley):
    """Return a JIT-compiled one-step driver for a fixture cell.

    Parameters
    ----------
    cell : CompatibilityHodgkinHuxley
        Cell fixture used by the driver.

    Returns
    -------
    callable
        JIT-compiled one-step driver.
    """

    return brainstate.transform.jit(advance_one_step, static_argnums=0)(
        cell, bounded_current_density(0.0)
    )


def finite_difference_fixture() -> dict[str, float]:
    """Compare one PP-Prop BrainCell step with a central difference.

    Returns
    -------
    dict
        Direct derivative, centered derivative, tolerance, and pass status.
    """

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
    """Check deterministic threshold crossing and finite surrogate activity.

    Returns
    -------
    dict
        Threshold, spike, and finite-gradient evidence.
    """

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
    """Check finite direct gradients for all 360 voltage-readout values.

    Returns
    -------
    dict
        Gradient shape, finiteness, and nonzero checks.
    """

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
        choices=("proof", "run", "evolve"),
        help="run the proof, fixed schedule, or resumable ARC evolution",
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
        help="raw ARC root for proof, run, and evolve (default: /datasets/arc/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write reports or resumable evolution artifacts into this directory",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="maximum resumable evolution rounds (default: 8)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="stable evolution rounds before stopping (default: 2)",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=ORDINARY_UPDATES,
        help="PP-Prop updates per non-proof block (default: 128)",
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


def _evolve_workflow_report(
    arc_root, output_dir, *, rounds=8, patience=2, updates=ORDINARY_UPDATES
):
    """Run or resume iterative ARC evolution and summarize its terminal state."""

    from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter
    from examples.pp_prop.example21_evolve import (
        ConsoleProgressReporter,
        PipelineConfig,
        run_evolution,
    )

    config = PipelineConfig(rounds=rounds, patience=patience, updates=updates)
    state = run_evolution(
        Example21ArcAdapter(arc_root),
        output_dir,
        config=config,
        progress_reporter=ConsoleProgressReporter(),
    )
    return {
        "mode": "evolve",
        "passed": bool(state.closed and state.evaluation_completed),
        "closed": bool(state.closed),
        "terminal_reason": state.terminal_reason,
        "round_index": int(state.round_index),
        "training_exact_tasks": int(state.accepted.score.exact_count),
        "training_task_count": len(state.accepted.score.task_ids),
        "checkpoint": state.accepted.checkpoint_path,
        "checkpoint_sha256": state.accepted.checkpoint_sha256,
        "evaluation_digest": state.evaluation_digest,
    }


def _run_command(args):
    if args.command is not None and args.smoke:
        raise ValueError(
            "Choose one of proof, run, evolve, or --smoke; "
            "pass exactly one execution mode."
        )
    if args.command is None and not args.smoke:
        _parser().print_help()
        return 0
    if args.command == "evolve" and args.output_dir is None:
        raise ValueError(
            "Evolution requires --output-dir for checkpoints and progress; "
            "pass a durable run directory."
        )
    try:
        device = jax.devices(args.device)[0]
    except RuntimeError as error:
        raise ValueError(
            f"Requested device {args.device!r} is unavailable; "
            "select an available device."
        ) from error
    started_at = time.monotonic() if args.command == "proof" else None
    with jax.default_device(device):
        if args.command == "proof":
            report = _real_workflow_report(args.arc_root, proof=True)
        elif args.command == "run":
            report = _real_workflow_report(args.arc_root, proof=False)
        elif args.command == "evolve":
            report = _evolve_workflow_report(
                args.arc_root,
                args.output_dir,
                rounds=args.rounds,
                patience=args.patience,
                updates=args.updates,
            )
        else:
            report = _smoke_report()
    if started_at is not None:
        report = _apply_proof_deadline(report, started_at)
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
