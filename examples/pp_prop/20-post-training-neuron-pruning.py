"""20 - Causal post-training neuron-and-edge pruning for Example 18.

Example 18 learns the tasks and evolves recurrent synapses. This example then
freezes learning, ranks recurrent neurons by task-aware contribution signals,
and functionally removes the lowest-ranked neurons. Fixed probes are rerun after
each coarse-to-fine lesion checkpoint to find the smallest observed contiguous
safe network along that ranking. A second greedy phase then tests every retained
neuron individually, accepts only safe removals, reranks, and repeats. It then
does the same for retained-to-retained edges and alternates the two phases until
neither phase can remove another coordinate.

The result is coordinate-wise locally minimal in neurons and recurrent edges on
fixed probes for one trained model, not a globally minimum subnetwork and not a
new learning algorithm. Example 19's exact twin partition is reused only to
describe whether removed neurons came from structurally interchangeable groups.
"""

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np


def _load_neighbor(number: str, module_name: str):
    path = pathlib.Path(__file__).resolve().with_name(number)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EX18 = _load_neighbor("18-structural-evolution.py", "_pp_prop_structural_evolution")
EX19 = _load_neighbor(
    "19-structural-evolution-cfsg-symmetry.py", "_pp_prop_cfsg_symmetry"
)


def _bind_device(requested: str) -> str:
    """Bind JAX to the requested backend and fail closed on a mismatch."""
    if requested not in ("gpu", "cpu", "auto"):
        raise ValueError("device must be 'gpu', 'cpu', or 'auto'")
    if requested != "auto":
        jax.config.update("jax_platform_name", requested)
    try:
        backend = jax.default_backend()
    except RuntimeError as error:
        raise RuntimeError(
            f"requested device {requested}, but that JAX backend is unavailable"
        ) from error
    if requested != "auto" and backend != requested:
        raise RuntimeError(f"requested device {requested}, bound backend is {backend}")
    return backend


def _normalize_task_rows(values: np.ndarray) -> np.ndarray:
    """Normalize every nonzero task row by its maximum absolute value."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite two-dimensional array")
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    return np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)


def _contribution_scores(
    rates: np.ndarray,
    readout_weight: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    task_mass: np.ndarray,
    n_rec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine task-specific activity, output, recurrence, and gradient signals."""
    rates = np.asarray(rates, dtype=np.float64)
    readout_weight = np.asarray(readout_weight, dtype=np.float64)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    values = np.asarray(values, dtype=np.float64)
    task_mass = np.asarray(task_mass, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] != n_rec:
        raise ValueError("rates must have shape (n_tasks, n_rec)")
    n_tasks = rates.shape[0]
    if n_tasks < 1 or readout_weight.shape != (n_rec, n_tasks):
        raise ValueError("readout_weight must have shape (n_rec, n_tasks)")
    if rows.ndim != 1 or cols.ndim != 1 or values.ndim != 1:
        raise ValueError("edge arrays must be one-dimensional and aligned")
    if not (rows.shape == cols.shape == values.shape):
        raise ValueError("edge arrays must be aligned")
    if task_mass.shape != (n_tasks, rows.size):
        raise ValueError("task_mass must have shape (n_tasks, n_edges)")
    finite = (
        np.all(np.isfinite(rates))
        and np.all(np.isfinite(readout_weight))
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(task_mass))
    )
    if not finite or np.any(rates < 0) or np.any(task_mass < 0):
        raise ValueError(
            "contribution inputs must be finite and nonnegative where required"
        )
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
        cols.dtype, np.integer
    ):
        raise ValueError("edge endpoints must be integers")
    if rows.size and (
        np.any(rows < 0)
        or np.any(rows >= n_rec)
        or np.any(cols < 0)
        or np.any(cols >= n_rec)
    ):
        raise ValueError("edge endpoint is outside [0, n_rec)")

    direct = rates * np.abs(readout_weight.T)
    outgoing_strength = np.zeros(n_rec, dtype=np.float64)
    np.add.at(outgoing_strength, cols, np.abs(values))
    relay = rates * outgoing_strength[None, :]
    incident_mass = np.zeros((n_tasks, n_rec), dtype=np.float64)
    for task in range(n_tasks):
        np.add.at(incident_mass[task], rows, task_mass[task])
        np.add.at(incident_mass[task], cols, task_mass[task])
    task_scores = (
        _normalize_task_rows(direct)
        + _normalize_task_rows(relay)
        + _normalize_task_rows(incident_mass)
    ) / 3.0
    scores = np.max(task_scores, axis=0)
    owners = np.argmax(task_scores, axis=0)
    return scores, task_scores, owners


def _edge_contribution_scores(
    rates: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    task_mass: np.ndarray,
    neuron_alive: np.ndarray,
    edge_alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank active edges from task-aware transmission and gradient signals."""
    rates = np.asarray(rates, dtype=np.float64)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    values = np.asarray(values, dtype=np.float64)
    task_mass = np.asarray(task_mass, dtype=np.float64)
    neuron_alive = np.asarray(neuron_alive, dtype=np.float64)
    edge_alive = np.asarray(edge_alive, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] != neuron_alive.size:
        raise ValueError("rates and neuron_alive must align by neuron")
    if rows.ndim != 1 or not (rows.shape == cols.shape == values.shape):
        raise ValueError("edge arrays must be one-dimensional and aligned")
    if task_mass.shape != (rates.shape[0], rows.size):
        raise ValueError("task_mass must have shape (n_tasks, n_edges)")
    if edge_alive.shape != rows.shape:
        raise ValueError("edge_alive must have shape (n_edges,)")
    if not np.all((neuron_alive == 0.0) | (neuron_alive == 1.0)):
        raise ValueError("neuron_alive must be binary")
    if not np.all((edge_alive == 0.0) | (edge_alive == 1.0)):
        raise ValueError("edge_alive must be binary")
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
        cols.dtype, np.integer
    ):
        raise ValueError("edge endpoints must be integers")
    if rows.size and (
        np.any(rows < 0)
        or np.any(cols < 0)
        or np.any(rows >= neuron_alive.size)
        or np.any(cols >= neuron_alive.size)
    ):
        raise ValueError("edge endpoint is outside the neuron mask")
    if (
        not np.all(np.isfinite(rates))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(task_mass))
        or np.any(rates < 0.0)
        or np.any(task_mass < 0.0)
    ):
        raise ValueError("edge contribution inputs must be finite and nonnegative")
    active = edge_alive * neuron_alive[rows] * neuron_alive[cols]
    transmission = rates[:, cols] * np.abs(values)[None, :] * active[None, :]
    gradient = task_mass * active[None, :]
    task_scores = (
        _normalize_task_rows(transmission) + _normalize_task_rows(gradient)
    ) / 2.0
    scores = np.max(task_scores, axis=0)
    owners = np.argmax(task_scores, axis=0)
    return scores, task_scores, owners


def _removal_order(scores: np.ndarray) -> np.ndarray:
    """Return neuron indices from least to most important, stably by index."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    return np.lexsort((np.arange(scores.size), scores))


def _coarse_removed_counts(n_rec: int, fraction: float) -> np.ndarray:
    """Return coarse removal counts including all-alive and one-retained ends."""
    if isinstance(n_rec, bool) or n_rec < 1:
        raise ValueError("n_rec must be a positive integer")
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("prune step fraction must be finite and in (0, 1)")
    step = max(1, math.ceil(n_rec * fraction))
    counts = np.arange(0, n_rec, step, dtype=int)
    if counts[-1] != n_rec - 1:
        counts = np.append(counts, n_rec - 1)
    return counts


def _alive_masks(order: np.ndarray, removed_counts: np.ndarray) -> np.ndarray:
    """Construct fixed-shape alive masks for cumulative removals."""
    order = np.asarray(order)
    removed_counts = np.asarray(removed_counts)
    if order.ndim != 1 or order.size < 1 or not np.issubdtype(order.dtype, np.integer):
        raise ValueError("order must be a nonempty integer permutation")
    if not np.array_equal(np.sort(order), np.arange(order.size)):
        raise ValueError("order must be a permutation of neuron indices")
    if removed_counts.ndim != 1 or not np.issubdtype(removed_counts.dtype, np.integer):
        raise ValueError("removed_counts must be a one-dimensional integer array")
    if (
        removed_counts.size < 1
        or np.any(removed_counts < 0)
        or np.any(removed_counts >= order.size)
    ):
        raise ValueError("removed_counts must stay in [0, n_rec)")
    rank = np.empty(order.size, dtype=int)
    rank[order] = np.arange(order.size)
    return (rank[None, :] >= removed_counts[:, None]).astype(np.float32)


def _select_safe_frontier(
    removed_counts: np.ndarray, accuracies: np.ndarray, target: float
) -> Dict[str, Any]:
    """Select the contiguous safe prefix and flag any later recovery."""
    removed_counts = np.asarray(removed_counts)
    accuracies = np.asarray(accuracies, dtype=np.float64)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target must be finite and in [0, 1]")
    valid_counts = (
        removed_counts.ndim == 1
        and removed_counts.size > 0
        and np.issubdtype(removed_counts.dtype, np.integer)
        and removed_counts[0] == 0
        and np.all(np.diff(removed_counts) > 0)
    )
    if not valid_counts:
        raise ValueError("removed_counts must start at zero and strictly increase")
    if (
        accuracies.ndim != 2
        or accuracies.shape[0] != removed_counts.size
        or accuracies.shape[1] < 1
        or not np.all(np.isfinite(accuracies))
        or np.any(accuracies < 0)
        or np.any(accuracies > 1)
    ):
        raise ValueError("accuracies must align with counts and stay in [0, 1]")
    valid = np.all(accuracies >= target, axis=1)
    failed = np.flatnonzero(~valid)
    if failed.size == 0:
        return {
            "baseline_eligible": True,
            "safe_removed": int(removed_counts[-1]),
            "first_failed_removed": None,
            "later_recovery": False,
        }
    first = int(failed[0])
    return {
        "baseline_eligible": bool(valid[0]),
        "safe_removed": int(removed_counts[max(0, first - 1)]),
        "first_failed_removed": int(removed_counts[first]),
        "later_recovery": bool(np.any(valid[first + 1 :])),
    }


def _refinement_counts(frontier: Dict[str, Any]) -> np.ndarray:
    """Return every unmeasured removal count inside the first failure bracket."""
    failed = frontier["first_failed_removed"]
    safe = int(frontier["safe_removed"])
    if not frontier["baseline_eligible"] or failed is None or failed == 0:
        return np.array([], dtype=int)
    return np.arange(safe + 1, int(failed) + 1, dtype=int)


def _alignment_owners(
    initial_owners: np.ndarray,
    final_owners: np.ndarray,
    removed: np.ndarray,
) -> np.ndarray:
    """Use meaningful ownership measurements on each side of the final mask."""
    initial_owners = np.asarray(initial_owners)
    final_owners = np.asarray(final_owners)
    removed = np.asarray(removed)
    if (
        initial_owners.ndim != 1
        or final_owners.shape != initial_owners.shape
        or removed.shape != initial_owners.shape
        or removed.dtype != np.bool_
        or not np.issubdtype(initial_owners.dtype, np.integer)
        or not np.issubdtype(final_owners.dtype, np.integer)
    ):
        raise ValueError("initial, final, and removed ownership arrays must align")
    return np.where(removed, initial_owners, final_owners)


def _alignment_summary(
    class_of: np.ndarray,
    removed: np.ndarray,
    owners: np.ndarray,
    n_tasks: int,
) -> Dict[str, Any]:
    """Summarize task ownership and exact-twin membership at the frontier."""
    class_of = np.asarray(class_of)
    removed = np.asarray(removed)
    owners = np.asarray(owners)
    if (
        class_of.ndim != 1
        or removed.shape != class_of.shape
        or owners.shape != class_of.shape
        or removed.dtype != np.bool_
        or not np.issubdtype(owners.dtype, np.integer)
    ):
        raise ValueError("class_of, removed, and owners must align by neuron")
    if n_tasks < 1 or np.any(owners < 0) or np.any(owners >= n_tasks):
        raise ValueError("owners must contain valid task indices")
    sizes = Counter(class_of.tolist())
    twin = np.array([sizes[label] >= 2 for label in class_of.tolist()])
    full = partial = 0
    for label, size in sizes.items():
        if size < 2:
            continue
        state = removed[class_of == label]
        full += int(np.all(state))
        partial += int(np.any(state) and not np.all(state))
    return {
        "removed_task_counts": np.bincount(owners[removed], minlength=n_tasks).tolist(),
        "retained_task_counts": np.bincount(
            owners[~removed], minlength=n_tasks
        ).tolist(),
        "removed_twin_neurons": int(np.sum(removed & twin)),
        "fully_removed_twin_classes": full,
        "partially_pruned_twin_classes": partial,
    }


_PROBE_CACHE: Dict[int, tuple] = {}


def _probe_arrays(config: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build Example 18's deterministic task-major probe trials and windows.

    Memoized per configuration object, because rebuilding costs a Python loop
    over every probe trial and several evaluators are handed the same
    configuration in one run. The cache keeps a reference to the configuration
    so its identity stays valid.
    """
    cached = _PROBE_CACHE.get(id(config))
    if cached is not None and cached[0] is config:
        return cached[1]
    layout = EX18._layout(config)
    trials = []
    windows = []
    for task in range(config.num_tricks):
        window = np.zeros(layout.n_step, dtype=np.float32)
        start = layout.response_start[task]
        window[start : start + layout.response_ticks] = 1.0
        for index in range(config.eval_trials_per_task):
            spikes, _, _ = EX18._make_trial(task, EX18._probe_seed(task, index), config)
            trials.append(spikes)
            windows.append(window)
    built = (jnp.stack(trials), jnp.asarray(np.stack(windows)))
    _PROBE_CACHE[id(config)] = (config, built)
    return built


def _probe_logit_evaluator(experiment: Any, config: Any):
    """Build a one-trial-at-a-time fixed-probe logit evaluator.

    Deliberately unbatched. The physical-compaction check compares this
    evaluator's raw logits between a masked wide model and its narrow rebuild
    at ``rtol=1e-5`` / ``atol=1e-6``. On GPU a different batch size selects
    different kernels, and that rounding is occasionally enough to move a
    marginal membrane potential across the spike threshold; one flipped spike
    shifts a logit by about 1e-4, far outside that tolerance. This path runs a
    handful of times per analysis, so there is nothing to gain by batching it
    and a published fail-closed guarantee to lose.
    """
    trials, windows = _probe_arrays(config)
    model = experiment.model
    rec_weight = model.rec_syn.comm.weight
    base_params = dict(rec_weight.value)
    base_values = base_params["weight"]

    def evaluate_mask(alive, edge_alive):
        masked_params = dict(base_params)
        masked_params["weight"] = base_values * edge_alive
        rec_weight.value = masked_params

        def evaluate_trial(spikes, window):
            brainstate.nn.reset_all_states(model, batch_size=1)

            def step(x):
                model.ff_syn(x)
                model.rec_syn(model.neu.get_spike() * alive)
                model.neu(0.0 * u.mA)
                masked_spikes = model.neu.get_spike() * alive
                return model.readout(masked_spikes), masked_spikes

            outputs, neuron_spikes = brainstate.transform.for_loop(step, spikes)
            logits = jnp.sum(outputs * window[:, None, None], axis=0)
            logits = logits / jnp.maximum(jnp.sum(window), 1.0)
            return logits[0], jnp.mean(neuron_spikes, axis=(0, 1))

        # Size the states before the trial loop carries them: the search leaves
        # the model at the probe batch, and the per-trial reset inside the body
        # must not change the carry shape.
        brainstate.nn.reset_all_states(model, batch_size=1)
        logits, rates = brainstate.transform.for_loop(evaluate_trial, trials, windows)
        task_rates = rates.reshape(
            config.num_tricks, config.eval_trials_per_task, config.n_rec
        ).mean(axis=1)
        rec_weight.value = base_params
        return logits, task_rates

    return evaluate_mask


def _probe_batch(config: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return step-major probe tensors for a single batched rollout.

    Returns
    -------
    spikes : jax.Array
        Input spikes of shape ``(n_step, n_trial, n_in)``.
    windows : jax.Array
        Response windows of shape ``(n_step, n_trial)``.
    window_sums : jax.Array
        Per-trial response-window length, floored at one, shape ``(n_trial,)``.
    """
    trials, windows = _probe_arrays(config)
    spikes = jnp.transpose(trials[:, :, 0, :], (1, 0, 2))
    return spikes, windows.T, jnp.maximum(jnp.sum(windows, axis=1), 1.0)


def _batched_logit_evaluator(experiment: Any, config: Any):
    """Build a fixed-probe logit evaluator that runs every trial at once.

    The probe trials are mutually independent, so they belong on the model's
    batch axis rather than in a sequential loop: one rollout of ``n_step`` steps
    at batch ``n_trial`` replaces ``n_trial`` rollouts of ``n_step`` steps at
    batch one. The recurrent projection takes a leading batch axis natively, so
    this needs no ``vmap`` and no sparse batching rule.

    Spike rates accumulate in the scan carry, because stacking them would cost
    one ``(n_trial, n_rec)`` array per step. Readout outputs are stacked and
    reduced in one pass instead, which keeps the logit summation shape the
    unbatched evaluator uses.

    Callers must size the model's states to ``n_trial`` before entering any
    compiled loop that carries them; :func:`_prime_probe_states` does that.
    """
    spikes, windows, window_sums = _probe_batch(config)
    n_step, n_trial = int(spikes.shape[0]), int(spikes.shape[1])
    model = experiment.model
    rec_weight = model.rec_syn.comm.weight
    denominator = window_sums[:, None]

    def evaluate_mask(alive, edge_alive):
        base_params = dict(rec_weight.value)
        masked_params = dict(base_params)
        masked_params["weight"] = base_params["weight"] * edge_alive
        rec_weight.value = masked_params
        brainstate.nn.reset_all_states(model, batch_size=n_trial)

        def step(rate_sum, inputs):
            current, window = inputs
            model.ff_syn(current)
            model.rec_syn(model.neu.get_spike() * alive)
            model.neu(0.0 * u.mA)
            masked_spikes = model.neu.get_spike() * alive
            output = model.readout(masked_spikes) * window[:, None]
            return rate_sum + masked_spikes, output

        rate_sum, outputs = brainstate.transform.scan(
            step,
            jnp.zeros((n_trial, config.n_rec), dtype=jnp.float32),
            (spikes, windows),
        )
        rec_weight.value = base_params
        logits = jnp.sum(outputs, axis=0) / denominator
        rates = (rate_sum / n_step).reshape(
            config.num_tricks, config.eval_trials_per_task, config.n_rec
        )
        return logits, jnp.mean(rates, axis=1)

    return evaluate_mask


def _prime_probe_batch(experiment: Any, config: Any) -> None:
    """Reset every model state to the probe-trial batch size."""
    trials, _ = _probe_arrays(config)
    brainstate.nn.reset_all_states(experiment.model, batch_size=int(trials.shape[0]))


def _prime_evaluator(evaluate_mask) -> None:
    """Size model states to the evaluator's batch before a carrying loop.

    A compiled ``for_loop`` or ``while_loop`` carries model state, so the state
    shape must already match the rollout's batch before the loop is traced; the
    reset inside the rollout then reproduces that same shape. Evaluators that
    do not drive a model, such as the synthetic oracles used in tests, expose
    no ``prime`` and need none.
    """
    prime = getattr(evaluate_mask, "prime", None)
    if prime is not None:
        prime()


def _mask_evaluator(experiment: Any, config: Any):
    """Build a transform-compatible accuracy evaluator for structural masks."""
    evaluate_logits = _batched_logit_evaluator(experiment, config)
    task_ids = jnp.repeat(jnp.arange(config.num_tricks), config.eval_trials_per_task)

    def evaluate_mask(alive, edge_alive):
        logits, task_rates = evaluate_logits(alive, edge_alive)
        predictions = jnp.argmax(logits, axis=1)
        correct = predictions == task_ids
        accuracies = correct.reshape(
            config.num_tricks, config.eval_trials_per_task
        ).mean(axis=1)
        return accuracies, task_rates

    evaluate_mask.prime = lambda: _prime_probe_batch(experiment, config)
    return evaluate_mask


def _evaluate_probe_logits(
    experiment: Any,
    config: Any,
    neuron_alive: np.ndarray,
    edge_alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-trial logits and task accuracy for one structural mask."""
    neuron_alive = np.asarray(neuron_alive, dtype=np.float32)
    edge_alive = np.asarray(edge_alive, dtype=np.float32)
    if neuron_alive.shape != (config.n_rec,) or not np.all(
        (neuron_alive == 0.0) | (neuron_alive == 1.0)
    ):
        raise ValueError("neuron_alive must be a binary (n_rec,) array")
    if edge_alive.shape != (experiment.task_mass.shape[1],) or not np.all(
        (edge_alive == 0.0) | (edge_alive == 1.0)
    ):
        raise ValueError("edge_alive must be a binary (n_edges,) array")
    evaluate_logits = _probe_logit_evaluator(experiment, config)

    @brainstate.transform.jit
    def evaluate(neurons, edges):
        return evaluate_logits(neurons, edges)

    logits, _ = evaluate(jnp.asarray(neuron_alive), jnp.asarray(edge_alive))
    logits = np.asarray(logits, dtype=np.float64)
    predictions = np.argmax(logits, axis=1)
    task_ids = np.repeat(np.arange(config.num_tricks), config.eval_trials_per_task)
    accuracy = (
        (predictions == task_ids)
        .reshape(config.num_tricks, config.eval_trials_per_task)
        .mean(axis=1)
    )
    return logits, accuracy


def _evaluate_alive_masks(
    experiment: Any, config: Any, alive_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate task accuracy and spike rates for many lesion masks in one driver."""
    edge_count = int(experiment.task_mass.shape[1])
    edge_masks = np.ones((np.asarray(alive_masks).shape[0], edge_count))
    return _evaluate_structural_masks(experiment, config, alive_masks, edge_masks)


def _evaluate_structural_masks(
    experiment: Any,
    config: Any,
    alive_masks: np.ndarray,
    edge_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate aligned neuron and recurrent-edge lesion masks."""
    alive_masks = np.asarray(alive_masks, dtype=np.float32)
    edge_masks = np.asarray(edge_masks, dtype=np.float32)
    if (
        alive_masks.ndim != 2
        or alive_masks.shape[1] != config.n_rec
        or alive_masks.shape[0] < 1
        or not np.all((alive_masks == 0.0) | (alive_masks == 1.0))
    ):
        raise ValueError("alive_masks must be a nonempty binary (masks, n_rec) array")
    if (
        edge_masks.ndim != 2
        or edge_masks.shape != (alive_masks.shape[0], experiment.task_mass.shape[1])
        or not np.all((edge_masks == 0.0) | (edge_masks == 1.0))
    ):
        raise ValueError("edge_masks must be binary and align with alive_masks")
    evaluate_mask = _mask_evaluator(experiment, config)

    @brainstate.transform.jit
    def evaluate(neuron_masks, recurrent_masks):
        _prime_evaluator(evaluate_mask)
        return brainstate.transform.for_loop(
            evaluate_mask, neuron_masks, recurrent_masks
        )

    accuracies, rates = evaluate(jnp.asarray(alive_masks), jnp.asarray(edge_masks))
    return np.asarray(accuracies, dtype=np.float64), np.asarray(rates, dtype=np.float64)


def _joint_fixed_point_prune(
    experiment: Any,
    config: Any,
    initial_alive: np.ndarray,
    target: float,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    task_mass: np.ndarray,
) -> Dict[str, Any]:
    """Alternate greedy neuron and edge phases to a coordinate-wise fixed point."""
    initial_alive = np.asarray(initial_alive, dtype=np.float32)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    values = np.asarray(values, dtype=np.float32)
    task_mass = np.asarray(task_mass, dtype=np.float32)
    n_rec = config.n_rec
    n_tasks = config.num_tricks
    if initial_alive.shape != (n_rec,) or not np.all(
        (initial_alive == 0.0) | (initial_alive == 1.0)
    ):
        raise ValueError("initial_alive must be a binary (n_rec,) array")
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target must be finite and in [0, 1]")
    if rows.shape != cols.shape or rows.shape != values.shape or rows.ndim != 1:
        raise ValueError("edge arrays must be one-dimensional and aligned")
    if task_mass.shape != (n_tasks, rows.size):
        raise ValueError("task_mass must have shape (n_tasks, n_edges)")

    evaluate_mask = _mask_evaluator(experiment, config)
    readout_weight = jnp.asarray(_readout_weight(experiment.model), dtype=jnp.float32)
    edge_rows = jnp.asarray(rows)
    edge_cols = jnp.asarray(cols)
    edge_values = jnp.asarray(values)
    edge_task_mass = jnp.asarray(task_mass)

    def normalize_rows(array):
        scale = jnp.max(jnp.abs(array), axis=1, keepdims=True)
        return array / jnp.where(scale > 0.0, scale, 1.0)

    def induced_edges(alive, edge_alive):
        return edge_alive * alive[edge_rows] * alive[edge_cols]

    def current_neuron_scores(rates, alive, edge_alive):
        edge_alive = induced_edges(alive, edge_alive)
        outgoing = (
            jnp.zeros(n_rec, dtype=rates.dtype)
            .at[edge_cols]
            .add(jnp.abs(edge_values) * edge_alive)
        )
        incident = jnp.zeros((n_tasks, n_rec), dtype=rates.dtype)
        weighted_mass = edge_task_mass * edge_alive[None, :]
        incident = incident.at[:, edge_rows].add(weighted_mass)
        incident = incident.at[:, edge_cols].add(weighted_mass)
        direct = rates * jnp.abs(readout_weight.T)
        relay = rates * outgoing[None, :]
        task_scores = (
            normalize_rows(direct) + normalize_rows(relay) + normalize_rows(incident)
        ) / 3.0
        return jnp.max(task_scores, axis=0), task_scores

    def current_edge_scores(rates, alive, edge_alive):
        active = induced_edges(alive, edge_alive)
        transmission = (
            rates[:, edge_cols] * jnp.abs(edge_values)[None, :] * active[None, :]
        )
        gradient = edge_task_mass * active[None, :]
        task_scores = (normalize_rows(transmission) + normalize_rows(gradient)) / 2.0
        return jnp.max(task_scores, axis=0), task_scores

    record_count = n_rec + rows.size + 2

    def run_phase(
        alive,
        edge_alive,
        total_accepted,
        removal_step,
        removal_accuracy,
        last_test_accuracy,
        accepted_per_pass,
        pass_index,
        cycle_index,
        *,
        neurons,
    ):
        initial = (
            jnp.asarray(1, dtype=jnp.int32),
            alive,
            edge_alive,
            total_accepted,
            removal_step,
            removal_accuracy,
            last_test_accuracy,
            accepted_per_pass,
            pass_index,
            jnp.asarray(0, dtype=jnp.int32),
        )

        def continue_pass(carry):
            accepted_last, alive_now, edges_now, *_, current_pass, _ = carry
            current_coordinates = alive_now if neurons else edges_now
            return jnp.logical_and(
                accepted_last > 0,
                jnp.logical_and(
                    jnp.sum(current_coordinates) > 0,
                    current_pass < record_count,
                ),
            )

        def run_pass(carry):
            (
                _,
                alive_now,
                edges_now,
                accepted_total,
                accepted_steps,
                accepted_accuracies,
                tested_accuracies,
                pass_counts,
                current_pass,
                phase_total,
            ) = carry
            _, rates = evaluate_mask(alive_now, edges_now)
            if neurons:
                scores, _ = current_neuron_scores(rates, alive_now, edges_now)
                current_coordinates = alive_now
            else:
                scores, _ = current_edge_scores(rates, alive_now, edges_now)
                current_coordinates = edges_now
            order = jnp.argsort(
                jnp.where(current_coordinates > 0, scores, jnp.inf), stable=True
            )
            retained_at_start = jnp.sum(current_coordinates, dtype=jnp.int32)
            candidate_initial = (
                jnp.asarray(0, dtype=jnp.int32),
                alive_now,
                edges_now,
                jnp.asarray(0, dtype=jnp.int32),
                accepted_total,
                accepted_steps,
                accepted_accuracies,
                tested_accuracies,
            )

            def continue_candidate(candidate_carry):
                return candidate_carry[0] < retained_at_start

            def test_candidate(candidate_carry):
                (
                    position,
                    accepted_alive,
                    accepted_edges,
                    accepted_this_pass,
                    candidate_total,
                    candidate_steps,
                    candidate_accuracies,
                    candidate_last,
                ) = candidate_carry
                coordinate = order[position]
                if neurons:
                    trial_alive = accepted_alive.at[coordinate].set(0.0)
                    trial_edges = induced_edges(trial_alive, accepted_edges)
                else:
                    trial_alive = accepted_alive
                    trial_edges = accepted_edges.at[coordinate].set(0.0)
                trial_accuracy, _ = evaluate_mask(trial_alive, trial_edges)
                accepted = jnp.all(trial_accuracy >= target)
                next_alive = jnp.where(accepted, trial_alive, accepted_alive)
                next_edges = jnp.where(accepted, trial_edges, accepted_edges)
                next_total = candidate_total + accepted.astype(jnp.int32)
                candidate_steps = candidate_steps.at[coordinate].set(
                    jnp.where(
                        accepted,
                        next_total - 1,
                        candidate_steps[coordinate],
                    )
                )
                candidate_accuracies = candidate_accuracies.at[coordinate].set(
                    jnp.where(
                        accepted,
                        trial_accuracy,
                        candidate_accuracies[coordinate],
                    )
                )
                candidate_last = candidate_last.at[coordinate].set(trial_accuracy)
                return (
                    position + 1,
                    next_alive,
                    next_edges,
                    accepted_this_pass + accepted.astype(jnp.int32),
                    next_total,
                    candidate_steps,
                    candidate_accuracies,
                    candidate_last,
                )

            candidate = brainstate.transform.while_loop(
                continue_candidate, test_candidate, candidate_initial
            )
            (
                _,
                next_alive,
                next_edges,
                accepted_this_pass,
                next_total,
                next_steps,
                next_accuracies,
                next_last,
            ) = candidate
            pass_counts = pass_counts.at[current_pass].set(accepted_this_pass)
            label = "neuron" if neurons else "edge"
            retained = jnp.sum(next_alive if neurons else next_edges, dtype=jnp.int32)
            jax.debug.print(
                "[20-joint] cycle {cycle} " + label + " pass {pass_number}: "
                "accepted {accepted}; retained {retained}",
                cycle=cycle_index + 1,
                pass_number=current_pass + 1,
                accepted=accepted_this_pass,
                retained=retained,
                ordered=True,
            )
            return (
                accepted_this_pass,
                next_alive,
                next_edges,
                next_total,
                next_steps,
                next_accuracies,
                next_last,
                pass_counts,
                current_pass + 1,
                phase_total + accepted_this_pass,
            )

        return brainstate.transform.while_loop(continue_pass, run_pass, initial)

    @brainstate.transform.jit
    def run(alive):
        _prime_evaluator(evaluate_mask)
        edge_alive = alive[edge_rows] * alive[edge_cols]
        initial = (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            alive,
            edge_alive,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.full(n_rec, -1, dtype=jnp.int32),
            jnp.full(rows.size, -1, dtype=jnp.int32),
            jnp.full((n_rec, n_tasks), jnp.nan, dtype=jnp.float32),
            jnp.full((rows.size, n_tasks), jnp.nan, dtype=jnp.float32),
            jnp.full((n_rec, n_tasks), jnp.nan, dtype=jnp.float32),
            jnp.full((rows.size, n_tasks), jnp.nan, dtype=jnp.float32),
            jnp.zeros(record_count, dtype=jnp.int32),
            jnp.zeros(record_count, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def continue_cycle(carry):
            cycle_index, edges_accepted_last, *_ = carry
            return jnp.logical_and(edges_accepted_last > 0, cycle_index < record_count)

        def run_cycle(carry):
            (
                cycle_index,
                _,
                alive_now,
                edges_now,
                neuron_total,
                edge_total,
                neuron_steps,
                edge_steps,
                neuron_accuracies,
                edge_accuracies,
                neuron_last,
                edge_last,
                neuron_counts,
                edge_counts,
                neuron_pass_index,
                edge_pass_index,
            ) = carry
            neuron_phase = run_phase(
                alive_now,
                edges_now,
                neuron_total,
                neuron_steps,
                neuron_accuracies,
                neuron_last,
                neuron_counts,
                neuron_pass_index,
                cycle_index,
                neurons=True,
            )
            (
                _,
                after_neurons,
                after_neuron_edges,
                neuron_total,
                neuron_steps,
                neuron_accuracies,
                neuron_last,
                neuron_counts,
                neuron_pass_index,
                _,
            ) = neuron_phase
            edge_phase = run_phase(
                after_neurons,
                after_neuron_edges,
                edge_total,
                edge_steps,
                edge_accuracies,
                edge_last,
                edge_counts,
                edge_pass_index,
                cycle_index,
                neurons=False,
            )
            (
                _,
                final_neurons,
                after_edges,
                edge_total,
                edge_steps,
                edge_accuracies,
                edge_last,
                edge_counts,
                edge_pass_index,
                edges_this_cycle,
            ) = edge_phase
            return (
                cycle_index + 1,
                edges_this_cycle,
                final_neurons,
                after_edges,
                neuron_total,
                edge_total,
                neuron_steps,
                edge_steps,
                neuron_accuracies,
                edge_accuracies,
                neuron_last,
                edge_last,
                neuron_counts,
                edge_counts,
                neuron_pass_index,
                edge_pass_index,
            )

        final = brainstate.transform.while_loop(continue_cycle, run_cycle, initial)
        (
            cycle_count,
            edges_accepted_last,
            final_alive,
            final_edge_alive,
            neuron_total,
            edge_total,
            neuron_steps,
            edge_steps,
            neuron_accuracies,
            edge_accuracies,
            neuron_last,
            edge_last,
            neuron_counts,
            edge_counts,
            neuron_pass_count,
            edge_pass_count,
        ) = final
        final_accuracy, final_rates = evaluate_mask(final_alive, final_edge_alive)
        final_scores, final_task_scores = current_neuron_scores(
            final_rates, final_alive, final_edge_alive
        )
        final_edge_scores, final_edge_task_scores = current_edge_scores(
            final_rates, final_alive, final_edge_alive
        )
        return (
            cycle_count,
            edges_accepted_last,
            final_alive,
            final_edge_alive,
            neuron_total,
            edge_total,
            neuron_steps,
            edge_steps,
            neuron_accuracies,
            edge_accuracies,
            neuron_last,
            edge_last,
            neuron_counts,
            edge_counts,
            neuron_pass_count,
            edge_pass_count,
            final_accuracy,
            final_scores,
            final_task_scores,
            final_edge_scores,
            final_edge_task_scores,
        )

    (
        cycle_count,
        edges_accepted_last,
        final_alive,
        final_edge_alive,
        neuron_total,
        edge_total,
        neuron_step,
        edge_step,
        neuron_removal_accuracy,
        edge_removal_accuracy,
        neuron_last_test_accuracy,
        edge_last_test_accuracy,
        neuron_accepted_per_pass,
        edge_accepted_per_pass,
        neuron_pass_count,
        edge_pass_count,
        final_accuracy,
        final_scores,
        final_task_scores,
        final_edge_scores,
        final_edge_task_scores,
    ) = run(jnp.asarray(initial_alive))
    cycle_count = int(np.asarray(cycle_count))
    neuron_pass_count = int(np.asarray(neuron_pass_count))
    edge_pass_count = int(np.asarray(edge_pass_count))
    final_alive = np.asarray(final_alive, dtype=np.float32)
    final_edge_alive = np.asarray(final_edge_alive, dtype=np.float32)
    neuron_step = np.asarray(neuron_step, dtype=int)
    edge_step = np.asarray(edge_step, dtype=int)
    neuron_removal_accuracy = np.asarray(neuron_removal_accuracy, dtype=np.float64)
    edge_removal_accuracy = np.asarray(edge_removal_accuracy, dtype=np.float64)
    retained = np.flatnonzero(final_alive > 0.0)
    retained_edges = np.flatnonzero(final_edge_alive > 0.0)
    accepted = np.flatnonzero(neuron_step >= 0)
    accepted = accepted[np.argsort(neuron_step[accepted])]
    accepted_edges = np.flatnonzero(edge_step >= 0)
    accepted_edges = accepted_edges[np.argsort(edge_step[accepted_edges])]
    final_scores = np.asarray(final_scores, dtype=np.float64)
    final_task_scores = np.asarray(final_task_scores, dtype=np.float64)
    final_edge_scores = np.asarray(final_edge_scores, dtype=np.float64)
    final_edge_task_scores = np.asarray(final_edge_task_scores, dtype=np.float64)
    original_live_live = final_alive[rows] * final_alive[cols]
    original_live_live_count = int(np.sum(original_live_live))
    active_edge_count = int(np.sum(final_edge_alive))
    return {
        "converged": bool(
            int(np.asarray(edges_accepted_last)) == 0
            and (retained.size == 0 or neuron_pass_count > 0)
        ),
        "cycle_count": cycle_count,
        "pass_count": neuron_pass_count,
        "neuron_pass_count": neuron_pass_count,
        "edge_pass_count": edge_pass_count,
        "accepted_per_pass": np.asarray(neuron_accepted_per_pass, dtype=int)[
            :neuron_pass_count
        ].tolist(),
        "neuron_accepted_per_pass": np.asarray(neuron_accepted_per_pass, dtype=int)[
            :neuron_pass_count
        ].tolist(),
        "edge_accepted_per_pass": np.asarray(edge_accepted_per_pass, dtype=int)[
            :edge_pass_count
        ].tolist(),
        "additional_removed": int(np.asarray(neuron_total)),
        "causally_removed_edges": int(np.asarray(edge_total)),
        "final_alive_mask": final_alive.astype(int).tolist(),
        "final_edge_alive_mask": final_edge_alive.astype(int).tolist(),
        "accepted_neurons": accepted.tolist(),
        "accepted_edges": accepted_edges.tolist(),
        "accepted_accuracies": neuron_removal_accuracy[accepted].tolist(),
        "accepted_edge_accuracies": edge_removal_accuracy[accepted_edges].tolist(),
        "final_accuracies": np.asarray(final_accuracy, dtype=np.float64).tolist(),
        "final_scores": final_scores.tolist(),
        "final_task_scores": final_task_scores.tolist(),
        "final_owners": np.argmax(final_task_scores, axis=0).tolist(),
        "final_edge_scores": final_edge_scores.tolist(),
        "final_edge_task_scores": final_edge_task_scores.tolist(),
        "final_edge_owners": np.argmax(final_edge_task_scores, axis=0).tolist(),
        "retained_indices": retained.tolist(),
        "retained_edge_indices": retained_edges.tolist(),
        "retained_single_ablation_accuracies": np.asarray(
            neuron_last_test_accuracy, dtype=np.float64
        )[retained].tolist(),
        "retained_single_edge_ablation_accuracies": np.asarray(
            edge_last_test_accuracy, dtype=np.float64
        )[retained_edges].tolist(),
        "retained_zero_score_count": int(np.sum(final_scores[retained] == 0.0)),
        "retained_zero_edge_score_count": int(
            np.sum(final_edge_scores[retained_edges] == 0.0)
        ),
        "stored_edge_count": int(rows.size),
        "incident_edge_count": int(rows.size - original_live_live_count),
        "final_original_live_live_edge_count": original_live_live_count,
        "causally_removed_live_live_edge_count": int(
            original_live_live_count - active_edge_count
        ),
        "final_active_edge_count": active_edge_count,
    }


def _readout_weight(model: Any) -> np.ndarray:
    """Extract the trained recurrent-neuron to task readout matrix."""
    readout = model.readout if hasattr(model.readout, "W") else model.readout.readout
    weight = np.asarray(u.get_mantissa(readout.W.value), dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("trained readout weight must be two-dimensional")
    return weight


def _compact_topology(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    neuron_alive: np.ndarray,
    edge_alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remap active live-live edges onto a compact recurrent index space."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    values = np.asarray(values, dtype=np.float64)
    neuron_alive = np.asarray(neuron_alive)
    edge_alive = np.asarray(edge_alive)
    if rows.ndim != 1 or not (rows.shape == cols.shape == values.shape):
        raise ValueError("edge arrays must be one-dimensional and aligned")
    if neuron_alive.ndim != 1 or not np.all((neuron_alive == 0) | (neuron_alive == 1)):
        raise ValueError("neuron_alive must be a one-dimensional binary array")
    if edge_alive.shape != rows.shape or not np.all(
        (edge_alive == 0) | (edge_alive == 1)
    ):
        raise ValueError("edge_alive must be binary and align with edge arrays")
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
        cols.dtype, np.integer
    ):
        raise ValueError("edge endpoints must be integers")
    if rows.size and (
        np.any(rows < 0)
        or np.any(cols < 0)
        or np.any(rows >= neuron_alive.size)
        or np.any(cols >= neuron_alive.size)
    ):
        raise ValueError("edge endpoint is outside the neuron mask")
    active = edge_alive.astype(bool)
    if np.any(
        active & ~(neuron_alive[rows].astype(bool) & neuron_alive[cols].astype(bool))
    ):
        raise ValueError("active edge is incident to a dead neuron")
    retained = np.flatnonzero(neuron_alive).astype(np.int64)
    if retained.size == 0:
        raise ValueError("physical compaction requires at least one retained neuron")
    old_to_new = np.full(neuron_alive.size, -1, dtype=np.int64)
    old_to_new[retained] = np.arange(retained.size, dtype=np.int64)
    return (
        retained,
        old_to_new[rows[active]],
        old_to_new[cols[active]],
        values[active],
    )


def _with_template_unit(values: np.ndarray, template: Any) -> Any:
    """Convert mantissas to float32 while preserving a parameter's unit."""
    array = jnp.asarray(values, dtype=jnp.float32)
    unit = getattr(template, "unit", None)
    return array * unit if unit is not None else array


def _compact_config(config: Any, n_rec: int, n_edges: int) -> Any:
    """Create an inference-only configuration with compact structural sizes."""
    values = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)
    values = dict(values)
    values["n_rec"] = int(n_rec)
    values["n_edges"] = int(n_edges)
    values["min_edges"] = min(int(values["min_edges"]), max(1, n_edges))
    values["max_edges"] = max(int(values["max_edges"]), max(1, n_edges))
    return SimpleNamespace(**values)


def _readout_module(model: Any) -> Any:
    """Return the underlying trained leaky readout for either task style."""
    return model.readout if hasattr(model.readout, "W") else model.readout.readout


def _install_compact_parameters(
    model: Any,
    ff_weight: np.ndarray,
    ff_bias: Optional[np.ndarray],
    readout_weight: np.ndarray,
) -> None:
    """Install dense compact parameters into a newly constructed inference net."""
    ff_params = dict(model.ff_syn.comm.weight.value)
    ff_params["weight"] = _with_template_unit(ff_weight, ff_params["weight"])
    if ff_bias is None:
        ff_params.pop("bias", None)
    elif "bias" not in ff_params:
        raise ValueError("bundle contains feed-forward bias for a bias-free model")
    else:
        ff_params["bias"] = _with_template_unit(ff_bias, ff_params["bias"])
    model.ff_syn.comm.weight.value = ff_params
    readout = _readout_module(model)
    readout.W.value = _with_template_unit(readout_weight, readout.W.value)


def _compact_experiment(
    config: Any,
    retained: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    ff_weight: np.ndarray,
    ff_bias: Optional[np.ndarray],
    readout_weight: np.ndarray,
) -> Dict[str, Any]:
    """Construct one compact inference experiment from explicit arrays."""
    compact_config = _compact_config(config, retained.size, rows.size)
    csr = EX18._to_csr(rows, cols, values, retained.size, compact_config.sparse_backend)
    with brainstate.random.seed_context(compact_config.seed + 1):
        model = EX18._Net(compact_config, csr)
    _install_compact_parameters(model, ff_weight, ff_bias, readout_weight)
    brainstate.nn.init_all_states(model, batch_size=1)
    experiment = SimpleNamespace(
        model=model,
        task_mass=np.zeros((compact_config.num_tricks, rows.size), dtype=np.float64),
    )
    return {
        "experiment": experiment,
        "config": compact_config,
        "original_neuron_indices": np.asarray(retained, dtype=np.int64),
        "rows": np.asarray(rows, dtype=np.int64),
        "cols": np.asarray(cols, dtype=np.int64),
        "values": np.asarray(values, dtype=np.float64),
    }


def _build_compact_model(
    experiment: Any,
    config: Any,
    neuron_alive: np.ndarray,
    edge_alive: np.ndarray,
) -> Dict[str, Any]:
    """Materialize the physically compact inference model for final masks."""
    rows, cols, values = EX18._current_topology(experiment)
    retained, compact_rows, compact_cols, compact_values = _compact_topology(
        rows, cols, values, neuron_alive, edge_alive
    )
    ff_params = experiment.model.ff_syn.comm.weight.value
    ff_weight = np.asarray(u.get_mantissa(ff_params["weight"]), dtype=np.float32)[
        :, retained
    ]
    ff_bias = ff_params.get("bias")
    if ff_bias is not None:
        ff_bias = np.asarray(u.get_mantissa(ff_bias), dtype=np.float32)[retained]
    readout_weight = _readout_weight(experiment.model)[retained, :]
    return _compact_experiment(
        config,
        retained,
        compact_rows,
        compact_cols,
        compact_values,
        ff_weight,
        ff_bias,
        readout_weight,
    )


def _save_compact_bundle(compact: Dict[str, Any], path: pathlib.Path) -> None:
    """Save a self-contained compact inference bundle."""
    model = compact["experiment"].model
    config = compact["config"]
    ff_params = model.ff_syn.comm.weight.value
    ff_bias = ff_params.get("bias")
    path = pathlib.Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        config_json=np.asarray(json.dumps(vars(config), sort_keys=True)),
        original_neuron_indices=np.asarray(
            compact["original_neuron_indices"], dtype=np.int64
        ),
        rows=np.asarray(compact["rows"], dtype=np.int64),
        cols=np.asarray(compact["cols"], dtype=np.int64),
        values=np.asarray(compact["values"], dtype=np.float32),
        ff_weight=np.asarray(u.get_mantissa(ff_params["weight"]), dtype=np.float32),
        has_ff_bias=np.asarray(ff_bias is not None),
        ff_bias=(
            np.asarray(u.get_mantissa(ff_bias), dtype=np.float32)
            if ff_bias is not None
            else np.empty(0, dtype=np.float32)
        ),
        readout_weight=np.asarray(_readout_weight(model), dtype=np.float32),
    )


def _load_compact_bundle(path: pathlib.Path) -> Dict[str, Any]:
    """Load and reconstruct a compact inference model from an NPZ bundle."""
    path = pathlib.Path(path).resolve()
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if int(bundle["format_version"]) != 1:
                raise ValueError("unsupported compact bundle format version")
            values = json.loads(str(bundle["config_json"]))
            config = SimpleNamespace(**values)
            retained = np.asarray(bundle["original_neuron_indices"], dtype=np.int64)
            rows = np.asarray(bundle["rows"], dtype=np.int64)
            cols = np.asarray(bundle["cols"], dtype=np.int64)
            recurrent_values = np.asarray(bundle["values"], dtype=np.float32)
            ff_weight = np.asarray(bundle["ff_weight"], dtype=np.float32)
            ff_bias = (
                np.asarray(bundle["ff_bias"], dtype=np.float32)
                if bool(bundle["has_ff_bias"])
                else None
            )
            readout_weight = np.asarray(bundle["readout_weight"], dtype=np.float32)
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid compact model bundle: {path}") from error
    if retained.size != config.n_rec or rows.shape != cols.shape:
        raise ValueError("compact bundle dimensions are inconsistent")
    layout = EX18._layout(config)
    if ff_weight.shape != (layout.n_in, config.n_rec):
        raise ValueError("compact feed-forward weight shape is inconsistent")
    if ff_bias is not None and ff_bias.shape != (config.n_rec,):
        raise ValueError("compact feed-forward bias shape is inconsistent")
    if readout_weight.shape != (config.n_rec, config.num_tricks):
        raise ValueError("compact readout weight shape is inconsistent")
    return _compact_experiment(
        config,
        retained,
        rows,
        cols,
        recurrent_values,
        ff_weight,
        ff_bias,
        readout_weight,
    )


def _inference_storage(model: Any) -> Dict[str, int]:
    """Count persistent parameter and CSR metadata storage for inference."""
    parameter_values = 0
    parameter_bytes = 0
    for state in model.states(brainstate.ParamState).values():
        for leaf in jax.tree_util.tree_leaves(state.value):
            array = np.asarray(u.get_mantissa(leaf))
            parameter_values += int(array.size)
            parameter_bytes += int(array.nbytes)
    sparse = model.rec_syn.comm.spar_mat
    csr_index_bytes = int(
        np.asarray(sparse.indices).nbytes + np.asarray(sparse.indptr).nbytes
    )
    return {
        "parameter_values": parameter_values,
        "parameter_bytes": parameter_bytes,
        "csr_index_bytes": csr_index_bytes,
        "total_bytes": parameter_bytes + csr_index_bytes,
    }


def _benchmark_compaction(
    experiment: Any,
    config: Any,
    neuron_alive: np.ndarray,
    edge_alive: np.ndarray,
    compact: Dict[str, Any],
    *,
    repetitions: int,
) -> Dict[str, float]:
    """Benchmark warmed masked and compact full-probe inference on one device."""
    if isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    def measure(current_experiment, current_config, neurons, edges):
        evaluate_logits = _probe_logit_evaluator(current_experiment, current_config)
        neurons = jnp.asarray(neurons, dtype=jnp.float32)
        edges = jnp.asarray(edges, dtype=jnp.float32)

        @brainstate.transform.jit
        def repeated(neuron_mask, edge_mask):
            def evaluate_once(_):
                logits, _ = evaluate_logits(neuron_mask, edge_mask)
                return logits

            return brainstate.transform.for_loop(evaluate_once, jnp.arange(repetitions))

        jax.block_until_ready(repeated(neurons, edges))
        started = time.perf_counter()
        jax.block_until_ready(repeated(neurons, edges))
        return (time.perf_counter() - started) * 1000.0 / repetitions

    compact_config = compact["config"]
    compact_experiment = compact["experiment"]
    masked_ms = measure(experiment, config, neuron_alive, edge_alive)
    compact_ms = measure(
        compact_experiment,
        compact_config,
        np.ones(compact_config.n_rec, dtype=np.float32),
        np.ones(compact["rows"].size, dtype=np.float32),
    )
    return {
        "masked_probe_ms": float(masked_ms),
        "compact_probe_ms": float(compact_ms),
        "speedup": float(masked_ms / compact_ms),
    }


def _analyze_compaction(
    experiment: Any,
    config: Any,
    fixed_point: Dict[str, Any],
    target: float,
    output: Optional[pathlib.Path],
    benchmark_repetitions: int,
) -> Dict[str, Any]:
    """Materialize, verify, benchmark, and optionally save the compact model."""
    if isinstance(benchmark_repetitions, bool) or benchmark_repetitions < 1:
        raise ValueError("benchmark_repetitions must be a positive integer")
    neuron_alive = np.asarray(fixed_point["final_alive_mask"], dtype=np.float32)
    edge_alive = np.asarray(fixed_point["final_edge_alive_mask"], dtype=np.float32)
    if not np.any(neuron_alive):
        return {
            "status": "skipped_empty_network",
            "reason": "physical _Net compaction requires at least one neuron",
        }
    compact = _build_compact_model(experiment, config, neuron_alive, edge_alive)
    masked_logits, masked_accuracy = _evaluate_probe_logits(
        experiment, config, neuron_alive, edge_alive
    )
    compact_config = compact["config"]
    compact_logits, compact_accuracy = _evaluate_probe_logits(
        compact["experiment"],
        compact_config,
        np.ones(compact_config.n_rec, dtype=np.float32),
        np.ones(compact["rows"].size, dtype=np.float32),
    )
    predictions_identical = bool(
        np.array_equal(
            np.argmax(masked_logits, axis=1),
            np.argmax(compact_logits, axis=1),
        )
    )
    logits_close = bool(
        np.allclose(masked_logits, compact_logits, rtol=1e-5, atol=1e-6)
    )
    compact_meets_target = bool(np.all(compact_accuracy >= target))
    if not (predictions_identical and logits_close and compact_meets_target):
        raise RuntimeError(
            "physical compaction failed masked-model equivalence or target "
            f"accuracy: predictions_identical={predictions_identical}, "
            f"logits_close={logits_close} (max abs error "
            f"{float(np.max(np.abs(masked_logits - compact_logits), initial=0.0)):.3e}"
            f"), compact_meets_target={compact_meets_target} "
            f"(compact {np.asarray(compact_accuracy).tolist()}, target {target})"
        )
    original_storage = _inference_storage(experiment.model)
    compact_storage = _inference_storage(compact["experiment"].model)
    timing = _benchmark_compaction(
        experiment,
        config,
        neuron_alive,
        edge_alive,
        compact,
        repetitions=benchmark_repetitions,
    )
    bundle_path = None
    bundle_bytes = None
    if output is not None:
        resolved = pathlib.Path(output).resolve()
        _save_compact_bundle(compact, resolved)
        bundle_path = str(resolved)
        bundle_bytes = resolved.stat().st_size
    return {
        "status": "complete",
        "n_rec": int(compact_config.n_rec),
        "n_edges": int(compact["rows"].size),
        "original_neuron_indices": compact["original_neuron_indices"].tolist(),
        "accuracies": compact_accuracy.tolist(),
        "predictions_identical": predictions_identical,
        "max_abs_logit_error": float(
            np.max(np.abs(masked_logits - compact_logits), initial=0.0)
        ),
        "original_storage": original_storage,
        "compact_storage": compact_storage,
        "storage_reduction_fraction": float(
            1.0 - compact_storage["total_bytes"] / original_storage["total_bytes"]
        ),
        "parameter_reduction_fraction": float(
            1.0
            - compact_storage["parameter_values"] / original_storage["parameter_values"]
        ),
        "benchmark_repetitions": int(benchmark_repetitions),
        **timing,
        "bundle_path": bundle_path,
        "bundle_bytes": bundle_bytes,
    }


def _merge_checkpoints(
    coarse_counts: np.ndarray,
    coarse_accuracies: np.ndarray,
    refined_counts: np.ndarray,
    refined_accuracies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge coarse and refined checkpoints into one sorted unique curve."""
    mapping = {
        int(count): np.asarray(accuracy, dtype=np.float64)
        for count, accuracy in zip(coarse_counts, coarse_accuracies)
    }
    mapping.update(
        {
            int(count): np.asarray(accuracy, dtype=np.float64)
            for count, accuracy in zip(refined_counts, refined_accuracies)
        }
    )
    counts = np.array(sorted(mapping), dtype=int)
    return counts, np.stack([mapping[int(count)] for count in counts])


def _analyze_pruning(
    experiment: Any,
    arm: Dict[str, Any],
    config: Any,
    target: float,
    step_fraction: float,
    *,
    compact_output: Optional[pathlib.Path] = None,
    benchmark_repetitions: int = 10,
) -> Dict[str, Any]:
    """Run the complete contribution ranking and causal pruning analysis."""
    _select_safe_frontier(np.array([0]), np.ones((1, config.num_tricks)), target)
    coarse_counts = _coarse_removed_counts(config.n_rec, step_fraction)
    baseline_accuracy, baseline_rates = _evaluate_alive_masks(
        experiment, config, np.ones((1, config.n_rec), dtype=np.float32)
    )
    rows, cols, values = EX18._current_topology(experiment)
    scores, task_scores, owners = _contribution_scores(
        baseline_rates[0],
        _readout_weight(experiment.model),
        rows,
        cols,
        values,
        experiment.task_mass,
        config.n_rec,
    )
    order = _removal_order(scores)
    baseline_frontier = _select_safe_frontier(np.array([0]), baseline_accuracy, target)
    refined_count = 0
    if not baseline_frontier["baseline_eligible"]:
        counts = np.array([0], dtype=int)
        accuracies = baseline_accuracy
        frontier = baseline_frontier
        status = "baseline_below_target"
        coarse_checkpoint_count = 1
    else:
        coarse_accuracies, _ = _evaluate_alive_masks(
            experiment, config, _alive_masks(order, coarse_counts)
        )
        coarse_frontier = _select_safe_frontier(
            coarse_counts, coarse_accuracies, target
        )
        refinement = _refinement_counts(coarse_frontier)
        refined_count = int(refinement.size)
        if refinement.size:
            refined_accuracies, _ = _evaluate_alive_masks(
                experiment, config, _alive_masks(order, refinement)
            )
        else:
            refined_accuracies = np.empty((0, config.num_tricks))
        counts, accuracies = _merge_checkpoints(
            coarse_counts,
            coarse_accuracies,
            refinement,
            refined_accuracies,
        )
        frontier = _select_safe_frontier(counts, accuracies, target)
        status = "complete"
        coarse_checkpoint_count = int(coarse_counts.size)

    initial_removed = int(frontier["safe_removed"])
    initial_alive = np.ones(config.n_rec, dtype=np.float32)
    initial_alive[order[:initial_removed]] = 0.0
    if status == "complete":
        fixed_point = _joint_fixed_point_prune(
            experiment,
            config,
            initial_alive,
            target,
            rows,
            cols,
            values,
            experiment.task_mass,
        )
        final_alive = np.asarray(fixed_point["final_alive_mask"], dtype=bool)
        final_owners = np.asarray(fixed_point["final_owners"], dtype=int)
        accepted_accuracies = np.asarray(
            fixed_point["accepted_accuracies"], dtype=np.float64
        ).reshape((-1, config.num_tricks))
        initial_position = int(np.flatnonzero(counts == initial_removed)[0])
        fixed_point_counts = np.arange(
            initial_removed,
            initial_removed + fixed_point["additional_removed"] + 1,
            dtype=int,
        )
        fixed_point_accuracies = np.concatenate(
            (accuracies[initial_position : initial_position + 1], accepted_accuracies),
            axis=0,
        )
    else:
        final_alive = initial_alive.astype(bool)
        final_owners = owners
        fixed_point_counts = np.array([0], dtype=int)
        fixed_point_accuracies = baseline_accuracy
        fixed_point = {
            "converged": False,
            "cycle_count": 0,
            "pass_count": 0,
            "accepted_per_pass": [],
            "neuron_accepted_per_pass": [],
            "edge_accepted_per_pass": [],
            "additional_removed": 0,
            "causally_removed_edges": 0,
            "final_alive_mask": initial_alive.astype(int).tolist(),
            "final_edge_alive_mask": np.ones(rows.size, dtype=int).tolist(),
            "accepted_neurons": [],
            "accepted_edges": [],
            "accepted_accuracies": [],
            "accepted_edge_accuracies": [],
            "final_accuracies": baseline_accuracy[0].tolist(),
            "final_scores": scores.tolist(),
            "final_task_scores": task_scores.tolist(),
            "final_owners": owners.tolist(),
            "retained_indices": np.flatnonzero(final_alive).tolist(),
            "retained_single_ablation_accuracies": [],
            "retained_single_edge_ablation_accuracies": [],
            "retained_zero_score_count": int(np.sum(scores == 0.0)),
            "retained_zero_edge_score_count": 0,
            "stored_edge_count": int(rows.size),
            "incident_edge_count": 0,
            "final_original_live_live_edge_count": int(rows.size),
            "causally_removed_live_live_edge_count": 0,
            "final_active_edge_count": int(rows.size),
        }
    removed = ~final_alive
    safe_removed = int(np.sum(removed))
    neighbors = EX19._validate_topology(config.n_rec, rows, cols)
    class_of = EX19._twin_partition(neighbors)
    alignment_owners = _alignment_owners(owners, final_owners, removed)
    alignment = _alignment_summary(
        class_of, removed, alignment_owners, config.num_tricks
    )
    compaction = (
        _analyze_compaction(
            experiment,
            config,
            fixed_point,
            target,
            compact_output,
            benchmark_repetitions,
        )
        if status == "complete"
        else {
            "status": "skipped_baseline_below_target",
            "reason": "the unpruned checkpoint did not meet the pruning target",
        }
    )
    return {
        "status": status,
        "target": float(target),
        "names": list(arm["trick_names"]),
        "baseline_accuracies": baseline_accuracy[0].tolist(),
        "scores": scores.tolist(),
        "task_scores": task_scores.tolist(),
        "owners": owners.tolist(),
        "removal_order": order.tolist(),
        "removed_counts": counts.tolist(),
        "accuracies": accuracies.tolist(),
        "initial_frontier_removed": initial_removed,
        "initial_frontier_retained": config.n_rec - initial_removed,
        "fixed_point_removed_counts": fixed_point_counts.tolist(),
        "fixed_point_accuracies": fixed_point_accuracies.tolist(),
        "fixed_point": fixed_point,
        "safe_removed": safe_removed,
        "safe_retained": config.n_rec - safe_removed,
        "first_failed_removed": frontier["first_failed_removed"],
        "later_recovery": bool(frontier["later_recovery"]),
        "coarse_checkpoint_count": coarse_checkpoint_count,
        "refined_checkpoint_count": refined_count,
        "alignment": alignment,
        "compaction": compaction,
    }


def _format_report(analysis: Dict[str, Any]) -> str:
    """Format the pruning result and its claim boundary in plain English."""
    names = analysis["names"]
    baseline = ", ".join(
        f"{name}={accuracy:.0%}"
        for name, accuracy in zip(names, analysis["baseline_accuracies"])
    )
    total = len(analysis["scores"])
    checkpoint = analysis.get("checkpoint_index")
    subject = (
        f"qualifying checkpoint {checkpoint}"
        if checkpoint is not None
        else "final trained model"
    )
    lines = [
        "=== Example 20: what joint pruning and compaction did ===",
        f"- JAX backend: {analysis.get('device', 'not recorded')}.",
        f"- Re-evaluated {subject}: {baseline}; required every task "
        f"to stay at or above {analysis['target']:.0%}.",
    ]
    if analysis["status"] == "baseline_below_target":
        lines.append(
            "- Pruning did not start because the unpruned final model was already "
            "below the requested target."
        )
    else:
        lines.append(
            f"- Initial fixed-ranking frontier: "
            f"{analysis.get('initial_frontier_retained', analysis['safe_retained'])}/"
            f"{total} neurons retained."
        )
        failed = analysis["first_failed_removed"]
        if failed is None:
            lines.append("- No evaluated lesion failed before the one-neuron endpoint.")
        else:
            lines.append(
                f"- The first failed checkpoint removed {failed} neurons; the "
                "previous contiguous checkpoints all met target."
            )
        if analysis["later_recovery"]:
            lines.append(
                "- Accuracy recovered at a later coarse checkpoint, but an "
                "intervening failure means that point is not called safely minimal."
            )
        fixed_point = analysis.get("fixed_point")
        if fixed_point is not None:
            pass_counts = ", ".join(
                str(count) for count in fixed_point["accepted_per_pass"]
            )
            edge_pass_counts = ", ".join(
                str(count) for count in fixed_point.get("edge_accepted_per_pass", [])
            )
            lines.extend(
                [
                    f"- Joint fixed point converged in "
                    f"{fixed_point.get('cycle_count', 1)} alternating cycle(s).",
                    f"- Neuron passes accepted removals: [{pass_counts}].",
                    f"- Edge passes accepted removals: [{edge_pass_counts}].",
                    f"- Joint locally minimal network: "
                    f"{analysis['safe_retained']}/{total} neurons "
                    f"({analysis['safe_removed']} removed total); final accuracy "
                    + ", ".join(
                        f"{name}={accuracy:.0%}"
                        for name, accuracy in zip(
                            names, fixed_point["final_accuracies"]
                        )
                    )
                    + ".",
                    f"- Retained neurons with exactly zero recomputed contribution "
                    f"score: {fixed_point['retained_zero_score_count']}.",
                    f"- Recurrent edges: {fixed_point.get('stored_edge_count', 0)} "
                    f"stored; {fixed_point.get('incident_edge_count', 0)} incident "
                    "to removed neurons; "
                    f"{fixed_point.get('final_original_live_live_edge_count', 0)} "
                    "original retained-to-retained; "
                    f"{fixed_point.get('causally_removed_live_live_edge_count', 0)} "
                    "of those causally removed; "
                    f"{fixed_point.get('final_active_edge_count', 0)} active.",
                ]
            )
        compaction = analysis.get("compaction")
        if compaction is not None and compaction["status"] == "complete":
            original_bytes = compaction["original_storage"]["total_bytes"]
            compact_bytes = compaction["compact_storage"]["total_bytes"]
            compact_accuracy = ", ".join(
                f"{name}={accuracy:.0%}"
                for name, accuracy in zip(names, compaction["accuracies"])
            )
            lines.extend(
                [
                    f"- Physical compact model: {compaction['n_rec']} neurons, "
                    f"{compaction['n_edges']} recurrent edges; {compact_accuracy}.",
                    f"- Masked/compact predictions are identical; maximum absolute "
                    f"logit error {compaction['max_abs_logit_error']:.3g}.",
                    f"- Persistent inference storage: {original_bytes:,} -> "
                    f"{compact_bytes:,} bytes "
                    f"({compaction['storage_reduction_fraction']:.1%} reduction).",
                    f"- Warmed full-probe timing over "
                    f"{compaction['benchmark_repetitions']} compiled repeats: "
                    f"masked={compaction['masked_probe_ms']:.3f} ms, "
                    f"compact={compaction['compact_probe_ms']:.3f} ms, "
                    f"speedup={compaction['speedup']:.2f}x.",
                ]
            )
            if compaction["bundle_path"] is not None:
                lines.append(
                    f"- Compact bundle: {compaction['bundle_path']} "
                    f"({compaction['bundle_bytes']:,} bytes)."
                )
            else:
                lines.append("- Compact bundle was verified but not saved.")
        elif compaction is not None:
            lines.append(
                f"- Physical compaction skipped: {compaction.get('reason', 'not eligible')}."
            )
    alignment = analysis["alignment"]
    removed_tasks = ", ".join(
        f"{name}={count}"
        for name, count in zip(names, alignment["removed_task_counts"])
    )
    retained_tasks = ", ".join(
        f"{name}={count}"
        for name, count in zip(names, alignment["retained_task_counts"])
    )
    lines.extend(
        [
            f"- Contribution ownership (initial for removed, recomputed for "
            f"retained), removed: {removed_tasks}; retained: {retained_tasks}.",
            f"- Exact-twin alignment: {alignment['removed_twin_neurons']} removed "
            "neurons belonged to non-singleton twin classes; "
            f"{alignment['fully_removed_twin_classes']} such classes were fully "
            f"removed and {alignment['partially_pruned_twin_classes']} partially "
            "pruned.",
            "- The terminal neuron and edge passes prove coordinate-wise local "
            "minimality only on the fixed probes; this is not a global minimum "
            "or generalization claim.",
        ]
    )
    return "\n".join(lines)


def _print_report(report: str) -> None:
    """Print a tagged pruning report."""
    print(f"[20-neuron-pruning]\n{report}", flush=True)


def _plot_pruning(analysis: Dict[str, Any], path: pathlib.Path) -> None:
    """Save contribution ranking and causal accuracy curve panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(analysis["scores"])
    owners = np.asarray(analysis["owners"])
    order = np.asarray(analysis.get("removal_order", _removal_order(scores)))
    counts = np.asarray(analysis["removed_counts"])
    accuracies = np.asarray(analysis["accuracies"])
    names = analysis["names"]
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    total = scores.size
    fig, (left, right, edges) = plt.subplots(1, 3, figsize=(14, 4.5))
    for task, name in enumerate(names):
        picked = owners[order] == task
        left.scatter(
            np.arange(total)[picked],
            scores[order][picked],
            s=12,
            color=colors[task % len(colors)],
            label=name,
        )
    initial_removed = analysis.get("initial_frontier_removed", analysis["safe_removed"])
    left.axvline(initial_removed, color="grey", linestyle=":", label="initial frontier")
    if analysis["safe_removed"] != initial_removed:
        left.axvline(
            analysis["safe_removed"],
            color="black",
            linestyle=":",
            label="fixed point count",
        )
    left.set_xlabel("neuron rank (least contributing first)")
    left.set_ylabel("normalized contribution score")
    left.set_title("Frozen-model contribution ranking")
    left.legend(fontsize=8)

    retained = total - counts
    for task, name in enumerate(names):
        right.plot(
            retained,
            accuracies[:, task],
            marker="o",
            color=colors[task % len(colors)],
            label=name,
        )
    fixed_counts = np.asarray(
        analysis.get("fixed_point_removed_counts", [initial_removed])
    )
    fixed_accuracies = np.asarray(
        analysis.get(
            "fixed_point_accuracies",
            [analysis["accuracies"][0]],
        )
    )
    fixed_retained = total - fixed_counts
    for task in range(len(names)):
        right.plot(
            fixed_retained,
            fixed_accuracies[:, task],
            color=colors[task % len(colors)],
            linewidth=2.5,
        )
    right.axhline(analysis["target"], color="black", linestyle="--", label="target")
    right.axvline(
        total - initial_removed,
        color="grey",
        linestyle=":",
        label="initial frontier",
    )
    if analysis["safe_retained"] != total - initial_removed:
        right.axvline(
            analysis["safe_retained"],
            color="black",
            linestyle=":",
            label="fixed point",
        )
    right.set_xscale("log", base=2)
    right.set_xlabel("retained recurrent neurons (log2 scale)")
    right.set_ylabel("fixed-probe accuracy")
    right.set_ylim(0.0, 1.02)
    right.set_title("Causal pruning curve")
    right.legend(fontsize=8)

    fixed_point = analysis.get("fixed_point", {})
    edge_labels = ("stored", "live-live", "active")
    edge_values = (
        fixed_point.get("stored_edge_count", 0),
        fixed_point.get("final_original_live_live_edge_count", 0),
        fixed_point.get("final_active_edge_count", 0),
    )
    edge_bars = edges.bar(
        edge_labels, edge_values, color=("0.65", "tab:blue", "tab:green")
    )
    edges.bar_label(edge_bars, fontsize=8)
    edges.set_ylabel("recurrent edges")
    edges.set_title("Joint structural compression")
    edges.set_ylim(bottom=0)
    fig.tight_layout()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    """Train Example 18, prune its evolved neurons, and report the result.

    Parameters
    ----------
    argv : list, optional
        Example 20 arguments plus any arguments accepted by Example 18. Uses
        ``sys.argv`` when omitted.

    Returns
    -------
    dict
        Example 18's result with a ``neuron_pruning`` analysis mapping.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prune-target", type=float, default=1.0)
    parser.add_argument("--prune-step-fraction", type=float, default=0.05)
    parser.add_argument(
        "--pruning-plot-output",
        type=pathlib.Path,
        default=pathlib.Path("neuron_pruning.png"),
    )
    parser.add_argument(
        "--compact-model-output",
        type=pathlib.Path,
        default=pathlib.Path("compacted_network.npz"),
    )
    parser.add_argument("--compact-benchmark-repetitions", type=int, default=10)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    args, forwarded = parser.parse_known_args(raw)
    if not math.isfinite(args.prune_target) or not 0.0 <= args.prune_target <= 1.0:
        parser.error("--prune-target must be finite and in [0, 1]")
    if args.compact_benchmark_repetitions < 1:
        parser.error("--compact-benchmark-repetitions must be positive")
    _coarse_removed_counts(1, args.prune_step_fraction)
    backend = _bind_device(args.device)
    has_task_style = any(
        value == "--task-style" or value.startswith("--task-style=")
        for value in forwarded
    )
    if not has_task_style:
        forwarded.extend(["--task-style", "temporal-credit", "--num-tricks", "4"])

    captured: Dict[str, Any] = {}

    def analyze_checkpoint(experiment, names, config, checkpoint, accuracies):
        if "analysis" in captured or min(accuracies) < args.prune_target:
            return
        analysis = _analyze_pruning(
            experiment,
            {"trick_names": names},
            config,
            args.prune_target,
            args.prune_step_fraction,
            compact_output=args.compact_model_output,
            benchmark_repetitions=args.compact_benchmark_repetitions,
        )
        analysis["checkpoint_index"] = checkpoint
        analysis["checkpoint_accuracies"] = list(accuracies)
        captured["analysis"] = analysis

    def analyze_final(experiment, arm, config):
        if "analysis" in captured:
            return
        analysis = _analyze_pruning(
            experiment,
            arm,
            config,
            args.prune_target,
            args.prune_step_fraction,
            compact_output=args.compact_model_output,
            benchmark_repetitions=args.compact_benchmark_repetitions,
        )
        analysis["checkpoint_index"] = None
        analysis["checkpoint_accuracies"] = None
        captured["analysis"] = analysis

    result = EX18.main(
        forwarded,
        evolve_posthoc=analyze_final,
        evolve_checkpoint=analyze_checkpoint,
    )
    if "analysis" not in captured:
        raise RuntimeError("Example 18 did not invoke the evolving-arm posthoc hook")
    analysis = captured["analysis"]
    analysis["device"] = backend
    plot_path = args.pruning_plot_output.resolve()
    _plot_pruning(analysis, plot_path)
    report = _format_report(analysis)
    _print_report(report)
    analysis["plot_path"] = str(plot_path)
    analysis["report"] = report
    result["neuron_pruning"] = analysis
    return result


if __name__ == "__main__":
    main()
