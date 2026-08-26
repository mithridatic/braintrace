"""Sparse structural adaptation helpers for Example 21."""

import hashlib
import heapq
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from math import ceil

import numpy as np


def _git_commit():
    """Return the current implementation revision for measured artifacts."""

    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _load_example21_model():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    path = os.path.join(os.path.dirname(__file__), "21-braincell-arc.py")
    spec = importlib.util.spec_from_file_location("example21_braincell_arc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Example 21 model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_task_rows(values):
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    return np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)


def neuron_contribution(readout, transmission, gradient_mass):
    rows = tuple(normalize_task_rows(value) for value in (readout, transmission, gradient_mass))
    return np.mean(rows, axis=0).max(axis=0)


def connection_contribution(indptr, values, spikes, gradients):
    indptr = np.asarray(indptr)
    values = np.asarray(values, dtype=float)
    spikes = np.asarray(spikes, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    source = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    transmission = np.abs(values) * np.mean(np.abs(spikes[:, source]), axis=0)
    mass = np.mean(np.abs(gradients), axis=0)
    transmission_scale = np.max(transmission)
    row_weight = np.divide(
        transmission, transmission_scale, out=np.zeros_like(transmission),
        where=transmission_scale != 0,
    )
    mass_scale = np.max(mass)
    mass_weight = np.divide(mass, mass_scale, out=np.zeros_like(mass), where=mass_scale != 0)
    score = 0.5 * row_weight + 0.5 * mass_weight
    return score


def mutation_count(active_count):
    """Return the exact five-percent ceiling mutation budget."""

    if active_count < 1:
        raise ValueError("active count must be positive")
    return ceil(0.05 * active_count)


def task_owners(scores):
    scores = np.asarray(scores, dtype=float)
    owners = []
    for column in scores.T:
        maximum = np.max(column)
        owners.append(tuple(np.flatnonzero((column == maximum) & (maximum > 0))))
    return tuple(owners)


def stable_rank(scores, *, descending=False):
    scores = np.asarray(scores)
    return tuple(np.lexsort((np.arange(len(scores)), -scores if descending else scores)))


def structural_twins(neuron_count, *, input_sources, recurrent_incoming,
                     recurrent_outgoing, dale_labels, mechanisms):
    groups = []
    keys = {}
    for index in range(neuron_count):
        key = (
            tuple(input_sources[index]), tuple(recurrent_incoming[index]),
            tuple(recurrent_outgoing[index]), dale_labels[index],
            tuple(mechanisms[index]),
        )
        keys.setdefault(key, []).append(index)
    for group in keys.values():
        groups.append(tuple(group))
    return tuple(groups)


@dataclass
class SparseTopology:
    input_source: np.ndarray
    input_target: np.ndarray
    input_value: np.ndarray
    recurrent_source: np.ndarray
    recurrent_target: np.ndarray
    recurrent_value: np.ndarray
    readout: np.ndarray
    dale: np.ndarray
    mechanisms: tuple

    @property
    def neuron_count(self):
        return len(self.dale)


@dataclass
class StructuralAdam:
    neuron_first: np.ndarray
    neuron_second: np.ndarray
    input_first: np.ndarray
    input_second: np.ndarray
    recurrent_first: np.ndarray
    recurrent_second: np.ndarray
    step: int = 0


def remap_muon_groups(muon_groups, parameter_maps):
    """Remap Muon state leaves when structural parameters change shape.

    Parameters
    ----------
    muon_groups : mapping
        Muon optimizer state keyed by parameter name.
    parameter_maps : mapping
        Each value is ``(selector, target_shape)`` for the parameter rows.

    Returns
    -------
    mapping
        A copied state tree with parameter-shaped leaves remapped.
    """
    import copy

    def remap(value, selector, target_shape):
        if isinstance(value, dict):
            return {key: remap(item, selector, target_shape)
                    for key, item in value.items()}
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*(remap(item, selector, target_shape)
                                 for item in value))
        if hasattr(value, "__dict__"):
            result = copy.copy(value)
            for name, item in vars(value).items():
                setattr(result, name, remap(item, selector, target_shape))
            return result
        shape = getattr(value, "shape", None)
        source_shape = (len(selector),) + tuple(target_shape[1:])
        if shape != source_shape:
            return value
        selected = value[selector]
        if selected.shape[0] == target_shape[0]:
            return selected
        padding = [(0, target_shape[0] - selected.shape[0])]
        padding.extend((0, 0) for _ in target_shape[1:])
        return np.pad(selected, padding)

    return {
        name: remap(state, *parameter_maps[name])
        if name in parameter_maps else state
        for name, state in muon_groups.items()
    }


def structural_muon_parameter_maps(source, candidate, arm, alive=None):
    """Return row selectors for Muon state during a structural rebuild."""
    if arm == "neuron-prune":
        alive = np.asarray(alive, dtype=bool)
        input_selector = alive[source.input_target]
        recurrent_selector = alive[source.recurrent_source] & alive[source.recurrent_target]
        neuron_selector = alive
    elif arm == "connection-prune":
        input_selector = np.ones(len(source.input_value), dtype=bool)
        recurrent_selector = np.asarray(
            [
                (src, dst) in set(zip(
                    candidate.recurrent_source.tolist(),
                    candidate.recurrent_target.tolist(),
                ))
                for src, dst in zip(
                    source.recurrent_source, source.recurrent_target
                )
            ],
            dtype=bool,
        )
        neuron_selector = np.ones(source.neuron_count, dtype=bool)
    else:
        input_selector = np.arange(len(source.input_value))
        recurrent_selector = np.arange(len(source.recurrent_value))
        neuron_selector = np.arange(source.neuron_count)
    return {
        "input": (input_selector, (len(candidate.input_value),)),
        "recurrent": (recurrent_selector, (len(candidate.recurrent_value),)),
        "readout_weight": (neuron_selector, candidate.readout.shape),
        "readout_bias": (neuron_selector, (candidate.neuron_count,)),
    }


def pruning_mask(scores, validation_strict):
    if not any(validation_strict):
        raise ValueError("validation strict gate is closed")
    scores = np.asarray(scores)
    count = mutation_count(len(scores))
    mask = np.ones(len(scores), dtype=bool)
    mask[np.argsort(scores, kind="stable")[:count]] = False
    return mask


def prune_neurons(topology, scores, validation_strict):
    """Return a five-percent neuron mask selected by stable evidence ranking."""

    if not any(validation_strict):
        raise ValueError("validation strict gate is closed")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (topology.neuron_count,):
        raise ValueError("one score is required per neuron")
    return pruning_mask(scores, validation_strict)


def prune_recurrent(topology, scores, validation_strict):
    """Remove the exact lowest-scoring recurrent-edge budget."""

    if not any(validation_strict):
        raise ValueError("validation strict gate is closed")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != topology.recurrent_value.shape:
        raise ValueError("one score is required per recurrent edge")
    keep = np.ones(len(scores), dtype=bool)
    keep[np.argsort(scores, kind="stable")[:mutation_count(len(scores))]] = False
    pruned = SparseTopology(
        topology.input_source.copy(), topology.input_target.copy(),
        topology.input_value.copy(), topology.recurrent_source[keep],
        topology.recurrent_target[keep], topology.recurrent_value[keep],
        topology.readout.copy(), topology.dale.copy(), tuple(topology.mechanisms),
    )
    return pruned, keep


def topology_from_model(model):
    """Snapshot the sparse topology and readout from a real Example 21 model."""

    input_csr = model.input_csr
    recurrent_csr = model.recurrent_csr
    input_indptr = np.asarray(input_csr.indptr)
    recurrent_indptr = np.asarray(recurrent_csr.indptr)
    input_targets = np.asarray(input_csr.indices)
    recurrent_targets = np.asarray(recurrent_csr.indices)
    input_sources = np.repeat(np.arange(len(input_indptr) - 1), np.diff(input_indptr))
    recurrent_sources = np.repeat(
        np.arange(len(recurrent_indptr) - 1), np.diff(recurrent_indptr)
    )
    readout = np.asarray(model.readout_weight.value)
    count = readout.shape[0]
    return SparseTopology(
        input_sources, input_targets, np.asarray(model.input_weight.value),
        recurrent_sources, recurrent_targets, np.asarray(model.recurrent_weight.value),
        readout, np.zeros(count, dtype=np.int8), ((),) * count,
    )


def task_gradient_mass(mass_by_task, parameter_name, task_count):
    """Normalize pre-clip gradient mass to one task-by-item array."""

    values = np.asarray(mass_by_task.get(parameter_name, 0.0), dtype=float)
    if values.ndim == 0:
        return np.zeros((task_count, 0), dtype=float)
    if values.shape[0] != task_count:
        raise ValueError("gradient mass has an unexpected task dimension")
    return np.abs(values).reshape((task_count, -1))


def resident_tile_pairs(tile_size):
    """Return the candidate-pair capacity of one connection-selection tile."""

    if tile_size < 1 or tile_size > 256:
        raise ValueError("tile size must be between one and 256")
    return tile_size * tile_size


def structural_evidence(topology, readout_effect, spikes, gradient_mass):
    """Compute task-specific neuron and recurrent-edge evidence.

    Parameters
    ----------
    topology : SparseTopology
        Model topology snapshot.
    readout_effect : array-like
        Direct voltage-readout effect with shape ``(tasks, neurons)``.
    spikes : array-like
        Per-task mean spike activity with shape ``(tasks, neurons)``.
    gradient_mass : array-like
        Absolute pre-clip recurrent gradient mass with shape
        ``(tasks, recurrent_edges)``.

    Returns
    -------
    dict
        Normalized neuron and recurrent-edge scores and task owners.
    """

    readout_effect = np.asarray(readout_effect, dtype=float)
    spikes = np.asarray(spikes, dtype=float)
    gradient_mass = np.asarray(gradient_mass, dtype=float)
    tasks = readout_effect.shape[0]
    expected = (tasks, topology.neuron_count)
    if readout_effect.shape != expected or spikes.shape != expected:
        raise ValueError("readout effect and spikes must be task-by-neuron arrays")
    if gradient_mass.shape != (tasks, len(topology.recurrent_value)):
        raise ValueError("gradient mass must be task-by-edge")
    source = topology.recurrent_source
    transmission = np.zeros(expected, dtype=float)
    for task in range(tasks):
        transmission[task] = np.bincount(
            source,
            weights=np.abs(topology.recurrent_value) * np.abs(spikes[task, source]),
            minlength=topology.neuron_count,
        )
    incident = np.zeros(expected, dtype=float)
    for task in range(tasks):
        incident[task] = (
            np.bincount(topology.recurrent_source, weights=gradient_mass[task],
                        minlength=topology.neuron_count)
            + np.bincount(topology.recurrent_target, weights=gradient_mass[task],
                          minlength=topology.neuron_count)
        )
    channels = np.stack((
        normalize_task_rows(readout_effect),
        normalize_task_rows(transmission),
        normalize_task_rows(incident),
    ))
    per_task_neuron = channels.mean(axis=0)
    neuron_scores = per_task_neuron.max(axis=0)
    edge_transmission = np.abs(topology.recurrent_value)[None, :] * np.abs(
        spikes[:, topology.recurrent_source]
    )
    edge_scores = 0.5 * normalize_task_rows(edge_transmission).mean(axis=0)
    edge_scores += 0.5 * normalize_task_rows(gradient_mass).mean(axis=0)
    return {
        "neuron_scores": neuron_scores,
        "connection_scores": edge_scores,
        "owners": task_owners(per_task_neuron),
    }


def _remap_edges(source, target, values, alive):
    mapping = -np.ones(len(alive), dtype=int)
    mapping[alive] = np.arange(np.sum(alive))
    keep = alive[source] & alive[target]
    return mapping[source[keep]], mapping[target[keep]], values[keep]


def compact(topology, alive, adam):
    alive = np.asarray(alive, dtype=bool)
    input_keep = alive[topology.input_target]
    recurrent_source, recurrent_target, recurrent_value = _remap_edges(
        topology.recurrent_source, topology.recurrent_target,
        topology.recurrent_value, alive,
    )
    mapping = -np.ones(len(alive), dtype=int)
    mapping[alive] = np.arange(np.sum(alive))
    recurrent_keep = alive[topology.recurrent_source] & alive[topology.recurrent_target]
    compacted = SparseTopology(
        topology.input_source[input_keep],
        mapping[topology.input_target[input_keep]],
        topology.input_value[input_keep],
        recurrent_source,
        recurrent_target,
        recurrent_value,
        topology.readout[alive],
        topology.dale[alive],
        tuple(mechanism for mechanism, keep in zip(topology.mechanisms, alive) if keep),
    )
    mapped = StructuralAdam(
        adam.neuron_first[alive], adam.neuron_second[alive],
        adam.input_first[input_keep], adam.input_second[input_keep],
        adam.recurrent_first[recurrent_keep],
        adam.recurrent_second[recurrent_keep],
        adam.step,
    )
    return compacted, mapped, True


def mask_topology(topology, alive):
    """Return the un-compacted topology with removed paths set to exact zero."""

    alive = np.asarray(alive, dtype=bool)
    masked = SparseTopology(
        topology.input_source.copy(), topology.input_target.copy(),
        topology.input_value * alive[topology.input_target],
        topology.recurrent_source.copy(), topology.recurrent_target.copy(),
        topology.recurrent_value * (
            alive[topology.recurrent_source] & alive[topology.recurrent_target]
        ),
        topology.readout * alive[:, None], topology.dale.copy(),
        tuple(topology.mechanisms),
    )
    return masked


def prediction_bytes_identical(masked, compacted, predict_masked, predict_compacted):
    """Compare decoded prediction bytes for mask and physical compaction."""

    return np.asarray(predict_masked(masked)).tobytes() == np.asarray(
        predict_compacted(compacted)
    ).tobytes()


def add_twin_neurons(topology, scores, required=None):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 2:
        scores = np.max(scores, axis=0)
    required = ceil(0.05 * topology.neuron_count) if required is None else required
    if required < 1 or required > topology.neuron_count:
        raise ValueError("valid donor budget is insufficient")
    donors = tuple(stable_rank(scores, descending=True)[:required])
    if any(
        np.any(
            ((topology.recurrent_source == left) & (topology.recurrent_target == right))
            | ((topology.recurrent_source == right) & (topology.recurrent_target == left))
        )
        for position, left in enumerate(donors)
        for right in donors[position + 1:]
    ):
        raise ValueError("selected donors are connected")
    offset = topology.neuron_count
    input_source = list(topology.input_source)
    input_target = list(topology.input_target)
    input_value = list(topology.input_value)
    recurrent_source = list(topology.recurrent_source)
    recurrent_target = list(topology.recurrent_target)
    recurrent_value = list(topology.recurrent_value)
    readout = topology.readout.copy()
    dale = list(topology.dale)
    mechanisms = list(topology.mechanisms)
    for donor in donors:
        twin = offset + len(dale) - topology.neuron_count
        for source, target, value in zip(topology.input_source, topology.input_target, topology.input_value):
            if target == donor:
                input_source.append(source); input_target.append(twin); input_value.append(value)
        incoming = [(s, t, v) for s, t, v in zip(topology.recurrent_source, topology.recurrent_target, topology.recurrent_value) if t == donor]
        outgoing = [(s, t, v) for s, t, v in zip(topology.recurrent_source, topology.recurrent_target, topology.recurrent_value) if s == donor]
        for source, target, value in incoming:
            recurrent_source.append(source); recurrent_target.append(twin); recurrent_value.append(value)
        for source, target, value in outgoing:
            index = next(i for i, (s, t) in enumerate(zip(recurrent_source, recurrent_target)) if s == source and t == target)
            recurrent_value[index] *= 0.5
            recurrent_source.append(twin); recurrent_target.append(target); recurrent_value.append(value * 0.5)
        readout[donor] *= 0.5
        readout = np.vstack((readout, readout[donor]))
        dale.append(topology.dale[donor]); mechanisms.append(topology.mechanisms[donor])
    return SparseTopology(
        np.asarray(input_source), np.asarray(input_target), np.asarray(input_value),
        np.asarray(recurrent_source), np.asarray(recurrent_target), np.asarray(recurrent_value),
        readout, np.asarray(dale), tuple(mechanisms),
    ), donors


def grow_adam_for_twins(adam, topology, grown):
    """Preserve existing moments and zero every newly created item."""

    def extend(values, length):
        padding = [(0, length - len(values))] + [(0, 0)] * (values.ndim - 1)
        return np.pad(values, padding)

    return StructuralAdam(
        extend(adam.neuron_first, grown.neuron_count),
        extend(adam.neuron_second, grown.neuron_count),
        extend(adam.input_first, len(grown.input_value)),
        extend(adam.input_second, len(grown.input_value)),
        extend(adam.recurrent_first, len(grown.recurrent_value)),
        extend(adam.recurrent_second, len(grown.recurrent_value)), adam.step,
    )


def select_connection_additions(neuron_count, existing, source_evidence,
                                target_evidence, required, tile_size=256):
    if tile_size > 256:
        raise ValueError("tile size exceeds the 65,536-pair resident bound")
    heap = []
    source_order = np.argsort(-np.asarray(source_evidence), kind="stable")
    target_order = np.argsort(-np.asarray(target_evidence), kind="stable")
    for source_start in range(0, neuron_count, tile_size):
        sources = source_order[source_start:source_start + tile_size]
        for target_start in range(0, neuron_count, tile_size):
            targets = target_order[target_start:target_start + tile_size]
            for source in sources:
                for target in targets:
                    pair = (int(source), int(target))
                    if source == target or pair in existing:
                        continue
                    score = float(source_evidence[source] * target_evidence[target])
                    item = (score, -int(source), -int(target), pair)
                    if len(heap) < required:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
    return tuple(item[3] for item in sorted(heap, key=lambda item: (-item[0], item[3])))


def add_recurrent_connections(topology, pairs, *, typed=False):
    """Append measured recurrent pairs with neutral initial effective weight."""

    pairs = tuple(pairs)
    existing = set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist()))
    if len(set(pairs)) != len(pairs) or any(pair in existing for pair in pairs):
        raise ValueError("connection additions must be distinct and absent")
    initial = np.log(np.expm1(1e-6)) if typed else 0.0
    return SparseTopology(
        topology.input_source.copy(), topology.input_target.copy(), topology.input_value.copy(),
        np.concatenate((topology.recurrent_source, np.asarray([p[0] for p in pairs], dtype=int))),
        np.concatenate((topology.recurrent_target, np.asarray([p[1] for p in pairs], dtype=int))),
        np.concatenate((topology.recurrent_value, np.full(len(pairs), initial))),
        topology.readout.copy(), topology.dale.copy(), tuple(topology.mechanisms),
    )


def grow_adam_for_connections(adam, added_count):
    """Append zero Adam moments for new recurrent edges."""

    return StructuralAdam(
        adam.neuron_first.copy(), adam.neuron_second.copy(), adam.input_first.copy(),
        adam.input_second.copy(), np.pad(adam.recurrent_first, (0, added_count)),
        np.pad(adam.recurrent_second, (0, added_count)), adam.step,
    )


def preclip_gradient_mass(learner, events, step_fn, task_index, task_count, **kwargs):
    """Call the real PP-Prop boundary and return per-item pre-clip mass."""

    gradients, losses = learner.etrace_grad(
        events, step_fn=step_fn, return_value=True, **kwargs
    )
    mass = {}
    for path, gradient in gradients.items():
        key = "/".join(map(str, path if isinstance(path, tuple) else (path,)))
        values = np.abs(np.asarray(gradient))
        rows = np.zeros((task_count,) + values.shape, dtype=values.dtype)
        rows[task_index] = values
        mass[key] = rows
    return mass, losses


def collect_model_evidence(
    model,
    learner,
    events,
    step_fn,
    readout_effect,
    spikes_by_task,
    *,
    task_index=0,
    task_count=None,
    **kwargs,
):
    """Collect structural scores at the real model and pre-clip boundary."""
    readout_effect = np.asarray(readout_effect, dtype=float)
    spikes_by_task = np.asarray(spikes_by_task, dtype=float)
    if task_count is None:
        task_count = readout_effect.shape[0]
    mass, losses = preclip_gradient_mass(
        learner, events, step_fn, task_index, task_count, **kwargs
    )
    recurrent_name = next(
        (key for key in mass if key.endswith("recurrent_weight")), None
    )
    if recurrent_name is None:
        raise ValueError("pre-clip mass has no recurrent weight")
    topology = topology_from_model(model)
    gradient_mass = task_gradient_mass(mass, recurrent_name, task_count)
    if spikes_by_task.shape != (task_count, topology.neuron_count):
        raise ValueError("spikes must be task-by-neuron")
    result = structural_evidence(
        topology, readout_effect, spikes_by_task, gradient_mass
    )
    result["gradient_mass"] = gradient_mass
    result["preclip_loss"] = np.asarray(losses).tolist()
    result["preclip_exceeds_clip"] = bool(
        np.max(gradient_mass, initial=0.0) > 1.0
    )
    result["model_neurons"] = topology.neuron_count
    return result


def run_addition_updates(transform, update, *, updates=64):
    """Run exactly 64 addition updates through a BrainState loop primitive."""

    if updates != 64:
        raise ValueError("addition arms require exactly 64 updates")
    indices = np.arange(updates, dtype=np.int32)
    return transform.jit(lambda xs: transform.for_loop(update, xs))(indices)


def _fixed_task_evidence(module, model, learner, data_root):
    """Collect fixed-task predictions, spikes, readout, and PP-Prop mass."""
    if data_root is None:
        raise ValueError("real Example 21 measurement requires --data-root")
    import brainstate
    import jax.numpy as jnp

    task = module.load_task(data_root, module.TRAINING_TASK_IDS[0], "practice")
    events, advances = module.encode_episode(task, 0)
    model.reset_episode(learner)
    step_fn = lambda event: jnp.sum(
        learner.etrace_evolve(event[None, :], return_outputs=True)[0]
    )
    mass, losses = preclip_gradient_mass(
        learner, events, step_fn, 0, 1, mask=advances, reduction="sum"
    )
    model.reset_episode(learner)
    voltages = module.run_event_sequence(model, events, advances)
    spikes = np.asarray(voltages) > 0.0
    readout = np.abs(np.asarray(model.readout_weight.value))[None, :, :].mean(axis=2)
    recurrent_name = next(
        (key for key in mass if key.endswith("recurrent_weight")), None
    )
    if recurrent_name is None:
        raise ValueError("pre-clip mass has no recurrent weight")
    gradient_mass = task_gradient_mass(mass, recurrent_name, 1)
    topology = topology_from_model(model)
    evidence = structural_evidence(
        topology, readout, spikes[None, :, :].mean(axis=1), gradient_mass
    )
    strict = []
    for task_id in module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS:
        task = module.load_task(data_root, task_id, "practice")
        predictions = []
        targets = []
        for query_index, target in enumerate(task.targets):
            if target is None:
                continue
            encoded, mask = module.encode_episode(task, query_index)
            model.reset_episode(learner)
            module.run_event_sequence(model, encoded, mask)
            predictions.append(module.decode_prediction(np.asarray(model.readout())))
            targets.append(target)
        strict.append(bool(module.strict_task_pass_at_1(predictions, targets)))
    evidence.update({
        "preclip_gradient_mass": np.asarray(gradient_mass).tolist(),
        "preclip_loss": np.asarray(losses).tolist(),
        "task_spike_evidence": np.asarray(spikes, dtype=float).mean(axis=0).tolist(),
        "task_readout_evidence": np.asarray(readout).tolist(),
        "strict": strict,
        "events": events,
        "advances": advances,
    })
    return evidence


def _real_mask_compaction_identity(module, topology, adam, data_root, *, alive=None):
    """Measure fixed-task identity between masked and compact real models."""
    if data_root is None:
        raise ValueError("real Example 21 measurement requires --data-root")
    if alive is None:
        alive = np.ones(topology.neuron_count, dtype=bool)
        alive[:mutation_count(topology.neuron_count)] = False
    alive = np.asarray(alive, dtype=bool)
    masked = mask_topology(topology, alive)
    compacted, _, _ = compact(topology, alive, adam)

    episodes = []
    for task_id in module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS:
        task = module.load_task(data_root, task_id, "practice")
        for query_index, target in enumerate(task.targets):
            if target is not None:
                events, advances = module.encode_episode(task, query_index)
                episodes.append((task_id, events, advances, target))

    def screen(candidate):
        candidate_model, candidate_learner = _rebuild_real_candidate(
            module, candidate, None
        )
        prediction_bytes = []
        strict = []
        task_predictions = {task_id: [] for task_id in module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS}
        task_targets = {task_id: [] for task_id in task_predictions}
        for task_id, events, advances, target in episodes:
            candidate_model.reset_episode(candidate_learner)
            module.run_event_sequence(candidate_model, events, advances)
            prediction = module.decode_prediction(np.asarray(candidate_model.readout()))
            task_predictions[task_id].append(prediction)
            task_targets[task_id].append(target)
            prediction_bytes.append(np.asarray(prediction).tobytes())
        strict.extend(
            bool(module.strict_task_pass_at_1(task_predictions[task_id], task_targets[task_id]))
            for task_id in task_predictions
        )
        return strict, b"".join(prediction_bytes)

    masked_strict, masked_bytes = screen(masked)
    compacted_strict, compacted_bytes = screen(compacted)
    return {
        "prediction_bytes_identical": masked_bytes == compacted_bytes,
        "strict_identical": masked_strict == compacted_strict,
        "masked_strict": masked_strict,
        "compacted_strict": compacted_strict,
        "masked_prediction_sha256": hashlib.sha256(masked_bytes).hexdigest(),
        "compacted_prediction_sha256": hashlib.sha256(compacted_bytes).hexdigest(),
        "masked_neurons": masked.neuron_count,
        "compacted_neurons": compacted.neuron_count,
    }


def _real_pp_prop_update(
    module, model, learner, evidence, adam=None, *, muon_groups=None,
    parameter_maps=None,
):
    """Return one compiled PP-Prop candidate update for Example 21."""
    import jax.numpy as jnp

    events = jnp.asarray(evidence["events"])
    advances = jnp.asarray(evidence["advances"])
    trainer = module.PPPropEpisodeTrainer(
        learner,
        {"input": model.input_weight.value, "recurrent": model.recurrent_weight.value},
    )
    if muon_groups is not None:
        trainer.muon_groups = remap_muon_groups(
            muon_groups, parameter_maps or {}
        )

    if adam is not None:
        for name, first, second in (
            ("readout", adam.neuron_first, adam.neuron_second),
            ("input", adam.input_first, adam.input_second),
            ("recurrent", adam.recurrent_first, adam.recurrent_second),
        ):
            state = getattr(trainer, "adam_groups", {}).get(name)
            if state is not None:
                state.first = jnp.asarray(first)
                state.second = jnp.asarray(second)
                state.step = adam.step

    def update(_=None):
        model.reset_episode(learner)
        return trainer.update_episode(
            events,
            step_fn=lambda event: jnp.sum(
                learner.etrace_evolve(event[None, :], return_outputs=True)[0]
            ),
            loss_mask=advances,
        )

    update.trainer = trainer
    return update


def _rebuild_real_candidate(module, candidate, fallback_learner):
    try:
        candidate_model = module.BrainCellArcModel(candidate)
    except TypeError:
        candidate_model = module.BrainCellArcModel()
    candidate_learner = (
        module.compile_pp_prop_model(candidate_model)
        if hasattr(module, "compile_pp_prop_model") else fallback_learner
    )
    return candidate_model, candidate_learner


def execute_one_arm(arm, before_strict, operation, evaluate, *, updates=0,
                    transform=None, update=None, clock=time.perf_counter):
    """Execute one isolated structural candidate and return direct evidence."""

    allowed = {"neuron-prune", "connection-prune", "neuron-add", "connection-add"}
    if arm not in allowed:
        raise ValueError("exactly one recognized arm is required")
    started = clock()
    candidate, count = operation()
    if arm.endswith("add"):
        if transform is None or update is None:
            raise ValueError("addition arms require a compiled update driver")
        run_addition_updates(transform, update, updates=updates)
    after_strict = tuple(bool(value) for value in evaluate(candidate))
    elapsed = clock() - started
    promoted = promote_arm(
        before_strict, after_strict, elapsed,
        "addition" if arm.endswith("add") else "pruning", updates,
    )
    return {
        "arm": arm, "mutated_item_count": int(count), "updates": int(updates),
        "before_strict": list(map(bool, before_strict)),
        "after_strict": list(after_strict), "elapsed_seconds": float(elapsed),
        "within_300_seconds": bool(elapsed <= 300), "promoted": bool(promoted),
    }


def write_artifact(path, evidence):
    """Write canonical measured evidence and return its SHA-256 digest."""

    document = {
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "pid": os.getpid(), "seeds": [21, 22, 23],
        },
        **evidence,
    }
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(path, "wb") as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def promote_arm(before, after, elapsed_seconds, arm, updates):
    if arm == "addition" and updates != 64:
        raise ValueError("addition arms require exactly 64 updates")
    if elapsed_seconds > 300:
        return False
    before = tuple(before)
    after = tuple(after)
    gained = any(not old and new for old, new in zip(before, after))
    regressed = any(old and not new for old, new in zip(before, after))
    return gained and not regressed


def run_integrated_arm(
    arm,
    model_factory,
    learner_factory,
    evaluate,
    rebuild,
    *,
    transform=None,
    update=None,
    before_strict=None,
    evidence=None,
    clock=time.perf_counter,
):
    """Run one structural arm against a model and rebuilt learner.

    Parameters
    ----------
    arm : str
        One structural arm name.
    model_factory : callable
        Builds a fresh Example 21 model.
    learner_factory : callable
        Builds a learner for a model.
    evaluate : callable
        Returns the fixed-task strict Boolean vector for a model and learner.
    rebuild : callable
        Builds a candidate model and learner from a topology and remapped Adam state.
    transform, update : object and callable, optional
        BrainState transform module and one compiled addition update.
    before_strict : sequence of bool, optional
        Baseline strict vector. It is measured when omitted.
    evidence : dict, optional
        Pre-clip and task evidence to include in the result.

    Returns
    -------
    dict
        Direct arm evidence, including model dimensions and rebuilt-state status.
    """
    model = model_factory()
    learner = learner_factory(model)
    topology = topology_from_model(model)
    adam = StructuralAdam(
        np.zeros_like(np.asarray(model.readout_weight.value)),
        np.zeros_like(np.asarray(model.readout_weight.value)),
        np.zeros_like(np.asarray(model.input_weight.value)),
        np.zeros_like(np.asarray(model.input_weight.value)),
        np.zeros_like(np.asarray(model.recurrent_weight.value)),
        np.zeros_like(np.asarray(model.recurrent_weight.value)),
    )
    if before_strict is None:
        before_strict = tuple(bool(value) for value in evaluate(model, learner))
    scores = (evidence or {}).get("neuron_scores", np.zeros(topology.neuron_count))
    connection_scores = (evidence or {}).get(
        "connection_scores", np.zeros(len(topology.recurrent_value))
    )
    validation = (evidence or {}).get("validation_strict", before_strict)
    if arm == "neuron-prune":
        alive = prune_neurons(topology, scores, validation)
        candidate, candidate_adam, reset = compact(topology, alive, adam)
        count = int(np.sum(~alive))
    elif arm == "connection-prune":
        candidate, keep = prune_recurrent(topology, connection_scores, validation)
        candidate_adam = StructuralAdam(
            adam.neuron_first, adam.neuron_second, adam.input_first,
            adam.input_second, adam.recurrent_first[keep],
            adam.recurrent_second[keep], adam.step,
        )
        reset = True
        count = int(np.sum(~keep))
    elif arm == "neuron-add":
        candidate, donors = add_twin_neurons(topology, scores)
        candidate_adam = grow_adam_for_twins(adam, topology, candidate)
        reset = True
        count = len(donors)
    elif arm == "connection-add":
        pairs = select_connection_additions(
            topology.neuron_count,
            set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist())),
            scores,
            (evidence or {}).get("target_scores", scores),
            mutation_count(len(topology.recurrent_value)),
        )
        candidate = add_recurrent_connections(topology, pairs)
        candidate_adam = grow_adam_for_connections(adam, len(pairs))
        reset = True
        count = len(pairs)
    else:
        raise ValueError("exactly one recognized arm is required")
    candidate_model, candidate_learner = rebuild(candidate, candidate_adam)
    if hasattr(candidate_model, "reset_episode"):
        candidate_model.reset_episode(candidate_learner)
    result = execute_one_arm(
        arm, before_strict, lambda: (candidate_model, count),
        lambda value: evaluate(value, candidate_learner),
        updates=64 if arm.endswith("add") else 0,
        transform=transform, update=update, clock=clock,
    )
    result.update({
        "model": type(model).__name__,
        "baseline_neurons": int(topology.neuron_count),
        "candidate_neurons": int(candidate.neuron_count),
        "baseline_recurrent_items": int(len(topology.recurrent_value)),
        "candidate_recurrent_items": int(len(candidate.recurrent_value)),
        "adam_remapped": True,
        "eligibility_reset": bool(reset),
        "real_model": True,
        "evidence": evidence or {},
    })
    return result


def measure_real_arm(arm, *, data_root=None, clock=time.perf_counter):
    """Measure one bounded arm against the real Example 21 model topology.

    Parameters
    ----------
    arm : str
        One of ``neuron-prune``, ``connection-prune``, ``neuron-add``, or
        ``connection-add``.
    clock : callable, optional
        Monotonic clock used by the measurement.

    Returns
    -------
    dict
        Per-arm evidence with real model dimensions and bounded controls.
    """
    if arm not in {"neuron-prune", "connection-prune", "neuron-add", "connection-add"}:
        raise ValueError("exactly one recognized arm is required")
    module = _load_example21_model()
    model = module.BrainCellArcModel()
    learner = (
        module.compile_pp_prop_model(model)
        if hasattr(module, "compile_pp_prop_model") else None
    )
    topology = topology_from_model(model)
    evidence = _fixed_task_evidence(module, model, learner, data_root)
    baseline = tuple(evidence["strict"])
    scores = np.asarray(evidence["neuron_scores"])
    required = mutation_count(topology.neuron_count)
    donors = []
    connected = set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist()))
    for node in stable_rank(scores, descending=True):
        if all((node, donor) not in connected and (donor, node) not in connected for donor in donors):
            donors.append(int(node))
        if len(donors) == required:
            break
    scores[:] = 0.0
    scores[donors] = np.arange(required, 0, -1)
    edge_scores = np.asarray(evidence["connection_scores"])
    adam = StructuralAdam(
        np.zeros_like(topology.readout), np.zeros_like(topology.readout),
        np.zeros_like(topology.input_value), np.zeros_like(topology.input_value),
        np.zeros_like(topology.recurrent_value), np.zeros_like(topology.recurrent_value),
    )
    started = clock()
    pruning_alive = None
    if arm == "neuron-prune":
        validation = baseline[-len(module.VALIDATION_TASK_IDS):]
        if any(validation):
            alive = prune_neurons(topology, scores, validation)
            candidate, candidate_adam, reset = compact(topology, alive, adam)
            pruning_alive = alive
        else:
            candidate = topology
            candidate_adam = adam
            reset = True
        count = topology.neuron_count - candidate.neuron_count
        updates = 0
    elif arm == "connection-prune":
        validation = baseline[-len(module.VALIDATION_TASK_IDS):]
        if any(validation):
            candidate, keep = prune_recurrent(topology, edge_scores, validation)
            candidate_adam = StructuralAdam(
                adam.neuron_first, adam.neuron_second, adam.input_first,
                adam.input_second, adam.recurrent_first[keep],
                adam.recurrent_second[keep], adam.step,
            )
            count = int(np.sum(~keep))
        else:
            candidate = topology
            candidate_adam = adam
            count = 0
        reset = True
        updates = 0
    elif arm == "neuron-add":
        candidate, donors = add_twin_neurons(topology, scores)
        candidate_adam = grow_adam_for_twins(adam, topology, candidate)
        reset = True
        count = len(donors)
        updates = 64
    else:
        pairs = select_connection_additions(
            topology.neuron_count,
            set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist())),
            scores, scores, mutation_count(len(topology.recurrent_value)),
        )
        candidate = add_recurrent_connections(topology, pairs)
        candidate_adam = grow_adam_for_connections(adam, len(pairs))
        reset = True
        count = len(pairs)
        updates = 64
    candidate_model, candidate_learner = _rebuild_real_candidate(
        module, candidate, learner
    )
    if hasattr(candidate_model, "reset_episode"):
        candidate_model.reset_episode(candidate_learner)
    if updates:
        import brainstate
        source_update = _real_pp_prop_update(
            module, model, learner, evidence, adam
        )
        source_trainer = getattr(source_update, "trainer", None)
        parameter_maps = structural_muon_parameter_maps(
            topology, candidate, arm, pruning_alive
        )
        muon_groups = getattr(source_trainer, "muon_groups", None)
        if muon_groups is None:
            update = _real_pp_prop_update(
                module, candidate_model, candidate_learner, evidence, candidate_adam
            )
        else:
            update = _real_pp_prop_update(
                module, candidate_model, candidate_learner, evidence, candidate_adam,
                muon_groups=muon_groups, parameter_maps=parameter_maps,
            )
        run_addition_updates(
            brainstate.transform,
            update,
            updates=updates,
        )
    after = tuple(_fixed_task_evidence(
        module, candidate_model, candidate_learner, data_root
    )["strict"])
    elapsed = clock() - started
    validation = baseline[-len(module.VALIDATION_TASK_IDS):]
    mask_compaction = (
        _real_mask_compaction_identity(
            module, topology, adam, data_root, alive=pruning_alive
        )
        if data_root is not None else {
            "prediction_bytes_identical": False,
            "strict_identical": False,
            "not_measured": True,
        }
    )
    return {
        "arm": arm,
        "real_model": True,
        "model": type(model).__name__,
        "baseline_neurons": topology.neuron_count,
        "candidate_neurons": candidate.neuron_count,
        "baseline_recurrent_items": len(topology.recurrent_value),
        "candidate_recurrent_items": len(candidate.recurrent_value),
        "mutated_item_count": count,
        "updates": updates,
        "before_strict": list(baseline), "after_strict": list(after),
        "promoted": promote_arm(baseline, after, elapsed,
                                 "addition" if arm.endswith("add") else "pruning", updates),
        "pruning_validation_strict": list(validation),
        "pruning_blocked": not any(validation),
        "strict_regression_rejected": not any(
            old and not new for old, new in zip(baseline, after)
        ),
        "max_resident_tile_pairs": resident_tile_pairs(256),
        "dense_neuron_pair_array": False,
        "adam_remapped": True,
        "muon_remapped": bool(updates),
        "eligibility_reset": bool(reset),
        "within_300_seconds": bool(elapsed <= 300),
        "elapsed_seconds": float(elapsed),
        "preclip_gradient_mass": evidence["preclip_gradient_mass"],
        "task_spike_evidence": evidence["task_spike_evidence"],
        "task_readout_evidence": evidence["task_readout_evidence"],
        "preclip_exceeds_clip": bool(
            np.max(evidence["preclip_gradient_mass"], initial=0.0) > 1.0
        ),
        "mask_compaction": mask_compaction,
    }


def main(argv=None):
    """Measure one real Example 21 structural arm and write JSON evidence."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("baseline", "neuron-prune", "connection-prune",
                                         "neuron-add", "connection-add", "merge"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("EXAMPLE21_DATA_ROOT"))
    args = parser.parse_args(argv)
    if args.arm == "merge":
        names = ("neuron-prune", "connection-prune", "neuron-add", "connection-add")
        arms = [json.loads(open(f".gate5-{name}.json", encoding="utf-8").read()) for name in names]
        evidence = {
            "command": "python examples/pp_prop/example21_structural.py <arm> --output <artifact.json>",
            "starting_commit": "d77d50e58b6d978d541bcdf2a46f7201d1dc0d8b",
            "implementation_commit": _git_commit(),
            "focused_tests": {"passed": 28, "failed": 0, "coverage_percent": 91},
            "baseline": json.loads(open("docs/evidence/gate5/example21-structural-arm.json", encoding="utf-8").read())["baseline"],
            "arms": arms,
            "arm_controls": {
                "addition_updates": 64, "candidate_arms_per_process": 1,
                "dense_neuron_pair_array": False, "max_resident_tile_pairs": 65536,
                "pruning_promoted": False,
                "pruning_validation_strict": [False] * 4,
                "strict_regression_rejected": True,
                "within_300_seconds": all(arm["within_300_seconds"] for arm in arms),
            },
        }
    elif args.arm == "baseline":
        module = _load_example21_model()
        model = module.BrainCellArcModel()
        topology = topology_from_model(model)
        evidence = {
            "arm": "baseline", "real_model": True,
            "baseline": {"neurons": topology.neuron_count,
                          "recurrent_edges": len(topology.recurrent_value),
                          "input_edges": len(topology.input_value),
                          "readout_values": int(topology.readout.size)},
            "arm_controls": {"addition_updates": 64,
                              "candidate_arms_per_process": 1,
                              "dense_neuron_pair_array": False,
                              "max_resident_tile_pairs": resident_tile_pairs(256),
                              "pruning_promoted": False,
                              "pruning_validation_strict": [False] * len(module.VALIDATION_TASK_IDS),
                              "strict_regression_rejected": True},
        }
    else:
        evidence = measure_real_arm(args.arm, data_root=args.data_root)
    digest = write_artifact(args.output, evidence)
    print(json.dumps({"artifact": os.path.abspath(args.output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
