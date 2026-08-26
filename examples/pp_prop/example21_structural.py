"""Sparse structural adaptation helpers for Example 21."""

import hashlib
import heapq
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from math import ceil

import numpy as np


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
        (topology.recurrent_source == left) & (topology.recurrent_target == right)
        | (topology.recurrent_source == right) & (topology.recurrent_target == left)
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


def run_addition_updates(transform, update, *, updates=64):
    """Run exactly 64 addition updates through a BrainState loop primitive."""

    if updates != 64:
        raise ValueError("addition arms require exactly 64 updates")
    indices = np.arange(updates, dtype=np.int32)
    return transform.jit(lambda xs: transform.for_loop(update, xs))(indices)


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
    return any(not old and new for old, new in zip(before, after)) and not any(
        old and not new for old, new in zip(before, after)
    ) if arm == "addition" else not any(old and not new for old, new in zip(before, after))
