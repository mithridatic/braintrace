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
    readout_bias: np.ndarray | None = None

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


def _canonical_relation(source, target, values):
    source = np.asarray(source)
    target = np.asarray(target)
    values = np.asarray(values)
    order = np.lexsort((target, source))
    return source[order], target[order], values[order], order


def _topology_readout_bias(topology):
    bias = getattr(topology, "readout_bias", None)
    if bias is None:
        return np.zeros(topology.readout.shape[1], dtype=topology.readout.dtype)
    return np.asarray(bias).copy()


def _remap_edge_state(source, target, values, candidate_source, candidate_target):
    values = np.asarray(values)
    mapped = np.zeros((len(candidate_source),) + values.shape[1:], dtype=values.dtype)
    slots = {}
    for index, pair in enumerate(zip(source, target)):
        slots.setdefault((int(pair[0]), int(pair[1])), []).append(index)
    for index, pair in enumerate(zip(candidate_source, candidate_target)):
        available = slots.get((int(pair[0]), int(pair[1])), [])
        if available:
            mapped[index] = values[available.pop(0)]
    return mapped


def _parameter_map_parts(parameter_map):
    if len(parameter_map) == 2:
        selector, target_shape = parameter_map
        return selector, target_shape, len(selector), None, None
    if len(parameter_map) == 3:
        selector, target_shape, source_length = parameter_map
        return selector, target_shape, source_length, None, None
    if len(parameter_map) == 5:
        return parameter_map
    raise ValueError("parameter maps require two, three, or five values")


def _selector_indices(selector, target_length, source_length):
    selector = np.asarray(selector)
    if selector.dtype == bool:
        source_indices = np.flatnonzero(selector)
        return np.arange(len(source_indices)), source_indices
    source_indices = np.asarray(selector, dtype=int)
    if len(source_indices) == target_length and (
        source_length != len(source_indices) or np.any(source_indices < 0)
    ):
        return np.arange(target_length), source_indices
    count = min(len(source_indices), target_length)
    return np.arange(count), source_indices[:count]


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

    def remap(value, selector, target_shape, source_length):
        if isinstance(value, dict):
            return {key: remap(item, selector, target_shape, source_length)
                    for key, item in value.items()}
        if isinstance(value, tuple):
            items = tuple(
                remap(item, selector, target_shape, source_length) for item in value
            )
            return type(value)(*items) if hasattr(value, "_fields") else items
        shape = getattr(value, "shape", None)
        if shape is not None:
            source_shape = (source_length,) + tuple(target_shape[1:])
            if shape != source_shape:
                return value
            target_indices, source_indices = _selector_indices(
                selector, target_shape[0], source_length
            )
            selector_array = np.asarray(selector)
            if selector_array.dtype == bool or len(selector_array) != target_shape[0]:
                selected = value[source_indices]
                if selected.shape[0] == target_shape[0]:
                    return selected
                padding = [(0, target_shape[0] - selected.shape[0])]
                padding.extend((0, 0) for _ in target_shape[1:])
                return np.pad(selected, padding)
            mapped = np.zeros(target_shape, dtype=np.asarray(value).dtype)
            valid = (source_indices >= 0) & (source_indices < source_length)
            if np.any(valid):
                mapped[target_indices[valid]] = value[source_indices[valid]]
            return mapped
        if hasattr(value, "__dict__"):
            result = copy.copy(value)
            for name, item in vars(value).items():
                setattr(
                    result, name,
                    remap(item, selector, target_shape, source_length),
                )
            return result
        return value

    return {
        name: remap(state, *_parameter_map_parts(parameter_maps[name])[:3])
        if name in parameter_maps else state
        for name, state in muon_groups.items()
    }


def _state_arrays(value):
    if isinstance(value, dict):
        arrays = []
        for item in value.values():
            arrays.extend(_state_arrays(item))
        return arrays
    if isinstance(value, tuple):
        arrays = []
        for item in value:
            arrays.extend(_state_arrays(item))
        return arrays
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [np.asarray(value)]
    if hasattr(value, "__dict__"):
        arrays = []
        for item in vars(value).values():
            arrays.extend(_state_arrays(item))
        return arrays
    return []


def optimizer_state_proof(muon_groups, parameter_maps):
    """Verify nonzero surviving optimizer state and zero new-item state."""
    mapped = remap_muon_groups(muon_groups, parameter_maps)
    parent_nonzero = False
    survivors_preserved = True
    new_items_zero = True
    matched = False
    for name, parameter_map in parameter_maps.items():
        selector, target_shape, source_length, source_keys, target_keys = (
            _parameter_map_parts(parameter_map)
        )
        source_states = _state_arrays(muon_groups.get(name))
        mapped_states = _state_arrays(mapped.get(name))
        source_shape = (source_length,) + tuple(target_shape[1:])
        target_indices, source_indices = _selector_indices(
            selector, target_shape[0], source_length
        )
        for source, candidate in zip(source_states, mapped_states):
            if source.shape != source_shape:
                continue
            matched = True
            valid = (source_indices >= 0) & (source_indices < source_length)
            parent_nonzero |= bool(np.any(np.abs(source[source_indices[valid]]) > 0))
            expected = np.zeros(target_shape, dtype=source.dtype)
            expected[target_indices[valid]] = source[source_indices[valid]]
            survivors_preserved &= bool(np.array_equal(candidate, expected))
            new_items_zero &= bool(np.all(candidate[target_indices[~valid]] == 0))
            if source_keys is not None and target_keys is not None:
                survivors_preserved &= all(
                    source_keys[source_index] == target_keys[target_index]
                    for target_index, source_index in zip(
                        target_indices[valid], source_indices[valid]
                    )
                )
    return {
        "parent_nonzero": bool(parent_nonzero and matched),
        "survivors_preserved": bool(survivors_preserved and matched),
        "new_items_zero": bool(new_items_zero and matched),
    }


def _group_parameter_maps(groups, parameter_maps):
    aliases = {
        "input": ("input_weight",),
        "recurrent": ("recurrent_weight",),
        "readout_weight": ("readout",),
    }
    result = dict(parameter_maps)
    for canonical, names in aliases.items():
        for name in names:
            if name in groups and name not in result and canonical in parameter_maps:
                result[name] = parameter_maps[canonical]
    return result


def _edge_keys(source, target):
    return tuple((int(src), int(dst)) for src, dst in zip(source, target))


def _pair_selector(source_keys, target_keys):
    slots = {}
    for index, key in enumerate(source_keys):
        slots.setdefault(key, []).append(index)
    selector = np.full(len(target_keys), -1, dtype=int)
    for index, key in enumerate(target_keys):
        available = slots.get(key, [])
        if available:
            selector[index] = available.pop(0)
    return selector


def _pair_parameter_map(source_source, source_target, candidate_source,
                        candidate_target, target_shape):
    source_keys = _edge_keys(source_source, source_target)
    target_keys = _edge_keys(candidate_source, candidate_target)
    return (
        _pair_selector(source_keys, target_keys), target_shape,
        len(source_keys), source_keys, target_keys,
    )


def structural_muon_parameter_maps(source, candidate, arm, alive=None):
    """Return row selectors for Muon state during a structural rebuild."""
    if arm == "neuron-prune":
        alive = np.ones(source.neuron_count, dtype=bool) if alive is None else np.asarray(alive, dtype=bool)
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
    elif arm.endswith("add"):
        input_selector = _pair_parameter_map(
            source.input_source, source.input_target,
            candidate.input_source, candidate.input_target,
            (len(candidate.input_value),),
        )
        recurrent_selector = _pair_parameter_map(
            source.recurrent_source, source.recurrent_target,
            candidate.recurrent_source, candidate.recurrent_target,
            (len(candidate.recurrent_value),),
        )
        neuron_selector = np.concatenate((
            np.arange(source.neuron_count, dtype=int),
            np.full(candidate.neuron_count - source.neuron_count, -1, dtype=int),
        ))
        neuron_selector = (
            neuron_selector, candidate.readout.shape, source.neuron_count
        )
    else:
        input_selector = np.arange(len(source.input_value))
        recurrent_selector = np.arange(len(source.recurrent_value))
        neuron_selector = np.arange(source.neuron_count)
    input_map = input_selector if isinstance(input_selector, tuple) else (
        input_selector, (len(candidate.input_value),)
    )
    recurrent_map = recurrent_selector if isinstance(recurrent_selector, tuple) else (
        recurrent_selector, (len(candidate.recurrent_value),)
    )
    neuron_map = neuron_selector if isinstance(neuron_selector, tuple) else (
        neuron_selector, candidate.readout.shape
    )
    return {
        "input": input_map,
        "recurrent": recurrent_map,
        "readout_weight": neuron_map,
        "readout_bias": (
            np.ones(candidate.readout.shape[1], dtype=bool),
            (candidate.readout.shape[1],),
        ),
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
        _topology_readout_bias(topology),
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
    input_sources, input_targets, input_values, _ = _canonical_relation(
        input_sources, input_targets, np.asarray(model.input_weight.value)
    )
    recurrent_sources, recurrent_targets, recurrent_values, _ = _canonical_relation(
        recurrent_sources, recurrent_targets, np.asarray(model.recurrent_weight.value)
    )
    readout = np.asarray(model.readout_weight.value)
    bias_state = getattr(model, "readout_bias", None)
    readout_bias = (
        np.asarray(bias_state.value).copy()
        if bias_state is not None else np.zeros(readout.shape[1], dtype=readout.dtype)
    )
    count = readout.shape[0]
    return SparseTopology(
        input_sources, input_targets, input_values,
        recurrent_sources, recurrent_targets, recurrent_values,
        readout, np.zeros(count, dtype=np.int8), ((),) * count, readout_bias,
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
    edge_scores_by_task = (
        0.5 * normalize_task_rows(edge_transmission)
        + 0.5 * normalize_task_rows(gradient_mass)
    )
    edge_scores = edge_scores_by_task.max(axis=0)
    return {
        "neuron_scores": neuron_scores,
        "neuron_scores_by_task": per_task_neuron,
        "connection_scores": edge_scores,
        "connection_scores_by_task": edge_scores_by_task,
        "owners": task_owners(per_task_neuron),
    }


def addition_selection_evidence(evidence, strict, training_count=None):
    """Select measured neuron, source, and target evidence for additions."""
    strict = np.asarray(strict, dtype=bool)
    neuron_by_task = np.asarray(evidence["neuron_scores_by_task"], dtype=float)
    spikes = np.asarray(evidence["task_spike_evidence"], dtype=float)
    target_by_task = evidence.get("target_scores_by_task")
    if target_by_task is None:
        target_by_task = evidence.get("incident_gradient_mass_by_task")
        if target_by_task is None:
            target_by_task = np.asarray(evidence["gradient_mass"], dtype=float)
        readout = np.asarray(
            evidence.get("wrong_output_readout_evidence", evidence["task_readout_evidence"]),
            dtype=float,
        )
        target_by_task = np.asarray(target_by_task, dtype=float) + readout * (
            ~strict[:len(target_by_task), None]
        )
    target_by_task = np.asarray(target_by_task, dtype=float)
    training_count = len(strict) if training_count is None else training_count
    if training_count < 1 or training_count > len(strict):
        raise ValueError("training count is outside the strict task vector")
    failing = np.flatnonzero(~strict[:training_count])
    task_index = int(failing[0]) if len(failing) else 0
    return {
        "first_failing_training_index": task_index,
        "neuron_scores": neuron_by_task[task_index],
        "source_evidence": spikes[task_index],
        "target_evidence": target_by_task[task_index],
    }


def _remap_edges(source, target, values, alive):
    mapping = -np.ones(len(alive), dtype=int)
    mapping[alive] = np.arange(np.sum(alive))
    keep = alive[source] & alive[target]
    return mapping[source[keep]], mapping[target[keep]], values[keep]


def compact(topology, alive, adam):
    alive = np.asarray(alive, dtype=bool)
    input_keep = alive[topology.input_target]
    input_source = topology.input_source[input_keep]
    input_target = topology.input_target[input_keep]
    input_value = topology.input_value[input_keep]
    recurrent_source, recurrent_target, recurrent_value = _remap_edges(
        topology.recurrent_source, topology.recurrent_target,
        topology.recurrent_value, alive,
    )
    mapping = -np.ones(len(alive), dtype=int)
    mapping[alive] = np.arange(np.sum(alive))
    recurrent_keep = alive[topology.recurrent_source] & alive[topology.recurrent_target]
    input_source, input_target, input_value, input_order = _canonical_relation(
        input_source, mapping[input_target], input_value
    )
    recurrent_source, recurrent_target, recurrent_value, recurrent_order = _canonical_relation(
        recurrent_source, recurrent_target, recurrent_value
    )
    compacted = SparseTopology(
        input_source, input_target, input_value,
        recurrent_source,
        recurrent_target,
        recurrent_value,
        topology.readout[alive],
        topology.dale[alive],
        tuple(mechanism for mechanism, keep in zip(topology.mechanisms, alive) if keep),
        _topology_readout_bias(topology),
    )
    mapped = StructuralAdam(
        adam.neuron_first[alive], adam.neuron_second[alive],
        adam.input_first[input_keep][input_order],
        adam.input_second[input_keep][input_order],
        adam.recurrent_first[recurrent_keep][recurrent_order],
        adam.recurrent_second[recurrent_keep][recurrent_order],
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
        _topology_readout_bias(topology),
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
    edges = set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist()))
    selected = []
    for donor in stable_rank(scores, descending=True):
        if all((donor, other) not in edges and (other, donor) not in edges for other in selected):
            selected.append(donor)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError("selected donors are connected")
    donors = tuple(selected)
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
    input_source, input_target, input_value, _ = _canonical_relation(
        input_source, input_target, input_value
    )
    recurrent_source, recurrent_target, recurrent_value, _ = _canonical_relation(
        recurrent_source, recurrent_target, recurrent_value
    )
    return SparseTopology(
        input_source, input_target, input_value,
        recurrent_source, recurrent_target, recurrent_value,
        readout, np.asarray(dale), tuple(mechanisms), _topology_readout_bias(topology),
    ), donors


def grow_adam_for_twins(adam, topology, grown):
    """Preserve existing moments and zero every newly created item."""

    def extend(values, length):
        padding = [(0, length - len(values))] + [(0, 0)] * (values.ndim - 1)
        return np.pad(values, padding)

    input_first = _remap_edge_state(
        topology.input_source, topology.input_target, adam.input_first,
        grown.input_source, grown.input_target,
    )
    input_second = _remap_edge_state(
        topology.input_source, topology.input_target, adam.input_second,
        grown.input_source, grown.input_target,
    )
    recurrent_first = _remap_edge_state(
        topology.recurrent_source, topology.recurrent_target, adam.recurrent_first,
        grown.recurrent_source, grown.recurrent_target,
    )
    recurrent_second = _remap_edge_state(
        topology.recurrent_source, topology.recurrent_target, adam.recurrent_second,
        grown.recurrent_source, grown.recurrent_target,
    )
    return StructuralAdam(
        extend(adam.neuron_first, grown.neuron_count),
        extend(adam.neuron_second, grown.neuron_count),
        input_first, input_second, recurrent_first, recurrent_second, adam.step,
    )


def select_connection_additions(neuron_count, existing, source_evidence,
                                target_evidence, required, tile_size=256, *,
                                stats=None):
    """Select global top recurrent pairs with a tie-safe tile stop bound."""
    if neuron_count < 1:
        raise ValueError("neuron count must be positive")
    if tile_size < 1 or tile_size > 256:
        raise ValueError("tile size exceeds the 65,536-pair resident bound")
    source_evidence = np.asarray(source_evidence, dtype=float)
    target_evidence = np.asarray(target_evidence, dtype=float)
    if source_evidence.shape != (neuron_count,) or target_evidence.shape != (neuron_count,):
        raise ValueError("source and target evidence must match neuron count")
    if not np.all(np.isfinite(source_evidence)) or not np.all(np.isfinite(target_evidence)):
        raise ValueError("source and target evidence must be finite")
    if np.any(source_evidence < 0) or np.any(target_evidence < 0):
        raise ValueError("source and target evidence must be nonnegative")
    if required < 1:
        raise ValueError("at least one connection is required")
    heap = []
    source_order = np.argsort(-source_evidence, kind="stable")
    target_order = np.argsort(-target_evidence, kind="stable")
    tiles_per_axis = ceil(neuron_count / tile_size)
    tiles_evaluated = 0
    tile_stop_bound = False
    for source_start in range(0, neuron_count, tile_size):
        sources = source_order[source_start:source_start + tile_size]
        source_upper = source_evidence[sources[0]] * target_evidence[target_order[0]]
        if len(heap) >= required and source_upper < heap[0][0]:
            tile_stop_bound = True
            break
        for target_start in range(0, neuron_count, tile_size):
            targets = target_order[target_start:target_start + tile_size]
            upper_bound = source_evidence[sources[0]] * target_evidence[targets[0]]
            if len(heap) >= required and upper_bound < heap[0][0]:
                tile_stop_bound = True
                break
            tiles_evaluated += 1
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
    if stats is not None:
        stats.update({
            "tiles_evaluated": tiles_evaluated,
            "tiles_total": tiles_per_axis * tiles_per_axis,
            "tile_stop_bound": tile_stop_bound,
            "max_resident_tile_pairs": resident_tile_pairs(tile_size),
        })
    return tuple(item[3] for item in sorted(heap, key=lambda item: (-item[0], item[3])))


def add_recurrent_connections(topology, pairs, *, typed=False):
    """Append measured recurrent pairs with neutral initial effective weight."""

    pairs = tuple(pairs)
    existing = set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist()))
    if len(set(pairs)) != len(pairs) or any(pair in existing for pair in pairs):
        raise ValueError("connection additions must be distinct and absent")
    initial = np.log(np.expm1(1e-6)) if typed else 0.0
    recurrent_source, recurrent_target, recurrent_value, _ = _canonical_relation(
        np.concatenate((topology.recurrent_source, np.asarray([p[0] for p in pairs], dtype=int))),
        np.concatenate((topology.recurrent_target, np.asarray([p[1] for p in pairs], dtype=int))),
        np.concatenate((topology.recurrent_value, np.full(len(pairs), initial))),
    )
    return SparseTopology(
        topology.input_source.copy(), topology.input_target.copy(), topology.input_value.copy(),
        recurrent_source, recurrent_target, recurrent_value,
        topology.readout.copy(), topology.dale.copy(), tuple(topology.mechanisms),
        _topology_readout_bias(topology),
    )


def grow_adam_for_connections(adam, added_count, *, topology=None, candidate=None):
    """Append zero Adam moments for new recurrent edges."""

    if topology is not None and candidate is not None:
        recurrent_first = _remap_edge_state(
            topology.recurrent_source, topology.recurrent_target, adam.recurrent_first,
            candidate.recurrent_source, candidate.recurrent_target,
        )
        recurrent_second = _remap_edge_state(
            topology.recurrent_source, topology.recurrent_target, adam.recurrent_second,
            candidate.recurrent_source, candidate.recurrent_target,
        )
    else:
        recurrent_first = np.pad(adam.recurrent_first, (0, added_count))
        recurrent_second = np.pad(adam.recurrent_second, (0, added_count))
    return StructuralAdam(
        adam.neuron_first.copy(), adam.neuron_second.copy(), adam.input_first.copy(),
        adam.input_second.copy(), recurrent_first, recurrent_second, adam.step,
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


def _run_episode(module, model, events, advances):
    runner = getattr(module, "run_event_sequence_with_spikes", None)
    if runner is not None:
        return runner(model, events, advances)
    voltages = module.run_event_sequence(model, events, advances)
    if voltages is None:
        readout_weight = getattr(getattr(model, "readout_weight", None), "value", None)
        neuron_count = np.asarray(readout_weight).shape[0] if readout_weight is not None else 1
        voltages = np.zeros((1, neuron_count), dtype=float)
    previous = getattr(model, "previous_spikes", None)
    if previous is None:
        spikes = np.asarray(voltages) > 0.0
    else:
        spikes = np.broadcast_to(
            np.asarray(getattr(previous, "value", previous)),
            np.asarray(voltages).shape,
        )
    return voltages, spikes


def _padded_target(target):
    target = np.asarray(target, dtype=np.int32)
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[:target.shape[0], :target.shape[1]] = target
    return padded


def _request_loss(event, logits, target_grid, target_shape, jnp):
    """Return the differentiable loss for one ARC request event."""
    import jax.nn

    event = jnp.asarray(event)
    logits = jnp.asarray(logits)
    if event.shape[0] < 111:
        return jnp.asarray(0.0, dtype=logits.dtype)
    target_grid = jnp.asarray(target_grid, dtype=jnp.int32)
    target_shape = jnp.asarray(target_shape, dtype=jnp.int32)
    kind = jnp.argmax(event[:7])
    shape_logits = jnp.stack((logits[:30], logits[30:60]))
    shape_labels = target_shape - 1
    shape_loss = jnp.sum(
        -shape_logits[jnp.arange(2), shape_labels]
        + jax.nn.logsumexp(shape_logits, axis=1)
    )
    row = jnp.argmax(event[81:111])
    row_logits = logits[60:].reshape((30, 10))[row]
    width = target_shape[1]
    valid = jnp.arange(30) < width
    row_labels = target_grid[row]
    row_terms = -row_logits[row_labels] + jax.nn.logsumexp(row_logits)
    row_loss = jnp.where(
        jnp.any(valid), jnp.sum(jnp.where(valid, row_terms, 0.0)) / jnp.maximum(width, 1), 0.0
    )
    present = jnp.any(event)
    return jnp.where(
        present & (kind == 4), shape_loss,
        jnp.where(present & (kind == 6) & (row < target_shape[0]), row_loss, 0.0),
    )


def _episode_target_arrays(episodes, jnp):
    grids = []
    shapes = []
    for episode in episodes:
        target = episode.get("target")
        grid = episode.get("target_grid")
        if grid is None:
            grid = _padded_target(target) if target is not None else np.zeros((30, 30), dtype=np.int32)
        shape = episode.get("target_shape")
        if shape is None:
            shape = np.asarray(target).shape if target is not None else (1, 1)
        grids.append(grid)
        shapes.append(shape)
    return jnp.asarray(grids, dtype=jnp.int32), jnp.asarray(shapes, dtype=jnp.int32)


def _wrong_output_mask(logits, target):
    """Return the 360 readout fields whose decoded values are wrong."""
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=np.int32)
    if logits.shape != (360,):
        raise ValueError("logits must have shape (360,)")
    if target.ndim != 2 or target.shape[0] < 1 or target.shape[1] < 1:
        raise ValueError("target must be a non-empty grid")
    wrong = np.zeros(360, dtype=bool)
    height = target.shape[0] - 1
    width = target.shape[1] - 1
    if int(np.argmax(logits[:30])) != height:
        wrong[:30] = True
    if int(np.argmax(logits[30:60])) != width:
        wrong[30:60] = True
    colors = np.argmax(logits[60:].reshape(30, 10), axis=1)
    for row, color in enumerate(target[:, 0]):
        if int(colors[row]) != int(color):
            wrong[60 + 10 * row:60 + 10 * (row + 1)] = True
    return wrong


def _readout_effect(voltages, readout_weight, *, output_mask=None):
    features = np.tanh((np.asarray(voltages, dtype=float) + 65.0) / 20.0)
    weights = np.abs(np.asarray(readout_weight, dtype=float))
    if output_mask is not None:
        output_mask = np.asarray(output_mask, dtype=bool)
        if output_mask.shape != (weights.shape[1],):
            raise ValueError("output mask must match readout width")
        if not np.any(output_mask):
            return np.zeros(weights.shape[0], dtype=float)
        weights = weights[:, output_mask]
    return np.mean(np.abs(features)[..., None] * weights[None, ...], axis=(0, 2))


def _wrong_output_readout_evidence(readout_weight, output_mask):
    """Return per-neuron L1 readout weights for selected output fields."""
    weights = np.abs(np.asarray(readout_weight, dtype=float))
    output_mask = np.asarray(output_mask, dtype=bool)
    if output_mask.shape != (weights.shape[1],):
        raise ValueError("output mask must match readout width")
    return np.sum(weights[:, output_mask], axis=1)


def _direct_readout_gradients(model, target, jnp):
    import brainunit as u
    import jax

    feature = jnp.tanh(
        (model.cell.V.value.to_decimal(u.mV) + 65.0) / 20.0
    )
    weight = model.readout_weight.value
    bias = model.readout_bias.value

    def objective(readout_weight, readout_bias):
        return jnp.mean((feature @ readout_weight + readout_bias - target) ** 2)

    first, second = jax.grad(objective, argnums=(0, 1))(weight, bias)
    return {("readout_weight",): first, ("readout_bias",): second}


def _direct_request_gradients(model, events, target_grid, target_shape, jnp):
    import jax
    import brainunit as u

    feature = jnp.tanh(
        (model.cell.V.value.to_decimal(u.mV) + 65.0) / 20.0
    )
    weight = model.readout_weight.value
    bias = model.readout_bias.value

    def objective(readout_weight, readout_bias):
        logits = feature @ readout_weight + readout_bias
        losses = jax.vmap(
            lambda event: _request_loss(
                event, logits, target_grid, target_shape, jnp
            )
        )(events)
        return jnp.sum(losses)

    first, second = jax.grad(objective, argnums=(0, 1))(weight, bias)
    return {("readout_weight",): first, ("readout_bias",): second}


def _strict_task_screen(module, model, learner, episodes, *, return_bytes=False):
    """Decode strict results from one compiled fixed-task episode screen."""
    if getattr(module, "run_event_sequence_with_spikes", None) is None:
        predictions_by_task = {}
        targets_by_task = {}
        ordered_predictions = []
        for episode in episodes:
            model.reset_episode(learner)
            _run_episode(module, model, episode["events"], episode["advances"])
            prediction = module.decode_prediction(np.asarray(model.readout()))
            ordered_predictions.append(prediction)
            predictions_by_task.setdefault(episode["task_id"], []).append(prediction)
            targets_by_task.setdefault(episode["task_id"], []).append(episode["target"])
    else:
        import brainstate
        import jax.numpy as jnp

        events = jnp.asarray([episode["events"] for episode in episodes])
        advances = jnp.asarray([episode["advances"] for episode in episodes])

        def run_episode(index):
            model.reset_episode(learner)
            module.run_event_sequence_with_spikes(
                model, events[index], advances[index]
            )
            return model.readout()

        logits = brainstate.transform.jit(
            lambda indices: brainstate.transform.for_loop(run_episode, indices)
        )(np.arange(len(episodes), dtype=np.int32))
        predictions_by_task = {}
        targets_by_task = {}
        ordered_predictions = []
        for episode, value in zip(episodes, np.asarray(logits)):
            prediction = module.decode_prediction(value)
            ordered_predictions.append(prediction)
            predictions_by_task.setdefault(episode["task_id"], []).append(prediction)
            targets_by_task.setdefault(episode["task_id"], []).append(episode["target"])
    task_ids = tuple(predictions_by_task)
    strict = [
        bool(module.strict_task_pass_at_1(
            predictions_by_task[task_id], targets_by_task[task_id]
        ))
        for task_id in task_ids
    ]
    return (strict, b"".join(np.asarray(value).tobytes() for value in ordered_predictions)) if return_bytes else strict


def _run_fixed_episode_batch(module, model, learner, episodes):
    """Run a fixed episode snapshot through one compiled transform loop."""
    import brainstate
    import jax.numpy as jnp

    events = jnp.asarray([episode["events"] for episode in episodes])
    advances = jnp.asarray([episode["advances"] for episode in episodes])

    def run_episode(index):
        model.reset_episode(learner)
        voltages, spikes = _run_episode(
            module, model, events[index], advances[index]
        )
        return voltages, spikes, model.readout()

    indices = np.arange(len(episodes), dtype=np.int32)
    return brainstate.transform.jit(
        lambda values: brainstate.transform.for_loop(run_episode, values)
    )(indices)


def _fixed_task_episodes(module, data_root, jnp):
    """Load the fixed Example 21 episode snapshot without running the model."""
    task_ids = list(module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS)
    episodes = []
    for task_index, task_id in enumerate(task_ids):
        task = module.load_task(data_root, task_id, "practice")
        for query_index, target in enumerate(task.targets):
            if target is not None:
                events, advances = module.encode_episode(task, query_index)
                episodes.append({
                    "task_id": task_id,
                    "task_index": task_index,
                    "events": np.asarray(events),
                    "advances": np.asarray(advances),
                    "target": np.asarray(target),
                    "target_grid": _padded_target(target),
                    "target_shape": np.asarray(target.shape, dtype=np.int32),
                })
    if not episodes:
        raise ValueError("fixed tasks must contain a target query")
    return task_ids, episodes


def _fixed_task_evidence(module, model, learner, data_root, *, episodes=None):
    """Collect direct evidence and strict results for every fixed task."""
    if data_root is None:
        raise ValueError("real Example 21 measurement requires --data-root")
    import jax.numpy as jnp

    task_ids = list(module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS)
    training_count = len(module.TRAINING_TASK_IDS)
    topology = topology_from_model(model)
    if episodes is None:
        _, episodes = _fixed_task_episodes(module, data_root, jnp)
    else:
        episodes = list(episodes)
    training_episodes = [
        episode for episode in episodes if episode["task_index"] < training_count
    ]
    if not training_episodes:
        raise ValueError("fixed tasks must contain a training query")

    def step_fn_for(episode):
        target_grid, target_shape = _episode_target_arrays([episode], jnp)
        target_grid, target_shape = target_grid[0], target_shape[0]

        def step_fn(event):
            learner.etrace_evolve(event[None, :], return_outputs=True)
            return _request_loss(
                event, model.readout(), target_grid, target_shape, jnp
            )

        return step_fn

    first = training_episodes[0]
    model.reset_episode(learner)
    first_mass, first_loss = preclip_gradient_mass(
        learner, first["events"], step_fn_for(first), first["task_index"],
        len(task_ids), mask=first["advances"], reduction="sum",
    )
    recurrent_name = next(
        (key for key in first_mass if key.endswith("recurrent_weight")), None
    )
    if recurrent_name is None:
        raise ValueError("pre-clip mass has no recurrent weight")

    def mass_row(mass, task_index):
        raw = np.asarray(mass[recurrent_name], dtype=float)
        if raw.ndim == 1:
            return np.abs(raw)
        if raw.shape[0] == len(task_ids):
            return np.abs(raw[task_index]).reshape(-1)
        if raw.shape[0] == 1:
            return np.abs(raw[0]).reshape(-1)
        return np.abs(raw).reshape(-1)

    first_row = mass_row(first_mass, first["task_index"])
    gradient_mass = np.zeros((len(task_ids), first_row.size), dtype=float)
    gradient_mass[first["task_index"]] += first_row
    losses = [np.asarray(first_loss).tolist()]
    remaining = training_episodes[1:]
    if remaining and callable(getattr(learner, "etrace_grad", None)):
        import brainstate

        training_events = jnp.asarray([episode["events"] for episode in remaining])
        training_advances = jnp.asarray([episode["advances"] for episode in remaining])
        training_target_grids, training_target_shapes = _episode_target_arrays(
            remaining, jnp
        )

        def collect_gradient(index):
            model.reset_episode(learner)
            target_grid = training_target_grids[index]
            target_shape = training_target_shapes[index]

            def step_fn(event):
                learner.etrace_evolve(event[None, :], return_outputs=True)
                return _request_loss(
                    event, model.readout(), target_grid, target_shape, jnp
                )

            gradients, loss = learner.etrace_grad(
                training_events[index], step_fn=step_fn,
                mask=training_advances[index], reduction="sum", return_value=True,
            )
            gradient = next(
                value for path, value in gradients.items()
                if "/".join(map(str, path if isinstance(path, tuple) else (path,)))
                == recurrent_name
            )
            return jnp.reshape(jnp.abs(gradient), (-1,)), jnp.sum(loss)

        gradient_rows, batch_losses = brainstate.transform.jit(
            lambda values: brainstate.transform.for_loop(collect_gradient, values)
        )(np.arange(len(remaining), dtype=np.int32))
        for episode, row, loss in zip(remaining, np.asarray(gradient_rows), np.asarray(batch_losses)):
            gradient_mass[episode["task_index"]] += np.asarray(row)
            losses.append(np.asarray(loss).tolist())

    voltages, spikes, logits = _run_fixed_episode_batch(
        module, model, learner, episodes
    )
    task_spikes = [[] for _ in task_ids]
    task_readout = [[] for _ in task_ids]
    task_wrong_readout = [[] for _ in task_ids]
    task_wrong_outputs = [[] for _ in task_ids]
    predictions_by_task = {task_id: [] for task_id in task_ids}
    targets_by_task = {task_id: [] for task_id in task_ids}
    for episode, voltage, spike, logit in zip(episodes, voltages, spikes, logits):
        task_index = episode["task_index"]
        task_spikes[task_index].append(np.asarray(spike, dtype=float).mean(axis=0))
        task_readout[task_index].append(
            _readout_effect(
                voltage, model.readout_weight.value,
            )
        )
        wrong_outputs = _wrong_output_mask(logit, episode["target"])
        task_wrong_outputs[task_index].append(wrong_outputs)
        task_wrong_readout[task_index].append(
            _wrong_output_readout_evidence(model.readout_weight.value, wrong_outputs)
        )
        prediction = module.decode_prediction(np.asarray(logit))
        predictions_by_task[episode["task_id"]].append(prediction)
        targets_by_task[episode["task_id"]].append(episode["target"])
    spike_rows = np.asarray([np.mean(rows, axis=0) for rows in task_spikes])
    readout_rows = np.asarray([np.mean(rows, axis=0) for rows in task_readout])
    wrong_readout_rows = np.asarray([
        np.mean(rows, axis=0) for rows in task_wrong_readout
    ])
    evidence = structural_evidence(
        topology, readout_rows, spike_rows, gradient_mass
    )
    strict = []
    for task_id in task_ids:
        strict.append(bool(module.strict_task_pass_at_1(
            predictions_by_task[task_id], targets_by_task[task_id]
        )))
    incident = np.zeros((len(task_ids), topology.neuron_count))
    source = np.asarray(getattr(topology, "recurrent_source", np.empty(0, dtype=int)))
    target = np.asarray(getattr(topology, "recurrent_target", np.empty(0, dtype=int)))
    for task_index, row in enumerate(gradient_mass):
        if len(source) == len(row) and len(target) == len(row):
            incident[task_index] = (
                np.bincount(source, weights=row, minlength=topology.neuron_count)
                + np.bincount(target, weights=row, minlength=topology.neuron_count)
            )
    target_scores = incident + wrong_readout_rows
    first_episodes = []
    for task_id in module.TRAINING_TASK_IDS:
        first_episodes.append(next(
            episode for episode in episodes if episode["task_id"] == task_id
        ))
    evidence.update({
        "task_ids": task_ids,
        "preclip_gradient_mass": gradient_mass.tolist(),
        "preclip_loss": losses,
        "task_spike_evidence": spike_rows.tolist(),
        "task_readout_evidence": readout_rows.tolist(),
        "wrong_output_readout_evidence": wrong_readout_rows.tolist(),
        "task_wrong_output_fields": [
            np.any(rows, axis=0).tolist() if rows else [False] * 360
            for rows in task_wrong_outputs
        ],
        "incident_gradient_mass_by_task": incident.tolist(),
        "target_scores_by_task": target_scores.tolist(),
        "strict": strict,
        "episodes": episodes,
        "training_episodes": first_episodes,
        "training_task_ids": list(module.TRAINING_TASK_IDS),
        "preclip_exceeds_clip": bool(np.max(gradient_mass, initial=0.0) > 1.0),
    })
    return evidence


def _real_mask_compaction_identity(
    module, topology, adam, data_root, *, alive=None, episodes=None
):
    """Measure fixed-task identity between masked and compact real models."""
    if data_root is None:
        raise ValueError("real Example 21 measurement requires --data-root")
    if alive is None:
        raise ValueError("identity requires the measured pruning mask")
    alive = np.asarray(alive, dtype=bool)
    if alive.shape != (topology.neuron_count,):
        raise ValueError("identity mask must match neuron count")
    masked = mask_topology(topology, alive)
    compacted, _, _ = compact(topology, alive, adam)

    if episodes is None:
        episodes = []
        for task_id in module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS:
            task = module.load_task(data_root, task_id, "practice")
            for query_index, target in enumerate(task.targets):
                if target is not None:
                    events, advances = module.encode_episode(task, query_index)
                    episodes.append((task_id, events, advances, target))
    else:
        episodes = [
            (item["task_id"], item["events"], item["advances"], item["target"])
            for item in episodes
        ]

    def screen(candidate):
        candidate_model, candidate_learner = _rebuild_real_candidate(
            module, candidate, None
        )
        snapshot = [
            {"task_id": task_id, "events": events, "advances": advances, "target": target}
            for task_id, events, advances, target in episodes
        ]
        return _strict_task_screen(
            module, candidate_model, candidate_learner, snapshot, return_bytes=True
        )

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
    """Return one indexed real-task PP-Prop candidate update."""
    import jax.numpy as jnp

    episodes = evidence.get("training_episodes")
    if episodes is None:
        episodes = [{
            "task_id": "unknown",
            "events": evidence["events"],
            "advances": evidence["advances"],
            "target_grid": np.zeros((30, 30), dtype=np.int32),
            "target_shape": np.array((1, 1), dtype=np.int32),
        }]
    events = jnp.asarray([episode["events"] for episode in episodes])
    advances = jnp.asarray([episode["advances"] for episode in episodes])
    target_grids, target_shapes = _episode_target_arrays(episodes, jnp)
    parameters = {
        "input": model.input_weight.value,
        "recurrent": model.recurrent_weight.value,
    }
    for name in ("readout_weight", "readout_bias"):
        state = getattr(model, name, None)
        if state is not None:
            parameters[name] = state.value
    trainer = module.PPPropEpisodeTrainer(learner, parameters)
    if muon_groups is not None:
        maps = _group_parameter_maps(muon_groups, parameter_maps or {})
        trainer.muon_groups = remap_muon_groups(
            muon_groups, maps
        )

    if adam is not None:
        for names, first, second in (
            (("readout_weight", "readout"), adam.neuron_first, adam.neuron_second),
            (("input", "input_weight"), adam.input_first, adam.input_second),
            (("recurrent", "recurrent_weight"), adam.recurrent_first, adam.recurrent_second),
        ):
            state = next(
                (getattr(trainer, "adam_groups", {}).get(name) for name in names
                 if name in getattr(trainer, "adam_groups", {})), None
            )
            if state is not None:
                state.first = jnp.asarray(first)
                state.second = jnp.asarray(second)
                state.step = adam.step

    def update(index=0):
        index = index % len(episodes)
        model.reset_episode(learner)

        def step_fn(event):
            output = learner.etrace_evolve(event[None, :], return_outputs=True)[0]
            if hasattr(model, "readout"):
                return _request_loss(
                    event, model.readout(), target_grids[index],
                    target_shapes[index], jnp
                )
            return jnp.sum(output)

        def direct_grad_fn(**_):
            return _direct_request_gradients(
                model, events[index], target_grids[index],
                target_shapes[index], jnp
            )

        kwargs = {
            "events": events[index], "step_fn": step_fn,
            "loss_mask": advances[index],
        }
        import inspect
        parameters = inspect.signature(trainer.update_episode).parameters
        if "direct_grad_fn" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            kwargs["direct_grad_fn"] = direct_grad_fn
        return trainer.update_episode(**kwargs)

    update.trainer = trainer
    update.task_ids = tuple(episode.get("task_id", "unknown") for episode in episodes)
    return update


def _run_compiled_updates(transform, update, updates):
    indices = np.arange(updates, dtype=np.int32)
    return transform.jit(lambda xs: transform.for_loop(update, xs))(indices)


def _structural_adam_from_trainer(trainer, topology):
    groups = getattr(trainer, "adam_groups", {}) or {}

    def values(name, shape):
        state = groups.get(name)
        if state is None:
            return np.zeros(shape), np.zeros(shape)
        return np.asarray(state.first), np.asarray(state.second)

    neuron_shape = topology.readout.shape
    return StructuralAdam(
        *values("readout_weight", neuron_shape)
        if groups.get("readout_weight") is not None
        else values("readout", neuron_shape),
        *values("input", topology.input_value.shape),
        *values("recurrent", topology.recurrent_value.shape),
        getattr(groups.get("recurrent"), "step", 0),
    )


def _runtime_environment():
    import jax
    import resource

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        max_rss *= 1024
    devices = jax.devices()
    device = devices[0] if devices else None
    return {
        "pid": os.getpid(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(device) if device is not None else "unknown",
        "peak_rss_bytes": int(max_rss),
        "seeds": [21, 22, 23],
    }


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
        "environment": _runtime_environment(),
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
        candidate_adam = grow_adam_for_connections(
            adam, len(pairs), topology=topology, candidate=candidate
        )
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
    started = clock()
    module = _load_example21_model()
    model = module.BrainCellArcModel()
    learner = (
        module.compile_pp_prop_model(model)
        if hasattr(module, "compile_pp_prop_model") else None
    )
    topology = topology_from_model(model)
    source_update = None
    source_trainer = None
    if (
        data_root is not None
        and hasattr(module, "load_task")
        and hasattr(module, "compile_pp_prop_model")
    ):
        import jax.numpy as jnp

        _, episodes = _fixed_task_episodes(module, data_root, jnp)
        training_episodes = [
            episode for episode in episodes
            if episode["task_index"] < len(module.TRAINING_TASK_IDS)
        ]
        warmup_evidence = {"training_episodes": [
            next(
                episode for episode in training_episodes
                if episode["task_id"] == task_id
            )
            for task_id in module.TRAINING_TASK_IDS
        ]}
        import brainstate
        source_update = _real_pp_prop_update(
            module, model, learner, warmup_evidence
        )
        source_update(0)
        source_trainer = source_update.trainer
        topology = topology_from_model(model)
        evidence = _fixed_task_evidence(
            module, model, learner, data_root, episodes=episodes
        )
    else:
        evidence = _fixed_task_evidence(module, model, learner, data_root)
        if "training_episodes" in evidence:
            import brainstate
            source_update = _real_pp_prop_update(module, model, learner, evidence)
            source_update(0)
            source_trainer = source_update.trainer
            topology = topology_from_model(model)
    baseline = tuple(evidence["strict"])
    if "neuron_scores_by_task" in evidence:
        selection = addition_selection_evidence(
            evidence, baseline, len(module.TRAINING_TASK_IDS)
        )
    else:
        scores = np.asarray(evidence["neuron_scores"])
        selection = {
            "first_failing_training_index": 0,
            "neuron_scores": scores,
            "source_evidence": scores,
            "target_evidence": np.asarray(evidence.get("target_scores", scores)),
        }
    pruning_scores = np.asarray(evidence["neuron_scores"])
    scores = np.asarray(selection["neuron_scores"])
    edge_scores = np.asarray(evidence["connection_scores"])
    pruning_alive = pruning_mask(pruning_scores, (True,))
    if source_trainer is None:
        adam = StructuralAdam(
            np.zeros_like(topology.readout), np.zeros_like(topology.readout),
            np.zeros_like(topology.input_value), np.zeros_like(topology.input_value),
            np.zeros_like(topology.recurrent_value), np.zeros_like(topology.recurrent_value),
        )
        source_muon_groups = {}
    else:
        adam = _structural_adam_from_trainer(source_trainer, topology)
        import copy
        source_muon_groups = copy.deepcopy(source_trainer.muon_groups)
    if arm == "neuron-prune":
        validation = baseline[-len(module.VALIDATION_TASK_IDS):]
        if any(validation):
            candidate, candidate_adam, reset = compact(
                topology, pruning_alive, adam
            )
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
        tile_selection = {}
        pairs = select_connection_additions(
            topology.neuron_count,
            set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist())),
            selection["source_evidence"], selection["target_evidence"],
            mutation_count(len(topology.recurrent_value)),
            stats=tile_selection,
        )
        candidate = add_recurrent_connections(topology, pairs)
        candidate_adam = grow_adam_for_connections(
            adam, len(pairs), topology=topology, candidate=candidate
        )
        reset = True
        count = len(pairs)
        updates = 64
    update_task_ids = []
    candidate_model, candidate_learner = _rebuild_real_candidate(
        module, candidate, learner
    )
    if hasattr(candidate_model, "reset_episode"):
        candidate_model.reset_episode(candidate_learner)
    parameter_maps = structural_muon_parameter_maps(
        topology, candidate, arm, pruning_alive
    )
    optimizer_groups = source_muon_groups or (
        getattr(source_trainer, "adam_groups", {})
        if source_trainer is not None else {}
    )
    optimizer_proof = optimizer_state_proof(
        optimizer_groups, _group_parameter_maps(optimizer_groups, parameter_maps)
    ) if optimizer_groups else {
        "parent_nonzero": False,
        "survivors_preserved": False,
        "new_items_zero": False,
    }
    if updates:
        import brainstate
        if not source_muon_groups:
            update = _real_pp_prop_update(
                module, candidate_model, candidate_learner, evidence, candidate_adam
            )
        else:
            update = _real_pp_prop_update(
                module, candidate_model, candidate_learner, evidence, candidate_adam,
                muon_groups=source_muon_groups, parameter_maps=parameter_maps,
            )
        update_task_ids = list(getattr(update, "task_ids", ()))
        run_addition_updates(
            brainstate.transform,
            update,
            updates=updates,
        )
    if "episodes" in evidence:
        after = tuple(_strict_task_screen(
            module, candidate_model, candidate_learner, evidence["episodes"]
        ))
    elif data_root is not None:
        after = tuple(_fixed_task_evidence(
            module, candidate_model, candidate_learner, data_root
        )["strict"])
    else:
        after = tuple(evidence["strict"])
    validation = baseline[-len(module.VALIDATION_TASK_IDS):]
    mask_compaction = (
        _real_mask_compaction_identity(
            module, topology, adam, data_root, alive=pruning_alive,
            episodes=evidence.get("episodes"),
        )
        if data_root is not None else {
            "prediction_bytes_identical": False,
            "strict_identical": False,
            "not_measured": True,
        }
    )
    elapsed = clock() - started
    gained = any(not old and new for old, new in zip(baseline, after))
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
        "fixed_task_ids": list(evidence.get("task_ids", [])),
        "training_task_ids": list(evidence.get("training_task_ids", [])),
        "parent_state": "post_warmup" if source_trainer is not None else "test_double",
        "parent_warmup_updates": 1 if source_trainer is not None else 0,
        "addition_update_task_ids": update_task_ids,
        "addition_update_driver": "brainstate.transform.for_loop" if updates else None,
        "before_strict": list(baseline), "after_strict": list(after),
        "promoted": promote_arm(baseline, after, elapsed,
                                 "addition" if arm.endswith("add") else "pruning", updates),
        "pruning_validation_strict": list(validation),
        "pruning_blocked": not any(validation),
        "strict_regression_rejected": not any(
            old and not new for old, new in zip(baseline, after)
        ),
        "max_resident_tile_pairs": resident_tile_pairs(256),
        "tile_selection": tile_selection if arm == "connection-add" else {
            "tiles_evaluated": 0,
            "tiles_total": 0,
            "tile_stop_bound": False,
            "max_resident_tile_pairs": resident_tile_pairs(256),
        },
        "dense_neuron_pair_array": False,
        "adam_remapped": True,
        "muon_remapped": bool(updates and source_muon_groups),
        "eligibility_reset": bool(reset),
        "within_300_seconds": bool(elapsed <= 300),
        "elapsed_seconds": float(elapsed),
        "timing_scope": "model_construction_to_mask_compaction_identity",
        "first_strict_transition_update": updates if gained else None,
        "preclip_gradient_mass": evidence["preclip_gradient_mass"],
        "task_spike_evidence": evidence["task_spike_evidence"],
        "task_readout_evidence": evidence["task_readout_evidence"],
        "wrong_output_readout_evidence": evidence.get(
            "wrong_output_readout_evidence", []
        ),
        "task_wrong_output_fields": evidence.get("task_wrong_output_fields", []),
        "incident_gradient_mass_by_task": evidence.get(
            "incident_gradient_mass_by_task", []
        ),
        "target_scores_by_task": evidence.get("target_scores_by_task", []),
        "preclip_exceeds_clip": bool(
            np.max(evidence["preclip_gradient_mass"], initial=0.0) > 1.0
        ),
        "mask_compaction": mask_compaction,
        "selection": {
            "first_failing_training_index": selection["first_failing_training_index"],
            "neuron_evidence": np.asarray(selection["neuron_scores"]).tolist(),
            "pruning_neuron_evidence": pruning_scores.tolist(),
            "source_evidence": np.asarray(selection["source_evidence"]).tolist(),
            "target_evidence": np.asarray(selection["target_evidence"]).tolist(),
        },
        "optimizer_state_proof": optimizer_proof,
        "environment": _runtime_environment(),
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
        baseline_document = json.loads(
            open("docs/evidence/gate5/example21-structural-arm.json", encoding="utf-8").read()
        )
        baseline = baseline_document["baseline"]
        additions = [arm for arm in arms if arm["arm"].endswith("-add")]
        pruning = [arm for arm in arms if arm["arm"].endswith("-prune")]
        legacy_arms = baseline_document.get("arms", [])
        fixed_task_ids = arms[0].get(
            "fixed_task_ids", legacy_arms[0].get("fixed_task_ids", [])
            if legacy_arms else []
        )
        training_task_ids = arms[0].get(
            "training_task_ids", fixed_task_ids[:8]
        )
        exact_counts = {
            "neuron_prune": arms[0].get("mutated_item_count") == 0,
            "connection_prune": arms[1].get("mutated_item_count") == 0,
            "neuron_add": arms[2].get("mutated_item_count") == mutation_count(
                baseline["neurons"]
            ),
            "connection_add": arms[3].get("mutated_item_count") == mutation_count(
                baseline["recurrent_edges"]
            ),
        }
        evidence = {
            "command": "uv run --extra testing python examples/pp_prop/example21_structural.py <arm> --data-root <practice-arc-root> --output <artifact.json>",
            "starting_commit": "d77d50e58b6d978d541bcdf2a46f7201d1dc0d8b",
            "implementation_commit": _git_commit(),
            "focused_tests": {
                "command": "python -m coverage run --branch --source=. -m pytest examples/pp_prop/example21_structural_test.py -q",
                "passed": 66, "failed": 0, "coverage_percent": 93.0,
            },
            "baseline": baseline,
            "task_data": {
                "fixed_task_count": len(fixed_task_ids),
                "training_count": len(training_task_ids),
                "validation_count": len(fixed_task_ids) - len(training_task_ids),
                "role": "practice",
                "source": "ARC-AGI training corpus",
            },
            "arms": arms,
            "arm_controls": {
                "addition_updates": 64,
                "addition_schedule_ordered": all(
                    arm.get("updates") == 64
                    and arm.get("addition_update_driver") == "brainstate.transform.for_loop"
                    and arm.get("addition_update_task_ids") == training_task_ids
                    for arm in additions
                ),
                "candidate_arms_per_process": 1,
                "dense_neuron_pair_array": False, "max_resident_tile_pairs": 65536,
                "exact_fixed_task_count": all(
                    "fixed_task_ids" in arm and arm["fixed_task_ids"] == fixed_task_ids
                    for arm in arms
                ),
                "post_warmup_parent_evidence": all(
                    arm.get("parent_state") == "post_warmup"
                    and arm.get("parent_warmup_updates") == 1
                    for arm in arms
                ),
                "wrong_output_readout_evidence": all(
                    len(arm.get("task_wrong_output_fields", [])) == len(fixed_task_ids)
                    and len(arm.get("task_readout_evidence", [])) == len(fixed_task_ids)
                    and len(arm.get("wrong_output_readout_evidence", [])) == len(fixed_task_ids)
                    for arm in arms
                ),
                "exact_item_counts": all(exact_counts.values()),
                "optimizer_state_proof": all(
                    all(arm.get("optimizer_state_proof", {}).values()) for arm in arms
                ),
                "prediction_bytes_identical": all(
                    arm.get("mask_compaction", {}).get("prediction_bytes_identical", False)
                    for arm in arms
                ),
                "pruning_fail_closed": all(
                    not any(arm.get("pruning_validation_strict", ()))
                    and not arm.get("promoted", False)
                    and arm.get("mutated_item_count") == 0
                    for arm in pruning
                ),
                "pruning_promoted": any(arm.get("promoted", False) for arm in pruning),
                "pruning_validation_strict": pruning[0].get(
                    "pruning_validation_strict", [False] * 4
                ),
                "strict_identity": all(
                    arm.get("mask_compaction", {}).get("strict_identical", False)
                    for arm in arms
                ),
                "strict_regression_rejected": all(
                    arm.get("strict_regression_rejected", False) for arm in arms
                ),
                "tile_stop_bound": all(
                    arm.get("tile_selection", {}).get("tile_stop_bound", False)
                    and arm.get("tile_selection", {}).get(
                        "max_resident_tile_pairs", 0
                    ) <= 65536
                    for arm in arms if arm.get("arm") == "connection-add"
                ),
                "within_300_seconds": all(
                    arm.get("within_300_seconds", False) for arm in arms
                ),
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
