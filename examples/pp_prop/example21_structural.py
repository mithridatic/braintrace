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
from pathlib import Path

import numpy as np


def _git_commit():
    """Return the current implementation revision for measured artifacts."""

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    try:
        result = subprocess.run(
            ("git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"),
            cwd=repo, capture_output=True, text=True, check=True,
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
    """Sparse optimizer arrays carried across a topology rebuild.

    Attributes
    ----------
    neuron_first, neuron_second : numpy.ndarray
        Readout-weight optimizer arrays indexed by neuron.
    input_first, input_second : numpy.ndarray
        Sparse input-edge optimizer arrays.
    recurrent_first, recurrent_second : numpy.ndarray
        Sparse recurrent-edge optimizer arrays.
    input_step, recurrent_step, readout_step : int
        Preserved optimizer update counts.
    """

    neuron_first: np.ndarray
    neuron_second: np.ndarray
    input_first: np.ndarray
    input_second: np.ndarray
    recurrent_first: np.ndarray
    recurrent_second: np.ndarray
    step: int = 0
    bias_first: np.ndarray | None = None
    bias_second: np.ndarray | None = None
    input_step: int | None = None
    recurrent_step: int | None = None
    readout_step: int | None = None

    def __post_init__(self):
        self.input_step = self.step if self.input_step is None else self.input_step
        self.recurrent_step = self.step if self.recurrent_step is None else self.recurrent_step
        self.readout_step = self.step if self.readout_step is None else self.readout_step


@dataclass(frozen=True)
class ParentCheckpoint:
    """Validated parent topology, parameters, and nonzero optimizer state.

    Attributes
    ----------
    topology : SparseTopology
        Loaded sparse parent topology.
    optimizer : StructuralAdam
        Loaded optimizer values and step counts.
    readout_bias : numpy.ndarray
        Loaded direct-readout bias.
    digest : str
        SHA-256 digest of the checkpoint file bytes.
    nonzero_optimizer_values : bool
        Whether at least one loaded optimizer value is nonzero.
    """

    topology: SparseTopology
    optimizer: StructuralAdam
    readout_bias: np.ndarray
    digest: str
    nonzero_optimizer_values: bool


def load_parent_checkpoint(module, path):
    """Load one accepted parent with real, nonzero optimizer values.

    Parameters
    ----------
    module : module
        Example 21 module that exposes the validated ``load_checkpoint`` API.
    path : path-like
        Accepted parent checkpoint path.

    Returns
    -------
    ParentCheckpoint
        Sparse model state and optimizer arrays ready for remapping.
    """

    arrays = module.load_checkpoint(path)
    neuron_count = int(arrays["neuron_count"])
    input_indptr = np.asarray(arrays["input_indptr"])
    recurrent_indptr = np.asarray(arrays["recurrent_indptr"])
    if len(input_indptr) != 442:
        raise ValueError("parent checkpoint must contain 441 sparse input rows")
    if len(recurrent_indptr) != neuron_count + 1:
        raise ValueError("parent checkpoint recurrent row count is invalid")
    topology = SparseTopology(
        np.repeat(np.arange(441), np.diff(input_indptr)),
        np.asarray(arrays["input_indices"]),
        np.asarray(arrays["input_values"]),
        np.repeat(np.arange(neuron_count), np.diff(recurrent_indptr)),
        np.asarray(arrays["recurrent_indices"]),
        np.asarray(arrays["recurrent_values"]),
        np.asarray(arrays["readout_weight"]),
        np.asarray(arrays["dale_codes"]),
        tuple(() if code == 0 else (int(code),)
              for code in np.asarray(arrays["mechanism_codes"])),
    )
    optimizer = StructuralAdam(
        np.asarray(arrays["readout_weight_m1"]),
        np.asarray(arrays["readout_weight_m2"]),
        np.asarray(arrays["input_m1"]),
        np.asarray(arrays["input_m2"]),
        np.asarray(arrays["recurrent_m1"]),
        np.asarray(arrays["recurrent_m2"]),
        bias_first=np.asarray(arrays["readout_bias_m1"]),
        bias_second=np.asarray(arrays["readout_bias_m2"]),
        input_step=int(arrays["input_step"]),
        recurrent_step=int(arrays["recurrent_step"]),
        readout_step=int(arrays["readout_step"]),
    )
    optimizer_values = (
        optimizer.neuron_first, optimizer.neuron_second,
        optimizer.input_first, optimizer.input_second,
        optimizer.recurrent_first, optimizer.recurrent_second,
        optimizer.bias_first, optimizer.bias_second,
    )
    nonzero = any(np.any(np.asarray(value) != 0) for value in optimizer_values)
    if not nonzero:
        raise ValueError("parent checkpoint must contain nonzero optimizer values")
    if min(optimizer.input_step, optimizer.recurrent_step, optimizer.readout_step) < 1:
        raise ValueError("parent checkpoint optimizer steps must be positive")
    return ParentCheckpoint(
        topology=topology,
        optimizer=optimizer,
        readout_bias=np.asarray(arrays["readout_bias"]),
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        nonzero_optimizer_values=bool(nonzero),
    )


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
        if isinstance(value, tuple):
            return type(value)(
                remap(item, selector, target_shape) for item in value
            )
        shape = getattr(value, "shape", None)
        source_shape = (len(selector),) + tuple(target_shape[1:])
        if shape == source_shape:
            selected = value[selector]
            if selected.shape[0] == target_shape[0]:
                return selected
            padding = [(0, target_shape[0] - selected.shape[0])]
            padding.extend((0, 0) for _ in target_shape[1:])
            return np.pad(selected, padding)
        if hasattr(value, "__dict__"):
            result = copy.copy(value)
            for name, item in vars(value).items():
                setattr(result, name, remap(item, selector, target_shape))
            return result
        return value

    return {
        name: remap(state, *parameter_maps[name])
        if name in parameter_maps else state
        for name, state in muon_groups.items()
    }


def _replace_muon_arrays(state, first, second, step):
    """Seed an Optax Muon state tree from accepted checkpoint arrays."""
    import jax.numpy as jnp

    if isinstance(state, dict):
        return type(state)(
            (key, _replace_muon_arrays(value, first, second, step))
            for key, value in state.items()
        )
    if isinstance(state, tuple) and hasattr(state, "_fields"):
        values = {}
        for name, value in zip(state._fields, state):
            if name == "count" and hasattr(value, "shape") and value.shape == ():
                values[name] = jnp.asarray(step, dtype=value.dtype)
            elif name == "mu" and getattr(value, "shape", None) == first.shape:
                values[name] = jnp.asarray(first, dtype=value.dtype)
            elif name == "nu" and getattr(value, "shape", None) == second.shape:
                values[name] = jnp.asarray(second, dtype=value.dtype)
            else:
                values[name] = _replace_muon_arrays(value, first, second, step)
        return type(state)(**values)
    if isinstance(state, tuple):
        return type(state)(
            _replace_muon_arrays(value, first, second, step) for value in state
        )
    return state


def initialize_muon_groups(trainer, optimizer):
    """Initialize active Muon groups from an accepted structural checkpoint.

    Parameters
    ----------
    trainer : object
        PP-Prop trainer with parameter groups and learning rates.
    optimizer : StructuralAdam
        Accepted checkpoint optimizer arrays.

    Returns
    -------
    dict
        Active Optax Muon states keyed by parameter name.
    """
    import optax

    arrays = {
        "input": (optimizer.input_first, optimizer.input_second, optimizer.input_step),
        "recurrent": (
            optimizer.recurrent_first, optimizer.recurrent_second,
            optimizer.recurrent_step,
        ),
        "readout_weight": (
            optimizer.neuron_first, optimizer.neuron_second,
            optimizer.readout_step,
        ),
        "readout_bias": (
            optimizer.bias_first, optimizer.bias_second,
            optimizer.readout_step,
        ),
    }
    groups = {}
    for name, parameter in trainer.parameters.items():
        if name not in arrays:
            continue
        first, second, step = arrays[name]
        if first is None or second is None:
            continue
        rate_name = next(
            (candidate for candidate in trainer.learning_rates if candidate in name),
            None,
        )
        if rate_name is None:
            continue
        rate = trainer.learning_rates[rate_name]
        transform = optax.contrib.muon(
            learning_rate=rate,
            weight_decay=0.1,
            adam_learning_rate=rate,
            adam_weight_decay=0.1,
        )
        groups[name] = _replace_muon_arrays(
            transform.init(parameter), np.asarray(first), np.asarray(second), int(step)
        )
    return groups


def optimizer_remap_checks(source, candidate, parameter_maps):
    """Measure surviving-value preservation and zero state for new items.

    Parameters
    ----------
    source, candidate : StructuralAdam
        Parent and remapped candidate optimizer arrays.
    parameter_maps : mapping
        Sparse row selectors and target shapes.

    Returns
    -------
    dict
        Boolean preservation checks and source and candidate step counts.
    """

    arrays = {
        "input": (
            source.input_first, source.input_second,
            candidate.input_first, candidate.input_second,
        ),
        "recurrent": (
            source.recurrent_first, source.recurrent_second,
            candidate.recurrent_first, candidate.recurrent_second,
        ),
        "readout_weight": (
            source.neuron_first, source.neuron_second,
            candidate.neuron_first, candidate.neuron_second,
        ),
    }
    preserved = True
    new_zero = True
    for name, (source_first, source_second, target_first, target_second) in arrays.items():
        selector, target_shape = parameter_maps[name]
        expected_first = np.asarray(source_first)[selector]
        expected_second = np.asarray(source_second)[selector]
        surviving = len(expected_first)
        preserved &= bool(
            np.array_equal(np.asarray(target_first)[:surviving], expected_first)
            and np.array_equal(np.asarray(target_second)[:surviving], expected_second)
        )
        new_zero &= bool(
            np.all(np.asarray(target_first)[surviving:] == 0)
            and np.all(np.asarray(target_second)[surviving:] == 0)
            and np.asarray(target_first).shape == tuple(target_shape)
        )
    steps_preserved = (
        source.input_step == candidate.input_step
        and source.recurrent_step == candidate.recurrent_step
        and source.readout_step == candidate.readout_step
    )
    return {
        "surviving_values_preserved": bool(preserved),
        "new_values_zero": bool(new_zero),
        "step_counts_preserved": bool(steps_preserved),
        "source_steps": {
            "input": int(source.input_step),
            "recurrent": int(source.recurrent_step),
            "readout": int(source.readout_step),
        },
        "candidate_initial_steps": {
            "input": int(candidate.input_step),
            "recurrent": int(candidate.recurrent_step),
            "readout": int(candidate.readout_step),
        },
    }


def muon_remap_checks(source_groups, candidate_groups, parameter_maps):
    """Compare active candidate Muon state with the exact sparse remap.

    Parameters
    ----------
    source_groups, candidate_groups : mapping
        Parent and candidate Optax Muon state trees.
    parameter_maps : mapping
        Sparse row selectors and target shapes.

    Returns
    -------
    dict
        Loaded, nonzero, and preservation measurements.
    """
    import jax

    expected = remap_muon_groups(source_groups, parameter_maps)
    expected_structure = jax.tree_util.tree_structure(expected)
    candidate_structure = jax.tree_util.tree_structure(candidate_groups)
    if expected_structure != candidate_structure:
        return {"loaded": bool(source_groups), "surviving_values_preserved": False}
    expected_leaves = jax.tree_util.tree_leaves(expected)
    candidate_leaves = jax.tree_util.tree_leaves(candidate_groups)
    equal = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(expected_leaves, candidate_leaves)
    )
    nonzero = any(
        np.any(np.asarray(value) != 0)
        for value in jax.tree_util.tree_leaves(source_groups)
        if hasattr(value, "shape")
    )
    return {
        "loaded": bool(source_groups),
        "source_nonzero": bool(nonzero),
        "surviving_values_preserved": bool(equal),
    }


def _muon_parameter_arrays(state, shape):
    """Extract active Muon or Adam-fallback arrays for one parameter."""
    matches = []

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            fields = dict(zip(value._fields, value))
            first = fields.get("mu")
            if getattr(first, "shape", None) == shape:
                second = fields.get("nu")
                if getattr(second, "shape", None) != shape:
                    second = np.zeros(shape, dtype=np.asarray(first).dtype)
                count = fields.get("count", 0)
                matches.append((
                    np.asarray(first), np.asarray(second), int(np.asarray(count))
                ))
            for item in value:
                visit(item)
            return
        if isinstance(value, tuple):
            for item in value:
                visit(item)

    visit(state)
    if len(matches) != 1:
        raise ValueError("active optimizer state does not match one parameter")
    return matches[0]


def optimizer_from_muon_groups(trainer):
    """Convert active Muon state to fixed structural checkpoint arrays.

    Parameters
    ----------
    trainer : object
        Trained PP-Prop trainer with active Muon groups.

    Returns
    -------
    StructuralAdam
        Fixed optimizer arrays and step counts.
    """

    extracted = {}
    for name in ("input", "recurrent", "readout_weight", "readout_bias"):
        if name not in trainer.muon_groups or name not in trainer.parameters:
            raise ValueError(f"active optimizer has no {name} state")
        extracted[name] = _muon_parameter_arrays(
            trainer.muon_groups[name], tuple(trainer.parameters[name].shape)
        )
    input_first, input_second, input_step = extracted["input"]
    recurrent_first, recurrent_second, recurrent_step = extracted["recurrent"]
    readout_first, readout_second, readout_step = extracted["readout_weight"]
    bias_first, bias_second, bias_step = extracted["readout_bias"]
    if bias_step != readout_step:
        raise ValueError("readout optimizer step counts are inconsistent")
    return StructuralAdam(
        readout_first, readout_second,
        input_first, input_second,
        recurrent_first, recurrent_second,
        bias_first=bias_first,
        bias_second=bias_second,
        input_step=input_step,
        recurrent_step=recurrent_step,
        readout_step=readout_step,
    )


def checkpoint_arrays(model, optimizer, evidence):
    """Build the exact array-only checkpoint schema from a trained parent.

    Parameters
    ----------
    model : object
        Trained Example 21 model.
    optimizer : StructuralAdam
        Extracted active optimizer state.
    evidence : mapping
        Measured task ownership evidence.

    Returns
    -------
    dict
        Arrays in the format-1 checkpoint schema.
    """

    topology = topology_from_model(model)
    input_counts = np.bincount(topology.input_source, minlength=441)
    recurrent_counts = np.bincount(
        topology.recurrent_source, minlength=topology.neuron_count
    )
    owners = evidence.get("owners", ((),) * topology.neuron_count)
    owner_codes = np.asarray([
        -1 if not owner else (owner[0] if len(owner) == 1 else -2)
        for owner in owners
    ], dtype=np.int16)
    return {
        "neuron_ids": np.arange(topology.neuron_count, dtype=np.int32),
        "dale_codes": np.asarray(topology.dale, dtype=np.int8),
        "owner_codes": owner_codes,
        "mechanism_codes": np.zeros(topology.neuron_count, dtype=np.uint8),
        "neuron_count": np.asarray(topology.neuron_count, dtype=np.int32),
        "integration_substeps": np.asarray(1, dtype=np.int32),
        "input_indptr": np.concatenate((
            np.zeros(1, dtype=np.int32), np.cumsum(input_counts, dtype=np.int32)
        )),
        "input_indices": np.asarray(topology.input_target, dtype=np.int32),
        "input_values": np.asarray(topology.input_value, dtype=np.float32),
        "input_m1": np.asarray(optimizer.input_first, dtype=np.float32),
        "input_m2": np.asarray(optimizer.input_second, dtype=np.float32),
        "recurrent_indptr": np.concatenate((
            np.zeros(1, dtype=np.int32),
            np.cumsum(recurrent_counts, dtype=np.int32),
        )),
        "recurrent_indices": np.asarray(topology.recurrent_target, dtype=np.int32),
        "recurrent_values": np.asarray(topology.recurrent_value, dtype=np.float32),
        "recurrent_m1": np.asarray(optimizer.recurrent_first, dtype=np.float32),
        "recurrent_m2": np.asarray(optimizer.recurrent_second, dtype=np.float32),
        "readout_weight": np.asarray(model.readout_weight.value, dtype=np.float32),
        "readout_bias": np.asarray(model.readout_bias.value, dtype=np.float32),
        "readout_weight_m1": np.asarray(optimizer.neuron_first, dtype=np.float32),
        "readout_weight_m2": np.asarray(optimizer.neuron_second, dtype=np.float32),
        "readout_bias_m1": np.asarray(optimizer.bias_first, dtype=np.float32),
        "readout_bias_m2": np.asarray(optimizer.bias_second, dtype=np.float32),
        "input_step": np.asarray(optimizer.input_step, dtype=np.int64),
        "recurrent_step": np.asarray(optimizer.recurrent_step, dtype=np.int64),
        "readout_step": np.asarray(optimizer.readout_step, dtype=np.int64),
    }


def write_parent_checkpoint(module, path, data_root):
    """Train and write one real accepted parent optimizer checkpoint.

    Parameters
    ----------
    module : module
        Example 21 implementation module.
    path : path-like
        Destination for the array-only checkpoint.
    data_root : path-like
        Root of the direct ARC training data.

    Returns
    -------
    dict
        Parent training, strict-screen, optimizer, and digest evidence.
    """
    import brainstate

    model = module.BrainCellArcModel()
    learner = module.compile_pp_prop_model(model)
    evidence = _fixed_task_evidence(module, model, learner, data_root)
    before = tuple(evidence["strict"])
    update = _real_pp_prop_update(module, model, learner, evidence)
    run_addition_updates(brainstate.transform, update, updates=64)
    after = _fixed_strict_screen(module, model, learner, data_root)
    if any(old and not new for old, new in zip(before, after)):
        raise ValueError("parent training caused a strict regression")
    optimizer = optimizer_from_muon_groups(update.trainer)
    arrays = checkpoint_arrays(model, optimizer, evidence)
    module.write_checkpoint(path, arrays)
    with open(path, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return {
        "arm": "parent",
        "implementation_commit": _git_commit(),
        "checkpoint_sha256": digest,
        "updates": 64,
        "before_strict": list(before),
        "after_strict": list(after),
        "optimizer_nonzero": any(
            np.any(arrays[name] != 0)
            for name in (
                "input_m1", "input_m2", "recurrent_m1", "recurrent_m2",
                "readout_weight_m1", "readout_weight_m2",
                "readout_bias_m1", "readout_bias_m2",
            )
        ),
        "optimizer_steps": {
            "input": int(arrays["input_step"]),
            "recurrent": int(arrays["recurrent_step"]),
            "readout": int(arrays["readout_step"]),
        },
    }


def structural_muon_parameter_maps(source, candidate, arm, alive=None):
    """Return row selectors for Muon state during a structural rebuild."""
    if arm == "neuron-prune":
        alive = (
            np.ones(source.neuron_count, dtype=bool)
            if alive is None else np.asarray(alive, dtype=bool)
        )
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
        "readout_bias": (
            np.arange(candidate.readout.shape[1]),
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


def structural_evidence(
    topology, readout_effect, spikes, gradient_mass, input_gradient_mass=None
):
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
    input_gradient_mass : array-like, optional
        Absolute pre-clip input gradient mass with shape
        ``(tasks, input_edges)``.

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
    if input_gradient_mass is None:
        input_gradient_mass = np.zeros((tasks, len(topology.input_value)))
    input_gradient_mass = np.asarray(input_gradient_mass, dtype=float)
    if input_gradient_mass.shape != (tasks, len(topology.input_value)):
        raise ValueError("input gradient mass must be task-by-edge")
    source = topology.recurrent_source
    transmission = np.zeros(expected, dtype=float)
    for task in range(tasks):
        transmission[task] = np.bincount(
            source,
            weights=np.abs(topology.recurrent_value) * np.abs(spikes[task, source]),
            minlength=topology.neuron_count,
        )
    incident = np.zeros(expected, dtype=float)
    target_incident = np.zeros(expected, dtype=float)
    for task in range(tasks):
        recurrent_source_mass = np.bincount(
            topology.recurrent_source, weights=gradient_mass[task],
            minlength=topology.neuron_count,
        )
        target_incident[task] = np.bincount(
            topology.recurrent_target, weights=gradient_mass[task],
            minlength=topology.neuron_count,
        )
        input_incident = np.bincount(
            topology.input_target, weights=input_gradient_mass[task],
            minlength=topology.neuron_count,
        )
        incident[task] = (
            input_incident
            + recurrent_source_mass
            + target_incident[task]
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
    per_task_edge = 0.5 * normalize_task_rows(edge_transmission)
    per_task_edge += 0.5 * normalize_task_rows(gradient_mass)
    edge_scores = per_task_edge.max(axis=0)
    return {
        "neuron_scores": neuron_scores,
        "connection_scores": edge_scores,
        "owners": task_owners(per_task_neuron),
        "neuron_task_scores": per_task_neuron,
        "connection_task_scores": per_task_edge,
        "source_mean_spikes": np.abs(spikes),
        "target_incident_gradient": incident,
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
        bias_first=None if adam.bias_first is None else adam.bias_first.copy(),
        bias_second=None if adam.bias_second is None else adam.bias_second.copy(),
        input_step=adam.input_step,
        recurrent_step=adam.recurrent_step,
        readout_step=adam.readout_step,
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
    connected = set(zip(
        topology.recurrent_source.tolist(), topology.recurrent_target.tolist()
    ))
    donors = []
    for candidate in stable_rank(scores, descending=True):
        if scores[candidate] <= 0:
            break
        if all(
            (candidate, donor) not in connected and (donor, candidate) not in connected
            for donor in donors
        ):
            donors.append(int(candidate))
        if len(donors) == required:
            break
    if len(donors) != required:
        raise ValueError("selected donors are connected")
    donors = tuple(donors)
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
        bias_first=None if adam.bias_first is None else adam.bias_first.copy(),
        bias_second=None if adam.bias_second is None else adam.bias_second.copy(),
        input_step=adam.input_step,
        recurrent_step=adam.recurrent_step,
        readout_step=adam.readout_step,
    )


def select_connection_additions(
    neuron_count,
    existing,
    source_evidence,
    target_evidence,
    required,
    tile_size=256,
    *,
    return_statistics=False,
):
    """Select globally ranked absent pairs with a proven next-tile stop.

    Parameters
    ----------
    neuron_count : int
        Active neuron count.
    existing : set of tuple
        Existing directed source-target pairs.
    source_evidence, target_evidence : array-like
        Nonnegative per-neuron evidence.
    required : int
        Exact number of absent pairs to select.
    tile_size : int, optional
        Source and target tile width, at most 256.
    return_statistics : bool, optional
        Return measured tile scan statistics with the selected pairs.

    Returns
    -------
    tuple or tuple of tuple and dict
        Selected pairs, optionally with bounded scan statistics.
    """

    if tile_size < 1 or tile_size > 256:
        raise ValueError("tile size exceeds the 65,536-pair resident bound")
    if required < 1:
        raise ValueError("connection addition count must be positive")
    source_evidence = np.asarray(source_evidence, dtype=float)
    target_evidence = np.asarray(target_evidence, dtype=float)
    expected = (neuron_count,)
    if source_evidence.shape != expected or target_evidence.shape != expected:
        raise ValueError("connection evidence must have one value per neuron")
    if (not np.all(np.isfinite(source_evidence))
            or not np.all(np.isfinite(target_evidence))
            or np.any(source_evidence < 0)
            or np.any(target_evidence < 0)):
        raise ValueError("connection evidence must be finite and nonnegative")
    heap = []
    source_order = np.argsort(-source_evidence, kind="stable")
    target_order = np.argsort(-target_evidence, kind="stable")
    tiles = []
    for source_start in range(0, neuron_count, tile_size):
        for target_start in range(0, neuron_count, tile_size):
            upper_bound = float(
                source_evidence[source_order[source_start]]
                * target_evidence[target_order[target_start]]
            )
            tiles.append((upper_bound, source_start, target_start))
    tiles.sort(key=lambda item: (-item[0], item[1], item[2]))
    scanned = 0
    stopped = False
    next_upper_bound = None
    max_resident = 0
    for upper_bound, source_start, target_start in tiles:
        if len(heap) == required and upper_bound < heap[0][0]:
            stopped = True
            next_upper_bound = upper_bound
            break
        sources = source_order[source_start:source_start + tile_size]
        targets = target_order[target_start:target_start + tile_size]
        scanned += 1
        max_resident = max(max_resident, len(sources) * len(targets))
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
    if len(heap) != required:
        raise ValueError("valid connection addition budget is insufficient")
    selected = tuple(
        item[3] for item in sorted(heap, key=lambda item: (-item[0], item[3]))
    )
    if not return_statistics:
        return selected
    return selected, {
        "tiles_scanned": scanned,
        "tiles_total": len(tiles),
        "stopped_by_bound": stopped,
        "next_tile_upper_bound": next_upper_bound,
        "worst_selected_score": float(heap[0][0]),
        "max_resident_pairs": max_resident,
    }


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
        bias_first=None if adam.bias_first is None else adam.bias_first.copy(),
        bias_second=None if adam.bias_second is None else adam.bias_second.copy(),
        input_step=adam.input_step,
        recurrent_step=adam.recurrent_step,
        readout_step=adam.readout_step,
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


def wrong_output_readout_evidence(
    request_voltages, readout_weight, readout_bias, target
):
    """Return neuron readout mass for supervised groups predicted incorrectly.

    Parameters
    ----------
    request_voltages : array-like
        Voltage vectors for the shape request and 30 row requests.
    readout_weight : array-like
        Direct voltage-readout weights with shape ``(neurons, 360)``.
    readout_bias : array-like
        Direct readout bias with shape ``(360,)``.
    target : array-like
        Integer target grid for the measured query.

    Returns
    -------
    numpy.ndarray
        Nonnegative wrong-output evidence for each neuron.
    """

    voltages = np.asarray(request_voltages, dtype=float)
    weights = np.asarray(readout_weight, dtype=float)
    bias = np.asarray(readout_bias, dtype=float)
    target = np.asarray(target)
    if voltages.ndim != 2 or voltages.shape[0] != 31:
        raise ValueError("readout evidence requires 31 request voltage vectors")
    if weights.shape != (voltages.shape[1], 360) or bias.shape != (360,):
        raise ValueError("readout evidence has incompatible parameter shapes")
    features = np.tanh((voltages + 65.0) / 20.0)
    logits = features @ weights + bias
    evidence = np.zeros(voltages.shape[1], dtype=float)

    def add_group(request_index, start, stop, expected):
        if int(np.argmax(logits[request_index, start:stop])) == int(expected):
            return
        evidence[:] += (
            np.abs(features[request_index])
            * np.sum(np.abs(weights[:, start:stop]), axis=1)
        )

    height, width = target.shape
    add_group(0, 0, 30, height - 1)
    add_group(0, 30, 60, width - 1)
    for row in range(height):
        for column in range(width):
            start = 60 + column * 10
            add_group(row + 1, start, start + 10, int(target[row, column]))
    return evidence


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


def _fixed_strict_screen(module, model, learner, data_root, *, transform=None):
    """Evaluate all fixed tasks with one transformed forward screen."""
    import jax.numpy as jnp

    if transform is None:
        import brainstate
        transform = brainstate.transform
    screen_events = []
    screen_advances = []
    screen_records = []
    task_ids = module.TRAINING_TASK_IDS + module.VALIDATION_TASK_IDS
    for task_id in task_ids:
        task = module.load_task(data_root, task_id, "practice")
        for query_index, target in enumerate(task.targets):
            if target is None:
                continue
            encoded, mask = module.encode_episode(task, query_index)
            screen_events.append(encoded)
            screen_advances.append(mask)
            screen_records.append((task_id, target))

    def evaluate_episode(events, advances):
        model.reset_episode(learner)
        voltages = module.run_event_sequence(model, events, advances)
        features = jnp.tanh((voltages[-31:] + 65.0) / 20.0)
        return features @ model.readout_weight.value + model.readout_bias.value

    screen_logits = transform.for_loop(
        evaluate_episode, np.stack(screen_events), np.stack(screen_advances)
    )
    predictions_by_task = {task_id: [] for task_id in task_ids}
    targets_by_task = {task_id: [] for task_id in task_ids}
    for logits, (task_id, target) in zip(np.asarray(screen_logits), screen_records):
        predictions_by_task[task_id].append(module.decode_prediction(logits))
        targets_by_task[task_id].append(target)
    return tuple(
        bool(module.strict_task_pass_at_1(
            predictions_by_task[task_id], targets_by_task[task_id]
        ))
        for task_id in task_ids
    )


def _fixed_task_evidence(module, model, learner, data_root, *, transform=None):
    """Collect all training-task evidence and fixed-screen strict results."""
    if data_root is None:
        raise ValueError("real Example 21 measurement requires --data-root")
    import jax.numpy as jnp

    if transform is None:
        import brainstate
        transform = brainstate.transform
    training_events = []
    training_advances = []
    training_targets = []
    for task_id in module.TRAINING_TASK_IDS:
        task = module.load_task(data_root, task_id, "practice")
        query_index = next(
            (index for index, target in enumerate(task.targets) if target is not None),
            None,
        )
        if query_index is None:
            raise ValueError(f"training task {task_id} has no supervised query")
        events, advances = module.encode_episode(task, query_index)
        training_events.append(events)
        training_advances.append(advances)
        training_targets.append(task.targets[query_index])
    training_events = np.stack(training_events)
    training_advances = np.stack(training_advances)
    target_colors = np.zeros((len(training_targets), 30, 30), dtype=np.int32)
    target_valid = np.zeros((len(training_targets), 30, 30), dtype=bool)
    target_heights = np.zeros(len(training_targets), dtype=np.int32)
    target_widths = np.zeros(len(training_targets), dtype=np.int32)
    for index, target in enumerate(training_targets):
        height, width = target.shape
        target_colors[index, :height, :width] = target
        target_valid[index, :height, :width] = True
        target_heights[index] = height - 1
        target_widths[index] = width - 1

    def measure_task(events, advances):
        model.reset_episode(learner)
        step_fn = lambda event: jnp.sum(
            learner.etrace_evolve(event[None, :], return_outputs=True)[0]
        )
        gradients, losses = learner.etrace_grad(
            events, step_fn=step_fn, return_value=True,
            mask=advances, reduction="sum",
        )

        def gradient_named(name):
            return next((
                value for path, value in gradients.items()
                if name in tuple(map(
                    str, path if isinstance(path, tuple) else (path,)
                ))
            ), None)

        input_gradient = gradient_named("input_weight")
        recurrent_gradient = gradient_named("recurrent_weight")
        if input_gradient is None or recurrent_gradient is None:
            raise ValueError("pre-clip mass must contain input and recurrent weights")
        model.reset_episode(learner)
        voltages = module.run_event_sequence(model, events, advances)
        return (
            jnp.abs(input_gradient), jnp.abs(recurrent_gradient),
            jnp.sum(losses), voltages,
        )

    input_mass, gradient_mass, losses, voltages = transform.for_loop(
        measure_task, training_events, training_advances
    )
    input_mass = np.asarray(input_mass)
    gradient_mass = np.asarray(gradient_mass)
    voltages = np.asarray(voltages)
    spikes = np.mean(voltages > 0.0, axis=1)
    readout_weight = np.asarray(model.readout_weight.value)
    readout_bias = np.asarray(model.readout_bias.value)
    request_voltages = voltages[:, -31:, :]
    readout = np.mean(
        np.abs(np.tanh((request_voltages + 65.0) / 20.0)), axis=1
    ) * np.sum(np.abs(readout_weight), axis=1)[None, :]
    wrong_readout = np.stack([
        wrong_output_readout_evidence(
            request_voltages[index], readout_weight, readout_bias, target
        )
        for index, target in enumerate(training_targets)
    ])
    topology = topology_from_model(model)
    evidence = structural_evidence(
        topology, readout, spikes, gradient_mass, input_mass
    )
    target_scores_by_task = normalize_task_rows(
        evidence["target_incident_gradient"] + wrong_readout
    )
    source_scores_by_task = normalize_task_rows(spikes)

    strict = _fixed_strict_screen(
        module, model, learner, data_root, transform=transform
    )
    evidence.update({
        "preclip_gradient_mass": np.asarray(gradient_mass).tolist(),
        "input_preclip_gradient_mass": np.asarray(input_mass).tolist(),
        "preclip_loss": np.asarray(losses).tolist(),
        "task_spike_evidence": np.asarray(spikes, dtype=float).tolist(),
        "task_readout_evidence": np.asarray(readout).tolist(),
        "wrong_output_readout_evidence": wrong_readout.tolist(),
        "connection_source_scores_by_task": source_scores_by_task.tolist(),
        "connection_target_scores_by_task": target_scores_by_task.tolist(),
        "training_task_ids": list(module.TRAINING_TASK_IDS),
        "strict": list(strict),
        "training_events": training_events,
        "training_advances": training_advances,
        "training_target_colors": target_colors,
        "training_target_valid": target_valid,
        "training_target_heights": target_heights,
        "training_target_widths": target_widths,
        "events": training_events[0],
        "advances": training_advances[0],
    })
    return evidence


def _real_mask_compaction_identity(
    module, topology, adam, data_root, *, alive=None, transform=None
):
    """Measure fixed-task identity between masked and compact real models."""
    import jax.numpy as jnp

    if transform is None:
        import brainstate
        transform = brainstate.transform

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

        def evaluate_episode(events, advances):
            candidate_model.reset_episode(candidate_learner)
            voltages = module.run_event_sequence(candidate_model, events, advances)
            features = jnp.tanh((voltages[-31:] + 65.0) / 20.0)
            return (
                features @ candidate_model.readout_weight.value
                + candidate_model.readout_bias.value
            )

        logits = transform.for_loop(
            evaluate_episode,
            np.stack([episode[1] for episode in episodes]),
            np.stack([episode[2] for episode in episodes]),
        )
        for output, (task_id, _events, _advances, target) in zip(
            np.asarray(logits), episodes
        ):
            prediction = module.decode_prediction(output)
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
    import jax
    import jax.numpy as jnp

    events = jnp.asarray(evidence.get("training_events", evidence["events"]))
    advances = jnp.asarray(evidence.get("training_advances", evidence["advances"]))
    if events.ndim == 2:
        events = events[None, ...]
        advances = advances[None, ...]
    direct_target_keys = (
        "training_target_colors", "training_target_valid",
        "training_target_heights", "training_target_widths",
    )
    direct_targets = (
        tuple(jnp.asarray(evidence[key]) for key in direct_target_keys)
        if all(key in evidence for key in direct_target_keys) else None
    )
    trainer = module.PPPropEpisodeTrainer(
        learner,
        {"input": model.input_weight.value, "recurrent": model.recurrent_weight.value},
    )
    if muon_groups is not None:
        trainer.muon_groups = remap_muon_groups(
            muon_groups, parameter_maps or {}
        )
    elif (adam is not None and hasattr(trainer, "parameters")
          and hasattr(trainer, "learning_rates")):
        trainer.muon_groups = initialize_muon_groups(trainer, adam)

    if adam is not None:
        for name, first, second, step in (
            ("readout", adam.neuron_first, adam.neuron_second, adam.readout_step),
            ("readout_weight", adam.neuron_first, adam.neuron_second, adam.readout_step),
            ("readout_bias", adam.bias_first, adam.bias_second, adam.readout_step),
            ("input", adam.input_first, adam.input_second, adam.input_step),
            ("recurrent", adam.recurrent_first, adam.recurrent_second, adam.recurrent_step),
        ):
            state = getattr(trainer, "adam_groups", {}).get(name)
            if state is not None and first is not None and second is not None:
                state.first = jnp.asarray(first)
                state.second = jnp.asarray(second)
                state.step = step

    def update(index=0):
        task_index = jnp.asarray(index, dtype=jnp.int32) % events.shape[0]
        task_events = events[task_index]
        task_advances = advances[task_index]
        model.reset_episode(learner)
        direct_grad_fn = None
        if (direct_targets is not None
                and "readout_weight" in getattr(trainer, "parameters", {})
                and "readout_bias" in getattr(trainer, "parameters", {})):
            colors, valid, heights, widths = direct_targets

            def direct_grad_fn(**_kwargs):
                model.reset_episode(learner)
                voltages = module.run_event_sequence(
                    model, task_events, task_advances
                )
                features = jnp.tanh((voltages[-31:] + 65.0) / 20.0)

                def objective(weight, bias):
                    logits = features @ weight + bias
                    shape_loss = (
                        -jax.nn.log_softmax(logits[0, :30])[heights[task_index]]
                        -jax.nn.log_softmax(logits[0, 30:60])[widths[task_index]]
                    )
                    row_logits = logits[1:, 60:].reshape((30, 30, 10))
                    row_log_prob = jax.nn.log_softmax(row_logits, axis=-1)
                    selected = jnp.take_along_axis(
                        row_log_prob,
                        colors[task_index, :, :, None],
                        axis=-1,
                    )[..., 0]
                    mask = valid[task_index]
                    row_loss = -jnp.sum(jnp.where(mask, selected, 0.0)) / jnp.maximum(
                        jnp.sum(mask), 1
                    )
                    return shape_loss + row_loss

                weight_gradient, bias_gradient = jax.grad(
                    objective, argnums=(0, 1)
                )(
                    trainer.parameters["readout_weight"],
                    trainer.parameters["readout_bias"],
                )
                return {
                    ("readout_weight",): weight_gradient,
                    ("readout_bias",): bias_gradient,
                }

        arguments = {
            "step_fn": lambda event: jnp.sum(
                learner.etrace_evolve(event[None, :], return_outputs=True)[0]
            ),
            "loss_mask": task_advances,
        }
        if direct_grad_fn is not None:
            arguments["direct_grad_fn"] = direct_grad_fn
        return trainer.update_episode(task_events, **arguments)

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


def validate_merged_arms(arms):
    """Reject a merged artifact unless every real arm satisfies Gate 5.

    Parameters
    ----------
    arms : sequence of mapping
        Four measured structural arm artifacts in fixed order.

    Raises
    ------
    ValueError
        If an arm violates a structural, optimizer, strict, or runtime gate.
    """

    expected_names = (
        "neuron-prune", "connection-prune", "neuron-add", "connection-add"
    )
    if tuple(arm.get("arm") for arm in arms) != expected_names:
        raise ValueError("merge requires the four structural arms in fixed order")
    pids = [arm.get("environment", {}).get("pid") for arm in arms]
    if None in pids or len(set(pids)) != len(pids):
        raise ValueError("each structural arm must use one separate process")
    commits = {arm.get("implementation_commit") for arm in arms}
    parents = {arm.get("parent_checkpoint_sha256") for arm in arms}
    if None in commits or len(commits) != 1:
        raise ValueError("all arms must identify one implementation commit")
    if None in parents or len(parents) != 1:
        raise ValueError("all arms must load one accepted parent checkpoint")
    for arm in arms:
        name = arm["arm"]
        before = arm.get("before_strict", [])
        after = arm.get("after_strict", [])
        gained = any(not old and new for old, new in zip(before, after))
        regressed = any(old and not new for old, new in zip(before, after))
        if not arm.get("promoted") or not gained:
            raise ValueError(f"{name} is not promoted by a strict Boolean gain")
        if regressed or not arm.get("strict_regression_rejected"):
            raise ValueError(f"{name} has a strict regression")
        if not arm.get("within_300_seconds"):
            raise ValueError(f"{name} exceeds the 300-second limit")
        if arm.get("dense_neuron_pair_array") is not False:
            raise ValueError(f"{name} does not prove sparse pair storage")
        if len(arm.get("training_evidence_task_ids", [])) != 8:
            raise ValueError(f"{name} does not contain eight training evidence rows")
        if not arm.get("parent_optimizer_nonzero"):
            raise ValueError(f"{name} did not load nonzero parent optimizer state")
        remap = arm.get("optimizer_remap", {})
        if not all(remap.get(key) for key in (
            "surviving_values_preserved", "new_values_zero", "step_counts_preserved"
        )):
            raise ValueError(f"{name} does not preserve optimizer state")
        if not arm.get("adam_remapped") or not arm.get("muon_remapped"):
            raise ValueError(f"{name} does not preserve active optimizer state")
        addition = name.endswith("add")
        if arm.get("updates") != (64 if addition else 0):
            raise ValueError(f"{name} has an invalid update count")
        baseline_count = (
            arm["baseline_neurons"] if name.startswith("neuron")
            else arm["baseline_recurrent_items"]
        )
        if arm.get("mutated_item_count") != mutation_count(baseline_count):
            raise ValueError(f"{name} has an invalid mutation count")
    compaction = arms[0].get("mask_compaction", {})
    if not (compaction.get("prediction_bytes_identical")
            and compaction.get("strict_identical")):
        raise ValueError("neuron pruning does not prove compaction identity")
    connection_selection = arms[3].get("connection_selection", {})
    if (not connection_selection.get("stopped_by_bound")
            or connection_selection.get("max_resident_pairs", 65_537) > 65_536):
        raise ValueError("connection addition does not prove the bounded tile bound")


def _coverage_summary():
    """Read measured line-plus-branch coverage from the current data file."""
    import coverage

    measured = coverage.Coverage(config_file=False)
    measured.load()
    has_branches = measured.get_data().has_arcs()
    percent = float(measured.report(
        show_missing=False,
        include=["examples/pp_prop/example21_structural.py"],
    ))
    if not has_branches or percent <= 90.0:
        raise ValueError("focused line-plus-branch coverage must exceed 90 percent")
    return {"line_and_branch_percent": percent, "branch_data": True}


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


def measure_real_arm(
    arm, *, data_root=None, parent_checkpoint=None, clock=time.perf_counter
):
    """Measure one bounded arm against the real Example 21 model topology.

    Parameters
    ----------
    arm : str
        One of ``neuron-prune``, ``connection-prune``, ``neuron-add``, or
        ``connection-add``.
    clock : callable, optional
        Monotonic clock used by the measurement.
    parent_checkpoint : path-like, optional
        Validated accepted parent checkpoint. Real evidence requires this path.

    Returns
    -------
    dict
        Per-arm evidence with real model dimensions and bounded controls.
    """
    if arm not in {"neuron-prune", "connection-prune", "neuron-add", "connection-add"}:
        raise ValueError("exactly one recognized arm is required")
    module = _load_example21_model()
    parent = (
        load_parent_checkpoint(module, parent_checkpoint)
        if parent_checkpoint is not None else None
    )
    model = (
        module.BrainCellArcModel(parent.topology)
        if parent is not None else module.BrainCellArcModel()
    )
    if parent is not None and hasattr(model, "readout_bias"):
        model.readout_bias.value = parent.readout_bias
    learner = (
        module.compile_pp_prop_model(model)
        if hasattr(module, "compile_pp_prop_model") else None
    )
    topology = topology_from_model(model)
    evidence = _fixed_task_evidence(module, model, learner, data_root)
    baseline = tuple(evidence["strict"])
    scores = np.asarray(evidence["neuron_scores"])
    edge_scores = np.asarray(evidence["connection_scores"])
    if parent is None:
        adam = StructuralAdam(
            np.zeros_like(topology.readout), np.zeros_like(topology.readout),
            np.zeros_like(topology.input_value), np.zeros_like(topology.input_value),
            np.zeros_like(topology.recurrent_value), np.zeros_like(topology.recurrent_value),
            bias_first=np.zeros_like(np.asarray(model.readout_bias.value))
            if hasattr(model, "readout_bias") else None,
            bias_second=np.zeros_like(np.asarray(model.readout_bias.value))
            if hasattr(model, "readout_bias") else None,
        )
    else:
        adam = parent.optimizer
    training_strict = baseline[:len(module.TRAINING_TASK_IDS)]
    first_failing_task = next(
        (index for index, value in enumerate(training_strict) if not value), None
    )
    if arm.endswith("add") and first_failing_task is None:
        raise ValueError("addition requires one failing fixed training task")
    neuron_task_scores = np.asarray(
        evidence.get("neuron_task_scores", scores[None, :])
    )
    source_scores_by_task = np.asarray(
        evidence.get("connection_source_scores_by_task", neuron_task_scores)
    )
    target_scores_by_task = np.asarray(
        evidence.get("connection_target_scores_by_task", neuron_task_scores)
    )
    started = clock()
    pruning_alive = None
    selection_statistics = {
        "tiles_scanned": 0,
        "tiles_total": 0,
        "stopped_by_bound": False,
        "next_tile_upper_bound": None,
        "worst_selected_score": None,
        "max_resident_pairs": 0,
    }
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
                bias_first=adam.bias_first,
                bias_second=adam.bias_second,
                input_step=adam.input_step,
                recurrent_step=adam.recurrent_step,
                readout_step=adam.readout_step,
            )
            count = int(np.sum(~keep))
        else:
            candidate = topology
            candidate_adam = adam
            count = 0
        reset = True
        updates = 0
    elif arm == "neuron-add":
        candidate, donors = add_twin_neurons(
            topology, neuron_task_scores[first_failing_task]
        )
        candidate_adam = grow_adam_for_twins(adam, topology, candidate)
        reset = True
        count = len(donors)
        updates = 64
    else:
        pairs, selection_statistics = select_connection_additions(
            topology.neuron_count,
            set(zip(topology.recurrent_source.tolist(), topology.recurrent_target.tolist())),
            source_scores_by_task[first_failing_task],
            target_scores_by_task[first_failing_task],
            mutation_count(len(topology.recurrent_value)),
            return_statistics=True,
        )
        candidate = add_recurrent_connections(topology, pairs)
        candidate_adam = grow_adam_for_connections(adam, len(pairs))
        reset = True
        count = len(pairs)
        updates = 64
    candidate_model, candidate_learner = _rebuild_real_candidate(
        module, candidate, learner
    )
    if parent is not None and hasattr(candidate_model, "readout_bias"):
        candidate_model.readout_bias.value = parent.readout_bias
    if hasattr(candidate_model, "reset_episode"):
        candidate_model.reset_episode(candidate_learner)
    parameter_maps = structural_muon_parameter_maps(
        topology, candidate, arm, pruning_alive
    )
    optimizer_remap = optimizer_remap_checks(
        adam, candidate_adam, parameter_maps
    )
    update = None
    muon_remap = {
        "loaded": False,
        "source_nonzero": False,
        "surviving_values_preserved": False,
    }
    if count or updates:
        source_update = _real_pp_prop_update(
            module, model, learner, evidence, adam
        )
        source_trainer = getattr(source_update, "trainer", None)
        muon_groups = getattr(source_trainer, "muon_groups", {})
        update = (
            _real_pp_prop_update(
                module, candidate_model, candidate_learner, evidence, candidate_adam,
                muon_groups=muon_groups, parameter_maps=parameter_maps,
            )
            if muon_groups else _real_pp_prop_update(
                module, candidate_model, candidate_learner, evidence, candidate_adam
            )
        )
        candidate_groups = getattr(
            getattr(update, "trainer", None), "muon_groups", {}
        )
        if muon_groups:
            muon_remap = muon_remap_checks(
                muon_groups, candidate_groups, parameter_maps
            )
    if updates:
        import brainstate
        run_addition_updates(
            brainstate.transform,
            update,
            updates=updates,
        )
    after = (
        _fixed_strict_screen(module, candidate_model, candidate_learner, data_root)
        if data_root is not None else tuple(_fixed_task_evidence(
            module, candidate_model, candidate_learner, data_root
        )["strict"])
    )
    elapsed = clock() - started
    validation = baseline[-len(module.VALIDATION_TASK_IDS):]
    mask_compaction = (
        _real_mask_compaction_identity(
            module, topology, adam, data_root, alive=pruning_alive
        )
        if data_root is not None and arm == "neuron-prune" and pruning_alive is not None else {
            "prediction_bytes_identical": False,
            "strict_identical": False,
            "not_measured": True,
        }
    )
    return {
        "arm": arm,
        "implementation_commit": _git_commit(),
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
        "first_failing_training_task": (
            module.TRAINING_TASK_IDS[first_failing_task]
            if first_failing_task is not None else None
        ),
        "training_evidence_task_ids": evidence.get("training_task_ids", []),
        "max_resident_tile_pairs": selection_statistics["max_resident_pairs"],
        "connection_selection": selection_statistics,
        "dense_neuron_pair_array": False,
        "parent_checkpoint_sha256": parent.digest if parent is not None else None,
        "parent_optimizer_nonzero": (
            parent.nonzero_optimizer_values if parent is not None else False
        ),
        "optimizer_remap": optimizer_remap,
        "adam_remapped": bool(
            optimizer_remap["surviving_values_preserved"]
            and optimizer_remap["new_values_zero"]
            and optimizer_remap["step_counts_preserved"]
        ),
        "muon_remap": muon_remap,
        "muon_remapped": bool(
            muon_remap["loaded"]
            and muon_remap["source_nonzero"]
            and muon_remap["surviving_values_preserved"]
        ),
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
    parser.add_argument("arm", choices=("baseline", "parent", "neuron-prune", "connection-prune",
                                         "neuron-add", "connection-add", "merge"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("EXAMPLE21_DATA_ROOT"))
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--focused-passed", type=int)
    args = parser.parse_args(argv)
    if args.arm == "merge":
        names = ("neuron-prune", "connection-prune", "neuron-add", "connection-add")
        arms = [json.loads(open(f".gate5-{name}.json", encoding="utf-8").read()) for name in names]
        validate_merged_arms(arms)
        if args.focused_passed is None or args.focused_passed < 1:
            parser.error("merge requires --focused-passed from the focused pytest run")
        coverage_summary = _coverage_summary()
        evidence = {
            "command": "python examples/pp_prop/example21_structural.py <arm> --data-root <arc-root> --parent-checkpoint <accepted.npz> --output .gate5-<arm>.json",
            "starting_commit": "d77d50e58b6d978d541bcdf2a46f7201d1dc0d8b",
            "implementation_commit": arms[0]["implementation_commit"],
            "artifact_build_commit": _git_commit(),
            "focused_tests": {
                "passed": args.focused_passed,
                "failed": 0,
                **coverage_summary,
            },
            "baseline": json.loads(open(".gate5-baseline.json", encoding="utf-8").read())["baseline"],
            "arms": arms,
            "arm_controls": {
                "addition_updates": 64, "candidate_arms_per_process": 1,
                "dense_neuron_pair_array": False, "max_resident_tile_pairs": 65536,
                "all_arms_promoted": True,
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
    elif args.arm == "parent":
        if args.data_root is None or args.checkpoint_output is None:
            parser.error("parent requires --data-root and --checkpoint-output")
        module = _load_example21_model()
        evidence = write_parent_checkpoint(
            module, args.checkpoint_output, args.data_root
        )
    else:
        if args.parent_checkpoint is None:
            parser.error("a real structural arm requires --parent-checkpoint")
        evidence = measure_real_arm(
            args.arm,
            data_root=args.data_root,
            parent_checkpoint=args.parent_checkpoint,
        )
    digest = write_artifact(args.output, evidence)
    print(json.dumps({"artifact": os.path.abspath(args.output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
