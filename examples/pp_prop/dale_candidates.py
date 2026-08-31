"""Measured Dale candidate selection and sparse sign constraints."""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from braintrace import sparse_matmul
from jax.errors import TracerBoolConversionError

_FLOOR = 1e-6


@dataclass(frozen=True)
class DaleMeasurements:
    """Per-neuron measurements from one accepted untyped parent."""

    parent_id: str
    rows: np.ndarray
    weights: np.ndarray
    activity: np.ndarray
    gradient_mass: np.ndarray
    task_ownership: np.ndarray
    lesion_evidence: np.ndarray
    type_signs: np.ndarray | None = None


@dataclass(frozen=True)
class DaleSelection:
    """Measured candidate indices for one parent checkpoint."""

    parent_id: str
    excitatory: np.ndarray
    inhibitory: np.ndarray
    excitatory_scores: np.ndarray
    inhibitory_scores: np.ndarray


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(values), initial=0.0)
    return np.zeros_like(values) if scale == 0 else values / scale


def _validate_measurements(measurements: DaleMeasurements) -> int:
    rows = np.asarray(measurements.rows)
    weights = np.asarray(measurements.weights)
    if rows.ndim != 1 or weights.ndim != 1 or rows.shape != weights.shape:
        raise ValueError("rows and weights must be one-dimensional arrays of equal length")
    arrays = (measurements.activity, measurements.gradient_mass,
              measurements.task_ownership, measurements.lesion_evidence)
    if any(np.asarray(item).ndim != 1 for item in arrays):
        raise ValueError("candidate measurements must be one-dimensional arrays")
    sizes = {np.asarray(item).size for item in arrays}
    if len(sizes) != 1:
        raise ValueError("candidate measurements must have one shared neuron count")
    if np.any(rows < 0) or np.any(rows >= next(iter(sizes))):
        raise ValueError("rows must address existing neurons")
    if measurements.type_signs is not None:
        signs = np.asarray(measurements.type_signs)
        if signs.shape != (next(iter(sizes)),) or not np.all(np.isin(signs, (-1, 0, 1))):
            raise ValueError("type signs must be one value per neuron and use -1, 0, or 1")
    return next(iter(sizes))


def measure_dale_candidates(
    measurements: DaleMeasurements,
    *,
    fraction: float = 0.05,
) -> DaleSelection:
    """Select measured excitatory and inhibitory candidates from one parent.

    Parameters
    ----------
    measurements : DaleMeasurements
        Measurements collected from the accepted untyped parent.
    fraction : float, optional
        Fraction of neurons selected independently for each arm.

    Returns
    -------
    DaleSelection
        Stable, independently ranked candidate sets.
    """
    neuron_count = _validate_measurements(measurements)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    type_signs = (
        np.zeros(neuron_count, dtype=np.int8)
        if measurements.type_signs is None
        else np.asarray(measurements.type_signs, dtype=np.int8)
    )
    untyped = np.flatnonzero(type_signs == 0)
    if not untyped.size:
        raise ValueError("parent has no untyped neurons")
    count = max(1, int(np.ceil(untyped.size * fraction)))
    rows = np.asarray(measurements.rows)
    weights = np.asarray(measurements.weights)
    signs = np.sign(weights)
    positive = np.zeros(neuron_count, dtype=np.float64)
    negative = np.zeros(neuron_count, dtype=np.float64)
    total = np.zeros(neuron_count, dtype=np.float64)
    np.add.at(positive, rows, signs > 0)
    np.add.at(negative, rows, signs < 0)
    np.add.at(total, rows, 1)
    coherence_e = np.divide(positive, total, out=np.zeros_like(total), where=total > 0)
    coherence_i = np.divide(negative, total, out=np.zeros_like(total), where=total > 0)
    common = sum(
        _normalise(np.asarray(values))
        for values in (
            measurements.activity,
            measurements.gradient_mass,
            measurements.task_ownership,
            measurements.lesion_evidence,
        )
    )
    scores_e = 0.2 * _normalise(coherence_e) + 0.8 * common / 4
    scores_i = 0.2 * _normalise(coherence_i) + 0.8 * common / 4
    order_e = untyped[np.lexsort((untyped, -scores_e[untyped]))[:count]]
    order_i = untyped[np.lexsort((untyped, -scores_i[untyped]))[:count]]
    return DaleSelection(
        measurements.parent_id,
        order_e.astype(np.int32),
        order_i.astype(np.int32),
        scores_e[order_e],
        scores_i[order_i],
    )


def inverse_softplus(values: jnp.ndarray) -> jnp.ndarray:
    """Convert positive values to raw coordinates for softplus."""
    values = jnp.maximum(values, jnp.finfo(values.dtype).tiny)
    return values + jnp.log(-jnp.expm1(-values))


def make_dale_weight_fn(type_signs: jnp.ndarray, floor: float = _FLOOR) -> Callable:
    """Build a differentiable sparse weight transform for typed edges.

    A zero type sign selects the raw signed coordinate. Non-zero signs select
    a positive softplus magnitude with the requested sign.
    """
    signs = jnp.asarray(type_signs)
    if floor <= 0:
        raise ValueError("floor must be positive")
    try:
        valid = bool(jnp.all(jnp.isin(signs, jnp.asarray((-1, 0, 1)))))
    except TracerBoolConversionError:
        valid = True
    if not valid:
        raise ValueError("type signs must be -1, 0, or 1")

    def weight_fn(raw: jnp.ndarray) -> jnp.ndarray:
        magnitude = jax_softplus(raw)
        return jnp.where(signs == 0, raw, signs * magnitude)

    return weight_fn


def sparse_dale_matmul(x, raw, sparse_mat, type_signs, *, floor: float = _FLOOR):
    """Apply the local Dale transform through BrainTrace sparse matmul."""
    return sparse_matmul(
        x,
        raw,
        sparse_mat=sparse_mat,
        weight_fn=make_dale_weight_fn(type_signs, floor),
    )


def jax_softplus(values: jnp.ndarray) -> jnp.ndarray:
    """Compute softplus without importing a second numerical namespace."""
    return jnp.logaddexp(values, 0)


def encode_dale_weights(effective: jnp.ndarray, type_signs: jnp.ndarray, floor: float = _FLOOR) -> jnp.ndarray:
    """Encode effective weights for a Dale `weight_fn` while preserving signs."""
    effective = jnp.asarray(effective)
    signs = jnp.asarray(type_signs)
    if effective.shape != signs.shape:
        raise ValueError("effective weights and type signs must have equal shape")
    if floor <= 0:
        raise ValueError("floor must be positive")
    typed = signs != 0
    return jnp.where(
        typed,
        inverse_softplus(jnp.maximum(jnp.abs(effective), floor)),
        effective,
    )


def effective_dale_weights(raw: jnp.ndarray, type_signs: jnp.ndarray) -> jnp.ndarray:
    """Return effective signed weights for a raw sparse coordinate array."""
    return make_dale_weight_fn(type_signs)(jnp.asarray(raw))


def project_dale_raw_weights(
    raw: jnp.ndarray, type_signs: jnp.ndarray, floor: float = _FLOOR
) -> jnp.ndarray:
    """Keep typed raw coordinates above the minimum effective magnitude."""
    raw = jnp.asarray(raw)
    signs = jnp.asarray(type_signs)
    if raw.shape != signs.shape:
        raise ValueError("raw weights and type signs must have equal shape")
    lower = inverse_softplus(jnp.asarray(floor, dtype=raw.dtype))
    return jnp.where(signs == 0, raw, jnp.maximum(raw, lower))


def validate_effective_signs(
    raw: jnp.ndarray, type_signs: jnp.ndarray, *, floor: float = _FLOOR
) -> bool:
    """Validate finite typed magnitudes and exact effective signs."""
    raw = jnp.asarray(raw)
    signs = jnp.asarray(type_signs)
    if raw.shape != signs.shape:
        raise ValueError("raw weights and type signs must have equal shape")
    effective = effective_dale_weights(raw, signs)
    typed = signs != 0
    return bool(
        jnp.all(jnp.isfinite(effective))
        and jnp.all(jnp.where(typed, jnp.abs(effective) >= floor, True))
        and jnp.all(jnp.where(typed, jnp.sign(effective) == signs, True))
    )


def deferred_biology_defaults() -> dict[str, bool]:
    """Return the inactive options for every deferred biological feature."""
    return {name: False for name in _DEFERRED_BIOLOGY}


_DEFERRED_BIOLOGY = (
    "ampa", "gabaa", "nmda", "hcn", "calcium_dependent_adaptation",
    "electrical_junctions", "multiple_compartments", "morphology",
    "neuromodulation", "persistent_episodic_memory", "extra_channels",
)


def validate_deferred_biology(**options: bool) -> None:
    """Reject optional biological mechanisms deferred from this experiment."""
    unknown = set(options) - set(_DEFERRED_BIOLOGY)
    if unknown:
        raise ValueError(f"unknown biology options: {', '.join(sorted(unknown))}")
    enabled = sorted(name for name, value in options.items() if value)
    if enabled:
        raise ValueError(f"deferred biology is disabled: {', '.join(enabled)}")


def strict_dale_gate(before, after, *, updates: int, elapsed_seconds: float | None = None) -> bool:
    """Return whether one measured Dale arm gains strict results without regression."""
    if updates != 128:
        raise ValueError("Dale arms require exactly 128 updates")
    if elapsed_seconds is not None and (
        not np.isfinite(elapsed_seconds) or elapsed_seconds > 300.0
    ):
        return False
    before = tuple(bool(value) for value in before)
    after = tuple(bool(value) for value in after)
    if len(before) != len(after):
        raise ValueError("strict vectors must have equal length")
    gained = any(not old and new for old, new in zip(before, after))
    regressed = any(old and not new for old, new in zip(before, after))
    return gained and not regressed


def _serialize_checkpoint(value) -> bytes:
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except (AttributeError, OSError, pickle.PickleError, TypeError) as error:
        raise ValueError("Dale parent must provide a serializable checkpoint") from error


def _validate_deferred_candidate(candidate) -> None:
    expected = deferred_biology_defaults()
    options = getattr(candidate, "biology_options", None)
    if not isinstance(options, dict) or set(options) != set(expected):
        raise ValueError("Dale candidate must declare every deferred biology option")
    validate_deferred_biology(**options)
    mechanisms = getattr(candidate, "mechanisms", None)
    if mechanisms is None:
        raise ValueError("Dale candidate must declare deferred mechanisms")
    if any(mechanisms):
        raise ValueError("Dale candidate enables deferred biology mechanisms")


def run_dale_arm(
    parent,
    candidate_indices,
    sign: int,
    build_candidate: Callable,
    update: Callable,
    strict: Callable,
    *,
    before_strict=None,
    transform=None,
    updates: int = 128,
    clock=time.perf_counter,
) -> dict:
    """Build and measure one isolated 128-update Dale candidate arm."""
    if sign not in (-1, 1):
        raise ValueError("Dale sign must be -1 or 1")
    if updates != 128:
        raise ValueError("Dale arms require exactly 128 updates")
    if transform is None:
        import brainstate
        transform = brainstate.transform
    started = clock()
    candidate = build_candidate(parent, np.asarray(candidate_indices), sign)
    if candidate is parent:
        raise ValueError("Dale candidate aliases its checkpoint parent")
    _validate_deferred_candidate(candidate)
    before = (
        tuple(bool(value) for value in strict(parent))
        if before_strict is None else tuple(bool(value) for value in before_strict)
    )

    def step(index):
        return update(candidate, index)

    indices = jnp.arange(updates, dtype=jnp.int32)
    transform.jit(lambda values: transform.for_loop(step, values))(indices)
    after = tuple(bool(value) for value in strict(candidate))
    elapsed = float(clock() - started)
    parent_id = getattr(parent, "parent_id", getattr(parent, "id", None))
    candidate_parent = getattr(candidate, "parent_id", None)
    if parent_id is not None and candidate_parent != parent_id:
        raise ValueError("Dale candidate does not preserve parent identity")
    return {
        "sign": sign,
        "candidate_indices": np.asarray(candidate_indices, dtype=np.int32).tolist(),
        "updates": updates,
        "before_strict": list(before),
        "after_strict": list(after),
        "elapsed_seconds": elapsed,
        "promoted": strict_dale_gate(
            before, after, updates=updates, elapsed_seconds=elapsed
        ),
        "candidate": candidate,
    }


def run_dale_candidates(
    parent,
    measurements: DaleMeasurements,
    build_candidate: Callable,
    update: Callable,
    strict: Callable,
    *,
    checkpoint: bytes | None = None,
    transform=None,
    updates: int = 128,
    clock=time.perf_counter,
) -> dict:
    """Run isolated excitatory and inhibitory arms from one parent checkpoint.

    Parameters
    ----------
    parent : object
        Accepted untyped parent used only for the baseline strict screen.
    measurements : DaleMeasurements
        Measurements collected from ``parent``.
    build_candidate, update, strict : callable
        Candidate construction, compiled update, and strict-screen callbacks.
    checkpoint : bytes, optional
        Serialized immutable parent checkpoint. When omitted, ``parent`` is
        serialized once before either arm starts.

    Returns
    -------
    dict
        Candidate selection, arm evidence, and checkpoint integrity evidence.
    """
    selection = measure_dale_candidates(measurements)
    parent_id = getattr(parent, "parent_id", getattr(parent, "id", None))
    if parent_id is not None and measurements.parent_id != parent_id:
        raise ValueError("Dale measurements do not belong to the accepted parent")
    parent_bytes = _serialize_checkpoint(parent)
    checkpoint_bytes = parent_bytes if checkpoint is None else bytes(checkpoint)
    if checkpoint is not None and checkpoint_bytes != parent_bytes:
        raise ValueError("Dale checkpoint does not match the accepted parent state")
    try:
        pickle.loads(checkpoint_bytes)
    except (EOFError, pickle.PickleError, TypeError, ValueError) as error:
        raise ValueError("Dale parent checkpoint is invalid") from error
    before = tuple(bool(value) for value in strict(parent))
    sources = []
    candidates_seen = []
    arms = []
    for sign, candidates in (
        (1, selection.excitatory), (-1, selection.inhibitory)
    ):
        source = pickle.loads(checkpoint_bytes)
        if source is parent or any(source is value for value in sources):
            raise ValueError("Dale checkpoint restore aliases a parent arm")
        source_id = getattr(source, "parent_id", getattr(source, "id", None))
        if parent_id is not None and source_id != parent_id:
            raise ValueError("Dale checkpoint does not match the accepted parent")
        sources.append(source)
        arm = run_dale_arm(
            source,
            candidates,
            sign,
            build_candidate,
            update,
            strict,
            before_strict=before,
            transform=transform,
            updates=updates,
            clock=clock,
        )
        candidate = arm["candidate"]
        if any(candidate is value for value in candidates_seen):
            raise ValueError("Dale arms share a candidate object")
        candidates_seen.append(candidate)
        if _serialize_checkpoint(source) != checkpoint_bytes:
            raise ValueError("Dale arm mutated its checkpoint parent")
        arms.append(arm)
    if _serialize_checkpoint(parent) != parent_bytes:
        raise ValueError("Dale arm mutated the accepted parent")
    return {
        "parent_id": parent_id,
        "candidate_count": int(selection.excitatory.size),
        "selection": selection,
        "arms": tuple(arms),
        "parent_checkpoint_unchanged": True,
    }
