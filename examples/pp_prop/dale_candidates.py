"""Measured Dale candidate selection and sparse sign constraints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from braintrace import sparse_matmul

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
    count = max(1, int(np.ceil(neuron_count * fraction)))
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
    order_e = np.lexsort((np.arange(neuron_count), -scores_e))[:count]
    order_i = np.lexsort((np.arange(neuron_count), -scores_i))[:count]
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
    if not bool(jnp.all(jnp.isin(signs, jnp.asarray((-1, 0, 1))))):
        raise ValueError("type signs must be -1, 0, or 1")

    def weight_fn(raw: jnp.ndarray) -> jnp.ndarray:
        magnitude = jax_softplus(raw) + floor
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
    if bool(jnp.any(typed & (signs * effective < floor))):
        raise ValueError("typed effective weights must have magnitude above the floor")
    return jnp.where(typed, inverse_softplus(jnp.abs(effective) - floor), effective)


def validate_deferred_biology(**options: bool) -> None:
    """Reject optional biological mechanisms deferred from this experiment."""
    deferred = {"ampa", "gabaa", "nmda", "neuromodulation", "extra_channels"}
    unknown = set(options) - deferred
    if unknown:
        raise ValueError(f"unknown biology options: {', '.join(sorted(unknown))}")
    enabled = sorted(name for name, value in options.items() if value)
    if enabled:
        raise ValueError(f"deferred biology is disabled: {', '.join(enabled)}")
