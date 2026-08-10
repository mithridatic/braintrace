"""Sealed optimizer schedules, clipping, and telemetry calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import braintools
import jax.numpy as jnp


def scheduled_learning_rate(update: int, total_updates: int, peak: float) -> float:
    """Apply 10% warmup, 70% hold, then cosine decay to 15% peak."""
    if total_updates <= 0 or update < 0 or update >= total_updates or peak <= 0.0:
        raise ValueError("invalid learning-rate schedule arguments")
    warmup_updates = max(1, math.ceil(0.1 * total_updates))
    hold_updates = max(warmup_updates, math.ceil(0.7 * total_updates))
    if update < warmup_updates:
        denominator = max(1, warmup_updates - 1)
        return peak * (0.1 + 0.9 * update / denominator)
    if update < hold_updates:
        return peak
    decay_updates = total_updates - hold_updates
    denominator = max(1, decay_updates - 1)
    decay_progress = (update - hold_updates) / denominator
    if decay_updates == 1:
        decay_progress = 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return peak * (0.15 + 0.85 * cosine)


class SealedLearningRateSchedule(braintools.optim.LRScheduler):
    """JIT-compatible implementation of the sealed three-stage schedule."""

    def __init__(self, peak: float, total_updates: int):
        if peak <= 0.0 or total_updates <= 0:
            raise ValueError("peak and total_updates must be positive")
        self.total_updates = total_updates
        self.warmup_updates = max(1, math.ceil(0.1 * total_updates))
        self.hold_updates = max(
            self.warmup_updates, math.ceil(0.7 * total_updates)
        )
        self.decay_updates = total_updates - self.hold_updates
        super().__init__(base_lr=peak)

    def get_lr(self):
        """Return the current scheduled learning rate as a one-item list."""
        update = self.last_epoch.value
        warm_denominator = max(1, self.warmup_updates - 1)
        warm = 0.1 + 0.9 * update / warm_denominator
        decay_denominator = max(1, self.decay_updates - 1)
        decay_progress = jnp.clip(
            (update - self.hold_updates) / decay_denominator, 0.0, 1.0
        )
        if self.decay_updates == 1:
            decay_progress = jnp.asarray(1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
        factor = jnp.where(
            update < self.warmup_updates,
            warm,
            jnp.where(update < self.hold_updates, 1.0, 0.15 + 0.85 * cosine),
        )
        return [self.base_lrs[0] * factor]


@dataclass(frozen=True)
class GradientTelemetry:
    """Summarize one optimizer group at one update."""

    raw_gradient_norm: float
    clipped_gradient_norm: float
    clip_event: bool
    weight_norm: float
    update_norm: float
    update_to_weight_ratio: float
    adam_first_moment_norm: float
    adam_second_moment_norm: float
    effective_learning_rate: float


def global_norm(leaves: list[np.ndarray]) -> float:
    """Return the Euclidean norm across an array tree's leaves."""
    return float(math.sqrt(sum(float(np.vdot(leaf, leaf).real) for leaf in leaves)))


def clip_leaves(
    leaves: list[np.ndarray], clip_norm: float | None
) -> tuple[list[np.ndarray], float, float, bool]:
    """Clip a group globally and return norms plus the clip event."""
    raw_norm = global_norm(leaves)
    if clip_norm is None or raw_norm <= clip_norm:
        return leaves, raw_norm, raw_norm, False
    scale = clip_norm / max(raw_norm, np.finfo(np.float64).tiny)
    clipped = [leaf * scale for leaf in leaves]
    return clipped, raw_norm, global_norm(clipped), True


def optimizer_telemetry(
    raw_gradients: list[np.ndarray],
    clipped_gradients: list[np.ndarray],
    weights_before: list[np.ndarray],
    weights_after: list[np.ndarray],
    first_moments: list[np.ndarray],
    second_moments: list[np.ndarray],
    learning_rate: float,
) -> GradientTelemetry:
    """Compute required gradient, weight, Adam, and update telemetry."""
    updates = [after - before for before, after in zip(weights_before, weights_after)]
    raw_norm = global_norm(raw_gradients)
    clipped_norm = global_norm(clipped_gradients)
    weight_norm = global_norm(weights_before)
    update_norm = global_norm(updates)
    return GradientTelemetry(
        raw_gradient_norm=raw_norm,
        clipped_gradient_norm=clipped_norm,
        clip_event=clipped_norm + 1e-12 < raw_norm,
        weight_norm=weight_norm,
        update_norm=update_norm,
        update_to_weight_ratio=update_norm / max(weight_norm, np.finfo(float).tiny),
        adam_first_moment_norm=global_norm(first_moments),
        adam_second_moment_norm=global_norm(second_moments),
        effective_learning_rate=float(learning_rate),
    )


def successive_halving_grid() -> tuple[tuple[float, float, float], ...]:
    """Return the sealed 27-combination development learning-rate grid."""
    return tuple(
        (readout, feedforward, recurrent)
        for readout in (1e-3, 3e-3, 1e-2)
        for feedforward in (3e-4, 1e-3, 3e-3)
        for recurrent in (1e-4, 3e-4, 1e-3)
    )
