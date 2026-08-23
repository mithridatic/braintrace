"""Balanced delayed-cue trial specifications and physical-rate encoding."""

from __future__ import annotations

import hashlib
import msgspec_json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from temporal_benchmark_config import HorizonSpec

N_INPUTS = 17
CLASS_CHANNELS = 8
GO_CHANNEL = 16


@dataclass(frozen=True)
class TrialSpec:
    """Identify a deterministic balanced delayed-cue trial."""

    trial_id: int
    label: int


def spike_probability(rate_hz: float, dt_seconds: float) -> float:
    """Convert a physical event rate into Bernoulli-bin probability."""
    if not math.isfinite(rate_hz) or rate_hz < 0.0:
        raise ValueError("rate_hz must be finite and nonnegative. Set rate_hz to a finite non-negative value.")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive. Set dt_seconds to a finite positive value.")
    return -math.expm1(-rate_hz * dt_seconds)


def balanced_trial_specs(count: int, split_seed: int) -> tuple[TrialSpec, ...]:
    """Create an exactly balanced, deterministically shuffled trial manifest."""
    if count <= 0 or count % 2:
        raise ValueError("Count must be positive and even. Set Count to a positive value.")
    labels = np.tile(np.asarray((0, 1), dtype=np.int8), count // 2)
    labels = labels[np.random.default_rng(split_seed).permutation(count)]
    return tuple(TrialSpec(index, int(label)) for index, label in enumerate(labels))


def trial_commitment(specs: Iterable[TrialSpec]) -> str:
    """Return the canonical SHA-256 commitment for ordered trial specs."""
    payload = [asdict(spec) for spec in specs]
    canonical = msgspec_json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def response_mask(horizon: HorizonSpec) -> np.ndarray:
    """Return the loss mask with supervision only in the response window."""
    mask = np.zeros(horizon.total_steps, dtype=np.float32)
    mask[-horizon.response_steps :] = 1.0
    return mask


def rate_template(
    label: int,
    horizon: HorizonSpec,
    cue_rate_hz: float,
    go_rate_hz: float,
) -> np.ndarray:
    """Create the label-conditioned physical-rate template for one trial."""
    if label not in (0, 1):
        raise ValueError("Label must be zero or one. Set Label to zero or one.")
    rates = np.zeros((horizon.total_steps, N_INPUTS), dtype=np.float64)
    cue_start = label * CLASS_CHANNELS
    rates[: horizon.cue_steps, cue_start : cue_start + CLASS_CHANNELS] = cue_rate_hz
    rates[-horizon.response_steps :, GO_CHANNEL] = go_rate_hz
    return rates


def encode_trials(
    specs: Iterable[TrialSpec],
    horizon: HorizonSpec,
    encoding_seed: int,
    *,
    cue_rate_hz: float,
    go_rate_hz: float,
    dt_seconds: float,
) -> np.ndarray:
    """Encode ordered trials with one shared deterministic spike realization."""
    ordered = tuple(specs)
    probabilities = np.stack(
        [
            1.0
            - np.exp(
                -rate_template(spec.label, horizon, cue_rate_hz, go_rate_hz)
                * dt_seconds
            )
            for spec in ordered
        ],
        axis=1,
    )
    random = np.random.default_rng(encoding_seed)
    return (random.random(probabilities.shape) < probabilities).astype(np.float32)


def response_is_label_independent(
    horizon: HorizonSpec, cue_rate_hz: float, go_rate_hz: float
) -> bool:
    """Prove analytically represented response rates do not depend on label."""
    zero = rate_template(0, horizon, cue_rate_hz, go_rate_hz)
    one = rate_template(1, horizon, cue_rate_hz, go_rate_hz)
    return bool(
        np.array_equal(zero[-horizon.response_steps :], one[-horizon.response_steps :])
    )
