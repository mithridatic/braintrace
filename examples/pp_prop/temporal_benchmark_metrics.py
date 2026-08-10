"""Scientific metrics, hierarchical confidence intervals, and gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260811


@dataclass(frozen=True)
class BundleValue:
    """Associate one scalar result with a paired bundle and split."""

    split_id: str
    bundle_id: str
    value: float


@dataclass(frozen=True)
class ConfidenceInterval:
    """Store an estimate and percentile 95% interval."""

    estimate: float
    lower: float
    upper: float
    resamples: int = BOOTSTRAP_RESAMPLES
    seed: int = BOOTSTRAP_SEED


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Report NLL and ensemble accuracy across shared encodings."""
    if logits.ndim != 3 or logits.shape[2] != 2:
        raise ValueError("logits must have shape (encodings, trials, 2)")
    if labels.shape != (logits.shape[1],):
        raise ValueError("labels do not match the trial axis")
    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    ensemble = probabilities.mean(axis=0)
    indices = np.arange(labels.size)
    nll = -np.log(np.maximum(ensemble[indices, labels], np.finfo(float).tiny)).mean()
    return {
        "ensemble_accuracy": float(np.mean(ensemble.argmax(axis=1) == labels)),
        "ensemble_nll": float(nll),
    }


def dynamics_metrics(spikes: np.ndarray, voltages: np.ndarray) -> dict[str, object]:
    """Summarize firing and membrane-voltage stability diagnostics."""
    if spikes.ndim != 3 or voltages.shape != spikes.shape:
        raise ValueError("spikes and voltages must share (steps, batch, neurons)")
    neuron_rates = spikes.mean(axis=(0, 1))
    voltage_values = voltages.reshape(-1)
    return {
        "mean_firing_spikes_per_neuron_step": float(spikes.mean()),
        "firing_rate_percentiles": {
            str(percentile): float(np.percentile(neuron_rates, percentile))
            for percentile in (5, 25, 50, 75, 95, 99)
        },
        "silent_neuron_fraction": float(np.mean(neuron_rates == 0.0)),
        "saturated_neuron_fraction": float(np.mean(neuron_rates >= 0.95)),
        "membrane_voltage_percentiles": {
            str(percentile): float(np.percentile(voltage_values, percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
    }


def gradient_comparison(pp_prop: np.ndarray, bptt: np.ndarray) -> dict[str, float]:
    """Compare flattened pp-prop and BPTT recurrent gradients."""
    left = np.asarray(pp_prop, dtype=np.float64).reshape(-1)
    right = np.asarray(bptt, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("gradient vectors must be nonempty and shape matched")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    denominator = max(left_norm * right_norm, np.finfo(float).tiny)
    return {
        "cosine_similarity": float(np.dot(left, right) / denominator),
        "sign_agreement": float(np.mean(np.sign(left) == np.sign(right))),
        "norm_ratio": float(left_norm / max(right_norm, np.finfo(float).tiny)),
        "relative_deviation": float(
            np.linalg.norm(left - right) / max(right_norm, np.finfo(float).tiny)
        ),
    }


def hierarchical_paired_interval(
    records: Iterable[BundleValue],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """Resample splits and then bundles within each selected split."""
    observed = tuple(records)
    if not observed or resamples <= 0:
        raise ValueError("records and resamples must be nonempty")
    by_split: dict[str, list[float]] = {}
    for record in observed:
        if not math.isfinite(record.value):
            raise ValueError("bootstrap values must be finite")
        by_split.setdefault(record.split_id, []).append(record.value)
    split_ids = tuple(sorted(by_split))
    random = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = random.choice(split_ids, size=len(split_ids), replace=True)
        values: list[float] = []
        for split_id in selected:
            available = np.asarray(by_split[str(split_id)], dtype=np.float64)
            values.extend(random.choice(available, size=available.size, replace=True))
        samples[index] = np.mean(values)
    estimate = float(np.mean([record.value for record in observed]))
    lower, upper = np.percentile(samples, (2.5, 97.5))
    return ConfidenceInterval(estimate, float(lower), float(upper), resamples, seed)


def paired_differences(
    left: Iterable[BundleValue], right: Iterable[BundleValue]
) -> tuple[BundleValue, ...]:
    """Return bundle-aligned left-minus-right results."""
    right_map = {record.bundle_id: record for record in right}
    differences: list[BundleValue] = []
    for record in left:
        matched = right_map.get(record.bundle_id)
        if matched is None or matched.split_id != record.split_id:
            raise ValueError("paired bundle sets do not match")
        differences.append(
            BundleValue(record.split_id, record.bundle_id, record.value - matched.value)
        )
    if len(differences) != len(right_map):
        raise ValueError("paired bundle sets do not match")
    return tuple(differences)


def stability_passes(dynamics: dict[str, object], recurrent_ratios: np.ndarray) -> bool:
    """Apply the sealed primary stability gates."""
    values = np.asarray(recurrent_ratios, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    mean = float(dynamics["mean_firing_spikes_per_neuron_step"])
    silent = float(dynamics["silent_neuron_fraction"])
    saturated = float(dynamics["saturated_neuron_fraction"])
    return (
        0.001 <= mean <= 0.30
        and silent <= 0.95
        and saturated < 0.05
        and float(np.percentile(values, 99)) <= 0.05
    )


def gate_report(
    intervals: dict[str, ConfidenceInterval],
    *,
    response_label_independent: bool,
    stability: bool,
) -> dict[str, object]:
    """Evaluate available episodic gates and fail closed on missing evidence."""
    required = {
        "bptt_accuracy",
        "readout_only_accuracy",
        "no_recurrence_accuracy",
        "pp_prop_accuracy",
        "pp_prop_minus_frozen_accuracy",
        "cosine_advantage_over_null",
        "pp_prop_small_update_loss_change",
    }
    missing = sorted(required - intervals.keys())
    if missing:
        return {"passed": False, "status": "incomplete", "missing": missing}
    gates = {
        "task_validity": (
            intervals["bptt_accuracy"].lower >= 0.80
            and intervals["readout_only_accuracy"].upper <= 0.60
            and intervals["no_recurrence_accuracy"].upper <= 0.60
            and response_label_independent
        ),
        "pp_prop_learning": (
            intervals["pp_prop_accuracy"].lower > 0.50
            and intervals["pp_prop_minus_frozen_accuracy"].lower > 0.0
        ),
        "gradient_evidence": (
            intervals["cosine_advantage_over_null"].lower > 0.0
            and intervals["pp_prop_small_update_loss_change"].upper < 0.0
        ),
        "stability": stability,
    }
    return {
        "passed": all(gates.values()),
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "intervals": {name: asdict(interval) for name, interval in intervals.items()},
    }
