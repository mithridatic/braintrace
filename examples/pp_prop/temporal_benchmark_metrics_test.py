"""Tests for ensemble metrics, hierarchical intervals, and gates."""

import numpy as np

from temporal_benchmark_metrics import (
    BundleValue,
    ConfidenceInterval,
    classification_metrics,
    gate_report,
    hierarchical_paired_interval,
    paired_differences,
    stability_passes,
)


def test_ensemble_classification_averages_shared_encodings() -> None:
    logits = np.asarray([[[5.0, 0.0], [0.0, 5.0]]] * 8)
    metrics = classification_metrics(logits, np.asarray([0, 1]))

    assert metrics["ensemble_accuracy"] == 1.0
    assert metrics["ensemble_nll"] < 0.01


def test_hierarchical_bootstrap_is_seed_reproducible() -> None:
    records = tuple(
        BundleValue(f"split{index // 4}", f"bundle{index}", index / 12)
        for index in range(12)
    )

    first = hierarchical_paired_interval(records, resamples=100, seed=8)
    repeated = hierarchical_paired_interval(records, resamples=100, seed=8)

    assert first == repeated
    assert first.lower <= first.estimate <= first.upper


def test_paired_difference_preserves_bundle_pairing() -> None:
    left = (BundleValue("s", "b", 0.8),)
    right = (BundleValue("s", "b", 0.5),)

    assert paired_differences(left, right)[0].value == 0.30000000000000004


def test_gate_report_passes_only_literal_scientific_thresholds() -> None:
    intervals = {
        "bptt_accuracy": ConfidenceInterval(0.9, 0.81, 0.95),
        "readout_only_accuracy": ConfidenceInterval(0.5, 0.45, 0.59),
        "no_recurrence_accuracy": ConfidenceInterval(0.5, 0.45, 0.59),
        "pp_prop_accuracy": ConfidenceInterval(0.7, 0.55, 0.8),
        "pp_prop_minus_frozen_accuracy": ConfidenceInterval(0.1, 0.01, 0.2),
        "cosine_advantage_over_null": ConfidenceInterval(0.2, 0.01, 0.4),
        "pp_prop_small_update_loss_change": ConfidenceInterval(-0.1, -0.2, -0.01),
    }

    report = gate_report(intervals, response_label_independent=True, stability=True)

    assert report["passed"] is True


def test_stability_gate_checks_firing_and_update_ratio() -> None:
    dynamics = {
        "mean_firing_spikes_per_neuron_step": 0.1,
        "silent_neuron_fraction": 0.2,
        "saturated_neuron_fraction": 0.0,
    }

    assert stability_passes(dynamics, np.asarray([0.01, 0.02]))
    assert not stability_passes(dynamics, np.asarray([0.1]))
