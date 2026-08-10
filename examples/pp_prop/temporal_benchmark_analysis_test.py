"""Tests for fail-closed sealed benchmark analysis."""

import json

import pytest

from temporal_benchmark_analysis import analyze_sealed_results
from temporal_benchmark_config import ARMS


def _accuracy(arm: str) -> float:
    return {
        "all_bptt": 0.90,
        "readout_only": 0.50,
        "no_recurrent_module": 0.50,
        "all_pp_prop": 0.90,
        "frozen_random_recurrence": 0.70,
    }.get(arm, 0.60)


def _document(arm: str, bundle_id: str) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "completed",
        "arm": arm,
        "bundle_id": bundle_id,
        "sealed_test_metrics": {"ensemble_accuracy": _accuracy(arm)},
        "response_label_independent": True,
        "dynamics": {
            "mean_firing_spikes_per_neuron_step": 0.1,
            "silent_neuron_fraction": 0.1,
            "saturated_neuron_fraction": 0.0,
        },
        "optimizer_telemetry": {"recurrent": {"update_to_weight_ratio": [0.001]}},
    }
    if arm == "all_pp_prop":
        result["gradient_evidence"] = {
            "probes": [
                {
                    "cosine_advantage_over_permuted_null": 0.4,
                    "pp_prop_small_update_loss_change": -0.1,
                }
            ]
        }
    return {
        "schema_version": 1,
        "sealed_test": True,
        "environment": {"source_dirty": False},
        "result": result,
    }


def _matrix(tmp_path):
    paths = []
    for arm in ARMS:
        for split in range(3):
            for topology in range(2):
                for weight in range(2):
                    bundle = f"split{split}-topology{topology}-weight{weight}"
                    path = tmp_path / f"{arm}-{bundle}.json"
                    path.write_text(
                        json.dumps(_document(arm, bundle)), encoding="utf-8"
                    )
                    paths.append(path)
    return paths


def test_complete_sealed_matrix_passes_all_episodic_gates(tmp_path) -> None:
    report = analyze_sealed_results(_matrix(tmp_path))

    assert report["passed"] is True
    assert all(report["gates"].values())


def test_incomplete_matrix_fails_closed(tmp_path) -> None:
    paths = _matrix(tmp_path)

    with pytest.raises(ValueError, match="exactly 12"):
        analyze_sealed_results(paths[:-1])


def test_duplicate_bundle_fails_closed(tmp_path) -> None:
    paths = _matrix(tmp_path)
    paths[11] = paths[0]

    with pytest.raises(ValueError, match="duplicate sealed bundles"):
        analyze_sealed_results(paths)
