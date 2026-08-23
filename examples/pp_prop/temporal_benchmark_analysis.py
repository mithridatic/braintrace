"""Fail-closed hierarchical analysis of sealed Example 17 results."""

from __future__ import annotations

import msgspec_json
from pathlib import Path
from typing import Iterable

import numpy as np

from temporal_benchmark_config import ARMS
from temporal_benchmark_metrics import (
    BundleValue,
    gate_report,
    hierarchical_paired_interval,
    paired_differences,
    stability_passes,
)

EXPECTED_BUNDLES = 12


def _load_result(path: Path) -> dict[str, object]:
    document = msgspec_json.loads(path.read_text(encoding="utf-8"))
    environment = document.get("environment", {})
    if document.get("schema_version") != 1 or document.get("sealed_test") is not True:
        raise ValueError(f"Unsealed or unsupported result: {path}. Use a supported option or change the configuration.")
    if (
        not isinstance(environment, dict)
        or environment.get("source_dirty") is not False
    ):
        raise ValueError(f"Sealed result lacks clean provenance: {path}. Provide the missing item named in the message.")
    result = document.get("result")
    if not isinstance(result, dict) or result.get("status") not in {
        "completed",
        "stopped",
    }:
        raise ValueError(f"Result is incomplete: {path}. Fix the input condition named in the error, then rerun the operation.")
    if not isinstance(result.get("sealed_test_metrics"), dict):
        raise ValueError(f"Result lacks sealed test metrics: {path}. Provide the missing item named in the message.")
    return result


def _split_id(bundle_id: str) -> str:
    split_id = bundle_id.split("-", maxsplit=1)[0]
    if split_id not in {"split0", "split1", "split2"}:
        raise ValueError(f"Invalid bundle split: {bundle_id}. Set the named field to a value in the stated range, then rerun the operation.")
    return split_id


def _arm_records(results: Iterable[dict[str, object]], arm: str):
    records = []
    for result in results:
        if result["arm"] != arm:
            continue
        bundle_id = str(result["bundle_id"])
        metrics = result["sealed_test_metrics"]
        assert isinstance(metrics, dict)
        records.append(
            BundleValue(
                _split_id(bundle_id), bundle_id, float(metrics["ensemble_accuracy"])
            )
        )
    if len(records) != EXPECTED_BUNDLES:
        raise ValueError(f"Arm {arm} must contain exactly 12 sealed bundles. Add exactly 12 sealed bundles to Arm {arm}.")
    if len({record.bundle_id for record in records}) != EXPECTED_BUNDLES:
        raise ValueError(f"Arm {arm} contains duplicate sealed bundles. Fix the input condition named in the error, then rerun the operation.")
    return tuple(records)


def _gradient_records(results: Iterable[dict[str, object]], field: str):
    records = []
    for result in results:
        if result["arm"] != "all_pp_prop":
            continue
        evidence = result.get("gradient_evidence")
        if not isinstance(evidence, dict) or not isinstance(
            evidence.get("probes"), list
        ):
            raise ValueError("Each pp-prop result requires gradient evidence. Provide the required value for Each pp-prop result.")
        values = [float(probe[field]) for probe in evidence["probes"]]
        bundle_id = str(result["bundle_id"])
        records.append(
            BundleValue(_split_id(bundle_id), bundle_id, float(np.mean(values)))
        )
    return tuple(records)


def _all_stable(results: Iterable[dict[str, object]]) -> bool:
    for result in results:
        if result["arm"] != "all_pp_prop":
            continue
        telemetry = result["optimizer_telemetry"]
        assert isinstance(telemetry, dict)
        recurrent = telemetry.get("recurrent")
        if not isinstance(recurrent, dict):
            return False
        ratios = np.asarray(recurrent["update_to_weight_ratio"], dtype=np.float64)
        dynamics = result["dynamics"]
        if not isinstance(dynamics, dict) or not stability_passes(dynamics, ratios):
            return False
    return True


def analyze_sealed_results(paths: Iterable[Path]) -> dict[str, object]:
    """Compute all episodic gates from a complete sealed result matrix."""
    results = tuple(_load_result(path) for path in paths)
    by_arm = {arm: _arm_records(results, arm) for arm in ARMS}
    intervals = {
        "bptt_accuracy": hierarchical_paired_interval(by_arm["all_bptt"]),
        "readout_only_accuracy": hierarchical_paired_interval(by_arm["readout_only"]),
        "no_recurrence_accuracy": hierarchical_paired_interval(
            by_arm["no_recurrent_module"]
        ),
        "pp_prop_accuracy": hierarchical_paired_interval(by_arm["all_pp_prop"]),
        "pp_prop_minus_frozen_accuracy": hierarchical_paired_interval(
            paired_differences(
                by_arm["all_pp_prop"], by_arm["frozen_random_recurrence"]
            )
        ),
        "cosine_advantage_over_null": hierarchical_paired_interval(
            _gradient_records(results, "cosine_advantage_over_permuted_null")
        ),
        "pp_prop_small_update_loss_change": hierarchical_paired_interval(
            _gradient_records(results, "pp_prop_small_update_loss_change")
        ),
    }
    independent = all(
        result.get("response_label_independent") is True for result in results
    )
    return gate_report(
        intervals,
        response_label_independent=independent,
        stability=_all_stable(results),
    )
