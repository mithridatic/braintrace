#!/usr/bin/env python3
"""Report Gate 2 metrics and neural-participation metrics from a run artifact.

Gate 2 asks whether the model has stopped losing to a predictor that uses no
neurons: shape >= 0.85 and pixel >= 0.6336, against the trivial
``copy_or_rule_shape`` floor. The participation block reports the two numbers
the recovery spec uses to describe how much of the model the connectome
actually is -- the recurrent share of parameters and the recurrent share of
drive -- so an edge-growth arm can be compared against its own baseline rather
than asserted.

Usage:  example21_gate2_report.py RESULT_JSON [RESULT_JSON ...]
"""

from __future__ import annotations

import json
import pathlib
import sys

# Trivial-predictor floors re-measured on the restored dataset, from
# docs/specs/2026-08-18-example21-arc-score-recovery.md section 2.
SHAPE_FLOOR = 0.8687
PIXEL_FLOOR = 0.6336


def _dig(node: object, *names: str) -> object:
    """Return the first value found under any of ``names``, depth-first."""
    if isinstance(node, dict):
        for name in names:
            if name in node:
                return node[name]
        for value in node.values():
            found = _dig(value, *names)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _dig(value, *names)
            if found is not None:
                return found
    return None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _arm(result: dict, name: str) -> dict:
    arms = _dig(result.get("evaluation", {}), "arms", "controls")
    if isinstance(arms, dict) and name in arms and isinstance(arms[name], dict):
        return arms[name]
    return result.get("evaluation", {}) if name == "intact" else {}


def _metrics(result: dict) -> dict[str, float | None]:
    evaluation = result.get("evaluation", {})
    by_effort = _dig(evaluation, "metrics_by_effort")
    best = {}
    if isinstance(by_effort, dict):
        # Highest effort is the reported configuration.
        key = max(by_effort, key=lambda item: int(item) if item.isdigit() else -1)
        best = by_effort.get(key, {}) if isinstance(by_effort.get(key), dict) else {}
    return {
        "shape": _number(best.get("shape_accuracy_diagnostic")),
        "pixel": _number(best.get("valid_cell_pixel_accuracy_diagnostic")),
        "pass@1": _number(best.get("strict_task_pass_at_1")),
        "pass@2": _number(best.get("strict_task_pass_at_2")),
        "square_fraction": _number(best.get("shape_square_fraction")),
    }


def _participation(result: dict) -> dict[str, float | None]:
    model = result.get("model", {})
    total = _number(model.get("parameter_count"))
    # One scalar per instantiated recurrent edge, so the edge count is the
    # connectome's parameter count.
    recurrent = _number(model.get("recurrent_edge_count"))
    steps = result.get("evaluation", {}).get("aggregate_trajectory") or []
    # Step zero is where the recovery spec measured the drive split.
    trajectory = steps[0] if isinstance(steps, list) and steps else {}
    return {
        "parameters": total,
        "recurrent_parameters": recurrent,
        "recurrent_parameter_share": (
            recurrent / total if recurrent and total else None
        ),
        "neurons": _number(model.get("neuron_count")),
        "edges": _number(model.get("recurrent_edge_count")),
        "feedforward_l2": _number(trajectory.get("mean_feedforward_current_l2")),
        "recurrent_l2": _number(trajectory.get("mean_recurrent_current_l2")),
        "peak_gpu_bytes": _number(_dig(result.get("device", {}), "peak_bytes_in_use")),
        "gpu_bytes_limit": _number(_dig(result.get("device", {}), "bytes_limit")),
    }


def _show(label: str, value: float | None, floor: float | None = None) -> str:
    if value is None:
        return f"  {label:<28} unavailable"
    verdict = ""
    if floor is not None:
        verdict = "  PASS" if value >= floor else f"  FAIL (floor {floor})"
    return f"  {label:<28} {value:.4f}{verdict}"


def main() -> int:
    paths = [pathlib.Path(argument) for argument in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = _metrics(result)
        participation = _participation(result)
        print(f"\n=== {path.parent.name} ===")
        print(" Gate 2 (beat the no-neuron predictor)")
        print(_show("shape accuracy", metrics["shape"], SHAPE_FLOOR))
        print(_show("pixel accuracy", metrics["pixel"], PIXEL_FLOOR))
        print(_show("shape square fraction", metrics["square_fraction"]))
        print(" Exact score (rule channel included)")
        print(_show("strict pass@1", metrics["pass@1"]))
        print(_show("strict pass@2", metrics["pass@2"]))
        print(" Neural participation")
        for name in (
            "neurons",
            "edges",
            "parameters",
            "recurrent_parameters",
            "recurrent_parameter_share",
            "feedforward_l2",
            "recurrent_l2",
        ):
            print(_show(name.replace("_", " "), participation[name]))
        feedforward = participation["feedforward_l2"]
        recurrent_drive = participation["recurrent_l2"]
        if feedforward and recurrent_drive:
            print(
                _show(
                    "recurrent drive share",
                    recurrent_drive / (feedforward + recurrent_drive),
                )
            )
        peak = participation["peak_gpu_bytes"]
        limit = participation["gpu_bytes_limit"]
        if peak and limit:
            print(
                f"  {'peak gpu':<28} {peak / 2**30:.2f} GiB of "
                f"{limit / 2**30:.2f} GiB ({peak / limit:.1%})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
