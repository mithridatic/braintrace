"""Score SP16 runs cycle by cycle against the registered predictions.

The spec (docs/specs/2026-09-12-h01-sodium-slow-inactivation.md) predicts per cycle, not
per train: spike 1 unchanged, a threshold step at spike 2 above the decision limit, a hold
from spike 5 on, and the counts inside their bands. This module reads a run's trace with
the same ``cycle_table`` used for the recorded cells and returns each registered check with
its reading, so that a decision file can cite the numbers.
"""

import json
import sys
from pathlib import Path

import numpy as np

from h01_topographic_cycles import cycle_table
from h01_topographic_stage0 import resample

E_PULSE = (1020., 2020.)


def load_run(path):
    """Uniform-grid time and voltage of a saved run (``.npz`` with time_ms and voltage_mv)."""
    data = np.load(path)
    current = data["applied_current_na"] if "applied_current_na" in data else data.get("current_na", np.zeros_like(data["time_ms"]))
    t, v, _ = resample(np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float), np.asarray(current, float))
    return t, v


def reproduction(run, reference, limits):
    """Stage 0: the run must equal the reference (count, spike-1 rise inside the band, thresholds flat)."""
    checks = [
        ("count equal", run["count"] == reference["count"], run["count"], reference["count"]),
    ]
    if run["count"] and reference["count"]:
        r1, f1 = run["spikes"][0], reference["spikes"][0]
        checks += [
            ("spike-1 rise within band", abs(r1["max_rise_v_s"]/f1["max_rise_v_s"]-1.) <= limits["rise_fraction"], r1["max_rise_v_s"], f1["max_rise_v_s"]),
            ("first spike within 1 ms", abs(r1["t_ms"]-f1["t_ms"]) <= 1., r1["t_ms"], f1["t_ms"]),
            ("every threshold within 0.1 mV", all(abs(a["threshold_mv"]-b["threshold_mv"]) <= .1 for a, b in zip(run["spikes"], reference["spikes"])),
             [round(s["threshold_mv"], 2) for s in run["spikes"]], [round(s["threshold_mv"], 2) for s in reference["spikes"]]),
        ]
    return _rows(checks)


def dose_310(run, reference, limits):
    """Stage 1 at 310 pA: spike 1 unchanged, a step at spike 2, a hold from spike 5, count 9 or 10."""
    checks = [("count 9 or 10", run["count"] in (9, 10), run["count"], reference["count"])]
    if run["count"] >= 2 and reference["count"]:
        r1, f1, sh = run["spikes"][0], reference["spikes"][0], run["shape"]
        checks += [
            ("spike-1 rise unchanged", abs(r1["max_rise_v_s"]/f1["max_rise_v_s"]-1.) <= limits["rise_fraction"], r1["max_rise_v_s"], f1["max_rise_v_s"]),
            ("first spike unchanged", abs(r1["t_ms"]-f1["t_ms"]) <= limits["first_spike_ms"], r1["t_ms"], f1["t_ms"]),
            ("threshold step at spike 2 above the limit", sh["spike_2"]["threshold_mv"] >= limits["threshold_mv"], sh["spike_2"]["threshold_mv"], limits["threshold_mv"]),
            ("spike-2 rise fraction (recorded 0.86)", True, sh["spike_2"]["rise_fraction"], .86),
        ]
        if run["count"] >= 10:
            hold = run["spikes"][9]["threshold_mv"]-run["spikes"][4]["threshold_mv"]
            checks.append(("hold from spike 5 (spike 10 minus spike 5 within the limit)", abs(hold) <= limits["threshold_mv"], hold, limits["threshold_mv"]))
    return _rows(checks)


def dose_200(run, reference, limits):
    """Stage 1 at 200 pA: spike 1 unchanged, count falls towards 1, level towards -67.1 mV."""
    checks = [("count falls (at most 3 at depth 0.6; recorded 1)", run["count"] <= 3, run["count"], reference["count"])]
    if run["count"] and reference["count"]:
        r1, f1 = run["spikes"][0], reference["spikes"][0]
        checks += [
            ("spike-1 rise unchanged", abs(r1["max_rise_v_s"]/f1["max_rise_v_s"]-1.) <= limits["rise_fraction"], r1["max_rise_v_s"], f1["max_rise_v_s"]),
            ("first spike unchanged", abs(r1["t_ms"]-f1["t_ms"]) <= limits["first_spike_ms"], r1["t_ms"], f1["t_ms"]),
        ]
    if "level_p50_mv" in run:
        checks.append(("post-spike level towards -67.1 (reference -65.0)", run["level_p50_mv"] < reference.get("level_p50_mv", -65.), run["level_p50_mv"], -67.1))
    return _rows(checks)


def _rows(checks):
    return [{"check": name, "pass": bool(ok), "run": value, "reference": ref} for name, ok, value, ref in checks]


def score(run_path, reference_path, kind, limits, pulse=E_PULSE):
    """Read both traces, tabulate their cycles and apply the registered checks of ``kind``."""
    t, v = load_run(run_path)
    run = cycle_table(t, v, pulse)
    run["level_p50_mv"] = float(np.median(v[(t >= 1800.) & (t < 2000.)]))
    t, v = load_run(reference_path)
    reference = cycle_table(t, v, pulse)
    reference["level_p50_mv"] = float(np.median(v[(t >= 1800.) & (t < 2000.)]))
    rows = {"reproduction": reproduction, "310": dose_310, "200": dose_200}[kind](run, reference, limits)
    return {"kind": kind, "run": str(run_path), "reference": str(reference_path), "checks": rows,
            "all_pass": all(r["pass"] for r in rows), "run_cycles": run, "reference_cycles": reference}


def main(argv):
    run_path, reference_path, kind = Path(argv[0]), Path(argv[1]), argv[2]
    manifest = json.loads(Path(argv[3]).read_text(encoding="utf-8")) if len(argv) > 3 else {"decision_limits": {"threshold_mv": 1., "rise_fraction": .035, "first_spike_ms": 23.}}
    result = score(run_path, reference_path, kind, manifest["decision_limits"])
    for row in result["checks"]:
        print(("PASS" if row["pass"] else "FAIL"), row["check"], "run", row["run"], "reference", row["reference"])
    print("all_pass", result["all_pass"])
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
