"""Decide the declared layer-2 calcium-removal prediction from direct traces."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def _record(stem, minimum_events=4):
    with np.load(stem.with_suffix(".npz")) as data:
        t, v, current = (data[key] for key in ("time_ms", "voltage_mv", "applied_current_na"))
    if (t.ndim != 1 or len(t) < 2 or v.shape != t.shape or current.shape != t.shape or
            not all(np.isfinite(a).all() for a in (t, v, current)) or
            not np.all(np.diff(t) > 0) or t[0] != 0 or t[-1] < 2020):
        raise ValueError("invalid trace or observation coverage")
    mask = (t >= 1020) & (t < 2020)
    if mask.sum() < 2 or v[mask][0] >= -20 or v[mask][-1] >= -20:
        raise ValueError("incomplete pulse-boundary event")
    events = spike_datums(t[mask], v[mask])
    if len(events) < minimum_events:
        raise ValueError("fewer than required complete events")
    if any(e["peak_sample_voltage_mv"] <= 0 for e in events[:minimum_events]):
        raise ValueError("required events lack positive peaks")
    between = (t > events[0]["peak_sample_time_ms"]) & (t < events[1]["rise_crossing_ms"])
    record = {"events": events,
              "first_three_intervals_ms": np.diff([e["rise_crossing_ms"] for e in events[:4]]).tolist(),
              "first_interspike_minimum_mv": float(v[between].min()),
              "first_peak_mv": events[0]["peak_sample_voltage_mv"]}
    return record, t, current


def compare_runs(control, candidate, human):
    """Check intervention isolation and the declared recovery prediction.

    Parameters
    ----------
    control, candidate : pathlib.Path
        Saved run stems on the selected common mesh.
    human : pathlib.Path
        The source-linked sweep-50 direct datum JSON.

    Returns
    -------
    dict
        Decision with all direct intervals and explicit invalidity reasons.
    """
    metas = [json.loads(p.with_suffix(".json").read_text()) for p in (control, candidate)]
    reasons = []
    for key in ("model_id", "specimen_id", "sweep", "neuron_version", "dt_ms", "solver",
                "input", "initial_mv", "temperature_c", "voltage_convention", "fit_sha256",
                "nseg_factor", "sections"):
        if key not in metas[0] or key not in metas[1] or metas[0][key] != metas[1][key]:
            reasons.append("changed or missing " + key)
    if any(m.get("input") != "source command only; bias not added" or
           m.get("added_bias_na", 0.) != 0. for m in metas):
        reasons.append("input is not the declared command-only control")
    expected = copy.deepcopy(metas[0]["applied_genome"])
    targets = [r for r in expected if r["section"] == "soma" and
               r["mechanism"] == "CaDynamics" and r["name"] == "decay_CaDynamics"]
    if len(targets) != 1:
        reasons.append("source calcium-removal target is not unique")
    else:
        targets[0]["value"] *= 2
        if expected != metas[1]["applied_genome"]:
            reasons.append("genome change is not exactly doubled somatic removal time")
    reference = json.loads(human.read_text())
    target = np.asarray(reference["intervals_ms"][:3])
    if (reference.get("specimen_id") != 541563728 or reference.get("sweep") != 50 or
            target.shape != (3,) or not np.isfinite(target).all() or not np.all(target > 0)):
        reasons.append("wrong or incomplete human datum")
    records, clocks, currents = [], [], []
    for stem in (control, candidate):
        try:
            record, clock, current = _record(stem)
            records.append(record)
            clocks.append(clock)
            currents.append(current)
        except ValueError as error:
            reasons.append(stem.name + ": " + str(error))
    if len(records) == 2:
        common = min(len(t) for t in clocks)
        if (not np.array_equal(clocks[0][:common], clocks[1][:common]) or
                not np.array_equal(currents[0][:common], currents[1][:common])):
            reasons.append("time or applied-current prefix changed")
    report = {"decision": "invalid", "invalid_reasons": reasons, "records": records,
              "qualification": "Conditional diagnostic prediction; no physiological qualification."}
    if reasons:
        return report
    errors = [np.asarray(r["first_three_intervals_ms"]) - target for r in records]
    improved = (np.abs(errors[1]) < np.abs(errors[0])).tolist()
    minimum_change = records[1]["first_interspike_minimum_mv"] - records[0]["first_interspike_minimum_mv"]
    peak_change = records[1]["first_peak_mv"] - records[0]["first_peak_mv"]
    checks = {"each_first_three_interval_error_smaller": all(improved),
              "first_minimum_not_deepened_over_0_1_mv": minimum_change >= -.1,
              "first_peak_change_within_0_1_mv": abs(peak_change) <= .1}
    report.update(decision="supported" if all(checks.values()) else "rejected", checks=checks,
                  interval_errors_ms=[e.tolist() for e in errors], each_interval_improved=improved,
                  first_minimum_change_mv=minimum_change, first_peak_change_mv=peak_change)
    return report


def main():
    """Write the direct recovery decision for a saved intervention pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "human", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = compare_runs(args.control, args.candidate, args.human)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], report.get("checks"), report["invalid_reasons"])


if __name__ == "__main__":
    main()
