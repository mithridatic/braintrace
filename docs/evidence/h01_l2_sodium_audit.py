"""Assess selective sodium recovery without hiding later event losses."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_l2_recovery_audit import _record


def compare_runs(control, candidate, human):
    """Check isolated recovery scaling and the declared first-interval targets.

    Parameters
    ----------
    control, candidate : pathlib.Path
        Factor-one and factor-two run stems from the same mechanism build.
    human : pathlib.Path
        Source-linked human sweep-50 direct datum.

    Returns
    -------
    dict
        Narrow prediction decision and complete observed model events.
    """
    metas = [json.loads(p.with_suffix(".json").read_text()) for p in (control, candidate)]
    reasons = []
    for key in ("model_id", "specimen_id", "sweep", "neuron_version", "dt_ms", "solver",
                "input", "added_bias_na", "initial_mv", "temperature_c", "voltage_convention",
                "fit_sha256", "nseg_factor", "sections", "applied_genome",
                "mechanism_source_sha256", "mechanism_library_sha256"):
        if key not in metas[0] or key not in metas[1] or metas[0][key] != metas[1][key]:
            reasons.append("changed or missing " + key)
    if [m.get("sodium_recovery_factor") for m in metas] != [1., 2.]:
        reasons.append("not the declared recovery-factor pair")
    reference = json.loads(human.read_text())
    intervals = reference.get("intervals_ms", [])
    target = intervals[0] if intervals else float("nan")
    if (reference.get("specimen_id") != 541563728 or reference.get("sweep") != 50 or
            not np.isfinite(target) or target <= 0):
        reasons.append("invalid human first-interval reference")
    records, clocks, currents = [], [], []
    for stem in (control, candidate):
        try:
            record, clock, current = _record(stem, minimum_events=2)
            records.append(record)
            clocks.append(clock)
            currents.append(current)
        except ValueError as error:
            reasons.append(stem.name + ": " + str(error))
    if len(records) == 2:
        length = min(len(t) for t in clocks)
        if (not np.array_equal(clocks[0][:length], clocks[1][:length]) or
                not np.array_equal(currents[0][:length], currents[1][:length])):
            reasons.append("time or input prefix changed")
    report = {"decision": "invalid", "invalid_reasons": reasons, "records": records,
              "qualification": "First-interval mechanism prediction only; later events remain separate targets."}
    if reasons:
        return report
    errors = [r["first_three_intervals_ms"][0] - target for r in records]
    onset = records[1]["events"][0]["rise_crossing_ms"] - records[0]["events"][0]["rise_crossing_ms"]
    peak = records[1]["first_peak_mv"] - records[0]["first_peak_mv"]
    checks = {"first_interval_error_smaller": abs(errors[1]) < abs(errors[0]),
              "first_onset_change_within_0_1_ms": abs(onset) <= .1,
              "first_peak_change_within_0_1_mv": abs(peak) <= .1}
    report.update(decision="supported" if all(checks.values()) else "rejected", checks=checks,
                  first_interval_errors_ms=errors, first_onset_change_ms=onset, first_peak_change_mv=peak,
                  event_counts=[len(r["events"]) for r in records],
                  first_minimum_change_mv=records[1]["first_interspike_minimum_mv"] -
                      records[0]["first_interspike_minimum_mv"])
    return report


def main():
    """Write the prediction audit for two fixed-step sodium-recovery runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "human", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = compare_runs(args.control, args.candidate, args.human)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], report.get("checks"), report["invalid_reasons"])


if __name__ == "__main__":
    main()
