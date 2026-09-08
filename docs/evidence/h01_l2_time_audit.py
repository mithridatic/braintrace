"""Check direct layer-2 event changes under successive time refinement."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def compare_runs(coarse, fine):
    """Compare saved runs against the declared temporal limits.

    Parameters
    ----------
    coarse, fine : pathlib.Path
        Run stems with JSON metadata and NPZ traces.

    Returns
    -------
    dict
        Decision, direct residuals, and explicit invalidity reasons.
    """
    metadata = [json.loads(stem.with_suffix(".json").read_text()) for stem in (coarse, fine)]
    invariant_keys = ("model_id", "specimen_id", "sweep", "neuron_version", "solver", "input",
                      "initial_mv", "temperature_c", "voltage_convention", "fit_sha256",
                      "applied_genome", "sections")
    reasons = ["changed or missing " + key for key in invariant_keys
               if key not in metadata[0] or key not in metadata[1] or metadata[0][key] != metadata[1][key]]
    biases = [row.get("added_bias_na", 0. if row.get("input") == "source command only; bias not added"
                      else np.nan) for row in metadata]
    for key in ("applied_passive_parameters", "leak_reversal_shift_mv", "sodium_opening_factor", "kv3_closing_factor"):
        if key in metadata[0] or key in metadata[1]:
            if key not in metadata[0] or key not in metadata[1] or metadata[0][key] != metadata[1][key]:
                reasons.append("changed or missing " + key)
    if not np.isfinite(biases).all() or biases[0] != biases[1]:
        reasons.append("changed or missing added bias")
    steps = [row.get("dt_ms", np.nan) for row in metadata]
    if not all(np.isfinite(dt) and dt > 0 for dt in steps) or steps[0] != 2 * steps[1]:
        reasons.append("time step was not halved")
    return _trace_report(coarse, fine, reasons)


def _trace_report(coarse, fine, reasons):
    """Retain every direct event and apply the shared numerical limits."""
    limits = {"rise_crossing_ms": .1, "peak_sample_voltage_mv": .1,
              "time_above_threshold_ms": .01}
    records = []
    for stem in (coarse, fine):
        with np.load(stem.with_suffix(".npz")) as trace:
            t, v = trace["time_ms"], trace["voltage_mv"]
        if (t.ndim != 1 or v.shape != t.shape or len(t) < 2 or
                not np.isfinite(t).all() or not np.isfinite(v).all() or
                not np.all(np.diff(t) > 0) or t[0] > 1020 or t[-1] < 2020):
            reasons.append(stem.name + ": invalid trace or pulse coverage")
            records.append([])
            continue
        mask = (t >= 1020) & (t < 2020)
        if mask.sum() < 2 or v[mask][0] >= -20 or v[mask][-1] >= -20:
            reasons.append(stem.name + ": missing or incomplete pulse-boundary event")
            records.append([])
            continue
        records.append(spike_datums(t[mask], v[mask]))
    if not all(records):
        reasons.append("missing complete events")
    differences = [{"ordinal": index + 1, **{key: b[key] - a[key] for key in limits}}
                   for index, (a, b) in enumerate(zip(*records))]
    maxima = {key: max((abs(row[key]) for row in differences), default=None) for key in limits}
    counts = [len(row) for row in records]
    same_signs = counts[0] == counts[1] and all(
        (a["peak_sample_voltage_mv"] > 0) == (b["peak_sample_voltage_mv"] > 0)
        for a, b in zip(*records))
    passed = not reasons and same_signs and all(maxima[key] <= value for key, value in limits.items())
    return {"decision": "invalid" if reasons else "supported" if passed else "rejected",
            "invalid_reasons": reasons, "counts": counts, "equal_peak_signs": same_signs,
            "limits": limits, "max_absolute_differences": maxima,
            "ordinal_differences_fine_minus_coarse": differences,
            "unmatched_events": {stem.name: events[min(counts):]
                                 for stem, events in zip((coarse, fine), records)},
            "qualification": "Temporal comparison only; no spatial or physiological qualification."}


def main():
    """Write the audit for two run stems supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare_runs(args.coarse, args.fine)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], report["counts"], report["max_absolute_differences"])


if __name__ == "__main__":
    main()
