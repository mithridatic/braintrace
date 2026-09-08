"""Audit uniform spatial refinement of the retrieved layer-2 source model."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_l2_time_audit import _trace_report


def compare_runs(coarse, fine, topology):
    """Check fixed setup and tripled section counts before event comparison.

    Parameters
    ----------
    coarse, fine : pathlib.Path
        Saved run stems for the two spatial meshes.
    topology : pathlib.Path
        Source setup replay with parent connections for legacy metadata.

    Returns
    -------
    dict
        Direct event differences, setup checks, and decision.
    """
    metas = [json.loads(p.with_suffix(".json").read_text()) for p in (coarse, fine)]
    replay = json.loads(topology.read_text())
    reasons = []
    for key in ("model_id", "specimen_id", "sweep", "neuron_version", "solver", "input",
                "dt_ms", "initial_mv", "temperature_c", "voltage_convention", "fit_sha256",
                "applied_genome"):
        if key not in metas[0] or key not in metas[1] or metas[0][key] != metas[1][key]:
            reasons.append("changed or missing " + key)
    for key in ("sodium_recovery_factor", "mechanism_source_sha256",
                "mechanism_library_sha256", "calcium_removal_intervention",
                "applied_passive_parameters", "leak_reversal_shift_mv", "sodium_opening_factor", "kv3_closing_factor"):
        if key in metas[0] or key in metas[1]:
            if key not in metas[0] or key not in metas[1] or metas[0][key] != metas[1][key]:
                reasons.append("changed or missing " + key)
    if any(row.get("input") != "source command only; bias not added" or
           row.get("added_bias_na", 0.) != 0. for row in metas):
        reasons.append("not the declared command-only comparison")
    factors = [metas[0].get("nseg_factor", 1), metas[1].get("nseg_factor", 0)]
    if (any(not isinstance(f, int) or isinstance(f, bool) or f < 1 or f % 2 != 1 for f in factors)
            or factors[1] != 3 * factors[0]):
        reasons.append("not a successive odd-factor tripling")
    sections = [{s["name"]: s for s in row["sections"]} for row in metas]
    parents = replay["parent_sections"]
    if (set(sections[0]) != set(sections[1]) or set(sections[0]) != set(parents) or
            not replay.get("all_sections_reach_soma") or
            any(len(s) != len(m["sections"]) for s, m in zip(sections, metas))):
        reasons.append("section names, uniqueness, or topology replay invalid")
    replay_used = []
    for name in sorted(set(sections[0]) & set(sections[1])):
        a, b = sections[0][name], sections[1][name]
        if "parent" not in a:
            replay_used.append(name)
        parent = a.get("parent", parents.get(name, "MISSING"))
        if parent != b.get("parent", "MISSING"):
            reasons.append(name + ": parent changed or missing")
        if b["nseg"] != 3 * a["nseg"]:
            reasons.append(name + ": segment count did not triple")
        for key in ("length_um", "area_um2", "cm_uf_cm2", "ra_ohm_cm"):
            if not np.isclose(a[key], b[key], rtol=1e-10, atol=1e-10):
                reasons.append(name + ": changed " + key)
    report = _trace_report(coarse, fine, reasons)
    report["qualification"] = "Spatial comparison at one source input; no physiological qualification."
    report["baseline_parent_evidence"] = "setup replay for legacy metadata; not original-run parent recording"
    report["sections_using_replayed_parent"] = replay_used
    report["segment_counts"] = [sum(s["nseg"] for s in row["sections"]) for row in metas]
    return report


def main():
    """Write a spatial audit from two run stems and the source topology replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("coarse", "fine", "topology", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = compare_runs(args.coarse, args.fine, args.topology)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], report["counts"], report["max_absolute_differences"], report["invalid_reasons"])


if __name__ == "__main__":
    main()
