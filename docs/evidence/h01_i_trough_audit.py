"""Stage 0 of the I reserve: gate state and axial drive at every trough of the retained traces.

The open Y3 row is the post-trough drive: the human refires within 6 to 10 ms of a
-79 mV trough and the finalist does not. Two mechanisms on the recorded axon-first
pathway remain, sodium availability at the soma (recovery) and inward current from
the axon (drive). This audit reads the retained probes (soma NaTg h and m, Kv3 m,
soma and axon voltage, axial neighbours) at each trough, trough + 6 ms and
trough + 10 ms, finalist against source at both inputs, and applies the rule
pre-registered in ``docs/specs/2026-09-07-h01-i-reserve.md`` to name the arm the
single reserve evaluation tests. No model is run.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_pv_human_datums import spike_datums
from h01_spike_cycle_energetics import DETECTION_MV, landmarks, pv_currents

PULSE_MS = (270., 1270.)
OFFSETS_MS = (0., 6., 10.)
PROBES = {"soma_h": "NaTg_h", "soma_m": "NaTg_m", "kv3_m": "Kv3_1_m",
          "soma_mv": "voltage_mv", "axon_mv": "axon_voltage_mv"}
RULE = {"available_at_least": .7, "limited_below": .5, "offset_ms": 6.,
        "statistic": "median over unflagged cycles, smaller of the two inputs"}
DRIVE_FLAGS = {"scale": ["NaTg:axon:1.5"]}
RECOVERY_FLAGS = {"sodium_h_recovery_factor": .5}
STEMS = {"finalist": "e-kv3-close2", "source": "r-source"}
INPUTS = ("019", "027")
EVIDENCE = Path(__file__).resolve().parent


def trough_rows(data, geometry, pulse_ms=PULSE_MS, offsets=OFFSETS_MS):
    """Sample every probe at each trough and at fixed offsets after it.

    Parameters
    ----------
    data : mapping
        Arrays of a ``--observe-charge-balance`` run (``time_ms``, ``voltage_mv``,
        gate probes, ``axial_neighbour_*_mv``).
    geometry : dict
        ``charge_balance_geometry`` of the matching report.
    pulse_ms : tuple of float
        Pulse window handed to ``landmarks``.
    offsets : tuple of float
        Offsets after the trough, in ms, at which the probes are read.

    Returns
    -------
    list of dict
        One row per (cycle, offset) with the interpolated probes, the inward-positive
        axial current into the soma in nA, and ``after_next_spike`` when the sample
        time lies beyond the next spike's -20 mV rise crossing.
    """
    time = np.asarray(data["time_ms"], float)
    voltage = np.asarray(data["voltage_mv"], float)
    axial = pv_currents(data, geometry)["axial"]
    rises = np.array([e["rise_crossing_ms"] for e in spike_datums(time, voltage, DETECTION_MV)])
    rows = []
    for row in landmarks(time, voltage, pulse_ms):
        later = rises[rises > row["minimum_ms"]]
        next_rise = float(later[0]) if len(later) else None
        for offset in offsets:
            at = row["minimum_ms"]+offset
            sample = {name: float(np.interp(at, time, np.asarray(data[key], float))) for name, key in PROBES.items()}
            rows.append({"cycle": row["spike"], "trough_ms": row["minimum_ms"], "offset_ms": float(offset),
                         "at_ms": float(at), "after_next_spike": next_rise is not None and at > next_rise,
                         **sample, "axial_na": float(np.interp(at, time, axial))})
    return rows


def availability(rows, offset_ms=RULE["offset_ms"]):
    """Median soma ``h`` at ``offset_ms`` after the trough over cycles the next spike has not reached."""
    values = [r["soma_h"] for r in rows if r["offset_ms"] == offset_ms and not r["after_next_spike"]]
    return float(np.median(values)) if values else None


def decide(h_by_input, rule=RULE):
    """Apply the pre-registered rule to the finalist's per-input availability.

    Parameters
    ----------
    h_by_input : dict
        Input name to median soma ``h`` at trough + 6 ms; ``None`` when no cycle qualified.

    Returns
    -------
    dict
        ``arm`` (``drive`` or ``recovery``), ``reading``, the value used, and the arm flags.
    """
    missing = [k for k, v in h_by_input.items() if v is None]
    if missing:
        raise ValueError(f"Input(s) {', '.join(missing)} have no unflagged trough + 6 ms sample; the rule cannot be applied.")
    used = min(h_by_input.values())
    if used >= rule["available_at_least"]:
        reading, arm = "available", "drive"
        text = f"soma h {used:.3f} >= {rule['available_at_least']} at trough + 6 ms: availability is not the limit"
    elif used < rule["limited_below"]:
        reading, arm = "limited", "recovery"
        text = f"soma h {used:.3f} < {rule['limited_below']} at trough + 6 ms: recovery is the limit"
    else:
        reading, arm = "ambiguous", "drive"
        text = f"soma h {used:.3f} between {rule['limited_below']} and {rule['available_at_least']}: ambiguous, the axon already leads"
    flags = DRIVE_FLAGS if arm == "drive" else RECOVERY_FLAGS
    return {"decision": f"{text}; the reserve tests post-trough {arm}", "arm": arm, "reading": reading,
            "h_used": used, "h_by_input": dict(h_by_input), "rule": dict(rule), "arm_flags": dict(flags)}


def candidate_flags(finalist_flags, arm_flags):
    """Finalist flag file with the arm's flags laid over it (the old scaling slot is untouched)."""
    return {**finalist_flags, **arm_flags}


def load_run(folder, stem):
    """Arrays and report of one retained run."""
    data = dict(np.load(Path(folder)/f"{stem}.npz"))
    report = json.loads((Path(folder)/f"{stem}.json").read_text())
    return data, report


def audit(folder, stems=STEMS, inputs=INPUTS, pulse_ms=PULSE_MS):
    """Rows for every model and input, the finalist decision, and the source control reading."""
    rows = {role: {} for role in stems}
    for role, stem in stems.items():
        for name in inputs:
            data, report = load_run(folder, f"{stem}-{name}")
            rows[role][name] = trough_rows(data, report["charge_balance_geometry"], pulse_ms)
    finalist = {name: availability(rows["finalist"][name]) for name in inputs}
    source = {name: availability(rows["source"][name]) for name in inputs}
    candidate = Path(folder)/f"{stems['finalist']}.candidate.json"
    finalist_flags = json.loads(candidate.read_text()) if candidate.exists() else {}
    decision = decide(finalist)
    decision["finalist_flags"] = finalist_flags
    decision["candidate_flags"] = candidate_flags(finalist_flags, decision["arm_flags"])
    return {"purpose": __doc__.strip().splitlines()[0], "folder": str(folder), "stems": dict(stems),
            "pulse_ms": list(pulse_ms), "offsets_ms": list(OFFSETS_MS), "probes": dict(PROBES),
            "axial_convention": "inward positive, sum((V_neighbour - V_soma) / R_axial) in nA",
            "rows": rows, "source_control": source, "decision": decision}


def _fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def render(report):
    """Markdown page: the retained rows per model and input, then the decision."""
    columns = ("cycle", "offset_ms", "at_ms", "after_next_spike", "soma_h", "soma_m", "kv3_m", "soma_mv", "axon_mv", "axial_na")
    lines = ["# I trough audit (SP4 stage 0)", "", report["purpose"], "",
             f"Runs: `{report['folder']}`, stems {report['stems']}, pulse {report['pulse_ms']} ms. "
             f"Axial current: {report['axial_convention']}.", ""]
    for role, per_input in report["rows"].items():
        for name, rows in per_input.items():
            lines += [f"## {role}, input {name}", "", "| "+" | ".join(columns)+" |", "|"+" --- |"*len(columns)]
            lines += ["| "+" | ".join(_fmt(r[c]) for c in columns)+" |" for r in rows]
            lines.append("")
    decision = report["decision"]
    lines += ["## Decision", "", f"Rule: {decision['rule']}", "",
              f"Finalist soma h at trough + 6 ms by input: {decision['h_by_input']}; used {decision['h_used']:.4g}.",
              f"Source control (same statistic): {report['source_control']}.", "",
              f"**{decision['decision']}.** Arm `{decision['arm']}`, flags {decision['arm_flags']}.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=EVIDENCE/"h01-i-energetic")
    parser.add_argument("--finalist", default=STEMS["finalist"])
    parser.add_argument("--source", default=STEMS["source"])
    parser.add_argument("--pulse-ms", type=float, nargs=2, default=PULSE_MS)
    parser.add_argument("--out", type=Path, default=EVIDENCE/"h01-i-trough-audit", help="stem for .json and .md")
    parser.add_argument("--decision", type=Path, default=EVIDENCE/"h01-i-reserve"/"stage-0-decision.json")
    args = parser.parse_args(argv)
    report = audit(args.folder, {"finalist": args.finalist, "source": args.source}, pulse_ms=tuple(args.pulse_ms))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    args.out.with_suffix(".md").write_text(render(report))
    decision = report["decision"]
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(json.dumps({"stage": "0", "audit": args.out.with_suffix(".json").name, **decision}, indent=2))
    print(json.dumps({k: decision[k] for k in ("decision", "arm", "h_by_input")}))


if __name__ == "__main__":
    main()
