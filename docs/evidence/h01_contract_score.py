"""Score every H01 candidate record row by row against the approved human contract.

The contract (``docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md``)
fixes one absolute allowance per measurement. Every residual is model minus human on
the recorded convention. No pooled score is formed. A row is a target when its
residual exceeds the allowance tenfold, real when it exceeds it twofold, and parked
otherwise. Numerical decision limits come from Dixon decision-limit factors applied to
paired numerical repeats (Hartshorne 2020, scan pages p199-p201).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums

FOLDER = Path(__file__).parent
CONTRACT_SPEC = "docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md"
ALLOWANCE = {"rise_crossing_ms": 1., "fall_crossing_ms": 1., "peak_sample_time_ms": 1.,
             "peak_sample_voltage_mv": 1., "time_above_threshold_ms": .05, "rise_to_peak_ms": .05,
             "peak_to_fall_ms": .05, "minimum_voltage_mv": 1., "minimum_delay_ms": 1.,
             "subthreshold_voltage_mv": 1.}
DERIVED = {"interval_ms": 2.}
DLF = {2: 2.95, 3: 1.81, 4: 1.47}
TIERS = ((10., "target"), (2., "real"), (0., "parked"))


def event_rows(time, voltage, human_events, pulse_ms):
    """Contract rows for the count, every event, and the derived intervals.

    Parameters
    ----------
    time, voltage : array
        Model trace on the stimulus clock, in ms and mV.
    human_events : list of dict
        Human event datums with the five per-event fields.
    pulse_ms : tuple
        Stimulus start and end; events are counted inside this window only.

    Returns
    -------
    list of dict
        One row per observation with residual, allowance, and verdict.
    """
    time, voltage = np.asarray(time, dtype=float), np.asarray(voltage, dtype=float)
    mask = (time >= pulse_ms[0]) & (time < pulse_ms[1])
    events = spike_datums(time[mask], voltage[mask])
    rows = [_row("event_count", "count", len(human_events), len(events), None,
                 verdict="pass" if len(events) == len(human_events) else "fail")]
    for i, human in enumerate(human_events, start=1):
        model = events[i-1] if i <= len(events) else None
        for field in ("rise_crossing_ms", "fall_crossing_ms", "peak_sample_time_ms",
                      "peak_sample_voltage_mv", "time_above_threshold_ms"):
            rows.append(_row(f"e{i}_{field}", field, human[field], model[field] if model else None, ALLOWANCE[field]))
        h_rise = human["peak_sample_time_ms"]-human["rise_crossing_ms"]
        h_fall = human["fall_crossing_ms"]-human["peak_sample_time_ms"]
        m_rise = model["peak_sample_time_ms"]-model["rise_crossing_ms"] if model else None
        m_fall = model["fall_crossing_ms"]-model["peak_sample_time_ms"] if model else None
        rows.append(_row(f"e{i}_rise_to_peak_ms", "rise_to_peak_ms", h_rise, m_rise, ALLOWANCE["rise_to_peak_ms"]))
        rows.append(_row(f"e{i}_peak_to_fall_ms", "peak_to_fall_ms", h_fall, m_fall, ALLOWANCE["peak_to_fall_ms"]))
    h_rises = [h["rise_crossing_ms"] for h in human_events]
    m_rises = [e["rise_crossing_ms"] for e in events]
    for i in range(1, len(human_events)):
        model = m_rises[i]-m_rises[i-1] if i < len(m_rises) else None
        rows.append(_row(f"i{i}_ms", "interval_ms", h_rises[i]-h_rises[i-1], model, DERIVED["interval_ms"], derived=True))
    return rows, events


def recovery_rows(time, voltage, events, recovery_datums, pulse_end_ms):
    """Rows for each complete recovery minimum voltage and its delay from the peak."""
    time, voltage = np.asarray(time, dtype=float), np.asarray(voltage, dtype=float)
    rows = []
    for datum in recovery_datums:
        i = datum["event_index"]
        label = f"m{i+1}"
        if i >= len(events):
            rows += [_row(f"{label}_voltage_mv", "minimum_voltage_mv", datum["minimum_sample_voltage_mv"], None, 1.),
                     _row(f"{label}_delay_ms", "minimum_delay_ms", datum["delay_from_peak_ms"], None, 1.)]
            continue
        start = events[i]["fall_crossing_ms"]
        stop = events[i+1]["rise_crossing_ms"] if i+1 < len(events) else pulse_end_ms
        window = (time > start) & (time < stop)
        if datum.get("terminal_window") or not window.any():
            rows += [_row(f"{label}_voltage_mv", "minimum_voltage_mv", datum["minimum_sample_voltage_mv"], None, 1.,
                          verdict="unavailable", note="terminal or empty window; endpoint comparison not issued"),
                     _row(f"{label}_delay_ms", "minimum_delay_ms", datum["delay_from_peak_ms"], None, 1.,
                          verdict="unavailable", note="terminal or empty window")]
            continue
        at = int(np.argmin(voltage[window]))
        rows.append(_row(f"{label}_voltage_mv", "minimum_voltage_mv", datum["minimum_sample_voltage_mv"],
                         float(voltage[window][at]), 1.))
        rows.append(_row(f"{label}_delay_ms", "minimum_delay_ms", datum["delay_from_peak_ms"],
                         float(time[window][at])-events[i]["peak_sample_time_ms"], 1.))
    return rows


def subthreshold_rows(time, voltage, samples):
    """Rows for the fixed subthreshold sample voltages; no extrapolation."""
    time, voltage = np.asarray(time, dtype=float), np.asarray(voltage, dtype=float)
    if time.ndim != 1 or voltage.shape != time.shape or time.size < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("Trace must be one-dimensional, ordered, and paired with its voltage.")
    if not (np.isfinite(time).all() and np.isfinite(voltage).all()):
        raise ValueError("Trace must be finite.")
    rows = []
    for sample in samples:
        requested = sample["requested_time_ms"]
        if requested < time[0] or requested > time[-1]:
            rows.append(_row(f"sub_{requested:.0f}_mv", "subthreshold_voltage_mv", sample["voltage_mv"], None, 1.,
                             verdict="unavailable", note="requested time outside the saved trace"))
            continue
        rows.append(_row(f"sub_{requested:.0f}_mv", "subthreshold_voltage_mv", sample["voltage_mv"],
                         float(np.interp(requested, time, voltage)), 1.))
    return rows


def _row(key, kind, human, model, allowance, verdict=None, note=None, derived=False):
    residual = None if model is None or human is None else float(model)-float(human)
    if verdict is None:
        verdict = "unavailable" if residual is None else ("pass" if abs(residual) <= allowance else "fail")
    ratio = None if residual is None or not allowance else abs(residual)/allowance
    tier = None if ratio is None else next(name for bound, name in TIERS if ratio >= bound)
    return {"key": key, "kind": kind, "human": human, "model": model, "residual": residual,
            "allowance": allowance, "ratio": ratio, "tier": tier, "verdict": verdict,
            "derived": derived, "note": note}


def dixon_limits(vectors, factor=None):
    """Per-key decision limit from two to four numerical repeats of the same physical model.

    Parameters
    ----------
    vectors : list of dict
        Residual rows keyed by observation, one per numerical setting.
    factor : float, optional
        Decision limit factor; defaults to the Dixon table value for ``len(vectors)``.

    Returns
    -------
    dict
        ``limit`` per key equals the range across repeats times the factor.
    """
    count = len(vectors)
    if count not in DLF:
        raise ValueError("Dixon decision limits need two, three, or four repeats.")
    factor = DLF[count] if factor is None else factor
    keys = set.intersection(*(set(v) for v in vectors))
    limits = {}
    for key in sorted(keys):
        values = [v[key] for v in vectors]
        if any(x is None for x in values):
            continue
        limits[key] = (max(values)-min(values))*factor
    return {"repeats": count, "factor": factor, "limits": limits}


def attach_limits(rows, limits):
    """Mark rows whose residual is inside numerical noise as unresolved."""
    for row in rows:
        limit = limits.get(row["key"]) if limits else None
        row["numerical_limit"] = limit
        if limit is not None and row["residual"] is not None and abs(row["residual"]) <= limit:
            row["verdict"] = "unresolved" if row["verdict"] == "fail" else row["verdict"]
    return rows


def score_trace(trace, datums):
    """All rows for one saved model trace against one human input's datums."""
    with np.load(trace) as data:
        time, voltage = np.asarray(data["time_ms"]), np.asarray(data["voltage_mv"])
    if datums.get("events") is not None:
        rows, events = event_rows(time, voltage, datums["events"], datums["pulse_ms"])
        rows += recovery_rows(time, voltage, events, datums.get("recovery", []), datums["pulse_ms"][1])
        return rows
    return subthreshold_rows(time, voltage, datums["samples"])


def load_datums():
    """Human datums per cell and input, on the pinned conventions."""
    requirements = json.loads((FOLDER/"h01-human-event-requirements.json").read_text())
    recovery = json.loads((FOLDER/"h01-human-recovery-datums.json").read_text())
    samples = json.loads((FOLDER/"h01-human-subthreshold-samples.json").read_text())
    out = {}
    for entry in requirements["inputs"]:
        if entry["purpose"] != "calibration" and entry["purpose"] != "candidate calibration trace":
            continue
        key = f"{entry['role']} {entry['command_na']:.2f}"
        record = next((r for r in recovery["records"] if r["reference"] in (entry["role"], f"{entry['role']} {entry['command_na']:.2f} nA")), None)
        out[key] = {"events": entry["events"], "pulse_ms": tuple(entry["pulse_ms"]),
                    "recovery": record["recovery_datums"] if record else []}
    for record in samples["records"]:
        out[f"{record['role']} {record['command_current_na']:.2f}"] = {"events": None, "samples": record["samples"]}
    return out


def summarize(rows):
    counts = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0)+1
    return counts


def render(report):
    lines = ["# Contract scorecard", "",
             "Every residual is model minus human. Allowances are the approved contract values;",
             "interval rows are derived from two crossing allowances and are diagnostic only.",
             "Tier: target = ratio >= 10, real = ratio >= 2, parked otherwise. A failed row inside its",
             "numerical decision limit is reported as unresolved, not as a pass.", "",
             f"Contract: `{CONTRACT_SPEC}` (sha256 {report['contract_sha256'][:12]}).", ""]
    for record in report["records"]:
        counts = record["summary"]
        lines += [f"## {record['cell']} {record['name']} at {record['input']}", "",
                  f"Trace `{record['trace']}`. Settings: {record['settings']}. Verdicts: {counts}.", "",
                  "| Row | Human | Model | Residual | Allowance | Ratio | Tier | Numerical limit | Verdict |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
        for row in record["rows"]:
            if row["verdict"] == "pass" and row["tier"] == "parked":
                continue
            lines.append("| "+" | ".join(_fmt(row[k]) for k in ("key", "human", "model", "residual", "allowance", "ratio", "tier", "numerical_limit", "verdict"))+" |")
        lines.append("")
    lines += ["## Targets (ratio >= 10) by cell", ""]
    for cell, keys in report["targets"].items():
        lines.append(f"- {cell}: "+", ".join(keys) if keys else f"- {cell}: none")
    return "\n".join(lines)+"\n"


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=FOLDER/"h01-contract-records.json")
    parser.add_argument("--output", type=Path, default=FOLDER/"h01-contract-scorecard")
    args = parser.parse_args(argv)
    manifest = json.loads(args.records.read_text())
    datums = load_datums()
    root = FOLDER.parents[1]
    limit_sets = {}
    for pair in manifest.get("numerical_pairs", []):
        vectors = [{r["key"]: r["residual"] for r in score_trace(FOLDER/t, datums[pair["input"]])} for t in pair["traces"]]
        limit_sets[pair["name"]] = dixon_limits(vectors)
        limit_sets[pair["name"]]["traces"] = pair["traces"]
        limit_sets[pair["name"]]["caveat"] = pair.get("caveat")
    report = {"contract": CONTRACT_SPEC, "contract_sha256": hashlib.sha256((root/CONTRACT_SPEC).read_bytes()).hexdigest(),
              "allowances": ALLOWANCE, "derived_allowances": DERIVED, "numerical_limit_sets": limit_sets,
              "records": [], "targets": {}}
    for record in manifest["records"]:
        limits = limit_sets.get(record.get("limits"), {}).get("limits")
        rows = attach_limits(score_trace(FOLDER/record["trace"], datums[record["input"]]), limits)
        entry = dict(record, rows=rows, summary=summarize(rows),
                     trace_sha256=hashlib.sha256((FOLDER/record["trace"]).read_bytes()).hexdigest())
        report["records"].append(entry)
        if record.get("role") == "candidate":
            cell = record["cell"]
            report["targets"].setdefault(cell, [])
            report["targets"][cell] += [f"{record['input']} {r['key']} ({r['residual']:+.3g})" for r in rows
                                        if r["tier"] == "target" and r["verdict"] == "fail"]
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".md").write_text(render(report))
    print(json.dumps({r["name"]+" "+r["input"]: r["summary"] for r in report["records"]}, indent=1))


if __name__ == "__main__":
    main()
