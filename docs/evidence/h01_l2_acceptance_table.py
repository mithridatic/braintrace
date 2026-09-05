"""Assemble the layer-2 acceptance table from the recorded human datums.

Every row names the measurement convention, the datum, the allowed error, the
basis for that error, and the pass rule. No allowance has been agreed, so every
row is flagged and no pass or fail is issued. Numerical decision limits come
from the frozen density candidate's tolerance, mesh, and minima comparisons.
"""

import json
from pathlib import Path

FOLDER = Path(__file__).parent
FIELDS = ("rise_crossing_ms", "fall_crossing_ms", "time_above_threshold_ms", "peak_sample_time_ms", "peak_sample_voltage_mv")
NO_ALLOWANCE = "none declared (engineering choice pending agreement)"


def _load(name):
    return json.loads((FOLDER/name).read_text())


def numerical_limits(tolerance, spatial, minima):
    """Largest recorded numerical change per observation kind for the frozen candidate."""
    fields = {k: max(tolerance["max_absolute_differences"][k], spatial["max_absolute_differences"][k]) for k in tolerance["limits"]}
    phase = max(tolerance["max_phase_difference_ms"], spatial["max_phase_difference_ms"])
    rows = [abs(d["voltage_mv"]) for c in minima["comparisons"].values() for d in c["individual_differences"]]
    return {"fields": fields, "phase_ms": phase, "minimum_voltage_mv": max(rows)}


def build_rows(requirements, recovery, subthreshold, candidate, limits):
    """Return one dictionary per required observation for the E calibration inputs."""
    human = next(i for i in requirements["inputs"] if i["role"] == "E")
    rows = [{"observation": "event count", "convention": requirements["event_matching"], "datum": len(human["events"]),
             "candidate_residual": f"{len(candidate['events'])-len(human['events'])} events; unmatched {candidate['unmatched_events']}",
             "numerical_limit": "equal counts required", "allowance": "exact", "rule": "equal count and no unmatched event"}]
    for index, (event, residual) in enumerate(zip(human["events"], candidate["event_residuals"]), start=1):
        for field in FIELDS:
            rows.append({"observation": f"event {index} {field}", "convention": "-20 mV crossings interpolated; peak on the sample grid",
                         "datum": event[field], "candidate_residual": residual[field],
                         "numerical_limit": limits["fields"].get(field, limits["phase_ms"]), "allowance": NO_ALLOWANCE, "rule": "undefined until allowance agreed"})
        for name, phase, error in zip(("rise to peak", "peak to fall"),
                                      (event["peak_sample_time_ms"]-event["rise_crossing_ms"], event["fall_crossing_ms"]-event["peak_sample_time_ms"]),
                                      candidate["phase_residuals_ms"][index-1]):
            rows.append({"observation": f"event {index} {name} ms", "convention": "sample peak minus rise crossing; fall crossing minus sample peak",
                         "datum": phase, "candidate_residual": error, "numerical_limit": limits["phase_ms"],
                         "allowance": NO_ALLOWANCE, "rule": "undefined until allowance agreed"})
    rises = [e["rise_crossing_ms"] for e in human["events"]]
    for index, (a, b, error) in enumerate(zip(rises, rises[1:], candidate["interval_errors_ms"]), start=1):
        rows.append({"observation": f"interval {index} ms", "convention": "rise crossing to next rise crossing", "datum": b-a,
                     "candidate_residual": error, "numerical_limit": limits["fields"]["rise_crossing_ms"],
                     "allowance": NO_ALLOWANCE, "rule": "undefined until allowance agreed"})
    record = next(r for r in recovery["records"] if r["reference"] == "E")
    for datum in record["recovery_datums"]:
        index = datum["event_index"]
        error = candidate["minimum_voltage_errors_mv"][index] if index < len(candidate["minimum_voltage_errors_mv"]) else "not computed"
        flag = "; terminal window (disposition pending)" if datum["terminal_window"] else ""
        rows.append({"observation": f"recovery minimum {index+1} voltage mV", "convention": recovery["definition"]+flag,
                     "datum": datum["minimum_sample_voltage_mv"], "candidate_residual": error,
                     "numerical_limit": limits["minimum_voltage_mv"], "allowance": NO_ALLOWANCE, "rule": "undefined until allowance agreed"})
        rows.append({"observation": f"recovery minimum {index+1} delay from peak ms", "convention": "minimum sample time minus peak sample time"+flag,
                     "datum": datum["delay_from_peak_ms"], "candidate_residual": "not computed on this convention (result records delay after fall)",
                     "numerical_limit": "not measured", "allowance": NO_ALLOWANCE, "rule": "undefined until allowance agreed"})
    samples = next(r for r in subthreshold["records"] if r["role"] == "E")
    for sample in samples["samples"]:
        rows.append({"observation": f"subthreshold {sample['requested_time_ms']:.0f} ms voltage mV",
                     "convention": f"sample at requested time under {samples['command_current_na']} nA", "datum": sample["voltage_mv"],
                     "candidate_residual": "see subthreshold result", "numerical_limit": "not measured per sample",
                     "allowance": NO_ALLOWANCE, "rule": "undefined until allowance agreed"})
    return rows


def render(rows, sources):
    """Markdown table with the explicit no-pass statement."""
    lines = ["# Layer-2 acceptance table", "",
             "No physiological allowance has been agreed. Every row is flagged. This table",
             "issues no pass and no fail; it fixes the conventions, datums, and the measured",
             "numerical decision limits that any allowance must exceed. Candidate residuals are",
             "the frozen h01-l2-kv3-ninety-ca133 model minus the human datum.", "",
             "| Observation | Convention | Datum | Candidate residual | Numerical limit | Allowed error | Pass rule |", "| --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else str(v)
    lines += ["| "+" | ".join(fmt(row[k]) for k in ("observation", "convention", "datum", "candidate_residual", "numerical_limit", "allowance", "rule"))+" |" for row in rows]
    lines += ["", "Sources: "+", ".join(f"[{s}]({s})" for s in sources), ""]
    return "\n".join(lines)


def main():
    sources = ("h01-human-event-requirements.json", "h01-human-recovery-datums.json", "h01-human-subthreshold-samples.json",
               "h01-l2-kv3-ninety-ca133-result.json", "h01-l2-density130-tolerance-result.json",
               "h01-l2-density130-spatial-result.json", "h01-l2-density130-minima-numerical-review.json")
    data = [_load(s) for s in sources]
    limits = numerical_limits(*data[4:])
    rows = build_rows(data[0], data[1], data[2], data[3]["candidate"], limits)
    (FOLDER/"h01-l2-acceptance-table.md").write_text(render(rows, sources))
    print(len(rows), "rows;", sum(r["allowance"] == NO_ALLOWANCE for r in rows), "without an agreed allowance")


if __name__ == "__main__":
    main()
