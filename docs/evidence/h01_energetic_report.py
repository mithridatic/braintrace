"""Per-boundary charge budgets per spike cycle from the Stage 3 reference runs.

For every candidate and input in an energetic manifest's output directory, load the
probed trace, build inward-positive currents in nA, segment cycles, and integrate
the charge through each boundary over three windows: the upstroke (threshold to
peak), the falling phase (peak to minimum) and the whole cycle. The closure
residual of the whole-cycle budget is the numerical floor of every budget.
"""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_spike_cycle_energetics import (
    charge_per_cycle, currents_at, early_late, l2_currents, landmarks, pv_currents, slope_v_s)

PULSES = {"h01_pv_neuron_reference.py": (270., 1270.), "h01_l2_neuron_reference.py": (1020., 2020.)}


def load_run(folder, name, input_name):
    """Arrays and report of one probed run."""
    stem = folder/f"{name}-{input_name}"
    return dict(np.load(stem.with_suffix(".npz"))), json.loads(stem.with_suffix(".json").read_text())


def currents_for(driver, data, report):
    """Inward-positive currents in nA for either driver's trace format."""
    geometry = report["charge_balance_geometry"]
    return pv_currents(data, geometry) if driver.startswith("h01_pv") else l2_currents(data, geometry)


def windows(row):
    """The three integration windows of one cycle row."""
    start = row["threshold_ms"] if row["threshold_ms"] is not None else row["cycle_start_ms"]
    return {"upstroke": (start, row["peak_ms"]), "falling": (row["peak_ms"], row["minimum_ms"]),
            "cycle": (row["cycle_start_ms"], row["minimum_ms"])}


def instants(time, voltage, row):
    """Times of the landmarks at which the boundary currents are read."""
    slope = slope_v_s(time, voltage)
    window = (time >= row["cycle_start_ms"]) & (time <= row["peak_ms"])
    rise = float(time[window][int(np.argmax(slope[window]))])
    return {"at_threshold": row["threshold_ms"] if row["threshold_ms"] is not None else row["cycle_start_ms"],
            "at_max_rise": rise, "at_peak": row["peak_ms"], "at_minimum": row["minimum_ms"]}


def budgets(time, voltage, currents, rows):
    """Charge per boundary in every window, and currents at every landmark, per cycle row."""
    out = []
    for row in rows:
        entry = {"spike": row["spike"], "cycle_ms": row["cycle_ms"], "minimum_mv": row["minimum_mv"],
                 "peak_mv": row["peak_mv"], "max_rise_v_s": row["max_rise_v_s"], "threshold_mv": row["threshold_mv"]}
        for label, (start, stop) in windows(row).items():
            pseudo = [{"spike": row["spike"], "cycle_start_ms": start, "minimum_ms": stop}]
            entry[label] = charge_per_cycle(time, currents, pseudo)[0]
        for label, at_ms in instants(time, voltage, row).items():
            entry[label] = currents_at(time, currents, at_ms)
            entry[label]["voltage_mv"] = float(np.interp(at_ms, time, voltage))
        out.append(entry)
    return out


def shares(budget, window="upstroke"):
    """Each boundary's share of the total inward charge of one window."""
    values = {k: v for k, v in budget[window].items() if k not in ("spike", "residual", "capacitive")}
    inward = sum(v for v in values.values() if v > 0.) or 1.
    return {k: v/inward for k, v in values.items()}


def run_budget(driver, folder, name, input_name):
    """Budgets of the early and late cycles of one run."""
    data, report = load_run(folder, name, input_name)
    pulse = PULSES[driver]
    rows = landmarks(data["time_ms"], data["voltage_mv"], pulse)
    split = early_late(rows)
    currents = currents_for(driver, data, report)
    selected = split["early"]+[r for r in split["late"] if r not in split["early"]]
    result = {"spikes": len(rows), "budgets": budgets(data["time_ms"], data["voltage_mv"], currents, selected),
              "boundaries": sorted(currents)}
    result["closure_residual_pc"] = max(abs(b["cycle"]["residual"]) for b in result["budgets"]) if result["budgets"] else None
    return result


def render(report):
    """Markdown tables: per run, per cycle, the charge per boundary in three windows."""
    lines = [f"# Charge budgets, {report['manifest']}", ""]
    for name, inputs in report["runs"].items():
        for input_name, result in inputs.items():
            lines += [f"## {name}, {input_name}: {result['spikes']} spikes, closure residual "
                      f"{result['closure_residual_pc']:.2e} pC", ""]
            for window in ("upstroke", "falling", "cycle", "at_max_rise", "at_minimum"):
                unit = "nA, inward positive" if window.startswith("at_") else "pC, inward positive"
                lines += [f"### {window} ({unit})", "",
                          "| spike | " + " | ".join(result["boundaries"]) + " | residual |",
                          "| --- |" + " --- |"*(len(result["boundaries"])+1)]
                for b in result["budgets"]:
                    total = b[window].get("residual", sum(b[window][k] for k in result["boundaries"]))
                    lines.append(f"| {b['spike']} | " + " | ".join(f"{b[window][k]:.4g}" for k in result["boundaries"])
                                 + f" | {total:.2e} |")
                lines.append("")
    return "\n".join(lines)


def main(argv=None):
    """Write ``<output_dir>-budgets.{json,md}`` for the runs present in the manifest's folder."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", default="R")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parents[2]
    folder = root/"docs/evidence"/manifest["output_dir"]
    report = {"manifest": args.manifest.name, "stage": args.stage, "runs": {}}
    for candidate in manifest["candidates"]:
        if candidate["stage"] != args.stage:
            continue
        for input_name in manifest["inputs"]:
            if (folder/f"{candidate['name']}-{input_name}.json").exists():
                report["runs"].setdefault(candidate["name"], {})[input_name] = run_budget(
                    manifest["driver"], folder, candidate["name"], input_name)
    stem = root/"docs/evidence"/f"{manifest['output_dir']}-stage-{args.stage.lower()}-budgets"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1))
    stem.with_suffix(".md").write_text(render(report))
    print(render(report)[:3000])


if __name__ == "__main__":
    main()
