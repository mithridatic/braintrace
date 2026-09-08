"""Write the Stage 1 isolation reports for the I and E cells from existing records."""

import json
from pathlib import Path

import h5py
import numpy as np

from docs.evidence.h01_isolation_repeatability import (
    LANDMARKS, branch_summary, contrast_rows, first_spike_vector, repeat_limits, subthreshold_contrast, tree)
from docs.evidence.h01_spike_cycle_energetics import landmarks, subthreshold_landmarks

JUNCTION_MV = -14.
CELLS = {
    "I": {"nwb": ".cache/human-pv/528687507_ephys.nwb", "pulse": (270., 1270.),
          "human_repeats": {"short square 750 pA": ((15, 16, 17, 19), (1020., 1100.)),
                            "long square 120 pA": ((40, 41, 42, 43), (1020., 2020.))},
          "model_repeats": {"source 0.19 nA, nseg x9 and x27": ("h01-pv-neuron-space9", "h01-pv-neuron-space27"),
                            "candidate 0.27 nA, nseg x9, x27, x81": ("h01-pv-calcium-response-004-300-027",
                                                                   "h01-pv-calcium-response-004-300-027-space27",
                                                                   "h01-pv-calcium-response-004-300-027-space81")},
          "inputs": {"0.19 nA": {"human": ".cache/human-pv/active-0.npz", "source": "h01-pv-neuron-space9",
                                 "candidate": "h01-pv-calcium-response-004-300"},
                     "0.27 nA": {"human": ".cache/human-pv/active-2.npz", "source": None,
                                 "candidate": "h01-pv-calcium-response-004-300-027"},
                     "-0.11 nA": {"human": ".cache/human-pv/passive-0.npz", "source": "h01-pv-neuron-hyper011",
                                  "candidate": None},
                     "-0.05 nA": {"human": ".cache/human-pv/passive-1.npz", "source": "h01-pv-neuron-hyper005",
                                  "candidate": None}}},
    "E": {"nwb": ".cache/human-pyramidal-l2/recording.nwb", "pulse": (1020., 2020.), "sub_window_ms": 40.,
          "human_repeats": {"long square 200 pA": ((56, 59, 60, 61, 62), (1020., 2020.))},
          "model_repeats": {"candidate sweep 50, atol 1e-10, 1e-11, nseg x3": ("h01-l2-kv3-ninety-ca133",
                                                                             "h01-l2-kv3-ninety-ca133-atol11",
                                                                             "h01-l2-kv3-ninety-ca133-space3")},
          "inputs": {"sweep 50": {"human": ".cache/human-pyramidal-l2/sweep-50.npz",
                                  "source": "h01-l2-campaign-fine/s0-source-fine-sweep50",
                                  "candidate": "h01-l2-campaign-fine/s0-candidate-fine-sweep50"},
                     "sweep 43": {"human": ".cache/human-pyramidal-l2/sweep-43.npz",
                                  "source": "h01-l2-campaign-coverage-recovery/s0-source-fine-sweep43",
                                  "candidate": "h01-l2-campaign-coverage-recovery/s0-candidate-fine-sweep43"}}},
}


def nwb_sweep(path, sweep):
    """Junction-corrected voltage in mV and time in ms of one NWB sweep."""
    with h5py.File(path, "r") as nwb:
        group = nwb[f"acquisition/timeseries/Sweep_{sweep}"]
        voltage = group["data"][()].astype(np.float64)*1000.+JUNCTION_MV
        rate = float(group["starting_time"].attrs["rate"])
    return np.arange(len(voltage))*1000./rate, voltage


def human_trace(path):
    """Time and voltage of a released human calibration export."""
    data = np.load(path)
    if "corrected_voltage_mv" in data:
        return data["time_ms"], data["corrected_voltage_mv"]
    return data["time"], data["voltage"]


def model_trace(folder, name):
    """Time and voltage of a saved model trace."""
    data = np.load(folder/f"{name}.npz")
    return data["time_ms"], data["voltage_mv"]


def human_limits(root, spec):
    """Decision limits from every human repeat set, merged by taking the largest."""
    merged, sets = {}, {}
    for label, (sweeps, window) in spec["human_repeats"].items():
        vectors = []
        for sweep in sweeps:
            time, voltage = nwb_sweep(root/spec["nwb"], sweep)
            vector = first_spike_vector(landmarks(time, voltage, window), window[0]) or {}
            vectors.append({**vector, **subthreshold_landmarks(time, voltage, window, spec.get("sub_window_ms", 100.))})
        sets[label] = repeat_limits(vectors)
        for key, value in sets[label]["limits"].items():
            merged[key] = max(merged.get(key, 0.), value)
    return merged, sets


def model_limits(folder, spec):
    """Decision limits from numerical repeats, merged by taking the largest per key."""
    merged, sets = {}, {}
    for label, names in spec["model_repeats"].items():
        traces = [model_trace(folder, n) for n in names]
        rows = [landmarks(*trace, spec["pulse"]) for trace in traces]
        count = min(len(r) for r in rows)
        vectors = [{key: r[0][key] for key in LANDMARKS if key != "cycle_ms"} for r in rows]
        cycle = [{"cycle_ms": max(r[i]["cycle_ms"] for i in range(1, count))} for r in rows]
        sub = [subthreshold_landmarks(*trace, spec["pulse"], spec.get("sub_window_ms", 100.)) for trace in traces]
        sets[label] = repeat_limits([{**v, **c, **s} for v, c, s in zip(vectors, cycle, sub)])
        for key, value in sets[label]["limits"].items():
            merged[key] = max(merged.get(key, 0.), value)
    return merged, sets


def cell_report(root, cell, spec):
    """Contrast rows for every input and model, with the isolation summary."""
    folder = root/"docs/evidence"
    human_lim, human_sets = human_limits(root, spec)
    model_lim, model_sets = model_limits(folder, spec)
    report = {"cell": cell, "human_limits": human_lim, "human_repeat_sets": human_sets,
              "model_limits": model_lim, "model_repeat_sets": model_sets, "inputs": {}}
    for label, paths in spec["inputs"].items():
        human_rows = landmarks(*human_trace(root/paths["human"]), spec["pulse"])
        entry = {"human_spikes": len(human_rows), "models": {}}
        for model in ("source", "candidate"):
            if paths[model] is None:
                entry["models"][model] = {"status": "no zero-bias trace on record; Stage 3 reference run"}
                continue
            model_time, model_voltage = model_trace(folder, paths[model])
            model_rows = landmarks(model_time, model_voltage, spec["pulse"])
            rows = contrast_rows(human_rows, model_rows, human_lim, model_lim)
            window = spec.get("sub_window_ms", 100.)
            sub = subthreshold_contrast(subthreshold_landmarks(*human_trace(root/paths["human"]), spec["pulse"], window),
                                        subthreshold_landmarks(model_time, model_voltage, spec["pulse"], window),
                                        human_lim, model_lim)
            rows += [r for r in sub if r["key"] == "return_mv" or not human_rows]
            entry["models"][model] = {"trace": paths[model], "model_spikes": len(model_rows),
                                      "rows": rows, "summary": branch_summary(rows)}
        report["inputs"][label] = entry
    all_rows = [r for e in report["inputs"].values() for m in e["models"].values() for r in m.get("rows", [])]
    report["summary"] = branch_summary(all_rows)
    report["tree"] = tree(cell, report["summary"])
    return report


def render(report):
    """Markdown tables of limits and search rows."""
    lines = [f"# Isolation split, {report['cell']} cell", "", "## Decision limits", "",
             "| Landmark | Human repeat limit | Model repeat limit |", "| --- | --- | --- |"]
    for key in sorted(set(report["human_limits"]) | set(report["model_limits"])):
        lines.append(f"| {key} | {report['human_limits'].get(key, float('nan')):.3g} | "
                     f"{report['model_limits'].get(key, float('nan')):.3g} |")
    for label, entry in report["inputs"].items():
        for model, data in entry["models"].items():
            lines += ["", f"## {label}, {model}", ""]
            if "rows" not in data:
                lines.append(data["status"])
                continue
            lines += [f"Human {entry['human_spikes']} spikes, model {data['model_spikes']}.", "",
                      "| Phase | Spike | Landmark | Human | Model | Contrast | Limit | Ratio | Tier |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in data["rows"]:
                ratio = "" if r["ratio"] is None else f"{r['ratio']:.1f}"
                limit = "" if r["limit"] is None else f"{r['limit']:.3g}"
                lines.append(f"| {r['phase']} | {r['spike']} | {r['key']} | {r['human']:.4g} | {r['model']:.4g} | "
                             f"{r['contrast']:+.4g} | {limit} ({r['limit_source']}) | {ratio} | {r['tier']} |")
    lines += ["", "## Tree", "", "```mermaid", report["tree"], "```", ""]
    return "\n".join(lines)


def main():
    """Write both isolation reports."""
    root = Path(__file__).resolve().parents[2]
    for cell, spec in CELLS.items():
        report = cell_report(root, cell, spec)
        name = f"h01-isolation-{cell.lower()}"
        (root/"docs/evidence"/f"{name}.json").write_text(json.dumps(report, indent=1))
        (root/"docs/evidence"/f"{name}.md").write_text(render(report))
        print(cell, json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
