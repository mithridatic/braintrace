"""Usable tier of the H01 acceptance: a retained per-cycle table with a QC view.

The stored form is one row per (input, cycle). Rate and adaptation ratio are
derived views over that table, computed for the verdict only. Each verdict
carries its resolvability: a limit whose human spread exceeds half of it is
`unresolvable`, never `pass` (Hartshorne 2020 p130). The approved 1 mV
contract is reported beside the usable verdict and is not relaxed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_isolation_repeatability import DLF
from h01_isolation_report import human_trace, model_trace, nwb_sweep
from h01_pv_human_datums import spike_datums
from h01_spike_cycle_energetics import DETECTION_MV, landmarks

CYCLICAL_FACTOR = 1.47
RATE_FRACTION = .15
ADAPTATION_FRACTION = .25
WIDTH_FRACTION = .2
AHP_MV = 2.
CONTRACT = {"ahp_mv": 1., "width_ms": .05}
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent

CELLS = {
    "I": {
        "pulse_ms": (270., 1270.),
        "inputs": {
            "0.19 nA": (".cache/human-pv/active-0.npz", "h01-i-energetic/e-kv3-close2-019"),
            "0.23 nA": (".cache/human-pv/active-1.npz", "h01-i-energetic/p-close2-023"),
            "0.27 nA": (".cache/human-pv/active-2.npz", "h01-i-energetic/e-kv3-close2-027"),
        },
        "repeats": (".cache/human-pv/528687507_ephys.nwb", (40, 41, 42, 43), (1020., 2020.)),
        "model": "candidate with somatic Kv3 close factor 2.0 (energetic search finalist)",
    },
    "E": {
        "pulse_ms": (1020., 2020.),
        "inputs": {
            "250 pA": (".cache/human-pyramidal-l2/sweep-50.npz", "h01-e-energetic/r-candidate-sweep50"),
            "310 pA": (".cache/human-pyramidal-l2/sweep-53.npz", "h01-e-energetic/p-candidate-sweep53"),
        },
        "repeats": (".cache/human-pyramidal-l2/recording.nwb", (56, 59, 60, 61, 62), (1020., 2020.)),
        "model": "kv3-ninety-ca133 candidate (energetic search, no E explanation passed)",
    },
}


def cycle_table(time, voltage, pulse_ms, label):
    """Stored form: one row per complete spike cycle inside the pulse."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    widths = {round(e["peak_sample_time_ms"], 6): e["time_above_threshold_ms"]
              for e in spike_datums(time, voltage, DETECTION_MV)}
    return [{"input": label, "cycle": r["spike"], "cycle_ms": r["cycle_ms"], "ahp_mv": r["minimum_mv"],
             "width_ms": widths[round(r["peak_ms"], 6)], "peak_mv": r["peak_mv"],
             "threshold_mv": r["threshold_mv"]}
            for r in landmarks(time, voltage, pulse_ms)]


def qc_view(table, pulse_ms):
    """Derived view for the verdict: count, rate, adaptation ratio, retained cycles."""
    cycles = [r["cycle_ms"] for r in table]
    full = cycles[1:]
    rate = 1000./float(np.mean(full)) if full else 0.
    ratio = full[-1]/full[0] if len(full) >= 2 else None
    return {"count": len(table), "count_rate_hz": 1000.*len(table)/(pulse_ms[1]-pulse_ms[0]),
            "rate_hz": rate, "adaptation_ratio": ratio, "width_ms": [r["width_ms"] for r in table],
            "ahp_mv": [r["ahp_mv"] for r in table]}


def repeat_spread(values):
    """Human repeat range times the Dixon factor (1.47 floor for five or more)."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return float(np.ptp(values))*DLF.get(len(values), 1.47)


def cyclical_spread(values):
    """Range over the cycles of one trace times the cyclical factor."""
    return float(np.ptp(values))*CYCLICAL_FACTOR if len(values) >= 2 else None


def usable_limits(human_view, repeats):
    """Pre-registered limits with their spread and basis, per usable row."""
    limits = {"rate_hz": {"limit": RATE_FRACTION*human_view["rate_hz"], "spread": None,
                          "basis": "15 percent of the human rate; no human repeats at this input"},
              "adaptation_ratio": {"limit": None if human_view["adaptation_ratio"] is None else
                                   ADAPTATION_FRACTION*human_view["adaptation_ratio"], "spread": None,
                                   "basis": "25 percent of the human ratio; no human repeats at this input"}}
    for key, limit, basis in (("width_ms", None, "20 percent of this human cycle's width"),
                              ("ahp_mv", AHP_MV, "2 mV of trough depth")):
        spread = repeat_spread([r.get(key) for r in repeats])
        source = "spread from human repeats"
        if spread is None:
            spread, source = cyclical_spread(human_view[key]), "spread from the human cyclical range"
        limits[key] = {"limit": limit, "spread": spread, "basis": f"{basis}; {source}"}
        if key == "width_ms":
            limits[key]["fraction"] = WIDTH_FRACTION
    return limits


def verdict(human, model, limit, spread):
    """pass, fail, unresolvable, or unavailable; the spread must be under half the limit."""
    if human is None or model is None or limit is None:
        return "unavailable"
    if spread is not None and spread > limit/2.:
        return "unresolvable"
    return "pass" if abs(model-human) <= limit else "fail"


def contract_verdict(key, human, model):
    """The approved contract beside the usable verdict; rows it lacks say so."""
    if key not in CONTRACT or human is None or model is None:
        return "no contract row"
    return "pass" if abs(model-human) <= CONTRACT[key] else "fail"


def compare(label, human_view, model_view, limits):
    """Verdict rows for one input: the derived rows, then every matched cycle."""
    rows = []
    for key in ("rate_hz", "adaptation_ratio"):
        rows.append({"input": label, "row": key, "cycle": None, "human": human_view[key],
                     "model": model_view[key], **limits[key],
                     "verdict": verdict(human_view[key], model_view[key], limits[key]["limit"], None),
                     "contract": "no contract row"})
    for key in ("width_ms", "ahp_mv"):
        for index, (h, m) in enumerate(zip(human_view[key], model_view[key]), start=1):
            cycle_limit = dict(limits[key])
            if "fraction" in cycle_limit:
                cycle_limit["limit"] = cycle_limit["fraction"]*h
            rows.append({"input": label, "row": key, "cycle": index, "human": h, "model": m, **cycle_limit,
                         "verdict": verdict(h, m, cycle_limit["limit"], cycle_limit["spread"]),
                         "contract": contract_verdict(key, h, m)})
    return rows


def human_repeats(root, spec):
    """First-cycle width and AHP from every human repeat sweep."""
    path, sweeps, window = spec["repeats"]
    vectors = []
    for sweep in sweeps:
        time, voltage = nwb_sweep(root/path, sweep)
        table = cycle_table(time, voltage, window, f"repeat {sweep}")
        vectors.append(table[0] if table else {})
    return vectors


def score_cell(cell, spec, root=ROOT):
    """Tables, views, limits and verdict rows for one cell."""
    repeats = human_repeats(root, spec)
    tables, rows, counts = {}, [], {}
    for label, (human_path, model_name) in spec["inputs"].items():
        human = cycle_table(*human_trace(root/human_path), spec["pulse_ms"], label)
        model = cycle_table(*model_trace(EVIDENCE, model_name), spec["pulse_ms"], label)
        views = qc_view(human, spec["pulse_ms"]), qc_view(model, spec["pulse_ms"])
        tables[label] = {"human": human, "model": model}
        counts[label] = {"human": views[0]["count"], "model": views[1]["count"]}
        rows.extend(compare(label, views[0], views[1], usable_limits(views[0], repeats)))
    return {"cell": cell, "model": spec["model"], "counts": counts, "repeats": repeats,
            "rows": rows, "tables": tables}


def _fmt(value):
    return "" if value is None else (f"{value:.3g}" if isinstance(value, float) else str(value))


def render(report):
    """Markdown: verdicts with both tiers, then the retained per-cycle tables."""
    lines = [f"# {report['cell']} cell usable tier", "", f"Model: {report['model']}. Contract column: the approved",
             "1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the",
             "human spread exceeds half the usable limit.", "", "## Counts (contract: exact)", "",
             "| Input | Human | Model | Contract |", "| --- | --- | --- | --- |"]
    for label, c in report["counts"].items():
        lines.append(f"| {label} | {c['human']} | {c['model']} | {'pass' if c['human'] == c['model'] else 'FAIL'} |")
    lines += ["", "## Verdicts", "", "| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['input']} | {r['row']} | {_fmt(r['cycle'])} | {_fmt(r['human'])} | {_fmt(r['model'])} | "
              f"{_fmt(r['limit'])} | {_fmt(r['spread'])} | {r['verdict']} | {r['contract']} | {r['basis']} |"
              for r in report["rows"]]
    lines += ["", "## Retained per-cycle tables", ""]
    for label, sides in report["tables"].items():
        for side, table in sides.items():
            lines += [f"### {label}, {side}", "", "| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |",
                      "| --- | --- | --- | --- | --- | --- |"]
            lines += [f"| {r['cycle']} | {r['cycle_ms']:.2f} | {r['ahp_mv']:.2f} | {r['width_ms']:.3f} | "
                      f"{r['peak_mv']:.2f} | {r['threshold_mv']:.2f} |" for r in table]
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=tuple(CELLS), required=True)
    args = parser.parse_args(argv)
    report = score_cell(args.cell, CELLS[args.cell])
    stem = EVIDENCE/f"h01-usable-tier-{args.cell.lower()}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(stem.with_suffix(".md"))


if __name__ == "__main__":
    main()
