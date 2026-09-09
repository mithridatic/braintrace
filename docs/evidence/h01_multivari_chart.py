"""Multivari charts in Hartshorne's form for the H01 E and I cells.

One measure per panel on the vertical axis, expressed as deviation from a nominal (the
human median of that measure for the cell). The horizontal axis groups observations into
nested families: cycle within sweep, sweep within input, input within machine (human or
model). Points of one sweep are joined; vertical rules separate inputs, heavier rules
separate machines; the family labels are stacked under the axis the way the book draws
them (Figure 104 and Figure 136). Each panel's title names the family that varies most,
computed as the range of family means, so the chart states the Matryoshka answer.

Input: ``h01-multivari-data.json`` from ``h01_multivari_data.py``. No trace is read.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
MEASURES = [("width_ms", "width (ms)"), ("ahp_mv", "AHP (mV)"), ("cycle_ms", "cycle (ms)"),
            ("rise_v_per_s", "max dV/dt (V/s)")]
COLOURS = {"human": "k", "model": "tab:red"}


def input_key(label):
    """Sort drives numerically; the human repeat group goes last."""
    if label == "repeat":
        return (1, 0.)
    value = float(label.split()[0])
    return (0, value*1e3 if "nA" in label else value)


def family_ranges(rows, measure):
    """Range of family means at each nesting level, taken within the parent family.

    The largest is the family that varies most: cycle within a sweep, sweep (repeat)
    within an input, input within a machine, and machine over everything.
    """
    rows = [r for r in rows if r[measure] is not None]

    def nested_range(parent, child):
        best = 0.
        for key in {parent(r) for r in rows}:
            groups = {}
            for r in rows:
                if parent(r) == key:
                    groups.setdefault(child(r), []).append(r[measure])
            if len(groups) >= 2:
                best = max(best, float(np.ptp([np.mean(v) for v in groups.values()])))
        return best

    sweep = lambda r: (r["machine"], r["input"], r["repeat"])  # noqa: E731
    within = [float(np.ptp([r[measure] for r in rows if sweep(r) == key])) for key in {sweep(r) for r in rows}]
    return {"cycle (within sweep)": float(np.median(within)) if within else 0.,
            "sweep (repeat)": nested_range(lambda r: (r["machine"], r["input"]), sweep),
            "input (drive)": nested_range(lambda r: r["machine"], lambda r: (r["machine"], r["input"])),
            "machine (human vs model)": nested_range(lambda r: 0, lambda r: r["machine"])}


def columns(rows):
    """x position of every row plus the divider and label positions."""
    order = sorted({(r["machine"], r["input"], r["repeat"]) for r in rows},
                   key=lambda t: (t[0] != "human", input_key(t[1]), str(t[2])))
    x, cursor, sweep_x, input_x, machine_x, dividers = {}, 0, {}, {}, {}, []
    last_input, last_machine = None, None
    for machine, label, sweep in order:
        if (machine, label) != last_input and last_input is not None:
            dividers.append((cursor-.5, 2. if machine != last_machine else 1.))
        cycles = sorted(r["cycle"] for r in rows if (r["machine"], r["input"], r["repeat"]) == (machine, label, sweep))
        for c in cycles:
            x[(machine, label, sweep, c)] = cursor
            cursor += 1
        sweep_x[(machine, label, sweep)] = cursor-len(cycles)/2-.5
        input_x.setdefault((machine, label), []).append(sweep_x[(machine, label, sweep)])
        machine_x.setdefault(machine, []).append(sweep_x[(machine, label, sweep)])
        last_input, last_machine = (machine, label), machine
        cursor += 1
    return x, dividers, sweep_x, input_x, machine_x, cursor


def draw_cell(cell, rows, out):
    rows = [r for r in rows if r["cell"] == cell]
    x, dividers, sweep_x, input_x, machine_x, width = columns(rows)
    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(max(12, width*.16), 11), sharex=True)
    answers = {}
    for axis, (measure, label) in zip(axes, MEASURES):
        nominal = float(np.median([r[measure] for r in rows if r["machine"] == "human" and r[measure] is not None]))
        for key in sweep_x:
            pts = sorted((x[(*key, r["cycle"])], r[measure]-nominal) for r in rows
                         if (r["machine"], r["input"], r["repeat"]) == key and r[measure] is not None)
            if pts:
                axis.plot(*zip(*pts), marker="o", ms=2.5, lw=.8, color=COLOURS[key[0]])
        for xpos, weight in dividers:
            axis.axvline(xpos, color="0.6" if weight == 1. else "k", lw=weight*.6)
        axis.axhline(0, color="0.8", lw=.6)
        ranges = family_ranges(rows, measure)
        top = max(ranges, key=ranges.get)
        answers[measure] = {"nominal": nominal, "ranges": ranges, "varies_most": top}
        axis.set_ylabel(f"{label}\ndev. from human median {nominal:.3g}", fontsize=8)
        axis.set_title(f"{label}: varies most by {top} (range {ranges[top]:.3g})", fontsize=9, loc="left")
    axis = axes[-1]
    axis.set_xticks([])
    for (machine, label), xs in input_x.items():
        axis.text(float(np.mean(xs)), -.06, label, transform=axis.get_xaxis_transform(), ha="center", fontsize=7)
    for machine, xs in machine_x.items():
        axis.text(float(np.mean(xs)), -.14, machine, transform=axis.get_xaxis_transform(), ha="center",
                  fontsize=9, color=COLOURS[machine])
    axis.text(-.5, -.06, "Input", transform=axis.get_xaxis_transform(), ha="right", fontsize=7)
    axis.text(-.5, -.14, "Machine", transform=axis.get_xaxis_transform(), ha="right", fontsize=9)
    axis.set_xlim(-1, width)
    fig.suptitle(f"{cell} cell multivari: points = spike cycles in order within one sweep; "
                 "rules separate inputs (thin) and human/model (thick); sweeps of the same input are repeats", fontsize=10)
    fig.tight_layout(rect=(0, .03, 1, .97))
    fig.savefig(out.with_suffix(".png"), dpi=130)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return answers


def main(data=EVIDENCE/"h01-multivari-data.json", out_dir=EVIDENCE):
    payload = json.loads(Path(data).read_text())
    answers = {cell: draw_cell(cell, payload["rows"], Path(out_dir)/f"h01-multivari-{cell.lower()}")
               for cell in ("E", "I") if any(r["cell"] == cell for r in payload["rows"])}
    (Path(out_dir)/"h01-multivari-chart.json").write_text(json.dumps(
        {"data": str(Path(data).name), "missing_model_traces": payload.get("missing", []),
         "answers": answers}, indent=2)+"\n")
    return answers


if __name__ == "__main__":
    print(json.dumps(main(*[Path(a) for a in sys.argv[1:3]]), indent=1))
