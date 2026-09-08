"""Matryoshka multivari of human against model on the phase plane (Hartshorne 2020, Ch. 3).

Rows are input levels (structural family), columns are the early and late parts of
the train (temporal family), each panel overlays three consecutive cycles (cyclical
family) of the phase-plane loop (elemental family). A third column plots the cycle
length against spike index. The family carrying the largest share of the
human-to-model contrast, in units of the combined decision limit, is named.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

from docs.evidence.h01_isolation_report import CELLS, human_trace, model_trace
from docs.evidence.h01_spike_cycle_energetics import early_late, landmarks, phase_plane

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = {"human": "black", "source": "tab:blue", "candidate": "tab:red"}
FAMILIES = ("elemental", "cyclical", "structural", "temporal")


def cycle_window(rows):
    """Start and stop times spanning a list of consecutive cycles."""
    return rows[0]["cycle_start_ms"], rows[-1]["minimum_ms"]


def traces_for(root, spec, label):
    """Human, source and candidate traces of one input; missing models are skipped."""
    paths = spec["inputs"][label]
    traces = {"human": human_trace(root/paths["human"])}
    for model in ("source", "candidate"):
        if paths[model] is not None:
            traces[model] = model_trace(root/"docs/evidence", paths[model])
    return traces


def panel(axis, traces, pulse, phase):
    """Overlay three consecutive cycles of every trace in the phase plane."""
    for name, (time, voltage) in traces.items():
        rows = early_late(landmarks(time, voltage, pulse))[phase]
        if not rows:
            continue
        v, dv = phase_plane(time, voltage, *cycle_window(rows))
        axis.plot(v, dv, color=COLOURS[name], lw=.8, label=name)
    axis.set_xlabel("V (mV)")
    axis.set_ylabel("dV/dt (V/s)")


def cycle_panel(axis, traces, pulse):
    """Cycle length against spike index for every trace."""
    for name, (time, voltage) in traces.items():
        rows = landmarks(time, voltage, pulse)
        axis.plot([r["spike"] for r in rows], [r["cycle_ms"] for r in rows], ".-", color=COLOURS[name], label=name)
    axis.set_xlabel("spike")
    axis.set_ylabel("cycle (ms)")


def figure(root, spec, labels):
    """Small multiples: inputs as rows; early, late and cycle-length columns."""
    fig, axes = plt.subplots(len(labels), 3, figsize=(12, 3.6*len(labels)), squeeze=False)
    for row, label in enumerate(labels):
        traces = traces_for(root, spec, label)
        panel(axes[row][0], traces, spec["pulse"], "early")
        panel(axes[row][1], traces, spec["pulse"], "late")
        cycle_panel(axes[row][2], traces, spec["pulse"])
        axes[row][0].set_title(f"{label}: early cycles 1-3")
        axes[row][1].set_title(f"{label}: late cycles")
        axes[row][2].set_title(f"{label}: cycle length")
    for column in (0, 1):
        limits = [a[column].get_ylim() for a in axes]+[a[column].get_xlim() for a in axes]
        for a in axes:
            a[column].set_xlim(min(l[0] for l in limits[len(axes):]), max(l[1] for l in limits[len(axes):]))
            a[column].set_ylim(min(l[0] for l in limits[:len(axes)]), max(l[1] for l in limits[:len(axes)]))
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def family_ranges(rows_by_input):
    """Contrast range per Matryoshka family, in units of the combined limit.

    Rows limited by one branch only (no human repeat for that landmark) are not
    ranked: the book's limit needs the repeatability of both systems (p204).

    Parameters
    ----------
    rows_by_input : dict
        Input label to the contrast rows of one model (``h01_isolation_report``).
    """
    per_key = {}
    for label, rows in rows_by_input.items():
        for row in rows:
            if not row["limit"] or row["phase"] == "subthreshold" or row.get("limit_source") != "human and model":
                continue
            entry = per_key.setdefault(row["key"], {})
            entry.setdefault(label, {}).setdefault(row["phase"], []).append(row["contrast"]/row["limit"])
    families = {}
    for key, inputs in per_key.items():
        early = [v for i in inputs.values() for v in i.get("early", [])]
        late = [v for i in inputs.values() for v in i.get("late", [])]
        firsts = [i["early"][0] for i in inputs.values() if i.get("early")]
        families[key] = {
            "elemental": float(np.mean([abs(f) for f in firsts])) if firsts else 0.,
            "cyclical": float(np.mean([np.ptp(i["early"]) for i in inputs.values() if len(i.get("early", [])) > 1] or [0.])),
            "structural": float(np.ptp(firsts)) if len(firsts) > 1 else 0.,
            "temporal": float(abs(np.mean(late)-np.mean(early))) if early and late else 0.}
    return families


def named_family(families):
    """The family with the largest total share across landmarks."""
    totals = {f: sum(v[f] for v in families.values()) for f in FAMILIES}
    return max(totals, key=totals.get), totals


def render(cell, model, families, named, totals):
    """Markdown table of family ranges per landmark."""
    lines = [f"# Matryoshka characterisation, {cell} cell, {model}", "",
             "Contrast ranges in units of the combined decision limit.", "",
             "| Landmark | " + " | ".join(FAMILIES) + " |", "| --- |" + " --- |"*len(FAMILIES)]
    for key, values in sorted(families.items()):
        lines.append(f"| {key} | " + " | ".join(f"{values[f]:.1f}" for f in FAMILIES) + " |")
    lines.append("| total | " + " | ".join(f"{totals[f]:.1f}" for f in FAMILIES) + " |")
    lines += ["", f"**Named family:** {named}", ""]
    return "\n".join(lines)


def main():
    """Write the multivari figures and Matryoshka tables for both cells."""
    root = Path(__file__).resolve().parents[2]
    for cell, spec in CELLS.items():
        report = json.loads((root/"docs/evidence"/f"h01-isolation-{cell.lower()}.json").read_text())
        spiking = [l for l, e in report["inputs"].items() if e["human_spikes"] > 0]
        figure(root, spec, spiking).savefig(root/"docs/evidence"/f"h01-multivari-{cell.lower()}.png", dpi=110)
        pages = []
        for model in ("source", "candidate"):
            rows = {l: report["inputs"][l]["models"][model].get("rows", []) for l in spiking}
            families = family_ranges(rows)
            named, totals = named_family(families)
            pages.append(render(cell, model, families, named, totals))
            print(cell, model, named, {k: round(v, 1) for k, v in totals.items()})
        (root/"docs/evidence"/f"h01-matryoshka2-{cell.lower()}.md").write_text("\n".join(pages))


if __name__ == "__main__":
    main()
