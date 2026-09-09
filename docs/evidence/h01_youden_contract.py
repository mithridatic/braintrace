"""Youden plots of the H01 usable-tier contracts: human value on x, model value on y.

Hartshorne's Youden plot puts the same quantity measured two ways on the two axes, so
bias reads as distance from the 45 degree line and repeatability as scatter along it.
Here the two ways are the human recording and the model at the same input. Each point
carries its contract limit as a vertical bar centred on the diagonal, so a point whose
marker lies outside its own bar is a failed row. Human repeat spread, where it exists,
is the horizontal bar.

Inputs are the committed usable-tier JSON files; no traces are read.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
SOURCES = {"E": EVIDENCE/"h01-e-usable"/"b3-all-inputs-usable.json",
           "I": EVIDENCE/"h01-usable-tier-i.json"}
METRICS = [("rate_hz", "rate (Hz)"), ("adaptation_ratio", "adaptation ratio"),
           ("width_ms", "spike width (ms)"), ("ahp_mv", "AHP (mV)")]
COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_rows(path):
    """Contract rows with both values present, keyed by metric."""
    data = json.loads(Path(path).read_text())
    rows = [r for r in data["rows"] if r["human"] is not None and r["model"] is not None]
    return data, {metric: [r for r in rows if r["row"] == metric] for metric, _ in METRICS}


def draw_panel(axis, rows, label):
    """One metric: points, per-point limit bars on the diagonal, human spread, 45 degree line."""
    inputs = sorted({r["input"] for r in rows})
    for colour, name in zip(COLOURS, inputs):
        pts = [r for r in rows if r["input"] == name]
        for r in pts:
            axis.plot([r["human"], r["human"]], [r["human"]-r["limit"], r["human"]+r["limit"]],
                      color="0.75", lw=1, zorder=1)
            if r.get("spread"):
                axis.plot([r["human"]-r["spread"], r["human"]+r["spread"]], [r["model"]]*2,
                          color=colour, lw=1, alpha=.5, zorder=2)
        marker = ["o" if r["verdict"] == "pass" else "x" for r in pts]
        for r, m in zip(pts, marker):
            axis.scatter(r["human"], r["model"], color=colour, marker=m, s=28, zorder=3,
                         label=name if r is pts[0] else None)
    values = [v for r in rows for v in (r["human"], r["model"])]
    lo, hi = min(values), max(values)
    pad = .08*(hi-lo or 1.)
    axis.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="k", lw=.8)
    axis.set_xlim(lo-pad, hi+pad)
    axis.set_ylim(lo-pad, hi+pad)
    axis.set_aspect("equal")
    axis.set_xlabel("human "+label)
    axis.set_ylabel("model "+label)
    passed = sum(r["verdict"] == "pass" for r in rows)
    axis.set_title(f"{label}: {passed}/{len(rows)} rows pass", fontsize=10)
    axis.legend(fontsize=7, loc="upper left")


def draw_cell(cell, path, out):
    """Four metric panels for one cell; returns per-metric pass counts."""
    data, by_metric = load_rows(path)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    counts = {}
    for axis, (metric, label) in zip(axes, METRICS):
        rows = by_metric[metric]
        counts[metric] = (sum(r["verdict"] == "pass" for r in rows), len(rows))
        if rows:
            draw_panel(axis, rows, label)
        else:
            axis.set_axis_off()
            axis.set_title(f"{label}: no rows", fontsize=10)
    fig.suptitle(f"{cell} cell Youden plot: {data['model']}. Grey bar = contract limit "
                 "on the diagonal; o pass, x fail; horizontal bar = human repeat spread", fontsize=10)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return counts


def main(out_dir=EVIDENCE):
    summary = {}
    for cell, path in SOURCES.items():
        summary[cell] = draw_cell(cell, path, Path(out_dir)/f"h01-youden-contract-{cell.lower()}")
    (Path(out_dir)/"h01-youden-contract.json").write_text(json.dumps(
        {"sources": {k: str(v.relative_to(EVIDENCE)) for k, v in SOURCES.items()},
         "pass_counts": summary}, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:2]), indent=1))
