"""SP15 stage 6: the cyclical family read cycle by cycle, from retained traces, no simulation.

Stage 0 compressed the cyclical family to one number per train (threshold first to last).
Hartshorne chapter 3 asks for the same position in consecutive cycles, and chapter 5 warns
that one compressed value loses the answer to the strategic question. This module tabulates
every spike of every retained train (threshold, maximum rise, time from pulse onset) for
both humans and both fits, and reads three shapes from the tables:

* the step from spike 1 to spike 2 and the level at spike 5, against the first spike;
* whether spike 1 is the same in every sweep (recovery between sweeps);
* whether the change is a step that then holds, or an accumulation across the train.

Outputs ``docs/evidence/h01-topographic/cycle-tables.json`` and ``.md``.
"""

import json
from pathlib import Path

import numpy as np

from h01_isolation_report import human_trace, nwb_sweep
from h01_spike_cycle_energetics import landmarks
from h01_topographic_stage0 import CELLS, OUT, ROOT, model_trace

E_STAIRCASE = (48, 49, 50, 51, 52, 53, 54, 55)          # 210 to 350 pA in 20 pA steps, one sweep each
E_REPEATS = (56, 57, 58, 59, 60, 61, 62)                # 200 pA, every repeat including the two with no spike
E_PULSE = (1020., 2020.)


def cycle_table(time, voltage, pulse):
    """Threshold, maximum rise and onset time of every spike in the pulse, plus the shape summary."""
    rows = landmarks(time, voltage, pulse)
    table = [{"spike": r["spike"], "threshold_mv": r["threshold_mv"], "max_rise_v_s": r["max_rise_v_s"],
              "t_ms": r["threshold_ms"]-pulse[0]} for r in rows]
    return {"count": len(table), "spikes": table, "shape": shape(table)}


def shape(table):
    """Step (spike 2 minus spike 1), level at spike 5 and at the last spike, all against spike 1."""
    if len(table) < 2:
        return None
    first = table[0]
    def delta(row):
        return {"threshold_mv": row["threshold_mv"]-first["threshold_mv"],
                "rise_fraction": row["max_rise_v_s"]/first["max_rise_v_s"], "t_ms": row["t_ms"]-first["t_ms"]}
    out = {"spike_2": delta(table[1]), "last": delta(table[-1])}
    if len(table) >= 5:
        out["spike_5"] = delta(table[4])
    return out


def human_e():
    path = ROOT/CELLS["E"]["human_dir"]/CELLS["E"]["repeat_file"]
    out = {}
    for sweep in E_STAIRCASE+E_REPEATS:
        t, v = nwb_sweep(path, sweep)
        out[f"sweep {sweep}"] = cycle_table(t, v, E_PULSE)
    return out


def human_i():
    c = CELLS["I"]
    out = {}
    for label, spec in c["inputs"].items():
        t, v = human_trace(ROOT/c["human_dir"]/spec[0])
        out[label] = cycle_table(t, v, c["pulse"])
    for sweep in c["repeat_sweeps"]:
        t, v = nwb_sweep(ROOT/c["human_dir"]/c["repeat_file"], sweep)
        out[f"repeat sweep {sweep}"] = cycle_table(t, v, c["repeat_pulse"])
    return out


def models(cell):
    c = CELLS[cell]
    out = {}
    for label, spec in c["inputs"].items():
        t, v, _ = model_trace(spec)
        out[label] = cycle_table(t, v, c["pulse"])
    return out


def first_spike_spread(tables):
    """Spread of spike 1 across sweeps: the recovery-between-sweeps reading."""
    th = [x["spikes"][0]["threshold_mv"] for x in tables.values() if x["count"]]
    rise = [x["spikes"][0]["max_rise_v_s"] for x in tables.values() if x["count"]]
    return {"sweeps_with_a_spike": len(th), "threshold_mv": {"min": min(th), "max": max(th), "sd": float(np.std(th, ddof=1))},
            "max_rise_v_s": {"min": min(rise), "max": max(rise), "sd": float(np.std(rise, ddof=1))}}


def markdown(result):
    lines = ["# SP15 stage 6: the cyclical family, cycle by cycle", "",
             "Retained traces only; thresholds and rises by the campaign's `landmarks()` on a uniform grid.",
             "Columns: spike number, threshold (mV), maximum rise (V/s), time from pulse onset (ms).", ""]
    for who, tables in result["tables"].items():
        lines += [f"## {who}", ""]
        for label, tab in tables.items():
            if not tab["count"]:
                lines += [f"**{label}**: no spike", ""]
                continue
            lines += [f"**{label}** ({tab['count']} spikes)", "", "| spike | threshold | rise | t |", "| ---: | ---: | ---: | ---: |"]
            lines += [f"| {s['spike']} | {s['threshold_mv']:.1f} | {s['max_rise_v_s']:.0f} | {s['t_ms']:.0f} |" for s in tab["spikes"][:14]]
            if tab["count"] > 14:
                lines.append(f"| ... | | | ({tab['count']-14} more) |")
            sh = tab["shape"]
            if sh:
                lines += ["", f"step at spike 2: {sh['spike_2']['threshold_mv']:+.1f} mV, rise x{sh['spike_2']['rise_fraction']:.2f}, "
                          f"{sh['spike_2']['t_ms']:.0f} ms after spike 1; last spike {sh['last']['threshold_mv']:+.1f} mV, rise x{sh['last']['rise_fraction']:.2f}"]
            lines.append("")
    lines += ["## Spike 1 across sweeps (recovery between sweeps)", ""]
    for who, spread in result["first_spike_spread"].items():
        lines.append(f"- {who}: {spread['sweeps_with_a_spike']} sweeps; threshold {spread['threshold_mv']['min']:.1f} to {spread['threshold_mv']['max']:.1f} mV "
                     f"(sd {spread['threshold_mv']['sd']:.2f}); rise {spread['max_rise_v_s']['min']:.0f} to {spread['max_rise_v_s']['max']:.0f} V/s (sd {spread['max_rise_v_s']['sd']:.1f})")
    lines += ["", "## Reading", ""] + result["reading"]
    return "\n".join(lines)+"\n"


READING = [
    "E human, 230 to 350 pA: the threshold steps up between spike 1 and spike 2, keeps moving through spike 5, then holds for the rest of the second at every drive; the maximum rise steps down at spike 2 by about an eighth and then drifts a further twentieth over the second. Spike 1 is the same in every sweep. At 200 pA the cell fires once and then holds below threshold for the rest of the pulse, and in two of seven repeats it does not fire.",
    "I human: at 0.27 nA the threshold steps up over the first five spikes and then holds; at 0.19 nA it climbs across the whole second as the intervals lengthen. The rise follows the threshold in both.",
    "Both fits hold threshold and rise flat at every drive.",
    "So the use-dependence enters within tens of milliseconds of a spike, does not recover inside the 800 ms hold after a single spike, and has recovered by the next sweep. A mechanism with one time constant of a second accumulates gradually and cannot make the step; SP16 is amended to separate entry from recovery, and its predictions are written per cycle.",
]


def main():
    tables = {"E human": human_e(), "E fit": models("E"), "I human": human_i(), "I fit": models("I")}
    result = {"stage": "6", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md",
              "method": "landmarks() per spike on the retained traces; E staircase sweeps 48-55 and every 200 pA repeat 56-62; I retained inputs and repeats",
              "tables": tables,
              "first_spike_spread": {"E human staircase": first_spike_spread({k: v for k, v in tables["E human"].items() if int(k.split()[1]) in E_STAIRCASE}),
                                     "E human 200 pA repeats": first_spike_spread({k: v for k, v in tables["E human"].items() if int(k.split()[1]) in E_REPEATS}),
                                     "I human repeats": first_spike_spread({k: v for k, v in tables["I human"].items() if k.startswith("repeat")})},
              "reading": READING}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"cycle-tables.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (OUT/"cycle-tables.md").write_text(markdown(result), encoding="utf-8")
    print("wrote", OUT/"cycle-tables.json")


if __name__ == "__main__":
    main()
