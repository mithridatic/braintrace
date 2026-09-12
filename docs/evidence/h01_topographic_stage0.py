"""SP15 stage 0: Matryoshka and z-strategy characterisation from retained traces (no simulation).

Spec: docs/specs/2026-09-12-h01-topographic-strategy.md. Three products, human over model,
for the E cell (B3 on Allen 541563728) and the I cell (Kv3 close 2.0 finalist):

* the repeat envelope (temporal family) from the human's own repeated sweeps;
* the family contrasts in units of that envelope (elemental, cyclical, structural);
* phase-plane small multiples (dV/dt against V, cycles 1, 2, middle, last) and the
  z-strategy load curves I_load = I_inj - C dV/dt against V through the post-spike
  trajectory, with C estimated the same way on both traces (initial slope at pulse onset).

No lever is named. The decision names the family and the branch with the largest contrast.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_isolation_report import nwb_sweep
from h01_spike_cycle_energetics import landmarks

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]/".cache/worktree-recovery-2026-09-09/h01-braincell/.cache"
OUT = EVIDENCE/"h01-topographic"
LATE_MS = (1800., 2000.)

CELLS = {
    "E": {"pulse": (1020., 2020.), "human_dir": "human-pyramidal-l2", "repeat_file": "recording.nwb",
          "repeat_sweeps": (56, 59, 60, 61, 62), "repeat_pulse": (1020., 2020.),
          "inputs": {"200 pA": ("sweep-56.npz", "h01-e-currents/m0-b3-currents-sweep56", .2),
                     "250 pA": ("sweep-50.npz", "h01-e-currents/m0-b3-currents-sweep50", .25),
                     "310 pA": ("sweep-53.npz", "h01-e-currents/m0-b3-currents-sweep53", .31)},
          "subthreshold": ("sweep-43.npz", None, .11), "level_window": LATE_MS},
    "I": {"pulse": (270., 1270.), "human_dir": "human-pv", "repeat_file": "528687507_ephys.nwb",
          "repeat_sweeps": (40, 41, 42, 43), "repeat_pulse": (1020., 2020.),
          "inputs": {"0.19 nA": ("active-0.npz", "h01-i-sk/s0-finalist-019", .19),
                     "0.27 nA": ("active-2.npz", "h01-i-sk/s0-finalist-027", .27)},
          "subthreshold": None, "level_window": (1050., 1250.)},
}


def slope_mv_ms(time, voltage):
    return np.gradient(np.asarray(voltage, float), np.asarray(time, float))


def capacitance_from_onset(time, voltage, step_na, onset_ms, window_ms=1.):
    """Effective capacitance (nF) from the initial slope after a current step: C = I / (dV/dt)."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    mask = (time >= onset_ms) & (time <= onset_ms+window_ms)
    if mask.sum() < 3:
        raise ValueError("Too few samples in the onset window.")
    slope = np.polyfit(time[mask], voltage[mask], 1)[0]
    if slope <= 0:
        raise ValueError("The onset slope must be positive for a depolarising step.")
    return step_na/slope


def smooth(time, voltage, width_ms=.5):
    """Boxcar of ``width_ms`` (odd sample count) before differentiating a sampled trace."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    step = float(np.median(np.diff(time)))
    n = max(1, int(round(width_ms/step)))
    n += 1-n % 2
    if n == 1:
        return voltage
    kernel = np.ones(n)/n
    padded = np.pad(voltage, n//2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_current(time, voltage, injected_na, capacitance_nf, width_ms=.5):
    """I_load = I_inj - C dV/dt (nA, outward positive): the current absorbed by channels and cable."""
    return np.asarray(injected_na, float)-capacitance_nf*slope_mv_ms(time, smooth(time, voltage, width_ms))


def binned_median(x, y, edges):
    """Median of ``y`` in bins of ``x`` (None where a bin has fewer than 5 samples)."""
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        centres.append(.5*(lo+hi))
        medians.append(float(np.median(y[k])) if k.sum() >= 5 else np.nan)
    return np.array(centres), np.array(medians)


def human_trace(cell, spec):
    data = np.load(ROOT/CELLS[cell]["human_dir"]/spec[0])
    if "corrected_voltage_mv" in data:
        t, v = data["time_ms"], data["corrected_voltage_mv"]
        i = data["total_current_na"] if "total_current_na" in data else data["command_current_na"]
    else:
        t, v = data["time"], data["voltage"]
        p = CELLS[cell]["pulse"]
        i = np.where((t >= p[0]) & (t < p[1]), spec[2], 0.)
    return np.asarray(t, float), np.asarray(v, float), np.asarray(i, float)


def model_trace(spec):
    data = np.load(EVIDENCE/(spec[1]+".npz"))
    i = data["applied_current_na"] if "applied_current_na" in data else data["current_na"]
    return resample(np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float), np.asarray(i, float))


def resample(time, voltage, current, step_ms=.02):
    """Uniform grid for CVode traces: every sample then carries equal time, so medians are time-weighted."""
    if time.size > 1 and np.allclose(np.diff(time), step_ms, atol=1e-6):
        return time, voltage, current
    grid = np.arange(time[0], time[-1]+step_ms/2, step_ms)
    return grid, np.interp(grid, time, voltage), np.interp(grid, time, current)


def window_p50(time, voltage, window):
    m = (time >= window[0]) & (time < window[1])
    return float(np.median(voltage[m])) if m.any() else None


def repeat_envelope(cell):
    """Per-repeat first spike, threshold, AHP, count and post-pulse-window level, with their spread."""
    c = CELLS[cell]
    rows = []
    for sweep in c["repeat_sweeps"]:
        t, v = nwb_sweep(ROOT/c["human_dir"]/c["repeat_file"], sweep)
        marks = landmarks(t, v, c["repeat_pulse"])
        rows.append({"sweep": sweep, "count": len(marks),
                     "first_spike_ms": marks[0]["threshold_ms"]-c["repeat_pulse"][0] if marks else None,
                     "threshold_mv": marks[0]["threshold_mv"] if marks else None,
                     "ahp_mv": marks[0]["minimum_mv"] if marks else None,
                     "max_rise_v_s": marks[0]["max_rise_v_s"] if marks else None,
                     "level_p50_mv": window_p50(t, v, (c["repeat_pulse"][1]-220., c["repeat_pulse"][1]-20.))})
    sigma = {}
    for key in ("count", "first_spike_ms", "threshold_mv", "ahp_mv", "max_rise_v_s", "level_p50_mv"):
        vals = [r[key] for r in rows if r[key] is not None]
        sigma[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return {"rows": rows, "sigma": sigma}


def cycle_rows(time, voltage, pulse):
    return landmarks(time, voltage, pulse)


def _ratio(diff, sigma):
    return None if sigma in (None, 0.) or diff is None else float(diff/sigma)


def family_contrasts(cell, envelope, human, model):
    """Elemental, cyclical and temporal contrasts of one cell in units of the repeat sigma."""
    c, s = CELLS[cell], envelope["sigma"]
    first_label = next(iter(c["inputs"]))
    ht, hv, _ = human[first_label]
    mt, mv, _ = model[first_label]
    hm, mm = cycle_rows(ht, hv, c["pulse"]), cycle_rows(mt, mv, c["pulse"])
    level_h, level_m = window_p50(ht, hv, c["level_window"]), window_p50(mt, mv, c["level_window"])
    rows = [
        {"family": "elemental", "row": f"post-spike level p50 {first_label}", "human": level_h, "model": level_m,
         "diff": level_m-level_h, "sigma_key": "level_p50_mv", "ratio": _ratio(level_m-level_h, s["level_p50_mv"])},
        {"family": "elemental", "row": f"count {first_label}", "human": len(hm), "model": len(mm),
         "diff": len(mm)-len(hm), "sigma_key": "count", "ratio": "sigma 0 (identical in every repeat)" if s["count"] == 0. else _ratio(len(mm)-len(hm), s["count"])},
        {"family": "elemental", "row": f"first spike ms {first_label}", "human": hm[0]["threshold_ms"]-c["pulse"][0] if hm else None,
         "model": mm[0]["threshold_ms"]-c["pulse"][0] if mm else None,
         "diff": (mm[0]["threshold_ms"]-hm[0]["threshold_ms"]) if hm and mm else None, "sigma_key": "first_spike_ms",
         "ratio": _ratio((mm[0]["threshold_ms"]-hm[0]["threshold_ms"]) if hm and mm else None, s["first_spike_ms"])},
        {"family": "elemental", "row": f"max rise of spike 1 (V/s) {first_label}", "human": hm[0]["max_rise_v_s"] if hm else None,
         "model": mm[0]["max_rise_v_s"] if mm else None,
         "diff": (mm[0]["max_rise_v_s"]-hm[0]["max_rise_v_s"]) if hm and mm else None, "sigma_key": "max_rise_v_s",
         "ratio": _ratio((mm[0]["max_rise_v_s"]-hm[0]["max_rise_v_s"]) if hm and mm else None, s["max_rise_v_s"])},
    ]
    last_label = list(c["inputs"])[-1]
    ht, hv, _ = human[last_label]
    mt, mv, _ = model[last_label]
    hm, mm = cycle_rows(ht, hv, c["pulse"]), cycle_rows(mt, mv, c["pulse"])
    climb_h = hm[-1]["threshold_mv"]-hm[0]["threshold_mv"] if len(hm) > 1 else None
    climb_m = mm[-1]["threshold_mv"]-mm[0]["threshold_mv"] if len(mm) > 1 else None
    rows.append({"family": "cyclical", "row": f"threshold climb first to last {last_label}", "human": climb_h, "model": climb_m,
                 "diff": None if None in (climb_h, climb_m) else climb_m-climb_h, "sigma_key": "threshold_mv",
                 "ratio": _ratio(None if None in (climb_h, climb_m) else climb_m-climb_h, s["threshold_mv"])})
    if len(hm) > 1 and len(mm) > 1:
        rows.append({"family": "cyclical", "row": f"cycle 2 ms {last_label}", "human": hm[1]["cycle_ms"], "model": mm[1]["cycle_ms"],
                     "diff": mm[1]["cycle_ms"]-hm[1]["cycle_ms"], "sigma_key": "cycle repeat limit 12.8 ms",
                     "ratio": float((mm[1]["cycle_ms"]-hm[1]["cycle_ms"])/12.8)})
    rows.append({"family": "temporal", "row": "repeat sigma (first spike ms / threshold mV / AHP mV / level mV)",
                 "human": [s["first_spike_ms"], s["threshold_mv"], s["ahp_mv"], s["level_p50_mv"]], "model": None,
                 "diff": None, "sigma_key": None, "ratio": 1.})
    return rows


STRUCTURAL = [
    {"row": "rheobase (human above model)", "E": "+ (1 vs 4 spikes at 200 pA)", "I": "+ (late-rate zero 169 vs 144 pA)"},
    {"row": "late gain (human steeper)", "E": "+ (1/5/10/13 vs 4/8/10/12)", "I": "+ (0.39 vs 0.26 Hz/pA)"},
    {"row": "threshold climbs along the train (model fixed)", "E": "+ (-56.4 to -52.8 mV at 310 pA)", "I": "+ (usable-tier cycle tables)"},
    {"row": "post-spike slowing shrinks with input (model brake grows)", "E": "+ (one spike then hold at 200 pA; 120 ms cycles at 310 pA)", "I": "+ (late cycle 100 ms at 0.19, 25 ms at 0.27 nA)"},
]


def phase_windows(rows, pulse):
    """Cycle windows (threshold-1 ms to minimum+1 ms) for cycles 1, 2, middle and last."""
    picks = sorted({0, 1 if len(rows) > 1 else 0, len(rows)//2, len(rows)-1})
    return [(f"cycle {k+1}", rows[k]["threshold_ms"]-1., rows[k]["minimum_ms"]+1.) for k in picks if rows]


def load_windows(rows, pulse):
    """Post-spike trajectories: after cycle 1 (to the next threshold or pulse end) and after the last spike."""
    out, end = [], pulse[1]-2.
    if rows:
        end1 = rows[1]["threshold_ms"] if len(rows) > 1 else end
        out.append(("after spike 1", rows[0]["minimum_ms"], end1))
        if len(rows) > 1:
            out.append(("after last spike", rows[-1]["minimum_ms"], end))
    return out


def characterise(cell):
    c = CELLS[cell]
    human = {k: human_trace(cell, v) for k, v in c["inputs"].items()}
    model = {k: model_trace(v) for k, v in c["inputs"].items()}
    envelope = repeat_envelope(cell)
    first = next(iter(c["inputs"]))
    cap = {"human_nf": capacitance_from_onset(*human[first][:2], c["inputs"][first][2], c["pulse"][0]),
           "model_nf": capacitance_from_onset(*model[first][:2], c["inputs"][first][2], c["pulse"][0]),
           "method": "initial slope over 1 ms after pulse onset at the first input; effective near-soma capacitance, same estimator on both traces"}
    return {"cell": cell, "envelope": envelope, "contrasts": family_contrasts(cell, envelope, human, model),
            "capacitance": cap, "_human": human, "_model": model}


def plot_cell(result, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cell, c = result["cell"], CELLS[result["cell"]]
    labels = list(c["inputs"])
    fig, axes = plt.subplots(4, len(labels), figsize=(4.2*len(labels), 11), squeeze=False)
    for j, label in enumerate(labels):
        ht, hv, _ = result["_human"][label]
        mt, mv, _ = result["_model"][label]
        hw, mw = phase_windows(cycle_rows(ht, hv, c["pulse"]), c["pulse"]), phase_windows(cycle_rows(mt, mv, c["pulse"]), c["pulse"])
        for i in range(4):
            ax = axes[i][j]
            for (name, a, b), t, v, col, lab in ((hw[min(i, len(hw)-1)], ht, hv, "#b35f48", "human"), (mw[min(i, len(mw)-1)], mt, mv, "#4f76a3", "model")) if hw and mw else ():
                m = (t >= a) & (t <= b)
                ax.plot(v[m], slope_mv_ms(t, v)[m], color=col, lw=.8, label=f"{lab} {name}")
            ax.set_title(f"{cell} {label}: {['cycle 1', 'cycle 2', 'middle', 'last'][i]}", fontsize=9)
            ax.set_xlabel("V (mV)"); ax.set_ylabel("dV/dt (mV/ms)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir/f"phase-planes-{cell.lower()}.png", dpi=110); plt.close(fig)
    fig, axes = plt.subplots(2, len(labels), figsize=(4.2*len(labels), 7), squeeze=False)
    ch, cmod = result["capacitance"]["human_nf"], result["capacitance"]["model_nf"]
    for j, label in enumerate(labels):
        ht, hv, hi = result["_human"][label]
        mt, mv, mi = result["_model"][label]
        hw, mw = load_windows(cycle_rows(ht, hv, c["pulse"]), c["pulse"]), load_windows(cycle_rows(mt, mv, c["pulse"]), c["pulse"])
        for i in range(2):
            ax = axes[i][j]
            for wins, t, v, cur, cap, col, lab in ((hw, ht, hv, hi, ch, "#b35f48", "human"), (mw, mt, mv, mi, cmod, "#4f76a3", "model")):
                if i < len(wins):
                    name, a, b = wins[i]
                    m = (t >= a) & (t <= b)
                    load = load_current(t, v, cur, cap)[m]
                    ax.plot(v[m], load, color=col, lw=.3, alpha=.25)
                    x, y = binned_median(v[m], load, np.arange(-90., -40., 1.))
                    ax.plot(x, y, color=col, lw=1.8, label=f"{lab} {name} (1 mV medians)")
            ax.axhline(0, color="k", lw=.4); ax.set_xlim(-90, -40); ax.set_ylim(-1., 1.)
            ax.set_title(f"{cell} {label}: load curve", fontsize=9)
            ax.set_xlabel("V (mV)"); ax.set_ylabel("I_inj - C dV/dt (nA, outward +)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir/f"load-curves-{cell.lower()}.png", dpi=110); plt.close(fig)


def load_curve_table(result, windows_mv=((-75., -70.), (-70., -65.), (-65., -60.))):
    """Median I_load in voltage bins through the post-spike trajectory after spike 1 at the first input."""
    c = CELLS[result["cell"]]
    label = next(iter(c["inputs"]))
    rows = []
    for who, (t, v, cur), cap in (("human", result["_human"][label], result["capacitance"]["human_nf"]),
                                  ("model", result["_model"][label], result["capacitance"]["model_nf"])):
        wins = load_windows(cycle_rows(t, v, c["pulse"]), c["pulse"])
        if not wins:
            continue
        name, a, b = wins[0]
        m = (t >= a) & (t <= b)
        load = load_current(t, v, cur, cap)
        for lo, hi in windows_mv:
            k = m & (v >= lo) & (v < hi)
            rows.append({"who": who, "window": name, "input": label, "v_bin_mv": [lo, hi],
                         "i_load_p50_na": float(np.median(load[k])) if k.sum() > 5 else None, "samples": int(k.sum())})
    return rows


def decide(results):
    named = []
    for r in results:
        big = max((x for x in r["contrasts"] if isinstance(x["ratio"], float) and x["family"] != "temporal"),
                  key=lambda x: abs(x["ratio"]), default=None)
        named.append({"cell": r["cell"], "largest_contrast": big})
    return {"stage": "0", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md",
            "decision": "family and branch named; no lever named",
            "largest_contrasts": named, "structural": STRUCTURAL,
            "cells": [{k: v for k, v in r.items() if not k.startswith("_")} | {"load_curve_bins": load_curve_table(r)} for r in results]}


def render(decision):
    lines = ["# SP15 stage 0: Matryoshka and load-curve characterisation (retained traces)", ""]
    for cell in decision["cells"]:
        lines += [f"## {cell['cell']} cell", "", "Repeat envelope (temporal family):", "",
                  "| sweep | count | first spike (ms) | threshold (mV) | AHP (mV) | max rise (V/s) | late level p50 (mV) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in cell["envelope"]["rows"]:
            lines.append(f"| {r['sweep']} | {r['count']} | {_f(r['first_spike_ms'])} | {_f(r['threshold_mv'])} | {_f(r['ahp_mv'])} | {_f(r['max_rise_v_s'])} | {_f(r['level_p50_mv'])} |")
        s = cell["envelope"]["sigma"]
        lines += [f"| sigma | {_f(s['count'])} | {_f(s['first_spike_ms'])} | {_f(s['threshold_mv'])} | {_f(s['ahp_mv'])} | {_f(s['max_rise_v_s'])} | {_f(s['level_p50_mv'])} |", "",
                  "Family contrasts (model minus human, in units of the repeat sigma):", "",
                  "| family | row | human | model | diff | ratio to sigma |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for x in cell["contrasts"]:
            lines.append(f"| {x['family']} | {x['row']} | {_f(x['human'])} | {_f(x['model'])} | {_f(x['diff'])} | {_f(x['ratio'])} |")
        cap = cell["capacitance"]
        lines += ["", f"Effective capacitance from the onset slope: human {cap['human_nf']*1000:.1f} pF, model {cap['model_nf']*1000:.1f} pF ({cap['method']}).", "",
                  "Load curve after spike 1 (median I_inj - C dV/dt by voltage bin, nA outward positive):", "",
                  "| who | window | V bin (mV) | I_load p50 (nA) | samples |", "| --- | --- | --- | ---: | ---: |"]
        for b in cell["load_curve_bins"]:
            lines.append(f"| {b['who']} | {b['window']} | {b['v_bin_mv']} | {_f(b['i_load_p50_na'])} | {b['samples']} |")
        lines.append("")
    lines += ["## Structural family (both cells against both fits)", "", "| pattern | E | I |", "| --- | --- | --- |"]
    for r in decision["structural"]:
        lines.append(f"| {r['row']} | {r['E']} | {r['I']} |")
    lines += ["", "## Largest contrasts", ""]
    for n in decision["largest_contrasts"]:
        b = n["largest_contrast"]
        lines.append(f"- {n['cell']}: {b['family']} / {b['row']}: diff {_f(b['diff'])}, {_f(b['ratio'])} sigma." if b else f"- {n['cell']}: none")
    lines += ["", f"Decision: {decision['decision']}.", ""]
    return "\n".join(lines)


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) < 100 else f"{x:.1f}"
    if isinstance(x, list):
        return ", ".join(_f(y) for y in x)
    return str(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", default=["E", "I"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = [characterise(c) for c in args.cells]
    for r in results:
        plot_cell(r, args.out)
    decision = decide(results)
    (args.out/"stage-0-decision.json").write_text(json.dumps(decision, indent=2, default=float)+"\n", encoding="utf-8")
    (args.out/"stage-0-decision.md").write_text(render(decision)+"\n", encoding="utf-8")
    print(render(decision))


if __name__ == "__main__":
    main()
