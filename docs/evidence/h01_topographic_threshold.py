"""SP15 stage 7: the take-off against the approach, from retained traces, no simulation.

Stage 6 read the cyclical family cycle by cycle and SP16 showed that the fits' threshold does
not answer to sodium availability. Before any lever is named again, this module reads the
elemental question in Y only: is the take-off a fixed voltage, or is it set by the trajectory
that reaches it? For every spike of every retained train it tabulates the threshold under two
definitions and the approach rate in the 10 to 2 ms before it. It also records the seven
200 pA twins (what a spike leaves in the level 800 ms later) and the subthreshold staircase
(the recorded cell's own state space). Sweep 54 stays sealed.

Outputs ``docs/evidence/h01-topographic/threshold-approach.{json,md,png}``.
"""

import json

import h5py
import numpy as np

from h01_isolation_report import human_trace, nwb_sweep
from h01_spike_cycle_energetics import landmarks
from h01_topographic_stage0 import CELLS, OUT, ROOT, model_trace

E_NWB = ROOT/CELLS["E"]["human_dir"]/CELLS["E"]["repeat_file"]
I_NWB = ROOT/CELLS["I"]["human_dir"]/CELLS["I"]["repeat_file"]
E_PULSE = (1020., 2020.)
E_STAIRCASE = (48, 49, 50, 51, 52, 53, 55)      # 210 to 350 pA; sweep 54 (330 pA) is a sealed holdout
E_REPEATS = (56, 59, 60, 61, 62)                # 200 pA repeats with one spike
E_TWINS = (56, 57, 58, 59, 60, 61, 62)          # every 200 pA repeat, spike or not
E_SHORT = (27, 28, 29, 30, 31)                  # 1260 pA, 3 ms short squares
E_SHORT_WINDOW = (1020., 1100.)
E_SUBTHRESHOLD = tuple(range(32, 48))           # -110 to +190 pA long squares
APPROACH_WINDOW_MS = (10., .5)                  # at most 10 ms before the threshold, ending 0.5 ms before it
SLOPE_V_S = 10.
LEVEL_WINDOWS = {"rest": (200., 1000.), "steady": (1820., 2020.), "post_100": (2020., 2100.),
                 "post_300": (2100., 2300.), "post_1000": (3000., 4000.)}
SIGMA = {"E": {"threshold_mv": .25, "level_mv": .30}, "I": {"threshold_mv": .38, "level_mv": .17}}


def sweep_meta(path, sweep):
    """Command amplitude and holding current of one NWB sweep, in pA."""
    with h5py.File(path, "r") as nwb:
        group = nwb[f"acquisition/timeseries/Sweep_{sweep}"]
        return {"amplitude_pa": float(group["aibs_stimulus_amplitude_pa"][()]),
                "bias_pa": float(group["bias_current"][()])*1e12}


def approach_rate(time, voltage, threshold_ms, floor_ms, window=APPROACH_WINDOW_MS):
    """Least-squares slope, mV/ms, over the window before the threshold, never before ``floor_ms``.

    The floor is the previous spike's trough, or the pulse onset for spike 1, so a fast cycle
    or a short pulse measures its own trough-to-threshold (or onset-to-threshold) ramp.
    """
    start, stop = max(threshold_ms-window[0], floor_ms), threshold_ms-window[1]
    mask = (time >= start) & (time <= stop)
    if stop-start < 1. or mask.sum() < 4:
        return float("nan")
    return float(np.polyfit(time[mask], voltage[mask], 1)[0])


def first_crossing(time, voltage, row, slope=SLOPE_V_S):
    """Second definition: voltage at the first crossing of ``slope`` on the way up to the peak.

    Not applicable (nan) when the approach itself is faster than the criterion, as under the
    3 ms short squares, whose ramp crosses 10 V/s at the pulse onset.
    """
    if row.get("approach_mv_per_ms", 0.) >= slope*.5:
        return float("nan")
    rate = np.gradient(voltage, time)
    start = np.searchsorted(time, row["threshold_ms"]-8.)
    stop = np.searchsorted(time, row["peak_ms"])
    above = np.flatnonzero(rate[start:stop] >= slope)
    return float(voltage[start+above[0]]) if len(above) else float("nan")


def threshold_rows(time, voltage, pulse):
    """One row per spike: both thresholds, the approach rate, the rise and the onset time."""
    time, voltage = np.asarray(time, dtype=float), np.asarray(voltage, dtype=float)
    rows, floor = [], pulse[0]
    for r in landmarks(time, voltage, pulse):
        r["approach_mv_per_ms"] = approach_rate(time, voltage, r["threshold_ms"], floor)
        rows.append({"spike": r["spike"], "threshold_mv": r["threshold_mv"], "crossing_mv": first_crossing(time, voltage, r),
                     "approach_mv_per_ms": r["approach_mv_per_ms"], "max_rise_v_s": r["max_rise_v_s"], "t_ms": r["threshold_ms"]-pulse[0]})
        floor = r["minimum_ms"]
    return rows


def human_e():
    out = {}
    for sweep in E_STAIRCASE+E_REPEATS:
        out[f"sweep {sweep}"] = dict(sweep_meta(E_NWB, sweep), spikes=threshold_rows(*nwb_sweep(E_NWB, sweep), E_PULSE))
    for sweep in E_SHORT:
        out[f"short sweep {sweep}"] = dict(sweep_meta(E_NWB, sweep), spikes=threshold_rows(*nwb_sweep(E_NWB, sweep), E_SHORT_WINDOW))
    return out


def human_i():
    c = CELLS["I"]
    out = {label: {"amplitude_pa": spec[2]*1000., "spikes": threshold_rows(*human_trace(ROOT/c["human_dir"]/spec[0]), c["pulse"])}
           for label, spec in c["inputs"].items()}
    for sweep in c["repeat_sweeps"]:
        out[f"repeat sweep {sweep}"] = dict(sweep_meta(I_NWB, sweep), spikes=threshold_rows(*nwb_sweep(I_NWB, sweep), c["repeat_pulse"]))
    return out


def models(cell):
    c = CELLS[cell]
    return {label: {"amplitude_pa": spec[2]*1000., "spikes": threshold_rows(*model_trace(spec)[:2], c["pulse"])}
            for label, spec in c["inputs"].items()}


def relation(tables, key="threshold_mv", spike=1):
    """Take-off against approach: the slide over the approach range; ``spike=None`` takes every spike."""
    pts = [(s["approach_mv_per_ms"], s[key]) for tab in tables.values() for s in tab["spikes"] if spike in (None, s["spike"])]
    pts = [(a, v) for a, v in pts if np.isfinite(a) and np.isfinite(v)]
    if len(pts) < 2:
        return None
    a, v = np.array(pts).T
    slope = float(np.polyfit(a, v, 1)[0])
    return {"n": len(pts), "approach_range": [float(a.min()), float(a.max())], "threshold_range": [float(v.min()), float(v.max())],
            "slide_mv": float(v[np.argmax(a)]-v[np.argmin(a)]), "slope_mv_per_mv_ms": slope}


def later_spikes_residual(tables, fit):
    """Spikes 2 and later against the spike-1 relation, at their own approach rates."""
    if fit is None:
        return None
    pts = [(s["approach_mv_per_ms"], s["threshold_mv"]) for tab in tables.values() for s in tab["spikes"] if s["spike"] > 1]
    pts = [(a, v) for a, v in pts if np.isfinite(a)]
    if not pts:
        return None
    a, v = np.array(pts).T
    one = [(s["approach_mv_per_ms"], s["threshold_mv"]) for tab in tables.values() for s in tab["spikes"] if s["spike"] == 1]
    a1, v1 = np.array(one).T
    intercept = float(np.mean(v1)-fit["slope_mv_per_mv_ms"]*np.mean(a1))
    resid = v-(fit["slope_mv_per_mv_ms"]*a+intercept)
    return {"n": len(pts), "median_residual_mv": float(np.median(resid)), "min_residual_mv": float(resid.min()),
            "below_relation": int((resid < 0).sum())}


def window_median(time, voltage, window):
    mask = (time >= window[0]) & (time < window[1])
    return float(np.median(voltage[mask]))


def twins():
    """The 200 pA repeats: count and levels, spike or not."""
    out = {}
    for sweep in E_TWINS:
        t, v = nwb_sweep(E_NWB, sweep)
        row = dict(sweep_meta(E_NWB, sweep), count=len(landmarks(t, v, E_PULSE)))
        row.update({k: window_median(t, v, w) for k, w in LEVEL_WINDOWS.items()})
        out[f"sweep {sweep}"] = row
    return out


def twins_contrast(table, sigma):
    """Steady-level difference between the sweeps with a spike and without, in sigma."""
    with_spike = [r["steady"] for r in table.values() if r["count"]]
    without = [r["steady"] for r in table.values() if not r["count"]]
    diff = float(np.mean(with_spike)-np.mean(without))
    return {"with_spike_mv": with_spike, "without_mv": without, "diff_mv": diff, "sigmas": abs(diff)/sigma}


def staircase():
    """Subthreshold long squares: levels, the early extreme and the post-pulse extreme, per sweep."""
    out = {}
    for sweep in E_SUBTHRESHOLD:
        t, v = nwb_sweep(E_NWB, sweep)
        row = dict(sweep_meta(E_NWB, sweep))
        row.update({k: window_median(t, v, w) for k, w in LEVEL_WINDOWS.items()})
        early = v[(t >= 1020.) & (t < 1320.)]
        post = v[(t >= 2020.) & (t < 2320.)]
        sign = 1. if row["amplitude_pa"] > 0 else -1.
        row["early_extreme_mv"] = float(early.max() if sign > 0 else early.min())
        row["post_extreme_mv"] = float(post.min() if sign > 0 else post.max())
        row["sag_mv"] = sign*(row["early_extreme_mv"]-row["steady"])
        out[f"sweep {sweep}"] = row
    return out


def input_resistance(table):
    """Steady minus rest against command, fitted within each holding-current group; MOhm."""
    groups = {}
    for row in table.values():
        groups.setdefault(round(row["bias_pa"], 1), []).append((row["amplitude_pa"], row["steady"]-row["rest"]))
    out = {}
    for bias, pts in groups.items():
        a, dv = np.array(sorted(pts)).T
        out[f"bias {bias:+.1f} pA"] = {"n": len(pts), "amplitude_range_pa": [float(a[0]), float(a[-1])],
                                       "megaohm": float(np.polyfit(a, dv, 1)[0]*1000.)}
    return out


def _check(cell, prediction, value, sigma, passed):
    return {"cell": cell, "prediction": prediction, "value_mv": float(value), "sigmas": abs(float(value))/sigma, "pass": bool(passed)}


def decide(result):
    """The registered predictions of stage 7, each with its number and verdict, per cell."""
    rel, later, tw = result["relations"], result["later_spikes"], result["twins_contrast"]
    checks = []
    for cell in ("E", "I"):
        sig, human, fit = SIGMA[cell]["threshold_mv"], rel[f"{cell} human"], rel[f"{cell} fit"]
        checks.append(_check(cell, "human spike-1 take-off slides more than 3 sigma across its approach range",
                             human["threshold_mv"]["slide_mv"], sig, abs(human["threshold_mv"]["slide_mv"]) > 3*sig))
        checks.append(_check(cell, "the slide has the same sign under both threshold definitions (where the second applies)",
                             human["crossing_mv"]["slide_mv"], sig,
                             np.sign(human["crossing_mv"]["slide_mv"]) == np.sign(human["threshold_mv"]["slide_mv"])))
        wider = np.ptp(fit["threshold_mv"]["approach_range"]) > np.ptp(human["crossing_mv"]["approach_range"])   # long pulses only
        checks.append(_check(cell, "fit take-off moves less than 1 sigma across every spike of every train, over a wider approach range than the human's long pulses",
                             np.ptp(fit["threshold_mv"]["threshold_range"]), sig, np.ptp(fit["threshold_mv"]["threshold_range"]) < sig and wider))
        checks.append(_check(cell, "human later spikes sit above the spike-1 relation by more than 3 sigma (median)",
                             later[f"{cell} human"]["median_residual_mv"], sig, later[f"{cell} human"]["median_residual_mv"] > 3*sig))
    checks.append(_check("E", "the twins' steady levels agree within 3 sigma of the level spread", tw["diff_mv"], SIGMA["E"]["level_mv"], tw["sigmas"] < 3.))
    verdict = {cell: "PASS" if all(c["pass"] for c in checks if c["cell"] == cell) else "FAIL" for cell in ("E", "I")}
    return {"checks": checks, "verdict": verdict}


def _table_lines(label, tab):
    lines = [f"**{label}**" + (f" ({tab['amplitude_pa']:.0f} pA" + (f", bias {tab['bias_pa']:+.1f} pA)" if "bias_pa" in tab else ")") if "amplitude_pa" in tab else ""), ""]
    if not tab["spikes"]:
        return lines[:1]+[": no spike", ""]
    lines += ["| spike | approach mV/ms | threshold | 10 V/s crossing | rise V/s | t ms |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {s['spike']} | {s['approach_mv_per_ms']:.2f} | {s['threshold_mv']:.1f} | {s['crossing_mv']:.1f} | {s['max_rise_v_s']:.0f} | {s['t_ms']:.0f} |" for s in tab["spikes"][:8]]
    if len(tab["spikes"]) > 8:
        lines.append(f"| ... | | | | | ({len(tab['spikes'])-8} more) |")
    return lines+[""]


def markdown(result):
    lines = ["# SP15 stage 7: the take-off against the approach", "",
             "Retained traces only. Threshold: campaign definition (last 10 V/s crossing before the maximum rise); "
             "second column: first 10 V/s crossing on the way up. Approach: slope over at most 10 ms before the threshold, ending 0.5 ms before it, "
             "never before the previous trough or the pulse onset. Sweep 54 sealed.", ""]
    for who, tables in result["tables"].items():
        lines += [f"## {who}", ""]
        for label, tab in tables.items():
            lines += _table_lines(label, tab)
    lines += ["## Take-off against approach (humans: spike 1 of each train; fits: every spike)", ""]
    for who, rel in result["relations"].items():
        r = rel["threshold_mv"]
        lines.append(f"- {who}: n {r['n']}; approach {r['approach_range'][0]:.2f} to {r['approach_range'][1]:.2f} mV/ms; threshold "
                     f"{r['threshold_range'][0]:.1f} to {r['threshold_range'][1]:.1f} mV; slide {r['slide_mv']:+.1f} mV "
                     f"(crossing definition {rel['crossing_mv']['slide_mv']:+.1f} mV)")
    lines += ["", "## Later spikes against the spike-1 relation", ""]
    lines += [f"- {who}: n {v['n']}; median residual {v['median_residual_mv']:+.1f} mV, minimum {v['min_residual_mv']:+.1f} mV, {v['below_relation']} below the relation"
              for who, v in result["later_spikes"].items() if v]
    lines += ["", "## The 200 pA twins", "", "| sweep | spikes | rest | steady 1820-2020 | post 100 ms | post 300 ms | post 1 s |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {k.split()[1]} | {r['count']} | {r['rest']:.2f} | {r['steady']:.2f} | {r['post_100']:.2f} | {r['post_300']:.2f} | {r['post_1000']:.2f} |" for k, r in result["twins"].items()]
    tc = result["twins_contrast"]
    lines += ["", f"Steady level with a spike minus without: {tc['diff_mv']:+.2f} mV, {tc['sigmas']:.1f} sigma.", ""]
    lines += ["## Subthreshold staircase", "", "| sweep | pA | bias pA | rest | steady | sag | post extreme |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {k.split()[1]} | {r['amplitude_pa']:.0f} | {r['bias_pa']:+.1f} | {r['rest']:.2f} | {r['steady']:.2f} | {r['sag_mv']:.2f} | {r['post_extreme_mv']:.2f} |" for k, r in result["staircase"].items()]
    lines += [""]+[f"- input resistance, {k}: {v['megaohm']:.1f} MOhm over {v['amplitude_range_pa'][0]:.0f} to {v['amplitude_range_pa'][1]:.0f} pA (n {v['n']})" for k, v in result["input_resistance"].items()]
    lines += ["", "## Decision", ""]+[f"- {c['cell']} {'PASS' if c['pass'] else 'FAIL'}: {c['prediction']} ({c['value_mv']:+.2f} mV, {c['sigmas']:.1f} sigma)"
                                       for c in result["decision"]["checks"]]
    lines += ["", f"Verdict: E {result['decision']['verdict']['E']}, I {result['decision']['verdict']['I']}", "", "## Reading", ""]+result["reading"]
    return "\n".join(lines)+"\n"


def plot(result, path):
    import matplotlib, matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex="row", sharey="row")
    for ax, who in zip(axes.flat, ("E human", "E fit", "I human", "I fit")):
        for label, tab in result["tables"][who].items():
            first = [s for s in tab["spikes"] if s["spike"] == 1]
            later = [s for s in tab["spikes"] if s["spike"] > 1]
            ax.plot([s["approach_mv_per_ms"] for s in first], [s["threshold_mv"] for s in first], "o", ms=6, label=label)
            ax.plot([s["approach_mv_per_ms"] for s in later], [s["threshold_mv"] for s in later], "x", ms=4, color="0.5")
        ax.set_title(who); ax.set_xlabel("approach, mV/ms (up to 10 ms before threshold)"); ax.set_ylabel("threshold, mV")
        ax.set_xscale("log"); ax.grid(alpha=.3); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Take-off against approach: filled = spike 1 of each train, x = later spikes")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


READING = [
    "E human, spike 1: the take-off is -54.5 to -55.2 mV when the approach is 0.17 to 0.22 mV/ms (200 pA), -57.2 mV at 0.75 mV/ms (350 pA), "
    "and -61.9 mV under the 3 ms short squares that approach at 7.2 mV/ms; the slide over the long squares is 2.2 mV (9 sigma) under both "
    "threshold definitions. The E fit takes off at -57.0 to -57.2 mV at every spike of every train, from 0.23 to 1.16 mV/ms (0.8 sigma). "
    "The recorded take-off is set by the trajectory that reaches it; the fit's is a fixed voltage.",
    "E human, later spikes: all 41 sit above the spike-1 relation, by 1.9 mV in the median (8 sigma), so what a spike leaves behind raises "
    "the take-off beyond what the trajectory alone gives; the fit's later spikes sit on its fixed voltage.",
    "I human, spike 1: -57.7 to -58.5 mV at 0.7 mV/ms (120 pA), -60.5 mV at 1.1 (0.19 nA), -61.7 mV at 1.7 (0.27 nA): a 4.0 mV slide "
    "(10 sigma of the spike-1 spread) under both definitions. The I fit slides 0.7 mV over 0.44 to 1.72 mV/ms and holds within 0.3 mV "
    "along each train: not fixed to 1 sigma, but a fifth of the recorded slide. The I later spikes sit 1.0 mV above the relation in the "
    "median, inside 3 sigma, with five fast-burst spikes below it; the I along-train rows have no repeat spread of their own.",
    "Twins: the 200 pA sweeps with one spike (56, 59 to 62) and without (57, 58) have the same steady level 800 ms later (+0.16 mV, "
    "0.5 sigma) and the same recovery after the pulse, so a spike leaves nothing in the level; the Y4 level row was a steady-state "
    "difference between the cells at 200 pA, not a post-spike effect.",
    "The recorded cell's rheobase sits at 200 pA, where its own repeats give 0 or 1 spike; a count at that drive is inside the "
    "recording's spread and is not scored. Its input resistance is 75 MOhm over -110 to 70 pA and 91 MOhm over 90 to 190 pA, fitted "
    "separately because the holding current differs by 6.2 pA between the two groups; the post-pulse undershoot deepens with the drive.",
]


def main():
    tables = {"E human": human_e(), "E fit": models("E"), "I human": human_i(), "I fit": models("I")}
    relations = {who: {key: relation(tab, key, None if who.endswith("fit") else 1) for key in ("threshold_mv", "crossing_mv")}
                 for who, tab in tables.items()}
    later = {who: later_spikes_residual(tab, relations[who]["threshold_mv"]) for who, tab in tables.items()}
    tw, st = twins(), staircase()
    result = {"stage": "7", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "sealed": [54],
              "sigma": SIGMA, "tables": tables, "relations": relations, "later_spikes": later,
              "twins": tw, "twins_contrast": twins_contrast(tw, SIGMA["E"]["level_mv"]),
              "staircase": st, "input_resistance": input_resistance(st), "reading": READING}
    result["decision"] = decide(result)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"threshold-approach.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"threshold-approach.md").write_text(markdown(result), encoding="utf-8")
    plot(result, OUT/"threshold-approach.png")
    print("wrote", OUT/"threshold-approach.json", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
