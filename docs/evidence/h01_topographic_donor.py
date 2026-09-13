"""SP15 stage 13: the base as a sub-system. Toronto HL23PYR unchanged, driven as the E recording
is driven, read with the campaign's readers through the recording's chain (stage 12), beside
B3 (system A) and the recording.

Outputs ``docs/evidence/h01-topographic/stage-13.{json,md,png}``.
"""

import json

import numpy as np

from h01_spike_cycle_energetics import landmarks
from h01_topographic_chain import apply_chain, soma_trace
from h01_topographic_onset import onset, rate_of_rise
from h01_topographic_stage0 import OUT
from h01_topographic_threshold import E_PULSE, approach_rate

RUNS = {"0.2": "h01-e-donor/s13-hl23pyr-step200", "0.25": "h01-e-donor/s13-hl23pyr-step250", "0.31": "h01-e-donor/s13-hl23pyr-step310", "0.4": "h01-e-donor/s13-hl23pyr-step400"}
B3_RUNS = {"0.2": "h01-e-currents/m0-b3-currents-sweep56", "0.25": "h01-e-currents/m0-b3-currents-sweep50", "0.31": "h01-e-cable-load/c10-area-310-n9-sweep53"}
RECORDED_SWEEPS = {"0.2": "sweep 56", "0.25": "sweep 50", "0.31": "sweep 53"}
RECORDED_RISE_BAND = (336., 351.)
RECORDED = {"rise_v_s": 348., "fall_v_s": -104., "climb_2_mv": 1.4, "climb_5_mv": 3.6, "rise_5_over_1": .86, "take_off_slope_mv_per_mv_ms": -4.3, "counts": {"0.2": 1, "0.25": 5, "0.31": 10}}
LIMITS = {"rise_fraction": .15, "climb_fraction": .5, "rise_5_over_1_min": .8, "count_310": (5, 15)}
SPAN_BAND_MV = (3.5, 5.7)                         # recorded 10-100 V/s onset span; a wider reading is a reader failure at a fast approach
RECORDED_APPROACH_RANGE = (.2, .75)                # the recording's long-square spike-1 approaches (stage 7)
RECORDED_SLOPE_PER_DECADE = -3.8


def train_rows(t, v, pulse=E_PULSE):
    """Per spike: threshold (last 10 V/s crossing), take-off (first 10 V/s crossing), rise, fall, peak, approach."""
    rows = []
    rate = rate_of_rise(v)
    prev = pulse[0]
    for r in landmarks(t, v, pulse):
        o = onset(t, v, rate, r)
        rows.append({"spike": r["spike"], "t_ms": r["threshold_ms"], "threshold_mv": r["threshold_mv"], "take_off_mv": o["low_mv"], "span_mv": o["span_mv"],
                     "rise_v_s": r["max_rise_v_s"], "fall_v_s": r["max_fall_v_s"], "peak_mv": r["peak_mv"], "approach_mv_per_ms": approach_rate(t, v, r["threshold_ms"], prev)})
        prev = r["minimum_ms"]
    return rows


def read_run(base, chain):
    tr = soma_trace(base)
    if tr is None:
        return {"status": "not run"}
    t, v = tr
    rows = train_rows(t, apply_chain(t, v, chain["tau_p_us"], chain["fc_khz"]))
    return {"status": "read" if rows else "no spike", "count": len(rows), "spikes": rows, "rest_mv": float(np.median(v[(t > 950.) & (t < 1020.)]))}


def climb(rows):
    """Threshold at spike 2 and 5 against spike 1 and the rise ratio, from the cycle-by-cycle rows."""
    if len(rows) < 5:
        return None
    return {"climb_2_mv": rows[1]["threshold_mv"]-rows[0]["threshold_mv"], "climb_5_mv": rows[4]["threshold_mv"]-rows[0]["threshold_mv"],
            "rise_5_over_1": rows[4]["rise_v_s"]/rows[0]["rise_v_s"], "thresholds_mv": [r["threshold_mv"] for r in rows[:6]]}


def take_off_relation(runs):
    """Spike-1 take-off against approach across the drives that fire: slope, mV per mV/ms."""
    pts = [(r["spikes"][0]["approach_mv_per_ms"], r["spikes"][0]["threshold_mv"]) for r in runs.values()
           if r.get("status") == "read" and np.isfinite(r["spikes"][0]["approach_mv_per_ms"]) and r["spikes"][0]["span_mv"] <= SPAN_BAND_MV[1]]
    if len(pts) < 2:
        return None
    a, v = np.array(pts).T
    overlap = a.min() <= RECORDED_APPROACH_RANGE[1] and a.max() >= RECORDED_APPROACH_RANGE[0]
    return {"n": len(pts), "approach_range": [float(a.min()), float(a.max())], "take_off_range": [float(v.min()), float(v.max())],
            "slope_mv_per_mv_ms": float(np.polyfit(a, v, 1)[0]) if np.ptp(a) > 1e-6 else None,
            "slope_mv_per_decade": float(np.polyfit(np.log10(a), v, 1)[0]) if np.ptp(a) > 1e-6 else None,
            "overlaps_recorded_range": bool(overlap), "excluded_out_of_band_spans": int(sum(1 for r in runs.values() if r.get("status") == "read" and r["spikes"][0]["span_mv"] > SPAN_BAND_MV[1]))}


def decide(result):
    checks = []
    def add(reading, prediction, value, passed, note=""):
        checks.append({"reading": reading, "prediction": prediction, "value": None if value is None else float(value), "pass": None if passed is None else bool(passed), "note": note})
    runs = result["hl23pyr"]
    fired = [d for d in RUNS if runs[d].get("status") == "read"]
    if not fired:
        add("a", "spike-1 rise through the chain within 15 percent of 348 V/s", None, None, "no spike at any drive")
        return {"checks": checks, "verdict": "no reading"}
    first = runs[fired[0]]["spikes"][0]
    matched = runs["0.31"]["spikes"][0]["rise_v_s"] if runs["0.31"].get("status") == "read" else None
    add("a", f"spike-1 rise through the chain within 15 percent of 348 V/s (lowest firing drive {fired[0]} nA)", first["rise_v_s"], abs(first["rise_v_s"]-RECORDED["rise_v_s"]) < LIMITS["rise_fraction"]*RECORDED["rise_v_s"],
        f"at the matched drive 0.31 nA {matched:.0f}; B3 through the chain {result['b3']['0.31']['spikes'][0]['rise_v_s']:.0f}" if matched else f"B3 through the chain {result['b3']['0.31']['spikes'][0]['rise_v_s']:.0f}")
    trains = [(d, climb(runs[d]["spikes"])) for d in fired if climb(runs[d]["spikes"])]
    if trains:
        d, c = trains[0]
        m = climb(runs["0.31"]["spikes"]) if runs["0.31"].get("status") == "read" else None
        add("b", f"threshold climbs at least half the recorded +1.4 (spike 2) and +3.6 mV (spike 5), {d} nA train", c["climb_5_mv"],
            c["climb_2_mv"] >= LIMITS["climb_fraction"]*RECORDED["climb_2_mv"] and c["climb_5_mv"] >= LIMITS["climb_fraction"]*RECORDED["climb_5_mv"],
            f"spike 2 {c['climb_2_mv']:+.2f}, spike 5 {c['climb_5_mv']:+.2f} mV" + (f"; at the matched drive 0.31 nA {m['climb_2_mv']:+.2f} / {m['climb_5_mv']:+.2f}" if m else "") + "; B3 -0.02 / +0.24")
        add("b", "rise of spike 5 above 0.8 of spike 1", c["rise_5_over_1"], c["rise_5_over_1"] > LIMITS["rise_5_over_1_min"], f"recorded 0.86; B3 {climb(result['b3']['0.31']['spikes'])['rise_5_over_1']:.2f}")
    else:
        add("b", "threshold climb along a train of at least five spikes", None, None, "no drive gave five spikes")
    add("c", "spike-1 fall through the chain near the recorded -104 V/s (informational)", first["fall_v_s"], None, f"B3 {result['b3']['0.31']['spikes'][0]['fall_v_s']:.0f}")
    rel = result["take_off_relation"]
    add("d", "spike-1 take-off against approach (informational): recorded -3.8 mV per decade over 0.2 to 0.75 mV/ms, B3 flat", rel["slope_mv_per_decade"] if rel else None, None,
        (f"HL23PYR {rel['slope_mv_per_decade'] if rel['slope_mv_per_decade'] is None else round(rel['slope_mv_per_decade'], 2)} mV per decade over {rel['approach_range'][0]:.2f} to {rel['approach_range'][1]:.2f} mV/ms, "
         + ("overlapping" if rel["overlaps_recorded_range"] else "DISJOINT from") + f" the recorded range; {rel['excluded_out_of_band_spans']} drive excluded for an out-of-band span (reader failure at a fast approach)") if rel else "fewer than two firing drives")
    c310 = runs["0.31"].get("count")
    if c310 is not None:
        add("e", "0.31 nA count within 5 to 15 (rise not bought with the count)", c310, LIMITS["count_310"][0] <= c310 <= LIMITS["count_310"][1], f"recorded 10, B3 10; counts at 0.2/0.25/0.31/0.4: " + "/".join(str(runs[d].get("count", "-")) for d in RUNS))
    scored = [c for c in checks if c["pass"] is not None]
    verdict = "PASS" if all(c["pass"] for c in scored) else "FAIL"
    return {"checks": checks, "verdict": verdict}


def markdown(result):
    lines = ["# SP15 stage 13: the base as a sub-system (HL23PYR beside B3 and the recording)", "",
             "Every model quantity is read through the recording's chain (stage 12: pipette pole "
             f"{result['chain']['tau_p_us']:.1f} us, corner {result['chain']['fc_khz']:.0f} kHz).", "",
             "| drive nA | system | count | rest mV | spike-1 rise | fall | peak | threshold | take-off | approach | span |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(label, drive, r):
        if r.get("status") != "read":
            return f"| {drive} | {label} | {r.get('count', '')} {r.get('status', '')} | | | | | | | | |"
        s = r["spikes"][0]
        return (f"| {drive} | {label} | {r['count']} | {r.get('rest_mv', float('nan')):.1f} | {s['rise_v_s']:.0f} | {s['fall_v_s']:.0f} | {s['peak_mv']:.1f} | {s['threshold_mv']:.2f} | "
                f"{s['take_off_mv']:.2f} | {s['approach_mv_per_ms']:.2f} | {s['span_mv']:.2f} |")
    for drive in RUNS:
        lines.append(row("HL23PYR", drive, result["hl23pyr"][drive]))
        if drive in result["b3"]:
            lines.append(row("B3", drive, result["b3"][drive]))
        if drive in result["recorded"]:
            lines.append(row("recording", drive, result["recorded"][drive]))
    lines += ["", "## Threshold along the train (cycle by cycle, spikes 1 to 6)", ""]
    for label, group in (("HL23PYR", result["hl23pyr"]), ("B3", result["b3"]), ("recording", result["recorded"])):
        for drive, r in group.items():
            c = climb(r["spikes"]) if r.get("status") == "read" else None
            if c:
                lines.append(f"- {label} {drive} nA: " + ", ".join(f"{x:.2f}" for x in c["thresholds_mv"]) + f" mV; spike 2 {c['climb_2_mv']:+.2f}, spike 5 {c['climb_5_mv']:+.2f}; rise 5/1 {c['rise_5_over_1']:.2f}")
    rel = result["take_off_relation"]
    lines += ["", f"HL23PYR spike-1 take-off against approach: {rel}" if rel else "", "", "## Decision", ""]
    for c in result["decision"]["checks"]:
        status = "informational" if c["pass"] is None and c["value"] is not None else ("no reading" if c["pass"] is None else ("pass" if c["pass"] else "FAIL"))
        value = "n/a" if c["value"] is None else f"{c['value']:+.3g}"
        lines.append(f"- ({c['reading']}) {status}: {c['prediction']} ({value}{'; ' + c['note'] if c['note'] else ''})")
    lines += ["", f"Verdict: {result['decision']['verdict']}", "", "## Reading", ""] + [f"- {r}" for r in result["reading"]] + [""]
    return "\n".join(lines)


def plot(result, chain, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from h01_isolation_report import nwb_sweep
    from h01_topographic_stage0 import resample
    from h01_topographic_threshold import E_NWB
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for label, base, style in (("HL23PYR 0.31 nA", RUNS["0.31"], "C3-"), ("B3 0.31 nA", B3_RUNS["0.31"], "C0-")):
        tr = soma_trace(base)
        if tr is None:
            continue
        t, v = tr; v = apply_chain(t, v, chain["tau_p_us"], chain["fc_khz"])
        m = landmarks(t, v, E_PULSE)
        if m:
            w = (t >= m[0]["threshold_ms"]-1.) & (t <= m[0]["threshold_ms"]+2.)
            ax.plot(v[w], rate_of_rise(v)[w], style, lw=1, label=label)
    th, vh = nwb_sweep(E_NWB, 53); th, vh, _ = resample(np.asarray(th, float), np.asarray(vh, float), np.zeros(len(th)))
    m = landmarks(th, vh, E_PULSE); w = (th >= m[0]["threshold_ms"]-1.) & (th <= m[0]["threshold_ms"]+2.)
    ax.plot(vh[w], rate_of_rise(vh)[w], "k-", lw=1, label="recording sweep 53")
    ax.set(title="spike 1 at 0.31 nA, phase plane, all through the chain", xlabel="mV", ylabel="V/s"); ax.legend(fontsize=8)
    ax = axes[1]
    for label, group, style in (("HL23PYR", result["hl23pyr"], "C3o-"), ("B3", result["b3"], "C0s-"), ("recording", result["recorded"], "k^-")):
        for drive, r in group.items():
            if r.get("status") == "read" and len(r["spikes"]) >= 2:
                ax.plot([s["spike"] for s in r["spikes"][:10]], [s["threshold_mv"]-r["spikes"][0]["threshold_mv"] for s in r["spikes"][:10]], style, lw=.8, ms=3, label=f"{label} {drive}")
    ax.set(title="threshold along the train, relative to spike 1", xlabel="spike", ylabel="mV"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


READING = [
    "The rise is not the fitting pipeline. Two independently fitted human L2/3 models, B3 (Allen, fitted to this very cell's sweeps) "
    "and HL23PYR (Toronto, a population fit with human-shifted NaTg on a different anatomy), both rise at 570 to 600 V/s through the "
    "recording's chain against the recorded 348, at every drive. What the two share is the sodium equation family (Colbert and Pan "
    "2002, shifted and re-dosed) and the ideal current clamp; what differs between them (densities, shifts, anatomy, initial segment) "
    "does not move the rise. Replacing the base does not close that gap.",
    "The threshold climb has a positive control. HL23PYR climbs along its train, +1.38 mV at spike 2 and +1.78 by spike 5 at 0.31 nA "
    "(+2.24 / +2.36 at 0.4), where B3 gives -0.02 / +0.24 and the recording +1.38 / +3.63, with the rise kept (0.95 of spike 1; "
    "recorded 0.86). The registered 'at least half' fired by the letter on the lowest firing drive (0.2 nA: +0.77 / +0.90) and sits at "
    "the limit at the matched drive. The shape differs: HL23PYR's climb is a step in the first interval that then holds, the "
    "recording's is that step plus an accumulation of +2.2 mV more over the next three spikes. So a human-fitted base reproduces the "
    "step and not the accumulation.",
    "HL23PYR is the wrong cell in excitability: it rests at -74 mV where this cell rests at -84 under the same bias, fires 16, 18, 20 "
    "and 24 spikes where the recording fires 1, 5 and 10, and its 0.4 nA take-off reading is a reader failure at a 2.6 mV/ms approach "
    "(excluded). Its spike-1 take-off against approach is read over 1.1 to 1.9 mV/ms, disjoint from the recorded 0.2 to 0.75, and is "
    "not compared as a value. Its repolarisation, -80 V/s, is slower than B3's -94 and further from the recorded -104.",
    "The elemental picture, read from the same plot: through the chain the three spike-1 loops coincide from the take-off to about "
    "-40 mV (the recording, B3 and HL23PYR all pass 200 V/s near -45 mV), and separate above it, where the two fits accelerate to "
    "550 to 575 V/s and the recording levels at 340. The fits' excess rise lives above -40 mV, in the phase the soma's own sodium "
    "carries, not in the phase delivered by the initiation site; the recording's repolarisation is faster and its loop rounder.",
    "Consequence for the search: the base stays B3; the second half (HL23PYR biophysics on the 541563728 anatomy) is worth one run "
    "only for the climb, not for the rise. The rise stays the steepest element and now has the same value on two bases that differ in "
    "everything but their sodium equations, which is the sub-system the next split has to swap or dose as a characteristic curve, "
    "read through the chain.",
]


def main():
    s12 = json.loads((OUT/"stage-12.json").read_text(encoding="utf-8"))
    chain = {"tau_p_us": s12["E"]["chain"]["tau_p_us"], "fc_khz": s12["E"]["chain"]["fc_khz"]}
    threshold = json.loads((OUT/"threshold-approach.json").read_text(encoding="utf-8"))
    recorded = {}
    for drive, sweep in RECORDED_SWEEPS.items():
        spikes = threshold["tables"]["E human"][sweep]["spikes"]
        recorded[drive] = {"status": "read", "count": len(spikes), "spikes": [dict(s, take_off_mv=s.get("crossing_mv", float("nan")), fall_v_s=float("nan"), peak_mv=float("nan"), span_mv=float("nan")) for s in spikes]}
    # the recording's spike-1 fall and peak from the trace itself
    from h01_isolation_report import nwb_sweep
    from h01_topographic_stage0 import resample
    from h01_topographic_threshold import E_NWB
    for drive, sweep in RECORDED_SWEEPS.items():
        th, vh = nwb_sweep(E_NWB, int(sweep.split()[1])); th, vh, _ = resample(np.asarray(th, float), np.asarray(vh, float), np.zeros(len(th)))
        rows = train_rows(th, vh)
        recorded[drive] = {"status": "read", "count": len(rows), "spikes": rows, "rest_mv": float(np.median(vh[(th > 950.) & (th < 1020.)]))}
    result = {"stage": "13", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "chain": chain, "limits": LIMITS, "recorded_reference": RECORDED,
              "hl23pyr": {d: read_run(b, chain) for d, b in RUNS.items()}, "b3": {d: read_run(b, chain) for d, b in B3_RUNS.items()}, "recorded": recorded, "reading": READING}
    result["take_off_relation"] = take_off_relation(result["hl23pyr"])
    result["decision"] = decide(result)
    (OUT/"stage-13.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-13.md").write_text(markdown(result), encoding="utf-8")
    plot(result, chain, OUT/"stage-13.png")
    print("wrote", OUT/"stage-13.json", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
