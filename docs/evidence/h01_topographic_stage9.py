"""SP15 stage 9: the take-off's own variables.

Three readings, registered in the strategy spec (stage 9 execution addendum):

(a) E recording, retained traces: the later-spike take-off residual against the spike-1
    relation (stage 7) as a function of the preceding interval, at the spike's own approach.
    A step that recovers with the interval falls with it; a fixed step is flat.
(b) E fit under current ramps (``h01-e-takeoff/``): the first 10 V/s crossing at the soma,
    at ``axon[0](0.5)`` and at ``axon[1](0.5)`` against the measured approach, one ramp per run.
(c) I fit with axon probes (``h01-i-takeoff/``): the order of the 10 V/s crossings at the
    soma and four axon points, and the onset span at each.

Outputs ``docs/evidence/h01-topographic/stage-9.{json,md}``.
"""

import json
from pathlib import Path

import numpy as np

from h01_spike_cycle_energetics import landmarks
from h01_topographic_onset import onset, rate_of_rise
from h01_topographic_stage0 import EVIDENCE, OUT, resample
from h01_topographic_threshold import E_PULSE, SIGMA, approach_rate

E_RUNS = {"ramp01": ("h01-e-takeoff/s9-b3-n9-ramp01", 1.1), "ramp1": ("h01-e-takeoff/s9-b3-n9-ramp1", 11.),
          "ramp10": ("h01-e-takeoff/s9-b3-n9-ramp10", 110.)}
E_POINTS = {"soma": "voltage_mv", "axon0": "axon0_voltage_mv", "axon1": "axon_voltage_mv"}
I_RUN = "h01-i-takeoff/s9-finalist-probes-019-probes"
I_PULSE = (270., 1270.)
I_POINTS = {"soma": "voltage_mv", "axon0_0.1": "axon0_0.1_voltage_mv", "axon0_0.5": "axon0_0.5_voltage_mv",
            "axon0_0.9": "axon0_0.9_voltage_mv", "axon1_0.5": "axon1_0.5_voltage_mv"}
LIMITS = {"e_residual_mv": 3*SIGMA["E"]["threshold_mv"], "e_ramp_take_off_mv": 1., "i_lead_ms": .1, "i_span_mv": 8.}


def residual_by_interval(threshold_result):
    """(a) E recording: later-spike residual against the spike-1 relation, with the preceding interval."""
    rel = threshold_result["relations"]["E human"]["threshold_mv"]
    tables = threshold_result["tables"]["E human"]
    first = [(s["approach_mv_per_ms"], s["threshold_mv"]) for tab in tables.values() for s in tab["spikes"] if s["spike"] == 1 and np.isfinite(s["approach_mv_per_ms"])]
    a1, v1 = np.array(first).T
    intercept = float(np.mean(v1)-rel["slope_mv_per_mv_ms"]*np.mean(a1))
    rows = []
    for label, tab in tables.items():
        spikes = tab["spikes"]
        for prev, s in zip(spikes, spikes[1:]):
            if np.isfinite(s["approach_mv_per_ms"]):
                rows.append({"train": label, "spike": s["spike"], "interval_ms": s["t_ms"]-prev["t_ms"], "approach_mv_per_ms": s["approach_mv_per_ms"],
                             "residual_mv": s["threshold_mv"]-(rel["slope_mv_per_mv_ms"]*s["approach_mv_per_ms"]+intercept)})
    return rows


def interval_relation(rows):
    """Residual against log interval: the slope, and the medians in short (< 30 ms), middle and long (> 100 ms) intervals."""
    iv = np.array([r["interval_ms"] for r in rows]); res = np.array([r["residual_mv"] for r in rows])
    if iv.size < 3:
        return None
    slope = float(np.polyfit(np.log10(iv), res, 1)[0])
    bands = {"short_under_30_ms": res[iv < 30.], "middle": res[(iv >= 30.) & (iv <= 100.)], "long_over_100_ms": res[iv > 100.]}
    med = {k: {"n": int(v.size), "median_mv": float(np.median(v))} if v.size else None for k, v in bands.items()}
    present = [m["median_mv"] for m in med.values() if m]
    return {"n": int(iv.size), "interval_range_ms": [float(iv.min()), float(iv.max())], "slope_mv_per_decade": slope,
            "bands": med, "band_spread_mv": float(max(present)-min(present))}


def load_points(base, points):
    """Uniform-grid traces at every recorded point of one run; None if the run is not on disk."""
    path = EVIDENCE/(base+".npz")
    if not path.exists():
        return None
    data = np.load(path)
    t = np.asarray(data["time_ms"], float)
    out = {}
    for name, key in points.items():
        if key in data.files:
            grid, v, _ = resample(t, np.asarray(data[key], float), np.zeros(len(t)))
            out[name] = (grid, v)
    return out


def point_rows(traces, pulse, floor_ms):
    """Per spike: at each point the 10 V/s crossing (time, voltage), the span and the lead over the soma."""
    grid, soma = traces["soma"]
    rows = []
    prev_min = floor_ms
    for r in landmarks(grid, soma, pulse):
        row = {"spike": r["spike"], "approach_mv_per_ms": approach_rate(grid, soma, r["threshold_ms"], prev_min), "points": {}}
        for name, (_, v) in traces.items():
            o = onset(grid, v, rate_of_rise(v), r)
            row["points"][name] = {"take_off_mv": o["low_mv"], "span_mv": o["span_mv"], "t_ms": o["low_ms"]}
        soma_t = row["points"]["soma"]["t_ms"]
        for name in row["points"]:
            row["points"][name]["lead_ms"] = soma_t-row["points"][name]["t_ms"]
        rows.append(row)
        prev_min = r["minimum_ms"]
    return rows


def e_ramps():
    """(b) the E fit under the three ramps: spike-1 take-off at each point against the measured approach."""
    out = {}
    for label, (base, rate) in E_RUNS.items():
        traces = load_points(base, E_POINTS)
        if traces is None:
            out[label] = {"rate_pa_per_ms": rate, "status": "not run"}
            continue
        rows = point_rows(traces, E_PULSE, E_PULSE[0])
        out[label] = {"rate_pa_per_ms": rate, "status": "no spike" if not rows else "read", "count": len(rows), "spike_1": rows[0] if rows else None}
    return out


def e_ramp_contrast(ramps):
    """Range of the spike-1 take-off at each point across the ramps that spiked."""
    read = [r for r in ramps.values() if r.get("status") == "read"]
    if len(read) < 2:
        return None
    out = {"n_ramps": len(read), "approach_range_mv_per_ms": [min(r["spike_1"]["approach_mv_per_ms"] for r in read), max(r["spike_1"]["approach_mv_per_ms"] for r in read)]}
    for name in E_POINTS:
        vals = [r["spike_1"]["points"][name]["take_off_mv"] for r in read if name in r["spike_1"]["points"]]
        if vals:
            out[name] = {"take_off_mv": vals, "range_mv": float(np.ptp(vals))}
    return out


def i_probes():
    """(c) the I fit with probes: crossing order and spans at five points, every spike."""
    traces = load_points(I_RUN, I_POINTS)
    if traces is None:
        return {"status": "not run"}
    rows = point_rows(traces, I_PULSE, I_PULSE[0])
    leads = {name: [r["points"][name]["lead_ms"] for r in rows if np.isfinite(r["points"][name]["lead_ms"])] for name in I_POINTS if name != "soma"}
    spans = {name: [r["points"][name]["span_mv"] for r in rows if np.isfinite(r["points"][name]["span_mv"])] for name in I_POINTS}
    earliest_spike_1 = max(rows[0]["points"], key=lambda n: rows[0]["points"][n]["lead_ms"]) if rows else None
    return {"status": "read", "count": len(rows), "rows": rows,
            "max_lead_ms": {n: float(max(v)) for n, v in leads.items() if v}, "min_span_mv": {n: float(min(v)) for n, v in spans.items() if v},
            "earliest_point_spike_1": earliest_spike_1, "earliest_lead_spike_1_ms": rows[0]["points"][earliest_spike_1]["lead_ms"] if rows else None}


def decide(result):
    checks = []
    a = result["e_interval"]["relation"]
    if a:
        checks.append({"cell": "E", "reading": "a", "prediction": "recovering step: residual falls with interval by more than 3 sigma (0.75 mV) between short and long intervals",
                       "value": a["band_spread_mv"], "recovering": bool(a["band_spread_mv"] > LIMITS["e_residual_mv"] and a["slope_mv_per_decade"] < 0.),
                       "fixed": bool(a["band_spread_mv"] <= LIMITS["e_residual_mv"])})
    b = result["e_ramps"]["contrast"]
    if b:
        worst = max(b[n]["range_mv"] for n in E_POINTS if n in b)
        checks.append({"cell": "E", "reading": "b", "prediction": "fit take-off at every point moves less than 1 mV across the ramps", "value": worst, "pass": bool(worst < LIMITS["e_ramp_take_off_mv"])})
    else:
        checks.append({"cell": "E", "reading": "b", "prediction": "fit take-off at every point moves less than 1 mV across the ramps", "value": None,
                       "pass": None, "note": "fewer than two ramps read: " + ", ".join(f"{k} {v['status']}" for k, v in result["e_ramps"]["runs"].items())})
    c = result["i_probes"]
    if c.get("status") == "read":
        lead_ok = all(v <= LIMITS["i_lead_ms"] for v in c["max_lead_ms"].values())
        span_ok = all(v > LIMITS["i_span_mv"] for v in c["min_span_mv"].values())
        checks.append({"cell": "I", "reading": "c", "prediction": "no probe leads the soma by more than 0.1 ms and every span exceeds 8 mV", "value": max(c["max_lead_ms"].values()),
                       "pass": bool(lead_ok and span_ok), "min_span_mv": min(c["min_span_mv"].values()),
                       "rejection": ("fired by the letter: the earliest point at spike 1 is axon1_0.5" if c["earliest_point_spike_1"] == "axon1_0.5" else "not fired")
                                    + f" (lead {c['earliest_lead_spike_1_ms']:+.3f} ms); the model's axon ends at axon[1], there is no point beyond it"})
    return {"checks": checks}


def markdown(result):
    lines = ["# SP15 stage 9: the take-off's own variables", ""]
    a = result["e_interval"]
    lines += ["## (a) E recording: later-spike residual against the preceding interval (retained traces)", "",
              "Residual = take-off minus the spike-1 relation at the spike's own approach (stage 7).", "",
              "| train | spike | interval ms | approach mV/ms | residual mV |", "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['train']} | {r['spike']} | {r['interval_ms']:.0f} | {r['approach_mv_per_ms']:.2f} | {r['residual_mv']:+.2f} |" for r in a["rows"]]
    rel = a["relation"]
    if rel:
        lines += ["", f"n {rel['n']}; intervals {rel['interval_range_ms'][0]:.0f} to {rel['interval_range_ms'][1]:.0f} ms; slope {rel['slope_mv_per_decade']:+.2f} mV per decade of interval; "
                  + "; ".join(f"{k} median {v['median_mv']:+.2f} mV (n {v['n']})" for k, v in rel["bands"].items() if v) + f"; band spread {rel['band_spread_mv']:.2f} mV"]
    lines += ["", "## (b) E fit under current ramps: spike-1 take-off at three points", "",
              "| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span | axon0 lead ms | axon1 lead ms |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, r in result["e_ramps"]["runs"].items():
        if r.get("spike_1"):
            p = r["spike_1"]["points"]
            lines.append(f"| {r['rate_pa_per_ms']} | {r['status']} | {r['count']} | {r['spike_1']['approach_mv_per_ms']:.2f} | {p['soma']['take_off_mv']:.1f} | {p.get('axon0', {}).get('take_off_mv', float('nan')):.1f} | "
                         f"{p['axon1']['take_off_mv']:.1f} | {p['soma']['span_mv']:.1f} | {p.get('axon0', {}).get('lead_ms', float('nan')):+.2f} | {p['axon1']['lead_ms']:+.2f} |")
        else:
            lines.append(f"| {r['rate_pa_per_ms']} | {r['status']} | | | | | | | | |")
    b = result["e_ramps"]["contrast"]
    if b:
        lines += ["", f"Across {b['n_ramps']} ramps, approach {b['approach_range_mv_per_ms'][0]:.2f} to {b['approach_range_mv_per_ms'][1]:.2f} mV/ms: " + "; ".join(f"{n} take-off range {b[n]['range_mv']:.2f} mV" for n in E_POINTS if n in b)]
    c = result["i_probes"]
    lines += ["", "## (c) I fit with axon probes, 0.19 nA: 10 V/s crossing order and span at five points", ""]
    if c.get("status") == "read":
        lines += ["| spike | " + " | ".join(f"{n} lead ms / take-off / span" for n in I_POINTS) + " |", "| ---: | " + " | ".join("---" for _ in I_POINTS) + " |"]
        for r in c["rows"][:4]+c["rows"][-1:]:
            lines.append(f"| {r['spike']} | " + " | ".join(f"{r['points'][n]['lead_ms']:+.2f} / {r['points'][n]['take_off_mv']:.1f} / {r['points'][n]['span_mv']:.1f}" for n in I_POINTS) + " |")
        lines += ["", f"{c['count']} spikes; largest lead over the soma: " + ", ".join(f"{n} {v:+.2f} ms" for n, v in c["max_lead_ms"].items())
                  + "; smallest span: " + ", ".join(f"{n} {v:.1f} mV" for n, v in c["min_span_mv"].items())]
    else:
        lines.append("not run")
    lines += ["", "## Decision", ""]
    for ch in result["decision"]["checks"]:
        verdict = ("recovering" if ch.get("recovering") else "fixed" if ch.get("fixed") else "neither") if ch["reading"] == "a" else ("PASS" if ch.get("pass") else "no reading" if ch.get("pass") is None else "FAIL")
        extra = f"; {ch['note']}" if "note" in ch else (f"; rejection {ch['rejection']}" if "rejection" in ch else "")
        val = f"{ch['value']:+.2f}" if ch["value"] is not None else "n/a"
        lines.append(f"- {ch['cell']} ({ch['reading']}) {verdict}: {ch['prediction']} ({val}{extra})")
    lines += ["", "## Reading", ""]+result["reading"]
    return "\n".join(lines)+"\n"


READING = [
    "(a) E recording: the later spikes sit above the spike-1 relation by +2.0 mV after intervals under 30 ms, +2.2 mV at 30 to 100 ms "
    "and +1.5 mV after intervals over 100 ms (up to 734 ms), a band spread of 0.69 mV against the 0.75 mV limit: a fixed step within "
    "the limit, with a downward trend of 0.6 mV per decade of interval that the limit does not resolve. What a spike leaves in the "
    "recorded take-off does not recover inside the pulse; stage 6 read that it has recovered by the next sweep.",
    "(b) E fit under ramps, spike 1: the somatic take-off is -57.1 to -57.2 mV from 0.37 to 2.53 mV/ms (0.07 mV), as in the step "
    "pulses. The take-off at the axon points is not fixed: axon[1](0.5) falls from -54.7 to -58.8 mV (4.1 mV) and axon[0](0.5) from "
    "-55.7 to -58.1 mV (2.4 mV) as the approach quickens, the axon leading the soma by 0.30 ms at every rate. At the axon's take-off "
    "the soma sits at -58.3, -58.6 and -59.5 mV (1.2 mV range) while the axon is 3.6, 2.8 and 0.7 mV above it: under a slow approach "
    "the axon's own inward current carries it ahead of the soma before it takes off; under a fast one the electrode drives the soma "
    "and the axon lags. Prediction (b) fails at both axon points and holds at the soma.",
    "So the fit does contain a take-off that slides with the approach, of the recorded sign and size (recorded soma: -2.2 mV from "
    "0.2 to 0.75 mV/ms, -7 mV to the 7 mV/ms short square; fit axon: -4.1 mV from 0.37 to 2.53 mV/ms), at its initiation site; its "
    "soma does not show it because the soma is a load 45 um down a 1 um stub that reads the arrival of the axonal spike at one "
    "voltage. Along the retained trains the fit's axon take-off is a function of the approach only (-55.1 at the 310 pA onset, -54.3 "
    "at every later spike and at the slow 200 pA onset); the recorded +1.9 mV after a spike has no counterpart at either site.",
    "(c) I fit with probes: no probe leads the soma by more than 0.08 ms (axon[1](0.5) at spike 1; later spikes soma first) and "
    "every span is 8.9 mV or more (soma 12.8 to 14.4): the whole 60 um axon and the soma turn over together. The registered "
    "rejection names axon[1](0.5) as the earliest point at spike 1, by 0.08 ms, four samples; the model's axon ends there, so there "
    "is no point to move outward to. The finalist's spike is a whole-cell turnover because its axon is a 60 um stub with no site "
    "that fires first; the stage-8 I reading is confirmed on a second run with five points.",
    "Reading of the stage: in the E fit the recorded take-off signature lives at the initiation site and is hidden from the soma "
    "by the coupling between them; in the recording the soma shows it. The next split is an input split, registered as stage 10: "
    "hold every channel and change the coupling between the soma and the site.",
]


def main():
    threshold_result = json.loads((OUT/"threshold-approach.json").read_text(encoding="utf-8"))
    rows = residual_by_interval(threshold_result)
    ramps = e_ramps()
    result = {"stage": "9", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "limits": LIMITS,
              "e_interval": {"rows": rows, "relation": interval_relation(rows)},
              "e_ramps": {"runs": ramps, "contrast": e_ramp_contrast(ramps)},
              "i_probes": i_probes(), "reading": READING}
    result["decision"] = decide(result)
    (OUT/"stage-9.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-9.md").write_text(markdown(result), encoding="utf-8")
    print("wrote", OUT/"stage-9.json")
    for ch in result["decision"]["checks"]:
        print(ch)


if __name__ == "__main__":
    main()
