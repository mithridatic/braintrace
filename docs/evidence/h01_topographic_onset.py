"""SP15 stage 8: soft or hard take-off, from retained traces, no simulation.

Stage 7 read that the recorded take-off is set by the trajectory that reaches it while the
fit's is one voltage at every approach. Stage 8 asks, in Y only, whether the somatic onset
is a gradual take-off (the local membrane turning over) or a kink (a current arriving from
elsewhere), at every spike, in both recordings at the soma and in both fits at the soma and
at their retained axonal observation point (E ``axon[1](0.5)``, I ``axon[0](0.5)``).

Definitions (spec, stage 8 addendum): uniform 0.02 ms grid; rate of rise as a central
difference over 0.1 ms; onset read between the first 10 V/s and the 100 V/s crossings:
span (mV covered), rapidness (slope of dV/dt against V from 10 to 50 V/s, per ms), and for
the fits the axon lead (axonal 10 V/s crossing minus somatic, ms).

Outputs ``docs/evidence/h01-topographic/onset-shape.{json,md,png}``.
"""

import json

import numpy as np

from h01_isolation_report import human_trace, nwb_sweep
from h01_spike_cycle_energetics import landmarks
from h01_topographic_stage0 import CELLS, EVIDENCE, OUT, ROOT, resample
from h01_topographic_threshold import E_NWB, E_PULSE, E_REPEATS, E_STAIRCASE, I_NWB

RISE_LOW_V_S, RISE_MID_V_S, RISE_HIGH_V_S = 10., 50., 100.
DERIVATIVE_HALF_WIDTH = 2                      # samples: central difference over 0.1 ms on the 0.02 ms grid
KINK_SPAN_MV, GRADUAL_SPAN_MV = 1., 2.
AXON_POINT = {"E": "axon[1](0.5), 45 um out on the 60 um stub", "I": "axon[0](0.5), first axon section"}
SP16_E_TRACES = {"depth 0": "h01-e-cable-load/c10-area-310-n9-sweep53", "depth 0.3": "h01-e-sodium-slow/s1-d03-n9-sweep53",
                 "depth 0.6": "h01-e-sodium-slow/s1-d06-n9-sweep53"}      # 310 pA, nseg 9; the SP16 stage-1 series


def rate_of_rise(voltage, step_ms=.02, half=DERIVATIVE_HALF_WIDTH):
    """Central difference over 2*half samples, V/s (= mV/ms); the ends repeat the nearest value."""
    rate = np.empty_like(voltage)
    rate[half:-half] = (voltage[2*half:]-voltage[:-2*half])/(2*half*step_ms)
    rate[:half], rate[-half:] = rate[half], rate[-half-1]
    return rate


def crossing_index(rate, start, stop, level):
    """First index in [start, stop) where the rate reaches ``level``; None if it never does."""
    above = np.flatnonzero(rate[start:stop] >= level)
    return int(start+above[0]) if len(above) else None


def onset(time, voltage, rate, row):
    """Span and rapidness of one spike's onset, read between the 10 and 100 V/s crossings."""
    start, stop = np.searchsorted(time, row["threshold_ms"]-8.), np.searchsorted(time, row["peak_ms"])
    low, mid, high = (crossing_index(rate, start, stop, level) for level in (RISE_LOW_V_S, RISE_MID_V_S, RISE_HIGH_V_S))
    if low is None or high is None or mid is None or mid <= low:
        return {"span_mv": float("nan"), "rapidness_per_ms": float("nan"), "low_ms": float("nan"), "low_mv": float("nan")}
    slope = np.polyfit(voltage[low:mid+1], rate[low:mid+1], 1)[0] if mid-low >= 2 else (rate[mid]-rate[low])/max(voltage[mid]-voltage[low], 1e-9)
    return {"span_mv": float(voltage[high]-voltage[low]), "rapidness_per_ms": float(slope),
            "low_ms": float(time[low]), "low_mv": float(voltage[low])}


def onset_rows(time, voltage, pulse, axon=None):
    """One row per spike: somatic onset, and the axon point's onset and lead when given."""
    time, voltage, _ = resample(np.asarray(time, float), np.asarray(voltage, float), np.zeros(len(time)))
    rate = rate_of_rise(voltage)
    axon_rate = rate_of_rise(axon) if axon is not None else None
    rows = []
    for r in landmarks(time, voltage, pulse):
        row = {"spike": r["spike"], "threshold_mv": r["threshold_mv"], "soma": onset(time, voltage, rate, r)}
        if axon is not None:
            row["axon"] = onset(time, axon, axon_rate, r)
            row["axon_lead_ms"] = row["soma"]["low_ms"]-row["axon"]["low_ms"]
            soma_low = np.searchsorted(time, row["soma"]["low_ms"])
            row["axon_mv_at_soma_10"] = float(axon[soma_low]) if np.isfinite(row["soma"]["low_ms"]) else float("nan")
        rows.append(row)
    return rows


def human_e():
    return {f"sweep {s}": onset_rows(*nwb_sweep(E_NWB, s), E_PULSE) for s in E_STAIRCASE+E_REPEATS}


def human_i():
    c = CELLS["I"]
    out = {label: onset_rows(*human_trace(ROOT/c["human_dir"]/spec[0]), c["pulse"]) for label, spec in c["inputs"].items()}
    out.update({f"repeat sweep {s}": onset_rows(*nwb_sweep(I_NWB, s), c["repeat_pulse"]) for s in c["repeat_sweeps"]})
    return out


def model_with_axon(spec):
    data = np.load(EVIDENCE/(spec[1]+".npz"))
    t = np.asarray(data["time_ms"], float)
    grid, v, axon = resample(t, np.asarray(data["voltage_mv"], float), np.asarray(data["axon_voltage_mv"], float))
    return grid, v, axon


def models(cell):
    c = CELLS[cell]
    out = {}
    for label, spec in c["inputs"].items():
        t, v, axon = model_with_axon(spec)
        out[label] = onset_rows(t, v, c["pulse"], axon)
    return out


def sp16_ais():
    """The E fit's somatic and axonal 10 V/s crossings along the 310 pA train under the SP16 gate, from the retained stage-1 traces."""
    out = {}
    for label, base in SP16_E_TRACES.items():
        path = EVIDENCE/(base+".npz")
        if not path.exists():
            out[label] = None
            continue
        t, v, axon = model_with_axon((None, base))
        rows = onset_rows(t, v, E_PULSE, axon)
        out[label] = [{"spike": r["spike"], "soma_10_mv": r["soma"]["low_mv"], "axon_10_mv": r["axon"]["low_mv"], "axon_lead_ms": r["axon_lead_ms"]}
                      for r in rows if r["spike"] in (1, 2, 5, len(rows))]
    return out


def spike_1(tables, key):
    vals = [tab[0]["soma"][key] for tab in tables.values() if tab]
    return [v for v in vals if np.isfinite(v)]


def summary(tables, site="soma"):
    """Spike-1 and later-spike span and rapidness at one observation site."""
    first = [tab[0][site] for tab in tables.values() if tab and np.isfinite(tab[0][site]["span_mv"])]
    later = [row[site] for tab in tables.values() for row in tab[1:] if np.isfinite(row[site]["span_mv"])]
    def stats(rows, key):
        vals = np.array([r[key] for r in rows])
        return {"n": int(vals.size), "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())} if vals.size else None
    return {"spike_1": {k: stats(first, k) for k in ("span_mv", "rapidness_per_ms")},
            "later": {k: stats(later, k) for k in ("span_mv", "rapidness_per_ms")}}


def axon_lead(tables):
    leads = np.array([row["axon_lead_ms"] for tab in tables.values() for row in tab if np.isfinite(row["axon_lead_ms"])])
    at10 = np.array([row["axon_mv_at_soma_10"] for tab in tables.values() for row in tab if np.isfinite(row["axon_mv_at_soma_10"])])
    first = [tab[0] for tab in tables.values() if tab]
    return {"n": int(leads.size), "min_ms": float(leads.min()), "median_ms": float(np.median(leads)),
            "axon_first_every_spike": bool((leads > 0).all()), "axon_mv_at_soma_10_median": float(np.median(at10)),
            "spike_1": [{"lead_ms": r["axon_lead_ms"], "axon_mv_at_soma_10": r["axon_mv_at_soma_10"]} for r in first]}


def repeat_sigma(tables, sweeps):
    spans = spike_1({k: v for k, v in tables.items() if k in sweeps}, "span_mv")
    rapid = spike_1({k: v for k, v in tables.items() if k in sweeps}, "rapidness_per_ms")
    return {"n": len(spans), "span_mv": float(np.std(spans, ddof=1)), "span_range_mv": float(np.ptp(spans)),
            "rapidness_per_ms": float(np.std(rapid, ddof=1))}


def classify(span):
    return "kink" if span < KINK_SPAN_MV else "gradual" if span > GRADUAL_SPAN_MV else "unresolved"


def decide(result):
    checks = []
    for cell in ("E", "I"):
        hs, fs, sig = result["summary"][f"{cell} human"]["soma"], result["summary"][f"{cell} fit"]["soma"], result["sigma"][cell]
        fit_span_max, human_span = fs["spike_1"]["span_mv"]["max"], hs["spike_1"]["span_mv"]["median"]
        fit_later_max = fs["later"]["span_mv"]["max"] if fs["later"]["span_mv"] else fit_span_max
        lead = result["axon_lead"][f"{cell} fit"]
        checks += [
            {"cell": cell, "prediction": "fit somatic onset is a kink (span under 1 mV) at every spike", "value": max(fit_span_max, fit_later_max),
             "pass": max(fit_span_max, fit_later_max) < KINK_SPAN_MV},
            {"cell": cell, "prediction": "fit axon point crosses 10 V/s before the soma at every spike", "value": lead["min_ms"], "pass": lead["axon_first_every_spike"]},
            {"cell": cell, "prediction": "recording spike-1 span is resolved (not 1 to 2 mV) and its repeats spread under 1 mV", "value": human_span,
             "pass": classify(human_span) != "unresolved" and sig["span_range_mv"] < 1.},
            {"cell": cell, "prediction": "recording and fit spike-1 spans differ by more than 3 sigma of the repeat spread",
             "value": human_span-fs["spike_1"]["span_mv"]["median"], "sigmas": abs(human_span-fs["spike_1"]["span_mv"]["median"])/sig["span_mv"],
             "pass": abs(human_span-fs["spike_1"]["span_mv"]["median"]) > 3*sig["span_mv"]},
        ]
        if cell == "E":
            checks.append({"cell": cell, "prediction": "E recording spike-1 onset is gradual (span over 2 mV)", "value": human_span, "pass": human_span > GRADUAL_SPAN_MV})
    for c in checks:
        c["pass"] = bool(c["pass"]); c["value"] = float(c["value"])
    reading, rejection = {}, {}
    for cell in ("E", "I"):
        h, f = result["summary"][f"{cell} human"]["soma"]["spike_1"]["span_mv"]["median"], result["summary"][f"{cell} fit"]["soma"]["spike_1"]["span_mv"]["median"]
        reading[cell] = f"recording {classify(h)} ({h:.1f} mV), fit {classify(f)} ({f:.1f} mV), fit/recording {f/h:.1f}"
        if cell == "I":
            reading[cell] += "; the absolute bands were registered for the E cell and are not applied to I, which is read as the ratio"
        rng = result["sigma"][cell]["span_range_mv"]
        rejection[cell] = f"fired: repeat span range {rng:.2f} mV over 1 mV; the reading is provisional" if rng > 1. else "not fired"
    verdict = {cell: "PASS" if all(c["pass"] for c in checks if c["cell"] == cell) else "FAIL" for cell in ("E", "I")}
    return {"checks": checks, "classes": reading, "rejection": rejection, "verdict": verdict}


def markdown(result):
    lines = ["# SP15 stage 8: soft or hard take-off", "",
             "Retained traces only, uniform 0.02 ms grid, rate of rise as a central difference over 0.1 ms. Onset read from the first 10 V/s "
             "crossing to the 100 V/s crossing: span (mV), rapidness (slope of dV/dt against V from 10 to 50 V/s, per ms). Fits also at their axon point "
             f"(E {AXON_POINT['E']}; I {AXON_POINT['I']}): lead of the axonal 10 V/s crossing over the somatic one, and the axon voltage when the soma crosses 10 V/s.", ""]
    for who, tables in result["tables"].items():
        lines += [f"## {who}", ""]
        has_axon = any(tab and "axon" in tab[0] for tab in tables.values())
        head = "| train | spike | threshold | soma span | soma rapidness |" + (" axon span | axon lead ms | axon mV at soma 10 V/s |" if has_axon else "")
        lines += [head, "| --- | ---: | ---: | ---: | ---: |" + (" ---: | ---: | ---: |" if has_axon else "")]
        for label, tab in tables.items():
            for row in tab[:6]:
                cells = f"| {label} | {row['spike']} | {row['threshold_mv']:.1f} | {row['soma']['span_mv']:.2f} | {row['soma']['rapidness_per_ms']:.0f} |"
                if has_axon:
                    cells += f" {row['axon']['span_mv']:.2f} | {row['axon_lead_ms']:.3f} | {row['axon_mv_at_soma_10']:.1f} |"
                lines.append(cells)
            if len(tab) > 6:
                lines.append(f"| {label} | ... | | | |" + (" | | |" if has_axon else "") + f" ({len(tab)-6} more)")
        lines.append("")
    lines += ["## Summary (median, min to max)", ""]
    for who, summ in result["summary"].items():
        for site, s in summ.items():
            for phase in ("spike_1", "later"):
                sp, ra = s[phase]["span_mv"], s[phase]["rapidness_per_ms"]
                if sp:
                    lines.append(f"- {who}, {site}, {phase.replace('_', ' ')}: span {sp['median']:.2f} mV ({sp['min']:.2f} to {sp['max']:.2f}), rapidness {ra['median']:.0f}/ms ({ra['min']:.0f} to {ra['max']:.0f}), n {sp['n']}")
    lines += ["", "## Axon lead (fits)", ""]
    lines += [f"- {who}: n {v['n']}; lead {v['median_ms']:.3f} ms median, {v['min_ms']:.3f} min; axon first at every spike: {v['axon_first_every_spike']}; "
              f"axon at {v['axon_mv_at_soma_10_median']:.1f} mV (median over spikes) when the soma crosses 10 V/s; spike 1 of each train: "
              + "; ".join(f"lead {r['lead_ms']:+.3f} ms, axon {r['axon_mv_at_soma_10']:.1f} mV" for r in v["spike_1"])
              for who, v in result["axon_lead"].items()]
    lines += ["", "## The E fit's take-off at the soma and at the axon point under the SP16 gate (310 pA, retained stage-1 traces)", "",
              "| gate depth | spike | soma 10 V/s mV | axon 10 V/s mV | axon lead ms |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, rows in result["sp16_ais"].items():
        lines += [f"| {label} | {r['spike']} | {r['soma_10_mv']:.1f} | {r['axon_10_mv']:.1f} | {r['axon_lead_ms']:.2f} |" for r in rows] if rows else [f"| {label} | trace not retained locally | | | |"]
    lines += ["", "## Repeat spread of spike 1", ""]
    lines += [f"- {cell}: n {s['n']}; span sd {s['span_mv']:.2f} mV (range {s['span_range_mv']:.2f}); rapidness sd {s['rapidness_per_ms']:.1f}/ms" for cell, s in result["sigma"].items()]
    d = result["decision"]
    lines += ["", "## Decision", ""]+[f"- {c['cell']} {'PASS' if c['pass'] else 'FAIL'}: {c['prediction']} ({c['value']:+.2f}" + (f", {c['sigmas']:.1f} sigma)" if "sigmas" in c else ")") for c in d["checks"]]
    lines += ["", f"Classes: E {d['classes']['E']}; I {d['classes']['I']}", f"Registered rejection: E {d['rejection']['E']}; I {d['rejection']['I']}",
              f"Verdict: E {d['verdict']['E']}, I {d['verdict']['I']}", "", "## Reading", ""]+result["reading"]
    return "\n".join(lines)+"\n"


def plot(result, path):
    """Phase-plane onsets, spike 1 of each train: recording over fit soma, fit axon dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cell in zip(axes, ("E", "I")):
        for label, (t, v, axon) in result["_traces"][cell].items():
            rate = rate_of_rise(v)
            ax.plot(v, rate, lw=1., label=label)
            if axon is not None:
                ax.plot(axon, rate_of_rise(axon), lw=.8, ls="--", color=ax.lines[-1].get_color(), alpha=.6)
        ax.set_xlim(-70., -30.); ax.set_ylim(-5., 150.); ax.axhline(RISE_LOW_V_S, color="0.7", lw=.5); ax.axhline(RISE_HIGH_V_S, color="0.7", lw=.5)
        ax.set_xlabel("V, mV"); ax.set_ylabel("dV/dt, V/s"); ax.set_title(f"{cell}: spike-1 onset, recording and fit soma (fit axon dashed)"); ax.legend(fontsize=6); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def onset_windows(cell):
    """Spike-1 windows (4 ms before to 1 ms after the threshold) for the plot."""
    c, out = CELLS[cell], {}
    if cell == "E":
        for s in (48, 53, 56):
            t, v = nwb_sweep(E_NWB, s); r = landmarks(t, v, E_PULSE)
            if r:
                m = (t >= r[0]["threshold_ms"]-4.) & (t <= r[0]["threshold_ms"]+1.); out[f"human sweep {s}"] = (t[m], v[m], None)
    else:
        for label, spec in c["inputs"].items():
            t, v = human_trace(ROOT/c["human_dir"]/spec[0]); t, v, _ = resample(np.asarray(t, float), np.asarray(v, float), np.zeros(len(t))); r = landmarks(t, v, c["pulse"])
            if r:
                m = (t >= r[0]["threshold_ms"]-4.) & (t <= r[0]["threshold_ms"]+1.); out[f"human {label}"] = (t[m], v[m], None)
    for label, spec in c["inputs"].items():
        t, v, axon = model_with_axon(spec); r = landmarks(t, v, c["pulse"])
        if r:
            m = (t >= r[0]["threshold_ms"]-4.) & (t <= r[0]["threshold_ms"]+1.); out[f"fit {label}"] = (t[m], v[m], axon[m])
    return out


READING = [
    "E: the somatic onset has the same shape in the recording and in the fit. From 10 to 100 V/s the recording covers 5.0 mV "
    "(3.5 to 5.7 over 12 trains, repeat sd 0.26) and the fit 5.1 mV (3.4 to 5.5); the rapidness is 34 against 41 per ms. Both are "
    "gradual by the registered class, and they differ by 0.4 sigma in span. Yet in the fit the axon point crosses 10 V/s 0.28 ms before "
    "the soma at every one of 22 spikes and is at about -40 mV, mid-spike, when the soma starts: the somatic onset of this fit is driven "
    "by an axonal spike that has already fired, and it still turns over as gradually as the recording. Prediction 1 is refuted: an "
    "imposed take-off does not show as a kink at this soma. The onset shape therefore does not discriminate the two take-off mechanisms "
    "that stage 7 separated; what differs is the voltage at which the turnover begins, fixed in the fit, trajectory-set in the recording.",
    "I (provisional: the registered rejection fired, the four repeats spread 1.16 mV in span). The recording's onset covers 2.7 mV "
    "from 10 to 100 V/s (1.9 to 3.8; rapidness 52 per ms), gradual by the absolute band registered for the E cell but a fifth of its "
    "fit's, which turns over across 14.2 mV (rapidness 5 per ms): 20 sigma of the repeat spread and ten times its range. The reading "
    "stands on that ratio, not on the band, until a repeat set resolves the span; stage 9's I probe run serves as that confirmation. In the fit the soma and the first axon section rise "
    "together, the soma crossing 10 V/s 0.18 ms before the axon point and the axon point overtaking it only above 50 V/s; the earlier "
    "axon-first readings were taken at a high fixed voltage and describe the peak, not the onset. The I fit has no sharp initiation: "
    "its spike begins as a whole-cell turnover, where the recorded cell's begins five times more abruptly, as a spike arriving from a "
    "site that fires first. This is the largest elemental contrast in the I cell after the count, and it is a contrast of the initiation site, an input.",
    "E fit, the take-off at the axon point: -54.3 to -55.2 mV across every spike and drive, sliding 0.9 mV from 0.25 to 1.16 mV/ms while "
    "the soma holds at -57.2; under the SP16 gate at depth 0.6 the axon point's take-off climbs 2.1 mV by spike 10 (-55.0 to -52.9) and the "
    "soma's 1.5 mV, together, with the lead unchanged at 0.3 ms. So the fit's take-off is set at or proximal to the axon point (45 um out; the point fires first, so initiation is at or "
    "before it), and there it answers to a "
    "40 percent loss of sodium availability by only 2 mV: a dense insertion with fast kinetics crosses its own threshold at nearly the same "
    "voltage whatever its availability. The recording moves 2.2 mV with the approach alone and 1.9 mV more after one spike, which is more "
    "than availability at such a site can give.",
    "Both-gradual was not a registered reading for E; it is recorded as a no-discrimination outcome of this split, not as evidence for "
    "either branch. The next E split has to act on the timing of the axonal spike relative to the somatic approach, which is where the "
    "fixed take-off is made; the next I split is the initiation site's density, length and coupling.",
]


def main():
    tables = {"E human": human_e(), "E fit": models("E"), "I human": human_i(), "I fit": models("I")}
    summ = {who: {"soma": summary(tab, "soma")} for who, tab in tables.items()}
    for cell in ("E", "I"):
        summ[f"{cell} fit"]["axon"] = summary(tables[f"{cell} fit"], "axon")
    result = {"stage": "8", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "axon_point": AXON_POINT,
              "tables": tables, "summary": summ,
              "axon_lead": {f"{cell} fit": axon_lead(tables[f"{cell} fit"]) for cell in ("E", "I")},
              "sigma": {"E": repeat_sigma(tables["E human"], {f"sweep {s}" for s in E_REPEATS}),
                        "I": repeat_sigma(tables["I human"], {f"repeat sweep {s}" for s in CELLS["I"]["repeat_sweeps"]})},
              "sp16_ais": sp16_ais(), "reading": READING}
    result["decision"] = decide(result)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"onset-shape.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"onset-shape.md").write_text(markdown(result), encoding="utf-8")
    result["_traces"] = {cell: onset_windows(cell) for cell in ("E", "I")}
    plot(result, OUT/"onset-shape.png")
    print("wrote", OUT/"onset-shape.json", result["decision"]["classes"], result["decision"]["verdict"])


if __name__ == "__main__":
    main()
