"""SP15 stage 10: the coupling between the soma and the initiation site, every channel held.

Reads ``h01-e-coupling/`` (three arms: stub diameter x2 with the total axonal NaTs held, x2
with the density held, x4 with the total held) with the stage-9 runs at 1 um as the control:
the spike-1 take-off at the soma, ``axon[0](0.5)`` and ``axon[1](0.5)`` against the measured
approach under the three ramps, the somatic onset span, the axon lead, and the 310 pA train
(count and the take-off cycle by cycle) for the x2 arms.

Outputs ``docs/evidence/h01-topographic/stage-10.{json,md,png}``.
"""

import json

import numpy as np

from h01_topographic_stage0 import OUT
from h01_topographic_stage9 import E_POINTS, E_PULSE, load_points, point_rows

ARMS = {"control 1 um": ("h01-e-takeoff/s9-b3-n9", 1., "3.814"), "x2 total held": ("h01-e-coupling/s10-d2-total", 2., "1.907"),
        "x2 density held": ("h01-e-coupling-b/s10-d2-density", 2., "3.814"), "x4 total held": ("h01-e-coupling-c/s10-d4-total", 4., "0.9535")}
RAMPS = {"ramp01": 1.1, "ramp1": 11., "ramp10": 110.}
STEP = "sweep53"
LIMITS = {"soma_slide_x4_mv": 1., "soma_slide_x2_mv": .5, "x2_arms_agree_mv": .5, "axon_lead_x4_ms": .15, "span_mv": (3.5, 5.7), "count_310": (8, 12)}


def arm_ramps(base):
    """Spike-1 rows of the three ramps of one arm; status per ramp."""
    out = {}
    for label, rate in RAMPS.items():
        traces = load_points(f"{base}-{label}", E_POINTS)
        if traces is None:
            out[label] = {"rate_pa_per_ms": rate, "status": "not run"}
            continue
        rows = point_rows(traces, E_PULSE, E_PULSE[0])
        out[label] = {"rate_pa_per_ms": rate, "status": "read" if rows else "no spike", "count": len(rows), "spike_1": rows[0] if rows else None}
    return out


def arm_step(base):
    """The 310 pA train of one arm: count and the take-off at the three points, cycle by cycle."""
    traces = load_points(f"{base}-{STEP}", E_POINTS)
    if traces is None:
        return {"status": "not run"}
    rows = point_rows(traces, E_PULSE, E_PULSE[0])
    return {"status": "read" if rows else "no spike", "count": len(rows),
            "cycles": [{"spike": r["spike"], "approach_mv_per_ms": r["approach_mv_per_ms"],
                        **{f"{n}_take_off_mv": r["points"][n]["take_off_mv"] for n in E_POINTS if n in r["points"]},
                        "soma_span_mv": r["points"]["soma"]["span_mv"], "axon1_lead_ms": r["points"]["axon1"]["lead_ms"] if "axon1" in r["points"] else float("nan")}
                       for r in rows]}


def slide(ramps, point):
    """Take-off at ``point`` from the slowest to the fastest ramp that spiked, and the approach range."""
    read = [r for r in ramps.values() if r.get("status") == "read" and point in r["spike_1"]["points"]]
    if len(read) < 2:
        return None
    read.sort(key=lambda r: r["spike_1"]["approach_mv_per_ms"])
    values = [r["spike_1"]["points"][point]["take_off_mv"] for r in read]
    return {"n": len(read), "approach_range": [read[0]["spike_1"]["approach_mv_per_ms"], read[-1]["spike_1"]["approach_mv_per_ms"]],
            "take_off_mv": values, "slide_mv": values[-1]-values[0], "range_mv": float(np.ptp(values))}


def summarise(arm):
    ramps = arm["ramps"]
    read = [r for r in ramps.values() if r.get("status") == "read"]
    spans = [r["spike_1"]["points"]["soma"]["span_mv"] for r in read]
    leads = [r["spike_1"]["points"]["axon1"]["lead_ms"] for r in read if "axon1" in r["spike_1"]["points"]]
    return {"soma": slide(ramps, "soma"), "axon0": slide(ramps, "axon0"), "axon1": slide(ramps, "axon1"),
            "soma_span_range_mv": [min(spans), max(spans)] if spans else None, "axon1_lead_range_ms": [min(leads), max(leads)] if leads else None,
            "count_310": arm["step"].get("count")}


def decide(result):
    s = {name: result["arms"][name]["summary"] for name in ARMS}
    lo, hi = LIMITS["span_mv"]
    checks = []
    def add(arm, prediction, value, passed, note=""):
        checks.append({"arm": arm, "prediction": prediction, "value": None if value is None else float(value), "pass": None if passed is None else bool(passed), "note": note})
    x4 = s["x4 total held"]
    if x4["soma"]:
        add("x4 total held", "somatic take-off falls by more than 1 mV from the slowest to the fastest ramp", x4["soma"]["slide_mv"], x4["soma"]["slide_mv"] < -LIMITS["soma_slide_x4_mv"])
        add("x4 total held", "axon[1] lead below 0.15 ms", x4["axon1_lead_range_ms"][1] if x4["axon1_lead_range_ms"] else None,
            x4["axon1_lead_range_ms"][1] < LIMITS["axon_lead_x4_ms"] if x4["axon1_lead_range_ms"] else None)
    else:
        add("x4 total held", "somatic take-off falls by more than 1 mV", None, None, "fewer than two ramps read")
    for name in ("x2 total held", "x2 density held"):
        a = s[name]
        if a["soma"]:
            add(name, "somatic take-off falls by more than 0.5 mV", a["soma"]["slide_mv"], a["soma"]["slide_mv"] < -LIMITS["soma_slide_x2_mv"])
        else:
            add(name, "somatic take-off falls by more than 0.5 mV", None, None, "fewer than two ramps read")
    if s["x2 total held"]["soma"] and s["x2 density held"]["soma"]:
        diff = s["x2 total held"]["soma"]["slide_mv"]-s["x2 density held"]["soma"]["slide_mv"]
        add("x2 arms", "the two x2 slides agree within 0.5 mV (coupling, not density)", diff, abs(diff) < LIMITS["x2_arms_agree_mv"])
    for name in ("x2 total held", "x2 density held", "x4 total held"):
        a = s[name]
        if a["soma_span_range_mv"]:
            add(name, "somatic onset span stays within 3.5 to 5.7 mV", a["soma_span_range_mv"][1], lo <= a["soma_span_range_mv"][0] and a["soma_span_range_mv"][1] <= hi)
        if a["count_310"] is not None:
            add(name, "310 pA count within 2 of 10", a["count_310"], LIMITS["count_310"][0] <= a["count_310"] <= LIMITS["count_310"][1])
    verdict = "no reading" if any(c["pass"] is None for c in checks) else ("PASS" if all(c["pass"] for c in checks) else "FAIL")
    return {"checks": checks, "verdict": verdict}


def markdown(result):
    lines = ["# SP15 stage 10: the coupling between the soma and the site", "",
             "Every channel held; the axon stub's diameter changed (source 1 um), with the axonal NaTs density divided by the area factor in the total-held arms. "
             "Spike-1 take-off = first 10 V/s crossing at each point; approach measured over up to 10 ms before the somatic threshold.", ""]
    for name, arm in result["arms"].items():
        base, diam, nats = ARMS[name]
        lines += [f"## {name} (diameter {diam} um, NaTs axon {nats} S/cm2)", "",
                  "| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span mV | axon1 lead ms |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for label, r in arm["ramps"].items():
            if r.get("spike_1"):
                p = r["spike_1"]["points"]
                g = lambda n, k: p[n][k] if n in p else float("nan")
                lines.append(f"| {r['rate_pa_per_ms']} | {r['status']} | {r['count']} | {r['spike_1']['approach_mv_per_ms']:.2f} | {g('soma','take_off_mv'):.1f} | {g('axon0','take_off_mv'):.1f} | {g('axon1','take_off_mv'):.1f} | {g('soma','span_mv'):.1f} | {g('axon1','lead_ms'):+.2f} |")
            else:
                lines.append(f"| {r['rate_pa_per_ms']} | {r['status']} | | | | | | | |")
        su = arm["summary"]
        if su["soma"]:
            lines += ["", f"Slide, slowest to fastest ramp ({su['soma']['approach_range'][0]:.2f} to {su['soma']['approach_range'][1]:.2f} mV/ms): "
                      + "; ".join(f"{n} {su[n]['slide_mv']:+.2f} mV" for n in ("soma", "axon0", "axon1") if su[n])]
        st = arm["step"]
        if st.get("status") == "read":
            lines += ["", f"310 pA step: {st['count']} spikes", "", "| spike | approach | soma take-off | axon0 | axon1 | soma span | axon1 lead |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            lines += [f"| {c['spike']} | {c['approach_mv_per_ms']:.2f} | {c['soma_take_off_mv']:.1f} | {c.get('axon0_take_off_mv', float('nan')):.1f} | {c.get('axon1_take_off_mv', float('nan')):.1f} | {c['soma_span_mv']:.1f} | {c['axon1_lead_ms']:+.2f} |"
                      for c in st["cycles"][:5]+st["cycles"][-1:]]
        elif st.get("status"):
            lines += ["", f"310 pA step: {st['status']}"]
        lines.append("")
    lines += ["## Decision", ""]
    for c in result["decision"]["checks"]:
        v = "no reading" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
        val = "n/a" if c["value"] is None else f"{c['value']:+.2f}"
        lines.append(f"- {c['arm']} {v}: {c['prediction']} ({val}{'; '+c['note'] if c['note'] else ''})")
    lines += ["", f"Verdict: {result['decision']['verdict']}", "", "## Reading", ""]+result["reading"]
    return "\n".join(lines)+"\n"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, point in zip(axes, ("soma", "axon0", "axon1")):
        for name, arm in result["arms"].items():
            read = sorted([r for r in arm["ramps"].values() if r.get("status") == "read" and point in r["spike_1"]["points"]], key=lambda r: r["spike_1"]["approach_mv_per_ms"])
            if read:
                ax.plot([r["spike_1"]["approach_mv_per_ms"] for r in read], [r["spike_1"]["points"][point]["take_off_mv"] for r in read], "o-", label=name)
        ax.set_xscale("log"); ax.set_title(f"{point}: spike-1 take-off against approach"); ax.set_xlabel("approach, mV/ms"); ax.grid(alpha=.3)
    axes[0].set_ylabel("take-off (10 V/s crossing), mV"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


READING = [
    "The coupling makes the somatic take-off a function of the approach. Slowest to fastest ramp (about 0.4 to 2.6 mV/ms): control "
    "-0.07 mV; x2 with the total axonal sodium held -2.9 mV; x2 with the density held -2.1 mV; x4 with the total held -3.4 mV. The "
    "recorded soma slides -2.2 mV from 0.2 to 0.75 mV/ms. The axon lead falls from 0.30 ms to 0.16, 0.20 and 0.04 ms.",
    "Two registered rejections fire by the letter. The two x2 arms differ by 0.8 mV (limit 0.5): the density modulates the slide, "
    "but both arms slide by thirty times the control's, so the coupling is the main effect and the density a secondary one. The "
    "somatic onset span leaves the recorded 3.5 to 5.7 mV band in every arm: 6.1 to 8.9 mV at x2 with the total held, 10.7 to 11.5 "
    "at x4, where the site has merged into the soma and the spike is becoming the whole-cell turnover of the I fit; at x2 with the "
    "density held the span is 4.1, 5.3 and 6.0 mV under the ramps and 3.8 to 5.4 along the 310 pA train, outside the band by 0.3 mV "
    "at one ramp only. The coupling therefore changes the spike as well as its reading, least in the density-held x2 arm.",
    "The density-held x2 arm is the closest of anything this campaign has run to the recorded take-off: -55.2, -55.7 and -57.3 mV at "
    "0.45, 0.99 and 2.50 mV/ms against the recording's -54.5 to -55.2 at 0.2 mV/ms and -57.2 at 0.75; span 4.1 to 6.0 against 3.5 to "
    "5.7; 310 pA count 9 against 10. It does not reproduce the two other recorded elements: along the train the take-off holds at "
    "-55.1 to -55.5 mV (the recording steps up 1.9 mV after the first spike), and the spike-1 rise is 655 V/s against 348 (the "
    "x2 total-held arm 579), so the coupling is not the rise lever either.",
    "Reading: the recorded soma reads its initiation site through a coupling tighter than the Allen stub, and the approach signature "
    "of the take-off follows that input and not a channel. What remains of the recorded take-off is what a spike leaves behind, "
    "+1.9 mV that no arm produces, and the rise, which no coupling touches.",
]


def main():
    arms = {}
    for name, (base, _, _) in ARMS.items():
        arm = {"ramps": arm_ramps(base), "step": arm_step(base) if name != "x4 total held" else {"status": "not registered"}}
        arm["summary"] = summarise(arm)
        arms[name] = arm
    result = {"stage": "10", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "limits": LIMITS, "arms": arms, "reading": READING}
    result["decision"] = decide(result)
    (OUT/"stage-10.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-10.md").write_text(markdown(result), encoding="utf-8")
    plot(result, OUT/"stage-10.png")
    print("wrote", OUT/"stage-10.json", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
