"""SP15 stage 16 part 2: the threshold climb as accommodation at the initiation site.

Part 1 (recorded in the spec) refused the somatic dose: somatic outward current accumulates
across the train while the take-off does not move, because at the take-off the axon delivers
about thirty times the accumulated somatic brake. So the climb is dosed where the spike starts.
The axonal sodium's recovery from inactivation is dosed on the stage-10 thickened-stub geometry,
whose somatic take-off tracks the site, with the same strongest dose on the default 1 um stub as
the coupling control.

Outputs ``docs/evidence/h01-topographic/stage-16.{json,md,png}``.
"""

import json

import numpy as np

from h01_topographic_donor import climb
from h01_topographic_stage0 import OUT
from h01_topographic_swap import accuracy, read, recorded

CONTROL = "h01-e-coupling-b/s10-d2-density-sweep53"          # h_recovery_factor 1.0, the arm's own control
DOSES = {"1.0": CONTROL,
         "0.5": "h01-e-climb/s16-axrec-050-sweep53",
         "0.25": "h01-e-climb/s16-axrec-025-sweep53",
         "0.125": "h01-e-climb/s16-axrec-0125-sweep53"}
STUB1 = "h01-e-climb/s16-axrec-0125-stub1-sweep53"           # coupling control, 1 um stub, strongest dose
B3_RUN = "h01-e-currents/m0-b3-currents-sweep53"
RECORDED = {"climb_2_mv": 1.38, "climb_5_mv": 3.63, "rise_5_over_1": .86, "count": 10}
TARGET_STEP_MV = .69                                          # registered: at least half the recorded step
COUNT_BAND = (5, 15)
RISE_FRACTION = .15                                           # of this arm's own control


def row_of(r):
    """Climb and the guard quantities of one run.

    The registered cell (c) is the FIRST-INTERVAL step, which needs only two spikes, so climb_2 is
    read directly from the first two spikes; the campaign's climb() helper is used for the spike-5
    step and the rise ratio and returns nothing on trains shorter than five.
    """
    row = {"status": r.get("status"), "count": r.get("count", 0)}
    if r.get("status") != "read" or not r["spikes"]:
        return row
    sp = r["spikes"]
    row.update({"rise_v_s": sp[0]["rise_v_s"], "take_off_mv": sp[0]["take_off_mv"],
                "fall_v_s": sp[0]["fall_v_s"], "rest_mv": r.get("rest_mv")})
    if len(sp) >= 2:
        row["climb_2_mv"] = sp[1]["threshold_mv"]-sp[0]["threshold_mv"]
    c = climb(sp)
    if c:
        row.update({"climb_5_mv": c["climb_5_mv"], "rise_5_over_1": c["rise_5_over_1"]})
    return row


def series(chain):
    """Climb and the guard quantities at each dose, through the recording's chain."""
    return {dose: row_of(read(base, chain)) for dose, base in DOSES.items()}


def monotone_up(values):
    xs = [v for v in values if v is not None and np.isfinite(v)]
    return len(xs) >= 2 and all(b >= a-1e-9 for a, b in zip(xs, xs[1:]))


def decide(result):
    checks = []

    def add(reading, prediction, value, passed, note=""):
        checks.append({"reading": reading, "prediction": prediction, "value": value, "pass": passed, "note": note})
    s = result["series"]
    order = ["1.0", "0.5", "0.25", "0.125"]                   # falling recovery factor = more accommodation
    read_doses = [d for d in order if s.get(d, {}).get("status") == "read"]
    if len(read_doses) < 2:
        add("c", "the first-interval threshold step grows as the axonal recovery factor falls", None, None,
            "fewer than two doses read")
        return {"verdict": "no reading", "checks": checks}
    steps = [s[d].get("climb_2_mv") for d in read_doses]
    add("c", f"some dose reaches at least half the recorded first-interval step (+{TARGET_STEP_MV} of +1.38 mV), "
             "and the climb is monotone as the recovery factor falls",
        {d: (round(s[d]["climb_2_mv"], 2) if s[d].get("climb_2_mv") is not None else None) for d in read_doses},
        any(x is not None and x >= TARGET_STEP_MV for x in steps) and monotone_up(steps),
        f"recorded +1.38; B3 -0.02; HL23PYR +1.38")
    ctrl = s.get("1.0", {})
    best_d, best = None, None
    for d in read_doses:
        x = s[d].get("climb_2_mv")
        if x is not None and (best is None or x > best):
            best, best_d = x, d
    ok_guard = None
    if best_d and ctrl.get("rise_v_s"):
        cnt = s[best_d]["count"]
        rise = s[best_d].get("rise_v_s")
        ok_guard = (COUNT_BAND[0] <= cnt <= COUNT_BAND[1]
                    and rise is not None and abs(rise-ctrl["rise_v_s"]) <= RISE_FRACTION*ctrl["rise_v_s"])
    add("d", "at the best dose the count stays within 5 to 15 and the spike-1 rise within 15 percent of the arm's control",
        {"dose": best_d, "count": s.get(best_d, {}).get("count"),
         "rise_v_s": s.get(best_d, {}).get("rise_v_s"), "control_rise_v_s": ctrl.get("rise_v_s")},
        ok_guard, "the climb must not be bought by crippling the cell")
    stub = result["coupling_control"]
    coupled = s.get("0.125", {}).get("climb_2_mv")
    stub_step = stub.get("climb_2_mv") if stub.get("status") == "read" else None
    add("f", "the same strongest dose on the default 1 um stub gives at most a fraction of the coupled arm's climb",
        {"coupled_2um": (round(coupled, 2) if coupled is not None else None),
         "stub_1um": (round(stub_step, 2) if stub_step is not None else None)},
        (None if coupled is None or stub_step is None else stub_step < .5*coupled),
        "if they are equal the coupling is not required and stage 10's account of the take-off is wrong")
    acc = result["accuracy"]
    add("g", "percent accuracy of the best climb dose beside B3's 71.8",
        {"b3_pct": round(acc["b3"]["mean_pct"], 1) if acc.get("b3") else None,
         "best_climb_pct": round(acc["best_climb"]["mean_pct"], 1) if acc.get("best_climb") else None},
        None, "reported; this arm's baseline rise is further from the recording than plain B3's")
    fired = [c for c in checks if c["reading"] in ("c", "d") and c["pass"] is False]
    verdict = "PASS" if not fired and any(c["reading"] == "c" and c["pass"] for c in checks) else "MIXED"
    return {"verdict": verdict, "checks": checks}


def markdown(result):
    d = result["decision"]
    s = result["series"]
    lines = ["# Stage 16: the threshold climb at the initiation site", "",
             "Part 1 refused the somatic dose (outward current accumulates across the train while the take-off",
             "does not move; at take-off the axon delivers ~0.33 nA against a ~0.012 nA somatic brake).",
             "Part 2 doses the axonal sodium's recovery on the stage-10 thickened-stub geometry.", "",
             f"Verdict: **{d['verdict']}**", "",
             "## The characteristic curve (axonal sodium recovery factor -> threshold climb)", "",
             "| recovery factor | climb spike 2 (mV) | climb spike 5 (mV) | count | spike-1 rise (V/s) | rise 5/1 |",
             "|---|---|---|---|---|---|"]
    for dose in ["1.0", "0.5", "0.25", "0.125"]:
        r = s.get(dose, {})
        if r.get("status") != "read":
            lines.append(f"| {dose} | {r.get('status', 'not run')} | - | {r.get('count', 0)} | - | - |")
            continue
        lines.append(f"| {dose} | {r.get('climb_2_mv', float('nan')):+.2f} | {r.get('climb_5_mv', float('nan')):+.2f} | "
                     f"{r['count']} | {r.get('rise_v_s', float('nan')):.0f} | {r.get('rise_5_over_1', float('nan')):.2f} |")
    st = result["coupling_control"]
    lines += ["", "## Coupling control (same strongest dose, default 1 um stub)", ""]
    if st.get("status") == "read":
        lines.append(f"climb spike 2 = {st.get('climb_2_mv', float('nan')):+.2f} mV, count {st.get('count')}, "
                     f"rise {st.get('rise_v_s', float('nan')):.0f} V/s")
    else:
        lines.append(f"status: {st.get('status')}")
    lines += ["", f"Recorded: climb +1.38 / +3.63 mV, rise ratio 0.86, count 10. B3: -0.02 / +0.24. HL23PYR: +1.38 / +1.78.",
              "", "## Percent accuracy (ten elements, the stage-15 definition)", ""]
    acc = result["accuracy"]
    if acc.get("b3") and acc.get("best_climb"):
        lines += ["| element | recorded | B3 | best climb dose | B3 % | climb % |", "|---|---|---|---|---|---|"]
        for k, v in acc["b3"]["elements"].items():
            w = acc["best_climb"]["elements"].get(k)
            lines.append(f"| {k} | {v['recorded']:.2f} | {v['model']:.2f} | {w['model']:.2f} | "
                         f"{v['accuracy_pct']:.0f} | {w['accuracy_pct']:.0f} |" if w else
                         f"| {k} | {v['recorded']:.2f} | {v['model']:.2f} | - | {v['accuracy_pct']:.0f} | - |")
        lines.append(f"| **mean** | | | | **{acc['b3']['mean_pct']:.1f}** | **{acc['best_climb']['mean_pct']:.1f}** |")
    lines += ["", "## Cells", ""]
    for c in d["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "--"}[c["pass"]]
        lines.append(f"- ({c['reading']}) {mark}: {c['prediction']} -> {c['value']}. {c['note']}")
    lines += ["", "## Reading", ""]+[f"- {r}" for r in result["reading"]]
    return "\n".join(lines)+"\n"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = result["series"]
    order = ["1.0", "0.5", "0.25", "0.125"]
    xs = [float(d) for d in order if s.get(d, {}).get("climb_2_mv") is not None]
    ys = [s[d]["climb_2_mv"] for d in order if s.get(d, {}).get("climb_2_mv") is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(xs, ys, "o-", color="C0", label="2 um stub (coupled)")
    st = result["coupling_control"]
    if st.get("status") == "read" and st.get("climb_2_mv") is not None:
        ax1.plot([0.125], [st["climb_2_mv"]], "s", color="C3", label="1 um stub (control)")
    ax1.axhline(RECORDED["climb_2_mv"], color="k", ls="--", lw=1, label="recorded +1.38")
    ax1.axhline(TARGET_STEP_MV, color="0.5", ls=":", lw=1, label="target +0.69")
    ax1.invert_xaxis()
    ax1.set(xlabel="axonal sodium recovery factor (falling = more accommodation)",
            ylabel="threshold step spike 2 - spike 1 (mV)", title="the climb against the dose")
    ax1.legend(fontsize=7)
    acc = result["accuracy"]
    if acc.get("b3") and acc.get("best_climb"):
        names = list(acc["b3"]["elements"])
        x = np.arange(len(names))
        ax2.bar(x-.2, [acc["b3"]["elements"][k]["accuracy_pct"] for k in names], .4,
                label=f"B3 ({acc['b3']['mean_pct']:.0f}%)", color="C3")
        ax2.bar(x+.2, [acc["best_climb"]["elements"].get(k, {}).get("accuracy_pct", 0) for k in names], .4,
                label=f"best climb ({acc['best_climb']['mean_pct']:.0f}%)", color="C0")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax2.set(ylabel="accuracy (%)", title="per-element accuracy at 310 pA")
        ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


READING = []


def main():
    s12 = json.loads((OUT/"stage-12.json").read_text(encoding="utf-8"))
    chain = {"tau_p_us": s12["E"]["chain"]["tau_p_us"], "fc_khz": s12["E"]["chain"]["fc_khz"]}
    rec = recorded(chain)
    s = series(chain)
    stub_row = row_of(read(STUB1, chain))
    acc = {"b3": accuracy(read(B3_RUN, chain), rec)}
    best, best_dose = None, None
    for dose, base in DOSES.items():
        a = accuracy(read(base, chain), rec)
        if a and (best is None or a["mean_pct"] > best["mean_pct"]):
            best, best_dose = a, dose
    acc["best_climb"], acc["best_dose"] = best, best_dose
    result = {"stage": "16", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "chain": chain,
              "recorded_reference": RECORDED, "series": s, "coupling_control": stub_row,
              "accuracy": acc, "reading": READING}
    result["decision"] = decide(result)
    (OUT/"stage-16.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-16.md").write_text(markdown(result), encoding="utf-8")
    plot(result, OUT/"stage-16.png")
    print("wrote", OUT/"stage-16.json", result["decision"]["verdict"],
          "| best climb", best_dose, None if not best else round(best["mean_pct"], 1))


if __name__ == "__main__":
    main()
