"""SP15 stage 15: the human-for-rodent sodium swap on the human cell, and the percent accuracy.

Arm A swaps the somatic sodium equations of B3 (human cell, human anatomy, human-fitted
densities, Allen's rodent-lineage NaTs) for the Toronto human NaTg with HL23PYR's human
voltage shifts, dosed across four densities, and reads the characteristic curve of the
spike-1 rise at 310 pA through the stage-12 chain. Arm B (HL23PYR biophysics on this cell's
anatomy) is recorded as a no-reading: the published Toronto template cannot construct on this
morphology and editing it would break the split's unchanged condition.

The percent accuracy was defined in the spec before the runs: ten elements at 310 pA, each
scored ``max(0, 1 - |model - recorded| / scale)``, the scale being |recorded| for quantities
with a meaningful zero and a stated voltage span for the level quantities. The mean is a
lossy summary of the kind chapter 2 warns about; the per-element table beside it carries the
information.

Outputs ``docs/evidence/h01-topographic/stage-15.{json,md,png}``.
"""

import json

import numpy as np

from h01_topographic_chain import apply_chain, soma_trace
from h01_topographic_donor import climb, train_rows
from h01_topographic_stage0 import OUT, resample
from h01_topographic_threshold import E_NWB

DOSES = {"0.068": "h01-e-human-sodium/s15-natg-0068-sweep53",
         "0.136": "h01-e-human-sodium/s15-natg-0136-sweep53",
         "0.272": "h01-e-human-sodium/s15-natg-0272-sweep53",
         "0.544": "h01-e-human-sodium/s15-natg-0544-sweep53"}
B3_RUN = "h01-e-currents/m0-b3-currents-sweep53"
RECORDED_SWEEP = 53
RISE_FRACTION = .15                      # registered band around the recorded rise
COUNT_BAND = (5, 15)                     # registered: the rise must not be bought with the count
LEVEL_SCALE_MV = 100.                    # stated scale for voltage levels (peak, threshold, take-off, rest)
ARM_B = {"status": "no reading",
         "cause": "the published HL23PYR template cannot construct on the 541563728 morphology: "
                  "biophys_HL23PYR raises 'section in the object was deleted' at distribute_channels, "
                  "with the full reconstructed axon and again with the axon removed. Editing the "
                  "published template would break the split's unchanged condition, so the arm is "
                  "closed as a no-reading. Three of four evaluations spent; the fourth is unspent.",
         "registered_cells": {"e": "step (spike 2 - spike 1) against the recorded +1.38 mV",
                              "f": "accumulation (spike 5 - spike 1) against the recorded +3.63 mV"}}


def read(base, chain):
    """Spike rows of one run through the recording's chain."""
    tr = soma_trace(base)
    if tr is None:
        return {"status": "not run"}
    t, v = tr
    rows = train_rows(t, apply_chain(t, v, chain["tau_p_us"], chain["fc_khz"]))
    return {"status": "read" if rows else "no spike", "count": len(rows), "spikes": rows,
            "rest_mv": float(np.median(v[(t > 950.) & (t < 1020.)]))}


def recorded(chain):
    """The recorded sweep 53, already at the pipette (the chain is not applied twice)."""
    from h01_isolation_report import nwb_sweep
    t, v = nwb_sweep(E_NWB, RECORDED_SWEEP)
    t, v, _ = resample(np.asarray(t, float), np.asarray(v, float), np.zeros(len(t)))
    rows = train_rows(t, v)
    return {"status": "read", "count": len(rows), "spikes": rows,
            "rest_mv": float(np.median(v[(t > 950.) & (t < 1020.)]))}


def curve(runs):
    """The characteristic curve: spike-1 rise against the inserted human sodium density."""
    points = [{"dose_s_cm2": float(d), "rise_v_s": r["spikes"][0]["rise_v_s"], "count": r["count"]}
              for d, r in runs.items() if r.get("status") == "read"]
    points.sort(key=lambda p: p["dose_s_cm2"])
    return points


def crossing(points, target):
    """Dose at which the curve crosses the recorded rise, by linear interpolation; None if outside."""
    for a, b in zip(points, points[1:]):
        lo, hi = sorted((a["rise_v_s"], b["rise_v_s"]))
        if lo <= target <= hi and a["rise_v_s"] != b["rise_v_s"]:
            f = (target-a["rise_v_s"])/(b["rise_v_s"]-a["rise_v_s"])
            return {"dose_s_cm2": a["dose_s_cm2"]+f*(b["dose_s_cm2"]-a["dose_s_cm2"]), "inside_series": True}
    return {"dose_s_cm2": None, "inside_series": False}


def elements(run, rec):
    """The ten registered elements of one run beside the recording, at 310 pA."""
    if run.get("status") != "read" or not run["spikes"]:
        return None
    s, rs = run["spikes"][0], rec["spikes"][0]
    c, rc = climb(run["spikes"]), climb(rec["spikes"])
    out = {"rise_v_s": (s["rise_v_s"], rs["rise_v_s"], abs(rs["rise_v_s"])),
           "fall_v_s": (s["fall_v_s"], rs["fall_v_s"], abs(rs["fall_v_s"])),
           "peak_mv": (s["peak_mv"], rs["peak_mv"], LEVEL_SCALE_MV),
           "threshold_mv": (s["threshold_mv"], rs["threshold_mv"], LEVEL_SCALE_MV),
           "take_off_mv": (s["take_off_mv"], rs["take_off_mv"], LEVEL_SCALE_MV),
           "count": (float(run["count"]), float(rec["count"]), abs(float(rec["count"]))),
           "rest_mv": (run["rest_mv"], rec["rest_mv"], LEVEL_SCALE_MV)}
    if c and rc:
        out["climb_2_mv"] = (c["climb_2_mv"], rc["climb_2_mv"], abs(rc["climb_2_mv"]))
        out["climb_5_mv"] = (c["climb_5_mv"], rc["climb_5_mv"], abs(rc["climb_5_mv"]))
        out["rise_5_over_1"] = (c["rise_5_over_1"], rc["rise_5_over_1"], abs(rc["rise_5_over_1"]))
    return out


def accuracy(run, rec):
    """Percent accuracy per element and the unweighted mean, on the definition registered in the spec."""
    el = elements(run, rec)
    if el is None:
        return None
    per = {}
    for name, (model, ref, scale) in el.items():
        if not (np.isfinite(model) and np.isfinite(ref)) or not scale:
            continue
        per[name] = {"model": float(model), "recorded": float(ref),
                     "accuracy_pct": float(max(0., 1.-abs(model-ref)/scale)*100.)}
    mean = float(np.mean([p["accuracy_pct"] for p in per.values()])) if per else float("nan")
    return {"elements": per, "mean_pct": mean, "n_elements": len(per)}


def decide(result):
    checks = []

    def add(reading, prediction, value, passed, note=""):
        checks.append({"reading": reading, "prediction": prediction, "value": value, "pass": passed, "note": note})
    pts = result["curve"]
    if len(pts) < 2:
        add("a", "characteristic curve of the rise against the human sodium density", None, None, "fewer than two doses read")
        return {"verdict": "no reading", "checks": checks}
    rises = [p["rise_v_s"] for p in pts]
    monotone = all(b >= a-1e-9 for a, b in zip(rises, rises[1:]))
    add("a", "the rise rises monotonically with the inserted human sodium density (the dose acts through the somatic sodium)",
        {p["dose_s_cm2"]: round(p["rise_v_s"], 1) for p in pts}, monotone,
        f"crossing of the recorded 348 V/s: {result['crossing']}")
    target = result["recorded_reference"]["rise_v_s"]
    inband = [p for p in pts if abs(p["rise_v_s"]-target) < RISE_FRACTION*target]
    holds = [p for p in inband if COUNT_BAND[0] <= p["count"] <= COUNT_BAND[1]]
    add("b", "some dose puts the rise within 15 percent of the recorded 348 V/s with the 310 pA count still within 5 to 15",
        {"in_rise_band": [p["dose_s_cm2"] for p in inband], "and_count_holds": [p["dose_s_cm2"] for p in holds]},
        bool(holds), "the rise must not be bought with the count")
    b3 = result["accuracy"]["b3"]
    best = result["accuracy"].get("best_human")
    add("c", "the human sodium equations reach the recorded rise where the rodent equations reach it only by removing nearly all somatic sodium (stage-11 intercept ~320 V/s)",
        {"b3_rodent_rise_v_s": round(result["b3"]["spikes"][0]["rise_v_s"], 1) if result["b3"].get("status") == "read" else None,
         "best_human_dose": result.get("best_dose")}, None, "reported beside the stage-11 rodent series 655/586/505")
    add("d", "percent accuracy over the ten registered elements improves against B3",
        {"b3_pct": round(b3["mean_pct"], 1) if b3 else None,
         "human_pct": round(best["mean_pct"], 1) if best else None},
        bool(best and b3 and best["mean_pct"] > b3["mean_pct"]), "lossy summary; the per-element table is the reading")
    verdict = "PASS" if (monotone and holds) else "MIXED" if pts else "no reading"
    return {"verdict": verdict, "checks": checks}


def markdown(result):
    d = result["decision"]
    lines = ["# Stage 15: the human-for-rodent sodium swap on the human cell", "",
             "Arm A: B3's somatic rodent NaTs (gbar 2.641 S/cm2) zeroed and the Toronto human NaTg inserted on the",
             "soma with HL23PYR's human voltage shifts (vshiftm 13, vshifth 15, slopem 7), dosed at 310 pA.",
             "Everything else held: the human anatomy of Allen 541563728, its human-fitted densities, the recorded",
             "command, nseg 9, CVode 1e-10, 34 C. All fast quantities through the stage-12 chain.", "",
             f"Verdict: **{d['verdict']}**", "",
             "## The characteristic curve (spike-1 rise against the inserted human sodium density)", "",
             "| NaTg soma (S/cm2) | rise through chain (V/s) | count at 310 pA |", "|---|---|---|"]
    for p in result["curve"]:
        lines.append(f"| {p['dose_s_cm2']:.3f} | {p['rise_v_s']:.0f} | {p['count']} |")
    lines += ["", f"Recorded 348 V/s; B3 with the rodent equations 570 V/s at gbar 2.641, count 10.",
              f"Crossing of the recorded rise: {result['crossing']}.", "",
              "## Percent accuracy over the ten registered elements at 310 pA", "",
              "| element | recorded | B3 (rodent eq.) | best human eq. | B3 % | human % |", "|---|---|---|---|---|---|"]
    b3, best = result["accuracy"]["b3"], result["accuracy"].get("best_human")
    if b3:
        for name, p in b3["elements"].items():
            hp = (best or {}).get("elements", {}).get(name)
            lines.append(f"| {name} | {p['recorded']:.2f} | {p['model']:.2f} | "
                         f"{hp['model']:.2f} | {p['accuracy_pct']:.0f} | {hp['accuracy_pct']:.0f} |"
                         if hp else f"| {name} | {p['recorded']:.2f} | {p['model']:.2f} | - | {p['accuracy_pct']:.0f} | - |")
        lines.append(f"| **mean** | | | | **{b3['mean_pct']:.1f}** | "
                     f"**{best['mean_pct']:.1f}** |" if best else f"| **mean** | | | | **{b3['mean_pct']:.1f}** | - |")
    lines += ["", f"Best human-sodium dose: {result.get('best_dose')} S/cm2.", "",
              "Voltage levels (peak, threshold, take-off, rest) are scored against a stated 100 mV span, so they",
              "score high by construction and lift both means equally; the comparison between the two columns is",
              "the reading, not the absolute figure.", "",
              "## Arm B (the climb on this cell's anatomy): no reading", "",
              ARM_B["cause"], "", "## Cells", ""]
    for c in d["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "--"}[c["pass"]]
        lines.append(f"- ({c['reading']}) {mark}: {c['prediction']} -> {c['value']}. {c['note']}")
    lines += ["", "## Reading", ""]+[f"- {r}" for r in result["reading"]]
    return "\n".join(lines)+"\n"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    pts = result["curve"]
    ax1.plot([p["dose_s_cm2"] for p in pts], [p["rise_v_s"] for p in pts], "o-", color="C0", label="human NaTg on the human cell")
    ax1.axhline(result["recorded_reference"]["rise_v_s"], color="k", ls="--", lw=1, label="recording 348 V/s")
    if result["b3"].get("status") == "read":
        ax1.axhline(result["b3"]["spikes"][0]["rise_v_s"], color="C3", ls=":", lw=1.2, label="B3 rodent NaTs 570 V/s")
    ax1.set(xlabel="inserted somatic NaTg (S/cm2)", ylabel="spike-1 rise through the chain (V/s)",
            title="characteristic curve of the rise")
    ax1.legend(fontsize=7)
    b3, best = result["accuracy"]["b3"], result["accuracy"].get("best_human")
    if b3 and best:
        names = list(b3["elements"])
        x = np.arange(len(names))
        ax2.bar(x-.2, [b3["elements"][n]["accuracy_pct"] for n in names], .4, label=f"B3 rodent ({b3['mean_pct']:.0f}%)", color="C3")
        ax2.bar(x+.2, [best["elements"][n]["accuracy_pct"] for n in names], .4, label=f"human eq. ({best['mean_pct']:.0f}%)", color="C0")
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
    runs = {d: read(b, chain) for d, b in DOSES.items()}
    b3 = read(B3_RUN, chain)
    pts = curve(runs)
    target = rec["spikes"][0]["rise_v_s"]
    acc = {"b3": accuracy(b3, rec)}
    best_dose, best = None, None
    for d, r in runs.items():
        a = accuracy(r, rec)
        if a and (best is None or a["mean_pct"] > best["mean_pct"]):
            best, best_dose = a, d
    acc["best_human"] = best
    acc["per_dose"] = {d: accuracy(r, rec) for d, r in runs.items()}
    result = {"stage": "15", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "chain": chain,
              "recorded_reference": {"rise_v_s": target, "count": rec["count"]},
              "runs": runs, "b3": b3, "recorded": rec, "curve": pts,
              "crossing": crossing(pts, target), "accuracy": acc, "best_dose": best_dose,
              "arm_b": ARM_B, "reading": READING}
    result["decision"] = decide(result)
    (OUT/"stage-15.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-15.md").write_text(markdown(result), encoding="utf-8")
    plot(result, OUT/"stage-15.png")
    print("wrote", OUT/"stage-15.json", result["decision"]["verdict"],
          "| B3", None if not acc["b3"] else round(acc["b3"]["mean_pct"], 1),
          "| human", None if not best else round(best["mean_pct"], 1))


if __name__ == "__main__":
    main()
