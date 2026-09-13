"""SP15 stage 11: the rise dosed through the somatic sodium on the x2 density-held geometry,
and what a spike leaves behind, read from the retained traces.

(a) Rise. Three doses of the somatic NaTs density factor on the stage-10 x2 density-held
geometry at 310 pA: 0.9 (the retained ``s10-d2-density-sweep53`` run), 0.7 and 0.5
(``h01-e-rise/``). Per dose: spike-1 rise (``max_rise_v_s`` of the campaign landmarks), the
somatic take-off (10 V/s crossing) with its approach, and the count. The take-off move is
read raw and corrected for the approach difference by this geometry's ramp slope.

(b) What a spike leaves behind. Short squares 27 to 31 at 1260 pA: sweep 27 is unprimed
(five non-spiking sweeps before it), 28 to 31 each follow one spike by seconds; the take-off
of each against the unprimed one at the same approach. And the later-spike residual of the
long-square trains against the spike index (stage-7 tables).

Outputs ``docs/evidence/h01-topographic/stage-11.{json,md}``.
"""

import json

import h5py
import numpy as np

from h01_spike_cycle_energetics import landmarks
from h01_topographic_onset import onset, rate_of_rise
from h01_topographic_stage0 import OUT
from h01_topographic_stage9 import load_points, point_rows, residual_by_interval
from h01_topographic_threshold import E_NWB, E_PULSE, E_SHORT, SIGMA

DOSES = {"0.9": "h01-e-coupling-b/s10-d2-density-sweep53", "0.7": "h01-e-rise/s11-na07-sweep53", "0.5": "h01-e-rise/s11-na05-sweep53"}
SOMA = {"soma": "voltage_mv"}
SLOPE_MV_PER_DECADE = -2.8                       # stage 10, x2 density held, soma take-off per decade of approach
RECORDED = "sweep 53"                            # 310 pA long square of the recording (stage-7 table)
SHORT_PRIOR = 21                                 # the last short square that spiked before sweep 27
LIMITS = {"take_off_move_mv": .5, "count_310": (8, 12), "step_mv": 1.9, "residual_growth_mv": 3*SIGMA["E"]["threshold_mv"]}
STEP_ELEMENT = 1.9


def dose_row(base):
    """Spike-1 rise, somatic take-off with its approach, and the count of one 310 pA run."""
    traces = load_points(base, SOMA)
    if traces is None:
        return {"status": "not run"}
    grid, soma = traces["soma"]
    marks = landmarks(grid, soma, E_PULSE)
    if not marks:
        return {"status": "no spike", "count": 0}
    rows = point_rows(traces, E_PULSE, E_PULSE[0])
    first = rows[0]
    return {"status": "read", "count": len(marks), "rise_v_s": marks[0]["max_rise_v_s"], "take_off_mv": first["points"]["soma"]["take_off_mv"],
            "span_mv": first["points"]["soma"]["span_mv"], "approach_mv_per_ms": first["approach_mv_per_ms"],
            "rise_by_spike_v_s": [m["max_rise_v_s"] for m in marks]}


def take_off_move(row, ref, slope=SLOPE_MV_PER_DECADE):
    """Raw take-off difference from the reference dose, and the same corrected for the approach by the ramp slope."""
    raw = row["take_off_mv"]-ref["take_off_mv"]
    decades = np.log10(row["approach_mv_per_ms"])-np.log10(ref["approach_mv_per_ms"])
    return {"raw_mv": float(raw), "approach_decades": float(decades), "corrected_mv": float(raw-slope*decades)}


def rise_series(recorded_rise):
    """(a) the three doses with their moves from 0.9, and the recorded rise beside them."""
    rows = {dose: dose_row(base) for dose, base in DOSES.items()}
    ref = rows["0.9"]
    for dose, row in rows.items():
        if row["status"] == "read" and ref["status"] == "read" and dose != "0.9":
            row["move"] = take_off_move(row, ref)
    return {"doses": rows, "recorded_rise_v_s": recorded_rise}


def short_square_timing(path=E_NWB, sweeps=E_SHORT, prior=SHORT_PRIOR):
    """Seconds from the previous spike to each short-square spike (sweep start times plus the 1020 ms pulse)."""
    with h5py.File(path, "r") as nwb:
        start = {s: float(nwb[f"acquisition/timeseries/Sweep_{s}"]["starting_time"][()]) for s in (prior,)+tuple(sweeps)}
    order = (prior,)+tuple(sweeps)
    return {s: start[s]-start[p] for p, s in zip(order, order[1:])}


def short_square_rows(tables, timing):
    """(b) primed against unprimed: the short-square take-offs at the same approach, with the seconds since the last spike."""
    rows = []
    for sweep in E_SHORT:
        spike = tables[f"short sweep {sweep}"]["spikes"][0]
        rows.append({"sweep": sweep, "since_last_spike_s": timing[sweep], "approach_mv_per_ms": spike["approach_mv_per_ms"], "take_off_mv": spike["threshold_mv"]})
    unprimed = rows[0]["take_off_mv"]
    for r in rows:
        r["minus_unprimed_mv"] = r["take_off_mv"]-unprimed
    primed = [r["minus_unprimed_mv"] for r in rows[1:]]
    sigma = float(np.std([r["take_off_mv"] for r in rows], ddof=1))
    return {"rows": rows, "sigma_mv": sigma, "primed_median_mv": float(np.median(primed)), "primed_range_mv": [min(primed), max(primed)],
            "primed_seconds_range": [min(r["since_last_spike_s"] for r in rows[1:]), max(r["since_last_spike_s"] for r in rows[1:])]}


def residual_by_spike(threshold_result):
    """(b) later-spike residual by spike index: spike 2, spikes 3 to 5, spikes 6 and later, and the slope per spike."""
    rows = residual_by_interval(threshold_result)
    idx = np.array([r["spike"] for r in rows]); res = np.array([r["residual_mv"] for r in rows])
    bands = {"spike_2": res[idx == 2], "spikes_3_to_5": res[(idx >= 3) & (idx <= 5)], "spikes_6_plus": res[idx >= 6]}
    med = {k: {"n": int(v.size), "median_mv": float(np.median(v))} if v.size else None for k, v in bands.items()}
    growth = med["spikes_6_plus"]["median_mv"]-med["spike_2"]["median_mv"] if med["spikes_6_plus"] and med["spike_2"] else None
    return {"n": int(idx.size), "bands": med, "growth_mv": growth, "slope_mv_per_spike": float(np.polyfit(idx, res, 1)[0]) if idx.size >= 3 else None}


def decide(result):
    checks = []
    def add(reading, prediction, value, passed, note=""):
        checks.append({"reading": reading, "prediction": prediction, "value": None if value is None else float(value), "pass": None if passed is None else bool(passed), "note": note})
    doses = result["rise"]["doses"]
    read = [d for d in DOSES if doses[d]["status"] == "read"]
    if len(read) == 3:
        rises = [doses[d]["rise_v_s"] for d in DOSES]
        add("a", "spike-1 rise falls monotonically over 0.9, 0.7, 0.5", rises[0]-rises[-1], rises[0] > rises[1] > rises[2])
    else:
        add("a", "spike-1 rise falls monotonically over 0.9, 0.7, 0.5", None, None, "not every dose read: " + ", ".join(f"{d} {doses[d]['status']}" for d in DOSES))
    for dose in ("0.7", "0.5"):
        row = doses[dose]
        if row["status"] == "read" and "move" in row:
            add("a", f"take-off at {dose}, corrected for the approach, moves less than 0.5 mV from 0.9", row["move"]["corrected_mv"], abs(row["move"]["corrected_mv"]) < LIMITS["take_off_move_mv"],
                f"raw {row['move']['raw_mv']:+.2f} mV over {row['move']['approach_decades']:+.2f} decades of approach")
            add("a", f"count at {dose} within 8 to 12", row["count"], LIMITS["count_310"][0] <= row["count"] <= LIMITS["count_310"][1])
        else:
            add("a", f"take-off at {dose} moves less than 0.5 mV", None, None, row["status"])
    ss = result["short_squares"]
    three = 3*ss["sigma_mv"]
    add("b", "primed short-square take-offs sit within 3 sigma (0.75 mV, the long-square repeat sigma) of the unprimed one (the step does not outlast seconds)", ss["primed_median_mv"],
        max(abs(v) for v in ss["primed_range_mv"]) < LIMITS["residual_growth_mv"], f"the five short squares' own 3 sigma is {three:.2f} mV; a persisting step would read about {STEP_ELEMENT:+.1f}")
    rb = result["residual_by_spike"]
    if rb["growth_mv"] is not None:
        add("b", "later-spike residual does not grow with the count (spikes 6+ against spike 2 within 3 sigma, 0.75 mV)", rb["growth_mv"], abs(rb["growth_mv"]) < LIMITS["residual_growth_mv"])
    verdict = "no reading" if any(c["pass"] is None for c in checks) else ("PASS" if all(c["pass"] for c in checks) else "FAIL")
    return {"checks": checks, "verdict": verdict}


def markdown(result):
    lines = ["# SP15 stage 11: the rise dose, and what a spike leaves behind", ""]
    lines += ["## (a) Rise against the somatic NaTs density, x2 density-held geometry, 310 pA", "",
              f"Recorded spike-1 rise (sweep 53): {result['rise']['recorded_rise_v_s']:.0f} V/s.", "",
              "| density | status | count | rise V/s | take-off mV | approach mV/ms | span mV | move raw mV | move corrected mV |", "|---|---|---|---|---|---|---|---|---|"]
    for dose, row in result["rise"]["doses"].items():
        if row["status"] != "read":
            lines.append(f"| {dose} | {row['status']} | | | | | | | |")
            continue
        mv = row.get("move")
        lines.append(f"| {dose} | read | {row['count']} | {row['rise_v_s']:.0f} | {row['take_off_mv']:.2f} | {row['approach_mv_per_ms']:.2f} | {row['span_mv']:.1f} | "
                     + (f"{mv['raw_mv']:+.2f} | {mv['corrected_mv']:+.2f} |" if mv else "ref | ref |"))
    ss = result["short_squares"]
    lines += ["", "## (b) Short squares 27 to 31 at 1260 pA: primed against unprimed (retained traces)", "",
              "| sweep | s since last spike | approach mV/ms | take-off mV | minus unprimed mV |", "|---|---|---|---|---|"]
    for r in ss["rows"]:
        lines.append(f"| {r['sweep']} | {r['since_last_spike_s']:.1f} | {r['approach_mv_per_ms']:.2f} | {r['take_off_mv']:.2f} | {r['minus_unprimed_mv']:+.2f} |")
    lines += ["", f"Sigma of the five: {ss['sigma_mv']:.2f} mV; primed median {ss['primed_median_mv']:+.2f} mV at {ss['primed_seconds_range'][0]:.1f} to {ss['primed_seconds_range'][1]:.1f} s.", ""]
    rb = result["residual_by_spike"]
    lines += ["## (b) Later-spike residual by spike index, long-square trains (stage-7 tables)", "",
              "| band | n | median residual mV |", "|---|---|---|"]
    for k, v in rb["bands"].items():
        lines.append(f"| {k} | {v['n']} | {v['median_mv']:+.2f} |" if v else f"| {k} | 0 | |")
    lines += ["", f"Slope {rb['slope_mv_per_spike']:+.3f} mV per spike over {rb['n']} later spikes; spikes 6+ minus spike 2: {rb['growth_mv']:+.2f} mV." if rb["growth_mv"] is not None else "", ""]
    lines += ["## Decision", ""]
    for c in result["decision"]["checks"]:
        status = "no reading" if c["pass"] is None else ("pass" if c["pass"] else "FAIL")
        value = "n/a" if c["value"] is None else f"{c['value']:+.2f}"
        lines.append(f"- ({c['reading']}) {status}: {c['prediction']} ({value}{'; ' + c['note'] if c['note'] else ''})")
    lines += ["", f"Verdict: {result['decision']['verdict']}", "", "## Reading", ""] + [f"- {r}" for r in result["reading"]] + [""]
    return "\n".join(lines)


READING = [
    "(a) On the x2 density-held geometry at 310 pA the spike-1 rise follows the somatic NaTs density monotonically: 655, 586 and 505 V/s "
    "at 0.9, 0.7 and 0.5, about 375 V/s per unit of the factor over the range read, and the somatic take-off holds: -55.46, -55.36 and "
    "-55.32 mV at 0.63, 0.61 and 0.59 mV/ms (corrected for the approach +0.07 and +0.06 mV; limit 0.5). The count is 9 at every dose, "
    "so the somatic density is a rise lever that touches neither the take-off nor the count here. The onset span widens with the dose, "
    "4.1, 5.0 and 5.4 mV, still inside the recorded 3.5 to 5.7. Every registered prediction holds and no rejection fired.",
    "(a) qualified: the slope is read between 0.5 and 0.9 only. The recorded 348 V/s lies 157 V/s below the lowest dose; carried "
    "outside the range at the same slope it would need a factor near 0.1, which is not a reading, only a statement that the somatic "
    "density alone does not reach the recorded rise inside the range read while the span stays in band.",
    "(b) The step does not outlast seconds. Sweep 27 fires unprimed (30.7 s after the last spike, five silent sweeps between); sweeps "
    "28, 29 and 30 each fire 4.2, 4.2 and 2.7 s after one spike and sweep 31 15.8 s after one, all at 7.2 to 7.3 mV/ms; their take-offs "
    "sit -0.16, +0.56, +0.31 and +0.22 mV from the unprimed one, median +0.27, where a persisting step would read +1.9. The yardstick "
    "is the independent long-square repeat sigma, 0.25 mV (3 sigma 0.75): every primed deviation is inside it; the spread of the five "
    "short squares themselves (0.28 mV) only agrees with it and is not the test, since a step present in the primed four would widen it. Inside a train the step is full-sized after one spike (spike 2 residual +1.83 mV, n 6) and does not grow with "
    "the count (spikes 3 to 5 +1.63, spikes 6 and later +2.23; 6+ minus 2 is +0.40 mV, limit 0.75; +0.09 mV per spike over 41). So it "
    "is a per-spike step, present in full after one spike, not recovering measurably within 0.73 s (stage 9) and gone within 2.7 s.",
    "(b) limits: the read does not reach inside the first 2.7 s after a spike, and it assumes the step is as visible at 7 mV/ms as at 0.2 "
    "to 0.75 (it is read here at -62 mV, where the recorded relation has slid 6 mV from the long-square regime). The registered "
    "next-long-square read was not taken: the first spiking long square starts 173 s after sweep 31.",
]


def main():
    threshold = json.loads((OUT/"threshold-approach.json").read_text(encoding="utf-8"))
    tables = threshold["tables"]["E human"]
    result = {"stage": "11", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "limits": LIMITS,
              "rise": rise_series(tables[RECORDED]["spikes"][0]["max_rise_v_s"]),
              "short_squares": short_square_rows(tables, short_square_timing()),
              "residual_by_spike": residual_by_spike(threshold), "reading": READING}
    result["decision"] = decide(result)
    (OUT/"stage-11.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-11.md").write_text(markdown(result), encoding="utf-8")
    print("wrote", OUT/"stage-11.json", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
