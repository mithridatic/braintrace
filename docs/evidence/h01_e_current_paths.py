"""SP11 E cell: the soma current paths of unchanged B3 between spikes, 200 against 310 pA.

Stage 0 checks that the probed rerun reproduces ``g0-b3`` (counts, peak times within
0.1 ms, troughs within 0.1 mV) and then measures, for every interspike window
(trough to next threshold) and for the post-train tail, the mean of each recorded
soma current in nA (inward positive), the net drift current and each term's share.
The CLI first saves direct observation and multivari artifacts. Historical current
share comparisons remain descriptive; they do not establish causal admissibility.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from h01_spike_cycle_energetics import l2_currents, landmarks

from docs.evidence.h01_direct_observations import first_crossing_ms, window_stats

EVIDENCE = Path(__file__).resolve().parent
PULSE_MS = (1020., 2020.)
TRANSFER_MS = .1
TRANSFER_MV = .1
INPUTS = {"sweep56": "200 pA", "sweep50": "250 pA", "sweep53": "310 pA"}
LEVERS = ("Nap", "Im", "SK", "Ih", "pas", "K_P", "K_T", "Kv3_1", "NaTs", "Ca_HVA", "Ca_LVA", "axial")


def load_run(folder, stem):
    data = dict(np.load(folder/f"{stem}.npz"))
    report = json.loads((folder/f"{stem}.json").read_text(encoding="utf-8"))
    return data, report


def window_means(time, currents, start, stop):
    """Mean current in nA of every boundary over ``[start, stop]`` (trapezoid over the recorded grid)."""
    if start >= stop or time[0] > start or time[-1] < stop:
        return None
    mask = (time >= start) & (time <= stop)
    if mask.sum() < 2:
        return None
    t = time[mask]
    return {name: float(np.trapezoid(np.asarray(values, float)[mask], t)/(t[-1]-t[0])) for name, values in currents.items()}


def interspike_windows(rows, voltage_time, voltage):
    """Trough-to-next-threshold windows plus the post-train tail to the pulse end."""
    out = []
    for index, row in enumerate(rows):
        stop = rows[index+1]["threshold_ms"] if index+1 < len(rows) else PULSE_MS[1]
        out.append({"window": f"after spike {row['spike']}", "start_ms": row["minimum_ms"], "stop_ms": stop,
                    "tail": index+1 == len(rows),
                    "voltage_change_mv": float(np.interp(stop, voltage_time, voltage)-row["minimum_mv"])})
    return out


def drift_rows(data, report):
    """Per-window mean currents, the drift (capacitive) current and each term's share of the balance."""
    time, soma = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    currents = l2_currents(data, report["charge_balance_geometry"])
    rows = landmarks(time, soma, PULSE_MS)
    out = []
    for window in interspike_windows(rows, time, soma):
        means = window_means(time, currents, window["start_ms"], window["stop_ms"])
        if means is None:
            continue
        total = sum(abs(v) for k, v in means.items() if k not in ("capacitive", "applied"))
        out.append({**window, "duration_ms": window["stop_ms"]-window["start_ms"], "means_na": means,
                    "drift_na": -means.get("capacitive", float("nan")),
                    "shares": {k: abs(v)/total if total else None for k, v in means.items() if k not in ("capacitive", "applied")}})
    return out


def cycle_summary(data):
    time, soma = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    rows = landmarks(time, soma, PULSE_MS)
    axon_ms = first_crossing_ms(time, np.asarray(data["axon_voltage_mv"], float), PULSE_MS[0])
    soma_ms = first_crossing_ms(time, soma, PULSE_MS[0])
    return {"count": len(rows), "peak_ms": [r["peak_ms"] for r in rows], "trough_mv": [r["minimum_mv"] for r in rows],
            "cycle_ms": [r["cycle_ms"] for r in rows],
            "axon_first": axon_ms is not None and soma_ms is not None and axon_ms < soma_ms,
            "axon_leads_ms": None if None in (axon_ms, soma_ms) else soma_ms-axon_ms}


def stage0_bands(label, summary, reference):
    """Reproduction bands against the retained ``g0-b3`` usable-tier cycle table of one input."""
    ref_peaks = [r["peak_ms"] for r in reference]
    ref_troughs = [r["ahp_mv"] for r in reference]
    same = summary["count"] == len(reference)
    peak_dev = max((abs(a-b) for a, b in zip(summary["peak_ms"], ref_peaks)), default=None) if same else None
    trough_dev = max((abs(a-b) for a, b in zip(summary["trough_mv"], ref_troughs)), default=None) if same else None
    return [{"row": "count", "predicted": len(reference), "observed": summary["count"], "held": same, "input": label},
            {"row": "peak_time_max_dev_ms", "predicted": f"<= {TRANSFER_MS}", "observed": peak_dev,
             "held": peak_dev is not None and peak_dev <= TRANSFER_MS, "input": label},
            {"row": "trough_max_dev_mv", "predicted": f"<= {TRANSFER_MV}", "observed": trough_dev,
             "held": trough_dev is not None and trough_dev <= TRANSFER_MV, "input": label},
            {"row": "axon_first", "predicted": True, "observed": summary["axon_first"], "held": summary["axon_first"], "input": label}]


PLATEAU_WINDOWS = {"late_pulse": (1800., 2000.), "mid_pulse": (1300., 1500.), "post_pulse": (2100., 2300.)}


def _window_stats(time, voltage, window):
    return window_stats(time, voltage, window)


def plateau_contrast(data, human_path, applied_na):
    """Human and model voltage statistics in the mid-pulse, late-pulse and post-pulse windows.

    Sample percentiles remain explicitly labeled QC summaries. Neither a partial
    post-pulse window nor a static level ratio establishes incremental resistance
    or the current required for a causal repair; those fields remain unavailable.
    """
    human = np.load(human_path)
    ht, hv = human["time_ms"], human["corrected_voltage_mv"]
    mt, mv = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    rows = {name: {"human": _window_stats(ht, hv, w), "model": _window_stats(mt, mv, w)} for name, w in PLATEAU_WINDOWS.items()}
    late = rows["late_pulse"]
    offset_mv = (late["model"]["p50_mv"]-late["human"]["p50_mv"]
                 if all(x["p50_mv"] is not None for x in late.values()) else None)
    return {"windows": rows, "model_input_resistance_mohm": None,
            "late_pulse_offset_mv": offset_mv, "offset_na": None,
            "note": "A pulse/post-pulse voltage ratio does not establish incremental resistance or a required causal current."}


def lever_rule(paths, offset_na=None):
    """Apply the registered lever rule to the 200 pA and 310 pA interspike balances.

    ``paths`` maps input label to its drift rows. The 200 pA reference window is the
    mean over its interspike windows (tail excluded); the 310 pA reference is the mean
    over its late interspike windows (last three, tail excluded).
    """
    def mean_over(rows, key, take):
        picked = [r for r in rows if not r["tail"]]
        picked = picked[-take:] if take else picked
        if not picked:
            return {}
        names = picked[0][key]
        return {n: float(np.mean([r[key][n] for r in picked])) for n in names}
    low, high = paths["200 pA"], paths["310 pA"]
    low_means, high_means = mean_over(low, "means_na", 0), mean_over(high, "means_na", 3)
    low_shares, high_shares = mean_over(low, "shares", 0), mean_over(high, "shares", 3)
    drift_low = float(np.mean([r["drift_na"] for r in low if not r["tail"]])) if any(not r["tail"] for r in low) else None
    verdicts = {}
    for name in LEVERS:
        if name not in low_means:
            continue
        carries = drift_low is not None and abs(low_means[name]) >= abs(drift_low)
        selective = low_shares.get(name) is not None and high_shares.get(name) is not None and high_shares[name] < low_shares[name]
        carries_offset = None if offset_na is None else bool(abs(low_means[name]) >= abs(offset_na))
        verdicts[name] = {"mean_200pa_na": low_means[name], "mean_310pa_late_na": high_means.get(name),
                          "share_200pa": low_shares.get(name), "share_310pa_late": high_shares.get(name),
                          "carries_drift": bool(carries), "selective_for_low_drive": bool(selective),
                          "carries_plateau_offset": carries_offset,
                          "admissible": None}
    return {"drift_200pa_na": drift_low, "plateau_offset_na": offset_na, "levers": verdicts,
            "admissible": None, "causal_verdict": "not_established",
            "note": "Current means and shares describe local observations. They cannot exclude interventions or establish a required missing current; voltage and states feed back."}


def decide(folder, reference_usable, human_cache=None):
    reference = json.loads(reference_usable.read_text(encoding="utf-8"))["tables"]
    bands, paths, summaries, plateaus = [], {}, {}, {}
    for name, label in INPUTS.items():
        data, report = load_run(folder, f"m0-b3-currents-{name}")
        summaries[label] = cycle_summary(data)
        bands += stage0_bands(label, summaries[label], reference[label]["model"])
        paths[label] = drift_rows(data, report)
        if human_cache is not None:
            plateaus[label] = plateau_contrast(data, human_cache/f"sweep-{name[len('sweep'):]}.npz",
                                               float(np.mean(np.asarray(data["applied_current_na"], float)[(np.asarray(data["time_ms"], float) >= 1800.) & (np.asarray(data["time_ms"], float) < 2000.)])))
    held = all(b["held"] for b in bands)
    offset = plateaus["200 pA"]["offset_na"] if "200 pA" in plateaus else None
    rule = lever_rule(paths, offset) if held else {"note": "not applied: stage 0 bands missed"}
    log = json.loads((folder/"campaign-log.json").read_text(encoding="utf-8"))
    return {"stage": "0", "candidate": "m0-b3-currents",
            "candidate_sha256": json.loads((folder/"m0-b3-currents-sweep53.json").read_text(encoding="utf-8"))["candidate_json"]["sha256"],
            "reference": str(reference_usable.relative_to(EVIDENCE)).replace("\\", "/"),
            "verdict": "PASS" if held else "FAIL",
            "decision": "B3 reproduced; direct observations required; no causal exclusion from current shares" if held else "reproduction failed; stop",
            "bands": bands, "bands_held": sum(b["held"] for b in bands), "bands_total": len(bands),
            "summaries": summaries, "current_paths": paths, "plateau_contrast": plateaus, "lever_rule": rule,
            "runs": [{"name": e["name"], "evaluation_index": e["evaluation_index"],
                      "runs": [{k: r[k] for k in ("input", "seconds", "returncode", "aborted")} for r in e["runs"]]} for e in log]}


def render(decision):
    lines = [f"# SP11 stage 0: B3 current paths ({decision['verdict']})", "", decision["decision"], "",
             "| Input | Band | Predicted | Observed | Held |", "| --- | --- | --- | --- | --- |"]
    direct = decision.get("direct_observation")
    if direct:
        lines[2:2] = [f"[Direct observations and required charts]({Path(direct['markdown']).name})", "",
                      "Inspect those observations first. The tables below are secondary QC summaries, not causal exclusions.", ""]
    lines += [f"| {b['input']} | {b['row']} | {b['predicted']} | {b['observed']} | {'yes' if b['held'] else 'NO'} |" for b in decision["bands"]]
    for label, rows in decision["current_paths"].items():
        lines += ["", f"## {label}: interspike mean currents (nA, inward positive)", "",
                  "| Window | ms | dV mV | drift | applied | axial | NaTs | Nap | Im | SK | Ih | pas | K_P | K_T | Kv3_1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            m = r["means_na"]
            cells = [f"{m.get(k, float('nan')):.4f}" for k in ("applied", "axial", "NaTs", "Nap", "Im", "SK", "Ih", "pas", "K_P", "K_T", "Kv3_1")]
            lines.append(f"| {r['window']}{' (tail)' if r['tail'] else ''} | {r['duration_ms']:.1f} | {r['voltage_change_mv']:.2f} | {r['drift_na']:.4f} | "+" | ".join(cells)+" |")
    if decision.get("plateau_contrast"):
        lines += ["", "## Plateau contrast (p50 mV; human | model)", "",
                  "| Input | mid pulse 1300-1500 | late pulse 1800-2000 | post pulse 2100-2300 | model R_in (MOhm) | late offset (mV) | offset (nA) |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        for label, c in decision["plateau_contrast"].items():
            w = c["windows"]
            cells = [f"{w[k]['human']['p50_mv']} / {w[k]['model']['p50_mv']}" for k in ("mid_pulse", "late_pulse", "post_pulse")]
            lines.append(f"| {label} | "+" | ".join(cells)+f" | unavailable | {c['late_pulse_offset_mv']} | unavailable |")
    rule = decision["lever_rule"]
    if "levers" in rule:
        offset = rule.get("plateau_offset_na")
        lines += ["", f"## Lever rule (drift at 200 pA {rule['drift_200pa_na']:.5f} nA; plateau offset {offset if offset is None else round(offset, 4)} nA)", "",
                  "| Lever | mean 200 pA | mean 310 pA late | share 200 | share 310 late | carries drift | selective | carries offset | admissible |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
        for name, v in rule["levers"].items():
            lines.append(f"| {name} | {v['mean_200pa_na']:.4f} | {v['mean_310pa_late_na']:.4f} | {v['share_200pa']:.3f} | {v['share_310pa_late']:.3f} | "
                         f"{v['carries_drift']} | {v['selective_for_low_drive']} | {v['carries_plateau_offset']} | {v['admissible']} |")
        lines += ["", "Causal admissibility: not established (not an exclusion).", "", rule["note"]]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=EVIDENCE/"h01-e-currents")
    parser.add_argument("--reference", type=Path, default=EVIDENCE/"h01-e-gain"/"g0-b3-usable.json")
    parser.add_argument("--human-cache", type=Path, help="directory holding sweep-<n>.npz human exports (plateau contrast)")
    parser.add_argument("--output-stem", default="stage-0-reanalysis", help="Separate from the historical registered decision.")
    args = parser.parse_args(argv)
    from docs.evidence.h01_direct_report import write_bundle
    direct = write_bundle([args.folder/f"m0-b3-currents-{name}" for name in INPUTS],
                          args.folder/f"{args.output_stem}-direct",
                          [args.human_cache/f"sweep-{name[5:]}.npz" for name in INPUTS] if args.human_cache else [])
    decision = decide(args.folder, args.reference, args.human_cache)
    decision["direct_observation"] = direct
    (args.folder/f"{args.output_stem}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.folder/f"{args.output_stem}.md").write_text(render(decision), encoding="utf-8")
    print(render(decision))


if __name__ == "__main__":
    main()
