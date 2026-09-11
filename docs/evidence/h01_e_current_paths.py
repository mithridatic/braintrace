"""SP11 E cell: the soma current paths of unchanged B3 between spikes, 200 against 310 pA.

Stage 0 checks that the probed rerun reproduces ``g0-b3`` (counts, peak times within
0.1 ms, troughs within 0.1 mV) and then measures, for every interspike window
(trough to next threshold) and for the post-train tail, the mean of each recorded
soma current in nA (inward positive), the net drift current and each term's share.
The lever rule of ``docs/specs/2026-09-11-h01-e-current-measurement.md`` is applied
to those numbers: a lever must carry at least the 200 pA drift current there and a
smaller share of the balance at 310 pA than at 200 pA.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_initiation_score import first_crossing_ms
from h01_spike_cycle_energetics import l2_currents, landmarks

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


def lever_rule(paths):
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
        verdicts[name] = {"mean_200pa_na": low_means[name], "mean_310pa_late_na": high_means.get(name),
                          "share_200pa": low_shares.get(name), "share_310pa_late": high_shares.get(name),
                          "carries_drift": bool(carries), "selective_for_low_drive": bool(selective),
                          "admissible": bool(carries and selective)}
    return {"drift_200pa_na": drift_low, "levers": verdicts,
            "admissible": [n for n, v in verdicts.items() if v["admissible"]]}


def decide(folder, reference_usable):
    reference = json.loads(reference_usable.read_text(encoding="utf-8"))["tables"]
    bands, paths, summaries = [], {}, {}
    for name, label in INPUTS.items():
        data, report = load_run(folder, f"m0-b3-currents-{name}")
        summaries[label] = cycle_summary(data)
        bands += stage0_bands(label, summaries[label], reference[label]["model"])
        paths[label] = drift_rows(data, report)
    held = all(b["held"] for b in bands)
    rule = lever_rule(paths) if held else {"note": "not applied: stage 0 bands missed"}
    log = json.loads((folder/"campaign-log.json").read_text(encoding="utf-8"))
    return {"stage": "0", "candidate": "m0-b3-currents",
            "candidate_sha256": json.loads((folder/"m0-b3-currents-sweep53.json").read_text(encoding="utf-8"))["candidate_json"]["sha256"],
            "reference": str(reference_usable.relative_to(EVIDENCE)).replace("\\", "/"),
            "verdict": "PASS" if held else "FAIL",
            "decision": "B3 reproduced with soma currents recorded; lever rule applied" if held else "reproduction failed; stop",
            "bands": bands, "bands_held": sum(b["held"] for b in bands), "bands_total": len(bands),
            "summaries": summaries, "current_paths": paths, "lever_rule": rule,
            "runs": [{"name": e["name"], "evaluation_index": e["evaluation_index"],
                      "runs": [{k: r[k] for k in ("input", "seconds", "returncode", "aborted")} for r in e["runs"]]} for e in log]}


def render(decision):
    lines = [f"# SP11 stage 0: B3 current paths ({decision['verdict']})", "", decision["decision"], "",
             "| Input | Band | Predicted | Observed | Held |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {b['input']} | {b['row']} | {b['predicted']} | {b['observed']} | {'yes' if b['held'] else 'NO'} |" for b in decision["bands"]]
    for label, rows in decision["current_paths"].items():
        lines += ["", f"## {label}: interspike mean currents (nA, inward positive)", "",
                  "| Window | ms | dV mV | drift | applied | axial | NaTs | Nap | Im | SK | Ih | pas | K_P | K_T | Kv3_1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            m = r["means_na"]
            cells = [f"{m.get(k, float('nan')):.4f}" for k in ("applied", "axial", "NaTs", "Nap", "Im", "SK", "Ih", "pas", "K_P", "K_T", "Kv3_1")]
            lines.append(f"| {r['window']}{' (tail)' if r['tail'] else ''} | {r['duration_ms']:.1f} | {r['voltage_change_mv']:.2f} | {r['drift_na']:.4f} | "+" | ".join(cells)+" |")
    rule = decision["lever_rule"]
    if "levers" in rule:
        lines += ["", f"## Lever rule (drift at 200 pA {rule['drift_200pa_na']:.4f} nA)", "",
                  "| Lever | mean 200 pA | mean 310 pA late | share 200 | share 310 late | carries drift | selective | admissible |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for name, v in rule["levers"].items():
            lines.append(f"| {name} | {v['mean_200pa_na']:.4f} | {v['mean_310pa_late_na']:.4f} | {v['share_200pa']:.3f} | {v['share_310pa_late']:.3f} | "
                         f"{v['carries_drift']} | {v['selective_for_low_drive']} | {v['admissible']} |")
        lines += ["", f"Admissible levers: {rule['admissible'] or 'none'}"]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=EVIDENCE/"h01-e-currents")
    parser.add_argument("--reference", type=Path, default=EVIDENCE/"h01-e-gain"/"g0-b3-usable.json")
    args = parser.parse_args(argv)
    decision = decide(args.folder, args.reference)
    (args.folder/"stage-0-decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.folder/"stage-0-current-paths.md").write_text(render(decision), encoding="utf-8")
    print(render(decision))


if __name__ == "__main__":
    main()
