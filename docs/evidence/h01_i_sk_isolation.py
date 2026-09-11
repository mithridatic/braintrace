"""SP10 I cell: score the axonal-SK isolation stages against their registered bands.

Stage 0 is the platform transfer of the finalist ``e-kv3-close2`` onto the local
executor: counts, cycle 2 and troughs 1-3 must reproduce the container run within
the registered limits. Stage 1 compares ``s1-sk-axon0`` with ``s0-finalist`` from the
same executor: the late trough-to-threshold delay, the count, the early cycle and
the waveform bands of ``docs/specs/2026-09-11-h01-i-sk-isolation.md``. Nothing here
re-tunes a band after the run.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_initiation_score import first_crossing_ms
from h01_pv_human_datums import spike_datums
from h01_spike_cycle_energetics import DETECTION_MV, landmarks, pv_currents

EVIDENCE = Path(__file__).resolve().parent
PULSE_MS = (270., 1270.)
SAMPLE_AFTER_TROUGH_MS = 6.
CYCLE_LIMIT_MS = 12.8
TRANSFER_MS = .1
TRANSFER_MV = .1
WIDTH_TOL_MS = .006
PEAK_AT_MOST_MV = 22.
BLOCK_MV, BLOCK_MS = -20., 5.
INPUTS = {"019": "0.19 nA", "027": "0.27 nA"}
CONTAINER = {"019": {"count": 14, "cycle2_ms": 34.76, "troughs_mv": (-80.13, -80.54, -80.65)},
             "027": {"count": 37, "cycle2_ms": 16.37, "troughs_mv": (-78.99, -79.30, -79.61)}}


def load_run(folder, stem):
    """Arrays and report of one run stem."""
    data = dict(np.load(folder/f"{stem}.npz"))
    report = json.loads((folder/f"{stem}.json").read_text(encoding="utf-8"))
    return data, report


def cycle_rows(data, report):
    """Per-cycle landmarks with the trough-to-next-threshold delay and the axonal SK state at trough+6 ms."""
    time, soma = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    rows = landmarks(time, soma, PULSE_MS)
    events = [e for e in spike_datums(time, soma, DETECTION_MV) if PULSE_MS[0] <= e["rise_crossing_ms"] < PULSE_MS[1]]
    axial = pv_currents(data, report["charge_balance_geometry"])["axial"]
    out = []
    for index, row in enumerate(rows):
        following = rows[index+1]["threshold_ms"] if index+1 < len(rows) else None
        at = row["minimum_ms"]+SAMPLE_AFTER_TROUGH_MS
        sample = {key: float(np.interp(at, time, np.asarray(data[key], float)))
                  for key in ("axon_sk_current_ma_cm2", "axon_calcium_mm", "axon_sk_gate") if key in data}
        out.append({"cycle": row["spike"], "cycle_ms": row["cycle_ms"], "threshold_ms": row["threshold_ms"],
                    "peak_mv": row["peak_mv"], "trough_ms": row["minimum_ms"], "trough_mv": row["minimum_mv"],
                    "width_ms": events[index]["time_above_threshold_ms"] if index < len(events) else None,
                    "delay_to_next_threshold_ms": None if following is None else following-row["minimum_ms"],
                    "sample_ms": at, "sample_after_next_spike": following is not None and at > following,
                    "axial_na_at_sample": float(np.interp(at, time, axial)), **{k+"_at_sample": v for k, v in sample.items()}})
    return out


def initiation(data):
    """Axon-first check of the first spike under the PV pulse window."""
    time = np.asarray(data["time_ms"], float)
    soma_ms = first_crossing_ms(time, np.asarray(data["voltage_mv"], float), PULSE_MS[0])
    axon_ms = first_crossing_ms(time, np.asarray(data["axon_voltage_mv"], float), PULSE_MS[0])
    return {"soma_crossing_ms": soma_ms, "axon_crossing_ms": axon_ms,
            "axon_leads_ms": None if None in (soma_ms, axon_ms) else soma_ms-axon_ms,
            "axon_first": axon_ms is not None and soma_ms is not None and axon_ms < soma_ms}


def depolarisation_block(data):
    """Longest continuous time above -20 mV inside the pulse, and whether it exceeds the block limit."""
    time, soma = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    mask = (time >= PULSE_MS[0]) & (time < PULSE_MS[1])
    above = soma[mask] >= BLOCK_MV
    longest, start = 0., None
    for t, up in zip(time[mask], above):
        if up and start is None:
            start = t
        elif not up and start is not None:
            longest, start = max(longest, t-start), None
    if start is not None:
        longest = max(longest, time[mask][-1]-start)
    return {"longest_above_ms": float(longest), "blocked": bool(longest > BLOCK_MS)}


def late_delay(rows):
    """The last measured trough-to-threshold delay (the penultimate cycle's)."""
    delays = [r["delay_to_next_threshold_ms"] for r in rows if r["delay_to_next_threshold_ms"] is not None]
    return delays[-1] if delays else None


def _row(name, predicted, observed, held, label):
    return {"row": name, "predicted": predicted, "observed": observed, "held": bool(held), "input": label}


def stage0_bands(name, rows, init):
    """Transfer bands of one input against the container baseline."""
    base, label = CONTAINER[name], INPUTS[name]
    cycle2 = rows[1]["cycle_ms"] if len(rows) > 1 else None
    out = [_row("count", base["count"], len(rows), len(rows) == base["count"], label),
           _row("cycle2_ms", f"{base['cycle2_ms']} +/- {TRANSFER_MS}", cycle2,
                cycle2 is not None and abs(cycle2-base["cycle2_ms"]) <= TRANSFER_MS, label)]
    for cycle, reference in enumerate(base["troughs_mv"], start=1):
        trough = rows[cycle-1]["trough_mv"] if len(rows) >= cycle else None
        out.append(_row(f"trough{cycle}_mv", f"{reference} +/- {TRANSFER_MV}", trough,
                        trough is not None and abs(trough-reference) <= TRANSFER_MV, label))
    out.append(_row("axon_first", True, init["axon_first"], init["axon_first"], label))
    return out


def stage1_bands(name, base_rows, arm_rows, init, block):
    """Single-change bands of ``s1-sk-axon0`` against ``s0-finalist`` at one input."""
    label = INPUTS[name]
    base_late, arm_late = late_delay(base_rows), late_delay(arm_rows)
    shortened = None if None in (base_late, arm_late) else base_late-arm_late
    cycle2 = (base_rows[1]["cycle_ms"], arm_rows[1]["cycle_ms"]) if min(len(base_rows), len(arm_rows)) > 1 else (None, None)
    width1 = arm_rows[0]["width_ms"] if arm_rows else None
    peak = max((r["peak_mv"] for r in arm_rows), default=None)
    sk = arm_rows[-2]["axon_sk_current_ma_cm2_at_sample"] if len(arm_rows) > 1 else None
    out = [_row("axon_sk_current_at_sample", "0 by construction", sk, sk is not None and abs(sk) < 1e-12, label),
           _row("late_delay_shortens_ms", f"> {CYCLE_LIMIT_MS} (from {base_late})", shortened,
                shortened is not None and shortened > CYCLE_LIMIT_MS, label),
           _row("count", f"> {len(base_rows)}", len(arm_rows), len(arm_rows) > len(base_rows), label),
           _row("cycle2_change_ms", f"|change| < {CYCLE_LIMIT_MS} (from {cycle2[0]})",
                None if None in cycle2 else cycle2[1]-cycle2[0],
                None not in cycle2 and abs(cycle2[1]-cycle2[0]) < CYCLE_LIMIT_MS, label),
           _row("width1_ms", f"0.221 +/- {WIDTH_TOL_MS}", width1, width1 is not None and abs(width1-.221) <= WIDTH_TOL_MS, label),
           _row("peak_mv", f"<= {PEAK_AT_MOST_MV}", peak, peak is not None and peak <= PEAK_AT_MOST_MV, label)]
    for cycle in (1, 2, 3):
        base_t = base_rows[cycle-1]["trough_mv"] if len(base_rows) >= cycle else None
        arm_t = arm_rows[cycle-1]["trough_mv"] if len(arm_rows) >= cycle else None
        out.append(_row(f"trough{cycle}_mv", f"within 1 of {base_t}", arm_t,
                        None not in (base_t, arm_t) and abs(arm_t-base_t) <= 1., label))
    out.append(_row("axon_first", True, init["axon_first"], init["axon_first"], label))
    out.append(_row("no_depolarisation_block", f"<= {BLOCK_MS} ms above {BLOCK_MV} mV", block["longest_above_ms"], not block["blocked"], label))
    return out


def stage1_rejection(bands):
    """Registered rejection: late delay not shortened at either input, spikes lost, or block."""
    by = {(b["input"], b["row"]): b for b in bands}
    not_shortened = [label for label in INPUTS.values() if not by[(label, "late_delay_shortens_ms")]["held"]]
    lost = [label for label in INPUTS.values() if by[(label, "count")]["observed"] < int(str(by[(label, "count")]["predicted"]).split()[-1])]
    blocked = [label for label in INPUTS.values() if not by[(label, "no_depolarisation_block")]["held"]]
    return {"delay_not_shortened": not_shortened, "spikes_lost": lost, "depolarisation_block": blocked,
            "met": bool(not_shortened or lost or blocked)}


def decide(stage, folder):
    """Build the decision record of one stage from the run files in ``folder``."""
    runs = {}
    for stem in (("s0-finalist",) if stage == "0" else ("s0-finalist", "s1-sk-axon0")):
        for name in INPUTS:
            data, report = load_run(folder, f"{stem}-{name}")
            runs[(stem, name)] = {"rows": cycle_rows(data, report), "initiation": initiation(data),
                                  "block": depolarisation_block(data), "sha256": report["candidate_json"]["sha256"]}
    bands = []
    for name in INPUTS:
        if stage == "0":
            bands += stage0_bands(name, runs[("s0-finalist", name)]["rows"], runs[("s0-finalist", name)]["initiation"])
        else:
            arm = runs[("s1-sk-axon0", name)]
            bands += stage1_bands(name, runs[("s0-finalist", name)]["rows"], arm["rows"], arm["initiation"], arm["block"])
    reject = stage1_rejection(bands) if stage == "1" else {"met": not all(b["held"] for b in bands)}
    held = all(b["held"] for b in bands)
    log = json.loads((folder/"campaign-log.json").read_text(encoding="utf-8"))
    return {"stage": stage, "candidate": "s0-finalist" if stage == "0" else "s1-sk-axon0",
            "candidate_sha256": runs[("s0-finalist" if stage == "0" else "s1-sk-axon0", "019")]["sha256"],
            "baseline_sha256": runs[("s0-finalist", "019")]["sha256"],
            "verdict": "PASS" if held and not reject["met"] else "FAIL",
            "decision": ("transfer reproduced: every band held" if stage == "0" and held else
                         "transfer failed: a band missed" if stage == "0" else
                         "axonal SK is the direct path of the late delay under the finalist settings: every band held, rejection not met"
                         if held and not reject["met"] else "rejection met or a band missed; see bands"),
            "bands": bands, "bands_held": sum(b["held"] for b in bands), "bands_total": len(bands),
            "rejection": reject,
            "cycles": {f"{stem}-{name}": run["rows"] for (stem, name), run in runs.items()},
            "initiation": {f"{stem}-{name}": run["initiation"] for (stem, name), run in runs.items()},
            "depolarisation_block": {f"{stem}-{name}": run["block"] for (stem, name), run in runs.items()},
            "runs": [{"name": e["name"], "evaluation_index": e["evaluation_index"],
                      "runs": [{k: r[k] for k in ("input", "seconds", "returncode", "aborted")} for r in e["runs"]]} for e in log]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("0", "1"), required=True)
    parser.add_argument("--folder", type=Path, default=EVIDENCE/"h01-i-sk")
    args = parser.parse_args(argv)
    decision = decide(args.stage, args.folder)
    out = args.folder/f"stage-{args.stage}-decision.json"
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({k: decision[k] for k in ("verdict", "decision", "bands_held", "bands_total", "rejection")}, indent=1))
    for band in decision["bands"]:
        print(band["input"], band["row"], "held" if band["held"] else "MISSED", band["predicted"], band["observed"])


if __name__ == "__main__":
    main()
