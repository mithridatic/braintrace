"""Unitary I-to-E response of the H01 measured pair, connected against disconnected.

``run`` launches the closing-restored circuit wrapper for each named condition
(sequentially; about 170 s per 30 ms run). ``score`` reads the saved traces and
retains, per condition, the E soma and receptor-site difference (connected minus
disconnected) after the I spike: amplitude, latency to 10 percent, decay time
constant, and the change in E events. Predictions are registered in
``docs/specs/2026-09-07-h01-usable-circuit-plan.md`` (W4).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
CACHE = ROOT/".cache/h01"
PYTHON = ROOT/".cache/validation/Scripts/python.exe"
WRAPPER = ["-m", "docs.evidence.h01_i_source_closing_circuit", "--connectivity", "measured",
           "--solver", "h01_staggered_scan", "--i-current-na", "1", "--i-delay-ms", "8", "--i-pulse-ms", "3"]
LITERATURE = {"inhibitory_weight_us": .0031, "inhibitory_reversal_mv": -75., "inhibitory_tau_ms": 4.18}
ASSUMED = {"inhibitory_weight_us": .02, "inhibitory_reversal_mv": -80., "inhibitory_tau_ms": 5.}


def condition_args(control, hold_na, synapse, duration_ms=40.):
    """Wrapper arguments for one condition: E held by a constant current for the run."""
    args = list(WRAPPER)+["--control", control, "--e-current-na", str(hold_na), "--e-delay-ms", "0",
                          "--e-pulse-ms", str(duration_ms), "--duration-ms", str(duration_ms)]
    for key, value in synapse.items():
        args += ["--"+key.replace("_", "-"), str(value)]
    return args


def run_condition(name, args, cache=CACHE, python=PYTHON):
    """Run one condition into ``cache/pair-<name>``; return the output stem."""
    stem = cache/f"pair-{name}"
    command = [str(python)]+args+["--output", str(stem)]
    subprocess.run(command, check=True, cwd=ROOT, env={**dict(__import__("os").environ), "PYTHONPATH": "."})
    return stem


def event_times(data, role):
    return data["time_ms"][np.asarray(data[role+"_events"]).astype(bool)]


def difference_rows(connected, disconnected, site_key):
    """Amplitude, latency and decay of the connected-minus-disconnected trace after the I spike."""
    time = connected["time_ms"]
    diff = connected[site_key]-disconnected[site_key]
    i_events = event_times(connected, "I")
    if not len(i_events):
        return {"note": "no I spike; nothing delivered"}
    onset = float(i_events[0])
    after = time >= onset
    t, d = time[after], diff[after]
    peak_index = int(np.argmax(np.abs(d)))
    amplitude = float(d[peak_index])
    reach = np.flatnonzero(np.abs(d[:peak_index+1]) >= .1*abs(amplitude))
    latency = float(t[reach[0]]-onset) if len(reach) else None
    tail = d[peak_index:]
    decayed = np.flatnonzero(np.abs(tail) <= abs(amplitude)*np.exp(-1.))
    tau = float(t[peak_index+decayed[0]]-t[peak_index]) if len(decayed) else None
    return {"i_spike_ms": onset, "amplitude_mv": amplitude, "peak_ms": float(t[peak_index]),
            "latency_to_10pct_ms": latency, "decay_to_37pct_ms": tau,
            "pre_spike_max_abs_difference_mv": float(np.max(np.abs(diff[~after]))) if (~after).any() else 0.}


def score_pair(connected, disconnected):
    """Rows for the E soma and the receptor site plus the E event change."""
    return {"soma": difference_rows(connected, disconnected, "E_voltage"),
            "receptor_site": difference_rows(connected, disconnected, "E_incoming_voltage"),
            "e_events_connected_ms": event_times(connected, "E").tolist(),
            "e_events_disconnected_ms": event_times(disconnected, "E").tolist(),
            "e_hold_mv_before_i_spike": float(np.interp(event_times(connected, "I")[0]-.5, connected["time_ms"],
                                                        disconnected["E_voltage"])) if len(event_times(connected, "I")) else None}


def render(report):
    lines = ["# I-to-E pair response (measured contact 8105899)", "",
             "Connected minus disconnected at the E soma and at the receptor site after the first I spike.",
             "", "| Condition | E hold (mV) | Site | Amplitude (mV) | Latency (ms) | Decay to 37 % (ms) | Pre-spike difference (mV) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in report["conditions"].items():
        for site in ("soma", "receptor_site"):
            row = r[site]
            if "note" in row:
                lines.append(f"| {name} | {r['e_hold_mv_before_i_spike']} | {site} | {row['note']} | | | |")
                continue
            lines.append(f"| {name} | {r['e_hold_mv_before_i_spike']:.1f} | {site} | {row['amplitude_mv']:.4f} | "
                         f"{row['latency_to_10pct_ms']} | {row['decay_to_37pct_ms']} | {row['pre_spike_max_abs_difference_mv']:.2e} |")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "score"))
    parser.add_argument("--hold-na", type=float, default=.5)
    parser.add_argument("--conditions", nargs="*", default=["disconnected", "literature", "assumed"])
    args = parser.parse_args(argv)
    if args.mode == "run":
        for name in args.conditions:
            synapse = ASSUMED if name == "assumed" else LITERATURE
            control = "disconnected" if name == "disconnected" else "ei"
            run_condition(f"{name}-hold{args.hold_na:g}", condition_args(control, args.hold_na, synapse))
        return
    disconnected = np.load(CACHE/f"pair-disconnected-hold{args.hold_na:g}.npz")
    report = {"hold_na": args.hold_na, "conditions": {}}
    for name in [c for c in args.conditions if c != "disconnected"]:
        connected = np.load(CACHE/f"pair-{name}-hold{args.hold_na:g}.npz")
        report["conditions"][name] = score_pair(connected, disconnected)
    (EVIDENCE/"h01-ie-pair-response.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (EVIDENCE/"h01-ie-pair-response.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({k: v["soma"] for k, v in report["conditions"].items()}))


if __name__ == "__main__":
    main()
