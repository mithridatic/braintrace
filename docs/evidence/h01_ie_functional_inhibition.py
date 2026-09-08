"""Functional inhibition at the H01 measured I-to-E pair (SP5, stage 1).

Four arms through the closing-restored circuit wrapper: ``disconnected``, the
literature receptor at the ``measured`` site, the same receptor at the E
``soma`` (inferred placement), and the dt-halving partner ``measured-halved``.
E is driven by a registered constant current; I fires a pulse train timed to
land between the control's E spikes. ``score`` retains, per control E cycle,
the cycle length, the spike-time shift and the IPSP at the receptor site and
the soma, derives the decision limit from the halving pair, and writes
``h01-ie-inhibition/decision.json``. Spec:
``docs/specs/2026-09-07-h01-functional-inhibition.md``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import h01_ie_pair_response as pair
except ImportError:  # run as a plain file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import h01_ie_pair_response as pair  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
OUT = EVIDENCE/"h01-ie-inhibition"
SPEC = "docs/specs/2026-09-07-h01-functional-inhibition.md"
E_DRIVE_NA, E_DRIVE_ALTERNATE_NA = .6, .8
I_SPIKE_LATENCY_MS, I_PULSE_MS = 6.78, 3.
DT_MS, WINDOW_MS, BENCHMARK_MS = .005, 300., 100.
PROVISIONAL_START_MS, PROVISIONAL_PERIOD_MS = 20., 25.
DLF_TWO, SOMA_NULL_MV, SOMA_PLACEMENT_MV, IPSP_WINDOW_MS = 2.95, .05, .3, 20.
ARMS = {"disconnected": ("disconnected", "measured", DT_MS), "measured": ("ei", "measured", DT_MS),
        "soma": ("ei", "soma", DT_MS), "measured-halved": ("ei", "measured", DT_MS/2)}
HUMAN_TIERS = {"contract_1mv": "not applicable: no human recording of this pair",
               "usable": "not applicable: the usable tier scores a single-cell train against a human datum"}


def sibling_cache(root=ROOT):
    """``root/.cache`` when it holds the H01 archive, else the sibling ``h01-braincell`` cache.

    The ``h01-pair`` worktree is a sparse checkout without ``.cache``; the archive and the
    validation interpreter live in the sibling worktree.
    """
    local = root/".cache"
    return local if (local/"h01"/"proofread104.zip").exists() else root.parent/"h01-braincell"/".cache"


def provisional_onsets(window_ms=WINDOW_MS):
    """Registered periodic I pulse onsets for the control (E does not depend on them)."""
    return np.arange(PROVISIONAL_START_MS, window_ms-I_PULSE_MS, PROVISIONAL_PERIOD_MS).tolist()


def between_spike_onsets(e_spikes_ms, latency_ms=I_SPIKE_LATENCY_MS, pulse_ms=I_PULSE_MS):
    """One I pulse per control cycle, its spike aimed at the cycle midpoint.

    Parameters
    ----------
    e_spikes_ms : sequence of float
        Control E spike times.
    latency_ms : float, optional
        Pulse-onset-to-I-spike latency (6.78 ms measured in W4).
    pulse_ms : float, optional
        Pulse duration; onsets closer than this to the previous one are dropped.

    Returns
    -------
    list of float
        Ascending onsets, none before 0 ms.
    """
    spikes = np.asarray(e_spikes_ms, dtype=float)
    onsets = []
    for start, end in zip(spikes[:-1], spikes[1:]):
        onset = (start+end)/2.-latency_ms
        if onset >= 0. and (not onsets or onset-onsets[-1] > pulse_ms):
            onsets.append(float(round(onset, 3)))
    return onsets


def arm_args(arm, onsets, duration_ms, *, drive_na=E_DRIVE_NA, cache=None):
    """Wrapper arguments for one arm: literature receptor, registered E drive, I train."""
    control, site, dt = ARMS[arm]
    args = pair.condition_args(control, drive_na, pair.LITERATURE, duration_ms)
    args += ["--dt-ms", str(dt), "--receptor-site", site, "--cache", str((cache or sibling_cache())/"h01")]
    return args+["--i-pulse-onsets-ms"]+[str(o) for o in onsets]


def stem_for(arm, duration_ms, cache=None):
    """Output stem (no dot) under the H01 cache."""
    return (cache or sibling_cache())/"h01"/f"inh-{arm}-{int(round(duration_ms))}ms"


def command_for(arm, onsets, duration_ms, *, drive_na=E_DRIVE_NA, cache=None, python=None):
    """The exact process command for one arm."""
    cache = cache or sibling_cache()
    python = python or cache/"validation"/"Scripts"/"python.exe"
    return [str(python)]+arm_args(arm, onsets, duration_ms, drive_na=drive_na, cache=cache)+["--output", str(stem_for(arm, duration_ms, cache))]


def run_arm(arm, onsets, duration_ms, *, drive_na=E_DRIVE_NA, out=OUT, cache=None, python=None, runner=subprocess.run):
    """Run one arm, append its wall time and events to ``out/run-log.json``, return the record."""
    cache = cache or sibling_cache()
    command = command_for(arm, onsets, duration_ms, drive_na=drive_na, cache=cache, python=python)
    started = time.perf_counter()
    runner(command, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": "."})
    record = {"arm": arm, "duration_ms": duration_ms, "drive_na": drive_na, "seconds": round(time.perf_counter()-started, 1),
              "command": command, "seconds_per_simulated_ms": None}
    record["seconds_per_simulated_ms"] = round(record["seconds"]/duration_ms, 3)
    stem = stem_for(arm, duration_ms, cache)
    if stem.with_suffix(".npz").exists():
        data = np.load(stem.with_suffix(".npz"))
        record.update(e_events_ms=pair.event_times(data, "E").tolist(), i_events_ms=pair.event_times(data, "I").tolist())
    out.mkdir(parents=True, exist_ok=True)
    log = out/"run-log.json"
    entries = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    log.write_text(json.dumps(entries+[record], indent=1), encoding="utf-8")
    return record


def _extremum(time, diff, start, stop):
    inside = (time >= start) & (time < stop)
    if not inside.any():
        return None
    values = diff[inside]
    return float(values[np.argmax(np.abs(values))])


def cycle_rows(connected, disconnected):
    """Per control E cycle: length, connected cycle, spike shift, I spikes, site and soma IPSP.

    The IPSP window runs from the first I spike in the cycle to 20 ms later or to 2 ms before
    the earlier of the two arms' next spikes, so a shifted spike does not enter the difference.
    """
    time = connected["time_ms"]
    control, arm = pair.event_times(disconnected, "E"), pair.event_times(connected, "E")
    i_spikes = pair.event_times(connected, "I")
    soma = connected["E_voltage"]-disconnected["E_voltage"]
    site = connected["E_incoming_voltage"]-disconnected["E_incoming_voltage"]
    rows = []
    for k in range(len(control)-1):
        start, end = float(control[k]), float(control[k+1])
        landed = i_spikes[(i_spikes >= start) & (i_spikes < end)]
        row = {"cycle": k, "start_ms": start, "cycle_ms": end-start, "i_spikes_ms": landed.tolist(),
               "shift_ms": float(arm[k]-control[k]) if k < len(arm) else None,
               "connected_cycle_ms": float(arm[k+1]-arm[k]) if k+1 < len(arm) else None,
               "soma_ipsp_mv": None, "site_ipsp_mv": None}
        if len(landed):
            stop = min(end, float(arm[k+1]) if k+1 < len(arm) else end)-2.
            stop = min(stop, float(landed[0])+IPSP_WINDOW_MS)
            row["soma_ipsp_mv"], row["site_ipsp_mv"] = (_extremum(time, d, float(landed[0]), stop) for d in (soma, site))
        rows.append(row)
    return rows


def _largest(values):
    values = [v for v in values if v is not None]
    return max(values, key=abs) if values else None


def summarise(rows, connected, disconnected):
    """Arm summary: counts, largest signed IPSPs, largest shift and lengthening."""
    return {"e_count_connected": int(len(pair.event_times(connected, "E"))),
            "e_count_disconnected": int(len(pair.event_times(disconnected, "E"))),
            "i_count": int(len(pair.event_times(connected, "I"))),
            "soma_ipsp_mv": _largest([r["soma_ipsp_mv"] for r in rows]),
            "site_ipsp_mv": _largest([r["site_ipsp_mv"] for r in rows]),
            "max_shift_ms": _largest([r["shift_ms"] for r in rows]),
            "max_lengthening_ms": _largest([r["connected_cycle_ms"]-r["cycle_ms"] for r in rows
                                            if r["connected_cycle_ms"] is not None]),
            "cycles": rows}


def halving_limits(rows, halved_rows, dt_ms=DT_MS, dlf=DLF_TWO):
    """Decision limits from the dt-halving pair: mean per-cycle range x DLF, floored at one step."""
    if len(rows) != len(halved_rows) or any((a["shift_ms"] is None) != (b["shift_ms"] is None)
                                            or (a["connected_cycle_ms"] is None) != (b["connected_cycle_ms"] is None)
                                            for a, b in zip(rows, halved_rows)):
        return {"resolved": False, "reason": "halving pair differs in E spike count", "shift_ms": None, "cycle_ms": None}
    limits = {"resolved": True, "dlf": dlf, "floor_ms": dt_ms}
    for key, name in (("shift_ms", "shift_ms"), ("connected_cycle_ms", "cycle_ms")):
        ranges = [abs(a[key]-b[key]) for a, b in zip(rows, halved_rows) if a[key] is not None]
        limits[name] = max(float(np.mean(ranges))*dlf, dt_ms) if ranges else dt_ms
        limits[name+"_ranges"] = ranges
    return limits


def beyond_limit(summary, limits):
    """Which cycles exceed the limits; a spike-count change is beyond any limit."""
    count_change = summary["e_count_connected"] != summary["e_count_disconnected"]
    if not limits["resolved"]:
        return {"functional_inhibition": None, "spikes_beyond_limit": [], "cycles_beyond_limit": [], "count_change": count_change}
    spikes = [r["cycle"] for r in summary["cycles"] if r["shift_ms"] is not None and abs(r["shift_ms"]) > limits["shift_ms"]]
    cycles = [r["cycle"] for r in summary["cycles"] if r["connected_cycle_ms"] is not None
              and r["connected_cycle_ms"]-r["cycle_ms"] > limits["cycle_ms"]]
    return {"functional_inhibition": bool(spikes or cycles or count_change), "spikes_beyond_limit": spikes,
            "cycles_beyond_limit": cycles, "count_change": count_change}


def _below(value, bound):
    return None if value is None else abs(value) < bound


def decide(summaries, limits):
    """Score the registered predictions and the placement hypothesis from the arm summaries."""
    verdicts = {arm: beyond_limit(s, limits) for arm, s in summaries.items()}
    measured, soma = summaries["measured"], summaries["soma"]
    delayed = [c for c in verdicts["soma"]["spikes_beyond_limit"]
               if soma["cycles"][c]["shift_ms"] is not None and soma["cycles"][c]["shift_ms"] > 0]
    predictions = {
        "measured_soma_ipsp_below_0.05_mv": _below(measured["soma_ipsp_mv"], SOMA_NULL_MV),
        "measured_no_spike_or_cycle_beyond_limit": None if limits["resolved"] is False else not verdicts["measured"]["functional_inhibition"],
        "soma_ipsp_at_least_0.3_mv": None if soma["soma_ipsp_mv"] is None else abs(soma["soma_ipsp_mv"]) >= SOMA_PLACEMENT_MV,
        "soma_spike_delayed_beyond_limit": None if not limits["resolved"] else bool(delayed)}
    rejected = _below(soma["soma_ipsp_mv"], SOMA_NULL_MV)
    placement = ("not decided" if rejected is None else "rejected: the soma-placed receptor also moves the soma below 0.05 mV"
                 if rejected else "not rejected: placement changes the soma response")
    return {"predictions": predictions, "verdicts": verdicts, "placement_hypothesis": placement,
            "functional_inhibition": {arm: v["functional_inhibition"] for arm, v in verdicts.items()}}


def score(loader, out=OUT, date="2026-09-07"):
    """Build and write ``decision.json`` from the saved arm traces via ``loader(arm)``."""
    disconnected = loader("disconnected")
    summaries, rows = {}, {}
    for arm in ("measured", "soma", "measured-halved"):
        connected = loader(arm)
        rows[arm] = cycle_rows(connected, disconnected)
        summaries[arm] = summarise(rows[arm], connected, disconnected)
    limits = halving_limits(rows["measured"], rows["measured-halved"])
    decision = {"date": date, "spec": SPEC, "stage": 1, "gate": "functional inhibition: one E spike shifted or one cycle "
                "lengthened beyond the DLF limit from the halving pair, or an E spike count change",
                "e_drive_na": E_DRIVE_NA, "limits": limits, "arms": summaries, **decide(summaries, limits),
                "receptor_placement": {"measured": "measured endpoint, cable_location [2805, 0.93]",
                                       "soma": "inferred: explicit perisomatic hypothesis"},
                "human_tiers": HUMAN_TIERS,
                "qualification": "Diagnostic on unpromoted profiles through the closing-restored I wrapper; not human qualified."}
    out.mkdir(parents=True, exist_ok=True)
    (out/"decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    (EVIDENCE/"h01-ie-inhibition-result.md").write_text(render(decision), encoding="utf-8")
    return decision


def _fmt(value, digits=3):
    return "none" if value is None else f"{value:.{digits}f}"


def render(decision):
    """Result page: per-arm summary table, per-cycle rows, predictions and placement verdict."""
    lines = ["# Functional inhibition at the measured I-to-E pair (SP5 stage 1)", "",
             f"Decision JSON: `h01-ie-inhibition/decision.json` ({decision['date']}). E drive {decision['e_drive_na']} nA "
             "(registered). Receptor placement `soma` is an inferred perisomatic hypothesis, not the measured endpoint.",
             "", "| Arm | E spikes (arm / control) | I spikes | Soma IPSP (mV) | Site IPSP (mV) | Largest shift (ms) | "
             "Largest lengthening (ms) | Functional inhibition |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, s in decision["arms"].items():
        lines.append(f"| {arm} | {s['e_count_connected']} / {s['e_count_disconnected']} | {s['i_count']} | {_fmt(s['soma_ipsp_mv'], 4)} | "
                     f"{_fmt(s['site_ipsp_mv'])} | {_fmt(s['max_shift_ms'])} | {_fmt(s['max_lengthening_ms'])} | "
                     f"{decision['functional_inhibition'][arm]} |")
    limits = decision["limits"]
    lines += ["", f"Limits (halving pair, DLF {DLF_TWO}): shift {_fmt(limits.get('shift_ms'))} ms, cycle {_fmt(limits.get('cycle_ms'))} ms; "
              f"resolved {limits['resolved']}.", "", "| Prediction | Held |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in decision["predictions"].items()]
    lines += ["", f"Placement hypothesis: {decision['placement_hypothesis']}.", "",
              "Human tiers: "+"; ".join(f"{k} {v}" for k, v in decision["human_tiers"].items())+".", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "schedule", "run", "score"))
    parser.add_argument("--arm", choices=tuple(ARMS), default="disconnected")
    parser.add_argument("--window-ms", type=float, default=WINDOW_MS)
    parser.add_argument("--drive-na", type=float, default=E_DRIVE_NA)
    args = parser.parse_args(argv)
    cache = sibling_cache()
    schedule = OUT/"schedule.json"
    if args.mode == "benchmark":
        print(json.dumps(run_arm("disconnected", provisional_onsets(BENCHMARK_MS), BENCHMARK_MS, drive_na=args.drive_na)))
    elif args.mode == "schedule":
        data = np.load(stem_for("disconnected", args.window_ms, cache).with_suffix(".npz"))
        onsets = between_spike_onsets(pair.event_times(data, "E"))
        OUT.mkdir(parents=True, exist_ok=True)
        schedule.write_text(json.dumps({"window_ms": args.window_ms, "control_e_spikes_ms": pair.event_times(data, "E").tolist(),
                                        "i_pulse_onsets_ms": onsets}, indent=1), encoding="utf-8")
        print(json.dumps(onsets))
    elif args.mode == "run":
        onsets = (provisional_onsets(args.window_ms) if args.arm == "disconnected"
                  else json.loads(schedule.read_text(encoding="utf-8"))["i_pulse_onsets_ms"])
        print(json.dumps(run_arm(args.arm, onsets, args.window_ms, drive_na=args.drive_na)))
    else:
        decision = score(lambda arm: np.load(stem_for(arm, args.window_ms, cache).with_suffix(".npz")), out=OUT)
        print(json.dumps({"functional_inhibition": decision["functional_inhibition"], "predictions": decision["predictions"]}))


if __name__ == "__main__":
    main()
