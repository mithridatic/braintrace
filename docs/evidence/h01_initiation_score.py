"""Score an E run for the Stage A initiation-site prediction.

For the first spike: which site crosses the detection voltage first (axon probe
or soma), whether the soma shows a shoulder (a local dV/dt maximum inside the
registered voltage and slope bands before the main rise), and the landmark rows.
The usable-tier cycle table is appended so the same run feeds Stage B.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_pv_human_datums import spike_datums
from h01_spike_cycle_energetics import DETECTION_MV, landmarks, slope_v_s
from h01_usable_tier import cycle_table, qc_view

SHOULDER_MV = (-55., -45.)
SHOULDER_V_S = (100., 300.)
PROFILE_MV = (-52., -50., -47.5, -45., -42., -35.)
PULSE_MS = (1020., 2020.)
EVIDENCE = Path(__file__).resolve().parent


def first_crossing_ms(time, voltage, start_ms, level=DETECTION_MV):
    """Interpolated time of the first upward crossing of ``level`` after ``start_ms``, or None."""
    events = [e for e in spike_datums(np.asarray(time, float), np.asarray(voltage, float), level)
              if e["rise_crossing_ms"] >= start_ms]
    return events[0]["rise_crossing_ms"] if events else None


def shoulder(time, voltage, threshold_ms, peak_ms):
    """Local dV/dt maximum inside the shoulder bands between threshold and peak, or None."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    slope = slope_v_s(time, voltage)
    window = np.flatnonzero((time >= threshold_ms) & (time <= peak_ms))
    if len(window) < 3:
        return None
    s, v, t = slope[window], voltage[window], time[window]
    local = np.flatnonzero((s[1:-1] >= s[:-2]) & (s[1:-1] > s[2:]))+1
    hits = [i for i in local if SHOULDER_MV[0] <= v[i] <= SHOULDER_MV[1] and SHOULDER_V_S[0] <= s[i] <= SHOULDER_V_S[1]]
    if not hits:
        return None
    i = hits[0]
    return {"time_ms": float(t[i]), "voltage_mv": float(v[i]), "slope_v_s": float(s[i])}


def slope_profile(time, voltage, threshold_ms, peak_ms, levels=PROFILE_MV):
    """dV/dt at fixed voltages on the upstroke: a shoulder is a plateau of this profile."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    slope = slope_v_s(time, voltage)
    window = (time >= threshold_ms) & (time <= peak_ms)
    v, s = voltage[window], slope[window]
    order = np.argsort(v)
    return {f"{level:g}": float(np.interp(level, v[order], s[order])) if v.min() <= level <= v.max() else None
            for level in levels}


def score_run(time, soma, axon, pulse_ms=PULSE_MS):
    """Initiation-site rows for the first spike plus the usable-tier view."""
    rows = landmarks(time, soma, pulse_ms)
    if not rows:
        return {"spikes": 0, "note": "no complete spike inside the pulse"}
    first = rows[0]
    soma_ms = first_crossing_ms(time, soma, pulse_ms[0])
    axon_ms = first_crossing_ms(time, axon, pulse_ms[0])
    table = cycle_table(time, soma, pulse_ms, "run")
    return {"spikes": len(rows), "soma_crossing_ms": soma_ms, "axon_crossing_ms": axon_ms,
            "axon_leads_ms": None if axon_ms is None else soma_ms-axon_ms,
            "axon_first": axon_ms is not None and axon_ms < soma_ms,
            "shoulder": shoulder(time, soma, first["threshold_ms"], first["peak_ms"]),
            "slope_profile_v_s": slope_profile(time, soma, first["threshold_ms"], first["peak_ms"]),
            "first_spike": first, "cycles_ms": [r["cycle_ms"] for r in rows],
            "usable_view": qc_view(table, pulse_ms), "table": table}


def load_run(path):
    data = np.load(path)
    return data["time_ms"], data["voltage_mv"], data["axon_voltage_mv"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run stem inside docs/evidence, e.g. h01-e-usable/a-axon-na-sweep50")
    args = parser.parse_args(argv)
    report = score_run(*load_run(EVIDENCE/args.run.with_suffix(".npz")))
    out = EVIDENCE/(str(args.run)+"-initiation.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("table", "usable_view")}))


if __name__ == "__main__":
    main()
