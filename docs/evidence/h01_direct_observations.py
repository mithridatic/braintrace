"""Select a few H01 cycles without discarding their direct temporal observations."""

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums
from docs.evidence.h01_spike_cycle_energetics import landmarks


def validate_trace(time, voltage):
    """Validate a shared finite, strictly increasing observation clock.

    Parameters
    ----------
    time, voltage : array_like
        Matching one-dimensional samples in ms and mV.

    Returns
    -------
    tuple of numpy.ndarray
        Validated time and voltage arrays.
    """
    t, v = np.asarray(time, float), np.asarray(voltage, float)
    if t.ndim != 1 or t.size < 2 or v.shape != t.shape:
        raise ValueError("Need matching one-dimensional time/voltage arrays with >=2 samples.")
    if not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Time and voltage must be finite; time must strictly increase.")
    return t, v


def window_stats(time, voltage, window):
    """Summarize a covered window while keeping sample and time weighting distinct.

    Parameters
    ----------
    time, voltage : array_like
        Observation clock in ms and response in mV.
    window : tuple of float
        Requested start and end times, both required to be covered.

    Returns
    -------
    dict
        Coverage verdict, time mean, and explicitly labeled sample percentiles.
    """
    t, v = validate_trace(time, voltage)
    lo, hi = map(float, window)
    covered = np.isfinite([lo, hi]).all() and t[0] <= lo < hi <= t[-1]
    mask = (t >= lo) & (t < hi)
    result = {"status": "available" if covered else "unavailable", "requested_ms": [lo, hi],
                  "recorded_ms": [float(t[0]), float(t[-1])], "sample_count": int(mask.sum()),
                  "mean_mv": None, "sample_mean_mv": None, "p10_mv": None, "p50_mv": None,
                  "percentile_basis": "unweighted original samples; not elapsed time"}
    if not covered:
        return result
    inside = (t > lo) & (t < hi)
    grid = np.r_[lo, t[inside], hi]
    values = np.interp(grid, t, v)
    result["mean_mv"] = float(np.trapezoid(values, grid) / (hi-lo))
    if mask.any():
        result.update(sample_mean_mv=float(v[mask].mean()), p10_mv=float(np.percentile(v[mask], 10)),
                      p50_mv=float(np.percentile(v[mask], 50)))
    return result


def first_crossing_ms(time, voltage, start_ms, level=-20.):
    """Return the first complete event's interpolated rise crossing.

    Parameters
    ----------
    time, voltage : array_like
        Clock and voltage samples.
    start_ms, level : float
        Earliest event time and detection voltage in mV.

    Returns
    -------
    float or None
        Crossing time; None when no complete event is present.
    """
    t, v = validate_trace(time, voltage)
    return next((e["rise_crossing_ms"] for e in spike_datums(t, v, level)
                 if e["rise_crossing_ms"] >= start_ms), None)


def select_windows(time, voltage, pulse_ms):
    """Select onset, first/second/last complete recovery, and the terminal tail.

    Parameters
    ----------
    time, voltage : array_like
        Complete original trace, in ms and mV.
    pulse_ms : tuple of float
        Command start and stop on the original clock.

    Returns
    -------
    tuple of list
        Window descriptions and all observed event landmarks. Uncovered windows
        remain explicit; they are never extrapolated.
    """
    t, v = validate_trace(time, voltage)
    lo, hi = map(float, pulse_ms)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Invalid pulse window.")
    # The existing landmark finder needs a recorded interval, not an extrapolated one.
    observed = (max(lo, t[0]), min(hi, t[-1]))
    rows = landmarks(t, v, observed) if observed[0] < observed[1] else []
    events = spike_datums(t, v)
    rises = np.flatnonzero((v[:-1] < -20.) & (v[1:] >= -20.))
    incomplete = [float(t[i+1]) for i in rises if lo <= t[i+1] < hi and not any(
        e["rise_crossing_ms"] <= t[i+1] <= e["fall_crossing_ms"] <= observed[1] for e in events)]
    windows = []
    def add(kind, cycle, start, stop):
        covered = start is not None and stop is not None and t[0] <= start < stop <= t[-1]
        windows.append({"kind": kind, "cycle": cycle, "start_ms": start, "stop_ms": stop,
                            "status": "available" if covered else "unavailable"})
    if not rows:
        add("no_complete_spike_pulse" if incomplete else "no_spike_pulse", None, lo, hi)
        for start in incomplete:
            add("incomplete_spike", None, start, None)
        return windows, rows
    add("onset", 1, lo, rows[0]["threshold_ms"])
    for i in sorted({0, min(1, len(rows)-1), len(rows)-1}):
        add("spike", rows[i]["spike"], rows[i]["threshold_ms"], rows[i]["minimum_ms"])
    recoveries = list(range(len(rows)-1))
    chosen = sorted(set(recoveries[:2] + recoveries[-1:]))
    for i in chosen:
        add("recovery", rows[i]["spike"], rows[i]["minimum_ms"], rows[i+1]["threshold_ms"])
    add("tail", rows[-1]["spike"], rows[-1]["minimum_ms"], hi)
    for start in incomplete:
        add("incomplete_spike", None, start, None)
    return windows, rows


def observe(time, voltage, pulse_ms, currents=None, metadata=None, offsets_ms=(0., 2., 6., 20., 60., 200.)):
    """Retain direct samples and conjugate pairs for a small nested cycle sample.

    Parameters
    ----------
    time, voltage : array_like
        Full original trace in ms and mV.
    pulse_ms : tuple of float
        Command boundary on that clock.
    currents : dict, optional
        Named inward-positive local currents in nA on the same clock.
    metadata : dict, optional
        Provenance, units, and spatial boundary supplied by the adapter.
    offsets_ms : tuple of float
        Fixed times after a window's start; never extrapolated beyond its end.

    Returns
    -------
    tuple
        JSON-safe observation record and arrays retaining every selected sample.
    """
    t, v = validate_trace(time, voltage)
    currents = currents or {}
    for name, values in currents.items():
        a = np.asarray(values)
        if a.shape != t.shape or not np.isfinite(a).all():
            raise ValueError(f"Current {name} must be finite and share the voltage clock.")
    if any(not np.isfinite(x) or x < 0 for x in offsets_ms):
        raise ValueError("Phase offsets must be finite and nonnegative.")
    windows, rows = select_windows(t, v, pulse_ms)
    arrays = {}
    for index, window in enumerate(windows):
        window["array_prefix"] = prefix = f"w{index}"
        window["phase_samples"] = []
        if window["status"] != "available":
            continue
        lo, hi = window["start_ms"], window["stop_ms"]
        # Preserve original interior samples and explicitly interpolate boundary endpoints.
        grid = np.r_[lo, t[(t > lo) & (t < hi)], hi]
        arrays[f"{prefix}_time_ms"] = grid
        arrays[f"{prefix}_voltage_mv"] = np.interp(grid, t, v)
        for name, values in currents.items():
            samples = np.interp(grid, t, values)
            arrays[f"{prefix}_current_{name}_na"] = samples
            charge = np.r_[0., np.cumsum((samples[:-1]+samples[1:])*.5*np.diff(grid))]
            arrays[f"{prefix}_charge_{name}_pc"] = charge
        phases = [(f"+{offset:g}ms", lo+offset) for offset in offsets_ms]
        if window["kind"] == "spike":
            row = rows[window["cycle"]-1]
            phases = [("threshold", lo), ("peak", row["peak_ms"]), ("trough", hi)]
        for phase, at in phases:
            available = at <= hi if window["kind"] == "spike" else at < hi
            window["phase_samples"].append({"offset_ms": at-lo, "time_ms": at,
                "phase": phase,
                "status": "available" if available else "beyond_window",
                "voltage_mv": float(np.interp(at, t, v)) if available else None,
                "currents_na": {k: float(np.interp(at, t, a)) for k, a in currents.items()} if available else {}})
    return {"schema": "h01-direct-observation-v1", "pulse_ms": list(pulse_ms), "metadata": metadata or {},
                "windows": windows, "event_landmarks": rows, "current_evidence": "available" if currents else "unavailable",
                "causal_verdict": "not_established", "human_qualification": "not_assessed",
                "selection": "onset; first, second, last complete recovery; tail; original samples retained"}, arrays
