"""Direct event comparison with explicit invalid-window and missing-spike gates."""
import numpy as np
from docs.evidence.h01_pv_human_datums import spike_datums

PEAK_KEYS = {"sample": "peak_sample_voltage_mv", "interpolated": "peak_interpolated_voltage_mv"}


def interpolated_peak_voltage(time, voltage, index):
    """Vertex of the parabola through the three samples around a discrete maximum.

    General three-point (Lagrange) form, exact for a parabola on any grid. Falls back to
    the sampled value when the triple is flat or the vertex leaves the three-sample span.
    """
    if index <= 0 or index >= len(time)-1:
        return float(voltage[index])
    t0, t1, t2 = (float(x) for x in time[index-1:index+2])
    v0, v1, v2 = (float(x) for x in voltage[index-1:index+2])
    a = (v0/((t0-t1)*(t0-t2))+v1/((t1-t0)*(t1-t2))+v2/((t2-t0)*(t2-t1)))
    if not a < 0.:
        return v1
    b = (-v0*(t1+t2)/((t0-t1)*(t0-t2))-v1*(t0+t2)/((t1-t0)*(t1-t2))-v2*(t0+t1)/((t2-t0)*(t2-t1)))
    vertex = -b/(2.*a)
    if not t0 <= vertex <= t2:
        return v1
    c = (v0*t1*t2/((t0-t1)*(t0-t2))+v1*t0*t2/((t1-t0)*(t1-t2))+v2*t0*t1/((t2-t0)*(t2-t1)))
    return float(a*vertex*vertex+b*vertex+c)


def _events_with_both_peaks(times, voltage):
    events = spike_datums(times, voltage)
    for event in events:
        index = int(np.flatnonzero(times == event["peak_sample_time_ms"])[0])
        event["peak_interpolated_voltage_mv"] = interpolated_peak_voltage(times, voltage, index)
    return events


def compare_spike_transfer(reference, actual, *, start_ms=270., stop_ms=300., peak_method="sample"):
    """Compare every complete event in a fixed shared observation window.

    Parameters
    ----------
    reference, actual : mapping
        Arrays named time_ms and voltage_mv in source voltage coordinates.
    start_ms, stop_ms : float, optional
        Shared window. Both inputs must cover it with no boundary spike.
    peak_method : {"sample", "interpolated"}, optional
        Which peak enters the gate: the sampled maximum (default, every earlier result)
        or the parabolic vertex through the three samples around it. Both peaks are
        reported on every event and in every signed error either way.

    Returns
    -------
    dict
        Each event, signed differences, and strict independent acceptance gates.
    """
    if peak_method not in PEAK_KEYS:
        raise ValueError("peak_method must be 'sample' or 'interpolated'.")
    if not np.isfinite([start_ms, stop_ms]).all() or start_ms >= stop_ms:
        raise ValueError("Require a finite increasing comparison window.")
    records = []
    for trace in (reference, actual):
        t, v = np.asarray(trace["time_ms"]), np.asarray(trace["voltage_mv"])
        if (t.ndim != 1 or v.shape != t.shape or len(t) < 2
                or not np.isfinite(t).all() or not np.isfinite(v).all()
                or not np.all(np.diff(t) > 0) or t[0] > start_ms or t[-1] < stop_ms):
            raise ValueError("Trace must be finite, ordered, and cover the full comparison window.")
        keep = (t >= start_ms) & (t <= stop_ms)
        times, voltage = t[keep], v[keep]
        if len(times) < 2 or voltage[0] >= -20. or voltage[-1] >= -20.:
            raise ValueError("Window boundary cuts a spike or has too few samples.")
        records.append(_events_with_both_peaks(times, voltage))
    expected, observed = records
    limits = {"rise_crossing_ms": .1, PEAK_KEYS[peak_method]: .1, "time_above_threshold_ms": .01}
    reported = ("rise_crossing_ms", *PEAK_KEYS.values(), "time_above_threshold_ms")
    same_count = len(expected) == len(observed) and len(expected) > 0
    errors = [{k: a[k]-r[k] for k in reported} for r, a in zip(expected, observed)]
    gates = {k: same_count and all(abs(row[k]) <= limit for row in errors) for k, limit in limits.items()}
    return dict(reference_events=expected, actual_events=observed, signed_errors=errors,
                same_nonzero_event_count=same_count, limits=limits, gates=gates, peak_method=peak_method,
                passed=same_count and all(gates.values()), window_ms=[start_ms, stop_ms],
                qualification="Numerical spike transfer only; human and circuit validation remain separate.")
