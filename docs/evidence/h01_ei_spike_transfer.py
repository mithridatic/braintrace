"""Direct event comparison with explicit invalid-window and missing-spike gates."""
import numpy as np
from docs.evidence.h01_pv_human_datums import spike_datums


def compare_spike_transfer(reference, actual, *, start_ms=270., stop_ms=300.):
    """Compare every complete event in a fixed shared observation window.

    Parameters
    ----------
    reference, actual : mapping
        Arrays named time_ms and voltage_mv in source voltage coordinates.
    start_ms, stop_ms : float, optional
        Shared window. Both inputs must cover it with no boundary spike.

    Returns
    -------
    dict
        Each event, signed differences, and strict independent acceptance gates.
    """
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
        records.append(spike_datums(times, voltage))
    expected, observed = records
    limits = {"rise_crossing_ms": .1, "peak_sample_voltage_mv": .1, "time_above_threshold_ms": .01}
    same_count = len(expected) == len(observed) and len(expected) > 0
    errors = [{k: a[k]-r[k] for k in limits} for r, a in zip(expected, observed)]
    gates = {k: same_count and all(abs(row[k]) <= limit for row in errors) for k, limit in limits.items()}
    return dict(reference_events=expected, actual_events=observed, signed_errors=errors,
                same_nonzero_event_count=same_count, limits=limits, gates=gates,
                passed=same_count and all(gates.values()), window_ms=[start_ms, stop_ms],
                qualification="Numerical spike transfer only; human and circuit validation remain separate.")
