"""Preserve right-limit current samples at continuous-voltage event boundaries."""

import numpy as np


def right_limit_recording(time, voltage, current):
    """Select final samples at duplicate times without changing sample values.

    Parameters
    ----------
    time, voltage, current : array_like
        Matching one-dimensional raw time, voltage, and current arrays.

    Returns
    -------
    dict
        Derived arrays and their indices in the unchanged raw recording.

    Raises
    ------
    ValueError
        If arrays are malformed, time decreases, voltage jumps at a
        duplicate time, or fewer than two distinct times remain.
    """
    time, voltage, current = map(np.asarray, (time, voltage, current))
    if (time.ndim != 1 or len(time) < 2 or voltage.shape != time.shape or current.shape != time.shape
            or not all(np.isfinite(a).all() for a in (time, voltage, current))):
        raise ValueError("Raw arrays must be matching, finite, and one-dimensional.")
    steps = np.diff(time)
    if np.any(steps < 0):
        raise ValueError("Raw time must not decrease.")
    duplicates = steps == 0
    if np.any(voltage[1:][duplicates] != voltage[:-1][duplicates]):
        raise ValueError("Duplicate-time voltage must be exactly continuous.")
    indices = np.flatnonzero(np.r_[steps > 0, True])
    if len(indices) < 2:
        raise ValueError("At least two distinct times are required.")
    return {"time_ms": time[indices], "voltage_mv": voltage[indices],
            "applied_current_na": current[indices], "raw_sample_index": indices}
