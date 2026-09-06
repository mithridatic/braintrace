"""Per-cycle phase-plane landmarks and charge budgets for H01 cell traces.

The characteristic is the conjugate pair of the electrical domain per spike cycle
(Hartshorne 2020, Z strategy): membrane voltage as effort, membrane current as flow,
charge as displacement. A cycle runs from one post-spike minimum to the next. From
voltage alone the phase-plane loop ``(V, dV/dt)`` gives the elemental landmarks;
when the trace carries currents, the charge through each boundary per cycle closes
against the capacitive charge, and the closure residual is the numerical floor.
"""

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums

THRESHOLD_SLOPE_V_S = 10.
DETECTION_MV = -20.
DENSITY_TO_NA = .01


def slope_v_s(time, voltage):
    """Central-difference dV/dt in V/s (mV/ms) on a non-uniform grid."""
    return np.gradient(np.asarray(voltage, float), np.asarray(time, float))


def _minimum(time, voltage, start_ms, stop_ms):
    mask = (time >= start_ms) & (time <= stop_ms)
    index = np.flatnonzero(mask)[int(np.argmin(voltage[mask]))]
    return float(time[index]), float(voltage[index])


def _threshold(time, voltage, slope, peak_ms, floor_ms):
    """Last upward crossing of the slope criterion before the peak."""
    mask = (time >= floor_ms) & (time <= peak_ms)
    rise = np.flatnonzero(mask)[int(np.argmax(slope[mask]))]
    below = np.flatnonzero(mask & (slope < THRESHOLD_SLOPE_V_S) & (np.arange(len(time)) < rise))
    if not len(below):
        return None, None
    index = below[-1]+1
    return float(time[index]), float(voltage[index])


def landmarks(time, voltage, pulse_ms):
    """Phase-plane landmarks of every complete spike inside the pulse window.

    Parameters
    ----------
    time, voltage : array_like
        Trace in ms and mV.
    pulse_ms : tuple of float
        Pulse start and end; the first cycle opens at the pulse start.

    Returns
    -------
    list of dict
        One row per spike with threshold, peak, slopes, minimum and cycle length.
    """
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    slope = slope_v_s(time, voltage)
    events = [e for e in spike_datums(time, voltage, DETECTION_MV)
              if pulse_ms[0] <= e["rise_crossing_ms"] < pulse_ms[1]]
    rows, previous_min = [], pulse_ms[0]
    for index, event in enumerate(events):
        peak_ms = event["peak_sample_time_ms"]
        next_rise = events[index+1]["rise_crossing_ms"] if index+1 < len(events) else pulse_ms[1]
        if next_rise <= event["fall_crossing_ms"]:
            continue
        minimum_ms, minimum_mv = _minimum(time, voltage, event["fall_crossing_ms"], next_rise)
        threshold_ms, threshold_mv = _threshold(time, voltage, slope, peak_ms, previous_min)
        window = (time >= previous_min) & (time <= minimum_ms)
        rows.append({"spike": index+1, "threshold_ms": threshold_ms, "threshold_mv": threshold_mv,
                     "peak_mv": event["peak_sample_voltage_mv"], "peak_ms": peak_ms,
                     "max_rise_v_s": float(slope[window].max()), "max_fall_v_s": float(slope[window].min()),
                     "minimum_mv": minimum_mv, "minimum_ms": minimum_ms,
                     "cycle_ms": minimum_ms-previous_min, "cycle_start_ms": previous_min})
        previous_min = minimum_ms
    return rows


def phase_plane(time, voltage, start_ms, stop_ms):
    """Voltage and slope samples of one window, for a phase-plane overlay."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    mask = (time >= start_ms) & (time <= stop_ms)
    return voltage[mask], slope_v_s(time, voltage)[mask]


def density_to_na(density_ma_cm2, area_um2, outward_positive=True):
    """Convert a NEURON density current to an inward-positive current in nA."""
    sign = -1. if outward_positive else 1.
    return sign*np.asarray(density_ma_cm2, float)*area_um2*DENSITY_TO_NA


def pv_currents(data, geometry):
    """Inward-positive currents in nA from a PV charge-balance trace.

    Parameters
    ----------
    data : mapping
        Arrays of ``h01_pv_neuron_reference.py --observe-charge-balance``.
    geometry : dict
        ``charge_balance_geometry`` of the matching report.
    """
    area = geometry["area_um2"]
    currents = {"applied": np.asarray(data["total_current_na"], float)}
    for key in data:
        if key.endswith("_current_ma_cm2") and not key.startswith(("ina", "ik", "ica", "axon_")):
            currents[key.replace("_current_ma_cm2", "")] = density_to_na(data[key], area)
    axial = np.zeros_like(currents["applied"])
    for neighbour in geometry["neighbours"]:
        axial += (np.asarray(data[neighbour["voltage_key"]], float)-np.asarray(data["voltage_mv"], float))/neighbour["resistance_mohm"]
    currents["axial"] = axial
    return currents


def l2_currents(data, geometry):
    """Inward-positive currents in nA from an L2 trace with ``--record-soma-currents``.

    Soma density keys are ``soma_<MECH>_ma_cm2`` plus ``soma_icap_ma_cm2``; axial
    current comes from the recorded neighbour voltages and ``charge_balance_geometry``.
    """
    area = geometry["area_um2"]
    currents = {"applied": np.asarray(data["applied_current_na"], float)}
    for key in data:
        if key.startswith("soma_") and key.endswith("_ma_cm2"):
            name = key[len("soma_"):-len("_ma_cm2")]
            currents["capacitive" if name == "icap" else name] = density_to_na(data[key], area)
    axial = np.zeros_like(currents["applied"])
    for neighbour in geometry["neighbours"]:
        axial += (np.asarray(data[neighbour["voltage_key"]], float)-np.asarray(data["voltage_mv"], float))/neighbour["resistance_mohm"]
    currents["axial"] = axial
    return currents


def charge_per_cycle(time, currents, rows):
    """Charge in pC through each boundary over every cycle, with the closure residual.

    Capacitive charge is the displacement; every other boundary is a source or a
    sink. ``residual`` is the sum of all boundaries, which a closed balance leaves
    at the integrator's floor.
    """
    time = np.asarray(time, float)
    budgets = []
    for row in rows:
        mask = (time >= row["cycle_start_ms"]) & (time <= row["minimum_ms"])
        budget = {name: float(np.trapezoid(np.asarray(values, float)[mask], time[mask]))
                  for name, values in currents.items()}
        budget["residual"] = float(sum(budget.values()))
        budget["spike"] = row["spike"]
        budgets.append(budget)
    return budgets


def subthreshold_landmarks(time, voltage, pulse_ms, window_ms=100.):
    """Mean voltages of the late pulse plateau and of the post-pulse return.

    ``plateau_mv`` averages the last ``window_ms`` of the pulse; ``return_mv`` averages
    the second ``window_ms`` after the pulse ends, where the slow return of the human
    trace lives. Spikes are not excluded, so use these on subthreshold windows or
    read them as the mean of the whole trajectory.
    """
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    plateau = (time >= pulse_ms[1]-window_ms) & (time < pulse_ms[1])
    after = (time >= pulse_ms[1]+window_ms) & (time < pulse_ms[1]+2.*window_ms)
    return {"plateau_mv": float(voltage[plateau].mean()), "return_mv": float(voltage[after].mean())}


def currents_at(time, currents, at_ms):
    """Every boundary's current in nA at one instant, by linear interpolation."""
    time = np.asarray(time, float)
    return {name: float(np.interp(at_ms, time, np.asarray(values, float))) for name, values in currents.items()}


def early_late(rows, count=3):
    """Split spike rows into the first and last ``count`` cycles."""
    return {"early": rows[:count], "late": rows[-count:] if len(rows) >= count else []}
