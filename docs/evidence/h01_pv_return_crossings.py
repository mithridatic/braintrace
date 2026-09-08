"""Observe first downward voltage crossings after the first spike."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def downward_crossings(time, voltage, start, stop, levels):
    """Return adjacent-sample downward crossings within an explicit interval.

    Parameters
    ----------
    time, voltage : ndarray
        Sample times in ms and voltages in mV.
    start, stop : float
        First peak and next upward crossing times in ms.
    levels : iterable of float
        Voltage levels in mV.

    Returns
    -------
    list of dict
        First crossings, elapsed times, and crossing multiplicities.
    """
    rows = []
    for level in levels:
        hits = np.flatnonzero((voltage[:-1] > level) & (voltage[1:] <= level))
        instants = time[hits] + ((level - voltage[hits]) * (time[hits+1]-time[hits])
                                / (voltage[hits+1]-voltage[hits]))
        instants = instants[(instants >= start) & (instants < stop)]
        instant = float(instants[0]) if len(instants) else None
        rows.append({"level_mv": float(level), "first_time_ms": instant,
                     "time_from_peak_ms": None if instant is None else instant-start,
                     "crossing_count": len(instants)})
    return rows


def main():
    """Save comparisons on the original clocks without extrapolation."""
    folder = Path(__file__).parent
    records = []
    for current, suffix, index in ((.19, "019", 0), (.27, "027", 2)):
        human = np.load(folder.parents[1] / f".cache/human-pv/active-{index}.npz")
        control = np.load(folder / f"h01-pv-slope5-{suffix}.npz")
        half = np.load(folder / f"h01-pv-calva-half-{suffix}.npz")
        traces = (("human", human["time"], human["voltage"]),
                  ("control", control["time_ms"], control["voltage_mv"]),
                  ("half", half["time_ms"], half["voltage_mv"]))
        row = {"current_na": current}
        for name, time, voltage in traces:
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            assert len(events) >= 2 and all(e["peak_sample_voltage_mv"] > 0. for e in events[:2])
            row[name] = downward_crossings(time, voltage, events[0]["peak_sample_time_ms"],
                                          events[1]["rise_crossing_ms"], (-20., -40., -50., -60., -65., -70., -75.))
        records.append(row)
    report = {"records": records, "reserved_trace_used": False,
              "limit": "Direct falling-phase timing; not a unique channel attribution"}
    (folder / "h01-pv-return-crossings.json").write_text(json.dumps(report, indent=2))
    for row in records:
        print(row["current_na"])
        for name in ("human", "control", "half"):
            print(name, [(r["level_mv"], r["time_from_peak_ms"]) for r in row[name]])


if __name__ == "__main__":
    main()
