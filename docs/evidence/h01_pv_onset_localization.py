"""Locate onset error on the original clock without asserting a unique cause."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def crossings(time, voltage, stop):
    """Return first upward crossings and their multiplicity before the first peak."""
    records = []
    for level in (-80., -75., -70., -65., -60., -55., -50., -45., -40., -30., -20.):
        hits = np.flatnonzero((voltage[:-1] < level) & (voltage[1:] >= level)
                             & (time[:-1] >= 270.) & (time[1:] <= stop))
        instant = None
        if len(hits):
            j = hits[0]
            instant = float(time[j]+(level-voltage[j])*(time[j+1]-time[j])/(voltage[j+1]-voltage[j]))
        records.append({"voltage_mv": level, "first_crossing_ms": instant, "crossing_count": len(hits)})
    return records


def main():
    """Save the two-current crossing comparison and raw onset windows."""
    folder = Path(__file__).parent
    records = []
    for current, suffix, index in ((.19, "019", 0), (.27, "027", 2)):
        human = np.load(folder.parents[1] / f".cache/human-pv/active-{index}.npz")
        model = np.load(folder / f"h01-pv-neuron-waveform-018-{suffix}.npz")
        rows = {}
        for name, time, voltage in (("human", human["time"], human["voltage"]),
                                    ("model", model["time_ms"], model["voltage_mv"])):
            selected = (time >= 270.) & (time < 1270.)
            first = spike_datums(time[selected], voltage[selected])[0]
            rows[name] = crossings(time, voltage, first["peak_sample_time_ms"])
        comparison = []
        for measured, predicted in zip(rows["human"], rows["model"]):
            a, b = measured["first_crossing_ms"], predicted["first_crossing_ms"]
            comparison.append({"voltage_mv": measured["voltage_mv"],
                               "human_time_ms": a, "model_time_ms": b,
                               "error_ms": None if a is None or b is None else b-a,
                               "human_crossing_count": measured["crossing_count"]})
        records.append({"current_na": current, "crossings": comparison})
    report = {"records": records, "reserved_trace_used": False,
              "limit": "Voltage-error localization only; no causal parameter identification"}
    (folder / "h01-pv-onset-localization.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
