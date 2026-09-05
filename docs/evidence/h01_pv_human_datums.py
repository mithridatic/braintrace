"""Record direct sample and crossing datums from the released human PV traces."""

import hashlib
import json
from pathlib import Path

import numpy as np


def spike_datums(time, voltage, threshold=-20.):
    """Return complete above-threshold events without changing the trace."""
    above = voltage >= threshold
    rises = np.flatnonzero(~above[:-1] & above[1:])
    falls = np.flatnonzero(above[:-1] & ~above[1:])

    def crossing(index):
        return float(time[index]+(threshold-voltage[index])*(time[index+1]-time[index])/
                     (voltage[index+1]-voltage[index]))

    events = []
    for start in rises:
        following = falls[falls > start]
        if not len(following):
            continue
        stop = int(following[0])
        peak = start+1+int(np.argmax(voltage[start+1:stop+1]))
        events.append({"rise_crossing_ms": crossing(start), "fall_crossing_ms": crossing(stop),
                       "time_above_threshold_ms": crossing(stop)-crossing(start),
                       "peak_sample_time_ms": float(time[peak]), "peak_sample_voltage_mv": float(voltage[peak])})
    return events


def main():
    """Write datums, source hashes, and prospective fitting boundaries."""
    folder = Path(__file__).parent
    cache = folder.parents[1]/".cache/human-pv"
    records = []
    for index, current in enumerate((.19, .23, .27)):
        path = cache/f"active-{index}.npz"
        data = np.load(path)
        time, voltage = data["time"], data["voltage"]
        selection = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selection], voltage[selection])
        assert len(events) == (12, 31, 43)[index]
        baseline = voltage[(time >= 200.) & (time < 270.)]
        records.append({"current_na": current, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "sample_step_ms": float(np.median(np.diff(time))),
                        "baseline_mean_mv": float(baseline.mean()),
                        "baseline_sample_sd_mv": float(baseline.std(ddof=1)),
                        "baseline_sd_meaning": "within-trace fluctuation; not biological between-cell uncertainty",
                        "fitting_role": "reserved from future parameter fitting" if index == 1 else "candidate calibration trace",
                        "spikes": events})
    report = {"source": "ModelDB 267587 L5PV released human active.pkl exports",
              "stimulus_window_ms": [270., 1270.], "baseline_window_ms": [200., 270.],
              "crossing_threshold_mv": -20.,
              "duration_definition": "time above -20 mV, with linear crossing interpolation; not AP half-width",
              "voltage_offset_applied": False, "peak_alignment_applied": False,
              "holdout_limit": "all traces have already been inspected; 0.23 nA is a prospective fitting holdout, not untouched external validation",
              "records": records}
    (folder/"h01-pv-human-datums.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{ "current_na": row["current_na"], "spike_count": len(row["spikes"]),
                       "first_spike": row["spikes"][0]} for row in records]))


if __name__ == "__main__":
    main()
