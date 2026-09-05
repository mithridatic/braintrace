"""Compare directly sampled voltage minima between the first few spikes."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save unaligned post-spike minima with explicit interval endpoints."""
    folder = Path(__file__).parent
    records = []
    for current, suffix, index in ((.19, "019", 0), (.27, "027", 2)):
        human = np.load(folder.parents[1] / f".cache/human-pv/active-{index}.npz")
        model = np.load(folder / f"h01-pv-combined-015-{suffix}.npz")
        rows = {}
        for name, time, voltage in (("human", human["time"], human["voltage"]),
                                    ("model", model["time_ms"], model["voltage_mv"])):
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            minima = []
            for first, second in zip(events[:3], events[1:4]):
                start, stop = first["peak_sample_time_ms"], second["rise_crossing_ms"]
                ids = np.flatnonzero((time >= start) & (time < stop))
                assert len(ids)
                j = ids[np.argmin(voltage[ids])]
                minima.append({"previous_peak_ms": start, "next_rise_ms": stop,
                               "minimum_sample_time_ms": float(time[j]),
                               "minimum_sample_voltage_mv": float(voltage[j]),
                               "time_from_peak_ms": float(time[j]-start),
                               "previous_peak_positive": first["peak_sample_voltage_mv"] > 0.,
                               "next_peak_positive": second["peak_sample_voltage_mv"] > 0.})
            rows[name] = minima
        records.append({"current_na": current, **rows})
    report = {"records": records, "reserved_trace_used": False,
              "limit": "Sampled interval minima, not threshold-relative AHP amplitude or causal channel identification"}
    (folder / "h01-pv-postspike-minima.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
