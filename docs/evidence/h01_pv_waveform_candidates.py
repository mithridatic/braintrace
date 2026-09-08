"""Screen inferred inactivation candidates against direct human spike datums."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Write waveform errors without promoting a physiologically untested fit."""
    folder = Path(__file__).parent
    human = np.load(folder.parents[1] / ".cache/human-pv/active-0.npz")
    selected = (human["time"] >= 270.) & (human["time"] < 1270.)
    target_time, target_voltage = human["time"][selected], human["voltage"][selected]
    target_events = spike_datums(target_time, target_voltage)
    keys = ("peak_sample_voltage_mv", "time_above_threshold_ms")
    records = []
    for factor, stem in ((1., "h01-pv-neuron-split-control"),
                          (.5, "h01-pv-neuron-split-closure"),
                          (.25, "h01-pv-neuron-waveform-025"),
                          (.1, "h01-pv-neuron-waveform-010")):
        trace = np.load(folder / (stem+".npz"))
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        errors = {key: events[0][key]-target_events[0][key] for key in keys} if events else None
        residual = np.interp(target_time, time, voltage)-target_voltage
        records.append({"closure_factor": factor, "trace": stem, "events": events,
                        "events_with_peak_above_zero": sum(e["peak_sample_voltage_mv"] > 0. for e in events),
                        "first_event_errors": errors,
                        "unaligned_voltage_rms_mv": float(np.sqrt(np.mean(residual**2)))})
    baseline = records[0]["first_event_errors"]
    for row in records:
        errors = row["first_event_errors"]
        row["improves_both_first_event_errors"] = errors is not None and all(abs(errors[k]) < abs(baseline[k]) for k in keys)
    report = {"target": "released active-0, 0.19 nA, already junction corrected",
              "human_events": target_events, "records": records,
              "holdout": "0.23 nA not used",
              "qualification": "Screen only; inferred kinetics, no numerical or physiological acceptance"}
    (folder / "h01-pv-waveform-candidates.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{k: r[k] for k in ("closure_factor", "first_event_errors", "improves_both_first_event_errors")}
                      | {"complete_minus20_excursions": len(r["events"]),
                         "events_with_peak_above_zero": r["events_with_peak_above_zero"]} for r in records], indent=2))


if __name__ == "__main__":
    main()
