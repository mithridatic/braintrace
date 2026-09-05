"""Evaluate the full-cell inactivation-slope intervention at two inputs."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save complete events, first errors, and direct post-spike minima."""
    folder = Path(__file__).parent
    references = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())
    records = []
    for suffix, reference in zip(("019", "027"), references["records"]):
        stem = "h01-pv-slope5-" + suffix
        meta = json.loads((folder / (stem + ".json")).read_text())
        control = json.loads((folder / f"h01-pv-combined-015-{suffix}.json").read_text())
        for key in ("current_na", "geometry", "integration", "nseg_factor",
                    "initial_voltage_mv", "temperature_c", "bias_na", "axon_calcium_decay_ms",
                    "sodium_h_tau_factor", "sodium_h_recovery_factor", "conductance_intervention"):
            assert meta[key] == control[key], key
        assert meta["sodium_h_slope_mv"] == 5.
        assert control.get("sodium_h_slope_mv") is None
        trace = np.load(folder / (stem + ".npz"))
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        control_trace = np.load(folder / f"h01-pv-combined-015-{suffix}.npz")
        control_selected = (control_trace["time_ms"] >= 270.) & (control_trace["time_ms"] < 1270.)
        control_events = spike_datums(control_trace["time_ms"][control_selected],
                                     control_trace["voltage_mv"][control_selected])
        control_late = [e for e in control_events if e["rise_crossing_ms"] >= 500.
                        and e["peak_sample_voltage_mv"] > 0.]
        late = [e for e in events if e["rise_crossing_ms"] >= 500. and e["peak_sample_voltage_mv"] > 0.]
        minima = []
        for first, second in zip(events[:3], events[1:4]):
            ids = np.flatnonzero((time >= first["peak_sample_time_ms"])
                                 & (time < second["rise_crossing_ms"]))
            assert len(ids)
            j = ids[np.argmin(voltage[ids])]
            minima.append({"time_ms": float(time[j]), "voltage_mv": float(voltage[j]),
                           "time_from_peak_ms": float(time[j] - first["peak_sample_time_ms"]),
                           "next_peak_positive": second["peak_sample_voltage_mv"] > 0.})
        human = reference["human_events"][0]
        errors = {key: events[0][key] - human[key] for key in (
            "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
        records.append({"current_na": meta["current_na"], "events": events,
                        "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                        "late_positive_events": late, "first_errors": errors,
                        "control_late_positive_events": control_late,
                        "postspike_minima": minima})
    assert records[1]["current_na"] == .27
    report = {"records": records,
              "high_input_late_spike_rescue": (len(records[1]["late_positive_events"]) >= 2
                                               and not records[1]["control_late_positive_events"]),
              "reserved_trace_used": False,
              "limit": "Raw slope changes equilibrium and kinetics; rescue is not human validation"}
    (folder / "h01-pv-slope5-audit.json").write_text(json.dumps(report, indent=2))
    print("Rescue:", report["high_input_late_spike_rescue"])
    for row in records:
        print(row["current_na"], "events", len(row["events"]), "late positive", len(row["late_positive_events"]),
              "first errors", row["first_errors"], "minima", row["postspike_minima"])


if __name__ == "__main__":
    main()
