"""Evaluate the first voltage return after a somatic Ca_LVA intervention."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain events and distinguish failed predictions from invalid comparisons."""
    folder = Path(__file__).parent
    targets = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())["records"]
    return_targets = json.loads((folder / "h01-pv-postspike-minima.json").read_text())["records"]
    records = []
    for index, suffix in enumerate(("019", "027")):
        pair, metas = {}, {}
        for name, prefix in (("control", "h01-pv-slope5-"), ("half", "h01-pv-calva-half-")):
            stem = prefix + suffix
            meta = json.loads((folder / (stem + ".json")).read_text())
            metas[name] = meta
            trace = np.load(folder / (stem + ".npz"))
            time, voltage = trace["time_ms"], trace["voltage_mv"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            valid = len(events) >= 2 and all(e["peak_sample_voltage_mv"] > 0. for e in events[:2])
            minimum = None
            if valid:
                ids = np.flatnonzero((time >= events[0]["peak_sample_time_ms"])
                                     & (time < events[1]["rise_crossing_ms"]))
                assert len(ids)
                j = ids[np.argmin(voltage[ids])]
                minimum = {"time_ms": float(time[j]), "voltage_mv": float(voltage[j]),
                           "time_from_peak_ms": float(time[j] - events[0]["peak_sample_time_ms"])}
            human = targets[index]["human_events"][0]
            errors = {key: events[0][key] - human[key] for key in (
                "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
            pair[name] = {"events": events, "first_minimum": minimum, "first_errors": errors,
                          "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist()}
        for key in ("current_na", "geometry", "integration", "nseg_factor", "initial_voltage_mv",
                    "temperature_c", "bias_na", "axon_calcium_decay_ms", "sodium_h_tau_factor",
                    "sodium_h_recovery_factor", "sodium_h_slope_mv", "conductance_intervention"):
            assert metas["control"][key] == metas["half"][key], key
        assert metas["control"].get("somatic_calva_factor", 1.) == 1.
        assert metas["half"]["somatic_calva_factor"] == .5
        valid = all(pair[name]["first_minimum"] is not None for name in pair)
        difference = (pair["half"]["first_minimum"]["voltage_mv"]
                      - pair["control"]["first_minimum"]["voltage_mv"]) if valid else None
        decision = ("supported" if difference <= -1. else "rejected") if valid else "invalid_missing_positive_spikes"
        records.append({"current_na": metas["half"]["current_na"], **pair,
                        "human_first_minimum": return_targets[index]["human"][0],
                        "minimum_voltage_change_mv": difference, "directional_decision": decision})
    report = {"records": records, "reserved_trace_used": False,
              "limit": "Coupled conductance intervention; no exclusive mediation or human validation"}
    (folder / "h01-pv-calva-return-audit.json").write_text(json.dumps(report, indent=2))
    for row in records:
        print(row["current_na"], row["directional_decision"], row["minimum_voltage_change_mv"])
        print("minimum", row["half"]["first_minimum"], "first errors", row["half"]["first_errors"],
              "complete events", len(row["half"]["events"]))


if __name__ == "__main__":
    main()
