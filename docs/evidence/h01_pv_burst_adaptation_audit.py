"""Compare early and late individual intervals after a Kv3 closing change."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save all events and evaluate the predeclared first-pair criterion."""
    folder = Path(__file__).parent
    targets = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())["records"]
    records = []
    for index, suffix in enumerate(("019", "027")):
        human = targets[index]["human_events"]
        human_intervals = [human[i+1]["rise_crossing_ms"]-human[i]["rise_crossing_ms"] for i in range(3)]
        human_late = [e for e in human if e["rise_crossing_ms"] >= 1000. and e["peak_sample_voltage_mv"] > 0.]
        human_late_interval = human_late[1]["rise_crossing_ms"]-human_late[0]["rise_crossing_ms"] if len(human_late) >= 2 else None
        pair, metas = {}, {}
        for name, prefix in (("control", "h01-pv-calcium-entry-"), ("candidate", "h01-pv-burst-adaptation-")):
            stem = prefix + suffix
            metas[name] = json.loads((folder / (stem + ".json")).read_text())
            data = np.load(folder / (stem + ".npz"))
            selected = (data["time_ms"] >= 270.) & (data["time_ms"] < 1270.)
            events = spike_datums(data["time_ms"][selected], data["voltage_mv"][selected])
            initial = []
            for i in range(3):
                valid = len(events) > i+1 and all(e["peak_sample_voltage_mv"] > 0. for e in events[:i+2])
                interval = events[i+1]["rise_crossing_ms"]-events[i]["rise_crossing_ms"] if valid else None
                initial.append({"ordinal": i+1, "interval_ms": interval,
                                "human_interval_ms": human_intervals[i],
                                "error_ms": None if interval is None else interval-human_intervals[i]})
            late = [e for e in events if e["rise_crossing_ms"] >= 1000. and e["peak_sample_voltage_mv"] > 0.]
            interval = late[1]["rise_crossing_ms"]-late[0]["rise_crossing_ms"] if len(late) >= 2 else None
            errors = {key: events[0][key]-human[0][key] for key in (
                "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
            pair[name] = {"events": events, "initial_intervals": initial, "first_errors": errors,
                          "all_intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                          "late_pair": late[:2], "late_interval_ms": interval,
                          "late_error_ms": None if interval is None or human_late_interval is None else interval-human_late_interval}
        for key in ("current_na", "geometry", "integration", "nseg_factor", "initial_voltage_mv",
                    "temperature_c", "bias_na", "axon_calcium_decay_ms", "axon_calcium_gamma",
                    "sodium_h_tau_factor", "sodium_h_recovery_factor", "sodium_h_slope_mv",
                    "somatic_calva_factor", "somatic_kv3_factor", "somatic_kv3_tau_factor", "conductance_intervention"):
            assert metas["control"][key] == metas["candidate"][key], key
        assert metas["control"]["somatic_kv3_close_factor"] == 1.
        assert metas["candidate"]["somatic_kv3_close_factor"] == .5
        valid = all(row["error_ms"] is not None for value in pair.values() for row in value["initial_intervals"][:2])
        decision = "invalid_missing_initial_positive_spikes"
        if valid:
            improves = all(abs(pair["candidate"]["initial_intervals"][i]["error_ms"])
                           < abs(pair["control"]["initial_intervals"][i]["error_ms"]) for i in range(2))
            decision = "supported" if improves else "rejected"
        records.append({"current_na": metas["candidate"]["current_na"], **pair,
                        "first_pair_improvement": decision, "human_late_pair": human_late[:2],
                        "human_late_interval_ms": human_late_interval})
    report = {"records": records, "reserved_trace_used": False,
              "limit": "First-pair improvement is not a validated burst, waveform, or train"}
    (folder / "h01-pv-burst-adaptation-audit.json").write_text(json.dumps(report, indent=2))
    for row in records:
        print(row["current_na"], row["first_pair_improvement"])
        candidate = row["candidate"]
        print("initial intervals", candidate["initial_intervals"], "late", candidate["late_interval_ms"],
              "late error", candidate["late_error_ms"], "events", len(candidate["events"]),
              "first errors", candidate["first_errors"])


if __name__ == "__main__":
    main()
