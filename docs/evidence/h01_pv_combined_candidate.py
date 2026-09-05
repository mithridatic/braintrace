"""Compare a combined sodium candidate on both calibration inputs."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain direct events and evaluate the predeclared dominance screen."""
    folder = Path(__file__).parent
    previous = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())
    records = []
    for suffix, reference in zip(("019", "027"), previous["records"]):
        stem = "h01-pv-combined-015-" + suffix
        data = np.load(folder / (stem + ".npz"))
        meta = json.loads((folder / (stem + ".json")).read_text())
        assert meta["current_na"] == reference["current_na"]
        assert meta["sodium_h_tau_factor"] == .15 and meta["sodium_h_recovery_factor"] == 1.
        assert meta["conductance_intervention"] == {
            "mechanism": "NaTg", "factor": 1.1, "region": "soma"}
        assert meta["nseg_factor"] == 9 and meta["integration"]["cvode_atol"] == 1e-10
        assert meta["bias_na"] == 0. and meta["axon_calcium_decay_ms"] is None
        assert meta["initial_voltage_mv"] == -80. and meta["temperature_c"] == 34.
        selected = (data["time_ms"] >= 270.) & (data["time_ms"] < 1270.)
        events = spike_datums(data["time_ms"][selected], data["voltage_mv"][selected])
        target = reference["human_events"][0]
        fields = ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")
        errors = {key: events[0][key] - target[key] for key in fields} if events else None
        old_errors = reference["first_event_errors"]
        improved = {key: abs(errors[key]) <= abs(old_errors[key]) for key in fields} if events else {}
        records.append({"current_na": reference["current_na"], "events": events,
                        "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                        "first_errors": errors, "previous_first_errors": old_errors,
                        "absolute_error_does_not_increase": improved,
                        "complete_positive_first_spike": bool(events and events[0]["peak_sample_voltage_mv"] > 0.)})
    dominates = all(r["complete_positive_first_spike"] and all(
        r["absolute_error_does_not_increase"].values()) for r in records)
    strict = any(r["first_errors"] and any(abs(r["first_errors"][key]) < abs(
        r["previous_first_errors"][key]) for key in fields) for r in records)
    report = {"records": records, "dominates_previous_first_event_errors": bool(dominates and strict),
              "reserved_trace_used": False,
              "qualification": "Combined candidate screen, not isolated causality or human validation"}
    (folder / "h01-pv-combined-candidate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, "records": [{k: v for k, v in r.items()
          if k not in ("events", "intervals_ms")} for r in records]}, indent=2))


if __name__ == "__main__":
    main()
