"""Evaluate the predeclared regional sodium onset split."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save every event and separate onset decisions from waveform errors."""
    folder = Path(__file__).parent
    target = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())
    human = target["records"][0]["human_events"][0]
    records = {}
    for region in ("control", "soma", "axon", "all"):
        stem = ("h01-pv-neuron-waveform-018-019" if region == "control"
                else "h01-pv-onset-na110-" + region)
        meta = json.loads((folder / (stem + ".json")).read_text())
        assert meta["current_na"] == .19 and meta["nseg_factor"] == 9
        assert meta["sodium_h_tau_factor"] == .18
        assert meta["sodium_h_recovery_factor"] == 1.
        assert meta["bias_na"] == 0.
        # The retained control predates the calcium-removal CLI option.
        # New intervention records must explicitly include the field.
        if region == "control":
            assert meta.get("axon_calcium_decay_ms") is None
        else:
            assert meta["axon_calcium_decay_ms"] is None
        assert meta["integration"]["cvode_atol"] == 1e-10
        assert meta["initial_voltage_mv"] == -80. and meta["temperature_c"] == 34.
        if region != "control":
            assert meta["conductance_intervention"] == {
                "mechanism": "NaTg", "factor": 1.1, "region": region}
        data = np.load(folder / (stem + ".npz"))
        selected = (data["time_ms"] >= 270.) & (data["time_ms"] < 1270.)
        events = spike_datums(data["time_ms"][selected], data["voltage_mv"][selected])
        assert events and events[0]["peak_sample_voltage_mv"] > 0.
        fields = ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")
        records[region] = {"events": events, "first_errors": {
            key: events[0][key] - human[key] for key in fields}}
    control = records["control"]
    for region in ("soma", "axon", "all"):
        record = records[region]
        advance = control["events"][0]["rise_crossing_ms"] - record["events"][0]["rise_crossing_ms"]
        record["onset_advance_ms"] = advance
        record["advance_at_least_1ms"] = advance >= 1.
        record["shape_errors_do_not_increase"] = all(
            abs(record["first_errors"][key]) <= abs(control["first_errors"][key])
            for key in ("peak_sample_voltage_mv", "time_above_threshold_ms"))
    matches = [region for region in ("soma", "axon") if abs(
        records[region]["onset_advance_ms"] - records["all"]["onset_advance_ms"]) <= .1]
    report = {"records": records, "human_first_event": human,
              "single_region_matches_combined_within_0_1ms": matches,
              "unique_sufficient_region_under_test": matches[0] if len(matches) == 1 else None,
              "reserved_trace_used": False,
              "qualification": "Diagnostic model intervention; no human validation or parameter promotion"}
    (folder / "h01-pv-onset-sodium-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: {k: v for k, v in value.items() if k != "events"}
                      for key, value in records.items()}, indent=2))
    print("Regional match:", matches)


if __name__ == "__main__":
    main()
