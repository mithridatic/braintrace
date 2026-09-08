"""Compare axonal calcium removal with direct gate, current, and voltage records."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Evaluate onset separation while retaining complete event trajectories."""
    folder = Path(__file__).parent
    records = {}
    for case in ("019", "019-slow", "027-slow"):
        trace = np.load(folder / f"h01-pv-neuron-calcium-{case}.npz")
        time = trace["time_ms"]
        expected = .39879086425875565*trace["axon_sk_gate"]*(trace["axon_voltage_mv"]+85.)
        current_error = float(np.max(np.abs(expected-trace["axon_sk_current_ma_cm2"])))
        assert current_error < 1e-10
        if case == "019":
            old = np.load(folder / "h01-pv-neuron-waveform-018-019.npz")
            assert np.array_equal(time, old["time_ms"])
            assert np.array_equal(trace["voltage_mv"], old["voltage_mv"])
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], trace["voltage_mv"][selected])
        observations = []
        for event in events:
            instant = event["rise_crossing_ms"]
            observations.append({"time_ms": instant,
                                 **{key: float(np.interp(instant, time, trace[key]))
                                    for key in ("axon_voltage_mv", "axon_calcium_mm", "axon_sk_gate", "axon_sk_current_ma_cm2")}})
        records[case] = {"events": events, "axon_states_at_somatic_rise": observations,
                         "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                         "axon_current_identity_error_ma_cm2": current_error}
    delay = records["019-slow"]["events"][0]["rise_crossing_ms"]-records["019"]["events"][0]["rise_crossing_ms"]
    sk = json.loads((folder / "h01-pv-sk-region-audit.json").read_text())
    sk_delay = sk["first_crossing_changes_ms"]["axon"]
    report = {"records": records, "first_crossing_delay_ms": delay,
              "axon_density_intervention_delay_ms": sk_delay,
              "smaller_onset_delay_than_density_change": abs(delay) < abs(sk_delay),
              "qualification": "Candidate only; observations do not establish a unique mediator or human physiology"}
    (folder / "h01-pv-calcium-candidate-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"delay_ms": delay, "smaller_than_density_delay": abs(delay) < abs(sk_delay),
                      "cases": {case: {"count": len(row["events"]), "first": row["events"][0],
                                       "last_interval_ms": row["intervals_ms"][-1]} for case, row in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
