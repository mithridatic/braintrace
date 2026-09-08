"""Record event timing and matched-voltage recovery states for SK candidates."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Write direct comparisons without inferring mediation from spike counts."""
    folder = Path(__file__).parent
    records = []
    for current, suffix, index in ((.19, "019", 0), (.27, "027", 2)):
        human = np.load(folder.parents[1] / f".cache/human-pv/active-{index}.npz")
        hs = (human["time"] >= 270.) & (human["time"] < 1270.)
        target = spike_datums(human["time"][hs], human["voltage"][hs])
        cases = {}
        for factor, middle in ((1, ""), (2, "sk2-")):
            stem = f"h01-pv-neuron-waveform-018-{middle}{suffix}"
            trace = np.load(folder / (stem+".npz"))
            time, voltage = trace["time_ms"], trace["voltage_mv"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            rises = np.array([e["rise_crossing_ms"] for e in events])
            crossings = np.flatnonzero((voltage[:-1] < -60.) & (voltage[1:] >= -60.))
            recovery = []
            for left, right in zip(events[:-1], events[1:]):
                candidates = crossings[(time[crossings] > left["fall_crossing_ms"])
                                       & (time[crossings] < right["rise_crossing_ms"])]
                if not len(candidates):
                    recovery.append(None)
                    continue
                j = candidates[0]
                instant = time[j]+(-60.-voltage[j])*(time[j+1]-time[j])/(voltage[j+1]-voltage[j])
                recovery.append({"time_ms": float(instant), "voltage_mv": -60.,
                                 **{key: float(np.interp(instant, time, trace[key]))
                                    for key in ("calcium_mm", "sk_gate", "SK_current_ma_cm2")}})
            # Check the recorded local current against conductance, gate, and driving voltage.
            expected = factor*3.566097771013125e-05*trace["sk_gate"]*(voltage+85.)
            identity_error = float(np.max(np.abs(expected-trace["SK_current_ma_cm2"])))
            assert identity_error < 1e-10
            cases[str(factor)] = {"events": events, "intervals_ms": np.diff(rises).tolist(),
                                  "recovery_at_minus60": recovery,
                                  "somatic_sk_current_identity_error_ma_cm2": identity_error}
        records.append({"current_na": current, "human_events": target, "cases": cases})
    report = {"records": records, "reserved_trace_used": False,
              "limit": "Soma current and calcium observed. Soma and axon densities changed together; their separate causal effects are not identified."}
    (folder / "h01-pv-sk-candidate-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{ "current_na": r["current_na"], "human_count": len(r["human_events"]),
                       "cases": {k: {"event_count": len(v["events"]), "first": v["events"][0],
                                      "last_interval_ms": v["intervals_ms"][-1]} for k, v in r["cases"].items()}}
                      for r in records], indent=2))


if __name__ == "__main__":
    main()
