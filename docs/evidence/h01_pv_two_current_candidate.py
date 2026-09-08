"""Compare the intermediate kinetic candidate with both calibration traces."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain every event and unaligned residual for the two input currents."""
    folder = Path(__file__).parent
    records = []
    for current, suffix, index in ((.19, "019", 0), (.27, "027", 2)):
        stem = "h01-pv-neuron-waveform-018-"+suffix
        model = np.load(folder / (stem+".npz"))
        meta = json.loads((folder / (stem+".json")).read_text())
        assert meta["current_na"] == current
        assert meta["sodium_h_tau_factor"] == .18
        assert meta["sodium_h_recovery_factor"] == 1.
        human = np.load(folder.parents[1] / f".cache/human-pv/active-{index}.npz")
        selected = (human["time"] >= 270.) & (human["time"] < 1270.)
        time, voltage = human["time"][selected], human["voltage"][selected]
        target_events = spike_datums(time, voltage)
        m_selected = (model["time_ms"] >= 270.) & (model["time_ms"] < 1270.)
        events = spike_datums(model["time_ms"][m_selected], model["voltage_mv"][m_selected])
        residual = np.interp(time, model["time_ms"], model["voltage_mv"])-voltage
        np.savez_compressed(folder / (stem+"-human-residual.npz"), time_ms=time, residual_mv=residual)
        fields = ("peak_sample_voltage_mv", "time_above_threshold_ms", "rise_crossing_ms")
        errors = {key: events[0][key]-target_events[0][key] for key in fields} if events else None
        records.append({"current_na": current, "model_events": events, "human_events": target_events,
                        "positive_peak_events": sum(e["peak_sample_voltage_mv"] > 0. for e in events),
                        "first_event_errors": errors,
                        "unaligned_rms_mv": float(np.sqrt(np.mean(residual**2)))})
    report = {"inferred_closure_factor": .18, "records": records,
              "reserved_current_na": .23, "reserved_trace_used": False,
              "qualification": "Candidate comparison; no physiological acceptance or promotion"}
    (folder / "h01-pv-two-current-candidate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{ "current_na": r["current_na"], "first_event_errors": r["first_event_errors"],
                        "model_event_count": len(r["model_events"]), "human_event_count": len(r["human_events"]),
                        "positive_peak_events": r["positive_peak_events"]} for r in records], indent=2))


if __name__ == "__main__":
    main()
