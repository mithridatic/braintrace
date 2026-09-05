"""Evaluate whether the sodium timing effect survives removal of SK current."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain direct events and apply the predeclared rejection criterion."""
    folder = Path(__file__).parent
    records = {}
    for condition, prefix in (("present", "split"), ("removed", "no-sk")):
        cases = {}
        for case in ("control", "closure"):
            stem = f"h01-pv-neuron-{prefix}-{case}"
            trace = np.load(folder / (stem+".npz"))
            meta = json.loads((folder / (stem+".json")).read_text())
            assert meta["current_na"] == .19 and meta["bias_na"] == 0.
            assert meta["nseg_factor"] == 9 and meta["initial_voltage_mv"] == -80.
            assert meta["sodium_h_recovery_factor"] == 1.
            assert meta["sodium_h_tau_factor"] == (1. if case == "control" else .5)
            maximum_sk = float(np.max(np.abs(trace["SK_current_ma_cm2"])))
            if condition == "removed":
                assert meta["conductance_intervention"] == {"mechanism": "SK", "factor": 0., "region": "all"}
                assert maximum_sk == 0.
            time = trace["time_ms"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], trace["voltage_mv"][selected])
            cases[case] = {"trace": stem, "events": events,
                           "maximum_somatic_sk_current_ma_cm2": maximum_sk,
                           "first_interval_ms": (events[1]["rise_crossing_ms"]-events[0]["rise_crossing_ms"])
                           if len(events) >= 2 else None}
        records[condition] = cases
    removed = records["removed"]
    comparable = all(len(row["events"]) >= 2 for row in removed.values())
    shift = (removed["closure"]["events"][1]["rise_crossing_ms"]
             - removed["control"]["events"][1]["rise_crossing_ms"]) if comparable else None
    result = "rejected at tested conditions" if shift is not None and abs(shift) > 1. else "unresolved"
    report = {"claim": "SK current is necessary for the later timing effect of faster sodium inactivation",
              "records": records, "second_crossing_shift_without_sk_ms": shift,
              "necessity_claim": result,
              "limit": "Removal changes operating point; no mediation fraction or human physiological validation"}
    (folder / "h01-pv-sk-necessity-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"decision": result, "second_crossing_shift_ms": shift,
                      "first_intervals_ms": {condition: {case: row["first_interval_ms"] for case, row in cases.items()}
                                             for condition, cases in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
