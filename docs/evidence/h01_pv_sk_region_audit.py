"""Evaluate the regional SK intervention against direct event times."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain every event and check the predeclared regional timing criterion."""
    folder = Path(__file__).parent
    records = {}
    names = {"control": "h01-pv-neuron-waveform-018-019",
             "both": "h01-pv-neuron-waveform-018-sk2-019",
             "soma": "h01-pv-neuron-sk2-soma-019", "axon": "h01-pv-neuron-sk2-axon-019"}
    for case, stem in names.items():
        trace = np.load(folder / (stem+".npz"))
        meta = json.loads((folder / (stem+".json")).read_text())
        assert meta["current_na"] == .19 and meta["nseg_factor"] == 9
        assert meta["sodium_h_tau_factor"] == .18 and meta["sodium_h_recovery_factor"] == 1.
        if case in ("soma", "axon"):
            assert meta["conductance_intervention"] == {"mechanism": "SK", "factor": 2., "region": case}
        time = trace["time_ms"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], trace["voltage_mv"][selected])
        records[case] = {"events": events,
                         "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist()}
    first = {case: row["events"][0]["rise_crossing_ms"] for case, row in records.items()}
    matches = [case for case in ("soma", "axon") if abs(first[case]-first["both"]) < .1]
    report = {"records": records, "first_crossing_ms": first,
              "first_crossing_changes_ms": {case: value-first["control"] for case, value in first.items()},
              "regional_sufficiency": matches[0] if len(matches) == 1 else "unresolved",
              "limit": "Sufficiency for the specified onset effect only; not a unique mediator or validated human fit"}
    (folder / "h01-pv-sk-region-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("first_crossing_changes_ms", "regional_sufficiency")}
                     | {"event_counts": {case: len(row["events"]) for case, row in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
