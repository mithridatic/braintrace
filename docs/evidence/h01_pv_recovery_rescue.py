"""Evaluate recovery-only rescue of the combined candidate's late spikes."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main(slow=False):
    """Save direct event records and the predeclared late-spike decision.

    Parameters
    ----------
    slow : bool, optional
        Read the factor-2 intervention instead of the factor-0.5 intervention.
    """
    folder = Path(__file__).parent
    intervention_prefix = "h01-pv-recovery-slow-" if slow else "h01-pv-recovery-rescue-"
    targets = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())
    records = []
    for suffix, target in zip(("019", "027"), targets["records"]):
        pair = {}
        metas = {}
        for name, prefix in (("control", "h01-pv-combined-015-"),
                             ("recovery", intervention_prefix)):
            stem = prefix + suffix
            meta = json.loads((folder / (stem + ".json")).read_text())
            metas[name] = meta
            trace = np.load(folder / (stem + ".npz"))
            time, voltage = trace["time_ms"], trace["voltage_mv"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            late = [event for event in events if event["rise_crossing_ms"] >= 500.
                    and event["peak_sample_voltage_mv"] > 0.]
            human = target["human_events"][0]
            errors = {key: events[0][key] - human[key] for key in (
                "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
            pair[name] = {"events": events, "late_positive_events": late,
                          "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                          "first_errors": errors}
        for key in ("current_na", "nseg_factor", "integration", "initial_voltage_mv",
                    "temperature_c", "bias_na", "axon_calcium_decay_ms", "geometry",
                    "sodium_h_tau_factor", "conductance_intervention"):
            assert metas["control"][key] == metas["recovery"][key], key
        assert metas["control"]["sodium_h_recovery_factor"] == 1.
        assert metas["recovery"]["sodium_h_recovery_factor"] == (2. if slow else .5)
        assert metas["recovery"]["current_na"] == target["current_na"]
        records.append({"current_na": target["current_na"], **pair})
    high = records[1]
    assert high["current_na"] == .27
    rescue = (len(high["control"]["late_positive_events"]) == 0
              and len(high["recovery"]["late_positive_events"]) >= 2)
    report = {"records": records, "predeclared_rescue_supported": rescue,
              "recovery_factor": 2. if slow else .5,
              "reserved_trace_used": False,
              "limit": "Conditional recovery intervention; no human or numerical qualification"}
    (folder / (intervention_prefix + "audit.json")).write_text(json.dumps(report, indent=2))
    print("Rescue supported:", rescue)
    for record in records:
        for name in ("control", "recovery"):
            row = record[name]
            print(record["current_na"], name, "complete events", len(row["events"]),
                  "late positive events", len(row["late_positive_events"]),
                  "first errors", row["first_errors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow", action="store_true")
    main(slow=parser.parse_args().slow)
