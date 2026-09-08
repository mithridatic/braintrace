"""Evaluate the predeclared closure-versus-recovery waveform prediction."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Write direct events and a conditional decision for the split test."""
    folder = Path(__file__).parent
    grid = np.arange(270., 1270., .005)
    original = np.load(folder / "h01-pv-neuron-bias-control.npz")
    reference = np.interp(grid, original["time_ms"], original["voltage_mv"])
    records = {}
    for case in ("control", "closure", "recovery"):
        trace = np.load(folder / f"h01-pv-neuron-split-{case}.npz")
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        delta = np.interp(grid, time, voltage)-reference
        if case == "control":
            assert np.max(np.abs(delta)) < .01
        records[case] = {"events": events, "spike_count": len(events),
                         "response_change_rss_mv": float(np.linalg.norm(delta)),
                         "maximum_voltage_change_mv": float(np.max(np.abs(delta)))}
    widths = {case: row["events"][0]["time_above_threshold_ms"] if row["events"] else None
              for case, row in records.items()}
    closure_change = None if widths["closure"] is None else widths["control"]-widths["closure"]
    recovery_change = None if widths["recovery"] is None else widths["control"]-widths["recovery"]
    supported = (closure_change is not None and recovery_change is not None
                 and closure_change > .02 and abs(recovery_change) < abs(closure_change))
    report = {"conditions": "0.19 nA; zero bias; -80 mV initial; mesh factor 9; CVode atol 1e-10",
              "records": records, "first_width_ms": widths,
              "closure_width_reduction_ms": closure_change,
              "recovery_width_reduction_ms": recovery_change,
              "predeclared_directional_prediction_supported": supported,
              "rss_sample_count": len(grid),
              "limit": "Diagnostic state-dependent intervention in soma and axon; no human kinetic identification or numerical qualification"}
    (folder / "h01-pv-recovery-causal-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"widths": widths, "counts": {k: v["spike_count"] for k, v in records.items()},
                      "supported": supported}, indent=2))


if __name__ == "__main__":
    main()
