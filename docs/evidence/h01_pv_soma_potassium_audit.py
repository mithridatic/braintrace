"""Record direct waveform effects of somatic Kv3 conductance increases."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Write event datums and fixed-grid intervention effects."""
    folder = Path(__file__).parent
    reference = np.load(folder / "h01-pv-neuron-bias-control.npz")
    grid = np.arange(270., 1270., .005)
    reference_voltage = np.interp(grid, reference["time_ms"], reference["voltage_mv"])
    records = []
    for factor, name in ((1, "h01-pv-neuron-bias-control"),
                         (2, "h01-pv-neuron-soma-kv3-2"),
                         (4, "h01-pv-neuron-soma-kv3-4")):
        trace = np.load(folder / (name + ".npz"))
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        delta = np.interp(grid, time, voltage)-reference_voltage
        records.append({"trace": name, "soma_kv3_factor": factor,
                        "events": events, "spike_count": len(events),
                        "response_change_rss_mv": float(np.linalg.norm(delta)),
                        "response_change_rms_mv": float(np.sqrt(np.mean(delta**2)))})
    human = np.load(folder.parents[1] / ".cache/human-pv/active-0.npz")
    selected = (human["time"] >= 270.) & (human["time"] < 1270.)
    human_events = spike_datums(human["time"][selected], human["voltage"][selected])
    report = {"conditions": "0.19 nA; zero bias; initial -80 mV; mesh factor 9; CVode atol 1e-10",
              "sample_count": len(grid), "window_ms": [270., 1270.],
              "threshold_mv": -20., "duration_definition": "time above threshold, not AP half-width",
              "records": records, "human_events": human_events,
              "qualification": "Conditional conductance intervention; no candidate qualified"}
    (folder / "h01-pv-soma-potassium-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{k: r[k] for k in ("soma_kv3_factor", "spike_count", "response_change_rss_mv")}
                      for r in records]))


if __name__ == "__main__":
    main()
