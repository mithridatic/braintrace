"""Compare source initialization conventions and the retained slow gate states."""

import json
from pathlib import Path

import numpy as np

folder = Path(__file__).parent
sample_times = np.array([0., 200., 270., 500., 1270.])
records = []
traces = []
for initial in (80, 81):
    data = np.load(folder/f"h01-pv-neuron-initial{initial}-gates.npz")
    traces.append(data)
    metadata = json.loads((folder/f"h01-pv-neuron-initial{initial}-gates.json").read_text())
    original = np.load(folder/("h01-pv-neuron-space9.npz" if initial == 80 else "h01-pv-neuron-initial81.npz"))
    # Observer registration must leave the original trajectory unchanged.
    np.testing.assert_array_equal(data["time_ms"], original["time_ms"])
    np.testing.assert_array_equal(data["voltage_mv"], original["voltage_mv"])
    record = {"initial_voltage_mv": -initial, "spike_times_ms": metadata["spike_times_ms"],
              "observer_changed_voltage": False}
    for field in ("voltage_mv", "ih_gate", "nap_h_gate"):
        record[field] = np.interp(sample_times, data["time_ms"], data[field]).tolist()
    records.append(record)
oracle = json.loads((folder/"h01-pv-rate-oracle.json").read_text())
index = oracle["voltage_mv"].index(-86.2)
grid = np.arange(270., 1270., .005)
delta = (np.interp(grid, traces[1]["time_ms"], traces[1]["voltage_mv"])
         - np.interp(grid, traces[0]["time_ms"], traces[0]["voltage_mv"]))
report = {"conditions": "same mesh factor 9, CVode atol 1e-10, source channels and current step",
          "response_change_rss_mv": float(np.linalg.norm(delta)),
          "response_change_rms_mv": float(np.sqrt(np.mean(delta**2))),
          "response_grid_ms": .005, "response_window_ms": [270., 1270.],
          "rss_meaning": "response to the stated 1 mV initialization intervention; not a biological uncertainty estimate",
          "sample_times_ms": sample_times.tolist(),
          "independent_rate_reference_voltage_mv": -86.2,
          "ih_tau_ms": oracle["rates"]["Ih"]["m"]["tau_ms"][index],
          "nap_h_tau_ms": oracle["rates"]["Nap"]["h"]["tau_ms"][index],
          "records": records,
          "max_paired_spike_time_change_ms": float(np.max(np.abs(np.array(records[0]["spike_times_ms"])-records[1]["spike_times_ms"]))) }
(folder/"h01-pv-initial-state-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
