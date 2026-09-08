"""Separate conductance response changes from residuals to the human trace."""

import json
from pathlib import Path

import numpy as np

folder = Path(__file__).parent
reference = np.load(folder/"h01-pv-neuron-space9.npz")
reference_meta = json.loads((folder/"h01-pv-neuron-space9.json").read_text())
human = np.load(folder.parents[1]/".cache/human-pv/active-0.npz")
selected = (human["time"] >= 270.) & (human["time"] < 1270.)
human_time, human_voltage = human["time"][selected], human["voltage"][selected]
grid = np.arange(270., 1270., .005)
baseline = np.interp(grid, reference["time_ms"], reference["voltage_mv"])
baseline_human_residual = np.interp(human_time, reference["time_ms"], reference["voltage_mv"])-human_voltage
records = []
for mechanism in ("NaTg", "Kv3_1", "SK"):
    name = f"h01-pv-neuron-scale-{mechanism}"
    trace = np.load(folder/(name+".npz"))
    metadata = json.loads((folder/(name+".json")).read_text())
    change = np.interp(grid, trace["time_ms"], trace["voltage_mv"])-baseline
    residual = np.interp(human_time, trace["time_ms"], trace["voltage_mv"])-human_voltage
    records.append({"mechanism": mechanism, "density_multiplier": .9,
                    "response_change_rss_mv": float(np.linalg.norm(change)),
                    "response_change_rms_mv": float(np.sqrt(np.mean(change**2))),
                    "human_residual_rss_mv": float(np.linalg.norm(residual)),
                    "human_residual_rms_mv": float(np.sqrt(np.mean(residual**2))),
                    "spike_times_ms": metadata["spike_times_ms"],
                    "spike_peaks_mv": metadata["spike_peaks_mv"],
                    "first_peak_change_mv": metadata["spike_peaks_mv"][0]-reference_meta["spike_peaks_mv"][0]})
report = {"conditions": "0.19 nA; initial -80 mV; mesh factor 9; CVode atol 1e-10",
          "intervention": "reduce only the named conductance by 10 percent in soma and axon",
          "response_change_sample_count": len(grid), "human_residual_sample_count": len(human_time),
          "window_ms": [270., 1270.], "voltage_offset_applied": False, "peak_alignment_applied": False,
          "baseline_human_residual_rms_mv": float(np.sqrt(np.mean(baseline_human_residual**2))),
          "baseline_spike_times_ms": reference_meta["spike_times_ms"],
          "baseline_spike_peaks_mv": reference_meta["spike_peaks_mv"],
          "records": records,
          "qualification": "conditional physical sensitivity; no candidate is physiologically qualified"}
(folder/"h01-pv-conductance-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps([{k: r[k] for k in ("mechanism", "response_change_rss_mv", "human_residual_rms_mv", "first_peak_change_mv")}
                  for r in records]))
