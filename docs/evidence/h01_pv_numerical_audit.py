"""Quantify controlled mesh and time-step effects without aligning spikes."""

import json
from pathlib import Path

import numpy as np

folder = Path(__file__).parent
grid = np.arange(270., 1270., .005)
pairs = [
    ("braincell_time_step", "h01-pv-braincell-019", "h01-pv-braincell-019-fine",
     "dt 0.005 to 0.0025 ms; maximum CV length 10 um unchanged"),
    ("braincell_mesh_10_to_5", "h01-pv-braincell-019-fine", "h01-pv-braincell-019-space5",
     "maximum CV length 10 to 5 um; dt 0.0025 ms unchanged"),
    ("braincell_mesh_5_to_2p5", "h01-pv-braincell-019-space5", "h01-pv-braincell-019-space2p5",
     "maximum CV length 5 to 2.5 um; dt 0.0025 ms unchanged"),
    ("braincell_time_step_at_2p5", "h01-pv-braincell-019-space2p5", "h01-pv-braincell-019-space2p5-fine",
     "dt 0.0025 to 0.00125 ms; maximum CV length 2.5 um unchanged"),
    ("neuron_mesh_1_to_3", "h01-pv-neuron-observed", "h01-pv-neuron-space3",
     "each section nseg multiplied by 3; CVode atol 1e-10 unchanged"),
    ("neuron_mesh_3_to_9", "h01-pv-neuron-space3", "h01-pv-neuron-space9",
     "each section nseg multiplied by 3 again; CVode atol 1e-10 unchanged"),
    ("neuron_mesh_9_to_27", "h01-pv-neuron-space9", "h01-pv-neuron-space27",
     "each section nseg multiplied by 3 again; CVode atol 1e-10 unchanged"),
    ("neuron_mesh_27_to_81", "h01-pv-neuron-space27", "h01-pv-neuron-space81",
     "each section nseg multiplied by 3 again; CVode atol 1e-10 unchanged"),
    ("neuron_soma_only", "h01-pv-neuron-observed", "h01-pv-neuron-soma9",
     "soma nseg multiplied by 9 only; CVode atol 1e-10 unchanged"),
    ("neuron_axon_only", "h01-pv-neuron-observed", "h01-pv-neuron-axon9",
     "axon nseg multiplied by 9 only; CVode atol 1e-10 unchanged"),
    ("neuron_dendrites_only", "h01-pv-neuron-observed", "h01-pv-neuron-dendrites9",
     "dendrite nseg multiplied by 9 only; CVode atol 1e-10 unchanged"),
]
results = []
for factor, before, after, intervention in pairs:
    a, b = np.load(folder/(before+".npz")), np.load(folder/(after+".npz"))
    delta = np.interp(grid, b["time_ms"], b["voltage_mv"])-np.interp(grid, a["time_ms"], a["voltage_mv"])
    ma, mb = [json.loads((folder/(name+".json")).read_text()) for name in (before, after)]
    ta, tb = np.array(ma["spike_times_ms"]), np.array(mb["spike_times_ms"])
    results.append({"factor": factor, "intervention": intervention,
                    "before": before, "after": after,
                    "response_change_rss_mv": float(np.linalg.norm(delta)),
                    "response_change_rms_mv": float(np.sqrt(np.mean(delta**2))),
                    "response_change_max_mv": float(np.max(np.abs(delta))),
                    "spike_count_before": len(ta), "spike_count_after": len(tb),
                    "max_paired_spike_time_change_ms": float(np.max(np.abs(tb-ta))) if len(ta) == len(tb) else None})
report = {"window_ms": [270., 1270.], "grid_step_ms": .005, "sample_count": len(grid),
          "meaning": "RSS of direct voltage response changes for stated numerical interventions; not biological uncertainty",
          "results": results}
(folder/"h01-pv-numerical-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
