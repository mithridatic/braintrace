"""Compare the measured-bias intervention with its zero-bias control."""

import json
from pathlib import Path

import numpy as np


def main():
    """Check current accounting and report direct response changes."""
    folder = Path(__file__).parent
    control = np.load(folder / "h01-pv-neuron-bias-control.npz")
    changed = np.load(folder / "h01-pv-neuron-bias-measured.npz")
    previous = np.load(folder / "h01-pv-neuron-initial80-gates.npz")
    grid = np.arange(270., 1270., .005)
    control_grid = np.interp(grid, control["time_ms"], control["voltage_mv"])
    previous_grid = np.interp(grid, previous["time_ms"], previous["voltage_mv"])
    parity = float(np.max(np.abs(control_grid-previous_grid)))
    assert parity < .01, parity
    metadata = {}
    for name, trace in (("control", control), ("measured", changed)):
        meta = json.loads((folder / f"h01-pv-neuron-bias-{name}.json").read_text())
        settled = trace["time_ms"] > .001
        assert np.max(np.abs(trace["bias_current_na"][settled]-meta["bias_na"])) < 1e-12
        assert np.array_equal(trace["total_current_na"], trace["current_na"]+trace["bias_current_na"])
        metadata[name] = {
            "bias_na": meta["bias_na"], "baseline_mean_mv": meta["baseline_mean_mv"],
            "spike_times_ms": meta["spike_times_ms"], "spike_peaks_mv": meta["spike_peaks_mv"],
            "onset_states": {key: float(np.interp(270., trace["time_ms"], trace[key]))
                             for key in ("voltage_mv", "ih_gate", "nap_h_gate", "calcium_mm")},
        }
    delta = np.interp(grid, changed["time_ms"], changed["voltage_mv"])-control_grid
    report = {
        "conditions": "0.19 nA; mesh factor 9; initial -80 mV; CVode atol 1e-10",
        "previous_control_maximum_difference_mv": parity,
        "window_ms": [270., 1270.], "sample_count": len(grid),
        "response_change_rss_mv": float(np.linalg.norm(delta)),
        "response_change_rms_mv": float(np.sqrt(np.mean(delta**2))),
        "records": metadata,
        "qualification": "Bias-only intervention. Not full acquisition history or a validated candidate.",
    }
    (folder / "h01-pv-bias-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
