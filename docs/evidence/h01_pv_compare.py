"""Compare direct human PV recordings with the released NEURON cell response."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent
CACHE = ROOT.parents[1] / ".cache/human-pv"
results = []
fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
for row, (current, suffix) in enumerate(((.19, "019"), (.23, "023"), (.27, "027"))):
    recording_path = CACHE / f"active-{row}.npz"
    model_path = ROOT / f"h01-pv-neuron-{suffix}.npz"
    recorded = np.load(recording_path)
    modeled = np.load(model_path)
    source_time, source_v = recorded["time"], recorded["voltage"]
    mask = (source_time >= 270.) & (source_time < 1270.)
    times, observed = source_time[mask], source_v[mask]
    predicted = np.interp(times, modeled["time_ms"], modeled["voltage_mv"])
    residual = predicted-observed
    np.savez_compressed(ROOT / f"h01-pv-residual-{suffix}.npz", time_ms=times,
                        observed_mv=observed, predicted_mv=predicted, residual_mv=residual)
    human_peaks, _ = find_peaks(observed, height=0., prominence=40.)
    model_peaks, _ = find_peaks(predicted, height=0., prominence=40.)
    results.append({"current_na": current, "window_ms": [270., 1270.],
        "recording_sha256": hashlib.sha256(recording_path.read_bytes()).hexdigest(),
        "model_trace_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "observed_spike_times_ms": times[human_peaks].tolist(),
        "predicted_spike_times_ms": times[model_peaks].tolist(),
        "observed_peak_voltages_mv": observed[human_peaks].tolist(),
        "predicted_peak_voltages_mv": predicted[model_peaks].tolist(),
        "residual_root_sum_square_mv": float(np.linalg.norm(residual)),
        "residual_rms_mv": float(np.sqrt(np.mean(residual**2))),
        "sample_count": len(times), "peak_alignment_applied": False,
        "voltage_offset_applied": False,
        "qualification": "released reference does not reproduce the full measured voltage response"})
    for col, limits in enumerate(((270, 1270), (270, 330))):
        ax = axes[row, col]
        ax.plot(times, observed, color="#222222", linewidth=.7, label="Human recording")
        ax.plot(times, predicted, color="#ca6523", linewidth=.8, alpha=.8, label="Released model in NEURON")
        ax.set(xlim=limits, ylabel="Voltage (mV)", xlabel="Recorded stimulus time (ms)",
               title=f"{current:.2f} nA: same clock and voltage reference")
        ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Human PV recording versus published model: waveform agreement is not established")
fig.savefig(ROOT / "h01-pv-direct-comparison.svg")
fig.savefig(ROOT / "h01-pv-direct-comparison.png", dpi=130)
(ROOT / "h01-pv-direct-comparison.json").write_text(json.dumps(results, indent=2))
print(json.dumps([{k: row[k] for k in ("current_na", "residual_rms_mv", "sample_count")}
                  for row in results], indent=2))
