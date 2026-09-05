"""Compare released human hyperpolarizing traces on their original clock."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).parent
cache = folder.parents[1]/".cache/human-pv"
fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
reports = []
for index, (current, suffix) in enumerate(((-.11, "011"), (-.05, "005"))):
    human_path = cache/f"passive-{index}.npz"
    model_path = folder/f"h01-pv-neuron-hyper{suffix}.npz"
    human, model = np.load(human_path), np.load(model_path)
    selected = (human["time"] >= 200.) & (human["time"] < 1500.)
    time, observed = human["time"][selected], human["voltage"][selected]
    predicted = np.interp(time, model["time_ms"], model["voltage_mv"])
    residual = predicted-observed
    windows = {}
    for name, start, stop in (("baseline", 200., 270.), ("stimulus", 270., 1270.),
                              ("late_stimulus", 1170., 1270.), ("recovery", 1270., 1500.)):
        mask = (time >= start) & (time < stop)
        windows[name] = {"start_ms": start, "stop_ms": stop,
                         "human_mean_mv": float(observed[mask].mean()),
                         "model_mean_mv": float(predicted[mask].mean()),
                         "residual_rms_mv": float(np.sqrt(np.mean(residual[mask]**2))),
                         "residual_rss_mv": float(np.linalg.norm(residual[mask]))}
    reports.append({"current_na": current, "windows": windows,
                    "human_sha256": hashlib.sha256(human_path.read_bytes()).hexdigest(),
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "voltage_offset_applied": False, "time_alignment_applied": False,
                    "model": "all active mechanisms retained; NEURON nseg factor 9, CVode atol 1e-10"})
    np.savez_compressed(folder/f"h01-pv-hyper{suffix}-residual.npz", time_ms=time,
                        observed_mv=observed, predicted_mv=predicted, residual_mv=residual)
    axes[index].plot(time, observed, color="#222222", lw=.8, label="Human recording")
    axes[index].plot(time, predicted, color="#ca6523", lw=1, label="Published model")
    axes[index].set(xlabel="Time from initialization (ms)", ylabel="Soma voltage (mV)",
                    title=f"{current:.2f} nA; original time and voltage reference")
    axes[index].legend()
fig.savefig(folder/"h01-pv-hyperpolarization.svg")
fig.savefig(folder/"h01-pv-hyperpolarization.png", dpi=140)
(folder/"h01-pv-hyperpolarization.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports))
