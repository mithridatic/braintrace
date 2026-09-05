"""Plot direct simulated responses, without aligning their spike peaks."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
curves = []
for name, label in [("h01-active-aligned-2p5", "Na 25 / K 10"),
                    ("h01-active-aligned-k20", "Na 25 / K 20"),
                    ("h01-active-aligned-na27-k20", "Na 27 / K 20")]:
    data = json.loads((ROOT / (name + ".json")).read_text())
    v = np.asarray(data["result"]["voltage_mv"]).ravel()
    t = (np.arange(len(v))+1)*data["result"]["dt_ms"]
    axes[0].plot(t, v, label=label, linewidth=1.4)
    curves.append(v)
axes[0].axvspan(2, 5, color="grey", alpha=.12, label="0.5 nA input")
axes[0].set(title="Simulated H01 soma voltage: change one conductance at a time",
            ylabel="Voltage (mV)", xlabel="Time from simulation start (ms)", xlim=(1, 10))
axes[0].legend(title="Base density (mS/cm²)", fontsize=9, loc="upper right")
axes[1].plot(t, curves[1]-curves[0], label="K 10 → 20; Na fixed at 25")
axes[1].plot(t, curves[2]-curves[1], label="Na 25 → 27; K fixed at 20")
axes[1].axhline(0, color="grey", linewidth=.6)
axes[1].set(title="Signed response change at the same time; no peak alignment",
            ylabel="Voltage change (mV)", xlabel="Time from simulation start (ms)", xlim=(1, 10))
axes[1].legend(fontsize=9)
data = json.loads((ROOT / "h01-active-five-pulses-k20.json").read_text())
v = np.asarray(data["result"]["voltage_mv"]).ravel()
t = (np.arange(len(v))+1)*data["result"]["dt_ms"]
axes[2].plot(t, v, color="#17607c", linewidth=1)
for onset in 2+25*np.arange(5):
    axes[2].axvspan(onset, onset+3, color="grey", alpha=.2)
axes[2].set(title="Five simulated pulses: Na 25 / K 20; shaded regions show input",
            ylabel="Voltage (mV)", xlabel="Time from simulation start (ms)", xlim=(0, 130))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
fig.suptitle("Direct model behavior — biological validation remains pending", fontsize=14)
fig.savefig(ROOT / "h01-causal-traces.svg")
fig.savefig(ROOT / "h01-causal-traces.png", dpi=130)
