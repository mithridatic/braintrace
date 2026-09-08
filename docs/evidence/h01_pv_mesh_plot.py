"""Plot direct voltage traces for isolated numerical interventions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).parent
cases = [("Original mesh", "h01-pv-neuron-observed"),
         ("Soma refined 9-fold", "h01-pv-neuron-soma9"),
         ("Axon refined 9-fold", "h01-pv-neuron-axon9"),
         ("Dendrites refined 9-fold", "h01-pv-neuron-dendrites9")]
fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharey=True)
for row, (label, name) in enumerate(cases):
    data = np.load(folder/(name+".npz"))
    for col, bounds in enumerate(((270, 1270), (300, 400))):
        ax = axes[row, col]
        selected = (data["time_ms"] >= bounds[0]) & (data["time_ms"] <= bounds[1])
        ax.plot(data["time_ms"][selected], data["voltage_mv"][selected], color="#245e91", lw=.8)
        ax.set_xlim(*bounds)
        ax.set_ylim(-95, 55)
        ax.grid(alpha=.2)
        ax.set_title(label, loc="left", fontsize=10)
        if col == 0:
            ax.set_ylabel("Soma voltage (mV)")
        if row == 3:
            ax.set_xlabel("Time from initialization (ms)")
fig.suptitle("Same cell parameters and current; change only the electrical mesh\n"
             "NEURON CVode tolerance 1e-10; no spike alignment", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(folder/"h01-pv-mesh-interventions.png", dpi=160)
fig.savefig(folder/"h01-pv-mesh-interventions.svg")
