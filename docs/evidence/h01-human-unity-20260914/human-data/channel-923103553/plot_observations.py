"""Render unfiltered human nucleated-patch observations, preserving source clocks."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify': False, 'font.size': 9})
SOURCE = json.loads((ROOT / 'source.json').read_bytes())


def load(sweep):
    """Load an export after verifying its byte identity.

    Parameters
    ----------
    sweep : int
        Retained channel-discovery sweep.

    Returns
    -------
    dict
        Arrays on their original clock and voltage reference.
    """
    path = ROOT / f'sweep-{sweep}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE['files'][path.name]
    with np.load(path) as data:
        return dict(data)


families = [('Total-current protocol', [98, 102, 106]),
            ('100 ms prepulse protocol', [117, 121, 125]),
            ('Sustained-current protocol', [132, 136, 140])]
colors = ['#176b9b', '#b16c00', '#9c2b51']
fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
current_limits = []
for row, (title, sweeps) in enumerate(families):
    for sn, color in zip(sweeps, colors):
        a = load(sn)
        t, i, v = a['time_ms'], a['total_current_pa'], a['command_voltage_mv']
        window = (t >= 990) & (t <= 2200)
        onset = (t >= 1099) & (t <= 1120)
        axes[row, 0].plot(t[window], i[window], color=color, lw=.5, label=f'Sweep {sn}')
        axes[row, 1].plot(t[window], v[window], color=color, lw=1)
        axes[row, 2].plot(t[onset], i[onset], color=color, lw=.5)
        current_limits.extend([float(i[window].min()), float(i[window].max())])
    axes[row, 0].set_title(title)
    axes[row, 0].legend(loc='upper right')
    axes[row, 0].set_ylabel('Total amplifier current (pA)')
    axes[row, 1].set_ylabel('DAC + holding command (mV)')
    axes[row, 2].set_ylabel('Total amplifier current (pA)')
    for col in [0, 1]: axes[row, col].set_xlim(990, 2200)
    axes[row, 1].set_ylim(-105, 85)
    axes[row, 2].set_xlim(1099, 1120)
for row in range(3):
    for col in [0, 2]:
        axes[row, col].set_ylim(min(current_limits) * 1.1, max(current_limits) * 1.1)
for ax in axes.flat:
    ax.set_xlabel('Original sweep time (ms)')
    ax.grid(alpha=.2)
axes[0, 1].set_title('Command; actual patch voltage unavailable')
axes[0, 2].set_title('Step onset detail; includes capacitive artifacts')
fig.suptitle('Human L1 LAMP5 NMBR | specimen 923103580 | raw current, no leak/junction correction')
fig.savefig(ROOT / 'steps.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
recovery = [(151, 2405.), (160, 2450.), (169, 6400.)]
limits = []
for row, (sn, second) in enumerate(recovery):
    a = load(sn)
    t, i, v = a['time_ms'], a['total_current_pa'], a['command_voltage_mv']
    window = (t >= 2000) & (t <= 6750)
    detail = (t >= second - 1) & (t <= second + 30)
    axes[row, 0].plot(t[window], i[window], color=colors[row], lw=.5)
    axes[row, 1].plot(t[window], v[window], color=colors[row], lw=1)
    axes[row, 2].plot(t[detail] - second, i[detail], color=colors[row], lw=.5)
    limits.extend([float(i[window].min()), float(i[window].max())])
    for col in [0, 1]:
        axes[row, col].set_xlim(2000, 6750)
        axes[row, col].set_xlabel('Original sweep time (ms)')
        if t[-1] < 6750:
            axes[row, col].axvspan(t[-1], 6750, color='#dddddd')
    axes[row, 1].set_ylim(-105, 85)
    axes[row, 2].set_xlim(-1, 30)
    axes[row, 2].set_xlabel(f'Time from second onset at {second:g} ms')
    axes[row, 0].set_title(f'Sweep {sn}; recovery interval {second - 2400:g} ms')
    axes[row, 0].set_ylabel('Total amplifier current (pA)')
    axes[row, 1].set_ylabel('DAC + holding command (mV)')
    axes[row, 2].set_ylabel('Total amplifier current (pA)')
for row in range(3):
    for col in [0, 2]: axes[row, col].set_ylim(min(limits) * 1.1, max(limits) * 1.1)
for ax in axes.flat: ax.grid(alpha=.2)
fig.suptitle('Raw recovery protocols | gray = outside recorded array | onset detail explicitly aligned')
fig.savefig(ROOT / 'recovery.png', dpi=150)
plt.close(fig)
