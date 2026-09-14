"""Render paired currents and same-command baseline windows without averaging."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify': False, 'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
baseline_fig, baseline_axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
for col, sn in enumerate([123, 132, 141]):
    prefix = root / f'sweep-{sn}'
    m = json.loads(prefix.with_suffix('.json').read_bytes())
    assert hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest() == m['arrays_sha256']
    with np.load(prefix.with_suffix('.npz')) as a:
        phase = a['phase_ms']
        axes[0, col].plot(phase, a['first_current_pa'], lw=.5, label='First pulse')
        axes[0, col].plot(phase, a['second_current_pa'], lw=.5, label='Second pulse')
        fit = (phase >= 10) & (phase < 280)
        axes[1, col].plot(phase[fit], a['difference_pa'][fit], lw=.5)
        baseline_axes[0, col].plot(a['initial_baseline_time_ms'], a['initial_baseline_current_pa'], lw=.5)
        baseline_axes[1, col].plot(a['final_baseline_time_ms'] - m['second_onset_ms'] - 300,
                                    a['final_baseline_current_pa'], lw=.5)
    axes[0, col].set_title(f'Sweep {sn}; gap {m["gap_ms"]:g} ms')
    axes[0, col].legend()
    axes[0, col].set_ylim(-100, 1800)
    axes[1, col].set_ylim(-20, 750)
    axes[0, col].set_ylabel('Total amplifier current (pA)')
    axes[1, col].set_ylabel('Second minus first current (pA)')
    baseline_axes[0, col].set_title(f'Sweep {sn}; initial holding window')
    baseline_axes[1, col].set_title(f'Final window after pulse ending at {m["second_onset_ms"]+300:g} ms')
    baseline_axes[0, col].set_xlabel('Original sweep time (ms)')
    baseline_axes[1, col].set_xlabel('Time after second pulse end (ms)')
for ax in axes.flat:
    ax.set_xlabel('Time after each pulse onset (ms)')
    ax.grid(alpha=.2)
for ax in baseline_axes.flat:
    ax.set_ylim(-10, 45)
    ax.set_ylabel('Total amplifier current (pA)')
    ax.grid(alpha=.2)
fig.suptitle('Human PAX6 paired +60 mV pulses | no drift correction | onset artifacts retained above')
baseline_fig.suptitle('Initial and final holding near -20 mV | all baseline-window samples retained')
fig.savefig(root / 'paired-currents.png', dpi=150)
baseline_fig.savefig(root / 'baseline-windows.png', dpi=150)
plt.close('all')
