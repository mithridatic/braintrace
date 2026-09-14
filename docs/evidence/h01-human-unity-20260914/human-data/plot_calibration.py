"""Render the three exported calibration observations without downsampling."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(3, 3, figsize=(13, 8), constrained_layout=True)
for column, sweep in enumerate((4, 14, 16)):
    prefix = root/'calibration-r2'/f'sweep-{sweep}'
    meta = json.loads(prefix.with_suffix('.json').read_bytes())
    with np.load(prefix.with_suffix('.npz')) as data:
        time = data['time_ms']
        voltage = data['recorded_voltage_mv']
        command = data['command_current_pa']
        holding = float(data['holding_current_pa'])
        axes[0, column].plot(time, voltage, linewidth=.55, color='#183b56')
        axes[0, column].axvspan(meta['last_recorded_time_ms'], 6600,
                               color='#dddddd', alpha=.5, label='No recorded voltage')
        axes[0, column].set(title=f'Sweep {sweep}', xlim=(0, 6600), ylim=(-85, 40),
                            ylabel='Recorded voltage (mV)', xlabel='Sweep time (ms)')
        axes[1, column].plot(time, command, linewidth=.7, label='NWB stimulus')
        axes[1, column].axhline(holding, linewidth=.8, color='#ac3d25',
                               linestyle='--', label='Amplifier holding current')
        axes[1, column].set(xlim=(0, 3000), ylim=(-40, 130), ylabel='Command current (pA)',
                            xlabel='Sweep time (ms)')
        axes[2, column].plot(time, voltage, linewidth=.7, color='#183b56')
        axes[2, column].set(xlim=(580, 750), ylim=(-85, 40), ylabel='Recorded voltage (mV)',
                            xlabel='Sweep time (ms); pulse starts at 600 ms')
    for row in (0, 1):
        axes[row, column].axvspan(600, 1600, color='#76b7b2', alpha=.12)
    for row in range(3):
        axes[row, column].grid(alpha=.15)
axes[0, 0].legend(fontsize=7, loc='upper right')
axes[1, 0].legend(fontsize=7, loc='upper right')
fig.suptitle('Human L1 specimen 811953283: calibration data, recorded voltage reference\n'
             'All samples retained; gray indicates trailing storage padding. Command is not ionic current.',
             fontsize=12)
fig.savefig(root/'calibration-observations.png', dpi=180)
plt.close(fig)
