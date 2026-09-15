"""Plot source-paired responses, absent-spike windows and all residuals."""

from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parent
data = np.load(out/'observations.npz')
candidate = json.loads((out/'frozen-candidate.json').read_text())
strength = candidate['strength']['peak_pa']
t = data['relative_ms']
shown = np.concatenate([data[f's{s}_post_mv'][[0, 1, 4, 5]].ravel()
                        for s in [0, 1, 14, 15, 16, 29]])
centered_shown = np.concatenate([(data[f's{s}_post_mv']-data[f's{s}_baseline_mv'][:, None])[[0, 1, 4, 5]].ravel()
                                 for s in [0, 1, 14, 15, 16, 29]])
for sweeps, name in [([0, 1, 14], 'calibration'), ([15, 16, 29], 'validation')]:
    fig, axes = plt.subplots(6, 4, figsize=(16, 16), sharex=True)
    for row, sweep in enumerate(sweeps):
        for col, slot in enumerate([0, 1, 4, 5]):
            raw = data[f's{sweep}_post_mv'][slot]
            base = data[f's{sweep}_baseline_mv'][slot]
            pred = strength*data[f's{sweep}_unit_prediction_mv_per_pa'][slot]
            ax, centered = axes[row*2, col], axes[row*2+1, col]
            ax.plot(t, raw, lw=.7, label='Recorded postsynaptic')
            ax.plot(t, base+pred, lw=.9, label='Prediction + measured baseline')
            ax.set_ylim(shown.min()-.1, shown.max()+.1)
            centered.plot(t, raw-base, lw=.7, label='Recorded, pre-baseline subtracted')
            centered.plot(t, pred, lw=.9, label='Frozen template, new strength')
            centered.set_ylim(centered_shown.min()-.1, centered_shown.max()+.1)
            label = 'NO SPIKE; epoch-clock anchor' if slot == 4 else 'recorded spike anchor'
            ax.set_title(f'Sweep {sweep}, slot {slot}: {label}', fontsize=9)
            ax.set_ylabel('Raw post (mV)'); centered.set_ylabel('Change (mV)')
            centered.set_xlabel('Time from labeled anchor (ms)')
            for panel in [ax, centered]:
                panel.axvline(0, lw=.5, color='gray'); panel.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f'{name.capitalize()}: original full windows, common voltage scales')
    fig.tight_layout(); fig.savefig(out/f'{name}-responses.png', dpi=120)

residual = np.stack([data[f's{s}_post_mv'][:, 500:]-data[f's{s}_baseline_mv'][:, None]
                     -strength*data[f's{s}_unit_prediction_mv_per_pa'][:, 500:] for s in range(30)])
limit = abs(residual).max()
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
for ax, offset in zip(axes, [0, 15]):
    im = ax.imshow(residual[offset:offset+15].reshape(90, -1), origin='lower', aspect='auto',
                   extent=[0, 50, -.5, 89.5], cmap='RdBu_r', vmin=-limit, vmax=limit,
                   interpolation='nearest')
    ax.set_yticks(np.arange(15)*6+2.5, labels=[str(s) for s in range(offset, offset+15)])
    ax.set_xlabel('Time from labeled anchor (ms)')
    ax.set_ylabel('Sweep; six rows each, slot 4 has no recorded spike')
    fig.colorbar(im, ax=ax, label='Human minus prediction (mV)')
fig.tight_layout(); fig.savefig(out/'all-slot-residuals.png', dpi=120)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
for ax, sweep in zip(axes.flat, [0, 1, 14, 15, 16, 29]):
    raw = data[f's{sweep}_electrical_raw_mv']
    baseline = data[f's{sweep}_electrical_baseline_samples_mv'].mean()
    ax.plot(data['electrical_relative_ms'], raw-baseline, lw=.6, label='Recorded response')
    ax.plot(data['electrical_relative_ms'], data['electrical_prediction_mv'], lw=1, label='Frozen electrical model')
    ax.set_title(f'Sweep {sweep}'); ax.set_xlabel('Time from -10 pA Step onset (ms)')
    ax.set_ylabel('Voltage change (mV)'); ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out/'electrical-transfer.png', dpi=120)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
for ax, sweep in zip(axes.flat, [0, 1, 14, 15, 16, 29]):
    for slot in [0, 1, 4, 5]:
        ax.plot(t, data[f's{sweep}_pre_mv'][slot], lw=.7,
                label=f'Slot {slot}'+(' (no spike)' if slot == 4 else ''))
    ax.set_title(f'Sweep {sweep}'); ax.set_xlabel('Time from labeled anchor (ms)')
    ax.set_ylabel('Recorded presynaptic IN3 (mV)'); ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out/'presynaptic-inputs.png', dpi=120)
