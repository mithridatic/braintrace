"""Render individual human test-pulse responses and every sweep's residuals."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parent
data = np.load(out/'observations.npz')
result = json.loads((out/'result.json').read_text())
t = data['relative_ms']
model = data['prediction_mv']
selected = [0, 1, 14, 15, 16, 29]
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
for ax, sweep in zip(axes.flat, selected):
    baseline = result['per_sweep'][sweep]['baseline_mv']
    ax.plot(t, data[f's{sweep}_raw_mv'], lw=.6, label='Original human voltage')
    ax.plot(t, model+baseline, lw=1.4, color='tab:orange', label='Frozen circuit prediction')
    ax.axhline(baseline, color='gray', ls=':', label='Pre-pulse baseline')
    ax.axvline(200, color='gray', lw=.5)
    ax.set_title(f'Sweep {sweep}: {result["per_sweep"][sweep]["group"]}')
    ax.set_ylabel('Absolute voltage (mV)')
    ax.set_xlabel('Time from command onset (ms)')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Human postsynaptic response to -10 pA, 0-200 ms; all original samples')
fig.tight_layout()
fig.savefig(out/'absolute-responses.png', dpi=140)

fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
for ax, sweep in zip(axes.flat, selected):
    baseline = result['per_sweep'][sweep]['baseline_mv']
    ax.plot(t, data[f's{sweep}_raw_mv']-baseline, lw=.6, label='Human minus pre-pulse baseline')
    ax.plot(t, model, lw=1.4, color='tab:orange', label='Frozen circuit prediction')
    ax.axvline(200, color='gray', lw=.5)
    ax.set_title(f'Sweep {sweep}: {result["per_sweep"][sweep]["group"]}')
    ax.set_ylabel('Voltage change (mV)')
    ax.set_xlabel('Time from command onset (ms)')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Same circuit in every sweep; no response-derived offsets or sample exclusion')
fig.tight_layout()
fig.savefig(out/'centered-responses.png', dpi=140)

residuals = np.stack([data[f's{s}_raw_mv']-result['per_sweep'][s]['baseline_mv']-model
                      for s in range(30)])
limit = np.abs(residuals).max()
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for ax, group, offset in zip(axes, ['Calibration', 'Validation'], [0, 15]):
    im = ax.imshow(residuals[offset:offset+15], origin='lower', aspect='auto',
                   extent=[t[0], t[-1], offset-.5, offset+14.5],
                   cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
    ax.set_title(group)
    ax.set_yticks(range(offset, offset+15))
    ax.set_ylabel('Original sweep')
    ax.axvline(200, color='black', lw=.6)
    fig.colorbar(im, ax=ax, label='Human minus circuit (mV)')
axes[-1].set_xlabel('Time from command onset (ms)')
fig.suptitle('All 30 residual trajectories; shared scale retains extrema')
fig.tight_layout()
fig.savefig(out/'all-residuals.png', dpi=140)

landmarks = []
for sweep in selected:
    for phase in [0, 1, 5, 20, 100, 199.98, 200, 205, 220, 300, 400]:
        index = int(round(phase/.02))
        landmarks.append(dict(sweep=sweep, phase_ms=float(t[index]),
                               source_index=int(data['pulse_source_indices'][index]),
                               human_mv=float(data[f's{sweep}_raw_mv'][index]),
                               baseline_mv=result['per_sweep'][sweep]['baseline_mv'],
                               prediction_change_mv=float(model[index])))
(out/'direct-samples.json').write_text(json.dumps(landmarks, indent=2)+'\n', encoding='utf-8')
print('Rendered absolute, centered and all-residual views; retained 66 direct phase samples.')
