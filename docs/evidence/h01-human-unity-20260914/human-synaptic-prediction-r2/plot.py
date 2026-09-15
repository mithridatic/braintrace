"""Render individual human EPSPs, current predictions and retained discrepancies."""

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
selected = [0, 1, 14, 15, 16, 29]
shown = [data[f's{s}_post_mv'][e]-data[f's{s}_baseline_mv'][e]
         for s in selected for e in [0, 1, 5]]
low, high = min(v.min() for v in shown), max(v.max() for v in shown)
for sweeps, name in [([0, 1, 14], 'calibration-events'), ([15, 16, 29], 'validation-events')]:
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
    for row, sweep in enumerate(sweeps):
        for col, event in enumerate([0, 1, 5]):
            ax = axes[row, col]
            ax.plot(t, data[f's{sweep}_post_mv'][event]-data[f's{sweep}_baseline_mv'][event],
                    lw=.7, label='Human, fixed pre-event baseline subtracted')
            ax.plot(t, data[f's{sweep}_prediction_mv'][event], color='tab:orange', lw=1.2,
                    label='Frozen synaptic + cell prediction')
            ax.axvline(0, color='gray', lw=.5)
            ax.axhline(0, color='gray', ls=':', lw=.5)
            ax.set_ylim(low-.1, high+.1)
            ax.set_title(f'Sweep {sweep}, event {event}, source {data[f"s{sweep}_accepted_crossings"][event]}')
            ax.set_xlabel('Time from recorded presynaptic crossing (ms)')
            ax.set_ylabel('Postsynaptic voltage change (mV)')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(name.replace('-', ' ').capitalize()+': original samples, shared scales')
    fig.tight_layout(); fig.savefig(out/(name+'.png'), dpi=140)

residuals = np.stack([data[f's{s}_post_mv'][:, 500:]-data[f's{s}_baseline_mv'][:, None]
                      - data[f's{s}_prediction_mv'][:, 500:] for s in range(30)])
limit = np.abs(residuals).max()
fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharex=True)
for ax, group, offset in zip(axes, ['Calibration', 'Validation'], [0, 15]):
    im = ax.imshow(residuals[offset:offset+15].reshape(90, -1), origin='lower', aspect='auto',
                   extent=[0, 50, -.5, 89.5], cmap='RdBu_r', vmin=-limit, vmax=limit,
                   interpolation='nearest')
    ax.set_title(group)
    ax.set_yticks(np.arange(15)*6+2.5, labels=[str(s) for s in range(offset, offset+15)])
    ax.set_ylabel('Original sweep; six event rows per sweep, ordinals 0-5')
    ax.set_xlabel('Time from presynaptic crossing (ms)')
    fig.colorbar(im, ax=ax, label='Human minus prediction (mV)')
fig.suptitle('All 180 event residuals; full extrema retained')
fig.tight_layout(); fig.savefig(out/'all-event-residuals.png', dpi=140)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(t, data['s0_inferred_current_pa'][0])
axes[0].set_ylabel('Inferred effective current (pA)')
axes[1].plot(t, data['s0_prediction_mv'][0], color='tab:orange')
axes[1].set_ylabel('Predicted voltage change (mV)')
axes[1].set_xlabel('Time from recorded presynaptic crossing (ms)')
for ax in axes: ax.grid(alpha=.2)
fig.suptitle('Frozen candidate, sweep 0/event 0; current is inferred, not measured')
fig.tight_layout(); fig.savefig(out/'inferred-current-and-voltage.png', dpi=140)

original = np.load(out.parent/'human-synaptic-prediction/unexpected-crossing.npz')
idx = original['source_indices']
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for sweep in [0, 2]:
    axes[0].plot(original[f's{sweep}_time_ms'], original[f's{sweep}_pre_mv'], label=f'Sweep {sweep}', lw=.8)
    axes[2].plot(original[f's{sweep}_time_ms'], original[f's{sweep}_post_mv'], label=f'Sweep {sweep}', lw=.8)
axes[1].plot(idx*.02, data['prior_parser_command_pa'][idx], label='Prior parser reconstruction', ls='--')
axes[1].plot(idx*.02, data['reconstructed_command_pa'][idx], label='Corrected source-epoch reconstruction')
for ax, label in zip(axes, ['Presynaptic voltage (mV)', 'Reconstructed DAC0 command (pA)', 'Postsynaptic voltage (mV)']):
    ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(fontsize=8)
axes[0].axhline(0, color='gray', lw=.5)
axes[-1].set_xlabel('Original within-sweep time (ms)')
fig.suptitle('Fifth pulse restored from source epochs; original voltages unchanged')
fig.tight_layout(); fig.savefig(out/'corrected-command.png', dpi=140)

points = []
for sweep in selected:
    for event in [0, 1, 5]:
        for phase in [-5, 0, 2, 5, 20, 50]:
            i = int(round((phase+10)/.02))
            points.append(dict(sweep=sweep, event=event, phase_ms=phase,
                               source_index=int(data[f's{sweep}_indices'][event, i]),
                               pre_mv=float(data[f's{sweep}_pre_mv'][event, i]),
                               post_mv=float(data[f's{sweep}_post_mv'][event, i]),
                               baseline_mv=float(data[f's{sweep}_baseline_mv'][event]),
                               predicted_change_mv=float(data[f's{sweep}_prediction_mv'][event, i]),
                               inferred_current_pa=float(data[f's{sweep}_inferred_current_pa'][event, i])))
(out/'direct-samples.json').write_text(json.dumps(points, indent=2)+'\n', encoding='utf-8')
print('Rendered five direct/residual figures and retained 108 phase observations.')
