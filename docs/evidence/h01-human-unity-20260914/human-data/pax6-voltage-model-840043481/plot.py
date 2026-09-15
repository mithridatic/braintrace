"""Render every original prediction sample and focused command-aligned views."""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
inputs = json.loads((out / 'training-inputs.json').read_bytes())
with np.load(out / 'training-predictions.npz') as source:
    arrays = dict(source)
records = inputs['records']
fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
fig_r, axes_r = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
for row, record in enumerate(records):
    mask = arrays['protocol_ids'] == row
    t = arrays['time_ms'][mask]
    observed, predicted = arrays['observed_pa'][mask], arrays['filtered_pa'][mask]
    axes.flat[row].plot(t, observed, lw=.6, label='Human amplifier current')
    axes.flat[row].plot(t, predicted, lw=1, label='Filtered model prediction')
    axes.flat[row].set(title=f"Sweep {record['sweep']}: {record['family']}",
                       xlabel='Original sweep time (ms)', ylabel='Baseline-centered current (pA)')
    axes_r.flat[row].plot(t, predicted - observed, lw=.6)
    axes_r.flat[row].axhline(0, color='gray', lw=.5)
    axes_r.flat[row].set(title=f"Sweep {record['sweep']}: RMSE {np.sqrt(np.mean((predicted-observed)**2)):.3f} pA",
                         xlabel='Original sweep time (ms)', ylabel='Model minus human (pA)')
axes.flat[-1].axis('off')
axes_r.flat[-1].axis('off')
axes.flat[0].legend(fontsize=7)
fig.suptitle('All human training responses: every original scoring sample')
fig_r.suptitle('All training residuals: no phase or failed sample removed')
fig.savefig(out / 'all-training.png', dpi=150)
fig_r.savefig(out / 'all-training-residuals.png', dpi=150)
plt.close(fig)
plt.close(fig_r)

fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
for r, sweep in enumerate([70, 78, 103]):
    row = inputs['training_sweeps'].index(sweep)
    for c, (start, stop, title) in enumerate([(44, 70, 'Early control, including onsets'),
                                            (1105, 2090, 'Depolarized phase after 5 ms'),
                                            (2105, 2590, 'Tail after 5 ms')]):
        mask = (arrays['protocol_ids'] == row) & (arrays['time_ms'] >= start) & (arrays['time_ms'] < stop)
        axes[r, c].plot(arrays['time_ms'][mask], arrays['observed_pa'][mask], lw=.6, label='Human')
        axes[r, c].plot(arrays['time_ms'][mask], arrays['filtered_pa'][mask], lw=1, label='Model')
        axes[r, c].set(title=f'Sweep {sweep}: {title}', xlabel='Original sweep time (ms)', ylabel='Current (pA)')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Direct waveform detail; complete onset samples remain in the full views')
fig.savefig(out / 'training-detail.png', dpi=150)
plt.close(fig)

decision = json.loads((out / 'decision.json').read_bytes())
if decision['prediction_responses_opened']:
    with np.load(out / 'frozen-protocol-predictions.npz') as source:
        predictions = dict(source)
    with np.load(out / 'prediction-observations.npz') as source:
        observed = source['observed_pa']
    freeze = json.loads((out / 'prediction-freeze.json').read_bytes())
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig_r, axes_r = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for row, record in enumerate(freeze['records']):
        mask = predictions['protocol_ids'] == row
        t = predictions['time_ms'][mask]
        axes.flat[row].plot(t, observed[mask], lw=.6, label='Previously unopened human response')
        axes.flat[row].plot(t, predictions['filtered_pa'][mask], lw=1, label='Frozen full model')
        axes.flat[row].plot(t, predictions['passive_filtered_pa'][mask], lw=.8, label='Frozen observation-only reference')
        axes.flat[row].set(title=f"Prediction sweep {record['sweep']}", xlabel='Original sweep time (ms)', ylabel='Current (pA)')
        axes_r.flat[row].plot(t, predictions['filtered_pa'][mask] - observed[mask], lw=.6)
        axes_r.flat[row].axhline(0, color='gray', lw=.5)
        axes_r.flat[row].set(title=f"Prediction residual: sweep {record['sweep']}", xlabel='Original sweep time (ms)', ylabel='Model minus human (pA)')
    axes.flat[0].legend(fontsize=7)
    fig.savefig(out / 'all-protocol-predictions.png', dpi=150)
    fig_r.savefig(out / 'all-protocol-residuals.png', dpi=150)
    plt.close(fig)
    plt.close(fig_r)
