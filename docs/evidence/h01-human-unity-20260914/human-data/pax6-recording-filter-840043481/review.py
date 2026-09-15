"""Verify source membership and render the fixed control predictions."""

from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
prior = out.parent / 'pax6-voltage-model-840043481'
inputs = json.loads((out / 'inputs.json').read_bytes())
candidate = json.loads((out / 'frozen-candidate.json').read_bytes())
decision = json.loads((out / 'decision.json').read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(out / 'inputs.npz') == inputs['observations_sha256']
assert sha(prior / 'training-source.npz') == inputs['original_arrays_sha256']
assert sha(out / 'inputs.json') == candidate['inputs_sha256']
with np.load(out / 'predictions.npz') as f:
    a = dict(f)
with np.load(prior / 'training-source.npz') as raw, np.load(prior / 'training-predictions.npz') as old:
    for row, meta in enumerate(inputs['records']):
        mask = a['protocol_ids'] == row
        index = a['source_indices'][mask]
        np.testing.assert_array_equal(a['observed_pa'][mask], raw[f"s{meta['sweep']}_total_current_pa"][index] - meta['baseline_pa'])
        np.testing.assert_array_equal(a['time_ms'][mask], raw[f"s{meta['sweep']}_time_ms"][index])
        np.testing.assert_array_equal(a['old_model_pa'][mask], old['filtered_pa'][old['protocol_ids'] == row][index - 875])
        assert np.all(a['fitting_mask'][mask] == (meta['sweep'] in inputs['training_sweeps']))
np.testing.assert_array_equal(a['residual_pa'], a['predicted_pa'] - a['observed_pa'])
early = (~a['fitting_mask']) & (a['time_ms'] < 70)
rms = lambda x: float(np.sqrt(np.mean(x**2)))
assert rms(a['residual_pa'][early]) == decision['new_early_rmse_pa']
assert rms(a['old_model_pa'][early] - a['observed_pa'][early]) == decision['old_early_rmse_pa']

fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
fig_zoom, zoom = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
for axis, detail, sweep in zip(axes.flat, zoom.flat, inputs['excluded_from_fit_sweeps']):
    row = next(i for i, r in enumerate(inputs['records']) if r['sweep'] == sweep)
    mask = (a['protocol_ids'] == row) & (a['time_ms'] < 70)
    tight = mask & (a['time_ms'] >= 44.92) & (a['time_ms'] <= 46.)
    for target, selected, origin in [(axis, mask, 0), (detail, tight, 45)]:
        for key, label, width in [('observed_pa', 'Human', .7), ('old_model_pa', 'Previous model', .8),
                                   ('predicted_pa', 'Fixed Bessel correction', 1)]:
            target.plot(a['time_ms'][selected]-origin, a[key][selected], lw=width,
                        marker='o' if origin else None, ms=2, label=label)
        target.set(title=f'Excluded-from-fit control {sweep}', xlabel='Time / onset phase (ms)', ylabel='Current (pA)')
        target.axhline(0, color='gray', lw=.5)
axes[0, 0].legend(fontsize=7)
zoom[0, 0].legend(fontsize=7)
fig.savefig(out / 'excluded-control-predictions.png', dpi=150)
fig_zoom.savefig(out / 'excluded-control-onsets.png', dpi=150)
plt.close(fig)
plt.close(fig_zoom)
row = next(i for i, r in enumerate(inputs['records']) if r['sweep'] == 83)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
for axis, lo, hi in zip(axes, [35, 1099.92, 1110], [3100, 1101, 2090]):
    mask = (a['protocol_ids'] == row) & (a['time_ms'] >= lo) & (a['time_ms'] < hi)
    for key, label in [('observed_pa', 'Human'), ('old_model_pa', 'Previous model'), ('predicted_pa', 'Fixed correction')]:
        axis.plot(a['time_ms'][mask], a[key][mask], lw=.7, label=label)
    axis.set(xlabel='Original sweep time (ms)', ylabel='Current (pA)')
axes[0].legend(fontsize=8)
fig.suptitle('Excluded long -10 mV control 83: complete current and waveform detail')
fig.savefig(out / 'excluded-long-control.png', dpi=150)
plt.close(fig)
long_mask = a['protocol_ids'] == row
assert rms(a['residual_pa'][long_mask]) == decision['new_long_rmse_pa']
verification = dict(status='passed', source_observations_verified=len(a['observed_pa']),
                    training_samples=int(a['fitting_mask'].sum()), excluded_early_samples=int(early.sum()),
                    long_control_samples=int(long_mask.sum()), original_clocks_and_previous_predictions_verified=True,
                    candidate_and_source_hashes_verified=True, independent_validation=False, scores_promoted=False)
with (out / 'verification.json').open('x', encoding='utf-8') as f:
    json.dump(verification, f, indent=2)
print(json.dumps(verification))
