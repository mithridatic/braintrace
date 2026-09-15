"""Preserve five human calibration tail responses and source-bound observations."""

from pathlib import Path
import hashlib
import json
import sys
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_l1_voltage_clamp import read_sweep

start = time.perf_counter()
source = repo / '.cache/human-source-followup/pax6-840043481.nwb'
prior = out.parent / 'pax6-recovery-state-840043481/next-protocol-commands.json'
inventory = json.loads(prior.read_bytes())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if (out / 'observations.json').exists():
    raise FileExistsError('Use a new evidence prefix.')
records, arrays, checked = [], {}, 0
fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
windows = [(35, 70, 0, 'Early control: raw current'),
           (1100, 2100, 1100, 'Common +70 mV conditioning: raw current'),
           (2100, 2600, 2100, 'Full tail onset: raw current'),
           (2105, 2600, 2100, 'Tail after 5 ms: raw current'),
           (2105, 2600, 2100, 'Tail minus its late mean (not leak corrected)'),
           (2600, 3600, 2600, 'Final phase: raw current; sweep 99 has no transition')]
fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

with h5py.File(source, 'r') as original:
    for reference in inventory:
        sweep = reference['sweep']
        data, meta = read_sweep(source, sweep, session='840043481')
        assert meta['command_segments'] == reference['command_segments']
        assert meta['source_sha256'] == reference['source_sha256']
        assert meta['samples'] == 90000 and meta['sample_rate_hz'] == 25000
        t, current = data['time_ms'], data['total_current_pa']
        # Independent conversion from raw source arrays, including every sample.
        for group, key, scale in [
            ('acquisition', 'total_current_pa', 1e12),
            ('stimulus/presentation', 'dac_voltage_mv', 1e3),
        ]:
            suffix = 'AD0' if group == 'acquisition' else 'DA0'
            raw = original[f'{group}/data_{sweep:05d}_{suffix}/data']
            expected = (raw[()].astype(np.float64) * raw.attrs['conversion']
                        + raw.attrs.get('offset', 0)) * scale
            np.testing.assert_array_equal(data[key], expected)
            checked += expected.size
        np.testing.assert_allclose(t, np.arange(90000) * 0.04, rtol=0, atol=1e-12)
        baseline_mask = (t >= 1000) & (t < 1090)
        late_mask = (t >= 2550) & (t < 2590)
        assert baseline_mask.sum() == 2250 and late_mask.sum() == 1000
        baseline, late = float(current[baseline_mask].mean()), float(current[late_mask].mean())
        phases = np.array([5., 20., 60., 200., 490.])
        indices = ((2100 + phases) * 25).astype(int)
        np.testing.assert_allclose(t[indices], 2100 + phases, rtol=0, atol=1e-10)
        tail_voltage = float(data['command_voltage_mv'][52500])
        row = dict(metadata=meta, baseline_pa=baseline, tail_late_mean_pa=late,
                   tail_command_mv=tail_voltage, landmark_indices=indices.tolist(),
                   landmark_phases_ms=phases.tolist(), raw_pa=current[indices].tolist(),
                   initial_centered_pa=(current[indices] - baseline).tolist(),
                   late_difference_pa=(current[indices] - late).tolist())
        records.append(row)
        for key, values in data.items():
            arrays[f's{sweep}_{key}'] = values
        arrays[f's{sweep}_baseline_indices'] = np.flatnonzero(baseline_mask)
        arrays[f's{sweep}_late_indices'] = np.flatnonzero(late_mask)
        label = f'{sweep}: {tail_voltage:.2f} mV'
        for n, (axis, (a, b, origin, title)) in enumerate(zip(axes.flat, windows)):
            mask = (t >= a) & (t < b)
            values = current - late if n == 4 else current
            axis.plot(t[mask] - origin, values[mask], lw=.7, label=label)
            axis.set(title=title, xlabel='Phase / source time (ms)', ylabel='Current (pA)')
        ax2[0].plot(t, data['command_voltage_mv'], lw=.8, label=label)
        ax2[1].plot(phases, current[indices] - late, 'o-', label=label)

for axis in axes.flat:
    axis.axhline(0, color='gray', lw=.5)
    axis.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.suptitle('Human PAX6 840043506: all five tail recordings, original samples')
fig.savefig(out / 'direct-tails.png', dpi=150)
plt.close(fig)
ax2[0].set(xlabel='Original sweep time (ms)', ylabel='Reconstructed command (mV)',
           title='Command is not measured patch voltage')
ax2[0].legend(fontsize=8)
ax2[1].set(xlabel='Tail phase (ms)', ylabel='Current minus late mean (pA)',
           title='Exact original landmark samples; lines connect observations only')
ax2[1].axhline(0, color='gray', lw=.5)
fig2.savefig(out / 'commands-and-landmarks.png', dpi=150)
plt.close(fig2)
with (out / 'observations.npz').open('xb') as stream:
    np.savez_compressed(stream, **arrays)
result = dict(source_path=str(source.relative_to(repo)), source_sha256=sha(source),
              command_inventory_sha256=sha(prior), records=records,
              response_samples_verified=checked // 2, dac_samples_verified=checked // 2,
              external_response_opened=False, whole_cell_response_opened=False,
              scores_promoted=False, model_parameters_changed=False,
              independent_validation=False, analysis_seconds=time.perf_counter() - start,
              arrays_sha256=sha(out / 'observations.npz'))
with (out / 'observations.json').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'records'}))
for row in records:
    print(json.dumps({key: value for key, value in row.items() if key != 'metadata'}))
