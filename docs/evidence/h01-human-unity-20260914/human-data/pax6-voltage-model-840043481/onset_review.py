"""Retain the measured control-onset timing behind the model's sharp residuals."""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
inputs = json.loads((out / 'training-inputs.json').read_bytes())
records = []
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
shown = [70, 78, 103]
with np.load(out / 'training-source.npz') as source, np.load(out / 'training-predictions.npz') as model:
    for row, meta in enumerate(inputs['records']):
        sweep = meta['sweep']
        phase = np.arange(-2, 26) / 25
        on_indices = 1125 + np.arange(-2, 26)
        off_indices = 1376 + np.arange(-2, 26)
        current = source[f's{sweep}_total_current_pa']
        predicted = model['filtered_pa'][model['protocol_ids'] == row]
        on = current[on_indices] - meta['baseline_pa']
        off_baseline = float(current[1300:1375].mean())
        off = current[off_indices] - off_baseline
        model_on = predicted[on_indices - 875]
        model_off = predicted[off_indices - 875] - float(predicted[1300 - 875:1375 - 875].mean())
        records.append(dict(sweep=sweep, phase_ms=phase.tolist(),
                            on_source_indices=on_indices.tolist(), off_source_indices=off_indices.tolist(),
                            human_on_pa=on.tolist(), human_off_pa=off.tolist(),
                            model_on_pa=model_on.tolist(), model_off_pa=model_off.tolist(),
                            off_baseline_pa=off_baseline))
        if sweep in shown:
            column = shown.index(sweep)
            for axis, human, prediction, name in [(axes[0, column], on, model_on, 'on'),
                                                   (axes[1, column], off, model_off, 'off')]:
                axis.plot(phase, human, 'o-', ms=3, lw=.8, label='Original human samples')
                axis.plot(phase, prediction, 'o-', ms=3, lw=.8, label='Model')
                axis.axvline(0, color='black', lw=.7, label='Command transition')
                axis.set(title=f'Sweep {sweep}, control {name}', xlabel='Phase from command transition (ms)',
                         ylabel='Current relative to pre-transition level (pA)')
                axis.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.suptitle('Control transient shape mismatch: direct review after the failed fit')
fig.savefig(out / 'control-onset-detail.png', dpi=150)
plt.close(fig)
with (out / 'control-onset-observations.json').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(dict(records=records, interpretation='post-fit diagnostic; no parameter update',
                                new_response_access=False), indent=2) + '\n')
print(json.dumps(dict(records=len(records), onset_samples_per_transition=len(phase))))
