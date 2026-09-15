"""Visualize retained descriptive fits and classify their recorded boundary hits."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify': False, 'font.size': 10})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharex=True)
decisions = []
for row, control in enumerate([79, 88]):
    prefix = root / f'decay-control-{control}'
    metadata = json.loads(prefix.with_suffix('.json').read_bytes())
    digest = hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest()
    assert digest == metadata['arrays_sha256']
    with np.load(prefix.with_suffix('.npz')) as a:
        axes[row, 0].plot(a['time_ms'], a['current_pa'], lw=.45, label='Conditional current')
        axes[row, 0].plot(a['time_ms'], a['fitted_current_pa'], lw=1.5, color='#c74900',
                          label='Descriptive fit')
        axes[row, 1].plot(a['time_ms'], a['residual_pa'], lw=.45)
        axes[row, 1].axhline(0, color='black', lw=.7)
    limited = any(metadata['active_bounds'])
    decisions.append(dict(control=control, source_json_sha256=hashlib.sha256(
        prefix.with_suffix('.json').read_bytes()).hexdigest(), boundary_limited=limited,
        accepted_for_kinetic_deployment=False,
        reason='Descriptive envelope only; offset reaches registered lower bound.'
               if limited else 'Descriptive envelope only; no channel-rate validation.'))
    axes[row, 0].set_title(f'Control {control}: C={metadata["offset_pa"]:.0f} pA at lower bound')
    axes[row, 0].set_ylabel('Current (pA)')
    axes[row, 0].legend()
    axes[row, 1].set_title(f'Residual RMS {metadata["residual_rms_pa"]:.2f} pA')
    axes[row, 1].set_ylabel('Observed minus fit (pA)')
for col in range(2):
    limits = [a.get_ylim() for a in axes[:, col]]
    for a in axes[:, col]: a.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
for a in axes.flat:
    a.set_xlabel('Original sweep time (ms)')
    a.grid(alpha=.2)
fig.suptitle('PAX6 conditional decay | optimizer convergence does not qualify channel kinetics')
fig.savefig(root / 'decay-review.png', dpi=150)
plt.close(fig)
path = root / 'decay-decision.json'
with path.open('x', encoding='utf-8') as f:
    f.write(json.dumps(dict(fits=decisions, external_currents_opened=False,
                           next_requirement='Joint constraints from prepulse and recovery protocols',
                           scores_promoted=False), indent=2) + '\n')
