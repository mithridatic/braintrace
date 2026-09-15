"""Review joint predictions, baseline sensitivity and every gap's residuals."""

from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify': False, 'font.size': 10})
data = {}
for name in ['unadjusted', 'baseline-sensitivity']:
    p = root / f'joint-{name}'
    m = json.loads(p.with_suffix('.json').read_bytes())
    assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest() == m['arrays_sha256']
    with np.load(p.with_suffix('.npz')) as a: data[name] = dict(a)
    data[name]['metadata'] = m
u, s = data['unadjusted'], data['baseline-sensitivity']
fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
for col, gap in enumerate([5, 50, 1000, 4000]):
    row = int(np.flatnonzero(u['gaps_ms'] == gap)[0])
    t = u['phase_ms']
    axes[0, col].plot(t, u['difference_pa'][row], lw=.55, label='Observed difference')
    axes[0, col].plot(t, u['predicted_pa'][row], lw=1.2, label='Unadjusted fit')
    axes[0, col].plot(t, s['predicted_pa'][row] + s['metadata']['baseline_changes_pa'][row],
                       lw=1.2, ls='--', label='Sensitivity fit + baseline change')
    axes[0, col].set_ylim(-20, 700)
    axes[0, col].set_title(f'Recovery gap {gap:g} ms')
    axes[0, col].set_ylabel('Second minus first current (pA)')
    axes[1, col].plot(t, u['residual_pa'][row], lw=.55)
    axes[1, col].axhline(0, color='black', lw=.7)
    axes[1, col].set_ylim(-100, 100)
    axes[1, col].set_ylabel('Observed minus unadjusted fit (pA)')
axes[0, 0].legend(fontsize=8)
for ax in axes.flat:
    ax.set_xlabel('Pulse phase (ms)')
    ax.grid(alpha=.2)
fig.suptitle('Conditional human PAX6 recovery fit | initial availability included | not physiological qualification')
fig.savefig(root / 'joint-fit.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
extent = max(abs(u['residual_pa'].min()), abs(u['residual_pa'].max()))
im = ax.imshow(u['residual_pa'], aspect='auto', interpolation='nearest', cmap='coolwarm',
               vmin=-extent, vmax=extent,
               extent=[u['phase_ms'][0], u['phase_ms'][-1], len(u['gaps_ms'])-.5, -.5])
ax.set_yticks(range(len(u['gaps_ms'])), [f'{v:g}' for v in u['gaps_ms']])
ax.set_ylabel('Recovery gap (ms); rows preserve condition identity')
ax.set_xlabel('Pulse phase (ms)')
ax.set_title('All 19 conditions: observed minus unadjusted prediction')
fig.colorbar(im, ax=ax, label='Residual current (pA)')
fig.savefig(root / 'all-residuals.png', dpi=150)
plt.close(fig)
