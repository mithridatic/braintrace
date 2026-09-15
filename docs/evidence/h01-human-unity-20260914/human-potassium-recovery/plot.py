"""Plot every observed recovery curve and predictions from the other group."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
result = json.loads((root/'result.json').read_bytes())
by_family = {family:{p['filename']:p for f in result['folds'] if f['family']==family for p in f['predictions']}
             for family in ('single','double')}
fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True)
resfig, resaxes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True)
for ax, rx, (name, p) in zip(axes.flat, resaxes.flat, sorted(by_family['double'].items()), strict=True):
    t = p['times_ms']
    ax.plot(t, p['observed_ratio'], 'ko-', markersize=3, label='Observed human ratio')
    for family, color in [('single','tab:orange'),('double','tab:blue')]:
        q = by_family[family][name]
        ax.plot(t, q['predicted_ratio'], '.--', color=color, label=family+'; other group fit')
        rx.plot(t, q['residual_ratio'], '.-', color=color, label=family)
    rx.axhline(0, color='k', linewidth=.5)
    for a in (ax,rx):
        a.set_xscale('log')
        a.set_title(name.removesuffix('.nwb'),fontsize=9)
        a.grid(alpha=.2)
    ax.set_ylim(0, 1.1)
    rx.set_ylim(-.45, .45)
axes.flat[0].legend(fontsize=6)
resaxes.flat[0].legend(fontsize=7)
for f, title, ylabel, filename in (
    (fig,'Human potassium recovery at commanded -80 mV, 34 C\nGrouped cross-validation; normalized source amplitudes, no original current traces',
     'Current amplitude / initial amplitude','grouped-predictions.png'),
    (resfig,'Residuals for every human recording: prediction minus observed ratio\nNo per-record amplitude adjustment; identical scales across recording groups',
     'Signed amplitude-ratio residual','grouped-residuals.png')):
    f.suptitle(title, fontsize=12)
    f.supxlabel('Recovery delay (ms, logarithmic scale)')
    f.supylabel(ylabel)
    f.tight_layout(rect=(.02,.02,1,.93))
    f.savefig(root/filename, dpi=150)
    plt.close(f)
print('Saved direct predictions and residuals for 12 human records.')
