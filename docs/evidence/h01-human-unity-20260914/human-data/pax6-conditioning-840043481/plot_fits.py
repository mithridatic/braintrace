"""Inspect original differences and conditional decay residuals at every voltage."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify':False,'font.size':9})
rows=json.loads((root/'inventory.json').read_bytes())
fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
rfig,raxes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
for row,ax,rx in zip(rows,axes.flat,raxes.flat):
    for scenario in ('unadjusted','baseline-sensitivity'):
        prefix=root/f'fit-{row["total"]}-{row["conditioned"]}-{scenario}'
        m=json.loads(prefix.with_suffix('.json').read_bytes())
        assert hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest()==m['arrays_sha256']
        with np.load(prefix.with_suffix('.npz')) as a:
            x=a['phase_ms']
            if scenario=='unadjusted':
                ax.plot(x,a['difference_pa'],lw=.55,label='Original difference')
                rx.plot(x,a['residual_pa'],lw=.55,label='Unadjusted residual')
            ax.plot(x,a['predicted_pa']+m['subtracted_baseline_difference_pa'],lw=1,label=scenario)
    for axis in (ax,rx):
        axis.axhline(0,color='black',lw=.5)
        axis.set_title(f'{row["test_command_mv"]:.3f} mV | {row["total"]}, {row["conditioned"]}')
        axis.set_xlabel('Original test phase (ms)'); axis.grid(alpha=.2)
    ax.set_ylim(-100,260);rx.set_ylim(-100,100)
    ax.set_ylabel('Total minus conditioned current (pA)');rx.set_ylabel('Observed minus fit (pA)')
axes[0,0].legend(fontsize=8)
fig.suptitle('Two positive decays without free offset | conditional diagnostic, not rate identification')
rfig.suptitle('All conditioning fit residuals | original samples retained')
fig.savefig(root/'fit-review.png',dpi=130);rfig.savefig(root/'fit-residuals.png',dpi=130)
plt.close('all')
