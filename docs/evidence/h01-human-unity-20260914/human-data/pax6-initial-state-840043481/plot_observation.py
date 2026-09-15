"""Show direct current trajectories before and after initial-state observation."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with np.load(root/'observation.npz') as source:
    a = dict(source)
fig, axes = plt.subplots(4,3,figsize=(14,11),sharex='col',sharey='col')
for row, index in enumerate((0,9,15,18)):
    for col, name in enumerate(('first','second','post')):
        ax = axes[row,col]
        phase = a['post_phase_ms'] if name == 'post' else a['pulse_phase_ms']
        ax.plot(phase,a[name+'_current_pa'][index],color='0.65',lw=.5,label='Human total current')
        ax.plot(phase,a[name+'_shared_predicted_pa'][index],color='#c44e52',lw=1.3,label='Shared state')
        ax.plot(phase,a[name+'_conditioned_predicted_pa'][index],color='#0072b2',lw=1.3,label='Observed initial state')
        ax.set_title(f'Sweep {123+index}; gap {a["gaps_ms"][index]:g} ms; {name}')
        ax.set_xlim((0,1000) if name == 'post' else (0,300))
        ax.set_ylim((-140,140) if name == 'post' else ((15,80) if name == 'first' else (0,700)))
        if name == 'first':ax.axvspan(10,60,color='#0072b2',alpha=.08)
        if col == 0:ax.set_ylabel('pA above original baseline')
        if row == 3:ax.set_xlabel('Original phase (ms)')
axes[0,0].legend(fontsize=7)
fig.suptitle('Frozen kinetics: initial-state variation fails to repair second-pulse mismatch\nCommand +60.006 mV during pulses; -19.994 mV during return. Command is not measured membrane voltage.')
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(root/'direct-comparison.png',dpi=150)
plt.close(fig)
