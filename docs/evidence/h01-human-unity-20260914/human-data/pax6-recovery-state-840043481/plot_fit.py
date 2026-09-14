"""Inspect both fitted pulses, holding return and every condition's residuals."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify':False,'font.size':9})
m=json.loads((root/'joint-fit.json').read_bytes())
assert hashlib.sha256((root/'joint-fit.npz').read_bytes()).hexdigest()==m['arrays_sha256']
with np.load(root/'joint-fit.npz') as a:a=dict(a)
fig,axes=plt.subplots(3,4,figsize=(17,11),constrained_layout=True)
for col,gap in enumerate((5.,50.,1000.,4000.)):
    index=int(np.flatnonzero(a['gaps_ms']==gap)[0])
    for row,name in enumerate(('first','second','post')):
        phase=a['post_phase_ms'] if name=='post' else a['pulse_phase_ms']
        axes[row,col].plot(phase,a[name+'_current_pa'][index],lw=.55,label='Original current minus initial baseline')
        axes[row,col].plot(phase,a[name+'_predicted_pa'][index],lw=1.1,label='Shared state candidate')
        axes[row,col].set_title(f'{name.capitalize()} response | recovery gap {gap:g} ms')
        axes[row,col].set_xlabel('Original phase after command transition (ms)')
        axes[row,col].set_ylabel('Baseline-centered current (pA)')
        axes[row,col].set_ylim((10,70) if name=='first' else ((0,750) if name=='second' else (-140,140)))
        axes[row,col].grid(alpha=.2)
axes[0,0].legend(fontsize=7)
fig.suptitle('One within-sweep state candidate | both original pulses and the measured holding return')
fig.savefig(root/'joint-state-fit.png',dpi=140);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(17,8),constrained_layout=True)
limit=max(float(np.abs(a[name+'_residual_pa']).max()) for name in ('first','second','post'))
for ax,name in zip(axes,('first','second','post')):
    phase=a['post_phase_ms'] if name=='post' else a['pulse_phase_ms']
    im=ax.imshow(a[name+'_residual_pa'],aspect='auto',interpolation='nearest',cmap='coolwarm',vmin=-limit,vmax=limit,
                 extent=[phase[0],phase[-1],18.5,-.5])
    ax.set_yticks(range(19),[f'{x:g}' for x in a['gaps_ms']])
    ax.set_xlabel('Original phase (ms)');ax.set_ylabel('Recovery gap (ms)')
    ax.set_title(name.capitalize()+' observed minus predicted')
fig.colorbar(im,ax=axes,label='Current residual (pA)')
fig.suptitle('All 19 conditions and all original fit samples | one common residual scale')
fig.savefig(root/'all-state-residuals.png',dpi=140);plt.close(fig)

points=[]
for row,gap in enumerate(a['gaps_ms']):
    samples=[]
    for name in ('first','second','post'):
        phase=a['post_phase_ms'] if name=='post' else a['pulse_phase_ms']
        for value in ((20.,60.,200.,600.,900.) if name=='post' else (20.,60.,200.)):
            k=int(np.flatnonzero(np.isclose(phase,value,atol=1e-8,rtol=0))[0])
            samples.append(dict(window=name,phase_ms=value,observed_pa=float(a[name+'_current_pa'][row,k]),
                                predicted_pa=float(a[name+'_predicted_pa'][row,k]),residual_pa=float(a[name+'_residual_pa'][row,k])))
    points.append(dict(sweep=123+row,gap_ms=float(gap),samples=samples))
(root/'direct-phase-observations.json').write_text(json.dumps(points,indent=2)+'\n',encoding='utf-8',newline='\n')

source_path=root.parent/'pax6-recovery-840043481/sweep-141.npz'
source_meta=json.loads(source_path.with_suffix('.json').read_bytes())
assert hashlib.sha256(source_path.read_bytes()).hexdigest()==source_meta['arrays_sha256']
with np.load(source_path) as original:original=dict(original)
end=source_meta['second_onset_ms']+300.
phase=original['source_time_ms']-end
window=(phase>=120)&(phase<155)
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
axes[0].plot(original['source_time_ms'][window],original['source_total_current_pa'][window],lw=.7)
axes[0].set_ylabel('Raw amplifier current (pA)')
axes[1].plot(original['source_time_ms'][window],original['source_command_voltage_mv'][window],lw=.8)
axes[1].set_ylabel('Amplifier command (mV)');axes[1].set_ylim(-21,-19)
axes[1].set_xlabel('Original sweep time (ms)')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Sweep 141 holding-return excursion | original samples retained | no cause assigned')
fig.savefig(root/'holding-excursion.png',dpi=140);plt.close(fig)
