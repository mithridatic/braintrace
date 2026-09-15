"""Inspect early controls, independent gain predictions and the changing leak trace."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify':False,'font.size':9})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inventory=json.loads((root/'inventory.json').read_bytes())
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
control_limits=[]
for col,(name,sweeps) in enumerate([('Total protocols',range(70,79)),('Long leak controls',range(79,89)),('Conditioning protocols',range(89,98))]):
    for sweep in sweeps:
        m=json.loads((root/f'control-{sweep}.json').read_bytes())
        assert sha(root/f'control-{sweep}.npz')==m['arrays_sha256']
        with np.load(root/f'control-{sweep}.npz') as a:
            t=a['phase_ms'];i=a['centered_current_pa']
            control_limits.extend([float(i.min()),float(i.max())])
            axes[0,col].plot(t,i,lw=.5,label=str(sweep))
            w=(t>=2)&(t<10.04)
            axes[1,col].plot(t[w],i[w],lw=.6)
    axes[0,col].set_title(name);axes[0,col].legend(ncol=3,fontsize=7)
    axes[1,col].set_ylim(-2,16)
for ax in axes[0]:ax.set_ylim(min(control_limits)*1.05,max(control_limits)*1.05)
for ax in axes.flat:
    ax.set_xlabel('Phase after 45 ms command onset (ms)')
    ax.set_ylabel('Current minus initial baseline (pA)');ax.grid(alpha=.2)
fig.suptitle('All 28 +10 mV controls | original samples and return artifacts | no membrane conductance claim')
fig.savefig(root/'all-controls.png',dpi=140);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
rfig,rax=plt.subplots(figsize=(12,7),constrained_layout=True)
residuals=[]
gain_limits=[]
for sweep in range(89,98):
    m=json.loads((root/f'gain-72-{sweep}.json').read_bytes())
    assert sha(root/f'gain-72-{sweep}.npz')==m['arrays_sha256']
    with np.load(root/f'gain-72-{sweep}.npz') as a:
        post=a['main_phase_ms']>=5
        residuals.append(a['main_residual_pa'][post])
        if sweep in (89,93,97):
            col=(sweep-89)//4
            cp=a['control_phase_ms'];cw=a['control_fit_mask']
            gain_limits.extend([float(a['control_target_pa'][cw].min()),float(a['control_target_pa'][cw].max()),
                                float(a['control_predicted_pa'][cw].min()),float(a['control_predicted_pa'][cw].max())])
            axes[0,col].plot(cp[cw],a['control_target_pa'][cw],lw=.7,label='Target control')
            axes[0,col].plot(cp[cw],a['control_predicted_pa'][cw],lw=.7,label='Gain x control 72')
            mp=a['main_phase_ms'][post]
            axes[1,col].plot(mp,a['main_target_pa'][post],lw=.7,label='Target main response')
            axes[1,col].plot(mp,a['main_predicted_pa'][post],lw=.7,label='Independent prediction')
            axes[0,col].set_title(f'Target {sweep}; control-fitted gain {m["gain"]:.3f}')
            axes[1,col].set_ylim(10,90)
            axes[0,col].set_xlabel('Small pulse phase (ms)');axes[1,col].set_xlabel('Main response phase (ms)')
for ax in axes[0]:ax.set_ylim(min(-10.,min(gain_limits)*1.05),max(gain_limits)*1.05)
for ax in axes.flat:
    ax.set_ylabel('Baseline-centered current (pA)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Gain fitted only on the small pulse | prediction at the same -20 mV command')
fig.savefig(root/'gain-predictions.png',dpi=140);plt.close(fig)
residuals=np.asarray(residuals)
limit=float(np.max(abs(residuals)))
im=rax.imshow(residuals,aspect='auto',interpolation='nearest',cmap='coolwarm',vmin=-limit,vmax=limit,
              extent=[5.,99.96,8.5,-.5])
rax.set_yticks(range(9),list(range(89,98)));rax.set_ylabel('Target conditioning sweep')
rax.set_xlabel('Original phase after main command onset (ms)')
rax.set_title('Observed minus control-gain prediction | all nine targets, all post-5 ms samples')
rfig.colorbar(im,ax=rax,label='Current residual (pA)')
rfig.savefig(root/'all-gain-residuals.png',dpi=140);plt.close(rfig)

fig,axes=plt.subplots(3,3,figsize=(15,10),constrained_layout=True)
observations=[]
leak_limits=[[],[]]
for col,sweep in enumerate((79,80,81)):
    c=json.loads((root/f'control-{sweep}.json').read_bytes())
    p=root/c['source_arrays']['path'];assert sha(p)==c['source_arrays']['sha256']
    with np.load(p) as a:a=dict(a)
    t,i,v=a['time_ms'],a['total_current_pa'],a['command_voltage_mv']
    for row,(lo,hi) in enumerate(((35.,65.),(900.,2200.),(1105.,2200.))):
        w=(t>=lo)&(t<hi);axes[row,col].plot(t[w],i[w],lw=.55)
        if row<2:leak_limits[row].extend([float(i[w].min()),float(i[w].max())])
        axes[row,col].set_xlabel('Original sweep time (ms)');axes[row,col].set_ylabel('Raw total current (pA)')
        axes[row,col].grid(alpha=.2)
    axes[0,col].set_title(f'Sweep {sweep}; acquisition starts {c["absolute_start_s"]:.3f} s')
    axes[2,col].set_ylim(-25,-5)
    points=[]
    for phase in (900.,1090.,1120.,1200.,1500.,1800.,2090.,2150.):
        k=int(np.flatnonzero(np.isclose(t,phase,rtol=0,atol=1e-8))[0])
        points.append(dict(time_ms=phase,current_pa=float(i[k]),command_mv=float(v[k])))
    observations.append(dict(sweep=sweep,source_arrays=c['source_arrays'],points=points))
for row in (0,1):
    for ax in axes[row]:ax.set_ylim(min(leak_limits[row])*1.05,max(leak_limits[row])*1.05)
fig.suptitle('Adjacent leak controls | complete transients above; magnified late current below')
fig.savefig(root/'leak-transition.png',dpi=140);plt.close(fig)
(root/'leak-observations.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf-8',newline='\n')
