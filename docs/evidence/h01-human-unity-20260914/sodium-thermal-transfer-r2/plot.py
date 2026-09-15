"""Compare frozen sodium predictions with every human response and direct gates."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/'result.json').read_bytes())
fig,axes=plt.subplots(4,4,figsize=(14,9),sharex=True,sharey=True)
for ax,b,c in zip(axes.flat,r['baseline_records'],r['candidate_records'],strict=True):
    ap=np.arange(1,201)
    ax.plot(ap,b['observed'],color='black',linewidth=.8,label='Human')
    ax.plot(ap,b['predicted'],color='tab:orange',label='Published scaling')
    ax.plot(ap,c['predicted'],color='tab:blue',label='Fitted scaling')
    ax.set_title(b['filename'].removesuffix('.nwb')+'\n'+b['split'],fontsize=8)
    ax.set_ylim(0,1.2);ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=7)
fig.suptitle('Human sodium Hwide response: two-scalar candidate rejected\nOne calibration group; two separate validation groups; no per-record gain')
fig.supxlabel('AP number');fig.supylabel('Sodium peak / AP1 peak')
fig.tight_layout(rect=(.02,.02,1,.93));fig.savefig(root/'all-human-predictions.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(12,10),sharex='col',sharey='row')
for name,color in [('baseline','tab:orange'),('candidate','tab:blue')]:
    a=np.load(root/(name+'-direct.npz'))
    peak1=max(a['ap1_inward_proxy_right'].max(),a['ap1_inward_proxy_left'].max())
    for col,ap in enumerate((1,5,200)):
        t=a[f'ap{ap}_time_ms'];t=t-t[125]
        if name=='baseline':axes[0,col].plot(t,a[f'ap{ap}_voltage_mv'],color='black')
        axes[1,col].plot(t,a[f'ap{ap}_inward_proxy_right']/peak1,color=color,label=name)
        axes[2,col].plot(t,a[f'ap{ap}_m'],color=color)
        axes[3,col].plot(t,a[f'ap{ap}_h'],color=color)
        axes[0,col].set_title(f'AP {ap}')
        for ax in axes[:,col]:ax.grid(alpha=.2)
        axes[3,col].set_xlabel('ms relative to source voltage peak')
axes[0,0].set_ylabel('Command (mV)')
axes[1,0].set_ylabel('Inward proxy / AP1 peak')
axes[2,0].set_ylabel('Activation m');axes[3,0].set_ylabel('Availability h')
axes[1,0].legend(fontsize=8)
fig.suptitle('Identical command; measured-effect hypothesis changes gate dynamics\nProxy current and gates are model outputs, not recorded human ionic currents')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/'direct-model-comparison.png',dpi=150);plt.close(fig)
print('Saved all 16 human comparisons and direct AP1/AP5/AP200 model samples.')
