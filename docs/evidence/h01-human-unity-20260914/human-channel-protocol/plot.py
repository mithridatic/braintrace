"""Render retained source AP windows and individual human amplitude sequences."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
a=np.load(root/'commands.npz')
t=a['train_time_ms'];v=a['train_original_mv'];peaks=a['train_peak_indices']
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
direct=[]
for ax,number in zip(axes,[1,5,200],strict=True):
    peak=peaks[number-1];offset=t-t[peak];mask=(offset>=-2)&(offset<=4)
    ax.plot(offset[mask],v[mask],color='gray',label='Original waveform')
    ax.plot(offset[mask],v[mask]-10,color='tab:blue',label='Sodium command (-10 mV)')
    ax.set_title(f'AP {number}, source peak {t[peak]:.3f} ms')
    ax.grid(alpha=.2)
    direct.append(dict(ap_number=number,source_peak_index=int(peak),
        source_indices=np.flatnonzero(mask).tolist(),time_ms=t[mask].tolist(),
        original_mv=v[mask].tolist(),sodium_mv=(v[mask]-10).tolist()))
axes[0].legend(fontsize=8)
axes[0].set_ylabel('Voltage (mV)')
fig.supxlabel('Time relative to source voltage peak (ms)')
fig.suptitle('Source voltage samples: command input, not measured ionic current')
fig.tight_layout()
fig.savefig(root/'source-voltage-windows.png',dpi=150)
plt.close(fig)
(root/'direct-voltage-windows.json').write_text(json.dumps(direct,indent=2)+'\n',encoding='utf-8',newline='\n')
records=json.loads((root/'human-hwide-responses.json').read_bytes())['selected']
fig,axes=plt.subplots(4,4,figsize=(14,9),sharex=True,sharey=True)
assert len(records)==len(axes.flat)
for ax,record in zip(axes.flat,records,strict=True):
    y=np.asarray([np.nan if x is None else x for x in record['ratios']])
    ax.plot(np.arange(1,201),y,color='black',linewidth=.8)
    ax.plot([1,5,200],y[[0,4,199]],'o',color='tab:blue',markersize=3)
    ax.set_title(record['filename'].removesuffix('.nwb'),fontsize=8)
    ax.grid(alpha=.2)
    ax.set_ylim(0,1.2)
fig.suptitle('Human sodium responses to Hwide input\nPublished normalized amplitudes; all available AP values, no physiological QC or model fit')
fig.supxlabel('AP number')
fig.supylabel('Relative sodium current amplitude')
fig.tight_layout(rect=(.02,.02,1,.94))
fig.savefig(root/'human-hwide-trajectories.png',dpi=150)
plt.close(fig)
print('Saved all 16 human amplitude trajectories and complete AP1/AP5/AP200 voltage windows.')
