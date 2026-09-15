"""Inspect human recovery observations before selecting a replacement fit."""

from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
na=json.loads((root/'sodium-human-records.json').read_bytes())['records']
k=json.loads((root/'potassium-human-records.json').read_bytes())['records']
fig,axes=plt.subplots(1,3,figsize=(15,5))
points=[]
for i,record in enumerate(na):
    row={re.sub(r'_\s*(\d+)$',r'_\1',name):value for name,value in record.items()}
    pairs=[]
    for step in range(1,7):
        v,tau,gof=(row.get(prefix+str(step)) for prefix in ('ia_volt_','ia_tau_h_rec_','ia_gof_h_rec_'))
        if v is None or tau is None:continue
        pairs.append((v,tau))
        points.append(dict(channel='sodium',filename=record['Filename'],excel_row=record['excel_row'],
            step=step,voltage_mv=v,tau_ms=tau,gof=gof))
    if pairs:
        pairs=np.array(pairs)
        axes[0].plot(pairs[:,0],pairs[:,1],'.-',lw=.6,alpha=.65)
axes[0].set_xlabel('Reported recovery voltage (mV)')
axes[0].set_ylabel('Reported sodium recovery time constant (ms)')
axes[0].set_title('Human Na, 25 C; each line is one recording')
for ax,key,title in zip(axes[1:],('rec_tau1','rec_tau2'),('Fast','Slow')):
    for i,record in enumerate(k):
        tau=record.get(key)
        if tau is None:continue
        ax.plot(i+1,tau,'o',color='#0072b2')
        points.append(dict(channel='potassium',filename=record['Filename'],excel_row=record['excel_row'],
            protocol=record.get('rec_protocol_name'),component=key,tau_ms=tau))
    ax.set_xlabel('Human potassium recording index (1-13)')
    ax.set_ylabel('Reported recovery time constant (ms)')
    ax.set_title(title+' human K recovery, 34 C')
    ax.set_xlim(.5,13.5)
fig.suptitle('Human-only derived experimental observations; no model fit or physiological QC\nMissing entries remain missing. Original current traces are not supplied by these workbooks.')
fig.tight_layout(rect=(0,0,1,.88))
fig.savefig(root/'human-recovery-observations.png',dpi=150)
plt.close(fig)
(root/'direct-observations.json').write_text(json.dumps(points,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
