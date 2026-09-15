"""Retain same-command comparisons and unobserved intervals between acquisitions."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
if (root/'history-observations.json').exists():
    raise FileExistsError('History observations already exist in this evidence prefix.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plt.rcParams.update({'path.simplify':False,'font.size':9})
fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
saved,rows={},[]
reference_command=None
for sweep in [72,*range(89,98)]:
    m=json.loads((root/f'sweep-{sweep}.json').read_bytes())
    assert sha(root/f'sweep-{sweep}.npz')==m['arrays_sha256']
    with np.load(root/f'sweep-{sweep}.npz') as a:
        a=dict(a)
    onset=1100. if sweep==72 else 1000.
    t=a['time_ms'];current=a['total_current_pa'];command=a['command_voltage_mv']
    early=(t>=onset)&(t<onset+100.)
    before=(t>=900)&(t<990)
    baseline=float(current[before].mean())
    phase=t[early]-onset
    if reference_command is None: reference_command=command[early]
    assert np.array_equal(command[early],reference_command)
    saved[f's{sweep}_phase_ms']=phase
    saved[f's{sweep}_source_time_ms']=t[early]
    saved[f's{sweep}_current_pa']=current[early]
    saved[f's{sweep}_baseline_time_ms']=t[before]
    saved[f's{sweep}_baseline_current_pa']=current[before]
    points=[]
    for value in (20.,60.,90.):
        index=int(np.flatnonzero(np.isclose(phase,value,atol=1e-8,rtol=0))[0])
        points.append(dict(phase_ms=value,source_time_ms=float(t[early][index]),
                           current_pa=float(current[early][index]),baseline_centered_pa=float(current[early][index]-baseline)))
    rows.append(dict(sweep=sweep,source_json_sha256=sha(root/f'sweep-{sweep}.json'),
                     absolute_start_s=m['absolute_start_s'],actual_onset_ms=onset,
                     preceding_hold_since_test_pulse_ms=onset-55.04,
                     initial_baseline_pa=baseline,instrument=m['instrument'],points=points))
    for col,shift in ((0,0.),(1,baseline)):
        color='black' if sweep==72 else None
        lw=1.7 if sweep==72 else .55
        axes[0,col].plot(phase,current[early]-shift,lw=lw,color=color,label=f'{sweep}')
        post=phase>=5
        axes[1,col].plot(phase[post],current[early][post]-shift,lw=lw,color=color)
    if sweep in (72,91):
        common=(t>=onset)&(t<onset+1000.)
        assert np.allclose(command[common],reference_command[0],atol=1e-7,rtol=0)
        saved[f's{sweep}_long_phase_ms']=t[common]-onset
        saved[f's{sweep}_long_source_time_ms']=t[common]
        saved[f's{sweep}_long_current_pa']=current[common]
        post=common&(t>=onset+5)
        for col,shift in ((0,0.),(1,baseline)):
            axes[2,col].plot(t[post]-onset,current[post]-shift,lw=.6,label=f'{sweep}')
for col in (0,1):
    axes[0,col].set_ylim(-100,850)
    axes[1,col].set_ylim(-30,110);axes[2,col].set_ylim(-30,110)
    axes[0,col].set_title('All identical early commands; onset transient retained')
    axes[1,col].set_title('Same samples beyond 5 ms; total sweep 72 in black')
    axes[2,col].set_title('Full shared -20 mV command; sweeps 72 and 91, beyond 5 ms')
    axes[0,col].legend(ncol=5,fontsize=8);axes[2,col].legend()
    for ax in axes[:,col]:
        ax.set_ylabel('Total current (pA)' if col==0 else 'Current minus own baseline (pA)')
        ax.set_xlabel('Time after actual depolarizing command onset (ms)');ax.grid(alpha=.2)
fig.suptitle('Human PAX6 identical command comparison | alignment does not establish identical initial state')
fig.savefig(root/'same-command-history.png',dpi=140)
plt.close(fig)
np.savez_compressed(root/'same-command-history.npz',**saved)
recovery=root.parent/'pax6-recovery-840043481'
sequence=[];previous_end=None
for sweep in range(123,142):
    path=recovery/f'sweep-{sweep}.json';m=json.loads(path.read_bytes());s=m['source']
    start=s['absolute_start_s'];end=start+s['samples']/s['sample_rate_hz']
    sequence.append(dict(sweep=sweep,source_json_sha256=sha(path),start_s=start,exclusive_end_s=end,
                         preceding_unrecorded_s=None if previous_end is None else start-previous_end,
                         recovery_gap_ms=m['gap_ms']))
    previous_end=end
fits=json.loads((root/'fit-summary.json').read_bytes())
proximity=[]
for fit in fits:
    p=np.asarray(fit['parameters']);lo=np.array([0.,0.,1.,1.]);hi=np.full(4,10000.)
    distance=np.minimum(p-lo,hi-p)/(hi-lo)
    proximity.append(dict(total_sweep=fit['total_sweep'],scenario=fit['scenario'],
                          relative_distance_to_bound=distance.tolist(),within_one_millionth_of_range=(distance<=1e-6).tolist(),
                          optimizer_active_bounds=fit['active_bounds']))
result=dict(same_command_early_mv=float(reference_command[0]),same_command_samples_preserved=True,
            rows=rows,arrays_sha256=sha(root/'same-command-history.npz'),recovery_sequence=sequence,
            bound_proximity=proximity,
            interpretation='Total sweep 72 and conditioning sweeps have different currents at the same actual command and phase. Holding duration and prior histories differ; stationarity is not established. No unique cause identified.',
            external_response_opened=False,physiological_qualification=False)
(root/'history-observations.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Saved ten early histories, two full common commands and all recovery acquisition gaps.')
