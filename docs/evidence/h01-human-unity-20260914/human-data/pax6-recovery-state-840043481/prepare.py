"""Bind both recovery pulses and the holding return on their original sample indices."""

from pathlib import Path
import hashlib
import json

import numpy as np

root=Path(__file__).resolve().parent
prior=root.parent/'pax6-recovery-840043481'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (root/'inputs.npz').exists():raise FileExistsError('Inputs already exist in this evidence prefix.')
prior_receipt=json.loads((prior/'analysis-receipt.json').read_bytes())
pin='30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944'
data={k:[] for k in ('first_raw_pa','second_raw_pa','post_raw_pa','first_source_time_ms',
                     'second_source_time_ms','post_source_time_ms','initial_baseline_pa','gaps_ms')}
records=[]
for sweep in range(123,142):
    path=prior/f'sweep-{sweep}.npz';jp=path.with_suffix('.json')
    assert sha(jp)==prior_receipt['files'][jp.name]
    m=json.loads(jp.read_bytes());assert sha(path)==m['arrays_sha256']
    assert m['source']['source_sha256']==pin and m['source']['sample_rate_hz']==25000.
    with np.load(path) as a:a=dict(a)
    t,i,v=a['source_time_ms'],a['source_total_current_pa'],a['source_command_voltage_mv']
    assert np.allclose(np.diff(t),.04,rtol=0,atol=1e-8)
    assert np.isfinite(i).all() and np.isfinite(v).all()
    windows=[(m['first_onset_ms'],300.,m['pulse_command_mv']),
             (m['second_onset_ms'],300.,m['pulse_command_mv']),
             (m['second_onset_ms']+300.,1000.,m['holding_command_mv'])]
    indices=[]
    for name,(onset,duration,level) in zip(('first','second','post'),windows):
        start=np.flatnonzero(np.isclose(t,onset,rtol=0,atol=1e-8))
        assert len(start)==1
        start=int(start[0]);stop=start+int(duration*25)
        assert stop<=len(t) and np.allclose(v[start:stop],level,rtol=0,atol=1e-7)
        phase=np.arange(stop-start)/25.
        assert np.allclose(t[start:stop]-onset,phase,rtol=0,atol=1e-8)
        data[name+'_raw_pa'].append(i[start:stop]);data[name+'_source_time_ms'].append(t[start:stop])
        indices.append(dict(window=name,start_index=start,stop_index=stop,onset_ms=onset,command_mv=level))
    assert m['pulse_duration_ms']==300.
    assert abs(m['pulse_command_mv']-60.00630371272564)<1e-7
    assert abs(m['gap_command_mv']+89.99370341189206)<1e-7
    assert abs(m['holding_command_mv']+19.993700087070465)<1e-7
    gap_start=indices[0]['stop_index'];gap_stop=indices[1]['start_index']
    assert np.allclose(v[gap_start:gap_stop],m['gap_command_mv'],atol=1e-7,rtol=0)
    assert np.isclose((gap_stop-gap_start)/25,m['gap_ms'],rtol=0,atol=1e-8)
    baseline=float(a['initial_baseline_current_pa'].mean())
    assert baseline==m['initial_baseline_mean_pa']
    data['initial_baseline_pa'].append(baseline);data['gaps_ms'].append(m['gap_ms'])
    records.append(dict(sweep=sweep,source_json=dict(path='../'+prior.name+'/'+jp.name,sha256=sha(jp)),
                        source_npz=dict(path='../'+prior.name+'/'+path.name,sha256=sha(path)),windows=indices,
                        initial_baseline_pa=baseline))
data={k:np.asarray(v) for k,v in data.items()}
data['pulse_phase_ms']=np.arange(7500)/25.
data['post_phase_ms']=np.arange(25000)/25.
for name in ('first','second','post'):
    data[name+'_current_pa']=data[name+'_raw_pa']-data['initial_baseline_pa'][:,None]
np.savez_compressed(root/'inputs.npz',**data)
metadata=dict(source_sha256=pin,source_session='840043481',source_specimen='840043506',
              arrays_sha256=sha(root/'inputs.npz'),records=records,
              coordinate_note='Common phases are exact sample indices / 25 kHz; original timestamps retained and checked to 1e-8 ms. No interpolation.',
              raw_currents_preserved=True,baseline='same measured initial baseline subtracted from all three responses',
              external_response_opened=False)
(root/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Prepared all 19 first pulses, second pulses and 1000 ms holding returns.')
