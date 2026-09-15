"""Check bounded initial-state variation against frozen recovery predictions."""

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[2]))
from h01_l1_initial_state import observe_state, conditioned_currents
from h01_l1_recovery_state import state_currents

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, obj: p.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8', newline='\n')
prior = root.parent/'pax6-recovery-state-840043481'
receipt = json.loads((prior/'analysis-receipt.json').read_bytes())
assert sha(prior/'analysis-receipt.json') == '53c302417b91e1c2974d0db5c0f07b188aedbb28dedc78ac56367b9757b9883f'
for name in ('inputs.json', 'inputs.npz', 'joint-fit.json', 'joint-fit.npz'):
    assert sha(prior/name) == receipt['files'][name]
if (root/'frozen-candidate.json').exists():
    raise FileExistsError('Evidence prefix already contains a frozen candidate.')
fit = json.loads((prior/'joint-fit.json').read_bytes())
p = np.array(fit['parameters'])
write(root/'frozen-candidate.json', dict(source_commit='59531e9', parameters=p.tolist(),
      parameter_order=fit['parameter_order'], sources={name:receipt['files'][name]
      for name in ('inputs.json', 'inputs.npz', 'joint-fit.json', 'joint-fit.npz')},
      external_response_opened=False, estimation_window_ms=[10,60],
      scoring_windows_ms=dict(first=[60,280], second=[10,280], post=[10,980]),
      independent_validation=False))
inputs = json.loads((prior/'inputs.json').read_bytes())
with np.load(prior/'inputs.npz') as source:
    data = dict(source)
durations, baselines, source_records = [], [], []
for row, record in enumerate(inputs['records']):
    path = prior/record['source_npz']['path']
    assert sha(path) == record['source_npz']['sha256']
    with np.load(path) as original:
        t = original['source_time_ms']
        v = original['source_command_voltage_mv']
        first = record['windows'][0]['start_index']
        changes = np.flatnonzero(np.abs(np.diff(v[:first+1])) > 1e-7)+1
        hold_start = int(changes[changes < first][-1])
        assert abs(t[first]-2100.) < 1e-8 and abs(t[hold_start]-55.04) < 1e-8
        assert np.allclose(v[hold_start:first], -19.993700087070465, rtol=0, atol=1e-7)
        assert np.allclose(np.diff(t[hold_start:first+1]), .04, rtol=0, atol=1e-8)
        b = original['initial_baseline_time_ms']-t[first]
        assert len(b) == 2250
        assert np.isclose(original['initial_baseline_current_pa'].mean(), data['initial_baseline_pa'][row])
        durations.append(t[first]-t[hold_start])
        baselines.append(b)
        source_records.append(dict(sweep=record['sweep'], source_npz=record['source_npz'],
              holding_start_index=hold_start, holding_stop_index=first,
              holding_start_ms=float(t[hold_start]), holding_stop_ms=float(t[first]),
              holding_command_mv=float(v[hold_start])))
assert np.allclose(baselines, baselines[0], rtol=0, atol=1e-8)
# Exact sample-index baseline phases, checked against every original timestamp.
b = np.arange(2250)/25.-100.
assert np.allclose(baselines, b, rtol=0, atol=1e-8)
started = time.monotonic()
arrays, metadata = observe_state(p, data['pulse_phase_ms'], data['first_current_pa'], durations, b)
old = state_currents(p, data['gaps_ms'], data['pulse_phase_ms'], data['post_phase_ms'])
new = conditioned_currents(p, data['gaps_ms'], data['pulse_phase_ms'], data['post_phase_ms'],
                           arrays['initial_availability'], arrays['baseline_availability'])
metadata.update(elapsed_seconds=time.monotonic()-started, source_records=source_records,
                source_session='840043481', source_specimen='840043506',
                nwb_sha256=inputs['source_sha256'], external_response_opened=False,
                execution='local CPU; closed-form algebra and bounded linear fits',
                frozen_candidate_sha256=sha(root/'frozen-candidate.json'),
                scoring={}, direct_samples=[])
arrays.update(data)
for name, shared, prediction in zip(('first','second','post'), old, new):
    phase = data['post_phase_ms'] if name == 'post' else data['pulse_phase_ms']
    start, stop = {'first':(60,280),'second':(10,280),'post':(10,980)}[name]
    mask = (phase >= start)&(phase < stop)
    current = data[name+'_current_pa']
    arrays.update({name+'_shared_predicted_pa':shared, name+'_conditioned_predicted_pa':prediction,
                   name+'_conditioned_residual_pa':current-prediction, name+'_scoring_mask':mask})
    metadata['scoring'][name] = dict(window_ms=[start,stop], sample_count=int(mask.sum()),
        shared_rms_pa=float(np.sqrt(np.mean((current[:,mask]-shared[:,mask])**2))),
        conditioned_rms_pa=float(np.sqrt(np.mean((current[:,mask]-prediction[:,mask])**2))),
        shared_per_sweep_rms_pa=np.sqrt(np.mean((current[:,mask]-shared[:,mask])**2,axis=1)).tolist(),
        conditioned_per_sweep_rms_pa=np.sqrt(np.mean((current[:,mask]-prediction[:,mask])**2,axis=1)).tolist())
    for row in (0,9,15,18):
        for phase_value in (10.,60.,200.):
            index = int(round(phase_value*25))
            metadata['direct_samples'].append(dict(sweep=123+row, gap_ms=float(data['gaps_ms'][row]),
                window=name, phase_ms=float(phase[index]), raw_current_pa=float(data[name+'_raw_pa'][row,index]),
                centered_current_pa=float(current[row,index]), shared_prediction_pa=float(shared[row,index]),
                conditioned_prediction_pa=float(prediction[row,index])))
np.savez_compressed(root/'observation.npz', **arrays)
metadata.update(arrays_sha256=sha(root/'observation.npz'),
                holding_start_availability=arrays['holding_start_availability'].tolist(),
                initial_availability=arrays['initial_availability'].tolist(),
                holding_memory=arrays['holding_memory'].tolist(),
                design_singular_values=arrays['design_singular_values'].tolist(),
                active_bounds=arrays['active_bounds'].tolist())
write(root/'observation.json', metadata)
print(json.dumps({'scoring': metadata['scoring'], 'holding_memory': metadata['holding_memory'][0],
                  'states':metadata['holding_start_availability'], 'elapsed_seconds':metadata['elapsed_seconds']},indent=2))
