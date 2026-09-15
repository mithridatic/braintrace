"""Fit the registered relaxation hypothesis to the same preserved observations."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_pax6_relaxation_current import prepare_commands, predict_active, INITIAL, LOWER, UPPER, ORDER

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = out.parent / 'pax6-active-refit-840043481'
observer_path = out.parent / 'pax6-recording-filter-840043481/frozen-candidate.json'
inputs = json.loads((prior / 'training-inputs.json').read_bytes())
assert sha(prior / 'training-source.npz') == inputs['arrays_sha256']
assert sha(observer_path) == inputs['observer_sha256']
observer = json.loads(observer_path.read_bytes())
assert sha(repo / 'docs/evidence/h01_pax6_recording_filter.py') == observer['helper_sha256']
began = time.perf_counter()


def write(name, data):
    with (out / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


with np.load(prior / 'training-source.npz') as source:
    raw = dict(source)
ids, times, values, weights = [], [], [], []
for row, record in enumerate(inputs['records']):
    sweep = record['sweep']
    index = raw[f's{sweep}_fit_indices']
    count = raw[f's{sweep}_fit_weights']
    assert index[0] == 875 and count.sum() == record['full_scoring_samples']
    np.testing.assert_array_equal(count, np.diff(np.r_[index, record['samples']]))
    ids.append(np.full(len(index), row, int))
    times.append(raw[f's{sweep}_time_ms'][index])
    values.append(raw[f's{sweep}_total_current_pa'][index]-record['baseline_pa'])
    weights.append(np.sqrt(count))
ids, times, observed, weight = (np.concatenate(x) for x in (ids, times, values, weights))
context = prepare_commands(raw['command_starts'], raw['command_stops'], raw['command_volts'],
                           ids, times, inputs['observation_parameters'])
inputs.update(source_arrays_path='../pax6-active-refit-840043481/training-source.npz',
              previous_inputs_sha256=sha(prior / 'training-inputs.json'),
              previous_candidate_sha256=sha(prior / 'frozen-candidate.json'),
              observations_and_fitting_masks_unchanged=True)
write('training-inputs.json', inputs)
evaluations = 0


def residual(parameters):
    global evaluations
    error = (predict_active(parameters, context)-observed)*weight
    if not np.iscomplexobj(parameters):
        evaluations += 1
        trial = dict(evaluation=evaluations, seconds=time.perf_counter()-began, parameters=parameters.tolist(),
                     quadrature_rmse_pa=float(np.sqrt(np.sum(error**2)/np.sum(weight**2))))
        with (out / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(trial, allow_nan=False)+'\n')
        if evaluations == 1 or evaluations % 5 == 0:
            print(json.dumps(trial), flush=True)
    return error


result = least_squares(residual, INITIAL, bounds=(LOWER, UPPER), jac='cs', x_scale='jac',
                       max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
proximity = np.minimum(result.x-LOWER, UPPER-result.x)/(UPPER-LOWER)
fit = dict(parameter_order=ORDER, parameters=result.x.tolist(), initial=INITIAL.tolist(),
           lower=LOWER.tolist(), upper=UPPER.tolist(), optimizer_success=bool(result.success),
           message=str(result.message), evaluations=int(result.nfev), active_bounds=result.active_mask.tolist(),
           relative_bound_distance=proximity.tolist(), parameter_at_bound=bool(np.any(proximity <= 1e-6)),
           fit_seconds=time.perf_counter()-began, cost=float(result.cost), source_sha256=inputs['source_sha256'],
           training_inputs_sha256=sha(out / 'training-inputs.json'), observer_sha256=sha(observer_path),
           model_source_sha256=sha(repo / 'docs/evidence/h01_pax6_relaxation_current.py'),
           dependency_sha256={name: sha(repo / 'docs/evidence' / name) for name in
                              ['h01_pax6_bessel_current.py', 'h01_pax6_voltage_model.py', 'h01_pax6_recording_filter.py']},
           spec_sha256=sha(repo / 'docs/specs/2026-09-14-h01-pax6-independent-relaxation.md'),
           scores_promoted=False, physiology_qualified=False)
write('frozen-candidate.json', fit)
print(json.dumps(fit), flush=True)
