"""Run the registered active fit while preserving every real optimization trial."""

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
from docs.evidence.h01_l1_voltage_clamp import read_sweep, PAX6_SHA256
from docs.evidence.h01_pax6_bessel_current import prepare_commands, predict_active, INITIAL, LOWER, UPPER, ORDER

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source = repo / '.cache/human-source-followup/pax6-840043481.nwb'
observer_path = out.parent / 'pax6-recording-filter-840043481/frozen-candidate.json'
observer = json.loads(observer_path.read_text())
assert sha(source) == PAX6_SHA256
assert sha(repo / 'docs/evidence/h01_pax6_recording_filter.py') == observer['helper_sha256']
training = [70, 72, 74, 76, 78, 79, 83, 88, 99, 101, 103, 89, 93, 97, 123, 132, 141]
began = time.perf_counter()


def write(name, data):
    with (out / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


records, arrays, fit_ids, fit_times, fit_values, fit_weights = [], {}, [], [], [], []
for row, sweep in enumerate(training):
    data, metadata = read_sweep(source, sweep, session='840043481')
    t, current = data['time_ms'], data['total_current_pa']
    baseline_indices = np.flatnonzero((t >= 35) & (t < 44))
    assert len(baseline_indices) == 225
    baseline = float(current[baseline_indices].mean())
    selection = np.arange(len(t)) % 25 == 0
    for segment in metadata['command_segments'][1:]:
        selection[segment['start_index']:segment['start_index'] + 50] = True
    selection &= t >= 35
    indices = np.flatnonzero(selection)
    weights = np.diff(np.r_[indices, len(t)])
    assert weights.sum() == len(t) - indices[0]
    metadata.update(baseline_indices=baseline_indices.tolist(), baseline_pa=baseline,
                    fit_samples=len(indices), full_scoring_samples=int((t >= 35).sum()))
    records.append(metadata)
    arrays.update({f's{sweep}_{key}': value for key, value in data.items()})
    arrays[f's{sweep}_fit_indices'] = indices
    arrays[f's{sweep}_fit_weights'] = weights
    fit_ids.append(np.full(len(indices), row, int))
    fit_times.append(t[indices])
    fit_values.append(current[indices] - baseline)
    fit_weights.append(np.sqrt(weights))
width = max(len(row['command_segments']) for row in records)
starts, stops, volts = [], [], []
for row in records:
    segments = row['command_segments']
    padding = width - len(segments)
    starts.append([s['start_ms'] for s in segments] + [segments[-1]['stop_ms']] * padding)
    stops.append([s['stop_ms'] for s in segments] + [segments[-1]['stop_ms']] * padding)
    volts.append([s['command_mv'] for s in segments] + [segments[-1]['command_mv']] * padding)
arrays.update(command_starts=np.array(starts), command_stops=np.array(stops), command_volts=np.array(volts))
ids, times, observed, weight = (np.concatenate(x) for x in (fit_ids, fit_times, fit_values, fit_weights))
with (out / 'training-source.npz').open('xb') as stream:
    np.savez_compressed(stream, **arrays)
write('training-inputs.json', dict(source_sha256=PAX6_SHA256, records=records,
                                 arrays_sha256=sha(out / 'training-source.npz'), training_sweeps=training,
                                 selected_samples=len(times), represented_samples=int(round(np.sum(weight**2))),
                                 fit_objective='time-quadrature weighted selected original samples',
                                 observer_sha256=sha(observer_path), observation_parameters=observer['parameters'],
                                 prediction_responses_opened=False))
context = prepare_commands(starts, stops, volts, ids, times, observer['parameters'])
evaluations = 0


def residual(parameters):
    global evaluations
    result = (predict_active(parameters, context) - observed) * weight
    if not np.iscomplexobj(parameters):
        evaluations += 1
        trial = dict(evaluation=evaluations, seconds=time.perf_counter()-began,
                     parameters=parameters.tolist(), quadrature_rmse_pa=float(np.sqrt(np.sum(result**2)/np.sum(weight**2))))
        with (out / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(trial, allow_nan=False) + '\n')
        if evaluations == 1 or evaluations % 5 == 0:
            print(json.dumps(trial), flush=True)
    return result


result = least_squares(residual, INITIAL, bounds=(LOWER, UPPER), jac='cs',
                       x_scale='jac', max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
proximity = np.minimum(result.x-LOWER, UPPER-result.x)/(UPPER-LOWER)
fit = dict(parameter_order=ORDER, parameters=result.x.tolist(), initial=INITIAL.tolist(),
           lower=LOWER.tolist(), upper=UPPER.tolist(), optimizer_success=bool(result.success),
           message=str(result.message), evaluations=int(result.nfev), active_bounds=result.active_mask.tolist(),
           relative_bound_distance=proximity.tolist(), parameter_at_bound=bool(np.any(proximity <= 1e-6)),
           fit_seconds=time.perf_counter()-began, cost=float(result.cost), source_sha256=PAX6_SHA256,
           training_inputs_sha256=sha(out / 'training-inputs.json'), observer_sha256=sha(observer_path),
           model_source_sha256=sha(repo / 'docs/evidence/h01_pax6_bessel_current.py'),
           spec_sha256=sha(repo / 'docs/specs/2026-09-14-h01-pax6-active-refit.md'),
           scores_promoted=False, physiology_qualified=False)
write('frozen-candidate.json', fit)
print(json.dumps(fit), flush=True)
