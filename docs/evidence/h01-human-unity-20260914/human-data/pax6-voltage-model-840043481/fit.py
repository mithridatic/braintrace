"""Fit one registered voltage-dependent candidate, then freeze prediction access."""

from pathlib import Path
import hashlib
import json
import sys
import time

import h5py
import numpy as np
from scipy.optimize import least_squares

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_l1_voltage_clamp import read_sweep, command_segments, PAX6_SHA256
from docs.evidence.h01_l1_recording import notebook_value
from docs.evidence.h01_pax6_voltage_model import predict, INITIAL, LOWER, UPPER, ORDER

source = repo / '.cache/human-source-followup/pax6-840043481.nwb'
training = [70, 72, 74, 76, 78, 79, 83, 88, 99, 101, 103]
prediction_sweeps = [105, 106, 107, 109, 110, 111]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(source) == PAX6_SHA256
began = time.perf_counter()


def write(name, data):
    with (out / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def pack(records):
    width = max(len(row['command_segments']) for row in records)
    starts, stops, volts = [], [], []
    for row in records:
        segments = row['command_segments']
        padding = width - len(segments)
        starts.append([s['start_ms'] for s in segments] + [segments[-1]['stop_ms']] * padding)
        stops.append([s['stop_ms'] for s in segments] + [segments[-1]['stop_ms']] * padding)
        volts.append([s['command_mv'] for s in segments] + [segments[-1]['command_mv']] * padding)
    return tuple(np.array(x) for x in (starts, stops, volts))


records, original_arrays, fit_ids, fit_times, fit_values, fit_weights = [], {}, [], [], [], []
for row, sweep in enumerate(training):
    data, metadata = read_sweep(source, sweep, session='840043481')
    t, current = data['time_ms'], data['total_current_pa']
    baseline_indices = np.flatnonzero((t >= 35) & (t < 44))
    assert len(baseline_indices) == 225
    baseline = float(current[baseline_indices].mean())
    selection = np.arange(len(t)) % 25 == 0
    for segment in metadata['command_segments'][1:]:
        selection[segment['start_index']:segment['start_index'] + 250] = True
    selection &= t >= 35
    indices = np.flatnonzero(selection)
    weights = np.diff(np.r_[indices, len(t)])
    assert weights.sum() == len(t) - indices[0]
    metadata.update(baseline_indices=baseline_indices.tolist(), baseline_pa=baseline,
                    fit_samples=len(indices), full_scoring_samples=int((t >= 35).sum()))
    records.append(metadata)
    original_arrays.update({f's{sweep}_{key}': value for key, value in data.items()})
    original_arrays[f's{sweep}_fit_indices'] = indices
    original_arrays[f's{sweep}_fit_weights'] = weights
    fit_ids.append(np.full(len(indices), row, int))
    fit_times.append(t[indices])
    fit_values.append(current[indices] - baseline)
    fit_weights.append(np.sqrt(weights))
commands = pack(records)
ids, times, observed, weight = (np.concatenate(x) for x in (fit_ids, fit_times, fit_values, fit_weights))
with (out / 'training-source.npz').open('xb') as stream:
    np.savez_compressed(stream, **original_arrays)
write('training-inputs.json', dict(source_sha256=PAX6_SHA256, records=records,
                                 arrays_sha256=sha(out / 'training-source.npz'),
                                 training_sweeps=training, prediction_sweeps=prediction_sweeps,
                                 selected_samples=len(times), represented_samples=int(np.sum(weight**2)),
                                 fit_objective='time-quadrature weighted selected original samples',
                                 prediction_responses_opened=False))
evaluations = 0


def residual(parameters):
    global evaluations
    result = (predict(parameters, *commands, ids, times) - observed) * weight
    if not np.iscomplexobj(parameters):
        evaluations += 1
        if evaluations == 1 or evaluations % 5 == 0:
            print(json.dumps(dict(evaluation=evaluations, seconds=time.perf_counter() - began,
                                  quadrature_rmse_pa=float(np.sqrt(np.sum(result**2) / np.sum(weight**2))))), flush=True)
    return result


result = least_squares(residual, INITIAL, bounds=(LOWER, UPPER), jac='cs',
                       x_scale='jac', max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
proximity = np.minimum(result.x - LOWER, UPPER - result.x) / (UPPER - LOWER)
kinetic_indices = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 18])
at_bound = bool(np.any(proximity[kinetic_indices] <= 1e-6))
fit = dict(parameter_order=ORDER, parameters=result.x.tolist(), initial=INITIAL.tolist(),
           lower=LOWER.tolist(), upper=UPPER.tolist(), optimizer_success=bool(result.success),
           message=str(result.message), evaluations=int(result.nfev),
           active_bounds=result.active_mask.tolist(), relative_bound_distance=proximity.tolist(),
           kinetic_parameter_at_bound=at_bound, fit_seconds=time.perf_counter() - began,
           cost=float(result.cost), source_sha256=PAX6_SHA256,
           training_inputs_sha256=sha(out / 'training-inputs.json'),
           model_source_sha256=sha(repo / 'docs/evidence/h01_pax6_voltage_model.py'),
           spec_sha256=sha(repo / 'docs/specs/2026-09-14-h01-pax6-voltage-model.md'),
           scores_promoted=False, physiology_qualified=False)
write('frozen-candidate.json', fit)
print(json.dumps(fit), flush=True)

# One batched prediction covers every original training sample used for scoring.
full_ids = np.concatenate([np.full(row['full_scoring_samples'], j, int) for j, row in enumerate(records)])
full_times = np.concatenate([original_arrays[f's{sweep}_time_ms'][875:] for sweep in training])
full_observed = np.concatenate([original_arrays[f's{sweep}_total_current_pa'][875:] - row['baseline_pa']
                                for sweep, row in zip(training, records)])
full = predict(result.x, *commands, full_ids, full_times, components=True)
full.update(protocol_ids=full_ids, time_ms=full_times, observed_pa=full_observed,
            residual_pa=full['filtered_pa'] - full_observed)
with (out / 'training-predictions.npz').open('xb') as stream:
    np.savez_compressed(stream, **full)
write('training-errors.json', [dict(sweep=sweep, samples=int((full_ids == row).sum()),
                                    rmse_pa=float(np.sqrt(np.mean(full['residual_pa'][full_ids == row]**2))))
                               for row, sweep in enumerate(training)])

if not result.success or at_bound:
    write('decision.json', dict(status='rejected before new response access',
                               optimizer_success=bool(result.success), kinetic_parameter_at_bound=at_bound,
                               prediction_responses_opened=False, external_responses_opened=False,
                               whole_cell_responses_opened=False, scores_promoted=False))
    raise SystemExit(0)

# Inspect only command arrays and notebook fields before freezing all predictions.
prediction_records, target_commands, target_times = [], [], []
with h5py.File(source, 'r') as nwb:
    notebook = nwb['general/labnotebook/ITC18USB_Dev_0']
    keys = notebook['numericalKeys'].asstr()[0].tolist()
    values = notebook['numericalValues'][()]
    for sweep in prediction_sweeps:
        group = nwb[f'stimulus/presentation/data_{sweep:05d}_DA0']
        data = group['data']
        assert data.attrs['unit'] == 'volts'
        assert notebook_value(keys, values, 'V-Clamp Holding Enable', sweep, scope=0) == 1
        holding = notebook_value(keys, values, 'V-Clamp Holding Level', sweep, scope=0)
        command = (data[:].astype(float) * data.attrs['conversion'] + data.attrs.get('offset', 0)) * 1000 + holding
        rate = float(group['starting_time'].attrs['rate'])
        assert rate == 25000 and np.isfinite(command).all()
        t = np.arange(len(command)) / rate * 1000
        prediction_records.append(dict(sweep=sweep, command_segments=command_segments(command, rate),
                                       samples=len(command), full_scoring_samples=len(command) - 875))
        target_commands.append(command)
        target_times.append(t)
target_batch = pack(prediction_records)
target_ids = np.concatenate([np.full(len(t) - 875, j, int) for j, t in enumerate(target_times)])
target_clock = np.concatenate([t[875:] for t in target_times])
predicted = predict(result.x, *target_batch, target_ids, target_clock, components=True)
passive = result.x.copy()
passive[:3] = 0
predicted['passive_filtered_pa'] = predict(passive, *target_batch, target_ids, target_clock)
predicted.update(protocol_ids=target_ids, time_ms=target_clock)
with (out / 'frozen-protocol-predictions.npz').open('xb') as stream:
    np.savez_compressed(stream, **predicted)
write('prediction-freeze.json', dict(candidate_sha256=sha(out / 'frozen-candidate.json'),
                                    predictions_sha256=sha(out / 'frozen-protocol-predictions.npz'),
                                    records=prediction_records, response_arrays_opened=False))

# Candidate and protocol predictions are now immutable artifacts; open targets once.
target_arrays, target_observed = {}, []
for sweep, expected_command in zip(prediction_sweeps, target_commands):
    data, metadata = read_sweep(source, sweep, session='840043481')
    np.testing.assert_array_equal(data['command_voltage_mv'], expected_command)
    baseline = float(data['total_current_pa'][875:1100].mean())
    target_observed.append(data['total_current_pa'][875:] - baseline)
    target_arrays.update({f's{sweep}_{key}': value for key, value in data.items()})
    write(f'prediction-source-{sweep}.json', dict(metadata=metadata, baseline_pa=baseline))
target_arrays['observed_pa'] = np.concatenate(target_observed)
with (out / 'prediction-observations.npz').open('xb') as stream:
    np.savez_compressed(stream, **target_arrays)
errors = predicted['filtered_pa'] - target_arrays['observed_pa']
reference_errors = predicted['passive_filtered_pa'] - target_arrays['observed_pa']
new_rmse, reference_rmse = (float(np.sqrt(np.mean(x**2))) for x in (errors, reference_errors))
rows = [dict(sweep=sweep, candidate_rmse_pa=float(np.sqrt(np.mean(errors[target_ids == j]**2))),
             passive_rmse_pa=float(np.sqrt(np.mean(reference_errors[target_ids == j]**2))))
        for j, sweep in enumerate(prediction_sweeps)]
passed = new_rmse <= .75 * reference_rmse and all(row['candidate_rmse_pa'] <= 1.05 * row['passive_rmse_pa'] for row in rows)
write('decision.json', dict(status='eligible for further qualification' if passed else 'prediction rejected',
                           candidate_rmse_pa=new_rmse, passive_rmse_pa=reference_rmse,
                           individual_sweeps=rows, prediction_responses_opened=True,
                           external_responses_opened=False, whole_cell_responses_opened=False,
                           scores_promoted=False, physiology_qualified=False))
