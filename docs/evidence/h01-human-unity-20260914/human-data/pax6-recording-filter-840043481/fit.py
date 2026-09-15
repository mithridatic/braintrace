"""Fit the documented recording filter to controls and test excluded controls."""

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
from docs.evidence.h01_l1_recording import notebook_value
from docs.evidence.h01_pax6_recording_filter import predict_control, INITIAL, LOWER, UPPER, ORDER, pole_pairs

prior = out.parent / 'pax6-voltage-model-840043481'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source = repo / '.cache/human-source-followup/pax6-840043481.nwb'
inputs = json.loads((prior / 'training-inputs.json').read_bytes())
assert sha(source) == inputs['source_sha256']
assert sha(prior / 'training-source.npz') == inputs['arrays_sha256']
training = [70, 74, 78, 79, 88]
validation = [72, 76, 83, 99, 101, 103]
records = inputs['records']


def write(name, data):
    with (out / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


metadata = []
with h5py.File(source, 'r') as nwb:
    notebook = nwb['general/labnotebook/ITC18USB_Dev_0']
    keys, units = (notebook['numericalKeys'].asstr()[i].tolist() for i in (0, 1))
    values = notebook['numericalValues'][()]
    for record in records:
        fields = {key: dict(value=notebook_value(keys, values, key, record['sweep'], scope=0),
                             stored_unit=units[keys.index(key)])
                  for key in ['Hardware Type', 'LPF Cutoff', 'Secondary LPF Cutoff',
                              'Scaled Out Signal', 'Scale Factor Units'] if key in keys}
        assert fields['LPF Cutoff']['value'] == 2000
        assert fields['Hardware Type']['value'] == 1
        metadata.append(dict(sweep=record['sweep'], headstage=0, fields=fields))
write('recorded-filter.json', dict(source_sha256=inputs['source_sha256'], records=metadata,
                                 cutoff_hz_used=2000, order=4, normalization='analog -3 dB magnitude',
                                 metadata_description='https://alleninstitute.github.io/MIES/labnotebook-descriptions.html',
                                 manufacturer_manual='https://neurophysics.ucsd.edu/Manuals/Axon%20Instruments/MultiClamp_700B.pdf',
                                 manual_page=144, secondary_filter_cascaded=False,
                                 pole_pair_constants=[x.tolist() for x in pole_pairs()]))

edge_times, edge_steps = [], []
for record in records:
    segments = record['command_segments']
    times = [segment['start_ms'] for segment in segments[1:]]
    steps = np.diff([segment['command_mv'] for segment in segments]).tolist()
    padding = 5 - len(times)
    edge_times.append(times + [segments[-1]['stop_ms']] * padding)
    edge_steps.append(steps + [0.] * padding)
edge_times, edge_steps = np.array(edge_times), np.array(edge_steps)

ids, times, observed, prior_predicted, indices, roles = [], [], [], [], [], []
with np.load(prior / 'training-source.npz') as original, np.load(prior / 'training-predictions.npz') as old:
    for row, record in enumerate(records):
        sweep = record['sweep']
        # All eleven early controls; full traces only for the three long controls.
        index = np.arange(875, record['samples'] if sweep in [79, 83, 88] else 1750)
        t = original[f's{sweep}_time_ms'][index]
        current = original[f's{sweep}_total_current_pa'][index] - record['baseline_pa']
        previous = old['filtered_pa'][old['protocol_ids'] == row][index - 875]
        ids.append(np.full(len(index), row, int))
        times.append(t)
        observed.append(current)
        prior_predicted.append(previous)
        indices.append(index)
        roles.append(np.full(len(index), sweep in training, bool))
ids, times, observed, prior_predicted, indices, roles = (np.concatenate(x) for x in (ids, times, observed, prior_predicted, indices, roles))
arrays = dict(protocol_ids=ids, time_ms=times, source_indices=indices, observed_pa=observed,
              old_model_pa=prior_predicted, fitting_mask=roles, edge_times_ms=edge_times, edge_steps_mv=edge_steps)
with (out / 'inputs.npz').open('xb') as stream:
    np.savez_compressed(stream, **arrays)
write('inputs.json', dict(source_sha256=inputs['source_sha256'], prior_inputs_sha256=sha(prior / 'training-inputs.json'),
                         original_arrays_sha256=sha(prior / 'training-source.npz'), prior_predictions_sha256=sha(prior / 'training-predictions.npz'),
                         observations_sha256=sha(out / 'inputs.npz'), training_sweeps=training,
                         excluded_from_fit_sweeps=validation, records=records, independent_validation=False,
                         novel_responses_opened=False, selected_samples=int(roles.sum()),
                         evaluation_samples=int((~roles).sum())))
began = time.perf_counter()
calls = 0


def residual(p):
    global calls
    values = predict_control(p, edge_times, edge_steps, ids[roles], times[roles]) - observed[roles]
    if not np.iscomplexobj(p):
        calls += 1
        if calls == 1 or calls % 5 == 0:
            print(json.dumps(dict(evaluation=calls, seconds=time.perf_counter() - began,
                                  rmse_pa=float(np.sqrt(np.mean(values**2))))), flush=True)
    return values


result = least_squares(residual, INITIAL, bounds=(LOWER, UPPER), jac='cs', x_scale='jac',
                       max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
proximity = np.minimum(result.x - LOWER, UPPER - result.x) / (UPPER - LOWER)
write('frozen-candidate.json', dict(parameter_order=ORDER, parameters=result.x.tolist(),
                                  initial=INITIAL.tolist(), lower=LOWER.tolist(), upper=UPPER.tolist(),
                                  optimizer_success=bool(result.success), message=str(result.message),
                                  evaluations=result.nfev, active_bounds=result.active_mask.tolist(),
                                  relative_bound_distance=proximity.tolist(), seconds=time.perf_counter() - began,
                                  cutoff_hz=2000, observation_model_only=True,
                                  inputs_sha256=sha(out / 'inputs.json'),
                                  helper_sha256=sha(repo / 'docs/evidence/h01_pax6_recording_filter.py')))
# Fixed parameters are written before evaluating any excluded control.
prediction = predict_control(result.x, edge_times, edge_steps, ids, times)
arrays.update(predicted_pa=prediction, residual_pa=prediction-observed)
with (out / 'predictions.npz').open('xb') as stream:
    np.savez_compressed(stream, **arrays)
early = (~roles) & (times < 70)
rmse = lambda values: float(np.sqrt(np.mean(values**2)))
rows = []
for sweep in validation:
    row = inputs['training_sweeps'].index(sweep)
    mask = (ids == row) & (times < 70)
    rows.append(dict(sweep=sweep, samples=int(mask.sum()), old_rmse_pa=rmse(prior_predicted[mask]-observed[mask]),
                     new_rmse_pa=rmse(prediction[mask]-observed[mask])))
long_mask = ids == inputs['training_sweeps'].index(83)
old_early, new_early = rmse(prior_predicted[early]-observed[early]), rmse(prediction[early]-observed[early])
old_long, new_long = rmse(prior_predicted[long_mask]-observed[long_mask]), rmse(prediction[long_mask]-observed[long_mask])
gates = dict(converged=bool(result.success), latency_interior=bool(proximity[4] > 1e-6),
             early_improvement=new_early <= .5 * old_early,
             no_early_control_worse=all(r['new_rmse_pa'] <= 1.05 * r['old_rmse_pa'] for r in rows),
             long_control_not_worse=new_long <= 1.05 * old_long)
decision = dict(status='accepted for conditional observation correction' if all(gates.values()) else 'rejected',
                gates=gates, old_early_rmse_pa=old_early, new_early_rmse_pa=new_early,
                early_rmse_reduction_pct=100*(1-new_early/old_early),
                old_long_rmse_pa=old_long, new_long_rmse_pa=new_long, individual_early_controls=rows,
                scores_promoted=False, physiology_qualified=False, new_response_access=False)
write('decision.json', decision)
print(json.dumps(decision), flush=True)
