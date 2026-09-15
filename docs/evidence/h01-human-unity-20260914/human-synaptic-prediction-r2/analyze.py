"""Fit and test one shared human synaptic-current candidate without refitting the cell."""

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pyabf
from pyabf.waveform import EpochTable

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT/'docs/evidence'))
from h01_paired_response import spike_windows
from h01_abf_pulses import command_from_epochs, commanded_crossings
from h01_synaptic_prediction import fit_synapse, synaptic_waveforms, train_prediction

started = time.monotonic()
cell_path = OUT.parent/'human-paired-passive/frozen-candidate.json'
assert hashlib.sha256(cell_path.read_bytes()).hexdigest() == 'a4cec3ab1f9f3ccb012586f77f78cd42941b38b5ddbf7330064389721e4d8c60'
cell = json.loads(cell_path.read_text())
settings = dict(resistance_mohm=cell['resistance_mohm'], membrane_tau_ms=cell['tau_ms'])
source_path = ROOT/'.cache/human-paired-synapse/2015_11_04_0076.abf'
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == cell['source_sha256']
abf = pyabf.ABF(str(source_path))
assert abf.adcNames[:2] == ['IN 0', 'IN 2'] and abf.adcUnits[:2] == ['mV', 'mV']
assert abf.dataRate == 50000 and abf.sweepCount == 30
stored = dict(relative_ms=np.arange(-500, 2501)*.02)
records = {}
unassigned_crossings = {}
command_table = EpochTable(abf, 0)


def extract(sweep):
    """Extract one complete original six-spike paired observation.

    Parameters
    ----------
    sweep : int
        Original ABF sweep number.

    Returns
    -------
    tuple of ndarray
        Scored lags, baseline lags and baseline-centered observed voltage.
    """
    abf.setSweep(sweep, channel=0)
    pre, clock = abf.sweepY.copy(), abf.sweepX.copy()
    abf.setSweep(sweep, channel=1)
    assert np.array_equal(clock, abf.sweepX)
    command = command_from_epochs(command_table.epochWaveformsBySweep[sweep])
    assigned, unassigned = commanded_crossings(pre, command)
    unassigned_crossings[sweep] = unassigned.tolist()
    all_events = spike_windows(clock, pre, abf.sweepY, rate_hz=abf.dataRate)
    assert len(assigned) == 6
    selected = np.searchsorted(all_events['accepted_crossings'], assigned)
    np.testing.assert_array_equal(all_events['accepted_crossings'][selected], assigned)
    events = {key: all_events[key][selected] for key in
              ['indices', 'time_s', 'pre_mv', 'post_mv', 'accepted_crossings']}
    stored[f's{sweep}_unassigned_crossings'] = unassigned
    if sweep == 0:
        stored['reconstructed_command_pa'] = command
        stored['prior_parser_command_pa'] = command_table.epochWaveformsBySweep[sweep].getWaveform()
    else:
        np.testing.assert_array_equal(command, stored['reconstructed_command_pa'])
    indices, crossings = events['indices'], events['accepted_crossings']
    baseline = events['post_mv'][:, :400].mean(axis=-1)
    centered = events['post_mv'][:, 500:]-baseline[:, None]
    lags = (indices[:, 500:, None]-crossings[None, None, :])*.02
    before = (indices[:, :400, None]-crossings[None, None, :])*.02
    for key in ['indices', 'time_s', 'pre_mv', 'post_mv', 'accepted_crossings']:
        stored[f's{sweep}_{key}'] = events[key]
    stored[f's{sweep}_baseline_mv'] = baseline
    records[sweep] = (lags, before, centered)
    return lags, before, centered


calibration = [extract(s) for s in range(15)]
lags, before, measured = [np.stack([x[i] for x in calibration]) for i in range(3)]
fit = fit_synapse(lags, before, measured, **settings)
candidate = dict(synaptic_parameters=fit['parameters'],
                 parameter_names=['peak_pa', 'rise_ms', 'decay_minus_rise_ms', 'delay_ms'],
                 frozen_cell=settings, cell_candidate_sha256=hashlib.sha256(cell_path.read_bytes()).hexdigest(),
                 source_sha256=cell['source_sha256'], calibration_sweeps=list(range(15)),
                 validation_sweeps=list(range(15, 30)))
(OUT/'frozen-candidate.json').write_text(json.dumps(candidate, indent=2)+'\n', encoding='utf-8')
frozen_sha = hashlib.sha256((OUT/'frozen-candidate.json').read_bytes()).hexdigest()
(OUT/'fit.json').write_text(json.dumps(fit, indent=2)+'\n', encoding='utf-8')
validation = [extract(s) for s in range(15, 30)]
assert hashlib.sha256((OUT/'frozen-candidate.json').read_bytes()).hexdigest() == frozen_sha
parameters = candidate['synaptic_parameters']
sweep_errors, event_errors = [], []
all_observed, all_predicted = [], []
for sweep in range(30):
    score_lags, baseline_lags, observed = records[sweep]
    predicted = train_prediction(score_lags[None], baseline_lags[None], parameters, **settings)[0]
    crossings = stored[f's{sweep}_accepted_crossings']
    full_lags = (stored[f's{sweep}_indices'][:, :, None]-crossings[None, None, :])*.02
    current, voltage = synaptic_waveforms(full_lags, parameters, **settings)
    stored[f's{sweep}_inferred_current_pa'] = current.sum(axis=-1)
    raw_prediction = voltage.sum(axis=-1)
    model_baseline = raw_prediction[:, :400].mean(axis=-1)
    stored[f's{sweep}_model_baseline_mv'] = model_baseline
    stored[f's{sweep}_prediction_mv'] = raw_prediction-model_baseline[:, None]
    np.testing.assert_allclose(stored[f's{sweep}_prediction_mv'][:, 500:], predicted, atol=1e-12, rtol=0)
    base_error = float(np.sqrt(np.mean(observed**2)))
    model_error = float(np.sqrt(np.mean((observed-predicted)**2)))
    sweep_errors.append(dict(sweep=sweep, group='calibration' if sweep < 15 else 'validation',
                             reference_rmse_mv=base_error, candidate_rmse_mv=model_error,
                             error_ratio=model_error/base_error))
    for event in range(6):
        base = float(np.sqrt(np.mean(observed[event]**2)))
        error = float(np.sqrt(np.mean((observed[event]-predicted[event])**2)))
        event_errors.append(dict(sweep=sweep, event=event, reference_rmse_mv=base,
                                 candidate_rmse_mv=error, error_ratio=error/base))
    all_observed.append(observed)
    all_predicted.append(predicted)
observed, predicted = np.stack(all_observed), np.stack(all_predicted)
reference_error = float(np.sqrt(np.mean(observed[15:]**2)))
candidate_error = float(np.sqrt(np.mean((observed[15:]-predicted[15:])**2)))
failures = [e['sweep'] for e in sweep_errors[15:] if e['error_ratio'] > 1.05]
accepted = (fit['success'] and fit['active_mask'] == [0]*4
            and candidate_error <= .75*reference_error and not failures)
ordinal_errors = [dict(event=e,
                       reference_rmse_mv=float(np.sqrt(np.mean(observed[15:, e]**2))),
                       candidate_rmse_mv=float(np.sqrt(np.mean((observed[15:, e]-predicted[15:, e])**2))))
                  for e in range(6)]
np.savez_compressed(OUT/'observations.npz', **stored)
result = dict(candidate=candidate, candidate_sha256=frozen_sha, per_sweep=sweep_errors,
              unassigned_crossings=unassigned_crossings,
              per_event=event_errors, validation_by_event_ordinal=ordinal_errors,
              validation_reference_rmse_mv=reference_error,
              validation_candidate_rmse_mv=candidate_error,
              validation_improvement_fraction=1-candidate_error/reference_error,
              validation_failed_sweeps=failures, no_active_bounds=fit['active_mask'] == [0]*4,
              accepted_for_further_qualification=bool(accepted),
              physiology_qualified=False, scores_promoted=False,
              elapsed_seconds=time.monotonic()-started)
(OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['per_sweep', 'per_event']}, indent=2))
