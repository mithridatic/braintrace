"""Test fixed cell/kinetic transfer while fitting only a second connection's strength."""

from pathlib import Path
import csv
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
from h01_abf_pulses import command_from_epochs, commanded_crossings
from h01_paired_response import spike_windows
from h01_paired_passive import pulse_response
from h01_synaptic_prediction import synaptic_waveforms, train_prediction
from h01_synaptic_transfer import fit_connection_strength

started = time.monotonic()
acquisition = json.loads((OUT/'acquisition.json').read_text())
source = ROOT/acquisition['path']
assert hashlib.sha256(source.read_bytes()).hexdigest() == acquisition['sha256']
cell_path = OUT.parent/'human-paired-passive/frozen-candidate.json'
kernel_path = OUT.parent/'human-synaptic-prediction-r2/frozen-candidate.json'
assert hashlib.sha256(cell_path.read_bytes()).hexdigest() == 'a4cec3ab1f9f3ccb012586f77f78cd42941b38b5ddbf7330064389721e4d8c60'
assert hashlib.sha256(kernel_path.read_bytes()).hexdigest() == '788bfaaa50870f06cac22443b4d4793fa01634c70bae23452cbc1d6405be5061'
cell, kernel = json.loads(cell_path.read_text()), json.loads(kernel_path.read_text())
settings = kernel['frozen_cell']
unit_parameters = np.asarray(kernel['synaptic_parameters']).copy()
original_strength = float(unit_parameters[0])
unit_parameters[0] = 1
abf = pyabf.ABF(str(source))
assert abf.adcNames == ['IN 0', 'IN 2', 'IN 3'] and abf.adcUnits == ['mV']*3
assert abf._adcSection.nADCNum == [0, 2, 3]
assert abf.sweepCount == 30 and abf.dataRate == 50000 and abf.protocol == 'Stim5Hz_3to012'
with (OUT.parent/'human-paired-synapse/Data_fig2_Pyr_FS.csv').open(newline='', encoding='utf-8-sig') as handle:
    rows = list(csv.DictReader(handle))
pair, = [r for r in rows if r['File/folder name'] == source.name]
first, = [r for r in rows if r['File/folder name'] == '2015_11_04_0076.abf']
assert pair['species'] == 'Human' and pair['Type'] == 'EXC' and pair['SpikeCh'] == 'IN3'
assert all(pair[k] == first[k] for k in ['Date', 'Slice', 'Cluster', 'ResponseCh'])
metadata = dict(pair=pair, adc_names=abf.adcNames, physical_adc_numbers=abf._adcSection.nADCNum,
                adc_units=abf.adcUnits, protocol=abf.protocol, recorded_datetime=abf.abfDateTimeString,
                sample_rate_hz=abf.dataRate, sweep_count=abf.sweepCount, source_sha256=acquisition['sha256'])
(OUT/'recording.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
tables = [EpochTable(abf, ch) for ch in range(4)]
stored = dict(relative_ms=np.arange(-500, 2501)*.02, electrical_relative_ms=np.arange(20001)*.02)
electrical_prediction = pulse_response(stored['electrical_relative_ms'], cell['resistance_mohm'],
                                      cell['tau_ms'], cell['series_mohm'])
stored['electrical_prediction_mv'] = electrical_prediction
electrical_errors, electrical_observations = [], []
for sweep in range(30):
    expected = np.zeros(abf.sweepPointCount); expected[153125:163125] = -10
    for table in tables[:3]:
        np.testing.assert_array_equal(command_from_epochs(table.epochWaveformsBySweep[sweep]), expected)
    abf.setSweep(sweep, channel=1)
    raw = abf.sweepY[153125:173126].copy()
    pre = abf.sweepY[150625:152625].copy()
    assert raw.size == 20001 and np.isfinite(raw).all() and np.isfinite(pre).all()
    baseline = float(pre.mean())
    measured = raw-baseline
    stored[f's{sweep}_electrical_raw_mv'] = raw
    stored[f's{sweep}_electrical_baseline_samples_mv'] = pre
    electrical_observations.append(measured)
    ref = float(np.sqrt(np.mean(measured**2)))
    error = float(np.sqrt(np.mean((measured-electrical_prediction)**2)))
    electrical_errors.append(dict(sweep=sweep, baseline_mv=baseline, reference_rmse_mv=ref,
                                  candidate_rmse_mv=error, error_ratio=error/ref))
electrical_observations = np.stack(electrical_observations)
e_ref = float(np.sqrt(np.mean(electrical_observations[15:]**2)))
e_error = float(np.sqrt(np.mean((electrical_observations[15:]-electrical_prediction)**2)))
e_failures = [e['sweep'] for e in electrical_errors[15:] if e['error_ratio'] > 1.05]
electrical_pass = e_error <= .75*e_ref and not e_failures
records, extras = {}, {}


def extract(sweep):
    """Retain one source-mapped six-event observation and unit-strength template.

    Parameters
    ----------
    sweep : int
        Original ABF sweep number.

    Returns
    -------
    tuple of ndarray
        Unit-current prediction and centered human response on scored samples.
    """
    abf.setSweep(sweep, channel=2)
    clock, pre = abf.sweepX.copy(), abf.sweepY.copy()
    abf.setSweep(sweep, channel=1)
    command = command_from_epochs(tables[3].epochWaveformsBySweep[sweep])
    assigned, unassigned = commanded_crossings(pre, command)
    assert len(assigned) == 6
    extras[sweep] = unassigned.tolist()
    all_events = spike_windows(clock, pre, abf.sweepY, rate_hz=abf.dataRate)
    chosen = np.searchsorted(all_events['accepted_crossings'], assigned)
    np.testing.assert_array_equal(all_events['accepted_crossings'][chosen], assigned)
    events = {k: all_events[k][chosen] for k in ['indices', 'time_s', 'pre_mv', 'post_mv', 'accepted_crossings']}
    indices = events['indices']
    baseline = events['post_mv'][:, :400].mean(axis=-1)
    observed = events['post_mv'][:, 500:]-baseline[:, None]
    lags = (indices[:, 500:, None]-assigned[None, None, :])*.02
    before = (indices[:, :400, None]-assigned[None, None, :])*.02
    template = train_prediction(lags[None], before[None], unit_parameters, **settings)[0]
    for key, value in events.items(): stored[f's{sweep}_{key}'] = value
    stored[f's{sweep}_baseline_mv'] = baseline
    stored[f's{sweep}_unassigned_crossings'] = unassigned
    if sweep == 0: stored['presynaptic_command_pa'] = command
    else: np.testing.assert_array_equal(stored['presynaptic_command_pa'], command)
    records[sweep] = (template, observed)
    return template, observed


calibration = [extract(s) for s in range(15)]
strength = fit_connection_strength(np.stack([x[0] for x in calibration]),
                                   np.stack([x[1] for x in calibration]))
candidate = dict(strength=strength, unit_synaptic_parameters=unit_parameters.tolist(),
                 frozen_cell=settings, original_strength_pa=original_strength,
                 source_sha256=acquisition['sha256'],
                 original_synaptic_candidate_sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
                 original_cell_candidate_sha256=hashlib.sha256(cell_path.read_bytes()).hexdigest(),
                 calibration_sweeps=list(range(15)), validation_sweeps=list(range(15, 30)))
(OUT/'frozen-candidate.json').write_text(json.dumps(candidate, indent=2)+'\n', encoding='utf-8')
frozen_sha = hashlib.sha256((OUT/'frozen-candidate.json').read_bytes()).hexdigest()
validation = [extract(s) for s in range(15, 30)]
assert hashlib.sha256((OUT/'frozen-candidate.json').read_bytes()).hexdigest() == frozen_sha
sweep_errors, event_errors, observed_all, predicted_all, original_all = [], [], [], [], []
for sweep in range(30):
    template, observed = records[sweep]
    pred = template*strength['peak_pa']
    original_pred = template*original_strength
    crossings = stored[f's{sweep}_accepted_crossings']
    full_lags = (stored[f's{sweep}_indices'][:, :, None]-crossings[None, None, :])*.02
    current, voltage = synaptic_waveforms(full_lags, unit_parameters, **settings)
    unit_voltage = voltage.sum(axis=-1)
    unit_voltage -= unit_voltage[:, :400].mean(axis=-1)[:, None]
    stored[f's{sweep}_unit_prediction_mv_per_pa'] = unit_voltage
    stored[f's{sweep}_inferred_current_pa'] = current.sum(axis=-1)*strength['peak_pa']
    np.testing.assert_allclose(unit_voltage[:, 500:]*strength['peak_pa'], pred, atol=1e-12, rtol=0)
    ref = float(np.sqrt(np.mean(observed**2)))
    error = float(np.sqrt(np.mean((observed-pred)**2)))
    original_error = float(np.sqrt(np.mean((observed-original_pred)**2)))
    sweep_errors.append(dict(sweep=sweep, reference_rmse_mv=ref, candidate_rmse_mv=error,
                             original_strength_rmse_mv=original_error, error_ratio=error/ref))
    for event in range(6):
        baseline_error = float(np.sqrt(np.mean(observed[event]**2)))
        prediction_error = float(np.sqrt(np.mean((observed[event]-pred[event])**2)))
        event_errors.append(dict(sweep=sweep, event=event, reference_rmse_mv=baseline_error,
                                 candidate_rmse_mv=prediction_error, error_ratio=prediction_error/baseline_error))
    observed_all.append(observed); predicted_all.append(pred); original_all.append(original_pred)
obs, pred, original_pred = np.stack(observed_all), np.stack(predicted_all), np.stack(original_all)
reference = float(np.sqrt(np.mean(obs[15:]**2)))
error = float(np.sqrt(np.mean((obs[15:]-pred[15:])**2)))
original_error = float(np.sqrt(np.mean((obs[15:]-original_pred[15:])**2)))
failures = [e['sweep'] for e in sweep_errors[15:] if e['error_ratio'] > 1.05]
synaptic_pass = strength['interior'] and error <= .75*reference and not failures
np.savez_compressed(OUT/'observations.npz', **stored)
result = dict(candidate=candidate, candidate_sha256=frozen_sha,
              electrical=dict(per_sweep=electrical_errors, validation_reference_rmse_mv=e_ref,
                              validation_candidate_rmse_mv=e_error, failed_sweeps=e_failures, passed=bool(electrical_pass)),
              synaptic=dict(per_sweep=sweep_errors, per_event=event_errors,
                            validation_reference_rmse_mv=reference, validation_candidate_rmse_mv=error,
                            validation_original_strength_rmse_mv=original_error,
                            validation_improvement_fraction=1-error/reference,
                            failed_sweeps=failures, passed=bool(synaptic_pass)),
              unassigned_crossings=extras, transfer_passed=bool(electrical_pass and synaptic_pass),
              physiology_qualified=False, scores_promoted=False, elapsed_seconds=time.monotonic()-started)
(OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(strength=strength, electrical_pass=bool(electrical_pass),
                     synaptic_pass=bool(synaptic_pass), reference_rmse_mv=reference,
                     candidate_rmse_mv=error, original_strength_rmse_mv=original_error,
                     failed_sweeps=failures, elapsed_seconds=result['elapsed_seconds']), indent=2))
