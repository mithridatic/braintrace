"""Fit early human test pulses, freeze the circuit, then score later sweeps."""

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
sys.path.insert(0, str(ROOT / 'docs/evidence'))
from h01_paired_passive import fit_response, pulse_response

started = time.monotonic()
source = json.loads((OUT.parent / 'human-paired-synapse/recording.json').read_text())
path = ROOT / source['source']['path']
assert hashlib.sha256(path.read_bytes()).hexdigest() == source['source']['sha256']
abf = pyabf.ABF(str(path))
assert abf.adcNames[1] == 'IN 2' and abf.adcUnits[1] == 'mV'
assert abf._adcSection.nADCNum[1] == 2
assert abf.dataRate == 50000 and abf.sweepCount == 30
tables = [EpochTable(abf, ch) for ch in [1, 2, 3]]
start, stop = 153125, 163125
relative = np.arange(20001) * .02
stored = dict(relative_ms=relative, pulse_source_indices=np.arange(start, start+20001),
              baseline_source_indices=np.arange(start-2500, start-500))
voltages, baselines = {}, {}


def extract(sweep):
    """Check the pulse and retain original postsynaptic observations.

    Parameters
    ----------
    sweep : int
        Original zero-based ABF sweep.

    Returns
    -------
    ndarray
        Pulse/return voltage after subtraction of the pre-pulse baseline.
    """
    commands = [t.epochWaveformsBySweep[sweep].getWaveform() for t in tables]
    assert all(np.array_equal(commands[0], c) for c in commands[1:])
    expected = np.zeros(abf.sweepPointCount)
    expected[start:stop] = -10
    assert np.array_equal(commands[0], expected)
    abf.setSweep(sweep, channel=1)
    v = abf.sweepY.copy()
    assert np.isfinite(v).all()
    baseline = float(np.mean(v[start-2500:start-500]))
    raw = v[start:start+20001]
    assert raw.size == relative.size
    stored[f's{sweep}_raw_mv'] = raw
    stored[f's{sweep}_source_time_s'] = abf.sweepX[start:start+20001].copy()
    stored[f's{sweep}_baseline_mv'] = v[start-2500:start-500]
    stored[f's{sweep}_baseline_time_s'] = abf.sweepX[start-2500:start-500].copy()
    stored[f's{sweep}_command_pa'] = commands[0][start:start+20001]
    baselines[sweep] = baseline
    voltages[sweep] = raw - baseline
    return voltages[sweep]


calibration = np.stack([extract(s) for s in range(15)])
fit = fit_response(relative, calibration)
candidate = {k: fit[k] for k in ['resistance_mohm', 'tau_ms', 'series_mohm']}
candidate['calibration_sweeps'] = list(range(15))
candidate['validation_sweeps'] = list(range(15, 30))
candidate['source_sha256'] = source['source']['sha256']
frozen = json.dumps(candidate, indent=2)+'\n'
(OUT / 'frozen-candidate.json').write_text(frozen, encoding='utf-8')
frozen_sha = hashlib.sha256((OUT / 'frozen-candidate.json').read_bytes()).hexdigest()
(OUT / 'fit.json').write_text(json.dumps(fit, indent=2)+'\n', encoding='utf-8')
validation = np.stack([extract(s) for s in range(15, 30)])
assert hashlib.sha256((OUT / 'frozen-candidate.json').read_bytes()).hexdigest() == frozen_sha
prediction = pulse_response(relative, candidate['resistance_mohm'], candidate['tau_ms'], candidate['series_mohm'])
stored['prediction_mv'] = prediction
errors = []
for sweep in range(30):
    measured = voltages[sweep]
    baseline_error = float(np.sqrt(np.mean(measured**2)))
    candidate_error = float(np.sqrt(np.mean((measured-prediction)**2)))
    errors.append(dict(sweep=sweep, group='calibration' if sweep < 15 else 'validation',
                       baseline_mv=baselines[sweep], baseline_rmse_mv=baseline_error,
                       candidate_rmse_mv=candidate_error, error_ratio=candidate_error/baseline_error))
base_rmse = float(np.sqrt(np.mean(validation**2)))
new_rmse = float(np.sqrt(np.mean((validation-prediction)**2)))
failures = [e['sweep'] for e in errors[15:] if e['error_ratio'] > 1.05]
interior = fit['active_mask'][:2] == [0, 0]
accepted = fit['success'] and interior and new_rmse <= .75*base_rmse and not failures
np.savez_compressed(OUT / 'observations.npz', **stored)
result = dict(candidate_sha256=frozen_sha, candidate=candidate,
              calibration_sweeps=list(range(15)), validation_sweeps=list(range(15, 30)),
              per_sweep=errors, validation_baseline_rmse_mv=base_rmse,
              validation_candidate_rmse_mv=new_rmse,
              validation_improvement_fraction=1-new_rmse/base_rmse,
              validation_failed_sweeps=failures, resistance_tau_interior=interior,
              accepted_for_subsequent_synaptic_work=bool(accepted),
              physiology_qualified=False, scores_promoted=False,
              elapsed_seconds=time.monotonic()-started,
              source=source['source'], adc_physical_number=2, recorded_channel_index=1,
              verified_identical_dacs=[1, 2, 3], command_transitions=[start, stop],
              postsynaptic_filter_hz=abf._adcSection.fTelegraphFilter[1],
              source_clock='Original times retained; model time is relative integer sample index times 0.02 ms.')
(OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['per_sweep']}, indent=2))
