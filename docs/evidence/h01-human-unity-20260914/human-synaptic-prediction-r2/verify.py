"""Verify source-indexed EPSPs and frozen predictions independently of fitting."""

from pathlib import Path
import hashlib
import inspect
import json
import re
import sys
from xml.etree import ElementTree

import numpy as np
import pyabf
import pyabf.waveform
import scipy

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
data = np.load(OUT/'observations.npz')
result = json.loads((OUT/'result.json').read_text())
candidate_path = OUT/'frozen-candidate.json'
assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == result['candidate_sha256']
candidate = json.loads(candidate_path.read_text())
cell_path = OUT.parent/'human-paired-passive/frozen-candidate.json'
assert hashlib.sha256(cell_path.read_bytes()).hexdigest() == candidate['cell_candidate_sha256']
source = ROOT/'.cache/human-paired-synapse/2015_11_04_0076.abf'
assert hashlib.sha256(source.read_bytes()).hexdigest() == candidate['source_sha256']
abf = pyabf.ABF(str(source))
assert abf.adcNames[:2] == ['IN 0', 'IN 2'] and abf.adcUnits[:2] == ['mV', 'mV']
epochs = pyabf.waveform.EpochTable(abf, 0).epochWaveformsBySweep[0]
epoch_record = {k: getattr(epochs, k) for k in ['p1s', 'p2s', 'types', 'levels', 'pulsePeriods', 'pulseWidths']}
(OUT/'source-epochs.json').write_text(json.dumps(epoch_record, indent=2)+'\n', encoding='utf-8')
expected_command = epochs.getWaveform().copy()
np.testing.assert_array_equal(data['prior_parser_command_pa'], expected_command)
expected_command[68125:68275] = 2500
np.testing.assert_array_equal(data['reconstructed_command_pa'], expected_command)
starts = np.flatnonzero(np.diff((expected_command > 0).astype(int)) == 1)+1
np.testing.assert_array_equal(starts, [28125, 38125, 48125, 58125, 68125, 93275])
failure_samples = np.load(OUT.parent/'human-synaptic-prediction/unexpected-crossing.npz')
for sweep in [0, 2]:
    begin = sweep*abf.sweepPointCount
    indices = failure_samples['source_indices']
    np.testing.assert_array_equal(failure_samples[f's{sweep}_pre_mv'], abf.data[0, begin+indices])
    np.testing.assert_array_equal(failure_samples[f's{sweep}_post_mv'], abf.data[1, begin+indices])
    np.testing.assert_array_equal(failure_samples[f's{sweep}_command_pa'], data['prior_parser_command_pa'][indices])
a, tr, gap, delay = candidate['synaptic_parameters']
td = tr+gap
r = candidate['frozen_cell']['resistance_mohm']
tm = candidate['frozen_cell']['membrane_tau_ms']
peak_time = tr*td/(td-tr)*np.log(td/tr)
norm = np.exp(-peak_time/td)-np.exp(-peak_time/tr)
observed, predicted = [], []
max_voltage_difference = 0.
max_current_difference = 0.
for sweep in range(30):
    begin = sweep*abf.sweepPointCount
    indices = data[f's{sweep}_indices']
    crosses = data[f's{sweep}_accepted_crossings']
    assert indices.shape == (6, 3001)
    assert np.all((crosses >= starts) & (crosses < starts+150))
    pre = abf.data[0, begin+indices]
    post = abf.data[1, begin+indices].astype(float)
    np.testing.assert_array_equal(data[f's{sweep}_pre_mv'], pre)
    np.testing.assert_array_equal(data[f's{sweep}_post_mv'], post)
    np.testing.assert_array_equal(data[f's{sweep}_time_s'], indices*abf.dataSecPerPoint)
    baseline = post[:, :400].mean(axis=-1)
    np.testing.assert_array_equal(data[f's{sweep}_baseline_mv'], baseline)
    x = np.maximum((indices[:, :, None]-crosses[None, None, :])*.02-delay, 0)
    current = a/norm*(np.exp(-x/td)-np.exp(-x/tr))
    decay_filter = td/(td-tm)*(np.exp(-x/td)-np.exp(-x/tm))
    rise_filter = tr/(tr-tm)*(np.exp(-x/tr)-np.exp(-x/tm))
    voltage = r*a/1000/norm*(decay_filter-rise_filter)
    raw_prediction = voltage.sum(axis=-1)
    centered_prediction = raw_prediction-raw_prediction[:, :400].mean(axis=-1)[:, None]
    max_voltage_difference = max(max_voltage_difference,
        float(np.abs(data[f's{sweep}_prediction_mv']-centered_prediction).max()))
    max_current_difference = max(max_current_difference,
        float(np.abs(data[f's{sweep}_inferred_current_pa']-current.sum(axis=-1)).max()))
    np.testing.assert_allclose(data[f's{sweep}_prediction_mv'], centered_prediction, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data[f's{sweep}_inferred_current_pa'], current.sum(axis=-1), atol=1e-10, rtol=0)
    obs = post[:, 500:]-baseline[:, None]
    pred = centered_prediction[:, 500:]
    observed.append(obs); predicted.append(pred)
    for event in range(6):
        record = result['per_event'][sweep*6+event]
        assert record['sweep'] == sweep and record['event'] == event
        assert abs(record['reference_rmse_mv']-float(np.sqrt(np.mean(obs[event]**2)))) < 1e-12
        assert abs(record['candidate_rmse_mv']-float(np.sqrt(np.mean((obs[event]-pred[event])**2)))) < 1e-12
    record = result['per_sweep'][sweep]
    assert abs(record['candidate_rmse_mv']-float(np.sqrt(np.mean((obs-pred)**2)))) < 1e-12
    all_crosses = np.flatnonzero((abf.data[0, begin:begin+199999] < 0)
                                & (abf.data[0, begin+1:begin+200000] >= 0))+1
    extras = all_crosses[~np.isin(all_crosses, crosses)]
    np.testing.assert_array_equal(data[f's{sweep}_unassigned_crossings'], extras)
    assert extras.tolist() == ([68280] if sweep == 2 else [])
observed, predicted = np.stack(observed), np.stack(predicted)
base = float(np.sqrt(np.mean(observed[15:]**2)))
error = float(np.sqrt(np.mean((observed[15:]-predicted[15:])**2)))
assert abs(base-result['validation_reference_rmse_mv']) < 1e-12
assert abs(error-result['validation_candidate_rmse_mv']) < 1e-12
fit = json.loads((OUT/'fit.json').read_text())
accepted = fit['success'] and fit['active_mask'] == [0]*4 and error <= .75*base
accepted = accepted and all(e['error_ratio'] <= 1.05 for e in result['per_sweep'][15:])
assert accepted == result['accepted_for_further_qualification']
points = json.loads((OUT/'direct-samples.json').read_text())
assert len(points) == 108
for point in points:
    index = point['sweep']*abf.sweepPointCount+point['source_index']
    assert point['pre_mv'] == float(abf.data[0, index])
    assert point['post_mv'] == float(abf.data[1, index])
tests = ElementTree.parse(OUT/'tests.xml').getroot()
assert len(tests.findall('.//testcase')) == 60
assert not tests.findall('.//failure') and not tests.findall('.//error')
coverage = json.loads((OUT/'coverage.json').read_text())
assert all(f['summary']['percent_covered'] > 90 for f in coverage['files'].values())
for folder in [OUT, OUT.parent/'human-synaptic-prediction']:
    for link in re.findall(r'\]\(([^)]+)\)', (folder/'README.md').read_text()):
        if '://' not in link:
            assert (folder/link).is_file(), link
paths = [p for folder in [OUT, OUT.parent/'human-synaptic-prediction']
         for p in folder.iterdir() if p.is_file() and p.name != 'verification.json']
paths += [ROOT/'docs/evidence'/name for name in ['h01_synaptic_prediction.py',
          'h01_synaptic_prediction_test.py', 'h01_abf_pulses.py', 'h01_abf_pulses_test.py']]
paths += [ROOT/'docs/specs/2026-09-14-h01-human-synaptic-prediction.md']
receipt = dict(source_sha256=candidate['source_sha256'], source_events_verified=180,
               paired_voltage_samples_verified=1080360, scored_samples_verified=450180,
               direct_samples_verified=108, unassigned_crossings_retained=1,
               max_voltage_difference_mv=max_voltage_difference, max_current_difference_pa=max_current_difference,
               decision_recomputed=True, tests_passed=60, scores_promoted=False,
               pyabf_waveform_source_sha256=hashlib.sha256(Path(inspect.getfile(pyabf.waveform)).read_bytes()).hexdigest(),
               environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pyabf=pyabf.__version__),
               files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
(OUT/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['files', 'environment']}, indent=2))
