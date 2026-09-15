"""Independently verify original samples, analytical fit and transfer rejection."""

from pathlib import Path
import hashlib
import json
import re
import sys
from xml.etree import ElementTree
import numpy as np
import pyabf

out = Path(__file__).resolve().parent
root = out.parents[3]
sys.path.insert(0, str(root/'docs/evidence'))
from h01_abf_pulses import command_from_epochs
from pyabf.waveform import EpochTable
data = np.load(out/'observations.npz')
result = json.loads((out/'result.json').read_text())
c = json.loads((out/'frozen-candidate.json').read_text())
assert hashlib.sha256((out/'frozen-candidate.json').read_bytes()).hexdigest() == result['candidate_sha256']
source = root/'.cache/human-paired-synapse/2015_11_04_0077.abf'
assert hashlib.sha256(source.read_bytes()).hexdigest() == c['source_sha256']
abf = pyabf.ABF(str(source))
first = pyabf.ABF(str(root/'.cache/human-paired-synapse/2015_11_04_0076.abf'))
prior_command = np.load(out.parent/'human-synaptic-prediction-r2/observations.npz')['reconstructed_command_pa']
for waveform in EpochTable(first, 0).epochWaveformsBySweep:
    np.testing.assert_array_equal(command_from_epochs(waveform), prior_command)
cell_path = out.parent/'human-paired-passive/frozen-candidate.json'
kernel_path = out.parent/'human-synaptic-prediction-r2/frozen-candidate.json'
assert hashlib.sha256(cell_path.read_bytes()).hexdigest() == c['original_cell_candidate_sha256']
assert hashlib.sha256(kernel_path.read_bytes()).hexdigest() == c['original_synaptic_candidate_sha256']
cell = json.loads(cell_path.read_text())
electrical_time = np.arange(20001)*.02
decay_step = np.exp(-np.minimum(electrical_time, 200)/cell['tau_ms'])
epred_independent = -.01*(cell['resistance_mohm']*(1-decay_step)
                         *np.exp(-np.maximum(electrical_time-200, 0)/cell['tau_ms'])
                         +cell['series_mohm']*(electrical_time < 200))
np.testing.assert_allclose(epred_independent, data['electrical_prediction_mv'], atol=1e-14, rtol=0)
amplitude = c['strength']['peak_pa']
_, rise, gap, delay = c['unit_synaptic_parameters']
decay = rise+gap
r, tau = c['frozen_cell']['resistance_mohm'], c['frozen_cell']['membrane_tau_ms']
peak_time = rise*decay/gap*np.log(decay/rise)
norm = np.exp(-peak_time/decay)-np.exp(-peak_time/rise)
observations, templates, electrical = [], [], []
direct = []
max_voltage_difference = 0.
max_baseline_precision_difference = 0.
for sweep in range(30):
    begin = sweep*abf.sweepPointCount
    pre = abf.data[2, begin:begin+abf.sweepPointCount]
    crosses = np.flatnonzero((pre[:-1] < 0) & (pre[1:] >= 0))+1
    np.testing.assert_array_equal(data[f's{sweep}_accepted_crossings'], crosses)
    assert crosses.size == 5
    starts = np.array([28125, 38125, 48125, 58125, 93225])
    assert np.all((crosses >= starts) & (crosses < starts+150))
    anchors = np.insert(crosses, 4, 68125)
    idx = anchors[:, None]+np.arange(-500, 2501)
    np.testing.assert_array_equal(data[f's{sweep}_indices'], idx)
    np.testing.assert_array_equal(data[f's{sweep}_observation_anchors'], anchors)
    np.testing.assert_array_equal(data[f's{sweep}_pre_mv'], pre[idx])
    post = abf.data[1, begin+idx]
    np.testing.assert_array_equal(data[f's{sweep}_post_mv'], post)
    np.testing.assert_array_equal(data[f's{sweep}_time_s'], idx*abf.dataSecPerPoint)
    baseline = post[:, :400].mean(axis=1)
    np.testing.assert_array_equal(data[f's{sweep}_baseline_mv'], baseline)
    max_baseline_precision_difference = max(max_baseline_precision_difference,
        float(abs(baseline-post[:, :400].astype(float).mean(axis=1)).max()))
    x = np.maximum((idx[:, :, None]-crosses[None, None, :])*.02-delay, 0)
    filtered_decay = decay/(decay-tau)*(np.exp(-x/decay)-np.exp(-x/tau))
    filtered_rise = rise/(rise-tau)*(np.exp(-x/rise)-np.exp(-x/tau))
    unit = (r/1000/norm*(filtered_decay-filtered_rise)).sum(axis=-1)
    unit -= unit[:, :400].mean(axis=1)[:, None]
    max_voltage_difference = max(max_voltage_difference,
        float(abs(unit-data[f's{sweep}_unit_prediction_mv_per_pa']).max()))
    np.testing.assert_allclose(unit, data[f's{sweep}_unit_prediction_mv_per_pa'], atol=1e-13, rtol=0)
    current = amplitude/norm*(np.exp(-x/decay)-np.exp(-x/rise))
    np.testing.assert_allclose(current.sum(axis=-1), data[f's{sweep}_inferred_current_pa'], atol=1e-10, rtol=0)
    observations.append(post[:, 500:]-baseline[:, None]); templates.append(unit[:, 500:])
    eraw = abf.data[1, begin+153125:begin+173126]
    ebase = abf.data[1, begin+150625:begin+152625]
    np.testing.assert_array_equal(data[f's{sweep}_electrical_raw_mv'], eraw)
    np.testing.assert_array_equal(data[f's{sweep}_electrical_baseline_samples_mv'], ebase)
    electrical.append(eraw-ebase.mean())
    if sweep in [0, 1, 14, 15, 16, 29]:
        for slot in [0, 1, 4, 5]:
            for offset in [0, 100, 500]:
                index = int(anchors[slot]+offset)
                direct.append(dict(sweep=sweep, slot=slot, source_index=index,
                                   anchor_kind='epoch clock; no spike' if slot == 4 else 'recorded spike',
                                   pre_mv=float(pre[index]), post_mv=float(abf.data[1, begin+index])))
obs, template = np.stack(observations), np.stack(templates)
# Independent SVD least squares, instead of the driver's dot-product quotient.
fitted = float(np.linalg.lstsq(template[:15].reshape(-1, 1), obs[:15].astype(float).ravel(), rcond=None)[0][0])
assert abs(fitted-amplitude) < 1e-10
pred = template*amplitude
for sweep in range(30):
    for slot in range(6):
        e = result['synaptic']['per_event'][sweep*6+slot]
        assert e['sweep'] == sweep and e['event'] == slot
        assert abs(e['candidate_rmse_mv']-np.sqrt(np.mean((obs[sweep, slot]-pred[sweep, slot])**2))) < 1e-12
        assert abs(e['reference_rmse_mv']-np.sqrt(np.mean(obs[sweep, slot]**2))) < 1e-12
for mask, record in [(np.ones(6, dtype=bool), result['synaptic']),
                     (np.array([True, True, True, True, False, True]), result['actual_spike_windows'])]:
    y, p = obs[15:, mask], pred[15:, mask]
    ref, error = float(np.sqrt(np.mean(y*y))), float(np.sqrt(np.mean((y-p)**2)))
    ratios = np.sqrt(np.mean((y-p)**2, axis=(1, 2)))/np.sqrt(np.mean(y*y, axis=(1, 2)))
    assert abs(ref-record['validation_reference_rmse_mv']) < 1e-12
    assert abs(error-record['validation_candidate_rmse_mv']) < 1e-12
    assert (np.flatnonzero(ratios > 1.05)+15).tolist() == record['failed_sweeps']
    assert not (error <= .75*ref and np.all(ratios <= 1.05))
eobs = np.stack(electrical)
epred = data['electrical_prediction_mv']
e_ref, e_error = np.sqrt(np.mean(eobs[15:]**2)), np.sqrt(np.mean((eobs[15:]-epred)**2))
assert abs(e_ref-result['electrical']['validation_reference_rmse_mv']) < 1e-12
assert abs(e_error-result['electrical']['validation_candidate_rmse_mv']) < 1e-12
assert e_error <= .75*e_ref
assert all(np.sqrt(np.mean((y-epred)**2)) <= 1.05*np.sqrt(np.mean(y*y)) for y in eobs[15:])
assert result['electrical']['passed'] and not result['transfer_passed']
tests = ElementTree.parse(out/'tests.xml').getroot()
assert len(tests.findall('.//testcase')) == 82 and not tests.findall('.//failure')
coverage = json.loads((out/'coverage.json').read_text())
assert all(x['summary']['percent_covered'] > 90 for x in coverage['files'].values())
for folder in [out, out.parent/'human-second-connection']:
    for link in re.findall(r'\]\(([^)]+)\)', (folder/'README.md').read_text()):
        if '://' not in link:
            assert (folder/link).is_file(), link
(out/'direct-samples.json').write_text(json.dumps(direct, indent=2)+'\n', encoding='utf-8')
receipt = dict(source_sha256=c['source_sha256'], source_spikes=150, observation_slots=180,
               paired_voltage_samples_verified=1080360, scored_samples=450180,
               electrical_samples_verified=660030, direct_samples_verified=len(direct),
               strength_independently_verified_pa=fitted,
               max_unit_voltage_difference_mv_per_pa=max_voltage_difference,
               max_float32_baseline_difference_from_float64_mv=max_baseline_precision_difference,
               tests_passed=82, gates_independently_verified=True, scores_promoted=False)
receipt['first_record_commands_unchanged_sweeps'] = 30
(out/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
