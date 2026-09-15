"""Independently verify source samples, circuit prediction and frozen decision."""

from pathlib import Path
import hashlib
import json
import re
import sys
from xml.etree import ElementTree

import numpy as np
import pyabf
import scipy
from pyabf.waveform import EpochTable

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
result = json.loads((OUT/'result.json').read_text())
data = np.load(OUT/'observations.npz')
frozen_path = OUT/'frozen-candidate.json'
assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == result['candidate_sha256']
candidate = json.loads(frozen_path.read_text())
source_path = ROOT/result['source']['path']
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == result['source']['sha256']
abf = pyabf.ABF(str(source_path))
assert abf.adcNames[1] == 'IN 2' and abf._adcSection.nADCNum[1] == 2
for channel in [1, 2, 3]:
    unit_index = abf._dacSection.lDACChannelUnitsIndex[channel]
    assert abf._stringsSection._indexedStrings[unit_index] == 'pA'
tables = [EpochTable(abf, channel) for channel in [1, 2, 3]]
t = data['relative_ms']
r, tau, rs = [candidate[k] for k in ['resistance_mohm', 'tau_ms', 'series_mohm']]
independent = np.empty_like(t)
on = t < 200
independent[on] = -.01*(r*(1-np.exp(-t[on]/tau))+rs)
independent[~on] = -.01*r*(1-np.exp(-200/tau))*np.exp(-(t[~on]-200)/tau)
np.testing.assert_allclose(data['prediction_mv'], independent, rtol=0, atol=1e-12)
assert candidate['calibration_sweeps'] == list(range(15))
assert candidate['validation_sweeps'] == list(range(15, 30))
validation = []
max_clock_difference = 0.
for sweep in range(30):
    begin = sweep*abf.sweepPointCount
    raw = abf.data[1, begin+153125:begin+173126]
    pre = abf.data[1, begin+150625:begin+152625]
    np.testing.assert_array_equal(data[f's{sweep}_raw_mv'], raw)
    np.testing.assert_array_equal(data[f's{sweep}_baseline_mv'], pre)
    source_time = np.arange(abf.sweepPointCount)*abf.dataSecPerPoint
    np.testing.assert_array_equal(data[f's{sweep}_source_time_s'], source_time[153125:173126])
    max_clock_difference = max(max_clock_difference,
        float(np.max(np.abs((source_time[153125:173126]-source_time[153125])*1000-t))))
    expected = np.zeros(abf.sweepPointCount)
    expected[153125:163125] = -10
    for table in tables:
        np.testing.assert_array_equal(table.epochWaveformsBySweep[sweep].getWaveform(), expected)
    np.testing.assert_array_equal(data[f's{sweep}_command_pa'], expected[153125:173126])
    assert result['per_sweep'][sweep]['baseline_mv'] == float(np.mean(pre))
    centered = raw-float(np.mean(pre))
    baseline_error = float(np.sqrt(np.mean(centered**2)))
    candidate_error = float(np.sqrt(np.mean((centered-independent)**2)))
    assert abs(result['per_sweep'][sweep]['baseline_rmse_mv']-baseline_error) < 1e-12
    assert abs(result['per_sweep'][sweep]['candidate_rmse_mv']-candidate_error) < 1e-12
    if sweep >= 15:
        validation.append(centered)
validation = np.stack(validation)
reference = float(np.sqrt(np.mean(validation**2)))
error = float(np.sqrt(np.mean((validation-independent)**2)))
assert abs(reference-result['validation_baseline_rmse_mv']) < 1e-12
assert abs(error-result['validation_candidate_rmse_mv']) < 1e-12
assert error <= .75*reference
assert all(e['error_ratio'] <= 1.05 for e in result['per_sweep'][15:])
assert result['accepted_for_subsequent_synaptic_work'] and not result['scores_promoted']
landmarks = json.loads((OUT/'direct-samples.json').read_text())
assert len(landmarks) == 66
for point in landmarks:
    absolute_index = point['sweep']*abf.sweepPointCount+point['source_index']
    assert point['human_mv'] == float(abf.data[1, absolute_index])
    local_index = point['source_index']-153125
    assert abs(point['prediction_change_mv']-independent[local_index]) < 1e-12
tests = ElementTree.parse(OUT/'tests.xml').getroot()
assert len(tests.findall('.//testcase')) == 34
assert not tests.findall('.//failure') and not tests.findall('.//error')
coverage = json.loads((OUT/'coverage.json').read_text())
assert coverage['totals']['percent_covered'] == 100
for link in re.findall(r'\]\(([^)]+)\)', (OUT/'README.md').read_text()):
    if '://' not in link:
        assert (OUT/link).is_file(), link
paths = [p for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json']
paths += [ROOT/'docs/evidence/h01_paired_passive.py',
          ROOT/'docs/evidence/h01_paired_passive_test.py',
          ROOT/'docs/specs/2026-09-14-h01-human-paired-passive.md']
receipt = dict(source_windows_verified=30, source_response_samples_verified=600030,
               source_baseline_samples_verified=60000, reconstructed_commands_verified=90,
               prediction_max_difference_mv=float(np.max(np.abs(data['prediction_mv']-independent))),
               original_clock_vs_index_clock_max_difference_ms=max_clock_difference,
               direct_samples_verified=len(landmarks),
               decision_recomputed=True, physiology_qualified=False, scores_promoted=False,
               tests_passed=34, helper_tests_passed=17, helper_line_coverage_percent=100,
               environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                pyabf=pyabf.__version__, execution='Local CPU closed-form circuit; no neuronal rollout'),
               files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
(OUT/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}, indent=2))
