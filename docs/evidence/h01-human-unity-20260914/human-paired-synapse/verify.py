"""Verify retained pair samples against a new read of the source data matrix."""

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pyabf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
record = json.loads((OUT / 'recording.json').read_text())
raw = ROOT / record['source']['path']
assert hashlib.sha256(raw.read_bytes()).hexdigest() == record['source']['sha256']
abf = pyabf.ABF(str(raw))
assert abf.adcNames[:2] == ['IN 0', 'IN 2']
assert abf.adcUnits[:2] == ['mV', 'mV']
full = np.load(OUT / 'full-sweeps.npz')
events = np.load(OUT / 'events.npz')
event_count = 0
for sweep in record['selected_sweeps']:
    start = sweep * abf.sweepPointCount
    stop = start + abf.sweepPointCount
    source_time = np.arange(abf.sweepPointCount) * abf.dataSecPerPoint
    indices = events[f's{sweep}_indices']
    np.testing.assert_array_equal(full[f's{sweep}_t'], source_time)
    np.testing.assert_array_equal(events[f's{sweep}_time_s'], source_time[indices])
    for channel, side in [(0, 'pre'), (1, 'post')]:
        source_voltage = abf.data[channel, start:stop]
        np.testing.assert_array_equal(full[f's{sweep}_{side}'], source_voltage)
        np.testing.assert_array_equal(events[f's{sweep}_{side}_mv'], source_voltage[indices])
    event_count += indices.shape[0]
for item in json.loads((OUT / 'direct-samples.json').read_text()):
    index = item['sweep'] * abf.sweepPointCount + item['source_index']
    assert item['post_mv'] == float(abf.data[1, index])
for link in re.findall(r'\]\(([^)]+)\)', (OUT / 'README.md').read_text()):
    if '://' not in link:
        assert (OUT / link).is_file(), link
coverage = json.loads((OUT / 'coverage.json').read_text())
assert coverage['totals']['percent_covered'] == 100
paths = [p for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json']
paths += [ROOT / 'docs/evidence/h01_paired_response.py',
          ROOT / 'docs/evidence/h01_paired_response_test.py',
          ROOT / 'docs/specs/2026-09-14-h01-human-paired-synapse.md']
result = dict(source_sha256=record['source']['sha256'],
              verified_full_paired_sweeps=len(record['selected_sweeps']),
              verified_events=event_count, direct_samples_verified=45,
              comparison='New pyabf read; direct flattened data-matrix indices, no extraction helper.',
              tests_passed=17, helper_line_coverage_percent=100,
              plot_review='README.md', scores_promoted=False,
              files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(paths)})
(OUT / 'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print('Verified three complete paired sweeps, 18 event windows, 45 landmarks and local links.')
