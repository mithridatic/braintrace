"""Check new filter metadata while proving all prior human export arrays unchanged."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_l1_voltage_clamp import read_sweep

prior = out.parent / 'pax6-voltage-model-840043481'
metadata = json.loads((out / 'recorded-filter.json').read_bytes())
source = repo / '.cache/human-source-followup/pax6-840043481.nwb'
count = 0
with np.load(prior / 'training-source.npz') as original:
    for row in metadata['records']:
        arrays, exported = read_sweep(source, row['sweep'], session='840043481')
        for key, values in arrays.items():
            np.testing.assert_array_equal(values, original[f"s{row['sweep']}_{key}"])
        for key, field in row['fields'].items():
            assert exported['instrument'][key] == dict(value=field['value'], unit=field['stored_unit'])
        count += exported['samples']
receipt = dict(status='passed', source_sha256=metadata['source_sha256'],
               current_command_and_clock_samples_unchanged=count, sweeps=11,
               new_filter_fields_match_original_notebook=True,
               exporter_sha256=hashlib.sha256((repo / 'docs/evidence/h01_l1_voltage_clamp.py').read_bytes()).hexdigest())
with (out / 'export-verification.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
