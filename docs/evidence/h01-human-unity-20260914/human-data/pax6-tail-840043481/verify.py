"""Independently verify saved tail observations against original HDF5 samples."""

from pathlib import Path
import hashlib
import json

import h5py
import numpy as np

out = Path(__file__).resolve().parent
repo = out.parents[4]
receipt = json.loads((out / 'observations.json').read_bytes())
source = repo / receipt['source_path']
assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['source_sha256']
assert hashlib.sha256((out / 'observations.npz').read_bytes()).hexdigest() == receipt['arrays_sha256']
readings = []
with h5py.File(source, 'r') as original, np.load(out / 'observations.npz') as saved:
    for row in receipt['records']:
        meta = row['metadata']
        sweep = meta['sweep']
        group = original[f'acquisition/data_{sweep:05d}_AD0']
        data = group['data']
        raw = (data[:].astype('float64') * float(data.attrs['conversion'])
               + float(data.attrs.get('offset', 0))) * 1e12
        np.testing.assert_array_equal(raw, saved[f's{sweep}_total_current_pa'])
        assert raw.shape == (90000,) and np.isfinite(raw).all()
        assert float(np.asarray(group['starting_time'][()]).item()) == meta['absolute_start_s']
        assert float(group['starting_time'].attrs['rate']) == 25000
        np.testing.assert_array_equal(saved[f's{sweep}_baseline_indices'], np.arange(25000, 27250))
        np.testing.assert_array_equal(saved[f's{sweep}_late_indices'], np.arange(63750, 64750))
        assert float(raw[25000:27250].mean()) == row['baseline_pa']
        assert float(raw[63750:64750].mean()) == row['tail_late_mean_pa']
        np.testing.assert_array_equal(raw[[52625, 53000, 54000, 57500, 64750]], row['raw_pa'])
        np.testing.assert_allclose(np.array(row['raw_pa']) - row['baseline_pa'], row['initial_centered_pa'], rtol=0, atol=0)
        np.testing.assert_allclose(np.array(row['raw_pa']) - row['tail_late_mean_pa'], row['late_difference_pa'], rtol=0, atol=0)
        readings.append(dict(sweep=sweep, conditioning_phases_ms=[10, 60, 200, 990],
                             conditioning_raw_pa=raw[[27750, 29000, 32500, 52250]].tolist(),
                             final_raw_mean_pa=float(raw[87500:89750].mean())))
result = dict(source_sha256=receipt['source_sha256'], verified_response_samples=450000,
              verified_baseline_and_late_windows=10, verified_tail_landmarks=25,
              conditioning_readings=readings, result='passed',
              independent_validation=False, physiology_qualified=False)
with (out / 'verification.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps(result))
