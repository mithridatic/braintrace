"""Verify raw-source membership, saved predictions and the candidate decision."""

from pathlib import Path
import hashlib
import json
import sys

import h5py
import numpy as np

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_pax6_voltage_model import predict

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = json.loads((out / 'training-inputs.json').read_bytes())
candidate = json.loads((out / 'frozen-candidate.json').read_bytes())
decision = json.loads((out / 'decision.json').read_bytes())
errors = json.loads((out / 'training-errors.json').read_bytes())
assert sha(repo / 'docs/evidence/h01_pax6_voltage_model.py') == candidate['model_source_sha256']
assert sha(repo / 'docs/specs/2026-09-14-h01-pax6-voltage-model.md') == candidate['spec_sha256']
assert sha(out / 'training-inputs.json') == candidate['training_inputs_sha256']
assert sha(out / 'training-source.npz') == inputs['arrays_sha256']
source = repo / '.cache/human-source-followup/pax6-840043481.nwb'
assert sha(source) == inputs['source_sha256']
verified_raw, verified_scores = 0, 0
with h5py.File(source, 'r') as nwb, np.load(out / 'training-source.npz') as raw_saved, np.load(out / 'training-predictions.npz') as scored:
    for index, metadata in enumerate(inputs['records']):
        sweep = metadata['sweep']
        group = nwb[f'acquisition/data_{sweep:05d}_AD0']
        raw = group['data']
        current = (raw[:].astype(float) * float(raw.attrs['conversion'])
                   + float(raw.attrs.get('offset', 0))) * 1e12
        np.testing.assert_array_equal(current, raw_saved[f's{sweep}_total_current_pa'])
        assert float(np.asarray(group['starting_time'][()]).item()) == metadata['absolute_start_s']
        assert float(group['starting_time'].attrs['rate']) == 25000
        t = np.arange(len(current)) / 25000 * 1000
        np.testing.assert_array_equal(t, raw_saved[f's{sweep}_time_ms'])
        baseline = float(current[875:1100].mean())
        assert baseline == metadata['baseline_pa']
        selection = np.arange(len(current)) % 25 == 0
        for segment in metadata['command_segments'][1:]:
            selection[segment['start_index']:segment['start_index'] + 250] = True
        selection[:875] = False
        fit_indices = np.flatnonzero(selection)
        np.testing.assert_array_equal(fit_indices, raw_saved[f's{sweep}_fit_indices'])
        np.testing.assert_array_equal(np.diff(np.r_[fit_indices, len(t)]), raw_saved[f's{sweep}_fit_weights'])
        mask = scored['protocol_ids'] == index
        np.testing.assert_array_equal(scored['observed_pa'][mask], current[875:] - baseline)
        np.testing.assert_array_equal(scored['time_ms'][mask], t[875:])
        error = scored['filtered_pa'][mask] - scored['observed_pa'][mask]
        np.testing.assert_array_equal(error, scored['residual_pa'][mask])
        assert float(np.sqrt(np.mean(error**2))) == errors[index]['rmse_pa']
        assert len(error) == errors[index]['samples']
        segments = metadata['command_segments']
        starts = [[s['start_ms'] for s in segments]]
        stops = [[s['stop_ms'] for s in segments]]
        volts = [[s['command_mv'] for s in segments]]
        queries = np.unique(np.linspace(0, len(error) - 1, 100).astype(int))
        repeated = predict(candidate['parameters'], starts, stops, volts,
                           np.zeros(len(queries), int), t[875:][queries])
        np.testing.assert_allclose(repeated, scored['filtered_pa'][mask][queries], rtol=1e-12, atol=1e-10)
        verified_raw += len(current)
        verified_scores += len(error)

if not decision['prediction_responses_opened']:
    assert not (out / 'prediction-observations.npz').exists()
    assert not candidate['optimizer_success'] or candidate['kinetic_parameter_at_bound']
    assert decision['status'] == 'rejected before new response access'
else:
    freeze = json.loads((out / 'prediction-freeze.json').read_bytes())
    assert sha(out / 'frozen-candidate.json') == freeze['candidate_sha256']
    assert sha(out / 'frozen-protocol-predictions.npz') == freeze['predictions_sha256']
    with np.load(out / 'prediction-observations.npz') as measured, np.load(out / 'frozen-protocol-predictions.npz') as predictions:
        residual = predictions['filtered_pa'] - measured['observed_pa']
        reference = predictions['passive_filtered_pa'] - measured['observed_pa']
        assert float(np.sqrt(np.mean(residual**2))) == decision['candidate_rmse_pa']
        assert float(np.sqrt(np.mean(reference**2))) == decision['passive_rmse_pa']
        for row, record in enumerate(decision['individual_sweeps']):
            mask = predictions['protocol_ids'] == row
            assert float(np.sqrt(np.mean(residual[mask]**2))) == record['candidate_rmse_pa']
            assert float(np.sqrt(np.mean(reference[mask]**2))) == record['passive_rmse_pa']
        passed = decision['candidate_rmse_pa'] <= .75 * decision['passive_rmse_pa'] and all(
            r['candidate_rmse_pa'] <= 1.05 * r['passive_rmse_pa'] for r in decision['individual_sweeps'])
        assert passed == (decision['status'] == 'eligible for further qualification')

verification = dict(status='passed', original_training_current_samples=verified_raw,
                    full_training_scoring_samples=verified_scores,
                    separately_recomputed_prediction_samples=1100,
                    source_and_candidate_hashes_verified=True,
                    prediction_access_and_decision_verified=True,
                    physiology_qualified=False, scores_promoted=False)
with (out / 'verification.json').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(verification, indent=2) + '\n')
print(json.dumps(verification))
