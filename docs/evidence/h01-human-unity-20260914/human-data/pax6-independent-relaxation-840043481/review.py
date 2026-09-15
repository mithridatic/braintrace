"""Score complete acquired responses and preserve the prospective access gate."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_pax6_relaxation_current import prepare_commands, predict_active

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rms = lambda a: float(np.sqrt(np.mean(a**2)))
inputs = json.loads((out / 'training-inputs.json').read_bytes())
candidate = json.loads((out / 'frozen-candidate.json').read_bytes())
assert sha(out / 'training-inputs.json') == candidate['training_inputs_sha256']
assert sha(out / inputs['source_arrays_path']) == inputs['arrays_sha256']
assert sha(repo / 'docs/evidence/h01_pax6_relaxation_current.py') == candidate['model_source_sha256']
assert sha(repo / 'docs/specs/2026-09-14-h01-pax6-independent-relaxation.md') == candidate['spec_sha256']
observer_dir = out.parent / 'pax6-recording-filter-840043481'
assert sha(observer_dir / 'frozen-candidate.json') == candidate['observer_sha256']
for name, digest in candidate['dependency_sha256'].items():
    assert sha(repo / 'docs/evidence' / name) == digest
records = inputs['records']
with np.load(out / inputs['source_arrays_path']) as source:
    raw = dict(source)
ids = np.concatenate([np.full(r['full_scoring_samples'], j, int) for j, r in enumerate(records)])
times = np.concatenate([raw[f"s{r['sweep']}_time_ms"][875:] for r in records])
observed = np.concatenate([raw[f"s{r['sweep']}_total_current_pa"][875:]-r['baseline_pa'] for r in records])
ctx = prepare_commands(raw['command_starts'], raw['command_stops'], raw['command_volts'], ids, times, inputs['observation_parameters'])
full = predict_active(candidate['parameters'], ctx, components=True)
full.update(protocol_ids=ids, time_ms=times, observed_pa=observed, residual_pa=full['filtered_pa']-observed)
with (out / 'training-predictions.npz').open('xb') as stream:
    np.savez_compressed(stream, **full)

with np.load(out.parent / 'pax6-active-refit-840043481/training-predictions.npz') as saved:
    previous_fit = dict(saved)
np.testing.assert_array_equal(ids, previous_fit['protocol_ids'])
np.testing.assert_array_equal(times, previous_fit['time_ms'])
np.testing.assert_array_equal(observed, previous_fit['observed_pa'])
errors = []
for j, r in enumerate(records):
    mask = ids == j
    fit_indices = raw[f"s{r['sweep']}_fit_indices"]
    fit_weights = raw[f"s{r['sweep']}_fit_weights"]
    assert fit_indices[0] == 875
    np.testing.assert_array_equal(fit_weights, np.diff(np.r_[fit_indices, r['samples']]))
    errors.append(dict(sweep=r['sweep'], samples=int(mask.sum()), candidate_rmse_pa=rms(full['residual_pa'][mask]),
                       previous_rmse_pa=rms(previous_fit['residual_pa'][mask]),
                       passive_rmse_pa=rms(full['passive_pa'][mask]-observed[mask])))

observer_inputs = json.loads((observer_dir / 'inputs.json').read_bytes())
with np.load(observer_dir / 'predictions.npz') as saved:
    old = dict(saved)
controls = []
for sweep in [72, 76, 83, 99, 101, 103]:
    row = next(j for j, r in enumerate(records) if r['sweep'] == sweep)
    old_row = next(j for j, r in enumerate(observer_inputs['records']) if r['sweep'] == sweep)
    mask = (ids == row) & (times < 70)
    previous = (old['protocol_ids'] == old_row) & (old['time_ms'] < 70)
    np.testing.assert_array_equal(times[mask], old['time_ms'][previous])
    np.testing.assert_array_equal(observed[mask], old['observed_pa'][previous])
    np.testing.assert_allclose(full['passive_pa'][mask], old['predicted_pa'][previous], atol=1e-10, rtol=1e-10)
    controls.append(dict(sweep=sweep, window='35-70 ms', candidate_rmse_pa=rms(full['residual_pa'][mask]),
                         observer_rmse_pa=rms(old['residual_pa'][previous])))
row = next(j for j, r in enumerate(records) if r['sweep'] == 83)
old_row = next(j for j, r in enumerate(observer_inputs['records']) if r['sweep'] == 83)
mask, previous = ids == row, old['protocol_ids'] == old_row
np.testing.assert_array_equal(observed[mask], old['observed_pa'][previous])
controls.append(dict(sweep=83, window='35 ms onward', candidate_rmse_pa=rms(full['residual_pa'][mask]),
                     observer_rmse_pa=rms(old['residual_pa'][previous])))
preserved = all(r['candidate_rmse_pa'] <= 1.05*r['observer_rmse_pa'] for r in controls)
prerequisite = candidate['optimizer_success'] and not candidate['parameter_at_bound'] and preserved
decision = dict(status='ready for frozen prospective predictions' if prerequisite else 'rejected before new response access',
                optimizer_success=candidate['optimizer_success'], parameter_at_bound=candidate['parameter_at_bound'],
                observer_preserved=preserved, full_prediction_finite=True, full_samples=len(times),
                full_rmse_pa=rms(full['residual_pa']), previous_full_rmse_pa=rms(previous_fit['residual_pa']),
                same_sample_improvement_percent=100*(1-rms(full['residual_pa'])/rms(previous_fit['residual_pa'])), passive_rmse_pa=rms(full['passive_pa']-observed),
                training_errors=errors, observation_controls=controls, prediction_responses_opened=False,
                external_responses_opened=False, whole_cell_responses_opened=False,
                scores_promoted=False, physiology_qualified=False)
with (out / 'decision.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(decision, indent=2) + '\n')
print(json.dumps(decision), flush=True)

# Rendering loops only arrange already calculated sample arrays; no simulation loop.
for name, selected in [('depolarization', [70, 74, 78]), ('conditioning', [89, 93, 97]),
                        ('recovery', [123, 132, 141]), ('tails', [99, 101, 103])]:
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    for panels, sweep in zip(axes, selected):
        row = next(j for j, r in enumerate(records) if r['sweep'] == sweep)
        mask = ids == row
        t = times[mask]
        panels[0].plot(raw[f's{sweep}_time_ms'], raw[f's{sweep}_command_voltage_mv'], lw=.7)
        panels[0].set(ylabel=f'Sweep {sweep}: DAC command (mV)')
        panels[1].plot(t, observed[mask], lw=.6, label='Human amplifier current')
        panels[1].plot(t, previous_fit['filtered_pa'][mask], color='gray', lw=.7, label='Previous candidate')
        panels[1].plot(t, full['filtered_pa'][mask], lw=.8, label='New candidate')
        panels[1].set(ylabel='Current (pA)')
        panels[2].plot(t, full['residual_pa'][mask], lw=.6)
        panels[2].set(ylabel='Candidate minus human (pA)')
        for ax in panels:
            ax.set(xlabel='Original sweep time (ms)')
            ax.axhline(0, color='gray', lw=.4)
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(f'{name}: complete original samples; training responses')
    fig.savefig(out / f'{name}.png', dpi=120)
    plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
for panels, sweep in zip(axes, [123, 132, 141]):
    row = next(j for j, r in enumerate(records) if r['sweep'] == sweep)
    transitions = [s for s in records[row]['command_segments'] if s['start_ms'] >= 2000]
    for ax, segment in zip(panels, transitions[:3]):
        onset = segment['start_ms']
        mask = (ids == row) & (times >= onset-.2) & (times < min(onset+30, segment['stop_ms']))
        ax.plot(times[mask]-onset, observed[mask], lw=.7, label='Human')
        ax.plot(times[mask]-onset, previous_fit['filtered_pa'][mask], color='gray', lw=.7, label='Previous')
        ax.plot(times[mask]-onset, full['filtered_pa'][mask], lw=.9, label='New candidate')
        ax.set(title=f'{sweep}: edge at {onset:g} ms', xlabel='Time after DAC edge (ms)', ylabel='Current (pA)')
axes[0, 0].legend()
fig.savefig(out / 'recovery-onsets.png', dpi=120)
plt.close(fig)
fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
reference_row = next(j for j, r in enumerate(records) if r['sweep'] == 103)
for panels, sweep in zip(axes, [123, 132, 141]):
    row = next(j for j, r in enumerate(records) if r['sweep'] == sweep)
    for ax, bounds in zip(panels, [(44.92, 46.), (46., 65.)]):
        mask = (ids == row) & (times >= bounds[0]) & (times < bounds[1])
        reference = (ids == reference_row) & (times >= bounds[0]) & (times < bounds[1])
        ax.plot(times[reference]-45, observed[reference], color='gray', lw=.7, label='Human 103: -90 mV hold')
        ax.plot(times[mask]-45, observed[mask], lw=.8, label=f'Human {sweep}: -20 mV hold')
        ax.plot(times[mask]-45, full['passive_pa'][mask], lw=.8, label='Frozen passive observer')
        ax.plot(times[mask]-45, full['filtered_pa'][mask], lw=.9, label='New active plus observer')
        ax.set(xlabel='Time after control DAC edge (ms)', ylabel='Baseline-relative current (pA)')
axes[0, 0].legend(fontsize=7)
fig.suptitle('Acquired small controls: holding voltage is confounded with acquisition order')
fig.savefig(out / 'holding-controls.png', dpi=120)
plt.close(fig)
with (out / 'verification.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(source_and_candidate_hashes_verified=True, original_samples_scored=len(times),
                                 source_masks_and_time_weights_verified=True, frozen_observer_predictions_matched=True,
                                 prediction_responses_opened=False, physiology_qualified=False), indent=2)+'\n')
