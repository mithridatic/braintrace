"""Run the preregistered human-only effective recovery estimate on local CPU."""

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
repo = root.parents[3]
sys.path.insert(0, str(root.parents[1]))
from h01_k_recovery import extract_curves, fit_recovery, recovery

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
write = lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
if (root/'result.json').exists():
    raise FileExistsError('Preserve completed evidence; use a new prefix.')
started = time.monotonic()
source = root.parent/'human-pyramidal-channels/potassium-human-records.json'
sealed = json.loads((source.parent/'analysis-receipt.json').read_bytes())
assert sha(source) == sealed['artifact_sha256'][source.name]
curves, excluded = extract_curves(json.loads(source.read_bytes())['records'])
groups = sorted({c['group'] for c in curves})
assert groups == ['H21.29.190', 'H21.29.191']
write(root/'inputs.json', dict(source_file=source.relative_to(repo).as_posix(),
    source_sha256=sha(source), curves=curves, excluded=excluded,
    source_temperature_c=34, recovery_command_mv=-80,
    original_experimental_current_traces_acquired=False,
    physiological_qc_applied=False, species='Human'))


def predict(curve, fit):
    t = np.asarray(curve['times_ms'])
    observed = np.asarray(curve['ratios'])
    predicted = recovery(t, fit['parameters'], fit['family'])
    residual = predicted-observed
    return dict(filename=curve['filename'], group=curve['group'], times_ms=t.tolist(),
        observed_ratio=observed.tolist(), predicted_ratio=predicted.tolist(),
        residual_ratio=residual.tolist(), rmse=float(np.sqrt(np.mean(residual**2))),
        phase_rmse={name:float(np.sqrt(np.mean(residual[mask]**2))) if mask.any() else None
                    for name,mask in [('early',t<=300),('intermediate',(t>300)&(t<=2000)),('late',t>2000)]})


all_fits, folds = {}, []
for family in ('single', 'double'):
    for held_group in groups:
        training = [c for c in curves if c['group'] != held_group]
        validation = [c for c in curves if c['group'] == held_group]
        assert not {c['filename'] for c in training} & {c['filename'] for c in validation}
        fit = fit_recovery(training, family)
        predictions = [predict(c, fit) for c in validation]
        folds.append(dict(family=family, held_group=held_group,
            training_identities=[c['filename'] for c in training],
            validation_identities=[c['filename'] for c in validation],
            fit=fit, predictions=predictions,
            rmse=float(np.sqrt(np.mean([p['rmse']**2 for p in predictions])))))
        write(root/'partial-fits.json', dict(completed_folds=folds))
    fit = fit_recovery(curves, family)
    all_fits[family] = dict(fit=fit, predictions=[predict(c, fit) for c in curves])

score = {family:float(np.sqrt(np.mean([p['rmse']**2 for f in folds if f['family']==family for p in f['predictions']])))
         for family in ('single','double')}
group_scores = {group:{f['family']:f['rmse'] for f in folds if f['held_group']==group} for group in groups}
preferred = score['double'] < score['single'] and all(s['double'] <= s['single'] for s in group_scores.values())
result = dict(status='double_exponential_preferred_for_protocol_followup' if preferred else 'double_exponential_not_selected',
    folds=folds, all_record_estimates=all_fits, grouped_validation_rmse=score,
    per_group_validation_rmse=group_scores, double_selected=preferred,
    elapsed_seconds=time.monotonic()-started, host='local Windows CPU',
    original_nwb_traces_acquired=False, parameters_installed=False, scores_promoted=False,
    current_constants_are_channel_gate_rates=False, independent_donor_validation=False,
    blind_holdout=False, recorded_before_fit_spec_sha256=sha(repo/'docs/specs/2026-09-14-h01-human-potassium-recovery.md'),
    executed_files_sha256={p.relative_to(repo).as_posix():sha(p) for p in
        (Path(__file__).resolve(), repo/'docs/evidence/h01_k_recovery.py')})
write(root/'result.json', result)
print(json.dumps({k:result[k] for k in ('status','grouped_validation_rmse','per_group_validation_rmse','elapsed_seconds')}))
