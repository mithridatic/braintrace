"""Fit and test two sodium rate factors against distinct human recording groups."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import site
import sys
import time

root=Path(__file__).resolve().parent
repo=root.parents[3]
primary=repo.parent.parent
site.addsitedir(str(primary/'.venv/Lib/site-packages'))
sys.path.insert(0,str(repo))
sys.path.insert(0,str(root.parents[1]))
import numpy as np
from scipy.optimize import least_squares
from h01_sodium_transfer import prepare_command,evaluate

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
if (root/'result.json').exists():raise FileExistsError('Preserve completed run.')
started=time.monotonic()
inputs=root.parent/'human-channel-protocol'
seal=json.loads((inputs/'analysis-receipt.json').read_bytes())
for name in ('commands.npz','human-hwide-responses.json'):
    assert sha(inputs/name)==seal['artifact_sha256'][name]
with np.load(inputs/'commands.npz') as a:
    voltage=a['train_sodium_mv'];peaks=a['train_peak_indices']
assert peaks.size==200
records=json.loads((inputs/'human-hwide-responses.json').read_bytes())['selected']
train=[r for r in records if r['filename'].startswith('H21.29.194.')]
validation=[r for r in records if not r['filename'].startswith('H21.29.194.')]
assert len(train)==7 and len(validation)==9
assert {'.'.join(r['filename'].split('.')[:3]) for r in validation}=={'H21.29.195','H21.29.197'}
target=np.array([r['ratios'] for r in train]);assert target.shape==(7,200) and np.isfinite(target).all()
prepared=prepare_command(voltage,peaks)
baseline_factors=np.repeat(2.3**.9,2)
baseline=evaluate(prepared,baseline_factors)
print(json.dumps(dict(stage='baseline',elapsed_seconds=time.monotonic()-started,
    calibration_rmse=float(np.sqrt(np.mean((baseline[0]-target)**2))),
    first_five=baseline[0][:5].tolist(),ap200=float(baseline[0][-1]))),flush=True)
evaluations=[]


def objective(log_factors):
    factors=np.exp(log_factors)
    ratio=evaluate(prepared,factors)[0]
    residual=(ratio-target).ravel()/np.sqrt(target.size)
    evaluations.append(dict(factors=factors.tolist(),rmse=float(np.linalg.norm(residual)),
                            elapsed_seconds=time.monotonic()-started))
    write(root/'evaluations.json',evaluations)
    print(json.dumps(dict(stage='fit',evaluation=len(evaluations),**evaluations[-1])),flush=True)
    return residual


fit=least_squares(objective,np.log(baseline_factors),bounds=(np.log([.25,.25]),np.log([8,8])),
                  max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-8)
factors=np.exp(fit.x)
frozen=dict(factors=factors.tolist(),optimizer_success=bool(fit.success),message=fit.message,
    nfev=fit.nfev,actual_evaluations=len(evaluations),cost=float(fit.cost),
    calibration_identities=[r['filename'] for r in train],
    validation_identities=[r['filename'] for r in validation],
    elapsed_seconds=time.monotonic()-started)
write(root/'frozen-candidate.json',frozen)
print(json.dumps(dict(stage='candidate_frozen',**frozen)),flush=True)
candidate=evaluate(prepared,factors)


def scores(ratio):
    rows=[]
    for record in records:
        observed=np.asarray(record['ratios']);residual=ratio-observed
        rows.append(dict(filename=record['filename'],group='.'.join(record['filename'].split('.')[:3]),
            split='calibration' if record['filename'].startswith('H21.29.194.') else 'validation',
            observed=observed.tolist(),predicted=ratio.tolist(),residual=residual.tolist(),
            rmse=float(np.sqrt(np.mean(residual**2))),
            ap1_ap5_ap200_residual=residual[[0,4,199]].tolist()))
    return rows


base_scores=scores(baseline[0]);candidate_scores=scores(candidate[0])
groups={group:{name:float(np.sqrt(np.mean([r['rmse']**2 for r in rows if r['group']==group])))
               for name,rows in [('baseline',base_scores),('candidate',candidate_scores)]}
        for group in ('H21.29.194','H21.29.195','H21.29.197')}
validation_rmse={name:float(np.sqrt(np.mean([r['rmse']**2 for r in rows if r['split']=='validation'])))
                 for name,rows in [('baseline',base_scores),('candidate',candidate_scores)]}
improvement=1-validation_rmse['candidate']/validation_rmse['baseline']
per_record_ok=all(c['rmse']<=1.05*b['rmse'] for b,c in zip(base_scores,candidate_scores,strict=True) if b['split']=='validation')
group_ok=all(groups[g]['candidate']<groups[g]['baseline'] for g in ('H21.29.195','H21.29.197'))
refined=prepare_command(voltage,peaks,substeps=2)
numerical={}
for name,rate_factors,coarse in [('baseline',baseline_factors,baseline),('candidate',factors,candidate)]:
    fine=evaluate(refined,rate_factors)
    delta=float(np.max(np.abs(coarse[0]-fine[0])))
    numerical[name]=dict(maximum_ratio_change=delta,passed=delta<=.001,fine_ratios=fine[0].tolist())
    direct={}
    for ap in (1,5,200):
        idx=np.arange(peaks[ap-1]-125,peaks[ap-1]+251)
        for key,value in [('source_index',idx),('time_ms',idx*.008),('voltage_mv',voltage[idx]),
                           ('m',coarse[2][idx,0]),('h',coarse[2][idx,1]),('inward_proxy_right',coarse[3][idx]),('inward_proxy_left',coarse[4][idx])]:
            direct[f'ap{ap}_{key}']=value
    np.savez_compressed(root/(name+'-direct.npz'),**direct)
accepted=bool(fit.success and group_ok and improvement>=.05 and per_record_ok and all(n['passed'] for n in numerical.values()))
result=dict(status='candidate_passes_comparative_gate' if accepted else 'candidate_rejected',accepted=accepted,
    baseline_factors=baseline_factors.tolist(),candidate_factors=factors.tolist(),group_rmse=groups,
    validation_rmse=validation_rmse,validation_relative_improvement=improvement,
    validation_groups_improved=group_ok,no_validation_record_worsens_over_five_percent=per_record_ok,
    numerical=numerical,baseline_records=base_scores,candidate_records=candidate_scores,
    elapsed_seconds=time.monotonic()-started,host='local CPU',simulation='compiled exact-gate scan; ideal held command',
    original_current_traces_used=False,measured_human_Q10=False,parameters_installed=False,scores_promoted=False,
    environment={name:importlib.metadata.version(name) for name in ('numpy','scipy','jax','brainstate')},
    source_sha256={p.relative_to(repo).as_posix():sha(p) for p in (inputs/'commands.npz',inputs/'human-hwide-responses.json',
        repo/'braintrace/datasets/h01_wilbers.py',repo/'docs/evidence/h01_sodium_transfer.py',
        repo/'docs/specs/2026-09-14-h01-sodium-thermal-transfer.md',Path(__file__).resolve())})
write(root/'result.json',result)
print(json.dumps({k:result[k] for k in ('status','validation_rmse','validation_relative_improvement','group_rmse','elapsed_seconds')}),flush=True)
