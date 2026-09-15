"""Evaluate frozen sodium reference-temperature recovery against human source rows."""

from pathlib import Path
import hashlib
import json
import shutil
import site
import sys
import time
import numpy as np

out = Path(__file__).resolve().parent
root = out.parents[3]
site.addsitedir(str(root.parent.parent/'.venv/Lib/site-packages'))
sys.path[:0] = [str(root), str(root/'docs/evidence')]
from braintrace.datasets.h01_wilbers import sodium_rates
from h01_sodium_recovery_assay import recovery_waveforms, fit_recovery_constant

if (out/'result.json').exists():
    raise FileExistsError('Preserve the completed reference comparison.')
started = time.monotonic()
source = out.parent/'human-pyramidal-channels/sodium-human-records.json'
history = out.parent/'human-channel-history/comparison.json'
human = json.loads(source.read_text())['records']
qc = {r['filename']: r for r in json.loads(history.read_text())['human_recovered_fields']}
assert len(human) == len(qc) == 19 and {r['Filename'] for r in human} == qc.keys()
observations = []
for row in human:
    old = qc[row['Filename']]
    assert row['excel_row'] == old['current_excel_row']
    for i in range(1, 8):
        v, tau, quality = [row.get(k+'_'+str(i)) for k in ['ia_volt', 'ia_tau_h_rec', 'ia_gof_h_rec']]
        reasons = []
        if old['recovered_fields']['exclude'] != 0: reasons.append('author exclusion')
        if row['RsComp'] is None or row['RsComp'] <= 50: reasons.append('compensation criterion')
        if v is None or tau is None or quality is None: reasons.append('missing observation')
        else:
            assert np.isfinite([v, tau, quality]).all()
            if quality <= .90: reasons.append('recovery fit quality')
            if v > -75: reasons.append('author recovery voltage restriction')
            if v < -90: reasons.append('outside model assay comparison domain')
            if tau <= 0: reasons.append('nonpositive observed time constant')
        observations.append(dict(filename=row['Filename'], excel_row=row['excel_row'],
                                 condition=i, voltage_mv=v, observed_tau_ms=tau,
                                 source_fit_quality=quality, source_exclude=old['recovered_fields']['exclude'],
                                 rscomp=row['RsComp'], eligible=not reasons, ineligibility=reasons))
volts = np.array(sorted({r['voltage_mv'] for r in observations if r['eligible']}))
assert np.array_equal(volts, [-90, -85, -80, -75])
hold = np.asarray(sodium_rates(-120., temperature_c=25.))
test = np.asarray(sodium_rates(0., temperature_c=25.))
rec = np.asarray(sodium_rates(volts, temperature_c=25.)).T
model = recovery_waveforms(hold[:2], test[:2], test[2:], rec[:, :2], rec[:, 2:])
refined = recovery_waveforms(hold[:2], test[:2], test[2:], rec[:, :2], rec[:, 2:], dt_ms=.01)
fitted = [fit_recovery_constant(model['delays_ms'], peaks) for peaks in model['peaks']]
fine = [fit_recovery_constant(refined['delays_ms'], peaks) for peaks in refined['peaks']]
predictions = []
for index, v in enumerate(volts):
    a, b = fitted[index], fine[index]
    difference = abs(a['tau_ms']-b['tau_ms'])/abs(b['tau_ms']) if a['available'] and b['available'] else None
    predictions.append(dict(voltage_mv=float(v), current_recovery_fit=a, refined_current_recovery_fit=b,
                            internal_h_tau_ms=float(rec[index, 3]), relative_refinement_difference=difference,
                            numerical_pass=bool(difference is not None and difference <= .01)))
lookup = {p['voltage_mv']: p for p in predictions}
for obs in observations:
    if obs['eligible']:
        p = lookup[obs['voltage_mv']]
        predicted = p['current_recovery_fit']['tau_ms']
        obs['predicted_tau_ms'] = predicted
        obs['signed_error_ms'] = predicted-obs['observed_tau_ms'] if predicted is not None else None
        obs['relative_error'] = obs['signed_error_ms']/obs['observed_tau_ms'] if predicted is not None else None
sources = [source, history, root/'braintrace/datasets/h01_wilbers.py',
           root/'docs/evidence/h01_sodium_recovery_assay.py',
           root/'docs/specs/2026-09-14-h01-sodium-reference-recovery.md']
metadata = json.loads((out.parent/'human-channel-history/versions.json').read_text())['data'][0]
for fid in [318563, 318629]:
    p = root/'.cache/human-source-followup/wilbers-pyramidal'/f'{fid}-na_human.mod'
    entry = next(f['dataFile'] for f in metadata['files'] if f['dataFile']['id'] == fid)
    assert p.stat().st_size == entry['filesize']
    assert hashlib.sha1(p.read_bytes()).hexdigest() == entry['checksum']['value']
    shutil.copyfile(p, out/p.name)
    sources.append(out/p.name)
np.savez_compressed(out/'model-observations.npz', voltage_mv=volts, recovery_rates=rec,
                    holding_rates=hold, test_rates=test, **model,
                    refined_second_pulse_time_ms=refined['second_pulse_time_ms'],
                    refined_peaks=refined['peaks'])
eligible = [r for r in observations if r['eligible']]
finite = [r for r in eligible if r['signed_error_ms'] is not None]
result = dict(temperature_c=25, records=observations, predictions=predictions,
              retained_source_conditions=len(observations), eligible_conditions=len(eligible),
              model_available_conditions=len(finite), numerical_pass=all(p['numerical_pass'] for p in predictions),
              comparison_rmse_ms=float(np.sqrt(np.mean([r['signed_error_ms']**2 for r in finite]))) if finite else None,
              model_parameters_changed=False, physiology_qualified=False, scores_promoted=False,
              source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              elapsed_seconds=time.monotonic()-started)
(out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['records', 'source_sha256']}, indent=2))
