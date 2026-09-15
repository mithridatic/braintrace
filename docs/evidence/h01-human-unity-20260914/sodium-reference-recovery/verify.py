"""Bind original human observations and independently check source rates and errors."""

from pathlib import Path
import hashlib
import json
import re
from xml.etree import ElementTree
import numpy as np

out = Path(__file__).resolve().parent
root = out.parents[3]
r = json.loads((out/'result.json').read_text())
d = np.load(out/'model-observations.npz')
for path, expected in r['source_sha256'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
original = json.loads((out.parent/'human-pyramidal-channels/sodium-human-records.json').read_text())
by_name = {x['Filename']: x for x in original['records']}
qc = {x['filename']: x for x in json.loads((out.parent/'human-channel-history/comparison.json').read_text())['human_recovered_fields']}
assert len(r['records']) == 133
for obs in r['records']:
    source = by_name[obs['filename']]; i = obs['condition']
    assert obs['excel_row'] == source['excel_row']
    assert obs['source_exclude'] == qc[obs['filename']]['recovered_fields']['exclude']
    for name, field in [('voltage_mv', 'ia_volt'), ('observed_tau_ms', 'ia_tau_h_rec'), ('source_fit_quality', 'ia_gof_h_rec')]:
        assert obs[name] == source.get(field+'_'+str(i))
    v, tau, quality = obs['voltage_mv'], obs['observed_tau_ms'], obs['source_fit_quality']
    expected = (obs['source_exclude'] == 0 and source['RsComp'] > 50
                and v is not None and tau is not None and quality is not None
                and -90 <= v <= -75 and tau > 0 and quality > .90)
    assert obs['eligible'] == expected
    if expected:
        p = next(p for p in r['predictions'] if p['voltage_mv'] == v)
        assert obs['predicted_tau_ms'] == p['current_recovery_fit']['tau_ms']
        assert obs['signed_error_ms'] == obs['predicted_tau_ms']-tau

# Read published numeric constants without importing/executing source code.
mod = (out/'318629-na_human.mod').read_text()
names = ['th_act_inf', 'q_act_inf', 'tha', 'qa', 'Ra', 'Rb', 'th_inact_inf',
         'q_inact_inf', 'thi1', 'thi2', 'qi1', 'qi2', 'Rd', 'Rg']
p = {name: float(re.search(r'^\s*'+name+r'\s*=\s*([-+0-9.eE]+)', mod, re.M)[1]) for name in names}
v = np.r_[-120., 0., d['voltage_mv']]
def _rate(voltage, threshold, coefficient, slope):
    delta = voltage-threshold
    return coefficient*delta/(-np.expm1(-delta/slope))
ma = _rate(v, p['tha'], p['Ra'], p['qa'])
mb = _rate(-v, -p['tha'], p['Rb'], p['qa'])
ha = _rate(v, p['thi1'], p['Rd'], p['qi1'])
hb = _rate(-v, -p['thi2'], p['Rg'], p['qi2'])
rates = np.array([1/(1+np.exp((p['th_act_inf']-v)/p['q_act_inf'])),
                  1/(1+np.exp((v-p['th_inact_inf'])/p['q_inact_inf'])),
                  1/(ma+mb), 1/(ha+hb)]).T
np.testing.assert_allclose(rates, np.vstack([d['holding_rates'], d['test_rates'], d['recovery_rates']]), atol=1e-12, rtol=0)
hold, test, recovery = rates[0], rates[1], rates[2:]
conditioned = test[:2]+(hold[:2]-test[:2])*np.exp(-10/test[2:])
recovered = recovery[:, None, :2]+(conditioned-recovery[:, None, :2])*np.exp(-d['delays_ms'][None, :, None]/recovery[:, None, 2:])
states = test[:2]+(recovered[:, :, None, :]-test[:2])*np.exp(-d['second_pulse_time_ms'][None, None, :, None]/test[2:])
np.testing.assert_allclose(states, d['second_pulse_states'], atol=1e-13, rtol=0)
np.testing.assert_allclose(d['inward_proxy'], 141*states[..., 0]**3*states[..., 1], atol=1e-11, rtol=0)
for index, pred in enumerate(r['predictions']):
    fit = pred['current_recovery_fit']; a, tau, offset = fit['fitted_parameters']
    y = d['peaks'][index]-d['peaks'][index, -1]
    yf = a*np.exp(-(d['delays_ms']-d['delays_ms'][0])/tau)+offset
    r2 = 1-np.sum((y-yf)**2)/np.sum((y-y.mean())**2)
    assert abs(r2-fit['r_squared']) < 1e-14 and fit['available'] and r2 > .95
    fine = pred['refined_current_recovery_fit']['tau_ms']
    assert abs(pred['relative_refinement_difference']-abs(tau-fine)/fine) < 1e-14
    assert pred['numerical_pass'] == (abs(tau-fine)/fine <= .01)
eligible = [x for x in r['records'] if x['eligible']]
assert len(eligible) == 76
errors = np.array([x['signed_error_ms'] for x in eligible])
assert abs(np.sqrt(np.mean(errors**2))-r['comparison_rmse_ms']) < 1e-13
bound = json.loads((out/'shared-prediction-bound.json').read_text())
within, bias = 0., 0.
for pred in r['predictions']:
    y = np.array([x['observed_tau_ms'] for x in eligible if x['voltage_mv'] == pred['voltage_mv']])
    within += np.sum((y-y.mean())**2)
    bias += y.size*(y.mean()-pred['current_recovery_fit']['tau_ms'])**2
assert abs(within+bias-np.sum(errors**2)) < 1e-10
assert abs(np.sqrt(within/76)-bound['unrestricted_shared_prediction_minimum_rmse_ms']) < 1e-13
tests = ElementTree.parse(out/'tests.xml').getroot()
assert len(tests.findall('.//testcase')) == 18 and not tests.findall('.//failure')
coverage = json.loads((out/'coverage.json').read_text())
assert all(x['summary']['percent_covered'] == 100 for x in coverage['files'].values())
receipt = dict(source_conditions_verified=133, eligible_conditions_verified=76,
               published_source_rate_values_verified=24, second_pulse_gate_values_verified=int(states.size),
               all_errors_verified=True, numerical_gate_verified=True, shared_prediction_bound_verified=True,
               tests_passed=18, helper_line_coverage_percent=100, scores_promoted=False)
(out/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
