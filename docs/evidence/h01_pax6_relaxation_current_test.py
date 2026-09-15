"""Check the hypothesis change against the archived family and convolution."""
import numpy as np
import pytest
from scipy.integrate import quad
from docs.evidence.h01_pax6_bessel_current import predict_active as old_predict
from docs.evidence.h01_pax6_bessel_current_test import context, OBS
from docs.evidence.h01_pax6_recording_filter import bessel_impulse
from docs.evidence.h01_pax6_relaxation_current import INITIAL, LOWER, UPPER, prepare_commands, predict_active


def test_exact_reduction_to_archived_model():
    p = INITIAL.copy()
    p[13:] = p[6:8]
    np.testing.assert_array_equal(predict_active(p, context()), old_predict(p[:13], context()))


def test_independent_relaxation_preserves_asymptote_and_passive():
    times = np.array([10.3, 11., 20., 99999.])
    ctx = prepare_commands([[0, 10]], [[10, 100000]], [[-90, 10]], np.zeros(len(times), int), times, OBS)
    p = INITIAL.copy()
    changed = p.copy()
    changed[13] = -45
    first, second = predict_active(p, ctx), predict_active(changed, ctx)
    assert np.max(np.abs(first[:3]-second[:3])) > .1
    np.testing.assert_allclose(first[-1], second[-1], atol=1e-10)
    p[:3] = 0
    np.testing.assert_array_equal(predict_active(p, ctx), ctx['passive_pa'])


def test_new_relaxation_against_quadrature():
    p = INITIAL
    times = np.array([10.3, 11., 20., 39.])
    ctx = context(times)
    def states(v):
        m = 1/(1+np.exp((p[3]-v)/p[4]))
        h = 1/(1+np.exp((v-p[6])/p[7]))
        q = 1/(1+np.exp((v-p[13])/p[14]))
        return np.array([m,h,h]), np.array([p[5],p[8]+(p[10]-p[8])*q,p[9]+(p[11]-p[9])*q])
    initial, _ = states(-90)
    target, tau = states(30)
    def current(v, s):
        return (v-p[12])*s[0]**2*(p[0]*s[1]+p[1]*s[2]+p[2])
    base = current(-90, initial)
    def forcing(t):
        return current(30, target+(initial-target)*np.exp(-t/tau))-base
    expected = np.array([quad(lambda x: forcing(x)*bessel_impulse(t-x), 0, t, epsabs=1e-9)[0]
                         for t in times-10-OBS[4]])
    np.testing.assert_allclose(predict_active(p,ctx,components=True)['filtered_active_pa'], expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('parameter', range(15))
def test_all_derivatives(parameter):
    ctx = context()
    p = INITIAL.astype(complex)
    p[parameter] += 1e-25j
    derivative = predict_active(p,ctx).imag/1e-25
    high, low = INITIAL.copy(), INITIAL.copy()
    high[parameter] += 1e-4
    low[parameter] -= 1e-4
    finite = (predict_active(high,ctx)-predict_active(low,ctx))/2e-4
    np.testing.assert_allclose(derivative,finite,rtol=1e-4,atol=1e-7)
    assert np.max(np.abs(derivative)) > 1e-9


@pytest.mark.parametrize('fault', ['shape','nan','vtau','ktau'])
def test_invalid_parameters(fault):
    p = INITIAL.copy()
    if fault == 'shape': p = p[:-1]
    elif fault == 'nan': p[13] = np.nan
    elif fault == 'vtau': p[13] = LOWER[13]-1
    else: p[14] = UPPER[14]+1
    with pytest.raises(ValueError): predict_active(p, context())


def test_nonfinite_prediction():
    ctx = context()
    ctx['passive_pa'][0] = np.nan
    with pytest.raises(FloatingPointError): predict_active(INITIAL,ctx)
