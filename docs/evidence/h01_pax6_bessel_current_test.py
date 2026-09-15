"""Independent convolution, state carry and sensitivity checks."""

import numpy as np
import pytest
from scipy.integrate import quad
from docs.evidence.h01_pax6_recording_filter import bessel_impulse, predict_control
from docs.evidence.h01_pax6_bessel_current import INITIAL, prepare_commands, predict_active

OBS = np.array([.2464313753349, 1.67019387655, 2.31670968226, 1.11036876524, .11959825201])


def context(times=None):
    if times is None:
        times = np.array([0., 9., 10.13, 10.3, 11., 20., 39., 40.2, 41., 55., 60.3, 70., 95.])
    return prepare_commands([[0, 10, 40, 60]], [[10, 40, 60, 100]],
                            [[-90, 30, -90, 30]], np.zeros(len(times), int), times, OBS)


def test_passive_limit_and_initial_equilibrium():
    ctx = context()
    p = INITIAL.copy()
    p[:3] = 0
    expected = predict_control(OBS, [[0, 10, 40, 60]], [[0, 120, -120, 120]], ctx['ids'], ctx['times'])
    np.testing.assert_array_equal(predict_active(p, ctx), expected)
    np.testing.assert_allclose(predict_active(INITIAL, ctx)[:2], 0, atol=1e-12)


def test_active_against_independent_quadrature():
    p = INITIAL
    times = np.array([10.13, 10.3, 11., 20., 39.])
    actual = predict_active(p, context(times), components=True)['filtered_active_pa']
    def gates(v):
        m = 1 / (1 + np.exp((p[3] - v) / p[4]))
        h = 1 / (1 + np.exp((v - p[6]) / p[7]))
        return np.array([m, h, h]), np.array([p[5], p[8] + (p[10]-p[8])*h, p[9] + (p[11]-p[9])*h])
    initial, _ = gates(-90)
    final, tau = gates(30)
    holding = (-90-p[12])*initial[0]**2*(p[0]*initial[1]+p[1]*initial[2]+p[2])
    def current(t):
        state = final + (initial-final)*np.exp(-t/tau)
        return (30-p[12])*state[0]**2*(p[0]*state[1]+p[1]*state[2]+p[2])-holding
    expected = np.array([quad(lambda s: current(s)*bessel_impulse(t-s), 0, t,
                              epsabs=1e-9)[0] for t in times-10-OBS[4]])
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_subdivision_filter_carry_and_batch_reset():
    ctx = context()
    subdivided = prepare_commands([[0, 5, 10, 20, 40, 50, 60, 80]],
                                 [[5, 10, 20, 40, 50, 60, 80, 100]],
                                 [[-90, -90, 30, 30, -90, -90, 30, 30]],
                                 ctx['ids'], ctx['times'], OBS)
    np.testing.assert_allclose(predict_active(INITIAL, ctx), predict_active(INITIAL, subdivided), atol=1e-10)
    batch = prepare_commands(np.repeat(ctx['start'], 2, axis=0), np.repeat(ctx['stop'], 2, axis=0),
                             np.repeat(ctx['voltage'], 2, axis=0), np.ones(len(ctx['ids']), int), ctx['times'], OBS)
    np.testing.assert_array_equal(predict_active(INITIAL, ctx), predict_active(INITIAL, batch))


def test_padding_long_hold_and_owned_arrays():
    start, stop, v = np.array([[0., 10., 100000.]]), np.array([[10., 100000., 100000.]]), np.array([[-90., 30., 30.]])
    ctx = prepare_commands(start, stop, v, [0], [99999.], OBS)
    v[:] = 0
    p = INITIAL
    m = 1/(1+np.exp((p[3]-30)/p[4]))
    h = 1/(1+np.exp((30-p[6])/p[7]))
    m0 = 1/(1+np.exp((p[3]+90)/p[4]))
    h0 = 1/(1+np.exp((-90-p[6])/p[7]))
    expected = (30-p[12])*m*m*((p[0]+p[1])*h+p[2])-(-90-p[12])*m0*m0*((p[0]+p[1])*h0+p[2])+120*OBS[0]
    np.testing.assert_allclose(predict_active(p, ctx), expected, atol=1e-10)


@pytest.mark.parametrize('parameter', range(13))
def test_complex_step_every_parameter(parameter):
    ctx = context()
    p = INITIAL.astype(complex)
    p[parameter] += 1e-25j
    derivative = predict_active(p, ctx).imag/1e-25
    high, low = INITIAL.copy(), INITIAL.copy()
    step = 1e-4
    high[parameter] += step
    low[parameter] -= step
    finite = (predict_active(high, ctx)-predict_active(low, ctx))/(2*step)
    np.testing.assert_allclose(derivative, finite, rtol=1e-4, atol=1e-7)
    assert np.max(np.abs(derivative)) > 1e-9


@pytest.mark.parametrize('fault', ['shape','gap','negative','nan','empty','padding','id','float_id','end','time_nan','obs','obs_bounds'])
def test_invalid_coverage(fault):
    a, b, v = np.array([[0., 10., 20.]]), np.array([[10., 20., 30.]]), np.array([[-90., 30., -90.]])
    ids, t, obs = np.array([0]), np.array([12.]), OBS.copy()
    if fault == 'shape': a = a[0]
    elif fault == 'gap': a[0, 1] = 11
    elif fault == 'negative': b[0, 1] = 9
    elif fault == 'nan': v[0, 1] = np.nan
    elif fault == 'empty': ids, t = np.array([], int), np.array([])
    elif fault == 'padding': b[0, 0] = a[0, 1] = 0
    elif fault == 'id': ids[0] = 1
    elif fault == 'float_id': ids = ids.astype(float)
    elif fault == 'end': t[0] = 30
    elif fault == 'time_nan': t[0] = np.nan
    elif fault == 'obs': obs = obs[:-1]
    elif fault == 'obs_bounds': obs[4] = -1
    with pytest.raises(ValueError): prepare_commands(a, b, v, ids, t, obs)


@pytest.mark.parametrize('fault', ['shape', 'nan', 'bounds'])
def test_invalid_parameters(fault):
    p = INITIAL.copy()
    if fault == 'shape': p = p[:-1]
    elif fault == 'nan': p[0] = np.nan
    else: p[0] = -1
    with pytest.raises(ValueError): predict_active(p, context())


def test_nonfinite_output_rejected():
    ctx = context()
    ctx['passive_pa'][0] = np.nan
    with pytest.raises(FloatingPointError): predict_active(INITIAL, ctx)
