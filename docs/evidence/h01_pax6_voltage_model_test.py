"""Independent numerical and physical-limit checks for the current candidate."""

import numpy as np
import pytest
from scipy.integrate import quad

from docs.evidence.h01_pax6_voltage_model import INITIAL, filtered_exponential, predict


def protocol():
    return np.array([[0., 10.]]), np.array([[10., 110.]]), np.array([[-90., -40.]])


def passive():
    p = INITIAL.copy()
    p[[0, 1, 2, 13, 14, 16]] = 0
    return p


def test_leak_filter_oracle_and_components():
    p = passive()
    p[13] = 1
    t = np.array([0., 5., 10., 10.04, 10.1, 11., 50.])
    values = predict(p, *protocol(), np.zeros(len(t), int), t, components=True)
    expected = 50 * -np.expm1(-np.maximum(t - 10, 0) / p[18])
    np.testing.assert_allclose(values['filtered_pa'], expected, atol=1e-12)
    np.testing.assert_array_equal(values['active_relative_pa'], 0)
    np.testing.assert_array_equal(values['recording_transient_pa'], 0)
    np.testing.assert_array_equal(values['leak_pa'], np.where(t >= 10, 50, 0))


@pytest.mark.parametrize('tau', [.08, .08000000001, .2, 4.])
def test_filtered_capacitive_step(tau):
    p = passive()
    p[14], p[15] = 2., tau
    phase = np.array([0., .04, .1, .5, 3.])
    value = predict(p, *protocol(), np.zeros(len(phase), int), 10 + phase)
    expected = np.array([quad(lambda x: 100 / tau * np.exp(-x / tau)
                              * np.exp(-(t - x) / p[18]) / p[18], 0, t,
                              epsabs=1e-11)[0] for t in phase])
    np.testing.assert_allclose(value, expected, rtol=1e-8, atol=1e-9)


def test_active_current_against_independent_quadrature():
    p = INITIAL.copy()
    p[[13, 14, 16]] = 0
    v0, v1 = -90., -40.
    m0, m1 = (1 / (1 + np.exp((p[3] - v) / p[4])) for v in (v0, v1))
    h0, h1 = (1 / (1 + np.exp((v - p[6]) / p[7])) for v in (v0, v1))
    tau1, tau2 = p[8] + (p[10] - p[8]) * h1, p[9] + (p[11] - p[9]) * h1
    initial = (v0 - p[12]) * m0**2 * ((p[0] + p[1]) * h0 + p[2])
    def forcing(t):
        m = m1 + (m0 - m1) * np.exp(-t / p[5])
        a = h1 + (h0 - h1) * np.exp(-t / tau1)
        b = h1 + (h0 - h1) * np.exp(-t / tau2)
        return (v1 - p[12]) * m**2 * (p[0] * a + p[1] * b + p[2]) - initial
    phase = np.array([.04, .3, 2., 10.])
    expected = np.array([quad(lambda x: forcing(x) * np.exp(-(t - x) / p[18]) / p[18],
                              0, t, epsabs=1e-10)[0] for t in phase])
    values = predict(p, *protocol(), np.zeros(len(phase), int), 10 + phase)
    np.testing.assert_allclose(values, expected, rtol=1e-8, atol=1e-9)


def test_subdivision_and_long_hold_composition():
    p = INITIAL.copy()
    t = np.array([0., 2., 10., 10.04, 50., 10000., 100010.])
    a = predict(p, [[0, 10]], [[10, 100100]], [[-90, -40]], np.zeros(len(t), int), t)
    b = predict(p, [[0, 4, 10, 20, 5000, 90000]],
                [[4, 10, 20, 5000, 90000, 100100]],
                [[-90, -90, -40, -40, -40, -40]], np.zeros(len(t), int), t)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
    assert np.isfinite(a).all()


def test_batch_padding_and_state_reset():
    starts = np.array([[0, 10, 110], [0, 5, 40]])
    stops = np.array([[10, 110, 110], [5, 40, 110]])
    volts = np.array([[-90, -40, -40], [-20, 60, -90]])
    ids, t = np.array([0, 0, 1, 1]), np.array([10.1, 70, 5.1, 60])
    combined = predict(INITIAL, starts, stops, volts, ids, t)
    first = predict(INITIAL, starts[:1, :2], stops[:1, :2], volts[:1, :2], [0, 0], t[:2])
    second = predict(INITIAL, starts[1:], stops[1:], volts[1:], [0, 0], t[2:])
    np.testing.assert_allclose(combined, np.r_[first, second], atol=1e-12)


@pytest.mark.parametrize('parameter', [0, 3, 5, 6, 9, 12, 14, 17, 18])
def test_complex_step_derivative(parameter):
    p = INITIAL.astype(complex)
    p[parameter] += 1e-25j
    derivative = predict(p, *protocol(), [0, 0], [10.1, 40.]).imag / 1e-25
    a, b = INITIAL.copy(), INITIAL.copy()
    step = 1e-5
    a[parameter] += step
    b[parameter] -= step
    finite = (predict(a, *protocol(), [0, 0], [10.1, 40.])
              - predict(b, *protocol(), [0, 0], [10.1, 40.])) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=1e-5, atol=1e-7)


def test_equal_rate_complex_derivative():
    rate, tau, t = 5., .2, .7
    derivative = filtered_exponential(rate + 1e-25j, t, tau).imag / 1e-25
    expected = -np.exp(-t / tau) * t**2 / (2 * tau)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)


@pytest.mark.parametrize('fault', ['parameter_shape', 'parameter_nan', 'bounds', 'shape',
                                 'gap', 'negative_duration', 'initial_time', 'zero',
                                 'middle_padding', 'nonfinite_command', 'query_shape',
                                 'query_float_id', 'query_negative_id', 'query_missing_id',
                                 'query_negative_time', 'query_missing_time', 'query_nan', 'empty'])
def test_invalid_input(fault):
    p = INITIAL.copy()
    starts, stops, voltage = protocol()
    ids, t = np.array([0]), np.array([20.])
    if fault == 'parameter_shape': p = p[:-1]
    elif fault == 'parameter_nan': p[0] = np.nan
    elif fault == 'bounds': p[5] = 0
    elif fault == 'shape': starts = starts[0]
    elif fault == 'gap': starts[0, 1] += 1
    elif fault == 'negative_duration': stops[0, 1] = 9
    elif fault == 'initial_time': starts[0, 0] = 1
    elif fault == 'zero': starts[:] = 0; stops[:] = 0
    elif fault == 'middle_padding': starts[0, 1] = 0; stops[0, 0] = 0
    elif fault == 'nonfinite_command': voltage[0, 0] = np.nan
    elif fault == 'query_shape': t = np.array([1., 2.])
    elif fault == 'query_float_id': ids = np.array([0.])
    elif fault == 'query_negative_id': ids = np.array([-1])
    elif fault == 'query_missing_id': ids = np.array([1])
    elif fault == 'query_negative_time': t[0] = -1
    elif fault == 'query_missing_time': t[0] = 110
    elif fault == 'query_nan': t[0] = np.nan
    elif fault == 'empty': ids, t = np.array([], int), np.array([])
    with pytest.raises(ValueError):
        predict(p, starts, stops, voltage, ids, t)
