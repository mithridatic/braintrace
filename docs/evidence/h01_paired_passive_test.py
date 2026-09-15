"""Analytic and synthetic-oracle tests for a human-pulse candidate."""

import numpy as np
import pytest

from h01_paired_passive import fit_response, pulse_response


def test_units_charge_step_return_and_series_jump():
    t = np.array([-1., 0., 10., 199.999999, 200., 210., 10000.])
    y = pulse_response(t, 100., 10., 20.)
    assert y[0] == 0
    assert y[1] == -.2
    assert y[2] == pytest.approx(-1*(1-np.exp(-1))-.2)
    assert y[4] == pytest.approx(-1*(1-np.exp(-20)))
    assert y[4]-y[3] == pytest.approx(.2, abs=1e-7)
    assert y[5] == pytest.approx(y[4]*np.exp(-1))
    assert abs(y[-1]) < 1e-100


def test_zero_current_and_sign_reversal():
    t = np.arange(-10., 401.)
    np.testing.assert_array_equal(pulse_response(t, 100., 10., 0., current_pa=0.), 0.)
    np.testing.assert_array_equal(pulse_response(t, 100., 10., 0., current_pa=10.),
                                  -pulse_response(t, 100., 10., 0.))


@pytest.mark.parametrize('time', [[], [[0, 1]], [0, 0], [1, 0], [0, np.nan]])
def test_invalid_clock(time):
    with pytest.raises(ValueError):
        pulse_response(time, 100., 10., 0.)


@pytest.mark.parametrize('parameters', [(0, 10, 0), (100, 0, 0),
                                      (100, 10, -1), (100, np.nan, 0)])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        pulse_response([0, 1], *parameters)


def test_invalid_duration_and_current():
    with pytest.raises(ValueError):
        pulse_response([0, 1], 100, 10, 0, duration_ms=0)
    with pytest.raises(ValueError):
        pulse_response([0, 1], 100, 10, 0, current_pa=np.inf)


def test_fit_recovers_independently_constructed_circuit():
    t = np.arange(401.)
    expected = np.empty_like(t)
    on = t < 200
    expected[on] = -10*(80*(1-np.exp(-t[on]/20))+5)/1000
    expected[~on] = -10*80*(1-np.exp(-10))*np.exp(-(t[~on]-200)/20)/1000
    fit = fit_response(t, np.stack([expected, expected]))
    assert fit['success']
    assert fit['resistance_mohm'] == pytest.approx(80, rel=1e-6)
    assert fit['tau_ms'] == pytest.approx(20, rel=1e-6)
    assert fit['series_mohm'] == pytest.approx(5, rel=1e-6)
    assert len(fit['evaluations']) >= fit['nfev']


@pytest.mark.parametrize('data', [[], [[0]], [[0, np.nan]], [0, 1]])
def test_invalid_calibration(data):
    with pytest.raises(ValueError):
        fit_response([0, 1], data)
