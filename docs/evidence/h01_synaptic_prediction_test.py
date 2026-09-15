"""Independent convolution, causality and observation-operator checks."""

import numpy as np
import pytest
from scipy.integrate import quad

from h01_synaptic_prediction import fit_synapse, synaptic_waveforms, train_prediction


@pytest.mark.parametrize('membrane_tau', [.3, 13., 20.])
def test_exact_convolution_including_coincident_time_constants(membrane_tau):
    amplitude, rise, decay, delay, resistance = 60., .3, 13., 1., 80.
    peak_time = rise*decay/(decay-rise)*np.log(decay/rise)
    norm = np.exp(-peak_time/decay)-np.exp(-peak_time/rise)
    for phase in [1., 5., 50.]:
        expected = quad(lambda u: amplitude*(np.exp(-u/decay)-np.exp(-u/rise))/norm
                        * resistance/membrane_tau*np.exp(-(phase-u)/membrane_tau)/1000,
                        0, phase, epsabs=1e-11)[0]
        _, voltage = synaptic_waveforms(phase+delay, [amplitude, rise, decay-rise, delay],
                                        resistance_mohm=resistance, membrane_tau_ms=membrane_tau)
        assert voltage == pytest.approx(expected, rel=1e-9, abs=1e-11)


def test_current_peak_causality_and_long_time_limit():
    peak_time = .3*3/2.7*np.log(10)
    current, voltage = synaptic_waveforms(np.array([-10., 0., 1., 1+peak_time, 10000.]),
                                         [60, .3, 2.7, 1], resistance_mohm=80, membrane_tau_ms=13)
    np.testing.assert_array_equal(current[:3], 0)
    np.testing.assert_array_equal(voltage[:3], 0)
    assert current[3] == pytest.approx(60)
    assert abs(voltage[-1]) < 1e-100


def lag_arrays(times, before, spikes):
    return (spikes[:, None]+times[None, :])[:, :, None]-spikes[None, None, :], \
           (spikes[:, None]+before[None, :])[:, :, None]-spikes[None, None, :]


def test_superposition_and_preceding_spike_baseline_tail():
    p = [60, .3, 2.7, 1]
    t, baseline, spikes = np.array([0., 2., 5.]), np.array([-10., -2.]), np.array([0., 20.])
    lags, pre = lag_arrays(t, baseline, spikes)
    predicted = train_prediction(lags[None], pre[None], p, resistance_mohm=80, membrane_tau_ms=13)
    settings = dict(resistance_mohm=80, membrane_tau_ms=13)
    first = synaptic_waveforms(t, p, **settings)[1]
    np.testing.assert_allclose(predicted[0, 0], first, atol=1e-14)
    second = first+synaptic_waveforms(t+20, p, **settings)[1]
    before = synaptic_waveforms(baseline+20, p, **settings)[1].mean()
    assert before > 0
    np.testing.assert_allclose(predicted[0, 1], second-before, atol=1e-14)


@pytest.mark.parametrize('parameters', [[1, 2, 3], [60, 0, 1, 0], [60, 1, 0, 0],
                                      [60, 1, 1, -1], [np.nan, 1, 1, 0]])
def test_bad_parameters(parameters):
    with pytest.raises(ValueError):
        synaptic_waveforms([0, 1], parameters, resistance_mohm=80, membrane_tau_ms=13)


def test_nonfinite_clock_and_cell_parameters():
    for time, resistance, tau in [([np.nan], 80, 13), ([0], np.inf, 13), ([0], 80, 0)]:
        with pytest.raises(ValueError):
            synaptic_waveforms(time, [60, .3, 2.7, 1], resistance_mohm=resistance, membrane_tau_ms=tau)


@pytest.mark.parametrize('score,baseline', [(np.zeros((1, 2)), np.zeros((1, 2))),
    (np.zeros((0, 2, 3, 2)), np.zeros((0, 2, 4, 2))),
    (np.zeros((1, 2, 3, 2)), np.zeros((1, 3, 4, 2)))])
def test_bad_window_shapes(score, baseline):
    with pytest.raises(ValueError):
        train_prediction(score, baseline, [60, .3, 2.7, 1], resistance_mohm=80, membrane_tau_ms=13)


def test_fit_recovery_and_invalid_observations():
    t = np.arange(0, 50.1, .2)
    lags, pre = lag_arrays(t, np.arange(-10, -2, .2), np.array([0., 200.]))
    lags, pre = lags[None], pre[None]
    settings = dict(resistance_mohm=80, membrane_tau_ms=13)
    expected = train_prediction(lags, pre, [80, .5, 3.5, 1.5], **settings)
    fit = fit_synapse(lags, pre, expected, **settings)
    assert fit['success']
    np.testing.assert_allclose(fit['parameters'], [80, .5, 3.5, 1.5], rtol=1e-4)
    assert len(fit['evaluations']) >= fit['nfev']
    with pytest.raises(ValueError):
        fit_synapse(lags, pre, expected[..., :-1], **settings)
