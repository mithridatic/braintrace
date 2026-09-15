"""Analytic gate and matched recovery-observation tests."""

import numpy as np
import pytest
from h01_sodium_recovery_assay import relax, recovery_waveforms, fit_recovery_constant


def test_analytic_relaxation_and_composition():
    a = relax(.9, .2, 10., np.array([0., 10., 1000.]))
    np.testing.assert_allclose(a, [.9, .2+.7/np.e, .2])
    np.testing.assert_allclose(relax(relax(.9, .2, 10., 3.), .2, 10., 7.), a[1])


@pytest.mark.parametrize('args', [(-.1, .2, 1, 0), (.5, 1.1, 1, 0),
    (.5, .2, 0, 1), (.5, .2, 1, -1), (.5, .2, np.nan, 1), ([], [], [], [])])
def test_invalid_states(args):
    with pytest.raises(ValueError): relax(*args)


def test_assay_has_all_delays_and_strict_second_pulse_window():
    a = recovery_waveforms([.01, .9], [.9, .01], [.1, 1.], [[.02, .8]], [[.1, 20.]])
    np.testing.assert_array_equal(a['delays_ms'], np.arange(1., 13.)**2)
    assert a['second_pulse_states'].shape == (1, 12, 99, 2)
    np.testing.assert_allclose(a['second_pulse_time_ms'][[0, -1]], [.02, 1.98])
    h0 = .01+(.9-.01)*np.exp(-10)
    expected_h = .8+(h0-.8)*np.exp(-a['delays_ms']/20)
    np.testing.assert_allclose(a['recovered_states'][0, :, 1], expected_h)
    assert np.all((a['second_pulse_states'] >= 0) & (a['second_pulse_states'] <= 1))
    b = recovery_waveforms([.01, .9], [.9, .01], [.1, 1.], [[.02, .8]], [[.1, 20.]], dt_ms=.01)
    assert b['second_pulse_states'].shape[-2] == 199


@pytest.mark.parametrize('hold,dt', [([.1], .02), ([.1, .9], .03)])
def test_invalid_assay_shape_or_grid(hold, dt):
    with pytest.raises(ValueError):
        recovery_waveforms(hold, [.9, .01], [.1, 1.], [[.02, .8]], [[.1, 20.]], dt_ms=dt)


def test_known_current_recovery_oracle_and_gain_invariance():
    t = np.arange(1., 13.)**2
    y = .3+.7*(1-np.exp(-t/27.))
    for gain in [1., 100.]:
        result = fit_recovery_constant(t, gain*y)
        assert result['available'] and abs(result['tau_ms']-27) < 1e-6


@pytest.mark.parametrize('t,y', [([1, 2], [1, 2]), ([1, 2, 3, 4], [1, 2, 3]),
    ([1, 1, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [1, np.nan, 3, 4]),
    ([1, 2, 3, 4], [0, 1, 2, 3])])
def test_invalid_recovery_observations(t, y):
    with pytest.raises(ValueError): fit_recovery_constant(t, y)


def test_constant_response_is_unavailable():
    assert not fit_recovery_constant([1, 4, 9, 16], [1, 1, 1, 1])['available']


def test_optimizer_failure_is_unavailable(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('oracle failure')
    monkeypatch.setattr('h01_sodium_recovery_assay.curve_fit', failed)
    assert not fit_recovery_constant([1, 4, 9, 16], [1, 2, 3, 4])['available']
