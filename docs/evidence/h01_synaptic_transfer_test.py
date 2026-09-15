"""Strength-only calibration cannot change the frozen waveform or inputs."""

import numpy as np
import pytest

from h01_synaptic_transfer import fit_connection_strength


def test_strength_matches_linear_least_squares_and_preserves_inputs():
    x = np.array([[0, .01, .02], [.001, .008, .025]])
    y = np.array([[.01, .4, .7], [.03, .28, 1.]])
    before_x, before_y = x.copy(), y.copy()
    reference = np.linalg.lstsq(x.reshape(-1, 1), y.ravel(), rcond=None)[0][0]
    result = fit_connection_strength(x, y)
    assert result['interior'] and result['peak_pa'] == pytest.approx(reference)
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


@pytest.mark.parametrize('strength', [-1, 0, 2001])
def test_inadmissible_strength_is_explicitly_bound_active(strength):
    x = np.array([.01, .02, .005])
    result = fit_connection_strength(x, strength*x)
    assert not result['interior']
    assert result['unconstrained_peak_pa'] == pytest.approx(strength)
    assert result['peak_pa'] in [.01, 2000]


@pytest.mark.parametrize('x,y', [([], []), ([0, 1], [1]), ([np.nan], [1]),
                               ([1], [np.inf]), ([0, 0], [1, 1])])
def test_unusable_calibration_is_rejected(x, y):
    with pytest.raises(ValueError): fit_connection_strength(x, y)


@pytest.mark.parametrize('low,high', [(0, 1), (1, 1), (2, 1), (.01, np.inf)])
def test_invalid_bounds(low, high):
    with pytest.raises(ValueError): fit_connection_strength([1], [1], minimum_pa=low, maximum_pa=high)
