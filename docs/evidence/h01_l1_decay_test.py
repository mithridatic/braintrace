"""Check descriptive fitting against a known envelope and missing observations."""

import numpy as np
import pytest

from h01_l1_decay import fit_decay


def test_known_two_scale_oracle_and_datum():
    t = np.arange(1100., 2100., 1.)
    x = t - 1110
    y = 20 + 300 * np.exp(-x / 25) + 500 * np.exp(-x / 300)
    arrays, result = fit_decay(t, y)
    np.testing.assert_allclose(result['time_constants_ms'], [25, 300], rtol=1e-7)
    np.testing.assert_allclose(result['amplitudes_pa'], [300, 500], rtol=1e-7)
    assert result['offset_pa'] == pytest.approx(20, abs=1e-7)
    assert result['optimizer_success'] and not result['physiological_qualification']
    np.testing.assert_allclose(arrays['residual_pa'], 0, atol=1e-7)
    np.testing.assert_array_equal(arrays['time_ms'], t[(t >= 1110) & (t < 2090)])


@pytest.mark.parametrize('fault', ['nan', 'shape', 'duplicate', 'short', 'late', 'window',
                                  'few_samples'])
def test_invalid_observations_do_not_produce_fit(fault):
    t, y = np.arange(1100., 2100.), np.ones(1000)
    window = (1110., 2090.)
    if fault == 'nan': y[5] = np.nan
    elif fault == 'shape': y = y[:-1]
    elif fault == 'duplicate': t[2] = t[1]
    elif fault == 'short': t, y = t[:500], y[:500]
    elif fault == 'late': t, y = t[100:], y[100:]
    elif fault == 'window': window = (2100, 1110)
    else: t, y = np.array([1100., 1200., 2100.]), np.ones(3)
    with pytest.raises(ValueError): fit_decay(t, y, window=window)


def test_approximately_linear_tail_exposes_parameter_boundary():
    t = np.arange(1100., 2100.)
    _, result = fit_decay(t, 700 - .3 * (t - 1110))
    assert result['optimizer_success']
    assert result['boundary_limited']
    assert not result['physiological_qualification']
