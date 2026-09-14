"""Independent state-transition oracle and source-coordinate checks."""

import numpy as np
import pytest

from h01_l1_recovery_fit import fit_recovery, recovery_prediction


def observation():
    gaps = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000.])
    phase = np.arange(0., 301., 5.)
    p = np.array([180, 480, 45, 450, 65, 900, .08, .04])
    # Independent population-wise state calculation, not the prediction helper.
    current = np.zeros((len(gaps), len(phase)))
    for a, d, r, initial in zip(p[:2], p[2:4], p[4:6], p[6:]):
        after_first = initial * np.exp(-300 / d)
        before_second = after_first + (1 - after_first) * (-np.expm1(-gaps / r))
        first = a * initial * np.exp(-(phase - 10) / d)
        second = a * before_second[:, None] * np.exp(-(phase[None, :] - 10) / d)
        current += second - first
    return gaps, phase, current, p


def test_closed_form_matches_state_oracle_and_fit_recovers_parameters():
    gaps, phase, current, expected = observation()
    np.testing.assert_allclose(recovery_prediction(expected, gaps, phase), current, atol=1e-12)
    arrays, meta = fit_recovery(gaps, phase, current)
    assert meta['optimizer_success'] and not meta['physiological_qualification']
    np.testing.assert_allclose(meta['parameters'], expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(arrays['residual_pa'], 0, atol=1e-5)


@pytest.mark.parametrize('fault', ['nan', 'shape', 'gaps', 'duplicate_phase', 'missing', 'few'])
def test_invalid_inputs_rejected(fault):
    g, t, y, _ = observation()
    if fault == 'nan': y[0, 0] = np.nan
    elif fault == 'shape': y = y[:-1]
    elif fault == 'gaps': g[0] = g[1]
    elif fault == 'duplicate_phase': t[0] = t[1]
    elif fault == 'missing': t, y = t[:-10], y[:, :-10]
    else:
        t = np.r_[np.arange(-100., -76.), 10., 280., 300.]
        y = np.zeros((len(g), len(t)))
    with pytest.raises(ValueError): fit_recovery(g, t, y)
