"""Pulse pairing, drift retention and original-clock tests."""

import numpy as np
import pytest

from h01_l1_recovery import paired_recovery


def source(gap=50.):
    t = np.arange(0., 3600 + gap, .5)
    first = (t >= 2100) & (t < 2400)
    second = (t >= 2400 + gap) & (t < 2700 + gap)
    v = np.full_like(t, -20.)
    v[first | second] = 60
    v[(t >= 2400) & (t < 2400 + gap)] = -90
    i = .001 * t
    i[first] += 10 * np.exp(-(t[first] - 2100) / 100)
    i[second] += 110 * np.exp(-(t[second] - 2400 - gap) / 100)
    return dict(time_ms=t, command_voltage_mv=v, total_current_pa=i)


def test_exact_pairing_preserves_drift_and_original_times():
    a = source()
    # Ensure the final baseline endpoint has an actual observation.
    a = {k: np.r_[v, v[-1] + .5 if k == 'time_ms' else v[-1]] for k, v in a.items()}
    result, meta = paired_recovery(a)
    phase = result['phase_ms']
    expected = 100 * np.exp(-phase / 100) + .35
    np.testing.assert_allclose(result['difference_pa'], expected, atol=1e-12)
    np.testing.assert_array_equal(result['first_time_ms'], 2100 + phase)
    np.testing.assert_array_equal(result['second_time_ms'], 2450 + phase)
    assert meta['gap_ms'] == 50
    assert meta['baseline_change_pa'] > 1
    assert not meta['drift_correction_applied'] and not meta['ionic_isolation_qualified']


@pytest.mark.parametrize('fault', ['nan', 'shape', 'clock', 'extra', 'pulse_length',
                                  'pulse_voltage', 'gap_voltage', 'return_voltage', 'missing'])
def test_invalid_protocols_fail(fault):
    a = source()
    t, v = a['time_ms'], a['command_voltage_mv']
    if fault == 'nan': a['total_current_pa'][0] = np.nan
    elif fault == 'shape': a['total_current_pa'] = a['total_current_pa'][:-1]
    elif fault == 'clock': t[2] = t[1]
    elif fault == 'extra': v[(t >= 1500) & (t < 1600)] = 5
    elif fault == 'pulse_length': v[(t >= 2750) & (t < 2800)] = 60
    elif fault == 'pulse_voltage': v[(t >= 2450) & (t < 2750)] = 61
    elif fault == 'gap_voltage': v[(t >= 2400) & (t < 2450)] = 70
    elif fault == 'return_voltage': v[t >= 2750] = -21
    else: a = {k: x[t < 3200] for k, x in a.items()}
    with pytest.raises(ValueError): paired_recovery(a)
