"""Analytic current oracle and protocol-boundary checks for control subtraction."""

import numpy as np
import pytest

from h01_l1_leak import linear_step_estimate


def trace(step, offset=0.):
    t = np.arange(0., 3100., .1)
    window = (t >= 1100) & (t < 2100)
    command = np.where(window, -90. + step, -90.)
    response = np.zeros_like(t)
    # Analytic passive step response; this generates data, not a model rollout.
    response[window] = step * (.5 + 4 * np.exp(-(t[window] - 1100) / 2))
    return dict(time_ms=t, command_voltage_mv=command, total_current_pa=response + offset)


def test_analytic_passive_plus_active_oracle_and_offsets():
    signal, control = trace(160, 31.), trace(-10, -8.)
    t = signal['time_ms']
    w = (t >= 1100) & (t < 2100)
    active = 900 * (1 - np.exp(-(t[w] - 1100) / 3)) * np.exp(-(t[w] - 1100) / 400)
    signal['total_current_pa'][w] += active
    arrays, meta = linear_step_estimate(signal, control)
    np.testing.assert_allclose(arrays['conditional_current_pa'], active, atol=1e-12)
    np.testing.assert_array_equal(arrays['time_ms'], t[w])
    assert meta['scale'] == -16
    assert meta['signal_baseline_mean_pa'] == 31
    assert meta['control_baseline_mean_pa'] == -8
    assert not meta['ionic_isolation_qualified']


@pytest.mark.parametrize('fault', ['short', 'late_start', 'nan', 'duplicate_time', 'irregular',
                                  'clock', 'holding', 'prepulse', 'extra_step', 'no_return',
                                  'zero_step', 'shape', 'short_array'])
def test_invalid_or_incompatible_traces_fail(fault):
    signal, control = trace(160), trace(-10)
    t = control['time_ms']
    if fault == 'short': control = {k: a[t < 2000] for k, a in control.items()}
    elif fault == 'late_start': control = {k: a[t >= 1000] for k, a in control.items()}
    elif fault == 'nan': control['total_current_pa'][0] = np.nan
    elif fault == 'duplicate_time': t[1] = t[0]
    elif fault == 'irregular': t[1] += .01
    elif fault == 'clock': control['time_ms'] = t - .001
    elif fault == 'holding': control['command_voltage_mv'] += 1
    elif fault == 'prepulse': control['command_voltage_mv'][(t >= 1000) & (t < 1100)] += 70
    elif fault == 'extra_step': control['command_voltage_mv'][(t >= 1600) & (t < 1700)] += 1
    elif fault == 'no_return': control['command_voltage_mv'][t >= 2100] = -100
    elif fault == 'zero_step': control['command_voltage_mv'][:] = -90
    elif fault == 'shape': control['total_current_pa'] = np.zeros((2, 2))
    else: control = {k: a[:2] for k, a in control.items()}
    with pytest.raises(ValueError): linear_step_estimate(signal, control)


@pytest.mark.parametrize('baseline,pulse', [((990, 900), (1100, 2100)),
                                           ((900, 1200), (1100, 2100)),
                                           ((np.nan, 990), (1100, 2100))])
def test_invalid_windows(baseline, pulse):
    with pytest.raises(ValueError):
        linear_step_estimate(trace(160), trace(-10), baseline=baseline, pulse=pulse)


def test_insufficient_samples_and_constant_holding_tolerance():
    with pytest.raises(ValueError, match='Insufficient'):
        linear_step_estimate(trace(160), trace(-10), baseline=(900, 900.01))
    control = trace(-10)
    control['command_voltage_mv'] += .05
    arrays, meta = linear_step_estimate(trace(160), control)
    np.testing.assert_allclose(arrays['conditional_current_pa'], 0, atol=1e-10)
    assert meta['scale'] == pytest.approx(-16)


def test_different_sampling_rates_are_not_silently_interpolated():
    control = {k: a[::2] for k, a in trace(-10).items()}
    with pytest.raises(ValueError, match='clocks differ'):
        linear_step_estimate(trace(160), control)
