"""Pair human recovery-protocol currents without silently correcting drift."""

import numpy as np

from docs.evidence.h01_l1_voltage_clamp import command_segments


def paired_recovery(arrays):
    """Extract two equal-voltage pulses and retain their original time boundaries.

    Parameters
    ----------
    arrays : dict
        Source-bound time_ms, total_current_pa and command_voltage_mv arrays.

    Returns
    -------
    tuple of dict
        Paired pulse and baseline arrays, plus actual command and timing metadata.
        Differences include potential drift and imperfect capacitive cancellation.
    """
    t, current, command = (np.asarray(arrays[k], dtype=float)
                           for k in ('time_ms', 'total_current_pa', 'command_voltage_mv'))
    if (t.ndim != 1 or len(t) < 3 or current.shape != t.shape or command.shape != t.shape
            or not all(np.isfinite(a).all() for a in (t, current, command))):
        raise ValueError('Invalid aligned samples.')
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-8, atol=1e-9):
        raise ValueError('A uniform increasing source clock is required.')
    rate = 1000. / dt[0]
    segments = command_segments(command, rate)
    later = [s for s in segments if t[s['start_index']] >= 1000.]
    if len(later) != 4:
        raise ValueError('Expected first pulse, recovery gap, second pulse and return.')
    first, gap, second, returned = later
    a, b = first['start_index'], first['stop_index']
    c, d = second['start_index'], second['stop_index']
    if a == 0 or b - a != d - c or not np.isclose((b - a) * dt[0], 300., atol=1e-6):
        raise ValueError('Recovery pulses must both last 300 ms.')
    holding = float(command[a - 1])
    if (abs(first['command_mv'] - second['command_mv']) > 1e-7
            or abs(returned['command_mv'] - holding) > 1e-7
            or not gap['command_mv'] < holding < first['command_mv']):
        raise ValueError('Pulse, recovery or holding commands are incompatible.')
    phase = t[a:b] - t[a]
    if not np.allclose(phase, t[c:d] - t[c], rtol=0, atol=1e-8):
        raise ValueError('Pulse phase clocks differ.')
    before = (t[a] - 100., t[a] - 10.)
    after = (t[d] + 500., t[d] + 900.)
    if t[0] > before[0] or t[-1] < after[1]:
        raise ValueError('Baseline windows are not fully observed.')
    masks = [(t >= lo) & (t < hi) for lo, hi in (before, after)]
    if any(m.sum() < 2 or not np.allclose(command[m], holding, rtol=0, atol=1e-7)
           for m in masks):
        raise ValueError('Baseline window is not observed at the holding command.')
    pre, post = masks
    result = dict(phase_ms=phase, first_time_ms=t[a:b], second_time_ms=t[c:d],
                  first_current_pa=current[a:b], second_current_pa=current[c:d],
                  difference_pa=current[c:d] - current[a:b],
                  initial_baseline_time_ms=t[pre], initial_baseline_current_pa=current[pre],
                  final_baseline_time_ms=t[post], final_baseline_current_pa=current[post])
    metadata = dict(gap_ms=float(t[c] - t[b]), first_onset_ms=float(t[a]),
                    second_onset_ms=float(t[c]), pulse_duration_ms=float((b - a) * dt[0]),
                    pulse_command_mv=first['command_mv'], gap_command_mv=gap['command_mv'],
                    holding_command_mv=holding, initial_baseline_mean_pa=float(current[pre].mean()),
                    final_baseline_mean_pa=float(current[post].mean()),
                    baseline_change_pa=float(current[post].mean() - current[pre].mean()),
                    drift_correction_applied=False, ionic_isolation_qualified=False)
    return result, metadata
