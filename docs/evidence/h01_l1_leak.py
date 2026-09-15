"""Conditional linear-control subtraction with explicit waveform safeguards."""

import numpy as np


def linear_step_estimate(signal, control, *, baseline=(900., 990.), pulse=(1100., 2100.)):
    """Subtract a scaled control while preserving each time-resolved component.

    Parameters
    ----------
    signal, control : dict
        Exported arrays with time_ms, total_current_pa and command_voltage_mv.
        The caller must bind both arrays to the same recorded specimen/session.
    baseline : tuple of float, optional
        Half-open baseline window in ms. Its mean removes a constant offset only.
    pulse : tuple of float, optional
        Half-open response window in ms, with an observed return to baseline.

    Returns
    -------
    tuple of dict
        Arrays of raw baseline-centered response, scaled control and conditional
        difference; metadata including scale and baseline offsets. This is not
        proof of linear passive behavior or isolated potassium current.
    """
    if not (len(baseline) == len(pulse) == 2
            and np.isfinite((*baseline, *pulse)).all()
            and baseline[0] < baseline[1] <= pulse[0] < pulse[1]):
        raise ValueError('Invalid ordered analysis windows.')
    traces = []
    for arrays in (signal, control):
        t, current, command = (np.asarray(arrays[k], dtype=float)
                               for k in ('time_ms', 'total_current_pa', 'command_voltage_mv'))
        if (t.ndim != 1 or len(t) < 3 or current.shape != t.shape or command.shape != t.shape
                or not all(np.isfinite(x).all() for x in (t, current, command))):
            raise ValueError('Invalid time, current or command samples.')
        dt = np.diff(t)
        if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-8, atol=1e-9):
            raise ValueError('A uniform increasing clock is required; no interpolation is permitted.')
        if t[0] > baseline[0] or t[-1] < pulse[1]:
            raise ValueError('Incomplete baseline, pulse or return coverage.')
        b = (t >= baseline[0]) & (t < baseline[1])
        w = (t >= pulse[0]) & (t < pulse[1])
        pre = (t >= baseline[0]) & (t < pulse[0])
        if b.sum() < 2 or w.sum() < 2:
            raise ValueError('Insufficient window samples.')
        holding = float(command[b][0])
        if not np.allclose(command[pre], holding, rtol=0, atol=1e-7):
            raise ValueError('Nonconstant pre-step history; a prepulse cannot use simple scaling.')
        level = float(command[w][0])
        if not np.allclose(command[w], level, rtol=0, atol=1e-7):
            raise ValueError('Pulse command is not constant.')
        returned = command[np.flatnonzero(t >= pulse[1])[0]]
        if abs(returned - holding) > 1e-7:
            raise ValueError('Command does not return to baseline at the requested pulse end.')
        delta = level - holding
        if abs(delta) <= 1e-7:
            raise ValueError('A nonzero voltage step is required.')
        offset = float(np.mean(current[b]))
        traces.append((t, w, current[w] - offset, holding, delta, offset))
    st, sw, response, sh, sd, so = traces[0]
    ct, _, leak, ch, cd, co = traces[1]
    if not np.array_equal(st, ct):
        raise ValueError('Signal and control clocks differ.')
    if abs(sh - ch) > .1:
        raise ValueError('Signal and control holding voltages differ by more than 0.1 mV.')
    scale = sd / cd
    scaled = leak * scale
    return dict(time_ms=st[sw], centered_response_pa=response, scaled_control_pa=scaled,
                conditional_current_pa=response - scaled), dict(
                    signal_step_mv=sd, control_step_mv=cd, scale=scale,
                    signal_baseline_mean_pa=so, control_baseline_mean_pa=co,
                    signal_holding_mv=sh, control_holding_mv=ch,
                    baseline_ms=list(baseline), pulse_ms=list(pulse),
                    assumption='linear control response scales across command amplitudes',
                    ionic_isolation_qualified=False)
