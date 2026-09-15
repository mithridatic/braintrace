"""Exact constant-voltage sodium states and a current-recovery observation."""

import numpy as np
from scipy.optimize import curve_fit


def relax(initial, equilibrium, tau_ms, elapsed_ms):
    """Evaluate bounded gate relaxation at constant voltage.

    Parameters
    ----------
    initial, equilibrium, tau_ms, elapsed_ms : array_like
        Broadcastable initial/steady fractions, positive time constants and
        nonnegative elapsed times, in milliseconds.

    Returns
    -------
    ndarray
        Analytic gate state at the requested elapsed times.
    """
    a, b, tau, t = np.broadcast_arrays(*[np.asarray(x, dtype=float)
                                       for x in [initial, equilibrium, tau_ms, elapsed_ms]])
    if (not all(np.isfinite(x).all() for x in [a, b, tau, t]) or a.size == 0
            or np.any((a < 0) | (a > 1) | (b < 0) | (b > 1))
            or np.any(tau <= 0) or np.any(t < 0)):
        raise ValueError('Require finite bounded fractions, positive tau and nonnegative time.')
    return b+(a-b)*np.exp(-t/tau)


def recovery_waveforms(holding_equilibrium, test_equilibrium, test_tau_ms,
                       recovery_equilibrium, recovery_tau_ms, *, dt_ms=.02):
    """Evaluate the fixed two-pulse assay for all recovery voltages and delays.

    Parameters
    ----------
    holding_equilibrium, test_equilibrium, test_tau_ms : array_like
        Two-element m/h arrays for -120 mV equilibrium and the 0 mV test pulse.
    recovery_equilibrium, recovery_tau_ms : array_like
        Arrays shaped (voltages, 2) for the recovery holding voltages.
    dt_ms : float, optional
        Peak observation grid, either the source 0.02 ms or refinement 0.01 ms.

    Returns
    -------
    dict
        All original assay delays, second-pulse times, states, inward-current
        proxies and peaks. Current proxies use a fixed 141 mV driving force.
    """
    hold, test, tau = [np.asarray(x, dtype=float) for x in
                       [holding_equilibrium, test_equilibrium, test_tau_ms]]
    rec, rtau = np.asarray(recovery_equilibrium, dtype=float), np.asarray(recovery_tau_ms, dtype=float)
    if (hold.shape != (2,) or test.shape != (2,) or tau.shape != (2,)
            or rec.ndim != 2 or rec.shape[1] != 2 or rec.shape != rtau.shape
            or rec.shape[0] == 0 or dt_ms not in [.01, .02]):
        raise ValueError('Invalid assay arrays or source/refinement observation grid.')
    delays = np.arange(1., 13.)**2
    time = np.arange(1, round(2/dt_ms))*dt_ms
    # The initial two-ms hold is already at equilibrium.
    conditioned = relax(hold, test, tau, 10.)
    recovered = relax(conditioned, rec[:, None, :], rtau[:, None, :], delays[None, :, None])
    states = relax(recovered[:, :, None, :], test, tau, time[None, None, :, None])
    current = 141*states[..., 0]**3*states[..., 1]
    return dict(delays_ms=delays, second_pulse_time_ms=time, conditioned_state=conditioned,
                recovered_states=recovered, second_pulse_states=states,
                inward_proxy=current, peaks=current.max(axis=-1))


def fit_recovery_constant(delays_ms, peaks):
    """Apply the source free-amplitude/offset exponential observation fit.

    Parameters
    ----------
    delays_ms, peaks : array_like
        Complete finite delays and positive second-pulse inward peaks.

    Returns
    -------
    dict
        Fitted recovery constant and R-squared; failed fits remain unavailable.
    """
    x, y = np.asarray(delays_ms, dtype=float), np.asarray(peaks, dtype=float)
    if (x.ndim != 1 or x.size < 4 or x.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0) or np.any(x < 0) or np.any(y <= 0)):
        raise ValueError('Require complete increasing delays and positive finite peaks.')
    x, y = x-x[0], y-y[-1]
    variance = float(np.sum((y-y.mean())**2))
    if variance == 0:
        return dict(available=False, tau_ms=None, reason='No recovery variation.')
    def exponential(t, amplitude, tau, offset):
        return amplitude*np.exp(-t/tau)+offset
    try:
        params, _ = curve_fit(exponential, x, y, p0=(y[0], 15., y[0]*.1), maxfev=800)
    except RuntimeError as exc:
        return dict(available=False, tau_ms=None, reason=str(exc))
    r2 = float(1-np.sum((y-exponential(x, *params))**2)/variance)
    valid = bool(np.isfinite(params).all() and params[1] > 0 and r2 > .95)
    return dict(available=valid, tau_ms=float(params[1]) if valid else None,
                fitted_parameters=params.tolist(), r_squared=r2)
