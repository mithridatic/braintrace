"""Closed-form electrical response fitted only to a human test-pulse record."""

import numpy as np
from scipy.optimize import least_squares


def pulse_response(time_ms, resistance_mohm, tau_ms, series_mohm,
                   *, current_pa=-10., duration_ms=200.):
    """Evaluate a parallel RC pulse response with an instantaneous series term.

    Parameters
    ----------
    time_ms : array_like
        Strictly increasing observation times relative to command onset.
    resistance_mohm, tau_ms, series_mohm : float
        Positive effective resistance/time constant and nonnegative series term.
    current_pa, duration_ms : float, optional
        Rectangular applied command and its positive duration.

    Returns
    -------
    ndarray
        Predicted voltage changes in mV, without a baseline voltage.
    """
    t = np.asarray(time_ms, dtype=float)
    p = np.asarray([resistance_mohm, tau_ms, series_mohm, current_pa, duration_ms], dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Require a finite, strictly increasing observation clock.')
    if not np.isfinite(p).all() or resistance_mohm <= 0 or tau_ms <= 0 or series_mohm < 0 or duration_ms <= 0:
        raise ValueError('Invalid electrical response parameters.')
    active_time = np.clip(t, 0, duration_ms)
    after_time = np.maximum(t - duration_ms, 0)
    membrane = resistance_mohm * (-np.expm1(-active_time / tau_ms)) * np.exp(-after_time / tau_ms)
    instantaneous = series_mohm * ((t >= 0) & (t < duration_ms))
    return current_pa * (membrane + instantaneous) / 1000


def fit_response(time_ms, centered_mv):
    """Fit one shared circuit to calibration responses under frozen bounds.

    Parameters
    ----------
    time_ms : array_like
        Relative source clock, including pulse and return.
    centered_mv : array_like
        Finite calibration voltages, shaped (sweeps, samples), after subtraction
        of baselines estimated independently from each pre-pulse window.

    Returns
    -------
    dict
        Fitted parameters, convergence information and every optimizer call.
    """
    t = np.asarray(time_ms, dtype=float)
    y = np.asarray(centered_mv, dtype=float)
    pulse_response(t, 100., 10., 1.)
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != t.size or not np.isfinite(y).all():
        raise ValueError('Require complete finite calibration sweeps on one clock.')
    history = []

    def residual(parameters):
        r, tau = np.exp(parameters[:2])
        series = parameters[2]
        predicted = pulse_response(t, r, tau, series)
        errors = predicted[None, :] - y
        history.append(dict(resistance_mohm=float(r), tau_ms=float(tau),
                            series_mohm=float(series), rmse_mv=float(np.sqrt(np.mean(errors**2)))))
        return errors.ravel()

    fit = least_squares(residual, [np.log(100.), np.log(10.), 1.],
                        bounds=([np.log(1.), np.log(.05), 0.],
                                [np.log(1000.), np.log(200.), 100.]),
                        max_nfev=60)
    return dict(resistance_mohm=float(np.exp(fit.x[0])),
                tau_ms=float(np.exp(fit.x[1])), series_mohm=float(fit.x[2]),
                success=bool(fit.success), message=fit.message,
                nfev=int(fit.nfev), active_mask=fit.active_mask.tolist(),
                evaluations=history)
