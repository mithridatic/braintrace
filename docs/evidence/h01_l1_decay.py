"""Descriptive post-transient decay fitting; not a voltage-gated channel model."""

import numpy as np
from scipy.optimize import least_squares


def fit_decay(time_ms, current_pa, *, window=(1110., 2090.)):
    """Fit two nonnegative exponential amplitudes on a complete observed window.

    Parameters
    ----------
    time_ms, current_pa : array-like
        Original time and conditional-current samples, with no interpolation.
    window : tuple of float, optional
        Half-open fit window in ms; its end must be covered by an observation.

    Returns
    -------
    tuple of dict
        Original fit samples, predictions and residuals; descriptive parameters
        and optimization diagnostics. No physiological gate is promoted.
    """
    t, current = np.asarray(time_ms, float), np.asarray(current_pa, float)
    if (t.ndim != 1 or len(t) < 3 or current.shape != t.shape
            or not np.isfinite(t).all() or not np.isfinite(current).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Finite aligned observations with increasing time are required.')
    if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError('Invalid fit window.')
    if t[0] > window[0] or t[-1] < window[1]:
        raise ValueError('The fit window is not fully observed.')
    mask = (t >= window[0]) & (t < window[1])
    if mask.sum() < 24:
        raise ValueError('Insufficient observations for descriptive fitting.')
    x, y = t[mask] - window[0], current[mask]
    lower = np.array([0., 0., -1000., 1., 1.])
    upper = np.array([10000., 10000., 1000., 10000., 10000.])

    def prediction(p):
        return p[2] + p[0] * np.exp(-x / p[3]) + p[1] * np.exp(-x / p[4])

    result = least_squares(lambda p: prediction(p) - y, [300., 300., 0., 30., 500.],
                           bounds=(lower, upper), x_scale='jac', max_nfev=500,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    p = result.x
    order = np.argsort(p[3:])
    fitted = prediction(p)
    parameters = dict(amplitudes_pa=p[:2][order].tolist(), time_constants_ms=p[3:][order].tolist(),
                      offset_pa=float(p[2]), origin_ms=window[0], window_ms=list(window),
                      optimizer_success=bool(result.success), evaluations=int(result.nfev),
                      optimizer_message=str(result.message), active_bounds=result.active_mask.tolist(),
                      active_bounds_parameter_order=['A1', 'A2', 'C', 'tau1', 'tau2'],
                      boundary_limited=bool(np.any(result.active_mask)),
                      minimum_relative_bound_distance=float(np.min(np.minimum(p - lower, upper - p)
                                                                 / (upper - lower))),
                      residual_rms_pa=float(np.sqrt(np.mean((y - fitted) ** 2))),
                      physiological_qualification=False,
                      interpretation='conditional descriptive decay; no molecular identity or rate law')
    return dict(time_ms=t[mask], current_pa=y, fitted_current_pa=fitted,
                residual_pa=y - fitted), parameters
