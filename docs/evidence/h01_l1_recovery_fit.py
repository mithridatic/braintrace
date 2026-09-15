"""Conditional availability fitting across human paired-pulse recovery protocols."""

import numpy as np
from scipy.optimize import least_squares


def recovery_prediction(parameters, gaps_ms, phase_ms):
    """Evaluate two availability populations using closed-form state transitions.

    Parameters
    ----------
    parameters : array-like
        A1,A2,d1,d2,r1,r2,f1,f2: pA, ms, ms, and initial fractions.
    gaps_ms, phase_ms : array-like
        Recovery intervals and pulse phases in ms. Phase origin is 10 ms.

    Returns
    -------
    numpy.ndarray
        Second-minus-first current, one row per gap and column per phase.
    """
    p = np.asarray(parameters)
    a, decay, recovery, fraction = p[:2], p[2:4], p[4:6], p[6:]
    gaps = np.asarray(gaps_ms)[:, None]
    phase = np.asarray(phase_ms)[None, :] - 10.
    available = 1 - (1 - fraction * np.exp(-300. / decay)) * np.exp(-gaps / recovery) - fraction
    return (available * a) @ np.exp(-phase / decay[:, None])


def fit_recovery(gaps_ms, phase_ms, difference_pa):
    """Fit all recovery conditions jointly without per-sweep nuisance offsets.

    Parameters
    ----------
    gaps_ms : array-like
        At least four distinct, increasing positive recovery intervals.
    phase_ms : array-like
        Original increasing phase samples covering 10-280 ms.
    difference_pa : array-like
        Finite paired-current matrix, shape (number of gaps, number of phases).

    Returns
    -------
    tuple of dict
        Original fit data, predictions, residuals and unqualified parameters.
        Molecular identities, voltage-dependent rates and physiology are not
        established by this conditional two-voltage fit.
    """
    gaps, phase, difference = (np.asarray(x, dtype=float)
                              for x in (gaps_ms, phase_ms, difference_pa))
    if (gaps.ndim != 1 or phase.ndim != 1 or len(gaps) < 4 or len(phase) < 24
            or difference.shape != (len(gaps), len(phase))
            or not all(np.isfinite(x).all() for x in (gaps, phase, difference))
            or np.any(gaps <= 0) or np.any(np.diff(gaps) <= 0)
            or np.any(np.diff(phase) <= 0)):
        raise ValueError('Invalid paired-current matrix or original coordinates.')
    if phase[0] > 10 or phase[-1] < 280:
        raise ValueError('Fit phase window is not fully observed.')
    window = (phase >= 10) & (phase < 280)
    x, y = phase[window], difference[:, window]
    if len(x) < 24:
        raise ValueError('Insufficient fit-phase samples.')
    lower = [0., 0., 1., 1., 1., 1., 0., 0.]
    upper = [2000., 2000., 10000., 10000., 20000., 20000., 1., 1.]
    result = least_squares(lambda p: (recovery_prediction(p, gaps, x) - y).ravel(),
                           [200., 500., 30., 500., 50., 1000., .05, .05],
                           bounds=(lower, upper), x_scale='jac', max_nfev=500,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    predicted = recovery_prediction(result.x, gaps, x)
    residual = y - predicted
    meta = dict(parameter_order=['A1', 'A2', 'd1', 'd2', 'r1', 'r2', 'f1', 'f2'],
                parameters=result.x.tolist(), optimizer_success=bool(result.success),
                evaluations=int(result.nfev), optimizer_message=str(result.message),
                active_bounds=result.active_mask.tolist(), boundary_limited=bool(np.any(result.active_mask)),
                residual_rms_pa=float(np.sqrt(np.mean(residual ** 2))),
                per_gap_residual_rms_pa=np.sqrt(np.mean(residual ** 2, axis=1)).tolist(),
                gaps_ms=gaps.tolist(), phase_window_ms=[10., 280.],
                physiological_qualification=False,
                assumption='two conditional availability populations at recorded -90/+60 mV commands')
    return dict(gaps_ms=gaps, phase_ms=x, difference_pa=y, predicted_pa=predicted,
                residual_pa=residual), meta
