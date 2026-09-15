"""Calibrate only a connection's strength against a frozen current template."""

import numpy as np


def fit_connection_strength(template_mv_per_pa, observed_mv, *, minimum_pa=.01, maximum_pa=2000.):
    """Find the bounded least-squares strength without changing the template.

    Parameters
    ----------
    template_mv_per_pa, observed_mv : array_like
        Matching finite arrays of the frozen unit-current prediction and human
        calibration voltages. All supplied observations receive equal weight.
    minimum_pa, maximum_pa : float, optional
        Positive finite strength bounds fixed before validation.

    Returns
    -------
    dict
        Bounded and unconstrained peak currents, and whether the optimum is
        strictly inside the bounds. A bound-active result is not a passing fit.
    """
    x, y = np.asarray(template_mv_per_pa, dtype=float), np.asarray(observed_mv, dtype=float)
    if x.shape != y.shape or x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Require matching complete finite template and observations.')
    if not np.isfinite([minimum_pa, maximum_pa]).all() or not 0 < minimum_pa < maximum_pa:
        raise ValueError('Invalid prespecified positive strength bounds.')
    denominator = float(np.vdot(x, x))
    numerator = float(np.vdot(x, y))
    if denominator <= 0 or not np.isfinite([denominator, numerator]).all():
        raise ValueError('The template has no finite identifiable strength.')
    raw = numerator/denominator
    return dict(peak_pa=float(np.clip(raw, minimum_pa, maximum_pa)),
                unconstrained_peak_pa=raw, interior=bool(minimum_pa < raw < maximum_pa))
