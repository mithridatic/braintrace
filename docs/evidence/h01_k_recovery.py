"""Estimate effective recovery responses from identified human measurements."""

import re

import numpy as np
from scipy.optimize import least_squares


def extract_curves(records):
    """Select complete human recovery observations without physiological filtering.

    Parameters
    ----------
    records : iterable of dict
        Original selected human Figure 4 workbook rows.

    Returns
    -------
    tuple of list and list
        Per-record curves with identities and explicit exclusions.
    """
    curves, excluded, seen = [], [], set()
    for r in records:
        name = r.get('Filename', '')
        if not isinstance(name, str) or not re.fullmatch(r'H\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.nwb', name) or name in seen or r.get('Species') != 'Human':
            raise ValueError('Missing, duplicate or nonhuman identity.')
        seen.add(name)
        duplicates = []
        for key, value in r.items():
            match = re.fullmatch(r'amp1torec_\s*(8|9|10)\.1', key)
            if match:
                if value != r.get('amp1torec_'+match[1]):
                    raise ValueError('Duplicate source columns disagree.')
                duplicates.append(key)
        if r.get('rec_protocol_name') is None:
            excluded.append(dict(filename=name, reason='missing recovery protocol'))
            continue
        if r['rec_protocol_name'] != 'K_IKrecv2_1_h0_DA_0':
            raise ValueError('Unregistered recovery protocol.')
        t, y, missing = [], [], []
        for i in range(1, 11):
            time, ratio = r.get(f'recdelay_{i}'), r.get(f'amp1torec_{i}')
            if time is None and ratio is None:
                missing.append(i)
                continue
            if time is None or ratio is None or not np.isfinite([time, ratio]).all():
                raise ValueError('Incomplete or nonfinite observation pair.')
            t.append(time)
            y.append(ratio)
        if len(t) < 6 or np.any(np.diff(t) <= 0) or min(t) < 0:
            raise ValueError('Insufficient or invalid recovery delays.')
        curves.append(dict(filename=name, group='.'.join(name.split('.')[:3]),
                           times_ms=t, ratios=y, missing_indices=missing,
                           duplicate_columns_verified=duplicates))
    if not curves:
        raise ValueError('No eligible human recovery curves.')
    return curves, excluded


def recovery(times_ms, parameters, family):
    """Evaluate a constrained current-amplitude recovery curve.

    Parameters
    ----------
    times_ms : array_like
        Nonnegative recovery intervals in ms.
    parameters : array_like
        r0, recovered fraction, then log(tau) for single; r0, recovered
        fraction, fast fraction, log(tfast), log(tslow) for double.
    family : {'single', 'double'}
        Exponential observation family; these are not channel-gate rates.

    Returns
    -------
    ndarray
        Predicted dimensionless current amplitude relative to initial amplitude.
    """
    t = np.asarray(times_ms)
    p = parameters
    if family == 'single':
        recovered = -np.expm1(-t / np.exp(p[2]))
    elif family == 'double':
        recovered = p[2] * -np.expm1(-t / np.exp(p[3])) + (1-p[2]) * -np.expm1(-t / np.exp(p[4]))
    else:
        raise ValueError('Unknown recovery family.')
    return p[0] + (1-p[0]) * p[1] * recovered


def fit_recovery(curves, family):
    """Fit equally weighted recording curves with fixed bounds and three starts.

    Parameters
    ----------
    curves : list of dict
        Each record supplies increasing times_ms and paired finite ratios.
    family : {'single', 'double'}
        Preregistered exponential family.

    Returns
    -------
    dict
        Best converged parameters, all starts, cost and identifiability diagnostics.
    """
    if family not in ('single', 'double') or not curves:
        raise ValueError('Unknown family or empty fitting input.')
    times, ratios, weights = [], [], []
    for c in curves:
        t, y = np.asarray(c['times_ms']), np.asarray(c['ratios'])
        if t.ndim != 1 or t.size < 2 or t.shape != y.shape or not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t) <= 0) or min(t) < 0:
            raise ValueError('Invalid fit observations.')
        times.append(t)
        ratios.append(y)
        weights.append(np.full(t.size, 1/np.sqrt(len(curves)*t.size)))
    t, y, w = np.concatenate(times), np.concatenate(ratios), np.concatenate(weights)
    if family == 'single':
        lower, upper = [0, 0, np.log(.1)], [1, 1, np.log(100000)]
        starts = [[.15, .7, np.log(tau)] for tau in (200, 1000, 4000)]
    else:
        lower, upper = [0, 0, 0, np.log(.1), np.log(1000)], [1, 1, 1, np.log(1000), np.log(100000)]
        starts = [[.15, .7, .5, np.log(tf), np.log(ts)] for tf, ts in ((50, 1500), (200, 4000), (700, 15000))]
    outcomes, converged = [], []
    for initial in starts:
        result = least_squares(lambda p: (recovery(t, p, family)-y)*w, initial,
                               bounds=(lower, upper), max_nfev=2000,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        outcomes.append(dict(initial=initial, parameters=result.x.tolist(),
                             success=bool(result.success), nfev=result.nfev,
                             cost=float(result.cost), message=result.message))
        if result.success:
            converged.append(result)
    if not converged:
        raise RuntimeError('No converged recovery estimate; retain failure.')
    best = min(converged, key=lambda result: result.cost)
    condition = float(np.linalg.cond(best.jac))
    normalized = (best.x-np.asarray(lower))/(np.asarray(upper)-np.asarray(lower))
    return dict(family=family, parameters=best.x.tolist(), cost=float(best.cost),
                jacobian_condition=condition if np.isfinite(condition) else None,
                bound_distance_fraction=np.minimum(normalized, 1-normalized).tolist(),
                starts=outcomes)
