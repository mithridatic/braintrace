"""Conditional current differences at explicitly matched human test commands."""

import numpy as np
from scipy.optimize import least_squares


def match_commands(total, conditioned):
    """Match unique partners by their measured constant test-window command.

    Parameters
    ----------
    total, conditioned : dict
        Sweep identifiers mapped to test command in mV.

    Returns
    -------
    list of tuple
        One-to-one pairs, ordered by total sweep identifier.
    """
    if not total or not all(np.isfinite(v) for v in [*total.values(), *conditioned.values()]):
        raise ValueError('Finite nonempty command maps are required.')
    pairs = []
    for sweep, voltage in sorted(total.items()):
        candidates = [s for s, v in conditioned.items() if abs(v-voltage) <= 1e-7]
        if len(candidates) != 1:
            raise ValueError('Test command has missing or ambiguous partners.')
        pairs.append((sweep, candidates[0]))
    if len({b for _, b in pairs}) != len(conditioned) or len(pairs) != len(conditioned):
        raise ValueError('Commands do not form a complete one-to-one pairing.')
    return pairs


def paired_conditioning(total, conditioned):
    """Retain pulse differences, baseline shifts and original time coordinates.

    Parameters
    ----------
    total, conditioned : dict
        Aligned time_ms, total_current_pa and command_voltage_mv source arrays.
        The test occupies 1100-2100 ms; conditioning occupies 1000-1100 ms.

    Returns
    -------
    tuple of dict
        Direct paired samples and protocol metadata. No ionic isolation or
        drift correction is asserted.
    """
    sources = [tuple(np.asarray(x[k], dtype=float) for k in
                     ('time_ms', 'total_current_pa', 'command_voltage_mv'))
               for x in (total, conditioned)]
    for t, current, command in sources:
        if (t.ndim != 1 or len(t) < 3 or current.shape != t.shape or command.shape != t.shape
                or not all(np.isfinite(a).all() for a in (t, current, command))
                or np.any(np.diff(t) <= 0)
                or not np.allclose(np.diff(t), t[1]-t[0], rtol=1e-8, atol=1e-9)):
            raise ValueError('Finite aligned samples on a uniform clock are required.')
    t, ia, va = sources[0]
    tb, ib, vb = sources[1]
    if t.shape != tb.shape or not np.allclose(t, tb, rtol=0, atol=1e-8):
        raise ValueError('Original source clocks differ.')
    edges = []
    for value in (900., 990., 1000., 1100., 2100., 2200.):
        found = np.flatnonzero(np.isclose(t, value, rtol=0, atol=1e-8))
        if len(found) != 1:
            raise ValueError('Complete on-grid baseline, conditioning, test and return required.')
        edges.append(int(found[0]))
    a, base_end, cond, start, stop, returned = edges
    holding, test, conditioning = va[a], va[start], vb[cond]
    windows = [(va[a:start], holding), (vb[a:cond], holding),
               (va[start:stop], test), (vb[start:stop], test),
               (vb[cond:start], conditioning), (va[stop:returned], holding),
               (vb[stop:returned], holding)]
    if (conditioning <= holding or test <= holding
            or any(not np.allclose(v, level, rtol=0, atol=1e-7) for v, level in windows)):
        raise ValueError('Incompatible holding, conditioning, test or return commands.')
    arrays = dict(phase_ms=t[start:stop]-1100., total_time_ms=t[start:stop],
                  conditioned_time_ms=tb[start:stop], total_current_pa=ia[start:stop],
                  conditioned_current_pa=ib[start:stop], difference_pa=ia[start:stop]-ib[start:stop],
                  baseline_time_ms=t[a:base_end], total_baseline_pa=ia[a:base_end],
                  conditioned_baseline_pa=ib[a:base_end])
    meta = dict(test_command_mv=float(test), holding_command_mv=float(holding),
                conditioning_command_mv=float(conditioning), conditioning_window_ms=[1000.,1100.],
                test_window_ms=[1100.,2100.], baseline_window_ms=[900.,990.],
                baseline_difference_pa=float((ia[a:base_end]-ib[a:base_end]).mean()),
                drift_correction_applied=False, ionic_isolation_qualified=False)
    return arrays, meta


def fit_difference(phase_ms, difference_pa):
    """Fit two nonnegative exponentials without a free steady offset.

    Parameters
    ----------
    phase_ms, difference_pa : array-like
        Original finite phase and difference samples covering 10-980 ms.

    Returns
    -------
    tuple of dict
        Original samples, fitted predictions, residuals and conditional parameters.
        Parameters are A1, A2 in pA and d1, d2 in ms; amplitudes refer to phase 10.
    """
    phase, y = np.asarray(phase_ms, dtype=float), np.asarray(difference_pa, dtype=float)
    if (phase.ndim != 1 or len(phase) < 24 or y.shape != phase.shape
            or not np.isfinite(phase).all() or not np.isfinite(y).all()
            or np.any(np.diff(phase) <= 0) or phase[0] > 10 or phase[-1] < 980):
        raise ValueError('Invalid or incomplete difference samples.')
    mask = (phase >= 10) & (phase < 980)
    x, y = phase[mask], y[mask]
    if len(x) < 24:
        raise ValueError('Insufficient fit-window samples.')
    def predict(p):
        return p[:2] @ np.exp(-(x[None,:]-10)/p[2:,None])
    result = least_squares(lambda p: predict(p)-y, [100.,300.,30.,500.],
                           bounds=([0.,0.,1.,1.], [10000.]*4), x_scale='jac',
                           max_nfev=500, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    predicted = predict(result.x)
    residual = y-predicted
    arrays = dict(phase_ms=x, difference_pa=y, predicted_pa=predicted, residual_pa=residual)
    meta = dict(parameter_order=['A1','A2','d1','d2'], parameters=result.x.tolist(),
                optimizer_success=bool(result.success), evaluations=int(result.nfev),
                optimizer_message=str(result.message), active_bounds=result.active_mask.tolist(),
                boundary_limited=bool(np.any(result.active_mask)),
                residual_rms_pa=float(np.sqrt(np.mean(residual**2))),
                phase_window_ms=[10.,980.], physiological_qualification=False)
    return arrays, meta
