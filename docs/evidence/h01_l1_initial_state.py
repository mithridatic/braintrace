"""Observe bounded initial availability with frozen human recovery parameters."""

import numpy as np
from scipy.optimize import lsq_linear

from h01_l1_recovery_state import LOWER, UPPER, ORDER


def _parameters(parameters):
    p = np.asarray(parameters, dtype=float)
    if (p.shape != (15,) or not np.isfinite(p).all()
            or np.any(p < LOWER) or np.any(p > UPPER)):
        raise ValueError('Invalid frozen parameters.')
    return p


def _clock(values, uniform=False):
    t = np.asarray(values, dtype=float)
    if (t.ndim != 1 or not len(t) or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid phase clock.')
    if uniform and (len(t) < 2 or not np.allclose(np.diff(t), t[1]-t[0], rtol=0, atol=1e-8)):
        raise ValueError('Clock must be uniform.')
    return t


def conditioned_currents(parameters, gaps_ms, pulse_phase_ms, post_phase_ms,
                         initial_availability, baseline_availability):
    """Carry observed states through paired 300 ms pulses and holding return.

    Parameters
    ----------
    parameters : array-like
        Fifteen frozen parameters in recovery-state order.
    gaps_ms, pulse_phase_ms, post_phase_ms : array-like
        Nonnegative gap durations and original phase coordinates in ms.
    initial_availability, baseline_availability : array-like
        Availability at first onset and mean availability in the recorded
        baseline, respectively. Each has shape (number of gaps, 2).

    Returns
    -------
    tuple of numpy.ndarray
        First, second and return currents in pA relative to the original
        baseline. These conditional currents are not isolated ionic currents.
    """
    p = _parameters(parameters)
    gaps = np.asarray(gaps_ms, dtype=float)
    pulse, post = _clock(pulse_phase_ms), _clock(post_phase_ms)
    states = [np.asarray(x, dtype=float) for x in (initial_availability, baseline_availability)]
    if (gaps.ndim != 1 or not len(gaps) or not np.isfinite(gaps).all()
            or np.any(gaps < 0) or pulse[0] < 0 or post[0] < 0
            or any(x.shape != (len(gaps), 2) or not np.isfinite(x).all()
                   or np.any(x < 0) or np.any(x > 1) for x in states)):
        raise ValueError('Invalid gaps or availability states.')
    h0, hb = states
    a, d, r, f = p[:2], p[2:4], p[4:6], p[6:8]
    c, rho, eta, s, off = p[8], p[9], p[10:12], p[12:14], p[14]
    end_first = h0 * np.exp(-300/d)
    start_second = 1-(1-end_first)*np.exp(-gaps[:, None]/r)
    end_second = start_second*np.exp(-300/d)
    offset = c-rho*np.sum(a*eta*(hb-f), axis=1)
    envelope = np.exp(-pulse[None, :]/d[:, None])
    first = offset[:, None]+(h0*a)@envelope
    second = offset[:, None]+(start_second*a)@envelope
    h = f[None, :, None]+(end_second-f)[:, :, None]*np.exp(-post[None, None, :]/s[None, :, None])
    factor = rho*(eta[:, None]+(1-eta[:, None])*np.exp(-post[None, :]/off))
    returned = np.einsum('j,ijt,jt->it', a, h, factor)-rho*np.sum(a*eta*hb, axis=1)[:, None]
    return first, second, returned


def observe_state(parameters, pulse_phase_ms, first_current_pa,
                  holding_duration_ms, baseline_phase_ms):
    """Estimate two bounded holding-start states from first-pulse 10-60 ms.

    Parameters
    ----------
    parameters : array-like
        Frozen recovery parameters. No global parameters are optimized.
    pulse_phase_ms : array-like
        Uniform original first-pulse clock, covering the estimation window.
    first_current_pa : array-like
        Original baseline-centered currents, one row per sweep.
    holding_duration_ms : array-like
        Verified uninterrupted holding durations before first onset per sweep.
    baseline_phase_ms : array-like
        Uniform original baseline clock spanning -100 to -10 ms, end excluded.

    Returns
    -------
    tuple of dict
        State estimates, original estimation samples, predictions and numerical
        diagnostics; metadata explicitly identifies calibration-only inference.
    """
    p = _parameters(parameters)
    t, b = _clock(pulse_phase_ms, True), _clock(baseline_phase_ms, True)
    current = np.asarray(first_current_pa, dtype=float)
    duration = np.asarray(holding_duration_ms, dtype=float)
    if (current.ndim != 2 or current.shape[1] != len(t) or not len(current)
            or not np.isfinite(current).all() or duration.shape != (len(current),)
            or not np.isfinite(duration).all() or np.any(duration+b[0] < 0)):
        raise ValueError('Invalid aligned currents or observed holding interval.')
    if (t[0] > 10 or t[-1]+t[1]-t[0] < 60-1e-8
            or not np.isclose(b[0], -100, rtol=0, atol=1e-8)
            or not np.isclose(b[-1]+b[1]-b[0], -10, rtol=0, atol=1e-8)):
        raise ValueError('Incomplete estimation or baseline window.')
    mask = (t >= 10) & (t < 60)
    if mask.sum() < 24:
        raise ValueError('Insufficient original estimation samples.')
    a, d, f, rho, eta, s = p[:2], p[2:4], p[6:8], p[9], p[10:12], p[12:14]
    memory = np.exp(-duration[:, None]/s)
    baseline_memory = np.exp(-(duration[:, None, None]+b[None, :, None])/s).mean(axis=1)
    design = a*(memory[:, None, :]*np.exp(-t[mask, None]/d)-rho*eta*baseline_memory[:, None, :])
    singular = np.linalg.svd(design, compute_uv=False)
    if np.any(singular[:, 1] <= singular[:, 0]*1e-12):
        raise ValueError('Two-state observation matrix is rank deficient.')
    shared = p[8]+np.exp(-t[mask, None]/d)@(a*f)
    intercept = shared-np.einsum('itj,j->it', design, f)
    observed = current[:, mask]
    # Independent bounded data fits, not a repeated membrane-model rollout.
    results = [lsq_linear(matrix, y-base, bounds=(0, 1), tol=1e-12, max_iter=100)
               for matrix, y, base in zip(design, observed, intercept)]
    if not all(result.success for result in results):
        raise ValueError('Bounded state estimation did not converge.')
    z = np.stack([result.x for result in results])
    predicted = intercept+np.einsum('itj,ij->it', design, z)
    arrays = dict(holding_start_availability=z, initial_availability=f+(z-f)*memory,
                  baseline_availability=f+(z-f)*baseline_memory,
                  holding_duration_ms=duration.copy(), baseline_phase_ms=b.copy(),
                  holding_memory=memory, baseline_memory=baseline_memory,
                  estimation_mask=mask, estimation_phase_ms=t[mask],
                  estimation_current_pa=observed, estimation_predicted_pa=predicted,
                  estimation_residual_pa=observed-predicted,
                  design_singular_values=singular,
                  active_bounds=np.stack([result.active_mask for result in results]),
                  optimizer_status=np.array([result.status for result in results]))
    metadata = dict(global_parameters_changed=False, independent_validation=False,
                    physiological_qualification=False, parameter_order=ORDER,
                    parameters=p.tolist(), estimation_window_ms=[10, 60],
                    assumption='Frozen holding kinetics and bounded state at start of verified holding interval.',
                    numerical_rank_relative_tolerance=1e-12)
    return arrays, metadata
