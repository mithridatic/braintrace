"""Exact four-pole Bessel observation responses for human clamp controls."""

from functools import lru_cache

import numpy as np
from scipy.signal import bessel


ORDER = ['gL_ns', 'C0_pf', 'C1_pf', 'tau_ms', 'latency_ms']
INITIAL = np.array([.3, 2., 2., 1., .12])
LOWER = np.array([0., 0., 0., .04, 0.])
UPPER = np.array([3., 100., 100., 500., .3])


@lru_cache(maxsize=8)
def pole_pairs(cutoff_hz=2000.):
    """Return conjugate-pair constants for the documented magnitude-cutoff filter.

    Parameters
    ----------
    cutoff_hz : float, optional
        Positive -3 dB cutoff in Hz. The pinned human recording reports 2000.

    Returns
    -------
    tuple of numpy.ndarray
        Real pole part, positive imaginary pole part, and real/imaginary residue
        parts, all expressed using milliseconds as the time coordinate.
    """
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError('Invalid filter cutoff.')
    _, poles, gain = bessel(4, 2 * np.pi * cutoff_hz / 1000,
                            analog=True, output='zpk', norm='mag')
    differences = poles[:, None] - poles[None, :]
    np.fill_diagonal(differences, 1)
    residues = gain / np.prod(differences, axis=1)
    selected = poles.imag > 0
    return poles[selected].real, poles[selected].imag, residues[selected].real, residues[selected].imag


def bessel_impulse(phase_ms, cutoff_hz=2000.):
    """Evaluate the causal Bessel impulse response, retaining complex derivatives.

    Parameters
    ----------
    phase_ms : array-like
        Query phases in ms. Negative phases precede the impulse and return zero.
    cutoff_hz : float, optional
        Fixed magnitude cutoff in Hz.

    Returns
    -------
    numpy.ndarray
        Impulse response in inverse ms, with the same shape as phase_ms.
    """
    phase = np.asarray(phase_ms)
    if not np.isfinite(phase).all():
        raise ValueError('Nonfinite response phase.')
    alpha, beta, real, imag = pole_pairs(cutoff_hz)
    t = np.where(phase.real > 0, phase, 0)[..., None]
    result = np.sum(2 * np.exp(alpha * t)
                    * (real * np.cos(beta * t) - imag * np.sin(beta * t)), axis=-1)
    return np.where(phase.real > 0, result, 0)


def bessel_exponential(rate_per_ms, phase_ms, cutoff_hz=2000.):
    """Filter a causal exponential input with the fixed analog Bessel response.

    Parameters
    ----------
    rate_per_ms, phase_ms : array-like
        Broadcastable nonnegative decay rates and phases in ms. Negative phases
        return zero. Complex perturbations support derivative calculation.
    cutoff_hz : float, optional
        Fixed -3 dB cutoff in Hz.

    Returns
    -------
    numpy.ndarray
        Filtered unit-amplitude exponential; rate zero gives the step response.
    """
    rate, phase = np.broadcast_arrays(np.asarray(rate_per_ms), np.asarray(phase_ms))
    if not np.isfinite(rate).all() or np.any(rate.real < 0) or not np.isfinite(phase).all():
        raise ValueError('Invalid exponential input or phase.')
    alpha, beta, real, imag = pole_pairs(cutoff_hz)
    t = np.where(phase.real > 0, phase, 0)[..., None]
    d = rate[..., None] + alpha
    denominator = d**2 + beta**2
    a = (real * d + imag * beta) / denominator
    b = (imag * d - real * beta) / denominator
    ea = np.exp(alpha * t)
    result = np.sum(2 * (a * (ea * np.cos(beta * t) - np.exp(-rate[..., None] * t))
                         - b * ea * np.sin(beta * t)), axis=-1)
    return np.where(phase.real > 0, result, 0)


def predict_control(parameters, edge_times_ms, edge_steps_mv, protocol_ids, time_ms,
                    *, cutoff_hz=2000.):
    """Predict small-control currents from original voltage-command edges.

    Parameters
    ----------
    parameters : array-like
        gL (nS), unresolved fast charge C0 (pF), slow charge C1 (pF), its tau
        (ms), and observation latency (ms). These are conditional estimates.
    edge_times_ms, edge_steps_mv : array-like
        Two-dimensional matching arrays of command-edge times and signed voltage
        changes. Zero-amplitude padding is permitted, in nondecreasing time order.
    protocol_ids, time_ms : array-like
        Protocol row and original time for each query. Time is in ms.
    cutoff_hz : float, optional
        Recorded primary-output Bessel cutoff, kept fixed during fitting.

    Returns
    -------
    numpy.ndarray
        Predicted baseline-relative amplifier current in pA. This does not
        isolate ionic current or establish the actual membrane voltage.
    """
    p = np.asarray(parameters)
    edges, steps = np.asarray(edge_times_ms, float), np.asarray(edge_steps_mv, float)
    ids, times = np.asarray(protocol_ids), np.asarray(time_ms, float)
    if (p.shape != (5,) or not np.isfinite(p).all()
            or np.any(p.real < LOWER) or np.any(p.real > UPPER)):
        raise ValueError('Invalid observation parameters.')
    if (edges.ndim != 2 or not edges.size or steps.shape != edges.shape
            or not np.isfinite(edges).all() or not np.isfinite(steps).all()
            or np.any(edges < 0) or np.any(np.diff(edges, axis=1) < 0)):
        raise ValueError('Invalid original command edges.')
    if (ids.ndim != 1 or not ids.size or times.shape != ids.shape
            or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
            or np.any(ids >= len(edges)) or not np.isfinite(times).all() or np.any(times < 0)):
        raise ValueError('Invalid observation queries.')
    phase = times[:, None] - edges[ids] - p[4]
    response = (p[0] * bessel_exponential(0., phase, cutoff_hz)
                + p[1] * bessel_impulse(phase, cutoff_hz)
                + p[2] / p[3] * bessel_exponential(1 / p[3], phase, cutoff_hz))
    result = np.sum(steps[ids] * response, axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite observation prediction.')
    return result
