"""Conditional human voltage-clamp currents with exact command-segment solutions."""

import numpy as np


ORDER = ['g1', 'g2', 'gs', 'vm', 'km', 'tm', 'vh', 'kh', 'd1', 'd2',
         'r1', 'r2', 'E', 'gL', 'C1', 'tc1', 'C2', 'tc2', 'tf']
INITIAL = np.array([1.5, 2., 1., -20., 15., 3., -50., 10., 25., 700.,
                    80., 500., -85., .3, 2., .2, 10., 20., .08])
LOWER = np.array([0., 0., 0., -60., 2., .04, -100., 2., 1., 50.,
                  1., 10., -120., 0., 0., .04, 0., 1., .04])
UPPER = np.array([50., 50., 50., 20., 35., 100., 0., 35., 300., 20000.,
                  3000., 20000., -50., 3., 100., 10., 100., 500., 3.])


def affine_starts(decay, offset, initial):
    """Compose stable affine transitions without a repeatedly executed model loop.

    Parameters
    ----------
    decay, offset : numpy.ndarray
        Nonnegative integrated decay and additive endpoint terms, shape
        (protocol, segment, state). Decay may have a complex-step perturbation.
    initial : numpy.ndarray
        Initial state, shape (protocol, state).

    Returns
    -------
    numpy.ndarray
        State at the beginning of every segment, with the same shape as decay.
    """
    size = decay.shape[1]
    accumulated = np.concatenate([np.zeros_like(decay[:, :1]),
                                  np.cumsum(decay, axis=1)], axis=1)
    preceding = np.arange(size)[:, None] > np.arange(size)[None, :]
    distance = accumulated[:, :-1, None, :] - accumulated[:, None, 1:, :]
    propagation = np.where(preceding[None, :, :, None],
                           np.exp(-np.where(preceding[None, :, :, None], distance, 0)), 0)
    return (initial[:, None, :] * np.exp(-accumulated[:, :-1])
            + np.einsum('bijs,bjs->bis', propagation, offset))


def filtered_exponential(rate, phase, filter_tau):
    """Return the zero-initial-state first-order filter of exp(-rate * phase).

    Parameters
    ----------
    rate, phase : array-like
        Broadcastable forcing decay rates in inverse ms and nonnegative phases
        in ms. Rates may carry a complex-step perturbation.
    filter_tau : float or complex
        Positive filter time constant in ms.

    Returns
    -------
    numpy.ndarray
        Filtered forcing, including the continuous equal-rate limit.
    """
    rate, phase = np.asarray(rate), np.asarray(phase)
    denominator = 1 - rate * filter_tau
    a, b = np.exp(-rate * phase), np.exp(-phase / filter_tau)
    # Local series retains complex-step derivatives at equal rates.
    delta = (1 / filter_tau - rate) * phase
    limit = b * phase / filter_tau * (1 + delta / 2 + delta**2 / 6 + delta**3 / 24)
    numerator, denominator = np.broadcast_arrays(a - b, denominator)
    result = np.array(np.broadcast_to(limit, numerator.shape), copy=True)
    np.divide(numerator, denominator, out=result, where=np.abs(delta) >= 1e-4)
    return result


def predict(parameters, starts_ms, stops_ms, commands_mv, protocol_ids, time_ms,
            *, components=False):
    """Predict conditional measured currents for a batch of command protocols.

    Parameters
    ----------
    parameters : array-like
        Nineteen parameters in ORDER, in nS, pF, mV and ms as documented there.
        Complex perturbations are supported for derivative calculation.
    starts_ms, stops_ms, commands_mv : array-like
        Two-dimensional equally shaped segment arrays. Protocols start at zero
        and cover time contiguously. Trailing zero-duration padding is allowed.
    protocol_ids : array-like of int
        Protocol row for each observation query.
    time_ms : array-like
        Original query times in ms, within the selected protocol's coverage.
    components : bool, optional
        Also return unfiltered active, leak and recording-transient components.

    Returns
    -------
    numpy.ndarray or dict
        Filtered baseline-relative current in pA, or that current and conditional
        components. Commands are not measured patch voltages; components are not
        experimentally isolated ionic currents.
    """
    p = np.asarray(parameters)
    starts, stops, voltage = (np.asarray(x, dtype=float)
                              for x in (starts_ms, stops_ms, commands_mv))
    ids, times = np.asarray(protocol_ids), np.asarray(time_ms, dtype=float)
    if (p.shape != (19,) or not np.isfinite(p).all()
            or np.any(p.real < LOWER) or np.any(p.real > UPPER)):
        raise ValueError('Invalid candidate parameters.')
    if (starts.ndim != 2 or not starts.size or stops.shape != starts.shape
            or voltage.shape != starts.shape or not np.isfinite(starts).all()
            or not np.isfinite(stops).all() or not np.isfinite(voltage).all()
            or np.any(starts[:, 0] != 0) or np.any(stops < starts)
            or np.any(starts[:, 1:] != stops[:, :-1])
            or np.any(stops[:, -1] <= 0)):
        raise ValueError('Invalid or discontinuous command coverage.')
    duration = stops - starts
    if np.any((duration[:, :-1] == 0) & (duration[:, 1:] > 0)):
        raise ValueError('Zero-duration padding must be trailing.')
    if (ids.ndim != 1 or times.shape != ids.shape or not ids.size
            or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
            or np.any(ids >= len(starts)) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(times >= stops[ids, -1])):
        raise ValueError('Invalid observation query or missing command coverage.')

    m_inf = 1 / (1 + np.exp((p[3] - voltage) / p[4]))
    h_inf = 1 / (1 + np.exp((voltage - p[6]) / p[7]))
    equilibrium = np.stack([m_inf, h_inf, h_inf, voltage, voltage], axis=-1)
    tau = np.stack([np.broadcast_to(p[5], voltage.shape),
                    p[8] + (p[10] - p[8]) * h_inf,
                    p[9] + (p[11] - p[9]) * h_inf,
                    np.broadcast_to(p[15], voltage.shape),
                    np.broadcast_to(p[17], voltage.shape)], axis=-1)
    integrated = duration[..., None] / tau
    states = affine_starts(integrated, equilibrium * (-np.expm1(-integrated)),
                           equilibrium[:, 0])
    delta = states - equilibrium
    m_coeff = np.stack([m_inf**2, 2 * m_inf * delta[..., 0], delta[..., 0]**2], axis=-1)
    m_rate = np.broadcast_to(np.arange(3) / p[5], m_coeff.shape)
    h_base = np.stack([h_inf, h_inf, np.ones_like(h_inf)], axis=-1)
    h_delta = np.concatenate([delta[..., 1:3], np.zeros_like(delta[..., :1])], axis=-1)
    h_coeff = np.stack([h_base, h_delta], axis=-1)
    h_rate = np.stack([np.zeros_like(h_base),
                       np.concatenate([1 / tau[..., 1:3], np.zeros_like(delta[..., :1])], axis=-1)], axis=-1)
    active_coeff = ((voltage - p[12])[..., None, None, None]
                    * p[:3][None, None, :, None, None]
                    * m_coeff[..., None, :, None] * h_coeff[..., :, None, :])
    active_rate = m_rate[..., None, :, None] + h_rate[..., :, None, :]
    shape = voltage.shape + (18,)
    active_coeff, active_rate = active_coeff.reshape(shape), active_rate.reshape(shape)
    hold = voltage[:, :1]
    active_initial = ((hold - p[12]) * m_inf[:, :1]**2
                      * ((p[0] + p[1]) * h_inf[:, :1] + p[2]))
    leak = p[13] * (voltage - hold)
    transient_coeff = -delta[..., 3:5] * np.array([p[14] / p[15], p[16] / p[17]])
    coefficients = np.concatenate([active_coeff, transient_coeff,
                                    (leak - active_initial)[..., None]], axis=-1)
    rates = np.concatenate([active_rate, 1 / tau[..., 3:5],
                             np.zeros_like(voltage[..., None])], axis=-1)
    filter_offset = np.sum(coefficients * filtered_exponential(rates, duration[..., None], p[18]), axis=-1)
    filter_starts = affine_starts((duration / p[18])[..., None], filter_offset[..., None],
                                  np.zeros((len(starts), 1), dtype=p.dtype))[..., 0]
    segment = np.sum(times[:, None] >= starts[ids], axis=1) - 1
    phase = times - starts[ids, segment]
    query_coeff, query_rate = coefficients[ids, segment], rates[ids, segment]
    filtered = (filter_starts[ids, segment] * np.exp(-phase / p[18])
                + np.sum(query_coeff * filtered_exponential(query_rate, phase[:, None], p[18]), axis=-1))
    if not np.isfinite(filtered).all():
        raise FloatingPointError('Nonfinite model prediction.')
    if not components:
        return filtered
    exponentials = query_coeff * np.exp(-query_rate * phase[:, None])
    return dict(filtered_pa=filtered,
                active_relative_pa=exponentials[:, :18].sum(axis=1) - active_initial[ids, 0],
                recording_transient_pa=exponentials[:, 18:20].sum(axis=1),
                leak_pa=leak[ids, segment],
                command_mv=voltage[ids, segment])
