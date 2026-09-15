"""Human active-current candidate observed through the fixed recording response."""

import numpy as np

from docs.evidence.h01_pax6_voltage_model import (
    affine_starts, INITIAL as OLD_INITIAL, LOWER as OLD_LOWER, UPPER as OLD_UPPER,
    ORDER as OLD_ORDER,
)
from docs.evidence.h01_pax6_recording_filter import pole_pairs, predict_control

INITIAL, LOWER, UPPER = (x[:13].copy() for x in (OLD_INITIAL, OLD_LOWER, OLD_UPPER))
ORDER = OLD_ORDER[:13]


def prepare_commands(starts_ms, stops_ms, commands_mv, protocol_ids, time_ms, observation):
    """Precompute fixed command and observation geometry for repeated fitting calls.

    Parameters
    ----------
    starts_ms, stops_ms, commands_mv : array-like
        Matching (protocol, segment) arrays. Commands start at zero, are
        contiguous and may have trailing zero-duration padding.
    protocol_ids, time_ms : array-like
        Protocol row and original sample time for each observation query.
    observation : array-like
        Frozen gL, C0, C1, tau and latency from the accepted control correction.

    Returns
    -------
    dict
        Owned arrays and fixed Bessel propagators, for use with predict_active.
    """
    start, stop, voltage = (np.array(x, dtype=float, copy=True)
                            for x in (starts_ms, stops_ms, commands_mv))
    ids, times = np.array(protocol_ids, copy=True), np.array(time_ms, dtype=float, copy=True)
    obs = np.array(observation, dtype=float, copy=True)
    if (start.ndim != 2 or not start.size or stop.shape != start.shape or voltage.shape != start.shape
            or any(not np.isfinite(x).all() for x in (start, stop, voltage))
            or np.any(start[:, 0] != 0) or np.any(stop < start)
            or np.any(start[:, 1:] != stop[:, :-1]) or np.any(stop[:, -1] <= 0)):
        raise ValueError('Invalid or incomplete command coverage.')
    duration = stop - start
    if np.any((duration[:, :-1] == 0) & (duration[:, 1:] > 0)):
        raise ValueError('Command padding must be trailing.')
    if (ids.ndim != 1 or not ids.size or times.shape != ids.shape
            or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
            or np.any(ids >= len(start)) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(times >= stop[ids, -1])):
        raise ValueError('Invalid query or missing observation coverage.')
    if obs.shape != (5,) or not np.isfinite(obs).all():
        raise ValueError('Invalid frozen observation parameters.')
    edge_times = start.copy()
    edge_steps = np.concatenate([np.zeros_like(voltage[:, :1]), np.diff(voltage, axis=1)], axis=1)
    passive = predict_control(obs, edge_times, edge_steps, ids, times)
    shifted = np.maximum(times - obs[4], 0)
    segment = np.sum(shifted[:, None] >= start[ids], axis=1) - 1
    phase = shifted - start[ids, segment]
    alpha, beta, real, imag = pole_pairs()
    prior = np.arange(start.shape[1])[:, None] > np.arange(start.shape[1])[None, :]
    distance = np.where(prior[None], start[:, :, None] - stop[:, None, :], 0)
    previous_cos = np.where(prior[None, :, :, None], np.exp(alpha * distance[..., None]) * np.cos(beta * distance[..., None]), 0)
    previous_sin = np.where(prior[None, :, :, None], np.exp(alpha * distance[..., None]) * np.sin(beta * distance[..., None]), 0)
    return dict(start=start, stop=stop, voltage=voltage, ids=ids, times=times,
                observation=obs, duration=duration, phase=phase, segment=segment,
                alpha=alpha, beta=beta, residue_real=real, residue_imag=imag,
                duration_cos=np.exp(alpha * duration[..., None]) * np.cos(beta * duration[..., None]),
                duration_sin=np.exp(alpha * duration[..., None]) * np.sin(beta * duration[..., None]),
                query_cos=np.exp(alpha * phase[:, None]) * np.cos(beta * phase[:, None]),
                query_sin=np.exp(alpha * phase[:, None]) * np.sin(beta * phase[:, None]),
                previous_cos=previous_cos, previous_sin=previous_sin, passive_pa=passive)


def _mode_forcing(coefficients, rates, phase, cosine, sine, alpha, beta):
    d = rates[..., None] + alpha
    denominator = d**2 + beta**2
    exponential = np.exp(-rates * phase[..., None])[..., None]
    difference = cosine[..., None, :] - exponential
    real = (d * difference + beta * sine[..., None, :]) / denominator
    imag = (d * sine[..., None, :] - beta * difference) / denominator
    return (np.sum(coefficients[..., None] * real, axis=-2),
            np.sum(coefficients[..., None] * imag, axis=-2))


def predict_active(parameters, context, *, components=False):
    """Evaluate exact gated current and its stateful Bessel-filtered observation.

    Parameters
    ----------
    parameters : array-like
        Thirteen active-current parameters in ORDER. Complex perturbations are
        supported for derivative calculation; original bounds remain enforced.
    context : dict
        Fixed geometry returned by prepare_commands.
    components : bool, optional
        Return total, active-observation and frozen passive predictions separately.

    Returns
    -------
    numpy.ndarray or dict
        Conditional baseline-relative amplifier current in pA. This is not an
        independently validated ionic-current or membrane-voltage prediction.
    """
    p = np.asarray(parameters)
    if (p.shape != (13,) or not np.isfinite(p).all()
            or np.any(p.real < LOWER) or np.any(p.real > UPPER)):
        raise ValueError('Invalid active-current parameters.')
    v, duration = context['voltage'], context['duration']
    m = 1 / (1 + np.exp((p[3] - v) / p[4]))
    h = 1 / (1 + np.exp((v - p[6]) / p[7]))
    equilibrium = np.stack([m, h, h], axis=-1)
    tau = np.stack([np.broadcast_to(p[5], v.shape),
                    p[8] + (p[10] - p[8]) * h,
                    p[9] + (p[11] - p[9]) * h], axis=-1)
    integrated = duration[..., None] / tau
    states = affine_starts(integrated, equilibrium * -np.expm1(-integrated), equilibrium[:, 0])
    delta = states - equilibrium
    mc = np.stack([m**2, 2 * m * delta[..., 0], delta[..., 0]**2], axis=-1)
    mr = np.broadcast_to(np.arange(3) / p[5], mc.shape)
    hc = np.stack([np.stack([h, h, np.ones_like(h)], axis=-1),
                   np.concatenate([delta[..., 1:], np.zeros_like(delta[..., :1])], axis=-1)], axis=-1)
    hr = np.stack([np.zeros_like(hc[..., 0]),
                   np.concatenate([1 / tau[..., 1:], np.zeros_like(delta[..., :1])], axis=-1)], axis=-1)
    coefficients = ((v - p[12])[..., None, None, None] * p[:3][None, None, :, None, None]
                    * mc[..., None, :, None] * hc[..., :, None, :]).reshape(v.shape + (18,))
    rates = (mr[..., None, :, None] + hr[..., :, None, :]).reshape(v.shape + (18,))
    initial = (v[:, :1] - p[12]) * m[:, :1]**2 * ((p[0] + p[1]) * h[:, :1] + p[2])
    coefficients = np.concatenate([coefficients, -np.broadcast_to(initial, v.shape)[..., None]], axis=-1)
    rates = np.concatenate([rates, np.zeros_like(v[..., None])], axis=-1)
    real_end, imag_end = _mode_forcing(coefficients, rates, duration,
                                      context['duration_cos'], context['duration_sin'],
                                      context['alpha'], context['beta'])
    pc, ps = context['previous_cos'], context['previous_sin']
    real_start = np.einsum('bijp,bjp->bip', pc, real_end) - np.einsum('bijp,bjp->bip', ps, imag_end)
    imag_start = np.einsum('bijp,bjp->bip', ps, real_end) + np.einsum('bijp,bjp->bip', pc, imag_end)
    ids, segment = context['ids'], context['segment']
    real_force, imag_force = _mode_forcing(coefficients[ids, segment], rates[ids, segment], context['phase'],
                                          context['query_cos'], context['query_sin'], context['alpha'], context['beta'])
    qc, qs = context['query_cos'], context['query_sin']
    real_mode = qc * real_start[ids, segment] - qs * imag_start[ids, segment] + real_force
    imag_mode = qs * real_start[ids, segment] + qc * imag_start[ids, segment] + imag_force
    active = np.sum(2 * (context['residue_real'] * real_mode - context['residue_imag'] * imag_mode), axis=-1)
    total = active + context['passive_pa']
    if not np.isfinite(total).all():
        raise FloatingPointError('Nonfinite filtered active-current prediction.')
    if components:
        return dict(filtered_pa=total, filtered_active_pa=active, passive_pa=context['passive_pa'])
    return total
