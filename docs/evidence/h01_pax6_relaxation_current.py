"""Conditional active current with independent relaxation voltage dependence."""

import numpy as np
from docs.evidence.h01_pax6_bessel_current import (
    prepare_commands, _mode_forcing, INITIAL as OLD_INITIAL,
    LOWER as OLD_LOWER, UPPER as OLD_UPPER, ORDER as OLD_ORDER,
)
from docs.evidence.h01_pax6_voltage_model import affine_starts

INITIAL = np.r_[OLD_INITIAL, -20., 10.]
LOWER = np.r_[OLD_LOWER, -60., 2.]
UPPER = np.r_[OLD_UPPER, 20., 35.]
ORDER = list(OLD_ORDER) + ['vtau', 'ktau']


def predict_active(parameters, context, *, components=False):
    """Evaluate exact gated current and its stateful Bessel-filtered observation.

    Parameters
    ----------
    parameters : array-like
        Fifteen active-current parameters in ORDER. Complex perturbations are
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
    if (p.shape != (15,) or not np.isfinite(p).all()
            or np.any(p.real < LOWER) or np.any(p.real > UPPER)):
        raise ValueError('Invalid active-current parameters.')
    v, duration = context['voltage'], context['duration']
    m = 1 / (1 + np.exp((p[3] - v) / p[4]))
    h = 1 / (1 + np.exp((v - p[6]) / p[7]))
    q = 1 / (1 + np.exp((v - p[13]) / p[14]))
    equilibrium = np.stack([m, h, h], axis=-1)
    tau = np.stack([np.broadcast_to(p[5], v.shape),
                    p[8] + (p[10] - p[8]) * q,
                    p[9] + (p[11] - p[9]) * q], axis=-1)
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
