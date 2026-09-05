"""Wilbers 2023 human cortical channel kinetics in millivolts and milliseconds.

Source: doi:10.34894/L5J0SD, version 3.0, Current_clamp/mod files
318563 (sodium) and 318566 (potassium), CC0. These are population current
models, not molecular channel subtype assignments or H01 measurements.
The source's singularity guard is replaced with its analytic limit.
"""

import jax.numpy as jnp
from jax.nn import sigmoid


def rate_trap(voltage_mv, threshold_mv, scale, slope_mv):
    """Evaluate a removable-singularity rate with a finite derivative.

    Parameters
    ----------
    voltage_mv, threshold_mv, slope_mv : array_like
        Voltage, rate threshold and positive slope, in millivolts.
    scale : array_like
        Rate coefficient in inverse milliseconds per millivolt.

    Returns
    -------
    array
        Rate in inverse milliseconds.
    """
    x = (jnp.asarray(voltage_mv) - threshold_mv) / slope_mv
    small = jnp.abs(x) < 1e-4
    safe = jnp.where(small, 1., jnp.abs(x))
    regular = safe / -jnp.expm1(-safe) * jnp.where(x < 0, jnp.exp(-safe), 1.)
    return scale * slope_mv * jnp.where(small, 1. + x * (0.5 + x / 12.), regular)


def sodium_rates(voltage_mv, *, temperature_c=34., gate=None):
    """Return human sodium steady states and time constants.

    Parameters
    ----------
    voltage_mv : array_like
        Channel voltage in mV, after the caller's reference shift.
    temperature_c : float, optional
        Celsius temperature; source reference is 25 degrees.
    gate : str or None, optional
        Specific gate identifier ('m' or 'h') to compute only that gate.

    Returns
    -------
    tuple
        Activation and inactivation steady states, then their time constants
        in ms if gate is None. If gate is specified, returns (equilibrium, tau_ms)
        for that gate. Temperature scales time constants, not steady states.
    """
    v = jnp.asarray(voltage_mv)
    phi = 2.3 ** ((temperature_c - 25.) / 10)
    if gate == "m":
        ma = rate_trap(v, -58.59235606617833, .2139849270413559, 9.21455371864071)
        mb = rate_trap(-v, 58.59235606617833, .2139849270413559, 9.21455371864071)
        return (sigmoid((v + 42.29145668954657) / 10.231152896063074), 1 / (phi * (ma + mb)))
    if gate == "h":
        ha = rate_trap(v, -39.298676911255, .0476391878313753, 7.21636386592805)
        hb = rate_trap(-v, 77.16674532461128, .011951655023359499, 4.749450048393884)
        return (sigmoid(-(v + 64.48) / 11.02), 1 / (phi * (ha + hb)))
    ma = rate_trap(v, -58.59235606617833, .2139849270413559, 9.21455371864071)
    mb = rate_trap(-v, 58.59235606617833, .2139849270413559, 9.21455371864071)
    ha = rate_trap(v, -39.298676911255, .0476391878313753, 7.21636386592805)
    hb = rate_trap(-v, 77.16674532461128, .011951655023359499, 4.749450048393884)
    return (sigmoid((v + 42.29145668954657) / 10.231152896063074),
            sigmoid(-(v + 64.48) / 11.02), 1/(phi*(ma+mb)), 1/(phi*(ha+hb)))


def potassium_rates(voltage_mv, *, temperature_c=34., gate=None):
    """Return human potassium steady states and fast time constants.

    Parameters
    ----------
    voltage_mv : array_like
        Channel voltage in mV, after the caller's reference shift.
    temperature_c : float, optional
        Celsius temperature; source reference is 34 degrees.
    gate : str or None, optional
        Specific gate identifier ('m' or 'h') to compute only that gate.

    Returns
    -------
    tuple
        Activation and shared inactivation steady states, activation time
        constant and fast-inactivation time constant in ms if gate is None.
        If gate is specified, returns (equilibrium, tau_ms) for that gate.
        Slow inactivation uses 450 ms and recovery 2789.7929428108187 ms
        in the source and must be selected using the slow gate state,
        separately from this function.
    """
    v = jnp.asarray(voltage_mv)
    phi = 2.3 ** ((temperature_c - 34.) / 10)
    if gate == "m":
        ma = rate_trap(v, -29.961798016904638, .018459250585411965, 6.590405688881463)
        mb = rate_trap(-v, 29.961798016904638, .018459250585411965, 6.590405688881463)
        return (sigmoid((v + 18.406163539758502) / 19.199550273441083), 1 / (phi * (ma + mb)))
    if gate == "h":
        ha = rate_trap(v, 40.5659148938157, .0016901435659310753, 26.773929244740913)
        hb = rate_trap(-v, 75.93822659307075, .0004968791312639546, .40183382749400054)
        return (sigmoid(-(v + 52.61) / 13.09), 1 / (phi * (ha + hb)))
    ma = rate_trap(v, -29.961798016904638, .018459250585411965, 6.590405688881463)
    mb = rate_trap(-v, 29.961798016904638, .018459250585411965, 6.590405688881463)
    ha = rate_trap(v, 40.5659148938157, .0016901435659310753, 26.773929244740913)
    hb = rate_trap(-v, 75.93822659307075, .0004968791312639546, .40183382749400054)
    return (sigmoid((v + 18.406163539758502) / 19.199550273441083),
            sigmoid(-(v + 52.61) / 13.09), 1/(phi*(ma+mb)), 1/(phi*(ha+hb)))
