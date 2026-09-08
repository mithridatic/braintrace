"""Channel equations for the published human-fitted HL5BN1 reference cell.

ModelDB 267587, commit 82cdd91bc93942ba19315371330a2412e064baf5.
Kinetics include nonhuman sources. They are not H01 channel measurements.
Rate parameters match the released fitted cell at its fixed 34 Celsius setting.
"""

import jax.numpy as jnp
from jax.nn import sigmoid

from .h01_wilbers import rate_trap


_Q = 2.3**1.3


def _pair(alpha, beta, q=1.):
    total = alpha+beta
    return alpha/total, 1./(q*total)


def _natg(v, ca, gate=None, component=None):
    if gate == "m":
        if component == 0:
            a = rate_trap(v, -38., .182, 9.)
            b = rate_trap(-v, 38., .124, 9.)
            return a / (a + b)
        if component == 1:
            return 1. / (_Q * (rate_trap(v, -38., .182, 9.) + rate_trap(-v, 38., .124, 9.)))
        return _pair(rate_trap(v, -38., .182, 9.), rate_trap(-v, 38., .124, 9.), _Q)
    if gate == "h":
        if component == 0:
            a = rate_trap(-v, 56., .015, 6.)
            b = rate_trap(v, -56., .015, 6.)
            return a / (a + b)
        if component == 1:
            return 1. / (_Q * (rate_trap(-v, 56., .015, 6.) + rate_trap(v, -56., .015, 6.)))
        return _pair(rate_trap(-v, 56., .015, 6.), rate_trap(v, -56., .015, 6.), _Q)
    return {"m": _pair(rate_trap(v, -38., .182, 9.), rate_trap(-v, 38., .124, 9.), _Q),
            "h": _pair(rate_trap(-v, 56., .015, 6.), rate_trap(v, -56., .015, 6.), _Q)}


def _nap(v, ca, gate=None, component=None):
    if gate == "m":
        if component == 0:
            return sigmoid((v + 52.6) / 4.6)
        if component == 1:
            total = rate_trap(v, -38., .182, 6.) + rate_trap(-v, 38., .124, 6.)
            return 6. / (_Q * total)
        total = rate_trap(v, -38., .182, 6.) + rate_trap(-v, 38., .124, 6.)
        return (sigmoid((v + 52.6) / 4.6), 6. / (_Q * total))
    if gate == "h":
        if component == 0:
            return sigmoid(-(v + 48.8) / 10.)
        if component == 1:
            slow = rate_trap(-v, 17., 2.88e-6, 4.63) + rate_trap(v, -64.4, 6.94e-6, 2.63)
            return 1. / (_Q * slow)
        slow = rate_trap(-v, 17., 2.88e-6, 4.63) + rate_trap(v, -64.4, 6.94e-6, 2.63)
        return (sigmoid(-(v + 48.8) / 10.), 1. / (_Q * slow))
    total = rate_trap(v, -38., .182, 6.) + rate_trap(-v, 38., .124, 6.)
    slow = rate_trap(-v, 17., 2.88e-6, 4.63) + rate_trap(v, -64.4, 6.94e-6, 2.63)
    return {"m": (sigmoid((v + 52.6) / 4.6), 6. / (_Q * total)),
            "h": (sigmoid(-(v + 48.8) / 10.), 1. / (_Q * slow))}


def _kp(v, ca, gate=None, component=None):
    x = v + 10.
    if gate == "m":
        if component == 0:
            return sigmoid((x + 1.) / 12.)
        tau = jnp.where(x < -50., 1.25 + 175.03 * jnp.exp(.026 * x), 1.25 + 13. * jnp.exp(-.026 * x)) / _Q
        if component == 1:
            return tau
        return (sigmoid((x + 1.) / 12.), tau)
    if gate == "h":
        if component == 0:
            return sigmoid(-(x + 54.) / 11.)
        tau = (360. + (1010. + 24. * (x + 55.)) * jnp.exp(-((x + 75.) / 48.) ** 2)) / _Q
        if component == 1:
            return tau
        return (sigmoid(-(x + 54.) / 11.), tau)
    tau_m = jnp.where(x < -50., 1.25 + 175.03 * jnp.exp(.026 * x), 1.25 + 13. * jnp.exp(-.026 * x)) / _Q
    tau_h = (360. + (1010. + 24. * (x + 55.)) * jnp.exp(-((x + 75.) / 48.) ** 2)) / _Q
    return {"m": (sigmoid((x + 1.) / 12.), tau_m),
            "h": (sigmoid(-(x + 54.) / 11.), tau_h)}


def _kt(v, ca, gate=None, component=None):
    x = v + 10.
    if gate == "m":
        if component == 0:
            return sigmoid(x / 19.)
        tau = (.34 + .92 * jnp.exp(-((x + 71.) / 59.) ** 2)) / _Q
        if component == 1:
            return tau
        return (sigmoid(x / 19.), tau)
    if gate == "h":
        if component == 0:
            return sigmoid(-(x + 66.) / 10.)
        tau = (8. + 49. * jnp.exp(-((x + 73.) / 23.) ** 2)) / _Q
        if component == 1:
            return tau
        return (sigmoid(-(x + 66.) / 10.), tau)
    return {"m": (sigmoid(x / 19.), (.34 + .92 * jnp.exp(-((x + 71.) / 59.) ** 2)) / _Q),
            "h": (sigmoid(-(x + 66.) / 10.), (8. + 49. * jnp.exp(-((x + 73.) / 23.) ** 2)) / _Q)}


def _kv3(v, ca, gate=None, component=None):
    if component == 0:
        return sigmoid((v - 18.7) / 9.7)
    if component == 1:
        return 4. * sigmoid((v + 46.56) / 44.14)
    res = (sigmoid((v - 18.7) / 9.7), 4. * sigmoid((v + 46.56) / 44.14))
    return res if gate == "m" else {"m": res}


def _im(v, ca, gate=None, component=None):
    if component == 0:
        a = .0033 * jnp.exp(.1 * (v + 35.))
        b = .0033 * jnp.exp(-.1 * (v + 35.))
        return a / (a + b)
    if component == 1:
        return 1. / (_Q * (.0033 * jnp.exp(.1 * (v + 35.)) + .0033 * jnp.exp(-.1 * (v + 35.))))
    res = _pair(.0033 * jnp.exp(.1 * (v + 35.)), .0033 * jnp.exp(-.1 * (v + 35.)), _Q)
    return res if gate == "m" else {"m": res}


def _ih(v, ca, gate=None, component=None):
    if component == 0:
        a = rate_trap(-v, 262.07471272583734, .001 * 78.52193065082363, 19.041504011554068)
        b = .001 * 1.3115816183914442 * jnp.exp((v + 4.162478599311996) / 75.56131984386032)
        return a / (a + b)
    if component == 1:
        a = rate_trap(-v, 262.07471272583734, .001 * 78.52193065082363, 19.041504011554068)
        b = .001 * 1.3115816183914442 * jnp.exp((v + 4.162478599311996) / 75.56131984386032)
        return 1. / (a + b)
    alpha = rate_trap(-v, 262.07471272583734, .001 * 78.52193065082363, 19.041504011554068)
    beta = .001 * 1.3115816183914442 * jnp.exp((v + 4.162478599311996) / 75.56131984386032)
    res = _pair(alpha, beta)
    return res if gate == "m" else {"m": res}


def _cahva(v, ca, gate=None, component=None):
    if gate == "m":
        if component == 0:
            a = rate_trap(v, -27., .055, 3.8)
            b = .94 * jnp.exp((-75. - v) / 17.)
            return a / (a + b)
        if component == 1:
            a = rate_trap(v, -27., .055, 3.8)
            b = .94 * jnp.exp((-75. - v) / 17.)
            return 1. / (a + b)
        return _pair(rate_trap(v, -27., .055, 3.8), .94 * jnp.exp((-75. - v) / 17.))
    if gate == "h":
        if component == 0:
            a = .000457 * jnp.exp((-13. - v) / 50.)
            b = .0065 * sigmoid((v + 15.) / 28.)
            return a / (a + b)
        if component == 1:
            a = .000457 * jnp.exp((-13. - v) / 50.)
            b = .0065 * sigmoid((v + 15.) / 28.)
            return 1. / (a + b)
        return _pair(.000457 * jnp.exp((-13. - v) / 50.), .0065 * sigmoid((v + 15.) / 28.))
    return {"m": _pair(rate_trap(v, -27., .055, 3.8), .94 * jnp.exp((-75. - v) / 17.)),
            "h": _pair(.000457 * jnp.exp((-13. - v) / 50.), .0065 * sigmoid((v + 15.) / 28.))}


def _calva(v, ca, gate=None, component=None):
    x = v + 10.
    if gate == "m":
        if component == 0:
            return sigmoid((x + 30.) / 6.)
        if component == 1:
            return (5. + 20. * sigmoid(-(x + 25.) / 5.)) / _Q
        return (sigmoid((x + 30.) / 6.), (5. + 20. * sigmoid(-(x + 25.) / 5.)) / _Q)
    if gate == "h":
        if component == 0:
            return sigmoid(-(x + 80.) / 6.4)
        if component == 1:
            return (20. + 50. * sigmoid(-(x + 40.) / 7.)) / _Q
        return (sigmoid(-(x + 80.) / 6.4), (20. + 50. * sigmoid(-(x + 40.) / 7.)) / _Q)
    return {"m": (sigmoid((x + 30.) / 6.), (5. + 20. * sigmoid(-(x + 25.) / 5.)) / _Q),
            "h": (sigmoid(-(x + 80.) / 6.4), (20. + 50. * sigmoid(-(x + 40.) / 7.)) / _Q)}


def _sk(v, ca, gate=None, component=None):
    if component == 1:
        return jnp.ones_like(ca)
    safe = jnp.maximum(ca, 1e-30)
    activated = sigmoid(4.8 * jnp.log(safe / .00043))
    inf = jnp.where(ca < 1e-4, 0., activated)
    if component == 0:
        return inf
    res = (inf, jnp.ones_like(ca))
    return res if gate == "z" else {"z": res}


_RATES = {"NaTg": _natg, "Nap": _nap, "K_P": _kp, "K_T": _kt,
          "Kv3_1": _kv3, "Im": _im, "Ih": _ih, "Ca_HVA": _cahva,
          "Ca_LVA": _calva, "SK": _sk}


def pv_rates(mechanism, voltage_mv, calcium_mm=1e-4, *, gate=None, component=None):
    """Return gate equilibria and time constants for the fitted PV model.

    Parameters
    ----------
    mechanism : str
        One of NaTg, Nap, K_P, K_T, Kv3_1, Im, Ih, Ca_HVA, Ca_LVA, or SK.
        Keep this argument static when compiling with JAX.
    voltage_mv : array_like
        Membrane voltage in millivolts, without an extra voltage shift.
    calcium_mm : array_like, optional
        Intracellular calcium in millimolar. Only SK uses this value.
    gate : str or None, optional
        Specific gate identifier (e.g. 'm', 'h', 'z') to compute only that gate.
    component : int or None, optional
        0 for equilibrium only, 1 for tau_ms only, None for both.

    Returns
    -------
    dict or tuple or array
        If gate is None, dictionary mapping gate name to (equilibrium, tau_ms).
        If gate is specified and component is None, the (equilibrium, tau_ms) pair for that gate.
        If component is specified (0 or 1), the corresponding array for that gate.

    Notes
    -----
    The voltage-gated source equations use a fixed 34 Celsius convention.
    Singular limits replace source voltage perturbations at isolated poles.
    This function describes the released fitted kinetics, not their validation
    against human recordings or an arbitrary-temperature extension.
    """
    if mechanism not in _RATES:
        raise ValueError(f"Unknown PV mechanism: {mechanism!r}.")
    return _RATES[mechanism](jnp.asarray(voltage_mv), jnp.asarray(calcium_mm), gate=gate, component=component)
