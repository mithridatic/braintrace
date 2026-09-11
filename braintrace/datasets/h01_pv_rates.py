"""Channel equations for the published human-fitted HL5BN1 reference cell.

ModelDB 267587, commit 82cdd91bc93942ba19315371330a2412e064baf5.
Kinetics include nonhuman sources. They are not H01 channel measurements.
Rate parameters match the released fitted cell at its fixed 34 Celsius setting.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid
from braincell._misc import is_traced_value

from .h01_wilbers import rate_trap


_Q = 2.3**1.3


def _is_jax(x):
    return is_traced_value(x) or isinstance(x, (jax.core.Tracer, jax.Array))


def _sig(x):
    if _is_jax(x):
        return sigmoid(x)
    arr = np.asarray(x, dtype=np.float64)
    return 1. / (1. + np.exp(-arr))


def _exp(x):
    if _is_jax(x):
        return jnp.exp(x)
    return np.exp(x)


def _where(cond, x, y):
    if _is_jax(cond) or _is_jax(x) or _is_jax(y):
        return jnp.where(cond, x, y)
    return np.where(cond, x, y)


def _max(x, y):
    if _is_jax(x) or _is_jax(y):
        return jnp.maximum(x, y)
    return np.maximum(x, y)


def _log(x):
    if _is_jax(x):
        return jnp.log(x)
    return np.log(x)


def _ones_like(x):
    if _is_jax(x):
        return jnp.ones_like(x)
    return np.ones_like(x)


def _pair(alpha, beta, q=1.):
    total = alpha+beta
    inf = alpha/total
    if _is_jax(inf):
        inf = jnp.clip(inf, 0., 1.)
    elif isinstance(inf, np.ndarray):
        inf = np.clip(inf, 0., 1.)
    elif isinstance(inf, (int, float)):
        inf = max(0., min(1., float(inf)))
    return inf, 1./(q*total)


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
            return _sig((v + 52.6) / 4.6)
        if component == 1:
            total = rate_trap(v, -38., .182, 6.) + rate_trap(-v, 38., .124, 6.)
            return 6. / (_Q * total)
        total = rate_trap(v, -38., .182, 6.) + rate_trap(-v, 38., .124, 6.)
        return (_sig((v + 52.6) / 4.6), 6. / (_Q * total))
    if gate == "h":
        if component == 0:
            return _sig(-(v + 48.8) / 10.)
        if component == 1:
            slow = rate_trap(-v, 17., 2.88e-6, 4.63) + rate_trap(v, -64.4, 6.94e-6, 2.63)
            return 1. / (_Q * slow)
        slow = rate_trap(-v, 17., 2.88e-6, 4.63) + rate_trap(v, -64.4, 6.94e-6, 2.63)
        return (_sig(-(v + 48.8) / 10.), 1. / (_Q * slow))
    total = rate_trap(v, -38., .182, 6.) + rate_trap(-v, 38., .124, 6.)
    slow = rate_trap(-v, 17., 2.88e-6, 4.63) + rate_trap(v, -64.4, 6.94e-6, 2.63)
    return {"m": (_sig((v + 52.6) / 4.6), 6. / (_Q * total)),
            "h": (_sig(-(v + 48.8) / 10.), 1. / (_Q * slow))}


def _kp(v, ca, gate=None, component=None):
    x = v + 10.
    if gate == "m":
        if component == 0:
            return _sig((x + 1.) / 12.)
        tau = _where(x < -50., 1.25 + 175.03 * _exp(.026 * x), 1.25 + 13. * _exp(-.026 * x)) / _Q
        if component == 1:
            return tau
        return (_sig((x + 1.) / 12.), tau)
    if gate == "h":
        if component == 0:
            return _sig(-(x + 54.) / 11.)
        tau = (360. + (1010. + 24. * (x + 55.)) * _exp(-((x + 75.) / 48.) ** 2)) / _Q
        if component == 1:
            return tau
        return (_sig(-(x + 54.) / 11.), tau)
    tau_m = _where(x < -50., 1.25 + 175.03 * _exp(.026 * x), 1.25 + 13. * _exp(-.026 * x)) / _Q
    tau_h = (360. + (1010. + 24. * (x + 55.)) * _exp(-((x + 75.) / 48.) ** 2)) / _Q
    return {"m": (_sig((x + 1.) / 12.), tau_m),
            "h": (_sig(-(x + 54.) / 11.), tau_h)}


def _kt(v, ca, gate=None, component=None):
    x = v + 10.
    if gate == "m":
        if component == 0:
            return _sig(x / 19.)
        tau = (.34 + .92 * _exp(-((x + 71.) / 59.) ** 2)) / _Q
        if component == 1:
            return tau
        return (_sig(x / 19.), tau)
    if gate == "h":
        if component == 0:
            return _sig(-(x + 66.) / 10.)
        tau = (8. + 49. * _exp(-((x + 73.) / 23.) ** 2)) / _Q
        if component == 1:
            return tau
        return (_sig(-(x + 66.) / 10.), tau)
    return {"m": (_sig(x / 19.), (.34 + .92 * _exp(-((x + 71.) / 59.) ** 2)) / _Q),
            "h": (_sig(-(x + 66.) / 10.), (8. + 49. * _exp(-((x + 73.) / 23.) ** 2)) / _Q)}


def _kv3(v, ca, gate=None, component=None):
    if component == 0:
        return _sig((v - 18.7) / 9.7)
    if component == 1:
        return 4. * _sig((v + 46.56) / 44.14)
    res = (_sig((v - 18.7) / 9.7), 4. * _sig((v + 46.56) / 44.14))
    return res if gate == "m" else {"m": res}


def _im(v, ca, gate=None, component=None):
    if component == 0:
        a = .0033 * _exp(.1 * (v + 35.))
        b = .0033 * _exp(-.1 * (v + 35.))
        return a / (a + b)
    if component == 1:
        return 1. / (_Q * (.0033 * _exp(.1 * (v + 35.)) + .0033 * _exp(-.1 * (v + 35.))))
    res = _pair(.0033 * _exp(.1 * (v + 35.)), .0033 * _exp(-.1 * (v + 35.)), _Q)
    return res if gate == "m" else {"m": res}


def _ih(v, ca, gate=None, component=None):
    if component == 0:
        a = rate_trap(-v, 262.07471272583734, .001 * 78.52193065082363, 19.041504011554068)
        b = .001 * 1.3115816183914442 * _exp((v + 4.162478599311996) / 75.56131984386032)
        return a / (a + b)
    if component == 1:
        a = rate_trap(-v, 262.07471272583734, .001 * 78.52193065082363, 19.041504011554068)
        b = .001 * 1.3115816183914442 * _exp((v + 4.162478599311996) / 75.56131984386032)
        return 1. / (a + b)
    alpha = rate_trap(-v, 262.07471272583734, .001 * 78.52193065082363, 19.041504011554068)
    beta = .001 * 1.3115816183914442 * _exp((v + 4.162478599311996) / 75.56131984386032)
    res = _pair(alpha, beta)
    return res if gate == "m" else {"m": res}


def _cahva(v, ca, gate=None, component=None):
    if gate == "m":
        if component == 0:
            a = rate_trap(v, -27., .055, 3.8)
            b = .94 * _exp((-75. - v) / 17.)
            return a / (a + b)
        if component == 1:
            a = rate_trap(v, -27., .055, 3.8)
            b = .94 * _exp((-75. - v) / 17.)
            return 1. / (a + b)
        return _pair(rate_trap(v, -27., .055, 3.8), .94 * _exp((-75. - v) / 17.))
    if gate == "h":
        if component == 0:
            a = .000457 * _exp((-13. - v) / 50.)
            b = .0065 * _sig((v + 15.) / 28.)
            return a / (a + b)
        if component == 1:
            a = .000457 * _exp((-13. - v) / 50.)
            b = .0065 * _sig((v + 15.) / 28.)
            return 1. / (a + b)
        return _pair(.000457 * _exp((-13. - v) / 50.), .0065 * _sig((v + 15.) / 28.))
    return {"m": _pair(rate_trap(v, -27., .055, 3.8), .94 * _exp((-75. - v) / 17.)),
            "h": _pair(.000457 * _exp((-13. - v) / 50.), .0065 * _sig((v + 15.) / 28.))}


def _calva(v, ca, gate=None, component=None):
    x = v + 10.
    if gate == "m":
        if component == 0:
            return _sig((x + 30.) / 6.)
        if component == 1:
            return (5. + 20. * _sig(-(x + 25.) / 5.)) / _Q
        return (_sig((x + 30.) / 6.), (5. + 20. * _sig(-(x + 25.) / 5.)) / _Q)
    if gate == "h":
        if component == 0:
            return _sig(-(x + 80.) / 6.4)
        if component == 1:
            return (20. + 50. * _sig(-(x + 40.) / 7.)) / _Q
        return (_sig(-(x + 80.) / 6.4), (20. + 50. * _sig(-(x + 40.) / 7.)) / _Q)
    return {"m": (_sig((x + 30.) / 6.), (5. + 20. * _sig(-(x + 25.) / 5.)) / _Q),
            "h": (_sig(-(x + 80.) / 6.4), (20. + 50. * _sig(-(x + 40.) / 7.)) / _Q)}


def _sk(v, ca, gate=None, component=None):
    if component == 1:
        return _ones_like(ca)
    safe = _max(ca, 1e-30)
    activated = _sig(4.8 * _log(safe / .00043))
    inf = _where(ca < 1e-4, 0., activated)
    if component == 0:
        return inf
    res = (inf, _ones_like(ca))
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
    v = jnp.asarray(voltage_mv) if _is_jax(voltage_mv) else np.asarray(voltage_mv)
    ca = jnp.asarray(calcium_mm) if _is_jax(calcium_mm) else np.asarray(calcium_mm)
    return _RATES[mechanism](v, ca, gate=gate, component=component)
