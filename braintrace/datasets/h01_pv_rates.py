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


def _natg(v, ca):
    return {"m": _pair(rate_trap(v, -38., .182, 9.), rate_trap(-v, 38., .124, 9.), _Q),
            "h": _pair(rate_trap(-v, 56., .015, 6.), rate_trap(v, -56., .015, 6.), _Q)}


def _nap(v, ca):
    total = rate_trap(v, -38., .182, 6.)+rate_trap(-v, 38., .124, 6.)
    slow = rate_trap(-v, 17., 2.88e-6, 4.63)+rate_trap(v, -64.4, 6.94e-6, 2.63)
    return {"m": (sigmoid((v+52.6)/4.6), 6./(_Q*total)),
            "h": (sigmoid(-(v+48.8)/10.), 1./(_Q*slow))}


def _kp(v, ca):
    x = v+10.
    tau = jnp.where(x < -50., 1.25+175.03*jnp.exp(.026*x), 1.25+13.*jnp.exp(-.026*x))/_Q
    return {"m": (sigmoid((x+1.)/12.), tau),
            "h": (sigmoid(-(x+54.)/11.), (360.+(1010.+24.*(x+55.))*jnp.exp(-((x+75.)/48.)**2))/_Q)}


def _kt(v, ca):
    x = v+10.
    return {"m": (sigmoid(x/19.), (.34+.92*jnp.exp(-((x+71.)/59.)**2))/_Q),
            "h": (sigmoid(-(x+66.)/10.), (8.+49.*jnp.exp(-((x+73.)/23.)**2))/_Q)}


def _kv3(v, ca):
    return {"m": (sigmoid((v-18.7)/9.7), 4.*sigmoid((v+46.56)/44.14))}


def _im(v, ca):
    return {"m": _pair(.0033*jnp.exp(.1*(v+35.)), .0033*jnp.exp(-.1*(v+35.)), _Q)}


def _ih(v, ca):
    alpha = rate_trap(-v, 262.07471272583734, .001*78.52193065082363, 19.041504011554068)
    beta = .001*1.3115816183914442*jnp.exp((v+4.162478599311996)/75.56131984386032)
    return {"m": _pair(alpha, beta)}


def _cahva(v, ca):
    return {"m": _pair(rate_trap(v, -27., .055, 3.8), .94*jnp.exp((-75.-v)/17.)),
            "h": _pair(.000457*jnp.exp((-13.-v)/50.), .0065*sigmoid((v+15.)/28.))}


def _calva(v, ca):
    x = v+10.
    return {"m": (sigmoid((x+30.)/6.), (5.+20.*sigmoid(-(x+25.)/5.))/_Q),
            "h": (sigmoid(-(x+80.)/6.4), (20.+50.*sigmoid(-(x+40.)/7.))/_Q)}


def _sk(v, ca):
    safe = jnp.maximum(ca, 1e-30)
    activated = sigmoid(4.8*jnp.log(safe/.00043))
    return {"z": (jnp.where(ca < 1e-4, 0., activated), jnp.ones_like(ca))}


_RATES = {"NaTg": _natg, "Nap": _nap, "K_P": _kp, "K_T": _kt,
          "Kv3_1": _kv3, "Im": _im, "Ih": _ih, "Ca_HVA": _cahva,
          "Ca_LVA": _calva, "SK": _sk}


def pv_rates(mechanism, voltage_mv, calcium_mm=1e-4):
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

    Returns
    -------
    dict
        Gate name mapped to a pair: dimensionless equilibrium and tau in ms.

    Notes
    -----
    The voltage-gated source equations use a fixed 34 Celsius convention.
    Singular limits replace source voltage perturbations at isolated poles.
    This function describes the released fitted kinetics, not their validation
    against human recordings or an arbitrary-temperature extension.
    """
    if mechanism not in _RATES:
        raise ValueError(f"Unknown PV mechanism: {mechanism!r}.")
    return _RATES[mechanism](jnp.asarray(voltage_mv), jnp.asarray(calcium_mm))
