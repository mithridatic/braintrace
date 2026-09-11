"""Experimental positive calcium solve; not installed in an H01 integrator."""

import brainstate
import jax
import jax.lax
import jax.numpy as jnp


def calcium_backward_euler(c_old, voltage, conductance, dt, decay, flux_factor,
                           nernst_factor, c_rest=1e-4, c_out=2.):
    """Solve the signed calcium equation with implicit Nernst feedback.

    Parameters
    ----------
    c_old, c_rest, c_out : array_like
        Positive concentrations in mM.
    voltage : array_like
        Frozen membrane voltage in mV.
    conductance : array_like
        Nonnegative frozen calcium conductance in mS/cm2.
    dt, decay : array_like
        Positive step and removal times in ms.
    flux_factor : array_like
        Nonnegative conversion from inward mA/cm2 to mM/ms.
    nernst_factor : array_like
        Positive RT/(zF) in mV.

    Returns
    -------
    Array
        Positive backward-Euler concentration, or NaN for invalid inputs.

    Notes
    -----
    Voltage and channel gates are frozen; calcium-dependent Nernst reversal
    remains inside the solve. No concentration or signed-current clipping is
    applied. This experimental helper does not change the default H01 solver.
    """
    c_old, voltage, conductance, dt, decay, flux_factor, nernst_factor, c_rest, c_out = jnp.broadcast_arrays(*(
        jnp.asarray(value) for value in
        (c_old, voltage, conductance, dt, decay, flux_factor, nernst_factor, c_rest, c_out)))
    valid = ((c_old > 0) & (conductance >= 0) & (dt > 0) & (decay > 0)
             & (flux_factor >= 0) & (nernst_factor > 0) & (c_rest > 0) & (c_out > 0))
    for value in (c_old, voltage, conductance, dt, decay, flux_factor, nernst_factor, c_rest, c_out):
        valid = valid & jnp.isfinite(value)
    old_log, rest_log = jnp.log(c_old), jnp.log(c_rest)
    equilibrium_log = jnp.log(c_out)-voltage/nernst_factor
    lower = jnp.minimum(jnp.minimum(old_log, rest_log), equilibrium_log)
    upper = jnp.maximum(jnp.maximum(old_log, rest_log), equilibrium_log)
    a = 1.+dt/decay
    b = dt*flux_factor*(conductance*1e-3)*nernst_factor
    d = c_old+dt*c_rest/decay+b*equilibrium_log

    zero_cond_res = d / a

    def bisect_step(bounds, unused):
        lo, hi = bounds
        mid = lo + (hi - lo) * 0.5
        residual = a * jnp.exp(mid) + b * mid - d
        return (jnp.where(residual <= 0, mid, lo),
                jnp.where(residual > 0, mid, hi)), None

    (lo, hi), _ = jax.lax.scan(bisect_step, (lower, upper), xs=None, length=6)
    x = (lo + hi) * 0.5

    def halley_step(x, unused):
        ex = jnp.exp(x)
        fx = a * ex + b * x - d
        f1 = a * ex + b
        f2 = a * ex
        dx = (2.0 * fx * f1) / (2.0 * f1 * f1 - fx * f2 + 1e-30)
        return x - dx, None

    x_final, _ = jax.lax.scan(halley_step, x, xs=None, length=4)
    result = jnp.where(b <= 0.0, zero_cond_res, jnp.exp(x_final))
    return jnp.where(valid & jnp.isfinite(result) & (result > 0), result, jnp.nan)
