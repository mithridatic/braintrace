"""Explicit source-history diffusion and ion conversion kernels.

The history kernel follows RusakovLab/BRAINCELL CommonBackend.py at
8acd1c23032a17af59ae1f813d142735fef8689c (BSD-3-Clause; see the pinned reference
license). It is a concentration boundary-source model, not a molecule impulse.
"""

import jax.numpy as jnp
from jax.scipy.special import erfc
import numpy as np

MOLECULES_PER_MM_UM3 = 6.02214076e5


def nernst_mv(inside_mm, outside_mm, *, temperature_c=34., valence=1.):
    """Compute Nernst reversal voltage; nonpositive concentrations produce NaN.

    Parameters
    ----------
    inside_mm, outside_mm : array-like
        Intracellular and extracellular concentrations in mM.
    temperature_c : float, optional
        Temperature in Celsius.
    valence : float, optional
        Nonzero ionic charge number.

    Returns
    -------
    array
        Reversal voltage in mV, without concentration clipping.
    """
    ci, co = jnp.asarray(inside_mm), jnp.asarray(outside_mm)
    valence = jnp.asarray(valence)
    valid = (ci > 0) & (co > 0) & jnp.isfinite(ci) & jnp.isfinite(co)
    valid = valid & (temperature_c > -273.15) & jnp.isfinite(temperature_c) & (valence != 0) & jnp.isfinite(valence)
    return jnp.where(valid, 8.31446261815324*(temperature_c+273.15)/96.48533212/valence*jnp.log(co/ci), jnp.nan)


def outward_current_amount_rate(current_na, *, valence=1.):
    """Convert outward current to outward ion amount rate.

    Parameters
    ----------
    current_na : array-like
        Outward-positive total ionic current in nA.
    valence : float, optional
        Nonzero ionic charge number.

    Returns
    -------
    array
        Outward mM um^3/ms. Apply opposite signs to the two reservoirs.
    """
    return jnp.where((valence != 0) & jnp.isfinite(valence), jnp.asarray(current_na)*1e6/(96485.33212*valence), jnp.nan)


class SourceHistory:
    """Reference-compatible extracellular concentration source geometry.

    Parameters
    ----------
    source_xyz, destination_xyz : array-like
        Source and observation coordinates in um, with shape (n, 3).
    diameter_um : array-like
        Positive source diameters in um.
    diffusion_um2_ms : float, optional
        Apparent diffusion coefficient including tortuosity; default 0.3.
    uptake_ms : float or None, optional
        First-order clearance time; default 600 ms, None disables clearance.

    Notes
    -----
    Use on bounded reference cases: work is proportional to source count,
    observation count and history length. No history truncation is implicit.
    """

    def __init__(self, source_xyz, destination_xyz, diameter_um,
                 *, diffusion_um2_ms=.3, uptake_ms=600.):
        src, dst, diameter = map(lambda x: np.asarray(x, dtype=float),
                                (source_xyz, destination_xyz, diameter_um))
        if (src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3 or
                diameter.shape != (len(src),) or not np.isfinite(src).all() or
                not np.isfinite(dst).all() or not np.isfinite(diameter).all() or np.any(diameter <= 0)):
            raise ValueError("Finite xyz coordinates and one positive diameter per source required")
        if not np.isfinite(diffusion_um2_ms) or diffusion_um2_ms <= 0:
            raise ValueError("Diffusion must be positive and finite")
        if uptake_ms is not None and (not np.isfinite(uptake_ms) or uptake_ms <= 0):
            raise ValueError("Uptake time must be positive and finite")
        self.distance = jnp.asarray(np.linalg.norm(dst[:, None, :] - src[None, :, :], axis=-1))
        self.geometry = jnp.asarray(diameter)/(jnp.asarray(diameter)+self.distance+1e-8)
        self.diffusion = diffusion_um2_ms
        self.uptake = uptake_ms

    def sample(self, time_ms, interval_bounds_ms, concentrations_mm, *, baseline_mm=0.):
        """Sample piecewise-constant source contributions without state clipping.

        Parameters
        ----------
        time_ms : scalar
            Observation time in ms.
        interval_bounds_ms : array
            Strictly increasing interval bounds, validated by the caller.
        concentrations_mm : array
            Source concentrations, shape (n_intervals, n_sources).
        baseline_mm : scalar, optional
            Constant concentration subtracted from each source interval.

        Returns
        -------
        array
            Concentration increments at the observation sites. Baseline reservoir
            evolution is separate and must be added by the caller if required.
        """
        bounds, values = jnp.asarray(interval_bounds_ms), jnp.asarray(concentrations_mm)
        if bounds.ndim != 1 or values.shape != (len(bounds)-1, self.distance.shape[1]):
            raise ValueError("History must match interval and source counts")
        def kernel(ages):
            safe_age = jnp.where(ages > 0, ages, 1.)
            term = erfc(self.distance[..., None]/jnp.sqrt(4*self.diffusion*safe_age))
            if self.uptake is not None:
                term = term*jnp.exp(-safe_age/self.uptake)
            return jnp.where(ages > 0, term, 0.)
        response = kernel(time_ms-bounds[:-1])-kernel(time_ms-bounds[1:])
        result = jnp.sum(self.geometry[..., None]*response*(values-baseline_mm).T, axis=(1, 2))
        valid = (jnp.all(jnp.isfinite(bounds)) & jnp.all(jnp.diff(bounds) > 0) &
                 jnp.all(jnp.isfinite(values)) & jnp.isfinite(time_ms) & jnp.isfinite(baseline_mm))
        return jnp.where(valid, result, jnp.nan)
