"""Astrocyte calcium reaction equations translated from BRAINCELL cadifus.mod.

Reference revision 8acd1c23032a17af59ae1f813d142735fef8689c; BSD-3-Clause,
RusakovLab/BRAINCELL. Pinned source and license: docs/biology/reference.
The ER concentration is a fixed reservoir, not a conserved dynamic ER store.
"""

from dataclasses import dataclass

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

@dataclass(frozen=True)
class CalciumParameters:
    """Fixed rates for the paper's astrocyte calcium experiment.

    Parameters
    ----------
    stationary_mm, mobile_mm : float
        Total initial stationary and mobile buffer concentrations in mM.
    resting_mm, ip3_resting_mm : float
        Initial free calcium and IP3 in mM.
    diffusion_um2_ms, mobile_diffusion_um2_ms : float
        Calcium and mobile buffer diffusivity in um^2/ms.
    pump_threshold_mm, pump_velocity_um_ms : float
        Plasma pump threshold and surface-transfer velocity.
    er_mm, alpha : float
        Fixed ER reservoir concentration and relative ER mechanism abundance.

    Notes
    -----
    Defaults follow paper methods, not unmodified repository defaults. Binding
    on-rates are 1000/(mM ms), dissociation constants .01/.00024 mM; SERCA vmax
    is 3.75e-6 mM/ms. Geometry-specific source-current coupling is external.
    """

    stationary_mm: float = .07
    mobile_mm: float = .0075
    resting_mm: float = .00005
    ip3_resting_mm: float = .00005
    diffusion_um2_ms: float = .3
    mobile_diffusion_um2_ms: float = .05
    pump_threshold_mm: float = .00002
    pump_velocity_um_ms: float = .008
    er_mm: float = .4
    alpha: float = 1.

    def __post_init__(self):
        values = np.asarray(tuple(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values < 0) or self.resting_mm <= 0 or self.er_mm <= self.resting_mm:
            raise ValueError("Calcium parameters must be finite, nonnegative and have ER above resting calcium")


def kir41_current_ma_cm2(voltage_mv, reversal_mv, outside_mm, *, conductance_ms_cm2=.4):
    """Evaluate the pinned Kir4.1 outward current density.

    Parameters
    ----------
    voltage_mv, reversal_mv : array-like
        Membrane and potassium reversal voltages in mV.
    outside_mm : array-like
        Extracellular potassium in mM.
    conductance_ms_cm2 : float, optional
        Fixed conductance density in mS/cm^2; paper value 0.4.

    Returns
    -------
    array
        Outward current in mA/cm^2, including source NormK=0.81.
    """
    v, ek, ko = map(jnp.asarray, (voltage_mv, reversal_mv, outside_mm))
    rectifier = jax.nn.sigmoid(-(v-.81*ek+105.82)/19.23)
    current = .001*conductance_ms_cm2*(v-.81*ek+14.83)*jnp.sqrt(ko*rectifier)
    return jnp.where((ko > 0) & (conductance_ms_cm2 >= 0), current, jnp.nan)


def initial_calcium(parameters):
    """Return one segment's 25-state shell vector at source equilibrium.

    Parameters
    ----------
    parameters : CalciumParameters
        Fixed physical rates and initial conditions.

    Returns
    -------
    array
        Four each of Ca, free/bound stationary buffer, free/bound mobile buffer,
        open-able IP3R gate, followed by one shared segment IP3 concentration.
    """
    p = parameters
    fs = p.stationary_mm*.01/(.01+p.resting_mm)
    fm = p.mobile_mm*.00024/(.00024+p.resting_mm)
    values = [p.resting_mm, fs, p.stationary_mm-fs, fm,
              p.mobile_mm-fm, .0002/(p.resting_mm+.0002)]
    return jnp.concatenate((jnp.repeat(jnp.asarray(values), 4), jnp.asarray([p.ip3_resting_mm])))


def calcium_derivative(state, parameters, *, surface_to_volume=0., glutamate_mm=0.,
                       influx_mm_ms=None, clamped_ip3_mm=None):
    """Evaluate local buffer, IP3 and ER reactions without spatial transport.

    Parameters
    ----------
    state : array
        One 25-state segment vector, as returned by ``initial_calcium``.
    parameters : CalciumParameters
        Fixed rates; only state and external inputs are differentiated.
    surface_to_volume : float, optional
        Outer membrane area / outer shell volume, in 1/um.
    glutamate_mm : scalar, optional
        Concentration driving the source PLC-beta pathway.
    influx_mm_ms : array, optional
        External calcium influx per shell in mM/ms, excluding this pump.
    clamped_ip3_mm : scalar or None, optional
        Explicit experimental IP3 clamp; None evolves the source IP3 equation.

    Returns
    -------
    tuple
        Derivative, signed ER-to-cytoplasm rate per shell, and outward plasma
        pump concentration rate in the outer shell (mM/ms).
    """
    p = parameters
    ca, fs, bs, fm, bm, gate = state[:24].reshape(6, 4)
    ip3 = state[24] if clamped_ip3_mm is None else clamped_ip3_mm
    safe = jnp.maximum(ca, 1e-9)  # Source denominator/rate protection, not state clipping.
    stationary = 1000*ca*fs-10*bs
    mobile = 1000*ca*fm-.24*bm
    activation = (ip3/(ip3+.0008))*(safe/(safe+.0003))*gate
    channel = .0035*(1-safe/p.er_mm)*activation**3
    pump_er = 3.75e-6*safe**2/(safe**2+.00027**2)
    c0, i0 = p.resting_mm, p.ip3_resting_mm
    a0 = i0/(i0+.0008)*c0/(c0+.0003)*(.0002/(c0+.0002))
    leak = (3.75e-6*c0**2/(c0**2+.00027**2)-.0035*(1-c0/p.er_mm)*a0**3)/(1-c0/p.er_mm)
    er = p.alpha*(channel-pump_er+leak*(1-safe/p.er_mm))
    pump = jnp.where(ca[0] > p.pump_threshold_mm,
        p.pump_velocity_um_ms*surface_to_volume*(ca[0]-p.pump_threshold_mm), 0.)
    influx = jnp.zeros(4) if influx_mm_ms is None else jnp.asarray(influx_mm_ms)
    dc = (er-stationary-mobile+influx).at[0].add(-pump)
    dh = 2.7*(.0002-(safe+.0002)*gate)
    kg = .0013+.01*safe[0]/(safe[0]+.0006)
    vg = .2e-6*glutamate_mm/(glutamate_mm+kg)
    vd = .02e-6/(1+ip3/.0015)*safe[0]**2/(safe[0]**2+.0001**2)
    vk = 2e-6*safe[0]**4/(safe[0]**4+.7**4)*ip3/(ip3+.001)
    # The source flux/COMPARTMENT area ratio is two, not one.
    di = 2*(vg+vd-vk-.04e-3*ip3) if clamped_ip3_mm is None else jnp.asarray(0.)
    derivative = jnp.concatenate((jnp.stack((dc, -stationary, stationary, -mobile, mobile, dh)).reshape(-1), jnp.atleast_1d(di)))
    return derivative, er, pump


def calcium_reaction_step(state, parameters, *, dt_ms, surface_to_volume=0.,
                          glutamate_mm=0., influx_mm_ms=None, clamped_ip3_mm=None):
    """Implicitly advance one segment's local reactions and report validity.

    Parameters
    ----------
    state : array
        One physical segment's 25 states.
    parameters : CalciumParameters
        Immutable biological parameters.
    dt_ms : float
        Positive interval, validated by the caller.
    surface_to_volume, glutamate_mm, influx_mm_ms, clamped_ip3_mm : optional
        Inputs documented in ``calcium_derivative``.

    Returns
    -------
    tuple
        Updated state, validity flag and maximum equation residual. The local
        25-by-25 solve is independent of the number of cable compartments.
    """
    state = jnp.asarray(state)
    if clamped_ip3_mm is not None:
        state = state.at[24].set(clamped_ip3_mm)

    def equation(y):
        rate = calcium_derivative(y, parameters, surface_to_volume=surface_to_volume,
            glutamate_mm=glutamate_mm, influx_mm_ms=influx_mm_ms, clamped_ip3_mm=clamped_ip3_mm)[0]
        return y-state-dt_ms*rate

    def solve(f, initial):
        def unfinished(carry):
            _, iteration, error = carry
            return (iteration < 12) & (error > 1e-13)
        def step(carry):
            y, iteration, _ = carry
            delta = jnp.linalg.solve(jax.jacfwd(f)(y), f(y))
            updated = y-delta
            return updated, iteration+1, jnp.max(jnp.abs(f(updated)))
        initial_error = jnp.max(jnp.abs(f(initial)))
        return brainstate.transform.while_loop(unfinished, step,
            (initial, jnp.asarray(0, dtype=jnp.int32), initial_error))[0]

    def tangent(g, value):
        return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(value)), value)

    updated = jax.lax.custom_root(equation, state, solve, tangent)
    residual = jnp.max(jnp.abs(equation(updated)))
    valid = (jnp.all(jnp.isfinite(updated)) & jnp.all(updated >= 0) &
             jnp.all(updated[20:24] <= 1) & (residual < 1e-10) & (dt_ms > 0))
    return updated, valid, residual
