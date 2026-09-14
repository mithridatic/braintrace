"""Shared conservative extracellular potassium and molecule-injection GABA."""

import brainstate
import jax.numpy as jnp
import numpy as np

from .extracellular import MOLECULES_PER_MM_UM3, nernst_mv, outward_current_amount_rate
from .astrocyte import kir41_current_ma_cm2


class MembraneMap:
    """Map each physical membrane compartment to one extracellular volume.

    Parameters
    ----------
    indices : array-like
        Extracellular index for each membrane compartment; repeats are allowed.
    inside_volumes : array-like
        Positive intracellular volumes in um^3, one per membrane compartment.
    outside_count : int
        Number of extracellular volumes.

    Notes
    -----
    This is a nearest-volume map. Geometry builders must document its spatial
    resolution and qualify it under refinement before physiological use.
    """

    def __init__(self, indices, inside_volumes, outside_count):
        idx, volumes = np.asarray(indices), np.asarray(inside_volumes, dtype=float)
        if (type(outside_count) is not int or outside_count <= 0 or idx.ndim != 1 or
                not np.issubdtype(idx.dtype, np.integer) or np.any(idx < 0) or
                np.any(idx >= outside_count) or volumes.shape != idx.shape or
                not np.isfinite(volumes).all() or np.any(volumes <= 0)):
            raise ValueError('Invalid membrane-to-volume map')
        self.indices = jnp.asarray(idx, dtype=jnp.int32)
        self.inside_volumes = jnp.asarray(volumes)
        self.outside_count = outside_count

    def scatter(self, rates):
        """Sum compartment amount rates into extracellular volumes.

        Parameters
        ----------
        rates : array
            Signed outward amounts in mM um^3/ms per membrane compartment.

        Returns
        -------
        array
            Summed rates per extracellular volume.
        """
        rates = jnp.asarray(rates)
        if rates.shape != self.indices.shape:
            raise ValueError('Membrane rates must match the placement map')
        return jnp.zeros(self.outside_count, dtype=rates.dtype).at[self.indices].add(rates)


class ChemicalEnvironment(brainstate.nn.Module):
    """Own shared extracellular concentrations and intracellular potassium.

    Parameters
    ----------
    potassium_transport, gaba_transport : DiffusionGraph
        Separate diffusivities/uptake on identical extracellular volumes and dt.
    membranes : MembraneMap
        Joint neuronal and astrocyte membrane-to-volume mapping.
    inside_potassium_mm : array-like
        Initial potassium per mapped intracellular compartment.
    outside_potassium_mm : float, optional
        Initial concentration and fixed boundary bath; default 2.5 mM.

    Notes
    -----
    Outward currents are physical total ionic currents, never total membrane
    current. They are debited from the matching intracellular pool exactly once.
    GABA release is a molecule impulse, distinct from BRAINCELL's concentration
    boundary-source history kernel. Uptake is a recorded removed pool.
    """

    def __init__(self, potassium_transport, gaba_transport, membranes,
                 inside_potassium_mm, *, outside_potassium_mm=2.5):
        super().__init__()
        k, g = potassium_transport, gaba_transport
        inside = np.asarray(inside_potassium_mm, dtype=float)
        if (k.dt_ms != g.dt_ms or not np.array_equal(k.volumes, g.volumes) or
                len(k.volumes) != membranes.outside_count):
            raise ValueError('Chemical species must share geometry and time interval')
        if (inside.shape != membranes.indices.shape or not np.isfinite(inside).all() or
                np.any(inside <= 0) or not np.isfinite(outside_potassium_mm) or outside_potassium_mm <= 0):
            raise ValueError('Initial potassium must be positive and match membranes')
        if np.any(np.asarray(k.uptake) != 0):
            raise ValueError('Potassium uptake needs an explicit receiving intracellular pool')
        self.k_transport, self.gaba_transport, self.membranes = k, g, membranes
        self.dt_ms = k.dt_ms
        self.initial_inside = jnp.asarray(inside)
        self.bath = jnp.full_like(k.volumes, outside_potassium_mm)
        self.potassium = brainstate.HiddenState(self.bath)
        self.inside_potassium = brainstate.HiddenState(self.initial_inside)
        self.gaba = brainstate.HiddenState(jnp.zeros_like(g.volumes))
        self.k_boundary_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.gaba_boundary_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.gaba_removed_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.gaba_released_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.balance_error = brainstate.ShortTermState(jnp.zeros(2))
        self.valid = brainstate.ShortTermState(jnp.asarray(True))
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def reversal_mv(self, *, temperature_c=34.):
        """Return dynamic K reversal per membrane compartment.

        Parameters
        ----------
        temperature_c : float, optional
            Temperature in Celsius, default 34.

        Returns
        -------
        array
            Nernst voltage in mV from both current concentration pools.
        """
        return nernst_mv(self.inside_potassium.value,
                        self.potassium.value[self.membranes.indices], temperature_c=temperature_c)

    def update(self, outward_potassium_na, gaba_molecules):
        """Advance one physical interval and retain transport/exchange ledgers.

        Parameters
        ----------
        outward_potassium_na : array
            Outward-positive K-only current per mapped membrane in nA.
        gaba_molecules : array
            Nonnegative molecule release per extracellular volume this interval.

        Returns
        -------
        tuple
            Potassium and GABA concentrations, both in mM. Check ``valid``.
        """
        amount = outward_current_amount_rate(outward_potassium_na)
        source = self.membranes.scatter(amount)
        molecules = jnp.asarray(gaba_molecules)
        if molecules.shape != self.gaba.value.shape:
            raise ValueError('GABA release must match extracellular volumes')
        injected = molecules/MOLECULES_PER_MM_UM3
        k = self.k_transport.step(self.potassium.value, source_amount_per_ms=source, reservoir_mm=self.bath)
        g = self.gaba_transport.step(self.gaba.value, source_amount_per_ms=injected/self.dt_ms)
        old_inside = self.inside_potassium.value
        inside = old_inside-self.dt_ms*amount/self.membranes.inside_volumes
        exchange_error = jnp.sum((inside-old_inside)*self.membranes.inside_volumes)+self.dt_ms*jnp.sum(source)
        self.potassium.value, self.gaba.value = k.concentration, g.concentration
        self.inside_potassium.value = inside
        self.k_boundary_amount.value = self.k_boundary_amount.value+k.boundary_amount
        self.gaba_boundary_amount.value = self.gaba_boundary_amount.value+g.boundary_amount
        self.gaba_removed_amount.value = self.gaba_removed_amount.value+g.removed_amount
        self.gaba_released_amount.value = self.gaba_released_amount.value+jnp.sum(injected)
        self.balance_error.value = self.balance_error.value+jnp.stack((k.balance_error+exchange_error, g.balance_error))
        self.valid.value = (self.valid.value & k.valid & g.valid & jnp.all(inside > 0) &
                            jnp.all(jnp.isfinite(inside)) & jnp.all(molecules >= 0))
        self.tick.value = self.tick.value+1
        return k.concentration, g.concentration

    def reset_state(self):
        """Restore both species, intracellular pools, all ledgers and clock."""
        self.potassium.value = self.bath
        self.inside_potassium.value = self.initial_inside
        self.gaba.value = jnp.zeros_like(self.gaba.value)
        for state in (self.k_boundary_amount, self.gaba_boundary_amount, self.gaba_removed_amount,
                      self.gaba_released_amount, self.balance_error, self.tick):
            state.value = jnp.zeros_like(state.value)
        self.valid.value = jnp.asarray(True)


def kir41_current_na(voltage_mv, reversal_mv, outside_mm, membrane_area_um2,
                     *, conductance_ms_cm2=.4):
    """Convert source Kir4.1 current density to outward total current.

    Parameters
    ----------
    voltage_mv, reversal_mv, outside_mm : array-like
        Local membrane voltage, K reversal and extracellular concentration.
    membrane_area_um2 : array-like
        Positive astrocyte membrane areas in um^2.
    conductance_ms_cm2 : float, optional
        Fixed Kir4.1 density, default .4 mS/cm^2.

    Returns
    -------
    array
        Outward-positive nA; invalid areas return NaN.
    """
    area = jnp.asarray(membrane_area_um2)
    current = kir41_current_ma_cm2(voltage_mv, reversal_mv, outside_mm,
                                   conductance_ms_cm2=conductance_ms_cm2)*area*.01
    return jnp.where((area > 0) & jnp.isfinite(area), current, jnp.nan)
