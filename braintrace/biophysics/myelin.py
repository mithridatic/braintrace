"""Explicit axon/sheath potassium exchange and fixed source pump laws.

Source pump equations: pinned BRAINCELL kpump.mod and nkpump.mod (see
BRAINCELL-LICENSE.txt). Spatial transport here is conservative finite volume,
not an exact transcription of the repository's RxD reaction discretization.
"""

import brainstate
import jax.numpy as jnp
import numpy as np

from .environment import ChemicalEnvironment, MembraneMap
from .shells import radial_shell_graph


def linear_pump_ma_cm2(inside_mm, *, resting_mm=110., maximum_ma_cm2=.1):
    """Evaluate the source linear outward K pump current density.

    Parameters
    ----------
    inside_mm : array-like
        Intracellular potassium in mM.
    resting_mm, maximum_ma_cm2 : float, optional
        Fixed source reference concentration and current scale.

    Returns
    -------
    array
        Signed outward current density in mA/cm^2. Invalid pools/rates give NaN.
    """
    ci = jnp.asarray(inside_mm)
    valid = (ci > 0) & jnp.isfinite(ci) & (resting_mm > 0) & jnp.isfinite(resting_mm)
    valid = valid & (maximum_ma_cm2 >= 0) & jnp.isfinite(maximum_ma_cm2)
    return jnp.where(valid, maximum_ma_cm2*(ci/resting_mm-1), jnp.nan)


def nak_pump_ma_cm2(voltage_mv, sodium_inside_mm, potassium_outside_mm, *,
                    temperature_c=34., maximum_ma_cm2=.06, km_sodium_mm=10., km_potassium_mm=2.5):
    """Evaluate source Na/K ATPase current with its 3:2 stoichiometry.

    Parameters
    ----------
    voltage_mv, sodium_inside_mm, potassium_outside_mm : array-like
        Local membrane voltage and concentrations.
    temperature_c, maximum_ma_cm2, km_sodium_mm, km_potassium_mm : float, optional
        Fixed source temperature/current/half-saturation parameters.

    Returns
    -------
    tuple
        Outward sodium and potassium current densities in mA/cm^2. Sodium is
        three pump cycles outward; potassium is two inward. A coupled driver
        must account for both species or explicitly declare a sodium reservoir.
    """
    v, na, k = map(jnp.asarray, (voltage_mv, sodium_inside_mm, potassium_outside_mm))
    valid = (na > 0) & (k > 0) & (v != -200.) & jnp.isfinite(v+na+k)
    valid = valid & (temperature_c > -273.15) & jnp.isfinite(temperature_c)
    valid = valid & (maximum_ma_cm2 >= 0) & (km_sodium_mm > 0) & (km_potassium_mm > 0)
    valid = valid & jnp.isfinite(maximum_ma_cm2+km_sodium_mm+km_potassium_mm)
    cycle = 3.**((temperature_c-37)/10)*maximum_ma_cm2*(v+150)/(v+200)
    cycle = cycle/((1+(km_sodium_mm/na)**1.5)*(1+km_potassium_mm/k))
    cycle = jnp.where(valid, cycle, jnp.nan)
    return 3*cycle, -2*cycle


class MyelinPotassium(brainstate.nn.Module):
    """Conserve radial potassium between explicit axon and sheath pools.

    Parameters
    ----------
    length_um, radii_um, diffusion_um2_ms : array-like
        Segment lengths, common annular boundaries and diffusivity per shell.
        First radius is the axon surface; outer zones can carry lower D.
    myelinated : array-like of bool
        Sealed outer boundary for myelinated segments; fixed bath otherwise.
    sheath_shell : int
        Zero-based shell adjacent to the sheath membrane.
    sheath_volumes_um3 : array-like
        Explicit intracellular sheath volumes, one per myelinated segment.
    inside_mm : array-like
        Initial K pools: all axon segments followed by myelinated sheath segments.
    dt_ms, bath_mm : float, optional
        Physical integration interval and fixed external bath concentration.

    Notes
    -----
    There is no longitudinal extracellular diffusion in this paper mode. This
    models the source's Schwann-like construction, not validated human cortical
    oligodendrocytes. Electrical membrane laws are supplied by coupled cables;
    this owner does not invent sheath channels or capacitance.
    """

    def __init__(self, length_um, radii_um, diffusion_um2_ms, myelinated,
                 sheath_shell, sheath_volumes_um3, inside_mm, *, dt_ms=.005, bath_mm=2.5):
        super().__init__()
        length, radii, mask = np.asarray(length_um), np.asarray(radii_um), np.asarray(myelinated)
        if (mask.dtype != np.bool_ or mask.shape != length.shape or type(sheath_shell) is not int or
                not 0 <= sheath_shell < len(radii)-1 or not len(radii) or radii[0] <= 0):
            raise ValueError('Invalid sheath mask, shell or axon radius')
        self.transport = radial_shell_graph(length, radii, diffusion_um2_ms, dt_ms=dt_ms,
                                            outer_reservoir=~mask, longitudinal=False)
        self.segment_count, self.shell_count = len(length), len(radii)-1
        sheath = np.flatnonzero(mask)
        volumes = np.asarray(sheath_volumes_um3)
        if volumes.shape != (len(sheath),):
            raise ValueError('Explicit sheath volumes must match myelinated segments')
        placements = np.r_[np.arange(len(length))*self.shell_count, sheath*self.shell_count+sheath_shell]
        intracellular = np.r_[np.pi*radii[0]**2*length, volumes]
        mapping = MembraneMap(placements, intracellular, len(self.transport.volumes))
        self.environment = ChemicalEnvironment(self.transport, self.transport, mapping, inside_mm,
                                               outside_potassium_mm=bath_mm)
        self.sheath_segments = tuple(sheath.tolist())

    def update(self, axon_outward_na, sheath_outward_na):
        """Advance axon and sheath exchange exactly once per physical interval.

        Parameters
        ----------
        axon_outward_na, sheath_outward_na : arrays
            K-only outward total currents in nA in their declared segment order.

        Returns
        -------
        array
            Extracellular mM potassium, shaped (segment, radial_shell).
        """
        axon, sheath = jnp.asarray(axon_outward_na), jnp.asarray(sheath_outward_na)
        if axon.shape != (self.segment_count,) or sheath.shape != (len(self.sheath_segments),):
            raise ValueError('Currents must match axon and sheath segment counts')
        k, _ = self.environment.update(jnp.concatenate((axon, sheath)), jnp.zeros_like(self.environment.gaba.value))
        return k.reshape(self.segment_count, self.shell_count)

    def reset_state(self):
        """Reset concentrations, exchange ledgers and physical clock."""
        self.environment.reset_state()
