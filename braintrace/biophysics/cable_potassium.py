"""BrainCell potassium ion whose pools are owned by a shared environment."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from braincell.ion import PotassiumFixed
from braincell.mech import register_ion

from .extracellular import nernst_mv


@register_ion('EnvironmentPotassium')
class EnvironmentPotassium(PotassiumFixed):
    """Use externally integrated concentrations for channel reversal.

    Parameters
    ----------
    size, E, Ci, Co, valence, name, **channels
        Same declarations as BrainCell PotassiumFixed. Before environment
        binding, the declared values support ordinary runtime construction.

    Notes
    -----
    Bind before tracing any cable steps. Concentrations advance only in the
    chemical environment, avoiding a second ion integrator for the same pool.
    """

    @property
    def Ci(self):
        """Intracellular K concentration at runtime points, in mM."""
        binding = getattr(self, '_environment_binding', None)
        return self._initial_ci if binding is None else binding.point_inside()

    @Ci.setter
    def Ci(self, value):
        self._initial_ci = value

    @property
    def Co(self):
        """Extracellular K concentration at runtime points, in mM."""
        binding = getattr(self, '_environment_binding', None)
        return self._initial_co if binding is None else binding.point_outside()

    @Co.setter
    def Co(self, value):
        self._initial_co = value

    @property
    def E(self):
        """Nernst voltage from current pools once bound, in mV."""
        binding = getattr(self, '_environment_binding', None)
        if binding is None:
            return self._initial_e
        return nernst_mv(self.Ci.to_decimal(u.mM), self.Co.to_decimal(u.mM),
                        temperature_c=binding.temperature_c)*u.mV

    @E.setter
    def E(self, value):
        self._initial_e = value


class CablePotassiumBinding:
    """Bind a one-cell BrainCell population to contiguous intracellular pools.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized cell painted with exactly one EnvironmentPotassium ion.
    environment : ChemicalEnvironment
        Shared potassium and GABA state owner.
    offset : int, optional
        First intracellular compartment index; cell CV ordering is preserved.
    temperature_c : float, optional
        Fixed bath temperature, default 34 C.

    Notes
    -----
    Current uses CV midpoint density and each CV membrane area exactly once.
    Endpoint interpolation nodes must never contribute another membrane area.
    """

    def __init__(self, cell, environment, *, offset=0, temperature_c=34.):
        runtime = getattr(cell, '_runtime', None)
        if runtime is None or tuple(cell.pop_size) not in ((), (1,)):
            raise ValueError('Binding requires an initialized single-cell population')
        ions = [ion for ion in runtime.ions.values() if isinstance(ion, EnvironmentPotassium)]
        if len(ions) != 1:
            raise ValueError('Paint exactly one EnvironmentPotassium ion before initialization')
        if (type(offset) is not int or offset < 0 or offset+runtime.n_cv > len(environment.initial_inside)
                or not np.isfinite(temperature_c) or temperature_c <= -273.15):
            raise ValueError('Invalid intracellular pool range or temperature')
        if getattr(ions[0], '_environment_binding', None) is not None:
            raise ValueError('Potassium ion is already bound')
        self.cell, self.environment, self.ion = cell, environment, ions[0]
        self.indices = jnp.arange(offset, offset+runtime.n_cv)
        self.temperature_c = temperature_c
        self.midpoints = jnp.asarray(runtime.node_tree.cv_to_mid_node_id, dtype=jnp.int32)
        self.area_cm2 = jnp.asarray(runtime.cv_area.to_decimal(u.cm**2))
        # Voltage's CV-to-point bridge zeros boundary nodes. Concentrations
        # must instead retain the owning CV's positive physical pools there.
        self.point_cv = jnp.asarray([point.roles[0].cv_id for point in runtime.node_tree.nodes], dtype=jnp.int32)
        self.ion._environment_binding = self

    def point_inside(self):
        """Gather intracellular concentrations from each point's owning CV."""
        values = self.environment.inside_potassium.value[self.indices]
        return values[self.point_cv].reshape(self.ion.varshape)*u.mM

    def point_outside(self):
        """Gather extracellular concentrations from each point's owning CV."""
        outside = self.environment.membranes.indices[self.indices]
        values = self.environment.potassium.value[outside]
        return values[self.point_cv].reshape(self.ion.varshape)*u.mM

    def outward_current_na(self):
        """Return step-start K-only outward current per CV, in nA.

        Returns
        -------
        array
            Total ionic current. BrainCell current is inward-positive, so this
            conversion negates its mA/cm^2 value and multiplies area and 1e6.
        """
        voltage = self.cell._cv_to_point_unchecked(self.cell.V.value)
        density = self.ion.current(voltage, include_external=True).to_decimal(u.mA/u.cm**2)
        return -density[..., self.midpoints].reshape(-1)*self.area_cm2*1e6


class PotassiumCoupling(brainstate.nn.Module):
    """Advance a shared environment from supported neuronal K bindings.

    Parameters
    ----------
    environment : ChemicalEnvironment
        Chemical state owner.
    bindings : sequence of CablePotassiumBinding
        Each intracellular compartment must occur exactly once. Astrocytes or
        other membrane sources need an explicit separate coupled driver.
    """

    def __init__(self, environment, bindings):
        super().__init__()
        indices = np.concatenate([np.asarray(b.indices) for b in bindings]) if bindings else np.array([], int)
        if (any(b.environment is not environment for b in bindings) or
                not np.array_equal(np.sort(indices), np.arange(len(environment.initial_inside)))):
            raise ValueError('Bindings must cover intracellular compartments exactly once')
        self.environment = environment
        self.bindings = tuple(bindings)

    def currents(self):
        """Sample old-state outward K current once before the cable advances."""
        result = jnp.zeros_like(self.environment.inside_potassium.value)
        for binding in self.bindings:
            result = result.at[binding.indices].set(binding.outward_current_na())
        return result

    def update(self, current, gaba_molecules=None):
        """Apply the sampled current and explicit GABA release for this tick.

        Parameters
        ----------
        current : array
            Output of ``currents`` captured before the cable update.
        gaba_molecules : array or None, optional
            Molecule release per extracellular volume; defaults to zero.

        Returns
        -------
        tuple
            Updated extracellular K/GABA concentrations in mM.
        """
        release = jnp.zeros_like(self.environment.gaba.value) if gaba_molecules is None else gaba_molecules
        return self.environment.update(current, release)

    def reset_state(self):
        """Reset the shared chemical owner; binding geometry stays fixed."""
        self.environment.reset_state()
