"""Physical coupling of neuronal spikes, glial cables and astrocyte calcium."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from braincell.network.delivery import population_spike

from .cable_potassium import PotassiumCoupling


class NeuroGlialCoupling(brainstate.nn.Module):
    """Share one chemical clock and complete K pools across neurons and glia.

    Parameters
    ----------
    environment : ChemicalEnvironment
        Joint extracellular domain and intracellular K owner.
    neuron_bindings, astrocyte_bindings : sequence of CablePotassiumBinding
        Initialized distinct cables covering every intracellular pool once.
        Neuronal bindings alone define the event-source and ARC domain.
    neuron_roles : sequence of str
        Explicit E/I roles in neuronal binding order.
    glutamate : TransmitterField
        Explicit glutamate kinetics on identical extracellular volumes and dt.
    calcium : AstrocyteCalcium
        One segment per glial CV, in concatenated glial binding order.
    calcium_volume_indices : array-like
        Extracellular indices matching the glial K membrane mapping.
    gaba_release, glutamate_release : release-site owner or None
        Event-driven sources, with fixed site geometry/dose and RNG state.
    gaba_sources, glutamate_sources : sequence of int
        Distinct neuronal binding indices, matching each release-owner row.
        GABA requires I sources, glutamate E sources. None requires no rows.
    basis : str
        Explicit anatomy and biological assumptions for this assembly.

    Notes
    -----
    currents() runs before neuronal dynamics. update() reads end-of-step
    neuronal spikes, advances glial cables, applies the saved K exchanges,
    then advances transmitter diffusion and calcium. New fields feed the next
    neuronal tick. Calcium membrane charge feedback is not implemented here.
    This driver is not a complete astrocyte physiology or myelin model.
    """

    def __init__(self, environment, neuron_bindings, astrocyte_bindings, neuron_roles,
                 glutamate, calcium, calcium_volume_indices, *, gaba_release, glutamate_release,
                 gaba_sources, glutamate_sources, basis):
        super().__init__()
        neurons, astro = tuple(neuron_bindings), tuple(astrocyte_bindings)
        if not neurons or not astro or len({id(b.cell) for b in neurons+astro}) != len(neurons+astro):
            raise ValueError('Distinct neuronal and glial cables are required')
        if any(not jnp.issubdtype(b.cell.V.value.dtype, jnp.floating) for b in neurons+astro):
            raise ValueError('Coupled cable voltage states must use floating point initial values')
        roles = tuple(neuron_roles)
        if len(roles) != len(neurons) or any(role not in ('E', 'I') for role in roles):
            raise ValueError('Explicit E/I roles must match neuronal bindings')
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError('Explicit anatomy and biological basis required')
        self.potassium_driver = PotassiumCoupling(environment, neurons+astro)
        if (glutamate.transport.dt_ms != environment.dt_ms or calcium.dt_ms != environment.dt_ms or
                not np.array_equal(glutamate.transport.volumes, environment.k_transport.volumes)):
            raise ValueError('Coupled species require identical volumes and clocks')
        ids = np.asarray(calcium_volume_indices)
        pools = np.concatenate([np.asarray(binding.indices) for binding in astro])
        expected = np.asarray(environment.membranes.indices)[pools]
        if (ids.shape != expected.shape or not np.issubdtype(ids.dtype, np.integer) or
                not np.array_equal(ids, expected) or len(calcium.initial) != len(pools)):
            raise ValueError('Calcium segments must match the glial membrane map')
        ca_volumes = np.asarray(calcium.calcium_transport.volumes).reshape(-1, 4).sum(axis=1)
        if not np.allclose(ca_volumes, np.asarray(environment.membranes.inside_volumes)[pools], rtol=1e-10, atol=1e-12):
            raise ValueError('Calcium and glial K intracellular volumes differ')
        self.gaba_sources = self._sources(gaba_sources, gaba_release, roles, 'I', environment.membranes.outside_count)
        self.glutamate_sources = self._sources(glutamate_sources, glutamate_release, roles, 'E', environment.membranes.outside_count)
        self.environment, self.bindings = environment, neurons
        self.astrocytes = tuple(binding.cell for binding in astro)
        self.glutamate, self.calcium = glutamate, calcium
        self.calcium_volume_indices = jnp.asarray(ids, dtype=jnp.int32)
        self.gaba_release, self.glutamate_release = gaba_release, glutamate_release
        self.basis, self.neuron_roles = basis, roles
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))
        self.valid = brainstate.ShortTermState(jnp.asarray(True))

    @staticmethod
    def _sources(indices, release, roles, role, outside_count):
        ids = tuple(indices)
        if (any(type(i) is not int or not 0 <= i < len(roles) for i in ids) or
                len(ids) != len(set(ids)) or any(roles[i] != role for i in ids)):
            raise ValueError('Release source indices must match distinct declared E/I roles')
        if release is None:
            if ids:
                raise ValueError('Source indices require a release owner')
        elif release.volume_indices.shape[0] != len(ids) or release.outside_count != outside_count:
            raise ValueError('Release site rows and volumes must match source maps')
        return ids

    def currents(self):
        """Sample old-state outward K currents for every neuronal/glial pool.

        Returns
        -------
        array
            Total outward nA per intracellular pool, captured exactly once.
        """
        return self.potassium_driver.currents()

    def update(self, current):
        """Advance glia and chemistry after the neuronal cable step.

        Parameters
        ----------
        current : array
            Previously sampled neuronal and glial K currents in nA.

        Returns
        -------
        tuple
            Extracellular K, GABA and glutamate concentrations in mM.
        """
        spikes = jnp.stack([population_spike(binding.cell.spike.value).reshape(()) for binding in self.bindings])
        zero = jnp.zeros_like(self.environment.gaba.value)
        gaba = zero if self.gaba_release is None else self.gaba_release(spikes[jnp.asarray(self.gaba_sources, dtype=jnp.int32)])
        glu = zero if self.glutamate_release is None else self.glutamate_release(spikes[jnp.asarray(self.glutamate_sources, dtype=jnp.int32)])
        for cell in self.astrocytes:
            cell._update_dynamics()
        k, g = self.environment.update(current, gaba)
        glutamate = self.glutamate.update(glu)
        self.calcium.update(glutamate[self.calcium_volume_indices])
        self.tick.value += 1
        finite = jnp.asarray(True)
        for cell in self.astrocytes:
            cell._set_current_time(self.tick.value*self.environment.dt_ms*u.ms)
            finite = finite & jnp.all(jnp.isfinite(cell.V.value.to_decimal(u.mV)))
        self.valid.value = self.valid.value & self.environment.valid.value & self.glutamate.valid.value & self.calcium.valid.value & finite
        return k, g, glutamate

    def reset_state(self):
        """Reset all glial, chemical, release and validity state together."""
        self.environment.reset_state()
        self.glutamate.reset_state()
        self.calcium.reset_state()
        for release in (self.gaba_release, self.glutamate_release):
            if release is not None:
                release.reset_state()
        for cell in self.astrocytes:
            cell.reset_state()
        self.tick.value = jnp.zeros_like(self.tick.value)
        self.valid.value = jnp.asarray(True)
