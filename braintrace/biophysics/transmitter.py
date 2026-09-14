"""Conserved released transmitter fields with explicitly supplied kinetics."""

import brainstate
import jax.numpy as jnp

from .extracellular import MOLECULES_PER_MM_UM3
from .gaba import GabaReleaseSites


class GlutamateReleaseSites(GabaReleaseSites):
    """Reuse site sampling with explicit glutamate dose and site count.

    Parameters
    ----------
    volume_indices : array-like
        Extracellular indices for each source's distinct release sites.
    outside_count : int
        Number of extracellular volumes.
    active_sites : int
        Selected sites per source spike; no inferred biological default.
    molecules_per_site : float
        Explicit molecule dose; not inferred from a bath concentration.
    seed : int, optional
        BrainState random stream seed.
    """

    def __init__(self, volume_indices, outside_count, *, active_sites, molecules_per_site, seed=21):
        super().__init__(volume_indices, outside_count, active_sites=active_sites,
                         molecules_per_site=molecules_per_site, seed=seed)


class TransmitterField(brainstate.nn.Module):
    """Advance a zero-initialized extracellular molecule field and its ledgers.

    Parameters
    ----------
    transport : DiffusionGraph
        Explicit physical volumes, transport, clearance and clock.

    Notes
    -----
    The external reservoir is zero concentration. Uptake is an accounted
    removed pool, not an implied intracellular transporter or metabolic model.
    """

    def __init__(self, transport):
        super().__init__()
        self.transport = transport
        self.concentration = brainstate.HiddenState(jnp.zeros_like(transport.volumes))
        self.released_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.removed_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.boundary_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.balance_error = brainstate.ShortTermState(jnp.asarray(0.))
        self.valid = brainstate.ShortTermState(jnp.asarray(True))
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def update(self, molecules):
        """Inject a molecule impulse and advance one physical transport tick.

        Parameters
        ----------
        molecules : array-like
            Nonnegative molecule counts per extracellular volume.

        Returns
        -------
        array
            New concentrations in mM. Check the cumulative validity flag.
        """
        values = jnp.asarray(molecules)
        if values.shape != self.concentration.value.shape:
            raise ValueError('Transmitter molecules must match extracellular volumes')
        amount = values/MOLECULES_PER_MM_UM3
        result = self.transport.step(self.concentration.value,
                                     source_amount_per_ms=amount/self.transport.dt_ms)
        self.concentration.value = result.concentration
        self.released_amount.value += jnp.sum(amount)
        self.removed_amount.value += result.removed_amount
        self.boundary_amount.value += result.boundary_amount
        self.balance_error.value += result.balance_error
        self.valid.value = self.valid.value & result.valid & jnp.all(values >= 0)
        self.tick.value += 1
        return result.concentration

    def reset_state(self):
        """Clear transmitter, all exchange ledgers, validity and clock."""
        for state in (self.concentration, self.released_amount, self.removed_amount,
                      self.boundary_amount, self.balance_error, self.tick):
            state.value = jnp.zeros_like(state.value)
        self.valid.value = jnp.asarray(True)
