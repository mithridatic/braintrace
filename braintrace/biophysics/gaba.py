"""Fixed-site GABA release and source-derived extrasynaptic receptor occupancy."""

import brainstate
import jax.numpy as jnp
import numpy as np


def tonic_occupancy(concentration_mm, *, ec50_mm=1e-4, hill=1.):
    """Evaluate GabaUpdateLinear.mod's equilibrium receptor occupancy.

    Parameters
    ----------
    concentration_mm : array-like
        Extracellular GABA, in mM.
    ec50_mm, hill : float, optional
        Fixed positive half-activation and Hill exponent; source defaults.

    Returns
    -------
    array
        Fractional activation. Negative concentrations return NaN rather than
        following the source's negative-concentration clipping.
    """
    c = jnp.asarray(concentration_mm)
    # This algebra preserves the finite derivative at zero for source hill=1.
    power = c**hill
    occupancy = power/(ec50_mm**hill+power)
    valid = (c >= 0) & jnp.isfinite(c) & (ec50_mm > 0) & jnp.isfinite(ec50_mm) & (hill > 0) & jnp.isfinite(hill)
    return jnp.where(valid, occupancy, jnp.nan)


class GabaReleaseSites(brainstate.nn.Module):
    """Choose a fixed number of distinct sites per source and physical tick.

    Parameters
    ----------
    volume_indices : array-like
        Placement (n_sources, n_sites) into the extracellular domain. Repeated
        volume IDs are allowed; the physical release sites remain distinct.
    outside_count : int
        Extracellular volume count.
    active_sites : int, optional
        Sites selected without replacement, default 2000 of the paper's 6000.
    molecules_per_site : float, optional
        Released molecules per selected site/spike, default 3000.
    seed : int, optional
        Episode seed for BrainState random state.

    Notes
    -----
    Sampling on silent physical ticks keeps the stream independent of spike
    counts. Padding consumes nothing. This is conserved molecule injection,
    not the source-history concentration boundary experiment.
    """

    def __init__(self, volume_indices, outside_count, *, active_sites=2000,
                 molecules_per_site=3000., seed=21):
        super().__init__()
        ids = np.asarray(volume_indices)
        if (ids.ndim != 2 or not ids.shape[0] or not ids.shape[1] or
                not np.issubdtype(ids.dtype, np.integer) or type(outside_count) is not int or
                outside_count <= 0 or np.any(ids < 0) or np.any(ids >= outside_count)):
            raise ValueError('Invalid release-site placements')
        if (type(active_sites) is not int or not 0 <= active_sites <= ids.shape[1] or
                not np.isfinite(molecules_per_site) or molecules_per_site < 0):
            raise ValueError('Invalid site count or molecule release')
        self.volume_indices = jnp.asarray(ids, dtype=jnp.int32)
        self.outside_count, self.active_sites = outside_count, active_sites
        self.molecules_per_site, self.seed = molecules_per_site, seed
        self.rng = brainstate.random.RandomState(seed)
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def update(self, source_spikes, *, advance=True):
        """Return molecule counts per volume for a real or padding tick.

        Parameters
        ----------
        source_spikes : array
            Spike indicator or differentiable amplitude per source.
        advance : bool, optional
            Whether physical time advances; False preserves RNG and tick.

        Returns
        -------
        array
            Molecule counts in extracellular volume ordering.
        """
        spikes = jnp.asarray(source_spikes)
        if spikes.shape != (self.volume_indices.shape[0],):
            raise ValueError('Spike count must match release sources')
        def active():
            ranks = self.rng.uniform(size=self.volume_indices.shape)
            selected = jnp.argsort(ranks, axis=1)[:, :self.active_sites]
            target = jnp.take_along_axis(self.volume_indices, selected, axis=1)
            amounts = jnp.broadcast_to(spikes[:, None]*self.molecules_per_site, target.shape)
            self.tick.value = self.tick.value+1
            return jnp.zeros(self.outside_count).at[target.reshape(-1)].add(amounts.reshape(-1))
        return brainstate.transform.cond(advance, active, lambda: jnp.zeros(self.outside_count))

    def reset_state(self):
        """Restore the episode random stream and physical clock."""
        self.rng.seed(self.seed)
        self.tick.value = jnp.zeros_like(self.tick.value)
