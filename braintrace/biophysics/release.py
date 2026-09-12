"""Fixed-probability release with BrainState-owned reproducible random streams."""

import brainstate
import jax.numpy as jnp
import numpy as np


def probability_profile(distance_um, *, kind="uniform", mean=.36, slope_per_um=5.2e-4,
                        centre_um=400., width_um=200.):
    """Construct a spatial probability profile with the requested discrete mean.

    Parameters
    ----------
    distance_um : array-like
        Nonnegative dendritic path distances, one per synapse.
    kind : str, optional
        ``uniform``, ``linear`` or ``bell``.
    mean : float, optional
        Fixed mean release probability, default 0.36.
    slope_per_um : float, optional
        Linear slope, in 1/um.
    centre_um, width_um : float, optional
        Bell centre and positive width in um. Its amplitude is 0.30.

    Returns
    -------
    ndarray
        Probabilities. Impossible mean/shape combinations raise, never clip.
    """
    x = np.asarray(distance_um, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Distances must be a nonempty nonnegative finite vector")
    if (not np.isfinite([mean, slope_per_um, centre_um, width_um]).all() or
            not 0 <= mean <= 1 or width_um <= 0):
        raise ValueError("Invalid probability profile parameters")
    if kind == "uniform":
        return np.full_like(x, mean)
    if kind == "linear":
        raw = slope_per_um*x
    elif kind == "bell":
        raw = .30*np.exp(-.5*((x-centre_um)/width_um)**2)
    else:
        raise ValueError("Unknown probability profile")
    probabilities = raw - raw.mean() + mean
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Spatial shape cannot have this mean without invalid probabilities")
    return probabilities


class ReleaseState(brainstate.nn.Module):
    """Sample one fixed-probability release mask per contact and cable tick.

    Parameters
    ----------
    probability : array-like
        Immutable release probabilities in [0, 1].
    seed : int, optional
        BrainState random-stream seed.

    Notes
    -----
    A masked padding step consumes no random draw. A real silent cable step
    consumes a draw, making streams independent of weight-dependent spike counts.
    Save/restore both ``rng.value`` and ``tick.value`` at checkpoint boundaries.
    Probabilities and masks are not trainable parameters.
    """

    def __init__(self, probability, *, seed=21):
        super().__init__()
        p = np.asarray(probability, dtype=float)
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be a finite vector in [0, 1]")
        self.probability = jnp.asarray(p)
        self.seed = seed
        self.rng = brainstate.random.RandomState(seed)
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def update(self, spikes, *, advance=True):
        """Return released events, leaving all state unchanged for padding.

        Parameters
        ----------
        spikes : array
            Per-contact spike indicators or differentiable spike amplitudes.
        advance : bool, optional
            Whether to advance the random stream and physical tick.

        Returns
        -------
        array
            Spike amplitudes multiplied by a Bernoulli mask.
        """
        spikes = jnp.asarray(spikes)
        if spikes.shape != self.probability.shape:
            raise ValueError("Spike vector must match release probabilities")
        def active():
            mask = self.rng.uniform(size=self.probability.shape) < self.probability
            self.tick.value = self.tick.value + 1
            return spikes * mask
        return brainstate.transform.cond(advance, active, lambda: jnp.zeros_like(spikes))

    def reset_state(self):
        """Reset the random stream and tick to the configured episode origin."""
        self.rng.seed(self.seed)
        self.tick.value = jnp.zeros_like(self.tick.value)
