"""GABA source cardinality, collision accounting and receptor response."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .gaba import GabaReleaseSites, tonic_occupancy


def test_occupancy_source_limits_and_derivative():
    np.testing.assert_allclose(tonic_occupancy(jnp.array([0., 1e-4, 1e-3])), [0., .5, 10/11], rtol=1e-6)
    assert np.isnan(tonic_occupancy(-1.))
    assert np.isnan(tonic_occupancy(1., ec50_mm=-1.))
    assert float(jax.grad(tonic_occupancy)(0.)) == pytest.approx(1e4)


def test_paper_site_selection_without_replacement_reset_and_padding():
    source = GabaReleaseSites(np.arange(6000)[None, :], 6000)
    run = brainstate.transform.jit(source.update)
    first = np.asarray(run(jnp.ones(1)))
    assert np.count_nonzero(first) == 2000 and first.sum() == 2000*3000
    assert np.all((first == 0) | (first == 3000))
    key, tick = source.rng.value, source.tick.value
    assert not np.any(run(jnp.ones(1), advance=False))
    assert jnp.array_equal(source.rng.value, key) and source.tick.value == tick
    second = np.asarray(run(jnp.ones(1)))
    assert not np.array_equal(first, second)
    source.reset_state()
    np.testing.assert_array_equal(run(jnp.ones(1)), first)


def test_collision_counts_and_silent_physical_ticks():
    source = GabaReleaseSites(np.array([[0, 0, 1], [0, 1, 1]]), 2, active_sites=3)
    run = brainstate.transform.jit(source.update)
    np.testing.assert_array_equal(run(jnp.array([1., 2.])), [12000., 15000.])
    key = source.rng.value
    np.testing.assert_array_equal(run(jnp.zeros(2)), [0., 0.])
    assert not jnp.array_equal(source.rng.value, key) and source.tick.value == 2
    with pytest.raises(ValueError):
        source.update(jnp.ones(1))


@pytest.mark.parametrize('ids,count,active', [([], 1, 0), ([[1.]], 2, 1), ([[2]], 2, 1),
    ([[0]], 0, 1), ([[0]], 1, 2), ([[0]], 1, -1)])
def test_invalid_release_configuration(ids, count, active):
    with pytest.raises(ValueError):
        GabaReleaseSites(ids, count, active_sites=active)
