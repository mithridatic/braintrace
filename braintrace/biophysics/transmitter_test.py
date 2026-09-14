"""Molecule injection, sink accounting, finite derivatives and reset."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .transmitter import GlutamateReleaseSites, TransmitterField
from .transport import DiffusionGraph
from .extracellular import MOLECULES_PER_MM_UM3


def test_impulse_transport_accounting_and_reset():
    with brainstate.environ.context(precision=64):
        graph = DiffusionGraph([1., 2.], [[0, 1]], [1.], dt_ms=.005,
                               boundary_conductance=[0., .1], uptake_per_ms=[.2, .2])
        field = TransmitterField(graph)
        release = GlutamateReleaseSites([[0, 1]], 2, active_sites=2, molecules_per_site=3000.)
        def advance(_):
            return field(release(jnp.ones(1)))
        actual = brainstate.transform.for_loop(advance, jnp.arange(3))
        assert field.valid.value and field.tick.value == release.tick.value == 3
        expected = 18000./MOLECULES_PER_MM_UM3
        np.testing.assert_allclose(field.released_amount.value, expected)
        np.testing.assert_allclose(jnp.sum(actual[-1]*graph.volumes)+field.removed_amount.value-field.boundary_amount.value,
                                   expected, rtol=1e-10)
        assert field.removed_amount.value > 0 and field.boundary_amount.value < 0
        field.reset_state()
        release.reset_state()
        np.testing.assert_array_equal(brainstate.transform.for_loop(advance, jnp.arange(3)), actual)
        with pytest.raises(ValueError, match='volumes'):
            field.update(jnp.ones(3))
        field.update(jnp.array([-1., 0.]))
        assert not field.valid.value


def test_molecule_to_neighbor_gradient():
    with brainstate.environ.context(precision=64):
        graph = DiffusionGraph([1., 1.], [[0, 1]], [1.], dt_ms=.005)
        def response(dose):
            return graph.step(jnp.zeros(2), source_amount_per_ms=jnp.array([dose, 0.])/
                              (MOLECULES_PER_MM_UM3*graph.dt_ms)).concentration[1]
        derivative = jax.jit(jax.grad(response))(3000.)
        expected = (response(3001.)-response(2999.))/2
        assert derivative > 0
        np.testing.assert_allclose(derivative, expected, rtol=1e-9)
