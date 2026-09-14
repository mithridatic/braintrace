"""Implicit coupled reaction-diffusion gradients against finite differences."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .astrocyte import CalciumParameters, initial_calcium
from .astrocyte_network import shell_transport
from .astrocyte_coupled import coupled_calcium_step
from .transport import DiffusionGraph


@pytest.mark.parametrize('external', ['boundary_conductance', 'uptake_per_ms'])
def test_intracellular_solver_rejects_unaccounted_reservoirs(external):
    p = CalciumParameters()
    graph = DiffusionGraph(np.ones(4), np.empty((0, 2), int), [], dt_ms=.00125,
                           **{external: np.ones(4)})
    with pytest.raises(ValueError, match='closed'):
        coupled_calcium_step(jnp.broadcast_to(initial_calcium(p), (1, 25)), p, (graph,)*3,
            dt_ms=.00125, surface_to_volume=jnp.ones(1), glutamate_mm=jnp.zeros(1),
            influx_mm_ms=jnp.zeros((1, 4)))


def test_coupled_neighbor_gradient_and_convergence():
    with brainstate.environ.context(precision=64):
        p = CalciumParameters(alpha=0., pump_velocity_um_ms=0.)
        args = ([1., 1.], [2., 2.], [[0, 1]])
        graphs = (shell_transport(*args, dt_ms=.00125, diffusion=.3),
                  shell_transport(*args, dt_ms=.00125, diffusion=.05),
                  shell_transport(*args, dt_ms=.00125, diffusion=.05, radial=False))
        state = jnp.broadcast_to(initial_calcium(p), (2, 25))
        def response(dose):
            result, valid, _ = coupled_calcium_step(state, p, graphs, dt_ms=.00125,
                surface_to_volume=jnp.ones(2), glutamate_mm=jnp.zeros(2),
                influx_mm_ms=jnp.zeros((2, 4)).at[0, 0].set(dose))
            return jnp.where(valid, result[1, 0], jnp.nan)
        response = jax.jit(response)
        dose, epsilon = 1e-4, 1e-7
        gradient = jax.jit(jax.grad(response))(dose)
        expected = (response(dose+epsilon)-response(dose-epsilon))/(2*epsilon)
        assert np.isfinite(gradient) and gradient > 0
        np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=1e-11)
