"""Independent cylinder geometry, topology and extracellular scaling tests."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from .shells import radial_shell_graph, cartesian_graph


def test_shell_volumes_radial_resistance_and_no_axial_connection():
    with brainstate.environ.context(precision=64):
        graph = radial_shell_graph([2., 3.], [1., 2., 3.], [.1, 1.], dt_ms=.1)
        np.testing.assert_allclose(graph.volumes, np.pi*np.array([6., 10., 9., 15.]))
        np.testing.assert_array_equal(graph.edges, [[0, 1], [2, 3]])
        resistance = np.log(2/1.5)/.1+np.log(2.5/2)
        np.testing.assert_allclose(graph.conductance, 2*np.pi*np.array([2., 3.])/resistance)
        result = graph.step(jnp.array([1., 0., 0., 0.]))
        assert result.valid and abs(result.balance_error) < 1e-12
        np.testing.assert_array_equal(result.concentration[2:], [0., 0.])


def test_axial_and_bath_boundaries():
    with brainstate.environ.context(precision=64):
        graph = radial_shell_graph([1., 1.], [0., 1.], [1.], dt_ms=.1,
                                   outer_reservoir=[False, True], longitudinal=True)
        assert graph.boundary[0] == 0 and graph.boundary[1] > 0
        result = graph.step(jnp.array([1., 1.]), reservoir_mm=jnp.zeros(2))
        assert result.boundary_amount < 0 and abs(result.balance_error) < 1e-12


def test_cartesian_fraction_and_tortuosity():
    with brainstate.environ.context(precision=64):
        graph = cartesian_graph((2, 2, 2), 2., diffusion_um2_ms=1., dt_ms=.1)
        assert len(graph.edges) == 12
        np.testing.assert_allclose(graph.volumes, 1.6)
        np.testing.assert_allclose(graph.conductance, .4/1.6**2)
        result = graph.step(jnp.full(8, 2.5), reservoir_mm=jnp.full(8, 2.5))
        assert result.valid
        np.testing.assert_allclose(result.concentration, 2.5, atol=1e-12)
        closed = cartesian_graph((1, 1, 1), 1., diffusion_um2_ms=1., dt_ms=.1, fixed_boundary=False)
        assert len(closed.edges) == 0 and closed.boundary[0] == 0


@pytest.mark.parametrize('args', [([], [0., 1.], [1.]), ([1.], [1., 0.], [1.]), ([1.], [0., 1.], [-1.])])
def test_invalid_shell_geometry(args):
    with pytest.raises(ValueError):
        radial_shell_graph(*args, dt_ms=.1)


def test_invalid_cartesian_geometry():
    for shape, spacing, fraction in (((0, 1, 1), 1., .2), ((1, 1, 1), 0., .2), ((1, 1, 1), 1., 2.)):
        with pytest.raises(ValueError):
            cartesian_graph(shape, spacing, diffusion_um2_ms=1., dt_ms=.1, volume_fraction=fraction)
