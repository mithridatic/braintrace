"""Numerical stress gates for the opt-in tree contraction prototype."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from h01_contraction_profile import schedule, solve


@pytest.mark.parametrize('parents', [[0], [0, 0, 1, 2, 3, 4, 5],
    [0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0]])
@pytest.mark.parametrize('stiffness', [1., 1.e4, 1.e6])
def test_contraction_matches_dense_stiff_systems(parents, stiffness):
    with brainstate.environ.context(precision=64):
        n = len(parents)
        nodes = np.arange(1, n)
        ancestors = np.asarray(parents[1:], dtype=np.int32)
        low = np.r_[0., -np.geomspace(1., stiffness, max(n-1, 1))[:n-1], 0.]
        up = low*np.linspace(.8, 1.2, n+1)
        matrix = np.zeros((n+1, n+1))
        matrix[nodes, ancestors] = low[nodes]
        matrix[ancestors, nodes] = up[nodes]
        diagonal = 1.+np.abs(matrix).sum(axis=1)
        matrix += np.diag(diagonal)
        expected = np.sin(np.arange(n+1)).astype(float)
        expected[-1] = 0.
        rhs = matrix@expected
        stages = schedule(parents)
        output = jax.jit(lambda rhs: solve(jnp.asarray(diagonal), rhs,
            jnp.asarray(low), jnp.asarray(up), stages))(jnp.asarray(rhs))
        np.testing.assert_allclose(output, expected, rtol=1.e-9, atol=1.e-9)
        np.testing.assert_allclose(output, np.linalg.solve(matrix, rhs), rtol=1.e-9, atol=1.e-9)
        assert np.linalg.norm(matrix@output-rhs, np.inf) <= 1.e-12*max(1., np.linalg.norm(rhs, np.inf))


def test_schedule_eliminates_every_nonroot_exactly_once():
    parents = np.arange(2048, dtype=np.int32)-1
    parents[0] = 0
    stages = schedule(parents)
    removed = np.concatenate([stage[0] for stage in stages])
    np.testing.assert_array_equal(np.sort(removed), np.arange(1, len(parents)))
    assert len(stages) <= 12
