"""Dense and batching gates for the experimental persistent GPU kernel."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from h01_contraction_profile import schedule
from h01_persistent_contraction import solve


@pytest.mark.parametrize('parents', [[0], [0, 0, 1, 2, 3],
                                   [0, 0, 0, 1, 1, 2, 2], [0]*1025])
def test_dense_batched_sentinel_and_nonmutation(parents):
    if jax.default_backend() != 'gpu':
        pytest.skip('Triton kernel requires GPU')
    with brainstate.environ.context(precision=64):
        n = len(parents)
        d = jnp.full(n+1, 4., dtype=jnp.float64)
        low = jnp.full(n+1, -.01, dtype=jnp.float64)
        up = low*1.3
        rhs = jnp.stack((jnp.sin(jnp.arange(n+1)), jnp.cos(jnp.arange(n+1)))).astype(jnp.float64)
        before = [np.asarray(x).copy() for x in (d, rhs, low, up)]
        stages = schedule(parents)
        output = jax.jit(jax.vmap(lambda r: solve(d, r, low, up, stages)))(rhs)
        matrix = np.diag(np.asarray(d))
        for i in range(1, n):
            matrix[i, parents[i]] = low[i]
            matrix[parents[i], i] = up[i]
        expected = np.linalg.solve(matrix, np.asarray(rhs).T).T
        np.testing.assert_allclose(output, expected, rtol=1e-11, atol=1e-12)
        for original, after in zip(before, (d, rhs, low, up)):
            np.testing.assert_array_equal(original, after)
