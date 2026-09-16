import numpy as np
import pytest
import brainstate
import jax
import jax.numpy as jnp

from braintrace.datasets import h01_dhs_contraction
from braintrace.datasets.h01_dhs_forest import build_schedule, dense_matrix, forest_roots, solve


@pytest.fixture(autouse=True)
def _precision():
    with brainstate.environ.context(precision=64):
        yield


def _random_tree(rng, n, offset):
    """Random tree over rows offset..offset+n-1 with the root first."""
    edges = [(offset+i, offset+int(rng.integers(0, i))) for i in range(1, n)]
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def _forest(rng, sizes):
    edges, offset = [], 0
    for size in sizes:
        edges.append(_random_tree(rng, size, offset))
        offset += size
    return np.concatenate(edges) if edges else np.zeros((0, 2), dtype=np.int64), offset


def _coefficients(rng, n):
    low = -rng.uniform(.1, 1., n+1)
    up = -rng.uniform(.1, 1., n+1)
    d = np.abs(low)+np.abs(up)+rng.uniform(1., 3., n+1)   # diagonally dominant
    d[n], low[n], up[n] = 1., 0., 0.
    return d, low, up


@pytest.mark.parametrize('sizes', [(1,), (3, 5), (7, 1, 12, 4), (30, 30, 30)])
def test_forest_solve_matches_dense_oracle(sizes):
    rng = np.random.default_rng(sum(sizes))
    edges, n = _forest(rng, sizes)
    d, low, up = _coefficients(rng, n)
    rhs = np.append(rng.normal(size=n), 0.)
    packs, roots = build_schedule(edges, n)
    assert roots.tolist() == list(np.cumsum((0,)+sizes[:-1]))
    solution = np.asarray(solve(jnp.asarray(d), jnp.asarray(rhs), jnp.asarray(low), jnp.asarray(up), packs, roots))
    expected = np.linalg.solve(dense_matrix(d, low, up, edges, n), rhs[:n])
    np.testing.assert_allclose(solution[:n], expected, rtol=1e-10, atol=1e-12)
    assert solution[n] == 0.


def test_single_tree_schedule_equals_the_single_tree_contraction():
    rng = np.random.default_rng(3)
    edges, n = _forest(rng, (25,))
    packs, roots = build_schedule(edges, n)
    single = h01_dhs_contraction.prepare(edges.astype(np.int32), n)
    assert roots.tolist() == [0]
    assert len(packs) == len(single)
    for mine, theirs in zip(packs, single):
        for a, b in zip(mine, theirs):
            np.testing.assert_array_equal(a, b)


def test_isolated_rows_have_no_stages():
    packs, roots = build_schedule(np.zeros((0, 2), dtype=np.int64), 4)
    assert packs == () and roots.tolist() == [0, 1, 2, 3]
    d, rhs = jnp.asarray([2., 4., 8., 16., 1.]), jnp.asarray([1., 1., 1., 1., 0.])
    np.testing.assert_allclose(np.asarray(solve(d, rhs, jnp.zeros(5), jnp.zeros(5), packs, roots)), [.5, .25, .125, .0625, 0.])


def test_stage_count_is_the_deepest_tree():
    chain = np.asarray([(i, i-1) for i in range(1, 40)])
    small = np.asarray([(40, 41)])
    packs, _ = build_schedule(np.concatenate([chain, small]), 42)
    single, _ = build_schedule(chain, 40)
    assert sum(len(p[0]) for p in packs) == sum(len(p[0]) for p in single)


def test_forest_roots_rejects_bad_edges():
    with pytest.raises(ValueError, match='exactly once'):
        forest_roots(np.asarray([(1, 0), (1, 0)]), 2)
    with pytest.raises(ValueError, match='out of range'):
        forest_roots(np.asarray([(5, 0)]), 2)
    with pytest.raises(ValueError, match='reachable'):
        build_schedule(np.asarray([(1, 2), (2, 1)]), 3)


def test_solve_is_jittable_and_differentiable():
    rng = np.random.default_rng(9)
    edges, n = _forest(rng, (6, 9))
    d, low, up = _coefficients(rng, n)
    packs, roots = build_schedule(edges, n)
    rhs = jnp.asarray(np.append(rng.normal(size=n), 0.))
    fn = jax.jit(lambda r: solve(jnp.asarray(d), r, jnp.asarray(low), jnp.asarray(up), packs, roots).sum())
    gradient = np.asarray(jax.grad(fn)(rhs))
    expected = np.linalg.solve(dense_matrix(d, low, up, edges, n).T, np.ones(n))
    np.testing.assert_allclose(gradient[:n], expected, rtol=1e-9, atol=1e-12)
