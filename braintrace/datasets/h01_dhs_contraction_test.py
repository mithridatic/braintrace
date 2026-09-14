"""Topology validation, bounded ownership, and full-system contraction oracles."""

from collections import OrderedDict
import gc

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from . import h01_dhs_contraction as subject


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(subject, '_CACHE', OrderedDict())
    monkeypatch.setattr(subject, '_CACHE_BYTES', 0)


@pytest.mark.parametrize('edges,n', [([[1, 0]], 0), ([[1, 0]], 1.5),
    ([[0, 1]], 2), ([[1, 0], [1, 0]], 3), ([[1, 2], [2, 1]], 3),
    ([[1, 1]], 2), ([[1, -1]], 2), ([[1, 3]], 2), ([1, 0], 2)])
def test_invalid_trees_are_rejected(edges, n):
    with pytest.raises(ValueError):
        subject.prepare(np.asarray(edges, dtype=np.int32), n)


def test_static_integral_edges_required():
    with pytest.raises(ValueError):
        subject.prepare([[1, 0]], 2)
    with pytest.raises(ValueError):
        subject.prepare(np.array([[1., 0.]]), 2)


def test_cache_tracks_content_shape_and_owner_lifetime():
    edges = np.array([[1, 0], [2, 1]], dtype=np.int32)
    first = subject.prepare(edges, 3)
    assert subject.prepare(edges, 3) is first
    with pytest.raises(ValueError):
        subject.prepare(edges, 3.)
    assert all(not value.flags.writeable for group in first for value in group)
    edges[1, 1] = 0
    assert subject.prepare(edges, 3) is not first
    edges.shape = (4,)
    with pytest.raises(ValueError):
        subject.prepare(edges, 3)
    edges.shape = (2, 2)
    subject.prepare(edges, 3)
    assert subject._CACHE_BYTES > 0
    del edges
    gc.collect()
    assert not subject._CACHE and subject._CACHE_BYTES == 0


def test_cache_limits_and_oversized_schedule(monkeypatch):
    arrays = [np.array([[1, 0], [2, 1]], dtype=np.int32) for _ in range(3)]
    first = subject.prepare(arrays[0], 3)
    size = subject._CACHE_BYTES
    monkeypatch.setattr(subject, '_CACHE_LIMIT_BYTES', size)
    second = subject.prepare(arrays[1], 3)
    assert len(subject._CACHE) == 1 and subject._CACHE_BYTES == size
    assert id(arrays[0]) not in subject._CACHE
    monkeypatch.setattr(subject, '_CACHE_LIMIT_BYTES', 10*size)
    monkeypatch.setattr(subject, '_CACHE_LIMIT_ENTRIES', 1)
    subject.prepare(arrays[2], 3)
    assert len(subject._CACHE) == 1 and id(arrays[2]) in subject._CACHE
    monkeypatch.setattr(subject, '_CACHE_LIMIT_BYTES', 0)
    oversized = np.array([[1, 0]], dtype=np.int32)
    assert subject.prepare(oversized, 2)
    assert id(oversized) not in subject._CACHE
    assert first and second


@pytest.mark.parametrize('n', [1, 7, 65])
def test_solve_preserves_inputs_and_matches_dense(n):
    with brainstate.environ.context(precision=64):
        edges = np.array([(i, (i-1)//2) for i in range(1, n)], dtype=np.int32).reshape(-1, 2)
        packs = subject.prepare(edges, n)
        args = (jnp.linspace(3., 4., n+1), jnp.sin(jnp.arange(n+1)),
                jnp.linspace(-.1, -.2, n+1), jnp.linspace(-.2, -.1, n+1))
        before = [np.asarray(value).copy() for value in args]
        d, rhs, low, up = args
        matrix = jnp.diag(d).at[edges[:, 0], edges[:, 1]].set(low[edges[:, 0]])
        matrix = matrix.at[edges[:, 1], edges[:, 0]].set(up[edges[:, 0]])
        actual = jax.jit(lambda *values: subject.solve(*values, packs))(*args)
        np.testing.assert_allclose(actual, jnp.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
        for value, previous in zip(args, before):
            np.testing.assert_array_equal(value, previous)


def test_grouping_bounds_padding_for_deep_chain():
    n = 2048
    edges = np.array([(i, i-1) for i in range(1, n)], dtype=np.int32)
    packs = subject.prepare(edges, n)
    valid = np.concatenate([group[0][group[0] < n] for group in packs])
    np.testing.assert_array_equal(np.sort(valid), np.arange(1, n))
    assert len(packs) == 3
    assert sum(group[0].size for group in packs) < 3*n
