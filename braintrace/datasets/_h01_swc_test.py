"""Geometry and annotation regression tests for the H01 conversion."""

import numpy as np
import pytest

from ._h01_swc import normalize


def test_voxels_and_radii_use_different_scales_and_keep_annotations():
    source = b"8 1 10 20 30 200 -1\n9 -1 20 20 40 100 8\n"
    rows, text = normalize(source)
    data = np.loadtxt(text.splitlines())
    np.testing.assert_allclose(data[:, 2:5], [[.32, .64, .99], [.64, .64, 1.32]])
    np.testing.assert_allclose(data[:, 5], [.2, .1])
    np.testing.assert_array_equal(data[:, [0, 1, 6]], [[1, 0, -1], [2, 0, 1]])
    np.testing.assert_array_equal(rows[:, 1], [1, -1])
    assert not rows.flags.writeable


def test_unsorted_nodes_preserve_parent_relations_and_branches():
    rows, text = normalize(b"5 0 2 0 0 32 7\n7 3 1 0 0 64 -1\n9 -1 1 1 0 32 7\n")
    converted = np.loadtxt(text.splitlines())
    np.testing.assert_array_equal(converted[:, 6], [-1, 1, 1])
    np.testing.assert_array_equal(rows[:, 0], [5, 7, 9])


@pytest.mark.parametrize("source,match", [
    (b"0 0 0 0 0 32\n", "seven finite"),
    (b"0 0 nan 0 0 32 -1\n", "seven finite"),
    (b"0.5 0 0 0 0 32 -1\n", "integers"),
    (b"-2 0 0 0 0 32 -1\n", "nonnegative"),
    (b"0 0 0 0 0 32 -1\n0 0 1 0 0 32 0\n", "unique"),
    (b"0 0 0 0 0 0 -1\n", "radii"),
    (b"0 0 0 0 0 -3 -1\n", "radii"),
    (b"0 0 0 0 0 32 -1\n1 0 1 0 0 32 -1\n", "one root"),
    (b"0 0 0 0 0 32 1\n1 0 1 0 0 32 0\n", "one root"),
    (b"0 0 0 0 0 32 -1\n1 0 1 0 0 32 4\n", "Missing parent"),
    (b"0 0 0 0 0 32 -1\n1 0 1 0 0 32 2\n2 0 2 0 0 32 1\n", "cycle"),
    (b"0 3 0 0 0 32 -1\n", "singleton"),
])
def test_invalid_geometry_is_rejected_without_inventing_repairs(source, match):
    with pytest.raises(ValueError, match=match):
        normalize(source)
