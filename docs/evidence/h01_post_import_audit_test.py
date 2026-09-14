"""Reject edge loss and coordinate/radius changes independently of total area."""

import numpy as np
import pytest

from h01_post_import_audit import compare_edges


EDGES = np.array([[0., 0., 0., 1., 0., 0., .2, .1],
                  [1., 0., 0., 2., 0., 0., .1, .1]])


def test_order_independent_identity():
    assert compare_edges(EDGES, EDGES[::-1]) == 0.


@pytest.mark.parametrize('actual', [EDGES[:1], np.empty((0, 8)), EDGES[:, :7],
                                   EDGES * np.nan])
def test_missing_or_nonfinite_edges(actual):
    with pytest.raises(ValueError):
        compare_edges(EDGES, actual)


@pytest.mark.parametrize('column', [0, 3, 6, 7])
def test_endpoint_or_radius_change(column):
    changed = EDGES.copy()
    changed[0, column] += .01
    with pytest.raises(ValueError):
        compare_edges(EDGES, changed)


def test_empty_geometry():
    with pytest.raises(ValueError):
        compare_edges(np.empty((0, 8)), np.empty((0, 8)))
