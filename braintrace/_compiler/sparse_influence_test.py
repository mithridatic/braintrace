"""Structural support and allocation checks for heterogeneous state blocks."""

import pytest

from .sparse_influence import SparseInfluence


def test_temporal_closure_keeps_cable_and_delayed_paths():
    # Current -> voltage -> channel -> queue -> second cell; an isolated cell
    # must not acquire any of those parameter dependencies.
    layout = SparseInfluence.build(
        [(3,), (2,), (5,), (7,), (4,)], [{0}, set(), set(), {1}, {2}],
        [set(), {0}, {1}, {2}, {4}], output_size=3)
    assert layout.outputs == ((0,), (0,), (0,), (0, 1), (2,))
    assert layout.elements == 3 + 2 + 5 + 14 + 4
    assert layout.colors[0] != layout.colors[1]
    assert layout.colors[2] == layout.colors[0]
    assert layout.color_count == 2


def test_cycles_close_without_adding_disconnected_outputs():
    layout = SparseInfluence.build([(2,), (3,), (1,)], [{0}, {1}, {2}],
                                   [{1}, {0}, set()], output_size=3)
    assert layout.outputs == ((0, 1), (0, 1), (2,))


def test_empty_support_and_scalar_shape():
    layout = SparseInfluence.build([(), (0,), (2,)], [set(), {0}, set()],
                                   [set(), set(), set()], output_size=1)
    assert layout.elements == 0
    assert layout.color_count == 1


@pytest.mark.parametrize('shapes,seeds,parents,size', [
    ([(2,)], [], [set()], 1),
    ([(2,)], [{0}], [], 1),
    ([(-1,)], [{0}], [set()], 1),
    ([(2,)], [{1}], [set()], 1),
    ([(2,)], [{0}], [{1}], 1),
    ([(2,)], [{-1}], [set()], 1),
    ([(2,)], [{0}], [set()], -1),
])
def test_invalid_structural_metadata(shapes, seeds, parents, size):
    with pytest.raises(ValueError):
        SparseInfluence.build(shapes, seeds, parents, output_size=size)


def test_allocation_limit_counts_closed_support_before_allocation():
    with pytest.raises(MemoryError, match='32'):
        SparseInfluence.build([(2,), (2,)], [{0}, {1}], [{1}, {0}],
                              output_size=2, itemsize=4, max_bytes=31)
    layout = SparseInfluence.build([(2,), (2,)], [{0}, {1}], [{1}, {0}],
                                   output_size=2, itemsize=4, max_bytes=32)
    assert layout.nbytes == 32
