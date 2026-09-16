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


def test_build_segmented_refines_blocks_by_declared_ports():
    """Two cells in one cable block, a contact from cell 0 into cell 1 through a ring block."""
    shapes = [(1, 6), (4, 1)]                      # cable block (two 3-compartment cells), ring buffer
    seeds = [{0, 1, 2}, {0, 2}]                    # block analysis: everything reaches everything
    parents = [{0, 1}, {0, 1}]
    blocks = [('segments', [0, 3, 6], [0, 1]), ('block', {0, ('contact', 0)}, {1, ('contact', 0)})]
    labels = [{0}, {1}, {('contact', 0)}]         # drive 0, drive 1, contact magnitude
    layout = SparseInfluence.build_segmented(shapes, seeds, parents, blocks, labels, output_size=3)
    assert layout.outputs == ((0, 1, 2), (0, 2))
    assert layout.slots[1] is None and layout.slots[0].shape == (1, 6, 3)
    assert layout.slots[0][0].tolist() == [[0, -1, -1]]*3+[[0, 1, 2]]*3
    assert layout.widths == (3, 2) and layout.elements == 6*3+4*2
    assert layout.colors == (0, 1, 2)
    with pytest.raises(MemoryError):
        SparseInfluence.build_segmented(shapes, seeds, parents, blocks, labels, output_size=3, max_bytes=8)


def test_build_segmented_segment_ports_can_be_directed():
    shapes = [(1, 4), (2, 2)]
    seeds = [{0, 1}, {0, 1}]
    parents = [{0, 1}, {0, 1}]
    ring = ('segments', [0, 1, 2], [({0, ('c', 0)}, {1, ('c', 0)}), ({1, ('c', 1)}, {0, ('c', 1)})])
    blocks = [('segments', [0, 2, 4], [0, 1]), ring]
    layout = SparseInfluence.build_segmented(shapes, seeds, parents, blocks, [{0}, {1}], output_size=2)
    assert layout.slots[0][0].tolist() == [[0, 1]]*4      # both cells reach each other through the two contacts
    assert layout.slots[1][0].tolist() == [[0, 1], [0, 1]]
    assert layout.color_count == 2


def test_build_segmented_rejects_bad_offsets_and_labels():
    with pytest.raises(ValueError, match='tile'):
        SparseInfluence.build_segmented([(1, 4)], [{0}], [set()], [('segments', [0, 3], [0])], [{0}], output_size=1)
    with pytest.raises(ValueError, match='label set'):
        SparseInfluence.build_segmented([(1, 4)], [{0}], [set()], [('segments', [0, 4], [0])], [], output_size=1)
    with pytest.raises(ValueError, match='Unknown'):
        SparseInfluence.build_segmented([(1, 4)], [{0}], [set()], [('rows',)], [{0}], output_size=1)
