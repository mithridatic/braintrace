"""Static sparse output support for heterogeneous recurrent state blocks."""

from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SparseInfluence:
    """Store temporally closed output support and collision-free JVP colors.

    Parameters
    ----------
    shapes : tuple of tuples
        Shapes of heterogeneous state blocks.
    outputs : tuple of tuples
        Sorted ETP output indices influencing each state block.
    colors : tuple of int
        Color of each output; outputs sharing a state block have distinct colors.
    elements : int
        Number of output-factor elements, excluding input factors and workspaces.
    nbytes : int
        Output-factor storage at the requested scalar item size.

    Notes
    -----
    Build from conservative compiler-derived dependencies. This representation
    does not prove the input dependency metadata; program analysis supplies it.
    """

    shapes: tuple
    outputs: tuple
    colors: tuple
    elements: int
    nbytes: int
    slots: tuple = None

    @property
    def widths(self):
        """Return the factor width (last axis) allocated for each state block.

        Returns
        -------
        tuple of int
            ``len(outputs[b])`` for a plain block; the slot-table width for a
            segmented block (see :meth:`build_segmented`).
        """
        if self.slots is None:
            return tuple(len(row) for row in self.outputs)
        return tuple(len(row) if table is None else int(table.shape[-1])
                     for row, table in zip(self.outputs, self.slots))

    @property
    def color_count(self):
        """Return the number of simultaneous directional-derivative passes.

        Returns
        -------
        int
            Zero for a layout with no output positions.
        """
        return max(self.colors, default=-1) + 1

    @classmethod
    def build(cls, shapes, seeds, parents, *, output_size, itemsize=8,
              max_bytes=2**30):
        """Close temporal support, color directions and check storage limits.

        Parameters
        ----------
        shapes : sequence of tuples
            Shapes of floating-point state blocks.
        seeds : sequence of collections of int
            Instantaneous ETP output dependencies for each block.
        parents : sequence of collections of int
            Previous-state blocks that can affect each next-state block.
        output_size : int
            Number of flattened ETP output positions.
        itemsize : int, optional
            Bytes per factor element; default eight.
        max_bytes : int, optional
            Maximum factor storage. Workspaces need a separate budget.

        Returns
        -------
        SparseInfluence
            Immutable layout; no numerical factor arrays are allocated.

        Raises
        ------
        ValueError
            Shapes, indices, counts or resource settings are invalid.
        MemoryError
            Temporally closed factors exceed the storage limit.
        """
        shapes = tuple(tuple(shape) for shape in shapes)
        seeds, parents = tuple(map(set, seeds)), tuple(map(set, parents))
        if len(shapes) != len(seeds) or len(shapes) != len(parents):
            raise ValueError('Every state block needs a shape, seed set and parent set')
        if any(not isinstance(n, Integral) or n < 0 for shape in shapes for n in shape):
            raise ValueError('State dimensions must be nonnegative integers')
        if not isinstance(output_size, Integral) or output_size < 0:
            raise ValueError('output_size must be a nonnegative integer')
        if any(not isinstance(n, Integral) or n <= 0 for n in (itemsize, max_bytes)):
            raise ValueError('itemsize and max_bytes must be positive integers')
        for rows, size in ((seeds, output_size), (parents, len(shapes))):
            if any(not isinstance(n, Integral) or not 0 <= n < size for row in rows for n in row):
                raise ValueError('Dependency index outside its declared range')

        outputs = _close(seeds, parents)
        elements = sum(prod(shape) * len(row) for shape, row in zip(shapes, outputs))
        nbytes = elements * itemsize
        if nbytes > max_bytes:
            raise MemoryError(f'Sparse output factors require {nbytes} bytes; limit is {max_bytes}')
        return cls(shapes, outputs, _color(outputs, output_size), elements, nbytes)

    @classmethod
    def build_segmented(cls, shapes, seeds, parents, blocks, output_labels, *, output_size,
                        itemsize=8, max_bytes=2**30):
        """Close support over declared segments of the state blocks.

        Parameters
        ----------
        shapes, seeds, parents : sequences
            As for :meth:`build`, from the block-level program analysis.
        blocks : sequence
            One declaration per block: ``('segments', offsets, owners)`` splits
            the block's last axis at ``offsets`` into segments; each owner is a
            label (the segment reads and feeds that label) or a
            ``(reads, feeds)`` pair of label sets. ``('block', reads, feeds)``
            keeps the block whole. A parent may influence a child only where
            the parent's ``feeds`` meet the child's ``reads`` (``None`` for
            all labels); an output seeds a block or segment only where its
            labels meet the ``reads``.
        output_labels : sequence of collections
            Labels each ETP output position belongs to; an output seeds only
            the segments and blocks of its labels.
        output_size, itemsize, max_bytes : int
            As for :meth:`build`.

        Returns
        -------
        SparseInfluence
            Layout whose ``slots[b]`` is an ``int32`` table ``shape + (width,)``
            of output positions per state position (``-1`` empty); its
            ``outputs`` rows are the unions per block.

        Notes
        -----
        The block-level analysis proves which blocks and outputs can interact;
        the declaration only refines *where* inside a block. Physics the
        declaration excludes (two segments coupled without a declared path)
        is not checked here; a gradient parity test against the unsegmented
        model is the check.
        """
        shapes = tuple(tuple(shape) for shape in shapes)
        if not (len(shapes) == len(seeds) == len(parents) == len(blocks)):
            raise ValueError('Every state block needs a shape, seed set, parent set and declaration')
        if len(output_labels) != output_size:
            raise ValueError('Every ETP output needs a label set')
        labels = [set(row) for row in output_labels]

        def ports(owner):
            # An owner is a label (reads = feeds = {label}) or a (reads, feeds) pair
            # of label sets, None meaning every label.
            if isinstance(owner, tuple) and len(owner) == 2 and all(
                    part is None or isinstance(part, (set, frozenset, list, tuple)) for part in owner):
                return (None if owner[0] is None else set(owner[0]), None if owner[1] is None else set(owner[1]))
            return ({owner}, {owner})

        virtual, members = [], []
        for index, (shape, spec) in enumerate(zip(shapes, blocks)):
            members.append([])
            if spec[0] == 'segments':
                offsets, owners = np.asarray(spec[1], dtype=np.int64), tuple(spec[2])
                if len(offsets) != len(owners)+1 or offsets[0] != 0 or offsets[-1] != shape[-1] or np.any(np.diff(offsets) < 0):
                    raise ValueError('Segment offsets must tile the last axis of block %d' % index)
                for segment, owner in enumerate(owners):
                    members[index].append(len(virtual))
                    virtual.append((index, segment, ports(owner), shape[:-1]+(int(offsets[segment+1]-offsets[segment]),)))
            elif spec[0] == 'block':
                members[index].append(len(virtual))
                virtual.append((index, None, ports((spec[1], spec[2])), shape))
            else:
                raise ValueError('Unknown block declaration %r' % (spec[0],))

        vseeds, vparents = [], []
        for index, _, (reads, _), _ in virtual:
            vseeds.append({o for o in seeds[index] if reads is None or labels[o] & reads})
            row = set()
            for parent in parents[index]:
                for other in members[parent]:
                    feeds = virtual[other][2][1]
                    if reads is None or feeds is None or reads & feeds:
                        row.add(other)
            vparents.append(row)
        rows = _close(vseeds, vparents)
        colors = _color(rows, output_size)
        outputs, slots, elements = [], [], 0
        for index, (shape, spec) in enumerate(zip(shapes, blocks)):
            union = sorted(set().union(*(rows[v] for v in members[index])))
            outputs.append(tuple(union))
            if spec[0] == 'block':
                slots.append(None)
                elements += prod(shape)*len(union)
                continue
            offsets = np.asarray(spec[1], dtype=np.int64)
            width = max((len(rows[v]) for v in members[index]), default=0)
            table = np.full(shape+(width,), -1, dtype=np.int32)
            for v in members[index]:
                _, segment, _, _ = virtual[v]
                row = rows[v]
                table[..., offsets[segment]:offsets[segment+1], :len(row)] = np.asarray(row, dtype=np.int32)
            table.flags.writeable = False
            slots.append(table)
            elements += prod(shape)*width
        nbytes = elements*itemsize
        if nbytes > max_bytes:
            raise MemoryError(f'Sparse output factors require {nbytes} bytes; limit is {max_bytes}')
        return cls(shapes, tuple(outputs), colors, elements, nbytes, tuple(slots))


def _close(seeds, parents):
    """Temporally close output support through the parent graph (static worklist)."""
    children = [set() for _ in seeds]
    for child, row in enumerate(parents):
        for parent in row:
            children[parent].add(child)
    support = [set(row) for row in seeds]
    pending = list(range(len(seeds)))
    queued = set(pending)
    while pending:
        parent = pending.pop()
        queued.remove(parent)
        for child in children[parent]:
            if not support[parent].issubset(support[child]):
                support[child].update(support[parent])
                if child not in queued:
                    pending.append(child)
                    queued.add(child)
    return tuple(tuple(sorted(row)) for row in support)


def _color(outputs, output_size):
    """Greedy colouring: outputs sharing a support row get distinct colours."""
    conflicts = [set() for _ in range(output_size)]
    for row in outputs:
        for output in row:
            conflicts[output].update(set(row) - {output})
    colors = []
    for output in range(output_size):
        forbidden = {colors[other] for other in conflicts[output] if other < output}
        color = 0
        while color in forbidden:
            color += 1
        colors.append(color)
    return tuple(colors)
