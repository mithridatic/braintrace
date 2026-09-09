"""Static sparse output support for heterogeneous recurrent state blocks."""

from dataclasses import dataclass
from math import prod
from numbers import Integral


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

        # Static compiler worklist, not a numerical model rollout. A dependency
        # can traverse arbitrarily many events, so close through every cycle.
        children = [set() for _ in shapes]
        for child, row in enumerate(parents):
            for parent in row:
                children[parent].add(child)
        support = [set(row) for row in seeds]
        pending = list(range(len(shapes)))
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
        outputs = tuple(tuple(sorted(row)) for row in support)
        elements = sum(prod(shape) * len(row) for shape, row in zip(shapes, outputs))
        nbytes = elements * itemsize
        if nbytes > max_bytes:
            raise MemoryError(f'Sparse output factors require {nbytes} bytes; limit is {max_bytes}')

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
        return cls(shapes, outputs, tuple(colors), elements, nbytes)
