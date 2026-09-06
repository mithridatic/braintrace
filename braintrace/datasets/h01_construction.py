"""Bound branch-index work during H01 compartment construction."""

from contextlib import contextmanager

import braincell


@contextmanager
def _indexed_morphology(morphology):
    original = morphology._branch_index
    original_branches = morphology.branch_by_order
    saved = {name: (name in morphology.__dict__, morphology.__dict__.get(name))
             for name in ("_branch_index", "branch_by_order")}
    stamp, indices = None, None
    branch_stamp, ordered_branches = None, None

    def index(node_id, *, order="default"):
        nonlocal stamp, indices
        if order != "default":
            return original(node_id, order=order)
        current = (len(morphology._nodes), morphology._next_id)
        if current != stamp:
            indices = morphology._branch_index_map(order="default")
            stamp = current
        return indices[node_id]

    def branches(*, order="default"):
        nonlocal branch_stamp, ordered_branches
        if order != "default":
            return original_branches(order=order)
        current = (len(morphology._nodes), morphology._next_id)
        if current != branch_stamp:
            ordered_branches = original_branches(order="default")
            branch_stamp = current
        return ordered_branches

    morphology._branch_index = index
    morphology.branch_by_order = branches
    try:
        yield
    finally:
        for name, (had_override, previous) in saved.items():
            if had_override:
                setattr(morphology, name, previous)
            else:
                delattr(morphology, name)


class H01Cell(braincell.Cell):
    """BrainCell cell with a construction-scoped branch-index lookup cache.

    Parameters
    ----------
    morpho : braincell.Morphology
        Source geometry, as accepted by ``braincell.Cell``.
    **kwargs
        Unchanged BrainCell cell parameters.

    Notes
    -----
    This changes index and ordered-branch lookup work only. The ordinary discretization,
    channel mechanisms, and solver remain in use. The temporary instance
    lookup is removed after each synchronous discretization attempt. Runtime
    initialization can clone the morphology; that clone receives its own
    temporary cache when discretized. No global methods are replaced.
    """

    @property
    def _discretization(self):
        with _indexed_morphology(self._morpho):
            return super()._discretization
