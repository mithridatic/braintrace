"""Remap original cable selections after explicit spine subdivision."""

import numpy as np


def remap_location(mapping, branch, x):
    """Map one original normalized location to a named subdivided branch.

    Parameters
    ----------
    mapping : dict
        Source mapping returned by ``add_spines``.
    branch : int
        Original branch index.
    x : float
        Original normalized location in [0, 1].

    Returns
    -------
    tuple
        New branch name and normalized coordinate. At an exact shared split,
        choose the proximal interval consistently; both share a cable endpoint.
    """
    if branch not in mapping or not np.isfinite(x) or not 0 <= x <= 1:
        raise ValueError('Location is outside the original source mapping')
    for lo, hi, name in mapping[branch]:
        if lo <= x <= hi:
            return name, (x-lo)/(hi-lo)
    raise ValueError('Location lies in a gap in the source mapping')


def remap_intervals(mapping, intervals):
    """Split source selections without losing or duplicating original cable.

    Parameters
    ----------
    mapping : dict
        Source mapping returned by ``add_spines``.
    intervals : sequence
        Original ``(branch_index, lo, hi)`` intervals, excluding added spines.

    Returns
    -------
    tuple
        ``(new_branch_name, lo, hi)`` intervals in new local coordinates.
    """
    result = []
    for branch, start, stop in intervals:
        if branch not in mapping or not np.isfinite([start, stop]).all() or not 0 <= start < stop <= 1:
            raise ValueError('Invalid source cable interval')
        covered = 0.
        for lo, hi, name in mapping[branch]:
            left, right = max(lo, start), min(hi, stop)
            if left < right:
                result.append((name, (left-lo)/(hi-lo), (right-lo)/(hi-lo)))
                covered += right-left
        if not np.isclose(covered, stop-start, rtol=0, atol=1e-12):
            raise ValueError('Source mapping has a gap or overlap')
    return tuple(result)


def remap_electrical_regions(mapping, source_regions, spines):
    """Carry a complete source partition onto parent-matched spine branches.

    Parameters
    ----------
    mapping : dict
        Subdivision map from ``add_spines``.
    source_regions : dict
        Electrical family to original interval triples.
    spines : sequence of Spine
        The exact additions used for subdivision.

    Returns
    -------
    dict
        Electrical family to named new intervals, including each neck and head.
        Every added spine inherits its parent's electrical family explicitly.
    """
    by_branch = {branch: [] for branch in mapping}
    for family, intervals in source_regions.items():
        for branch, lo, hi in intervals:
            if branch not in by_branch:
                raise ValueError('Electrical partition refers to an unknown source branch')
            by_branch[branch].append((lo, hi, family))
    for intervals in by_branch.values():
        end = 0.
        for lo, hi, _ in sorted(intervals):
            if not np.isfinite([lo, hi]).all() or lo != end or hi <= lo or hi > 1:
                raise ValueError('Source electrical partition has a gap or overlap')
            end = hi
        if end != 1.:
            raise ValueError('Source electrical partition must cover every branch')
    result = {family: list(remap_intervals(mapping, intervals)) for family, intervals in source_regions.items()}
    if len({spine.identity for spine in spines}) != len(spines):
        raise ValueError('Duplicate spine identity')
    for spine in spines:
        intervals = by_branch.get(spine.parent_branch, ())
        families = [family for lo, hi, family in intervals
                    if lo <= spine.parent_x < hi or spine.parent_x == hi == 1.]
        if len(families) != 1:
            raise ValueError('Spine attachment has no unique electrical parent')
        for part in ('neck', 'head'):
            result[families[0]].append(('added_'+spine.identity+'_'+part, 0., 1.))
    return {family: tuple(intervals) for family, intervals in result.items()}
