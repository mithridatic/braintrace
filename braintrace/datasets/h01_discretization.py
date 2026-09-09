"""Electrical mesh boundaries for explicitly selected active cable regions."""

from dataclasses import dataclass

from braincell import CVPolicy
from braincell.filter import RegionExpr


@dataclass(frozen=True)
class BoundaryAlignedCV(CVPolicy):
    """Retain a base mesh and split at every selected region boundary.

    Parameters
    ----------
    base : CVPolicy
        Underlying spatial refinement policy.
    region : RegionExpr
        Active region whose endpoints must be represented in the mesh.

    Notes
    -----
    This changes the discretization only. It does not change morphology or
    assert that the selected region is an anatomically verified compartment.
    Cuts within 1e-9 normalized branch position are merged to avoid
    degenerate intervals below BrainCell's geometric tolerance.
    """

    base: CVPolicy
    region: RegionExpr

    def resolve_cv_bounds(self, morpho, *, paint_rules=None):
        """Return branch intervals with the active boundaries inserted.

        Parameters
        ----------
        morpho : Morphology
            Morphology being discretized.
        paint_rules : tuple, optional
            Passed to the underlying policy unchanged.

        Returns
        -------
        tuple
            Complete, ordered branch intervals without zero-length pieces.
        """
        cache = getattr(morpho, "_h01_cv_bounds_cache", None)
        if cache is None:
            cache = {}
            try:
                object.__setattr__(morpho, "_h01_cv_bounds_cache", cache)
            except (AttributeError, TypeError):
                pass
        key = (id(self), id(paint_rules))
        if key in cache:
            return cache[key]

        base_bounds = self.base.resolve_cv_bounds(morpho, paint_rules=paint_rules)
        region_intervals = self.region.evaluate(morpho).intervals
        if not region_intervals:
            cache[key] = base_bounds
            return base_bounds

        modified_branches = {}
        for branch, lo, hi in region_intervals:
            if branch not in modified_branches:
                modified_branches[branch] = {p for interval in base_bounds[branch] for p in interval}
            modified_branches[branch].update((lo, hi))

        result = list(base_bounds)
        for branch, branch_cuts in modified_branches.items():
            points = [0.]
            for point in sorted(branch_cuts):
                # BrainCell requires normalized CV widths greater than 1e-9.
                # Source coordinate roundoff can otherwise add a cut beside 1.
                if point - points[-1] > 1e-9:
                    points.append(point)
            points[-1] = 1.
            result[branch] = tuple(zip(points[:-1], points[1:]))

        res_tuple = tuple(result)
        cache[key] = res_tuple
        return res_tuple
