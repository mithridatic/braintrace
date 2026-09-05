"""Active-region boundaries must remain explicit in the electrical mesh."""

import braincell
import brainunit as u
import numpy as np

from .h01_anatomy_test import imported
from .h01_discretization import BoundaryAlignedCV
from braincell.filter import RegionExpr, RegionMask


def test_nearly_identical_cuts_do_not_create_degenerate_cvs(imported):
    class RoundoffRegion(RegionExpr):
        def evaluate(self, morpho, cache=None):
            return RegionMask(((0, 1e-16, 1.-2e-16),))
    bounds = BoundaryAlignedCV(braincell.MaxCVLen(10*u.um), RoundoffRegion()).resolve_cv_bounds(imported.morphology)
    assert all(hi-lo > 1e-12 for branch in bounds for lo,hi in branch)


def test_aligned_policy_preserves_base_cuts_and_region_boundaries(imported):
    region = imported.anatomy().cable_neighborhood(30, radius_um=1.)
    base = braincell.MaxCVLen(10*u.um)
    bounds = BoundaryAlignedCV(base, region).resolve_cv_bounds(imported.morphology)
    original = base.resolve_cv_bounds(imported.morphology)
    for intervals, old in zip(bounds, original):
        assert intervals[0][0] == 0 and intervals[-1][1] == 1
        assert all(a < b for a,b in intervals)
        np.testing.assert_allclose([b for a,b in intervals[:-1]], [a for a,b in intervals[1:]])
        points = {p for interval in intervals for p in interval}
        assert {p for interval in old for p in interval} <= points
    for branch, lo, hi in region.intervals:
        points = {p for interval in bounds[branch] for p in interval}
        assert lo in points and hi in points


def test_alignment_with_whole_tree_does_not_add_empty_intervals(imported):
    region = imported.anatomy().cable_neighborhood(30, radius_um=100.)
    bounds = BoundaryAlignedCV(braincell.MaxCVLen(1*u.um), region).resolve_cv_bounds(imported.morphology)
    assert all(b > a for branch in bounds for a,b in branch)
