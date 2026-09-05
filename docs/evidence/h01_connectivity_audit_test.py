"""Reject insufficient or contradictory spatial identity evidence."""
import pytest
from docs.evidence.h01_connectivity_audit import resolve_identity


@pytest.mark.parametrize('labels,points,expected', [
    (['7', '7', '7'], [(1, 2, 3), (2, 2, 3), (3, 2, 3)], '7'),
    (['7', '7', '7'], [(1, 2, 3)]*3, None),
    (['0', '7', '7'], [(1, 2, 3), (2, 2, 3), (3, 2, 3)], None),
    (['7', '7', '8'], [(1, 2, 3), (2, 2, 3), (3, 2, 3)], None),
    (['8', '8', '8'], [(1, 2, 3), (2, 2, 3), (3, 2, 3)], None),
    ([], [], None),
])
def test_resolve_identity(labels, points, expected):
    witnesses = [dict(label=label, voxel=point) for label, point in zip(labels, points)]
    assert resolve_identity(witnesses, {'7'}) == expected
