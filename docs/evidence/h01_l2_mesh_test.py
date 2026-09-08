"""Check source compartment boundaries and invalid refinement factors."""

import numpy as np
import pytest

from docs.evidence.h01_l2_mesh import section_segments


@pytest.mark.parametrize("length,count", [(1., 1), (30., 1), (39.999, 1),
                                          (40., 3), (79.999, 3), (80., 5)])
@pytest.mark.parametrize("factor", [1, 3, np.int64(9)])
def test_source_boundary_and_uniform_refinement(length, count, factor):
    assert section_segments(length, factor) == count * factor


@pytest.mark.parametrize("factor", [0, -1, 2, 4, True, np.bool_(True), 3., "3"])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):
        section_segments(30., factor)


@pytest.mark.parametrize("length", [0., -1., np.nan, np.inf])
def test_invalid_length(length):
    with pytest.raises(ValueError):
        section_segments(length)
