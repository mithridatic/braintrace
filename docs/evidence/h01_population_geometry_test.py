"""Per-branch conservation and exact CV coverage checks."""

import numpy as np
import pytest

from h01_population_geometry import validate_cv_partition


BRANCHES = np.array([[6., 3.], [10., 5.]])
CVS = np.array([[0, 0, .5, 3, 1.5], [0, .5, 1, 3, 1.5], [1, 0, 1, 10, 5.]])


def test_reordered_partition_preserves_each_branch():
    report = validate_cv_partition(BRANCHES, CVS[::-1])
    assert report['cv_count'] == 3 and report['branch_count'] == 2
    assert report['maximum_relative_area_error'] == 0


@pytest.mark.parametrize('fault', ['cancel', 'gap', 'overlap', 'missing', 'nan',
                                  'negative', 'fractional_id', 'range', 'end', 'start', 'shape'])
def test_invalid_partition(fault):
    cvs = CVS.copy()
    if fault == 'cancel':cvs[0,3] += 1; cvs[2,3] -= 1
    if fault == 'gap':cvs[1,1] += .1
    if fault == 'overlap':cvs[1,1] -= .1
    if fault == 'missing':cvs = cvs[:2]
    if fault == 'nan':cvs[0,3] = np.nan
    if fault == 'negative':cvs[0,4] = -1
    if fault == 'fractional_id':cvs[0,0] = .2
    if fault == 'range':cvs[0,0] = 2
    if fault == 'end':cvs[2,2] = .9
    if fault == 'start':cvs[0,1] = .1
    if fault == 'shape':cvs = cvs[:,:4]
    with pytest.raises(ValueError):validate_cv_partition(BRANCHES, cvs)


@pytest.mark.parametrize('branches', [np.empty((0,2)), BRANCHES[:,:1], BRANCHES*np.nan,
                                      np.array([[0.,3.],[10.,5.]])])
def test_invalid_source_branch_observations(branches):
    with pytest.raises(ValueError):validate_cv_partition(branches, CVS)
