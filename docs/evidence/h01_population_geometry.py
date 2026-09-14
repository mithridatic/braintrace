"""Verify source-branch conservation through the production compartment mesh."""

import numpy as np


def validate_cv_partition(branches, cvs):
    """Check coverage, area and length independently for every source branch.

    Parameters
    ----------
    branches : array-like
        Rows of source lateral area in um2 and cable length in um.
    cvs : array-like
        Rows of branch ID, normalized proximal/distal coordinates, lateral
        area in um2 and cable length in um. Row order is irrelevant.

    Returns
    -------
    dict
        Counts and largest relative area/length discrepancies. Invalid or
        nonconservative partitions raise ValueError without a passing result.
    """
    b, c = np.asarray(branches, dtype=float), np.asarray(cvs, dtype=float)
    if (b.ndim != 2 or b.shape[1] != 2 or not len(b) or not np.isfinite(b).all()
            or np.any(b <= 0) or c.ndim != 2 or c.shape[1] != 5 or not len(c)
            or not np.isfinite(c).all() or np.any(c[:,3:] <= 0)
            or np.any(c[:,0] != np.floor(c[:,0])) or np.any(c[:,0] < 0)
            or np.any(c[:,0] >= len(b)) or np.any(c[:,1] < 0) or np.any(c[:,2] > 1)
            or np.any(c[:,2] <= c[:,1])):
        raise ValueError('Invalid source branch or CV observations.')
    c = c[np.lexsort((c[:,1], c[:,0]))]
    ids = c[:,0].astype(int)
    first = np.r_[True, ids[1:] != ids[:-1]]
    last = np.r_[ids[1:] != ids[:-1], True]
    if (not np.array_equal(ids[first], np.arange(len(b)))
            or not np.allclose(c[first,1], 0, rtol=0, atol=1e-12)
            or not np.allclose(c[last,2], 1, rtol=0, atol=1e-12)
            or not np.allclose(c[~last,2], c[~first,1], rtol=0, atol=1e-12)):
        raise ValueError('CV coverage has missing branches, gaps or overlaps.')
    totals = np.column_stack([np.bincount(ids, weights=c[:,k], minlength=len(b)) for k in (3,4)])
    error = np.abs(totals-b)/b
    if np.any(error > 1e-8):
        raise ValueError('Per-branch area or length is not conserved.')
    return dict(branch_count=len(b), cv_count=len(c),
                maximum_relative_area_error=float(error[:,0].max()),
                maximum_relative_length_error=float(error[:,1].max()))
