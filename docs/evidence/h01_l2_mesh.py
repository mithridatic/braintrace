"""Apply an explicit odd refinement to the source section count."""

import numpy as np


def section_segments(length_um, factor=1):
    """Compute the source count and apply a positive odd refinement.

    Parameters
    ----------
    length_um : float
        Finite positive section length in micrometers.
    factor : int, optional
        Positive odd multiplier of the original source count.

    Returns
    -------
    int
        Odd number of segments for this section.

    Raises
    ------
    ValueError
        If the length or factor is invalid.
    """
    if not np.isfinite(length_um) or length_um <= 0:
        raise ValueError("Section length must be positive and finite.")
    if (isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer))
            or factor < 1 or factor % 2 != 1):
        raise ValueError("Refinement factor must be a positive odd integer.")
    return (1 + 2 * int(length_um / 40.)) * int(factor)
