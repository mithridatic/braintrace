"""Apply the declared somatic Ih-density intervention."""

import copy

import numpy as np


def ih_density_fit(source, factor=1.):
    """Copy the fit and scale only somatic transient Ih density.

    Parameters
    ----------
    source : dict
        Source fit containing a genome list.
    factor : float, optional
        Positive finite density multiplier.

    Returns
    -------
    dict
        Independent fit with the selected density scaled.

    Raises
    ------
    ValueError
        If the factor, unique target, or resulting density is invalid.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Ih-density factor must be positive and finite.")
    result = copy.deepcopy(source)
    rows = [r for r in result["genome"] if r["section"] == "soma"
            and r["mechanism"] == "Ih" and r["name"] == "gbar_Ih"]
    if len(rows) != 1 or not np.isfinite(rows[0]["value"]) or rows[0]["value"] <= 0:
        raise ValueError("Exactly one positive finite somatic Ih density is required.")
    rows[0]["value"] *= factor
    if not np.isfinite(rows[0]["value"]):
        raise ValueError("Scaled Ih density must be finite.")
    return result
