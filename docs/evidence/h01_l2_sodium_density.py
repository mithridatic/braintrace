"""Apply the declared somatic sodium-density intervention."""

import copy

import numpy as np


def sodium_density_fit(source, factor=1.):
    """Copy the fit and scale only somatic transient sodium density.

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
        raise ValueError("Sodium-density factor must be positive and finite.")
    result = copy.deepcopy(source)
    rows = [r for r in result["genome"] if r["section"] == "soma"
            and r["mechanism"] == "NaTs" and r["name"] == "gbar_NaTs"]
    if len(rows) != 1 or not np.isfinite(rows[0]["value"]) or rows[0]["value"] <= 0:
        raise ValueError("Exactly one positive finite somatic sodium density is required.")
    rows[0]["value"] *= factor
    if not np.isfinite(rows[0]["value"]):
        raise ValueError("Scaled sodium density must be finite.")
    return result
