"""Apply a single explicit calcium-removal intervention to a copied fit."""

import copy

import numpy as np


def calcium_removal_fit(source, factor=1.):
    """Copy a source fit and scale only the somatic calcium-removal time.

    Parameters
    ----------
    source : dict
        Retrieved fit with a genome list.
    factor : float, optional
        Positive finite multiplier of the source removal time.

    Returns
    -------
    dict
        Independent fit with exactly one eligible decay row.

    Raises
    ------
    ValueError
        If the factor or uniquely identified source decay is invalid.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Calcium-removal factor must be positive and finite.")
    result = copy.deepcopy(source)
    rows = [row for row in result["genome"] if row["section"] == "soma"
            and row["mechanism"] == "CaDynamics" and row["name"] == "decay_CaDynamics"]
    if len(rows) != 1 or not np.isfinite(rows[0]["value"]) or rows[0]["value"] <= 0:
        raise ValueError("Exactly one positive finite somatic calcium decay is required.")
    rows[0]["value"] *= factor
    if not np.isfinite(rows[0]["value"]):
        raise ValueError("Scaled calcium-removal time must be finite.")
    return result
