"""Apply an explicit passive reversal shift without changing other fit values."""

import copy

import numpy as np


def shift_leak_reversal(source, shift_mv=0.):
    """Return a copied fit with its single passive reversal shifted.

    Parameters
    ----------
    source : dict
        Fit with one passive block and no genome reversal overrides.
    shift_mv : float, optional
        Finite additive shift in millivolts.

    Returns
    -------
    dict
        Independent modified fit.

    Raises
    ------
    ValueError
        If the source reversal, shift, or resulting value is invalid.
    """
    if not np.isfinite(shift_mv):
        raise ValueError("Leak reversal shift must be finite.")
    result = copy.deepcopy(source)
    if (len(result["passive"]) != 1 or
            any(r["name"] == "e_pas" for r in result["genome"]) or
            not np.isfinite(result["passive"][0].get("e_pas", np.nan))):
        raise ValueError("Exactly one finite passive reversal without genome overrides is required.")
    result["passive"][0]["e_pas"] += shift_mv
    if not np.isfinite(result["passive"][0]["e_pas"]):
        raise ValueError("Shifted reversal must be finite.")
    return result
