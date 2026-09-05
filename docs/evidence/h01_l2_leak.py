"""Scale passive leak without changing source regional ratios."""

import copy

import numpy as np


def leak_fit(source, factor=1.):
    """Copy the fit and scale the four regional passive densities.

    Parameters
    ----------
    source : dict
        Source fit with a genome list.
    factor : float, optional
        Positive finite common multiplier.

    Returns
    -------
    dict
        Independent fit with only the selected densities changed.

    Raises
    ------
    ValueError
        If the factor or regional target set is invalid.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Leak factor must be positive and finite.")
    result = copy.deepcopy(source)
    rows = [r for r in result["genome"] if r["name"] == "g_pas"]
    if (len(rows) != 4 or {r["section"] for r in rows} != {"soma", "axon", "dend", "apic"}
            or any(r["mechanism"] != "" or not np.isfinite(r["value"]) or r["value"] <= 0 for r in rows)):
        raise ValueError("Four unique positive source regional leak densities are required.")
    for row in rows:
        row["value"] *= factor
        if not np.isfinite(row["value"]):
            raise ValueError("Scaled leak density must be finite.")
    return result
