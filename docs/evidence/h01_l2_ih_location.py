"""Redistribute Ih while preserving integrated maximum conductance."""

import copy

import numpy as np


def distribute_ih(source, areas):
    """Return a copied fit with uniform soma and dendrite Ih density.

    Parameters
    ----------
    source : dict
        Fit with exactly one soma-only Ih target.
    areas : dict
        Soma, dend, and apic membrane areas in square micrometers.

    Returns
    -------
    tuple of dict
        Modified fit and conductance conservation record.

    Raises
    ------
    ValueError
        If source placement or regional areas are invalid.
    """
    regions = ("soma", "dend", "apic")
    if set(areas) != set(regions) or any(not np.isfinite(a) or a <= 0 for a in areas.values()):
        raise ValueError("Three positive finite regional areas are required.")
    result = copy.deepcopy(source)
    rows = [r for r in result["genome"] if r["mechanism"] == "Ih" or r["name"] == "gbar_Ih"]
    if (len(rows) != 1 or rows[0]["section"] != "soma" or rows[0]["mechanism"] != "Ih"
            or rows[0]["name"] != "gbar_Ih" or not np.isfinite(rows[0]["value"]) or rows[0]["value"] <= 0):
        raise ValueError("Exactly one positive finite soma-only Ih target is required.")
    total = rows[0]["value"] * areas["soma"] * 1e-8
    density = total / (sum(areas.values()) * 1e-8)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Redistributed density must be positive and finite.")
    rows[0]["value"] = density
    result["genome"].extend(dict(rows[0], section=s) for s in regions[1:])
    return result, {"areas_um2": areas, "source_maximum_conductance_s": total,
                    "distributed_maximum_conductance_s": density * sum(areas.values()) * 1e-8,
                    "uniform_density_s_cm2": density}
