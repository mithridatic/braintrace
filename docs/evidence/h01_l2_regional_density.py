"""Scale one mechanism density in named regions of the layer-2 fit."""

import copy

import numpy as np

REGIONS = ("soma", "axon", "dend", "apic")


def parse_regional_density(text):
    """Parse ``MECHANISM:REGION:FACTOR`` into its three validated parts."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Regional density must be MECHANISM:REGION:FACTOR.")
    mechanism, region, factor = parts[0], parts[1], float(parts[2])
    if region not in REGIONS+("all",):
        raise ValueError("Region must be one of soma, axon, dend, apic, or all.")
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Density factor must be non-negative and finite.")
    return {"mechanism": mechanism, "region": region, "factor": factor}


def regional_density_fit(source, mechanism, region, factor):
    """Copy the fit and scale ``gbar_<mechanism>`` rows in the selected regions.

    Parameters
    ----------
    source : dict
        Fit containing a genome list of section, mechanism, name, value rows.
    mechanism : str
        Mechanism name such as ``NaTs`` or ``Kv3_1``.
    region : str
        One of soma, axon, dend, apic, or all.
    factor : float
        Non-negative finite multiplier; zero removes the boundary reversibly.

    Returns
    -------
    dict
        Independent fit with every selected density scaled.

    Raises
    ------
    ValueError
        If no row matches or a scaled value is not finite.
    """
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Density factor must be non-negative and finite.")
    regions = REGIONS if region == "all" else (region,)
    result = copy.deepcopy(source)
    rows = [r for r in result["genome"] if r["section"] in regions
            and r["mechanism"] == mechanism and r["name"] == "gbar_"+mechanism]
    if not rows:
        raise ValueError(f"No {mechanism} density row in region {region}.")
    for row in rows:
        row["value"] *= factor
        if not np.isfinite(row["value"]):
            raise ValueError("Scaled density must be finite.")
    return result
