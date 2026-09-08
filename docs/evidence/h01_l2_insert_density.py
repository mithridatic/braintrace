"""Insert one mechanism at an absolute density into a region of the layer-2 fit.

The complement of :mod:`h01_l2_regional_density`, which scales rows that exist:
this adds a ``gbar_<mechanism>`` genome row to a region that has none, and a
reversal-potential row for that region cloned from the soma, because the Allen
fit names reversal potentials for the soma only and NEURON would otherwise
leave the new region at its default sodium reversal.
"""

import copy

import numpy as np

from h01_l2_regional_density import REGIONS


def parse_insert_density(text):
    """Parse ``MECHANISM:REGION:VALUE`` (value in S/cm2) into validated parts."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Insert density must be MECHANISM:REGION:VALUE.")
    mechanism, region, value = parts[0], parts[1], float(parts[2])
    if region not in REGIONS:
        raise ValueError("Region must be one of soma, axon, dend, apic.")
    if not np.isfinite(value) or value < 0:
        raise ValueError("Inserted density must be non-negative and finite.")
    return {"mechanism": mechanism, "region": region, "value": value}


def _erev_rows(fit):
    """The reversal-potential rows of a fit whose conditions are a list or a dict."""
    conditions = fit.get("conditions")
    if isinstance(conditions, list):
        conditions = conditions[0] if conditions else {}
    return (conditions or {}).get("erev", [])


def insert_density_fit(source, mechanism, region, value):
    """Copy the fit and add a ``gbar_<mechanism>`` row (and reversal row) to ``region``.

    Parameters
    ----------
    source : dict
        Fit with a genome list and, optionally, ``conditions`` with ``erev`` rows.
    mechanism : str
        Mechanism name such as ``NaTs``.
    region : str
        One of soma, axon, dend, apic.
    value : float
        Absolute density in S/cm2, non-negative and finite.

    Returns
    -------
    dict
        Independent fit with the new row appended.

    Raises
    ------
    ValueError
        If the region already carries a density row for the mechanism.
    """
    if region not in REGIONS:
        raise ValueError("Region must be one of soma, axon, dend, apic.")
    if not np.isfinite(value) or value < 0:
        raise ValueError("Inserted density must be non-negative and finite.")
    result = copy.deepcopy(source)
    name = "gbar_"+mechanism
    if any(r["section"] == region and r["name"] == name for r in result["genome"]):
        raise ValueError(f"{mechanism} already has a density row in region {region}; scale it instead.")
    result["genome"].append({"section": region, "mechanism": mechanism, "name": name, "value": value})
    erev = _erev_rows(result)
    if erev and not any(r["section"] == region for r in erev):
        erev.append(dict(next(r for r in erev if r["section"] == "soma"), section=region))
    return result
