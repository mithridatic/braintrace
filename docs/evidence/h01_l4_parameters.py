"""Turn an Allen perisomatic ``fit_parameters.json`` into the ``_h01_ei_parameters`` region tuples.

The Allen fit lists ``cm`` per section under ``passive``, and one genome row per parameter:
``gbar_<mechanism>`` densities, ``decay_CaDynamics`` / ``gamma_CaDynamics``, and ``g_pas``
rows with an empty mechanism. The output is
``(family, cm, g_pas, ((mechanism, density), ...), (decay, gamma) | None)`` for soma, axon,
dend and apic, channels in genome order. Applied to the cached L2 fit it must reproduce the
committed ``E_SOURCE``; applied to the L4 fit it defines ``L4_ALLEN_527952884_SOURCE``.
Specification: docs/specs/2026-09-07-h01-donor-allen-l4-import.md.
"""

import json
import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT/".cache/human-pyramidal-l4/source-model/fit_parameters.json"
L2_FIT = ROOT.parent/"h01-braincell/.cache/human-pyramidal-l2/541563728_fit.json"
FAMILIES = ("soma", "axon", "dend", "apic")


def region_tuples(fit):
    """The four region tuples read from a parsed Allen fit."""
    capacitance = {row["section"]: float(row["cm"]) for row in fit["passive"][0]["cm"]}
    leak = {row["section"]: float(row["value"]) for row in fit["genome"] if row["name"] == "g_pas"}
    regions = []
    for family in FAMILIES:
        if family not in capacitance or family not in leak:
            raise ValueError(f"No cm or g_pas for {family}.")
        rows = [row for row in fit["genome"] if row["section"] == family and row["mechanism"]]
        channels = []
        calcium = {}
        for row in rows:
            if row["mechanism"] == "CaDynamics":
                calcium[row["name"]] = float(row["value"])
            elif row["name"] == "gbar_"+row["mechanism"]:
                channels.append((row["mechanism"], float(row["value"])))
            else:
                raise ValueError(f"Unsupported genome row {row['name']} for {row['mechanism']}.")
        decay = None
        if calcium:
            decay = (calcium["decay_CaDynamics"], calcium["gamma_CaDynamics"])
        regions.append((family, capacitance[family], leak[family], tuple(channels), decay))
    return tuple(regions)


def physiology(fit):
    """Initial voltage, leak reversal, axial resistivity, reversals and temperature of the fit."""
    passive, conditions = fit["passive"][0], fit["conditions"][0]
    return {"initial_mv": conditions["v_init"], "e_pas_mv": passive["e_pas"], "ra_ohm_cm": passive["ra"],
            "celsius": conditions["celsius"], "ena_mv": conditions["erev"][0]["ena"],
            "ek_mv": conditions["erev"][0]["ek"], "junction_potential_mv": fit["fitting"][0]["junction_potential"]}


def main(argv=None):
    path = Path(argv[0]) if argv else FIT
    fit = json.loads(path.read_text())
    print("L4_ALLEN_527952884_SOURCE = "+pprint.pformat(region_tuples(fit), width=100))
    print("# physiology:", physiology(fit))


if __name__ == "__main__":
    main(sys.argv[1:])
