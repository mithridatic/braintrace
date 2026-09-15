"""Diagnostic split for the transfer block: scale only the dendritic passive load of a donor profile.

Evidence-only. The deployed profile is unchanged in production; this script wraps
``h01_anatomy_transfer_run`` and multiplies the ``dend``/``apic`` leak conductance
and capacitance of the resolved donor profile by ``--dend-load-scale`` before the
cell is painted. It asks one question of the retained H01 anatomy: is the
depolarised stable state that follows the first spikes a consequence of the
truncated component's small passive load? It does not qualify any anatomy or
profile and must never be promoted.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_anatomy_transfer_run as transfer  # noqa: E402
from braintrace.datasets import h01_ei_cell  # noqa: E402

PASSIVE_FAMILIES = ("dend", "apic")


def scale_passive_load(profile, scale):
    """Return the profile with dend/apic capacitance and leak multiplied by ``scale``."""
    if not scale > 0:
        raise ValueError("scale must be positive")
    regions = tuple((family, cm*scale if family in PASSIVE_FAMILIES else cm,
                     leak*scale if family in PASSIVE_FAMILIES else leak, channels, calcium)
                    for family, cm, leak, channels, calcium in profile.regions)
    return replace(profile, regions=regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dend-load-scale", type=float, required=True)
    args, passthrough = parser.parse_known_args()
    original = h01_ei_cell.get_donor_profile
    h01_ei_cell.get_donor_profile = lambda donor, mode="candidate": scale_passive_load(original(donor, mode=mode), args.dend_load_scale)
    sys.argv = [sys.argv[0], *passthrough]
    transfer.main()


if __name__ == "__main__":
    main()
