"""Parse ``biophys_HL5MN1.hoc`` into the ``_h01_ei_parameters`` region tuple format.

The hoc paints each ``forsec`` block (all, somatic, basal, axonal) with ``insert`` lists
and literal assignments, then sets densities through ``distribute_channels`` with
distribution type 0, slope 0 and unit scale, so every density equals the last argument.
Ih is inserted everywhere; where ``gbar_Ih`` is not set the mod-file default (1e-5 S/cm2)
applies. The output is ``(family, cm, g_pas, ((mechanism, density), ...), (decay, gamma))``
for soma, axon, dend (``basal``) and apic. Specification:
docs/specs/2026-09-07-h01-donor-hl5mn1-import.md.
"""

import pprint
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOC = ROOT/".cache/human-sst-l3/biophys_HL5MN1.hoc"
IH_DEFAULT_S_CM2 = 1e-5
FAMILIES = {"soma": "somatic", "axon": "axonal", "dend": "basal", "apic": "apical"}
ORDER = ("NaTg", "Nap", "K_P", "K_T", "Kv3_1", "Im", "SK", "Ca_HVA", "Ca_LVA", "Ih")


def blocks(text):
    """Map ``forsec $o1.<name>`` block name to its body text."""
    return {m.group(1): m.group(2) for m in re.finditer(r"forsec \$o1\.(\w+)\s*\{(.*?)\}", text, re.S)}


def assignments(body):
    """Literal ``name = value`` pairs of one block."""
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"(\w+)\s*=\s*(-?[\d.eE+-]+)", body)}


def inserted(body):
    """Mechanisms inserted in one block."""
    return re.findall(r"insert (\w+)", body)


def distributed(text):
    """``distribute_channels`` densities per (region, parameter); type 0, slope 0, scale 1 only."""
    out = {}
    pattern = r'distribute_channels\("(\w+)","(\w+)",(\d),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)'
    for region, name, kind, base, slope, a, b, scale in re.findall(pattern, text):
        if (int(kind), float(base), float(slope), float(a), float(b)) != (0, 1., 0., 0., 0.):
            raise ValueError(f"Unsupported distribution for {region} {name}.")
        out[(region, name)] = float(scale)
    return out


def region_tuples(text):
    """The four region tuples read from the hoc text."""
    parsed = blocks(text)
    common = assignments(parsed["all"])
    spread = distributed(text)
    regions = []
    for family, block in FAMILIES.items():
        body = parsed.get(block, "")
        local = assignments(body)
        mechanisms = set(inserted(parsed["all"]))|set(inserted(body))
        cm = local.get("cm", common["cm"])
        leak = local.get("g_pas", common["g_pas"])
        channels = []
        for mechanism in ORDER:
            if mechanism not in mechanisms:
                continue
            density = spread.get((family, "gbar_"+mechanism), local.get("gbar_"+mechanism))
            if density is None:
                density = common.get("gbar_"+mechanism, IH_DEFAULT_S_CM2 if mechanism == "Ih" else None)
            if density is None:
                raise ValueError(f"No density for {mechanism} in {family}.")
            channels.append((mechanism, density))
        calcium = None
        if "CaDynamics" in mechanisms:
            calcium = (spread[(family, "decay_CaDynamics")], local["gamma_CaDynamics"])
        regions.append((family, cm, leak, tuple(channels), calcium))
    return tuple(regions)


def passive(text):
    """Leak reversal and axial resistivity of the ``all`` block."""
    common = assignments(blocks(text)["all"])
    return {"e_pas_mv": common["e_pas"], "ra_ohm_cm": common["Ra"]}


def main(argv=None):
    path = Path(argv[0]) if argv else HOC
    text = path.read_text()
    print("SST_L3_HL5MN1_SOURCE = "+pprint.pformat(region_tuples(text), width=100))
    print("# passive:", passive(text))


if __name__ == "__main__":
    main(sys.argv[1:])
