"""Cell types of the H01 population beyond the E/I sign, and the donor registry.

The released H01 tags carry a layer and a morphology class for every cell. The
network builder reduces them to a Dale sign and assigns one of the frozen donor
profiles. This module keeps the full type, resolves the donor that fits it best
(``donor_for``), and says, per cell, whether that donor matches it in layer and
class, so that a population built from the registry can be described honestly.

The registry is keyed by donor key rather than by polarity, so a third human
donor is one new record here plus its physiology in ``h01_ei_profiles``.
"""

from dataclasses import dataclass

LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6", "WM")
CLASSES = {"pyramidal": "E", "excitatory/spiny-with-atypical-tree": "E", "interneuron": "I"}
MODIFIERS = ("bipolar", "sparsely-spiny", "web-like-interneuron", "lot-of-axon")
DONORS = {
    "l2-pyramidal-allen-541563728": {
        "layer": "L2", "cell_class": "pyramidal", "modifiers": (), "polarity": "E",
        "subtype": "regular-spiking pyramidal",
        "source": "Allen specimen 541563728, model 626170538 (layer 2 pyramidal)",
        "profile": "h01-l2-kv3-ninety-ca133", "channel_prefix": "H01L2",
        "sodium_reversal_mv": 53., "potassium_reversal_mv": -107.},
    "l5-pv-basket-hl5bn1": {
        "layer": "L5", "cell_class": "interneuron", "modifiers": (), "polarity": "I",
        "subtype": "parvalbumin basket",
        "source": "ModelDB 267587 HL5BN1 (human layer 5 basket neuron; layer read from the model name)",
        "profile": "h01-pv-regional-mesh-axon2187", "channel_prefix": "H01PV",
        "sodium_reversal_mv": 50., "potassium_reversal_mv": -85.},
    "l3-sst-interneuron-hl5mn1": {
        "layer": "L3", "cell_class": "interneuron", "modifiers": (), "polarity": "I",
        "subtype": "unknown (putative SST)",
        "source": "ModelDB 267587 HL5MN1 = Yao 2022 HL23SST (Allen human specimen 571700636, MTG layer 3, aspiny)",
        "profile": "h01-sst-l3-hl5mn1", "channel_prefix": "H01PV",
        "sodium_reversal_mv": 50., "potassium_reversal_mv": -85.},
    "l4-pyramidal-allen-527952884": {
        "layer": "L4", "cell_class": "pyramidal", "modifiers": (), "polarity": "E",
        "subtype": "regular-spiking pyramidal (spiny, apical truncated)",
        "source": "Allen specimen 527952884, model 626170709 (MTG layer 4 pyramidal, perisomatic fit)",
        "profile": "h01-l4-allen-527952884", "channel_prefix": "H01L2",
        "sodium_reversal_mv": 53., "potassium_reversal_mv": -107.},
}
# The polarity default is the first registered donor of each sign; later donors do not replace it.
DEFAULT_DONOR_KEYS = {"E": "l2-pyramidal-allen-541563728", "I": "l5-pv-basket-hl5bn1"}
DONORS_BY_POLARITY = {polarity: DONORS[key] for polarity, key in DEFAULT_DONOR_KEYS.items()}


@dataclass(frozen=True)
class H01CellType:
    """Layer, morphology class, modifiers and the polarity they imply."""

    layer: str
    cell_class: str
    modifiers: tuple
    polarity: str


def cell_type(tags):
    """Read one cell's released tags into a type.

    Parameters
    ----------
    tags : iterable of str
        Released H01 segment-property tags, e.g. ``["L2", "pyramidal", "neuron"]``.

    Returns
    -------
    H01CellType

    Raises
    ------
    ValueError
        If the tags carry no layer, no class, or more than one of either.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_cell_types import cell_type
        >>> cell_type(["L5", "bipolar", "neuron", "pyramidal"])
        H01CellType(layer='L5', cell_class='pyramidal', modifiers=('bipolar',), polarity='E')
    """
    tags = tuple(tags)
    layers = [t for t in tags if t in LAYERS]
    classes = [t for t in tags if t in CLASSES]
    if len(layers) != 1 or len(classes) != 1:
        raise ValueError(f"Tags must carry exactly one layer and one class: {tags}")
    modifiers = tuple(sorted(t for t in tags if t in MODIFIERS))
    return H01CellType(layers[0], classes[0], modifiers, CLASSES[classes[0]])


def donor_for(kind):
    """Resolve the donor key that fits a cell type best.

    Precedence: a donor equal in (layer, class, modifiers); then one equal in
    (layer, class) with no modifiers; then the first donor of the same class;
    then the polarity default. Registry order breaks ties.

    Parameters
    ----------
    kind : H01CellType

    Returns
    -------
    str
        A key of ``DONORS``.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_cell_types import cell_type, donor_for
        >>> donor_for(cell_type(["L4", "pyramidal", "neuron"]))
        'l4-pyramidal-allen-527952884'
        >>> donor_for(cell_type(["L5", "pyramidal", "neuron"]))
        'l2-pyramidal-allen-541563728'
        >>> donor_for(cell_type(["L3", "interneuron", "neuron"]))
        'l3-sst-interneuron-hl5mn1'
    """
    tests = (lambda d: (d["layer"], d["cell_class"], d["modifiers"]) == (kind.layer, kind.cell_class, kind.modifiers),
             lambda d: (d["layer"], d["cell_class"], d["modifiers"]) == (kind.layer, kind.cell_class, ()),
             lambda d: d["cell_class"] == kind.cell_class)
    for test in tests:
        for key, donor in DONORS.items():
            if test(donor):
                return key
    return DEFAULT_DONOR_KEYS[kind.polarity]


def donor_for_tags(tags, polarity):
    """Resolve a donor key from released tags, tolerating tags without a full type.

    Parameters
    ----------
    tags : iterable of str
        Released tags; may lack a layer (test fixtures carry only a class).
    polarity : str
        Verified Dale role, ``"E"`` or ``"I"``; the default when the tags do
        not read as a type.

    Returns
    -------
    str
        A key of ``DONORS``.
    """
    try:
        return donor_for(cell_type(tags))
    except ValueError:
        return DEFAULT_DONOR_KEYS[polarity]


def donor_match(kind):
    """Compare a cell type with the donor ``donor_for`` resolves for it.

    Returns
    -------
    dict
        ``donor`` (the profile name), ``donor_key``, ``match`` (``matched`` or
        the comma-joined mismatches among ``layer``, ``class``, ``modifier``)
        and ``note``.
    """
    key = donor_for(kind)
    donor = DONORS[key]
    mismatches = []
    if kind.layer != donor["layer"]:
        mismatches.append("layer")
    if kind.cell_class != donor["cell_class"]:
        mismatches.append("class")
    if kind.modifiers != donor["modifiers"]:
        mismatches.append("modifier")
    note = (f"interneuron subtype is not in the tags; the {donor['subtype']} donor is assumed"
            if kind.polarity == "I" and kind.cell_class == "interneuron" else "")
    return {"donor": donor["profile"], "donor_key": key,
            "match": ", ".join(mismatches) or "matched", "note": note}


def type_rows(nodes):
    """One row per node: cell id, type fields, donor key and donor match."""
    rows = []
    for node in nodes:
        kind = cell_type(node["tags"])
        rows.append({"cell_id": node["cell_id"], "layer": kind.layer, "cell_class": kind.cell_class,
                     "modifiers": list(kind.modifiers), "polarity": kind.polarity, **donor_match(kind)})
    return rows
