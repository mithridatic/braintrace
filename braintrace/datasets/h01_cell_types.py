"""Cell types of the H01 population beyond the E/I sign, and donor matching.

The released H01 tags carry a layer and a morphology class for every cell. The
network builder reduces them to a Dale sign and assigns one of two frozen donor
profiles. This module keeps the full type and says, per cell, whether the donor
that would be applied matches it in layer and class, so that a population built
from the two profiles can be described honestly.
"""

from dataclasses import dataclass

LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6", "WM")
CLASSES = {"pyramidal": "E", "excitatory/spiny-with-atypical-tree": "E", "interneuron": "I"}
MODIFIERS = ("bipolar", "sparsely-spiny", "web-like-interneuron", "lot-of-axon")
DONORS = {
    "E": {"layer": "L2", "cell_class": "pyramidal", "subtype": "regular-spiking pyramidal",
          "source": "Allen specimen 541563728, model 626170538 (layer 2 pyramidal)",
          "profile": "h01-l2-kv3-ninety-ca133"},
    "I": {"layer": "L5", "cell_class": "interneuron", "subtype": "parvalbumin basket",
          "source": "ModelDB 267587 HL5BN1 (human layer 5 basket neuron; layer read from the model name)",
          "profile": "h01-pv-regional-mesh-axon2187"},
}


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


def donor_match(kind):
    """Compare a cell type with the donor profile the builder would assign to it.

    Returns
    -------
    dict
        ``donor`` (the profile name), ``match`` (``matched`` or the comma-joined
        mismatches among ``layer``, ``class``, ``modifier``) and ``note``.
    """
    donor = DONORS[kind.polarity]
    mismatches = []
    if kind.layer != donor["layer"]:
        mismatches.append("layer")
    if kind.cell_class != donor["cell_class"]:
        mismatches.append("class")
    if kind.modifiers:
        mismatches.append("modifier")
    note = ("interneuron subtype is not in the tags; the parvalbumin donor is assumed"
            if kind.polarity == "I" and kind.cell_class == "interneuron" else "")
    return {"donor": donor["profile"], "match": ", ".join(mismatches) or "matched", "note": note}


def type_rows(nodes):
    """One row per node: cell id, type fields and donor match."""
    rows = []
    for node in nodes:
        kind = cell_type(node["tags"])
        rows.append({"cell_id": node["cell_id"], "layer": kind.layer, "cell_class": kind.cell_class,
                     "modifiers": list(kind.modifiers), "polarity": kind.polarity, **donor_match(kind)})
    return rows
