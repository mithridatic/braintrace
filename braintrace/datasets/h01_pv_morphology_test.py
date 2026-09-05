"""Independent section geometry and invalid topology checks."""

from copy import deepcopy
import json
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np
import pytest

from .h01_pv_morphology import make_pv_morphology


REFERENCE = json.loads((Path(__file__).resolve().parents[2] /
                       "docs/evidence/h01-pv-geometry-reference.json").read_text())


def test_every_section_preserves_neuron_length_area_and_attachment():
    with brainstate.environ.context(precision=64):
        morph = make_pv_morphology(REFERENCE)
        branches = {b.name.rsplit("_", 1)[0]+"["+b.name.rsplit("_", 1)[1]+"]": b for b in morph.branches}
        assert len(branches) == 29
        for section in REFERENCE["sections"]:
            node = branches[section["name"]]
            np.testing.assert_allclose(node.branch.length.to_decimal(u.um), section["length_um"], rtol=1e-12)
            np.testing.assert_allclose(node.branch.area.to_decimal(u.um**2), section["area_um2"], rtol=1e-10)
        edges = {edge.child.name: edge for edge in morph.edges}
        for section in REFERENCE["sections"][1:]:
            edge = edges[section["name"].replace("[", "_").replace("]", "")]
            assert edge.parent.name == section["parent"].replace("[", "_").replace("]", "")
            assert edge.parent_x == section["parent_x"]
            assert edge.child_x == section["child_x"]
        np.testing.assert_allclose(morph.total_area.to_decimal(u.um**2), 5746.131785301348, rtol=1e-10)


@pytest.mark.parametrize("change,error", [
    ("duplicate", "unique"), ("parent", "missing parent"),
    ("cycle", "cycle"), ("type", "Unsupported"), ("diameter", "Invalid"),
    ("arc", "Invalid"),
])
def test_invalid_source_geometry_is_rejected(change, error):
    reference = deepcopy(REFERENCE)
    sections = reference["sections"]
    if change == "duplicate":
        sections.append(sections[-1])
    elif change == "parent":
        sections[1]["parent"] = "missing"
    elif change == "cycle":
        sections[1]["parent"] = sections[2]["name"]
        sections[2]["parent"] = sections[1]["name"]
    elif change == "type":
        sections[-1]["name"] = "unknown[0]"
    elif change == "diameter":
        sections[0]["diameter3d_um"][0] = -1.
    else:
        sections[0]["arc_um"][0] = 1.
    with pytest.raises(ValueError, match=error):
        make_pv_morphology(reference)
