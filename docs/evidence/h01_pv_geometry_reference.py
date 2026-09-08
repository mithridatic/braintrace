"""Export the original NEURON template geometry without advancing a model."""

import json
from pathlib import Path

from neuron import h

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
h.load_file("/work/source/NeuronTemplate.hoc")
h.load_file("/work/source/biophys_HL5BN1.hoc")
cell = h.NeuronTemplate("/work/source/HL5BN1.swc")
h.biophys_HL5BN1(cell)
sections = []
for sec in cell.all:
    parent = sec.parentseg()
    sections.append({
        "name": sec.name().split(".", 1)[1],
        "parent": None if parent is None else parent.sec.name().split(".", 1)[1],
        "parent_x": None if parent is None else parent.x,
        "child_x": sec.orientation(), "length_um": sec.L,
        "diameter_um": sec.diam, "nseg": sec.nseg,
        "area_um2": sum(seg.area() for seg in sec),
        "arc_um": [sec.arc3d(i) for i in range(sec.n3d())],
        "diameter3d_um": [sec.diam3d(i) for i in range(sec.n3d())],
        "segment_centers": [{"x": seg.x, "area_um2": seg.area(),
                             "ri_megohm": seg.ri()} for seg in sec],
    })
report = {"source_commit": "82cdd91bc93942ba19315371330a2412e064baf5",
          "geometry": "NEURON Import3d plus original template axon replacement",
          "sections": sections}
Path("/evidence/h01-pv-geometry-reference.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"sections": len(sections), "segments": sum(s["nseg"] for s in sections),
                  "area_um2": sum(s["area_um2"] for s in sections)}))
