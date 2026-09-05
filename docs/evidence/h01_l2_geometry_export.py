"""Export the pinned Allen template geometry without running a response."""
import json
from pathlib import Path
from neuron import h
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
swc = h.Import3d_SWC_read()
swc.input("/cache/source-model/morphology.swc")
importer = h.Import3d_GUI(swc, 0)
h("objref this")
importer.instantiate(h.this)
h("soma[0] area(0.5)")
for sec in list(h.allsec()):
    if sec.name().startswith("axon"): h.delete_section(sec=sec)
h("create axon[2]")
for sec in h.axon:
    sec.L, sec.diam = 30., 1.
h.axon[0].connect(h.soma[0], .5, 0.)
h.axon[1].connect(h.axon[0], 1., 0.)
h.define_shape()
rows = []
for sec in h.allsec():
    parent = sec.parentseg()
    rows.append(dict(name=sec.name(), parent=None if parent is None else parent.sec.name(),
        parent_x=None if parent is None else parent.x, child_x=sec.orientation(),
        length_um=sec.L, diameter_um=sec(.5).diam,
        arc_um=[sec.arc3d(i) for i in range(sec.n3d())],
        diameter3d_um=[sec.diam3d(i) for i in range(sec.n3d())]))
Path("/evidence/h01-l2-geometry-reference.json").write_text(json.dumps(dict(
    source="Allen 541563728 model 626170538; original axon replacement", sections=rows), indent=2)+"\n")
print(len(rows), "sections exported; no response observed")
