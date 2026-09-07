"""Probe: does iterating cell.myelin on the HL5BN1 template touch deleted sections?"""
import faulthandler
import sys

faulthandler.enable()
from neuron import h

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
h.load_file("/work/source/NeuronTemplate.hoc")
h.load_file("/work/source/biophys_HL5BN1.hoc")
cell = h.NeuronTemplate("/work/source/HL5BN1.swc")
h.biophys_HL5BN1(cell)
print("all:", len(list(cell.all)), "axonal:", len(list(cell.axonal)), flush=True)
print("myelin count via hoc:", int(cell.myelin.count()) if hasattr(cell.myelin, "count") else "n/a", flush=True)
names = [s.name() for s in cell.all]
print("axon-family names in all:", [n for n in names if "axon" in n or "myelin" in n][:8], flush=True)
step = sys.argv[1] if len(sys.argv) > 1 else "iterate"
if step == "iterate":
    n = 0
    for m in cell.myelin:
        n += 1
        print("myelin section", n, flush=True)
        try:
            print("  name", m.name(), flush=True)
            print("  parentseg", m.parentseg(), flush=True)
        except Exception as error:
            print("  python exception:", type(error).__name__, error, flush=True)
    print("iterated", n, flush=True)
print("probe done", flush=True)
