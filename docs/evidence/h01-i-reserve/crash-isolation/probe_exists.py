from neuron import h
h.load_file("stdrun.hoc"); h.load_file("import3d.hoc")
h.load_file("/work/source/NeuronTemplate.hoc"); h.load_file("/work/source/biophys_HL5BN1.hoc")
cell = h.NeuronTemplate("/work/source/HL5BN1.swc"); h.biophys_HL5BN1(cell)
print("section_exists myelin 0:", h.section_exists("myelin", 0, cell))
print("section_exists myelin 1:", h.section_exists("myelin", 1, cell))
print("section_exists axon 0:", h.section_exists("axon", 0, cell), "axon 1:", h.section_exists("axon", 1, cell), "axon 2:", h.section_exists("axon", 2, cell))
print("section_exists soma 0:", h.section_exists("soma", 0, cell))
