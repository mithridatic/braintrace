"""Check the isolated kinetic change at matched, constant voltages in NEURON."""

import json
from pathlib import Path

import numpy as np
from neuron import h

h.load_file("stdrun.hoc")
sections, clamps, records = [], [], []
for voltage in (-80., -20., 20.):
    for factor in (1., .5):
        section = h.Section()
        section.insert("NaTg")
        section.gbar_NaTg = 0.
        section.vshifth_NaTg = 10.
        section.slopem_NaTg = 9.
        section.h_tau_factor_NaTg = factor
        clamp = h.SEClamp(section(.5))
        clamp.dur1, clamp.amp1, clamp.rs = 2., voltage, 1e-6
        sections.append(section)
        clamps.append(clamp)
        records.append((voltage, factor))
h.CVode().active(1)
h.CVode().atol(1e-12)
h.finitialize(-80.)
for section, (voltage, factor) in zip(sections, records):
    section.v = voltage
    section(.5).NaTg.m = 0.
    section(.5).NaTg.h = 0.
h.fcurrent()
h.CVode().re_init()
h.CVode().solve(.5)
observed = []
for section, (voltage, factor) in zip(sections, records):
    delta = voltage+56.
    alpha = -.015*delta/(1-np.exp(delta/6.))
    beta = .015*delta/(1-np.exp(-delta/6.))
    equilibrium = alpha/(alpha+beta)
    tau = factor/(alpha+beta)/(2.3**1.3)
    expected = equilibrium*(-np.expm1(-.5/tau))
    gate = section(.5).NaTg
    assert abs(section(.5).v-voltage) < 1e-8
    assert abs(gate.h-expected) < 1e-8, (h.t, voltage, factor, gate.h, expected)
    observed.append({"voltage_mv": voltage, "factor": factor, "h": gate.h,
                     "expected_h": expected, "h_inf": equilibrium, "h_tau_ms": tau,
                     "m": gate.m})
for index in range(0, len(observed), 2):
    assert abs(observed[index]["m"]-observed[index+1]["m"]) < 1e-10
Path("/evidence/h01-pv-inactivation-clamp.json").write_text(json.dumps(observed, indent=2))
print("Passed: h relaxation matches unchanged equilibrium and scaled tau; activation unchanged.")


