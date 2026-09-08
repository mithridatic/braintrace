"""Verify Kv3 timing against a closed-form fixed-voltage response."""

import itertools
import json
from pathlib import Path

import numpy as np
from neuron import h

h.load_file("stdrun.hoc")
cells = []
for voltage, initial, factor in itertools.product((-80., 20.), (0., 1.), (1., .5)):
    section = h.Section()
    section.insert("Kv3_1")
    section.gbar_Kv3_1 = 0.
    section.m_tau_factor_Kv3_1 = factor
    clamp = h.SEClamp(section(.5))
    clamp.dur1, clamp.amp1, clamp.rs = 2., voltage, 1e-6
    cells.append((section, clamp, voltage, initial, factor))
h.CVode().active(1)
h.CVode().atol(1e-12)
h.finitialize(-80.)
for section, clamp, voltage, initial, factor in cells:
    section.v = voltage
    section(.5).Kv3_1.m = initial
h.fcurrent()
h.CVode().re_init()
h.CVode().solve(.5)
assert abs(h.t - .5) < 1e-12
records = []
for section, clamp, voltage, initial, factor in cells:
    equilibrium = 1. / (1. + np.exp(-(voltage-18.7)/9.7))
    tau = factor * 4. / (1. + np.exp(-(voltage+46.56)/44.14))
    expected = equilibrium + (initial-equilibrium)*np.exp(-.5/tau)
    observed = section(.5).Kv3_1.m
    assert abs(section(.5).v-voltage) < 1e-8
    assert abs(observed-expected) < 1e-8
    records.append({"voltage_mv": voltage, "initial_m": initial, "tau_factor": factor,
                    "expected_m": float(expected), "observed_m": float(observed),
                    "absolute_error": float(abs(observed-expected))})
Path("/evidence/h01-pv-kv3-kinetics-clamp.json").write_text(json.dumps(records, indent=2))
print("Passed eight fixed-voltage opening and closing cases.")
