"""Check separate sodium closure and recovery rates at fixed voltage."""

import itertools
import json
from pathlib import Path

import numpy as np
from neuron import h

h.load_file("stdrun.hoc")
cells = []
for voltage, initial, closing, recovery in itertools.product(
        (-80., 20.), (0., 1.), (1., .5), (1., .5)):
    section = h.Section()
    section.insert("NaTg")
    section.gbar_NaTg = 0.
    section.vshifth_NaTg = 10.
    section.slopem_NaTg = 9.
    section.h_tau_factor_NaTg = closing
    section.h_recovery_factor_NaTg = recovery
    clamp = h.SEClamp(section(.5))
    clamp.dur1, clamp.amp1, clamp.rs = 2., voltage, 1e-6
    cells.append((section, clamp, voltage, initial, closing, recovery))
h.CVode().active(1)
h.CVode().atol(1e-12)
h.finitialize(-80.)
for section, clamp, voltage, initial, closing, recovery in cells:
    section.v = voltage
    section(.5).NaTg.m = 0.
    section(.5).NaTg.h = initial
h.fcurrent()
h.CVode().re_init()
h.CVode().solve(.5)
assert abs(h.t-.5) < 1e-12
records = []
for section, clamp, voltage, initial, closing, recovery in cells:
    delta = voltage+56.
    alpha = -.015*delta/(1-np.exp(delta/6.))
    beta = .015*delta/(1-np.exp(-delta/6.))
    equilibrium = alpha/(alpha+beta)
    factor = recovery if initial < equilibrium else closing
    tau = factor/(alpha+beta)/(2.3**1.3)
    expected = equilibrium+(initial-equilibrium)*np.exp(-.5/tau)
    observed = section(.5).NaTg.h
    assert abs(observed-expected) < 1e-8, (observed, expected)
    assert abs(section(.5).v-voltage) < 1e-8
    records.append({"voltage_mv": voltage, "initial_h": initial,
                    "closing_factor": closing, "recovery_factor": recovery,
                    "observed_h": observed, "expected_h": expected})
Path("/evidence/h01-pv-recovery-clamp.json").write_text(json.dumps(records, indent=2))
print("Passed all 16 matched-voltage closure/recovery checks at the exact endpoint.")
