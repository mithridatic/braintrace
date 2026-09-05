"""Check the implemented sodium inactivation slope at fixed voltage."""

import json
from pathlib import Path

import numpy as np
from neuron import h

h.load_file("stdrun.hoc")
section = h.Section()
section.insert("NaTg")
section.ena = 50.
section.gbar_NaTg = .4916517466528855 * 1.1
section.vshiftm_NaTg = 0.
section.vshifth_NaTg = 10.
section.slopem_NaTg = 9.
section.h_tau_factor_NaTg = .15
section.h_recovery_factor_NaTg = 1.
h.finitialize(-80.)
rows = []
for voltage in (-80., -70., -65., -56., -40., -30.):
    pair = []
    for slope in (6., 5.):
        section.slopeh_NaTg = slope
        # Static initialization invokes the source INITIAL gate equations.
        # Intermediate ASSIGNED variables are not exposed by this build.
        h.finitialize(voltage)
        mechanism = section(.5).NaTg
        expected = 1. / (1. + np.exp((voltage + 56.) / slope))
        error = abs(mechanism.h - expected)
        assert error < (1e-5 if voltage == -56. else 1e-10)
        delta = voltage + 56.
        rate_sum = .03*slope if delta == 0. else .015*delta/np.tanh(delta/(2.*slope))
        pair.append({"slope_mv": slope, "h_inf": float(mechanism.h),
                     "analytic_h_inf": float(expected), "absolute_error": float(error),
                     "m_inf": float(mechanism.m),
                     "source_equation_closing_tau_ms": float(.15/rate_sum/(2.3**1.3)),
                     "stationary_current_ma_cm2": float(section.gbar_NaTg * mechanism.m**3
                                                         * mechanism.h * (voltage - section.ena))})
    assert pair[0]["m_inf"] == pair[1]["m_inf"]
    if voltage < -56.:
        assert pair[1]["h_inf"] > pair[0]["h_inf"]
    elif voltage > -56.:
        assert pair[1]["h_inf"] < pair[0]["h_inf"]
    rows.append({"voltage_mv": voltage, "comparison": pair})
report = {"records": rows, "directional_prediction_passed": True,
          "limit": "Implemented fixed-voltage equilibrium check; slope also changes kinetics; no whole-cell or human validation"}
Path("/evidence/h01-pv-inactivation-equilibrium.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
