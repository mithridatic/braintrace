"""Check the modified gate against analytic fixed-voltage trajectories."""

import json
from pathlib import Path

import numpy as np
from neuron import h


def _vtrap(x):
    return 6. * (1 - x / 12.) if abs(x / 6.) < 1e-6 else x / np.expm1(x / 6.)


def main():
    """Advance independent zero-conductance sections and verify their gates."""
    h.load_file("stdrun.hoc")
    h.CVode().active(0)
    h.celsius = 34.
    h.dt = .00005
    h.steps_per_ms = 20000.
    qt = 2.3 ** ((34. - 23.) / 10.)
    cases = []
    for voltage in (-80., -60., -40., 0.):
        ha, hb = .015 * _vtrap(voltage + 66.), .015 * _vtrap(-(voltage + 66.))
        ma, mb = .182 * _vtrap(-(voltage + 40.)), .124 * _vtrap(voltage + 40.)
        hi, ht = ha / (ha + hb), 1. / ((ha + hb) * qt)
        mi, mt = ma / (ma + mb), 1. / ((ma + mb) * qt)
        for factor in (1., 2.):
            for phase, initial in (("recovery", 0.), ("closing", 1.), ("equilibrium", hi)):
                sec = h.Section(name="clamp_" + str(len(cases)))
                sec.L, sec.diam, sec.nseg = 10., 10., 1
                sec.insert("NaTs")
                channel = sec(.5).NaTs
                channel.gbar, channel.h_recovery_factor = 0., factor
                cases.append({"section": sec, "channel": channel, "voltage": voltage,
                              "factor": factor, "phase": phase, "initial": initial,
                              "hi": hi, "ht": ht * factor if phase == "recovery" else ht,
                              "mi": mi, "mt": mt,
                              "h_trace": h.Vector().record(channel._ref_h),
                              "m_trace": h.Vector().record(channel._ref_m),
                              "v_trace": h.Vector().record(sec(.5)._ref_v)})
    times = h.Vector().record(h._ref_t)
    h.finitialize(-65.)
    for case in cases:
        case["section"].v = case["voltage"]
        case["channel"].h = case["initial"]
        case["channel"].m = .5
    h.fcurrent()
    h.frecord_init()
    h.continuerun(2.)
    t = np.asarray(times)
    results, arrays = [], {"time_ms": t}
    for index, case in enumerate(cases):
        actual_h = np.asarray(case["h_trace"])
        actual_m = np.asarray(case["m_trace"])
        voltage = np.asarray(case["v_trace"])
        expected_h = case["hi"] + (case["initial"] - case["hi"]) * np.exp(-t / case["ht"])
        expected_m = case["mi"] + (.5 - case["mi"]) * np.exp(-t / case["mt"])
        h_error = float(np.max(np.abs(actual_h - expected_h)))
        m_error = float(np.max(np.abs(actual_m - expected_m)))
        fixed = bool(np.all(voltage == case["voltage"]))
        results.append({"voltage_mv": case["voltage"], "factor": case["factor"], "phase": case["phase"],
                        "h_equilibrium": case["hi"], "h_time_ms": case["ht"],
                        "max_h_error": h_error, "max_m_error": m_error, "voltage_fixed": fixed,
                        "passed": fixed and h_error <= 1e-8 and m_error <= 1e-8})
        arrays[f"h_{index}"] = actual_h
        arrays[f"m_{index}"] = actual_m
    root = Path(__file__).parent
    np.savez_compressed(root / "h01-l2-sodium-recovery-clamp.npz", **arrays)
    report = {"cases": results, "gate_error_limit": 1e-8,
              "decision": "supported" if all(r["passed"] for r in results) else "rejected",
              "qualification": "Fixed-voltage gate law only; full-cell source equivalence remains required."}
    (root / "h01-l2-sodium-recovery-clamp.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], len(results), "max h error", max(r["max_h_error"] for r in results),
          "max m error", max(r["max_m_error"] for r in results))


if __name__ == "__main__":
    main()
