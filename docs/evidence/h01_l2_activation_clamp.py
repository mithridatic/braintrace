"""Verify selective sodium opening against fixed-voltage analytic solutions."""

import hashlib
import json
from pathlib import Path

import numpy as np
from neuron import h

from h01_l2_recovery_clamp import _vtrap


def main():
    """Run independent gate cases in one compiled NEURON advance.

    Returns
    -------
    None
        Save all gate trajectories, comparisons, and mechanism hashes.
    """
    h.load_file("stdrun.hoc")
    h.CVode().active(0)
    h.celsius, h.dt, h.steps_per_ms = 34., .00005, 20000.
    qt = 2.3 ** 1.1
    cases = []
    for voltage in (-84., -60., -40., -20., 0., 20.):
        ma, mb = .182 * _vtrap(-(voltage + 40.)), .124 * _vtrap(voltage + 40.)
        ha, hb = .015 * _vtrap(voltage + 66.), .015 * _vtrap(-(voltage + 66.))
        mi, mt = ma / (ma + mb), 1. / ((ma + mb) * qt)
        hi, ht = ha / (ha + hb), 1. / ((ha + hb) * qt)
        for factor in (1., 2.):
            for phase, initial in (("opening", 0.), ("closing", 1.), ("equilibrium", mi)):
                sec = h.Section(name="opening_clamp_" + str(len(cases)))
                sec.L, sec.diam, sec.nseg = 10., 10., 1
                sec.insert("NaTs")
                channel = sec(.5).NaTs
                channel.gbar = 0.
                channel.h_recovery_factor, channel.m_opening_factor = 1., factor
                cases.append(dict(section=sec, channel=channel, voltage=voltage,
                                  factor=factor, phase=phase, initial=initial,
                                  mi=mi, mt=mt * factor if phase == "opening" else mt,
                                  hi=hi, ht=ht,
                                  m_trace=h.Vector().record(channel._ref_m),
                                  h_trace=h.Vector().record(channel._ref_h),
                                  v_trace=h.Vector().record(sec(.5)._ref_v)))
    times = h.Vector().record(h._ref_t)
    h.finitialize(-65.)
    for case in cases:
        case["section"].v = case["voltage"]
        case["channel"].m, case["channel"].h = case["initial"], .5
    h.fcurrent()
    h.frecord_init()
    h.continuerun(2.)
    t = np.asarray(times)
    arrays, results = {"time_ms": t}, []
    for index, case in enumerate(cases):
        am, ah, av = (np.asarray(case[k]) for k in ("m_trace", "h_trace", "v_trace"))
        em = case["mi"] + (case["initial"] - case["mi"]) * np.exp(-t / case["mt"])
        eh = case["hi"] + (.5 - case["hi"]) * np.exp(-t / case["ht"])
        me, he = float(np.max(np.abs(am - em))), float(np.max(np.abs(ah - eh)))
        fixed = bool(np.all(av == case["voltage"]))
        results.append(dict(voltage_mv=case["voltage"], factor=case["factor"],
                            phase=case["phase"], m_equilibrium=case["mi"],
                            effective_m_tau_ms=case["mt"], max_m_error=me,
                            max_h_error=he, voltage_fixed=fixed,
                            passed=fixed and me <= 1e-8 and he <= 1e-8))
        arrays.update({f"m_{index}": am, f"h_{index}": ah, f"v_{index}": av})
    root = Path(__file__).parent
    files = [*Path.cwd().glob("*.mod"), Path("x86_64/libnrnmech.so")]
    report = dict(cases=results, gate_error_limit=1e-8, dt_ms=float(h.dt),
                  temperature_c=float(h.celsius), solver="fixed step cnexp",
                  mechanism_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  decision="supported" if all(r["passed"] for r in results) else "rejected",
                  qualification="Gate law only; default whole-cell equivalence and physiological tests remain open.")
    np.savez_compressed(root / "h01-l2-sodium-activation-clamp.npz", **arrays)
    (root / "h01-l2-sodium-activation-clamp.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], len(results), "max m error", max(r["max_m_error"] for r in results),
          "max h error", max(r["max_h_error"] for r in results))


if __name__ == "__main__":
    main()
