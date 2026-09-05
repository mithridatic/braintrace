"""Compare isolated Kv3 closing against fixed-voltage analytic solutions."""

import hashlib
import json
from pathlib import Path

import numpy as np
from neuron import h


def main():
    """Run all gate cases in one compiled NEURON advance.

    Returns
    -------
    None
        Save every trajectory, direct error, and mechanism hash.
    """
    h.load_file("stdrun.hoc")
    h.CVode().active(0)
    h.celsius, h.dt, h.steps_per_ms = 34., .00005, 20000.
    cases = []
    for voltage in (-84., -60., -40., -20., 0., 20.):
        mi = 1. / (1. + np.exp((voltage - 18.7) / -9.7))
        mt = 4. / (1. + np.exp((voltage + 46.56) / -44.14))
        for factor in (1., .5):
            for phase, initial in (("opening", 0.), ("closing", 1.), ("equilibrium", mi)):
                sec = h.Section(name="kv3_clamp_" + str(len(cases)))
                sec.L, sec.diam, sec.nseg = 10., 10., 1
                sec.insert("Kv3_1")
                channel = sec(.5).Kv3_1
                channel.gbar, channel.m_closing_factor = 0., factor
                cases.append(dict(section=sec, channel=channel, voltage=voltage,
                                  factor=factor, phase=phase, initial=initial, mi=mi,
                                  mt=mt * factor if phase == "closing" else mt,
                                  m_trace=h.Vector().record(channel._ref_m),
                                  v_trace=h.Vector().record(sec(.5)._ref_v)))
    times = h.Vector().record(h._ref_t)
    h.finitialize(-65.)
    for case in cases:
        case["section"].v = case["voltage"]
        case["channel"].m = case["initial"]
    h.fcurrent()
    h.frecord_init()
    h.continuerun(2.)
    t = np.asarray(times)
    arrays, results = {"time_ms": t}, []
    for index, case in enumerate(cases):
        am, av = (np.asarray(case[k]) for k in ("m_trace", "v_trace"))
        expected = case["mi"] + (case["initial"] - case["mi"]) * np.exp(-t / case["mt"])
        error = float(np.max(np.abs(am - expected)))
        fixed = bool(np.all(av == case["voltage"]))
        results.append(dict(voltage_mv=case["voltage"], factor=case["factor"],
                            phase=case["phase"], equilibrium=case["mi"],
                            effective_tau_ms=case["mt"], max_gate_error=error,
                            voltage_fixed=fixed, passed=fixed and error <= 1e-8))
        arrays.update({f"m_{index}": am, f"v_{index}": av})
    assert len(results) == 36
    files = [*Path.cwd().glob("*.mod"), Path("x86_64/libnrnmech.so")]
    report = dict(cases=results, gate_error_limit=1e-8, dt_ms=float(h.dt),
                  temperature_c=float(h.celsius), solver="fixed step cnexp",
                  mechanism_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  decision="supported" if all(r["passed"] for r in results) else "rejected",
                  qualification="Gate law only; whole-cell equivalence and physiological tests remain open.")
    root = Path(__file__).parent
    np.savez_compressed(root / "h01-l2-kv3-closing-clamp.npz", **arrays)
    (root / "h01-l2-kv3-closing-clamp.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], len(results), "max gate error", max(r["max_gate_error"] for r in results))


if __name__ == "__main__":
    main()
