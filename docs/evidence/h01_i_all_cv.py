"""Record direct compartment voltages from the measured H01 I model."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np
from braincell.mech import MechanismProbe


@contextmanager
def _source_closing(enabled):
    """Scope the same I NaTg law override used by the circuit diagnostic."""
    if not enabled:
        yield
        return
    from braintrace.datasets.h01_pv_channels import _CHANNELS
    cls = _CHANNELS["NaTg"]
    original = cls.f_h_tau

    def restored(self, voltage, *ions):
        value = original(self, voltage, *ions)
        equilibrium = self._rates(voltage, ions)["h"][0]
        if not hasattr(self, "h"):
            return value
        return u.math.where(equilibrium <= self.h.value,
                            value/self.phase_factors["h"][1], value)

    cls.f_h_tau = restored
    try:
        yield
    finally:
        cls.f_h_tau = original


def _record(cell, *, dt_ms, duration_ms):
    """Run an uninitialized cell with no incoming network events."""
    if not np.isfinite([dt_ms, duration_ms]).all() or min(dt_ms, duration_ms) <= 0:
        raise ValueError("Time settings must be positive and finite.")
    with brainstate.environ.context(dt=dt_ms*u.ms):
        cell.init_state()
        times = u.math.arange(0.*u.ms, duration_ms*u.ms, dt_ms*u.ms)
        with brainstate.environ.context(t=0.*u.ms):
            cell._prepare_next_synapse_inputs()

        def step(t):
            with brainstate.environ.context(t=t):
                cell.update()
                return {"all_voltage": cell.V.value, "events": u.math.any(cell.spike.value > 0, axis=-1),
                        **cell.sample_probes()}

        traces = brainstate.transform.for_loop(step, times)
        cell._set_current_time(len(times)*dt_ms*u.ms)
    return times+dt_ms*u.ms, traces


def _add_gate_probes(cell, soma):
    """Observe the existing sodium gates without altering their equations."""
    for gate in ("m", "h"):
        cell.place(soma, MechanismProbe(mechanism="pv_NaTg", field=gate, name="NaTg_"+gate))


def main():
    """Save the measured I baseline with every CV voltage.

    Returns
    -------
    None
        Write the arrays and physical CV mapping at the requested output path.
    """
    from braintrace.datasets.h01 import H01Archive
    from braintrace.datasets.h01_annotations import H01Annotations
    from braintrace.datasets.h01_ei_circuit import make_h01_ei_circuit
    from examples.h01_ei_candidates import label_partition

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--duration-ms", type=float, default=8.)
    parser.add_argument("--restore-closing", action="store_true")
    args = parser.parse_args()
    with brainstate.environ.context(precision=64), _source_closing(args.restore_closing):
        cache = Path(".cache/h01")
        archive = H01Archive(cache/"proofread104.zip")
        components = {r: archive.load(n, component=0) for r, n in (("E", "4157825456"), ("I", "5584343344"))}
        parts = {r: label_partition(c) for r, c in components.items()}
        network, evidence = make_h01_ei_circuit(components, H01Annotations(cache),
            regions={r: p[0] for r, p in parts.items()}, region_basis={r: p[1] for r, p in parts.items()},
            solver="h01_staggered_scan", currents_na={"E": 1., "I": 1.})
        assert all(p["post"] != "I" for p in evidence["measured_contacts"])
        cell = network.populations["I"].cell
        _add_gate_probes(cell, components["I"].anatomy().soma_location())
        print("All-CV I recorder constructed", flush=True)
        times, traces = _record(cell, dt_ms=args.dt_ms, duration_ms=args.duration_ms)
        arrays = {"time_ms": np.asarray(times.to_decimal(u.ms))}
        for key, value in traces.items():
            unit = u.mV if key in ("all_voltage", "voltage", "output_voltage") else u.uS if key == "synaptic_conductance" else u.UNITLESS
            arrays[key] = np.asarray(value.to_decimal(unit) if isinstance(value, u.Quantity) else value)
            if not np.isfinite(arrays[key]).all():
                raise ValueError("Nonfinite recorded values: "+key)
        evidence.update(dt_ms=args.dt_ms, duration_ms=args.duration_ms,
            execution="Standalone I with no incoming edges; E is not simulated.",
            cv_columns=[dict(id=cv.id, branch_id=cv.branch_id, prox=cv.prox, dist=cv.dist) for cv in cell.cvs],
            array_axes="all_voltage: time, population, CV; CV order follows cv_columns.",
            qualification="Diagnostic baseline; compare original circuit traces before spatial interpretation.")
        if args.restore_closing:
            evidence["cells"]["I"]["diagnostic_override"] = dict(NaTg_h_closing_factor=1., candidate_value=.15,
                scope="all I NaTg regions; all times; E unchanged")
            evidence["qualification"] = "Diagnostic closing-time intervention; not a promoted or human-qualified model."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2)+"\n")
    print("All-CV I recorder complete", flush=True)


if __name__ == "__main__":
    main()
