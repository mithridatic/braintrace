"""Run the explicitly inferred two-cell H01 circuit and save direct traces.

python -m examples.h01_ei_circuit --control e_only --output .cache/h01/circuit-e
Run disconnected and ei with identical numerical and input settings for controls.
"""
import argparse
import json
from pathlib import Path
import brainstate
import brainunit as u
import numpy as np
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_ei_circuit import make_h01_ei_circuit
from examples.h01_ei_candidates import label_partition


def main():
    """Run a compiled diagnostic circuit with explicit provenance.

    Returns
    -------
    None
        Save per-cell voltages, conductances, output events, and provenance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("ei", "e_only", "i_only", "disconnected"), default="ei")
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--duration-ms", type=float, default=10.)
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--e-current-na", type=float, default=1.)
    parser.add_argument("--i-current-na", type=float, default=0.)
    args = parser.parse_args()
    archive, annotations = H01Archive(args.cache/"proofread104.zip"), H01Annotations(args.cache)
    with brainstate.environ.context(precision=64):
        components = {r: archive.load(nid, component=0) for r, nid in (("E", "810151953"), ("I", "678539249"))}
        parts = {r: label_partition(c) for r, c in components.items()}
        network, evidence = make_h01_ei_circuit(components, annotations,
            regions={r: p[0] for r, p in parts.items()}, region_basis={r: p[1] for r, p in parts.items()},
            control=args.control, max_cv_length_um=args.max_cv_um,
            currents_na={"E": args.e_current_na, "I": args.i_current_na})
        print("Circuit constructed:", args.control, flush=True)
        result = network.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms, spike_recording="population")
        arrays = {"time_ms": np.asarray(result.time.to_decimal(u.ms))+args.dt_ms}
        for role in ("E", "I"):
            for key, unit in (("voltage", u.mV), ("output_voltage", u.mV), ("synaptic_conductance", u.uS)):
                values = np.asarray(result.traces[role][key].to_decimal(unit)).ravel()
                if not np.isfinite(values).all():
                    raise RuntimeError(role+" "+key+" contains nonfinite values.")
                arrays[role+"_"+key] = values
            events = np.asarray(result.spikes[role]).ravel()
            arrays[role+"_events"] = events
            evidence["cells"][role]["emitted_events_ms"] = arrays["time_ms"][events.astype(bool)].tolist()
        evidence.update(dt_ms=args.dt_ms, duration_ms=args.duration_ms, delay_quantization="ceil",
                        sample_convention="end of step; Network start times plus dt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2)+"\n")
    print({r: evidence["cells"][r]["emitted_events_ms"] for r in ("E", "I")}, flush=True)


if __name__ == "__main__":
    main()
