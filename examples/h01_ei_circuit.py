"""Run the measured I-to-E two-cell H01 circuit and save direct traces.

python -m examples.h01_ei_circuit --control ei --output .cache/h01/circuit-e
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


MEASURED_IDENTITIES = (("E", "4157825456"), ("I", "5584343344"))
ILLUSTRATIVE_IDENTITIES = (("E", "810151953"), ("I", "678539249"))


def identities_for(connectivity):
    """Cell identities for the measured pair or the explicitly illustrative pair."""
    if connectivity == "measured":
        return MEASURED_IDENTITIES
    if connectivity == "illustrative":
        return ILLUSTRATIVE_IDENTITIES
    raise ValueError("connectivity must be measured or illustrative")


def default_i_current_na(connectivity, requested):
    """Explicit I drive, else 1 nA for the measured pair and 0 nA for the illustrative pair."""
    if requested is not None:
        return requested
    return 1. if connectivity == "measured" else 0.


def build_parser():
    """Command-line interface of the circuit runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("ei", "e_only", "i_only", "disconnected"), default="ei")
    parser.add_argument("--connectivity", choices=("measured", "illustrative"), default="measured")
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--duration-ms", type=float, default=10.)
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--solver", choices=("staggered", "h01_staggered_scan"), default="staggered")
    parser.add_argument("--e-current-na", type=float, default=1.)
    parser.add_argument("--i-current-na", type=float)
    parser.add_argument("--e-delay-ms", type=float, default=2.)
    parser.add_argument("--i-delay-ms", type=float, default=2.)
    parser.add_argument("--e-pulse-ms", type=float, default=3.)
    parser.add_argument("--i-pulse-ms", type=float, default=3.)
    parser.add_argument("--inhibitory-weight-us", type=float, default=.02)
    parser.add_argument("--inhibitory-reversal-mv", type=float, default=-80.)
    parser.add_argument("--inhibitory-tau-ms", type=float, default=5.)
    return parser


def main(argv=None):
    """Run a compiled diagnostic circuit with explicit provenance.

    Returns
    -------
    None
        Save per-cell voltages, conductances, output events, and provenance.
    """
    args = build_parser().parse_args(argv)
    archive, annotations = H01Archive(args.cache/"proofread104.zip"), H01Annotations(args.cache)
    with brainstate.environ.context(precision=64):
        identities = identities_for(args.connectivity)
        components = {r: archive.load(nid, component=0) for r, nid in identities}
        parts = {r: label_partition(c) for r, c in components.items()}
        network, evidence = make_h01_ei_circuit(components, annotations,
            regions={r: p[0] for r, p in parts.items()}, region_basis={r: p[1] for r, p in parts.items()},
            control=args.control, connectivity=args.connectivity, max_cv_length_um=args.max_cv_um, solver=args.solver,
            pulse_delays_ms={"E": args.e_delay_ms, "I": args.i_delay_ms},
            pulse_durations_ms={"E": args.e_pulse_ms, "I": args.i_pulse_ms},
            inhibitory_weight_us=args.inhibitory_weight_us, inhibitory_reversal_mv=args.inhibitory_reversal_mv,
            inhibitory_tau_ms=args.inhibitory_tau_ms,
            currents_na={"E": args.e_current_na, "I": default_i_current_na(args.connectivity, args.i_current_na)})
        print("Circuit constructed:", args.control, flush=True)
        result = network.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms, spike_recording="population")
        arrays = {"time_ms": np.asarray(result.time.to_decimal(u.ms))+args.dt_ms}
        for role in ("E", "I"):
            for key, unit in (("voltage", u.mV), ("output_voltage", u.mV), ("incoming_voltage", u.mV), ("synaptic_conductance", u.uS)):
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
