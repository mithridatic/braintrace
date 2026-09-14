"""Run selected E/I candidates on distinct measured H01 components, offline.

From the repository root:
python -m examples.h01_ei_candidates --cache .cache/h01 --output .cache/h01/ei-defaults

This example creates two separate cells. It does not supply measured partner
identities or a connected circuit. The I model borrows PV dynamics; the H01
source labels establish an interneuron, not a PV subtype.
"""
import argparse
import json
from pathlib import Path
import brainstate
import brainunit as u
import numpy as np
from braincell.filter import AllRegion
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell


def label_partition(imported):
    """Return the example's explicit sparse-label electrical partition.

    Parameters
    ----------
    imported : H01Component
        Original source component; its labels are retained.

    Returns
    -------
    regions : dict
        Soma, axon, and remaining basal-dendrite electrical regions.
    basis : str
        Measured-label basis and inferred boundary assumptions.
    """
    anatomy = imported.anatomy()
    soma = anatomy.region("soma", policy="sample_neighborhood")
    axon = (anatomy.region("axon", policy="sample_neighborhood") |
            anatomy.region("axon_initial_segment", policy="sample_neighborhood") |
            anatomy.region("myelinated_axon", policy="sample_neighborhood"))-soma
    regions = {"soma": soma, "axon": axon, "dend": AllRegion()-soma-axon}
    basis = ("Soma, axon, and AIS source samples extend halfway along adjoining edges. "
             "All remaining cable receives basal-dendrite parameters, including unclassified "
             "and astrocyte-labelled samples; this electrical assignment is inferred. "
             "No apical identity or myelin insulation is inferred. Source labels remain intact.")
    return regions, basis


def main():
    """Run both cells with compiled BrainCell time stepping.

    Returns
    -------
    None
        Write direct voltage traces and provenance beside the output prefix.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--output", type=Path, default=Path(".cache/h01/ei-defaults"))
    parser.add_argument("--duration-ms", type=float, default=10.)
    parser.add_argument("--dt-ms", type=float, default=.000625)  # qualified step, docs/evidence/h01-timestep-ladder.json
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--current-na", type=float, default=.1)
    parser.add_argument("--role", choices=("both", "E", "I"), default="both")
    parser.add_argument("--mode", choices=("candidate", "source"), default="candidate")
    args = parser.parse_args()
    if not np.isfinite([args.duration_ms, args.dt_ms, args.current_na]).all() or min(args.duration_ms, args.dt_ms) <= 0:
        parser.error("Time settings must be positive and finite; current must be finite.")
    archive = H01Archive(args.cache/"proofread104.zip")
    annotations = H01Annotations(args.cache)
    reports, traces = {}, {}
    with brainstate.environ.context(precision=64):
        # Static enumeration of two cells. Cell.run compiles each time loop.
        for role, neuron_id in (("E", "810151953"), ("I", "678539249")):
            if args.role not in ("both", role):
                continue
            imported = archive.load(neuron_id, component=0)
            regions, basis = label_partition(imported)
            cell, report = make_h01_ei_cell(imported, annotations, polarity=role,
                regions=regions, region_basis=basis, mode=args.mode, current_na=args.current_na,
                delay_ms=2., duration_ms=3., max_cv_length_um=args.max_cv_um)
            result = cell.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms)
            voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV)).ravel()
            if not np.isfinite(voltage).all():
                raise RuntimeError(role+" response contains nonfinite values.")
            traces[role+"_voltage_mv"] = voltage
            traces[role+"_time_ms"] = args.dt_ms*np.arange(1, len(voltage)+1)
            report.update(dt_ms=args.dt_ms, duration_ms=args.duration_ms,
                          current_na=args.current_na, final_soma_mv=float(voltage[-1]),
                          qualification="Execution demonstration; not physiological or circuit validation.")
            reports[role] = report
            print(role, neuron_id, report["borrowed_dynamics"]["name"], "completed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **traces)
    args.output.with_suffix(".json").write_text(json.dumps(reports, indent=2)+"\n")


if __name__ == "__main__":
    main()
