"""Experimental human-channel transfer onto H01 anatomy, pending calibration."""

import braincell
import brainunit as u
import numpy as np
import brainstate
import argparse
import json
from pathlib import Path
from braincell.filter import AllRegion
from braincell.mech import Channel, Ion, StateProbe

from . import h01_channels
from .h01 import H01Archive
from .h01_annotations import H01Annotations


def make_active_cell(imported, annotations, *, active_radius_um, current_na,
                     sodium_ms_cm2=9.601446023, potassium_ms_cm2=4.884943554,
                     max_cv_length_um=10., delay_ms=2., duration_ms=3., solver="staggered"):
    """Transfer human pyramidal kinetics to an explicitly inferred soma region.

    Parameters
    ----------
    imported : H01Component
        Connected source component with soma annotations.
    annotations : H01Annotations
        Source cell metadata; this configuration requires L2/L3 pyramidal tags.
    active_radius_um : float
        Inferred active radius along cable from the largest soma-labelled sample.
    current_na : float
        Current injection at that sample in nA.
    sodium_ms_cm2, potassium_ms_cm2 : float, optional
        Borrowed base conductance densities, pending H01 calibration.
    max_cv_length_um : float, optional
        Maximum discretization length in micrometers.
    delay_ms, duration_ms : float, optional
        Current pulse onset and duration in milliseconds.
    solver : str, optional
        BrainCell solver identifier, explicitly recorded in provenance.

    Returns
    -------
    cell : braincell.Cell
        Uninitialized experimental active cell.
    evidence : dict
        Measured source identity, borrowed dynamics and inferred region policy.

    Notes
    -----
    This first transfer activates a local soma neighborhood with passive
    remaining cable. It does not claim an identified AIS, dendritic active
    conductances, fitted H01 physiology, or reproduction of an entire neuron.
    """
    values = (active_radius_um, current_na, sodium_ms_cm2, potassium_ms_cm2,
              max_cv_length_um, delay_ms, duration_ms)
    if (not np.isfinite(values).all() or min(active_radius_um, max_cv_length_um, duration_ms) <= 0
            or min(sodium_ms_cm2, potassium_ms_cm2, delay_ms) < 0):
        raise ValueError("Invalid finite/positive active model parameters.")
    metadata = annotations.metadata(imported.neuron_id)
    if "pyramidal" not in metadata.tags or not {"L2", "L3"}.intersection(metadata.tags):
        raise ValueError("This channel transfer requires source L2/L3 pyramidal tags.")
    anatomy = imported.anatomy()
    soma = anatomy.soma_location()
    rows = imported.source_rows[imported.source_rows[:, 1] == 3]
    anchor = int(sorted(rows, key=lambda row: (-row[5], row[0]))[0][0])
    active = anatomy.cable_neighborhood(anchor, radius_um=active_radius_um)
    cell = braincell.Cell(imported.morphology, cv_policy=braincell.MaxCVLen(max_cv_length_um*u.um),
                          V_init=-70.0*u.mV, solver=solver)
    cell.paint(AllRegion(), braincell.CableProperty(membrane_capacitance=1*u.uF/u.cm**2,
                axial_resistivity=200*u.ohm*u.cm, resting_potential=-70*u.mV))
    cell.paint(AllRegion(), Channel("IL", g_max=.1*u.mS/u.cm**2, E=-70*u.mV))
    cell.paint(active, Ion("SodiumFixed", E=68*u.mV))
    cell.paint(active, Ion("PotassiumFixed", E=-86*u.mV))
    cell.paint(active, Channel("H01Na_Wilbers2023", g_max=sodium_ms_cm2*u.mS/u.cm**2))
    cell.paint(active, Channel("H01K_Wilbers2023", g_max=potassium_ms_cm2*u.mS/u.cm**2))
    cell.place(soma, StateProbe(field="v", name="voltage"))
    cell.place(soma, braincell.CurrentClamp(delay=delay_ms*u.ms, durations=duration_ms*u.ms,
                                          amplitudes=current_na*u.nA))
    evidence = {"measured_anatomy": imported.provenance, "source_tags": metadata.tags,
                "borrowed_dynamics": "Wilbers2023; Dataverse L5J0SD v3.0 Current_clamp/mod",
                "inferred_active_region": {"policy": active.policy, "anchor_source_id": anchor,
                                           "radius_um": active_radius_um, "intervals": active.intervals},
                "parameters": {"sodium_ms_cm2": sodium_ms_cm2, "potassium_ms_cm2": potassium_ms_cm2,
                               "capacitance_uf_cm2": 1., "axial_resistivity_ohm_cm": 200.,
                               "leak_ms_cm2": .1, "leak_mv": -70., "ena_mv": 68., "ek_mv": -86.,
                               "temperature_c": 34., "channel_voltage_shift_mv": -10.,
                               "current_na": current_na, "delay_ms": delay_ms, "duration_ms": duration_ms},
                "numerics": {"solver": solver, "max_cv_length_um": max_cv_length_um},
                "qualification": "experimental transfer; biological calibration and AIS model pending"}
    return cell, evidence


def main(argv=None):
    """Run an explicitly experimental active H01 transfer and save evidence.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments; defaults to the process arguments.

    Returns
    -------
    dict
        Source provenance, assumptions, parameters and the voltage trace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--neuron", default="810151953")
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--active-radius-um", type=float, required=True)
    parser.add_argument("--current-na", type=float, required=True)
    parser.add_argument("--sodium-ms-cm2", type=float, default=9.601446023)
    parser.add_argument("--potassium-ms-cm2", type=float, default=4.884943554)
    parser.add_argument("--max-cv-length-um", type=float, default=10.)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--duration-ms", type=float, default=20.)
    parser.add_argument("--pulse-ms", type=float, default=3.)
    parser.add_argument("--delay-ms", type=float, default=2.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite([args.dt_ms, args.duration_ms]).all() or min(args.dt_ms, args.duration_ms) <= 0:
        parser.error("Time step and run duration must be positive and finite.")
    with brainstate.environ.context(precision=64):
        imported = H01Archive(args.archive).load(args.neuron, component=args.component)
        cell, evidence = make_active_cell(imported, H01Annotations(args.annotations),
            active_radius_um=args.active_radius_um, current_na=args.current_na,
            sodium_ms_cm2=args.sodium_ms_cm2, potassium_ms_cm2=args.potassium_ms_cm2,
            max_cv_length_um=args.max_cv_length_um, delay_ms=args.delay_ms, duration_ms=args.pulse_ms)
        result = cell.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms)
        v = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    if not np.isfinite(v).all():
        raise RuntimeError("Non-finite voltage in active H01 transfer.")
    evidence["result"] = {"dt_ms": args.dt_ms, "duration_ms": args.duration_ms,
                          "precision_bits": v.dtype.itemsize*8, "n_cv": len(cell.cvs),
                          "min_mv": float(v.min()), "max_mv": float(v.max()),
                          "final_mv": float(v[-1]), "finite": True, "voltage_mv": v.tolist(),
                          "sample_time_convention": "end of step: (index+1)*dt"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence["result"].items() if k != "voltage_mv"}))
    return evidence


if __name__ == "__main__":  # pragma: no cover
    main()
