"""Record the transferred PV reference cell with the published current step."""

import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import asdict

import braincell
import brainstate
import brainunit as u
import numpy as np
from scipy.signal import find_peaks

from braintrace.datasets.h01_pv_cell import make_pv_cell
from braintrace.datasets.h01_pv_morphology import make_pv_morphology
from braintrace.datasets.h01_ei_profiles import get_ei_profile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dt-ms", type=float, default=.005)
parser.add_argument("--max-cv-um", type=float, default=10.)
parser.add_argument("--mesh-from", type=Path, help="NEURON run JSON whose per-section nseg becomes the mesh")
parser.add_argument("--current-na", type=float, default=.19)
parser.add_argument("--passive", action="store_true")
parser.add_argument("--mode", choices=("candidate", "source", "finalist"), default="candidate",
                    help="registry mode; finalist = candidate with the somatic Kv3 closing factor 2.0 (e-kv3-close2)")
parser.add_argument("--duration-ms", type=float, default=1500.)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not np.isfinite([args.dt_ms, args.duration_ms]).all() or min(args.dt_ms, args.duration_ms) <= 0:
    parser.error("Time step must be positive and finite.")
if not np.isfinite(args.max_cv_um) or args.max_cv_um <= 0:
    parser.error("Maximum compartment length must be positive and finite.")
reference = json.loads(Path(__file__).with_name("h01-pv-geometry-reference.json").read_text())


def neuron_counts_by_branch(geometry):
    """Map NEURON section names such as ``NeuronTemplate[0].soma[0]`` to branch counts."""
    counts = {}
    for section in geometry:
        name = section["name"].split(".", 1)[-1].replace("[", "_").replace("]", "")
        if name in counts:
            raise ValueError("Duplicate section name in NEURON geometry: "+section["name"])
        counts[name] = int(section["nseg"])
    return counts


mesh = {"policy": "MaxCVLen", "max_cv_length_um": args.max_cv_um}
policy = None
if args.mesh_from is not None:
    counts = neuron_counts_by_branch(json.loads(args.mesh_from.read_text())["geometry"])
    branches = [b.name for b in make_pv_morphology(reference).branches]
    missing = sorted(set(branches)-set(counts))
    if missing or len(counts) != len(branches):
        parser.error("NEURON geometry does not name every branch exactly once: "+", ".join(missing))
    per_branch = tuple(counts[b] for b in branches)
    policy = braincell.CVPerBranchList(per_branch)
    mesh = {"policy": "CVPerBranchList", "max_cv_length_um": None, "source": args.mesh_from.name,
            "source_sha256": hashlib.sha256(args.mesh_from.read_bytes()).hexdigest(),
            "branches": branches, "cv_per_branch": list(per_branch)}
with brainstate.environ.context(precision=64):
    cell = make_pv_cell(reference, args.current_na, max_cv_length_um=args.max_cv_um, active=not args.passive,
                        mode=args.mode, cv_policy=policy)
    result = cell.run(dt=args.dt_ms*u.ms, duration=args.duration_ms*u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV)).ravel()
    calcium = np.asarray(result.traces["calcium"].to_decimal(u.mM)).ravel() if not args.passive else np.array([])
    sk = np.asarray(result.traces["sk_gate"]).ravel() if not args.passive else np.array([])
time = (np.arange(len(voltage))+1)*args.dt_ms
assert np.isfinite(voltage).all() and np.isfinite(calcium).all()
peaks, _ = find_peaks(voltage, height=0., prominence=40.)
peaks = peaks[(time[peaks] >= 270.) & (time[peaks] < 1270.)]
np.savez_compressed(args.output.with_suffix(".npz"), time_ms=time, voltage_mv=voltage,
                    calcium_mm=calcium, sk_gate=sk)
report = {"source_commit": reference["source_commit"],
          "profile": asdict(get_ei_profile("I", mode=args.mode)), "duration_ms": args.duration_ms, "solver": "staggered",
          "active_channels": not args.passive,
          "dt_ms": args.dt_ms, "max_cv_length_um": mesh["max_cv_length_um"], "mesh": mesh,
          "current_na": args.current_na, "temperature_c": 34.,
          "initial_voltage_mv": -80., "stimulus_on_ms": 270., "stimulus_off_ms": 1270.,
          "sample_convention": "end of step; first sample at dt",
          "spike_times_ms": time[peaks].tolist(), "spike_peaks_mv": voltage[peaks].tolist(),
          "baseline_mean_mv": float(voltage[(time >= 200.) & (time < 270.)].mean()) if np.any((time >= 200.) & (time < 270.)) else None,
          "calcium_range_mm": [float(calcium.min()), float(calcium.max())] if calcium.size else None,
          "qualification": "first transfer response; numerical and human waveform validation incomplete"}
args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
