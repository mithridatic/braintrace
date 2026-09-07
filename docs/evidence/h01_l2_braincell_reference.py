"""Record the transferred layer-2 E reference cell under a recorded Allen sweep protocol."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import braincell
import brainstate
import brainunit as u
import numpy as np
from scipy.signal import find_peaks

from braintrace.datasets import h01_l2_cell
from braintrace.datasets.h01_pv_morphology import make_pv_morphology
from braintrace.datasets.h01_ei_profiles import get_ei_profile

PULSE_MS = (1020., 2020.)


def neuron_counts_by_branch(sections):
    """Map L2 NEURON report section names such as ``soma[0]`` to branch names and counts."""
    counts = {}
    for section in sections:
        name = section["name"].split(".", 1)[-1].replace("[", "_").replace("]", "")
        if name in counts:
            raise ValueError("Duplicate section name in NEURON report: "+section["name"])
        counts[name] = int(section["nseg"])
    return counts


def constant_segments(time_ms, current_na, stop_ms):
    """Collapse a sampled step protocol into consecutive constant segments covering ``[0, stop_ms]``."""
    time_ms, current_na = np.asarray(time_ms, dtype=float), np.asarray(current_na, dtype=float)
    if time_ms.ndim != 1 or time_ms.shape != current_na.shape or len(time_ms) < 2 or time_ms[0] != 0.:
        raise ValueError("Protocol requires matching one-dimensional samples starting at zero.")
    if not np.isfinite(current_na).all() or stop_ms <= 0 or stop_ms > time_ms[-1]:
        raise ValueError("Protocol requires finite current and a stop inside the recording.")
    keep = time_ms <= stop_ms
    times, current = time_ms[keep], current_na[keep]
    starts = np.r_[0, np.flatnonzero(np.diff(current) != 0.)+1]
    edges = np.r_[times[starts], stop_ms]
    durations, amplitudes = np.diff(edges), current[starts]
    positive = durations > 0
    return durations[positive].tolist(), amplitudes[positive].tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="human-pyramidal-l2 cache directory (read-only)")
    parser.add_argument("--sweep", type=int, choices=(43, 50, 53, 55), default=50)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--stop-ms", type=float, default=2100.)
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--mesh-from", type=Path, help="L2 NEURON report JSON whose per-section nseg becomes the mesh")
    parser.add_argument("--profile-key", "--mode", dest="profile_key", choices=("candidate", "source", "b3"),
                        default="candidate", help="registry mode; b3 = the frozen B3 experimental profile")
    parser.add_argument("--include-recorded-bias", action="store_true", default=True)
    parser.add_argument("--command-only", dest="include_recorded_bias", action="store_false")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite([args.dt_ms, args.stop_ms, args.max_cv_um]).all() or min(args.dt_ms, args.stop_ms, args.max_cv_um) <= 0:
        parser.error("Time step, stop, and compartment length must be positive and finite.")
    reference = json.loads(Path(__file__).with_name("h01-l2-geometry-reference.json").read_text())
    mesh, policy = {"policy": "MaxCVLen", "max_cv_length_um": args.max_cv_um}, None
    if args.mesh_from is not None:
        counts = neuron_counts_by_branch(json.loads(args.mesh_from.read_text())["sections"])
        branches = [b.name for b in make_pv_morphology(reference).branches]
        missing = sorted(set(branches)-set(counts))
        if missing or len(counts) != len(branches):
            parser.error("NEURON report does not name every branch exactly once: "+", ".join(missing))
        per_branch = tuple(counts[b] for b in branches)
        policy = braincell.CVPerBranchList(per_branch)
        mesh = {"policy": "CVPerBranchList", "max_cv_length_um": None, "source": args.mesh_from.name,
                "source_sha256": hashlib.sha256(args.mesh_from.read_bytes()).hexdigest(),
                "branches": branches, "cv_per_branch": list(per_branch)}
    waveform_path = args.cache/f"sweep-{args.sweep}.npz"
    with np.load(waveform_path) as source:
        bias = float(source["bias_current_na"]) if args.include_recorded_bias else 0.
        durations, amplitudes = constant_segments(source["time_ms"], source["command_current_na"], args.stop_ms)
    amplitudes = [a+bias for a in amplitudes]
    with brainstate.environ.context(precision=64):
        cell = h01_l2_cell.make_l2_cell(reference, mode=args.profile_key, current_na=amplitudes, delay_ms=0.,
                                        duration_ms=durations, max_cv_length_um=args.max_cv_um, cv_policy=policy)
        result = cell.run(dt=args.dt_ms*u.ms, duration=args.stop_ms*u.ms)
        voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV)).ravel()
        calcium = np.asarray(result.traces["calcium"].to_decimal(u.mM)).ravel()
    time = (np.arange(len(voltage))+1)*args.dt_ms
    assert np.isfinite(voltage).all() and np.isfinite(calcium).all()
    peaks, _ = find_peaks(voltage, height=0., prominence=40.)
    peaks = peaks[(time[peaks] >= PULSE_MS[0]) & (time[peaks] < PULSE_MS[1])]
    np.savez_compressed(args.output.with_suffix(".npz"), time_ms=time, voltage_mv=voltage, calcium_mm=calcium)
    profile = get_ei_profile("E", mode=args.profile_key)
    report = {"model_id": 626170538, "specimen_id": 541563728, "sweep": args.sweep,
              "waveform_sha256": hashlib.sha256(waveform_path.read_bytes()).hexdigest(),
              "profile": asdict(profile), "profile_key": args.profile_key, "solver": "staggered",
              "dt_ms": args.dt_ms, "stop_ms": args.stop_ms, "max_cv_length_um": mesh["max_cv_length_um"], "mesh": mesh,
              "input": "command plus recorded bias" if args.include_recorded_bias else "source command only; bias not added",
              "added_bias_na": bias, "segments": {"durations_ms": durations, "amplitudes_na": amplitudes},
              "pulse_ms": list(PULSE_MS), "temperature_c": 34., "initial_voltage_mv": profile.initial_mv,
              "sample_convention": "end of step; first sample at dt",
              "spike_times_ms": time[peaks].tolist(), "spike_peaks_mv": voltage[peaks].tolist(),
              "qualification": "E transfer response on donor geometry; numerical and human waveform validation incomplete"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("profile", "mesh")}))


if __name__ == "__main__":
    main()
