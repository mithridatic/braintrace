"""Record the transferred PV reference cell with the published current step."""

import argparse
import json
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np
from scipy.signal import find_peaks

from braintrace.datasets.h01_pv_cell import make_pv_cell

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dt-ms", type=float, default=.005)
parser.add_argument("--max-cv-um", type=float, default=10.)
parser.add_argument("--current-na", type=float, default=.19)
parser.add_argument("--passive", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not np.isfinite(args.dt_ms) or args.dt_ms <= 0:
    parser.error("Time step must be positive and finite.")
reference = json.loads(Path(__file__).with_name("h01-pv-geometry-reference.json").read_text())
with brainstate.environ.context(precision=64):
    cell = make_pv_cell(reference, args.current_na, max_cv_length_um=args.max_cv_um, active=not args.passive)
    result = cell.run(dt=args.dt_ms*u.ms, duration=1500.*u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV)).ravel()
    calcium = np.asarray(result.traces["calcium"].to_decimal(u.mM)).ravel() if not args.passive else np.array([])
    sk = np.asarray(result.traces["sk_gate"]).ravel() if not args.passive else np.array([])
time = (np.arange(len(voltage))+1)*args.dt_ms
assert np.isfinite(voltage).all() and np.isfinite(calcium).all()
peaks, _ = find_peaks(voltage, height=0., prominence=40.)
peaks = peaks[(time[peaks] >= 270.) & (time[peaks] < 1270.)]
np.savez_compressed(args.output.with_suffix(".npz"), time_ms=time, voltage_mv=voltage,
                    calcium_mm=calcium, sk_gate=sk)
report = {"source_commit": reference["source_commit"], "solver": "staggered",
          "active_channels": not args.passive,
          "dt_ms": args.dt_ms, "max_cv_length_um": args.max_cv_um,
          "current_na": args.current_na, "temperature_c": 34.,
          "initial_voltage_mv": -80., "stimulus_on_ms": 270., "stimulus_off_ms": 1270.,
          "sample_convention": "end of step; first sample at dt",
          "spike_times_ms": time[peaks].tolist(), "spike_peaks_mv": voltage[peaks].tolist(),
          "baseline_mean_mv": float(voltage[(time >= 200.) & (time < 270.)].mean()),
          "calcium_range_mm": [float(calcium.min()), float(calcium.max())] if calcium.size else None,
          "qualification": "first transfer response; numerical and human waveform validation incomplete"}
args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
