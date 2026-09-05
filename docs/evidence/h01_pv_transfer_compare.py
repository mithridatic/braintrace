"""Compare a BrainCell transfer trace with the accurate NEURON reference."""

import json
import argparse
from pathlib import Path

import numpy as np

folder = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--actual", default="h01-pv-braincell-019")
parser.add_argument("--reference", default="h01-pv-neuron-019-cvode10")
parser.add_argument("--output", default="h01-pv-transfer")
args = parser.parse_args()
actual = np.load(folder/(args.actual+".npz"))
reference = np.load(folder/(args.reference+".npz"))
time = actual["time_ms"]
predicted = actual["voltage_mv"]
target = np.interp(time, reference["time_ms"], reference["voltage_mv"])
residual = predicted-target
extras = {}
for field in ("calcium_mm", "sk_gate"):
    if field in actual and field in reference and actual[field].size and reference[field].size:
        extras[field] = actual[field]-np.interp(time, reference["time_ms"], reference[field])
report = {"reference": args.reference,
          "transfer": args.actual,
          "comparison": "same physical clock; no voltage or spike-time alignment",
          "windows": {}}
for name, start, stop in (("initial", 0., 20.), ("baseline", 200., 270.),
                          ("stimulus", 270., 1270.), ("recovery", 1270., 1500.)):
    selected = (time >= start) & (time < stop)
    errors = residual[selected]
    report["windows"][name] = {"start_ms": start, "stop_ms": stop,
        "max_error_mv": float(np.max(np.abs(errors))),
        "rms_error_mv": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error_mv": float(np.mean(errors))}
    for field, difference in extras.items():
        report["windows"][name][field] = {
            "max_error": float(np.max(np.abs(difference[selected]))),
            "rms_error": float(np.sqrt(np.mean(difference[selected]**2)))}
np.savez_compressed(folder/(args.output+"-residual.npz"), time_ms=time,
                    braincell_mv=predicted, neuron_mv=target, residual_mv=residual,
                    **{key+"_residual": value for key, value in extras.items()})
(folder/(args.output+"-comparison.json")).write_text(json.dumps(report, indent=2))
print(json.dumps(report))
