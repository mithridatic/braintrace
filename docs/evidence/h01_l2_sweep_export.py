"""Export one long-square sweep of the L2 recording in the driver's waveform format.

The released ``sweep-43.npz`` and ``sweep-50.npz`` carry reported and corrected
voltage, command current, the recorded bias and their sum. This exporter rebuilds
that format from ``recording.nwb`` for any long-square sweep, so that the closed
holdout (sweep 53) can be opened once for the Stage 4 prediction.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

JUNCTION_MV = -14.


def export_sweep(cache, sweep):
    """Arrays of one sweep in the driver's waveform format."""
    index = {row["sweep"]: row for row in json.loads((cache/"long-square-index.json").read_text())}
    entry = index[f"Sweep_{sweep}"]
    with h5py.File(cache/"recording.nwb", "r") as nwb:
        group = nwb[f"acquisition/timeseries/Sweep_{sweep}"]
        rate = float(group["starting_time"].attrs["rate"])
        reported = group["data"][()].astype(np.float64)*1000.
        command = nwb[f"stimulus/presentation/Sweep_{sweep}/data"][()].astype(np.float64)*1e9
    bias = float(entry["bias_a"])*1e9
    return {"time_ms": np.arange(len(reported))*1000./rate, "reported_voltage_mv": reported,
            "corrected_voltage_mv": reported+JUNCTION_MV, "command_current_na": command,
            "bias_current_na": np.float64(bias), "total_current_na": command+bias}


def matches(arrays, path):
    """Largest absolute difference against an existing export, per key."""
    existing = np.load(path)
    return {key: float(np.max(np.abs(np.asarray(existing[key], float)-np.asarray(arrays[key], float))))
            for key in arrays if key in existing}


def main(argv=None):
    """Write ``sweep-<n>.npz`` into the cache, or verify against an existing file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sweep", type=int, required=True)
    parser.add_argument("--verify", action="store_true", help="compare with the existing export instead of writing")
    args = parser.parse_args(argv)
    arrays = export_sweep(args.cache, args.sweep)
    target = args.cache/f"sweep-{args.sweep}.npz"
    if args.verify:
        print(json.dumps(matches(arrays, target)))
        return
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite a released export.")
    np.savez(target, **arrays)
    print(target, len(arrays["time_ms"]), "samples")


if __name__ == "__main__":
    main()
