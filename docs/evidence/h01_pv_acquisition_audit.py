"""Verify the released calibration traces against the original Allen NWB file."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    """Write the acquisition provenance report for the pinned recording."""
    root = Path(__file__).resolve().parents[2]
    cache = root / ".cache/human-pv"
    path = cache / "528687507_ephys.nwb"
    records = []
    with h5py.File(path, "r") as nwb:
        generated = [v.decode() for v in nwb["general/generated_by"][()]]
        assert generated == ["pipeline", "IVSCC", "version", "1.0"]
        assert int(nwb["general/aibs_specimen_id"][()]) == 528687520
        # Pipeline 1.0 already stores SI units. Its conversion attributes are stale.
        for sweep, kind, index in ((20, "passive", 0), (23, "passive", 1),
                                   (35, "active", 0), (39, "active", 2)):
            group = nwb[f"acquisition/timeseries/Sweep_{sweep}"]
            rate = float(group["starting_time"].attrs["rate"])
            assert rate == 50000.
            exported = np.load(cache / f"{kind}-{index}.npz")
            raw_mv = group["data"][()].astype(np.float64) * 1000.
            start = 37500
            target = exported["voltage"].astype(np.float64)
            expected = raw_mv[start:start + len(target)] - 14.
            error = target - expected
            assert np.max(np.abs(error)) < 2e-5
            assert np.max(np.abs(exported["time"] - np.arange(len(target))*.02)) < 1e-8
            stimulus = nwb[f"stimulus/presentation/Sweep_{sweep}/data"][()]
            records.append({
                "sweep": sweep, "export": f"{kind}-{index}.npz",
                "export_sha256": hashlib.sha256((cache / f"{kind}-{index}.npz").read_bytes()).hexdigest(),
                "sample_count": len(target), "crop_start_samples": start,
                "crop_start_ms": 750., "junction_correction_mv": -14.,
                "maximum_reconstruction_error_mv": float(np.max(np.abs(error))),
                "bias_current_pa": float(group["bias_current"][()]) * 1e12,
                "stimulus_levels_pa": (np.unique(stimulus).astype(float)*1e12).tolist(),
                "sampling_rate_hz": rate,
            })
    report = {
        "specimen_id": 528687520,
        "download_url": "https://api.brain-map.org/api/v2/well_known_file_download/618112937",
        "nwb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pipeline": generated, "records": records,
        "reserved_trace": "0.23 nA not read by this audit",
        "interpretation": "Released calibration voltages already include the -14 mV correction. Bias current is separate from stimulus levels.",
        "qualification": "Acquisition provenance only; model behavior is not validated.",
    }
    output = root / "docs/evidence/h01-pv-acquisition-audit.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
