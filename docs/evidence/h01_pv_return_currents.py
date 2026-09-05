"""Retain paired voltage and channel currents during the first voltage return."""

import json
from pathlib import Path

import numpy as np


def main():
    """Save current observations and check their ion-current sums."""
    folder = Path(__file__).parent
    reference = json.loads((folder / "h01-pv-slope5-audit.json").read_text())
    records = []
    for row, suffix in zip(reference["records"], ("019", "027")):
        data = np.load(folder / f"h01-pv-slope5-{suffix}.npz")
        time = data["time_ms"]
        errors = {}
        for ion, channels in (("ina", ("NaTg", "Nap")),
                              ("ik", ("K_P", "K_T", "Kv3_1", "Im", "SK")),
                              ("ica", ("Ca_HVA", "Ca_LVA"))):
            total = sum(data[name + "_current_ma_cm2"] for name in channels)
            error = float(np.max(np.abs(total - data[ion + "_ma_cm2"])))
            assert error < 1e-8
            errors[ion] = error
        first = row["events"][0]
        samples = []
        for label, instant in (("peak", first["peak_sample_time_ms"]),
                               ("falling_minus20", first["fall_crossing_ms"]),
                               ("minimum", row["postspike_minima"][0]["time_ms"])):
            samples.append({"event": label, "time_ms": instant,
                            "voltage_mv": float(np.interp(instant, time, data["voltage_mv"])),
                            "outward_current_ma_cm2": {
                                key.removesuffix("_current_ma_cm2"): float(np.interp(instant, time, data[key]))
                                for key in data.files if key.endswith("_current_ma_cm2")
                                and not key.startswith("axon_")}})
        records.append({"current_na": row["current_na"], "ion_sum_max_errors_ma_cm2": errors,
                        "samples": samples})
    report = {"records": records,
              "limit": "Local channel observations; excludes axial, leak, Ih and applied current, so not a complete current balance"}
    (folder / "h01-pv-return-currents.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
