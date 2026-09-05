"""Observe local current balance at the candidate's first upward crossings."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_onset_localization import crossings


def main():
    """Save direct crossing observations after checking probe neutrality."""
    folder = Path(__file__).parent
    records = []
    for suffix in ("019", "027"):
        base = f"h01-pv-onset-balance-{suffix}"
        trace = np.load(folder / (base + ".npz"))
        control = np.load(folder / f"h01-pv-neuron-waveform-018-{suffix}.npz")
        meta = json.loads((folder / (base + ".json")).read_text())
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        assert np.array_equal(time, control["time_ms"])
        assert np.array_equal(voltage, control["voltage_mv"])
        geometry = meta["charge_balance_geometry"]
        factor = geometry["area_um2"] * .01
        currents = {
            key.removesuffix("_ma_cm2") + "_na": trace[key] * factor
            for key in trace.files if key.endswith("_current_ma_cm2")
            and not key.startswith("axon_")
        }
        currents["ionic_outward_na"] = sum(trace[key] for key in (
            "ina_ma_cm2", "ik_ma_cm2", "ica_ma_cm2",
            "leak_current_ma_cm2", "Ih_current_ma_cm2")) * factor
        currents["axial_inward_na"] = sum(
            (trace[n["voltage_key"]] - voltage) / n["resistance_mohm"]
            for n in geometry["neighbours"])
        currents["applied_inward_na"] = trace["total_current_na"]
        residual = (currents["applied_inward_na"] + currents["axial_inward_na"]
                    - currents["ionic_outward_na"] - currents["capacitive_current_na"])
        selected = ((time > .001) & (abs(time - 270.) > .001)
                    & (abs(time - 1270.) > .001))
        maximum = float(np.max(np.abs(residual[selected])))
        assert maximum < 1e-5
        rows = []
        for crossing in crossings(time, voltage, meta["spike_times_ms"][0]):
            instant = crossing["first_crossing_ms"]
            assert instant is not None and crossing["crossing_count"] == 1
            rows.append({**crossing, "currents": {
                key: float(np.interp(instant, time, values))
                for key, values in currents.items()}})
        records.append({"current_na": meta["current_na"],
                        "probe_voltage_and_time_identical": True,
                        "maximum_balance_residual_na": maximum,
                        "location": geometry["location"], "crossings": rows})
    report = {"records": records, "reserved_trace_used": False,
              "convention": "Channel and capacitive currents outward positive; axial and applied inward positive",
              "limit": "Local current accounting, not a causal intervention or identification of human channel currents"}
    (folder / "h01-pv-onset-balance-audit.json").write_text(json.dumps(report, indent=2))
    for record in records:
        print(record["current_na"], record["maximum_balance_residual_na"])
        for row in record["crossings"]:
            if row["voltage_mv"] in (-75., -70., -65.):
                print(json.dumps(row))


if __name__ == "__main__":
    main()
