"""Check local charge balance with independently calculated axial currents."""

import json
from pathlib import Path

import numpy as np


def main():
    """Save current paths, residual traces, and first-spike observations."""
    folder = Path(__file__).parent
    trace = np.load(folder / "h01-pv-neuron-charge-balance.npz")
    control = np.load(folder / "h01-pv-neuron-bias-control.npz")
    metadata = json.loads((folder / "h01-pv-neuron-charge-balance.json").read_text())
    geometry = metadata["charge_balance_geometry"]
    assert np.array_equal(trace["time_ms"], control["time_ms"])
    assert np.array_equal(trace["voltage_mv"], control["voltage_mv"])
    time = trace["time_ms"]
    area_factor = geometry["area_um2"]*.01
    paths = {(n["location"]): (trace[n["voltage_key"]]-trace["voltage_mv"])/n["resistance_mohm"]
             for n in geometry["neighbours"]}
    axial = sum(paths.values())
    ionic = sum(trace[key] for key in ("ina_ma_cm2", "ik_ma_cm2", "ica_ma_cm2",
                                      "leak_current_ma_cm2", "Ih_current_ma_cm2"))*area_factor
    capacitive = trace["capacitive_current_ma_cm2"]*area_factor
    applied = trace["total_current_na"]
    residual = applied+axial-ionic-capacitive
    selection = (time > .001) & (abs(time-270.) > .001) & (abs(time-1270.) > .001)
    maximum = float(np.max(np.abs(residual[selection])))
    assert maximum < 1e-5
    # Independent integral check against the voltage-derived change in charge.
    selected = (time >= 302.) & (time <= 304.)
    local_time = time[selected]
    capacitance_nf = geometry["cm_uf_cm2"]*geometry["area_um2"]*1e-5
    charge_from_voltage = capacitance_nf*(trace["voltage_mv"][selected][-1]-trace["voltage_mv"][selected][0])
    charge_from_current = float(np.trapezoid(capacitive[selected], local_time))
    peak = metadata["spike_times_ms"][0]
    observations = {"applied_inward_na": float(np.interp(peak, time, applied)),
                    "axial_inward_na": float(np.interp(peak, time, axial)),
                    "ionic_outward_na": float(np.interp(peak, time, ionic)),
                    "capacitive_na": float(np.interp(peak, time, capacitive)),
                    "axial_paths_inward_na": {key: float(np.interp(peak, time, value)) for key, value in paths.items()}}
    report = {"location": geometry["location"], "area_um2": geometry["area_um2"],
              "maximum_balance_residual_na": maximum, "voltage_control_identical": True,
              "maximum_residual_if_axial_omitted_na": float(np.max(np.abs((residual-axial)[selection]))),
              "first_peak_time_ms": peak, "first_peak": observations,
              "charge_integral_check": {"actual_interval_ms": [float(local_time[0]), float(local_time[-1])],
                                        "from_voltage_pc": float(charge_from_voltage),
                                        "from_current_pc": charge_from_current,
                                        "difference_pc": float(charge_from_current-charge_from_voltage)},
              "limit": "Local accounting at one soma segment; no whole-cell energy or human physiological validation"}
    (folder / "h01-pv-charge-balance-audit.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(folder / "h01-pv-charge-balance-residual.npz", time_ms=time,
                        applied_inward_na=applied, axial_inward_na=axial,
                        ionic_outward_na=ionic, capacitive_na=capacitive, residual_na=residual)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
