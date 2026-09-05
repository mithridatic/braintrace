"""Check channel-current accounting and record first-spike gate observations."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Write directly observed current and gate values at defined events."""
    folder = Path(__file__).parent
    trace = np.load(folder / "h01-pv-neuron-spike-currents.npz")
    control = np.load(folder / "h01-pv-neuron-bias-control.npz")
    assert np.array_equal(trace["time_ms"], control["time_ms"])
    assert np.array_equal(trace["voltage_mv"], control["voltage_mv"])
    errors = {}
    for ion, mechanisms in (("ina", ("NaTg", "Nap")),
                             ("ik", ("K_P", "K_T", "Kv3_1", "Im", "SK")),
                             ("ica", ("Ca_HVA", "Ca_LVA"))):
        total = sum(trace[m+"_current_ma_cm2"] for m in mechanisms)
        errors[ion] = float(np.max(np.abs(trace[ion+"_ma_cm2"]-total)))
        assert errors[ion] < 1e-12
    time = trace["time_ms"]
    selected = (time >= 270.) & (time < 1270.)
    event = spike_datums(time[selected], trace["voltage_mv"][selected])[0]
    peak = event["peak_sample_time_ms"]
    points = {"rising_minus20": event["rise_crossing_ms"], "peak": peak,
              "peak_plus_0p1_ms": peak+.1, "peak_plus_0p3_ms": peak+.3,
              "falling_minus20": event["fall_crossing_ms"]}
    keys = ["voltage_mv", "NaTg_m", "NaTg_h", "Kv3_1_m"]
    keys += [key for key in trace.files if key.endswith("_current_ma_cm2")]
    observations = {name: {"time_ms": value,
                           **{key: float(np.interp(value, time, trace[key])) for key in keys}}
                    for name, value in points.items()}
    report = {"source": "h01-pv-neuron-spike-currents.npz",
              "conditions": "0.19 nA; zero bias; initial -80 mV; mesh factor 9; CVode atol 1e-10",
              "voltage_and_time_identical_to_control": True,
              "maximum_ion_current_sum_error_ma_cm2": errors,
              "current_sign": "outward positive", "observations": observations,
              "limit": "Somatic channel observations only; no full axial or capacitive balance; not human channel measurements"}
    (folder / "h01-pv-spike-current-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(errors))


if __name__ == "__main__":
    main()
