"""Check restored firing and individual event changes on two spatial meshes."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save ordinal event pairs, unmatched events, and post-spike minima."""
    folder = Path(__file__).parent
    rows, metas = [], []
    for factor, suffix in ((9, ""), (27, "-space27")):
        stem = "h01-pv-slope5-027" + suffix
        meta = json.loads((folder / (stem + ".json")).read_text())
        metas.append(meta)
        assert meta["nseg_factor"] == factor and meta["sodium_h_slope_mv"] == 5.
        trace = np.load(folder / (stem + ".npz"))
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        late = [e for e in events if e["rise_crossing_ms"] >= 500. and e["peak_sample_voltage_mv"] > 0.]
        minima = []
        for first, second in zip(events[:3], events[1:4]):
            ids = np.flatnonzero((time >= first["peak_sample_time_ms"])
                                 & (time < second["rise_crossing_ms"]))
            assert len(ids)
            j = ids[np.argmin(voltage[ids])]
            minima.append({"time_ms": float(time[j]), "voltage_mv": float(voltage[j]),
                           "time_from_peak_ms": float(time[j] - first["peak_sample_time_ms"])})
        rows.append({"spatial_factor": factor, "events": events,
                     "late_positive_events": late, "postspike_minima": minima})
    for key in ("current_na", "integration", "temperature_c", "initial_voltage_mv",
                "bias_na", "axon_calcium_decay_ms", "sodium_h_tau_factor",
                "sodium_h_recovery_factor", "sodium_h_slope_mv", "conductance_intervention"):
        assert metas[0][key] == metas[1][key], key
    assert len(metas[0]["geometry"]) == len(metas[1]["geometry"])
    for coarse, fine in zip(metas[0]["geometry"], metas[1]["geometry"]):
        assert fine["nseg"] == 3 * coarse["nseg"]
        assert coarse["name"] == fine["name"] and coarse["parent"] == fine["parent"]
        for key in ("length_um", "area_um2", "ra_ohm_cm", "cm_uf_cm2"):
            assert np.isclose(coarse[key], fine[key], rtol=1e-10, atol=1e-10), key
    control = json.loads((folder / "h01-pv-failure-mesh-audit.json").read_text())
    assert all(not row["late_positive_events"] for row in control["records"])
    common = min(len(row["events"]) for row in rows)
    differences = [{"ordinal": i + 1, **{
        key: rows[1]["events"][i][key] - rows[0]["events"][i][key]
        for key in ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")}}
        for i in range(common)]
    report = {"records": rows,
              "rescue_on_both_meshes": all(len(row["late_positive_events"]) >= 2 for row in rows),
              "ordinal_differences_fine_minus_coarse": differences,
              "unmatched_events": {str(row["spatial_factor"]): row["events"][common:] for row in rows},
              "limit": "Two-mesh rescue check; ordinal pairing is not proof of full convergence or physiological validity"}
    (folder / "h01-pv-slope5-mesh-audit.json").write_text(json.dumps(report, indent=2))
    print("Rescue on both meshes:", report["rescue_on_both_meshes"])
    for row in rows:
        print(row["spatial_factor"], "events", len(row["events"]), "late", len(row["late_positive_events"]),
              "minima", row["postspike_minima"])
    if differences:
        print("First difference", differences[0], "last paired difference", differences[-1])


if __name__ == "__main__":
    main()
