"""Check whether the combined candidate's late-spike failure survives refinement."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain event times and late voltage bounds on the two meshes."""
    folder = Path(__file__).parent
    rows, metas = [], []
    for factor, suffix in ((9, ""), (27, "-space27")):
        stem = "h01-pv-combined-015-027" + suffix
        meta = json.loads((folder / (stem + ".json")).read_text())
        metas.append(meta)
        assert meta["nseg_factor"] == factor
        trace = np.load(folder / (stem + ".npz"))
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        assert np.isfinite(voltage).all()
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        late = [e for e in events if e["rise_crossing_ms"] >= 500.
                and e["peak_sample_voltage_mv"] > 0.]
        late_samples = (time >= 500.) & (time < 1270.)
        rows.append({"spatial_factor": factor, "events": events,
                     "late_positive_events": late,
                     "late_sample_voltage_bounds_mv": [float(voltage[late_samples].min()),
                                                        float(voltage[late_samples].max())]})
    for key in ("current_na", "integration", "temperature_c", "initial_voltage_mv",
                "bias_na", "axon_calcium_decay_ms", "sodium_h_tau_factor",
                "sodium_h_recovery_factor", "conductance_intervention"):
        assert metas[0][key] == metas[1][key], key
    assert len(metas[0]["geometry"]) == len(metas[1]["geometry"])
    for coarse, fine in zip(metas[0]["geometry"], metas[1]["geometry"]):
        assert fine["nseg"] == 3 * coarse["nseg"]
        assert coarse["name"] == fine["name"] and coarse["parent"] == fine["parent"]
        # The reported diameter of a tapered section changes with discretization.
        # Retain that difference; check length and integrated area instead.
        for key in ("length_um", "area_um2", "ra_ohm_cm", "cm_uf_cm2"):
            assert np.isclose(coarse[key], fine[key], rtol=1e-10, atol=1e-10), key
    common = min(len(row["events"]) for row in rows)
    differences = [{key: rows[1]["events"][i][key] - rows[0]["events"][i][key]
                    for key in ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")}
                   for i in range(common)]
    report = {"records": rows,
              "reported_diameter_differences_um": {
                  a["name"]: b["diameter_um"]-a["diameter_um"]
                  for a, b in zip(metas[0]["geometry"], metas[1]["geometry"])},
              "no_late_positive_spikes_on_both_meshes": all(not row["late_positive_events"] for row in rows),
              "paired_event_differences_fine_minus_coarse": differences,
              "limit": "Two-mesh observation check, not full convergence or causal channel identification"}
    (folder / "h01-pv-failure-mesh-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
