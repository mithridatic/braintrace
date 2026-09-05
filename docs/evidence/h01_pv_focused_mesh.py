"""Compare an axon-focused mesh with the full factor-81 reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Check geometry and every ordinal onset against declared limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-factor", type=int, choices=(243, 729, 2187))
    args = parser.parse_args()
    folder = Path(__file__).parent
    records, metas = [], []
    cases = (("full", "h01-pv-calcium-response-004-300-027-space81"),
             ("focused", "h01-pv-regional-mesh-axon81"))
    if args.fine_factor:
        cases = tuple((str(f), f"h01-pv-regional-mesh-axon{f}")
                      for f in (args.fine_factor // 3, args.fine_factor))
    for name, stem in cases:
        meta = json.loads((folder / (stem + ".json")).read_text())
        metas.append(meta)
        with np.load(folder / (stem + ".npz")) as trace:
            t, v = trace["time_ms"], trace["voltage_mv"]
            assert np.isfinite(t).all() and np.isfinite(v).all()
            mask = (t >= 270.) & (t < 1270.)
            events = spike_datums(t[mask], v[mask])
        late = [e for e in events if e["rise_crossing_ms"] >= 1000.
                and e["peak_sample_voltage_mv"] > 0][:2]
        interval = late[1]["rise_crossing_ms"] - late[0]["rise_crossing_ms"] if len(late) == 2 else None
        records.append({"name": name, "events": events, "late_pair": late,
                        "late_interval_ms": interval, "segments": sum(s["nseg"] for s in meta["geometry"])})
    for key in ("current_na", "integration", "temperature_c", "initial_voltage_mv", "bias_na",
                "axon_calcium_decay_ms", "axon_calcium_gamma", "sodium_h_tau_factor",
                "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor",
                "somatic_kv3_factor", "somatic_kv3_tau_factor", "somatic_kv3_close_factor",
                "conductance_intervention", "source_commit", "neuron_version"):
        assert metas[0][key] == metas[1][key], key
    if args.fine_factor:
        for meta, factor in zip(metas, (args.fine_factor // 3, args.fine_factor)):
            assert meta["refine_region"] == "axon" and meta["nseg_factor"] == factor
            assert meta["unselected_nseg_factor"] == 9
    else:
        assert metas[0]["refine_region"] == "all" and metas[0]["nseg_factor"] == 81
        assert metas[1]["refine_region"] == "axon" and metas[1]["nseg_factor"] == 81
        assert metas[1]["unselected_nseg_factor"] == 9
    assert len(metas[0]["geometry"]) == len(metas[1]["geometry"])
    for full, focused in zip(metas[0]["geometry"], metas[1]["geometry"]):
        assert full["name"] == focused["name"] and full["parent"] == focused["parent"]
        family = full["name"].split(".", 1)[1].split("[", 1)[0]
        if args.fine_factor:
            assert focused["nseg"] == full["nseg"] * (3 if family == "axon" else 1)
        else:
            assert full["nseg"] == focused["nseg"] * (1 if family == "axon" else 9)
        for key in ("length_um", "area_um2", "ra_ohm_cm", "cm_uf_cm2"):
            assert np.isclose(full[key], focused[key], rtol=1e-10, atol=1e-10), key
    full, focused = records
    count_equal = len(full["events"]) == len(focused["events"])
    common = min(len(r["events"]) for r in records)
    differences = [{"ordinal": i + 1, **{
        key: focused["events"][i][key] - full["events"][i][key]
        for key in ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")}}
        for i in range(common)]
    classification_equal = count_equal and all((a["peak_sample_voltage_mv"] > 0) == (b["peak_sample_voltage_mv"] > 0)
                                              for a, b in zip(full["events"], focused["events"]))
    valid = common > 0 and all(r["late_interval_ms"] is not None for r in records)
    maximum = max(abs(d["rise_crossing_ms"]) for d in differences) if differences else None
    late = focused["late_interval_ms"] - full["late_interval_ms"] if valid else None
    passed = valid and classification_equal and maximum <= .1 and abs(late) <= .01
    report = {"records": records, "ordinal_differences_focused_minus_full": differences,
              "equal_counts": count_equal, "equal_event_classification": classification_equal,
              "max_onset_difference_ms": maximum, "late_interval_difference_ms": late,
              "limits_ms": {"every_onset": .1, "late_interval": .01},
              "decision": "supported" if passed else "rejected" if valid else "invalid_missing_events",
              "unmatched_events": {r["name"]: r["events"][common:] for r in records},
              "limit": "Agreement with the full factor-81 reference, not proof that the reference is converged or physiologically valid"}
    if args.fine_factor:
        report["ordinal_differences_fine_minus_coarse"] = report.pop("ordinal_differences_focused_minus_full")
        report["limit"] = "Successive axonal refinement at one input and parameter set; no physiological qualification"
    suffix = f"-{args.fine_factor}" if args.fine_factor else ""
    (folder / ("h01-pv-focused-mesh" + suffix + ".json")).write_text(json.dumps(report, indent=2))
    print(report["decision"], "max onset", maximum, "late difference", late,
          "counts", [len(r["events"]) for r in records], "segments", [r["segments"] for r in records])


if __name__ == "__main__":
    main()
