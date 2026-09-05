"""Audit selected calcium-candidate intervals on two spatial meshes."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save direct responses and apply the prospective numerical limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-factor", type=int, choices=(27, 81), default=27)
    args = parser.parse_args()
    folder = Path(__file__).parent
    rows, metas = [], []
    for factor in (args.fine_factor // 3, args.fine_factor):
        suffix = "" if factor == 9 else f"-space{factor}"
        stem = "h01-pv-calcium-response-004-300-027" + suffix
        meta = json.loads((folder / (stem + ".json")).read_text())
        assert meta["nseg_factor"] == factor
        metas.append(meta)
        with np.load(folder / (stem + ".npz")) as trace:
            time, voltage = trace["time_ms"], trace["voltage_mv"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
        early = None
        if len(events) >= 4 and all(e["peak_sample_voltage_mv"] > 0 for e in events[:4]):
            early = np.diff([e["rise_crossing_ms"] for e in events[:4]]).tolist()
        late = [e for e in events if e["rise_crossing_ms"] >= 1000.
                and e["peak_sample_voltage_mv"] > 0][:2]
        interval = late[1]["rise_crossing_ms"] - late[0]["rise_crossing_ms"] if len(late) == 2 else None
        rows.append({"factor": factor, "events": events, "initial_intervals_ms": early,
                     "late_pair": late, "late_interval_ms": interval})
    keys = ("current_na", "integration", "temperature_c", "initial_voltage_mv", "bias_na",
            "axon_calcium_decay_ms", "axon_calcium_gamma", "sodium_h_tau_factor",
            "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor",
            "somatic_kv3_factor", "somatic_kv3_tau_factor", "somatic_kv3_close_factor",
            "conductance_intervention", "refine_region")
    for key in keys:
        assert metas[0][key] == metas[1][key], key
    assert len(metas[0]["geometry"]) == len(metas[1]["geometry"])
    for coarse, fine in zip(metas[0]["geometry"], metas[1]["geometry"]):
        assert fine["nseg"] == 3 * coarse["nseg"]
        assert coarse["name"] == fine["name"] and coarse["parent"] == fine["parent"]
        for key in ("length_um", "area_um2", "ra_ohm_cm", "cm_uf_cm2"):
            assert np.isclose(coarse[key], fine[key], rtol=1e-10, atol=1e-10), key
    valid = all(r["initial_intervals_ms"] is not None and r["late_interval_ms"] is not None for r in rows)
    early_diff = (np.array(rows[1]["initial_intervals_ms"]) - rows[0]["initial_intervals_ms"]).tolist() if valid else None
    late_diff = rows[1]["late_interval_ms"] - rows[0]["late_interval_ms"] if valid else None
    passed = valid and all(abs(x) <= .05 for x in early_diff) and abs(late_diff) <= .25
    common = min(len(r["events"]) for r in rows)
    differences = [{"ordinal": i + 1, **{
        key: rows[1]["events"][i][key] - rows[0]["events"][i][key]
        for key in ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")}}
        for i in range(common)]
    report = {"records": rows, "initial_interval_changes_ms": early_diff,
              "late_interval_change_ms": late_diff,
              "diagnostic_limits_ms": {"each_initial_interval": .05, "late_interval": .25},
              "decision": "supported" if passed else "rejected" if valid else "invalid_missing_intervals",
              "ordinal_differences_fine_minus_coarse": differences,
              "unmatched_events": {str(r["factor"]): r["events"][common:] for r in rows},
              "limit": "Two-mesh interval diagnostic only; full trajectory and physiological qualification remain open"}
    output = "h01-pv-calcium-response-mesh" + ("-81" if args.fine_factor == 81 else "")
    (folder / (output + ".json")).write_text(json.dumps(report, indent=2))
    print(report["decision"], "initial changes", early_diff, "late change", late_diff)
    print("event counts", [len(r["events"]) for r in rows])
    if differences:
        print("first", differences[0], "last", differences[-1])


if __name__ == "__main__":
    main()
