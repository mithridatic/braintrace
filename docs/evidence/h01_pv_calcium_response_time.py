"""Compare all event onsets at two CVode tolerances on the same mesh."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain direct differences and apply prospective tolerance limits."""
    folder = Path(__file__).parent
    records, metas = [], []
    for atol, suffix in ((1e-8, "-atol8"), (1e-10, "")):
        stem = "h01-pv-calcium-response-004-300-027-space27" + suffix
        meta = json.loads((folder / (stem + ".json")).read_text())
        assert meta["integration"] == {"method": "CVode", "cvode_atol": atol}
        assert meta["nseg_factor"] == 27
        metas.append(meta)
        with np.load(folder / (stem + ".npz")) as trace:
            t, v = trace["time_ms"], trace["voltage_mv"]
            assert np.isfinite(t).all() and np.isfinite(v).all()
            mask = (t >= 270.) & (t < 1270.)
            events = spike_datums(t[mask], v[mask])
        early = None
        if len(events) >= 4 and all(e["peak_sample_voltage_mv"] > 0 for e in events[:4]):
            early = np.diff([e["rise_crossing_ms"] for e in events[:4]]).tolist()
        late = [e for e in events if e["rise_crossing_ms"] >= 1000.
                and e["peak_sample_voltage_mv"] > 0][:2]
        interval = late[1]["rise_crossing_ms"] - late[0]["rise_crossing_ms"] if len(late) == 2 else None
        records.append({"atol": atol, "events": events, "initial_intervals_ms": early,
                        "late_pair": late, "late_interval_ms": interval})
    for key in ("geometry", "current_na", "temperature_c", "initial_voltage_mv", "bias_na",
                "axon_calcium_decay_ms", "axon_calcium_gamma", "sodium_h_tau_factor",
                "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor",
                "somatic_kv3_factor", "somatic_kv3_tau_factor", "somatic_kv3_close_factor",
                "conductance_intervention", "refine_region", "source_commit", "neuron_version"):
        assert metas[0][key] == metas[1][key], key
    common = min(len(r["events"]) for r in records)
    differences = [{"ordinal": i + 1, **{
        key: records[1]["events"][i][key] - records[0]["events"][i][key]
        for key in ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")}}
        for i in range(common)]
    valid = all(r["initial_intervals_ms"] is not None and r["late_interval_ms"] is not None for r in records)
    early = (np.array(records[1]["initial_intervals_ms"]) - records[0]["initial_intervals_ms"]).tolist() if valid else None
    late = records[1]["late_interval_ms"] - records[0]["late_interval_ms"] if valid else None
    largest = max(abs(d["rise_crossing_ms"]) for d in differences) if differences else None
    equal_count = len(records[0]["events"]) == len(records[1]["events"])
    passed = valid and equal_count and largest <= .01 and all(abs(x) <= .01 for x in early) and abs(late) <= .01
    report = {"records": records, "ordinal_differences_tight_minus_loose": differences,
              "initial_interval_changes_ms": early, "late_interval_change_ms": late,
              "max_ordinal_onset_change_ms": largest, "equal_event_counts": equal_count,
              "limit_ms": .01,
              "decision": "supported" if passed else "rejected" if valid else "invalid_missing_intervals",
              "unmatched_events": {str(r["atol"]): r["events"][common:] for r in records},
              "qualification": "Tolerance sensitivity only; no spatial or physiological qualification"}
    (folder / "h01-pv-calcium-response-time.json").write_text(json.dumps(report, indent=2))
    print(report["decision"], "maximum onset change", largest, "early", early, "late", late,
          "event counts", [len(r["events"]) for r in records])


if __name__ == "__main__":
    main()
