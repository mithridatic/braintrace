"""Compare calcium response-time candidates on direct early and late intervals."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Retain direct timing errors and screen the third and late intervals."""
    folder = Path(__file__).parent
    target = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())["records"][0]["human_events"]
    human_initial = [target[i+1]["rise_crossing_ms"]-target[i]["rise_crossing_ms"] for i in range(3)]
    human_late = [e for e in target if e["rise_crossing_ms"] >= 1000. and e["peak_sample_voltage_mv"] > 0.][:2]
    assert len(human_late) == 2
    human_interval = human_late[1]["rise_crossing_ms"]-human_late[0]["rise_crossing_ms"]
    rows, metas = {}, {}
    for name, stem, gamma, decay in (("control", "h01-pv-burst-adaptation-019", .002, 1000.),
                                    ("004-300", "h01-pv-calcium-response-004-300", .004, 300.),
                                    ("008-150", "h01-pv-calcium-response-008-150", .008, 150.)):
        meta = json.loads((folder / (stem + ".json")).read_text())
        metas[name] = meta
        assert meta["axon_calcium_gamma"] == gamma and meta["axon_calcium_decay_ms"] == decay
        trace = np.load(folder / (stem + ".npz"))
        time, voltage = trace["time_ms"], trace["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        initial = []
        for i in range(3):
            valid = len(events) > i+1 and all(e["peak_sample_voltage_mv"] > 0. for e in events[:i+2])
            value = events[i+1]["rise_crossing_ms"]-events[i]["rise_crossing_ms"] if valid else None
            initial.append({"ordinal": i+1, "interval_ms": value,
                            "error_ms": None if value is None else value-human_initial[i]})
        late = [e for e in events if e["rise_crossing_ms"] >= 1000. and e["peak_sample_voltage_mv"] > 0.][:2]
        interval = late[1]["rise_crossing_ms"]-late[0]["rise_crossing_ms"] if len(late) == 2 else None
        late_error = None if interval is None else interval-human_interval
        first_errors = {key: events[0][key]-target[0][key] for key in (
            "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
        errors = [r["error_ms"] for r in initial]+[late_error]
        rows[name] = {"events": events, "initial_intervals": initial,
                      "all_intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                      "first_errors": first_errors, "late_pair": late, "late_interval_ms": interval,
                      "late_error_ms": late_error,
                      "root_sum_squared_four_interval_errors_ms": float(np.linalg.norm(errors)) if all(e is not None for e in errors) else None}
    for name in ("004-300", "008-150"):
        for key in ("current_na", "geometry", "integration", "nseg_factor", "initial_voltage_mv",
                    "temperature_c", "bias_na", "sodium_h_tau_factor", "sodium_h_recovery_factor",
                    "sodium_h_slope_mv", "somatic_calva_factor", "somatic_kv3_factor",
                    "somatic_kv3_tau_factor", "somatic_kv3_close_factor", "conductance_intervention"):
            assert metas[name][key] == metas["control"][key], key
        candidate, control = rows[name], rows["control"]
        pairs = ((candidate["initial_intervals"][2]["error_ms"], control["initial_intervals"][2]["error_ms"]),
                 (candidate["late_error_ms"], control["late_error_ms"]))
        valid = all(a is not None and b is not None for a, b in pairs)
        candidate["third_and_late_screen"] = ("supported" if all(abs(a) < abs(b) for a, b in pairs) else "rejected") if valid else "invalid_missing_intervals"
    report = {"records": rows, "human_initial_intervals_ms": human_initial,
              "human_late_pair": human_late, "human_late_interval_ms": human_interval,
              "reserved_trace_used": False,
              "limit": "Combined calibration candidates; four-interval RSS is a diagnostic error score, not uncertainty or full physiological validation"}
    (folder / "h01-pv-calcium-response-audit.json").write_text(json.dumps(report, indent=2))
    for name, row in rows.items():
        print(name, row.get("third_and_late_screen"), "initial", row["initial_intervals"],
              "late", row["late_interval_ms"], "RSS", row["root_sum_squared_four_interval_errors_ms"], "events", len(row["events"]))


if __name__ == "__main__":
    main()
