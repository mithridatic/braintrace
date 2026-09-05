"""Evaluate separate opening and closing contributions to the first return."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums
from docs.evidence.h01_pv_return_crossings import downward_crossings


def main(within_build=False):
    """Save each phase response and apply the predeclared attribution rule.

    Parameters
    ----------
    within_build : bool, optional
        Use the separately specified four-case comparison in the phase build.
    """
    folder = Path(__file__).parent
    build = np.load(folder / "h01-pv-kv3-phase-control.npz")
    previous = np.load(folder / "h01-pv-kv3-fast-019.npz")
    identical = (np.array_equal(build["time_ms"], previous["time_ms"])
                 and np.array_equal(build["voltage_mv"], previous["voltage_mv"]))
    if not within_build:
        assert identical, "Original attribution invalid: exact cross-build control failed."
    human = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())["records"][0]["human_events"][0]
    rows, metas = {}, {}
    for name, stem in (("source", "h01-pv-kv3-phase-source" if within_build else "h01-pv-calva-half-019"),
                       ("both", "h01-pv-kv3-phase-both" if within_build else "h01-pv-kv3-phase-control"),
                       ("opening", "h01-pv-kv3-phase-opening"), ("closing", "h01-pv-kv3-phase-closing")):
        metas[name] = json.loads((folder / (stem + ".json")).read_text())
        data = np.load(folder / (stem + ".npz"))
        time, voltage = data["time_ms"], data["voltage_mv"]
        selected = (time >= 270.) & (time < 1270.)
        events = spike_datums(time[selected], voltage[selected])
        valid = len(events) >= 2 and all(e["peak_sample_voltage_mv"] > 0. for e in events[:2])
        minimum, crossing = None, None
        if valid:
            start, stop = events[0]["peak_sample_time_ms"], events[1]["rise_crossing_ms"]
            crossing = downward_crossings(time, voltage, start, stop, (-65.,))[0]
            ids = np.flatnonzero((time >= start) & (time < stop))
            j = ids[np.argmin(voltage[ids])]
            minimum = {"time_ms": float(time[j]), "voltage_mv": float(voltage[j]),
                       "time_from_peak_ms": float(time[j]-start)}
        errors = {key: events[0][key]-human[key] for key in (
            "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
        rows[name] = {"events": events, "first_errors": errors, "first_minimum": minimum,
                      "falling_minus65": crossing,
                      "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist()}
    for name, meta in metas.items():
        for key in ("current_na", "geometry", "integration", "nseg_factor", "initial_voltage_mv",
                    "temperature_c", "bias_na", "axon_calcium_decay_ms", "sodium_h_tau_factor",
                    "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor", "conductance_intervention"):
            assert meta[key] == metas["source"][key], key
        assert meta.get("somatic_kv3_factor", 1.) == 1.
        expected_base, expected_close = {"source": (1., 1. if within_build else None),
                                         "both": (.5, .5 if within_build else None),
                                         "opening": (.5, 1.), "closing": (1., .5)}[name]
        assert meta.get("somatic_kv3_tau_factor", 1.) == expected_base
        assert meta.get("somatic_kv3_close_factor") == expected_close
    valid = all(row["falling_minus65"] is not None
                and row["falling_minus65"]["first_time_ms"] is not None for row in rows.values())
    advances = {name: rows["source"]["falling_minus65"]["time_from_peak_ms"]
                - row["falling_minus65"]["time_from_peak_ms"] for name, row in rows.items()} if valid else None
    decision = "invalid_missing_spikes_or_crossing"
    if valid:
        decision = ("supported" if abs(advances["opening"]-advances["both"]) <= .05
                    and advances["closing"] < .05 else "rejected")
    report = {"records": rows, "return_advances_ms": advances, "attribution_decision": decision,
              "build_control_identical": bool(identical), "within_build_factorial": within_build,
              "reserved_trace_used": False,
              "limit": "Conditional phase split at 0.19 nA; no unique later-train mediator or human validation"}
    filename = "h01-pv-kv3-phase-factorial.json" if within_build else "h01-pv-kv3-phase-audit.json"
    (folder / filename).write_text(json.dumps(report, indent=2))
    print(decision, advances)
    for name, row in rows.items():
        print(name, "events", len(row["events"]), "minimum", row["first_minimum"],
              "first errors", row["first_errors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--within-build", action="store_true")
    main(within_build=parser.parse_args().within_build)
