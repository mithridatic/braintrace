"""Evaluate the Kv3 conductance split through matched-voltage return times."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums
from docs.evidence.h01_pv_return_crossings import downward_crossings


def main(fast=False):
    """Save events and distinguish invalid comparisons from rejected predictions.

    Parameters
    ----------
    fast : bool, optional
        Evaluate the gate-speed intervention instead of doubled conductance.
    """
    folder = Path(__file__).parent
    label = "fast" if fast else "double"
    if fast:
        build = np.load(folder / "h01-pv-kv3-fast-control-019.npz")
        original = np.load(folder / "h01-pv-calva-half-019.npz")
        assert np.array_equal(build["time_ms"], original["time_ms"])
        assert np.array_equal(build["voltage_mv"], original["voltage_mv"])
    targets = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())["records"]
    records = []
    for index, suffix in enumerate(("019", "027")):
        pair, metas = {}, {}
        for name, prefix in (("control", "h01-pv-calva-half-"), (label, "h01-pv-kv3-" + label + "-")):
            stem = prefix + suffix
            metas[name] = json.loads((folder / (stem + ".json")).read_text())
            trace = np.load(folder / (stem + ".npz"))
            time, voltage = trace["time_ms"], trace["voltage_mv"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            valid = len(events) >= 2 and all(e["peak_sample_voltage_mv"] > 0. for e in events[:2])
            minimum, crossings = None, []
            if valid:
                start, stop = events[0]["peak_sample_time_ms"], events[1]["rise_crossing_ms"]
                crossings = downward_crossings(time, voltage, start, stop, (-40., -50., -60., -65., -70., -75.))
                ids = np.flatnonzero((time >= start) & (time < stop))
                assert len(ids)
                j = ids[np.argmin(voltage[ids])]
                minimum = {"time_ms": float(time[j]), "voltage_mv": float(voltage[j]),
                           "time_from_peak_ms": float(time[j]-start)}
            human = targets[index]["human_events"][0]
            errors = {key: events[0][key]-human[key] for key in (
                "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
            pair[name] = {"events": events, "first_errors": errors, "first_minimum": minimum,
                          "falling_crossings": crossings,
                          "intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist()}
        for key in ("current_na", "geometry", "integration", "nseg_factor", "initial_voltage_mv",
                    "temperature_c", "bias_na", "axon_calcium_decay_ms", "sodium_h_tau_factor",
                    "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor", "conductance_intervention"):
            assert metas["control"][key] == metas[label][key], key
        assert metas["control"].get("somatic_kv3_factor", 1.) == 1.
        assert metas[label]["somatic_kv3_factor"] == (1. if fast else 2.)
        assert metas["control"].get("somatic_kv3_tau_factor", 1.) == 1.
        if fast:
            assert metas[label]["somatic_kv3_tau_factor"] == .5
        times = [next((r["time_from_peak_ms"] for r in pair[name]["falling_crossings"]
                       if r["level_mv"] == -65.), None) for name in ("control", label)]
        advance = times[0]-times[1] if all(t is not None for t in times) else None
        decision = ("supported" if advance >= .1 else "rejected") if advance is not None else "invalid_missing_spikes_or_crossing"
        onset_shift = (pair[label]["events"][0]["rise_crossing_ms"]
                       - pair["control"]["events"][0]["rise_crossing_ms"]) if pair[label]["events"] and pair["control"]["events"] else None
        records.append({"current_na": metas[label]["current_na"], **pair,
                        "onset_shift_ms": onset_shift,
                        "onset_shift_within_1ms": None if onset_shift is None else abs(onset_shift) <= 1.,
                        "minus65_advance_ms": advance, "directional_decision": decision})
    report = {"records": records, "reserved_trace_used": False,
              "unchanged_speed_build_control_identical": True if fast else None,
              "limit": "Conditional conductance effect; no physiological or numerical qualification"}
    filename = "h01-pv-kv3-fast-audit.json" if fast else "h01-pv-kv3-return-audit.json"
    (folder / filename).write_text(json.dumps(report, indent=2))
    for row in records:
        print(row["current_na"], row["directional_decision"], row["minus65_advance_ms"])
        print("minimum", row[label]["first_minimum"], "first errors", row[label]["first_errors"],
              "complete events", len(row[label]["events"]), "onset shift", row["onset_shift_ms"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    main(fast=parser.parse_args().fast)
