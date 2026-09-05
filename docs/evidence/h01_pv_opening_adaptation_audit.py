"""Check individual late intervals after changing axonal calcium removal."""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main(entry=False):
    """Save direct event, calcium, and current observations for both inputs.

    Parameters
    ----------
    entry : bool, optional
        Compare increased entry scaling with the slow-removal control.
    """
    folder = Path(__file__).parent
    label = "entry" if entry else "slow"
    targets = json.loads((folder / "h01-pv-two-current-candidate.json").read_text())["records"]
    records = []
    for index, suffix in enumerate(("019", "027")):
        control_stem = ("h01-pv-kv3-phase-opening" if suffix == "019"
                        else "h01-pv-opening-adaptation-control-027")
        if entry:
            control_stem = "h01-pv-opening-adaptation-slow-" + suffix
        candidate_stem = ("h01-pv-calcium-entry-" if entry else "h01-pv-opening-adaptation-slow-") + suffix
        pair, metas = {}, {}
        for name, stem in (("control", control_stem), (label, candidate_stem)):
            metas[name] = json.loads((folder / (stem + ".json")).read_text())
            data = np.load(folder / (stem + ".npz"))
            time, voltage = data["time_ms"], data["voltage_mv"]
            selected = (time >= 270.) & (time < 1270.)
            events = spike_datums(time[selected], voltage[selected])
            late = [e for e in events if e["rise_crossing_ms"] >= 1000. and e["peak_sample_voltage_mv"] > 0.]
            interval = late[1]["rise_crossing_ms"]-late[0]["rise_crossing_ms"] if len(late) >= 2 else None
            expected = .39879086425875565*data["axon_sk_gate"]*(data["axon_voltage_mv"]+85.)
            error = float(np.max(np.abs(expected-data["axon_sk_current_ma_cm2"])))
            assert error < 1e-10
            human = targets[index]["human_events"][0]
            errors = {key: events[0][key]-human[key] for key in (
                "rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")} if events else None
            observations = [{"time_ms": e["rise_crossing_ms"], **{
                key: float(np.interp(e["rise_crossing_ms"], time, data[key]))
                for key in ("axon_voltage_mv", "axon_calcium_mm", "axon_sk_gate", "axon_sk_current_ma_cm2")}}
                for e in events]
            pair[name] = {"events": events, "first_errors": errors,
                          "all_intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist(),
                          "late_pair": late[:2], "late_interval_ms": interval,
                          "axon_states_at_somatic_rises": observations, "sk_identity_max_error": error}
        for key in ("current_na", "geometry", "integration", "nseg_factor", "initial_voltage_mv",
                    "temperature_c", "bias_na", "sodium_h_tau_factor", "sodium_h_recovery_factor",
                    "sodium_h_slope_mv", "somatic_calva_factor", "somatic_kv3_factor",
                    "somatic_kv3_tau_factor", "somatic_kv3_close_factor", "conductance_intervention"):
            assert metas["control"][key] == metas[label][key], key
        assert metas["control"]["axon_calcium_decay_ms"] == (1000. if entry else None)
        assert metas[label]["axon_calcium_decay_ms"] == 1000.
        assert metas["control"].get("axon_calcium_gamma") is None
        if entry:
            assert metas[label]["axon_calcium_gamma"] == .002
        valid = all(pair[name]["late_interval_ms"] is not None for name in pair)
        delta = pair[label]["late_interval_ms"]-pair["control"]["late_interval_ms"] if valid else None
        onset = (pair[label]["events"][0]["rise_crossing_ms"]-pair["control"]["events"][0]["rise_crossing_ms"]
                 if pair[label]["events"] and pair["control"]["events"] else None)
        human_late = [e for e in targets[index]["human_events"] if e["rise_crossing_ms"] >= 1000.
                      and e["peak_sample_voltage_mv"] > 0.]
        human_interval = human_late[1]["rise_crossing_ms"]-human_late[0]["rise_crossing_ms"] if len(human_late) >= 2 else None
        decision = ("supported" if delta >= 1. else "rejected") if valid else "invalid_missing_late_pair"
        if entry:
            decision = "invalid_missing_late_pair"
            if valid and human_interval is not None:
                improves = abs(pair[label]["late_interval_ms"]-human_interval) < abs(pair["control"]["late_interval_ms"]-human_interval)
                decision = "supported" if improves else "rejected"
        records.append({"current_na": metas[label]["current_na"], **pair,
                        "human_late_interval_ms": human_interval,
                        "human_late_pair": human_late[:2], "late_interval_change_ms": delta,
                        "late_interval_decision": decision,
                        "onset_shift_ms": onset, "onset_shift_below_0_1ms": None if onset is None else abs(onset) < .1})
    report = {"records": records, "reserved_trace_used": False,
              "criterion": "Reduced absolute human late-interval error" if entry else "Late interval increases by at least 1 ms",
              "limit": "Conditional calcium-removal effect; no exclusive SK mediation or human qualification"}
    filename = "h01-pv-calcium-entry-audit.json" if entry else "h01-pv-opening-adaptation-audit.json"
    (folder / filename).write_text(json.dumps(report, indent=2))
    for row in records:
        print(row["current_na"], row["late_interval_decision"], row["late_interval_change_ms"], "onset shift", row["onset_shift_ms"])
        for name in ("control", label):
            print(name, "events", len(row[name]["events"]), "late interval", row[name]["late_interval_ms"],
                  "first errors", row[name]["first_errors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="store_true")
    main(entry=parser.parse_args().entry)
