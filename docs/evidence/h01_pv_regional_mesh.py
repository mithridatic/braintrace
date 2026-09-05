"""Locate regional contributions to the selected late-interval mesh effect."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Verify each isolated mesh and retain its direct event responses."""
    folder = Path(__file__).parent
    stems = {"coarse": "h01-pv-calcium-response-004-300-027",
             "fine": "h01-pv-calcium-response-004-300-027-space27"}
    stems.update({r: "h01-pv-regional-mesh-" + r for r in ("axon", "soma", "dendrites")})
    metas, rows = {}, {}
    for name, stem in stems.items():
        metas[name] = json.loads((folder / (stem + ".json")).read_text())
        with np.load(folder / (stem + ".npz")) as trace:
            t, v = trace["time_ms"], trace["voltage_mv"]
            assert np.isfinite(t).all() and np.isfinite(v).all()
            mask = (t >= 270.) & (t < 1270.)
            events = spike_datums(t[mask], v[mask])
        late = [e for e in events if e["rise_crossing_ms"] >= 1000.
                and e["peak_sample_voltage_mv"] > 0][:2]
        interval = late[1]["rise_crossing_ms"] - late[0]["rise_crossing_ms"] if len(late) == 2 else None
        rows[name] = {"events": events, "late_pair": late, "late_interval_ms": interval,
                      "all_ordinal_intervals_ms": np.diff([e["rise_crossing_ms"] for e in events]).tolist()}
    for name in stems:
        for key in ("current_na", "integration", "temperature_c", "initial_voltage_mv", "bias_na",
                    "axon_calcium_decay_ms", "axon_calcium_gamma", "sodium_h_tau_factor",
                    "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor",
                    "somatic_kv3_factor", "somatic_kv3_tau_factor", "somatic_kv3_close_factor",
                    "conductance_intervention", "source_commit", "neuron_version"):
            assert metas[name][key] == metas["coarse"][key], (name, key)
        geometry = metas[name]["geometry"]
        assert len(geometry) == len(metas["coarse"]["geometry"])
        for coarse, actual in zip(metas["coarse"]["geometry"], geometry):
            assert actual["name"] == coarse["name"] and actual["parent"] == coarse["parent"]
            family = actual["name"].split(".", 1)[1].split("[", 1)[0]
            selected = name == "fine" or family == name or (name == "dendrites" and family in ("dend", "apic"))
            assert actual["nseg"] == coarse["nseg"] * (3 if selected else 1), (name, actual["name"])
            for key in ("length_um", "area_um2", "ra_ohm_cm", "cm_uf_cm2"):
                assert np.isclose(actual[key], coarse[key], rtol=1e-10, atol=1e-10), (name, key)
        if name in ("axon", "soma", "dendrites"):
            assert metas[name]["refine_region"] == name
            assert metas[name]["nseg_factor"] == 27 and metas[name]["unselected_nseg_factor"] == 9
    for name in ("axon", "soma", "dendrites"):
        value, coarse, fine = (rows[r]["late_interval_ms"] for r in (name, "coarse", "fine"))
        valid = all(x is not None for x in (value, coarse, fine))
        rows[name]["late_change_from_coarse_ms"] = value - coarse if valid else None
        rows[name]["late_difference_from_fine_ms"] = value - fine if valid else None
        rows[name]["reproduces_full_late_effect"] = ("supported" if abs(value-fine) <= .25 else "rejected") if valid else "invalid_missing_pair"
    report = {"records": rows, "late_sufficiency_limit_ms": .25,
              "limit": "Regional numerical sufficiency only; effects need not add and no unique biological cause is established"}
    (folder / "h01-pv-regional-mesh.json").write_text(json.dumps(report, indent=2))
    for name, row in rows.items():
        print(name, "events", len(row["events"]), "late", row["late_interval_ms"],
              "decision", row.get("reproduces_full_late_effect"))


if __name__ == "__main__":
    main()
