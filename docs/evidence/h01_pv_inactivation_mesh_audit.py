"""Check the predeclared sodium-inactivation effect across spatial meshes."""

import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def main():
    """Save direct events and the change in the intervention effect."""
    folder = Path(__file__).parent
    records = {}
    for factor, suffix in ((9, ""), (27, "-space27")):
        cases = {}
        for case in ("control", "closure"):
            stem = f"h01-pv-neuron-split-{case}{suffix}"
            trace = np.load(folder / (stem+".npz"))
            meta = json.loads((folder / (stem+".json")).read_text())
            assert meta["nseg_factor"] == factor
            assert meta["sodium_h_recovery_factor"] == 1.
            assert meta["sodium_h_tau_factor"] == (1. if case == "control" else .5)
            selected = (trace["time_ms"] >= 270.) & (trace["time_ms"] < 1270.)
            events = spike_datums(trace["time_ms"][selected], trace["voltage_mv"][selected])
            cases[case] = {"trace": stem, "events": events, "spike_count": len(events)}
        effect = (cases["control"]["events"][0]["time_above_threshold_ms"]
                  - cases["closure"]["events"][0]["time_above_threshold_ms"])
        records[str(factor)] = {"cases": cases, "first_width_reduction_ms": effect,
                                "prediction_supported": effect > .02}
    effect_change = records["27"]["first_width_reduction_ms"]-records["9"]["first_width_reduction_ms"]
    report = {"records": records, "effect_change_between_meshes_ms": effect_change,
              "limit": "Two-mesh robustness of one causal effect; not full spatial convergence or physiological validation"}
    (folder / "h01-pv-inactivation-mesh-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"effect_change_ms": effect_change,
                      "effects_ms": {k: v["first_width_reduction_ms"] for k, v in records.items()},
                      "counts": {k: {c: r["spike_count"] for c, r in v["cases"].items()} for k, v in records.items()}}))


if __name__ == "__main__":
    main()
