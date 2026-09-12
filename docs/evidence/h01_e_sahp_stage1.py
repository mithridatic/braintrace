"""SP13 stage 1: the surviving KsAHP candidate at 250, 310 and 110 pA.

Bands registered in ``docs/evidence/h01-e-sahp-tail-manifest.json`` (stage 1):
250 pA count 5-7 with cycles 4-5 of 250-350 ms; 310 pA count 9-10 with late cycles
(mean of the last three intervals) within 12.8 ms of 120 ms; every sweep-43 onset row
(1019-1120 ms) within 1 mV of the human sample and within 0.1 mV of ``g0-b3``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_e_current_paths import cycle_summary, load_run

EVIDENCE = Path(__file__).resolve().parent
LATE_MS = 12.8
ONSET_MS = (1019., 1120.)
HUMAN_MV = 1.
B3_MV = .1


def _band(row, predicted, observed, held, input_label):
    return {"row": row, "predicted": predicted, "observed": observed, "held": bool(held), "input": input_label}


def intervals(summary):
    """Spike intervals in ms; interval k is the cycle before spike k+1 (cycle numbering of the usable tables)."""
    return list(np.diff(summary["peak_ms"]))


def bands_250(summary):
    gaps = intervals(summary)
    cycles = {f"cycle_{k+1}_ms": gaps[k-1] for k in (3, 4) if k-1 < len(gaps)}
    rows = [_band("count", "5-7", summary["count"], 5 <= summary["count"] <= 7, "250 pA")]
    for k in (4, 5):
        value = cycles.get(f"cycle_{k}_ms")
        rows.append(_band(f"cycle_{k}_ms", "250-350", value, value is not None and 250. <= value <= 350., "250 pA"))
    return rows


def bands_310(summary):
    gaps = intervals(summary)
    late = float(np.mean(gaps[-3:])) if len(gaps) >= 3 else None
    return [_band("count", "9-10", summary["count"], 9 <= summary["count"] <= 10, "310 pA"),
            _band("late_cycle_mean_ms", f"120 +/- {LATE_MS}", late, late is not None and abs(late-120.) <= LATE_MS, "310 pA")]


def bands_43(data, reference_rows):
    """Onset rows: model interpolated at each sample time against the human sample and the g0-b3 model value."""
    t, v = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    rows = []
    for ref in reference_rows:
        if ref["kind"] != "subthreshold_voltage_mv":
            continue
        requested = float(ref["key"].split("_")[1])
        if not ONSET_MS[0] <= requested <= ONSET_MS[1]:
            continue
        model = float(np.interp(requested, t, v))
        rows.append(_band(f"{ref['key']}_vs_human", f"|x| <= {HUMAN_MV}", model-ref["human"], abs(model-ref["human"]) <= HUMAN_MV, "110 pA"))
        rows.append(_band(f"{ref['key']}_vs_b3", f"|x| <= {B3_MV}", model-ref["model"], abs(model-ref["model"]) <= B3_MV, "110 pA"))
    return rows


def decide(folder, name, b3_contract):
    reference_rows = json.loads(Path(b3_contract).read_text(encoding="utf-8"))["records"][0]["rows"]
    runs = {label: load_run(folder, f"{name}-{stem}") for label, stem in (("250 pA", "sweep50"), ("310 pA", "sweep53"), ("110 pA", "sweep43"))}
    s250, s310 = cycle_summary(runs["250 pA"][0]), cycle_summary(runs["310 pA"][0])
    s43 = cycle_summary(runs["110 pA"][0])
    bands = bands_250(s250)+bands_310(s310)+bands_43(runs["110 pA"][0], reference_rows)
    bands.append(_band("count", 0, s43["count"], s43["count"] == 0, "110 pA"))
    held = sum(b["held"] for b in bands)
    return {"stage": "1", "candidate": name, "bands": bands, "bands_held": held, "bands_total": len(bands),
            "decision": "PASS" if held == len(bands) else "FAIL",
            "summaries": {"250 pA": {**s250, "intervals_ms": intervals(s250)}, "310 pA": {**s310, "intervals_ms": intervals(s310)},
                          "110 pA": s43},
            "wall_seconds": {label: run[1].get("integration_seconds") for label, run in runs.items()},
            "candidate_sha256": runs["250 pA"][1].get("candidate_json", {}).get("sha256")}


def render(decision):
    lines = [f"# SP13 stage 1: {decision['decision']} ({decision['bands_held']} of {decision['bands_total']} bands)", "",
             "| Input | Row | Predicted | Observed | Held |", "| --- | --- | --- | ---: | --- |"]
    for b in decision["bands"]:
        obs = f"{b['observed']:.3f}" if isinstance(b["observed"], float) else b["observed"]
        lines.append(f"| {b['input']} | {b['row']} | {b['predicted']} | {obs} | {b['held']} |")
    for label in ("250 pA", "310 pA"):
        s = decision["summaries"][label]
        lines += ["", f"{label}: count {s['count']}; peaks {[round(p, 1) for p in s['peak_ms']]} ms; intervals {[round(g, 1) for g in s['intervals_ms']]} ms."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--b3-contract", type=Path, default=EVIDENCE/"h01-e-gain/g0-b3-sweep43-contract.json")
    parser.add_argument("--output-stem", default="stage-1-decision")
    args = parser.parse_args(argv)
    decision = decide(args.folder, args.candidate, args.b3_contract)
    (args.folder/f"{args.output_stem}.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
    (args.folder/f"{args.output_stem}.md").write_text(render(decision), encoding="utf-8")
    print(render(decision))


if __name__ == "__main__":
    main()
