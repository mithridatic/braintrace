"""Render the layer-2 family campaign result page from the recorded ranking files.

Reads ``stage-a-ranking.json`` (with per-group rankings), the optional Stage B
and Stage C vectors, and the campaign logs. Writes ``h01-l2-campaign-result.md``
beside the manifest. No number is typed by hand.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence import h01_l2_campaign_score as score

FOLDER = Path(__file__).parent
SHOW = ("event_count_model", "i1", "i2", "i3", "i4", "m1_v", "m2_v", "e1_rise_to_peak", "e1_peak_v", "sub_1120_v", "sub_2120_v")


def fmt(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "missing"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def residual_table(vectors, names):
    lines = ["| Run | " + " | ".join(SHOW) + " |", "| --- |" + " ---: |"*len(SHOW)]
    for name in names:
        row = vectors.get(name)
        if row is None:
            continue
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(k)) for k in SHOW) + " |")
    return "\n".join(lines)


def group_table(groups, families):
    lines = ["| Group | Keys | " + " | ".join(families) + " | Steep X |", "| --- | ---: |" + " ---: |"*len(families) + " --- |"]
    for group, row in groups.items():
        if group == "event_count":
            cells = [f"{row['families'][f]['into_source']:+d} / {row['families'][f]['out_of_candidate']:+d}" for f in families]
            lines.append(f"| event count change (into S / out of C) | 1 | " + " | ".join(cells) + f" | S {row['source']}, C {row['candidate']} |")
            continue
        cells = [fmt(row["families"].get(f, {}).get("combined")) for f in families]
        lines.append(f"| {group} | {len(row['keys'])} | " + " | ".join(cells) + f" | {row['steep_x'] or 'none'} |")
    return "\n".join(lines)


def prediction_check(vectors, limits, pair, stage_b):
    """Literal evaluation of the Stage B prediction on the recorded vectors."""
    into, out = vectors.get(f"ab-into-{pair}"), vectors.get(f"ab-out-{pair}")
    if into is None or out is None:
        return {"evaluated": False}
    checks = {"into_has_five_events": into["event_count_model"] == 5,
              "into_interval_2_and_4_positive": (into.get("i2") or -1) > 0 and (into.get("i4") or -1) > 0,
              "into_subthreshold_near_candidate": abs(into["sub_1120_v"]-vectors["candidate"]["sub_1120_v"]) < .05,
              "out_has_extra_events": out["event_count_model"] > 5,
              "out_late_intervals_negative": all((out.get(f"i{i}") or 0) < 0 for i in (2, 3, 4) if out.get(f"i{i}") is not None),
              "out_subthreshold_near_source": abs(out["sub_1120_v"]-vectors["source"]["sub_1120_v"]) < .05}
    return {"evaluated": True, "checks": checks, "holds": all(checks.values()), "prediction": stage_b}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    folder = FOLDER/manifest["output_dir"]
    ranking = json.loads((folder/"stage-a-ranking.json").read_text())
    limits = score.numerical_limits(*(json.loads((FOLDER/n).read_text()) for n in (
        "h01-l2-density130-tolerance-result.json", "h01-l2-density130-spatial-result.json",
        "h01-l2-density130-minima-numerical-review.json")))
    datums = score.load_datums()
    vectors = dict(ranking["vectors"])
    for candidate in manifest["candidates"]:
        if str(candidate["stage"]) in ("B", "C") and (folder/f"{candidate['name']}-sweep43.npz").exists():
            vectors[candidate["name"]] = {k: (None if not np.isfinite(v) else v) for k, v in score.score_candidate(folder, candidate["name"], datums).items()}
    pair = "F3F5"
    check = prediction_check(vectors, limits, pair, manifest.get("stage_b_prediction"))
    families = manifest["families"]
    names = ["source", "candidate", "corner"] + [f"{d}-{f}" for f in families for d in ("into", "out")] + [f"ab-into-{pair}", f"ab-out-{pair}", f"c-into-{pair}-atol11", f"c-out-{pair}-atol11"]
    stage_c = {}
    for direction in ("into", "out"):
        b, c = vectors.get(f"ab-{direction}-{pair}"), vectors.get(f"c-{direction}-{pair}-atol11")
        if b and c:
            control = vectors["source"] if direction == "into" else vectors["candidate"]
            changes = {k: c[k]-b[k] for k in ranking["keys"] if b.get(k) is not None and c.get(k) is not None}
            exceeding = {k: {"change": v, "limit": score.limit_for(k, limits)[0],
                             "claimed_contrast": None if control.get(k) is None else b[k]-control[k]}
                         for k, v in changes.items() if abs(v) > score.limit_for(k, limits)[0]}
            ratios = [abs(v)/abs(b[k]-control[k]) for k, v in changes.items() if control.get(k) is not None and b[k] != control[k]]
            stage_c[direction] = {"max_change": max(abs(v) for v in changes.values()), "event_count_unchanged": b["event_count_model"] == c["event_count_model"],
                                  "within_limits": not exceeding, "exceeding": exceeding,
                                  "max_change_over_claimed_contrast": max(ratios) if ratios else None}
    text = ["# Layer-2 family campaign result", "",
            "Specification: [2026-09-05-h01-l2-family-campaign](../specs/2026-09-05-h01-l2-family-campaign.md).",
            "Manifest: [" + args.manifest.name + "](" + args.manifest.name + "). Ranking: [stage-a-ranking.json](" + manifest["output_dir"] + "/stage-a-ranking.json).",
            "Every residual is model minus human. No physiological allowance is agreed; this page issues no pass.", "",
            "## Stage 0", "",
            "Coarse settings (nseg 3) fail the preservation rule on five phase measurements, so every family evaluation ran at nseg 9, CVode 1e-10, through 2140 ms. See [stage0-preservation.json](" + manifest["output_dir"] + "/stage0-preservation.json).", "",
            "## Stage A: family dissection", "",
            f"Ranked observations: {len(ranking['keys'])} keys whose source-to-candidate contrast exceeds five numerical decision limits. Pooled normalized RSS order: " + ", ".join(f"{f} {fmt(ranking['families'][f]['combined'])}" for f in ranking["order"]) + f". Pooled Steep X: {ranking['steep_x'] or 'none (no family exceeds the others in quadrature)'}.", "",
            group_table(ranking["groups"], families), "",
            "## Stage B: paired swap", "",
            json.dumps(check, indent=2) if check["evaluated"] else "Pending.", "",
            "## Stage C: tolerance recheck of the paired swap", "",
            json.dumps(stage_c, indent=2) if stage_c else "Pending.", "",
            "## Residuals", "", residual_table(vectors, names), ""]
    (FOLDER/"h01-l2-campaign-result.md").write_text("\n".join(text))
    (folder/"stage-b-prediction-check.json").write_text(json.dumps({"stage_b": check, "stage_c": stage_c}, indent=2))
    print(json.dumps({"stage_b_holds": check.get("holds"), "stage_c": stage_c}))


if __name__ == "__main__":
    main()
