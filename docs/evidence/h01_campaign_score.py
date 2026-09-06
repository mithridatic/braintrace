"""Score campaign stages against the contract and write the stage decision files.

Stage 0 fixes per-row Dixon decision limits from the numerical repeats of one
physical model and selects the rows whose source-to-candidate contrast exceeds five
limits. Stage A ranks sub-systems per observation group by Dixon-normalised RSS in
both swap directions; a Steep X exists when one sub-system exceeds the others
combined in quadrature. Every raw residual is retained; normalised values only rank.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_contract_score import dixon_limits, load_datums, score_trace

FOLDER = Path(__file__).parent
GROUPS = {"count": ("count",), "onsets": ("rise_crossing_ms", "fall_crossing_ms", "peak_sample_time_ms"),
          "peaks": ("peak_sample_voltage_mv",), "phases": ("time_above_threshold_ms", "rise_to_peak_ms", "peak_to_fall_ms"),
          "intervals": ("interval_ms",), "minima": ("minimum_voltage_mv", "minimum_delay_ms"),
          "subthreshold": ("subthreshold_voltage_mv",)}


def candidate_vector(folder, manifest, name, datums):
    """Residual rows for one candidate across every manifest input, keyed ``input:key``."""
    rows = {}
    for input_name, datum_key in manifest["input_datums"].items():
        trace = folder/f"{name}-{input_name}.npz"
        if not trace.exists():
            raise FileNotFoundError(trace)
        for row in score_trace(trace, datums[datum_key]):
            rows[f"{input_name}:{row['key']}"] = dict(row, input=input_name)
    return rows


def residuals(vector):
    return {k: v["residual"] for k, v in vector.items()}


def select_rows(source, candidate, limits, factor=5.):
    """Keys whose source-to-candidate contrast exceeds ``factor`` decision limits and that fail in either."""
    keys = []
    for key in source:
        s, c = source[key]["residual"], candidate.get(key, {}).get("residual")
        if s is None or c is None or key not in limits:
            continue
        if abs(c-s) > factor*limits[key] and (source[key]["verdict"] == "fail" or candidate[key]["verdict"] == "fail"):
            keys.append(key)
    return keys


def normalized_rss(base, swapped, keys, limits):
    """Root sum of squares of residual changes divided by their decision limits (zero limit -> skipped)."""
    terms = []
    for key in keys:
        a, b = base.get(key, {}).get("residual"), swapped.get(key, {}).get("residual")
        if a is None or b is None:
            raise ValueError(f"Incomplete observation {key}")
        if limits[key] > 0:
            terms.append(((b-a)/limits[key])**2)
    return float(np.sqrt(sum(terms))) if terms else 0.


def rank(vectors, subsystems, keys, limits):
    """Per-subsystem RSS in both directions and the sparsity decision for one key set."""
    rows = {}
    for name in subsystems:
        try:
            into = normalized_rss(vectors["source"], vectors[f"a-into-{name}"], keys, limits)
            out = normalized_rss(vectors["candidate"], vectors[f"a-out-{name}"], keys, limits)
            rows[name] = {"into_source": into, "out_of_candidate": out, "combined": float(np.hypot(into, out))}
        except (ValueError, KeyError) as error:
            rows[name] = {"into_source": None, "out_of_candidate": None, "combined": None, "incomplete": str(error)}
    order = sorted(rows, key=lambda n: -rows[n]["combined"] if rows[n]["combined"] is not None else float("inf"))
    complete = bool(order) and all(rows[n]["combined"] is not None for n in order)
    steep = complete and rows[order[0]]["combined"]**2 > sum(rows[n]["combined"]**2 for n in order[1:])
    return {"subsystems": rows, "order": order, "steep_x": order[0] if steep else None, "keys": keys}


def group_keys(keys, vector, group):
    kinds = GROUPS[group]
    return [k for k in keys if vector[k]["kind"] in kinds]


def control_names(manifest):
    """Stage 0 candidate names aliased by identity: source, candidate, and the numerical repeats."""
    names = {}
    for candidate in manifest["candidates"]:
        if str(candidate["stage"]) != "0":
            continue
        common = candidate.get("settings", {}) == manifest["common_settings"]
        if common and candidate["flags"] == manifest["source_flags"]:
            names["source"] = candidate["name"]
        elif common and candidate["flags"] == manifest["candidate_flags"]:
            names["candidate"] = candidate["name"]
        else:
            names[candidate["name"]] = candidate["name"]
    if "source" not in names or "candidate" not in names:
        raise ValueError("Stage 0 needs a source and a candidate at the common settings.")
    return names


def stage0(manifest, folder, datums):
    names = control_names(manifest)
    vectors, missing = {}, {}
    for alias, name in names.items():
        try:
            vectors[alias] = candidate_vector(folder, manifest, name, datums)
        except FileNotFoundError as error:
            if alias in ("source", "candidate"):
                raise
            missing[alias] = f"trace absent (aborted or not run): {Path(str(error)).name}"
    repeats = [vectors[alias] for alias in vectors if alias not in ("source",)]
    limits = dixon_limits([residuals(v) for v in repeats])
    keys = select_rows(vectors["source"], vectors["candidate"], limits["limits"])
    per_group = {g: group_keys(keys, vectors["source"], g) for g in GROUPS}
    counts = {alias: {k: v["model"] for k, v in vec.items() if v["kind"] == "count"} for alias, vec in vectors.items()}
    excluded_counts = [k for k, v in vectors["source"].items() if v["kind"] == "count" and k not in keys]
    decision = ("rank at the manifest common settings" if not excluded_counts
                else "count rows excluded at these settings: " + ", ".join(excluded_counts) + "; ranking uses the remaining rows")
    report = {"repeat_names": [alias for alias in vectors if alias != "source"], "missing_repeats": missing,
              "limits": limits, "selected_keys": keys,
              "selected_by_group": per_group, "event_counts": counts,
              "residuals": {alias: residuals(v) for alias, v in vectors.items()}, "decision": decision}
    (folder/"stage0-decision.json").write_text(json.dumps(report, indent=2))
    return report


def stage_a(manifest, folder, datums):
    decision = json.loads((folder/"stage0-decision.json").read_text())
    limits, keys = decision["limits"]["limits"], decision["selected_keys"]
    controls = control_names(manifest)
    names = {"source": controls["source"], "candidate": controls["candidate"]}
    names.update({c["name"]: c["name"] for c in manifest["candidates"] if str(c["stage"]) == "A"})
    vectors = {alias: candidate_vector(folder, manifest, name, datums) for alias, name in names.items()}
    groups = {}
    for group in GROUPS:
        selected = group_keys(keys, vectors["source"], group)
        groups[group] = rank(vectors, manifest["subsystems"], selected, limits) if selected else {"keys": [], "order": [], "steep_x": None}
    pooled = rank(vectors, manifest["subsystems"], keys, limits)
    counts = {alias: {k: v["model"] for k, v in vec.items() if v["kind"] == "count"} for alias, vec in vectors.items()}
    steep = {g: r["steep_x"] for g, r in groups.items() if r.get("steep_x")}
    largest = [n for n in pooled["order"] if pooled["subsystems"][n]["combined"] is not None][:2]
    report = {"groups": groups, "pooled": pooled, "event_counts": counts, "steep_x_by_group": steep,
              "residuals": {alias: residuals(v) for alias, v in vectors.items()},
              "two_largest": largest,
              "decision": ("steep X by group: "+json.dumps(steep) if steep else "no group reversed by a single swap")
              + "; Stage B pair: "+"+".join(largest)}
    (folder/"stage-a-ranking.json").write_text(json.dumps(report, indent=2))
    (folder/"stage-a-tree.mmd").write_text(tree(groups))
    return report


def tree(groups):
    lines = ["flowchart TD", "    Y[Candidate differs from human] --> G[Observation group]"]
    for group, ranking in groups.items():
        if not ranking.get("order"):
            continue
        node = group.replace(" ", "_")
        lines.append(f"    G --> {node}[{group}: {len(ranking['keys'])} rows]")
        for name in ranking["order"]:
            row = ranking["subsystems"][name]
            mark = "Steep X" if name == ranking["steep_x"] else "not dominant"
            value = "incomplete" if row["combined"] is None else f"RSS {row['combined']:.1f}"
            lines.append(f"    {node} --> {node}_{name}[{name}: {value}; {mark}]")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("0", "A"), required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    folder = FOLDER/manifest["output_dir"]
    datums = load_datums()
    report = stage0(manifest, folder, datums) if args.stage == "0" else stage_a(manifest, folder, datums)
    print(json.dumps({"decision": report["decision"], "event_counts": report["event_counts"]}, indent=1))


if __name__ == "__main__":
    main()
