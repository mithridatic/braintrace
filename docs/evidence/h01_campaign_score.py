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
    count_changes = {}
    for name in manifest["subsystems"]:
        count_changes[name] = {k: {"into_source": counts[f"a-into-{name}"][k]-counts["source"][k],
                                   "out_of_candidate": counts[f"a-out-{name}"][k]-counts["candidate"][k]}
                               for k in counts["source"]}
    magnitude = {n: sum(abs(v["into_source"])+abs(v["out_of_candidate"]) for v in c.values()) for n, c in count_changes.items()}
    count_limits = {k: decision["limits"]["limits"].get(k, 0.) for k in counts["source"]}
    count_movers = sorted((n for n in magnitude if any(abs(v["into_source"]) > count_limits[k] or abs(v["out_of_candidate"]) > count_limits[k]
                                                       for k, v in count_changes[n].items())), key=lambda n: -magnitude[n])
    largest = count_movers[:2] if count_movers else [n for n in pooled["order"] if pooled["subsystems"][n]["combined"] is not None][:2]
    report = {"groups": groups, "pooled": pooled, "event_counts": counts, "steep_x_by_group": steep,
              "count_changes": count_changes, "count_movers_by_magnitude": count_movers,
              "pairing_rule": "When any sub-system moves an event count beyond its decision limit, the two largest count movers form the Stage B pair; RSS rankings are incomplete where events vanish.",
              "residuals": {alias: residuals(v) for alias, v in vectors.items()},
              "two_largest": largest,
              "decision": ("steep X by group: "+json.dumps(steep) if steep else "no group reversed by a single swap")
              + "; Stage B pair: "+"+".join(largest)}
    (folder/"stage-a-ranking.json").write_text(json.dumps(report, indent=2))
    (folder/"stage-a-tree.mmd").write_text(tree(groups))
    return report


def cell_verdicts(vector):
    """Per-group verdict for one matrix cell: pass only when every row of the group passes."""
    out = {}
    for group, kinds in GROUPS.items():
        rows = [v for v in vector.values() if v["kind"] in kinds]
        if not rows:
            continue
        verdicts = {r["verdict"] for r in rows}
        out[group] = "pass" if verdicts == {"pass"} else ("unavailable" if "unavailable" in verdicts and "fail" not in verdicts else "fail")
    return out


def member_effects(cells, members, minima_key_filter, interval_key):
    """Average change of the recovery minima and of one interval when a member is added.

    Parameters
    ----------
    cells : dict
        Matrix code (e.g. ``"101"``) -> residual vector.
    members : list of str
        Member names in code order.
    minima_key_filter : callable
        Selects the minima voltage keys of a vector.
    interval_key : str
        The interval row used for the ratio, e.g. ``"sweep50:i2_ms"``.

    Returns
    -------
    dict
        Per member: mean minima change, interval change, their ratio, and the pairs used.
    """
    out = {}
    for index, member in enumerate(members):
        pairs, d_min, d_int = [], [], []
        for code, with_member in cells.items():
            if code[index] != "1":
                continue
            without = code[:index]+"0"+code[index+1:]
            if without not in cells:
                continue
            base = cells[without]
            keys = [k for k in with_member if minima_key_filter(k) and with_member[k]["residual"] is not None and base.get(k, {}).get("residual") is not None]
            if not keys or with_member.get(interval_key, {}).get("residual") is None or base.get(interval_key, {}).get("residual") is None:
                continue
            d_min.append(float(np.mean([abs(base[k]["residual"])-abs(with_member[k]["residual"]) for k in keys])))
            d_int.append(abs(with_member[interval_key]["residual"])-abs(base[interval_key]["residual"]))
            pairs.append((without, code))
        mean_min = float(np.mean(d_min)) if d_min else None
        mean_int = float(np.mean(d_int)) if d_int else None
        ratio = None if mean_min is None or mean_min <= 0 or not mean_int else mean_min/abs(mean_int)
        out[member] = {"minima_improvement_mv": mean_min, "interval_worsening_ms": mean_int, "ratio_mv_per_ms": ratio,
                       "pairs": pairs, "dose_invariant": None if len(d_min) < 2 else bool(np.std(np.array(d_min)/np.where(np.array(d_int) == 0, np.nan, np.array(d_int))) < .25*abs(ratio or 1.))}
    return out


def stage_matrix(manifest, folder, datums):
    members = list(manifest["members"])
    cells = {}
    for code, stem in manifest["existing_matrix_cells_traces"].items():
        cells[code] = {}
        for input_name, datum_key in manifest["input_datums"].items():
            trace = FOLDER/f"{stem[input_name]}.npz"
            for row in score_trace(trace, datums[datum_key]):
                cells[code][f"{input_name}:{row['key']}"] = dict(row, input=input_name)
    for candidate in manifest["candidates"]:
        if str(candidate["stage"]) == "A":
            code = candidate["name"].split("-")[0].lstrip("m")
            try:
                cells[code] = candidate_vector(folder, manifest, candidate["name"], datums)
            except FileNotFoundError:
                continue
    verdicts = {code: cell_verdicts(v) for code, v in cells.items()}
    counts = {code: {k: v["model"] for k, v in vec.items() if v["kind"] == "count"} for code, vec in cells.items()}
    passing = [code for code, groups in verdicts.items() if all(x == "pass" for x in groups.values())]
    interval_key = manifest.get("dose_interval_key", "sweep50:i2_ms")
    effects = member_effects(cells, members, lambda k: k.endswith("_voltage_mv") and ":m" in k, interval_key)
    ranked = sorted((m for m in effects if effects[m]["ratio_mv_per_ms"] is not None), key=lambda m: -effects[m]["ratio_mv_per_ms"])
    summary = {code: {"count": counts[code], "groups": verdicts[code],
                      "i2": cells[code].get(interval_key, {}).get("residual"),
                      "minima": [round(v["residual"], 3) for k, v in cells[code].items() if k.endswith("_voltage_mv") and ":m" in k and v["residual"] is not None]}
               for code in sorted(cells)}
    decision = ("PASS: "+", ".join(passing)) if passing else (
        "no cell passes every group; dose member "+ranked[0]+" (largest minima change per interval change)" if ranked else "no member effect could be computed")
    report = {"cells": summary, "member_effects": effects, "dose_member_order": ranked, "passing_cells": passing,
              "residuals": {code: residuals(v) for code, v in cells.items()}, "decision": decision}
    (folder/"stage-a-decision.json").write_text(json.dumps(report, indent=2))
    return {"decision": decision, "event_counts": counts, "verdicts": verdicts}


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
    parser.add_argument("--stage", choices=("0", "A", "matrix"), required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    folder = FOLDER/manifest["output_dir"]
    datums = load_datums()
    runner = {"0": stage0, "A": stage_a, "matrix": stage_matrix}[args.stage]
    report = runner(manifest, folder, datums)
    print(json.dumps({"decision": report["decision"], "event_counts": report["event_counts"]}, indent=1))


if __name__ == "__main__":
    main()
