"""Score campaign runs against the human datums and rank parameter families.

Every residual is model minus human on the stated convention. Normalized
values guide the family ranking only; raw residuals are retained. See
``docs/specs/2026-09-05-h01-l2-family-campaign.md``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_l2_acceptance_table import numerical_limits
from docs.evidence.h01_pv_human_datums import spike_datums

FOLDER = Path(__file__).parent
FIELDS = {"rise": "rise_crossing_ms", "fall": "fall_crossing_ms", "width": "time_above_threshold_ms",
          "peak_t": "peak_sample_time_ms", "peak_v": "peak_sample_voltage_mv"}
PULSE_MS = (1020., 2020.)


def active_residuals(time, voltage, human_events, recovery_datums):
    """Flat residual vector for the spiking input; missing events give NaN."""
    mask = (time >= PULSE_MS[0]) & (time < PULSE_MS[1])
    events = spike_datums(time[mask], voltage[mask])
    out = {"event_count_model": len(events), "event_count_human": len(human_events)}
    rises = [e["rise_crossing_ms"] for e in events]
    for i, human in enumerate(human_events, start=1):
        model = events[i-1] if i <= len(events) else None
        for key, field in FIELDS.items():
            out[f"e{i}_{key}"] = model[field]-human[field] if model else np.nan
        out[f"e{i}_rise_to_peak"] = ((model["peak_sample_time_ms"]-model["rise_crossing_ms"])
                                     - (human["peak_sample_time_ms"]-human["rise_crossing_ms"])) if model else np.nan
        out[f"e{i}_peak_to_fall"] = ((model["fall_crossing_ms"]-model["peak_sample_time_ms"])
                                     - (human["fall_crossing_ms"]-human["peak_sample_time_ms"])) if model else np.nan
    human_rises = [h["rise_crossing_ms"] for h in human_events]
    for i in range(1, len(human_events)):
        out[f"i{i}"] = (rises[i]-rises[i-1])-(human_rises[i]-human_rises[i-1]) if i < len(rises) else np.nan
    for datum in recovery_datums:
        i = datum["event_index"]
        if i < len(events):
            start = events[i]["fall_crossing_ms"]
            stop = events[i+1]["rise_crossing_ms"] if i+1 < len(events) else PULSE_MS[1]
            window = (time > start) & (time < stop)
            at = int(np.argmin(voltage[window]))
            out[f"m{i+1}_v"] = float(voltage[window][at])-datum["minimum_sample_voltage_mv"]
            out[f"m{i+1}_delay"] = (float(time[window][at])-events[i]["peak_sample_time_ms"])-datum["delay_from_peak_ms"]
        else:
            out[f"m{i+1}_v"] = out[f"m{i+1}_delay"] = np.nan
    return out


def subthreshold_residuals(time, voltage, samples):
    """Voltage residual at each requested sample time under the subthreshold input."""
    time, voltage = np.asarray(time), np.asarray(voltage)
    requested = np.asarray([s["requested_time_ms"] for s in samples], dtype=float)
    if (time.ndim != 1 or voltage.shape != time.shape or time.size < 2
            or not np.isfinite(time).all() or not np.isfinite(voltage).all()
            or not np.all(np.diff(time) > 0) or not np.isfinite(requested).all()):
        raise ValueError("Trace must be finite, ordered, and cover the requested samples.")
    if np.any(requested < time[0]) or np.any(requested > time[-1]):
        raise ValueError("Trace must cover every requested sample; extrapolation is forbidden.")
    return {f"sub_{s['requested_time_ms']:.0f}_v": float(np.interp(s["requested_time_ms"], time, voltage))-s["voltage_mv"]
            for s in samples}


def score_candidate(folder, name, datums):
    """Residual vector from a candidate's two sweep traces."""
    out = {}
    for sweep, scorer in (("50", lambda t, v: active_residuals(t, v, datums["events"], datums["recovery"])),
                          ("43", lambda t, v: subthreshold_residuals(t, v, datums["samples"]))):
        with np.load(folder/f"{name}-sweep{sweep}.npz") as data:
            out.update(scorer(np.asarray(data["time_ms"]), np.asarray(data["voltage_mv"])))
    return out


def limit_for(key, limits):
    """Measured numerical decision limit per observation kind; proxies are named."""
    fields = limits["fields"]
    if key.endswith("_peak_v"):
        return fields["peak_sample_voltage_mv"], "measured"
    if key.endswith("_width"):
        return fields["time_above_threshold_ms"], "measured"
    if key.endswith(("_rise_to_peak", "_peak_to_fall")):
        return limits["phase_ms"], "measured"
    if key.endswith("_v"):
        return limits["minimum_voltage_mv"], "measured" if key.startswith("m") else "proxy"
    return fields["rise_crossing_ms"], "measured" if not key.endswith("_delay") else "proxy"


def select_observations(source, candidate, limits, factor=5.):
    """Keys whose source-to-candidate contrast exceeds ``factor`` numerical limits."""
    keys = []
    for key in source:
        if key.startswith("event_count") or not np.isfinite([source[key], candidate.get(key, np.nan)]).all():
            continue
        if abs(candidate[key]-source[key]) > factor*limit_for(key, limits)[0]:
            keys.append(key)
    return keys


def normalized_rss(base, swapped, keys, limits):
    """Root sum of squares of residual changes divided by their numerical limits."""
    missing = [k for k in keys if not np.isfinite([base.get(k, np.nan), swapped.get(k, np.nan)]).all()]
    if missing or not keys:
        raise ValueError(f"Incomplete RSS observations: {missing or ['no selected observations']}")
    terms = [((swapped[k]-base[k])/limit_for(k, limits)[0])**2 for k in keys]
    return float(np.sqrt(sum(terms)))


def rank_families(vectors, families, keys, limits):
    """Per-family RSS in both swap directions and the sparsity decision."""
    rows = {}
    for family in families:
        try:
            rows[family] = {"into_source": normalized_rss(vectors["source"], vectors[f"into-{family}"], keys, limits),
                            "out_of_candidate": normalized_rss(vectors["candidate"], vectors[f"out-{family}"], keys, limits)}
            rows[family]["combined"] = float(np.hypot(rows[family]["into_source"], rows[family]["out_of_candidate"]))
        except ValueError as error:
            rows[family] = {"into_source": None, "out_of_candidate": None, "combined": None,
                            "incomplete_reason": str(error)}
        rows[family]["event_counts"] = {name: vectors[name].get("event_count_model")
                                       for name in ("source", "candidate", f"into-{family}", f"out-{family}")}
    ordered = sorted(rows, key=lambda f: -rows[f]["combined"] if rows[f]["combined"] is not None else float("inf"))
    complete = bool(ordered) and all(rows[f]["combined"] is not None for f in ordered)
    steep = complete and rows[ordered[0]]["combined"]**2 > sum(rows[f]["combined"]**2 for f in ordered[1:])
    return {"families": rows, "order": ordered, "steep_x": ordered[0] if steep else None, "keys": keys}


def preservation(coarse, fine, keys, limits, factor=5.):
    """Coarse settings preserve decisions when signs, orderings, and changes hold on every key."""
    failures = []
    names = list(coarse)
    for key in keys:
        c = {n: coarse[n].get(key, np.nan) for n in names}
        f = {n: fine[n].get(key, np.nan) for n in names}
        if not np.isfinite(list(c.values())+list(f.values())).all():
            failures.append((key, "missing"))
            continue
        if any(np.sign(c[n]) != np.sign(f[n]) for n in names):
            failures.append((key, "sign"))
        if sorted(names, key=lambda n: c[n]) != sorted(names, key=lambda n: f[n]):
            failures.append((key, "ordering"))
        contrasts = [abs(f[a]-f[b]) for a in names for b in names if a < b]
        if max(abs(c[n]-f[n]) for n in names) > min(contrasts)/factor:
            failures.append((key, "change exceeds one fifth of the smallest contrast"))
    return {"preserved": not failures, "failures": failures, "keys": keys}


GROUPS = {"event_count": ("event_count_model",), "intervals": ("i1", "i2", "i3", "i4"),
          "minima": tuple(f"m{i}_v" for i in range(1, 6)), "phases": tuple(f"e{i}_{p}" for i in range(1, 6) for p in ("rise_to_peak", "peak_to_fall", "width")),
          "peaks_and_onsets": tuple(f"e{i}_{p}" for i in range(1, 6) for p in ("rise", "fall", "peak_t", "peak_v")),
          "subthreshold": tuple(f"sub_{t}_v" for t in (1019, 1021, 1025, 1040, 1120, 1520, 2019, 2021, 2040, 2120))}


def group_ranking(vectors, families, keys, limits):
    """Family RSS per observation group, so sparsity is judged per behavior, not pooled.

    The event count group reports the signed count change per swap instead of an RSS.
    """
    report = {}
    for group, members in GROUPS.items():
        selected = [k for k in keys if k in members]
        if group == "event_count":
            count = lambda name: vectors.get(name, {}).get("event_count_model")
            change = lambda a, b: None if count(a) is None or count(b) is None else count(a)-count(b)
            rows = {f: {"into_source": change(f"into-{f}", "source"), "out_of_candidate": change(f"out-{f}", "candidate")} for f in families}
            report[group] = {"families": rows, "source": count("source"), "candidate": count("candidate")}
            continue
        if not selected:
            report[group] = {"families": {}, "keys": [], "order": [], "steep_x": None}
            continue
        ranked = rank_families(vectors, families, selected, limits)
        report[group] = {"families": ranked.get("families", {}), "keys": selected,
                         "order": ranked.get("order", []), "steep_x": ranked.get("steep_x")}
    return report


def tree_markdown(ranking):
    """Mermaid family tree for the causal page."""
    lines = ["flowchart TD", "    Y4[Layer-2 candidate: wrong timing and recovery] --> FAM[Parameter family]"]
    for family in ranking["order"]:
        row = ranking["families"][family]
        if row["combined"] is None:
            lines.append(f"    FAM --> {family}[{family}: incomplete observations]")
        else:
            mark = "Steep X" if family == ranking["steep_x"] else "not dominant"
            lines.append(f"    FAM --> {family}[{family}: RSS {row['combined']:.1f}; {mark}]")
    return "\n".join(lines)


def load_datums():
    requirements = json.loads((FOLDER/"h01-human-event-requirements.json").read_text())
    recovery = json.loads((FOLDER/"h01-human-recovery-datums.json").read_text())
    samples = json.loads((FOLDER/"h01-human-subthreshold-samples.json").read_text())
    return {"events": next(i for i in requirements["inputs"] if i["role"] == "E")["events"],
            "recovery": next(r for r in recovery["records"] if r["reference"] == "E")["recovery_datums"],
            "samples": next(r for r in samples["records"] if r["role"] == "E")["samples"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("preservation", "rank"), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    folder = FOLDER/manifest["output_dir"]
    datums = load_datums()
    limits = numerical_limits(*(json.loads((FOLDER/n).read_text()) for n in (
        "h01-l2-density130-tolerance-result.json", "h01-l2-density130-spatial-result.json",
        "h01-l2-density130-minima-numerical-review.json")))
    stage0 = manifest["stage0"]
    mesh = manifest.get("analysis_mesh", "coarse")
    if mesh not in ("coarse", "fine"):
        raise ValueError("analysis_mesh must be coarse or fine.")
    fine = {k: score_candidate(folder, v, datums) for k, v in stage0["fine"].items()}
    keys = select_observations(fine["source"], fine["candidate"], limits)
    if args.mode == "preservation":
        coarse = {k: score_candidate(folder, v, datums) for k, v in stage0["coarse"].items()}
        report = preservation(coarse, fine, keys, limits)
        report["limits"] = limits
        (folder/"stage0-preservation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"preserved": report["preserved"], "failures": report["failures"][:10], "keys": len(keys)}))
        return
    vectors = dict(fine) if mesh == "fine" else {
        k: score_candidate(folder, v, datums) for k, v in stage0["coarse"].items()}
    for candidate in manifest["candidates"]:
        if str(candidate["stage"]) == "A":
            vectors[candidate["name"].removeprefix("a-")] = score_candidate(folder, candidate["name"], datums)
    ranking = rank_families(vectors, manifest["families"], keys, limits)
    ranking["analysis_mesh"] = mesh
    ranking["groups"] = group_ranking(vectors, manifest["families"], keys, limits)
    ranking["vectors"] = {k: {kk: (None if not np.isfinite(vv) else vv) for kk, vv in v.items()} for k, v in vectors.items()}
    (folder/"stage-a-ranking.json").write_text(json.dumps(ranking, indent=2))
    (folder/"stage-a-tree.mmd").write_text(tree_markdown(ranking))
    print(json.dumps({"order": ranking["order"], "steep_x": ranking["steep_x"], "keys": len(keys)}))


if __name__ == "__main__":
    main()
