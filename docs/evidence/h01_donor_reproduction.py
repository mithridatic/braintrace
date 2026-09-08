"""Score a donor's NEURON reproduction against its registered prediction (SP6d).

For each registered input the model's -20 mV crossing count and first-crossing latency
after step onset are read from the ``source`` trace and from its dt-halved partner. The
pair difference times the two-repeat Dixon factor (2.95) is the decision limit of the
latency row; a pair that disagrees on the count leaves the time level open (no verdict).
Human values are re-read from the exported recording with the same detector and printed
beside the registered targets. Wall clocks come from the runner's ``campaign-log.json``;
container stderr files are listed, never summarised away.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_isolation_report import human_trace, model_trace
from h01_pv_human_datums import spike_datums

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]
DLF_PAIR = 2.95
DETECTION_MV = -20.

DONORS = {
    "sst": {
        "donor_key": "l3-sst-interneuron-hl5mn1",
        "output_dir": "h01-sst-reproduction",
        "manifest": "h01-sst-reproduction-manifest.json",
        "decision": "h01-donors/stage-hl5mn1-decision.json",
        "specification": "docs/specs/2026-09-07-h01-donor-hl5mn1-import.md",
        "inputs": {
            "100 pA": {"input": "100", "human": ".cache/human-sst-l3/active-0.npz", "sweep": 44, "pulse_ms": (270., 1270.),
                       "registered_count": 14, "repeat_counts": (14, 14, 13, 12), "registered_first_spike_ms": 24.16},
            "150 pA": {"input": "150", "human": ".cache/human-sst-l3/active-1.npz", "sweep": 35, "pulse_ms": (270., 1270.),
                       "registered_count": 34, "repeat_counts": None, "registered_first_spike_ms": 13.0},
        },
    },
    "l4": {
        "donor_key": "l4-pyramidal-allen-527952884",
        "output_dir": "h01-l4-reproduction",
        "manifest": "h01-l4-reproduction-manifest.json",
        "decision": "h01-donors/stage-allen-l4-decision.json",
        "specification": "docs/specs/2026-09-07-h01-donor-allen-l4-import.md",
        "inputs": {
            "100 pA": {"input": "69", "human": ".cache/human-pyramidal-l4/neuron-reference/sweep-69.npz", "sweep": 69,
                       "pulse_ms": (1020., 3020.), "registered_count": 20, "repeat_counts": (20, 17, 19, 20),
                       "registered_first_spike_ms": 32.18},
            "90 pA": {"input": "39", "human": ".cache/human-pyramidal-l4/neuron-reference/sweep-39.npz", "sweep": 39,
                      "pulse_ms": (1020., 2020.), "registered_count": 12, "repeat_counts": None,
                      "registered_first_spike_ms": 33.88},
        },
    },
}


def crossings(time, voltage, pulse_ms):
    """Count and first -20 mV rise crossing (ms after onset) inside the pulse; finiteness flag."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    mask = (time >= pulse_ms[0]) & (time < pulse_ms[1])
    events = spike_datums(time[mask], voltage[mask], DETECTION_MV)
    first = events[0]["rise_crossing_ms"]-pulse_ms[0] if events else None
    return {"count": len(events), "first_spike_ms": first, "finite": bool(np.isfinite(voltage).all()),
            "end_ms": float(time[-1])}


def count_verdict(model, registered, repeat_counts):
    """held (exact), within_one (prediction missed, rejection not met), rejected (off by more than 1)."""
    if model is None:
        return {"exact": "untested", "repeat_band": "untested"}
    difference = abs(model-registered)
    exact = "held" if difference == 0 else ("missed_within_one" if difference <= 1 else "rejected")
    band = None
    if repeat_counts:
        band = "held" if min(repeat_counts) <= model <= max(repeat_counts) else "missed"
    return {"exact": exact, "repeat_band": band}


def pair_limit(a, b):
    """Two-repeat Dixon decision limit of a paired numerical measurement."""
    if a is None or b is None:
        return None
    return abs(a-b)*DLF_PAIR


def latency_verdict(model, registered, limit):
    if model is None or limit is None:
        return "untested"
    return "held" if abs(model-registered) <= limit else "missed"


def score_input(label, spec, folder, root):
    """One input's rows: human re-read, source, dt-half, limits and verdicts."""
    human = crossings(*human_trace(root/spec["human"]), spec["pulse_ms"])
    traces = {}
    for name in ("source", "source-dt-half"):
        path = folder/f"{name}-{spec['input']}.npz"
        traces[name] = crossings(*model_trace(folder, f"{name}-{spec['input']}"), spec["pulse_ms"]) if path.exists() else None
    src, half = traces["source"], traces["source-dt-half"]
    counts = (src or {}).get("count"), (half or {}).get("count")
    pair_agrees = None if None in counts else counts[0] == counts[1]
    limit = pair_limit((src or {}).get("first_spike_ms"), (half or {}).get("first_spike_ms"))
    verdicts = {"count": count_verdict(counts[0], spec["registered_count"], spec["repeat_counts"]),
                "dt_pair_count": None if pair_agrees is None else ("agrees" if pair_agrees else "time level open"),
                "first_spike": latency_verdict((src or {}).get("first_spike_ms"), spec["registered_first_spike_ms"], limit)}
    if pair_agrees is False:
        verdicts["first_spike"] = "time level open"
    return {"label": label, "input": spec["input"], "sweep": spec["sweep"], "pulse_ms": list(spec["pulse_ms"]),
            "registered": {"count": spec["registered_count"], "repeat_counts": spec["repeat_counts"],
                           "first_spike_ms": spec["registered_first_spike_ms"]},
            "human_reread": human, "source": src, "source_dt_half": half,
            "first_spike_pair_limit_ms": limit, "verdicts": verdicts}


def wall_clocks(folder):
    """Runner rows (seconds, return code, aborted) and any persisted stderr files."""
    log = folder/"campaign-log.json"
    rows = json.loads(log.read_text()) if log.exists() else []
    clocks = [{"name": r["name"], "input": r["input"], "seconds": r["seconds"], "returncode": r["returncode"],
               "aborted": r["aborted"]} for r in rows]
    stderr = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in sorted(folder.glob("*.stderr.txt"))}
    return {"runs": clocks, "total_seconds": sum(r["seconds"] for r in clocks) if clocks else None,
            "stderr_files": {k: (v if v.strip() else "") for k, v in stderr.items()}}


def overall(inputs):
    """Reproduction verdict: reproduced, prediction_missed, rejected, time_level_open, or untested."""
    states = [i["verdicts"] for i in inputs]
    if any(v["count"]["exact"] == "untested" for v in states):
        return "untested"
    if any(v["dt_pair_count"] == "time level open" for v in states):
        return "time_level_open"
    if any(v["count"]["exact"] == "rejected" for v in states):
        return "rejected"
    if all(v["count"]["exact"] == "held" and v["first_spike"] == "held" for v in states):
        return "reproduced"
    return "prediction_missed"


def score_donor(name, root=ROOT, evidence=EVIDENCE):
    spec = DONORS[name]
    folder = evidence/spec["output_dir"]
    inputs = [score_input(label, s, folder, root) for label, s in spec["inputs"].items()]
    return {"donor_key": spec["donor_key"], "specification": spec["specification"], "manifest": spec["manifest"],
            "decision_json": spec["decision"], "detector": f"{DETECTION_MV:g} mV rise crossing inside the pulse",
            "pair_factor": DLF_PAIR, "inputs": inputs, "wall_clocks": wall_clocks(folder),
            "reproduction_verdict": overall(inputs)}


def _f(v):
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def render(report):
    lines = [f"# {report['donor_key']}: NEURON reproduction of the published fit", "",
             f"Spec `{report['specification']}`; manifest `{report['manifest']}`; decision `{report['decision_json']}`.",
             f"Detector: {report['detector']}. Latency limit = |dt pair difference| x {report['pair_factor']}.",
             "", f"**Reproduction verdict: {report['reproduction_verdict']}**", "",
             "## Counts and first spikes", "",
             "| Input | Sweep | Registered count | Repeat band | Human re-read | Model dt | Model dt/2 | dt pair | Count exact | Repeat row |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in report["inputs"]:
        r, v = i["registered"], i["verdicts"]
        band = f"{min(r['repeat_counts'])}-{max(r['repeat_counts'])}" if r["repeat_counts"] else "-"
        lines.append(f"| {i['label']} | {i['sweep']} | {r['count']} | {band} | {i['human_reread']['count']} | "
                     f"{_f((i['source'] or {}).get('count'))} | {_f((i['source_dt_half'] or {}).get('count'))} | "
                     f"{v['dt_pair_count']} | {v['count']['exact']} | {v['count']['repeat_band'] or '-'} |")
    lines += ["", "| Input | Registered first spike (ms) | Human re-read | Model dt | Model dt/2 | Pair limit | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for i in report["inputs"]:
        lines.append(f"| {i['label']} | {i['registered']['first_spike_ms']} | {_f(i['human_reread']['first_spike_ms'])} | "
                     f"{_f((i['source'] or {}).get('first_spike_ms'))} | {_f((i['source_dt_half'] or {}).get('first_spike_ms'))} | "
                     f"{_f(i['first_spike_pair_limit_ms'])} | {i['verdicts']['first_spike']} |")
    w = report["wall_clocks"]
    lines += ["", "## Wall clocks (runner-measured)", "", "| Candidate | Input | Seconds | rc | Aborted |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['name']} | {r['input']} | {r['seconds']:.1f} | {r['returncode']} | {r['aborted']} |" for r in w["runs"]]
    lines += ["", f"Total {_f(w['total_seconds'])} s. Stderr files: "
              + (", ".join(f"`{k}` ({'empty' if not v else str(len(v))+' chars'})" for k, v in w["stderr_files"].items()) or "none"), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", choices=tuple(DONORS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = score_donor(args.donor, root=args.root)
    stem = EVIDENCE/DONORS[args.donor]["output_dir"]
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(stem.with_suffix(".md"), report["reproduction_verdict"])


if __name__ == "__main__":
    main()
