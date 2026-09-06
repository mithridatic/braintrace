"""Render a campaign's stage decisions, counts, and search tree into one result page.

The page issues exactly one of PASS or FAIL from the recorded decisions; it adds no
new judgement. Raw residual files stay the evidence; this is the index.
"""

import argparse
import json
from pathlib import Path

FOLDER = Path(__file__).parent


def stage_files(folder):
    """Existing stage decision files in stage order."""
    order = ("stage0-decision.json", "stage-a-decision.json", "stage-a-ranking.json", "stage-b-decision.json",
             "stage-c-decision.json", "stage-d-decision.json")
    return [(name, json.loads((folder/name).read_text())) for name in order if (folder/name).exists()]


def wall_time(folder):
    """Total container seconds and the longest run from the runner log."""
    path = folder/"campaign-log.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    return {"runs": len(rows), "total_seconds": round(sum(r["seconds"] for r in rows), 1),
            "longest_seconds": round(max(r["seconds"] for r in rows), 1) if rows else 0,
            "aborted": [f"{r['name']} {r['input']}" for r in rows if r.get("aborted")]}


def verdict(stages):
    """PASS when any stage records a passing candidate, else FAIL-or-open."""
    for _, record in stages:
        if record.get("passing_cells"):
            return "PASS: " + ", ".join(record["passing_cells"])
        if str(record.get("decision", "")).startswith("PASS"):
            return record["decision"]
    return "No passing candidate recorded; the campaign is FAIL at its cap or still open"


def render(manifest, stages, timing, tree_text):
    lines = [f"# {manifest['output_dir']} result", "", manifest["purpose"], "",
             f"Cap {manifest['cap']} evaluations; abort {manifest['abort_seconds']} s per run; "
             f"inputs {', '.join(manifest['inputs'])}.", "", f"**Verdict.** {verdict(stages)}", ""]
    if timing:
        lines += [f"Container runs: {timing['runs']}, total {timing['total_seconds']} s, longest {timing['longest_seconds']} s"
                  + (f", aborted: {', '.join(timing['aborted'])}" if timing["aborted"] else "") + ".", ""]
    for name, record in stages:
        lines += [f"## {name}", "", f"Decision: {record.get('decision')}", ""]
        if record.get("event_counts"):
            lines += ["| Record | " + " | ".join(next(iter(record["event_counts"].values())).keys()) + " |",
                      "| --- | " + " | ".join("---:" for _ in next(iter(record["event_counts"].values()))) + " |"]
            lines += [f"| {alias} | " + " | ".join(str(v) for v in counts.values()) + " |" for alias, counts in record["event_counts"].items()]
            lines.append("")
        if record.get("notes"):
            lines += [f"- **{k}.** {v if isinstance(v, str) else json.dumps(v)}" for k, v in record["notes"].items()] + [""]
        if record.get("steep_x_by_group") is not None:
            lines += [f"Steep X by group: {json.dumps(record['steep_x_by_group'])}", ""]
        if record.get("cells"):
            lines += ["| Cell | Count | Groups | i2 (ms) | Minima (mV) |", "| --- | --- | --- | ---: | --- |"]
            lines += [f"| {code} | {json.dumps(c['count'])} | {json.dumps(c['groups'])} | {c['i2'] if c['i2'] is None else round(c['i2'], 2)} | {c['minima']} |"
                      for code, c in record["cells"].items()]
            lines.append("")
        if record.get("member_effects"):
            lines += ["| Member | Minima change (mV) | Interval change (ms) | Ratio |", "| --- | ---: | ---: | ---: |"]
            lines += [f"| {m} | {e['minima_change_mv']} | {e['interval_change_ms']} | {e['ratio_mv_per_ms']} |" for m, e in record["member_effects"].items()]
            lines.append("")
    if tree_text:
        lines += ["## Search tree", "", "```mermaid", tree_text, "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    folder = FOLDER/manifest["output_dir"]
    tree_path = folder/"stage-a-tree.mmd"
    text = render(manifest, stage_files(folder), wall_time(folder), tree_path.read_text() if tree_path.exists() else "")
    (FOLDER/f"{manifest['output_dir']}-result.md").write_text(text)
    print(verdict(stage_files(folder)))


if __name__ == "__main__":
    main()
