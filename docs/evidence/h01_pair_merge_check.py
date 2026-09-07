"""Merge check on the cell pair carrying the most candidate contacts in the C3 edge list.

Offline: reads the resolved edge list (and optional per-cell checkpoints) and reports
whether the two cells share a C3 label and whether any candidate's endpoints fall on one
proofread label. Writes ``h01-pair-merge-check.json`` and ``.md`` with the verdict
``suspected merge: yes / no / undetermined`` and the evidence per row.

Verdict rule (spec: docs/specs/2026-09-07-h01-connectivity-completion.md, SP7c):
``yes`` when a C3 label is shared by the two cells or a row reads one nonzero proofread
label at both endpoints; ``no`` when no label is shared and at least one row reads the
pre cell at the pre voxel and the post cell at the post voxel; ``undetermined`` otherwise.
"""

import argparse
import collections
import json
from pathlib import Path


def largest_pair(edges):
    """Unordered cell pair with the most candidate contacts, and its count."""
    counts = collections.Counter(frozenset((e["pre_cell"], e["post_cell"])) for e in edges)
    pair, count = counts.most_common(1)[0]
    return tuple(sorted(pair)), count


def classify_label(label, cell_a, cell_b):
    """Which cell a proofread label read at an endpoint belongs to."""
    if label == "0":
        return "background"
    if label == cell_a:
        return "cell_a"
    if label == cell_b:
        return "cell_b"
    return "third_label"


def row_evidence(edge, pair):
    """One candidate's endpoint evidence."""
    pre, post = edge["pre_proofread_label"], edge["post_proofread_label"]
    return {"annotation_id": edge["annotation_id"], "pre_cell": edge["pre_cell"], "post_cell": edge["post_cell"],
            "type": edge["type"], "pre_label": pre, "post_label": post,
            "pre_reads": classify_label(pre, *pair), "post_reads": classify_label(post, *pair),
            "pre_nearest_offset": edge.get("pre_nearest_offset"), "post_nearest_offset": edge.get("post_nearest_offset"),
            "same_nonzero_label_both_ends": pre != "0" and pre == post,
            "expected_cells_at_both_ends": pre == edge["pre_cell"] and post == edge["post_cell"],
            "endpoints_within_box": bool(edge.get("endpoints_within_box"))}


def shared_labels(cells_records, pair):
    """C3 ids attributed to both cells in one per-cell record dict (edge list or checkpoint)."""
    a, b = (set(cells_records.get(c, {}).get("c3_ids", [])) for c in pair)
    return sorted(a & b)


def verdict(shared, rows):
    if shared or any(r["same_nonzero_label_both_ends"] for r in rows):
        return "yes"
    if any(r["expected_cells_at_both_ends"] for r in rows):
        return "no"
    return "undetermined"


def check(edge_list, checkpoints):
    """Full merge-check record for the largest pair."""
    pair, count = largest_pair(edge_list["edges"])
    rows = [row_evidence(e, pair) for e in edge_list["edges"] if frozenset((e["pre_cell"], e["post_cell"])) == frozenset(pair)]
    sources = {"edge_list": edge_list["cells"], **{f"checkpoint:{name}": cells for name, cells in checkpoints.items()}}
    shared = {name: shared_labels(cells, pair) for name, cells in sources.items()}
    present = {name: [c for c in pair if c in cells] for name, cells in sources.items()}
    tally = collections.Counter((r["pre_cell"], r["type"]) for r in rows)
    reads = collections.Counter((r["pre_reads"], r["post_reads"]) for r in rows)
    result = verdict([x for v in shared.values() for x in v], rows)
    return {"pair": list(pair), "candidates": count, "verdict": f"suspected merge: {result}",
            "shared_c3_ids": shared, "cells_present_in_source": present,
            "cell_records": {c: {k: v for k, v in edge_list["cells"][c].items() if k != "c3_ids"}
                             | {"c3_ids_count": len(edge_list["cells"][c]["c3_ids"])} for c in pair},
            "emitted_shared_field": {k: v for k, v in edge_list.get("c3_ids_shared_by_cells", {}).items()
                                     if set(v) & set(pair)},
            "direction_type_mix": {f"{pre}->type{t}": n for (pre, t), n in sorted(tally.items())},
            "endpoint_reads": {f"{a}/{b}": n for (a, b), n in sorted(reads.items())},
            "rows_same_nonzero_label_both_ends": sum(r["same_nonzero_label_both_ends"] for r in rows),
            "rows_expected_cells_at_both_ends": sum(r["expected_cells_at_both_ends"] for r in rows),
            "rows_background_both_ends": sum(r["pre_reads"] == "background" == r["post_reads"] for r in rows),
            "rows_within_2voxel_box": sum(r["endpoints_within_box"] for r in rows),
            "rows": rows,
            "reading": ("The saved JSON can show a merge (shared C3 label or one label at both ends) or an "
                        "ordinary contact (expected cells at both ends); endpoints on background decide neither. "
                        "Decisive follow-up: read the C3 label at each endpoint voxel and test membership in each "
                        "cell's sampled c3_ids (online, one scattered_points call).")}


def render_markdown(record):
    lines = [f"# Merge check: pair {record['pair'][0]} / {record['pair'][1]}", "",
             f"**Verdict: {record['verdict']}** ({record['candidates']} candidate contacts).", "",
             "| Evidence | Value |", "| --- | --- |",
             f"| Shared C3 ids by source | {json.dumps(record['shared_c3_ids'])} |",
             f"| Cells present in each source | {json.dumps(record['cells_present_in_source'])} |",
             f"| Direction and type mix | {json.dumps(record['direction_type_mix'])} |",
             f"| Endpoint reads (pre/post) | {json.dumps(record['endpoint_reads'])} |",
             f"| Rows with one nonzero label at both ends | {record['rows_same_nonzero_label_both_ends']} |",
             f"| Rows with expected cells at both ends | {record['rows_expected_cells_at_both_ends']} |",
             f"| Rows background at both ends | {record['rows_background_both_ends']} |",
             f"| Rows within the 2-voxel box | {record['rows_within_2voxel_box']} |", "",
             record["reading"], "", "| Annotation | Pre | Post | Type | Pre reads | Post reads | Pre offset | Post offset |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in record["rows"]:
        lines.append(f"| {r['annotation_id']} | {r['pre_cell']} | {r['post_cell']} | {r['type']} | {r['pre_label']} "
                     f"({r['pre_reads']}) | {r['post_label']} ({r['post_reads']}) | {r['pre_nearest_offset']} | {r['post_nearest_offset']} |")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list", type=Path, default=Path("docs/evidence/h01-resolved-edge-list.json"))
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[Path("docs/evidence/h01-resolved-edge-list-3000.cells.json")])
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/h01-pair-merge-check.json"))
    args = parser.parse_args(argv)
    checkpoints = {p.name: json.loads(p.read_text())["cells"] for p in args.checkpoint if p.exists()}
    record = check(json.loads(args.edge_list.read_text()), checkpoints)
    record["sources"] = {"edge_list": str(args.edge_list), "checkpoints": [str(p) for p in args.checkpoint if p.exists()]}
    args.output.write_text(json.dumps(record, indent=1)+"\n")
    args.output.with_suffix(".md").write_text(render_markdown(record))
    print(record["verdict"], "|", json.dumps({k: record[k] for k in ("pair", "candidates", "shared_c3_ids", "endpoint_reads")}), flush=True)
    return record


if __name__ == "__main__":
    main()
