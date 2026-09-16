"""Control for C3-B: one kept cell exported from C3 beside its proofread SWC (recorded, not a gate).

Reads every ``{cell}.{k}.swc`` member of two archives in the proofread layout
(positions in 32 / 32 / 33 nm voxels, radius in nm, type codes per
``h01_anatomy.LABELS``) and records, per archive: component count, node count,
soma-labelled node count (type 3), cable length in um, the largest component's
node count and the type-code histogram. The C3 skeleton is the release's own
sparse skeletonisation of a non-proofread segment; the proofread SWC is a denser
skeleton of a corrected one, so the two are expected to differ in node count and
component count. The comparison is evidence, not an acceptance test.
"""

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

POSITION_UM = np.array([0.032, 0.032, 0.033])
SOMA_TYPE = 3
MEMBER = re.compile(r"([0-9]+)\.([0-9]+)\.swc\Z")
EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]


def summarize_member(source):
    """Nodes, soma nodes, cable (um) and type histogram of one SWC member in the proofread layout."""
    rows = np.loadtxt(io.BytesIO(source), ndmin=2)
    if rows.shape[1] != 7:
        raise ValueError("SWC member must have seven columns")
    codes = rows[:, 1].astype(int)
    index = {int(r[0]): i for i, r in enumerate(rows)}
    child = np.asarray([i for i, r in enumerate(rows) if int(r[6]) != -1], dtype=np.int64)
    parent = np.asarray([index[int(rows[i, 6])] for i in child], dtype=np.int64)
    points = rows[:, 2:5] * POSITION_UM
    cable = float(np.linalg.norm(points[child] - points[parent], axis=1).sum()) if len(child) else 0.0
    return {"nodes": int(len(rows)), "soma_nodes": int(np.count_nonzero(codes == SOMA_TYPE)),
            "cable_um": cable, "types": {str(k): int(v) for k, v in sorted(Counter(codes.tolist()).items())}}


def summarize_cell(archive_path, cell_id):
    """Per-archive summary of every component of ``cell_id``."""
    parts = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            match = MEMBER.fullmatch(name)
            if match and match[1] == str(cell_id):
                parts.append({"component": int(match[2]), **summarize_member(archive.read(name))})
    if not parts:
        raise KeyError(f"cell {cell_id} has no member in {Path(archive_path).name}")
    types = Counter()
    for part in parts:
        types.update({int(k): v for k, v in part["types"].items()})
    largest = max(parts, key=lambda p: p["nodes"])
    return {"archive": Path(archive_path).name, "components": len(parts),
            "nodes": sum(p["nodes"] for p in parts), "soma_nodes": sum(p["soma_nodes"] for p in parts),
            "cable_um": sum(p["cable_um"] for p in parts),
            "largest_component": largest["component"], "largest_nodes": largest["nodes"],
            "largest_has_soma": bool(largest["soma_nodes"]),
            "types": {str(k): int(v) for k, v in sorted(types.items())},
            "soma_components": [p["component"] for p in parts if p["soma_nodes"]]}


def compare(cell_id, proofread, c3):
    """The control record: both summaries and C3 / proofread ratios."""
    keys = ("components", "nodes", "soma_nodes", "cable_um")
    ratios = {k: (c3[k] / proofread[k] if proofread[k] else None) for k in keys}
    return {"cell_id": str(cell_id), "role": "control; recorded, not a gate",
            "proofread": proofread, "c3": c3, "c3_over_proofread": ratios,
            "note": "C3 is the release's sparse skeleton of the non-proofread segment; the proofread SWC is a "
                    "denser skeleton of the corrected one. Node and component counts are expected to differ."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", default="1684504313")
    parser.add_argument("--proofread", type=Path, default=ROOT / ".cache/h01/proofread104.zip")
    parser.add_argument("--c3-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    record = compare(args.cell, summarize_cell(args.proofread, args.cell), summarize_cell(args.c3_archive, args.cell))
    output = args.output or EVIDENCE / f"h01-c3-control-{args.cell}.json"
    output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("cell_id", "c3_over_proofread")}))
    return record


if __name__ == "__main__":
    main()
