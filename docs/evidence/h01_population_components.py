"""Component inventory of the 104 H01 cells: what a population build would model.

The archive holds disconnected components per cell; the builder never joins
them and loads only the component that carries a contact. This script reads
every SWC member of ``proofread104.zip`` directly (annotation column, node
count) and reports per cell: component count, soma-bearing components, the
largest component's node share, and whether every verified contact sits on a
soma-bearing component. No BrainCell import is needed.
"""

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

SOMA_CODE, AIS_CODE = 3, 5
MEMBER = re.compile(r"([0-9]+)\.([0-9]+)\.swc\Z")
EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]


def component_record(name, source):
    """Node count and annotation counts of one SWC member."""
    rows = np.loadtxt(io.BytesIO(source), ndmin=2)
    codes = rows[:, 1].astype(int)
    match = MEMBER.fullmatch(name)
    return {"cell_id": match[1], "component": int(match[2]), "nodes": int(len(rows)),
            "soma_nodes": int(np.count_nonzero(codes == SOMA_CODE)),
            "ais_nodes": int(np.count_nonzero(codes == AIS_CODE))}


def inventory(archive_path):
    """One record per component of every cell in the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        return [component_record(name, archive.read(name)) for name in sorted(archive.namelist())
                if MEMBER.fullmatch(name)]


def per_cell(records):
    """Per-cell summary rows from the component records."""
    cells = {}
    for record in records:
        cells.setdefault(record["cell_id"], []).append(record)
    rows = []
    for cell_id, parts in sorted(cells.items(), key=lambda item: int(item[0])):
        nodes = sum(p["nodes"] for p in parts)
        largest = max(parts, key=lambda p: p["nodes"])
        soma = [p["component"] for p in parts if p["soma_nodes"]]
        rows.append({"cell_id": cell_id, "components": len(parts), "nodes": nodes,
                     "soma_components": soma, "largest_component": largest["component"],
                     "largest_share": largest["nodes"]/nodes, "largest_has_soma": bool(largest["soma_nodes"]),
                     "fragments_without_soma": len(parts)-len(soma)})
    return rows


def contact_check(network, rows):
    """Whether each verified contact's endpoints sit on soma-bearing components."""
    soma = {r["cell_id"]: set(r["soma_components"]) for r in rows}
    checks = []
    for contact in network.get("contacts", []):
        for side in ("pre", "post"):
            placement = contact.get(f"{side}_placement") or {}
            cell = contact[f"{side}_cell"]
            checks.append({"annotation_id": contact["annotation_id"], "side": side, "cell_id": cell,
                           "component": placement.get("component"),
                           "on_soma_component": placement.get("component") in soma.get(cell, set())})
    return checks


def summarize(rows, checks):
    shares = [r["largest_share"] for r in rows]
    return {"cells": len(rows), "components": sum(r["components"] for r in rows),
            "cells_with_one_soma_component": sum(len(r["soma_components"]) == 1 for r in rows),
            "cells_with_several_soma_components": sum(len(r["soma_components"]) > 1 for r in rows),
            "cells_without_soma": sum(not r["soma_components"] for r in rows),
            "cells_where_largest_lacks_soma": sum(not r["largest_has_soma"] for r in rows),
            "largest_share_median": float(np.median(shares)), "largest_share_min": float(min(shares)),
            "contact_endpoints": len(checks),
            "contact_endpoints_on_soma_component": sum(c["on_soma_component"] for c in checks)}


def render(report):
    s = report["summary"]
    lines = ["# H01 population: components per cell", "",
             f"{s['cells']} cells, {s['components']} components. A population build that loads one component per cell",
             f"models a fragment unless that component carries the soma: {s['cells_where_largest_lacks_soma']} cells have",
             f"their largest component without a soma; {s['cells_with_several_soma_components']} cells have more than one",
             f"soma-bearing component; {s['cells_without_soma']} have none. Median share of a cell's nodes in its largest",
             f"component: {s['largest_share_median']:.2f} (minimum {s['largest_share_min']:.2f}).",
             f"Verified contact endpoints on a soma-bearing component: {s['contact_endpoints_on_soma_component']} of {s['contact_endpoints']}.",
             "", "| Cell | Components | Nodes | Soma components | Largest | Share | Largest has soma | Fragments |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['cell_id']} | {r['components']} | {r['nodes']} | {r['soma_components']} | {r['largest_component']} | "
              f"{r['largest_share']:.2f} | {r['largest_has_soma']} | {r['fragments_without_soma']} |" for r in report["cells"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/".cache/h01/proofread104.zip")
    parser.add_argument("--network", type=Path, default=EVIDENCE/"h01-verified-network.json")
    parser.add_argument("--output", type=Path, default=EVIDENCE/"h01-population-components")
    args = parser.parse_args(argv)
    rows = per_cell(inventory(args.archive))
    checks = contact_check(json.loads(args.network.read_text(encoding="utf-8")), rows)
    report = {"archive": args.archive.name, "summary": summarize(rows, checks), "cells": rows, "contacts": checks}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
