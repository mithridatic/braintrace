"""Import the selected H01 population before allocating electrical cells."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_network import plan_h01_cells


def main():
    """Validate pinned components and write incremental import evidence.

    Returns
    -------
    None
        Evidence is written to the explicit output path after each import.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topology = json.loads(args.topology.read_text())
    components = json.loads(args.components.read_text())
    plan = plan_h01_cells(topology, include_isolated=True, cells=104, components=components)
    selected = {identity: row["component"] for identity, row in plan["isolated_cells"].items()}
    for contact in topology["contacts"]:
        if contact["construction_ready"]:
            for side in ("pre", "post"):
                selected[contact[side + "_cell"]] = contact[side + "_placement"]["component"]
    archive = H01Archive(args.archive)
    report = {"status": "running", "rows": [], "versions": {
        name: importlib.metadata.version(name)
        for name in ("braincell", "brainstate", "brainunit", "jax", "jaxlib", "numpy")
    }, "source_sha256": {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.archive, args.topology, args.components)
    }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for identity in plan["simulated_cell_ids"]:
        began = time.perf_counter()
        imported = archive.load(identity, component=selected[identity])
        report["rows"].append({"source_id": identity, "component": selected[identity],
                               "source_nodes": len(imported.source_rows),
                               "source_sha256": imported.source_sha256,
                               "seconds": time.perf_counter() - began,
                               "errors": imported.report.error_count})
        report["elapsed_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{len(report['rows'])}/104 {identity}: {report['rows'][-1]['seconds']:.2f}s", flush=True)
    report["status"] = "pass" if len(report["rows"]) == 104 and not any(
        row["errors"] for row in report["rows"]) else "fail"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
