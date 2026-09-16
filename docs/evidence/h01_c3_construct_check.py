"""Construct and initialise every candidate of a C3 archive with the production builder (C3-B).

For each cell of a components inventory (``h01_population_components.py`` shape) the
largest component is loaded from the archive (explicit digest), painted with the deployed
donor profile for its polarity by ``make_h01_ei_cell`` (``_regions``, MaxCVLen 10 um,
implicit calcium solver, nothing tuned), registered in a one-cell ``braincell.Network``
and initialised by ``init_h01_network_states``. Nothing is run. The record per cell is
the component, polarity, donor, node and compartment counts, construction and
initialisation seconds and the error text when a stage fails. Output is checkpointed
after every cell; a rerun skips cells already passed and retries failures.

Tags for C3 ids come from the C3 cell table (``--cell-table``, ``H01SegmentProperties``);
the proofread table does not know them. Roles come from the candidates document (``polarity`` / ``donor`` / ``tags`` per cell,
``donor_for_tags`` resolving the donor) or from ``--polarity`` / ``--donor`` overrides.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from braintrace.datasets.h01_cell_types import DEFAULT_DONOR_KEYS, donor_for_tags  # noqa: E402

BASIS = ("Source soma/axon/AIS samples extend halfway along adjacent edges; remaining cable gets "
         "dendrite properties. Myelin is not modeled. This is the existing inferred electrical "
         "partition, not measured channel placement.")
EVIDENCE = Path(__file__).resolve().parent


def resolve_role(candidate, polarity=None, donor=None):
    """Polarity and donor key for one candidate record; overrides win, then the record, then the tags."""
    tags = list(candidate.get("tags") or [t for t in (candidate.get("layer"), candidate.get("cell_class")) if t])
    if polarity is None:
        polarity = candidate.get("polarity")
    if polarity is None:
        polarity = "I" if "interneuron" in tags or candidate.get("cell_class") == "interneuron" else "E"
    if donor is None:
        donor = candidate.get("donor") or candidate.get("donor_key")
    if donor is None:
        donor = donor_for_tags(tags, polarity) if tags else DEFAULT_DONOR_KEYS[polarity]
    return str(polarity), str(donor)


def candidate_index(document):
    """``{c3_id: record}`` from a candidates document (list, or under candidates/cells)."""
    items = document
    if isinstance(document, dict):
        items = document.get("candidates", document.get("cells", []))
    index = {}
    for item in items:
        if isinstance(item, dict):
            key = next((item[k] for k in ("c3_id", "neuron_id", "segment_id", "id") if k in item), None)
            if key is not None:
                index[str(key)] = item
    return index


def plan(inventory, candidates, polarity=None, donor=None):
    """One job per inventory cell: cell_id, component, polarity, donor."""
    jobs = []
    for row in inventory["cells"]:
        role = resolve_role(candidates.get(row["cell_id"], {}), polarity, donor)
        jobs.append({"cell_id": row["cell_id"], "component": row["largest_component"],
                     "polarity": role[0], "donor": role[1]})
    return jobs


def load_annotations(cache, cell_table=None):
    """The proofread annotation store, or the C3 cell table (``H01SegmentProperties``) when given."""
    from braintrace.datasets.h01_annotations import H01Annotations, H01SegmentProperties

    if cell_table is not None:
        return H01SegmentProperties(cell_table)
    return H01Annotations(cache)


def production_builder(archive, annotations, max_cv_um, solver, current_na, emit):
    """Closure that constructs and initialises one cell with the production path."""
    import braincell
    import brainstate

    from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
    from braintrace.datasets.h01_network import _regions, _register_cell
    from braintrace.datasets.h01_network_init import init_h01_network_states, process_rss_mb

    def build(job):
        with brainstate.environ.context(precision=64):
            started = time.perf_counter()
            imported = archive.load(job["cell_id"], component=job["component"])
            record = {"component_nodes": int(len(imported.source_rows)), "source_sha256": imported.source_sha256}
            cell, evidence = make_h01_ei_cell(imported, annotations, polarity=job["polarity"], donor=job["donor"],
                                              regions=_regions(imported), region_basis=BASIS,
                                              current_na=current_na, delay_ms=2., duration_ms=3.,
                                              max_cv_length_um=max_cv_um, pop_size=(1,), solver=solver)
            network = braincell.Network(name="h01_c3_construct")
            _register_cell(network, job["cell_id"], cell, evidence, imported, {}, current_na, emit)
            record.update(n_compartments=evidence.get("n_compartments"),
                          construction_seconds=time.perf_counter() - started)
            started = time.perf_counter()
            init_h01_network_states(network, progress=emit, heartbeat_seconds=30.)
            record.update(init_seconds=time.perf_counter() - started, rss_mb=process_rss_mb())
        return record

    return build


def check(jobs, builder, output, emit=print):
    """Run ``builder`` on every job not yet passed in ``output``; failures are retried; checkpoint after each."""
    output = Path(output)
    result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"cells": []}
    result["cells"] = [c for c in result["cells"] if c.get("passed")]
    done = {(c["cell_id"], c["component"]) for c in result["cells"]}
    result["status"] = "running"
    for job in jobs:
        if (job["cell_id"], job["component"]) in done:
            continue
        record = dict(job)
        tick = time.perf_counter()
        try:
            record.update(builder(job))
            record["passed"] = True
        except Exception as exc:  # noqa: BLE001 - every failure reason is recorded
            record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.perf_counter() - tick
        result["cells"].append(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=1), encoding="utf-8")
        emit(f"{len(result['cells'])}/{len(jobs)} {job['cell_id']} passed={record['passed']} "
             f"{record.get('error', '')} {record['seconds']:.0f} s")
    result.update(status="completed", cells_expected=len(jobs),
                  passed_count=sum(c["passed"] for c in result["cells"]),
                  passed=len(result["cells"]) == len(jobs) and all(c["passed"] for c in result["cells"]))
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv=None, builder=None):
    """Command line entry; ``builder`` is injectable for tests."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source")
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--polarity", choices=["E", "I"])
    parser.add_argument("--donor")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/h01")
    parser.add_argument("--cell-table", type=Path,
                        help="C3 segment-properties JSON supplying tags for C3 ids (else the proofread table).")
    parser.add_argument("--max-cv-um", type=float, default=10.)
    parser.add_argument("--solver", default="h01_staggered_calcium_implicit")
    parser.add_argument("--current-na", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inventory = json.loads(args.components.read_text(encoding="utf-8"))
    candidates = candidate_index(json.loads(args.candidates.read_text(encoding="utf-8"))) if args.candidates else {}
    jobs = plan(inventory, candidates, args.polarity, args.donor)
    started = time.perf_counter()
    emit = lambda message: print(f"[{time.perf_counter() - started:.0f}s] {message}", flush=True)  # noqa: E731
    if builder is None:
        from braintrace.datasets.h01 import H01Archive
        archive = H01Archive(args.archive, expected_sha256=args.expected_sha256, source=args.source)
        annotations = load_annotations(args.cache, args.cell_table)
        builder = production_builder(archive, annotations, args.max_cv_um, args.solver, args.current_na, emit)
    result = check(jobs, builder, args.output, emit)
    result.update(archive=args.archive.name, expected_sha256=args.expected_sha256, source=args.source,
                  cell_table=args.cell_table.name if args.cell_table else "cell_properties.json (proofread_104)",
                  max_cv_um=args.max_cv_um, solver=args.solver, seconds=time.perf_counter() - started,
                  scope="construct + initialise only; largest component per cell; deployed donor profile unchanged")
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    emit(f"{result['passed_count']}/{len(jobs)} constructed and initialised")
    return result


if __name__ == "__main__":
    main()
