"""Type tables for the C3 test-to-failure gate: the 40 candidates and the 17 kept cells.

Spec: docs/specs/2026-09-16-h01-c3-partner-expansion.md, step C. Writes two files in the
``h01-population-types.json`` shape (``rows`` of ``cell_id``, ``layer``, ``cell_class``,
``modifiers``, ``polarity``, ``donor``, ``donor_key``, ``match``, ``note``) so that
``h01_keep_drop_chain.rows_from_types`` and ``h01_anatomy_transfer_decision.decide_keep``
read them unchanged:

* ``types.json`` — the 40 C3 candidates of ``h01-c3-candidates.json`` with the donor
  assignment recorded there (``donor_match`` beside ``match``; donors are not reassigned),
  ``source: c3 non-proofread``, ordered interneurons first then pyramidal cells, each in
  the candidates file's rank order (the chain order).
* ``kept-types.json`` — the 17 kept cells of ``h01-keep-drop/decision.json`` with their
  rows from ``h01-population-types.json`` (proofread archive, unchanged donors).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from braintrace.datasets.h01_cell_types import DONORS

EVIDENCE = Path(__file__).resolve().parent
C3_SOURCE = "c3 non-proofread"


def candidate_rows(candidates, components=None):
    """Type rows of the candidates, interneurons first, each polarity in the candidates file's rank order.

    Parameters
    ----------
    candidates : list of dict
        ``candidates`` of ``h01-c3-candidates.json`` (``c3_id``, ``polarity``, ``layer``,
        ``cell_class``, ``modifiers``, ``donor_key``, ``donor_profile``, ``donor_match``).
    components : dict, optional
        ``h01-c3-candidates-components.json``; when given every candidate must have an
        inventory row (the archive the chain loads) and its component is recorded.

    Raises
    ------
    ValueError
        A donor key outside the registry, a donor whose polarity differs from the
        candidate's, a duplicate id, or a candidate absent from the components inventory.
    """
    inventory = None if components is None else {c["cell_id"]: c for c in components["cells"]}
    rows, seen = [], set()
    for polarity in ("I", "E"):
        for rank, candidate in enumerate((c for c in candidates if c["polarity"] == polarity), start=1):
            cell = str(candidate["c3_id"])
            donor = DONORS.get(candidate["donor_key"])
            if donor is None:
                raise ValueError(f"{cell}: donor {candidate['donor_key']} is not registered")
            if donor["polarity"] != polarity:
                raise ValueError(f"{cell}: donor {candidate['donor_key']} is {donor['polarity']}, candidate is {polarity}")
            if cell in seen:
                raise ValueError(f"{cell}: duplicate candidate")
            if inventory is not None and cell not in inventory:
                raise ValueError(f"{cell}: not in the components inventory")
            seen.add(cell)
            row = dict(cell_id=cell, layer=candidate["layer"], cell_class=candidate["cell_class"],
                       modifiers=list(candidate.get("modifiers") or []), polarity=polarity,
                       donor=candidate.get("donor_profile", donor["profile"]), donor_key=candidate["donor_key"],
                       match=candidate["donor_match"], donor_match=candidate["donor_match"],
                       note=("layer-only match: the donor's layer differs from the cell's" if candidate["donor_match"] != "matched" else ""),
                       source=C3_SOURCE, rank=rank, synapses_total_axon=candidate.get("synapses_total_axon"),
                       nsi=candidate.get("nsi"))
            if inventory is not None:
                row["component"] = inventory[cell]["largest_component"]
            rows.append(row)
    return rows


def kept_rows(population_types, decision):
    """Rows of the kept cells from the population types file, in the decision's ``kept`` order."""
    by_cell = {row["cell_id"]: row for row in population_types["rows"]}
    missing = [cell for cell in decision["kept"] if cell not in by_cell]
    if missing:
        raise ValueError(f"kept cells without a type row: {missing}")
    return [dict(by_cell[cell], source="proofread_104") for cell in decision["kept"]]


def summarize(rows):
    """Counts by donor, by match state and by polarity."""
    return dict(cells=len(rows), by_donor=dict(Counter(r["donor_key"] for r in rows)),
                by_match=dict(Counter(r["match"] for r in rows)), by_polarity=dict(Counter(r["polarity"] for r in rows)),
                by_layer_class=dict(Counter(f"{r['layer']} {r['cell_class']}" for r in rows)))


def build(candidates_doc, components_doc, population_types, decision):
    """The two documents (candidates types, kept types)."""
    rows = candidate_rows(candidates_doc["candidates"], components_doc)
    candidates = dict(source=C3_SOURCE, candidates_file="h01-c3-candidates.json", archive=components_doc["archive"],
                      archive_sha256=candidates_doc.get("archive_sha256"),
                      order="interneuron candidates first, then pyramidal, each in the candidates file's rank order",
                      donor_rule="donor_key as assigned in h01-c3-candidates.json (donor_for_tags); not reassigned here",
                      summary=summarize(rows), rows=rows)
    kept = kept_rows(population_types, decision)
    kept_doc = dict(source="h01-population-types.json rows of the h01-keep-drop/decision.json kept list",
                    archive="proofread104.zip", archive_sha256=population_types.get("archive_sha256"),
                    summary=summarize(kept), rows=kept)
    return candidates, kept_doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=EVIDENCE/"h01-c3-candidates.json")
    parser.add_argument("--components", type=Path, default=EVIDENCE/"h01-c3-candidates-components.json")
    parser.add_argument("--population-types", type=Path, default=EVIDENCE/"h01-population-types.json")
    parser.add_argument("--decision", type=Path, default=EVIDENCE/"h01-keep-drop"/"decision.json")
    parser.add_argument("--archive-sha256", default=None, help="digest of the candidate archive, recorded in types.json")
    parser.add_argument("--out-dir", type=Path, default=EVIDENCE/"h01-c3-keep-drop")
    args = parser.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
    candidates_doc = load(args.candidates)
    if args.archive_sha256:
        candidates_doc["archive_sha256"] = args.archive_sha256
    candidates, kept = build(candidates_doc, load(args.components), load(args.population_types), load(args.decision))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, document in (("types.json", candidates), ("kept-types.json", kept)):
        (args.out_dir/name).write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8", newline="\n")
        print(args.out_dir/name, document["summary"]["cells"], "rows", document["summary"]["by_donor"])


if __name__ == "__main__":
    main()
