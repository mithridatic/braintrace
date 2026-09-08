"""Type table of the 104 H01 cells and how many have a type-matched donor.

Reads the nodes of ``h01-verified-network.json`` (released tags per cell) and
writes ``h01-population-types.{json,md}``. Prediction registered in
``docs/specs/2026-09-07-h01-usable-circuit-plan.md`` (W6a): fewer than 40 of
the 104 cells match their donor in layer and class.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from braintrace.datasets.h01_cell_types import DONORS, type_rows

EVIDENCE = Path(__file__).resolve().parent


def summarize(rows):
    """Counts per (layer, class, modifiers), per match state, and the matched total."""
    by_type = Counter((r["layer"], r["cell_class"], ", ".join(r["modifiers"])) for r in rows)
    by_match = Counter(r["match"] for r in rows)
    return {"cells": len(rows), "matched": by_match.get("matched", 0),
            "by_match": dict(by_match.most_common()),
            "by_type": [{"layer": k[0], "cell_class": k[1], "modifiers": k[2], "count": v}
                        for k, v in by_type.most_common()],
            "types_without_donor": sorted({f"{r['layer']} {r['cell_class']}" + (f" ({', '.join(r['modifiers'])})"
                                            if r["modifiers"] else "") for r in rows if r["match"] != "matched"})}


def donor_notes(evidence=EVIDENCE):
    """Physiology label per donor from the stage decision JSONs under ``h01-donors/`` (absent file: none)."""
    notes = {}
    for path in sorted((evidence/"h01-donors").glob("stage-*-decision.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("donor_key") in DONORS:
            notes[record["donor_key"]] = {"status": record.get("reproduction_status", "untested"),
                                          "label": record.get("population_label", ""),
                                          "decision": f"h01-donors/{path.name}"}
    return notes


def render(report):
    """Markdown page: the donors, their physiology notes, the counts, and the 104-row table."""
    s = report["summary"]
    lines = ["# H01 population: cell types and donor match", "",
             f"{s['cells']} cells from the released tags; **{s['matched']} have a donor matched in layer and class**",
             "(prediction: fewer than 40). Every interneuron's subtype is unknown in the tags; the parvalbumin",
             "basket donor (HL5BN1) is assumed for 18 interneurons (L1/L2/L4/L5) and the putative-SST donor",
             "(HL5MN1) for the 7 L3 interneurons.", "", "## Donors", "", "| Donor key | Polarity | Profile | Source | Layer | Class |",
             "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {k} | {d['polarity']} | {d['profile']} | {d['source']} | {d['layer']} | {d['cell_class']} |"
              for k, d in DONORS.items()]
    notes = report.get("donor_notes") or {}
    if notes:
        lines += ["", "## Donor physiology notes", "",
                  "Reproduction of each imported donor's published fit in this project's NEURON container, and the",
                  "usable-tier verdicts on its own recordings. A donor without a note (the two original donors) is",
                  "scored on its own pages. A pass is not required to enter the population; the verdicts are labels.", "",
                  "| Donor key | Reproduction | Label | Decision |", "| --- | --- | --- | --- |"]
        lines += [f"| {k} | {n['status']} | {n['label']} | [{n['decision']}]({n['decision']}) |" for k, n in notes.items()]
    lines += ["", "## Match states", "", "| Match | Cells |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in s["by_match"].items()]
    lines += ["", "## Types", "", "| Layer | Class | Modifiers | Cells |", "| --- | --- | --- | --- |"]
    lines += [f"| {t['layer']} | {t['cell_class']} | {t['modifiers']} | {t['count']} |" for t in s["by_type"]]
    lines += ["", "## Cells", "", "| Cell | Layer | Class | Modifiers | Polarity | Donor key | Donor | Match |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['cell_id']} | {r['layer']} | {r['cell_class']} | {', '.join(r['modifiers'])} | {r['polarity']} | "
              f"{r['donor_key']} | {r['donor']} | {r['match']} |" for r in report["rows"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=EVIDENCE/"h01-verified-network.json")
    parser.add_argument("--output", type=Path, default=EVIDENCE/"h01-population-types")
    args = parser.parse_args(argv)
    network = json.loads(args.network.read_text(encoding="utf-8"))
    rows = type_rows(network["nodes"])
    report = {"source": args.network.name, "archive_sha256": network.get("archive_sha256"),
              "summary": summarize(rows), "donor_notes": donor_notes(), "rows": rows}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report["summary"]["by_match"]))


if __name__ == "__main__":
    main()
