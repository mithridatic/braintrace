"""Rank the C3 neurons that synapse with the 17 kept proofread cells and pick
the top 20 excitatory and top 20 inhibitory candidates.

Inputs: the neuron graph (``h01-c3-neuron-graph.json.gz``), the cell table
(``h01-c3-cell-table.json``), the cellmatrix mapping audit (released id to C3
id for the kept 17 and for the exclusion of all 104 proofread cells) and the
population-types rows (released ids of the 104). Output:
``docs/evidence/h01-c3-candidates.json`` (kept ids, candidates with donors,
the induced subgraph over kept and candidates, a summary with the registered
prediction's verdict) and a short ``h01-c3-candidates.md`` table.

Eligibility: E is ``pyramidal``, ``excitatory/spiny-with-atypical-tree`` or
``spiny-stellate`` with a layer tag (L1-L6, WM) and no ``possible-interneuron``
modifier; I is ``interneuron`` with a layer tag. Proofread cells and the kept
set itself are never candidates. Rank: synapses with the kept set in both
directions (descending), then NSI (descending), then C3 id.

Run from the worktree root on the box:
    /workspace/venv-c3/bin/python docs/evidence/h01_c3_partner_rank.py
"""

import collections
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

from docs.evidence.h01_c3_cell_table import load_cell_types

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "docs/evidence/h01-c3-neuron-graph.json.gz"
PROGRESS = ROOT / "docs/evidence/h01-c3-neuron-graph-progress.json"
TABLE = ROOT / "docs/evidence/h01-c3-cell-table.json"
MAPPING = ROOT / "docs/evidence/h01-cellmatrix-mapping-audit.json"
POPULATION = ROOT / "docs/evidence/h01-population-types.json"
OUT = ROOT / "docs/evidence/h01-c3-candidates.json"
OUT_MD = ROOT / "docs/evidence/h01-c3-candidates.md"
KEPT_RELEASED = (1669770671, 1684504313, 2001418787, 2103991145, 2252715458, 2848552900, 3111823553,
                 3571083397, 3761379470, 4010150634, 4197933517, 4437316933, 5013648003, 5173982155,
                 5439194879, 6833911543, 751294744)
E_CLASSES = ("pyramidal", "excitatory/spiny-with-atypical-tree", "spiny-stellate")
TOP_N = 20
MIN_PARTNERS = 20  # falsifier: fewer partners in total than this -> stop after writing the file
PREDICTED_GE2 = 40


def kept_c3_ids(audit_rows, released=KEPT_RELEASED):
    """``[{released_id, c3_id}]`` for the kept cells, in ``released`` order.

    Parameters
    ----------
    audit_rows : list of dict
        ``rows`` of the cellmatrix mapping audit (``existing_spatial_owners``, ``c3_ids``).
    released : iterable of int
        Released ids of the kept cells.

    Raises
    ------
    KeyError
        If a kept released id has no audit row.
    """
    by_released = {int(r["existing_spatial_owners"][0]): int(r["c3_ids"][0])
                   for r in audit_rows if r.get("existing_spatial_owners") and r.get("c3_ids")}
    return [{"released_id": int(rid), "c3_id": by_released[int(rid)]} for rid in released]


def proofread_ids(audit_rows, population_rows):
    """Every id (released and C3) of the 104 proofread cells."""
    ids = {int(r["cell_id"]) for r in population_rows}
    for r in audit_rows:
        ids.update(int(x) for x in r.get("existing_spatial_owners") or [])
        ids.update(int(x) for x in r.get("c3_ids") or [])
    return ids


def load_graph(path=GRAPH):
    """Rows of the neuron graph as dicts keyed by the file's columns."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    cols = doc["columns"]
    return [dict(zip(cols, row)) for row in doc["rows"]]


def partner_counts(rows, kept):
    """Synapse counts between every non-kept neuron and the kept set.

    Returns
    -------
    tuple
        ``(partners, per_kept, internal)``: ``partners[c3_id] = {to_kept, from_kept}``
        (partner -> kept, kept -> partner); ``per_kept[kept_id] = {synapses_in,
        synapses_out, partners (set), kept_partners (set)}``; ``internal`` the
        number of rows with both ends kept.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_partner_rank import partner_counts
        >>> rows = [{"pre": 5, "post": 1}, {"pre": 5, "post": 1}, {"pre": 1, "post": 6}, {"pre": 1, "post": 2}]
        >>> partners, per_kept, internal = partner_counts(rows, {1, 2})
        >>> partners[5], partners[6], internal
        ({'to_kept': 2, 'from_kept': 0}, {'to_kept': 0, 'from_kept': 1}, 1)
    """
    partners = collections.defaultdict(lambda: {"to_kept": 0, "from_kept": 0})
    per_kept = {k: {"synapses_in": 0, "synapses_out": 0, "partners": set(), "kept_partners": set()} for k in kept}
    internal = 0
    for row in rows:
        pre, post = row["pre"], row["post"]
        pre_k, post_k = pre in kept, post in kept
        if pre_k and post_k:
            internal += 1
            per_kept[pre]["synapses_out"] += 1
            per_kept[post]["synapses_in"] += 1
            per_kept[pre]["kept_partners"].add(post)
            per_kept[post]["kept_partners"].add(pre)
        elif post_k:
            partners[pre]["to_kept"] += 1
            per_kept[post]["synapses_in"] += 1
            per_kept[post]["partners"].add(pre)
        elif pre_k:
            partners[post]["from_kept"] += 1
            per_kept[pre]["synapses_out"] += 1
            per_kept[pre]["partners"].add(post)
    return dict(partners), per_kept, internal


def polarity_of(cell, vocab):
    """``"E"``, ``"I"`` or ``None`` per the eligibility rule."""
    if cell["layer"] not in vocab.LAYERS:
        return None
    if cell["cell_class"] in E_CLASSES and "possible-interneuron" not in cell["modifiers"]:
        return "E"
    if cell["cell_class"] == "interneuron":
        return "I"
    return None


def donor_fields(cell, polarity, vocab):
    """Donor key and match for one table row via ``donor_for_tags``."""
    tags = [cell["layer"], cell["cell_class"], "neuron", *cell["modifiers"]]
    key = vocab.donor_for_tags(tags, polarity)
    donor = vocab.DONORS[key]
    mods = tuple(sorted(m for m in cell["modifiers"] if m in vocab.MODIFIERS))
    mismatches = [name for name, same in (("layer", cell["layer"] == donor["layer"]),
                                          ("class", cell["cell_class"] == donor["cell_class"]),
                                          ("modifier", mods == tuple(donor["modifiers"]))) if not same]
    return {"donor_key": key, "donor_profile": donor["profile"], "donor_match": ", ".join(mismatches) or "matched"}


def rank_candidates(partners, table, excluded, vocab, top_n=TOP_N):
    """Top ``top_n`` eligible E and I partners by synapses with the kept set.

    Parameters
    ----------
    partners : dict
        From ``partner_counts``.
    table : dict
        ``c3_id -> cell table row``.
    excluded : set of int
        Ids never eligible (proofread cells, the kept set).
    vocab : module
        ``h01_cell_types``.

    Returns
    -------
    tuple
        ``(candidates, eligible_counts)`` with candidates ordered E then I.
    """
    pools = {"E": [], "I": []}
    for cid, counts in partners.items():
        cell = table.get(cid)
        if cid in excluded or cell is None:
            continue
        polarity = polarity_of(cell, vocab)
        if polarity is None:
            continue
        total = counts["to_kept"] + counts["from_kept"]
        pools[polarity].append((-total, -cell["NSI"], cid, cell, counts))
    eligible = {p: len(v) for p, v in pools.items()}
    out = []
    for polarity in ("E", "I"):
        for _, _, cid, cell, counts in sorted(pools[polarity])[:top_n]:
            out.append({"c3_id": cid, "polarity": polarity, "layer": cell["layer"], "cell_class": cell["cell_class"],
                        "modifiers": list(cell["modifiers"]), **donor_fields(cell, polarity, vocab),
                        "synapses_to_kept": counts["to_kept"], "synapses_from_kept": counts["from_kept"],
                        "synapses_total": counts["to_kept"] + counts["from_kept"], "nsi": cell["NSI"]})
    return out, eligible


def label_counts(rows):
    """Counts of ``type``, ``pre_class`` and ``post_class`` over graph rows (recorded, not interpreted)."""
    return {key: dict(sorted(collections.Counter(str(r.get(key)) for r in rows).items()))
            for key in ("type", "pre_class", "post_class")}


def induced_subgraph(rows, ids):
    """Graph rows with both ends in ``ids``."""
    return [r for r in rows if r["pre"] in ids and r["post"] in ids]


def summarize(partners, per_kept, internal, kept, candidates, eligible, excluded_hits, n_rows, row_labels=None):
    """Summary block with the registered prediction's verdict; ``row_labels`` are the graph's label counts."""
    ge2_to = sum(1 for c in partners.values() if c["to_kept"] >= 2)
    ge2_either = sum(1 for c in partners.values() if c["to_kept"] + c["from_kept"] >= 2)
    per = {str(k["c3_id"]): {"released_id": k["released_id"], "partners": len(per_kept[k["c3_id"]]["partners"]),
                             "kept_partners": sorted(per_kept[k["c3_id"]]["kept_partners"]),
                             "synapses_in": per_kept[k["c3_id"]]["synapses_in"],
                             "synapses_out": per_kept[k["c3_id"]]["synapses_out"]} for k in kept}
    every = all(v["partners"] > 0 for v in per.values())
    return {"graph_rows": n_rows, "row_labels": row_labels, "partners_total": len(partners), "partners_ge2_to_kept": ge2_to,
            "partners_ge2_either_direction": ge2_either, "kept_internal_rows": internal,
            "per_kept_cell_partner_counts": per, "eligible_partners": eligible,
            "partners_excluded_as_proofread": excluded_hits,
            "candidates": {"E": sum(1 for c in candidates if c["polarity"] == "E"),
                           "I": sum(1 for c in candidates if c["polarity"] == "I")},
            "prediction": {"text": "every kept cell has >= 1 neuron partner; >= 40 partners with >= 2 synapses "
                                   "to the kept set; falsifier: fewer than 20 partners in total",
                           "every_kept_has_partner": every, "partners_ge2_to_kept_ge40": ge2_to >= PREDICTED_GE2,
                           "partners_ge2_either_direction_ge40": ge2_either >= PREDICTED_GE2,
                           "falsifier_fewer_than_20_partners": len(partners) < MIN_PARTNERS,
                           "verdict": "supported" if every and ge2_to >= PREDICTED_GE2 else "refuted"}}


def markdown(doc):
    """Short Markdown report: kept-cell partner counts and the candidate table."""
    s = doc["summary"]
    lines = ["# C3 partners of the 17 kept cells (workstream A)", "",
             f"Graph rows {s['graph_rows']:,}; partners {s['partners_total']:,} "
             f"(>= 2 synapses to kept: {s['partners_ge2_to_kept']:,}; either direction: "
             f"{s['partners_ge2_either_direction']:,}); kept-kept rows {s['kept_internal_rows']}; "
             f"eligible E {s['eligible_partners']['E']:,}, I {s['eligible_partners']['I']:,}. "
             f"Prediction A: **{s['prediction']['verdict']}** "
             f"(every kept cell has a partner: {s['prediction']['every_kept_has_partner']}).", "",
             "| kept C3 id | released id | partners | in | out |", "| --- | --- | ---: | ---: | ---: |"]
    for cid, v in s["per_kept_cell_partner_counts"].items():
        lines.append(f"| {cid} | {v['released_id']} | {v['partners']} | {v['synapses_in']} | {v['synapses_out']} |")
    lines += ["", "| # | C3 id | pol | layer | class | modifiers | donor | match | to kept | from kept | NSI |",
              "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for i, c in enumerate(doc["candidates"], 1):
        lines.append(f"| {i} | {c['c3_id']} | {c['polarity']} | {c['layer']} | {c['cell_class']} | "
                     f"{', '.join(c['modifiers']) or '-'} | {c['donor_key']} | {c['donor_match']} | "
                     f"{c['synapses_to_kept']} | {c['synapses_from_kept']} | {c['nsi']} |")
    return "\n".join(lines) + "\n"


def build(rows, table_rows, audit_rows, population_rows, vocab, released=KEPT_RELEASED):
    """Assemble the candidates document from loaded inputs."""
    kept = kept_c3_ids(audit_rows, released)
    kept_set = {k["c3_id"] for k in kept}
    table = {int(r["id"]): r for r in table_rows}
    partners, per_kept, internal = partner_counts(rows, kept_set)
    excluded = proofread_ids(audit_rows, population_rows) | kept_set
    excluded_hits = sorted(cid for cid in partners if cid in excluded)
    candidates, eligible = rank_candidates(partners, table, excluded, vocab)
    ids = kept_set | {c["c3_id"] for c in candidates}
    summary = summarize(partners, per_kept, internal, kept, candidates, eligible, excluded_hits, len(rows),
                        label_counts(rows))
    return {"kept": kept, "candidates": candidates, "subgraph": induced_subgraph(rows, ids), "summary": summary}


def main():
    vocab = load_cell_types()
    rows = load_graph()
    table_rows = json.loads(TABLE.read_text(encoding="utf-8"))["rows"]
    audit_rows = json.loads(MAPPING.read_text(encoding="utf-8"))["rows"]
    population_rows = json.loads(POPULATION.read_text(encoding="utf-8"))["rows"]
    doc = build(rows, table_rows, audit_rows, population_rows, vocab)
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    doc = {"scope": "Partners of the 17 kept proofread cells in the C3 neuron synapse graph; top 20 E and top 20 I "
                    "eligible partners by synapses with the kept set (both directions), tie-break NSI",
           "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "inputs": {"graph": str(GRAPH.relative_to(ROOT)), "graph_sha256": progress.get("output", {}).get("sha256"),
                      "cell_table": str(TABLE.relative_to(ROOT)), "mapping_audit": str(MAPPING.relative_to(ROOT)),
                      "population_types": str(POPULATION.relative_to(ROOT))}, **doc}
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(doc), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc["summary"].items() if k != "per_kept_cell_partner_counts"}, indent=1))
    print("wrote", OUT, OUT.stat().st_size, "bytes;", OUT_MD)
    if doc["summary"]["prediction"]["falsifier_fewer_than_20_partners"]:
        print("STOP: fewer than", MIN_PARTNERS, "partners in total; the pool is what exists")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
