"""Rank the C3 neurons that synapse with the 17 kept proofread cells and pick
the top 20 excitatory and top 20 inhibitory candidates.

Inputs: the kept-partner graph (``h01-c3-kept-partner-graph.json.gz``, kept
cells joined by base-segment ownership; see ``h01_c3_kept_partner_graph.py``),
the cell table (``h01-c3-cell-table.json``), the cellmatrix mapping audit
(released id to C3 id for the kept 17 and for the exclusion of all 104
proofread cells) and the population-types rows (released ids of the 104).
Output: ``docs/evidence/h01-c3-candidates.json`` (kept ids, candidates with
donors, the induced subgraph over kept and candidates, a summary with the
registered prediction's verdict) and a short ``h01-c3-candidates.md`` table.
The earlier C3-id-only ranking is preserved as ``h01-c3-candidates-c3id-only.json``.

Counting: a synapse counts only where ``pre_class == "AXON"`` (a kept axon onto
a tabulated neuron's C3 segment, a tabulated neuron's C3 segment onto a kept
cell, or one kept cell onto another). Rows with any other presynaptic class
(DENDRITE, SOMA, AIS, UNKNOWN, CILIA) are tallied as ``dendrite_pre_rows_ignored``
beside every count and never ranked.

Eligibility: E is ``pyramidal``, ``excitatory/spiny-with-atypical-tree`` or
``spiny-stellate`` with a layer tag (L1-L6, WM) and no ``possible-interneuron``
modifier; I is ``interneuron`` with a layer tag. Proofread cells and the kept
set itself are never candidates. Rank: AXON-pre synapses with the kept set in
both directions (descending), then NSI (descending), then C3 id.

Run from the worktree root (laptop or box):
    PYTHONPATH=. python docs/evidence/h01_c3_partner_rank.py
"""

import collections
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

from docs.evidence.h01_c3_cell_table import load_cell_types

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "docs/evidence/h01-c3-kept-partner-graph.json.gz"
PROGRESS = ROOT / "docs/evidence/h01-c3-kept-partner-graph-progress.json"
TABLE = ROOT / "docs/evidence/h01-c3-cell-table.json"
MAPPING = ROOT / "docs/evidence/h01-cellmatrix-mapping-audit.json"
POPULATION = ROOT / "docs/evidence/h01-population-types.json"
OUT = ROOT / "docs/evidence/h01-c3-candidates.json"
OUT_MD = ROOT / "docs/evidence/h01-c3-candidates.md"
GRAPH_SOURCE = "kept-partner"
KEPT_RELEASED = (1669770671, 1684504313, 2001418787, 2103991145, 2252715458, 2848552900, 3111823553,
                 3571083397, 3761379470, 4010150634, 4197933517, 4437316933, 5013648003, 5173982155,
                 5439194879, 6833911543, 751294744)
E_CLASSES = ("pyramidal", "excitatory/spiny-with-atypical-tree", "spiny-stellate")
TOP_N = 20
MIN_PARTNERS = 20  # falsifier: fewer partners in total than this -> stop after writing the file
PREDICTED_GE2 = 40
AXON = "AXON"


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
    """Rows of a graph file as dicts keyed by the file's columns."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    cols = doc["columns"]
    return [dict(zip(cols, row)) for row in doc["rows"]]


def _new_partner():
    return {"to_kept": 0, "from_kept": 0, "to_kept_axon": 0, "from_kept_axon": 0}


def _new_kept():
    return {"synapses_in": 0, "synapses_out": 0, "synapses_in_axon": 0, "synapses_out_axon": 0,
            "partners": set(), "partners_any_class": set(), "kept_partners": set()}


def row_ends(row, kept_of_released):
    """``(pre, post, pre_kept, post_kept)`` with kept ends as kept C3 ids and the rest as row C3 ids."""
    pre_owner, post_owner = row.get("pre_owner"), row.get("post_owner")
    pre = kept_of_released[pre_owner] if pre_owner is not None else row["pre_c3"]
    post = kept_of_released[post_owner] if post_owner is not None else row["post_c3"]
    return pre, post, pre_owner is not None, post_owner is not None


def partner_counts(rows, kept_of_released):
    """Synapse counts between every tabulated neuron and the kept set, AXON-pre rows beside all rows.

    Parameters
    ----------
    rows : list of dict
        Kept-partner graph rows (``pre_owner``, ``post_owner``, ``pre_c3``,
        ``post_c3``, ``pre_class``).
    kept_of_released : dict
        Kept released id -> kept C3 id.

    Returns
    -------
    tuple
        ``(partners, per_kept, internal)``: ``partners[c3_id] = {to_kept, from_kept,
        to_kept_axon, from_kept_axon}``; ``per_kept[kept_c3] = {synapses_in,
        synapses_out, synapses_in_axon, synapses_out_axon, partners (AXON-pre),
        partners_any_class, kept_partners (AXON-pre)}``; ``internal = {rows,
        axon_rows: {(pre, post): count}, dendrite_pre_rows_ignored, same_cell_rows}``.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_partner_rank import partner_counts
        >>> rows = [{"pre_owner": None, "post_owner": 10, "pre_c3": 5, "post_c3": 1, "pre_class": "AXON"},
        ...         {"pre_owner": None, "post_owner": 10, "pre_c3": 5, "post_c3": 1, "pre_class": "DENDRITE"},
        ...         {"pre_owner": 10, "post_owner": None, "pre_c3": None, "post_c3": 6, "pre_class": "AXON"},
        ...         {"pre_owner": 10, "post_owner": 20, "pre_c3": None, "post_c3": 2, "pre_class": "AXON"},
        ...         {"pre_owner": 10, "post_owner": 10, "pre_c3": 1, "post_c3": 1, "pre_class": "AXON"}]
        >>> partners, per_kept, internal = partner_counts(rows, {10: 1, 20: 2})
        >>> partners[5]
        {'to_kept': 2, 'from_kept': 0, 'to_kept_axon': 1, 'from_kept_axon': 0}
        >>> dict(internal["axon_rows"]), internal["same_cell_rows"], sorted(per_kept[1]["partners"])
        ({(1, 2): 1}, 1, [5, 6])
    """
    partners = collections.defaultdict(_new_partner)
    per_kept = {k: _new_kept() for k in kept_of_released.values()}
    internal = {"rows": 0, "axon_rows": collections.Counter(), "dendrite_pre_rows_ignored": 0, "same_cell_rows": 0}
    for row in rows:
        pre, post, pre_k, post_k = row_ends(row, kept_of_released)
        axon = row.get("pre_class") == AXON
        if pre_k and post_k:
            if pre == post:
                internal["same_cell_rows"] += 1
                continue
            internal["rows"] += 1
            per_kept[pre]["synapses_out"] += 1
            per_kept[post]["synapses_in"] += 1
            if not axon:
                internal["dendrite_pre_rows_ignored"] += 1
                continue
            internal["axon_rows"][(pre, post)] += 1
            per_kept[pre]["synapses_out_axon"] += 1
            per_kept[post]["synapses_in_axon"] += 1
            per_kept[pre]["kept_partners"].add(post)
            per_kept[post]["kept_partners"].add(pre)
        elif post_k:
            partners[pre]["to_kept"] += 1
            per_kept[post]["synapses_in"] += 1
            per_kept[post]["partners_any_class"].add(pre)
            if axon:
                partners[pre]["to_kept_axon"] += 1
                per_kept[post]["synapses_in_axon"] += 1
                per_kept[post]["partners"].add(pre)
        elif pre_k:
            partners[post]["from_kept"] += 1
            per_kept[pre]["synapses_out"] += 1
            per_kept[pre]["partners_any_class"].add(post)
            if axon:
                partners[post]["from_kept_axon"] += 1
                per_kept[pre]["synapses_out_axon"] += 1
                per_kept[pre]["partners"].add(post)
    return dict(partners), per_kept, internal


def axon_total(counts):
    """Rank key: AXON-pre synapses with the kept set in both directions."""
    return counts["to_kept_axon"] + counts["from_kept_axon"]


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


def candidate_row(cid, polarity, cell, counts, vocab):
    """One candidate record (counts over all pre classes beside the AXON-pre subsets that rank)."""
    total = counts["to_kept"] + counts["from_kept"]
    return {"c3_id": cid, "polarity": polarity, "layer": cell["layer"], "cell_class": cell["cell_class"],
            "modifiers": list(cell["modifiers"]), **donor_fields(cell, polarity, vocab),
            "synapses_to_kept": counts["to_kept"], "synapses_from_kept": counts["from_kept"], "synapses_total": total,
            "synapses_to_kept_axon": counts["to_kept_axon"], "synapses_from_kept_axon": counts["from_kept_axon"],
            "synapses_total_axon": axon_total(counts), "dendrite_pre_rows_ignored": total - axon_total(counts),
            "nsi": cell["NSI"]}


def rank_candidates(partners, table, excluded, vocab, top_n=TOP_N):
    """Top ``top_n`` eligible E and I partners by AXON-pre synapses with the kept set.

    Parameters
    ----------
    partners : dict
        From ``partner_counts``; a partner with no AXON-pre synapse is not eligible.
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
        if cid in excluded or cell is None or axon_total(counts) == 0:
            continue
        polarity = polarity_of(cell, vocab)
        if polarity is None:
            continue
        pools[polarity].append((-axon_total(counts), -cell["NSI"], cid, cell, counts))
    eligible = {p: len(v) for p, v in pools.items()}
    out = []
    for polarity in ("E", "I"):
        for _, _, cid, cell, counts in sorted(pools[polarity])[:top_n]:
            out.append(candidate_row(cid, polarity, cell, counts, vocab))
    return out, eligible


def label_counts(rows):
    """Counts of ``type``, ``pre_class`` and ``post_class`` over graph rows (recorded, not interpreted)."""
    return {key: dict(sorted(collections.Counter(str(r.get(key)) for r in rows).items()))
            for key in ("type", "pre_class", "post_class")}


def induced_subgraph(rows, ids, kept_of_released):
    """Graph rows with both ends (kept ends by owner) in ``ids``."""
    return [r for r in rows if set(row_ends(r, kept_of_released)[:2]) <= ids]


def per_kept_summary(per_kept, kept):
    """Per kept cell: partner counts (AXON-pre and any class) and synapse counts in both directions."""
    out = {}
    for k in kept:
        p = per_kept[k["c3_id"]]
        out[str(k["c3_id"])] = {"released_id": k["released_id"], "partners": len(p["partners"]),
                                "partners_any_class": len(p["partners_any_class"]),
                                "kept_partners": sorted(p["kept_partners"]),
                                "synapses_in": p["synapses_in"], "synapses_out": p["synapses_out"],
                                "synapses_in_axon": p["synapses_in_axon"], "synapses_out_axon": p["synapses_out_axon"]}
    return out


def summarize(partners, per_kept, internal, kept, candidates, eligible, excluded_hits, n_rows, row_labels=None):
    """Summary block with the registered prediction's verdict; ``row_labels`` are the graph's label counts.

    ``partners_total`` and the ``>= 2`` counts are over AXON-pre synapses. The registered wording
    (search tree ``qc3_graph``) is ">= 40 partners with >= 2 synapses **to** the kept set" and sets
    the verdict; the brief's "to/from" reading is reported beside it as
    ``partners_ge2_either_direction_ge40`` and ``verdict_either_direction``.
    """
    axon_partners = {cid: c for cid, c in partners.items() if axon_total(c) > 0}
    ge2_to = sum(1 for c in axon_partners.values() if c["to_kept_axon"] >= 2)
    ge2_either = sum(1 for c in axon_partners.values() if axon_total(c) >= 2)
    per = per_kept_summary(per_kept, kept)
    every = all(v["partners"] > 0 for v in per.values())
    axon_rows = [{"pre": pre, "post": post, "count": n} for (pre, post), n in sorted(internal["axon_rows"].items())]
    return {"graph_source": GRAPH_SOURCE, "graph_rows": n_rows, "row_labels": row_labels,
            "counting_rule": "a synapse counts only where pre_class == AXON; other pre classes are "
                             "dendrite_pre_rows_ignored",
            "partners_total": len(axon_partners), "partners_any_class_total": len(partners),
            "partners_ge2_to_kept": ge2_to, "partners_ge2_either_direction": ge2_either,
            "kept_internal_rows": internal["rows"], "kept_internal_axon_rows": axon_rows,
            "kept_internal_dendrite_pre_rows_ignored": internal["dendrite_pre_rows_ignored"],
            "kept_same_cell_rows": internal["same_cell_rows"],
            "per_kept_cell_partner_counts": per, "eligible_partners": eligible,
            "partners_excluded_as_proofread": excluded_hits,
            "candidates": {"E": sum(1 for c in candidates if c["polarity"] == "E"),
                           "I": sum(1 for c in candidates if c["polarity"] == "I")},
            "prediction": {"text": "every kept cell has >= 1 neuron partner; >= 40 partners with >= 2 synapses "
                                   "to the kept set; falsifier: fewer than 20 partners in total",
                           "every_kept_has_partner": every, "partners_ge2_to_kept_ge40": ge2_to >= PREDICTED_GE2,
                           "partners_ge2_either_direction_ge40": ge2_either >= PREDICTED_GE2,
                           "falsifier_fewer_than_20_partners": len(axon_partners) < MIN_PARTNERS,
                           "verdict": "supported" if every and ge2_to >= PREDICTED_GE2 else "refuted",
                           "verdict_either_direction": "supported" if every and ge2_either >= PREDICTED_GE2
                           else "refuted"}}


DEFECT_NOTE = (
    "Why this graph replaces the C3-id-only ranking. The first ranking (`h01-c3-candidates-c3id-only.json`, "
    "from `h01-c3-neuron-graph.json.gz`) kept a synapse only when both `neuron_id` fields were tabulated C3 "
    "neurons. A C3 soma-bearing segment carries almost none of its own axon (axons are separate C3 fragments), "
    "so a kept cell's outgoing synapses never met that rule and its incoming synapses were seen only where the "
    "kept cell's dendrite happened to sit in the tabulated segment: 85 partners, 4 with >= 2 synapses. The kept "
    "17 are proofread cells whose reconstructions are known through base-segment membership "
    "(`.cache/h01/proofread-base-membership.json`), and every export row carries `base_neuron_id` for both "
    "sites, so `h01_c3_kept_partner_graph.py` rescanned the 166 shards keeping a row when a kept cell owns "
    "either base segment (or the row's `neuron_id` is a kept cell's C3 id) and the other side is a tabulated "
    "neuron or another kept cell. Only rows with `pre_class == AXON` count toward partners and ranks; the "
    "other presynaptic classes are recorded as `dendrite_pre_rows_ignored`. The tabulated partners themselves "
    "are still identified by their C3 segment, so a tabulated neuron's own axon output onto a kept cell is "
    "seen only where its axon sits in its soma-bearing segment; the progress file's "
    "`dropped_untabulated_partner` counts the kept-side synapses whose other end is an untabulated fragment.")


def markdown(doc):
    """Short Markdown report: kept-cell partner counts, kept-kept AXON rows and the candidate table."""
    s = doc["summary"]
    p = s["prediction"]
    lines = ["# C3 partners of the 17 kept cells (workstream A, kept-partner graph)", "",
             f"Graph source `{s['graph_source']}`; rows {s['graph_rows']:,}; partners with >= 1 AXON-pre synapse "
             f"{s['partners_total']:,} (any pre class: {s['partners_any_class_total']:,}); >= 2 AXON-pre synapses "
             f"to kept: {s['partners_ge2_to_kept']:,}; either direction: {s['partners_ge2_either_direction']:,}; "
             f"kept-kept rows {s['kept_internal_rows']} (AXON-pre pairs {len(s['kept_internal_axon_rows'])}, "
             f"same-cell rows {s['kept_same_cell_rows']}); eligible E {s['eligible_partners']['E']:,}, "
             f"I {s['eligible_partners']['I']:,}. Prediction A (registered wording, >= 2 to kept): "
             f"**{p['verdict']}**; to/from reading: **{p['verdict_either_direction']}** "
             f"(every kept cell has an AXON-pre partner: {p['every_kept_has_partner']}).", "", DEFECT_NOTE, "",
             "| kept C3 id | released id | partners (AXON-pre) | partners (any) | in | in AXON | out | out AXON |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cid, v in s["per_kept_cell_partner_counts"].items():
        lines.append(f"| {cid} | {v['released_id']} | {v['partners']} | {v['partners_any_class']} | "
                     f"{v['synapses_in']} | {v['synapses_in_axon']} | {v['synapses_out']} | {v['synapses_out_axon']} |")
    lines += ["", "Kept -> kept AXON-pre rows (pre C3 id, post C3 id, count): " + (
        "; ".join(f"{r['pre']} -> {r['post']} x{r['count']}" for r in s["kept_internal_axon_rows"]) or "none"), "",
        "| # | C3 id | pol | layer | class | modifiers | donor | match | to kept AXON | from kept AXON | ignored | NSI |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for i, c in enumerate(doc["candidates"], 1):
        lines.append(f"| {i} | {c['c3_id']} | {c['polarity']} | {c['layer']} | {c['cell_class']} | "
                     f"{', '.join(c['modifiers']) or '-'} | {c['donor_key']} | {c['donor_match']} | "
                     f"{c['synapses_to_kept_axon']} | {c['synapses_from_kept_axon']} | "
                     f"{c['dendrite_pre_rows_ignored']} | {c['nsi']} |")
    return "\n".join(lines) + "\n"


def build(rows, table_rows, audit_rows, population_rows, vocab, released=KEPT_RELEASED):
    """Assemble the candidates document from loaded inputs."""
    kept = kept_c3_ids(audit_rows, released)
    kept_of_released = {k["released_id"]: k["c3_id"] for k in kept}
    kept_set = set(kept_of_released.values())
    table = {int(r["id"]): r for r in table_rows}
    partners, per_kept, internal = partner_counts(rows, kept_of_released)
    excluded = proofread_ids(audit_rows, population_rows) | kept_set
    excluded_hits = sorted(cid for cid in partners if cid in excluded)
    candidates, eligible = rank_candidates(partners, table, excluded, vocab)
    ids = kept_set | {c["c3_id"] for c in candidates}
    summary = summarize(partners, per_kept, internal, kept, candidates, eligible, excluded_hits, len(rows),
                        label_counts(rows))
    return {"kept": kept, "candidates": candidates, "subgraph": induced_subgraph(rows, ids, kept_of_released),
            "summary": summary}


def main():
    vocab = load_cell_types()
    rows = load_graph()
    table_rows = json.loads(TABLE.read_text(encoding="utf-8"))["rows"]
    audit_rows = json.loads(MAPPING.read_text(encoding="utf-8"))["rows"]
    population_rows = json.loads(POPULATION.read_text(encoding="utf-8"))["rows"]
    doc = build(rows, table_rows, audit_rows, population_rows, vocab)
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    doc["summary"]["scan_totals"] = progress.get("totals")
    doc = {"scope": "Partners of the 17 kept proofread cells in the C3 kept-partner synapse graph (kept cells joined "
                    "by base-segment ownership); top 20 E and top 20 I eligible partners by AXON-pre synapses with "
                    "the kept set (both directions), tie-break NSI",
           "graph_source": GRAPH_SOURCE,
           "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "inputs": {"graph": str(GRAPH.relative_to(ROOT)), "graph_sha256": progress.get("output", {}).get("sha256"),
                      "graph_progress": str(PROGRESS.relative_to(ROOT)),
                      "cell_table": str(TABLE.relative_to(ROOT)), "mapping_audit": str(MAPPING.relative_to(ROOT)),
                      "population_types": str(POPULATION.relative_to(ROOT)),
                      "superseded": "docs/evidence/h01-c3-candidates-c3id-only.json"}, **doc}
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
