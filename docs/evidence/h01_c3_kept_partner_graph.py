"""Stream all 166 H01 C3 synapse-export shards and keep every synapse between
one of the 17 kept proofread cells and a tabulated C3 neuron, joining the kept
cells through **base-segment ownership** rather than the C3 ``neuron_id``.

Why a second scan. ``h01_c3_neuron_graph.py`` kept a row only when both
``neuron_id`` fields were tabulated C3 neurons. A C3 soma-bearing segment
carries almost none of its own axon (axons are separate C3 fragments), so a
kept cell's outgoing synapses never met that rule: 85 partners in total, 4
with two or more synapses. The kept 17 are proofread cells whose complete
axons are known through base-segment membership
(``.cache/h01/proofread-base-membership.json``, the same join as
``h01_full_export_join.py::load_owners``), and every export row carries the
base segment of both sites (``base_neuron_id``).

Keep rule (``classify``). A side is *kept* when its base segment is owned by a
kept cell **or** its ``neuron_id`` is a kept cell's C3 id; a side is
*tabulated* when its ``neuron_id`` is in ``h01-c3-cell-table.json``. A row is
kept as ``kept_to_kept`` (both sides kept, different cells), ``same_cell``
(both sides the same kept cell), ``kept_to_tabulated`` (pre kept, post
tabulated) or ``tabulated_to_kept`` (post kept, pre tabulated); classified
exactly once, in that order.

Graph row layout (``columns``): ``pre_owner, post_owner`` (kept released id or
null), ``pre_c3, post_c3`` (the row's ``neuron_id``, null when absent),
``type`` (1 inhibitory, 2 excitatory), ``x, y, z`` (8/8/33 nm voxels),
``pre_class, post_class``, ``confidence`` (4 decimals), ``shard`` (integer N of
``export{N:012d}``), ``index`` (record index within the shard).

Machinery (download with ``curl -4``, prefetch threads, ``spawn`` scan
workers, atomic progress, shard deletion, resume) is imported from
``h01_c3_neuron_graph``; only the keep context, the scan function, the cache
directory and the output paths differ.

Run from the worktree root on the box (receipts under var/c3-graph/kept-graph/):
    PYTHONPATH=. /workspace/venv-c3/bin/python docs/evidence/h01_c3_kept_partner_graph.py --workers 6
"""

import argparse
import collections
import datetime as dt
import json
import sys
import time
from pathlib import Path

from docs.evidence.h01_c3_neuron_graph import (BASE_URL, LISTING, TABLE, load_neuron_ids, merge_parts, part_path,
                                               run, sha256_path, shard_index, totals, write_json)

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "docs/evidence/h01-cellmatrix-mapping-audit.json"
MEMBERSHIP = ROOT / ".cache/h01/proofread-base-membership.json"
PROGRESS = ROOT / "docs/evidence/h01-c3-kept-partner-graph-progress.json"
GRAPH = ROOT / "docs/evidence/h01-c3-kept-partner-graph.json.gz"
CACHE = ROOT / ".cache/h01/c3-kept-graph"
COLUMNS = ("pre_owner", "post_owner", "pre_c3", "post_c3", "type", "x", "y", "z", "pre_class", "post_class",
           "confidence", "shard", "index")
KINDS = ("kept_to_kept", "same_cell", "kept_to_tabulated", "tabulated_to_kept")
KEPT_RELEASED = (1669770671, 1684504313, 2001418787, 2103991145, 2252715458, 2848552900, 3111823553,
                 3571083397, 3761379470, 4010150634, 4197933517, 4437316933, 5013648003, 5173982155,
                 5439194879, 6833911543, 751294744)


def load_kept_owners(audit_rows, membership, kept=KEPT_RELEASED):
    """Base-segment and C3-id ownership of the kept cells.

    Parameters
    ----------
    audit_rows : list of dict
        ``rows`` of the cellmatrix mapping audit (``record``,
        ``existing_spatial_owners``, ``c3_ids``).
    membership : list of dict
        The proofread base-membership file: one entry per proofread cell with
        ``file`` (the audit's ``record``) and ``regions`` (region name -> base ids).
    kept : iterable of int
        Released ids of the kept cells.

    Returns
    -------
    tuple
        ``(base_owner, c3_owner)``: ``base_owner[base_id] = released_id`` over
        every base segment of a kept cell; ``c3_owner[c3_id] = released_id``
        for the kept cells' C3 ids.

    Raises
    ------
    ValueError
        If a base segment is claimed by two kept cells, or a kept cell has no
        membership entry.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_kept_partner_graph import load_kept_owners
        >>> audit = [{"record": "a.json", "existing_spatial_owners": ["11"], "c3_ids": [21]},
        ...          {"record": "b.json", "existing_spatial_owners": ["12"], "c3_ids": [22]}]
        >>> membership = [{"file": "a.json", "regions": {"axon": [1, 2], "dendrite": [3]}},
        ...               {"file": "b.json", "regions": {"dendrite": [4]}}]
        >>> load_kept_owners(audit, membership, kept=(11,))
        ({1: 11, 2: 11, 3: 11}, {21: 11})
    """
    kept = {int(k) for k in kept}
    released_of = {r["record"]: int(r["existing_spatial_owners"][0]) for r in audit_rows
                   if r.get("existing_spatial_owners")}
    c3_owner = {int(c): released_of[r["record"]] for r in audit_rows for c in r.get("c3_ids") or []
                if r["record"] in released_of and released_of[r["record"]] in kept}
    base_owner, seen = {}, set()
    for entry in membership:
        owner = released_of.get(entry["file"])
        if owner not in kept:
            continue
        seen.add(owner)
        for ids in entry["regions"].values():
            for b in ids:
                b = int(b)
                if base_owner.get(b, owner) != owner:
                    raise ValueError(f"base segment {b} claimed by {base_owner[b]} and {owner}")
                base_owner[b] = owner
    missing = kept - seen
    if missing:
        raise ValueError(f"kept cells without membership: {sorted(missing)}")
    return base_owner, c3_owner


def side_owner(site, context):
    """Kept released id of one synapse side (base ownership first, then the kept C3 id), or ``None``."""
    base = site.get("base_neuron_id")
    owner = None if base is None else context["base_owner"].get(int(base))
    if owner is not None:
        return owner
    c3 = site.get("neuron_id")
    return None if c3 is None else context["c3_owner"].get(int(c3))


def classify(pre_owner, post_owner, pre_c3, post_c3, neuron_ids):
    """Row kind per the keep rule, or ``None``.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_kept_partner_graph import classify
        >>> classify(11, 12, None, 22, {22, 30})
        'kept_to_kept'
        >>> classify(11, 11, 21, 21, {21})
        'same_cell'
        >>> classify(11, None, None, 30, {22, 30})
        'kept_to_tabulated'
        >>> classify(None, 12, 30, None, {22, 30})
        'tabulated_to_kept'
        >>> classify(11, None, None, 99, {22, 30}) is None
        True
    """
    if pre_owner is not None and post_owner is not None:
        return "same_cell" if pre_owner == post_owner else "kept_to_kept"
    if pre_owner is not None:
        return "kept_to_tabulated" if post_c3 in neuron_ids else None
    if post_owner is not None:
        return "tabulated_to_kept" if pre_c3 in neuron_ids else None
    return None


def keep_row(record, index, context, shard):
    """``(kind, row)`` for one export record, or ``None``.

    Parameters
    ----------
    record : dict
        One Avro record (``pre_synaptic_site``, ``post_synaptic_partner`` with
        ``neuron_id`` and ``base_neuron_id``, ``type``, ``location``, ``confidence``).
    index : int
        Record index within the shard.
    context : dict
        ``base_owner``, ``c3_owner`` (from ``load_kept_owners``) and
        ``neuron_ids`` (tabulated C3 ids).
    shard : int
        Shard index recorded on the row.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_kept_partner_graph import keep_row
        >>> ctx = {"base_owner": {1: 11}, "c3_owner": {21: 11}, "neuron_ids": frozenset({21, 30})}
        >>> rec = {"pre_synaptic_site": {"neuron_id": None, "base_neuron_id": 1, "class_label": "AXON"},
        ...        "post_synaptic_partner": {"neuron_id": 30, "base_neuron_id": 9, "class_label": "DENDRITE"},
        ...        "type": 2, "location": {"x": 10, "y": 20, "z": 3}, "confidence": 0.98765}
        >>> keep_row(rec, 5, ctx, 7)
        ('kept_to_tabulated', [11, None, None, 30, 2, 10, 20, 3, 'AXON', 'DENDRITE', 0.9877, 7, 5])
    """
    pre, post = record["pre_synaptic_site"], record["post_synaptic_partner"]
    pre_owner, post_owner = side_owner(pre, context), side_owner(post, context)
    if pre_owner is None and post_owner is None:
        return None
    pre_c3, post_c3 = pre.get("neuron_id"), post.get("neuron_id")
    pre_c3 = None if pre_c3 is None else int(pre_c3)
    post_c3 = None if post_c3 is None else int(post_c3)
    kind = classify(pre_owner, post_owner, pre_c3, post_c3, context["neuron_ids"])
    if kind is None:
        return None
    loc = record["location"]
    conf = record.get("confidence")
    return kind, [pre_owner, post_owner, pre_c3, post_c3, record.get("type"), loc["x"], loc["y"], loc["z"],
                  pre.get("class_label"), post.get("class_label"),
                  None if conf is None else round(float(conf), 4), shard, index]


def scan_shard(path, shard, context, cache=CACHE):
    """Scan one downloaded shard in a worker process.

    Writes the kept rows to ``part_path(shard, cache)`` and returns the receipt
    ``{id, bytes, sha256, records, kept, kinds, axon_pre, scan_seconds}`` where
    ``kinds`` counts rows per ``KINDS`` and ``axon_pre`` the subset with
    ``pre_class == "AXON"``. Imports fastavro lazily (absent on the laptop).
    """
    from fastavro import reader

    path = Path(path)
    t0 = time.monotonic()
    digest = sha256_path(path)
    idx = shard_index(shard)
    kept, kinds, axon = [], collections.Counter(), collections.Counter()
    records = 0
    with path.open("rb") as f:
        for index, record in enumerate(reader(f)):
            records += 1
            hit = keep_row(record, index, context, idx)
            if hit is None:
                continue
            kind, row = hit
            kept.append(row)
            kinds[kind] += 1
            if row[8] == "AXON":
                axon[kind] += 1
    out = part_path(shard, cache)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out)
    return {"id": shard, "bytes": path.stat().st_size, "sha256": digest, "records": records, "kept": len(kept),
            "kinds": {k: kinds.get(k, 0) for k in KINDS}, "axon_pre": {k: axon.get(k, 0) for k in KINDS},
            "scan_seconds": round(time.monotonic() - t0, 3)}


def kind_totals(entries):
    """Per-kind row totals (all pre classes and the AXON-pre subset) summed over shard receipts.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_kept_partner_graph import kind_totals
        >>> kind_totals([{"kinds": {"kept_to_kept": 1}, "axon_pre": {"kept_to_kept": 1}},
        ...              {"kinds": {"kept_to_kept": 2, "same_cell": 3}, "axon_pre": {"same_cell": 1}}])
        {'rows': {'kept_to_kept': 3, 'same_cell': 3, 'kept_to_tabulated': 0, 'tabulated_to_kept': 0}, 'axon_pre_rows': {'kept_to_kept': 1, 'same_cell': 1, 'kept_to_tabulated': 0, 'tabulated_to_kept': 0}}
    """
    return {"rows": {k: sum(e.get("kinds", {}).get(k, 0) for e in entries) for k in KINDS},
            "axon_pre_rows": {k: sum(e.get("axon_pre", {}).get(k, 0) for e in entries) for k in KINDS}}


def new_progress(listing, workers, inputs):
    """Fresh progress document for this scan."""
    return {"scope": "H01 C3 synapse export (166 Avro shards) reduced to rows between a kept proofread cell "
                     "(base-segment ownership or kept C3 id) and a tabulated C3 neuron, or between two kept "
                     "cells; shards deleted after scanning",
            "source_prefix": BASE_URL, "input_hashes": inputs, "columns": list(COLUMNS), "kinds": list(KINDS),
            "units": {"type": "1 inhibitory, 2 excitatory", "xyz": "8/8/33 nm voxels",
                      "shard": "integer N of export{N:012d}", "index": "record index within the shard",
                      "confidence": "rounded to 4 decimals", "owner": "kept released id"},
            "kept_released_ids": list(KEPT_RELEASED), "workers": workers, "shards_total": len(listing),
            "shards": [], "totals": {},
            "started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_utc": None, "stopped_reason": None, "wall_seconds": 0.0}


def load_context(mapping=MAPPING, membership=MEMBERSHIP, table=TABLE):
    """Keep context for the workers: kept base and C3 ownership plus the tabulated neuron ids."""
    audit_rows = json.loads(Path(mapping).read_text(encoding="utf-8"))["rows"]
    base_owner, c3_owner = load_kept_owners(audit_rows, json.loads(Path(membership).read_text(encoding="utf-8")))
    return {"base_owner": base_owner, "c3_owner": c3_owner, "neuron_ids": load_neuron_ids(table)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=6, help="scan processes")
    ap.add_argument("--downloads", type=int, default=3, help="download threads")
    ap.add_argument("--limit", type=int, default=None, help="scan at most this many pending shards (measurement)")
    args = ap.parse_args(argv)
    listing = json.loads(LISTING.read_text(encoding="utf-8"))["shards"]
    if args.limit is not None:
        listing = dict(list(sorted(listing.items()))[:args.limit])
    context = load_context()
    inputs = {str(p.relative_to(ROOT)): sha256_path(p) for p in (LISTING, TABLE, MAPPING, MEMBERSHIP)}
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else new_progress(
        listing, args.workers, inputs)
    progress["workers"] = args.workers
    progress["shards_total"] = len(listing)
    progress["neuron_ids"] = len(context["neuron_ids"])
    progress["kept_base_segments"] = len(context["base_owner"])
    progress["kept_c3_ids"] = {str(k): v for k, v in sorted(context["c3_owner"].items())}
    complete = run(args.workers, args.downloads, listing, context, progress, cache=CACHE, scan=scan_shard,
                   progress_path=PROGRESS)
    progress["totals"] = {**totals(progress["shards"]), **kind_totals(progress["shards"])}
    write_json(PROGRESS, progress)
    print("STOP:", progress["stopped_reason"], "wall", progress["wall_seconds"], flush=True)
    if not complete:
        return 1
    t0 = time.monotonic()
    progress["output"] = merge_parts(sorted(listing), GRAPH, CACHE, columns=COLUMNS)
    progress["output"]["merge_seconds"] = round(time.monotonic() - t0, 3)
    progress["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(PROGRESS, progress)
    print("MERGED:", json.dumps(progress["output"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
