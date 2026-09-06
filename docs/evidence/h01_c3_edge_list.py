"""Directed contacts among the proofread-104 cells through the C3 relationship index.

Run with the isolated h01-inspect Python. Read-only: no model edge is created.

Route: sample skeleton nodes of every proofread cell, read the C3 segmentation label
at each node, query the C3 synapse index by those labels in both directions, and keep
annotations whose presynaptic and postsynaptic cells are two different proofread
cells. Surviving contacts are re-checked at both endpoints against the proofread
volume. Labels not touched by a sampled node are missed, so the result is a lower
bound on contacts, never proof of absence.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

from docs.evidence.h01_index_connectivity_audit import ANNOTATIONS, VOLUME, decode_related

C3_VOLUME = "https://storage.googleapis.com/h01-release/data/20210601/c3"
SWC_NM = (32., 32., 33.)
C3_NM = (8., 8., 33.)


def parse_swc(text):
    """Node positions in SWC voxel units from one released component file."""
    points = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError("SWC row needs seven columns.")
        points.append((float(parts[2]), float(parts[3]), float(parts[4])))
    return points


def to_c3_voxels(points):
    """Convert SWC voxel positions to C3 voxel indices."""
    return [tuple(int(round(v*s/c)) for v, s, c in zip(p, SWC_NM, C3_NM)) for p in points]


def sample(points, limit):
    """Evenly strided subset with at most ``limit`` points, keeping the first and last."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(points) <= limit:
        return list(points)
    stride = -(-len(points)//limit)
    chosen = points[::stride]
    if points[-1] != chosen[-1]:
        chosen.append(points[-1])
    return chosen[:limit] if len(chosen) > limit else chosen


def archive_cells(path):
    """Neuron id -> ordered component file names in the released archive."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    cells = collections.defaultdict(list)
    for name in names:
        match = re.fullmatch(r"(\d+)\.(\d+)\.swc", name)
        if match:
            cells[match.group(1)].append((int(match.group(2)), name))
    return {k: [n for _, n in sorted(v)] for k, v in cells.items()}


def assemble_edges(pre_index, post_index):
    """Contacts whose presynaptic and postsynaptic cells differ.

    Parameters
    ----------
    pre_index, post_index : dict
        annotation id -> set of cell ids found through that relationship.

    Returns
    -------
    list of tuple
        ``(annotation_id, pre_cell, post_cell)`` for unambiguous different-cell pairs.
    """
    edges = []
    for annotation in sorted(set(pre_index) & set(post_index)):
        pres, posts = pre_index[annotation], post_index[annotation]
        if len(pres) != 1 or len(posts) != 1:
            continue
        pre, post = next(iter(pres)), next(iter(posts))
        if pre != post:
            edges.append((annotation, pre, post))
    return edges


def checkpoint_key(limit, archive_path):
    return {"limit": limit, "archive": Path(archive_path).name}


def load_checkpoint(path, limit, archive_path):
    """Per-cell records saved by an earlier run with the same limit and archive, else empty."""
    if not Path(path).exists():
        return {}
    record = json.loads(Path(path).read_text())
    if record.get("key") != checkpoint_key(limit, archive_path):
        return {}
    return record["cells"]


def save_checkpoint(path, per_cell, limit, archive_path):
    Path(path).write_text(json.dumps({"key": checkpoint_key(limit, archive_path), "cells": per_cell}))


def neighbourhood(voxel, radius_xy=2, radius_z=1):
    """Voxels within a small box around ``voxel``, nearest first."""
    out = []
    for dz in range(-radius_z, radius_z+1):
        for dy in range(-radius_xy, radius_xy+1):
            for dx in range(-radius_xy, radius_xy+1):
                out.append((voxel[0]+dx, voxel[1]+dy, voxel[2]+dz))
    return sorted(out, key=lambda p: sum((a-b)**2 for a, b in zip(p, voxel)))


def nearest_label(labels, voxel, expected):
    """Offset (voxels) of the nearest sampled voxel carrying ``expected``, or None."""
    for point in neighbourhood(voxel):
        if str(int(labels[point])) == expected:
            return [point[i]-voxel[i] for i in range(3)]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--archive", default="proofread104.zip")
    parser.add_argument("--cells", nargs="*", help="Restrict to these neuron ids (pilot).")
    parser.add_argument("--limit", type=int, default=3000, help="Sampled nodes per cell.")
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/h01-resolved-edge-list.json"))
    args = parser.parse_args(argv)
    os.environ.setdefault("CLOUD_FILES_DIR", str(args.cache.resolve()/"cloudfiles"))
    from cloudvolume import CloudVolume
    from cloudvolume.cacheservice import CacheService
    from cloudvolume.datasource.precomputed.sharding import ShardReader, ShardingSpecification

    started = time.time()
    archive_path = args.cache/args.archive
    cells = archive_cells(archive_path)
    selected = {k: v for k, v in cells.items() if not args.cells or k in args.cells}
    c3 = CloudVolume("precomputed://"+C3_VOLUME, mip=0, progress=False, fill_missing=True, cache=False)
    proofread = CloudVolume("precomputed://"+VOLUME, mip=0, progress=False, fill_missing=True, cache=False)
    checkpoint = args.output.with_suffix(".cells.json")
    per_cell = load_checkpoint(checkpoint, args.limit, archive_path)
    label_to_cell = collections.defaultdict(set)
    for cell, entry in per_cell.items():
        for i in entry["c3_ids"]:
            label_to_cell[int(i)].add(cell)
    with zipfile.ZipFile(archive_path) as archive:
        for index, (cell, files) in enumerate(sorted(selected.items())):
            if cell in per_cell:
                continue
            t0 = time.time()
            points = []
            for name in files:
                points += parse_swc(archive.read(name).decode())
            voxels = sample(to_c3_voxels(points), args.limit)
            labels = c3.scattered_points(voxels)
            own = proofread.scattered_points(voxels)
            c3_ids = sorted({int(labels[v]) for v in voxels} - {0})
            own_labels = collections.Counter(str(int(own[v])) for v in voxels)
            per_cell[cell] = {"components": len(files), "nodes": len(points), "sampled": len(voxels),
                              "c3_ids": [str(i) for i in c3_ids],
                              "proofread_labels_at_samples": dict(own_labels),
                              "identity_consistent": set(own_labels) <= {cell, "0"},
                              "seconds": round(time.time()-t0, 1)}
            for i in c3_ids:
                label_to_cell[i].add(cell)
            save_checkpoint(checkpoint, per_cell, args.limit, archive_path)
            print("cell", index+1, cell, len(c3_ids), "c3 ids", per_cell[cell]["seconds"], "s", flush=True)
    shared = {str(k): sorted(v) for k, v in label_to_cell.items() if len(v) > 1}
    info = json.loads((args.cache/"c3-annotation-info.json").read_text())
    cache = CacheService(ANNOTATIONS, False, c3.config, meta=SimpleNamespace(cloudpath=ANNOTATIONS))
    indexes = {}
    records = {}
    for direction in ("pre_synaptic_cell", "post_synaptic_cell"):
        relation = next(x for x in info["relationships"] if x["id"] == direction)
        reader = ShardReader(ANNOTATIONS, cache, ShardingSpecification.from_dict(relation["sharding"]))
        found = collections.defaultdict(set)
        keys = sorted(label_to_cell)
        for offset in range(0, len(keys), 1000):
            payloads = reader.get_data(keys[offset:offset+1000], path=relation["key"])
            for identity, data in payloads.items():
                for record in decode_related(data, identity):
                    records[record["annotation_id"]] = record
                    for cell in label_to_cell[int(identity)]:
                        found[record["annotation_id"]].add(cell)
        indexes[direction] = found
        print(direction, len(found), "annotations", flush=True)
    edges = assemble_edges(indexes["pre_synaptic_cell"], indexes["post_synaptic_cell"])
    verified = []
    for annotation, pre, post in edges:
        record = records[annotation]
        pre_v = tuple(int(v) for v in record["pre_voxel"])
        post_v = tuple(int(v) for v in record["post_voxel"])
        labels = proofread.scattered_points(neighbourhood(pre_v)+neighbourhood(post_v))
        pre_offset, post_offset = nearest_label(labels, pre_v, pre), nearest_label(labels, post_v, post)
        verified.append({"annotation_id": annotation, "pre_cell": pre, "post_cell": post, "type": record["type"],
                         "pre_voxel": list(pre_v), "post_voxel": list(post_v),
                         "pre_proofread_label": str(int(labels[pre_v])), "post_proofread_label": str(int(labels[post_v])),
                         "pre_nearest_offset": pre_offset, "post_nearest_offset": post_offset,
                         "endpoints_verified": pre_offset == [0, 0, 0] and post_offset == [0, 0, 0],
                         "endpoints_within_box": pre_offset is not None and post_offset is not None})
    cells_with_edges = sorted({e["pre_cell"] for e in verified if e["endpoints_verified"]}
                              | {e["post_cell"] for e in verified if e["endpoints_verified"]})
    report = {"scope": "Lower bound on directed contacts among sampled proofread cells; absence is not established.",
              "archive": args.archive, "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              "c3_volume": C3_VOLUME, "proofread_volume": VOLUME, "annotations": ANNOTATIONS,
              "sample_limit_per_cell": args.limit, "cells": per_cell, "c3_ids_shared_by_cells": shared,
              "edges": verified, "cells_with_verified_edges": cells_with_edges,
              "counts": {"cells": len(per_cell), "c3_ids": len(label_to_cell), "edges": len(verified),
                         "verified_edges": sum(e["endpoints_verified"] for e in verified),
                         "edges_within_box": sum(e["endpoints_within_box"] for e in verified),
                         "cells_with_verified_edges": len(cells_with_edges)},
              "wall_seconds": round(time.time()-started, 1)}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["counts"]), "wall", report["wall_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
