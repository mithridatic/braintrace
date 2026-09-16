"""Export C3 release skeletons as an archive in the proofread-104 SWC layout (C3-B).

Runs on the box with ``/workspace/venv-c3`` (cloud-volume, numpy). For every C3
segment id: fetch the released skeleton (vertices in nm, radius in nm, edges),
read the subcompartment label at every vertex (64 x 64 x 66 nm label volume,
downloaded chunk by chunk in parallel; one 64^3 chunk per HTTP read instead of
one read per point), split the skeleton into connected components, drop
singleton components (the importer rejects them), root every component at its
soma-labelled vertex of largest radius (else its largest-radius vertex) and
write the rows the proofread importer expects::

    id type x y z radius parent

with ``x, y, z`` in 32 / 32 / 33 nm skeleton voxels, ``radius`` in nm, ``type``
= subcompartment label - 100 (0 unlabelled -> -1), ``parent`` -1 at the root.
Members are ``{c3_id}.{k}.swc`` with ``k`` in descending component size. The
archive's sha256 and a per-cell provenance record (non-proofread) are written
beside it. Nothing here is proofread; the import audit and the gate filter it.
"""

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

C3_VOLUME = "precomputed://https://storage.googleapis.com/h01-release/data/20210601/c3"
SUBCOMPARTMENTS = C3_VOLUME + "/subcompartments"
SWC_VOXEL_NM = np.array([32.0, 32.0, 33.0])
SOMA_LABEL = 103
SOURCE = "h01-release 20210601 c3 skeletons + subcompartments; non-proofread"
ROOT_POLICY = "soma-labelled vertex of largest radius, else largest radius; BFS tree"
ZIP_DATE = (2026, 9, 16, 0, 0, 0)
EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]


@dataclass(frozen=True)
class Skeleton:
    """A released skeleton: ``vertices`` (N, 3) nm, ``edges`` (M, 2), ``radius`` (N,) nm."""

    vertices: np.ndarray
    edges: np.ndarray
    radius: np.ndarray


def label_to_type(label):
    """SWC type code for one subcompartment label: ``label - 100``; 0 (unlabelled) -> -1."""
    label = int(label)
    return -1 if label == 0 else label - 100


def split_components(skeleton):
    """Connected components as arrays of vertex indices, largest first (ties by first vertex)."""
    n = len(skeleton.vertices)
    adjacency = [[] for _ in range(n)]
    for a, b in np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2):
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        members, pending = [start], deque([start])
        while pending:
            i = pending.popleft()
            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True
                    members.append(j)
                    pending.append(j)
        components.append(np.asarray(sorted(members), dtype=np.int64))
    return sorted(components, key=lambda c: (-len(c), int(c[0])))


def choose_root(labels, radius):
    """Index of the soma-labelled vertex with the largest radius, else the largest radius."""
    labels, radius = np.asarray(labels), np.asarray(radius, dtype=float)
    soma = np.flatnonzero(labels == SOMA_LABEL)
    if len(soma):
        return int(soma[np.argmax(radius[soma])])
    return int(np.argmax(radius))


def component_tree(skeleton, members, labels):
    """BFS tree of one component: (vertex order, parent position per order slot, edges dropped).

    Parameters
    ----------
    skeleton : Skeleton
        The whole skeleton.
    members : array_like
        Vertex indices of the component.
    labels : array_like
        Subcompartment label per skeleton vertex.

    Returns
    -------
    order : numpy.ndarray
        Component vertex indices in BFS order from the root.
    parents : list of int
        Position in ``order`` of each vertex's parent; -1 for the root.
    dropped : int
        Component edges not in the tree (cycles in the released skeleton).
    """
    members = np.asarray(members, dtype=np.int64)
    member_set = set(members.tolist())
    adjacency = {int(i): [] for i in members}
    n_edges = 0
    for a, b in np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2):
        if int(a) in member_set and int(b) in member_set:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
            n_edges += 1
    root = int(members[choose_root(np.asarray(labels)[members], np.asarray(skeleton.radius)[members])])
    position = {root: 0}
    order, parents, pending = [root], [-1], deque([root])
    while pending:
        i = pending.popleft()
        for j in adjacency[i]:
            if j not in position:
                position[j] = len(order)
                order.append(j)
                parents.append(position[i])
                pending.append(j)
    if len(order) != len(members):
        raise ValueError("component members are not connected")
    return np.asarray(order, dtype=np.int64), parents, n_edges - (len(order) - 1)


def swc_text(skeleton, labels, order, parents):
    """Proofread-layout rows for one rooted component (ids 0.., voxel positions, nm radius)."""
    voxels = np.asarray(skeleton.vertices, dtype=float)[order] / SWC_VOXEL_NM
    radius = np.asarray(skeleton.radius, dtype=float)[order]
    codes = [label_to_type(l) for l in np.asarray(labels)[order]]
    lines = [f"{i} {codes[i]} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {radius[i]:.6f} {parents[i]}"
             for i, v in enumerate(voxels)]
    return "\n".join(lines) + "\n"


def cable_um(vertices_nm, order, parents):
    """Summed parent-child distance of a rooted component, in micrometers."""
    points = np.asarray(vertices_nm, dtype=float)[order]
    child = np.asarray([i for i, p in enumerate(parents) if p != -1], dtype=np.int64)
    parent = np.asarray([p for p in parents if p != -1], dtype=np.int64)
    if not len(child):
        return 0.0
    return float(np.linalg.norm(points[child] - points[parent], axis=1).sum() / 1000.0)


def sample_labels(vertices_nm, resolution_nm, chunk_size, download, workers=32):
    """Label at every vertex, downloading each touched chunk once.

    Parameters
    ----------
    vertices_nm : array_like
        (N, 3) positions in nm.
    resolution_nm : array_like
        Voxel size of the label volume.
    chunk_size : array_like
        Chunk shape of the label volume in voxels.
    download : callable
        ``download(lo, hi) -> ndarray`` of shape ``hi - lo`` for a voxel box.
    workers : int
        Concurrent chunk downloads.

    Returns
    -------
    numpy.ndarray
        Integer label per vertex.
    """
    points = (np.asarray(vertices_nm, dtype=float) / np.asarray(resolution_nm, dtype=float)).astype(np.int64)
    size = np.asarray(chunk_size, dtype=np.int64)
    chunk_of = points // size
    chunks, inverse = np.unique(chunk_of, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    labels = np.zeros(len(points), dtype=np.int64)

    def fetch(k):
        lo = chunks[k] * size
        return k, np.asarray(download(lo, lo + size))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for k, block in pool.map(fetch, range(len(chunks))):
            idx = np.flatnonzero(inverse == k)
            local = points[idx] - chunks[k] * size
            labels[idx] = block[local[:, 0], local[:, 1], local[:, 2]]
    return labels


class CloudFetcher:
    """Skeletons and subcompartment labels from the released C3 precomputed volumes.

    Parameters
    ----------
    mip : int
        Label-volume mip (0 = 64 x 64 x 66 nm).
    workers : int
        Concurrent chunk downloads.
    """

    def __init__(self, mip=0, workers=32):
        self.mip, self.workers = int(mip), int(workers)
        self._skeletons = self._labels = None

    def _open(self):
        from cloudvolume import CloudVolume
        if self._skeletons is None:
            self._skeletons = CloudVolume(C3_VOLUME, mip=0, use_https=True, cache=False)
            self._labels = CloudVolume(SUBCOMPARTMENTS, mip=self.mip, use_https=True, cache=False,
                                       progress=False, fill_missing=True)

    def skeleton(self, c3_id):
        """Fetch one released skeleton (vertices and radius in nm)."""
        self._open()
        got = self._skeletons.skeleton.get(int(c3_id))
        return Skeleton(np.asarray(got.vertices, dtype=float), np.asarray(got.edges, dtype=np.int64),
                        np.asarray(got.radius, dtype=float))

    @property
    def resolution_nm(self):
        """Voxel size of the label volume at the chosen mip."""
        self._open()
        return [float(v) for v in self._labels.resolution]

    def labels(self, vertices_nm):
        """Subcompartment label at every vertex."""
        from cloudvolume import Bbox
        self._open()
        volume = self._labels

        def download(lo, hi):
            return np.asarray(volume.download(Bbox(lo, hi), mip=self.mip))[..., 0]

        return sample_labels(vertices_nm, volume.resolution, volume.chunk_size, download, self.workers)


def export_cell(fetcher, c3_id):
    """Members ``{name: bytes}`` and the provenance record for one C3 id."""
    started = time.perf_counter()
    skeleton = fetcher.skeleton(c3_id)
    labels = fetcher.labels(skeleton.vertices)
    members, parts, written, singletons, dropped_edges, cable = {}, [], 0, 0, 0, 0.0
    for component in split_components(skeleton):
        if len(component) < 2:
            singletons += 1
            continue
        order, parents, dropped = component_tree(skeleton, component, labels)
        length = cable_um(skeleton.vertices, order, parents)
        name = f"{c3_id}.{len(parts)}.swc"
        members[name] = swc_text(skeleton, labels, order, parents).encode("ascii")
        parts.append({"component": len(parts), "vertices": int(len(order)), "cable_um": length,
                      "soma_vertices": int(np.count_nonzero(labels[order] == SOMA_LABEL)),
                      "root_label": int(labels[order[0]]), "root_radius_nm": float(skeleton.radius[order[0]]),
                      "edges_dropped": int(dropped)})
        written += len(order)
        dropped_edges += dropped
        cable += length
    record = {"c3_id": str(c3_id), "components": len(parts), "components_total": len(parts) + singletons,
              "singletons_dropped": singletons, "vertices": int(len(skeleton.vertices)),
              "vertices_written": written, "edges": int(len(skeleton.edges)), "edges_dropped": dropped_edges,
              "label_counts": {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())},
              "cable_um": cable, "parts": parts, "source": SOURCE,
              "seconds": time.perf_counter() - started}
    return members, record


def candidate_ids(document):
    """C3 ids from a candidates document (a list, or ``candidates``/``cells``/``ids`` inside a dict)."""
    items = document
    if isinstance(document, dict):
        for key in ("candidates", "cells", "ids"):
            if key in document:
                items = document[key]
                break
        else:
            raise ValueError("candidates document has no candidates/cells/ids list")
    ids = []
    for item in items:
        if isinstance(item, dict):
            for key in ("c3_id", "neuron_id", "segment_id", "id"):
                if key in item:
                    item = item[key]
                    break
            else:
                raise ValueError(f"candidate without a c3 id: {item}")
        text = str(item)
        if text not in ids:
            ids.append(text)
    return ids


def write_archive(path, members):
    """Deterministic zip (sorted names, fixed timestamps); returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(zipfile.ZipInfo(name, ZIP_DATE), members[name], compress_type=zipfile.ZIP_DEFLATED)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_cell(stage, fetcher, c3_id, log=print):
    """Export one cell into ``stage`` unless its record is already there; returns the record."""
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    marker = stage / f"{c3_id}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    members, record = export_cell(fetcher, c3_id)
    for name, data in members.items():
        (stage / name).write_bytes(data)
    marker.write_text(json.dumps(record, indent=1), encoding="utf-8")
    log(f"{c3_id}: {record['components']} components, {record['vertices_written']} vertices, "
        f"{record['cable_um']:.0f} um, {record['seconds']:.0f} s")
    return record


def staged_members(stage, records):
    """Read the staged SWC members of every record back as ``{name: bytes}``."""
    members = {}
    for record in records:
        for part in record["parts"]:
            name = f"{record['c3_id']}.{part['component']}.swc"
            members[name] = (Path(stage) / name).read_bytes()
    return members


def run(ids, output, provenance, fetcher, stage=None, log=print):
    """Export ``ids`` to ``output`` (zip) and ``provenance`` (json); returns the provenance."""
    output, provenance = Path(output), Path(provenance)
    stage = Path(stage) if stage else output.with_name(output.stem + ".stage")
    started = time.perf_counter()
    records = [stage_cell(stage, fetcher, c3_id, log) for c3_id in ids]
    digest = write_archive(output, staged_members(stage, records))
    document = {"archive": output.name, "archive_sha256": digest, "source": SOURCE,
                "volume": C3_VOLUME, "subcompartments": SUBCOMPARTMENTS,
                "label_resolution_nm": fetcher.resolution_nm, "label_mip": getattr(fetcher, "mip", 0),
                "swc_layout": {"position": "32/32/33 nm skeleton voxels (nm / [32, 32, 33])", "radius": "nm",
                               "type": "subcompartment label - 100; 0 (unlabelled) -> -1", "root_parent": -1,
                               "member": "{c3_id}.{k}.swc, k in descending component size"},
                "root_policy": ROOT_POLICY, "singleton_components": "dropped (importer rejects them)",
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - started,
                "cells": records}
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(document, indent=1), encoding="utf-8")
    log(f"{output} sha256 {digest}; {len(records)} cells; {document['seconds']:.0f} s")
    return document


def main(argv=None, fetcher=None):
    """Command line entry; ``fetcher`` is injectable for tests."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, help="candidates JSON; its c3 ids are exported")
    parser.add_argument("--ids", nargs="*", default=[], help="explicit C3 ids (added to --candidates)")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/h01/c3-candidates-20260916.zip")
    parser.add_argument("--provenance", type=Path, default=EVIDENCE / "h01-c3-candidates-archive.json")
    parser.add_argument("--stage", type=Path, help="per-cell staging directory (default: beside --output)")
    parser.add_argument("--mip", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    ids = candidate_ids(json.loads(args.candidates.read_text(encoding="utf-8"))) if args.candidates else []
    ids += [i for i in map(str, args.ids) if i not in ids]
    if not ids:
        parser.error("no C3 ids: give --candidates and/or --ids")
    fetcher = fetcher or CloudFetcher(mip=args.mip, workers=args.workers)
    run(ids, args.output, args.provenance, fetcher, args.stage)


if __name__ == "__main__":
    main()
