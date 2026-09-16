"""Offline H01 node inventory and conservatively selected synaptic contacts.

Two selection routes feed ``prepare_connectivity``: the proofread endpoint audit
(``select_connectivity``, exact endpoints verified against the proofread volume) and the C3
neuron synapse graph (``select_c3_connectivity``, identity by the C3 ``neuron_id`` the
skeletons come from, so no endpoint voxel gate applies). Placement is the recorded
projection distance onto the cable; with ``blocker_distance_um`` a contact is blocked only
beyond that distance and its distance above ``max_distance_um`` is recorded.
"""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .h01 import ARCHIVE_SHA256, H01Archive, _digest
from .h01_annotations import H01Annotations
from ._h01_swc import _parse_swc_bytes
from . import h01_construction  # Ensure fast unit operations and caches are active

_CABLE_CACHE = {}
C3_SELECTION_SCHEMA = "h01-c3-selection-v1"


def _get_cell_cables(archive, cell_id):
    p = Path(archive.path)
    try:
        stat = p.stat()
        mtime = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        mtime = None
    comps = tuple(archive.components(cell_id))
    key = (str(p.resolve()), str(cell_id), comps, mtime)
    cached = _CABLE_CACHE.get(key)
    if cached is not None:
        return cached
    components_data = []
    with zipfile.ZipFile(archive.path) as source:
        for component in comps:
            name = f"{cell_id}.{component}.swc"
            raw = source.read(name)
            rows = _parse_swc_bytes(raw)
            lookup = {int(r[0]): i for i, r in enumerate(rows)}
            children = np.flatnonzero(rows[:, 6] != -1)
            if not len(children):
                continue
            parents = [lookup[int(rows[i, 6])] for i in children]
            start = rows[parents, 2:5] * [.032, .032, .033]
            vector = rows[children, 2:5] * [.032, .032, .033] - start
            length2 = np.sum(vector * vector, axis=1)
            if (length2 <= 0).any():
                raise ValueError("Source cable has a zero-length segment.")
            has_soma = bool((rows[:, 1] == 3).any())
            digest = hashlib.sha256(raw).hexdigest()
            components_data.append((component, start, vector, length2, has_soma, digest))
    if not components_data:
        raise ValueError("Endpoint cell has no cable component.")
    _CABLE_CACHE[key] = components_data
    return components_data


def cell_sign(tags):
    """Assign a Dale sign from the released cell-type tags.

    Parameters
    ----------
    tags : iterable of str
        Unmodified H01 cell tags.

    Returns
    -------
    int
        +1 for excitatory cells, -1 for interneurons.
    """
    tags = set(tags)
    excitatory = bool(tags & {"pyramidal", "excitatory/spiny-with-atypical-tree"})
    inhibitory = "interneuron" in tags
    if excitatory == inhibitory:
        raise ValueError("Cell needs one unambiguous E/I assignment.")
    return 1 if excitatory else -1


def select_connectivity(report, annotations):
    """Retain all cells and only contacts with exact verified endpoints.

    Parameters
    ----------
    report : dict
        Saved population endpoint audit.
    annotations : H01Annotations
        Source cell metadata.

    Returns
    -------
    dict
        Nodes, accepted contacts, and explicit excluded-contact reasons.
    """
    if report["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Endpoint audit uses a different morphology archive.")
    nodes = []
    for index, identity in enumerate(annotations.select()):
        tags = list(annotations.metadata(identity).tags)
        sign = cell_sign(tags)
        nodes.append(dict(index=index, cell_id=identity, tags=tags,
                          polarity="E" if sign == 1 else "I", dale_sign=sign))
    by_id = {n["cell_id"]: n for n in nodes}
    accepted, excluded, seen = [], [], {}
    for source in report["edges"]:
        edge = deepcopy(source)
        identity = edge["annotation_id"]
        if identity in seen:
            if seen[identity] != source:
                raise ValueError("Conflicting duplicate synapse annotation.")
            excluded.append(dict(annotation_id=identity, reason="duplicate annotation"))
            continue
        seen[identity] = deepcopy(source)
        reason = None
        if edge.get("endpoints_verified") is not True:
            reason = "exact endpoints not verified"
        else:
            for side in ("pre", "post"):
                cell = edge[f"{side}_cell"]
                if cell not in by_id:
                    reason = "endpoint cell absent from node inventory"
                elif (edge.get(f"{side}_proofread_label") != cell or
                      edge.get(f"{side}_nearest_offset") != [0, 0, 0] or
                      report["cells"].get(cell, {}).get("identity_consistent") is not True):
                    reason = "endpoint identity evidence is inconsistent"
                position = np.asarray(edge[f"{side}_voxel"], dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all() or (position < 0).any():
                    reason = "invalid endpoint coordinate"
            if reason is None:
                pre, post = by_id[edge["pre_cell"]], by_id[edge["post_cell"]]
                if edge["type"] != (2 if pre["dale_sign"] == 1 else 1):
                    reason = "synapse type conflicts with presynaptic Dale sign"
                else:
                    edge.update(pre_index=pre["index"], post_index=post["index"],
                                dale_sign=pre["dale_sign"], connection_type=pre["polarity"]+post["polarity"])
        if reason:
            excluded.append(dict(annotation_id=identity, reason=reason))
        else:
            accepted.append(edge)
    return dict(nodes=nodes, contacts=accepted, excluded_contacts=excluded,
                archive_sha256=ARCHIVE_SHA256,
                scope="Observed subset only. Missing connections do not establish biological absence.")


def c3_annotation_id(row):
    """Stable identity of one C3 graph row: ``c3-<shard>-<index>`` or a hash of the row.

    Parameters
    ----------
    row : dict
        C3 synapse graph row.

    Returns
    -------
    str
        Annotation identifier.
    """
    if row.get("shard") is not None and row.get("index") is not None:
        return f"c3-{row['shard']}-{row['index']}"
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "c3-"+hashlib.sha256(payload).hexdigest()[:16]


def _c3_nodes(cells):
    nodes = []
    for index, cell in enumerate(cells):
        tags = list(cell.get("tags", []))
        if "polarity" in cell and cell["polarity"] in ("E", "I"):
            sign = 1 if cell["polarity"] == "E" else -1
        else:
            sign = cell_sign(tags)
        nodes.append(dict(index=index, cell_id=str(cell["cell_id"]), tags=tags, polarity="E" if sign == 1 else "I",
                          dale_sign=sign, c3_id=None if cell.get("c3_id") is None else str(cell["c3_id"]),
                          archive_sha256=cell.get("archive_sha256", ARCHIVE_SHA256)))
    if len({n["cell_id"] for n in nodes}) != len(nodes):
        raise ValueError("Duplicate cell identity in the C3 cell list.")
    return nodes


def select_c3_connectivity(graph_rows, cells, c3_to_released):
    """Retain the listed cells and the C3 graph contacts among them.

    Parameters
    ----------
    graph_rows : iterable of dict
        C3 synapse graph rows: ``pre``, ``post`` (C3 neuron ids), ``type`` (1 inhibitory,
        2 excitatory), ``x``, ``y``, ``z`` (8/8/33 nm voxels), ``pre_class``, ``post_class``,
        ``confidence``; optional ``shard`` and ``index`` name the row.
    cells : sequence of dict
        Cells to retain: ``cell_id`` (released identity), ``tags`` (or ``polarity``), optional
        ``c3_id`` and ``archive_sha256`` (default: the proofread archive).
    c3_to_released : mapping
        C3 neuron id to released cell identity.

    Returns
    -------
    dict
        Nodes, accepted contacts and explicit excluded-contact reasons, in the shape
        ``select_connectivity`` produces, tagged ``schema = h01-c3-selection-v1``.
    """
    nodes = _c3_nodes(cells)
    by_id = {n["cell_id"]: n for n in nodes}
    mapping = {str(k): str(v) for k, v in c3_to_released.items()}
    accepted, excluded, seen = [], [], {}
    for source in graph_rows:
        row = deepcopy(source)
        identity = c3_annotation_id(row)
        if identity in seen:
            if seen[identity] != source:
                raise ValueError("Conflicting duplicate C3 graph row.")
            excluded.append(dict(annotation_id=identity, reason="duplicate annotation"))
            continue
        seen[identity] = deepcopy(source)
        pre, post = mapping.get(str(row["pre"])), mapping.get(str(row["post"]))
        position = np.asarray([row.get("x"), row.get("y"), row.get("z")], dtype=float)
        reason = None
        if pre not in by_id or post not in by_id:
            reason = "endpoint cell absent from node inventory"
        elif pre == post:
            reason = "presynaptic and postsynaptic cell are the same"
        elif position.shape != (3,) or not np.isfinite(position).all() or (position < 0).any():
            reason = "invalid endpoint coordinate"
        elif row["type"] != (2 if by_id[pre]["dale_sign"] == 1 else 1):
            reason = "synapse type conflicts with presynaptic Dale sign"
        if reason:
            excluded.append(dict(annotation_id=identity, reason=reason))
            continue
        voxel = [int(v) if float(v).is_integer() else float(v) for v in position]
        accepted.append(dict(annotation_id=identity, pre_cell=pre, post_cell=post, type=row["type"],
                             pre_voxel=list(voxel), post_voxel=list(voxel), endpoints_verified=True,
                             identity_basis="c3-neuron-id", pre_proofread_label=pre, post_proofread_label=post,
                             pre_nearest_offset=[0, 0, 0], post_nearest_offset=[0, 0, 0],
                             c3_pre=str(row["pre"]), c3_post=str(row["post"]),
                             pre_class=row.get("pre_class"), post_class=row.get("post_class"),
                             confidence=row.get("confidence"),
                             pre_index=by_id[pre]["index"], post_index=by_id[post]["index"],
                             dale_sign=by_id[pre]["dale_sign"],
                             connection_type=by_id[pre]["polarity"]+by_id[post]["polarity"]))
    return dict(schema=C3_SELECTION_SCHEMA, nodes=nodes, contacts=accepted, excluded_contacts=excluded,
                archives=sorted({n["archive_sha256"] for n in nodes}),
                scope="C3 neuron synapse graph subset; identity is the C3 neuron_id of the segmentation the "
                      "skeletons come from. Missing connections do not establish biological absence.")


def archive_digest(archive):
    """SHA256 an opened archive is pinned to, hashing the file when it carries none."""
    for name in ("sha256", "expected_sha256", "archive_sha256"):
        value = getattr(archive, name, None)
        if isinstance(value, str) and len(value) == 64:
            return value
    return _digest(Path(archive.path))


def archives_by_digest(archive_set):
    """Map each archive of a set to its digest; accepts a mapping or an iterable of archives."""
    found = archive_set.archives()
    if isinstance(found, dict):
        return {str(k): v for k, v in found.items()}
    return {archive_digest(a): a for a in found}


def _nearest_component(archive, cell_id, point):
    cables = _get_cell_cables(archive, cell_id)
    candidates = []
    for component, start, vector, length2, has_soma, digest in cables:
        fraction = np.clip(np.sum((point - start) * vector, axis=1) / length2, 0., 1.)
        diff = start + fraction[:, None] * vector - point
        distance = float(np.sqrt(np.sum(diff * diff, axis=1).min()))
        candidates.append((distance, component, has_soma, digest))
    candidates.sort()
    best = candidates[0]
    ambiguous = len(candidates) > 1 and abs(candidates[1][0] - best[0]) <= 1e-9
    return dict(distance_um=best[0], component=best[1], has_soma=best[2],
                source_swc_sha256=best[3], ambiguous=ambiguous)


def _source_lookup(cache, result, archive_set):
    """Per-cell archive and loader; the single proofread archive when no set is given."""
    if archive_set is None:
        archive = H01Archive(cache / "proofread104.zip")
        if set(archive.neuron_ids) != {n["cell_id"] for n in result["nodes"]}:
            raise ValueError("Morphology and annotation cell inventories differ.")
        return (lambda cell: archive), (lambda cell, component: archive.load(cell, component=component))
    by_digest = archives_by_digest(archive_set)
    digest_of = {}
    for node in result["nodes"]:
        digest = node.get("archive_sha256", ARCHIVE_SHA256)
        if digest not in by_digest:
            raise ValueError("Node names an archive the set does not hold: "+digest)
        if str(node["cell_id"]) not in {str(i) for i in by_digest[digest].neuron_ids}:
            raise ValueError(f"Cell {node['cell_id']} is absent from its archive {digest}.")
        digest_of[node["cell_id"]] = digest
    return (lambda cell: by_digest[digest_of[cell]]), (lambda cell, component: archive_set.load(cell, component, digest_of[cell]))


def prepare_connectivity(cache, report_path, *, max_distance_um=1., archive_set=None, blocker_distance_um=None):
    """Resolve verified contacts to source cable components without network access.

    Parameters
    ----------
    cache : path-like
        Directory containing proofread104.zip and cell_properties.json.
    report_path : path-like
        Saved population endpoint audit JSON, or a ``select_c3_connectivity`` document.
    max_distance_um : float, optional
        Distance to the cable centerline under which a placement is clean; with no
        ``blocker_distance_um`` it is also the blocker.
    archive_set : H01ArchiveSet, optional
        Archives resolved per node by ``archive_sha256``; required for a C3 selection.
    blocker_distance_um : float, optional
        When given, only a distance beyond it (or a failed projection, or a component without
        a soma) blocks construction; a distance above ``max_distance_um`` is recorded as
        ``placement_distance_um`` with ``within_max_distance`` false.

    Returns
    -------
    dict
        All node identities and verified contacts with construction status.
    """
    if not np.isfinite(max_distance_um) or max_distance_um <= 0:
        raise ValueError("Projection distance must be positive and finite.")
    if blocker_distance_um is not None and (not np.isfinite(blocker_distance_um) or blocker_distance_um < max_distance_um):
        raise ValueError("Blocker distance must be finite and at least the projection distance.")
    cache = Path(cache)
    raw = Path(report_path).read_bytes()
    document = json.loads(raw)
    if document.get("schema") == C3_SELECTION_SCHEMA:
        if archive_set is None:
            raise ValueError("A C3 selection resolves cells through an archive set.")
        result = deepcopy(document)
    else:
        result = select_connectivity(document, H01Annotations(cache))
    archive_for, load = _source_lookup(cache, result, archive_set)
    limit = max_distance_um if blocker_distance_um is None else blocker_distance_um
    result.update(endpoint_audit_sha256=hashlib.sha256(raw).hexdigest(),
                  max_distance_um=max_distance_um, blocker_distance_um=blocker_distance_um)
    for edge in result["contacts"]:
        blockers = []
        for side in ("pre", "post"):
            point = np.asarray(edge[f"{side}_voxel"]) * [.008, .008, .033]
            cell = edge[f"{side}_cell"]
            placement = _nearest_component(archive_for(cell), cell, point)
            placement["position_um"] = point.tolist()
            placement["placement_distance_um"] = placement["distance_um"]
            placement["within_max_distance"] = bool(placement["distance_um"] <= max_distance_um)
            if placement["ambiguous"] or placement["distance_um"] > limit:
                blockers.append(side+": ambiguous component or distance exceeds limit")
            else:
                imported = load(cell, placement["component"])
                projection = imported.anatomy().project([point], max_distance_um=limit)[0]
                if projection.status != "projected":
                    blockers.append(side+": "+projection.status)
                else:
                    placement["cable_location"] = list(projection.location.evaluate(imported.morphology).points[0])
            if not placement["has_soma"]:
                blockers.append(side+": source component has no soma; no cable connection to soma is supplied")
            edge[side+"_placement"] = placement
        edge.update(construction_ready=not blockers, blockers=blockers)
    result["counts"] = dict(nodes=len(result["nodes"]),
        excitatory=sum(n["dale_sign"] == 1 for n in result["nodes"]),
        inhibitory=sum(n["dale_sign"] == -1 for n in result["nodes"]),
        verified_contacts=len(result["contacts"]),
        construction_ready=sum(e["construction_ready"] for e in result["contacts"]),
        beyond_max_distance=sum(not e[s+"_placement"]["within_max_distance"]
                                for e in result["contacts"] for s in ("pre", "post")))
    return result
