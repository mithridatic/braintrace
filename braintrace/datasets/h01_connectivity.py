"""Offline H01 node inventory and conservatively selected synaptic contacts."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .h01 import ARCHIVE_SHA256, H01Archive
from .h01_annotations import H01Annotations
from ._h01_swc import _parse_swc_bytes


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


def _nearest_component(archive, cell_id, point):
    candidates = []
    with zipfile.ZipFile(archive.path) as source:
        for component in archive.components(cell_id):
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
            fraction = np.clip(np.sum((point-start)*vector, axis=1)/length2, 0., 1.)
            distance = float(np.linalg.norm(start+fraction[:, None]*vector-point, axis=1).min())
            candidates.append((distance, component, bool((rows[:, 1] == 3).any()),
                               hashlib.sha256(raw).hexdigest()))
    if not candidates:
        raise ValueError("Endpoint cell has no cable component.")
    candidates.sort()
    best = candidates[0]
    ambiguous = len(candidates) > 1 and abs(candidates[1][0]-best[0]) <= 1e-9
    return dict(distance_um=best[0], component=best[1], has_soma=best[2],
                source_swc_sha256=best[3], ambiguous=ambiguous)


def prepare_connectivity(cache, report_path, *, max_distance_um=1.):
    """Resolve verified contacts to source cable components without network access.

    Parameters
    ----------
    cache : path-like
        Directory containing proofread104.zip and cell_properties.json.
    report_path : path-like
        Saved population endpoint audit JSON.
    max_distance_um : float, optional
        Maximum accepted distance to the cable centerline.

    Returns
    -------
    dict
        All node identities and verified contacts with construction status.
    """
    if not np.isfinite(max_distance_um) or max_distance_um <= 0:
        raise ValueError("Projection distance must be positive and finite.")
    cache = Path(cache)
    raw = Path(report_path).read_bytes()
    result = select_connectivity(json.loads(raw), H01Annotations(cache))
    archive = H01Archive(cache / "proofread104.zip")
    if set(archive.neuron_ids) != {n["cell_id"] for n in result["nodes"]}:
        raise ValueError("Morphology and annotation cell inventories differ.")
    result.update(endpoint_audit_sha256=hashlib.sha256(raw).hexdigest(),
                  max_distance_um=max_distance_um)
    for edge in result["contacts"]:
        blockers = []
        for side in ("pre", "post"):
            point = np.asarray(edge[f"{side}_voxel"]) * [.008, .008, .033]
            placement = _nearest_component(archive, edge[f"{side}_cell"], point)
            placement["position_um"] = point.tolist()
            if placement["ambiguous"] or placement["distance_um"] > max_distance_um:
                blockers.append(side+": ambiguous component or distance exceeds limit")
            else:
                imported = archive.load(edge[f"{side}_cell"], component=placement["component"])
                projection = imported.anatomy().project([point], max_distance_um=max_distance_um)[0]
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
        construction_ready=sum(e["construction_ready"] for e in result["contacts"]))
    return result
