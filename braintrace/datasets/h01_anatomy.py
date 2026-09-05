"""Map H01 source annotations onto reusable BrainCell regions and locations."""

from dataclasses import dataclass
import hashlib

import brainunit as u
import numpy as np
from braincell.filter import LocsetExpr, LocsetMask, RegionExpr, RegionMask
from braincell import Morphology

from ._h01_swc import POSITION_UM

LABELS = {-1: "unclassified", 0: "axon", 1: "dendrite", 2: "astrocyte",
          3: "soma", 4: "cilium", 5: "axon_initial_segment",
          1000: "myelinated_axon", 1001: "myelinated_axon",
          1002: "myelinated_axon_internal_fragment", 1003: "myelinated_axon_internal_fragment",
          1004: "myelinated_axon", 1005: "myelinated_axon"}


def _label(code):
    return LABELS.get(int(code), f"unknown_{int(code)}")


def _geometry_signature(morphology):
    if not isinstance(morphology, Morphology):
        return None
    digest = hashlib.sha256()
    for view in morphology.branches:
        b = view.branch
        digest.update(str((view.index, b.type, b.n_segments)).encode())
        for array in (b.lengths, b.radii_proximal, b.radii_distal, b.points_proximal, b.points_distal):
            if array is None:
                digest.update(b"None")
            else:
                values = np.asarray(array.to_decimal(u.um), dtype="<f8")
                digest.update(values.tobytes())
    for edge in morphology.edges:
        digest.update(str((edge.parent.index, edge.child.index, edge.parent_x, edge.child_x)).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class _Locations(LocsetExpr):
    signature: str
    points: tuple

    def evaluate(self, morpho, cache=None):
        if _geometry_signature(morpho) != self.signature:
            raise ValueError("H01 locations belong to a different morphology.")
        return LocsetMask(self.points, tuple(f"h01_{b}_{x:.9g}" for b, x in self.points))


@dataclass(frozen=True)
class _Region(RegionExpr):
    signature: str
    intervals: tuple
    policy: str

    def evaluate(self, morpho, cache=None):
        if _geometry_signature(morpho) != self.signature:
            raise ValueError("H01 regions belong to a different morphology.")
        return RegionMask(self.intervals)


@dataclass(frozen=True)
class H01Projection:
    """A spatial projection with explicit rejection and distance evidence.

    Attributes
    ----------
    position_um : tuple of float
        Original query position in micrometers.
    distance_um : float
        Distance to the closest cable centerline, even when rejected.
    status : str
        ``projected``, ``too_far``, or ``ambiguous``.
    location : LocsetExpr or None
        BrainCell placement selection, supplied only for accepted projections.
    """

    position_um: tuple
    distance_um: float
    status: str
    location: object


class H01Anatomy:
    """Expose source samples as BrainCell selections without relabelling geometry.

    Parameters
    ----------
    imported : H01Component
        Imported connected component. Rebuild this adapter after changing geometry.

    Notes
    -----
    Original endpoint coordinates and source edges must match the morphology.
    Coincident source samples are rejected because coordinate-only identity is
    ambiguous. Selections are bound to this morphology, not other loaded cells.
    """

    def __init__(self, imported):
        self.imported = imported
        self.morphology = imported.morphology
        self._signature = _geometry_signature(self.morphology)
        rows = imported.source_rows
        xyz = rows[:, 2:5] * POSITION_UM
        keys = [tuple(np.round(p, 6)) for p in xyz]
        if len(set(keys)) != len(keys):
            raise ValueError("Coincident source points make annotation identity ambiguous.")
        lookup = dict(zip(keys, range(len(rows))))
        self._labels = np.array([_label(c) for c in rows[:, 1]])
        self._nodes = {}
        starts, ends, branch_ids, fractions, endpoints = [], [], [], [], []
        expected_edges = {frozenset((int(r[0]), int(r[6]))) for r in rows if r[6] != -1}
        actual_edges = []
        for view in self.morphology.branches:
            branch = view.branch
            lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
            if lengths.sum() <= 0:
                raise ValueError("Annotation mapping requires positive cable length.")
            bounds = np.r_[0., np.cumsum(lengths)] / lengths.sum()
            proximal = np.asarray(branch.points_proximal.to_decimal(u.um))
            distal = np.asarray(branch.points_distal.to_decimal(u.um))
            for j, (p, q) in enumerate(zip(proximal, distal)):
                try:
                    a, b = (lookup[tuple(np.round(v, 6))] for v in (p, q))
                except KeyError as exc:
                    raise ValueError("Morphology endpoints do not match H01 source samples.") from exc
                actual_edges.append(frozenset((int(rows[a, 0]), int(rows[b, 0]))))
                self._nodes.setdefault(int(rows[a, 0]), (view.index, float(bounds[j])))
                self._nodes.setdefault(int(rows[b, 0]), (view.index, float(bounds[j + 1])))
                starts.append(p)
                ends.append(q)
                branch_ids.append(view.index)
                fractions.append((bounds[j], bounds[j + 1]))
                endpoints.append((a, b))
        if len(actual_edges) != len(expected_edges) or set(actual_edges) != expected_edges:
            raise ValueError("Morphology edges do not preserve the source tree.")
        self._starts, self._ends = np.asarray(starts), np.asarray(ends)
        self._branches, self._fractions = np.asarray(branch_ids), np.asarray(fractions)
        self._endpoints = np.asarray(endpoints)
        self._vectors = self._ends - self._starts
        self._length2 = np.sum(self._vectors**2, axis=1)

    @property
    def label_counts(self):
        """Return source sample counts, including unclassified and unknown codes.

        Returns
        -------
        dict
            Decoded labels and counts; these are not cable coverage percentages.
        """
        labels, counts = np.unique(self._labels, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))

    @property
    def provenance(self):
        """Return the anatomical annotation policy and source identity.

        Returns
        -------
        dict
            Original source provenance plus annotation verification boundaries.
        """
        return {**self.imported.provenance,
                "annotation_verification": "source codes; per-sample manual review not supplied",
                "label_counts": self.label_counts,
                "soma_selection": "largest-radius source sample labelled soma; ties use smallest source ID"}

    def _check_label(self, label):
        if label not in set(LABELS.values()) | set(self._labels):
            raise ValueError(f"Unknown annotation label: {label!r}")

    def location(self, node_id):
        """Select one original source node for a clamp, probe, or synapse.

        Parameters
        ----------
        node_id : int
            Original SWC node ID, before normalization.

        Returns
        -------
        LocsetExpr
            One bound location; absent IDs raise ``KeyError``.
        """
        return _Locations(self._signature, (self._nodes[node_id],))

    def locations(self, label):
        """Select all source samples carrying a named annotation.

        Parameters
        ----------
        label : str
            Decoded name such as ``soma`` or ``axon_initial_segment``.

        Returns
        -------
        LocsetExpr
            Bound locations, possibly empty; no missing labels are inferred.
        """
        self._check_label(label)
        ids = self.imported.source_rows[self._labels == label, 0]
        return _Locations(self._signature, tuple(self._nodes[int(i)] for i in ids))

    def soma_location(self):
        """Select the largest-radius soma-labelled sample, never the arbitrary root.

        Returns
        -------
        LocsetExpr
            One location; raises ``ValueError`` when no soma sample is labelled.
        """
        soma = self.imported.source_rows[self._labels == "soma"]
        if not len(soma):
            raise ValueError("This component has no soma-labelled source sample.")
        row = sorted(soma, key=lambda r: (-r[5], r[0]))[0]
        return self.location(int(row[0]))

    def cable_neighborhood(self, node_id, *, radius_um):
        """Select a geodesic cable neighborhood as an explicit model assumption.

        Parameters
        ----------
        node_id : int
            Original source sample at the center of the neighborhood.
        radius_um : float
            Positive distance along the cable, in micrometers.

        Returns
        -------
        RegionExpr
            Geometry-bound intervals. This is not an anatomical classification;
            it may include unclassified or differently labelled samples.

        Notes
        -----
        Distance follows the connected source tree, not straight-line spatial
        proximity. Branches close in space do not become connected by selection.
        """
        if not np.isfinite(radius_um) or radius_um <= 0:
            raise ValueError("radius_um must be positive and finite.")
        rows = self.imported.source_rows
        ids = {int(row[0]): i for i, row in enumerate(rows)}
        anchor = ids[node_id]
        adjacency = [[] for _ in rows]
        lengths = np.sqrt(self._length2)
        for (a, b), length in zip(self._endpoints, lengths):
            adjacency[a].append((b, length))
            adjacency[b].append((a, length))
        distance = np.full(len(rows), np.inf)
        distance[anchor] = 0.
        stack = [anchor]
        while stack:
            a = stack.pop()
            for b, length in adjacency[a]:
                if np.isinf(distance[b]):
                    distance[b] = distance[a] + length
                    stack.append(b)
        intervals = []
        for (a, b), length, branch, (lo, hi) in zip(self._endpoints, lengths, self._branches, self._fractions):
            if distance[a] < distance[b]:
                left, right = 0., np.clip((radius_um-distance[a])/length, 0., 1.)
            else:
                left, right = 1-np.clip((radius_um-distance[b])/length, 0., 1.), 1.
            if right > left:
                intervals.append((int(branch), float(lo+left*(hi-lo)), float(lo+right*(hi-lo))))
        return _Region(self._signature, tuple(intervals),
                       f"inferred_geodesic_neighborhood:node={node_id},radius_um={radius_um}")

    def region(self, label, *, policy="strict"):
        """Select cable intervals using an explicit annotation boundary policy.

        Parameters
        ----------
        label : str
            Decoded source annotation name.
        policy : str, optional
            ``strict`` requires both segment endpoints to carry the label.
            ``sample_neighborhood`` extends each label halfway along incident
            segments. The latter is an inferred boundary, not measured anatomy.

        Returns
        -------
        RegionExpr
            Bound selection for ``cell.paint``. Unclassified samples stay so.
        """
        self._check_label(label)
        if policy not in ("strict", "sample_neighborhood"):
            raise ValueError("policy must be strict or sample_neighborhood.")
        intervals = []
        for branch, (lo, hi), (a, b) in zip(self._branches, self._fractions, self._endpoints):
            left, right = self._labels[a] == label, self._labels[b] == label
            if left and right:
                intervals.append((int(branch), float(lo), float(hi)))
            elif policy == "sample_neighborhood" and (left or right):
                mid = (lo + hi) / 2
                intervals.append((int(branch), float(lo if left else mid), float(mid if left else hi)))
        return _Region(self._signature, tuple(intervals), policy)

    def project(self, positions_um, *, max_distance_um):
        """Project physical points onto cable segments with explicit rejection.

        Parameters
        ----------
        positions_um : array-like, shape (n, 3)
            Absolute positions in micrometers, not voxel coordinates.
        max_distance_um : float
            Maximum centerline distance; distant components remain unmatched.

        Returns
        -------
        tuple of H01Projection
            One result per query, retaining rejected rows. Equidistant distinct
            cable sites are ambiguous; incident edges sharing a source node
            resolve to that node's canonical location.

        Notes
        -----
        This is geometric projection, not proof that a synapse belongs to this
        connected component. All queries must first be filtered by source cell ID.
        """
        positions = np.asarray(positions_um, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("positions_um must be a finite (n, 3) array.")
        if not np.isfinite(max_distance_um) or max_distance_um < 0:
            raise ValueError("max_distance_um must be finite and nonnegative.")
        result = []
        for point in positions:
            t = np.clip(np.sum((point - self._starts) * self._vectors, axis=1) / self._length2, 0., 1.)
            distances = np.linalg.norm(self._starts + t[:, None] * self._vectors - point, axis=1)
            best = int(np.argmin(distances))
            distance = float(distances[best])
            location, status = None, "too_far"
            if distance <= max_distance_um:
                candidates = np.flatnonzero(np.abs(distances - distance) <= 1e-9)
                sites = set()
                for candidate in candidates:
                    fraction = t[candidate]
                    if fraction <= 1e-10 or fraction >= 1 - 1e-10:
                        node = self._endpoints[candidate, int(fraction >= .5)]
                        sites.add(self._nodes[int(self.imported.source_rows[node, 0])])
                    else:
                        lo, hi = self._fractions[candidate]
                        sites.add((int(self._branches[candidate]), float(lo + fraction * (hi - lo))))
                status = "projected" if len(sites) == 1 else "ambiguous"
                if status == "projected":
                    location = _Locations(self._signature, tuple(sites))
            result.append(H01Projection(tuple(point), distance, status, location))
        return tuple(result)

    def project_synapses(self, synapses, *, max_distance_um):
        """Project CSV endpoints after checking every row's cell identity.

        Parameters
        ----------
        synapses : sequence of H01Synapse
            Rows returned by ``H01Annotations.synapses`` for this cell.
        max_distance_um : float
            Maximum accepted distance to the cable centerline.

        Returns
        -------
        tuple of H01Projection
            One result per input row, in identical order, including rejections.
        """
        if any(s.neuron_id != self.imported.neuron_id for s in synapses):
            raise ValueError("Synapse rows belong to a different H01 cell.")
        positions = np.asarray([s.position_um for s in synapses]).reshape(-1, 3)
        return self.project(positions, max_distance_um=max_distance_um)
