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
    cached = getattr(morphology, "_h01_geom_sig", None)
    if cached is not None:
        if isinstance(cached, str):
            return cached
        cached_sig, n_b, n_e = cached
        if len(morphology._nodes) == n_b:
            return cached_sig
    digest = hashlib.sha256()
    branches = morphology.branches
    indices = {id(view): index for index, view in enumerate(branches)}
    for index, view in enumerate(branches):
        b = view.branch
        digest.update(str((index, b.type, b.n_segments)).encode())
        for array in (b.lengths, b.radii_proximal, b.radii_distal, b.points_proximal, b.points_distal):
            if array is None:
                digest.update(b"None")
            else:
                if (isinstance(array, u.Quantity) and array.unit == u.um
                        and isinstance(array.mantissa, np.ndarray) and array.mantissa.dtype == np.float64):
                    values = array.mantissa
                else:
                    values = np.asarray(array.to_decimal(u.um), dtype="<f8")
                digest.update(values.tobytes())
    for edge in morphology.edges:
        digest.update(str((indices[id(edge.parent)], indices[id(edge.child)], edge.parent_x, edge.child_x)).encode())
    sig = digest.hexdigest()
    try:
        morphology._h01_geom_sig = (sig, len(morphology._nodes), len(morphology._nodes) - 1)
    except (AttributeError, TypeError):
        pass
    return sig


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
        xyz_round = np.round(xyz, 6)
        keys = [tuple(p) for p in xyz_round.tolist()]
        if len(set(keys)) != len(keys):
            raise ValueError("Coincident source points make annotation identity ambiguous.")
        lookup = dict(zip(keys, range(len(rows))))
        self._labels = np.array([_label(c) for c in rows[:, 1]])
        self._nodes = {}
        expected_edges = {(int(r[0]), int(r[6])) if int(r[0]) < int(r[6]) else (int(r[6]), int(r[0]))
                          for r in rows if r[6] != -1}
        actual_edges = []
        row_nodes = rows[:, 0].astype(int)

        branches_views = list(self.morphology.branches)
        total_segments = sum(
            len(v.branch.lengths.mantissa if isinstance(v.branch.lengths, u.Quantity) else v.branch.lengths)
            for v in branches_views
        )
        starts = np.empty((total_segments, 3), dtype=np.float64)
        ends = np.empty((total_segments, 3), dtype=np.float64)
        branch_ids = np.empty(total_segments, dtype=np.int32)
        fractions = np.empty((total_segments, 2), dtype=np.float64)
        endpoints = np.empty((total_segments, 2), dtype=np.int32)

        seg_offset = 0
        for branch_index, view in enumerate(branches_views):
            branch = view.branch
            if (isinstance(branch.lengths, u.Quantity) and branch.lengths.unit == u.um
                    and isinstance(branch.lengths.mantissa, np.ndarray) and branch.lengths.mantissa.dtype == np.float64):
                lengths = branch.lengths.mantissa
            else:
                lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
            l_sum = lengths.sum()
            if l_sum <= 0:
                raise ValueError("Annotation mapping requires positive cable length.")
            n_seg = len(lengths)
            bounds = np.empty(n_seg + 1, dtype=float)
            bounds[0] = 0.0
            bounds[1:] = np.cumsum(lengths) / l_sum
            # Pairwise sum and cumulative sum can round differently. A source
            # endpoint is exactly x=1 regardless of that reduction roundoff.
            bounds[-1] = 1.0

            if (isinstance(branch.points_proximal, u.Quantity) and branch.points_proximal.unit == u.um
                    and isinstance(branch.points_proximal.mantissa, np.ndarray)):
                proximal = branch.points_proximal.mantissa
            else:
                proximal = np.asarray(branch.points_proximal.to_decimal(u.um))

            if (isinstance(branch.points_distal, u.Quantity) and branch.points_distal.unit == u.um
                    and isinstance(branch.points_distal.mantissa, np.ndarray)):
                distal = branch.points_distal.mantissa
            else:
                distal = np.asarray(branch.points_distal.to_decimal(u.um))

            p_rnd = np.round(proximal, 6)
            q_rnd = np.round(distal, 6)
            p_keys = [tuple(v) for v in p_rnd.tolist()]
            q_keys = [tuple(v) for v in q_rnd.tolist()]

            starts[seg_offset:seg_offset + n_seg] = proximal
            ends[seg_offset:seg_offset + n_seg] = distal
            branch_ids[seg_offset:seg_offset + n_seg] = branch_index
            fractions[seg_offset:seg_offset + n_seg, 0] = bounds[:-1]
            fractions[seg_offset:seg_offset + n_seg, 1] = bounds[1:]

            for j in range(n_seg):
                try:
                    a = lookup[p_keys[j]]
                    b = lookup[q_keys[j]]
                except KeyError as exc:
                    raise ValueError("Morphology endpoints do not match H01 source samples.") from exc
                ra = row_nodes[a]
                rb = row_nodes[b]
                actual_edges.append((ra, rb) if ra < rb else (rb, ra))
                if ra not in self._nodes:
                    self._nodes[ra] = (branch_index, float(bounds[j]))
                if rb not in self._nodes:
                    self._nodes[rb] = (branch_index, float(bounds[j + 1]))
                endpoints[seg_offset + j, 0] = a
                endpoints[seg_offset + j, 1] = b
            seg_offset += n_seg

        if len(actual_edges) != len(expected_edges) or set(actual_edges) != expected_edges:
            raise ValueError("Morphology edges do not preserve the source tree.")
        self._starts = starts
        self._ends = ends
        self._branches = branch_ids
        self._fractions = fractions
        self._endpoints = endpoints
        self._vectors = self._ends - self._starts
        self._length2 = np.sum(self._vectors**2, axis=1)
        self._lengths = np.sqrt(self._length2)
        self._row_id_to_index = {int(row[0]): i for i, row in enumerate(rows)}
        self._adjacency = [[] for _ in rows]
        for idx in range(total_segments):
            a, b = int(endpoints[idx, 0]), int(endpoints[idx, 1])
            length = float(self._lengths[idx])
            self._adjacency[a].append((b, length))
            self._adjacency[b].append((a, length))

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
        anchor = self._row_id_to_index[node_id]
        distance = np.full(len(self._row_id_to_index), np.inf)
        distance[anchor] = 0.
        stack = [anchor]
        adj = self._adjacency
        while stack:
            a = stack.pop()
            d_a = distance[a]
            for b, length in adj[a]:
                if np.isinf(distance[b]):
                    distance[b] = d_a + length
                    stack.append(b)
        intervals = []
        for (a, b), length, branch, (lo, hi) in zip(self._endpoints, self._lengths, self._branches, self._fractions):
            if distance[a] < distance[b]:
                left, right = 0., np.clip((radius_um - distance[a]) / length, 0., 1.)
            else:
                left, right = 1. - np.clip((radius_um - distance[b]) / length, 0., 1.), 1.
            if right > left:
                intervals.append((int(branch), float(lo + left * (hi - lo)), float(lo + right * (hi - lo))))
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
        left = self._labels[self._endpoints[:, 0]] == label
        right = self._labels[self._endpoints[:, 1]] == label
        both = left & right
        intervals = []
        if policy == "strict":
            idx = np.flatnonzero(both)
            for i in idx:
                intervals.append((int(self._branches[i]), float(self._fractions[i, 0]), float(self._fractions[i, 1])))
        else:
            idx = np.flatnonzero(left | right)
            for i in idx:
                b = int(self._branches[i])
                lo, hi = float(self._fractions[i, 0]), float(self._fractions[i, 1])
                if both[i]:
                    intervals.append((b, lo, hi))
                elif left[i]:
                    intervals.append((b, lo, (lo + hi) / 2))
                else:
                    intervals.append((b, (lo + hi) / 2, hi))
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
            diff = self._starts + t[:, None] * self._vectors - point
            dist2 = np.sum(diff**2, axis=1)
            best = int(np.argmin(dist2))
            min_dist2 = dist2[best]
            distance = float(np.sqrt(min_dist2))
            location, status = None, "too_far"
            if distance <= max_distance_um:
                tol2 = 2.0 * distance * 1e-9 + 1e-18
                candidates = np.flatnonzero(dist2 - min_dist2 <= tol2)
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
