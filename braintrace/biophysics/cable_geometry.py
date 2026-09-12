"""Conservative chemical geometry from exact tapered cable control volumes."""

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json

import brainunit as u
import numpy as np

from .transport import DiffusionGraph


SHELL_FRACTIONS = np.array([11/36, 4/9, 2/9, 1/36])


def _integrals(branch, lo, hi):
    lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    rp = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    rd = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
    if (lengths.ndim != 1 or not len(lengths) or rp.shape != lengths.shape or rd.shape != lengths.shape or
            not all(np.isfinite(a).all() and np.all(a > 0) for a in (lengths, rp, rd))):
        raise ValueError('Chemical cable geometry requires positive finite source frusta')
    if not 0 <= lo < hi <= 1:
        raise ValueError('Invalid chemical CV interval')
    ends = np.cumsum(lengths)
    starts = np.r_[0., ends[:-1]]
    left, right = np.maximum(starts, lo*ends[-1]), np.minimum(ends, hi*ends[-1])
    mask = right > left
    span = (right-left)[mask]
    a = (rp+(rd-rp)*(left-starts)/lengths)[mask]
    b = (rp+(rd-rp)*(right-starts)/lengths)[mask]
    return np.array([np.sum(np.pi*span*(a*a+a*b+b*b)/3),
                     np.sum(np.pi*(a+b)*np.sqrt(span**2+(b-a)**2)),
                     np.sum(span), np.sum(span/(np.pi*a*b))])


@dataclass(frozen=True, init=False)
class CableChemicalGeometry:
    """Pin exact CV volumes, membrane areas and shared-boundary diffusion.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized cable with positive piecewise-frustum geometry. Midpoint
        branch attachments are unsupported; boundary junctions are eliminated
        using their complete shared-node roles.

    Notes
    -----
    Four shells use source annular fractions at every taper cross-section.
    Radial conductance is the source per-length law integrated along the CV.
    Axial transport uses exact half-CV integrals of ds/area, eliminating shared
    massless junctions. This conservative extension is not a 3-D taper solution.
    """

    _json: str

    def __init__(self, cell):
        if getattr(cell, '_runtime', None) is None:
            raise ValueError('Chemical geometry requires an initialized cable')
        cvs, branches = cell.cvs, cell.morpho.branches
        if [cv.id for cv in cvs] != list(range(len(cvs))):
            raise ValueError('Chemical geometry requires contiguous CV ordering')
        pieces = [[] for _ in branches]
        volumes, areas, lengths, halves, rows = [], [], [], [], []
        for cv in cvs:
            branch = branches[cv.branch_id].branch
            volume, area, length, resistance = _integrals(branch, cv.prox, cv.dist)
            middle = (cv.prox+cv.dist)/2
            half = [_integrals(branch, cv.prox, middle)[3], _integrals(branch, middle, cv.dist)[3]]
            electrical = [cv.area.to_decimal(u.um**2), cv.length.to_decimal(u.um),
                          (cv.r_axial/cv.ra).to_decimal(u.um**-1),
                          (cv.r_axial_prox/cv.ra).to_decimal(u.um**-1),
                          (cv.r_axial_dist/cv.ra).to_decimal(u.um**-1)]
            if not np.allclose([area, length, resistance, *half], electrical, rtol=1e-9, atol=1e-11):
                raise ValueError('Chemical geometry differs from electrical CV geometry')
            pieces[cv.branch_id].append((cv.prox, cv.dist))
            volumes.append(volume)
            areas.append(float(cv.area.to_decimal(u.um**2)))
            lengths.append(length)
            halves.append(half)
            rows.append([cv.id, cv.branch_id, cv.prox, cv.dist])
        for intervals in pieces:
            ordered = sorted(intervals)
            if (not ordered or ordered[0][0] != 0 or ordered[-1][1] != 1 or
                    any(a[1] != b[0] for a, b in zip(ordered[:-1], ordered[1:]))):
                raise ValueError('CV intervals must cover each source branch exactly once')
        conductances, seen, junctions = {}, set(), []
        for node in cell.node_tree.nodes:
            if node.kind == 'mid':
                if len(node.roles) != 1 or node.roles[0].position != 'mid':
                    raise ValueError('Midpoint attachments require a separate chemical discretization')
                continue
            junctions.append([(int(role.cv_id), role.position) for role in node.roles])
        # BrainCell collapses degree-two boundaries into direct midpoint edges.
        # Their two half-CV roles retain the eliminated junction provenance.
        for edge in cell.node_tree.edges:
            if len(edge.roles) > 1:
                if (len(edge.roles) != 2 or cell.node_tree.nodes[edge.parent_node_id].kind != 'mid' or
                        cell.node_tree.nodes[edge.child_node_id].kind != 'mid'):
                    raise ValueError('Unsupported collapsed chemical boundary')
                junctions.append([(int(role.cv_id), role.half) for role in edge.roles])
        for roles in junctions:
            if any(side not in ('prox', 'dist') or (i, side) in seen for i, side in roles):
                raise ValueError('Chemical boundary roles must uniquely cover CV ends')
            if len({i for i, _ in roles}) != len(roles):
                raise ValueError('A chemical junction cannot contain both ends of one CV')
            seen.update(roles)
            weights = {i: 1/halves[i][0 if side == 'prox' else 1] for i, side in roles}
            total = sum(weights.values())
            for i, j in combinations(sorted(weights), 2):
                conductances[i, j] = conductances.get((i, j), 0.)+weights[i]*weights[j]/total
        if seen != {(i, side) for i in range(len(cvs)) for side in ('prox', 'dist')}:
            raise ValueError('Chemical boundary mapping is incomplete')
        record = dict(schema='tapered-cable-chemistry-v1', cv_intervals=rows,
            volume_um3=volumes, area_um2=areas, length_um=lengths,
            half_resistance_per_um=halves, junctions=junctions,
            axial_edges=list(conductances), axial_conductance_um=list(conductances.values()),
            source_branches=[dict(length_um=b.branch.lengths.to_decimal(u.um).tolist(),
                radius_prox_um=b.branch.radii_proximal.to_decimal(u.um).tolist(),
                radius_dist_um=b.branch.radii_distal.to_decimal(u.um).tolist()) for b in branches],
            basis='Homothetic source shells; exact frustum volume/area and half-resistance; massless junction elimination')
        object.__setattr__(self, '_json', json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False))

    def to_dict(self):
        """Return detached geometry, CV ordering and discretization assumptions.

        Returns
        -------
        dict
            Canonical physical map, suitable for checkpoint manifests.
        """
        return json.loads(self._json)

    @property
    def sha256(self):
        """Return the immutable physical-map identity."""
        return hashlib.sha256(self._json.encode()).hexdigest()

    @property
    def volume_um3(self):
        """Return full CV volumes in um^3, in cable state ordering."""
        return np.asarray(self.to_dict()['volume_um3'])

    @property
    def area_um2(self):
        """Return actual lateral membrane areas in um^2."""
        return np.asarray(self.to_dict()['area_um2'])

    def transport(self, *, diffusion, dt_ms, radial=True):
        """Build a closed four-shell finite-volume transport graph.

        Parameters
        ----------
        diffusion : float
            Nonnegative diffusivity in um^2/ms.
        dt_ms : float
            Positive physical clock interval.
        radial : bool, optional
            Include source radial transport; disable for bound mobile buffer.

        Returns
        -------
        DiffusionGraph
            Conservative graph in CV-major, outer-to-inner shell order.
        """
        if not np.isfinite(diffusion) or diffusion < 0:
            raise ValueError('Diffusion must be finite and nonnegative')
        record = self.to_dict()
        edges, rates = [], []
        if radial:
            for i, length in enumerate(record['length_um']):
                for shell, factor in enumerate(np.pi*np.array([5., 3., 1.])):
                    edges.append((4*i+shell, 4*i+shell+1))
                    rates.append(diffusion*factor*length)
        for (i, j), conductance in zip(record['axial_edges'], record['axial_conductance_um']):
            for shell, fraction in enumerate(SHELL_FRACTIONS):
                edges.append((4*i+shell, 4*j+shell))
                rates.append(diffusion*conductance*fraction)
        volumes = self.volume_um3[:, None]*SHELL_FRACTIONS
        return DiffusionGraph(volumes.ravel(), np.asarray(edges, dtype=int).reshape(-1, 2), rates, dt_ms=dt_ms)
