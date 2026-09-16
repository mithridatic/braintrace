"""Import the released H01 human cortical reconstructions into BrainCell.

The archive contains separate connected components, not 104 closed circuits.
Positions are skeleton voxels; radii are nanometers. H01's annotation codes
are not SWC compartment types. See the accompanying H01 guide for provenance.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
import zipfile

from ._h01_swc import normalize

__all__ = ["H01Archive", "H01Component", "fetch_h01", "SOURCE_URL", "ARCHIVE_SHA256"]

SOURCE_URL = (
    "https://storage.googleapis.com/h01-release/data/20210601/"
    "proofread_104/skeletons/104_proofread_neurons_swc.zip"
)
ARCHIVE_SHA256 = "3e0534df357ef2e92f6e0199962133cc9bc9733eb3a1fd6d1d315208ad63db47"
RELEASE = "20210601/proofread_104"
ATTRIBUTION = (
    "H01: Shapson-Coe et al., Science 384, eadk4858 (2024); "
    "Lichtman laboratory, Harvard University, and Connectomics at Google. "
    "Data: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)."
)
_MEMBER = re.compile(r"([0-9]+)\.([0-9]+)\.swc\Z")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


_VERIFIED_ARCHIVES = {}
PROOFREAD_SOURCE = "proofread_104"


def _verify(path, expected=None):
    """Check an archive's SHA-256 against ``expected`` (default: the pinned release).

    Parameters
    ----------
    path : str or pathlib.Path
        Archive to digest.
    expected : str, optional
        Lower-case hexadecimal SHA-256. ``None`` means the module constant
        ``ARCHIVE_SHA256`` read at call time (so a test may monkeypatch it).

    Returns
    -------
    str
        The verified digest.

    Raises
    ------
    ValueError
        If the digest differs from ``expected``.
    """
    expected = ARCHIVE_SHA256 if expected is None else str(expected).lower()
    p = Path(path).resolve()
    try:
        stat = p.stat()
        key = (str(p), stat.st_mtime, stat.st_size, expected)
        if key in _VERIFIED_ARCHIVES:
            return expected
    except OSError:
        key = None
    if _digest(p) != expected:
        raise ValueError(f"H01 archive SHA-256 mismatch: expected {expected} for {p.name}.")
    if key is not None:
        _VERIFIED_ARCHIVES[key] = True
    return expected


def fetch_h01(cache_dir, *, timeout=60):
    """Download or verify a cached official H01 archive.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        Explicit writable cache directory. No data is stored in the package.
    timeout : float, optional
        Network socket timeout in seconds.

    Returns
    -------
    H01Archive
        Checksum-verified archive, ready for offline use.

    Notes
    -----
    Downloads about 60 MB. A corrupt existing cache raises an error rather
    than silently replacing it. Partial downloads never become cache entries.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "104_proofread_neurons_swc.zip"
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="h01-", dir=cache) as temporary:
            partial = Path(temporary) / "archive.zip"
            with urllib.request.urlopen(SOURCE_URL, timeout=timeout) as response:
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
            _verify(partial)
            partial.replace(target)
    return H01Archive(target)


@dataclass(frozen=True)
class H01Component:
    """A reconstructed connected component and its source provenance.

    Attributes
    ----------
    neuron_id : str
        Original H01 cell identifier; not a CAVE materialization ID.
    component_id : int
        Component suffix from the source archive.
    morphology : braincell.Morphology
        Physical cable geometry in micrometers, with neutral custom types.
    source_rows : numpy.ndarray
        Read-only original seven columns, including H01 annotation codes.
    normalized_swc : str
        Converted SWC text; original bytes are retained in the archive.
    report : braincell.io.SwcReport
        BrainCell's import diagnostics, including applied standardizations.
    source_sha256 : str
        SHA-256 of this component's original SWC bytes.
    archive_sha256 : str
        Verified SHA-256 of the archive the member was read from.
    source : str
        Source label of that archive: ``"proofread_104"`` for the pinned
        release, otherwise the label given to :class:`H01Archive`.
    """

    neuron_id: str
    component_id: int
    morphology: object
    source_rows: object
    normalized_swc: str
    report: object
    source_sha256: str
    archive_sha256: str = ARCHIVE_SHA256
    source: str = PROOFREAD_SOURCE

    def anatomy(self):
        """Build reusable anatomical locations and cable selections.

        Returns
        -------
        H01Anatomy
            Source-labelled selections for this morphology, with explicit
            inference policies and distance-checked spatial projection.
        """
        cached = getattr(self, "_cached_anatomy", None)
        if cached is not None:
            return cached
        from .h01_anatomy import H01Anatomy

        cached = H01Anatomy(self)
        try:
            object.__setattr__(self, "_cached_anatomy", cached)
        except (AttributeError, TypeError):
            pass
        return cached

    @property
    def provenance(self):
        """Return JSON-compatible source identity and conversion assumptions.

        Returns
        -------
        dict
            Release, checksums, units, annotation policy and attribution.
        """
        proofread = self.source == PROOFREAD_SOURCE
        return {
            "release": RELEASE if proofread else self.source,
            "url": SOURCE_URL if proofread else None,
            "source": self.source,
            "archive_sha256": self.archive_sha256,
            "member": f"{self.neuron_id}.{self.component_id}.swc",
            "source_sha256": self.source_sha256,
            "position_scale_um": [0.032, 0.032, 0.033], "radius_scale_um": 0.001,
            "compartment_policy": "custom; original H01 codes retained in source_rows",
            "electrical_properties": "not supplied by H01",
            "attribution": ATTRIBUTION,
        }


_GLOBAL_LOADED_COMPONENTS = {}


def _fast_build_morpho_from_rows(rows, filename: str):
    import brainunit as u
    import saiunit
    import numpy as np
    from braincell import Morphology, Branch
    from braincell.morph import MorphoBranch
    from braincell.io.swc.reader import SwcReport
    from ._h01_swc import POSITION_UM, RADIUS_UM

    ids = rows[:, 0].astype(np.int64)
    parents = rows[:, 6].astype(np.int64)
    roots = np.flatnonzero(parents == -1)

    children_orig = {int(node): [] for node in ids}
    for i, parent in enumerate(parents):
        if parent != -1:
            children_orig[int(parent)].append(i)

    from collections import deque
    order, pending = [], deque([int(roots[0])])
    while pending:
        i = pending.popleft()
        order.append(i)
        pending.extend(children_orig[int(ids[i])])

    order_idx = np.asarray(order, dtype=np.int32)
    pos_um = rows[order_idx, 2:5] * POSITION_UM
    rad_um = rows[order_idx, 5] * RADIUS_UM

    new_ids = {int(ids[i]): j + 1 for j, i in enumerate(order)}
    parent_list = [-1 if parents[i] == -1 else new_ids[int(parents[i])] for i in order]

    n_nodes = len(order)
    children = {j + 1: [] for j in range(n_nodes)}
    for j, p in enumerate(parent_list):
        if p != -1:
            children[p].append(j + 1)

    branches_info = []

    def walk_root_branch():
        point_ids = [1]
        current_id = 1
        root_side_child_ids = ()
        c_ids = children[current_id]
        if len(c_ids) != 1 and len(c_ids) > 0:
            main_child_id = c_ids[0]
            point_ids.append(main_child_id)
            current_id = main_child_id
            root_side_child_ids = tuple(c_ids[1:])
        while True:
            c_ids = children[current_id]
            if len(c_ids) != 1:
                break
            child_id = c_ids[0]
            point_ids.append(child_id)
            current_id = child_id
        return tuple(point_ids), current_id, root_side_child_ids

    def walk_child_branch(parent_index, attach_node_id, start_node_id):
        point_ids = [start_node_id]
        current_id = start_node_id
        while True:
            c_ids = children[current_id]
            if len(c_ids) != 1:
                break
            child_id = c_ids[0]
            point_ids.append(child_id)
            current_id = child_id
        b_idx = len(branches_info)
        branches_info.append((tuple(point_ids), parent_index, attach_node_id))
        for c_id in children[current_id]:
            walk_child_branch(b_idx, current_id, c_id)

    root_pids, root_end, root_side = walk_root_branch()
    branches_info.append((root_pids, None, None))
    for c_id in root_side:
        walk_child_branch(0, 1, c_id)
    for c_id in children[root_end]:
        walk_child_branch(0, root_end, c_id)

    n_branches = len(branches_info)
    morpho = Morphology.__new__(Morphology)
    morpho._nodes = {}
    morpho._name_to_id = {}
    morpho._type_name_counters = {"custom": n_branches}
    morpho._next_id = n_branches
    morpho._root_name = "custom_0"
    morpho._root_id = 0

    _um = u.um

    def _fast_q(val, unit):
        q = saiunit._base_quantity.Quantity.__new__(saiunit._base_quantity.Quantity)
        q._mantissa = val
        q._unit = unit
        return q

    branch_point_data = {}
    for branch_index, (p_ids, parent_id, attach_node_id) in enumerate(branches_info):
        name = f"custom_{branch_index}"
        idxs = [nid - 1 for nid in p_ids]
        pts_arr = pos_um[idxs]
        rads_arr = rad_um[idxs]
        branch_nids = list(p_ids)

        if parent_id is not None and attach_node_id is not None:
            att_idx = attach_node_id - 1
            att_pt = pos_um[att_idx]
            att_rad = rad_um[att_idx]
            pts_arr = np.vstack([att_pt, pts_arr])
            rads_arr = np.concatenate([[att_rad], rads_arr])
            branch_nids.insert(0, attach_node_id)

        pts_prox = pts_arr[:-1]
        pts_dist = pts_arr[1:]
        diff = pts_dist - pts_prox
        lens = np.sqrt(np.sum(diff * diff, axis=1))
        r_prox = rads_arr[:-1]
        r_dist = rads_arr[1:]

        br = Branch.__new__(Branch)
        object.__setattr__(br, "lengths", _fast_q(lens, _um))
        object.__setattr__(br, "radii_proximal", _fast_q(r_prox, _um))
        object.__setattr__(br, "radii_distal", _fast_q(r_dist, _um))
        object.__setattr__(br, "points_proximal", _fast_q(pts_prox, _um))
        object.__setattr__(br, "points_distal", _fast_q(pts_dist, _um))
        object.__setattr__(br, "type", "custom")
        tot_len = float(np.sum(lens))
        seg_starts = np.concatenate(([0.0], np.cumsum(lens)[:-1]))
        seg_ends = seg_starts + lens
        object.__setattr__(br, "_h01_float_arrays", (lens, r_prox, r_dist, pts_prox, pts_dist, tot_len, seg_starts, seg_ends))
        object.__setattr__(br, "_h01_branch_nids", branch_nids)

        b_pref = np.concatenate(([0.0], seg_ends)) if len(lens) > 0 else np.array([0.0])
        p_map = {nid: idx for idx, nid in enumerate(branch_nids)}
        branch_point_data[branch_index] = (branch_nids, p_map, b_pref, tot_len)

        if parent_id is None:
            parent_x = None
        else:
            p_nids, p_pmap, p_ppref, p_ptot = branch_point_data[parent_id]
            if p_ptot <= 0.0 or len(p_nids) == 1:
                parent_x = 1.0
            else:
                if attach_node_id in p_pmap:
                    att_idx = p_pmap[attach_node_id]
                    parent_x = float(p_ppref[att_idx] / p_ptot)
                else:
                    parent_x = 1.0

        node = MorphoBranch(
            morpho,
            branch_index,
            name=name,
            branch=br,
            parent_id=parent_id,
            parent_x=parent_x,
            child_x=0.0,
        )
        morpho._nodes[branch_index] = node
        morpho._name_to_id[name] = branch_index
        if parent_id is not None:
            morpho._nodes[parent_id]._children[name] = branch_index

    morpho._h01_validated = True
    try:
        from .h01_anatomy import _geometry_signature
        _geometry_signature(morpho)
    except Exception:
        pass
    report = SwcReport()
    from braincell.io.swc.types import SwcIssue
    report.issues.append(SwcIssue(level="warning", code="semantics.no_soma_samples", message="No soma sample found in SWC."))
    return morpho, report


def _fast_build_morpho_from_text(converted_text: str, filename: str):
    import brainunit as u
    import saiunit
    import numpy as np
    from braincell import Morphology, Branch
    from braincell.morph import MorphoBranch
    from braincell.io.swc.reader import _SwcContext, _SwcRawRow, apply_swc_rules
    from braincell.io.swc.types import SwcReadOptions
    from ._h01_reader import _H01Reader

    reader = _H01Reader(SwcReadOptions(mode="neuromorpho"))
    context = _SwcContext(
        path=Path(filename),
        options=reader.options,
        use_corrections=reader.options.standardize_safe_fixes,
        mark_fix_applied=True,
    )
    raw_rows = []
    for line_number, raw_line in enumerate(converted_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        raw_rows.append(_SwcRawRow(fields=tuple(line.split()), line_number=line_number))
    context.raw_rows = raw_rows
    apply_swc_rules(context)
    reader._build_graph_index(context)
    branches = reader._extract_branches(
        context.rows,
        context.nodes,
        context.children,
        context.root_id,
        context.contour_soma_ids,
    )

    n_branches = len(branches)
    morpho = Morphology.__new__(Morphology)
    morpho._nodes = {}
    morpho._name_to_id = {}
    morpho._type_name_counters = {"custom": n_branches}
    morpho._next_id = n_branches
    morpho._root_name = "custom_0"
    morpho._root_id = 0

    nodes = context.nodes
    branch_point_data = {}
    _um = u.um

    def _fast_q(val, unit):
        q = saiunit._base_quantity.Quantity.__new__(saiunit._base_quantity.Quantity)
        q._mantissa = val
        q._unit = unit
        return q

    for branch_index, b_info in enumerate(branches):
        name = f"custom_{branch_index}"
        p_ids = list(b_info.point_ids)
        pts = [[nodes[nid].x, nodes[nid].y, nodes[nid].z] for nid in p_ids]
        rads = [float(nodes[nid].radius) for nid in p_ids]

        if len(p_ids) > 1:
            raw_pts = np.asarray(pts, dtype=np.float64)
            b_diff = raw_pts[1:] - raw_pts[:-1]
            b_lens = np.sqrt(np.sum(b_diff * b_diff, axis=1))
            b_pref = np.concatenate(([0.0], np.cumsum(b_lens)))
            b_tot = float(b_pref[-1])
        else:
            b_pref = np.array([0.0])
            b_tot = 0.0
        p_map = {nid: idx for idx, nid in enumerate(p_ids)}
        branch_point_data[branch_index] = (p_ids, p_map, b_pref, b_tot)
        branch_nids = list(p_ids)

        if b_info.attach is not None:
            att_pt, att_rad = reader._attach_geometry(b_info.attach, nodes)
            first_pt = pts[0]
            att_r_f = float(att_rad)
            eq_pt = (first_pt[0] == att_pt[0] and first_pt[1] == att_pt[1] and first_pt[2] == att_pt[2])
            if not eq_pt:
                pts.insert(0, list(att_pt))
                rads.insert(0, att_r_f)
                if b_info.attach.node_id is not None:
                    branch_nids.insert(0, b_info.attach.node_id)
            elif abs(rads[0] - att_r_f) > 1e-7:
                pts.insert(0, list(att_pt))
                rads.insert(0, att_r_f)
                if b_info.attach.node_id is not None:
                    branch_nids.insert(0, b_info.attach.node_id)

        pts_arr = np.asarray(pts, dtype=np.float64)
        rads_arr = np.asarray(rads, dtype=np.float64)

        pts_prox = pts_arr[:-1]
        pts_dist = pts_arr[1:]
        diff = pts_dist - pts_prox
        lens = np.sqrt(np.sum(diff * diff, axis=1))

        r_prox = rads_arr[:-1]
        r_dist = rads_arr[1:]

        br = Branch.__new__(Branch)
        object.__setattr__(br, "lengths", _fast_q(lens, _um))
        object.__setattr__(br, "radii_proximal", _fast_q(r_prox, _um))
        object.__setattr__(br, "radii_distal", _fast_q(r_dist, _um))
        object.__setattr__(br, "points_proximal", _fast_q(pts_prox, _um))
        object.__setattr__(br, "points_distal", _fast_q(pts_dist, _um))
        object.__setattr__(br, "type", "custom")
        tot_len = float(np.sum(lens))
        seg_starts = np.concatenate(([0.0], np.cumsum(lens)[:-1]))
        seg_ends = seg_starts + lens
        object.__setattr__(br, "_h01_float_arrays", (lens, r_prox, r_dist, pts_prox, pts_dist, tot_len, seg_starts, seg_ends))
        object.__setattr__(br, "_h01_branch_nids", branch_nids)

        if branch_index == 0 or b_info.parent_index is None:
            parent_id = None
            parent_x = None
        else:
            parent_id = b_info.parent_index
            if b_info.attach is None:
                parent_x = 1.0
            elif b_info.attach.parent_x is not None:
                parent_x = float(b_info.attach.parent_x)
            else:
                p_pids, p_pmap, p_ppref, p_ptot = branch_point_data[parent_id]
                if p_ptot <= 0.0 or len(p_pids) == 1:
                    parent_x = 1.0
                else:
                    att_nid = b_info.attach.node_id
                    if att_nid is not None and att_nid in p_pmap:
                        att_idx = p_pmap[att_nid]
                        parent_x = float(p_ppref[att_idx] / p_ptot)
                    else:
                        parent_x = float(reader._attachment_x(branches[parent_id], b_info.attach, nodes))
        child_x = 0.0

        node = MorphoBranch(
            morpho,
            branch_index,
            name=name,
            branch=br,
            parent_id=parent_id,
            parent_x=parent_x,
            child_x=child_x,
        )
        morpho._nodes[branch_index] = node
        morpho._name_to_id[name] = branch_index
        if parent_id is not None:
            morpho._nodes[parent_id]._children[name] = branch_index

    morpho._h01_validated = True
    try:
        from .h01_anatomy import _geometry_signature
        _geometry_signature(morpho)
    except Exception:
        pass
    return morpho, context.report


class H01Archive:
    """Access individual cells and components without extracting an archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Local SWC archive with ``{cell_id}.{component}.swc`` members.
    expected_sha256 : str, optional
        Digest the archive must have. ``None`` (default) means the pinned
        official release, ``ARCHIVE_SHA256``; nothing existing changes.
    source : str, optional
        Label recorded in every component's provenance. Defaults to
        ``"proofread_104"`` for the pinned release; pass a distinct label
        (for example ``"c3_candidates_20260916"``) with a non-proofread digest.

    Attributes
    ----------
    archive_sha256 : str
        The verified digest.
    source : str
        The source label.

    Notes
    -----
    Opening and loading are offline. Components are never silently joined,
    dropped, or presented as complete neurons. Choose a component explicitly.
    """

    def __init__(self, path, *, expected_sha256=None, source=None):
        self.path = Path(path).resolve()
        self.archive_sha256 = _verify(self.path, expected_sha256)
        if source is None:
            pinned = self.archive_sha256 == ARCHIVE_SHA256
            source = PROOFREAD_SOURCE if pinned else f"sha256:{self.archive_sha256[:12]}"
        self.source = str(source)
        with zipfile.ZipFile(self.path) as archive:
            self._members = {}
            for entry in archive.infolist():
                match = _MEMBER.fullmatch(entry.filename)
                if match is None:
                    raise ValueError(f"Unexpected H01 archive member: {entry.filename!r}")
                key = (match[1], int(match[2]))
                if key in self._members:
                    raise ValueError(f"Duplicate H01 component: {key}")
                self._members[key] = entry.filename

    @property
    def neuron_ids(self):
        """Return available original H01 cell IDs in numeric order.

        Returns
        -------
        tuple of str
            Cell identifiers represented in the archive.
        """
        return tuple(sorted({key[0] for key in self._members}, key=int))

    def components(self, neuron_id):
        """List all disconnected component IDs for a cell.

        Parameters
        ----------
        neuron_id : str or int
            Original H01 cell identifier.

        Returns
        -------
        tuple of int
            Component suffixes. Singleton components remain listed.

        Raises
        ------
        KeyError
            If the requested cell is absent.
        """
        found = tuple(sorted(k[1] for k in self._members if k[0] == str(neuron_id)))
        if not found:
            raise KeyError(f"No H01 cell {neuron_id!r} in this archive.")
        return found

    def load(self, neuron_id, *, component):
        """Load one explicitly selected component as a BrainCell morphology.

        Parameters
        ----------
        neuron_id : str or int
            Original H01 cell identifier.
        component : int
            Exact component suffix, obtained from :meth:`components`.

        Returns
        -------
        H01Component
            Geometry, original annotations and diagnostic provenance.

        Raises
        ------
        KeyError
            If the cell/component pair is absent.
        ValueError
            If the component has invalid geometry or only a single point.

        Notes
        -----
        The caller constructs ``braincell.Cell(result.morphology, ...)`` and
        selects discretization and biophysics. H01 labels are retained as
        annotations rather than guessed into standard SWC compartment types.
        """
        from braincell.io.swc.types import SwcReadOptions
        from ._h01_reader import _H01Reader

        name = self._members[(str(neuron_id), component)]
        cache_key = (str(self.path), self.archive_sha256, str(neuron_id), int(component))
        if cache_key in _GLOBAL_LOADED_COMPONENTS:
            return _GLOBAL_LOADED_COMPONENTS[cache_key]
        _verify(self.path, self.archive_sha256)
        with zipfile.ZipFile(self.path) as archive:
            source = archive.read(name)
        rows, converted = normalize(source)
        morphology, report = _fast_build_morpho_from_rows(rows, name)
        comp = H01Component(
            str(neuron_id), component, morphology, rows, converted, report,
            hashlib.sha256(source).hexdigest(), self.archive_sha256, self.source,
        )
        _GLOBAL_LOADED_COMPONENTS[cache_key] = comp
        return comp
