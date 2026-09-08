"""Source-backed cell tags and synapse positions from H01 proofread_104."""

import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import urllib.request

import numpy as np

from .h01 import ATTRIBUTION, RELEASE

BASE_URL = "https://storage.googleapis.com/h01-release/data/20210601/proofread_104/"
ASSETS = {
    "cell_properties.json": (
        "segment_properties/info",
        "d8b9f54822460d4eb43fd7ca61897852a78c819bf229c8f27396c186f762716c",
    ),
    "synapse_locations.csv": (
        "synapse_locations.csv",
        "640ccb12c75b930f96373c7273bbf9688aa111b97b425ee22d8ef6db19f496be",
    ),
}
_FIELDS = ["104", "prepost", "x", "y", "z", "prex", "prey", "prez", "postx", "posty", "postz"]
_SYNAPSE_UM = np.array([.008, .008, .033])


def _verified_bytes(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"H01 annotation SHA-256 mismatch: {Path(path).name}")
    return data


def _provenance(name):
    source, checksum = ASSETS[name]
    return {
        "url": BASE_URL + source, "sha256": checksum, "release": RELEASE,
        "attribution": ATTRIBUTION,
        "verification": "released annotation; per-item manual review not supplied",
    }


def fetch_h01_annotations(cache_dir, *, timeout=60):
    """Explicitly download checksum-pinned H01 annotations for offline reuse.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        Directory for the JSON properties and approximately 19 MB CSV.
    timeout : float, optional
        Network timeout in seconds.

    Returns
    -------
    H01Annotations
        Validated annotation store. Existing corrupt files are not replaced.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for name, (source, checksum) in ASSETS.items():
        target = cache / name
        if not target.exists():
            with urllib.request.urlopen(BASE_URL + source, timeout=timeout) as response:
                data = response.read()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            with tempfile.TemporaryDirectory(prefix="h01-annotations-", dir=cache) as temporary:
                staged = Path(temporary) / name
                staged.write_bytes(data)
                _verified_bytes(staged, checksum)
                staged.replace(target)
        _verified_bytes(target, checksum)
    return H01Annotations(cache)


@dataclass(frozen=True)
class H01CellMetadata:
    """Whole-cell release annotations, independent of the selected component.

    Attributes
    ----------
    neuron_id : str
        Proofread_104 identifier.
    tags : tuple of str
        Released layer, cell type, and descriptive tags, without reinterpretation.
    measurements : mapping
        Original numeric properties (e.g. NSIe and NSIi), not fitted parameters.
    descriptions : mapping
        Released descriptions including units where supplied.
    provenance : mapping
        Source, checksum, attribution, and verification boundary.
    """

    neuron_id: str
    tags: tuple
    measurements: object
    descriptions: object
    provenance: object


@dataclass(frozen=True)
class H01Synapse:
    """One CSV row, retaining the cell's role and both physical endpoints.

    Attributes
    ----------
    neuron_id : str
        Proofread_104 cell ID, not the partner ID.
    source_row : int
        One-based CSV line number, including the header; not a biological ID.
    role : str
        ``pre`` or ``post`` for the named cell.
    center_um, pre_um, post_um : tuple of float
        Absolute coordinates in micrometers, converted from 8 x 8 x 33 nm voxels.

    Notes
    -----
    The CSV supplies neither per-synapse E/I labels nor partner neuron IDs.
    """

    neuron_id: str
    source_row: int
    role: str
    center_um: tuple
    pre_um: tuple
    post_um: tuple

    @property
    def position_um(self):
        """Return the endpoint belonging to this cell.

        Returns
        -------
        tuple of float
            Presynaptic or postsynaptic endpoint in micrometers.
        """
        return self.pre_um if self.role == "pre" else self.post_um


class H01Annotations:
    """Read the official proofread-cell annotation files offline.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        Directory populated by :func:`fetch_h01_annotations`.

    Notes
    -----
    Cell properties and the synapse CSV have different counts in this release.
    They remain separate observations; no per-synapse type is inferred from totals.
    """

    def __init__(self, cache_dir):
        self.path = Path(cache_dir).resolve()
        payload = _verified_bytes(self.path / "cell_properties.json", ASSETS["cell_properties.json"][1])
        _verified_bytes(self.path / "synapse_locations.csv", ASSETS["synapse_locations.csv"][1])
        data = json.loads(payload)
        if data.get("@type") != "neuroglancer_segment_properties":
            raise ValueError("Expected Neuroglancer segment properties.")
        inline = data["inline"]
        ids = inline["ids"]
        if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x.isdecimal() for x in ids):
            raise ValueError("Cell IDs must be unique decimal strings.")
        properties = inline["properties"]
        names = [p["id"] for p in properties]
        if len(set(names)) != len(names) or "tags" not in names:
            raise ValueError("Expected unique properties including tags.")
        for p in properties:
            if len(p["values"]) != len(ids):
                raise ValueError("Property values must align with cell IDs.")
            if p["id"] == "tags":
                if p["type"] != "tags" or any(
                    type(i) is not int or not 0 <= i < len(p["tags"])
                    for row in p["values"] for i in row
                ):
                    raise ValueError("Invalid tag indices.")
            elif p["type"] != "number" or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                for v in p["values"]
            ):
                raise ValueError("Expected finite numeric cell properties.")
        tags = properties[names.index("tags")]
        self._tags = frozenset(tags["tags"])
        self._cells = {}
        for i, cell_id in enumerate(ids):
            self._cells[cell_id] = H01CellMetadata(
                cell_id, tuple(tags["tags"][j] for j in tags["values"][i]),
                MappingProxyType({p["id"]: p["values"][i] for p in properties if p["id"] != "tags"}),
                MappingProxyType({p["id"]: p.get("description", "") for p in properties}),
                MappingProxyType(_provenance("cell_properties.json")),
            )

    def select(self, *tags):
        """Select cell IDs containing all requested source tags.

        Parameters
        ----------
        *tags : str
            Exact tags, for example ``L2`` and ``pyramidal``; unknown tags raise.

        Returns
        -------
        tuple of str
            Matching IDs in numeric order; no tags selects every cell.
        """
        if set(tags) - self._tags:
            raise ValueError(f"Unknown H01 tags: {set(tags) - self._tags}")
        return tuple(sorted((k for k, v in self._cells.items() if set(tags) <= set(v.tags)), key=int))

    def metadata(self, neuron_id):
        """Return immutable source annotations for one whole cell.

        Parameters
        ----------
        neuron_id : str or int
            Proofread_104 cell ID.

        Returns
        -------
        H01CellMetadata
            Source tags and measurements; missing IDs raise ``KeyError``.
        """
        return self._cells[str(neuron_id)]

    @property
    def synapse_provenance(self):
        """Return the synapse table's provenance and coordinate convention.

        Returns
        -------
        dict
            Source identity, scaling, and explicitly missing fields.
        """
        return {**_provenance("synapse_locations.csv"), "position_scale_um": [.008, .008, .033],
                "missing_fields": ["synapse_id", "partner_neuron_id", "ei_type", "conductance"]}

    def synapses(self, neuron_id, *, role=None):
        """Return this cell's CSV rows without deduplication or invented labels.

        Parameters
        ----------
        neuron_id : str or int
            Proofread_104 cell ID.
        role : str or None, optional
            Filter ``pre`` or ``post``; ``None`` returns both.

        Returns
        -------
        tuple of H01Synapse
            Rows in file order, carrying stable source line numbers.
        """
        neuron_id = self.metadata(neuron_id).neuron_id
        if role not in (None, "pre", "post"):
            raise ValueError("role must be pre, post, or None.")
        path = self.path / "synapse_locations.csv"
        _verified_bytes(path, ASSETS[path.name][1])
        result = []
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != _FIELDS:
                raise ValueError("Unexpected H01 synapse CSV schema.")
            for line, row in enumerate(reader, 2):
                if row["104"] != neuron_id:
                    continue
                if row["prepost"] not in ("pre", "post"):
                    raise ValueError(f"Invalid synapse role on row {line}.")
                xyz = np.array([float(row[f]) for f in _FIELDS[2:]]).reshape(3, 3)
                if not np.isfinite(xyz).all() or (xyz < 0).any():
                    raise ValueError(f"Invalid synapse coordinates on row {line}.")
                if role is None or row["prepost"] == role:
                    result.append(H01Synapse(neuron_id, line, row["prepost"],
                                             *(tuple(p) for p in xyz * _SYNAPSE_UM)))
        return tuple(result)
