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


def _verify(path):
    if _digest(path) != ARCHIVE_SHA256:
        raise ValueError("H01 archive SHA-256 mismatch; use the pinned official release.")


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
    """

    neuron_id: str
    component_id: int
    morphology: object
    source_rows: object
    normalized_swc: str
    report: object
    source_sha256: str

    def anatomy(self):
        """Build reusable anatomical locations and cable selections.

        Returns
        -------
        H01Anatomy
            Source-labelled selections for this morphology, with explicit
            inference policies and distance-checked spatial projection.
        """
        from .h01_anatomy import H01Anatomy

        return H01Anatomy(self)

    @property
    def provenance(self):
        """Return JSON-compatible source identity and conversion assumptions.

        Returns
        -------
        dict
            Release, checksums, units, annotation policy and attribution.
        """
        return {
            "release": RELEASE, "url": SOURCE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "member": f"{self.neuron_id}.{self.component_id}.swc",
            "source_sha256": self.source_sha256,
            "position_scale_um": [0.032, 0.032, 0.033], "radius_scale_um": 0.001,
            "compartment_policy": "custom; original H01 codes retained in source_rows",
            "electrical_properties": "not supplied by H01",
            "attribution": ATTRIBUTION,
        }


class H01Archive:
    """Access individual cells and components without extracting an archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Local official SWC archive. Validated against the pinned checksum.

    Notes
    -----
    Opening and loading are offline. Components are never silently joined,
    dropped, or presented as complete neurons. Choose a component explicitly.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        _verify(self.path)
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
        import braincell

        name = self._members[(str(neuron_id), component)]
        _verify(self.path)
        with zipfile.ZipFile(self.path) as archive:
            source = archive.read(name)
        rows, converted = normalize(source)
        with tempfile.TemporaryDirectory(prefix="braintrace-h01-") as temporary:
            path = Path(temporary) / name
            path.write_text(converted, encoding="utf-8")
            morphology, report = braincell.Morphology.from_swc(
                path, mode="neuromorpho", return_report=True,
            )
        return H01Component(
            str(neuron_id), component, morphology, rows, converted, report,
            hashlib.sha256(source).hexdigest(),
        )
