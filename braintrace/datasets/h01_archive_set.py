"""Resolve H01 components across several verified archives by archive digest.

A manifest or a topology names each cell's source by ``archive_sha256``. One
pinned proofread archive and one or more non-proofread candidate archives
(for example the C3 skeleton export) can be open at once; the set routes each
``(cell_id, component, archive_sha256)`` request to the archive whose verified
digest matches, and refuses a digest it does not hold.
"""

from pathlib import Path

from .h01 import H01Archive

__all__ = ["H01ArchiveSet"]


class H01ArchiveSet:
    """A mapping ``{archive_sha256: H01Archive}`` with digest-addressed loading.

    Parameters
    ----------
    archives : iterable of H01Archive
        Verified archives. Two archives with the same digest are rejected.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01 import H01Archive
        >>> from braintrace.datasets.h01_archive_set import H01ArchiveSet
        >>> archives = H01ArchiveSet([H01Archive(".cache/h01/proofread104.zip")])  # doctest: +SKIP
        >>> sorted(archives.archives())  # doctest: +SKIP
        ['3e0534df357ef2e92f6e0199962133cc9bc9733eb3a1fd6d1d315208ad63db47']
    """

    def __init__(self, archives=()):
        self._archives = {}
        for archive in archives:
            self.add(archive)

    @classmethod
    def from_paths(cls, paths):
        """Open archives from ``{expected_sha256: path}`` or ``{expected_sha256: (path, source)}``.

        Parameters
        ----------
        paths : dict
            Digest to archive path, or to a ``(path, source_label)`` pair.

        Returns
        -------
        H01ArchiveSet
            Every archive verified against its key.
        """
        archives = []
        for digest, spec in paths.items():
            path, source = (spec, None) if isinstance(spec, (str, Path)) else spec
            archives.append(H01Archive(path, expected_sha256=digest, source=source))
        return cls(archives)

    def add(self, archive):
        """Register one verified archive under its digest.

        Parameters
        ----------
        archive : H01Archive
            Archive to add.

        Raises
        ------
        ValueError
            If an archive with the same digest is already registered.
        """
        digest = archive.archive_sha256
        if digest in self._archives:
            raise ValueError(f"Archive {digest} is already registered ({archive.source}).")
        self._archives[digest] = archive

    def archives(self):
        """Return ``{archive_sha256: H01Archive}`` as a fresh dict.

        Returns
        -------
        dict
            Registered archives keyed by verified digest.
        """
        return dict(self._archives)

    def archive(self, archive_sha256):
        """Return the archive holding ``archive_sha256``.

        Parameters
        ----------
        archive_sha256 : str
            Verified digest of the wanted archive.

        Returns
        -------
        H01Archive
            The registered archive.

        Raises
        ------
        KeyError
            If no registered archive has that digest; the message lists the
            digests and source labels that are registered.
        """
        digest = str(archive_sha256).lower()
        if digest not in self._archives:
            known = ", ".join(f"{d[:12]} ({a.source})" for d, a in self._archives.items()) or "none"
            raise KeyError(f"No H01 archive with sha256 {digest}; registered: {known}.")
        return self._archives[digest]

    def load(self, cell_id, component, archive_sha256):
        """Load one component from the archive named by its digest.

        Parameters
        ----------
        cell_id : str or int
            Cell identifier inside that archive.
        component : int
            Component suffix.
        archive_sha256 : str
            Digest naming the archive.

        Returns
        -------
        H01Component
            The loaded component, whose provenance records that digest.
        """
        return self.archive(archive_sha256).load(cell_id, component=int(component))

    def __contains__(self, archive_sha256):
        return str(archive_sha256).lower() in self._archives

    def __len__(self):
        return len(self._archives)
