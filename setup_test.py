"""Tests for source-distribution packaging policy in :mod:`setup`."""

import pathlib
import tarfile


def test_source_distribution_excludes_scratch_directories() -> None:
    """Require the locally built sdist to omit root scratch evidence."""
    archives = sorted((pathlib.Path(__file__).parent / "dist").glob("*.tar.gz"))
    assert archives, "build an sdist before running the packaging test"
    with tarfile.open(archives[-1], "r:gz") as archive:
        members = archive.getnames()
    assert not any("/tmp/" in member for member in members)
