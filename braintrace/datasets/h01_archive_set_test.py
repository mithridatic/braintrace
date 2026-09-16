"""Tests for digest-addressed loading across several H01 archives (offline)."""

import hashlib
import io
import zipfile

import pytest

from . import h01
from .h01_archive_set import H01ArchiveSet

SOURCE = b"0 1 100 100 100 200 -1\n1 -1 200 100 100 100 0\n2 0 200 200 100 100 1\n"
OTHER = b"0 3 10 10 10 500 -1\n1 1 20 10 10 100 0\n"


def _zip(path, members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, source in members:
            archive.writestr(name, source)
    path.write_bytes(stream.getvalue())
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _two(tmp_path, monkeypatch):
    pinned = _zip(tmp_path / "pinned.zip", [("12.0.swc", SOURCE)])
    monkeypatch.setattr(h01, "ARCHIVE_SHA256", pinned)
    other = _zip(tmp_path / "c3.zip", [("77.0.swc", OTHER), ("12.0.swc", OTHER)])
    return pinned, other


def test_from_paths_verifies_and_labels_each_archive(tmp_path, monkeypatch):
    pinned, other = _two(tmp_path, monkeypatch)
    archives = H01ArchiveSet.from_paths({pinned: tmp_path / "pinned.zip",
                                         other: (tmp_path / "c3.zip", "c3_candidates")})
    assert len(archives) == 2 and pinned in archives and other in archives
    assert archives.archive(pinned).source == "proofread_104"
    assert archives.archive(other).source == "c3_candidates"
    assert set(archives.archives()) == {pinned, other}
    assert archives.archive(other.upper()).path.name == "c3.zip"


def test_load_routes_the_same_cell_id_by_digest(tmp_path, monkeypatch):
    pinned, other = _two(tmp_path, monkeypatch)
    archives = H01ArchiveSet.from_paths({pinned: tmp_path / "pinned.zip", other: tmp_path / "c3.zip"})
    a = archives.load(12, 0, pinned)
    b = archives.load("12", "0", other)
    assert a.source_sha256 == hashlib.sha256(SOURCE).hexdigest()
    assert b.source_sha256 == hashlib.sha256(OTHER).hexdigest()
    assert a.provenance["archive_sha256"] == pinned and a.provenance["source"] == "proofread_104"
    assert b.provenance["archive_sha256"] == other and b.provenance["source"].startswith("sha256:")
    assert b.provenance["url"] is None and b.provenance["release"] == b.provenance["source"]


def test_missing_digest_raises_listing_known_sources(tmp_path, monkeypatch):
    pinned, other = _two(tmp_path, monkeypatch)
    archives = H01ArchiveSet([h01.H01Archive(tmp_path / "pinned.zip")])
    with pytest.raises(KeyError, match="proofread_104"):
        archives.load(12, 0, other)
    with pytest.raises(KeyError, match="registered: none"):
        H01ArchiveSet().archive(other)


def test_wrong_digest_and_duplicate_digest_are_rejected(tmp_path, monkeypatch):
    pinned, other = _two(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        H01ArchiveSet.from_paths({pinned: tmp_path / "c3.zip"})
    archive = h01.H01Archive(tmp_path / "c3.zip", expected_sha256=other)
    with pytest.raises(ValueError, match="already registered"):
        H01ArchiveSet([archive, archive])
