"""Regression coverage for distinct H01 attachment points at large coordinates."""

import hashlib
import os
from pathlib import Path
import zipfile

import pytest

from . import h01


def test_distinct_voxel_leaf_is_preserved(tmp_path, monkeypatch):
    source = (b"0 0 100000 100000 100000 100 -1\n"
              b"1 0 100100 100000 100000 100 0\n"
              b"2 0 100101 100000 100000 100 1\n"
              b"3 0 100100 100100 100000 100 1\n")
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("5805562981.0.swc", source)
    monkeypatch.setattr(h01, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    result = h01.H01Archive(path).load("5805562981", component=0)
    assert len(result.source_rows) == 4
    assert result.source_rows[2, 2] - result.source_rows[1, 2] == 1
    assert result.report.error_count == 0


@pytest.mark.parametrize("identity", [
    "5965472721", "5805562981", "4365276903", "3111823553",
    "5687162964", "4641147055", "5136107765",
])
def test_released_attachment_regressions(identity):
    path = os.environ.get("H01_ARCHIVE")
    if path is None:
        pytest.skip("Set H01_ARCHIVE to run the pinned real-release regressions")
    result = h01.H01Archive(Path(path)).load(identity, component=0)
    assert result.report.error_count == 0
    assert result.neuron_id == identity
