"""Offline importer tests; real-release verification is documented separately."""

import hashlib
import io
import zipfile

import braincell
import brainunit as u
import numpy as np
import pytest

from . import h01

SOURCE = b"0 1 100 100 100 200 -1\n1 -1 200 100 100 100 0\n2 0 200 200 100 100 1\n"


def _archive(tmp_path, monkeypatch, members=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, source in members or [("12.0.swc", SOURCE), ("12.2.swc", SOURCE), ("2.0.swc", SOURCE)]:
            archive.writestr(name, source)
    payload = stream.getvalue()
    monkeypatch.setattr(h01, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    path = tmp_path / "archive.zip"
    path.write_bytes(payload)
    return path, payload


def test_offline_archive_load_and_provenance(tmp_path, monkeypatch):
    path, _ = _archive(tmp_path, monkeypatch)
    archive = h01.H01Archive(path)
    assert archive.neuron_ids == ("2", "12")
    assert archive.components(12) == (0, 2)
    result = archive.load(12, component=0)
    assert isinstance(result.morphology, braincell.Morphology)
    assert result.source_sha256 == hashlib.sha256(SOURCE).hexdigest()
    assert result.provenance["member"] == "12.0.swc"
    assert result.provenance["archive_sha256"] == h01.ARCHIVE_SHA256
    assert "CC BY 4.0" in result.provenance["attribution"]
    assert result.report.error_count == 0
    assert any(issue.code == "semantics.no_soma_samples" for issue in result.report.issues)
    # H01 code 1 must never turn a dendritic sample into a standard SWC soma.
    assert set(np.loadtxt(result.normalized_swc.splitlines())[:, 1]) == {0}
    with pytest.raises(KeyError):
        archive.components("missing")
    with pytest.raises(KeyError):
        archive.load(12, component=1)
    with pytest.raises(TypeError):
        archive.load(12)


def test_construct_and_run_compiled_braincell(tmp_path, monkeypatch):
    path, _ = _archive(tmp_path, monkeypatch)
    imported = h01.H01Archive(path).load(12, component=0)
    from braincell.filter import AllRegion, RootLocation
    from braincell.mech import Channel, StateProbe

    cell = braincell.Cell(imported.morphology, V_init=-65 * u.mV)
    cell.paint(AllRegion(), Channel("IL", g_max=.1 * u.mS / u.cm**2, E=-65 * u.mV))
    cell.place(RootLocation(0.), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(0.), braincell.CurrentClamp(
        delay=.05 * u.ms, durations=.1 * u.ms, amplitudes=.001 * u.nA,
    ))
    result = cell.run(dt=.025 * u.ms, duration=.2 * u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    assert voltage.shape == (8,)
    assert np.isfinite(voltage).all()
    assert voltage[-1] > voltage[0]
    assert voltage[0] == pytest.approx(-65, abs=.01)


def test_download_is_explicit_and_cached_offline(tmp_path, monkeypatch):
    _, payload = _archive(tmp_path, monkeypatch)
    requests = []

    def download(url, timeout):
        requests.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(h01.urllib.request, "urlopen", download)
    cache = tmp_path / "cache"
    result = h01.fetch_h01(cache, timeout=9)
    assert result.path.read_bytes() == payload
    assert requests == [(h01.SOURCE_URL, 9)]
    assert h01.fetch_h01(cache).neuron_ids == result.neuron_ids
    assert len(requests) == 1
    assert len(list(cache.iterdir())) == 1


def test_corrupt_cache_is_not_overwritten(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    path = tmp_path / "104_proofread_neurons_swc.zip"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        h01.fetch_h01(tmp_path)
    assert path.read_bytes() == b"corrupt"


def test_archive_changed_after_open_is_detected(tmp_path, monkeypatch):
    path, _ = _archive(tmp_path, monkeypatch)
    archive = h01.H01Archive(path)
    with zipfile.ZipFile(path, "w") as changed:
        changed.writestr("12.0.swc", SOURCE.replace(b"200", b"300"))
    with pytest.raises(ValueError, match="SHA-256"):
        archive.load(12, component=0)


@pytest.mark.parametrize("fail", [False, True])
def test_failed_download_never_leaves_a_cache_entry(tmp_path, monkeypatch, fail):
    _archive(tmp_path, monkeypatch)

    def download(*args, **kwargs):
        if fail:
            raise OSError("offline")
        return io.BytesIO(b"invalid bytes")

    monkeypatch.setattr(h01.urllib.request, "urlopen", download)
    cache = tmp_path / "cache"
    with pytest.raises((OSError, ValueError)):
        h01.fetch_h01(cache)
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize("name", ["../escape.swc", "/absolute.swc", "12.a.swc"])
def test_unexpected_archive_paths_are_not_extracted(tmp_path, monkeypatch, name):
    path, _ = _archive(tmp_path, monkeypatch, [(name, SOURCE)])
    with pytest.raises(ValueError, match="Unexpected"):
        h01.H01Archive(path)


def test_duplicate_components_are_rejected(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="Duplicate"):
        path, _ = _archive(tmp_path, monkeypatch, [("12.0.swc", SOURCE)] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        h01.H01Archive(path)
