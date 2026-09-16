"""Pinned annotation assets, metadata identity, and synapse unit tests."""

import gzip
import hashlib
import io
import json

import numpy as np
import pytest

from . import h01_annotations as annotations

CSV = ("104,prepost,x,y,z,prex,prey,prez,postx,posty,postz\n"
       "12,post,800,1,0,800,2,0,800,0,0\n"
       "12,pre,100,200,300,101,202,303,104,205,306\n"
       "13,post,100,200,300,101,202,303,104,205,306\n")
METADATA = {"@type": "neuroglancer_segment_properties", "inline": {
    "ids": ["12", "13"], "properties": [
        {"id": "tags", "type": "tags", "tags": ["L2", "pyramidal", "L3", "interneuron"], "values": [[0, 1], [2, 3]]},
        {"id": "NSI", "type": "number", "values": [388, 44], "description": "Number of incoming synapses"},
        {"id": "NSIe", "type": "number", "values": [138, 22]},
        {"id": "NSIi", "type": "number", "values": [250, 22]},
    ]}}


def _assets(tmp_path, monkeypatch, *, metadata=None, csv=CSV):
    payloads = {"cell_properties.json": json.dumps(METADATA if metadata is None else metadata).encode(),
                "synapse_locations.csv": csv.encode()}
    assets = {name: (annotations.ASSETS[name][0], hashlib.sha256(data).hexdigest()) for name, data in payloads.items()}
    monkeypatch.setattr(annotations, "ASSETS", assets)
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    return payloads


def test_metadata_selection_and_counts_are_not_csv_counts(tmp_path, monkeypatch):
    _assets(tmp_path, monkeypatch)
    store = annotations.H01Annotations(tmp_path)
    assert store.select() == ("12", "13")
    assert store.select("L2", "pyramidal") == ("12",)
    assert store.select("L2", "interneuron") == ()
    cell = store.metadata(12)
    assert cell.tags == ("L2", "pyramidal")
    assert cell.measurements["NSI"] == 388
    assert len(store.synapses(12, role="post")) == 1
    assert cell.descriptions["NSI"] == "Number of incoming synapses"
    assert cell.provenance["sha256"] == annotations.ASSETS["cell_properties.json"][1]
    with pytest.raises(TypeError):
        cell.measurements["NSI"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        store.select("layer2")
    with pytest.raises(KeyError):
        store.metadata(99)


def test_synapse_units_roles_and_source_rows(tmp_path, monkeypatch):
    _assets(tmp_path, monkeypatch)
    store = annotations.H01Annotations(tmp_path)
    rows = store.synapses(12)
    assert [r.source_row for r in rows] == [2, 3]
    assert [r.role for r in rows] == ["post", "pre"]
    np.testing.assert_allclose(rows[0].position_um, [6.4, 0, 0])
    np.testing.assert_allclose(rows[1].position_um, [101 * .008, 202 * .008, 303 * .033])
    np.testing.assert_allclose(rows[1].center_um, [.8, 1.6, 9.9])
    assert store.synapses(12, role="pre") == (rows[1],)
    assert "ei_type" in store.synapse_provenance["missing_fields"]
    with pytest.raises(ValueError, match="role"):
        store.synapses(12, role="incoming")


def test_gzip_download_and_offline_cache(tmp_path, monkeypatch):
    payloads = _assets(tmp_path, monkeypatch)
    requests = []

    def download(url, timeout):
        requests.append((url, timeout))
        name = next(name for name, (source, _) in annotations.ASSETS.items() if url.endswith(source))
        return io.BytesIO(gzip.compress(payloads[name]) if name.endswith("json") else payloads[name])

    monkeypatch.setattr(annotations.urllib.request, "urlopen", download)
    cache = tmp_path / "cache"
    assert annotations.fetch_h01_annotations(cache, timeout=9).select() == ("12", "13")
    assert len(requests) == 2 and requests[0][1] == 9
    annotations.fetch_h01_annotations(cache)
    assert len(requests) == 2
    assert set(p.name for p in cache.iterdir()) == set(payloads)
    (cache / "cell_properties.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        annotations.fetch_h01_annotations(cache)
    assert (cache / "cell_properties.json").read_bytes() == b"corrupt"


def test_failed_download_and_changed_csv(tmp_path, monkeypatch):
    _assets(tmp_path, monkeypatch)
    monkeypatch.setattr(annotations.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="SHA-256"):
        annotations.fetch_h01_annotations(tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []
    store = annotations.H01Annotations(tmp_path)
    (tmp_path / "synapse_locations.csv").write_text("changed")
    with pytest.raises(ValueError, match="SHA-256"):
        store.synapses(12)


@pytest.mark.parametrize("mutation", ["type", "duplicate_ids", "duplicate_props", "length", "tag", "tag_type", "nan", "bool"])
def test_invalid_metadata(tmp_path, monkeypatch, mutation):
    data = json.loads(json.dumps(METADATA))
    inline = data["inline"]
    props = inline["properties"]
    if mutation == "type": data["@type"] = "wrong"
    elif mutation == "duplicate_ids": inline["ids"] = ["12", "12"]
    elif mutation == "duplicate_props": props.append(props[0])
    elif mutation == "length": props[1]["values"] = []
    elif mutation == "tag": props[0]["values"][0] = [-1]
    elif mutation == "tag_type": props[0]["type"] = "number"
    elif mutation == "nan": props[1]["values"][0] = float("nan")
    elif mutation == "bool": props[1]["values"][0] = True
    _assets(tmp_path, monkeypatch, metadata=data)
    with pytest.raises(ValueError):
        annotations.H01Annotations(tmp_path)


@pytest.mark.parametrize("csv", [CSV.replace("104,", "id,", 1), CSV.replace("12,post", "12,wrong"),
                                  CSV.replace("800,1,0", "nan,1,0"), CSV.replace("800,1,0", "-1,1,0")])
def test_invalid_synapse_rows(tmp_path, monkeypatch, csv):
    _assets(tmp_path, monkeypatch, csv=csv)
    with pytest.raises(ValueError):
        annotations.H01Annotations(tmp_path).synapses(12)


def test_segment_properties_table_reads_any_release_and_has_no_synapses(tmp_path):
    path = tmp_path / "c3-segment-properties.json"
    payload = json.dumps(METADATA).encode()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    table = annotations.H01SegmentProperties(path, expected_sha256=digest.upper(), release="20210601/c3")
    assert table.sha256 == digest and table.select("L3", "interneuron") == ("13",)
    cell = table.metadata(12)
    assert cell.tags == ("L2", "pyramidal") and cell.measurements["NSI"] == 388
    assert cell.provenance["sha256"] == digest and cell.provenance["release"] == "20210601/c3"
    assert "non-proofread" in cell.provenance["verification"]
    assert table.synapse_provenance["synapse_table"].startswith("not supplied")
    with pytest.raises(LookupError):
        table.synapses(12)
    with pytest.raises(KeyError):
        table.metadata("99")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        annotations.H01SegmentProperties(path, expected_sha256="0" * 64)
    unpinned = annotations.H01SegmentProperties(path)
    assert unpinned.sha256 == digest
    path.write_text(json.dumps({"@type": "wrong"}))
    with pytest.raises(ValueError, match="segment properties"):
        annotations.H01SegmentProperties(path)
