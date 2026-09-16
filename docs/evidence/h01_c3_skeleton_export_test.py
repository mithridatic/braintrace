"""Tests for the C3 skeleton export (synthetic skeleton, injected fetcher, no network)."""

import hashlib
import json
import re
import zipfile

import numpy as np
import pytest

from braintrace.datasets._h01_swc import normalize
from braintrace.datasets.h01 import H01Archive
from docs.evidence.h01_c3_skeleton_export import (Skeleton, cable_um, candidate_ids, choose_root, component_tree,
                                                  export_cell, label_to_type, main, sample_labels, split_components,
                                                  swc_text)

# Vertices (nm): component A = 0..4 (a Y with a soma), component B = 5..7 (a triangle: one cycle),
# component C = 8 (singleton). Radii pick the roots; labels come from the fake volume.
VERTICES = np.array([[0, 0, 0], [320, 0, 0], [640, 0, 0], [640, 320, 0], [640, 0, 330],
                     [10000, 0, 0], [10320, 0, 0], [10000, 320, 0],
                     [50000, 50000, 50000]], dtype=float)
EDGES = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [5, 6], [6, 7], [7, 5]])
RADIUS = np.array([100, 900, 300, 50, 50, 200, 400, 30, 5000], dtype=float)
LABELS = np.array([101, 103, 103, 100, 0, 101, 101, 99, 103])


class FakeFetcher:
    """Returns the synthetic skeleton for any id and counts the calls."""

    mip = 0
    resolution_nm = [64.0, 64.0, 66.0]

    def __init__(self):
        self.calls = 0

    def skeleton(self, c3_id):
        self.calls += 1
        return Skeleton(VERTICES.copy(), EDGES.copy(), RADIUS.copy())

    def labels(self, vertices_nm):
        return LABELS.copy()


def test_label_to_type_matches_the_proofread_code_table():
    assert [label_to_type(l) for l in (0, 100, 101, 103, 105, 1100, 1103)] == [-1, 0, 1, 3, 5, 1000, 1003]


def test_components_are_split_largest_first_and_singletons_kept_for_the_caller():
    parts = split_components(Skeleton(VERTICES, EDGES, RADIUS))
    assert [p.tolist() for p in parts] == [[0, 1, 2, 3, 4], [5, 6, 7], [8]]


def test_root_prefers_the_largest_soma_vertex_then_the_largest_radius():
    assert choose_root([101, 103, 103], [900, 100, 300]) == 2
    assert choose_root([101, 101, 0], [100, 900, 300]) == 1


def test_component_tree_roots_at_the_soma_and_drops_cycle_edges():
    skeleton = Skeleton(VERTICES, EDGES, RADIUS)
    order, parents, dropped = component_tree(skeleton, [0, 1, 2, 3, 4], LABELS)
    assert order[0] == 1 and parents[0] == -1 and dropped == 0
    assert sorted(order.tolist()) == [0, 1, 2, 3, 4]
    assert all(parents[i] < i for i in range(1, 5))
    order, parents, dropped = component_tree(skeleton, [5, 6, 7], LABELS)
    assert order[0] == 6 and dropped == 1 and sorted(order.tolist()) == [5, 6, 7]
    with pytest.raises(ValueError, match="not connected"):
        component_tree(skeleton, [0, 5], LABELS)


def test_swc_text_is_the_proofread_layout_in_skeleton_voxels():
    skeleton = Skeleton(VERTICES, EDGES, RADIUS)
    order, parents, _ = component_tree(skeleton, [0, 1, 2, 3, 4], LABELS)
    text = swc_text(skeleton, LABELS, order, parents)
    first = text.splitlines()[0].split()
    assert first == ["0", "3", "10.000000", "0.000000", "0.000000", "900.000000", "-1"]
    rows, converted = normalize(text.encode())
    assert rows.shape == (5, 7) and sorted(rows[:, 1].tolist()) == [-1.0, 0.0, 1.0, 3.0, 3.0]
    assert "# H01" in converted
    assert cable_um(VERTICES, order, parents) == pytest.approx((320 + 320 + 320 + 330) / 1000)
    assert cable_um(VERTICES, np.array([8]), [-1]) == 0.0


def test_sample_labels_downloads_each_chunk_once_and_indexes_locally():
    calls = []

    def download(lo, hi):
        calls.append(tuple(lo))
        block = np.zeros(tuple(hi - lo), dtype=np.int64)
        block[:] = lo[0] * 1000 + lo[1] * 10 + lo[2]
        block[1, 2, 3] += 7
        return block

    vertices = np.array([[64 * 1, 64 * 2, 66 * 3], [64 * 1 + 10, 64 * 2, 66 * 3],
                         [64 * 100, 64 * 200, 66 * 300]], dtype=float)
    labels = sample_labels(vertices, [64, 64, 66], [64, 64, 64], download, workers=2)
    assert labels.tolist() == [7, 7, 64000 + 1920 + 256]
    assert sorted(calls) == [(0, 0, 0), (64, 192, 256)]


def test_export_cell_record_counts_components_vertices_labels_and_cable():
    members, record = export_cell(FakeFetcher(), 42)
    assert sorted(members) == ["42.0.swc", "42.1.swc"]
    assert record["components"] == 2 and record["components_total"] == 3 and record["singletons_dropped"] == 1
    assert record["vertices"] == 9 and record["vertices_written"] == 8
    assert record["edges"] == 7 and record["edges_dropped"] == 1
    assert record["label_counts"] == {"0": 1, "99": 1, "100": 1, "101": 3, "103": 3}
    assert record["parts"][0]["soma_vertices"] == 2 and record["parts"][0]["root_label"] == 103
    assert record["parts"][1]["root_radius_nm"] == 400.0
    assert record["cable_um"] == pytest.approx(1.29 + (320 + 320 * np.sqrt(2)) / 1000)
    assert "non-proofread" in record["source"]
    for name, data in members.items():
        normalize(data)


def test_candidate_ids_accepts_lists_dicts_and_id_keys():
    assert candidate_ids([1, "2", {"c3_id": 3}, {"neuron_id": "2"}]) == ["1", "2", "3"]
    assert candidate_ids({"candidates": [{"id": 9}], "other": 1}) == ["9"]
    with pytest.raises(ValueError, match="no candidates"):
        candidate_ids({"x": []})
    with pytest.raises(ValueError, match="without a c3 id"):
        candidate_ids([{"name": "a"}])


def test_main_writes_a_loadable_archive_and_provenance_and_resumes_from_stage(tmp_path):
    (tmp_path / "cand.json").write_text(json.dumps({"candidates": [{"c3_id": "7"}]}))
    output, provenance = tmp_path / "c3.zip", tmp_path / "prov.json"
    fetcher = FakeFetcher()
    main(["--candidates", str(tmp_path / "cand.json"), "--ids", "8", "7",
          "--output", str(output), "--provenance", str(provenance)], fetcher=fetcher)
    assert fetcher.calls == 2
    document = json.loads(provenance.read_text())
    assert document["archive_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert [c["c3_id"] for c in document["cells"]] == ["7", "8"]
    assert document["label_resolution_nm"] == [64.0, 64.0, 66.0] and "non-proofread" in document["source"]
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert names == ["7.0.swc", "7.1.swc", "8.0.swc", "8.1.swc"]
    assert all(re.fullmatch(r"([0-9]+)\.([0-9]+)\.swc", n) for n in names)
    archive = H01Archive(output, expected_sha256=document["archive_sha256"], source="c3_test")
    assert archive.neuron_ids == ("7", "8") and archive.components(7) == (0, 1)
    loaded = archive.load(7, component=0)
    assert loaded.provenance["source"] == "c3_test" and loaded.source_rows.shape == (5, 7)
    first_digest = document["archive_sha256"]
    main(["--ids", "7", "8", "--output", str(output), "--provenance", str(provenance)], fetcher=fetcher)
    assert fetcher.calls == 2 and json.loads(provenance.read_text())["archive_sha256"] == first_digest


def test_main_requires_at_least_one_id(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "x.zip"), "--provenance", str(tmp_path / "p.json")], fetcher=FakeFetcher())
