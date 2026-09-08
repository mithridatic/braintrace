"""Tests for the C3 edge-list assembly (pure functions; no network)."""

import io
import zipfile

import pytest

from docs.evidence.h01_c3_edge_list import (archive_cells, assemble_edges, nearest_label, neighbourhood, parse_swc,
                                            sample, to_c3_voxels)


def test_checkpoint_round_trip_and_key_mismatch(tmp_path):
    from docs.evidence.h01_c3_edge_list import load_checkpoint, save_checkpoint
    path = tmp_path/"x.cells.json"
    assert load_checkpoint(path, 10, "a.zip") == {}
    save_checkpoint(path, {"1": {"c3_ids": ["5"]}}, 10, "a.zip")
    assert load_checkpoint(path, 10, "a.zip") == {"1": {"c3_ids": ["5"]}}
    assert load_checkpoint(path, 11, "a.zip") == {}
    assert load_checkpoint(path, 10, "b.zip") == {}


def test_neighbourhood_is_nearest_first_and_nearest_label_reports_offset():
    box = neighbourhood((10, 10, 5))
    assert box[0] == (10, 10, 5) and len(box) == 5*5*3
    labels = {p: 0 for p in box}
    labels[(11, 10, 5)] = 7
    assert nearest_label(labels, (10, 10, 5), "7") == [1, 0, 0]
    assert nearest_label(labels, (10, 10, 5), "8") is None


def test_parse_swc_reads_positions_and_skips_comments():
    text = "# header\n0 1 100 200 10 1.5 -1\n1 3 104 200 10 1.0 0\n"
    assert parse_swc(text) == [(100., 200., 10.), (104., 200., 10.)]
    with pytest.raises(ValueError):
        parse_swc("0 1 100 200\n")


def test_to_c3_voxels_scales_xy_by_four_and_z_by_one():
    assert to_c3_voxels([(100., 200., 10.)]) == [(400, 800, 10)]


def test_sample_keeps_ends_and_respects_limit():
    points = list(range(10))
    assert sample(points, 20) == points
    chosen = sample(points, 4)
    assert len(chosen) <= 4 and chosen[0] == 0
    assert sample(points, 3)[0] == 0
    with pytest.raises(ValueError):
        sample(points, 0)


def test_archive_cells_groups_components_in_order(tmp_path):
    path = tmp_path/"a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("5.10.swc", "5.2.swc", "7.0.swc", "readme.txt"):
            archive.writestr(name, "")
    assert archive_cells(path) == {"5": ["5.2.swc", "5.10.swc"], "7": ["7.0.swc"]}


def test_assemble_edges_keeps_only_unambiguous_different_cell_pairs():
    pre = {"a": {"1"}, "b": {"1"}, "c": {"1", "2"}, "d": {"3"}}
    post = {"a": {"2"}, "b": {"1"}, "c": {"3"}, "e": {"1"}}
    assert assemble_edges(pre, post) == [("a", "1", "2")]
