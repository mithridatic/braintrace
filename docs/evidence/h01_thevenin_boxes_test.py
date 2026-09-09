"""Thevenin boxes: numbers come from the committed JSON, both figures are written."""

import json

import h01_thevenin_boxes as boxes


def test_find_returns_first_nested_hit_or_none():
    assert boxes._find({"a": [{"b": 1}], "c": {"bias_na": .03}}, "bias_na") == .03
    assert boxes._find({"a": [1, 2]}, "zzz") is None


def test_numbers_match_the_evidence_records():
    n = boxes.read_numbers()
    assert 30. < n["bias_pa"] < 33.
    assert n["circuit_ns"] == 20. and n["human_ns"] == 3.1
    assert n["circuit_rev_mv"] == -80 and n["human_rev_mv"] == -75


def test_main_writes_two_figures_and_a_record(tmp_path):
    boxes.main(tmp_path)
    for name in ("h01-thevenin-i-clamp", "h01-thevenin-ie-pair"):
        assert (tmp_path/f"{name}.png").exists() and (tmp_path/f"{name}.svg").exists()
    record = json.loads((tmp_path/"h01-thevenin-boxes.json").read_text())
    assert record["numbers"]["circuit_ns"]/record["numbers"]["human_ns"] > 6
