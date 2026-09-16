"""Tests for the C3 control comparison (synthetic archives, no network)."""

import json
import zipfile

import pytest

from docs.evidence import h01_c3_skeleton_control as module


def _swc(codes, step=100.0):
    return "\n".join(f"{i} {c} {i*step:.6f} 0.000000 0.000000 400.000000 {i-1}" for i, c in enumerate(codes)).encode()


@pytest.fixture
def archives(tmp_path):
    proofread, c3 = tmp_path / "proofread.zip", tmp_path / "c3.zip"
    with zipfile.ZipFile(proofread, "w") as z:
        z.writestr("5.0.swc", _swc([3, 3, 1, 1]))
        z.writestr("5.1.swc", _swc([0, 0]))
        z.writestr("6.0.swc", _swc([3]))
    with zipfile.ZipFile(c3, "w") as z:
        z.writestr("5.0.swc", _swc([3, 1], step=200.0))
        z.writestr("notes.txt", b"ignored")
    return proofread, c3


def test_summarize_member_counts_nodes_soma_and_cable():
    summary = module.summarize_member(_swc([3, 3, 1, 1]))
    assert summary["nodes"] == 4 and summary["soma_nodes"] == 2 and summary["types"] == {"1": 2, "3": 2}
    assert summary["cable_um"] == pytest.approx(3 * 100 * 0.032)
    assert module.summarize_member(_swc([0]))["cable_um"] == 0.0
    with pytest.raises(ValueError, match="seven columns"):
        module.summarize_member(b"0 1 2\n")


def test_summarize_cell_aggregates_only_that_cells_members(archives):
    proofread, c3 = archives
    summary = module.summarize_cell(proofread, 5)
    assert summary["components"] == 2 and summary["nodes"] == 6 and summary["soma_nodes"] == 2
    assert summary["largest_component"] == 0 and summary["largest_has_soma"] and summary["soma_components"] == [0]
    assert summary["types"] == {"0": 2, "1": 2, "3": 2}
    with pytest.raises(KeyError, match="no member"):
        module.summarize_cell(c3, 6)


def test_compare_records_ratios_and_main_writes_the_control_json(archives, tmp_path):
    proofread, c3 = archives
    record = module.main(["--cell", "5", "--proofread", str(proofread), "--c3-archive", str(c3),
                          "--output", str(tmp_path / "control.json")])
    written = json.loads((tmp_path / "control.json").read_text())
    assert written == record and record["cell_id"] == "5" and "not a gate" in record["role"]
    ratios = record["c3_over_proofread"]
    assert ratios["components"] == 0.5 and ratios["nodes"] == pytest.approx(2 / 6)
    assert ratios["soma_nodes"] == 0.5 and ratios["cable_um"] == pytest.approx((200 * 0.032) / (4 * 100 * 0.032))
    zero = module.compare("1", {"components": 0, "nodes": 0, "soma_nodes": 0, "cable_um": 0.0},
                          {"components": 1, "nodes": 2, "soma_nodes": 0, "cable_um": 1.0})
    assert zero["c3_over_proofread"] == {"components": None, "nodes": None, "soma_nodes": None, "cable_um": None}
