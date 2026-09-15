"""Tests for the SP15 stage-1 morphology-swap converter and scorer."""

import numpy as np
import pytest
import json
import zipfile
import subprocess
import sys
from pathlib import Path

import h01_e_morphology_swap as swap


SWC = """# test
0 1 1000 1000 1000 3000 -1
1 1 1010 1000 1000 2500 0
2 -1 1100 1000 1000 200 1
3 -1 1300 1000 1000 200 2
4 0 1000 1100 1000 150 0
5 5 1000 1150 1000 150 4
6 2 900 1000 1000 100 1
"""


def test_h01_dendrite_annotations_do_not_collapse_into_a_false_soma():
    # H01 code 3 is soma, 1 is dendrite and 2 is astrocyte, unlike standard SWC.
    source = '10 3 0 0 0 3000 -1\n20 1 1000 0 0 32 10\n30 1 2000 0 0 40 20\n'
    text, summary = swap.convert(source)
    rows = swap.parse_swc(text)
    assert len(rows) == 3
    assert rows[:, 1].tolist() == [0, 0, 0]
    assert rows[:, 2].tolist() == pytest.approx([0, 32, 64])
    assert rows[:, 5].tolist() == pytest.approx([3, .032, .04])
    assert rows[:, 6].tolist() == [-1, 1, 2]
    assert summary['source_annotation_by_node_id'] == {'10': 3, '20': 1, '30': 1}


def test_convert_preserves_source_nodes_edges_radii_and_annotations():
    text, summary = swap.convert(SWC)
    rows = swap.parse_swc(text)
    original = swap.parse_swc(SWC)
    assert summary['nodes_out'] == len(original) == len(rows)
    assert summary['soma_nodes_collapsed'] == 0
    assert np.all(rows[:, 1] == 0)
    reverse = summary['source_node_id_by_export_id']
    by_id = {int(row[0]): row for row in original}
    for row in rows:
        source_id = reverse[str(int(row[0]))]
        source = by_id[source_id]
        assert row[2:5] == pytest.approx(source[2:5] * swap.POSITION_UM)
        assert row[5] == pytest.approx(source[5] * .001)
        assert (-1 if row[6] == -1 else reverse[str(int(row[6]))]) == source[6]
        assert summary['source_annotation_by_node_id'][str(source_id)] == source[1]
    with pytest.raises(ValueError, match='soma substitution'):
        swap.convert(SWC, soma_radius_um=6.5)


def test_convert_rejects_multiple_roots_and_nonpositive_radius():
    with pytest.raises(ValueError):
        swap.convert("0 1 0 0 0 100 -1\n1 3 1 0 0 100 -1\n")
    with pytest.raises(ValueError):
        swap.convert("0 3 0 0 0 100 -1\n1 3 1 0 0 0 0\n")


def test_prepare_exports_anatomy_without_fabricated_donor(tmp_path):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('7.0.swc', SWC)
    output = tmp_path / 'source-preserved'
    swap.main(['prepare', '--cell-id', '7', '--archive', str(archive), '--out', str(output)])
    result = json.loads((output / 'anatomy.json').read_text())
    assert result['conversion']['nodes_out'] == 7
    assert result['physiology'] == 'unqualified'
    assert not (output / 'donor.json').exists()
    with pytest.raises(FileExistsError):
        swap.prepare('7', 0, output, archive=archive)


def test_prepare_cli_does_not_require_scorer_imports(tmp_path):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('7.0.swc', SWC)
    command = [sys.executable, str(Path(swap.__file__).resolve()), 'prepare',
               '--cell-id', '7', '--archive', str(archive), '--out', str(tmp_path / 'out')]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_parse_and_bfs_reject_malformed_geometry():
    with pytest.raises(ValueError):
        swap.parse_swc('1 2 3')
    with pytest.raises(ValueError):
        swap.bfs_order(np.array([1, 2]), np.array([-1, -1]))
    with pytest.raises(ValueError):
        swap.bfs_order(np.array([1, 2, 3]), np.array([-1, 3, 2]))


def test_decide_reads_mesh_control_then_anatomy_sensitivity():
    ctl = {"max_rise_v_s": 590., "count": 4, "first_spike_ms": 127., "threshold_mv": -57.}
    same = swap.decide(ctl, dict(ctl, max_rise_v_s=600.))
    assert same["mesh_ok"] and same["verdict"].startswith("function")
    moved = swap.decide(ctl, dict(ctl, max_rise_v_s=700.))
    assert moved["verdict"].startswith("inputs") and moved["delta_v_s"] == pytest.approx(110.)
    bad = swap.decide(dict(ctl, max_rise_v_s=400.), ctl)
    assert not bad["mesh_ok"] and "mesh" in bad["verdict"]
    silent = swap.decide(ctl, dict(ctl, max_rise_v_s=None, count=0))
    assert "did not spike" in silent["verdict"]


def _series(*pairs):
    return [{"dose": d, "upstroke": {"max_rise_v_s": r, "count": 4 if r else 0}} for d, r in pairs]


def test_graded_decide_reads_a_monotone_departure_as_the_input_branch():
    out = swap.graded_decide(_series(("x1", 653.), ("x1.5", 520.), ("x2", 430.), ("x3", 360.)))
    assert out["band_v_s"] == pytest.approx(3*abs(653.-swap.B3_NSEG9_RISE_V_S))
    assert out["monotone"] and out["verdict"].startswith("inputs")
    assert [r["outside_band"] for r in out["rows"]] == [False, False, True, True]


def test_graded_decide_reads_a_held_band_as_the_function_branch():
    out = swap.graded_decide(_series(("x1", 653.), ("x1.5", 640.), ("x2", 631.), ("x3", None)))
    assert out["doses_silent"] == ["x3"] and out["doses_spiking"] == ["x1", "x1.5", "x2"]
    assert out["verdict"].startswith("function") and "silence" in out["verdict"]
    held = swap.graded_decide(_series(("x1", 653.), ("x1.5", 640.), ("x2", 631.)))
    assert held["verdict"].endswith("across the whole series")


def test_graded_decide_flags_a_non_monotone_departure_and_a_missing_control():
    out = swap.graded_decide(_series(("x1", 653.), ("x1.5", 900.), ("x2", 400.)))
    assert not out["monotone"] and "not monotone" in out["verdict"]
    assert swap.graded_decide([])["verdict"] == "no control reading"
    assert swap.graded_decide(_series(("x1", None)))["verdict"] == "no control reading"


def test_graded_decide_refuses_a_branch_when_only_the_control_spikes():
    out = swap.graded_decide(_series(("x1", 653.), ("x1.5", None), ("x2", None), ("x3", None)))
    assert out["verdict"].startswith("no reading")
    assert out["doses_spiking"] == ["x1"] and out["doses_silent"] == ["x1.5", "x2", "x3"]


def test_graded_decide_rejects_a_control_that_matches_the_number_but_not_the_response():
    series = _series(("x1", 697.), ("x1.5", 607.), ("x2", 549.))
    series[0]["upstroke"]["count"] = 72
    out = swap.graded_decide(series, control_reference=639.1, reference_count=10)
    assert out["verdict"].startswith("control invalid") and "72" in out["verdict"]
    assert "rows" not in out
    series[0]["upstroke"]["count"] = 10
    ok = swap.graded_decide(series, control_reference=639.1, reference_count=10)
    assert ok["verdict"].startswith("function") and ok["control_count"] == 10


def test_upstroke_is_measured_on_a_uniform_grid(tmp_path):
    uniform = np.arange(1000., 2100., .02)
    spike = -70.+110.*np.exp(-((uniform-1120.)/.35)**2)
    adaptive = np.unique(np.concatenate([np.arange(1000., 2100., .5), np.arange(1119., 1122., .002)]))
    both = {}
    for name, t in (("uniform", uniform), ("adaptive", adaptive)):
        v = -70.+110.*np.exp(-((t-1120.)/.35)**2)
        p = tmp_path/f"{name}.npz"
        np.savez(p, time_ms=t, voltage_mv=v)
        both[name] = swap.upstroke(p)["max_rise_v_s"]
    assert both["adaptive"] == pytest.approx(both["uniform"], rel=.02)
