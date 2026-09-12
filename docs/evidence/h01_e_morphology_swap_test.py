"""Tests for the SP15 stage-1 morphology-swap converter and scorer."""

import numpy as np
import pytest

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


def test_convert_collapses_the_soma_scales_units_and_labels_subtrees():
    text, summary = swap.convert(SWC)
    rows = swap.parse_swc(text)
    assert summary["soma_nodes_collapsed"] == 2 and summary["nodes_out"] == 6
    assert rows[0, 1] == 1 and rows[0, 6] == -1
    assert rows[0, 5] == pytest.approx(3.)  # largest soma radius, nm to um
    assert rows[0, 2] == pytest.approx(1005*.032)  # centroid of the two soma nodes
    by_x = {round(r[2]/.032): int(r[1]) for r in rows[1:]}
    # the x-branch (x 1100, 1300) is the longest cable and becomes apical; the y-branch basal; x 900 stays axon
    assert by_x[1100] == 4 and by_x[1300] == 4 and by_x[1000] == 3 and by_x[900] == 2
    assert sum(1 for r in rows[1:] if r[6] == 1) == 3  # three soma children attach to the new soma
    assert rows[:, 2:5].max() < 50.  # micrometres, not voxels
    held, _ = swap.convert(SWC, soma_radius_um=6.5)
    assert swap.parse_swc(held)[0, 5] == pytest.approx(6.5)


def test_convert_rejects_multiple_roots_and_missing_soma():
    with pytest.raises(ValueError):
        swap.convert("0 1 0 0 0 100 -1\n1 3 1 0 0 100 -1\n")
    with pytest.raises(ValueError):
        swap.convert("0 3 0 0 0 100 -1\n1 3 1 0 0 100 0\n")


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
