"""Tests for the Youden pairing of contract residuals."""

import pytest

from docs.evidence.h01_youden_plot import paired_residuals, plot, square_limits


def _report():
    def rows(values):
        return [{"key": k, "kind": "rise_crossing_ms", "residual": v} for k, v in values.items()]
    return {"records": [
        {"name": "c", "input": "I 0.19", "rows": rows({"e1_rise_crossing_ms": .5, "e2_rise_crossing_ms": None, "e3_rise_crossing_ms": 2.})},
        {"name": "c", "input": "I 0.27", "rows": rows({"e1_rise_crossing_ms": -.5, "e2_rise_crossing_ms": 1., "e4_rise_crossing_ms": 3.})},
        {"name": "other", "input": "I 0.19", "rows": rows({"e1_rise_crossing_ms": 0.})}]}


def test_pairs_only_rows_finite_at_both_inputs():
    pairs = paired_residuals(_report(), "c", ("I 0.19", "I 0.27"), "rise_crossing_ms")
    assert pairs == [("e1_rise_crossing_ms", .5, -.5)]


def test_missing_input_raises():
    with pytest.raises(ValueError):
        paired_residuals(_report(), "other", ("I 0.19", "I 0.27"), "rise_crossing_ms")


def test_square_limits_cover_points_and_allowance():
    assert square_limits([("k", .5, -.5)], 1.) == (-1.1, 1.1)
    assert square_limits([("k", 5., -.5)], 1.) == pytest.approx((-5.5, 5.5))


def test_plot_writes_a_file(tmp_path):
    output = tmp_path/"y.png"
    plot([("e1_rise_crossing_ms", .5, -.5)], 1., "t", output)
    assert output.stat().st_size > 0
