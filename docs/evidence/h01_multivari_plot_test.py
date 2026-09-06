"""Tests for the phase-plane multivari and the Matryoshka family ranges."""

import numpy as np
import pytest

from docs.evidence.h01_multivari_plot import (
    cycle_panel, cycle_window, family_ranges, named_family, panel, render)


def _trace(scale=1.):
    time = np.arange(0., 200., .05)
    voltage = np.full_like(time, -80.)
    for centre in (30., 60., 90., 120., 150.):
        voltage += scale*100.*np.exp(-((time-centre)/.5)**2)-8.*np.exp(-((time-centre-2.)/1.5)**2)
    return time, voltage


def _row(key, phase, contrast, limit=1., source="human and model"):
    return {"key": key, "phase": phase, "contrast": contrast, "limit": limit, "limit_source": source}


def test_cycle_window_spans_first_start_to_last_minimum():
    rows = [{"cycle_start_ms": 1., "minimum_ms": 5.}, {"cycle_start_ms": 5., "minimum_ms": 9.}]
    assert cycle_window(rows) == (1., 9.)


def test_panels_draw_one_line_per_trace(tmp_path):
    import matplotlib.pyplot as plt
    traces = {"human": _trace(), "candidate": _trace(.8)}
    fig, (left, right) = plt.subplots(1, 2)
    panel(left, traces, (10., 190.), "early")
    cycle_panel(right, traces, (10., 190.))
    assert len(left.lines) == 2 and len(right.lines) == 2
    plt.close(fig)


def test_family_ranges_separate_the_four_families():
    rows = {"a": [_row("peak_mv", "early", 10.), _row("peak_mv", "early", 12.), _row("peak_mv", "late", 20.),
                  _row("peak_mv", "subthreshold", 99.), _row("cycle_ms", "early", 1., None),
                  _row("cycle_ms", "late", 500., .01, "model")],
            "b": [_row("peak_mv", "early", 4.), _row("peak_mv", "late", 4.)]}
    families = family_ranges(rows)
    assert set(families) == {"peak_mv"}
    peak = families["peak_mv"]
    assert peak["elemental"] == pytest.approx(7.)
    assert peak["cyclical"] == pytest.approx(2.)
    assert peak["structural"] == pytest.approx(6.)
    assert peak["temporal"] == pytest.approx(abs(np.mean([20., 4.])-np.mean([10., 12., 4.])))
    named, totals = named_family(families)
    assert named == "elemental" and totals["elemental"] == pytest.approx(7.)
    assert "Named family" in render("I", "candidate", families, named, totals)
