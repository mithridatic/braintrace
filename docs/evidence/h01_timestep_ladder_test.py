"""Tests for the timestep ladder.

Two properties matter here. The comparison must never interpolate — on a trace that swings to
+384 mV, resampling across a spike edge invents tens of millivolts the solver never produced. And
a single passing pair partway down a diverging ladder must not count as convergence.
"""

import numpy as np
import pytest

import h01_timestep_ladder as tl


def _pair(n, amp_coarse, amp_fine):
    """A bisection pair: the fine grid has twice the samples and starts at half the step."""
    dt = 1. / n
    tc = dt * np.arange(1, n + 1)
    tf = (dt / 2.) * np.arange(1, 2 * n + 1)
    return (tc, np.full(n, amp_coarse)), (tf, np.full(2 * n, amp_fine))


def test_compare_reads_the_fine_trace_at_the_coarse_sample_times():
    coarse, fine = _pair(100, 10., 10.5)
    c = tl.compare(coarse, fine)
    assert c["max_error_mv"] == pytest.approx(.5)
    assert c["n_compared"] == 100


def test_compare_refuses_a_pair_that_is_not_a_bisection():
    (tc, vc), _ = _pair(100, 10., 10.)
    tf = np.linspace(0., 1., 137)                       # arbitrary grid, not a bisection
    assert tl.compare((tc, vc), (tf, np.zeros(137))) is None


def test_compare_never_interpolates_across_an_edge():
    """A spike between coarse samples must not appear in the error: only matched times are read."""
    n = 50
    dt = 1. / n
    tc = dt * np.arange(1, n + 1)
    tf = (dt / 2.) * np.arange(1, 2 * n + 1)
    vc = np.zeros(n)
    vf = np.zeros(2 * n)
    vf[::2] = 400.                                      # huge values ONLY at unmatched times
    c = tl.compare((tc, vc), (tf, vf))
    assert c["max_error_mv"] == 0.


def test_qualified_step_requires_every_finer_pair_to_pass():
    diverging = [{"dt_coarse_ms": .01, "passes_gate": False},
                 {"dt_coarse_ms": .005, "passes_gate": True},
                 {"dt_coarse_ms": .0025, "passes_gate": False}]
    assert tl.qualified_step(diverging) is None
    converging = [{"dt_coarse_ms": .01, "passes_gate": False},
                  {"dt_coarse_ms": .005, "passes_gate": True},
                  {"dt_coarse_ms": .0025, "passes_gate": True}]
    assert tl.qualified_step(converging) == .005
    assert tl.qualified_step([]) is None


def test_the_recovered_ladder_closes_at_the_step_the_campaign_was_already_using():
    """The four adjacent pairs, read from the restored traces."""
    r = tl.report()
    errs = [p["max_error_mv"] for p in r["pairs"]]
    assert len(errs) == 4
    assert errs == sorted(errs, reverse=True)           # monotone refinement
    assert errs[0] == pytest.approx(6.365, abs=.01)     # matches the committed decision JSON
    assert r["qualified_dt_ms"] == pytest.approx(.000625)
    assert "QUALIFIED" in r["verdict"]


def test_the_error_halves_with_the_step_first_order():
    """Adjacent error ratios near 2 are the first-order convergence the transfer study found."""
    errs = [p["max_error_mv"] for p in tl.ladder()]
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    assert all(1.6 < r < 2.4 for r in ratios), ratios


def test_the_report_says_what_it_supersedes():
    r = tl.report()
    assert tl.DECISION in r["supersedes"]
    assert r["gate_mv"] == 1.
