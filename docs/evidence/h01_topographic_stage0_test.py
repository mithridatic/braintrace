"""Tests for the SP15 stage-0 characterisation helpers."""

import numpy as np
import pytest

import h01_topographic_stage0 as topo


def test_capacitance_from_onset_recovers_a_known_rc_cell():
    time = np.arange(0., 50., .02)
    c_nf, r_mohm, step = .1, 100., .2  # tau = 10 ms; initial slope = I/C = 2 mV/ms
    v = -70.+step*r_mohm*(1.-np.exp(-np.clip(time-10., 0., None)/(r_mohm*c_nf)))*(time >= 10.)
    c = topo.capacitance_from_onset(time, v, step, 10., window_ms=.5)
    assert c == pytest.approx(c_nf, rel=.05)
    with pytest.raises(ValueError):
        topo.capacitance_from_onset(time, np.full_like(time, -70.), step, 10.)


def test_load_current_is_injected_minus_capacitive():
    time = np.arange(0., 10., .1)
    v = -70.+2.*time  # 2 mV/ms
    load = topo.load_current(time, v, np.full_like(time, .2), .05)
    assert np.allclose(load[6:-6], .2-.05*2.)


def test_phase_and_load_windows_pick_cycles_and_post_spike_trajectories():
    rows = [{"threshold_ms": 100.+50.*k, "minimum_ms": 110.+50.*k} for k in range(5)]
    names = [w[0] for w in topo.phase_windows(rows, (0., 1000.))]
    assert names == ["cycle 1", "cycle 2", "cycle 3", "cycle 5"]
    wins = topo.load_windows(rows, (0., 1000.))
    assert wins[0] == ("after spike 1", 110., 150.) and wins[1] == ("after last spike", 310., 998.)
    assert topo.load_windows(rows[:1], (0., 1000.)) == [("after spike 1", 110., 998.)]
    assert topo.load_windows([], (0., 1000.)) == []


def test_smooth_and_binned_median():
    time = np.arange(0., 10., .1)
    noisy = np.sin(time)+np.where(np.arange(time.size) % 2 == 0, .5, -.5)
    assert np.abs(topo.smooth(time, noisy, .5)-np.sin(time))[5:-5].max() < .15
    assert topo.smooth(time, noisy, .01) is noisy
    x, y = topo.binned_median(np.array([0., .1, .2, .3, .4, 1.5]), np.array([1., 2., 3., 4., 5., 9.]), np.array([0., 1., 2.]))
    assert list(x) == [.5, 1.5] and y[0] == 3. and np.isnan(y[1])


def test_ratio_handles_missing_and_zero_sigma():
    assert topo._ratio(2., 1.) == 2.
    assert topo._ratio(2., None) is None and topo._ratio(2., 0.) is None and topo._ratio(None, 1.) is None


def test_resample_puts_cvode_traces_on_a_uniform_grid_and_leaves_uniform_ones_alone():
    t = np.array([0., .01, .05, .2, .21, 1.])
    g, v, i = topo.resample(t, t*10., np.ones_like(t), step_ms=.02)
    assert np.allclose(np.diff(g), .02) and np.allclose(v, g*10.) and np.all(i == 1.)
    u = np.arange(0., 1., .02)
    assert topo.resample(u, u, u)[0] is u
