"""Tests for the cycle-by-cycle tables (SP15 stage 6)."""

import numpy as np
import pytest

import h01_topographic_cycles as cyc


def _train(n_spikes, step_mv=2., first_th=-56., interval_ms=100.):
    """Synthetic train: a resting trace with n spikes whose thresholds step after spike 1."""
    t = np.arange(0., 3000., .02)
    v = np.full_like(t, -60.)   # no onset step: the spike must carry the steepest rise
    for k in range(n_spikes):
        onset = 1050.+k*interval_ms
        th = first_th+(step_mv if k else 0.)
        ramp = (t >= onset-10.) & (t < onset)        # slow approach, so the threshold is read at its end
        v[ramp] = -60.+(th+60.)*(t[ramp]-(onset-10.))/10.
        m = (t >= onset) & (t < onset+1.)
        v[m] = th+(40.-th)*np.sin(np.pi*(t[m]-onset))
        v[(t >= onset+1.) & (t < onset+4.)] = -70.
    return t, v


def test_shape_reports_the_step_and_the_levels_against_spike_1():
    table = [{"spike": i+1, "threshold_mv": -56.+(2. if i else 0.), "max_rise_v_s": 350.-(50. if i else 0.), "t_ms": 30.+100.*i}
             for i in range(6)]
    sh = cyc.shape(table)
    assert sh["spike_2"]["threshold_mv"] == pytest.approx(2.)
    assert sh["spike_2"]["rise_fraction"] == pytest.approx(300./350.)
    assert sh["spike_2"]["t_ms"] == pytest.approx(100.)
    assert sh["spike_5"]["t_ms"] == pytest.approx(400.)
    assert sh["last"]["threshold_mv"] == pytest.approx(2.)


def test_shape_is_none_for_fewer_than_two_spikes():
    assert cyc.shape([]) is None
    assert cyc.shape([{"spike": 1, "threshold_mv": -56., "max_rise_v_s": 350., "t_ms": 30.}]) is None


def test_cycle_table_counts_only_spikes_inside_the_pulse():
    t, v = _train(3)
    tab = cyc.cycle_table(t, v, (1020., 2020.))
    assert tab["count"] == 3
    assert [s["spike"] for s in tab["spikes"]] == [1, 2, 3]
    assert all(0. < s["t_ms"] < 1000. for s in tab["spikes"])
    assert tab["shape"]["spike_2"]["threshold_mv"] > 0.


def test_first_spike_spread_skips_sweeps_without_a_spike():
    tables = {"a": {"count": 1, "spikes": [{"threshold_mv": -55., "max_rise_v_s": 330.}]},
              "b": {"count": 0, "spikes": []},
              "c": {"count": 2, "spikes": [{"threshold_mv": -54., "max_rise_v_s": 334.}, {"threshold_mv": -52., "max_rise_v_s": 300.}]}}
    spread = cyc.first_spike_spread(tables)
    assert spread["sweeps_with_a_spike"] == 2
    assert spread["threshold_mv"] == {"min": -55., "max": -54., "sd": pytest.approx(np.std([-55., -54.], ddof=1))}
    assert spread["max_rise_v_s"]["max"] == 334.


def test_markdown_lists_every_table_and_marks_silent_sweeps():
    result = {"tables": {"E human": {"sweep 57": {"count": 0, "spikes": [], "shape": None},
                                     "sweep 53": {"count": 2, "spikes": [{"spike": 1, "threshold_mv": -56.4, "max_rise_v_s": 348., "t_ms": 36.},
                                                                         {"spike": 2, "threshold_mv": -55., "max_rise_v_s": 301., "t_ms": 47.}],
                                                  "shape": {"spike_2": {"threshold_mv": 1.4, "rise_fraction": .865, "t_ms": 11.},
                                                            "last": {"threshold_mv": 1.4, "rise_fraction": .865, "t_ms": 11.}}}}},
              "first_spike_spread": {"E": {"sweeps_with_a_spike": 1, "threshold_mv": {"min": -56.4, "max": -56.4, "sd": 0.},
                                           "max_rise_v_s": {"min": 348., "max": 348., "sd": 0.}}},
              "reading": ["reading line"]}
    text = cyc.markdown(result)
    assert "**sweep 57**: no spike" in text
    assert "| 2 | -55.0 | 301 | 47 |" in text
    assert "step at spike 2: +1.4 mV, rise x0.86, 11 ms after spike 1" in text
    assert text.rstrip().endswith("reading line")
