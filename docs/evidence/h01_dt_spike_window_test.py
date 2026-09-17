import numpy as np
import pytest

from docs.evidence.h01_dt_spike_window import build_report, compare_cells, spike_times, verdict


def _spike(length, at, dt, width_ms=.5, height=40.):
    trace = np.full(length, -70.)
    start = int(round(at/dt))
    trace[start:start+int(round(width_ms/dt))] = height
    return trace


def test_spike_times_counts_upward_crossings_only():
    trace = np.array([-70., -70., 5., 10., -30., -70., 2., -70.])
    assert spike_times(trace, .5).tolist() == [1., 3.]
    assert spike_times(np.full(4, 5.), .5).tolist() == []   # starts above: no crossing
    assert spike_times(np.full(4, -70.), .5).tolist() == []


def test_compare_cells_identical_traces_give_zero_delta_and_equal_counts():
    fine = np.stack([_spike(800, 20., .05), np.full(800, -70.)], axis=1)
    coarse = fine[7::8]
    result = compare_cells(fine, .05, coarse, .4, ['a', 'b'])
    assert result['max_abs_mv'] == 0.
    assert result['all_counts_equal'] and result['within_1mv']
    assert result['cells']['a']['reference_spikes'] == 1 and result['cells']['b']['reference_spikes'] == 0
    assert result['cells_spiking_reference'] == 1
    assert result['stride'] == 8 and result['compared_substeps'] == 100


def test_compare_cells_reports_shift_and_missing_spike():
    fine = np.stack([_spike(800, 20., .05), _spike(800, 10., .05)], axis=1)
    coarse = np.stack([_spike(100, 20.8, .4), np.full(100, -70.)], axis=1)
    result = compare_cells(fine, .05, coarse, .4, ['a', 'b'])
    assert result['cells']['a']['spike_shift_ms'] == pytest.approx([.8])
    assert result['cells']['b']['counts_equal'] is False
    assert not result['all_counts_equal'] and not result['within_1mv']
    assert result['max_abs_spike_shift_ms'] == pytest.approx(.8)


def test_compare_cells_rejects_non_multiple_timestep():
    with pytest.raises(ValueError):
        compare_cells(np.zeros((10, 1)), .3, np.zeros((5, 1)), .4, ['a'])


def test_verdict_picks_coarsest_passing():
    comparisons = {.005: dict(within_1mv=False, all_counts_equal=True),
                   .0025: dict(within_1mv=True, all_counts_equal=True),
                   .00125: dict(within_1mv=True, all_counts_equal=True)}
    result = verdict(comparisons)
    assert result['coarsest_passing_dt_ms'] == .0025
    assert result['failing_dt_ms'] == [.005]
    assert verdict({.005: dict(within_1mv=False, all_counts_equal=False)})['coarsest_passing_dt_ms'] is None


def test_build_report_joins_arms():
    fine = np.stack([_spike(800, 20., .05)], axis=1)
    arms = {.05: (dict(cells=['a'], compartments=3, settings=dict(dt_ms=.05), forward=dict(seconds_per_event_median=1.), stages={}), fine),
            .4: (dict(cells=['a'], compartments=3, settings=dict(dt_ms=.4), forward=dict(seconds_per_event_median=.1), stages={}), fine[7::8])}
    document = build_report(arms, .05)
    assert document['verdict']['coarsest_passing_dt_ms'] == .4
    assert document['reference_spikes'] == {'a': 1}
    assert '0.4' in document['comparisons']


def test_parity_report_same_dt(tmp_path):
    from docs.evidence.h01_dt_spike_window import parity_report
    fine = np.stack([_spike(800, 20., .05)], axis=1)
    report = dict(arm='a', cells=['a'], settings=dict(dt_ms=.05), forward=None)
    document = parity_report((report, fine), (dict(report, arm='b'), fine))
    assert document['comparison']['max_abs_mv'] == 0. and document['comparison']['all_counts_equal']
    with pytest.raises(ValueError):
        parity_report((report, fine), (dict(report, settings=dict(dt_ms=.1)), fine[1::2]))
