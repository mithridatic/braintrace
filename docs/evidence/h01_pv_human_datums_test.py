"""Direct crossing interpolation and complete-event boundaries."""

import numpy as np

from docs.evidence.h01_pv_human_datums import spike_datums


def test_crossings_use_physical_time_and_interpolate_threshold():
    events = spike_datums(np.array([0., 2., 3., 5.]), np.array([-30., 10., 20., -40.]))
    assert events == [{"rise_crossing_ms": .5, "fall_crossing_ms": 3.+4./3.,
                       "time_above_threshold_ms": 3.+4./3.-.5,
                       "peak_sample_time_ms": 3., "peak_sample_voltage_mv": 20.}]


def test_incomplete_edge_events_and_subthreshold_trace_are_excluded():
    assert spike_datums(np.arange(3.), np.array([0., -30., 0.])) == []
    assert spike_datums(np.arange(3.), np.array([-30., -25., -30.])) == []
