"""Meaningful channel-identity and original-sample boundary checks."""

import numpy as np
import pytest

from h01_paired_response import electrode_index, spike_windows


def test_electrode_identity_uses_labels_not_position():
    assert electrode_index(['IN 3', 'IN 0', 'IN 2'], ['mV'] * 3, 'IN2') == 2


@pytest.mark.parametrize('names,units,label', [
    (['IN 0'], [], 'IN0'), (['IN 0'], ['mV'], 'IN2'),
    (['IN 0', 'IN0'], ['mV', 'mV'], 'IN0'),
    (['IN 0'], ['pA'], 'IN0'), (['IN 0'], ['V'], 'IN0'),
])
def test_bad_electrode_mapping_fails(names, units, label):
    with pytest.raises(ValueError):
        electrode_index(names, units, label)


def source():
    t = np.arange(200) / 1000
    pre = np.full(200, -60.)
    pre[[5, 10, 25, 149, 150, 190]] = 10
    return t, pre, np.arange(200, dtype=float) - 80


def test_exact_source_samples_overlaps_and_boundaries():
    t, pre, post = source()
    result = spike_windows(t, pre, post, rate_hz=1000)
    np.testing.assert_array_equal(result['crossing_indices'], [5, 10, 25, 149, 190])
    np.testing.assert_array_equal(result['accepted_crossings'], [10, 25, 149])
    np.testing.assert_array_equal(result['boundary_crossings'], [5, 190])
    np.testing.assert_array_equal(result['indices'][0], np.arange(61))
    np.testing.assert_array_equal(result['indices'][-1], np.arange(139, 200))
    for name, original in [('time_s', t), ('pre_mv', pre), ('post_mv', post)]:
        np.testing.assert_array_equal(result[name], original[result['indices']])


def test_no_spike_has_no_invented_event():
    t, pre, post = source()
    result = spike_windows(t, pre * 0 - 60, post, rate_hz=1000)
    assert result['indices'].shape == (0, 61)
    assert result['post_mv'].shape == (0, 61)


@pytest.mark.parametrize('change', ['nan', 'shape', 'short', 'mismatch', 'clock'])
def test_invalid_source_rejected(change):
    t, pre, post = source()
    if change == 'nan':
        post[50] = np.nan
    elif change == 'shape':
        pre = pre[None, :]
    elif change == 'short':
        t, pre, post = t[:1], pre[:1], post[:1]
    elif change == 'mismatch':
        post = post[:-1]
    else:
        t[5] += .0001
    with pytest.raises(ValueError):
        spike_windows(t, pre, post, rate_hz=1000)


@pytest.mark.parametrize('settings', [dict(rate_hz=0), dict(rate_hz=np.nan),
    dict(rate_hz=1000, before_ms=-1), dict(rate_hz=1000, after_ms=.5)])
def test_invalid_grid_settings(settings):
    with pytest.raises(ValueError):
        spike_windows(*source(), **settings)
