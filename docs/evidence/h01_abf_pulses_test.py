"""Pulse-train boundary and command-associated spike regression tests."""

from types import SimpleNamespace
import numpy as np
import pytest

from h01_abf_pulses import command_from_epochs, commanded_crossings


def epochs():
    return SimpleNamespace(p1s=[0, 10, 41160], p2s=[10, 41160, 41200],
                           types=['Step', 'Pulse', 'Step'], levels=[0, 2500, 0],
                           pulsePeriods=[0, 10000, 0], pulseWidths=[0, 150, 0])


def test_final_pulse_is_retained_in_partial_final_period():
    e = epochs()
    e.p2s[1] = e.p1s[2] = 40160
    command = command_from_epochs(e)
    starts = np.flatnonzero((command[:-1] <= 0) & (command[1:] > 0))+1
    np.testing.assert_array_equal(starts, [10, 10010, 20010, 30010, 40010])
    assert np.count_nonzero(command == 2500) == 5*150


def test_associated_crossings_retain_unassigned_falling_phase_recrossing():
    command = np.zeros(30); command[5:10] = 10; command[20:25] = 10
    voltage = np.full(30, -60.); voltage[6:11] = 40; voltage[12] = .3; voltage[21:24] = 40
    assigned, extra = commanded_crossings(voltage, command)
    np.testing.assert_array_equal(assigned, [6, 21])
    np.testing.assert_array_equal(extra, [12])


@pytest.mark.parametrize('change', ['unsupported', 'gap', 'width', 'nan'])
def test_bad_epochs(change):
    e = epochs()
    if change == 'unsupported': e.types[1] = 'Ramp'
    elif change == 'gap': e.p1s[1] = 11
    elif change == 'width': e.pulseWidths[1] = 10001
    else: e.levels[1] = np.nan
    with pytest.raises(ValueError): command_from_epochs(e)


@pytest.mark.parametrize('voltage,command', [([0], [0]), ([0, 1], [0]),
    ([0, np.nan], [0, 1]), ([-60]*5, [0, 1, 1, 0, 0]),
    ([-60, 10, -60, 10, -60], [0, 1, 1, 1, 0])])
def test_invalid_or_ambiguous_commanded_events(voltage, command):
    with pytest.raises(ValueError): commanded_crossings(voltage, command)
