"""Verify the signed intervention without changing source commands."""

import numpy as np
import pytest

from docs.evidence.h01_l2_input import stimulus_current


def test_float32_source_is_promoted_before_bias_addition():
    command = np.array([0., .05, .11], dtype=np.float32)
    bias = -.003711858945210089
    expected = command.astype(np.float64) + bias
    assert np.max(np.abs((command + bias).astype(np.float64) - expected)) > 1e-12
    np.testing.assert_array_equal(stimulus_current(command, bias, include_bias=True), expected)


def test_default_preserves_command_and_owns_result():
    command = np.array([0., .05, 0., .25, 0.])
    result = stimulus_current(command, -.003711858945210089)
    np.testing.assert_array_equal(result, command)
    result[0] = 1
    assert command[0] == 0


@pytest.mark.parametrize("bias", [-.003711858945210089, 0., .004])
def test_bias_applies_to_baseline_and_every_pulse(bias):
    command = np.array([0., .05, 0., .25, 0.])
    result = stimulus_current(command, bias, include_bias=True)
    np.testing.assert_allclose(result - command, bias, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(command, [0., .05, 0., .25, 0.])


@pytest.mark.parametrize("command,bias,selected", [
    ([], 0., False), ([[0., .1]], 0., False), ([np.nan], 0., False),
    ([np.inf], 0., False), ([0.], np.nan, False), ([0.], np.inf, True),
    ([0.], [.1], True), ([0.], 0., "yes"),
])
def test_invalid_inputs_are_rejected(command, bias, selected):
    with pytest.raises(ValueError):
        stimulus_current(command, bias, include_bias=selected)


def test_ramp_command_is_zero_outside_the_window_linear_inside_and_capped():
    from docs.evidence.h01_l2_input import ramp_command
    time = np.array([0., 1019.98, 1020., 1120., 1520., 2019.98, 2020., 3000.])
    ramp = ramp_command(time, 11.)          # 11 pA/ms
    np.testing.assert_allclose(ramp, [0., 0., 0., 1.1, 2., 2., 0., 0.])
    with pytest.raises(ValueError):
        ramp_command(time, 0.)
    with pytest.raises(ValueError):
        ramp_command(np.array([[0.]]), 1.)
