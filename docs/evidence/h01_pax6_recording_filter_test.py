"""Independent transfer-function and causal-response checks for the correction."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.signal import bessel, tf2ss, freqs

from docs.evidence.h01_pax6_recording_filter import (
    INITIAL, pole_pairs, bessel_impulse, bessel_exponential, predict_control,
)


def system():
    numerator, denominator = bessel(4, 4 * np.pi, analog=True, norm='mag')
    return tf2ss(numerator, denominator)


def test_impulse_against_state_space():
    a, b, c, _ = system()
    t = np.array([.0001, .02, .04, .1, .2, .5, 1., 5.])
    expected = np.array([(c @ expm(a * x) @ b).item() for x in t])
    np.testing.assert_allclose(bessel_impulse(t), expected, rtol=1e-6, atol=1e-11)
    np.testing.assert_array_equal(bessel_impulse([-1., 0.]), 0)


@pytest.mark.parametrize('rate', [0., .001, .5, 5., 25., 1000.])
def test_exponential_against_augmented_state_space(rate):
    a, b, c, _ = system()
    augmented = np.zeros((5, 5))
    augmented[:4, :4], augmented[:4, 4], augmented[4, 4] = a, b[:, 0], -rate
    t = np.array([.0001, .04, .1, .3, 1., 10.])
    expected = np.array([(c @ expm(augmented * x)[:4, 4]).item() for x in t])
    np.testing.assert_allclose(bessel_exponential(rate, t), expected, rtol=1e-6, atol=2e-11)
    np.testing.assert_array_equal(bessel_exponential(rate, [-1., 0.]), 0)


def test_dc_and_cutoff_normalization():
    alpha, beta, real, imag = pole_pairs()
    poles, residues = alpha + 1j * beta, real + 1j * imag
    omega = 4 * np.pi
    response = np.sum(residues / (1j * omega - poles)
                       + residues.conj() / (1j * omega - poles.conj()))
    np.testing.assert_allclose(abs(response), 1 / np.sqrt(2), rtol=1e-12)
    np.testing.assert_allclose(bessel_exponential(0., 1000.), 1., rtol=1e-12)


def test_pulse_superposition_and_late_leak():
    p = INITIAL.copy()
    t = np.array([0, 44.9, 45.1, 45.5, 50., 55.3, 60., 100.])
    result = predict_control(p, [[45., 55.04]], [[10., -10.]], np.zeros(len(t), int), t)
    on = predict_control(p, [[45.]], [[10.]], np.zeros(len(t), int), t)
    off = predict_control(p, [[55.04]], [[-10.]], np.zeros(len(t), int), t)
    np.testing.assert_allclose(result, on + off, atol=1e-12)
    np.testing.assert_allclose(on[-1], 10 * p[0], atol=1e-10)
    np.testing.assert_allclose(result[-1], 0, atol=1e-10)


@pytest.mark.parametrize('parameter', range(5))
def test_complex_step_derivatives(parameter):
    t = np.array([45.2, 45.5, 47., 55.4, 60.])
    p = INITIAL.astype(complex)
    p[parameter] += 1e-25j
    args = ([[45., 55.04]], [[10., -10.]], np.zeros(len(t), int), t)
    derivative = predict_control(p, *args).imag / 1e-25
    upper, lower = INITIAL.copy(), INITIAL.copy()
    upper[parameter] += 1e-5
    lower[parameter] -= 1e-5
    finite = (predict_control(upper, *args) - predict_control(lower, *args)) / 2e-5
    np.testing.assert_allclose(derivative, finite, rtol=2e-5, atol=1e-6)


def test_batched_edges_and_padding():
    t = np.array([45.3, 46., 45.3, 60.])
    result = predict_control(INITIAL, [[45., 55.04], [45., 55.04]],
                              [[10., -10.], [-10., 0.]], [0, 0, 1, 1], t)
    separate = predict_control(INITIAL, [[45.]], [[-10.]], [0, 0], t[2:])
    np.testing.assert_allclose(result[2:], separate, atol=1e-12)


@pytest.mark.parametrize('fault', ['p_shape', 'p_nan', 'p_bounds', 'edge_shape', 'edge_nan',
                                 'edge_order', 'edge_negative', 'step_shape', 'step_nan',
                                 'id_float', 'id_negative', 'id_missing', 'time_shape',
                                 'time_negative', 'time_nan', 'empty'])
def test_reject_malformed_inputs(fault):
    p, edges, steps, ids, t = INITIAL.copy(), np.array([[45., 55.04]]), np.array([[10., -10.]]), np.array([0]), np.array([46.])
    if fault == 'p_shape': p = p[:-1]
    elif fault == 'p_nan': p[0] = np.nan
    elif fault == 'p_bounds': p[3] = 0
    elif fault == 'edge_shape': edges = edges[0]
    elif fault == 'edge_nan': edges[0, 0] = np.nan
    elif fault == 'edge_order': edges[0, 0] = 60
    elif fault == 'edge_negative': edges[0, 0] = -1
    elif fault == 'step_shape': steps = steps[0]
    elif fault == 'step_nan': steps[0, 0] = np.inf
    elif fault == 'id_float': ids = np.array([0.])
    elif fault == 'id_negative': ids[0] = -1
    elif fault == 'id_missing': ids[0] = 1
    elif fault == 'time_shape': t = np.array([45., 46.])
    elif fault == 'time_negative': t[0] = -1
    elif fault == 'time_nan': t[0] = np.nan
    elif fault == 'empty': ids, t = np.array([], int), np.array([])
    with pytest.raises(ValueError):
        predict_control(p, edges, steps, ids, t)


@pytest.mark.parametrize('cutoff', [0., -1., np.nan])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError): pole_pairs(cutoff)


def test_invalid_forcing_and_phase():
    with pytest.raises(ValueError): bessel_exponential(-1, .1)
    with pytest.raises(ValueError): bessel_exponential(1, np.nan)
    with pytest.raises(ValueError): bessel_impulse(np.nan)
