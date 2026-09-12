"""Real cable/contact integration of opt-in stochastic release."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_network_step_test import _contact_network
from .h01_arc_model import H01ArcModel


def build(tmp_path, probability):
    return H01ArcModel(_contact_network(tmp_path), ['pre', 'post'],
                       release_probability=probability, checkpoint_substeps=False)


def test_probability_one_preserves_cable_forward(tmp_path):
    with brainstate.environ.context(precision=64):
        baseline, stochastic = build(tmp_path, None), build(tmp_path, [1.])
        event = jnp.zeros(441)
        expected = brainstate.transform.jit(baseline.update)(event)
        actual = brainstate.transform.jit(stochastic.update)(event)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert stochastic.release.tick.value == 20


def test_zero_probability_removes_contact_and_padding_preserves_stream(tmp_path):
    with brainstate.environ.context(precision=64):
        model = build(tmp_path, [0.])
        initial = model.release.rng.value
        step = brainstate.transform.jit(model.step)
        step(jnp.zeros(441), False)
        assert model.release.tick.value == 0 and jnp.all(model.release.rng.value == initial)
        step(jnp.zeros(441))
        assert all(not np.any(buffer.value.mantissa) for buffer in model.stepper.ring_buffers)
        model.reset_episode()
        assert model.release.tick.value == 0 and jnp.all(model.release.rng.value == initial)


def test_release_manifest_cardinality(tmp_path):
    with pytest.raises(ValueError, match='placed contacts'):
        build(tmp_path, [.5, .5])
