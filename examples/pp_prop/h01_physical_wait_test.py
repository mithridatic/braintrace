"""Physical waits preserve padding, replay and exact cable clocks."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_network_step_test import _network
from .h01_arc_model import H01ArcModel
from .h01_physical_wait import PhysicalWait, wait_event_count


def test_requested_seconds_use_physical_clock():
    assert [wait_event_count(x) for x in (0., 1., 10., 20.)] == [0, 10000, 100000, 200000]
    for bad in (-1, float('nan'), .00015):
        with pytest.raises(ValueError):
            wait_event_count(bad)


def test_wait_chunks_match_uninterrupted_silence_and_completion_is_padding(tmp_path):
    model = H01ArcModel(_network(tmp_path), ['1'])
    model.reset_episode()
    wait = PhysicalWait(model, .0003)
    chunk = brainstate.transform.jit(lambda: wait.update(max_events=2))
    assert not chunk() and model.stepper.tick.value == 40
    assert chunk() and model.stepper.tick.value == 60
    first = np.asarray(model._soma())
    assert chunk() and model.stepper.tick.value == 60
    assert not np.any(model.drive.value)
    model.reset_episode()
    brainstate.transform.for_loop(lambda _: model.update(jnp.zeros(441)), jnp.arange(3))
    np.testing.assert_array_equal(first, model._soma())
    zero = PhysicalWait(model, 0.)
    assert brainstate.transform.jit(zero.update)()
    assert model.stepper.tick.value == 60
    with pytest.raises(ValueError):
        wait.update(max_events=0)


def test_wait_uses_learner_on_each_real_event(tmp_path):
    model = H01ArcModel(_network(tmp_path), ['1'])
    class Learner(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = brainstate.ShortTermState(jnp.asarray(0))
        def update(self, event):
            self.calls.value += 1
            return model.update(event)
    learner = Learner()
    wait = PhysicalWait(model, .0003, learner=learner)
    assert brainstate.transform.jit(wait.update)()
    assert learner.calls.value == 3 and model.stepper.tick.value == 60
