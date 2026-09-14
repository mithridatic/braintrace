"""Reproducible stream, event, padding and spatial probability tests."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from .release import ReleaseState, probability_profile


def test_fixed_probabilities_and_padding():
    model = ReleaseState([0., 1., .4])
    step = brainstate.transform.jit(model.update)
    key = model.rng.value
    np.testing.assert_array_equal(step(jnp.ones(3), advance=False), [0., 0., 0.])
    assert jnp.all(model.rng.value == key)
    assert model.tick.value == 0
    output = step(jnp.ones(3))
    assert output[0] == 0 and output[1] == 1 and model.tick.value == 1
    model.reset_state()
    np.testing.assert_array_equal(step(jnp.ones(3)), output)


def test_stream_checkpoint_and_binomial_statistics():
    model = ReleaseState(np.full(64, .36))
    def run():
        return brainstate.transform.for_loop(lambda _: model.update(jnp.ones(64)), jnp.arange(256))
    first = brainstate.transform.jit(run)()
    assert abs(np.mean(first)-.36) < .02
    key, tick = model.rng.value, model.tick.value
    expected = brainstate.transform.jit(run)()
    model.rng.value, model.tick.value = key, tick
    np.testing.assert_array_equal(brainstate.transform.jit(run)(), expected)
    assert model.tick.value == 512


@pytest.mark.parametrize('kind', ['uniform', 'linear', 'bell'])
def test_profiles_preserve_actual_discrete_mean(kind):
    values = probability_profile([0., 100., 400., 763.3], kind=kind)
    assert np.mean(values) == pytest.approx(.36)
    assert np.all((values >= 0) & (values <= 1))


@pytest.mark.parametrize('kwargs', [dict(distance_um=[]), dict(distance_um=[-1.]),
    dict(kind='unknown'), dict(mean=2.), dict(width_um=0.), dict(kind='linear', mean=0.)])
def test_invalid_profile(kwargs):
    args = dict(distance_um=[0., 100.])
    args.update(kwargs)
    with pytest.raises(ValueError):
        probability_profile(**args)


def test_invalid_probability_and_event_shape():
    for p in ([np.nan], [-.1], [[.5]]):
        with pytest.raises(ValueError):
            ReleaseState(p)
    with pytest.raises(ValueError):
        ReleaseState([.5]).update(jnp.ones(2))
