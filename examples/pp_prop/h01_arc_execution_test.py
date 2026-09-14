"""Unchanged ARC encoding, losses and exact decoding on multicompartment cells."""

from pathlib import Path
from types import SimpleNamespace

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_network_step_test import _network
from .arc_contracts import ARCTask, encode_episode
from .example21_arc_adapter import Example21ArcAdapter, SupervisedQuery, direct_query_metrics
from .h01_arc_model import H01ArcModel
from .h01_arc_execution import update_episode, score_episode


def test_encoded_arc_episode_updates_and_padding_does_not_advance(tmp_path):
    module = Example21ArcAdapter(Path('.'))._model()
    grid = np.array([[1]], dtype=np.uint8)
    task = ARCTask('00000001', ((grid, grid),), (grid,), (grid,), 'training')
    events, advances = encode_episode(task)
    query = SupervisedQuery(task.task_id, 0, events, advances, grid)
    payload = Example21ArcAdapter._episode_payload(query)
    with brainstate.environ.context(precision=64):
        model = H01ArcModel(_network(tmp_path, current_na=0), ['5805562981'])
        learner = braintrace.pp_prop.sparse(model)
        learner.compile_graph(jnp.zeros(441))
        trainer = module.PPPropEpisodeTrainer(learner, dict(input=model.input_weight.value,
            recurrent=model.recurrent_weight.value, readout_weight=model.readout_weight.value,
            readout_bias=model.readout_bias.value))
        session = SimpleNamespace(model=model, learner=learner, trainer=trainer)
        train = brainstate.transform.jit(lambda p: update_episode(session, p, module))
        result = train(jax.tree.map(jnp.asarray, payload))
        assert np.isfinite(np.asarray(result)).all() and int(trainer.updates) == 1
        assert int(model.stepper.tick.value) == int(np.sum(advances))*model.substeps
        before = {name: np.array(value) for name, value in trainer.parameters.items()}
        logits, activity = brainstate.transform.jit(lambda e, a: score_episode(session, e, a))(events, advances)
        assert logits.shape == (31, 360) and np.isfinite(logits).all()
        assert np.isfinite(activity).all()
        exact, loss = direct_query_metrics(np.asarray(logits), grid, module.decode_prediction)
        assert isinstance(exact, bool) and np.isfinite(loss)
        assert int(model.stepper.tick.value) == int(np.sum(advances))*model.substeps
        for name, value in trainer.parameters.items():
            np.testing.assert_array_equal(value, before[name])
        reference, reference_activity = brainstate.transform.jit(
            lambda e, a: original_score(session, e, a))(events, advances)
        np.testing.assert_allclose(logits, reference, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(activity, reference_activity)
        reference_exact, reference_loss = direct_query_metrics(np.asarray(reference), grid, module.decode_prediction)
        assert exact == reference_exact
        np.testing.assert_allclose(loss, reference_loss, rtol=1e-12, atol=1e-12)


class ScoreModel(brainstate.nn.Module):
    """Deterministic stateful fixture for the scoring execution contract."""

    def __init__(self, cells=104):
        super().__init__()
        self.voltage = brainstate.ShortTermState(jnp.full((cells,), -65., dtype=jnp.float64))
        self.ticks = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))
        self.readout_weight = brainstate.ParamState(
            jnp.sin(jnp.arange(cells*360, dtype=jnp.float64).reshape(cells, 360))*.01)
        self.readout_bias = brainstate.ParamState(jnp.linspace(-.1, .1, 360))

    def reset_episode(self, learner):
        self.voltage.value = jnp.full_like(self.voltage.value, -65.)
        self.ticks.value = jnp.asarray(0, dtype=jnp.int32)

    def step(self, event, advance):
        self.voltage.value = jnp.where(advance, self.voltage.value*.999+event[0]*.01,
                                      self.voltage.value)
        self.ticks.value = self.ticks.value+advance.astype(self.ticks.value.dtype)
        return self.voltage.value


def original_score(session, events, advances):
    """Frozen pre-optimization implementation used as the numerical oracle."""
    session.model.reset_episode(session.learner)

    def step(event, advance):
        voltage = session.model.step(event, advance)
        features = jnp.tanh((voltage+65.)/20.)
        activity = jnp.where(advance, jnp.abs(features), jnp.zeros_like(features))
        return features @ session.model.readout_weight.value+session.model.readout_bias.value, activity

    logits, activity = brainstate.transform.for_loop(step, jnp.asarray(events, dtype=jnp.float64), advances)
    return logits[-31:], jnp.sum(activity, axis=0)/jnp.maximum(jnp.sum(advances), 1)


@pytest.mark.parametrize('length,mask', [(1, 'all'), (17, 'none'), (31, 'all'),
                                        (193, 'mixed'), (2048, 'mixed')])
def test_score_matches_full_output_oracle(length, mask):
    with brainstate.environ.context(precision=64):
        model = ScoreModel()
        session = SimpleNamespace(model=model, learner=None)
        events = jnp.ones((length, 441))
        advances = (jnp.zeros(length, dtype=bool) if mask == 'none' else
                    jnp.ones(length, dtype=bool) if mask == 'all' else jnp.arange(length) % 3 != 0)
        expected = brainstate.transform.jit(lambda e, a: original_score(session, e, a))(events, advances)
        voltage, ticks = np.array(model.voltage.value), int(model.ticks.value)
        weights = np.array(model.readout_weight.value)
        actual = brainstate.transform.jit(lambda e, a: score_episode(session, e, a))(events, advances)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(model.voltage.value, voltage)
        np.testing.assert_array_equal(model.readout_weight.value, weights)
        assert int(model.ticks.value) == ticks == int(jnp.sum(advances))
        assert actual[0].shape == (min(length, 31), 360)


def test_query_scoring_reuses_trace_and_reads_current_parameters(monkeypatch):
    from . import h01_arc_execution as execution
    with brainstate.environ.context(precision=64):
        session = SimpleNamespace(model=ScoreModel(), learner=None)
        events, advances = jnp.ones((1, 31, 441)), jnp.ones((1, 31), dtype=bool)
        traces = []
        original = execution.score_episode

        def counted(*args):
            traces.append(1)
            return original(*args)

        monkeypatch.setattr(execution, 'score_episode', counted)
        before = execution.score_queries(session, events, advances)
        count = len(traces)
        session.model.readout_bias.value = session.model.readout_bias.value+1.
        after = execution.score_queries(session, events, advances)
        assert len(traces) == count
        np.testing.assert_allclose(after[0], before[0]+1., rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(after[1], before[1])
        other = SimpleNamespace(model=ScoreModel(), learner=None)
        fresh = execution.score_queries(other, events, advances)
        np.testing.assert_allclose(fresh[0], before[0], rtol=1e-12, atol=1e-12)
        assert len(traces) > count
