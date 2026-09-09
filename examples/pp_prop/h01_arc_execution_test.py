"""Unchanged ARC encoding, losses and exact decoding on multicompartment cells."""

from pathlib import Path
from types import SimpleNamespace

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np

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
        assert int(model.stepper.tick.value) == int(np.sum(advances))*20
        before = {name: np.array(value) for name, value in trainer.parameters.items()}
        logits, activity = brainstate.transform.jit(lambda e, a: score_episode(session, e, a))(events, advances)
        assert logits.shape == (31, 360) and np.isfinite(logits).all()
        assert np.isfinite(activity).all()
        exact, loss = direct_query_metrics(np.asarray(logits), grid, module.decode_prediction)
        assert isinstance(exact, bool) and np.isfinite(loss)
        assert int(model.stepper.tick.value) == int(np.sum(advances))*20
        for name, value in trainer.parameters.items():
            np.testing.assert_array_equal(value, before[name])
