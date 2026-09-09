"""Real-cell training continuation, morphology restore and optimizer transport."""

from pathlib import Path

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_anatomy_test import imported
from braintrace.datasets.h01 import H01Archive
from .example21_arc_adapter import Example21ArcAdapter
from .h01_checkpoint import store_asset
from .h01_topology import H01Topology
from .h01_runtime_test import _manifest
from .h01_session import H01Session, numerical_settings


def test_real_update_save_restore_and_clone_optimizer(imported, tmp_path):
    trainer_type = Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
    assets = tmp_path/'assets'
    digest = store_asset(tmp_path/'archive.zip', assets)
    archive = H01Archive(assets/digest)
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        doc = _manifest(imported).to_dict()
        doc['sources']['12']['archive_sha256'] = digest
        session = H01Session.build(H01Topology.from_dict(doc), archive, trainer_type,
                                  assets=(digest,), progress=lambda _: None)
        def update():
            session.trainer.reset_episode()
            def loss(event):
                session.learner(event)
                return jnp.mean(jnp.square(session.model.readout()))
            return session.trainer.update_episode(jnp.zeros((1, 441)), step_fn=loss)
        result = brainstate.transform.jit(update)()
        assert np.isfinite(np.asarray(result)).all()
        assert int(session.trainer.updates) == 1
        path = tmp_path/'episode.npz'
        identity = session.save(path, stage_id='test-stage', continuation={'schedule_sha256': 'b'*64, 'cursor': 1})
        restored = H01Session.restore(path, assets, trainer_type, expected_sha256=identity, progress=lambda _: None)
        assert int(restored.trainer.updates) == 1
        for expected, actual in zip(jax.tree.leaves(session.trainer.muon_groups), jax.tree.leaves(restored.trainer.muon_groups)):
            np.testing.assert_array_equal(actual, expected)
        for name, value in session.trainer.parameters.items():
            np.testing.assert_array_equal(restored.trainer.parameters[name], value)
        event = jnp.zeros(441)
        np.testing.assert_array_equal(brainstate.transform.jit(restored.model.update)(event),
                                      brainstate.transform.jit(session.model.update)(event))
        parent_head = np.array(restored.trainer.parameters['readout_weight'][0])
        cloned = restored.mutate(restored.topology.clone('12', stage='clone'), archive,
                                 progress=lambda _: None, release_parent=True)
        assert cloned.model.neuron_count == 2 and int(cloned.trainer.updates) == 1
        assert restored.model is None and restored.trainer is None
        np.testing.assert_array_equal(cloned.trainer.parameters['readout_weight'][1], parent_head)
        for leaf in jax.tree.leaves(cloned.trainer.muon_groups['readout_weight']):
            if np.shape(leaf) == (2, 360):
                assert np.count_nonzero(np.asarray(leaf)[1]) == 0


@pytest.mark.parametrize('mismatch', ['dependencies', 'implementation', 'precision'])
def test_incompatible_runtime_fails_before_morphology_construction(mismatch):
    from .h01_topology_test import _topology
    settings = numerical_settings()
    if mismatch == 'dependencies':
        settings['dependencies']['jax'] = 'different'
    elif mismatch == 'implementation':
        settings['implementation_sha256'] = {}
    with brainstate.environ.context(precision=32 if mismatch == 'precision' else 64):
        with pytest.raises(ValueError, match='dependencies|implementation|precision'):
            H01Session.build(_topology(), None, None, settings=settings)
