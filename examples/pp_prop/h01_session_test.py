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
        def update(target):
            target.trainer.reset_episode()
            def loss(event):
                target.learner(event)
                return jnp.mean(jnp.square(target.model.readout()))
            return target.trainer.update_episode(jnp.zeros((1, 441)), step_fn=loss)
        result = brainstate.transform.jit(lambda: update(session))()
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
        first = brainstate.transform.jit(lambda: update(session))()
        resumed = brainstate.transform.jit(lambda: update(restored))()
        np.testing.assert_array_equal(first, resumed)
        for left, right in zip(jax.tree.leaves((session.trainer.parameters, session.trainer.muon_groups)),
                               jax.tree.leaves((restored.trainer.parameters, restored.trainer.muon_groups))):
            np.testing.assert_array_equal(left, right)
        parent_head = np.array(restored.trainer.parameters['readout_weight'][0])
        restored._score_queries_compiled = object()
        cloned = restored.mutate(restored.topology.clone('12', stage='clone'), archive,
                                 progress=lambda _: None, release_parent=True)
        assert cloned.model.neuron_count == 2 and int(cloned.trainer.updates) == 2
        assert restored.model is None and restored.trainer is None
        assert not hasattr(restored, '_score_queries_compiled')
        np.testing.assert_array_equal(cloned.trainer.parameters['readout_weight'][1], parent_head)
        for leaf in jax.tree.leaves(cloned.trainer.muon_groups['readout_weight']):
            if np.shape(leaf) == (2, 360):
                assert np.count_nonzero(np.asarray(leaf)[1]) == 0


@pytest.mark.parametrize('mismatch', ['dependencies', 'implementation', 'precision', 'optimizer'])
def test_incompatible_runtime_fails_before_morphology_construction(mismatch):
    from .h01_topology_test import _topology
    settings = numerical_settings()
    if mismatch == 'dependencies':
        settings['dependencies']['jax'] = 'different'
    elif mismatch == 'optimizer':
        settings.pop('optimizer_policy')
    elif mismatch == 'implementation':
        settings['implementation_sha256'] = {}
    with brainstate.environ.context(precision=32 if mismatch == 'precision' else 64):
        with pytest.raises(ValueError, match='dependencies|implementation|precision|optimizer'):
            H01Session.build(_topology(), None, None, settings=settings)


def test_saved_optimizer_layout_mismatch_prevents_compilation(monkeypatch):
    from types import SimpleNamespace as Obj
    from . import h01_session as subject
    from .h01_muon_test import fixture
    adapter,params=fixture()
    model=Obj(**{name:Obj(value=params[group]) for name,group in
                 [('input_weight','input'),('recurrent_weight','recurrent'),
                  ('readout_weight','readout_weight'),('readout_bias','readout_bias')]})
    monkeypatch.setattr(subject,'build_network',lambda *a,**k:(None,None))
    monkeypatch.setattr(subject,'init_h01_network_states',lambda *a,**k:None)
    monkeypatch.setattr(subject,'H01ArcModel',lambda *a,**k:model)
    monkeypatch.setattr(subject,'model_optimizer',lambda *a,**k:adapter)
    def forbidden(*a,**k):
        raise AssertionError('Learner must not be constructed for an invalid optimizer layout')
    monkeypatch.setattr(subject.braintrace.pp_prop,'sparse',forbidden)
    settings=numerical_settings()
    settings['optimizer']=adapter.metadata()
    settings['optimizer']['layouts']['input']['coordinates'][0][1]=99
    with brainstate.environ.context(precision=64):
        with pytest.raises(ValueError,match='optimizer layout'):
            H01Session.build(Obj(to_dict=lambda:dict(active_cells=['a','b'])),None,None,settings=settings)


def test_biological_session_physical_wait_restore_includes_eligibility(imported, tmp_path):
    from types import SimpleNamespace
    from .h01_physical_wait import PhysicalWait
    trainer_type = Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
    archive = SimpleNamespace(load=lambda identity, component: imported)
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        topology = _manifest(imported).clone('12', stage='clone')
        child = topology.to_dict()['active_cells'][-1]
        topology = topology.add_contact('12', child, stage='contact')
        settings = numerical_settings()
        settings['biology'] = dict(schema='h01-biology-release-v1', release_probability=[.36])
        session = H01Session.build(topology, archive, trainer_type, settings=settings)
        session.model.reset_episode(session.learner)
        wait = PhysicalWait(session.model, .0003, learner=session.learner)
        step = brainstate.transform.jit(lambda: wait.update(max_events=1))
        assert not step()
        path = tmp_path/'physical-session.npz'
        digest = session.save_physical(path, wait=wait)
        with pytest.raises(ValueError, match='v2'):
            session.save(tmp_path/'legacy.npz', stage_id='legacy')
        assert not step()
        expected = np.array(session.model._soma())
        expected_factors = [np.array(f) for f in session.learner.factors.value]
        session.restore_physical(path, wait=wait, expected_sha256=digest)
        assert wait.elapsed_events.value == 1
        assert session.model.release.tick.value == session.model.substeps
        assert not step()
        np.testing.assert_array_equal(session.model._soma(), expected)
        for value, saved in zip(session.learner.factors.value, expected_factors):
            np.testing.assert_array_equal(value, saved)
        wrong_wait = SimpleNamespace(model=None, learner=None)
        with pytest.raises(ValueError, match='different'):
            session.save_physical(path, wait=wrong_wait)


def test_mutation_rejects_stale_biological_maps_before_touching_parent():
    from types import SimpleNamespace
    session = object.__new__(H01Session)
    session.settings = dict(biology=dict(schema='h01-biology-release-v1', release_probability=[.36]))
    parent = SimpleNamespace(stepper=SimpleNamespace(chemistry=None))
    session.model = parent
    with pytest.raises(ValueError, match='biological maps'):
        session.mutate(None, None, release_parent=True)
    assert session.model is parent


def test_spatial_session_rebuild_restore_and_eligibility_replay(imported, tmp_path):
    from types import SimpleNamespace
    from .h01_runtime_test import _spatial_fixture
    from .h01_physical_wait import PhysicalWait
    from braintrace.datasets.h01_biology import H01SpatialManifest
    trainer_type = Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
    archive = SimpleNamespace(load=lambda identity, component: imported)
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        topology, biology, _ = _spatial_fixture(imported)
        settings = numerical_settings()
        settings['biology'] = biology
        session = H01Session.build(topology, archive, trainer_type, settings=settings)
        session.model.reset_episode(session.learner)
        wait = PhysicalWait(session.model, .0003, learner=session.learner)
        advance = brainstate.transform.jit(lambda: wait.update(max_events=1))
        assert not advance()
        path = tmp_path/'spatial.npz'
        digest = session.save_physical(path, wait=wait)
        assert not advance()
        expected = np.array(session.model._soma())
        factors = [np.array(value) for value in session.learner.factors.value]
        rebuilt = H01Session.build(topology, archive, trainer_type, settings=settings)
        resumed = PhysicalWait(rebuilt.model, .0003, learner=rebuilt.learner)
        rebuilt.restore_physical(path, wait=resumed, expected_sha256=digest)
        assert not brainstate.transform.jit(lambda: resumed.update(max_events=1))()
        np.testing.assert_array_equal(rebuilt.model._soma(), expected)
        for actual, wanted in zip(rebuilt.learner.factors.value, factors):
            np.testing.assert_array_equal(actual, wanted)
        assert rebuilt.settings['biology'] == H01SpatialManifest(biology, topology.to_dict()).to_dict()
        # derived: two events, each advancing one event's worth of substeps
        assert int(rebuilt.model.release.tick.value) == 2*rebuilt.model.substeps
        with pytest.raises(ValueError, match='biological maps'):
            rebuilt.mutate(topology, archive)


@pytest.mark.parametrize('schema', ['h01-biology-spines-v1', 'unsupported'])
def test_invalid_biology_rejected_before_spatial_session_build(imported, schema):
    with brainstate.environ.context(precision=64):
        settings = numerical_settings()
        settings['biology'] = dict(schema=schema)
        with pytest.raises(ValueError, match='manifest'):
            H01Session.build(_manifest(imported), None, None, settings=settings)
