"""Ordered compiled callbacks retain the last completed episode after interruption."""

from types import SimpleNamespace

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from . import h01_episode_recovery as recovery
from .h01_checkpoint_test import _save
from .h01_checkpoint import load_checkpoint


class _Trainer:
    def __init__(self, parameters):
        self.params = brainstate.State(parameters)
        self.count = brainstate.State(jnp.asarray(0, dtype=jnp.int32))

    @property
    def parameters(self):
        return self.params.value

    @property
    def updates(self):
        return self.count.value

    @property
    def muon_groups(self):
        return {'count': self.count.value}


def test_compiled_episode_failure_replays_from_saved_parent(tmp_path, monkeypatch):
    path, digest, _ = _save(tmp_path)
    loaded = load_checkpoint(path, asset_root=tmp_path/'assets')
    with brainstate.environ.context(precision=64):
        trainer = _Trainer({k: jnp.asarray(v) for k, v in loaded['parameters'].items()})
        session = SimpleNamespace(trainer=trainer, topology=loaded['topology'], assets=(digest,),
            settings=loaded['metadata']['settings'], model=SimpleNamespace(input_csr=SimpleNamespace(
                indices=loaded['input_indices'], indptr=loaded['input_indptr'])))
        def update(session, payload, module):
            del payload, module
            session.trainer.count.value += 1
            session.trainer.params.value = dict(session.trainer.parameters,
                readout_bias=session.trainer.parameters['readout_bias']+1.)
            return jnp.asarray(1.), jnp.asarray(2.)
        monkeypatch.setattr(recovery, 'update_episode', update)
        original = recovery.save_checkpoint
        def interrupted(path, **kwargs):
            if kwargs['continuation']['cursor'] == 2:
                raise RuntimeError('simulated interruption before episode commit')
            return original(path, **kwargs)
        monkeypatch.setattr(recovery, 'save_checkpoint', interrupted)
        directory = tmp_path/'episodes'
        arguments = dict(directory=directory, stage_id='stage', parent_sha256='a'*64, schedule_sha256='b'*64)
        with pytest.raises(Exception, match='simulated interruption'):
            recovery.run_episodes(session, {'events': jnp.zeros((2, 1, 441))}, None, **arguments)
        checkpoint = recovery.recovery_checkpoint(directory, stage_id='stage', parent_sha256='a'*64,
            schedule_sha256='b'*64, asset_root=tmp_path/'assets')
        assert checkpoint[2] == 1
        saved = load_checkpoint(checkpoint[0], asset_root=tmp_path/'assets')
        np.testing.assert_array_equal(saved['parameters']['readout_bias'], np.ones(360))
        trainer.params.value = {k: jnp.asarray(v) for k, v in saved['parameters'].items()}
        trainer.count.value = jnp.asarray(1, dtype=jnp.int32)
        monkeypatch.setattr(recovery, 'save_checkpoint', original)
        recovery.run_episodes(session, {'events': jnp.zeros((1, 1, 441))}, None,
            **arguments, start_cursor=1, previous_sha256=checkpoint[1])
        checkpoint = recovery.recovery_checkpoint(directory, stage_id='stage', parent_sha256='a'*64,
            schedule_sha256='b'*64, asset_root=tmp_path/'assets')
        assert checkpoint[2] == 2
        saved = load_checkpoint(checkpoint[0], asset_root=tmp_path/'assets')
        np.testing.assert_array_equal(saved['parameters']['readout_bias'], np.full(360, 2.))
        with pytest.raises(ValueError, match='another parent'):
            recovery.recovery_checkpoint(directory, stage_id='stage', parent_sha256='c'*64,
                schedule_sha256='b'*64, asset_root=tmp_path/'assets')


def test_no_committed_episode_has_no_recovery(tmp_path):
    assert recovery.recovery_checkpoint(tmp_path, stage_id='s', parent_sha256='a'*64,
        schedule_sha256='b'*64, asset_root=tmp_path) is None
