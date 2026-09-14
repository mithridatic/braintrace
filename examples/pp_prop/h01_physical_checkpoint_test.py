"""Physical snapshot replay, typed RNG keys and fail-before-mutation checks."""

import json
from collections import namedtuple

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_network_step_test import _contact_network
from braintrace.biophysics.environment_test import make_environment
from .h01_arc_model import H01ArcModel
from .h01_physical_wait import PhysicalWait
from .h01_physical_checkpoint import save_physical_checkpoint, restore_physical_checkpoint, _pack
from .h01_checkpoint import _digest_file


Slots = namedtuple('Slots', 'moment count')


def fixture(tmp_path):
    model = H01ArcModel(_contact_network(tmp_path), ['1', '2'], release_probability=[.36])
    model.reset_episode()
    wait = PhysicalWait(model, .0005)
    chemistry = make_environment()
    roots = dict(model=model, wait=wait, chemistry=chemistry)
    return roots, Slots(jnp.array([.1, .2]), jnp.asarray(4))


def test_midwait_fresh_runtime_replay_preserves_every_state_and_optimizer(tmp_path):
    with brainstate.environ.context(precision=64):
        roots, optimizer = fixture(tmp_path)
        wait, chemistry = roots['wait'], roots['chemistry']
        brainstate.transform.jit(lambda: wait.update(max_events=2))()
        brainstate.transform.jit(chemistry.update)(jnp.array([.01, .02, .03]), jnp.array([3000., 0.]))
        manifest = dict(dt_ms=.005, geometry='test-cable-explicit', probability=[.36])
        path = tmp_path/'physical.npz'
        digest = save_physical_checkpoint(path, roots=roots, optimizer=optimizer, manifest=manifest)
        assert wait.elapsed_events.value == 2 and roots['model'].stepper.tick.value == 2 * roots['model'].substeps
        brainstate.transform.jit(lambda: wait.update(max_events=3))()
        expected, _, _ = _pack(roots, optimizer)
        fresh, template = fixture(tmp_path)
        restored_optimizer = restore_physical_checkpoint(path, roots=fresh, optimizer=template,
            manifest=manifest, expected_sha256=digest)
        assert isinstance(restored_optimizer, Slots)
        assert fresh['wait'].elapsed_events.value == 2
        assert fresh['model'].stepper.tick.value == fresh['model'].release.tick.value == 2 * fresh['model'].substeps
        brainstate.transform.jit(lambda: fresh['wait'].update(max_events=3))()
        actual, _, _ = _pack(fresh, restored_optimizer)
        assert actual.keys() == expected.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_corrupt_or_mismatched_snapshots_do_not_mutate_runtime(tmp_path):
    roots, optimizer = fixture(tmp_path)
    manifest = {'identity': 'geometry-and-rates'}
    path = tmp_path/'physical.npz'
    digest = save_physical_checkpoint(path, roots=roots, optimizer=optimizer, manifest=manifest)
    before, _, _ = _pack(roots, optimizer)
    args = dict(roots=roots, optimizer=optimizer, manifest=manifest, expected_sha256=digest)
    for override in ({'expected_sha256': '0'*64}, {'manifest': {'identity': 'changed'}}, {'max_bytes': 1},
                     {'optimizer': Slots(jnp.zeros(3), jnp.asarray(4))}, {'roots': {'model': roots['model']}}):
        with pytest.raises(ValueError):
            restore_physical_checkpoint(path, **(args | override))
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name]) for name in saved.files}
    name = next(key for key in arrays if key != 'physical_manifest')
    arrays[name] = np.ones_like(arrays[name])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='integrity'):
        restore_physical_checkpoint(path, **(args | {'expected_sha256': _digest_file(path)}))
    after, _, _ = _pack(roots, optimizer)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_legacy_and_missing_manifest_are_rejected(tmp_path):
    roots, optimizer = fixture(tmp_path)
    path = tmp_path/'legacy.npz'
    np.savez(path, h01_manifest=np.frombuffer(b'{}', dtype=np.uint8))
    with pytest.raises(ValueError, match='legacy'):
        restore_physical_checkpoint(path, roots=roots, optimizer=optimizer, manifest={'identity': 'test'},
                                    expected_sha256=_digest_file(path))
    with pytest.raises(ValueError, match='immutable'):
        save_physical_checkpoint(path, roots=roots, optimizer=optimizer, manifest={})
