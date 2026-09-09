"""Compiled episode execution with ordered, durable per-episode continuation."""

import json
import os
from pathlib import Path

import brainstate
import jax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np

from .h01_arc_execution import update_episode
from .h01_checkpoint import save_checkpoint, load_checkpoint


def recovery_checkpoint(directory, *, stage_id, parent_sha256, schedule_sha256, asset_root):
    """Verify the most recent durable episode for the exact scheduled parent.

    Parameters
    ----------
    directory, asset_root : path-like
        Stage recovery directory and immutable morphology store.
    stage_id, parent_sha256, schedule_sha256 : str
        Coordinator stage, original parent and deterministic schedule identities.

    Returns
    -------
    tuple or None
        Checkpoint path, SHA256 and next cursor, or None before the first commit.
    """
    directory = Path(directory)
    journal = directory/'resume.json'
    if not journal.exists():
        return None
    value = json.loads(journal.read_text())
    if (value['stage_id'], value['base_parent_sha256'], value['schedule_sha256']) != (
            stage_id, parent_sha256, schedule_sha256):
        raise ValueError('H01 episode recovery belongs to another parent or schedule')
    cursor = value['cursor']
    if type(cursor) is not int or cursor < 1:
        raise ValueError('Invalid H01 recovery cursor')
    path = directory/f'episode-{cursor:06d}.npz'
    loaded = load_checkpoint(path, asset_root=asset_root, expected_sha256=value['checkpoint_sha256'])
    boundary = loaded['metadata']['continuation']
    for name in ('stage_id', 'base_parent_sha256', 'schedule_sha256', 'cursor'):
        if boundary[name] != value[name]:
            raise ValueError('H01 episode checkpoint differs from its recovery journal')
    return path, value['checkpoint_sha256'], cursor


def run_episodes(session, payloads, module, *, directory, stage_id, parent_sha256,
                 schedule_sha256, start_cursor=0, previous_sha256=None):
    """Run repeated episodes in one BrainState loop, committing each boundary.

    Parameters
    ----------
    session : H01Session
        Parent or last-durable-episode runtime.
    payloads : dict
        Stacked remaining episode payloads, all sharing the event shape.
    module : module
        Existing Example 21 request loss and trainer implementation.
    directory : path-like
        Coordinator-owned stage recovery directory.
    stage_id, parent_sha256, schedule_sha256 : str
        Exact immutable stage, base parent and schedule identities.
    start_cursor : int, optional
        Number of previously committed scheduled episodes.
    previous_sha256 : str, optional
        Last durable episode hash; defaults to the base parent identity.

    Returns
    -------
    arrays
        Episode losses, gradient norms and durable cursors.

    Notes
    -----
    An ordered I/O callback writes only numerical parameters and optimizer
    state after each update. A crash before journal replacement replays that
    episode from the last saved parent. Physical and eligibility state reset
    at the next episode; repeated model execution stays inside the compiled loop.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = {'sha256': previous_sha256 or parent_sha256, 'cursor': start_cursor}
    indices, indptr = np.array(session.model.input_csr.indices), np.array(session.model.input_csr.indptr)

    def commit(parameters, optimizer, updates, cursor, loss, norm):
        cursor = int(cursor)
        if not np.isfinite(loss).all() or not np.isfinite(norm).all():
            raise ValueError('Cannot commit a nonfinite H01 learning episode')
        if cursor != pointer['cursor']+1:
            raise ValueError('H01 durable episode callbacks arrived out of order')
        boundary = dict(episode_boundary=True, completed_episodes=int(updates),
            stage_id=stage_id, base_parent_sha256=parent_sha256,
            schedule_sha256=schedule_sha256, cursor=cursor,
            parent_checkpoint_sha256=pointer['sha256'])
        path = directory/f'episode-{cursor:06d}.npz'
        digest = save_checkpoint(path, topology=session.topology, parameters=parameters,
            input_indices=indices, input_indptr=indptr, optimizer=optimizer,
            settings=session.settings, continuation=boundary, assets=session.assets)
        journal = dict(boundary, checkpoint_sha256=digest)
        temporary = directory/'resume.json.tmp'
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(journal, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory/'resume.json')
        pointer.update(sha256=digest, cursor=cursor)
        return np.asarray(cursor, dtype=np.int32)

    def episode(payload, cursor):
        loss, norm = update_episode(session, payload, module)
        durable = io_callback(commit, jax.ShapeDtypeStruct((), jnp.int32),
            session.trainer.parameters, session.trainer.muon_groups, session.trainer.updates, cursor, loss, norm,
            ordered=True)
        return loss, norm, durable

    count = len(payloads['events'])
    if count == 0:
        raise ValueError('A compiled H01 episode batch must be nonempty')
    driver = brainstate.transform.jit(lambda values: brainstate.transform.for_loop(
        episode, values, jnp.arange(start_cursor+1, start_cursor+count+1, dtype=jnp.int32)))
    return jax.block_until_ready(driver(jax.tree.map(jnp.asarray, payloads)))
