"""Bounded real ARC scoring, pp-prop update and H01 continuation verification."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

import brainstate
import jax
import numpy as np
import psutil

from examples.pp_prop.arc_contracts import load_corpus_manifest, load_task, encode_episode
from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter, SupervisedQuery, direct_query_metrics
from examples.pp_prop.h01_arc_adapter import H01ArcAdapter
from examples.pp_prop.h01_arc_execution import score_episode
from examples.pp_prop.h01_episode_recovery import run_episodes, recovery_checkpoint
from examples.pp_prop.h01_session import H01Session


def execute(session, query, module, directory, asset_root, phase):
    """Execute one complete query, commit its update and verify restored scoring.

    Parameters
    ----------
    session : H01Session
        Pinned actual source-cell runtime.
    query : SupervisedQuery
        Unmodified ARC encoding and query target.
    module : module
        Existing Example 21 loss, scoring and Muon implementation.
    directory, asset_root : path-like
        Probe artifacts and immutable source assets.
    phase : callable
        Timed operation wrapper accepting a label and zero-argument callable.

    Returns
    -------
    dict
        Measured query metrics, update counts and continuation agreement.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parent_path = directory/'parent.npz'
    parent_sha = (hashlib.sha256(parent_path.read_bytes()).hexdigest() if parent_path.exists() else
        phase('save_parent', lambda: session.save(parent_path, stage_id='arc-probe-parent')))
    score = brainstate.transform.jit(lambda e, a: score_episode(session, e, a))
    before_logits, _ = phase('score_before', lambda: jax.block_until_ready(score(query.events, query.advances)))
    before_metrics = direct_query_metrics(np.asarray(before_logits), query.target, module.decode_prediction)
    payload = Example21ArcAdapter._episode_payload(query)
    schedule_sha = hashlib.sha256(b''.join(np.asarray(payload[key]).tobytes()
                                           for key in sorted(payload))).hexdigest()
    arguments = dict(stage_id='arc-probe-update', parent_sha256=parent_sha, schedule_sha256=schedule_sha)
    episodes = directory/'episodes'
    recovered = recovery_checkpoint(episodes, asset_root=asset_root, **arguments)
    if recovered is None:
        result = phase('encoded_arc_update', lambda: run_episodes(session,
            jax.tree.map(lambda value: np.expand_dims(value, 0), payload), module,
            directory=episodes, **arguments))
        learning = dict(loss=float(result[0][0]), gradient_norm=float(result[1][0]))
        recovered = recovery_checkpoint(episodes, asset_root=asset_root, **arguments)
    else:
        learning = {'replayed_from_durable_boundary': True}
        session = H01Session.restore(recovered[0], asset_root, module.PPPropEpisodeTrainer,
                                     expected_sha256=recovered[1])
        score = brainstate.transform.jit(lambda e, a: score_episode(session, e, a))
    if recovered is None or recovered[2] != 1:
        raise ValueError('Expected exactly one committed ARC episode')
    after_logits, _ = phase('score_after', lambda: jax.block_until_ready(score(query.events, query.advances)))
    expected = np.array(after_logits)
    expected_parameters = jax.tree.map(np.array, session.trainer.parameters)
    expected_optimizer = jax.tree.map(np.array, session.trainer.muon_groups)
    count = int(session.trainer.updates)
    # The compiled closure retains the model until explicitly released.
    del score
    session.model = session.learner = session.trainer = None
    jax.clear_caches()
    gc.collect()
    restored = phase('restore_continuation', lambda: H01Session.restore(recovered[0], asset_root,
        module.PPPropEpisodeTrainer, expected_sha256=recovered[1]))
    for expected_tree, actual_tree in ((expected_parameters, restored.trainer.parameters),
                                      (expected_optimizer, restored.trainer.muon_groups)):
        if jax.tree.structure(expected_tree) != jax.tree.structure(actual_tree):
            raise ValueError('Restored continuation tree changed')
        for a, b in zip(jax.tree.leaves(expected_tree), jax.tree.leaves(actual_tree)):
            np.testing.assert_array_equal(a, b)
    if int(restored.trainer.updates) != count:
        raise ValueError('Restored completed-update count changed')
    restored_score = brainstate.transform.jit(lambda e, a: score_episode(restored, e, a))
    actual, _ = phase('score_restored', lambda: jax.block_until_ready(restored_score(query.events, query.advances)))
    np.testing.assert_array_equal(actual, expected)
    after_metrics = direct_query_metrics(expected, query.target, module.decode_prediction)
    return dict(status='arc_episode_update_restore_pass', task_id=query.task_id, query_index=query.query_index,
        advancing_events=int(np.sum(query.advances)), padded_events=len(query.advances),
        cells=len(restored.topology.to_dict()['active_cells']), completed_updates=count,
        score_before=dict(exact=before_metrics[0], loss=before_metrics[1]),
        score_after=dict(exact=after_metrics[0], loss=after_metrics[1]), learning=learning,
        checkpoint_sha256=recovered[1], restore_max_logit_error=0., physiology='unqualified')


def main():
    """Run a source-manifest probe with explicit wall-time and host-memory limits.

    Returns
    -------
    None
        Writes incremental timing evidence and durable episode checkpoints.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arc-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task', default='025d127b')
    parser.add_argument('--query', type=int, default=0)
    parser.add_argument('--wall-limit-seconds', type=float, default=900.)
    parser.add_argument('--rss-limit-gib', type=float, default=16.)
    args = parser.parse_args()
    if not np.isfinite([args.wall_limit_seconds, args.rss_limit_gib]).all() or min(
            args.wall_limit_seconds, args.rss_limit_gib) <= 0:
        parser.error('Resource limits must be finite and positive')
    manifest = load_corpus_manifest(args.arc_root)
    task = load_task(args.arc_root, args.task, manifest=manifest)
    events, advances = encode_episode(task, query_index=args.query)
    query = SupervisedQuery(task.task_id, args.query, events, advances, task.targets[args.query])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir/'report.json'
    report = dict(status='running', phases={}, limits=dict(wall_seconds=args.wall_limit_seconds,
        rss_gib=args.rss_limit_gib), task_sha256=next(source.source_sha256 for source in manifest.sources
        if source.task_id == args.task))
    started, stopped = time.monotonic(), threading.Event()
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['rss_gib'] = psutil.Process().memory_info().rss/2**30
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    def monitor():
        while not stopped.wait(1.):
            if time.monotonic()-started >= args.wall_limit_seconds or psutil.Process().memory_info().rss/2**30 >= args.rss_limit_gib:
                (args.output_dir/'limit.json').write_text(json.dumps(dict(status='resource_limit',
                    phase=report.get('active_phase'), elapsed_seconds=time.monotonic()-started)))
                os._exit(124)
    def phase(name, call):
        report['active_phase'] = name
        save()
        start = time.monotonic()
        value = call()
        report['phases'][name] = dict(seconds=time.monotonic()-start, status='pass')
        save()
        print(name, report['phases'][name], flush=True)
        return value
    threading.Thread(target=monitor, daemon=True).start()
    try:
        with brainstate.environ.context(precision=64):
            adapter = H01ArcAdapter(args.arc_root, args.manifest)
            parent = args.output_dir/'parent.npz'
            session = phase('construction_initialization_learning_graph',
                lambda: H01Session.restore(parent, adapter.asset_root, adapter._model().PPPropEpisodeTrainer)
                if parent.exists() else adapter._fresh_runtime())
            report['settings'] = session.settings
            report.update(execute(session, query, adapter._model(), args.output_dir, adapter.asset_root, phase))
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        stopped.set()
        save()


if __name__ == '__main__':
    main()
