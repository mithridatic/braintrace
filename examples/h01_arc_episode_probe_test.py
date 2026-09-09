"""Full encoded episode and durable continuation through the bounded probe."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import brainstate
import numpy as np
import pytest

from braintrace.datasets.h01_anatomy_test import imported
from braintrace.datasets.h01 import H01Archive
from examples import h01_arc_episode_probe as subject
from examples.pp_prop.arc_contracts import ARCTask, encode_episode
from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter, SupervisedQuery
from examples.pp_prop.h01_checkpoint import store_asset
from examples.pp_prop.h01_runtime_test import _manifest
from examples.pp_prop.h01_session import H01Session
from examples.pp_prop.h01_topology import H01Topology


def test_complete_probe_scores_updates_restores_and_resumes(imported, tmp_path):
    module = Example21ArcAdapter(Path('.'))._model()
    asset_root = tmp_path/'assets'
    digest = store_asset(tmp_path/'archive.zip', asset_root)
    document = _manifest(imported).to_dict()
    document['sources']['12']['archive_sha256'] = digest
    grid = np.ones((1, 1), dtype=np.uint8)
    task = ARCTask('00000001', ((grid, grid),), (grid,), (grid,), 'practice')
    events, advances = encode_episode(task)
    query = SupervisedQuery(task.task_id, 0, events, advances, grid)
    directory = tmp_path/'run'
    phases = []
    def phase(name, call):
        phases.append(name)
        return call()
    with brainstate.environ.context(precision=64):
        session = H01Session.build(H01Topology.from_dict(document), H01Archive(asset_root/digest),
                                   module.PPPropEpisodeTrainer, assets=(digest,))
        report = subject.execute(session, query, module, directory, asset_root, phase)
        assert report['status'] == 'arc_episode_update_restore_pass'
        assert report['completed_updates'] == 1 and report['restore_max_logit_error'] == 0.
        assert report['advancing_events'] == np.sum(advances)
        parent_bytes = (directory/'parent.npz').read_bytes()
        parent = H01Session.restore(directory/'parent.npz', asset_root, module.PPPropEpisodeTrainer)
        phases.clear()
        resumed = subject.execute(parent, query, module, directory, asset_root, phase)
        assert resumed['completed_updates'] == 1
        assert 'encoded_arc_update' not in phases
        assert resumed['learning']['replayed_from_durable_boundary']
        assert (directory/'parent.npz').read_bytes() == parent_bytes


@pytest.mark.parametrize('fail', [False, True])
def test_cli_preserves_task_identity_and_writes_failure_evidence(tmp_path, monkeypatch, fail):
    grid = np.ones((1, 1), dtype=np.uint8)
    task = ARCTask('025d127b', ((grid, grid),), (grid,), (grid,), 'practice')
    monkeypatch.setattr(subject, 'load_corpus_manifest', lambda root:
        SimpleNamespace(sources=[SimpleNamespace(task_id=task.task_id, source_sha256='a'*64)]))
    monkeypatch.setattr(subject, 'load_task', lambda *args, **kwargs: task)
    session = SimpleNamespace(settings={'pinned': True})
    monkeypatch.setattr(subject, 'H01ArcAdapter', lambda *args: SimpleNamespace(
        _fresh_runtime=lambda: session, _model=lambda: None, asset_root=tmp_path/'assets'))
    def execute(runtime, query, *args):
        assert runtime is session and query.task_id == task.task_id
        np.testing.assert_array_equal(query.target, grid)
        if fail:
            raise RuntimeError('measured execution failed')
        return {'status': 'arc_episode_update_restore_pass'}
    monkeypatch.setattr(subject, 'execute', execute)
    monkeypatch.setattr(sys, 'argv', ['probe', '--manifest', str(tmp_path/'manifest.json'),
        '--arc-root', str(tmp_path/'arc'), '--output-dir', str(tmp_path/'run')])
    if fail:
        with pytest.raises(RuntimeError, match='execution failed'):
            subject.main()
    else:
        subject.main()
    report = json.loads((tmp_path/'run'/'report.json').read_text())
    assert report['task_sha256'] == 'a'*64
    assert report['status'] == ('failed' if fail else 'arc_episode_update_restore_pass')
    assert report['limits'] == {'wall_seconds': 900., 'rss_gib': 16.}


def test_cli_rejects_unbounded_resource_values(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['probe', '--manifest', 'manifest.json', '--arc-root', 'arc',
        '--output-dir', str(tmp_path), '--wall-limit-seconds', 'nan'])
    with pytest.raises(SystemExit):
        subject.main()
    assert not (tmp_path/'report.json').exists()
