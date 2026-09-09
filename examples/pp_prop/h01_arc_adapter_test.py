"""H01 backend identities, coordinator promotion and inherited Dale policy."""

import json
from pathlib import Path

import numpy as np
import pytest

from .h01_checkpoint_test import _save
from .h01_checkpoint import load_checkpoint
from .h01_arc_adapter import H01ArcAdapter
from .example21_evolve import ScoreSnapshot, PipelineConfig


def _candidate(tmp_path):
    path, _, _ = _save(tmp_path)
    loaded = load_checkpoint(path, asset_root=tmp_path/'assets')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='h01-arc-manifest-v1',
        topology=loaded['topology'].to_dict(), settings=loaded['metadata']['settings'],
        assets=loaded['metadata']['assets'], asset_root='assets')))
    adapter = H01ArcAdapter(tmp_path/'arc', manifest)
    with np.load(path) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    metadata = json.loads(arrays['h01_manifest'].tobytes())
    metadata['continuation']['manifest_sha256'] = adapter._manifest_identity
    arrays['h01_manifest'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    path = adapter._candidate_path(tmp_path, 'initial')
    np.savez(path, **arrays)
    score = ScoreSnapshot(task_ids=('00000001',), task_exact=(False,), task_loss=(1.,), finite=True)
    candidate = adapter._snapshot_h01('initial', path, score, False)
    adapter._record_lineage(candidate, None)
    return adapter, candidate


def test_checkpoint_promotion_and_restore_preserve_all_identities(tmp_path):
    adapter, candidate = _candidate(tmp_path)
    assert adapter.restore(candidate).checkpoint_sha256 == candidate.checkpoint_sha256
    destination = tmp_path/'checkpoints'/'initial.npz'
    persisted = adapter.persist(candidate, destination, parent_checkpoint_sha256=None, stage_id='initial')
    assert persisted.checkpoint_sha256 == candidate.checkpoint_sha256
    assert not Path(candidate.checkpoint_path).exists()
    assert adapter.restore(persisted).parameters_sha256 == candidate.parameters_sha256
    repeated = adapter.persist(persisted, destination, parent_checkpoint_sha256=None, stage_id='initial')
    assert repeated.checkpoint_sha256 == persisted.checkpoint_sha256
    assert repeated.optimizer_sha256 == persisted.optimizer_sha256


def test_source_manifest_and_coordinator_parent_mismatch_fail(tmp_path):
    adapter, candidate = _candidate(tmp_path)
    with pytest.raises(ValueError, match='parent'):
        adapter.persist(candidate, tmp_path/'checkpoints'/'stage.npz', parent_checkpoint_sha256='a'*64, stage_id='stage')
    adapter._manifest_identity = 'b'*64
    with pytest.raises(ValueError, match='manifest'):
        adapter.restore(candidate)


def test_production_population_and_dale_identity_guards(tmp_path):
    from types import SimpleNamespace
    adapter, candidate = _candidate(tmp_path)
    with pytest.raises(ValueError, match='104'):
        adapter.initialize(PipelineConfig(), tmp_path)
    for arm in ('excitatory', 'inhibitory'):
        attempt = adapter.run_candidate(candidate, arm, None, SimpleNamespace(stage='dale'))
        assert attempt.status == 'blocked' and attempt.executed_updates == 0
        assert 'identities cannot change' in attempt.reason


def test_cli_exposes_h01_and_requires_manifest(tmp_path, monkeypatch):
    from .example21_arc_adapter import Example21ArcAdapter
    module = Example21ArcAdapter(Path('.'))._model()
    args = module._parser().parse_args(['evolve', '--model-backend', 'h01', '--h01-manifest', 'source.json'])
    assert args.model_backend == 'h01' and args.h01_manifest == Path('source.json')
    with pytest.raises(ValueError, match='requires --h01-manifest'):
        module._evolve_workflow_report(Path('.'), Path('.'), model_backend='h01')
    calls = []
    monkeypatch.setattr(module, '_evolve_workflow_report', lambda *args, **kwargs:
        calls.append(kwargs) or {'passed': True, 'mode': 'evolve'})
    assert module.main(['evolve', '--model-backend', 'h01', '--h01-manifest', 'source.json',
                        '--output-dir', str(tmp_path)]) == 0
    assert calls[0]['model_backend'] == 'h01' and calls[0]['h01_manifest'] == Path('source.json')


@pytest.mark.parametrize('stage,arm', [('neuron', 'add'), ('neuron', 'prune'), ('edge', 'add'), ('edge', 'prune')])
def test_mutation_arms_use_parent_rank_and_shared_schedule(tmp_path, monkeypatch, stage, arm):
    from dataclasses import replace
    from types import SimpleNamespace
    adapter, candidate = _candidate(tmp_path)
    topology = adapter.initial_topology.add_contact('5805562981', '5965472721', stage='seed')
    runtime = SimpleNamespace(topology=topology, trainer=SimpleNamespace(updates=7))
    children = []
    def mutate(child, archive, *, release_parent):
        assert release_parent
        children.append(child)
        return SimpleNamespace(topology=child, trainer=SimpleNamespace(updates=7))
    runtime.mutate = mutate
    monkeypatch.setattr(adapter, '_runtime_from_checkpoint', lambda _: runtime)
    monkeypatch.setattr(adapter, '_archive', lambda: None)
    evidence = SimpleNamespace(neuron_scores=np.array([3., 2., 1.]), edge_scores=np.array([.4]),
        activity=np.array([1., 2., 3.]), source_scores=np.array([3., 2., 1.]), target_scores=np.array([1., 2., 3.]))
    monkeypatch.setattr(adapter, '_parent_evidence', lambda *args: evidence)
    schedule = object()
    def train(child, supplied, context, parent, supplied_arm):
        assert supplied is schedule and parent.checkpoint_sha256 == candidate.checkpoint_sha256 and supplied_arm == arm
        child.trainer.updates += 128
        return child
    monkeypatch.setattr(adapter, '_train_scheduled', train)
    monkeypatch.setattr(adapter, '_write_runtime', lambda *args, **kwargs: replace(candidate,
        candidate_id=kwargs['candidate_id'], topology_changed=True))
    monkeypatch.setattr(adapter, '_record_lineage', lambda *args: None)
    context = SimpleNamespace(stage=stage, stage_id='stage', output_dir=tmp_path,
                              config=PipelineConfig(), score_task_ids=())
    attempt = adapter.run_candidate(candidate, arm, schedule, context)
    assert attempt.status == 'completed' and attempt.executed_updates == 128, attempt.reason
    child = children[0].to_dict()
    assert child['sources'] == topology.to_dict()['sources']
    if stage == 'neuron' and arm == 'add':
        assert child['instances'][child['active_cells'][-1]]['parent_id'] == '5805562981'
    elif stage == 'neuron':
        assert '7196644737' not in child['active_cells']
    elif arm == 'add':
        assert len(child['active_contacts']) == 2
    else:
        assert child['active_contacts'] == []


def test_partial_candidate_failure_reports_actual_update_count(tmp_path, monkeypatch):
    from types import SimpleNamespace
    adapter, candidate = _candidate(tmp_path)
    runtime = SimpleNamespace(topology=adapter.initial_topology, trainer=SimpleNamespace(updates=17))
    runtime.mutate = lambda topology, archive, **kwargs: runtime
    monkeypatch.setattr(adapter, '_runtime_from_checkpoint', lambda _: runtime)
    monkeypatch.setattr(adapter, '_archive', lambda: None)
    monkeypatch.setattr(adapter, '_parent_evidence', lambda *args: SimpleNamespace(neuron_scores=np.ones(3)))
    def interrupted(*args):
        runtime.trainer.updates += 3
        raise RuntimeError('interrupted after three updates')
    monkeypatch.setattr(adapter, '_train_scheduled', interrupted)
    context = SimpleNamespace(stage='neuron', stage_id='stage', output_dir=tmp_path,
                              config=PipelineConfig(), score_task_ids=())
    attempt = adapter.run_candidate(candidate, 'add', None, context)
    assert attempt.status == 'failed' and attempt.executed_updates == 3


@pytest.mark.parametrize('record', ['sources', 'instances', 'blocked_contacts'])
def test_restore_rejects_rewritten_source_provenance(tmp_path, record):
    """A matching declared manifest hash cannot authorize rewritten anatomy."""
    adapter, candidate = _candidate(tmp_path)
    path = Path(candidate.checkpoint_path)
    with np.load(path) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    metadata = json.loads(arrays['h01_manifest'].tobytes())
    topology = metadata['topology']
    if record == 'sources':
        next(iter(topology['sources'].values()))['source_sha256'] = 'f'*64
    elif record == 'instances':
        next(iter(topology['instances'].values()))['created_stage'] = 'rewritten'
    else:
        topology['blocked_contacts'] = [{'reason': 'rewritten'}]
    arrays['h01_manifest'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez(path, **arrays)
    candidate = adapter._snapshot_h01(candidate.candidate_id, path, candidate.score, False)
    with pytest.raises(ValueError, match='immutable source provenance'):
        adapter.restore(candidate)


def test_renderer_writes_coordinator_requested_png_path(tmp_path):
    adapter, candidate = _candidate(tmp_path)
    destination = tmp_path/'topology.png.tmp'
    adapter.render_topology(candidate, destination)
    assert destination.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


@pytest.mark.parametrize('cursor', [0, 127, 128])
def test_scheduled_training_uses_only_uncommitted_entries(tmp_path, monkeypatch, cursor):
    from types import SimpleNamespace
    from . import h01_arc_adapter as backend
    from .example21_evolve import UpdateSchedule, ScheduleEntry
    adapter, candidate = _candidate(tmp_path)
    runtime = SimpleNamespace(trainer=SimpleNamespace(updates=10))
    recovered = SimpleNamespace(trainer=SimpleNamespace(updates=10+cursor))
    schedule = UpdateSchedule(40, 168, tuple(ScheduleEntry(i, 'task', 0, 21, 'a'*64) for i in range(40, 168)))
    context = SimpleNamespace(stage_id='stage', output_dir=tmp_path)
    monkeypatch.setattr(backend, 'recovery_checkpoint', lambda *a, **k:
        (Path('episode.npz'), 'b'*64, cursor) if cursor else None)
    monkeypatch.setattr(backend.H01Session, 'restore', lambda *a, **k: recovered)
    monkeypatch.setattr(adapter, '_encoded_queries', lambda role: [SimpleNamespace(task_id='task', query_index=0)])
    monkeypatch.setattr(adapter, '_episode_payload', lambda query: {'events': np.zeros((1, 441))})
    calls = []
    def execute(session, payloads, module, **kwargs):
        calls.append(len(payloads['events']))
        assert kwargs['start_cursor'] == cursor
        assert kwargs['parent_sha256'] == candidate.checkpoint_sha256
        session.trainer.updates += len(payloads['events'])
    monkeypatch.setattr(backend, 'run_episodes', execute)
    result = adapter._train_scheduled(runtime, schedule, context, candidate, 'add')
    assert int(result.trainer.updates) == 138
    assert calls == ([128-cursor] if cursor < 128 else [])
    assert result.base_parent_sha256 == candidate.checkpoint_sha256


def test_interrupted_recovery_reports_committed_schedule_cursor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from . import h01_arc_adapter as backend
    from .example21_evolve import UpdateSchedule, ScheduleEntry
    adapter, candidate = _candidate(tmp_path)
    runtime = SimpleNamespace(trainer=SimpleNamespace(updates=10))
    schedule = UpdateSchedule(0, 128, tuple(ScheduleEntry(i, 'task', 0, 21, 'a'*64) for i in range(128)))
    context = SimpleNamespace(stage_id='stage', output_dir=tmp_path)
    cursors = iter([126, 127])
    monkeypatch.setattr(backend, 'recovery_checkpoint', lambda *a, **k: (Path('episode.npz'), 'b'*64, next(cursors)))
    monkeypatch.setattr(backend.H01Session, 'restore', lambda *a, **k:
        SimpleNamespace(trainer=SimpleNamespace(updates=136)))
    monkeypatch.setattr(adapter, '_encoded_queries', lambda role: [SimpleNamespace(task_id='task', query_index=0)])
    monkeypatch.setattr(adapter, '_episode_payload', lambda query: {'events': np.zeros((1, 441))})
    def interrupted(*args, **kwargs):
        raise RuntimeError('interrupted final episode')
    monkeypatch.setattr(backend, 'run_episodes', interrupted)
    with pytest.raises(RuntimeError, match='interrupted') as error:
        adapter._train_scheduled(runtime, schedule, context, candidate, 'add')
    assert error.value.executed_updates == 127
