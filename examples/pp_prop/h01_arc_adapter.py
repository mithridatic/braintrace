"""H01 backend for Example 21's existing protected evolution coordinator."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_network_init import process_rss_mb
from .example21_arc_adapter import Example21ArcAdapter, direct_query_metrics
from .h01_arc_execution import score_episode
from .h01_checkpoint import load_checkpoint
from .h01_episode_recovery import recovery_checkpoint, run_episodes
from .h01_session import H01Session
from .h01_topology import H01Topology


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class H01ArcAdapter(Example21ArcAdapter):
    """Reuse corpus isolation and coordination with real multicompartment cells.

    Parameters
    ----------
    arc_root : path-like
        Direct ARC corpus root used by Example 21.
    manifest : path-like
        H01 manifest containing topology, settings, assets and asset_root.
    """

    def __init__(self, arc_root, manifest):
        super().__init__(arc_root)
        self.manifest_path = Path(manifest).resolve()
        self.document = json.loads(self.manifest_path.read_text())
        if self.document.get('schema') != 'h01-arc-manifest-v1':
            raise ValueError('Expected an h01-arc-manifest-v1 source manifest')
        self.initial_topology = H01Topology.from_dict(self.document['topology'])
        self.asset_root = (self.manifest_path.parent/self.document['asset_root']).resolve()
        self._manifest_identity = _hash(self.document)

    def _archive(self):
        source = next(iter(self.initial_topology.to_dict()['sources'].values()))
        digest = source['archive_sha256']
        if digest not in self.document['assets']:
            raise ValueError('H01 source archive is not listed in immutable assets')
        return H01Archive(self.asset_root/digest)

    def _fresh_runtime(self):
        return H01Session.build(self.initial_topology, self._archive(), self._model().PPPropEpisodeTrainer,
            settings=self.document['settings'], assets=self.document['assets'])

    def _verify_provenance(self, loaded):
        initial, restored = self.initial_topology.to_dict(), loaded['topology'].to_dict()
        if (restored['sources'] != initial['sources'] or
                restored['blocked_contacts'] != initial['blocked_contacts'] or
                any(restored[section].get(key) != value
                    for section in ('instances', 'contacts')
                    for key, value in initial[section].items()) or
                loaded['metadata']['settings'] != self.document['settings'] or
                set(loaded['metadata']['assets']) != set(self.document['assets'])):
            raise ValueError('H01 checkpoint changed immutable source provenance or numerical settings')

    def _runtime_from_checkpoint(self, path):
        loaded = load_checkpoint(path, asset_root=self.asset_root)
        if loaded['metadata']['continuation'].get('manifest_sha256') != self._manifest_identity:
            raise ValueError('H01 checkpoint belongs to another source manifest')
        self._verify_provenance(loaded)
        runtime = H01Session.restore(path, self.asset_root, self._model().PPPropEpisodeTrainer)
        runtime.base_parent_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return runtime

    def _snapshot_h01(self, candidate_id, path, score, topology_changed):
        loaded = load_checkpoint(path, asset_root=self.asset_root)
        metadata, topology = loaded['metadata'], loaded['topology'].to_dict()
        arrays = metadata['arrays']
        identities = {prefix: _hash({k: v for k, v in arrays.items() if k.startswith(prefix)})
                      for prefix in ('parameter/', 'optimizer/')}
        # Encoder connectivity is part of topology, not a trainable magnitude.
        topology_hash = _hash(dict(topology=topology,
            encoder={key: arrays[key] for key in ('input_indices', 'input_indptr')}, settings=metadata['settings']))
        resources = self._evolve().ResourceUsage(
            persistent_bytes=sum(int(np.prod(v['shape']))*np.dtype(v['dtype']).itemsize for v in arrays.values()),
            checkpoint_bytes=Path(path).stat().st_size, neurons=len(topology['active_cells']),
            recurrent_edges=len(topology['active_contacts']),
            peak_host_ram_bytes=int((process_rss_mb(peak=True) or 0)*1024**2),
            device_memory_bytes=self._peak_device_memory_bytes())
        return self._evolve().CandidateSnapshot(candidate_id=candidate_id, checkpoint_path=str(Path(path).resolve()),
            checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            topology_sha256=topology_hash, parameters_sha256=identities['parameter/'],
            optimizer_sha256=identities['optimizer/'], score=score, resources=resources,
            topology_changed=topology_changed)

    def _write_runtime(self, runtime, *, role, candidate_id, path, topology_changed, task_ids=None):
        scored = self._score_runtime(runtime, role, task_ids=task_ids)
        runtime.save(path, stage_id=candidate_id.rsplit('-', 1)[0],
            parent_checkpoint_sha256=getattr(runtime, 'base_parent_sha256', None),
            continuation={'manifest_sha256': self._manifest_identity})
        candidate = self._snapshot_h01(candidate_id, path, scored.score, topology_changed)
        self._evidence_by_checkpoint[candidate.checkpoint_sha256] = scored
        return candidate

    def initialize(self, config, output_dir):
        """Initialize all 104 original cells through the existing coordinator.

        Parameters
        ----------
        config : PipelineConfig
            Existing schedule, optimizer and promotion policy.
        output_dir : path-like
            Coordinator artifact directory.

        Returns
        -------
        CandidateSnapshot
            Directly scored initial continuation.
        """
        if len(self.initial_topology.to_dict()['active_cells']) != 104:
            raise ValueError('Production H01 evolution must start with all 104 selected source cells')
        return super().initialize(config, output_dir)

    def restore(self, candidate):
        """Verify a checkpoint and its pinned H01 source manifest.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Coordinator-authenticated continuation.

        Returns
        -------
        CandidateSnapshot
            Same identities and score with refreshed resource measurements.
        """
        loaded = load_checkpoint(candidate.checkpoint_path, asset_root=self.asset_root,
                                 expected_sha256=candidate.checkpoint_sha256)
        if loaded['metadata']['continuation'].get('manifest_sha256') != self._manifest_identity:
            raise ValueError('H01 resume source manifest differs from this run')
        self._verify_provenance(loaded)
        restored = self._snapshot_h01(candidate.candidate_id, candidate.checkpoint_path,
                                     candidate.score, candidate.topology_changed)
        for name in ('topology_sha256', 'parameters_sha256', 'optimizer_sha256'):
            if getattr(restored, name) != getattr(candidate, name):
                raise ValueError('H01 checkpoint component differs from coordinator identity')
        return restored

    def persist(self, candidate, destination, *, parent_checkpoint_sha256, stage_id):
        """Atomically promote a verified candidate without rewriting its bytes.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Selected immutable candidate.
        destination : path-like
            Coordinator checkpoint path.
        parent_checkpoint_sha256 : str or None
            Authenticated stage parent.
        stage_id : str
            Exact coordinator stage identity.

        Returns
        -------
        CandidateSnapshot
            Same continuation at its durable destination.
        """
        candidate = self.restore(candidate)
        source, destination = Path(candidate.checkpoint_path), Path(destination)
        if destination.name != stage_id+'.npz' or not stage_id or any(c not in
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in stage_id):
            raise ValueError('Invalid H01 coordinator destination')
        if source.resolve() == destination.resolve():
            return candidate
        if self._persisted_lineage(candidate, source, destination) != parent_checkpoint_sha256:
            raise ValueError('H01 candidate parent differs from coordinator lineage')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        result = replace(candidate, checkpoint_path=str(destination.resolve()))
        source.unlink()
        self._provenance_path(source).unlink(missing_ok=True)
        self._temporary_paths.discard(source.resolve())
        return result

    def _score_runtime(self, runtime, role, *, task_ids=None):
        ids = self._scored_task_ids(self._manifest(role), task_ids)
        queries = tuple(q for q in self._encoded_queries(role) if q.task_id in ids)
        events = jnp.asarray(np.stack([q.events for q in queries]), dtype=jnp.float64)
        advances = jnp.asarray(np.stack([q.advances for q in queries]))
        execute = brainstate.transform.jit(lambda e, a: brainstate.transform.for_loop(
            lambda x, mask: score_episode(runtime, x, mask), e, a))
        logits, activity = execute(events, advances)
        logits, activity = np.asarray(logits), np.asarray(activity)
        metrics = {identity: [] for identity in ids}
        for query, output in zip(queries, logits):
            metrics[query.task_id].append(direct_query_metrics(output, query.target, self._model().decode_prediction))
        score = self._evolve().ScoreSnapshot(task_ids=ids,
            task_exact=tuple(all(value[0] for value in metrics[key]) for key in ids),
            task_loss=tuple(float(np.mean([value[1] for value in metrics[key]])) for key in ids),
            finite=bool(np.isfinite(logits).all() and np.isfinite(activity).all() and runtime.trainer.optimizer_is_finite()))
        doc = runtime.topology.to_dict()
        cell_index = {name: i for i, name in enumerate(doc['active_cells'])}
        magnitude = np.mean(activity, axis=0)
        head = np.asarray(runtime.trainer.parameters['readout_weight'])
        moments = [np.asarray(x) for x in jax.tree.leaves(runtime.trainer.muon_groups['readout_weight']) if np.shape(x) == head.shape]
        neuron = magnitude+sum((np.sum(np.abs(x), axis=1) for x in moments), np.zeros(len(magnitude)))
        edges = np.array([abs(float(runtime.trainer.parameters['recurrent'][i]))*magnitude[cell_index[doc['contacts'][key]['pre']]]
                          for i, key in enumerate(doc['active_contacts'])])
        return SimpleNamespace(score=score, neuron_scores=neuron, source_scores=neuron,
                               target_scores=neuron, edge_scores=edges, activity=magnitude)

    def _train_scheduled(self, runtime, schedule, context, parent, arm):
        if len(schedule.entries) != 128 or schedule.cursor_end-schedule.cursor_start != 128:
            raise ValueError('Production H01 requires the shared 128-update schedule')
        digest = _hash(asdict(schedule))
        stage = context.stage_id+'-'+arm
        directory = Path(context.output_dir)/'.episodes'/stage
        recovered = recovery_checkpoint(directory, stage_id=stage, parent_sha256=parent.checkpoint_sha256,
            schedule_sha256=digest, asset_root=self.asset_root)
        cursor, previous = 0, None
        if recovered is not None:
            path, previous, cursor = recovered
            runtime = H01Session.restore(path, self.asset_root, self._model().PPPropEpisodeTrainer,
                                         expected_sha256=previous)
        if cursor > 128:
            raise ValueError('H01 recovery cursor exceeds its scheduled block')
        if cursor < 128:
            lookup = {(q.task_id, q.query_index): q for q in self._encoded_queries('training')}
            payloads = [self._episode_payload(lookup[(entry.task_id, entry.query_index)]) for entry in schedule.entries[cursor:]]
            stacked = jax.tree.map(lambda *values: np.stack(values), *payloads)
            try:
                run_episodes(runtime, stacked, self._model(), directory=directory, stage_id=stage,
                    parent_sha256=parent.checkpoint_sha256, schedule_sha256=digest,
                    start_cursor=cursor, previous_sha256=previous)
            except Exception as error:
                # A failed compiled loop need not return its final State values.
                # The authenticated journal is the authoritative committed count.
                committed = recovery_checkpoint(directory, stage_id=stage,
                    parent_sha256=parent.checkpoint_sha256, schedule_sha256=digest,
                    asset_root=self.asset_root)
                error.executed_updates = committed[2] if committed is not None else cursor
                raise
        runtime.base_parent_sha256 = parent.checkpoint_sha256
        return runtime

    def train_parent(self, parent, schedule, context):
        """Train the accepted parent with durable scheduled episode boundaries.

        Parameters
        ----------
        parent, schedule, context : object
            Existing coordinator parent, shared schedule and stage context.

        Returns
        -------
        CandidateSnapshot
            Trained and directly scored continuation.
        """
        parent = self.restore(parent)
        runtime = self._runtime_from_checkpoint(parent.checkpoint_path)
        runtime = self._train_scheduled(runtime, schedule, context, parent, 'training')
        identity = context.stage_id+'-training'
        candidate = self._write_runtime(runtime, role='training', candidate_id=identity,
            path=self._candidate_path(context.output_dir, identity), topology_changed=False)
        self._record_lineage(candidate, parent.checkpoint_sha256)
        return candidate

    def run_candidate(self, parent, arm, schedule, context):
        """Execute one permitted H01 mutation from the immutable sibling parent.

        Parameters
        ----------
        parent, arm, schedule, context : object
            Existing coordinator candidate, mutation arm, schedule and stage.

        Returns
        -------
        CandidateAttempt
            Literal completed, ineligible or failed result.
        """
        evolve = self._evolve()
        kind = self._mutation_kind(context.stage, arm)
        if kind.startswith('dale-'):
            self.restore(parent)
            return evolve.CandidateAttempt.blocked(arm, 'Inherited H01 E/I identities cannot change', executed_updates=0)
        runtime, before = None, 0
        try:
            parent = self.restore(parent)
            runtime = self._runtime_from_checkpoint(parent.checkpoint_path)
            before = int(runtime.trainer.updates)
            evidence = self._parent_evidence(parent, runtime)
            doc = runtime.topology.to_dict()
            cells, contacts = doc['active_cells'], doc['active_contacts']
            ranks = sorted(cells, key=lambda key: (-evidence.neuron_scores[cells.index(key)], key))
            topology = runtime.topology
            if kind == 'neuron-add':
                topology = topology.clone(ranks[0], stage=context.stage_id)
            elif kind == 'neuron-prune':
                topology = topology.prune_cell(ranks[-1], stage=context.stage_id)
            elif kind == 'edge-prune':
                if not contacts:
                    return evolve.CandidateAttempt.blocked(arm, 'No active H01 contact to prune', executed_updates=0)
                identity = min(contacts, key=lambda key: (evidence.edge_scores[contacts.index(key)], key))
                topology = topology.prune_contact(identity, stage=context.stage_id)
            else:
                gradients = {(pre, post): float(evidence.source_scores[i]*evidence.target_scores[j])
                             for i, pre in enumerate(cells) for j, post in enumerate(cells) if pre != post}
                pairs = topology.rank_absent_pairs(activity=dict(zip(cells, evidence.activity)), gradients=gradients)
                if not pairs:
                    return evolve.CandidateAttempt.blocked(arm, 'No absent directed H01 pair', executed_updates=0)
                topology = topology.add_contact(*pairs[0], stage=context.stage_id)
            changed = topology.to_dict()
            if len(changed['active_cells']) > context.config.max_neurons or len(changed['active_contacts']) > context.config.max_recurrent_edges:
                return evolve.CandidateAttempt.blocked(arm, 'H01 mutation exceeds configured topology caps', executed_updates=0)
            runtime = runtime.mutate(topology, self._archive(), release_parent=True)
            before = int(runtime.trainer.updates)
            runtime = self._train_scheduled(runtime, schedule, context, parent, arm)
            identity = context.stage_id+'-'+arm
            candidate = self._write_runtime(runtime, role='training', candidate_id=identity,
                path=self._candidate_path(context.output_dir, identity), topology_changed=True,
                task_ids=context.score_task_ids)
            self._record_lineage(candidate, parent.checkpoint_sha256)
            return evolve.CandidateAttempt.completed(arm, candidate, executed_updates=128)
        except Exception as error:
            current = 0 if runtime is None else int(getattr(runtime.trainer, 'updates',
                                                            getattr(runtime, 'released_updates', before)))
            updates = getattr(error, 'executed_updates', 0 if runtime is None else max(0, current-before))
            return evolve.CandidateAttempt.failed(arm, f'{type(error).__name__}: {error}', executed_updates=updates)

    def render_topology(self, candidate, output_path):
        """Render inherited identities and anatomical or synthetic contacts.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Verified continuation.
        output_path : path-like
            Exact coordinator PNG destination, including temporary siblings.
        """
        self.restore(candidate)
        loaded = load_checkpoint(candidate.checkpoint_path, asset_root=self.asset_root)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        document = loaded['topology'].to_dict()
        cells = document['active_cells']
        angles = np.linspace(0., 2*np.pi, len(cells), endpoint=False)
        positions = {key: (np.cos(angle), np.sin(angle)) for key, angle in zip(cells, angles)}
        figure = Figure(figsize=(12, 12))
        FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        for key in document['active_contacts']:
            contact = document['contacts'][key]
            axis.annotate('', xy=positions[contact['post']], xytext=positions[contact['pre']],
                arrowprops=dict(arrowstyle='->', color='0.4',
                                linestyle='--' if contact['synthetic'] else '-'))
        for key in cells:
            instance = document['instances'][key]
            polarity = document['sources'][instance['source_id']]['polarity']
            x, y = positions[key]
            axis.scatter(x, y, color='#3075b5' if polarity == 'E' else '#ce593d',
                         marker='s' if instance['synthetic'] else 'o', s=60, zorder=3)
            axis.text(1.08*x, 1.08*y, key, fontsize=max(4, min(9, 300/len(cells))), ha='center')
        axis.set(title=f'H01: {len(cells)} cells, {len(document["active_contacts"])} contacts\n'
                 'Blue E / red I; squares are clones; dashed contacts are synthetic',
                 xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect='equal')
        axis.set_axis_off()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format='png', dpi=150)
