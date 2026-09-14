"""Trainable H01 runtime construction and durable episode-boundary restoration."""

import copy
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
import hashlib
import json
from pathlib import Path

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_biology import H01SpatialManifest
from braintrace.datasets.h01_network_init import init_h01_network_states
from .h01_arc_model import H01ArcModel
from .h01_checkpoint import load_checkpoint, save_checkpoint, restore_optimizer
from .h01_runtime import build_network
from .h01_muon import POLICY, model_optimizer
from .h01_remap import remap_mutation, remap_optimizer_group


@lru_cache(maxsize=1)
def _numerical_settings_cached():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/'braintrace'/'datasets').glob('h01*.py'))
    paths += sorted((root/'braintrace'/'biophysics').glob('*.py'))
    paths += sorted((root/'examples'/'pp_prop').glob('h01_physical*.py'))
    paths += [root/name for name in ('braintrace/_algorithm/sparse_pp_prop.py',
        'braintrace/_algorithm/sparse_io.py', 'braintrace/_compiler/sparse_io_graph.py',
        'braintrace/_compiler/sparse_support.py', 'braintrace/_compiler/sparse_influence.py',
        'examples/pp_prop/h01_arc_model.py', 'examples/pp_prop/h01_arc_execution.py',
        'examples/pp_prop/h01_muon.py', 'examples/pp_prop/21-braincell-arc.py',
        'examples/pp_prop/h01_session.py')]
    paths.append(root/'examples/pp_prop/h01_runtime.py')
    paths.append(root/'examples/pp_prop/h01_neuroglial.py')
    implementation = {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths if not path.name.endswith('_test.py')}
    # dt is the qualified step, docs/evidence/h01-timestep-ladder.json; substeps = event_ms / dt_ms = 0.1 / 0.000625 = 160
    return dict(optimizer_policy=POLICY, dt_ms=.000625, event_ms=.1, substeps=160, precision=64,
        solver='h01_staggered_calcium_implicit', max_cv_length_um=10., decay=.99,
        factor_limit_bytes=512*1024**2, seed=21, checkpoint_substeps=True,
        implementation_sha256=implementation,
        dependencies={name: version(name) for name in
            ('jax', 'jaxlib', 'brainstate', 'brainunit', 'braincell', 'brainevent', 'optax', 'numpy')})


def numerical_settings():
    """Return the pinned numerical clocks and installed dependency identities.

    Returns
    -------
    dict
        Settings persisted in every H01 checkpoint.
    """
    cached = _numerical_settings_cached()
    return copy.deepcopy(cached)


@dataclass
class H01Session:
    """Own one physical model, learner, optimizer and immutable topology.

    Parameters
    ----------
    topology, model, learner, trainer : object
        Active topology and actual executable training components.
    settings : dict
        Frozen numerical and dependency settings.
    assets : tuple of str
        Immutable morphology asset digests.
    """

    topology: object
    model: object
    learner: object
    trainer: object
    settings: dict
    assets: tuple

    @classmethod
    def build(cls, topology, archive, trainer_type, *, settings=None, assets=(),
              parameters=None, input_pattern=None, progress=None, optimizer_slots=None, biology_assets=None):
        """Construct, initialize and compile the full active H01 model.

        Parameters
        ----------
        topology : H01Topology
            Active topology with immutable source profiles.
        archive : H01Archive
            Verified source morphology archive.
        trainer_type : type
            Example 21's PPPropEpisodeTrainer.
        settings : dict, optional
            Pinned runtime settings.
        assets : tuple of str, optional
            Content-addressed source assets.
        parameters : dict, optional
            Restored or transported parameter values.
        input_pattern : tuple, optional
            Encoder CSR indices and indptr.
        progress : callable, optional
            Construction and initialization progress callback.
        optimizer_slots : mapping, optional
            Inherited parallel-contact plane assignments during mutation.
        biology_assets : mapping, optional
            Glial source SHA256 to local asset path for neuroglial construction.

        Returns
        -------
        H01Session
            Executable episode-boundary state.
        """
        settings = numerical_settings() if settings is None else copy.deepcopy(settings)
        installed = numerical_settings()
        if settings.get('optimizer_policy') != installed['optimizer_policy']:
            raise ValueError('Incompatible H01 optimizer checkpoint; legacy AdamW continuation is unsupported')
        if settings['dependencies'] != installed['dependencies']:
            raise ValueError('H01 checkpoint dependencies differ from this runtime')
        if settings['implementation_sha256'] != installed['implementation_sha256']:
            raise ValueError('Pinned H01 implementation differs from this runtime')
        if brainstate.environ.get('precision') != 64:
            raise ValueError('H01 sessions require an enclosing 64-bit precision context')
        biology = settings.get('biology')
        spatial = None
        neuroglial = None
        if isinstance(biology, dict) and biology.get('schema') == 'h01-biology-neuroglia-v1':
            from braintrace.datasets.h01_neuroglia import H01NeuroglialManifest
            neuroglial = H01NeuroglialManifest(biology, topology.to_dict())
            biology = settings['biology'] = neuroglial.to_dict()
            if biology['dt_ms'] != settings['dt_ms']:
                raise ValueError('Neuroglial and cable clocks differ')
            spatial = H01SpatialManifest(biology['spines'], topology.to_dict())
            probabilities = [biology['spines']['release_probability'][key] for key in topology.to_dict()['active_contacts']]
        elif isinstance(biology, dict) and biology.get('schema') == 'h01-biology-spines-v1':
            spatial = H01SpatialManifest(biology, topology.to_dict())
            # Persist the same canonical arrays validated by the manifest.
            # Python tuple directions otherwise become lists only on save,
            # making a freshly rebuilt session fail the identity comparison.
            biology = settings['biology'] = spatial.to_dict()
            probabilities = [biology['release_probability'][key] for key in topology.to_dict()['active_contacts']]
        elif biology is not None:
            if not isinstance(biology, dict) or biology.get('schema') != 'h01-biology-release-v1' or set(biology) != {'schema', 'release_probability'}:
                raise ValueError('Unsupported biology manifest; spatial assembly is not implicit')
            probabilities = biology['release_probability']
        else:
            probabilities = None
        network, records = build_network(topology, archive, solver=settings['solver'],
            max_cv_length_um=settings['max_cv_length_um'], progress=progress, biology=spatial,
            environment_potassium=neuroglial is not None)
        del records
        if neuroglial is not None:
            from .h01_neuroglial import prepare_neuroglia, attach_neuroglia
            glia = prepare_neuroglia(network, neuroglial, biology_assets)
        init_h01_network_states(network, progress=progress)
        model = H01ArcModel(network, topology.to_dict()['active_cells'], seed=settings['seed'],
                            dt_ms=settings['dt_ms'], input_pattern=input_pattern,
                            checkpoint_substeps=settings['checkpoint_substeps'],
                            release_probability=probabilities)
        if neuroglial is not None:
            attach_neuroglia(model, topology.to_dict(), neuroglial, glia)
        states = dict(input=model.input_weight, recurrent=model.recurrent_weight,
                      readout_weight=model.readout_weight, readout_bias=model.readout_bias)
        if parameters is not None:
            if parameters.keys() != states.keys():
                raise ValueError('H01 parameters require all four optimizer groups')
            for name, state in states.items():
                value = np.asarray(parameters[name])
                if value.shape != state.value.shape or not np.isfinite(value).all():
                    raise ValueError('Invalid restored H01 parameter: '+name)
                state.value = jnp.asarray(value, dtype=state.value.dtype)
        values = {name: state.value for name, state in states.items()}
        saved_optimizer = settings.get('optimizer')
        slots = optimizer_slots if optimizer_slots is not None else (
            saved_optimizer['slots'] if saved_optimizer is not None else None)
        optimizer = model_optimizer(model, values, inherited_slots=slots)
        if saved_optimizer is not None and optimizer_slots is None and saved_optimizer != optimizer.metadata():
            raise ValueError('H01 optimizer layout differs from checkpoint')
        settings['optimizer'] = optimizer.metadata()
        learner = braintrace.pp_prop.sparse(model, settings['decay'], max_bytes=settings['factor_limit_bytes'])
        learner.compile_graph(jnp.zeros(441))
        trainer = trainer_type(learner, values, optimizer_adapter=optimizer)
        return cls(topology, model, learner, trainer, settings, tuple(assets))

    def save(self, path, *, stage_id, parent_checkpoint_sha256=None, continuation=None):
        """Reset episode state and atomically save a completed-update boundary.

        Parameters
        ----------
        path : path-like
            Destination checkpoint.
        stage_id : str
            Coordinator continuation identity.
        parent_checkpoint_sha256 : str, optional
            Immutable parent checkpoint identity.
        continuation : dict, optional
            Additional schedule and cursor identities.

        Returns
        -------
        str
            Durable checkpoint digest.
        """
        if self.settings.get('biology') is not None or self.model.stepper.chemistry is not None:
            raise ValueError('Biological models require an explicit physical v2 checkpoint')
        self.model.reset_episode(self.learner)
        boundary = dict(continuation or {}, episode_boundary=True,
            completed_episodes=int(self.trainer.updates), stage_id=stage_id,
            parent_checkpoint_sha256=parent_checkpoint_sha256)
        return save_checkpoint(path, topology=self.topology, parameters=self.trainer.parameters,
            input_indices=self.model.input_csr.indices, input_indptr=self.model.input_csr.indptr,
            optimizer=self.trainer.muon_groups, settings=self.settings, continuation=boundary, assets=self.assets)

    def _physical_context(self, wait, biology_manifest):
        if wait.model is not self.model or wait.learner is not None and wait.learner is not self.learner:
            raise ValueError('Physical wait belongs to a different H01 session')
        configured = self.settings.get('biology')
        if isinstance(configured, dict) and configured.get('schema') == 'h01-biology-neuroglia-v1':
            identity = hashlib.sha256(json.dumps(configured, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if identity != getattr(self.model.stepper.chemistry, 'biology_sha256', None):
                raise ValueError('Session biology differs from constructed chemistry')
            if biology_manifest is not None and biology_manifest != configured:
                raise ValueError('Physical biology manifest differs from constructed session')
            biology_manifest = configured
        if self.model.stepper.chemistry is not None and not biology_manifest:
            raise ValueError('Spatial physical snapshots require an explicit immutable biology manifest')
        roots = dict(model=self.model, learner=self.learner, wait=wait)
        optimizer = dict(parameters=self.trainer.parameters, groups=self.trainer.muon_groups,
                         updates=self.trainer.updates)
        manifest = dict(topology=self.topology.to_dict(), settings=self.settings, assets=list(self.assets),
                        biology=biology_manifest or self.settings.get('biology'), wait_events=wait.total_events)
        return roots, optimizer, manifest

    def save_physical(self, path, *, wait, biology_manifest=None):
        """Save a complete wait boundary without resetting any physical state.

        Parameters
        ----------
        path : path-like
            Version-2 physical checkpoint destination.
        wait : PhysicalWait
            This session's wait driver, including its physical progress cursor.
        biology_manifest : dict or None, optional
            Full immutable spatial/rate manifest, required for attached chemistry.

        Returns
        -------
        str
            Durable checkpoint digest. Includes parameters, optimizer and traces.
        """
        from .h01_physical_checkpoint import save_physical_checkpoint
        roots, optimizer, manifest = self._physical_context(wait, biology_manifest)
        return save_physical_checkpoint(path, roots=roots, optimizer=optimizer, manifest=manifest)

    def restore_physical(self, path, *, wait, expected_sha256, biology_manifest=None):
        """Restore a wait boundary into an already constructed matching session.

        Parameters
        ----------
        path : path-like
            Version-2 physical checkpoint.
        wait : PhysicalWait
            Matching wait driver for this session.
        expected_sha256 : str
            Authenticated continuation identity.
        biology_manifest : dict or None, optional
            Same immutable biological configuration supplied at save time.
        """
        from .h01_physical_checkpoint import restore_physical_checkpoint
        roots, optimizer, manifest = self._physical_context(wait, biology_manifest)
        restored = restore_physical_checkpoint(path, roots=roots, optimizer=optimizer, manifest=manifest,
                                               expected_sha256=expected_sha256)
        self.trainer.parameters = restored['parameters']
        self.trainer.muon_groups = restored['groups']
        self.trainer.updates = restored['updates']

    @classmethod
    def restore(cls, path, asset_root, trainer_type, *, expected_sha256=None, progress=None):
        """Verify immutable assets and reconstruct an episode-boundary continuation.

        Parameters
        ----------
        path, asset_root : path-like
            Checkpoint and content-addressed asset directory.
        trainer_type : type
            Existing Example 21 episode trainer.
        expected_sha256 : str, optional
            Coordinator-authenticated checkpoint digest.
        progress : callable, optional
            Construction progress callback.

        Returns
        -------
        H01Session
            Restored parameters, optimizer moments and completed-update count.
        """
        from pathlib import Path
        loaded = load_checkpoint(path, asset_root=asset_root, expected_sha256=expected_sha256)
        metadata = loaded['metadata']
        source = next(iter(loaded['topology'].to_dict()['sources'].values()))
        digest = source['archive_sha256']
        if digest not in metadata['assets']:
            raise ValueError('Source archive is absent from the verified asset inventory')
        archive = H01Archive(Path(asset_root)/digest)
        result = cls.build(loaded['topology'], archive, trainer_type, settings=metadata['settings'],
            assets=metadata['assets'], parameters=loaded['parameters'],
            input_pattern=(loaded['input_indices'], loaded['input_indptr']), progress=progress)
        result.trainer.muon_groups = restore_optimizer(loaded, result.trainer.muon_groups)
        result.trainer.updates = jnp.asarray(metadata['continuation']['completed_episodes'], dtype=jnp.int32)
        return result

    def mutate(self, topology, archive, *, progress=None, release_parent=False):
        """Rebuild mutated cables with inherited parameters and survivor moments.

        Parameters
        ----------
        topology : H01Topology
            Permitted child topology.
        archive : H01Archive
            Verified morphology source.
        progress : callable, optional
            Construction progress callback.
        release_parent : bool, optional
            Release this session's physical runtime before child construction.
            The immutable parent checkpoint remains the recovery source.

        Returns
        -------
        H01Session
            Real child model with new moments initialized to zero.
        """
        if self.settings.get('biology') is not None or self.model.stepper.chemistry is not None:
            raise ValueError('Mutation requires rebuilding biological maps; implicit reuse is unsupported')
        transported = remap_mutation(self.topology, topology, self.trainer.parameters,
            self.model.input_csr.indices, self.model.input_csr.indptr)
        trainer_type = type(self.trainer)
        parent_shapes = {key: value.shape for key, value in self.trainer.parameters.items()}
        parent_groups = jax.tree.map(np.array, self.trainer.muon_groups)
        parent_updates = int(self.trainer.updates)
        if release_parent:
            import gc
            self.released_updates = parent_updates
            self.__dict__.pop('_score_queries_compiled', None)
            self.model = self.learner = self.trainer = None
            jax.clear_caches()
            gc.collect()
        result = self.build(topology, archive, trainer_type, settings=self.settings, assets=self.assets,
            parameters=transported['parameters'], input_pattern=(transported['input_indices'], transported['input_indptr']),
            progress=progress, optimizer_slots=self.settings['optimizer']['slots'])
        groups = {}
        for name, old in parent_groups.items():
            groups[name] = remap_optimizer_group(old, result.trainer.muon_groups[name],
                transported['survivor_maps'][name], parent_shapes[name],
                result.trainer.parameters[name].shape)
        result.trainer.muon_groups = groups
        result.trainer.updates = jnp.asarray(parent_updates, dtype=jnp.int32)
        return result
