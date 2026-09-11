"""Trainable H01 runtime construction and durable episode-boundary restoration."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
import hashlib
from pathlib import Path

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_network_init import init_h01_network_states
from .h01_arc_model import H01ArcModel
from .h01_checkpoint import load_checkpoint, save_checkpoint, restore_optimizer
from .h01_runtime import build_network
from .h01_remap import remap_mutation, remap_optimizer_group


@lru_cache(maxsize=1)
def _numerical_settings_cached():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/'braintrace'/'datasets').glob('h01*.py'))
    paths += [root/name for name in ('braintrace/_algorithm/sparse_pp_prop.py',
        'braintrace/_algorithm/sparse_io.py', 'braintrace/_compiler/sparse_io_graph.py',
        'braintrace/_compiler/sparse_support.py', 'braintrace/_compiler/sparse_influence.py',
        'examples/pp_prop/h01_arc_model.py', 'examples/pp_prop/h01_arc_execution.py')]
    implementation = {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths if not path.name.endswith('_test.py')}
    return dict(dt_ms=.005, event_ms=.1, substeps=20, precision=64,
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
    return dict(cached, implementation_sha256=dict(cached['implementation_sha256']),
                dependencies=dict(cached['dependencies']))


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
              parameters=None, input_pattern=None, progress=None):
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

        Returns
        -------
        H01Session
            Executable episode-boundary state.
        """
        settings = numerical_settings() if settings is None else dict(settings)
        installed = numerical_settings()
        if settings['dependencies'] != installed['dependencies']:
            raise ValueError('H01 checkpoint dependencies differ from this runtime')
        if settings['implementation_sha256'] != installed['implementation_sha256']:
            raise ValueError('Pinned H01 implementation differs from this runtime')
        if brainstate.environ.get('precision') != 64:
            raise ValueError('H01 sessions require an enclosing 64-bit precision context')
        network, records = build_network(topology, archive, solver=settings['solver'],
            max_cv_length_um=settings['max_cv_length_um'], progress=progress)
        del records
        init_h01_network_states(network, progress=progress)
        model = H01ArcModel(network, topology.to_dict()['active_cells'], seed=settings['seed'],
                            dt_ms=settings['dt_ms'], input_pattern=input_pattern,
                            checkpoint_substeps=settings['checkpoint_substeps'])
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
        learner = braintrace.pp_prop.sparse(model, settings['decay'], max_bytes=settings['factor_limit_bytes'])
        learner.compile_graph(jnp.zeros(441))
        trainer = trainer_type(learner, {name: state.value for name, state in states.items()})
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
        self.model.reset_episode(self.learner)
        boundary = dict(continuation or {}, episode_boundary=True,
            completed_episodes=int(self.trainer.updates), stage_id=stage_id,
            parent_checkpoint_sha256=parent_checkpoint_sha256)
        return save_checkpoint(path, topology=self.topology, parameters=self.trainer.parameters,
            input_indices=self.model.input_csr.indices, input_indptr=self.model.input_csr.indptr,
            optimizer=self.trainer.muon_groups, settings=self.settings, continuation=boundary, assets=self.assets)

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
        transported = remap_mutation(self.topology, topology, self.trainer.parameters,
            self.model.input_csr.indices, self.model.input_csr.indptr)
        trainer_type = type(self.trainer)
        parent_shapes = {key: value.shape for key, value in self.trainer.parameters.items()}
        parent_groups = jax.tree.map(np.array, self.trainer.muon_groups)
        parent_updates = int(self.trainer.updates)
        if release_parent:
            import gc
            self.released_updates = parent_updates
            self.model = self.learner = self.trainer = None
            jax.clear_caches()
            gc.collect()
        result = self.build(topology, archive, trainer_type, settings=self.settings, assets=self.assets,
            parameters=transported['parameters'], input_pattern=(transported['input_indices'], transported['input_indptr']),
            progress=progress)
        groups = {}
        for name, old in parent_groups.items():
            groups[name] = remap_optimizer_group(old, result.trainer.muon_groups[name],
                transported['survivor_maps'][name], parent_shapes[name],
                result.trainer.parameters[name].shape)
        result.trainer.muon_groups = groups
        result.trainer.updates = jnp.asarray(parent_updates, dtype=jnp.int32)
        return result
