"""Atomic physical-state snapshots with exact live-template restoration.

These v2 snapshots require already constructed, manifest-matched live objects.
They do not reinterpret legacy episode-only checkpoints as biological state.
"""

import json
import os
from pathlib import Path
import tempfile
import zipfile

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from .h01_checkpoint import _inventory, _digest_file


def _pack(roots, optimizer):
    states = brainstate.graph.states(roots)
    arrays, records, templates = {}, {}, {}
    values = [(json.dumps(list(path)), state.value, state) for path, state in states.items()]
    values.append(('optimizer', optimizer, None))
    for index, (path, value, state) in enumerate(sorted(values, key=lambda row: row[0])):
        leaves, tree = jax.tree.flatten(value)
        names, key_types = [], []
        for leaf_index, leaf in enumerate(leaves):
            name = f'state/{index:06d}/{leaf_index:06d}'
            typed_key = hasattr(leaf, 'dtype') and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            # Public JAX key serialization only; no random draws or seeds here.
            key_types.append(str(jax.random.key_impl(leaf)) if typed_key else None)
            arrays[name] = np.asarray(jax.random.key_data(leaf) if typed_key else leaf)
            names.append(name)
        records[path] = dict(tree=str(tree), leaves=names, keys=key_types,
                             state_type=None if state is None else type(state).__qualname__)
        templates[path] = (state, tree)
    return arrays, records, templates


def save_physical_checkpoint(path, *, roots, optimizer, manifest):
    """Save all supplied model/learner/wait states without resetting them.

    Parameters
    ----------
    path : path-like
        Atomic NPZ destination.
    roots : dict
        Named model, learner and wait roots. Include every state-owning driver.
    optimizer : pytree
        Full optimizer moments and counters, including plain array leaves.
    manifest : dict
        Immutable topology, biological geometry/rates, assets, source hashes,
        dependency versions and numerical clocks. Restore requires exact equality.

    Returns
    -------
    str
        SHA256 digest of the durable snapshot.

    Notes
    -----
    Call only outside compiled execution after a complete physical step. The
    caller retains responsibility for including every state owner in ``roots``.
    """
    if not manifest or not roots:
        raise ValueError('Physical snapshots require immutable manifest and live roots')
    arrays, records, _ = _pack(roots, optimizer)
    metadata = dict(schema='h01-physical-checkpoint-v2', manifest=manifest,
                    records=records, arrays=_inventory(arrays))
    arrays['physical_manifest'] = np.frombuffer(json.dumps(metadata, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode(), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.h01-physical-', suffix='.npz')
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _digest_file(path)


def restore_physical_checkpoint(path, *, roots, optimizer, manifest,
                                expected_sha256, max_bytes=2**31):
    """Verify every state before atomically replacing live values.

    Parameters
    ----------
    path, roots, optimizer, manifest
        Destination runtime templates matching ``save_physical_checkpoint``.
    expected_sha256 : str
        Authenticated checkpoint identity from the continuation record.
    max_bytes : int, optional
        Total uncompressed allocation bound.

    Returns
    -------
    pytree
        Restored optimizer with live container types. All supplied BrainState
        states are restored in place, including units and typed random keys.

    Notes
    -----
    This function never executes a serialized Python object. Pytree definitions
    come only from the manifest-matched live runtime.
    """
    if _digest_file(path) != expected_sha256:
        raise ValueError('Physical checkpoint digest mismatch')
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        if len(names) != len(set(names)) or sum(entry.file_size for entry in archive.infolist()) > max_bytes:
            raise ValueError('Physical checkpoint has duplicate entries or exceeds allocation limit')
    with np.load(path, allow_pickle=False) as archive:
        if 'physical_manifest' not in archive.files:
            raise ValueError('Expected v2 physical checkpoint; legacy import is not implicit')
        metadata = json.loads(archive['physical_manifest'].tobytes())
        if metadata.get('schema') != 'h01-physical-checkpoint-v2' or metadata['manifest'] != manifest:
            raise ValueError('Physical checkpoint manifest differs from this runtime')
        if set(archive.files)-{'physical_manifest'} != set(metadata['arrays']):
            raise ValueError('Physical checkpoint array inventory mismatch')
        arrays = {name: np.asarray(archive[name]) for name in metadata['arrays']}
    if _inventory(arrays) != metadata['arrays']:
        raise ValueError('Physical checkpoint array integrity mismatch')
    current, records, templates = _pack(roots, optimizer)
    if records != metadata['records']:
        raise ValueError('Physical checkpoint state tree differs from this runtime')
    if any(value.shape != current[name].shape or value.dtype != current[name].dtype for name, value in arrays.items()):
        raise ValueError('Physical checkpoint shape or dtype differs from this runtime')
    restored = []
    for path, record in records.items():
        state, tree = templates[path]
        leaves = [jnp.asarray(arrays[name]) if key is None else jax.random.wrap_key_data(jnp.asarray(arrays[name]), impl=key)
                  for name, key in zip(record['leaves'], record['keys'])]
        restored.append((state, tree.unflatten(leaves)))
    restored_optimizer = None
    for state, value in restored:
        if state is None:
            restored_optimizer = value
        else:
            state.value = value
    return restored_optimizer
