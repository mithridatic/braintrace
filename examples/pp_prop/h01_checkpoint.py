"""Versioned H01 episode-boundary checkpoints with verified immutable assets."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import jax
import numpy as np

from .h01_topology import H01Topology


_ASSET_DIGEST_CACHE = {}


def _digest_file(path):
    path_obj = Path(path)
    try:
        st = path_obj.stat()
        key = (str(path_obj.resolve()), st.st_mtime_ns, st.st_size)
        if key in _ASSET_DIGEST_CACHE:
            return _ASSET_DIGEST_CACHE[key]
    except OSError:
        key = None
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    result = digest.hexdigest()
    if key is not None:
        _ASSET_DIGEST_CACHE[key] = result
    return result


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def store_asset(source, asset_root):
    """Copy an immutable asset into a content-addressed store, once.

    Parameters
    ----------
    source, asset_root : path-like
        Source file and destination asset directory.

    Returns
    -------
    str
        SHA256 digest used as the asset filename.
    """
    source, root = Path(source), Path(asset_root)
    digest = _digest_file(source)
    root.mkdir(parents=True, exist_ok=True)
    destination = root/digest
    if destination.exists():
        if _digest_file(destination) != digest:
            raise ValueError('Existing content-addressed asset is corrupt')
        return digest
    descriptor, name = tempfile.mkstemp(dir=root, prefix='.asset-')
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if _digest_file(temporary) != digest:
            raise ValueError('Source asset changed while copying')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _validate_metadata(metadata):
    if metadata.get('schema') != 'h01-arc-checkpoint-v1':
        raise ValueError('Expected H01 checkpoint; cross-backend resume is unsupported')
    topology = H01Topology.from_dict(metadata['topology'])
    settings, continuation = metadata['settings'], metadata['continuation']
    dt, event, substeps = (settings[name] for name in ('dt_ms', 'event_ms', 'substeps'))
    if (not math.isfinite(dt) or dt <= 0 or event != .1 or type(substeps) is not int
            or substeps < 1 or not math.isclose(dt*substeps, event, rel_tol=0, abs_tol=1e-12)
            or settings['precision'] != 64 or not settings['solver'] or not settings['dependencies']):
        raise ValueError('H01 checkpoint has inconsistent pinned numerical clocks or dependencies')
    if (continuation.get('episode_boundary') is not True
            or type(continuation.get('completed_episodes')) is not int
            or continuation['completed_episodes'] < 0 or not continuation.get('stage_id')):
        raise ValueError('H01 continuation must be a durable episode boundary')
    parent = continuation.get('parent_checkpoint_sha256')
    if parent is not None and not _valid_digest(parent):
        raise ValueError('Invalid parent continuation digest')
    if not metadata['assets'] or any(not _valid_digest(value) for value in metadata['assets']):
        raise ValueError('Morphology assets must use content-addressed SHA256 identities')
    return topology


def _validate_arrays(topology, arrays):
    document = topology.to_dict()
    cells, contacts = len(document['active_cells']), len(document['active_contacts'])
    indices, indptr = arrays['input_indices'], arrays['input_indptr']
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or indptr.shape != (442,) or not np.issubdtype(indptr.dtype, np.integer)
            or indptr[0] != 0 or indptr[-1] != len(indices) or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0) or np.any(indices >= cells)):
        raise ValueError('Invalid H01 encoder connectivity')
    for start, end in zip(indptr[:-1], indptr[1:]):
        if end - start > 1 and len(set(indices[start:end])) != end-start:
            raise ValueError('An encoder feature cannot have duplicate cell targets')
    shapes = {'input': (len(indices),), 'recurrent': (contacts,),
              'readout_weight': (cells, 360), 'readout_bias': (360,)}
    for name, shape in shapes.items():
        value = arrays['parameter/'+name]
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError('Invalid H01 parameter shape or value: '+name)


def _inventory(arrays):
    result = {}
    for name, value in arrays.items():
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
            raise ValueError('Checkpoint arrays must be numeric and non-object')
        if not np.isfinite(value).all():
            raise ValueError('Checkpoint arrays must be finite')
        result[name] = {'shape': list(value.shape), 'dtype': value.dtype.str,
                        'sha256': hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()}
    return result


def save_checkpoint(path, *, topology, parameters, input_indices, input_indptr,
                    optimizer, settings, continuation, assets):
    """Atomically persist a complete H01 continuation at an episode boundary.

    Parameters
    ----------
    path : path-like
        Destination NPZ file.
    topology : H01Topology
        Immutable source, instance, contact and historical records.
    parameters : mapping
        Input, recurrent, readout_weight and readout_bias arrays.
    input_indices, input_indptr : arrays
        Encoder CSR connectivity in active-instance order.
    optimizer : pytree
        Actual grouped optimizer state, including moments and counters.
    settings : dict
        Pinned cable/event clocks, solver, precision and dependencies.
    continuation : dict
        Completed-episode count, stage ID and parent checkpoint identity.
    assets : sequence of str
        Digests of immutable files in the associated content-addressed store.

    Returns
    -------
    str
        SHA256 of the durable checkpoint bytes.
    """
    leaves, tree = jax.tree.flatten(optimizer)
    arrays = {'parameter/'+name: np.asarray(value) for name, value in parameters.items()}
    arrays.update(input_indices=np.asarray(input_indices), input_indptr=np.asarray(input_indptr))
    arrays.update({f'optimizer/{i:06d}': np.asarray(value) for i, value in enumerate(leaves)})
    metadata = dict(schema='h01-arc-checkpoint-v1', topology=topology.to_dict(),
        settings=settings, continuation=continuation, assets=list(assets),
        optimizer_tree=str(tree), optimizer_leaves=len(leaves), arrays=_inventory(arrays))
    _validate_metadata(metadata)
    _validate_arrays(topology, arrays)
    arrays['h01_manifest'] = np.frombuffer(json.dumps(metadata, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode(), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.h01-checkpoint-', suffix='.npz')
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


def load_checkpoint(path, *, asset_root, expected_sha256=None, max_bytes=2**31):
    """Verify and load a versioned H01 checkpoint without executing serialized code.

    Parameters
    ----------
    path, asset_root : path-like
        Checkpoint and content-addressed asset directory.
    expected_sha256 : str, optional
        Coordinator-authenticated checkpoint identity.
    max_bytes : int, optional
        Maximum total uncompressed archive size before array allocation.

    Returns
    -------
    dict
        Verified metadata, topology, parameters, encoder pattern and optimizer leaves.
    """
    if expected_sha256 is not None and _digest_file(path) != expected_sha256:
        raise ValueError('Checkpoint digest differs from the coordinator continuation')
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > max_bytes:
            raise ValueError('H01 checkpoint exceeds the allocation limit')
    with np.load(path, allow_pickle=False) as archive:
        if 'h01_manifest' not in archive.files:
            raise ValueError('Expected H01 checkpoint; cross-backend resume is unsupported')
        metadata = json.loads(archive['h01_manifest'].tobytes())
        topology = _validate_metadata(metadata)
        if set(archive.files) - {'h01_manifest'} != set(metadata['arrays']):
            raise ValueError('Checkpoint array inventory differs from its manifest')
        arrays = {name: np.array(archive[name]) for name in metadata['arrays']}
    if _inventory(arrays) != metadata['arrays']:
        raise ValueError('Checkpoint array digest, shape or dtype mismatch')
    _validate_arrays(topology, arrays)
    for digest in metadata['assets']:
        asset = Path(asset_root)/digest
        if not asset.is_file() or _digest_file(asset) != digest:
            raise ValueError('Missing or corrupt content-addressed morphology asset: '+digest)
    return dict(metadata=metadata, topology=topology,
        parameters={name.removeprefix('parameter/'): value for name, value in arrays.items() if name.startswith('parameter/')},
        input_indices=arrays['input_indices'], input_indptr=arrays['input_indptr'],
        optimizer_leaves=tuple(arrays[f'optimizer/{i:06d}'] for i in range(metadata['optimizer_leaves'])))


def restore_optimizer(loaded, template):
    """Restore optimizer leaves into the verified runtime's actual optimizer type.

    Parameters
    ----------
    loaded : dict
        Verified result from :func:`load_checkpoint`.
    template : pytree
        Newly initialized optimizer state for the same active topology.

    Returns
    -------
    pytree
        Restored optimizer with original moments, counters and container types.
    """
    leaves, tree = jax.tree.flatten(template)
    saved = loaded['optimizer_leaves']
    if str(tree) != loaded['metadata']['optimizer_tree'] or len(leaves) != len(saved):
        raise ValueError('H01 optimizer structure differs from the checkpoint')
    for value, expected in zip(saved, leaves):
        expected = np.asarray(expected)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('H01 optimizer shape or dtype differs from the checkpoint')
    return tree.unflatten(saved)
