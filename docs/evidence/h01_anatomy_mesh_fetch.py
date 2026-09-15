"""Bounded retrieval of one released H01 mesh with decoding provenance."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

SOURCE = 'https://h01-release.storage.googleapis.com/data/20210601/proofread_104'


def acquire_mesh(volume, cell_id, *, lod=0, max_mesh_bytes=64 * 1024**2):
    """Decode a single cell only after checking its manifest payload size.

    Parameters
    ----------
    volume : cloudvolume.CloudVolume
        Read-only volume for the released proofread cells.
    cell_id : int
        Released segment identifier.
    lod : int, optional
        Explicit nonnegative level; zero is the finest released mesh.
    max_mesh_bytes : int, optional
        Maximum compressed mesh payload requested after the metadata reads.

    Returns
    -------
    tuple
        Decoded mesh and a serializable provenance record. Vertices are already
        transformed by CloudVolume into physical nanometres.
    """
    if type(lod) is not int or lod < 0 or max_mesh_bytes <= 0 or cell_id <= 0:
        raise ValueError('Positive cell/cap and explicit nonnegative LOD required.')
    manifest = volume.mesh.get_manifest(cell_id)
    if manifest is None:
        raise ValueError('Mesh manifest unavailable for the requested cell.')
    if int(manifest.segment_id) != cell_id or lod >= manifest.num_lods:
        raise ValueError('Manifest cell or LOD does not match the request.')
    sizes = np.asarray(manifest.fragment_offsets[lod])
    positions = np.asarray(manifest.fragment_positions[lod])
    if (sizes.ndim != 1 or not len(sizes) or not np.isfinite(sizes).all()
            or (sizes < 0).any() or not np.equal(sizes, np.floor(sizes)).all()
            or len(sizes) != int(manifest.num_fragments_per_lod[lod])
            or positions.shape != (len(sizes), 3) or not np.isfinite(positions).all()):
        raise ValueError('Malformed mesh fragment sizes.')
    payload_bytes = sum(int(size) for size in sizes)
    if payload_bytes == 0 or payload_bytes > max_mesh_bytes:
        raise ValueError(f'Mesh payload {payload_bytes} bytes outside cap {max_mesh_bytes}.')
    mesh = volume.mesh.get(cell_id, lod=lod, concat=True, progress=False)[cell_id]
    return mesh, {
        'cell_id': str(cell_id), 'lod': lod, 'num_lods': int(manifest.num_lods),
        'payload_bytes': payload_bytes, 'max_mesh_bytes': max_mesh_bytes,
        'fragments': len(sizes), 'nonempty_fragments': int((sizes > 0).sum()),
        'fragment_sizes_bytes': sizes.tolist(),
        'fragment_positions': np.asarray(manifest.fragment_positions[lod]).tolist(),
        'manifest_shard_offset': int(manifest.shard_offset),
        'manifest_shard_path': str(manifest.path),
        'chunk_shape': np.asarray(manifest.chunk_shape).tolist(),
        'grid_origin': np.asarray(manifest.grid_origin).tolist(),
        'vertex_offsets': np.asarray(manifest.vertex_offsets).tolist(),
        'lod_scales': np.asarray(manifest.lod_scales).tolist(),
        'volume_metadata': volume.info, 'mesh_metadata': volume.mesh.meta.info,
        'vertices_unit': 'nm', 'transform_applied_by': 'cloudvolume mesh decoder',
        'domain': 'whole released segment; no selected-SWC-component correspondence yet',
        'physiology': 'unqualified',
    }


def main(argv=None):
    """Retrieve a bounded mesh and write a new, non-overwriting artifact pair.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments; None reads the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-id', type=int, default=955432427)
    parser.add_argument('--lod', type=int, default=0)
    parser.add_argument('--max-mesh-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args(argv)
    array_path, record_path = args.output.with_suffix('.npz'), args.output.with_suffix('.json')
    if array_path.exists() or record_path.exists():
        raise FileExistsError('Use a new output prefix to preserve evidence.')
    from cloudvolume import CloudVolume
    volume = CloudVolume(SOURCE, cache=str(args.cache.resolve()), parallel=1,
                         progress=False, use_https=True)
    print(f'manifest: cell={args.cell_id} lod={args.lod}', flush=True)
    mesh, record = acquire_mesh(volume, args.cell_id, lod=args.lod,
                                max_mesh_bytes=args.max_mesh_bytes)
    record['source'] = SOURCE
    record['versions'] = {name: importlib.metadata.version(name)
                          for name in ('cloud-volume', 'DracoPy', 'numpy')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with array_path.open('xb') as stream:
        np.savez_compressed(stream, vertices=mesh.vertices, faces=mesh.faces)
    record['array_sha256'] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    with record_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'written: {record_path}; payload={record["payload_bytes"]}', flush=True)


if __name__ == '__main__':
    main()
