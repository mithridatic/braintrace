"""Ensure large, missing or misidentified mesh data cannot bypass the preflight."""

from types import SimpleNamespace
from unittest.mock import Mock
import json
import sys

import numpy as np

import pytest

from h01_anatomy_mesh_fetch import acquire_mesh, main


def volume():
    manifest = SimpleNamespace(segment_id=7, num_lods=1, num_fragments_per_lod=[3], fragment_offsets=[[3, 0, 5]],
                               fragment_positions=[[[0, 0, 0], [1, 0, 0], [2, 0, 0]]],
                               shard_offset=100, path='x.shard', chunk_shape=[1, 1, 1],
                               grid_origin=[0, 0, 0], vertex_offsets=[[0, 0, 0]], lod_scales=[1])
    return SimpleNamespace(info={'type': 'segmentation'}, mesh=SimpleNamespace(
        get_manifest=Mock(return_value=manifest), get=Mock(return_value={7: 'decoded'}),
        meta=SimpleNamespace(info={'transform': [16, 16, 33]})))


def test_size_checked_before_mesh_is_requested():
    vol = volume()
    with pytest.raises(ValueError, match='outside cap'):
        acquire_mesh(vol, 7, max_mesh_bytes=7)
    vol.mesh.get.assert_not_called()
    mesh, record = acquire_mesh(vol, 7, max_mesh_bytes=8)
    assert mesh == 'decoded'
    assert record['payload_bytes'] == 8
    assert record['nonempty_fragments'] == 2
    assert record['vertices_unit'] == 'nm'
    assert record['physiology'] == 'unqualified'


@pytest.mark.parametrize('change', ['missing', 'wrong_cell', 'bad_lod', 'negative',
                                    'fraction', 'nan', 'empty', 'zero'])
def test_invalid_manifest_never_fetches_payload(change):
    vol = volume()
    manifest = vol.mesh.get_manifest.return_value
    kwargs = {}
    if change == 'missing':
        vol.mesh.get_manifest.return_value = None
    elif change == 'wrong_cell':
        manifest.segment_id = 8
    elif change == 'bad_lod':
        kwargs['lod'] = 1
    else:
        manifest.fragment_offsets = [{'negative': [-1], 'fraction': [.5],
                                      'nan': [float('nan')], 'empty': [], 'zero': [0]}[change]]
    with pytest.raises(ValueError):
        acquire_mesh(vol, 7, **kwargs)
    vol.mesh.get.assert_not_called()


@pytest.mark.parametrize('kwargs', [{'lod': -1}, {'lod': .5}, {'max_mesh_bytes': 0}])
def test_invalid_request_is_rejected(kwargs):
    with pytest.raises(ValueError):
        acquire_mesh(volume(), 7, **kwargs)


def test_missing_fragment_or_position_is_rejected_before_download():
    vol = volume()
    vol.mesh.get_manifest.return_value.fragment_offsets = [[3, 5]]
    with pytest.raises(ValueError, match='fragment'):
        acquire_mesh(vol, 7)
    vol.mesh.get.assert_not_called()
    vol = volume()
    vol.mesh.get_manifest.return_value.fragment_positions = [[[0, 0, 0]]]
    with pytest.raises(ValueError, match='fragment'):
        acquire_mesh(vol, 7)
    vol.mesh.get.assert_not_called()


def test_cli_preserves_decoded_units_hash_and_existing_evidence(tmp_path, monkeypatch):
    vol = volume()
    vol.mesh.get.return_value = {7: SimpleNamespace(vertices=np.zeros((3, 3)),
                                                   faces=np.array([[0, 1, 2]]))}
    monkeypatch.setitem(sys.modules, 'cloudvolume', SimpleNamespace(CloudVolume=lambda *a, **kw: vol))
    monkeypatch.setattr('h01_anatomy_mesh_fetch.importlib.metadata.version', lambda name: 'test')
    prefix = tmp_path / 'out' / 'mesh'
    args = ['--cell-id', '7', '--output', str(prefix), '--cache', str(tmp_path / 'cache')]
    main(args)
    record = json.loads(prefix.with_suffix('.json').read_text())
    assert record['vertices_unit'] == 'nm'
    assert len(record['array_sha256']) == 64
    assert record['versions']['DracoPy'] == 'test'
    with pytest.raises(FileExistsError):
        main(args)
