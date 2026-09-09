"""H01 checkpoint identity, numerical clocks, assets and optimizer restoration."""

import json
import numpy as np
import pytest

from .h01_checkpoint import store_asset, save_checkpoint, load_checkpoint, restore_optimizer
from .h01_topology_test import _topology


def _save(tmp_path):
    source = tmp_path/'source.swc'
    source.write_text('1 1 0 0 0 2 -1\n')
    digest = store_asset(source, tmp_path/'assets')
    topology = _topology()
    parameters = {'input': np.zeros(0), 'recurrent': np.zeros(0),
                  'readout_weight': np.zeros((3, 360)), 'readout_bias': np.zeros(360)}
    optimizer = {'step': np.array(3, dtype=np.int32), 'moment': np.ones((3, 360))}
    path = tmp_path/'checkpoint.npz'
    save_checkpoint(path, topology=topology, parameters=parameters,
        input_indices=np.zeros(0, dtype=np.int32), input_indptr=np.zeros(442, dtype=np.int32),
        optimizer=optimizer, settings={'dt_ms': .005, 'event_ms': .1, 'substeps': 20,
            'precision': 64, 'solver': 'h01_staggered_calcium_implicit', 'dependencies': {'test': '1'}},
        continuation={'episode_boundary': True, 'completed_episodes': 3,
                      'stage_id': 'stage-1', 'parent_checkpoint_sha256': None}, assets=[digest])
    return path, digest, optimizer


def test_roundtrip_retains_large_string_ids_clocks_and_optimizer(tmp_path):
    path, _, template = _save(tmp_path)
    loaded = load_checkpoint(path, asset_root=tmp_path/'assets')
    assert loaded['topology'].to_dict()['active_cells'][0] == '5805562981'
    assert loaded['metadata']['settings']['substeps'] == 20
    restored = restore_optimizer(loaded, template)
    np.testing.assert_array_equal(restored['moment'], template['moment'])
    assert int(restored['step']) == 3


@pytest.mark.parametrize('action', ['missing', 'corrupt'])
def test_missing_or_corrupt_morphology_asset_rejected(tmp_path, action):
    path, digest, _ = _save(tmp_path)
    asset = tmp_path/'assets'/digest
    if action == 'missing':
        asset.unlink()
    else:
        asset.write_bytes(b'changed')
    with pytest.raises(ValueError, match='asset'):
        load_checkpoint(path, asset_root=tmp_path/'assets')


def test_cross_backend_checkpoint_fails_clearly(tmp_path):
    path = tmp_path/'single-compartment.npz'
    np.savez(path, format_version=np.array(1))
    with pytest.raises(ValueError, match='cross-backend'):
        load_checkpoint(path, asset_root=tmp_path)


def test_single_compartment_loader_rejects_h01_backend(tmp_path):
    from .arc_contracts import load_checkpoint as legacy_load
    path, _, _ = _save(tmp_path)
    with pytest.raises(ValueError, match='cross-backend'):
        legacy_load(path)


def test_array_tampering_and_optimizer_template_mismatch(tmp_path):
    path, _, _ = _save(tmp_path)
    loaded = load_checkpoint(path, asset_root=tmp_path/'assets')
    with pytest.raises(ValueError, match='optimizer'):
        restore_optimizer(loaded, {'different': np.zeros(2)})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    arrays['parameter/readout_bias'][0] = 1.
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='digest'):
        load_checkpoint(path, asset_root=tmp_path/'assets')


def test_invalid_clock_and_mid_episode_save_are_rejected(tmp_path):
    path, _, _ = _save(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    metadata = json.loads(arrays['h01_manifest'].tobytes())
    metadata['continuation']['episode_boundary'] = False
    arrays['h01_manifest'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='episode boundary'):
        load_checkpoint(path, asset_root=tmp_path/'assets')


@pytest.mark.parametrize('section,key,value', [
    ('settings', 'dt_ms', .0025), ('settings', 'precision', 32),
    ('settings', 'dependencies', {}), ('continuation', 'parent_checkpoint_sha256', 'invalid'),
])
def test_pinned_metadata_changes_fail(tmp_path, section, key, value):
    path, _, _ = _save(tmp_path)
    with np.load(path) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    metadata = json.loads(arrays['h01_manifest'].tobytes())
    metadata[section][key] = value
    arrays['h01_manifest'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_checkpoint(path, asset_root=tmp_path/'assets')


def test_existing_asset_corruption_and_checkpoint_resource_limits(tmp_path):
    path, digest, _ = _save(tmp_path)
    assert store_asset(tmp_path/'source.swc', tmp_path/'assets') == digest
    with pytest.raises(ValueError, match='allocation limit'):
        load_checkpoint(path, asset_root=tmp_path/'assets', max_bytes=1)
    with pytest.raises(ValueError, match='coordinator'):
        load_checkpoint(path, asset_root=tmp_path/'assets', expected_sha256='a'*64)
    (tmp_path/'assets'/digest).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        store_asset(tmp_path/'source.swc', tmp_path/'assets')


def test_array_inventory_and_optimizer_dtype_are_verified(tmp_path):
    path, _, template = _save(tmp_path)
    loaded = load_checkpoint(path, asset_root=tmp_path/'assets')
    template['moment'] = template['moment'].astype(np.float32)
    with pytest.raises(ValueError, match='dtype'):
        restore_optimizer(loaded, template)
    with np.load(path) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    arrays['unexpected'] = np.zeros(1)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='inventory'):
        load_checkpoint(path, asset_root=tmp_path/'assets')


@pytest.mark.parametrize('problem', ['duplicate', 'outside', 'nonfinite', 'shape'])
def test_semantically_invalid_parameter_or_encoder_arrays_rejected(tmp_path, problem):
    from .h01_checkpoint import save_checkpoint
    path, _, optimizer = _save(tmp_path)
    loaded = load_checkpoint(path, asset_root=tmp_path/'assets')
    parameters = loaded['parameters']
    indices = np.array([0, 0] if problem == 'duplicate' else [0, 4], dtype=np.int32)
    indptr = np.array([0, *([2]*441)], dtype=np.int32)
    parameters['input'] = np.zeros(2)
    if problem in ('nonfinite', 'shape'):
        indices = np.array([0, 1], dtype=np.int32)
        parameters['readout_bias'] = np.full(360, np.nan) if problem == 'nonfinite' else np.zeros(359)
    with pytest.raises(ValueError, match='duplicate|connectivity|finite|shape'):
        save_checkpoint(path, topology=loaded['topology'], parameters=parameters,
            input_indices=indices, input_indptr=indptr, optimizer=optimizer,
            settings=loaded['metadata']['settings'], continuation=loaded['metadata']['continuation'],
            assets=loaded['metadata']['assets'])
