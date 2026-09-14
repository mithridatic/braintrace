"""Provenance, partial-domain, artifact-integrity and CLI regression tests."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from h01_anatomy_area_report import audit_case, main


@pytest.fixture
def case(tmp_path):
    raw = b'1 1 0 0 0 1000 -1\n2 1 100 0 0 1000 1\n'
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('7.0.swc', raw)
        z.writestr('7.1.swc', '1 2 1000 0 0 100 -1\n')
        z.writestr('70.0.swc', 'unrelated')
    (tmp_path / 'converted.swc').write_text('1 1 0 0 0 2 -1\n2 3 3.2 0 0 .05 1\n')
    donor = tmp_path / 'donor.json'
    donor.write_text(json.dumps({
        'morphology': 'converted.swc', 'conversion': {'soma_radius_um': 2},
        'morphology_source': {'member': '7.0.swc',
                              'member_sha256': hashlib.sha256(raw).hexdigest()}}))
    prefix = tmp_path / 'mesh'
    np.savez(prefix.with_suffix('.npz'), vertices=[[0, 0, 0], [1000, 0, 0], [0, 1000, 0]],
             faces=[[0, 1, 2]])
    prefix.with_suffix('.json').write_text(json.dumps({
        'cell_id': '7', 'vertices_unit': 'nm', 'lod': 0, 'domain': 'whole segment',
        'source': 'https://h01-release.storage.googleapis.com/data/20210601/proofread_104',
        'array_sha256': hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest()}))
    return archive, donor, prefix


def test_partial_component_never_becomes_whole_mesh_ratio(case):
    archive, donor, mesh = case
    result = audit_case(archive, donor, mesh_prefix=mesh)
    assert set(result['source_components']) == {'7.0.swc', '7.1.swc'}
    assert result['surface_mesh']['geometry']['area_um2'] == pytest.approx(.5)
    assert result['matched_membrane_area_ratio'] is None
    assert result['anatomy_transfer'] == 'unqualified'
    assert result['conversion_observations']['converted_nodes_at_radius_floor'] == 1
    # These are H01 annotations: the fixture has dendrite code 1, no soma code 3.
    assert result['conversion_observations']['source_soma_labeled_nodes'] == 0
    assert result['selected_source_geometry']['length_um'] == pytest.approx(3.2)


def test_wrong_archive_member_hash_rejected(case):
    archive, donor, _ = case
    record = json.loads(donor.read_text())
    record['morphology_source']['member_sha256'] = 'wrong'
    donor.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='registered transfer input'):
        audit_case(archive, donor)


@pytest.mark.parametrize('field', ['array_sha256', 'cell_id', 'source', 'vertices_unit'])
def test_wrong_mesh_metadata_rejected(case, field):
    archive, donor, prefix = case
    record = json.loads(prefix.with_suffix('.json').read_text())
    record[field] = 'wrong'
    prefix.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='mismatch'):
        audit_case(archive, donor, mesh_prefix=prefix)


def test_cli_works_without_mesh_and_never_overwrites(case, tmp_path):
    archive, donor, _ = case
    output = tmp_path / 'new' / 'audit.json'
    args = ['--archive', str(archive), '--donor-manifest', str(donor), '--output', str(output)]
    main(args)
    assert json.loads(output.read_text())['surface_mesh'] is None
    with pytest.raises(FileExistsError):
        main(args)
