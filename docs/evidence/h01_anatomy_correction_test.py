"""Mutation checks ensure the correction proves actual source preservation."""

import hashlib
import json
import zipfile

import pytest

from h01_anatomy_area_audit import read_swc
from h01_anatomy_correction import main, verify_export
from h01_e_morphology_swap import convert, prepare

SOURCE = '10 3 0 0 0 3000 -1\n20 1 1000 0 0 32 10\n30 -1 2000 0 0 40 20\n'


@pytest.mark.parametrize('mutation', ['position', 'radius', 'parent', 'label', 'mapping', 'missing_label'])
def test_geometry_or_annotation_changes_are_rejected(mutation):
    text, conversion = convert(SOURCE)
    exported = read_swc(text)
    if mutation == 'position':
        exported[1, 2] += 1
    elif mutation == 'radius':
        exported[1, 5] = .05
    elif mutation == 'parent':
        exported[2, 6] = 1
    elif mutation == 'label':
        conversion['source_annotation_by_node_id']['20'] = 3
    elif mutation == 'mapping':
        conversion['source_node_id_by_export_id']['2'] = 10
    else:
        del conversion['source_annotation_by_node_id']['30']
    with pytest.raises(ValueError):
        verify_export(read_swc(SOURCE), exported, conversion)


def test_correction_cli_checks_source_and_artifact_hashes(tmp_path):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('7.0.swc', SOURCE)
    prepared = tmp_path / 'corrected'
    new = prepare('7', 0, prepared, archive=archive)
    old = dict(new, conversion={'apical_cable_um': 1, 'basal_cable_um': 2})
    old['morphology'] = 'old.swc'
    (tmp_path / 'old.swc').write_text('1 1 0 0 0 6 -1\n2 3 100 0 0 1 1\n')
    donor_path = tmp_path / 'donor.json'
    donor_path.write_text(json.dumps(old))
    output = tmp_path / 'decision.json'
    args = ['--archive', str(archive), '--historical-donor', str(donor_path),
            '--corrected-anatomy', str(prepared / 'anatomy.json'), '--output', str(output)]
    main(args)
    result = json.loads(output.read_text())
    assert result['nodes_checked'] == 3 and result['edges_checked'] == 2
    assert result['status'] == 'passed'
    assert result['historical_transfer_interpretation'] == 'invalidated_by_conversion_defect'
    with pytest.raises(FileExistsError):
        main(args)
    args[-1] = str(tmp_path / 'other.json')
    (prepared / new['morphology']).write_text('changed')
    with pytest.raises(ValueError, match='Corrected SWC hash'):
        main(args)
    new['morphology_source']['member_sha256'] = 'bad'
    (prepared / 'anatomy.json').write_text(json.dumps(new))
    with pytest.raises(ValueError, match='different sources'):
        main(args)
    old['morphology_source']['member_sha256'] = 'bad'
    donor_path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='Original archive member'):
        main(args)
