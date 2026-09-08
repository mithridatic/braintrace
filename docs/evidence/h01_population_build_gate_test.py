"""Construction gate rejects partial populations and substituted anatomy."""

from copy import deepcopy
import json
import sys

import pytest

from docs.evidence import h01_population_build_gate as gate


@pytest.fixture
def evidence():
    ids = [str(i) for i in range(104)]
    sources = [dict(cell_id=i, component=0, source_sha256='source-'+i, passed=True) for i in ids]
    imports = dict(status='completed', passed=True, cells=sources, archive_sha256='archive')
    contacts = [dict(annotation_id='edge', pre_cell='0', post_cell='1', dale_sign=1,
                     construction_ready=True)]
    topology = dict(nodes=[dict(cell_id=i) for i in ids], contacts=contacts, archive_sha256='archive')
    cells = {i: dict(measured_anatomy=dict(member=i+'.0.swc', source_sha256='source-'+i,
                                        archive_sha256='archive'), donor='borrowed', n_compartments=1)
             for i in ids}
    build = dict(simulated_cell_ids=ids, cells=cells, compartments_by_cell={i:1 for i in ids},
                 n_compartments=104, donors={i:'borrowed' for i in ids}, control='ei',
                 contacts=[dict(contacts[0], enabled=True)], enabled_contacts=['edge'], removed_contacts=[])
    return build, imports, topology


def test_complete_build_does_not_promote_runtime_or_physiology(evidence):
    result = gate.audit_build(*evidence)
    assert result['status'] == 'passed'
    assert result['expected_cells'] == result['simulated_cells'] == 104
    assert result['runtime_qualified'] is False
    assert result['physiology_qualified'] is False


@pytest.mark.parametrize('defect', [
    'missing_cell', 'duplicate_cell', 'substituted_cell', 'missing_evidence',
    'source_hash', 'component', 'archive_hash', 'negative_count', 'fractional_count',
    'total_count', 'per_cell_count', 'missing_donor', 'changed_donor',
    'unsupported_contact', 'duplicate_contact', 'disabled_contact', 'changed_endpoint',
    'changed_sign', 'removed_contact', 'wrong_control', 'failed_import',
    'partial_import', 'duplicate_topology', 'topology_archive',
])
def test_rejects_corrupt_or_incomplete_evidence(evidence, defect):
    b, imports, topology = deepcopy(evidence)
    if defect == 'missing_cell': b['simulated_cell_ids'].pop()
    elif defect == 'duplicate_cell': b['simulated_cell_ids'].append('0')
    elif defect == 'substituted_cell': b['simulated_cell_ids'][-1] = 'other'
    elif defect == 'missing_evidence': b['cells'].pop('0')
    elif defect == 'source_hash': b['cells']['0']['measured_anatomy']['source_sha256'] = 'other'
    elif defect == 'component': b['cells']['0']['measured_anatomy']['member'] = '0.1.swc'
    elif defect == 'archive_hash': b['cells']['0']['measured_anatomy']['archive_sha256'] = 'other'
    elif defect == 'negative_count': b['compartments_by_cell']['0'] = -1
    elif defect == 'fractional_count': b['compartments_by_cell']['0'] = 1.5
    elif defect == 'total_count': b['n_compartments'] += 1
    elif defect == 'per_cell_count': b['cells']['0']['n_compartments'] += 1
    elif defect == 'missing_donor': b['donors'].pop('0')
    elif defect == 'changed_donor': b['donors']['0'] = 'different'
    elif defect == 'unsupported_contact': b['contacts'][0]['annotation_id'] = 'unverified'
    elif defect == 'duplicate_contact': b['contacts'].append(deepcopy(b['contacts'][0]))
    elif defect == 'disabled_contact': b['contacts'][0]['enabled'] = False
    elif defect == 'changed_endpoint': b['contacts'][0]['post_cell'] = '2'
    elif defect == 'changed_sign': b['contacts'][0]['dale_sign'] = -1
    elif defect == 'removed_contact': b['removed_contacts'] = ['edge']
    elif defect == 'wrong_control': b['control'] = 'disconnected'
    elif defect == 'failed_import': imports['cells'][0]['passed'] = False
    elif defect == 'partial_import': imports['status'] = 'running'
    elif defect == 'duplicate_topology': topology['nodes'].append({'cell_id':'0'})
    elif defect == 'topology_archive': topology['archive_sha256'] = 'other'
    result = gate.audit_build(b, imports, topology)
    assert result['status'] == 'failed'
    assert result['failures']


@pytest.mark.parametrize('valid', [False, True])
def test_cli_writes_hashed_verdict_and_rejects_failure(evidence, tmp_path, monkeypatch, valid):
    b, imports, topology = evidence
    if not valid:
        b['simulated_cell_ids'].pop()
    arguments = ['gate']
    for name, value in [('build', b), ('imports', imports), ('topology', topology)]:
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps(value))
        arguments += ['--'+name, str(path)]
    output = tmp_path/'decision.json'
    monkeypatch.setattr(sys, 'argv', arguments+['--output', str(output)])
    if valid:
        gate.main()
    else:
        with pytest.raises(SystemExit) as caught:
            gate.main()
        assert caught.value.code == 1
    result = json.loads(output.read_text())
    assert result['status'] == ('passed' if valid else 'failed')
    assert all(len(record['sha256']) == 64 for record in result['inputs'].values())
