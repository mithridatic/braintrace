"""Reject partial or inconsistent full-population runtime evidence."""

from copy import deepcopy
import hashlib
import json
import sys

import numpy as np
import pytest

from docs.evidence import h01_population_runtime_gate as gate


@pytest.fixture
def evidence():
    ids = [str(i) for i in range(104)]
    contacts = [dict(annotation_id='E', pre_cell='0', post_cell='1', dale_sign=1, enabled=True),
                dict(annotation_id='I', pre_cell='2', post_cell='3', dale_sign=-1, enabled=True)]
    reference = dict(simulated_cell_ids=ids, cells={i:dict(donor='borrowed') for i in ids},
                     donors={i:'borrowed' for i in ids}, compartments_by_cell={i:1 for i in ids},
                     n_compartments=104, contacts=contacts, control='ei',
                     enabled_contacts=['E','I'], removed_contacts=[])
    build = dict(deepcopy(reference), dt_ms=.005, duration_ms=.02)
    plan = dict(cells=104, control='ei', dt_ms=.005, duration_ms=.02)
    arrays = {'time_ms': np.arange(1,5)*.005}
    for identity in ids:
        arrays['cell_'+identity+'_voltage'] = np.full((4,1), -65.)
        arrays['cell_'+identity+'_output_voltage'] = np.full((4,1), -65.)
        arrays['cell_'+identity+'_events'] = np.zeros((4,1), dtype=bool)
    arrays['cell_0_events'][2,0] = True
    return build, reference, plan, arrays


@pytest.mark.parametrize('control', ['ei','e_only','i_only','disconnected'])
def test_all_controls_keep_full_population_and_report_events(evidence, control):
    build, reference, plan, arrays = evidence
    build['control'] = plan['control'] = control
    signs = {'ei':{-1,1}, 'e_only':{1}, 'i_only':{-1}, 'disconnected':set()}[control]
    for edge in build['contacts']:
        edge['enabled'] = edge['dale_sign'] in signs
    build['enabled_contacts'] = [e['annotation_id'] for e in build['contacts'] if e['enabled']]
    build['removed_contacts'] = [e['annotation_id'] for e in build['contacts'] if not e['enabled']]
    result = gate.audit_runtime(*evidence)
    assert result['status'] == 'passed'
    assert len(result['observations']) == 104
    assert result['observations'][0]['spike_count'] == 1
    assert result['observations'][0]['spike_times_ms'] == [.015]
    assert not result['activity_qualified'] and not result['physiology_qualified']


@pytest.mark.parametrize('defect', [
    'missing_voltage','missing_events','wrong_shape','wrong_time','nan_voltage','nan_events',
    'integer_events','wrong_model','wrong_control','wrong_contact','unknown_array',
    'wrong_duration','bad_dt','fractional_steps','duplicate_cell','wrong_cell_count','text_time',
])
def test_rejects_incomplete_or_corrupt_runtime(evidence, defect):
    b, reference, plan, arrays = evidence
    if defect == 'missing_voltage': arrays.pop('cell_103_voltage')
    elif defect == 'missing_events': arrays.pop('cell_103_events')
    elif defect == 'wrong_shape': arrays['cell_1_output_voltage'] = np.zeros((3,1))
    elif defect == 'wrong_time': arrays['time_ms'] += .005
    elif defect == 'nan_voltage': arrays['cell_1_voltage'][0,0] = np.nan
    elif defect == 'nan_events': arrays['cell_1_events'] = np.full((4,1), np.nan)
    elif defect == 'integer_events': arrays['cell_1_events'] = np.zeros((4,1), dtype=int)
    elif defect == 'wrong_model': b['cells']['0']['donor'] = 'changed'
    elif defect == 'wrong_control': b['control'] = 'disconnected'
    elif defect == 'wrong_contact': b['contacts'][0]['post_cell'] = '3'
    elif defect == 'unknown_array': arrays['cell_other_voltage'] = np.zeros((4,1))
    elif defect == 'wrong_duration': b['duration_ms'] = .01
    elif defect == 'bad_dt': plan['dt_ms'] = 0
    elif defect == 'fractional_steps': plan['duration_ms'] = .021
    elif defect == 'duplicate_cell': reference['simulated_cell_ids'].append('0')
    elif defect == 'wrong_cell_count': plan['cells'] = 40
    elif defect == 'text_time': arrays['time_ms'] = np.array(['bad']*4)
    result = gate.audit_runtime(*evidence)
    assert result['status'] == 'failed'
    assert result['failures']


@pytest.mark.parametrize('outcome', ['pass','bad_trace','bad_anchor'])
def test_cli_binds_reference_hash_and_writes_decision(evidence, tmp_path, monkeypatch, outcome):
    build, reference, plan, arrays = evidence
    paths = {}
    for name, value in [('build',build),('reference',reference),('plan',plan)]:
        paths[name] = tmp_path/(name+'.json')
        paths[name].write_text(json.dumps(value))
    digest = hashlib.sha256(paths['reference'].read_bytes()).hexdigest()
    anchor = dict(status='passed', inputs={'build':{'sha256':digest if outcome!='bad_anchor' else 'wrong'}})
    paths['construction-decision'] = tmp_path/'construction.json'
    paths['construction-decision'].write_text(json.dumps(anchor))
    if outcome == 'bad_trace': arrays['cell_0_voltage'][0,0] = np.inf
    paths['traces'] = tmp_path/'traces.npz'
    np.savez(paths['traces'], **arrays)
    paths['output'] = tmp_path/'decision.json'
    arguments = ['gate']
    for name,path in paths.items(): arguments.extend(['--'+name,str(path)])
    monkeypatch.setattr(sys,'argv',arguments)
    if outcome == 'bad_anchor':
        with pytest.raises(ValueError, match='gate-verified'):
            gate.main()
        assert not paths['output'].exists()
    else:
        if outcome == 'bad_trace':
            with pytest.raises(SystemExit) as error:
                gate.main()
            assert error.value.code == 1
        else:
            gate.main()
        result = json.loads(paths['output'].read_text())
        assert result['status'] == ('passed' if outcome=='pass' else 'failed')
        assert result['inputs']['reference']['sha256'] == digest


def _shrink(evidence, count):
    build, reference, plan, arrays = evidence
    ids = [str(i) for i in range(count)]
    for record in (reference, build):
        record['simulated_cell_ids'] = ids
        for key in ('cells', 'donors', 'compartments_by_cell'):
            record[key] = {i: record[key][i] for i in ids}
        record['n_compartments'] = count
    for name in list(arrays):
        if name != 'time_ms' and name.split('_', 2)[1] not in ids:
            arrays.pop(name)
    plan['cells'] = count


def test_kept_subset_passes_when_plan_count_matches(evidence):
    _shrink(evidence, 4)
    result = gate.audit_runtime(*evidence)
    assert result['status'] == 'passed' and len(result['observations']) == 4


def test_kept_subset_rejects_plan_count_mismatch(evidence):
    _shrink(evidence, 4)
    evidence[2]['cells'] = 5
    result = gate.audit_runtime(*evidence)
    assert 'reference and plan must cover exactly the same unique cells' in result['failures']
