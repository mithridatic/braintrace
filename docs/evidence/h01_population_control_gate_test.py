"""Exercise delivery, silence, isolation and missing evidence boundaries."""

from copy import deepcopy

import numpy as np
import pytest

from docs.evidence.h01_population_control_gate import ATOL_MV, audit_controls, CONTROLS
from docs.evidence.h01_population_runtime_gate_test import evidence


@pytest.fixture
def controls(evidence):
    _, reference, _, _ = evidence
    for edge in reference['contacts']:
        edge['delay_ms'] = .01
    runs = {}
    for control in CONTROLS:
        build = deepcopy(reference)
        build.update(control=control, dt_ms=.005, duration_ms=.06)
        signs = {'ei':{1,-1}, 'e_only':{1}, 'i_only':{-1}, 'disconnected':set()}[control]
        for edge in build['contacts']:
            edge['enabled'] = edge['dale_sign'] in signs
        build['enabled_contacts'] = [e['annotation_id'] for e in build['contacts'] if e['enabled']]
        build['removed_contacts'] = [e['annotation_id'] for e in build['contacts'] if not e['enabled']]
        plan = dict(cells=104, control=control, dt_ms=.005, duration_ms=.06)
        arrays = {'time_ms':np.arange(1,13)*.005}
        for identity in reference['simulated_cell_ids']:
            for suffix in ('voltage', 'output_voltage'):
                arrays['cell_'+identity+'_'+suffix] = np.full((12,1), -65.)
            arrays['cell_'+identity+'_events'] = np.zeros((12,1), dtype=bool)
        for edge in build['contacts']:
            arrays['cell_'+edge['pre_cell']+'_events'][0,0] = True
            prefix = 'cell_'+edge['post_cell']+'_syn_'+edge['annotation_id']
            conductance = np.zeros((12,1))
            if edge['enabled']:
                conductance[3:,0] = .01*np.exp(-np.arange(9)*.005/2)
            arrays[prefix+'_g'] = conductance
            arrays[prefix+'_voltage'] = -65.+conductance
        runs[control] = (build,plan,arrays)
    return reference,runs


def test_delivery_pass_does_not_promote_physiology(controls):
    result = audit_controls(*controls)
    assert result['status'] == 'passed', result
    assert len(result['observations']) == 4
    assert all(row['observable_source_events'] == 1 for row in result['observations'])
    assert not result['physiology_qualified']
    assert not result['functional_inhibition_qualified']


@pytest.mark.parametrize('defect', [
    'missing_control', 'connected_pairs', 'invalid_runtime', 'wrong_label', 'missing_probe',
    'missing_all_probes', 'outside_receiver', 'negative', 'disabled_delivery',
    'silent', 'missing_arrival', 'late_arrival', 'different_time',
])
def test_control_failures(controls, defect):
    reference,runs = controls
    a = runs['ei'][2]
    if defect == 'missing_control': runs.pop('i_only')
    elif defect == 'connected_pairs': reference['contacts'][1]['pre_cell'] = '1'
    elif defect == 'invalid_runtime': a['cell_103_voltage'][0,0] = np.nan
    elif defect == 'wrong_label': runs['ei'][1]['control'] = 'i_only'
    elif defect == 'missing_probe': a.pop('cell_1_syn_E_g')
    elif defect == 'missing_all_probes':
        for _,_,arrays in runs.values(): arrays.pop('cell_1_syn_E_g')
    elif defect == 'outside_receiver': runs['e_only'][2]['cell_103_voltage'][0,0] += .1
    elif defect == 'negative': a['cell_1_syn_E_g'][1,0] = -.1
    elif defect == 'disabled_delivery': runs['disconnected'][2]['cell_1_syn_E_g'][1,0] = .01
    elif defect == 'silent':
        for _,_,arrays in runs.values(): arrays['cell_0_events'][:] = False
    elif defect == 'missing_arrival': a['cell_1_syn_E_g'][:] = 0
    elif defect == 'late_arrival':
        a['cell_1_syn_E_g'][:] = 0
        a['cell_1_syn_E_g'][10,0] = .01
    elif defect == 'different_time':
        build,plan,arrays = runs['disconnected']
        build['dt_ms'] = plan['dt_ms'] = .01
        build['duration_ms'] = plan['duration_ms'] = .12
        arrays['time_ms'] *= 2
    result = audit_controls(reference,runs)
    assert result['status'] == 'failed'
    assert result['failures']


def test_nondeterministic_accumulation_below_tolerance_passes(controls):
    reference, runs = controls
    runs['e_only'][2]['cell_103_voltage'][0, 0] += 9e-7
    result = audit_controls(reference, runs)
    assert result['status'] == 'passed', result
    assert result['tolerance_mv'] == ATOL_MV == 1e-5
    assert result['max_abs_difference_mv'] == pytest.approx(9e-7)


def test_difference_above_tolerance_fails_with_the_receiver_message(controls):
    reference, runs = controls
    runs['e_only'][2]['cell_103_voltage'][0, 0] += 2e-5
    result = audit_controls(reference, runs)
    assert result['status'] == 'failed'
    assert 'e_only: change outside removed receivers: cell_103_voltage' in result['failures']
    assert result['max_abs_difference_mv'] == pytest.approx(2e-5)


def test_identical_controls_report_zero_difference(controls):
    result = audit_controls(*controls)
    assert result['max_abs_difference_mv'] == 0.
