"""Fullpopulation refinement checks reject omissions and late-cell failures."""
from copy import deepcopy
import numpy as np
import pytest
from docs.evidence.h01_population_runtime_gate_test import evidence
from docs.evidence.h01_population_refinement_gate import audit_refinement


def pair(evidence):
    build,reference,plan,arrays=evidence
    fb=deepcopy(build);fp=deepcopy(plan);fb['dt_ms']=fp['dt_ms']=.0025
    fa={k:np.repeat(v,2,axis=0) for k,v in arrays.items()}
    fa['time_ms']=np.arange(1,9)*.0025
    for key in fa:
        if key.endswith('_events'):fa[key][::2]=False
    return reference,(build,plan,arrays),(fb,fp,fa)


def test_all104_observed_and_silent_cells_do_not_promote_activity(evidence):
    r=audit_refinement(*pair(evidence))
    assert r['status']=='passed' and len(r['observations'])==104
    assert not r['activity_qualified'] and not r['physiology_qualified']


@pytest.mark.parametrize('site',['voltage','output_voltage'])
@pytest.mark.parametrize('error,passed',[(1.,True),(1.00001,False)])
def test_late_cell_voltage_tolerance(evidence,site,error,passed):
    ref,c,f=pair(evidence);f[2]['cell_103_'+site][7,0]+=error
    assert (audit_refinement(ref,c,f)['status']=='passed')==passed


@pytest.mark.parametrize('defect',['extra_event','missing_event','missing_cell','nan','wrong_dt','duration','control'])
def test_rejects_incomplete_or_unmatched_comparison(evidence,defect):
    ref,c,f=pair(evidence)
    if defect=='extra_event':f[2]['cell_103_events'][0,0]=True
    elif defect=='missing_event':f[2]['cell_0_events'][:]=False
    elif defect=='missing_cell':del f[2]['cell_103_voltage']
    elif defect=='nan':f[2]['cell_103_output_voltage'][0,0]=np.nan
    elif defect=='wrong_dt':f[1]['dt_ms']=.001
    elif defect=='duration':f[1]['duration_ms']=.01
    elif defect=='control':f[1]['control']='disconnected'
    assert audit_refinement(ref,c,f)['status']=='failed'


def test_counts_all_fine_events_not_only_matched_samples(evidence):
    ref,c,f=pair(evidence);f[2]['cell_0_events'][5,0]=False;f[2]['cell_0_events'][4,0]=True
    r=audit_refinement(ref,c,f)
    assert r['status']=='passed'
    assert r['observations'][0]['max_event_timing_error_ms']==pytest.approx(.0025)


@pytest.mark.parametrize('defect',['quarter_dt','different_duration','different_control','grid_roundoff'])
def test_individually_valid_runs_must_also_be_matched(evidence,defect):
    ref,c,f=pair(evidence)
    if defect in ('quarter_dt','different_duration'):
        for key,value in list(f[2].items()):
            f[2][key]=np.repeat(value,2,axis=0)
            if key.endswith('_events'):f[2][key][::2]=False
        if defect=='quarter_dt':f[0]['dt_ms']=f[1]['dt_ms']=.00125
        else:f[0]['duration_ms']=f[1]['duration_ms']=.04
        f[2]['time_ms']=np.arange(1,17)*f[1]['dt_ms']
    elif defect=='different_control':
        f[0]['control']=f[1]['control']='disconnected'
        for edge in f[0]['contacts']:edge['enabled']=False
        f[0]['enabled_contacts']=[];f[0]['removed_contacts']=['E','I']
    else:
        c[2]['time_ms']+=.9e-12;f[2]['time_ms']-=.9e-12
    r=audit_refinement(ref,c,f)
    assert r['status']=='failed'
    assert not any(message.startswith(('coarse:','fine:')) for message in r['failures'])


def test_equal_event_counts_can_fail_timing(evidence):
    ref,c,f=pair(evidence)
    for run in (c,f):
        run[0]['dt_ms']*=10;run[1]['dt_ms']*=10
        run[0]['duration_ms']*=10;run[1]['duration_ms']*=10
        run[2]['time_ms']*=10
    f[2]['cell_0_events'][:]=False;f[2]['cell_0_events'][0,0]=True
    r=audit_refinement(ref,c,f)
    assert r['status']=='failed'
    assert r['observations'][0]['max_event_timing_error_ms']>.05
