"""Small command controls and independent gain-transfer predictions."""

import numpy as np
import pytest

from h01_l1_control import extract_test_pulse, gain_transfer


def source():
    t=np.arange(0.,100.,.04)
    v=np.full_like(t,-90.);v[(t>=45)&(t<55.04)]=-80.
    i=np.full_like(t,-7.)
    pulse=(t>=45)&(t<55.04)
    i[pulse]+=3+50*np.exp(-(t[pulse]-45)/.3)
    return dict(time_ms=t,command_voltage_mv=v,total_current_pa=i)


def test_extract_preserves_clocks_and_baseline_offset():
    s=source();a,m=extract_test_pulse(s)
    np.testing.assert_array_equal(a['time_ms'],s['time_ms'][875:1625])
    np.testing.assert_array_equal(a['centered_current_pa'],a['total_current_pa']+7)
    assert m['baseline_mean_pa']==-7
    assert m['step_mv']==10
    assert m['apparent_late_slope_ns']==pytest.approx(.3,abs=1e-8)
    assert not m['membrane_conductance_qualified']


@pytest.mark.parametrize('fault',['nan','clock','short','rank','offgrid','extra','return','zero','late'])
def test_extract_rejects_invalid_protocol(fault):
    s=source()
    if fault=='nan':s['total_current_pa'][3]=np.nan
    if fault=='clock':s['time_ms'][3]+=.01
    if fault=='short':s={k:v[:1500] for k,v in s.items()}
    if fault=='rank':s['time_ms']=s['time_ms'][:,None]
    if fault=='offgrid':s['time_ms']+=.01
    if fault=='extra':s['command_voltage_mv'][1140]=-70
    if fault=='return':s['command_voltage_mv'][1400]=-70
    if fault=='zero':s['command_voltage_mv'][:]=-90
    if fault=='late':s['command_voltage_mv'][1300]=-70
    with pytest.raises(ValueError):extract_test_pulse(s)


def records(gain=1.6):
    s=source();r,_=extract_test_pulse(s)
    target={k:v.copy() for k,v in r.items()}
    target['centered_current_pa']*=gain
    phase=np.arange(0.,100.,.04)
    main=dict(phase_ms=phase,command_voltage_mv=np.full_like(phase,-20.),
              centered_current_pa=20+30*np.exp(-phase/40))
    other={k:v.copy() for k,v in main.items()};other['centered_current_pa']*=gain
    return r,target,main,other


def test_gain_fits_control_and_predicts_independent_main():
    r,t,m,n=records()
    a,meta=gain_transfer(r,t,m,n)
    assert meta['gain']==pytest.approx(1.6)
    np.testing.assert_allclose(a['main_predicted_pa'],n['centered_current_pa'],atol=1e-12)
    np.testing.assert_allclose(a['main_residual_pa'],0,atol=1e-12)
    np.testing.assert_array_equal(a['main_phase_ms'],m['phase_ms'])
    assert not meta['physiological_qualification']


def test_main_response_cannot_change_control_fitted_gain():
    r,t,m,n=records();n['centered_current_pa']+=20
    a,meta=gain_transfer(r,t,m,n)
    assert meta['gain']==pytest.approx(1.6)
    np.testing.assert_allclose(a['main_residual_pa'],20,atol=1e-12)


@pytest.mark.parametrize('fault',['zero','negative','nan','phase','command','mainphase','maincommand','shape','mask','shortmain'])
def test_gain_rejects_incompatible_inputs(fault):
    r,t,m,n=records()
    if fault=='zero':r['centered_current_pa'][:]=0
    if fault=='negative':t['centered_current_pa']*=-1
    if fault=='nan':n['centered_current_pa'][3]=np.nan
    if fault=='phase':t['phase_ms']+=.04
    if fault=='command':t['command_voltage_mv']+=1
    if fault=='mainphase':n['phase_ms']+=.04
    if fault=='maincommand':n['command_voltage_mv']+=1
    if fault=='shape':n['centered_current_pa']=n['centered_current_pa'][:-1]
    if fault=='mask':t['pulse_mask'][10]=True
    if fault=='shortmain':
        for d in (m,n):
            for k in d:d[k]=d[k][:-1]
    with pytest.raises(ValueError):gain_transfer(r,t,m,n)
