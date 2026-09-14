"""Initial-state observation from an earlier window with frozen global kinetics."""

import numpy as np
import pytest

from h01_l1_initial_state import observe_state,conditioned_currents
from h01_l1_recovery_state import INITIAL,state_currents


def synthetic():
    p=INITIAL.copy()
    pulse=np.arange(0.,300.,.5);baseline=np.arange(-100.,-10.,.5)
    duration=np.array([200.,400.,600.])
    z=np.array([[.2,.6],[.1,.8],[.3,.4]])
    memory=np.exp(-duration[:,None]/p[12:14])
    h0=p[6:8]+(z-p[6:8])*memory
    hb=np.empty_like(h0)
    for row,T in enumerate(duration):
        for j in range(2):
            samples=p[6+j]+(z[row,j]-p[6+j])*np.exp(-(T+baseline)/p[12+j])
            hb[row,j]=samples.mean()
    current=np.empty((3,len(pulse)))
    for row in range(3):
        current[row]=p[8]-p[9]*np.sum(p[:2]*p[10:12]*(hb[row]-p[6:8]))
        for j in range(2):current[row]+=p[j]*h0[row,j]*np.exp(-pulse/p[2+j])
    return p,pulse,baseline,duration,z,h0,hb,current


def test_observation_recovers_states_and_baseline_history():
    p,t,b,T,z,h0,hb,i=synthetic()
    a,m=observe_state(p,t,i,T,b)
    np.testing.assert_allclose(a['holding_start_availability'],z,atol=1e-8)
    np.testing.assert_allclose(a['initial_availability'],h0,atol=1e-8)
    np.testing.assert_allclose(a['baseline_availability'],hb,atol=1e-8)
    np.testing.assert_array_equal(a['estimation_phase_ms'],t[(t>=10)&(t<60)])
    np.testing.assert_allclose(a['estimation_predicted_pa'],a['estimation_current_pa'],atol=1e-8)
    assert not m['global_parameters_changed'] and not m['independent_validation']


def test_later_current_cannot_change_estimate():
    p,t,b,T,z,h0,hb,i=synthetic()
    a,_=observe_state(p,t,i,T,b)
    modified=i.copy();modified[:,t>=60]+=1000
    other,_=observe_state(p,t,modified,T,b)
    np.testing.assert_array_equal(a['holding_start_availability'],other['holding_start_availability'])


def test_conditioned_predictions_reduce_to_shared_state_and_carry_endpoints():
    p=INITIAL.copy();gaps=np.array([0.,50.,1e6]);pulse=np.array([0.,300.]);post=np.array([0.,1e6])
    initial=np.broadcast_to(p[6:8],(3,2)).copy()
    expected=state_currents(p,gaps,pulse,post)
    actual=conditioned_currents(p,gaps,pulse,post,initial,initial)
    for a,e in zip(actual,expected):np.testing.assert_allclose(a,e,atol=1e-12)
    different=initial.copy();different[:,1]=.3
    base=different+.01
    first,second,returned=conditioned_currents(p,gaps,pulse,post,different,base)
    assert second[0,0]==pytest.approx(first[0,-1])
    expected_long=p[9]*np.sum(p[:2]*p[10:12]*(p[6:8]-base),axis=1)
    np.testing.assert_allclose(returned[:,-1],expected_long,atol=1e-12)


@pytest.mark.parametrize('fault',['nan','shape','clock','coverage','baseline','holding','parameters','rank','sparse'])
def test_invalid_observation_inputs(fault):
    p,t,b,T,z,h0,hb,i=synthetic()
    if fault=='nan':i[0,0]=np.nan
    if fault=='shape':i=i[:,:-1]
    if fault=='clock':t[3]+=.1
    if fault=='coverage':t=t[30:];i=i[:,30:]
    if fault=='baseline':b=b[b<-20]
    if fault=='holding':T[0]=50
    if fault=='parameters':p[12]=-1
    if fault=='rank':p[0]=0
    if fault=='sparse':t=np.arange(0.,300.,5.);i=np.ones((3,len(t)))
    with pytest.raises(ValueError):observe_state(p,t,i,T,b)


def test_prediction_rejects_invalid_state_shapes_and_values():
    with pytest.raises(ValueError):conditioned_currents(INITIAL,[5.],[10.],[20.],[[1.1,0.]],[[0.,0.]])
    with pytest.raises(ValueError):conditioned_currents(INITIAL,[5.],[10.],[20.],[[.1,.2],[.2,.3]],[[.1,.2]])
