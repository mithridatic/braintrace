"""Independent state-transition and joint-current fit checks."""

import numpy as np
import pytest

from h01_l1_recovery_state import state_currents,fit_state_currents


PARAMETERS=np.array([170.,500.,32.,750.,60.,510.,.15,.01,30.,.5,.1,.2,300.,1500.,10.])


def oracle(parameters,gaps,pulse,post):
    a=parameters[:2];d=parameters[2:4];r=parameters[4:6];f=parameters[6:8]
    c,rho=parameters[8:10];eta=parameters[10:12];s=parameters[12:14];off=parameters[14]
    first=np.full((len(gaps),len(pulse)),c)
    second=first.copy();returned=np.zeros((len(gaps),len(post)))
    for j in range(2):
        h1end=f[j]*np.exp(-300/d[j])
        for row,gap in enumerate(gaps):
            h2=1+(h1end-1)*np.exp(-gap/r[j])
            end=h2*np.exp(-300/d[j])
            first[row]+=a[j]*f[j]*np.exp(-pulse/d[j])
            second[row]+=a[j]*h2*np.exp(-pulse/d[j])
            h=f[j]+(end-f[j])*np.exp(-post/s[j])
            factor=rho*(eta[j]+(1-eta[j])*np.exp(-post/off))
            returned[row]+=a[j]*(factor*h-rho*eta[j]*f[j])
    return first,second,returned


def test_populationwise_oracle_and_state_endpoints():
    gaps=np.array([0.,30.,4000.,1e6]);pulse=np.array([0.,10.,100.,300.]);post=np.array([0.,20.,900.,1e6])
    actual=state_currents(PARAMETERS,gaps,pulse,post)
    expected=oracle(PARAMETERS,gaps,pulse,post)
    for a,e in zip(actual,expected):np.testing.assert_allclose(a,e,atol=1e-12)
    first,second,returned=actual
    assert second[0,0]==pytest.approx(first[0,-1])
    assert second[-1,0]==pytest.approx(PARAMETERS[:2].sum()+PARAMETERS[8])
    np.testing.assert_allclose(returned[:,-1],0,atol=1e-12)
    expected_at_return=PARAMETERS[9]*((second[:,-1]-PARAMETERS[8])-np.sum(PARAMETERS[:2]*PARAMETERS[10:12]*PARAMETERS[6:8]))
    np.testing.assert_allclose(returned[:,0],expected_at_return,atol=1e-12)


def fit_data():
    gaps=np.array([5.,20.,50.,100.,300.,1000.,4000.])
    pulse=np.arange(0.,300.1,5.)
    post=np.arange(0.,1000.1,10.)
    p=PARAMETERS.copy();p[[0,1,2,3,4,5,8,9,14]]=[190.,470.,40.,680.,70.,580.,25.,.45,15.]
    return gaps,pulse,post,oracle(p,gaps,pulse,post)


def test_joint_fit_preserves_windows_and_predicts_synthetic_source():
    gaps,pulse,post,currents=fit_data()
    arrays,meta=fit_state_currents(gaps,pulse,post,*currents)
    np.testing.assert_array_equal(arrays['pulse_phase_ms'],pulse[(pulse>=10)&(pulse<280)])
    np.testing.assert_array_equal(arrays['post_phase_ms'],post[(post>=10)&(post<980)])
    for name in ('first','second','post'):
        np.testing.assert_allclose(arrays[name+'_predicted_pa'],arrays[name+'_current_pa'],atol=2e-4)
    assert meta['optimizer_success']
    assert not meta['physiological_qualification']
    assert len(meta['scaled_jacobian_singular_values'])==15


@pytest.mark.parametrize('fault',['gaps','pulse','post','shape','nan','short_pulse','short_post','sparse','few_gaps'])
def test_invalid_joint_data(fault):
    gaps,pulse,post,currents=fit_data();first,second,returned=currents
    if fault=='gaps':gaps[2]=gaps[1]
    if fault=='pulse':pulse[3]=pulse[2]
    if fault=='post':post[3]=post[2]
    if fault=='shape':first=first[:,:-1]
    if fault=='nan':returned[0,0]=np.nan
    if fault=='short_pulse':pulse=pulse[:-5];first=first[:,:-5];second=second[:,:-5]
    if fault=='short_post':post=post[:-5];returned=returned[:,:-5]
    if fault=='sparse':
        post=np.r_[np.arange(21.)/10,10.,980.,990.]
        returned=np.zeros((len(gaps),len(post)))
    if fault=='few_gaps':gaps=gaps[:3];first=first[:3];second=second[:3];returned=returned[:3]
    with pytest.raises(ValueError):fit_state_currents(gaps,pulse,post,first,second,returned)
