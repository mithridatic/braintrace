"""Joint human recovery currents with explicit within-sweep state carry."""

import numpy as np
from scipy.optimize import least_squares


INITIAL=np.array([170.,500.,32.,750.,60.,510.,.15,.01,30.,.5,.1,.2,300.,1500.,10.])
LOWER=np.array([0.,0.,1.,1.,1.,1.,0.,0.,-100.,0.,0.,0.,1.,1.,.04])
UPPER=np.array([2500.,2500.,10000.,10000.,20000.,20000.,1.,1.,500.,1.,1.,1.,20000.,20000.,1000.])
ORDER=['A1','A2','d1','d2','r1','r2','f1','f2','C','rho','eta1','eta2','s1','s2','a_off']


def state_currents(parameters,gaps_ms,pulse_phase_ms,post_phase_ms):
    """Evaluate closed-form availability through two pulses and return to holding.

    Parameters
    ----------
    parameters : array-like
        A1,A2,d1,d2,r1,r2,f1,f2,C,rho,eta1,eta2,s1,s2,a_off. Current amplitudes
        are in pA at phase zero, time constants in ms and fractions dimensionless.
    gaps_ms, pulse_phase_ms, post_phase_ms : array-like
        Recovery intervals and the separate original phase coordinates in ms.

    Returns
    -------
    tuple of numpy.ndarray
        First, second and post-pulse baseline-centered currents, each with one
        row per gap. These are conditional protocol currents, not isolated ions.
    """
    p=np.asarray(parameters)
    a,d,r,f=p[:2],p[2:4],p[4:6],p[6:8]
    c,rho,eta,s,off=p[8],p[9],p[10:12],p[12:14],p[14]
    gaps=np.asarray(gaps_ms);pulse=np.asarray(pulse_phase_ms);post=np.asarray(post_phase_ms)
    end_first=f*np.exp(-300/d)
    start_second=1-(1-end_first)*np.exp(-gaps[:,None]/r)
    end_second=start_second*np.exp(-300/d)
    envelope=np.exp(-pulse[None,:]/d[:,None])
    first=np.broadcast_to(c+(a*f)@envelope,(len(gaps),len(pulse)))
    second=c+(start_second*a)@envelope
    h=f[None,:,None]+(end_second-f)[:,:,None]*np.exp(-post[None,None,:]/s[None,:,None])
    factor=rho*(eta[:,None]+(1-eta[:,None])*np.exp(-post[None,:]/off))
    returned=np.einsum('j,ijt,jt->it',a,h,factor)-rho*np.sum(a*eta*f)
    return first,second,returned


def fit_state_currents(gaps_ms,pulse_phase_ms,post_phase_ms,first_pa,second_pa,post_pa):
    """Fit both original pulse responses and the return using one shared candidate.

    Parameters
    ----------
    gaps_ms : array-like
        At least four distinct increasing positive recovery gaps in ms.
    pulse_phase_ms, post_phase_ms : array-like
        Separate original increasing clocks covering 10-280 and 10-980 ms.
    first_pa, second_pa, post_pa : array-like
        Baseline-centered original currents, one row per gap. The same measured
        initial baseline must be subtracted from all three responses per sweep.

    Returns
    -------
    tuple of dict
        Original fit samples, predicted currents, residuals, state endpoints and
        conditional fit metadata. No physiological qualification is implied.
    """
    gaps,pulse,post=(np.asarray(x,dtype=float) for x in (gaps_ms,pulse_phase_ms,post_phase_ms))
    currents=[np.asarray(x,dtype=float) for x in (first_pa,second_pa,post_pa)]
    if (any(x.ndim!=1 for x in (gaps,pulse,post)) or len(gaps)<4
            or len(pulse)<24 or len(post)<24 or np.any(gaps<=0)
            or any(not np.isfinite(x).all() or np.any(np.diff(x)<=0) for x in (gaps,pulse,post))
            or any(not np.isfinite(x).all() for x in currents)
            or currents[0].shape!=(len(gaps),len(pulse))
            or currents[1].shape!=(len(gaps),len(pulse))
            or currents[2].shape!=(len(gaps),len(post))):
        raise ValueError('Invalid aligned recovery currents or original coordinates.')
    if pulse[0]>10 or pulse[-1]<280 or post[0]>10 or post[-1]<980:
        raise ValueError('The required pulse and return windows are incomplete.')
    pm=(pulse>=10)&(pulse<280);hm=(post>=10)&(post<980)
    if pm.sum()<24 or hm.sum()<24:
        raise ValueError('Insufficient original samples within fit windows.')
    pulse,post=pulse[pm],post[hm]
    observed=[currents[0][:,pm],currents[1][:,pm],currents[2][:,hm]]
    def residual(p):
        predicted=state_currents(p,gaps,pulse,post)
        return np.concatenate([(prediction-data).ravel() for prediction,data in zip(predicted,observed)])
    result=least_squares(residual,INITIAL,bounds=(LOWER,UPPER),x_scale='jac',
                         max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    predictions=state_currents(result.x,gaps,pulse,post)
    arrays=dict(gaps_ms=gaps,pulse_phase_ms=pulse,post_phase_ms=post)
    rms={}
    for name,data,prediction in zip(('first','second','post'),observed,predictions):
        error=data-prediction
        arrays.update({name+'_current_pa':data,name+'_predicted_pa':prediction,name+'_residual_pa':error})
        rms[name]=dict(all_rms_pa=float(np.sqrt(np.mean(error**2))),
                       per_gap_rms_pa=np.sqrt(np.mean(error**2,axis=1)).tolist())
    p=result.x
    end_first=p[6:8]*np.exp(-300/p[2:4])
    start_second=1-(1-end_first)*np.exp(-gaps[:,None]/p[4:6])
    arrays.update(first_end_availability=end_first,second_start_availability=start_second,
                  second_end_availability=start_second*np.exp(-300/p[2:4]))
    scales=np.maximum(abs(p),1.)
    singular=np.linalg.svd(result.jac*scales[None,:],compute_uv=False)
    proximity=np.minimum(p-LOWER,UPPER-p)/(UPPER-LOWER)
    metadata=dict(parameter_order=ORDER,parameters=p.tolist(),lower_bounds=LOWER.tolist(),
                  upper_bounds=UPPER.tolist(),optimizer_success=bool(result.success),
                  optimizer_message=str(result.message),evaluations=int(result.nfev),
                  active_bounds=result.active_mask.tolist(),relative_bound_distance=proximity.tolist(),
                  within_one_millionth_of_bound=(proximity<=1e-6).tolist(),
                  residuals=rms,scaled_jacobian_singular_values=singular.tolist(),
                  jacobian_column_scales=scales.tolist(),
                  scaled_jacobian_rank=int(np.sum(singular>singular[0]*1e-8)),
                  jacobian_rank_relative_tolerance=1e-8,
                  weighting='equal weight per original sample',pulse_window_ms=[10.,280.],
                  return_window_ms=[10.,980.],physiological_qualification=False,
                  assumption='shared initial availability equals holding asymptote; two availability populations and one off time constant')
    return arrays,metadata
