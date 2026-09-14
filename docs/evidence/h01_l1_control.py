"""Measured small-pulse controls and a bounded uniform-gain hypothesis."""

import numpy as np


def extract_test_pulse(source):
    """Retain the complete early command control and original sample boundaries.

    Parameters
    ----------
    source : dict
        time_ms, total_current_pa and command_voltage_mv from a bound recording.

    Returns
    -------
    tuple of dict
        Original and centered samples, masks and descriptive control metadata.
        The late response slope is not qualified membrane conductance.
    """
    t,i,v=(np.asarray(source[k],dtype=float) for k in
           ('time_ms','total_current_pa','command_voltage_mv'))
    if (t.ndim!=1 or len(t)<3 or i.shape!=t.shape or v.shape!=t.shape
            or not all(np.isfinite(x).all() for x in (t,i,v))
            or np.any(np.diff(t)<=0)
            or not np.allclose(np.diff(t),t[1]-t[0],rtol=1e-8,atol=1e-9)):
        raise ValueError('Finite aligned samples on a uniform clock required.')
    edges=[]
    for point in (35.,44.,45.,52.,55.,55.04,65.):
        found=np.flatnonzero(np.isclose(t,point,rtol=0,atol=1e-8))
        if len(found)!=1:raise ValueError('Complete on-grid control windows required.')
        edges.append(int(found[0]))
    a,b,start,late,late_end,stop,end=edges
    holding=v[a];step=v[start]-holding
    if (not np.isclose(step,10.,rtol=0,atol=1e-5)
            or not np.allclose(v[a:start],holding,rtol=0,atol=1e-7)
            or not np.allclose(v[start:stop],holding+step,rtol=0,atol=1e-7)
            or not np.allclose(v[stop:end],holding,rtol=0,atol=1e-7)):
        raise ValueError('Expected a constant +10 mV pulse and return to holding.')
    baseline=float(i[a:b].mean())
    indices=np.arange(a,end)
    arrays=dict(time_ms=t[a:end],phase_ms=t[a:end]-45.,total_current_pa=i[a:end],
                centered_current_pa=i[a:end]-baseline,command_voltage_mv=v[a:end],
                pulse_mask=(indices>=start)&(indices<stop),
                baseline_mask=(indices>=a)&(indices<b),
                late_mask=(indices>=late)&(indices<late_end))
    meta=dict(baseline_mean_pa=baseline,holding_mv=float(holding),step_mv=float(step),
              baseline_ms=[35.,44.],pulse_ms=[45.,55.04],late_ms=[52.,55.],
              apparent_late_slope_ns=float((i[late:late_end].mean()-baseline)/step),
              membrane_conductance_qualified=False)
    return arrays,meta


def _paired(reference,target):
    values=[]
    for data in (reference,target):
        phase,current,command=(np.asarray(data[k],dtype=float) for k in
                               ('phase_ms','centered_current_pa','command_voltage_mv'))
        if (phase.ndim!=1 or len(phase)<3 or current.shape!=phase.shape or command.shape!=phase.shape
                or not all(np.isfinite(x).all() for x in (phase,current,command))
                or np.any(np.diff(phase)<=0)
                or not np.allclose(np.diff(phase),phase[1]-phase[0],rtol=1e-8,atol=1e-9)):
            raise ValueError('Invalid paired control or main response.')
        values.append((phase,current,command))
    p,i,v=values[0];q,j,w=values[1]
    if (p.shape!=q.shape or not np.allclose(p,q,rtol=0,atol=1e-8)
            or not np.allclose(v,w,rtol=0,atol=1e-7)):
        raise ValueError('Paired phase clocks or command trajectories differ.')
    return p,i,j


def gain_transfer(reference_control,target_control,reference_main,target_main):
    """Fit a gain using the small pulse and predict the independent main response.

    Parameters
    ----------
    reference_control, target_control : dict
        Outputs of extract_test_pulse, including their exact pulse masks.
    reference_main, target_main : dict
        Matched phase_ms, command_voltage_mv and centered_current_pa arrays
        covering the first 100 ms after the actual large command onset.

    Returns
    -------
    tuple of dict
        Source responses, predictions, residuals and the control-fitted gain.
        Failure concerns this uniform multiplicative hypothesis only.
    """
    cp,ci,cj=_paired(reference_control,target_control)
    mp,mi,mj=_paired(reference_main,target_main)
    mask=np.asarray(reference_control['pulse_mask'])
    other=np.asarray(target_control['pulse_mask'])
    if (mask.dtype!=bool or mask.shape!=cp.shape or not np.array_equal(mask,other)
            or mask.sum()<3 or not np.isclose(mp[0],0.,atol=1e-8)
            or not np.isclose(mp[-1]+mp[1]-mp[0],100.,rtol=0,atol=1e-8)):
        raise ValueError('Exact control mask and complete 100 ms response required.')
    energy=float(ci[mask]@ci[mask])
    if energy<=0:raise ValueError('Control has no energy to identify gain.')
    gain=float(ci[mask]@cj[mask]/energy)
    if gain<=0:raise ValueError('The inferred gain is not positive.')
    arrays=dict(control_phase_ms=cp,control_reference_pa=ci,control_target_pa=cj,
                control_predicted_pa=gain*ci,control_residual_pa=cj-gain*ci,
                control_fit_mask=mask,main_phase_ms=mp,main_reference_pa=mi,
                main_target_pa=mj,main_predicted_pa=gain*mi,main_residual_pa=mj-gain*mi)
    meta=dict(gain=gain,gain_fit='least squares through zero on the small pulse only',
              control_fit_rms_pa=float(np.sqrt(np.mean((cj[mask]-gain*ci[mask])**2))),
              main_post5_residual_rms_pa=float(np.sqrt(np.mean((mj[mp>=5]-gain*mi[mp>=5])**2))),
              main_response_used_to_fit_gain=False,physiological_qualification=False)
    return arrays,meta
