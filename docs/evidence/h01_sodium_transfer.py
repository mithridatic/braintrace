"""Compiled ideal-command sodium response for human kinetic-scaling tests."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_wilbers import sodium_rates

jax.config.update('jax_enable_x64', True)


def prepare_command(voltage_mv, peak_indices, *, substeps=1):
    """Prepare source-held command intervals and complete AP observation windows.

    Parameters
    ----------
    voltage_mv : array_like
        Original 0.008-ms sodium command, including its voltage offset.
    peak_indices : array_like
        Original source sample indices of each voltage peak.
    substeps : int, optional
        One or two exact gate updates per held source interval.

    Returns
    -------
    dict
        Expanded command, 25 C gate equilibria/time constants and peak windows.
    """
    v=np.asarray(voltage_mv,dtype=float)
    peaks=np.asarray(peak_indices)
    if substeps not in (1,2) or v.ndim!=1 or v.size<2 or not np.isfinite(v).all():
        raise ValueError('Invalid voltage command or substep count.')
    if peaks.ndim!=1 or peaks.size==0 or not np.issubdtype(peaks.dtype,np.integer) or np.any(np.diff(peaks)<=0):
        raise ValueError('Invalid source peak indices.')
    if peaks.min()<125 or peaks.max()+250>=v.size:
        raise ValueError('Incomplete AP observation window.')
    expanded=np.r_[np.repeat(v[:-1],substeps),v[-1]]
    m,h,tm,th=sodium_rates(expanded,temperature_c=25.)
    windows=peaks[:,None]*substeps+np.arange(-125*substeps,250*substeps+1)
    return dict(voltage=jnp.asarray(expanded),equilibrium=jnp.asarray(np.stack([m,h],axis=1)),
                tau_ms=jnp.asarray(np.stack([tm,th],axis=1)),windows=jnp.asarray(windows),
                dt_ms=.008/substeps,substeps=substeps)


@brainstate.transform.jit
def gate_trajectory(equilibrium, tau_ms, multipliers, dt_ms):
    """Integrate held-voltage gates exactly with a compiled explicit-carry scan.

    Parameters
    ----------
    equilibrium, tau_ms : array_like
        Per-sample m/h equilibria and unscaled time constants.
    multipliers : array_like
        Positive activation/inactivation rate factors.
    dt_ms : float
        Time between expanded command samples.

    Returns
    -------
    array
        m/h gates at every sample, including initial equilibrium.
    """
    count=equilibrium.shape[0]-1
    block=1024
    padding=(-count)%block
    a=jnp.exp(-dt_ms*multipliers/tau_ms[:-1])
    b=equilibrium[:-1]*(1-a)
    a=jnp.pad(a,((0,padding),(0,0)),constant_values=1).reshape((-1,block,2))
    b=jnp.pad(b,((0,padding),(0,0))).reshape((-1,block,2))
    def compose(left,right):
        al,bl=left
        ar,br=right
        return ar*al,br+ar*bl
    def step(gates, inputs):
        prefix_a,prefix_b=jax.lax.associative_scan(compose,inputs)
        history=prefix_a*gates+prefix_b
        return history[-1],history
    initial=equilibrium[0]
    _,history=brainstate.transform.scan(step,initial,(a,b))
    return jnp.concatenate([initial[None,:],history.reshape((-1,2))[:count]],axis=0)


@brainstate.transform.jit
def predict(equilibrium, tau_ms, voltage, windows, log_multipliers, dt_ms):
    """Predict AP-normalized inward-current amplitudes and complete gate samples.

    Parameters
    ----------
    equilibrium, tau_ms, voltage, windows : array_like
        Prepared gate values, command voltage and source-indexed AP windows.
    log_multipliers : array_like
        Logarithms of the two positive rate factors.
    dt_ms : float
        Integration substep length in ms.

    Returns
    -------
    tuple
        Normalized amplitudes, peak proxies, gates and right/left current proxies.
    """
    gates=gate_trajectory(equilibrium,tau_ms,jnp.exp(log_multipliers),dt_ms)
    conductance=gates[:,0]**3*gates[:,1]
    right=conductance*(141.-voltage)
    left=conductance*(141.-jnp.r_[voltage[0],voltage[:-1]])
    peaks=jnp.max(jnp.maximum(right,left)[windows],axis=1)
    return peaks/peaks[0],peaks,gates,right,left


def evaluate(prepared, multipliers):
    """Evaluate a prepared command and reject invalid model responses.

    Parameters
    ----------
    prepared : dict
        Output of prepare_command.
    multipliers : array_like
        Two finite positive gate rate multipliers.

    Returns
    -------
    tuple of ndarray
        Normalized amplitudes, peaks, gate trajectory and two current proxies.
    """
    factors=np.asarray(multipliers,dtype=float)
    if factors.shape!=(2,) or not np.isfinite(factors).all() or np.any(factors<=0):
        raise ValueError('Invalid sodium rate multipliers.')
    args=[prepared[k] for k in ('equilibrium','tau_ms','voltage','windows')]
    outputs=tuple(np.asarray(x) for x in predict(*args,np.log(factors),prepared['dt_ms']))
    if any(not np.isfinite(x).all() for x in outputs) or np.any(outputs[1]<=0) or np.any((outputs[2]<0)|(outputs[2]>1)):
        raise ValueError('Nonfinite, nonpositive or invalid sodium response.')
    return outputs
