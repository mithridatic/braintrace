"""Analytic and boundary tests for compiled sodium-command responses."""

import jax.numpy as jnp
import numpy as np
import pytest

from h01_sodium_transfer import evaluate,gate_trajectory,prepare_command


def test_blocked_affine_scan_matches_sequential_across_boundaries():
    import brainstate
    x=jnp.arange(2051,dtype=jnp.float64)
    equilibrium=jnp.stack([.5+.4*jnp.sin(x/17),.5+.4*jnp.cos(x/29)],axis=1)
    tau=jnp.stack([.03+.02*jnp.sin(x/23),2+jnp.cos(x/41)],axis=1)
    factors=jnp.array([2.,.7])
    @brainstate.transform.jit
    def oracle(eq,ts):
        def step(g,inputs):
            target,t=inputs
            updated=target+(g-target)*jnp.exp(-.008*factors/t)
            return updated,updated
        _,history=brainstate.transform.scan(step,eq[0],(eq[:-1],ts[:-1]))
        return jnp.concatenate([eq[:1],history])
    np.testing.assert_allclose(np.asarray(gate_trajectory(equilibrium,tau,factors,.008)),
                               np.asarray(oracle(equilibrium,tau)),atol=2e-14,rtol=0)


def test_gate_step_matches_analytic_solution_and_initial_sample():
    equilibrium=np.array([[.1,.9]]+[[.7,.2]]*100)
    tau=np.tile([2.,5.],(101,1));factors=np.array([1.5,2.5])
    actual=np.asarray(gate_trajectory(jnp.asarray(equilibrium),jnp.asarray(tau),jnp.asarray(factors),.008))
    # The first interval retains the initial command; only later intervals step.
    elapsed=np.maximum(np.arange(101)-1,0)*.008
    expected=np.array([.7,.2])+(np.array([.1,.9])-np.array([.7,.2]))*np.exp(-elapsed[:,None]*factors/tau)
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)


def test_source_hold_substeps_and_constant_current_normalization():
    voltage=np.full(1000,-60.)
    p=prepare_command(voltage,np.array([200,600]),substeps=2)
    np.testing.assert_array_equal(np.asarray(p['voltage']),np.r_[np.repeat(voltage[:-1],2),voltage[-1]])
    assert p['voltage'].size==1999
    ratio,peaks,gates,right,left=evaluate(p,[2,2])
    np.testing.assert_allclose(ratio,[1,1],atol=1e-14)
    np.testing.assert_array_equal(right,left)
    assert peaks.min()>0 and gates.shape==(1999,2)


@pytest.mark.parametrize('voltage,peaks,substeps',[
    ([0,np.nan],[1],1), ([[0,1]],[1],1), ([0],[1],1),
    (np.zeros(1000),[200],3), (np.zeros(1000),[],1),
    (np.zeros(1000),[200.5],1), (np.zeros(1000),[200,200],1),
    (np.zeros(1000),[100],1), (np.zeros(1000),[800],1),
])
def test_incomplete_or_invalid_commands(voltage,peaks,substeps):
    with pytest.raises(ValueError):prepare_command(voltage,peaks,substeps=substeps)


@pytest.mark.parametrize('factors',[[0,1],[1,float('nan')],[1],[-1,1]])
def test_invalid_factors(factors):
    with pytest.raises(ValueError):evaluate({},factors)


def test_nonpositive_inward_response_rejected():
    p=prepare_command(np.full(1000,160.),np.array([200]))
    with pytest.raises(ValueError):evaluate(p,[1,1])
