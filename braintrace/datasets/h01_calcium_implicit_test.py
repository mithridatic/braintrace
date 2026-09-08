"""Independent numerical checks for the experimental calcium solve."""
import brainstate
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from .h01_calcium_implicit import calcium_backward_euler

K = 10000*.0008762096311710155/(2*96485.33212331002*.1)
R = 8.31446261815324*307.15/(2*96485.33212331002)*1000


def solve(c, v, g, dt=.005):
    return calcium_backward_euler(c, v, g, dt, 657.0460049891833, K, R)


def test_observed_negative_update_has_positive_implicit_solution():
    c, current, tau = 1.9281996466055362e-7, -.21562467119683992, 657.0460049891833
    frozen = c*np.exp(-.005/tau)+(1e-4+K*current*tau)*(-np.expm1(-.005/tau))
    assert frozen == pytest.approx(-2.959552815885944e-7, abs=1e-20)
    with brainstate.environ.context(precision=64):
        result = float(solve(c, 378.11889242275646, .0013121678452798395))
    assert result > 0
    assert result == pytest.approx(1.0563536991597979e-10, rel=1e-10)


@pytest.mark.parametrize('voltage', [-100., 0., 150., 400.])
@pytest.mark.parametrize('conductance', [0., .001, 1.])
def test_matches_independent_scalar_root(voltage, conductance):
    c, dt, tau = .0001, .005, 657.0460049891833
    def residual(y):
        value = np.exp(y)
        return value-c-dt*(K*conductance*(R*(np.log(2.)-y)-voltage)+(1e-4-value)/tau)
    expected = np.exp(brentq(residual, -700., 10., xtol=1e-13))
    with brainstate.environ.context(precision=64):
        result = float(solve(c, voltage, conductance, dt))
    assert result == pytest.approx(expected, rel=1e-11, abs=1e-20)


def test_vectorized_jit_signed_flux_and_zero_conductance():
    with brainstate.environ.context(precision=64):
        result = np.asarray(brainstate.transform.jit(solve)(np.array([1e-4]*3),
                            np.array([-70.,400.,0.]), np.array([.01,.01,0.])))
        removal = float(solve(.001, 0., 0.))
    assert result[0] > 1e-4 > result[1] > 0
    assert result[2] == pytest.approx(1e-4)
    assert removal == pytest.approx((.001+.005*1e-4/657.0460049891833)/(1+.005/657.0460049891833))


@pytest.mark.parametrize('args', [(-1.,0.,1.,.005,1.,K,R), (1e-4,0.,-1.,.005,1.,K,R),
    (1e-4,0.,1.,-.005,1.,K,R), (1e-4,np.nan,1.,.005,1.,K,R),
    (1e-4,0.,1.,.005,0.,K,R), (1e-4,0.,1.,.005,1.,-K,R)])
def test_invalid_parameters_fail_closed(args):
    with brainstate.environ.context(precision=64):
        assert np.isnan(float(calcium_backward_euler(*args)))


def test_refines_toward_continuous_signed_flux_ode():
    c0, v, g, tau, duration = .0001, 180., .01, 657.0460049891833, .4
    def rhs(t, y):
        c = np.exp(y[0])
        return [(K*g*(R*(np.log(2.)-y[0])-v)+(1e-4-c)/tau)/c]
    oracle = np.exp(solve_ivp(rhs, (0,duration), [np.log(c0)], method='Radau',
                             rtol=1e-11, atol=1e-12).y[0,-1])
    errors = []
    with brainstate.environ.context(precision=64):
        for count in (20,40,80):
            dt = duration/count
            def step(c, unused):
                return solve(c,v,g,dt), None
            final, _ = brainstate.transform.scan(step,c0,xs=None,length=count)
            errors.append(abs(float(final)-oracle))
    assert errors[1] < .6*errors[0]
    assert errors[2] < .6*errors[1]


def test_broadcasts_step_sizes_and_decay_with_scalar_state():
    with brainstate.environ.context(precision=64):
        result = np.asarray(calcium_backward_euler(.001, 0., 0., np.array([.005,.01]),
                            np.array([100.,200.]), K, R))
    np.testing.assert_allclose(result, (.001+np.array([.005,.01])*1e-4/np.array([100.,200.])) /
                              (1+np.array([.005,.01])/np.array([100.,200.])), rtol=1e-12)
