"""Human channel rate and initialization checks against source equations."""

import math

import brainstate
import brainunit as u
import jax
import numpy as np
import pytest

from .h01_wilbers import rate_trap, sodium_rates, potassium_rates


def test_source_guard_misses_the_removable_singularity():
    # Literal source branch tests v+threshold, but divides by v-threshold.
    v, threshold, scale, slope = -58.59235606617833, -58.59235606617833, .2139849270413559, 9.21455371864071
    assert abs((v + threshold) / slope) > 1e-6
    with pytest.raises(ZeroDivisionError):
        _ = scale * (v - threshold) / (1 - math.exp(-(v - threshold) / slope))
    assert float(rate_trap(v, threshold, scale, slope)) == pytest.approx(scale * slope)


def test_rate_limit_and_gradient():
    with brainstate.environ.context(precision=64):
        f = lambda v: rate_trap(v, -40., .2, 8.)
        assert float(f(-40.)) == pytest.approx(1.6)
        assert float(jax.grad(f)(-40.)) == pytest.approx(.1)
        np.testing.assert_allclose([f(-40 - 1e-8), f(-40 + 1e-8)], 1.6, atol=2e-9)


def test_float32_rate_gradients_do_not_overflow():
    with brainstate.environ.context(precision=32):
        gradient = jax.jit(jax.vmap(jax.grad(lambda v: sum(potassium_rates(v)))))
        assert np.isfinite(gradient(np.linspace(-120., 100., 100, dtype=np.float32))).all()


def test_rates_against_independent_scalar_equations():
    # Compare ordinary voltages to unmodified source algebra; limit tested above.
    def trap(v, th, a, q):
        return a * (v-th) / (1-math.exp(-(v-th)/q))
    for v in (-100., -70., -40., 0., 40.):
        rates = sodium_rates(v, temperature_c=34)
        phi = 2.3**.9
        mtau = 1/phi/(trap(v, -58.59235606617833, .2139849270413559, 9.21455371864071)
                      + trap(-v, 58.59235606617833, .2139849270413559, 9.21455371864071))
        htau = 1/phi/(trap(v, -39.298676911255, .0476391878313753, 7.21636386592805)
                      + trap(-v, 77.16674532461128, .011951655023359499, 4.749450048393884))
        np.testing.assert_allclose(rates, [1/(1+math.exp((-42.29145668954657-v)/10.231152896063074)),
                                         1/(1+math.exp((v+64.48)/11.02)), mtau, htau], rtol=2e-6)


def test_temperature_bounds_and_potassium_recovery():
    with brainstate.environ.context(precision=64):
        v = np.linspace(-120, 100, 441)
        for fn in (sodium_rates, potassium_rates):
            a = fn(v, temperature_c=24)
            b = fn(v, temperature_c=34)
            assert all(np.isfinite(x).all() for x in a)
            assert all(((x >= 0) & (x <= 1)).all() for x in a[:2])
            assert all((x > 0).all() for x in a[2:])
            np.testing.assert_allclose(a[:2], b[:2])
            np.testing.assert_allclose(np.asarray(a[2:]) / np.asarray(b[2:]), 2.3)


def test_targeted_wilbers_rates_match_full_tuple():
    with brainstate.environ.context(precision=64):
        v = np.linspace(-100, 40, 50)
        for fn in (sodium_rates, potassium_rates):
            full = fn(v)
            m_res = fn(v, gate="m")
            h_res = fn(v, gate="h")
            np.testing.assert_allclose(m_res[0], full[0])
            np.testing.assert_allclose(m_res[1], full[2])
            np.testing.assert_allclose(h_res[0], full[1])
            np.testing.assert_allclose(h_res[1], full[3])
