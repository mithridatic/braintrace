"""Source-history reference and dimensional conversion checks."""

import importlib.util
from pathlib import Path

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .extracellular import SourceHistory, nernst_mv, outward_current_amount_rate


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def test_nernst_and_faraday_conversions():
    assert nernst_mv(120., 2.5) == pytest.approx(-102.4652, abs=.002)
    assert nernst_mv(10., 140.) > 0
    assert nernst_mv(2., 2.) == 0
    # 96485.33212 nA for one ms transfers 1e6 mM um^3 for a monovalent ion.
    assert outward_current_amount_rate(96485.33212) == pytest.approx(1e6)
    assert outward_current_amount_rate(-1., valence=2.) < 0
    assert np.isnan(nernst_mv(0., 2.))
    assert np.isnan(nernst_mv(1., 2., valence=0.))
    assert np.isnan(outward_current_amount_rate(1., valence=0.))


@pytest.mark.parametrize('uptake', [None, 600.])
def test_pinned_braincell_source_kernel(uptake):
    path = Path(__file__).resolve().parents[2]/'docs/biology/reference/CommonBackend.py'
    spec = importlib.util.spec_from_file_location('braincell_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = np.array([[0., 0., 0.], [1., 2., 3.]])
    targets = np.array([[0., 0., 0.], [3., 1., 0.]])
    diameters = np.array([.1, .2])
    bounds = np.array([0., 1., 2., 5.])
    values = np.array([[.3, .4], [.1, .2], [0., 0.]])
    model = SourceHistory(sources, targets, diameters, uptake_ms=uptake)
    kernel = module.calcConcsInParallelXPU_kernel1.py_func
    for time in (0., 1., 1.5, 4., 10.):
        expected = [sum(kernel(bounds, values.reshape(-1), *sources.T, diameters,
            *targets.T, time, 0., .3, uptake is not None, uptake or 600., source, target)
            for source in range(2)) for target in range(2)]
        np.testing.assert_allclose(jax.jit(model.sample)(time, bounds, values), expected, atol=1e-12)
    derivative = jax.jacrev(lambda v: model.sample(1.5, bounds, v))(jnp.asarray(values))
    assert np.isfinite(derivative).all()


def test_bad_history_and_parameters():
    model = SourceHistory([[0., 0., 0.]], [[1., 0., 0.]], [.1])
    assert np.isnan(model.sample(1., [0., 0.], [[1.]])).all()
    with pytest.raises(ValueError):
        model.sample(1., [0., 1.], [[1., 2.]])
    for kwargs in (dict(diameter_um=[0.]), dict(source_xyz=[[0., 0.]]),
                   dict(diffusion_um2_ms=0.), dict(uptake_ms=0.)):
        config = dict(source_xyz=[[0., 0., 0.]], destination_xyz=[[1., 0., 0.]], diameter_um=[.1])
        config.update(kwargs)
        with pytest.raises(ValueError):
            SourceHistory(**config)
