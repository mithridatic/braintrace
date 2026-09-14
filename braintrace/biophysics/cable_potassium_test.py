"""Actual cable K current closes a shared chemical feedback path."""

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.filter import AllRegion
from braincell.mech import Ion, Channel

from braintrace.datasets.h01_network_step import H01NetworkStep
from braintrace.datasets.h01_network_step_test import _network
from .cable_potassium import EnvironmentPotassium, CablePotassiumBinding, PotassiumCoupling
from .environment import ChemicalEnvironment, MembraneMap
from .transport import DiffusionGraph


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def coupled(tmp_path):
    net = _network(tmp_path, current_na=0.)
    cell = net.populations['cell'].cell
    cell.paint(AllRegion(), Ion('EnvironmentPotassium', name='potassium'))
    cell.paint(AllRegion(), Channel('K_TM1991', g_max=10*u.mS/u.cm**2))
    step = H01NetworkStep(net)
    count = len(cell.cvs)
    edges = np.empty((0, 2), int)
    graph = DiffusionGraph(np.ones(count)*100., edges, [], dt_ms=.005)
    # Synthetic finite volumes for this coupling test, not claimed H01 geometry.
    env = ChemicalEnvironment(graph, graph, MembraneMap(np.arange(count), np.ones(count)*100., count), np.ones(count)*140)
    binding = CablePotassiumBinding(cell, env)
    chemistry = PotassiumCoupling(env, [binding])
    step.bind_chemistry(chemistry)
    step.reset_state()
    return step, binding, env


def test_real_cable_feedback_exchange_and_reset(tmp_path):
    step, binding, env = coupled(tmp_path)
    voltage = np.asarray(binding.cell.V.value.to_decimal(u.mV))
    current = brainstate.transform.jit(binding.outward_current_na)()
    assert np.all(np.asarray(current) > 0)
    before = jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    brainstate.transform.for_loop(lambda _: step.update(), jnp.arange(10))
    assert env.tick.value == step.tick.value == 10 and env.valid.value
    assert np.all(np.asarray(env.potassium.value) > 2.5)
    assert np.any(np.asarray(binding.cell.V.value.to_decimal(u.mV)) < voltage)
    after = jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    np.testing.assert_allclose(after, before, atol=1e-10)
    first = np.asarray(binding.cell.V.value.to_decimal(u.mV))
    step.reset_state()
    assert env.tick.value == 0 and step.tick.value == 0
    brainstate.transform.for_loop(lambda _: step.update(), jnp.arange(10))
    np.testing.assert_array_equal(binding.cell.V.value.to_decimal(u.mV), first)


def test_dynamic_reversal_changes_compiled_ionic_current(tmp_path):
    _, binding, env = coupled(tmp_path)
    current = brainstate.transform.jit(binding.outward_current_na)
    low = current()
    env.potassium.value = jnp.full_like(env.potassium.value, 10.)
    high = current()
    assert np.all(np.asarray(high) < np.asarray(low))
    info = binding.ion.pack_info()
    assert np.isfinite(np.asarray(info.E.to_decimal(u.mV))).all()
    assert np.allclose(info.Co.to_decimal(u.mM), 10.)
    assert np.allclose(info.Ci.to_decimal(u.mM), 140.)


def test_binding_validation(tmp_path):
    step, binding, env = coupled(tmp_path)
    with pytest.raises(ValueError, match='already bound'):
        CablePotassiumBinding(binding.cell, env)
    with pytest.raises(ValueError, match='pool range'):
        CablePotassiumBinding(binding.cell, env, offset=-1)
    with pytest.raises(ValueError, match='exactly once'):
        PotassiumCoupling(env, [binding, binding])
    with pytest.raises(ValueError, match='before stepping'):
        step.bind_chemistry(step.chemistry)
    bare = _network(tmp_path).populations['cell'].cell
    with pytest.raises(ValueError, match='initialized'):
        CablePotassiumBinding(bare, env)
    bare.init_state()
    with pytest.raises(ValueError, match='Paint exactly'):
        CablePotassiumBinding(bare, env)


def test_unbound_ion_preserves_declared_defaults():
    ion = EnvironmentPotassium(1)
    np.testing.assert_allclose(ion.E.to_decimal(u.mV), -95.)
    np.testing.assert_allclose(ion.Co.to_decimal(u.mM), 2.5)
