"""Actual cable current responds to spatial GABA without training receptors."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.mech import Channel
from braincell.filter import AllRegion

from .cable_gaba import EnvironmentGABA
from .cable_potassium import CablePotassiumBinding, PotassiumCoupling
from .cable_potassium_test import coupled
from braintrace.datasets.h01_network_step import H01NetworkStep


def test_spatial_gaba_receptor_current_and_voltage(tmp_path):
    with brainstate.environ.context(precision=64):
        old, _, env = coupled(tmp_path)
        cell = old.cells[0]
        # A new declaration rebuilds runtime. Bind only after rebuilding it.
        cell.reset()
        cell.paint(AllRegion(), Channel('EnvironmentGABA', name='tonic'))
        step = H01NetworkStep(old.network)
        binding = CablePotassiumBinding(cell, env)
        channel = next(node for node in cell.runtime.runtime_nodes.values() if isinstance(node, EnvironmentGABA))
        channel.bind(binding)
        step.bind_chemistry(PotassiumCoupling(env, [binding]))
        step.reset_state()
        voltage = cell._cv_to_point_unchecked(cell.V.value)
        current = brainstate.transform.jit(channel.current)
        assert np.all(current(voltage).to_decimal(u.mA/u.cm**2) == 0.)
        env.gaba.value = jnp.full_like(env.gaba.value, 1e-4)
        density = current(voltage).to_decimal(u.mA/u.cm**2)[..., binding.midpoints]
        np.testing.assert_allclose(density, -.00025, atol=1e-14)
        before = np.asarray(cell.V.value.to_decimal(u.mV))
        brainstate.transform.for_loop(lambda _: step.update(), jnp.arange(20))
        assert np.all(cell.V.value.to_decimal(u.mV) < before)
        assert env.valid.value
        assert not brainstate.graph.states(channel, brainstate.ParamState)
        with pytest.raises(ValueError, match='already bound'):
            channel.bind(binding)
        wrong = EnvironmentGABA(99)
        with pytest.raises(ValueError, match='shapes differ'):
            wrong.bind(binding)
