"""Actual neuronal release, glial cable feedback and whole-coupling replay."""

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.filter import AllRegion
from braincell.mech import Channel, Ion
from braincell.network.delivery import population_spike

from braintrace.datasets.h01_network_step import H01NetworkStep
from braintrace.datasets.h01_network_step_test import _network
from .cable_kir import BraincellKir41
from .cable_gaba import EnvironmentGABA
from .cable_potassium import CablePotassiumBinding
from .transport import DiffusionGraph
from .environment import ChemicalEnvironment, MembraneMap
from .gaba import GabaReleaseSites
from .transmitter import TransmitterField, GlutamateReleaseSites
from .astrocyte_network import AstrocyteCalcium
from .extracellular import MOLECULES_PER_MM_UM3
from .neuroglial import NeuroGlialCoupling


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        yield


def coupled_system(tmp_path, *, gaba_enabled=True, glutamate_enabled=True):
    net = _network(tmp_path)
    net.add_population('excitatory', _network(tmp_path).populations['cell'].cell)
    for population in net.populations.values():
        population.cell.paint(AllRegion(), Ion('EnvironmentPotassium', name='potassium'))
        population.cell.paint(AllRegion(), Channel('K_Leak', g_max=0.*u.mS/u.cm**2))
        population.cell.paint(AllRegion(), Channel('EnvironmentGABA', name='tonic'))
    step = H01NetworkStep(net)
    branch = braincell.Branch(lengths=np.array([1.])*u.um, radii_proximal=np.ones(1)*u.um,
        radii_distal=np.ones(1)*u.um, points_proximal=np.array([[0., 0., 0.]])*u.um,
        points_distal=np.array([[1., 0., 0.]])*u.um, type='dendrite')
    glia = braincell.Cell(braincell.Morphology(root_name='glia', root_branch=branch),
        V_init=-100.*u.mV, pop_size=(1,), cv_policy=braincell.MaxCVLen(10*u.um))
    glia.paint(AllRegion(), Ion('EnvironmentPotassium', name='potassium'))
    glia.paint(AllRegion(), Channel('BraincellKir41'))
    glia.init_state()
    assert glia.n_cv == 1
    counts = [cell.n_cv for cell in step.cells]
    indices = np.r_[np.zeros(counts[0], int), np.ones(counts[1], int), 1]
    volumes = np.r_[np.full(sum(counts), 100.), np.pi]
    k = DiffusionGraph([100., 100.], [[0, 1]], [10.], dt_ms=.005)
    g = DiffusionGraph([100., 100.], [[0, 1]], [10.], dt_ms=.005, uptake_per_ms=[1/600]*2)
    gt = DiffusionGraph([100., 100.], [[0, 1]], [10.], dt_ms=.005, uptake_per_ms=[.01]*2)
    env = ChemicalEnvironment(k, g, MembraneMap(indices, volumes, 2), np.full(len(indices), 140.), outside_potassium_mm=10.)
    bindings, offset = [], 0
    for cell in step.cells:
        binding = CablePotassiumBinding(cell, env, offset=offset)
        bindings.append(binding)
        channel = next(n for n in cell.runtime.runtime_nodes.values() if isinstance(n, EnvironmentGABA))
        channel.bind(binding)
        offset += cell.n_cv
    astro_binding = CablePotassiumBinding(glia, env, offset=offset)
    calcium = AstrocyteCalcium([1.], [2.], np.empty((0, 2), int))
    kwargs = dict(environment=env, neuron_bindings=bindings, astrocyte_bindings=[astro_binding],
        neuron_roles=('I', 'E'), glutamate=TransmitterField(gt), calcium=calcium, calcium_volume_indices=[1],
        gaba_release=GabaReleaseSites([[0, 1]], 2, active_sites=2) if gaba_enabled else None,
        glutamate_release=GlutamateReleaseSites([[1]], 2, active_sites=1,
            molecules_per_site=10*MOLECULES_PER_MM_UM3, seed=22) if glutamate_enabled else None,
        gaba_sources=[0] if gaba_enabled else [], glutamate_sources=[1] if glutamate_enabled else [],
        basis='Synthetic unit geometry and roles; Kir-only glial cable; diagnostic .1 mM glutamate impulse, not a measured vesicle dose')
    coupling = NeuroGlialCoupling(**kwargs)
    step.bind_chemistry(coupling)
    step.reset_state()
    return step, coupling, kwargs


def trajectory(step, count=10):
    def advance(_):
        step.update(sample_probes=False)
        # Native spike buffers may include several cable output sites.
        return (jnp.stack([population_spike(cell.spike.value).reshape(()) for cell in step.cells]),
                step.cells[0].V.value.to_decimal(u.mV), step.chemistry.calcium.state.value)
    return brainstate.transform.for_loop(advance, jnp.arange(count))


def test_evoked_release_k_buffering_and_receptor_feedback(tmp_path):
    step, coupling, kwargs = coupled_system(tmp_path)
    env = coupling.environment
    initial_k = jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    current = brainstate.transform.jit(coupling.currents)()
    assert current[-1] < 0
    initial_v = np.array(coupling.astrocytes[0].V.value.to_decimal(u.mV))
    spikes, voltage, ca = trajectory(step)
    assert np.any(spikes[:, 0]) and np.any(spikes[:, 1])
    assert coupling.valid.value and coupling.tick.value == env.tick.value == step.tick.value == 10
    assert env.inside_potassium.value[-1] > 140 and jnp.sum(env.potassium.value*env.k_transport.volumes) < 2000
    assert not np.array_equal(coupling.astrocytes[0].V.value.to_decimal(u.mV), initial_v)
    final_k = jnp.sum(env.potassium.value*env.k_transport.volumes)+jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)
    np.testing.assert_allclose(final_k, initial_k, rtol=0, atol=1e-8)
    np.testing.assert_allclose(env.gaba_released_amount.value, jnp.sum(spikes[:, 0])*6000/MOLECULES_PER_MM_UM3)
    np.testing.assert_allclose(coupling.glutamate.released_amount.value, jnp.sum(spikes[:, 1])*10.)
    assert coupling.calcium.state.value[0, 24] > coupling.calcium.initial[0, 24]
    control, _, _ = coupled_system(tmp_path, gaba_enabled=False)
    _, without_gaba, _ = trajectory(control)
    assert np.any(np.asarray(voltage[-1]) < np.asarray(without_gaba[-1]))
    no_glutamate, _, _ = coupled_system(tmp_path, glutamate_enabled=False)
    _, _, without_glutamate = trajectory(no_glutamate)
    assert np.max(np.asarray(ca[-1, :, :4]-without_glutamate[-1, :, :4])) > 1e-15
    step.reset_state()
    repeated = trajectory(step)
    for wanted, actual in zip((spikes, voltage, ca), repeated):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize('change', [
    dict(neuron_roles=('I',)), dict(basis=''), dict(calcium_volume_indices=[0]),
    dict(gaba_sources=[1]), dict(gaba_sources=[0, 0]), dict(glutamate_sources=[-1]),
    dict(gaba_release=None),
])
def test_invalid_coupling_maps_are_rejected(tmp_path, change):
    _, _, kwargs = coupled_system(tmp_path)
    with pytest.raises(ValueError):
        NeuroGlialCoupling(**dict(kwargs, **change))


def test_clock_geometry_and_complete_pool_ownership_guards(tmp_path):
    _, coupling, kwargs = coupled_system(tmp_path)
    with pytest.raises(ValueError, match='Distinct'):
        NeuroGlialCoupling(**dict(kwargs, astrocyte_bindings=kwargs['neuron_bindings'][:1]))
    graph = DiffusionGraph([100., 100.], [[0, 1]], [1.], dt_ms=.01)
    with pytest.raises(ValueError, match='clocks'):
        NeuroGlialCoupling(**dict(kwargs, glutamate=TransmitterField(graph)))
    wrong_ca = AstrocyteCalcium([2.], [2.], np.empty((0, 2), int))
    with pytest.raises(ValueError, match='intracellular volumes'):
        NeuroGlialCoupling(**dict(kwargs, calcium=wrong_ca))
    other = GabaReleaseSites([[0]], 3, active_sites=1)
    with pytest.raises(ValueError, match='rows and volumes'):
        NeuroGlialCoupling(**dict(kwargs, gaba_release=other))
    without_release = NeuroGlialCoupling(**dict(kwargs, gaba_release=None, gaba_sources=[],
                                               glutamate_release=None, glutamate_sources=[]))
    without_release.reset_state()
    with brainstate.environ.context(dt=.005*u.ms, t=0*u.ms):
        brainstate.transform.jit(lambda: without_release.update(without_release.currents()))()
    assert without_release.glutamate.released_amount.value == 0


def test_integer_voltage_cannot_change_dtype_after_first_physical_step(tmp_path):
    _, coupling, kwargs = coupled_system(tmp_path)
    glia = coupling.astrocytes[0]
    glia.V.value = jnp.asarray(glia.V.value.to_decimal(u.mV), dtype=jnp.int64)*u.mV
    with pytest.raises(ValueError, match='floating'):
        NeuroGlialCoupling(**kwargs)


def test_continuous_potassium_to_glial_voltage_gradient(tmp_path):
    step, coupling, _ = coupled_system(tmp_path)
    def response(outside_k):
        step.reset_state()
        coupling.environment.potassium.value = jnp.full_like(coupling.environment.potassium.value, outside_k)
        trajectory(step, count=2)
        return coupling.astrocytes[0].V.value.to_decimal(u.mV).reshape(())
    evaluate = brainstate.transform.jit(response)
    derivative = brainstate.transform.jit(brainstate.transform.grad(response))(10.)
    finite_difference = (evaluate(10.001)-evaluate(9.999))/.002
    assert np.isfinite(derivative) and abs(derivative) > 1e-8
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-4, atol=1e-8)


def test_fresh_coupled_checkpoint_restores_glia_calcium_and_both_rng_streams(tmp_path):
    from examples.pp_prop.h01_physical_checkpoint import save_physical_checkpoint, restore_physical_checkpoint, _pack
    step, coupling, _ = coupled_system(tmp_path)
    trajectory(step, count=2)
    optimizer = {'counter': jnp.asarray(3, dtype=jnp.int32)}
    manifest = {'fixture': 'two-neurons-one-kir-cable-calcium', 'basis': coupling.basis, 'dt_ms': .005}
    path = tmp_path/'neuroglial.npz'
    digest = save_physical_checkpoint(path, roots={'step': step}, optimizer=optimizer, manifest=manifest)
    trajectory(step, count=3)
    expected, records, _ = _pack({'step': step}, optimizer)
    assert any('astrocytes' in key for key in records)
    assert any('calcium' in key for key in records)
    assert any('glutamate_release' in key for key in records)
    fresh, restored, _ = coupled_system(tmp_path)
    fresh_arrays, fresh_records, _ = _pack({'step': fresh}, optimizer)
    mismatches = [(key, expected[name].shape, fresh_arrays[name].shape,
                   str(expected[name].dtype), str(fresh_arrays[name].dtype))
                  for key, record in fresh_records.items() for name in record['leaves']
                  if expected[name].shape != fresh_arrays[name].shape or expected[name].dtype != fresh_arrays[name].dtype]
    assert not mismatches, mismatches
    actual_optimizer = restore_physical_checkpoint(path, roots={'step': fresh}, optimizer=optimizer,
                                                   manifest=manifest, expected_sha256=digest)
    assert restored.tick.value == fresh.tick.value == 2
    trajectory(fresh, count=3)
    actual, _, _ = _pack({'step': fresh}, actual_optimizer)
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
