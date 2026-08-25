"""Focused BrainCell compatibility tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import braincell
import brainunit as u
import jax.numpy as jnp
import pytest


_PATH = Path(__file__).with_name("21-braincell-arc.py")
_SPEC = importlib.util.spec_from_file_location("braincell_arc", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixture)


def test_braincell_pin_and_imports():
    assert braincell.__version__ == "0.1.0"
    assert fixture.brainstate.__name__ == "brainstate"
    assert fixture.braintrace.__name__ == "braintrace"


def test_hodgkin_huxley_constructor_values_and_reset_state():
    cell = fixture.CompatibilityHodgkinHuxley()
    cell.init_state()
    cell.reset_state()
    assert cell.length.to_decimal(u.um) == pytest.approx(10.0)
    assert cell.radius.to_decimal(u.um) == pytest.approx(5.0)
    assert cell.C.to_decimal(u.uF / u.cm**2) == pytest.approx(1.0)
    assert cell.V_th.to_decimal(u.mV) == pytest.approx(0.0)
    assert jnp.allclose(cell.V.value.to_decimal(u.mV), -65.0)
    assert jnp.all(cell.spike.value == 0)
    assert jnp.allclose(cell.na.INa.p.value, 0.05293248, atol=1e-6)
    assert jnp.allclose(cell.na.INa.q.value, 0.5961208, atol=1e-6)
    assert jnp.allclose(cell.k.IK.p.value, 0.31767693, atol=1e-6)
    assert cell.na.E.to_decimal(u.mV) == pytest.approx(50.0)
    assert cell.k.E.to_decimal(u.mV) == pytest.approx(-77.0)
    assert cell.na.INa.g_max.to_decimal(u.mS / u.cm**2) == pytest.approx(120.0)
    assert cell.na.INa.temp.to_decimal(u.kelvin) == pytest.approx(309.15)
    assert cell.na.INa.temp_ref.to_decimal(u.kelvin) == pytest.approx(309.15)
    assert cell.na.INa.q10 == pytest.approx(3.0)
    assert cell.na.INa.V_sh.to_decimal(u.mV) == pytest.approx(-45.0)
    assert cell.k.IK.g_max.to_decimal(u.mS / u.cm**2) == pytest.approx(10.0)
    assert cell.k.IK.temp.to_decimal(u.kelvin) == pytest.approx(309.15)
    assert cell.k.IK.temp_ref.to_decimal(u.kelvin) == pytest.approx(309.15)
    assert cell.k.IK.q10 == pytest.approx(3.0)
    assert cell.k.IK.V_sh.to_decimal(u.mV) == pytest.approx(-45.0)
    assert cell.IL.g_max.to_decimal(u.mS / u.cm**2) == pytest.approx(0.03)
    assert cell.IL.E.to_decimal(u.mV) == pytest.approx(-54.387)


def test_csr_and_current_density_contract():
    csr = fixture.input_csr()
    assert csr.shape == (1, 4)
    assert jnp.array_equal(csr.indices, jnp.arange(4))
    assert jnp.array_equal(csr.indptr, jnp.asarray([0, 4]))
    assert jnp.allclose(csr.data, jnp.asarray([0.1, 0.0, 0.0, 0.0]))
    current = fixture.bounded_current_density(1.0)
    assert current.unit == u.mA / u.cm**2
    cell = fixture.CompatibilityHodgkinHuxley()
    cell.init_state()
    with pytest.raises(Exception, match="units do not match"):
        fixture.advance_one_step(cell, 1.0 * u.mA)


def test_finite_difference_fixture_has_declared_tolerance():
    result = fixture.finite_difference_fixture()
    assert result["absolute_error"] <= result["tolerance"]
    assert jnp.isfinite(result["pp_prop"])
    assert result["pp_prop"] != 0.0
    assert result["relations"] == 2.0
    assert result["finite_voltage"] == 1.0
    assert result["finite_gates"] == 1.0
    assert result["zero_spikes"] == 1.0
    assert result["reset_isolated"] == 1.0


def test_pp_prop_compiler_sees_input_and_recurrent_relations():
    relations = fixture.pp_prop_relation_fixture()
    assert len(relations) == 2
    assert all(relation.connected_hidden_paths for relation in relations)
    assert all(relation.trainable_vars for relation in relations)


def test_spike_path_fixture_is_finite_and_crosses_threshold():
    assert fixture.SPIKE_DRIVE == 20.0
    result = fixture.spike_path_fixture()
    assert result == {
        "threshold_crossed": True,
        "finite_voltage": True,
        "finite_spikes": True,
        "finite_gradient": True,
        "nonzero_gradient": True,
    }


def test_direct_readout_gradients_are_finite_and_nonzero():
    assert fixture.direct_readout_gradient_fixture() == {
        "shape": True,
        "finite": True,
        "height_nonzero": True,
        "width_nonzero": True,
        "color_nonzero": True,
    }


def test_production_topology_has_exact_source_rows_and_no_invalid_edges():
    inputs = fixture.input_topology()
    recurrent = fixture.recurrent_topology()
    assert inputs.shape == (fixture.N_INPUTS, fixture.N_NEURONS)
    assert recurrent.shape == (fixture.N_NEURONS, fixture.N_NEURONS)
    assert inputs.data.size == 14112
    assert recurrent.data.size == 16384
    assert jnp.all(jnp.diff(inputs.indptr) == 32)
    assert jnp.all(jnp.diff(recurrent.indptr) == 8)
    assert jnp.unique(inputs.indices).size > 1
    sources = jnp.repeat(jnp.arange(fixture.N_NEURONS), 8)
    assert jnp.all(sources != recurrent.indices)
    assert jnp.unique(jnp.stack((sources, recurrent.indices), axis=1), axis=0).shape[0] == 16384
    assert jnp.array_equal(
        recurrent.indices[:8], jnp.asarray([1, 2, 4, 8, 16, 32, 64, 128])
    )


def test_population_current_is_bounded_current_density():
    current = fixture.bounded_population_current(jnp.asarray([100.0]), jnp.asarray([-100.0]))
    assert current.unit == u.mA / u.cm**2
    assert float(current.to_decimal(u.mA / u.cm**2)[0]) == pytest.approx(0.01)


def test_gradient_clip_reports_unclipped_norm():
    clipped, norm = fixture.clip_gradient({"x": jnp.asarray([3.0, 4.0])})
    assert float(norm) == pytest.approx(5.0)
    assert jnp.linalg.norm(clipped["x"]) == pytest.approx(1.0)


def test_production_model_retains_parameters_across_episode_reset():
    model = fixture.BrainCellArcModel()
    input_before = model.input_weight.value.copy()
    recurrent_before = model.recurrent_weight.value.copy()
    model.cell.V.value = jnp.zeros_like(model.cell.V.value) * u.mV
    model.previous_spikes.value = jnp.ones_like(model.previous_spikes.value)
    model.reset_episode()
    assert jnp.array_equal(model.input_weight.value, input_before)
    assert jnp.array_equal(model.recurrent_weight.value, recurrent_before)
    assert jnp.allclose(model.cell.V.value.to_decimal(u.mV), -65.0)
    assert jnp.all(model.previous_spikes.value == 0)


def test_false_advance_preserves_biological_state_bitwise():
    model = fixture.BrainCellArcModel()
    voltage = model.cell.V.value.to_decimal(u.mV).copy()
    gates = tuple(gate.value.copy() for gate in (model.cell.na.INa.p, model.cell.na.INa.q, model.cell.k.IK.p))
    spikes = model.previous_spikes.value.copy()
    output = model.step(jnp.zeros((fixture.N_INPUTS,)), False)
    assert output.shape == (fixture.N_NEURONS,)
    assert jnp.array_equal(model.cell.V.value.to_decimal(u.mV), voltage)
    assert all(jnp.array_equal(gate.value, initial) for gate, initial in zip(
        (model.cell.na.INa.p, model.cell.na.INa.q, model.cell.k.IK.p), gates
    ))
    assert jnp.array_equal(model.previous_spikes.value, spikes)


def test_compiled_event_sequence_freezes_padding_and_returns_outputs():
    model = fixture.BrainCellArcModel()
    events = jnp.zeros((2, fixture.N_INPUTS), dtype=jnp.float32)
    before = model.cell.V.value.to_decimal(u.mV).copy()
    outputs = fixture.run_event_sequence(model, events, [False, False])
    assert outputs.shape == (2, fixture.N_NEURONS)
    assert jnp.array_equal(model.cell.V.value.to_decimal(u.mV), before)
    assert jnp.array_equal(outputs, jnp.zeros_like(outputs))


def test_padding_does_not_change_a_valid_sequence_result():
    events = jnp.zeros((2, fixture.N_INPUTS), dtype=jnp.float32)
    first = fixture.BrainCellArcModel()
    second = fixture.BrainCellArcModel()
    direct = fixture.run_event_sequence(first, events)
    padded = fixture.run_event_sequence(
        second, jnp.concatenate((events[:1], jnp.zeros((1, fixture.N_INPUTS)), events[1:])),
        [True, False, True],
    )
    assert jnp.array_equal(direct[1], padded[2])
    assert jnp.array_equal(
        first.cell.V.value.to_decimal(u.mV), second.cell.V.value.to_decimal(u.mV)
    )


def test_matched_integration_and_decoder_boundary_are_explicit():
    events = jnp.zeros((1, fixture.N_INPUTS), dtype=jnp.float32)
    check = fixture.matched_integration_check(events)
    assert check["finite"]
    assert check["max_voltage_difference"] <= 1.0
    model = fixture.BrainCellArcModel()
    before, after = fixture.decoder_boundary_intervention(model)
    assert jnp.any(before != after)


def test_trainer_schedule_keeps_episode_update_count_contract():
    assert fixture.update_schedule(fixture.PROOF_UPDATES, proof=True)[-1] == 7
    assert fixture.update_schedule(fixture.ORDINARY_UPDATES)[-1] == 63


def test_episode_loss_clip_adam_and_fixed_schedules():
    assert fixture.accumulate_masked_loss([1.0, 2.0, 3.0], [1, 0, 1]) == pytest.approx(4.0)
    gradient = {"x": jnp.asarray([3.0, 4.0])}
    state = fixture.AdamState({"x": jnp.zeros(2)}, {"x": jnp.zeros(2)})
    clipped, _ = fixture.clip_gradient(gradient)
    updated, next_state = fixture.adam_update({"x": jnp.zeros(2)}, clipped, state, 0.001)
    assert next_state.step == 1
    assert jnp.all(jnp.isfinite(updated["x"]))
    assert fixture.update_schedule(8, proof=True) == tuple(range(8))
    assert fixture.update_schedule(64) == tuple(range(64))
    with pytest.raises(ValueError):
        fixture.update_schedule(9, proof=True)


def test_grouped_adam_uses_declared_rates_and_finite_moments():
    parameters = {
        "input": jnp.zeros((1,)),
        "recurrent": jnp.zeros((1,)),
        "readout": jnp.zeros((1,)),
    }
    gradients = {name: jnp.ones((1,)) for name in parameters}
    updated, states = fixture.grouped_adam_update(parameters, gradients)
    assert float(updated["input"][0]) == pytest.approx(-0.001)
    assert float(updated["recurrent"][0]) == pytest.approx(-0.0003)
    assert float(updated["readout"][0]) == pytest.approx(-0.003)
    assert all(state.step == 1 for state in states.values())


def test_production_pp_prop_compile_has_two_temporal_relations():
    learner = fixture.compile_pp_prop_model(fixture.BrainCellArcModel())
    relations = learner.graph.hidden_param_op_relations
    assert len(relations) == 2
    assert all(relation.connected_hidden_paths for relation in relations)
    assert all(relation.trainable_vars for relation in relations)


def test_pp_prop_sequence_skips_false_events():
    model = fixture.BrainCellArcModel()
    learner = fixture.compile_pp_prop_model(model)
    before = model.cell.V.value.to_decimal(u.mV).copy()
    events = jnp.zeros((2, fixture.N_INPUTS), dtype=jnp.float32)
    outputs = fixture.run_pp_prop_sequence(learner, events, [False, False])
    assert outputs.shape == (2, fixture.N_NEURONS)
    assert jnp.array_equal(model.cell.V.value.to_decimal(u.mV), before)
