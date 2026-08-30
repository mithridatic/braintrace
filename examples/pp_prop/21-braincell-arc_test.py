"""Focused BrainCell compatibility tests."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import braincell
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
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


def test_clip_gradient_accepts_mixed_direct_and_etp_keys():
    gradient = {
        "readout_weight": jnp.ones((2, 2)),
        ("cell", "recurrent"): jnp.ones((2, 2)),
    }

    clipped, norm = fixture.clip_gradient(gradient)

    assert float(norm) == pytest.approx(np.sqrt(8.0))
    assert set(clipped) == set(gradient)
    assert all(np.all(np.isfinite(value)) for value in clipped.values())


def test_cli_help_is_a_real_command_path():
    result = subprocess.run(
        [sys.executable, str(_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--smoke" in result.stdout
    assert "proof" in result.stdout
    assert "finite_difference" not in result.stdout


def test_cli_smoke_writes_report(tmp_path):
    result = fixture.main(["--smoke", "--device", "cpu", "--output-dir", str(tmp_path)])
    assert result == 0
    report = tmp_path / "example21-smoke.json"
    assert report.exists()
    assert '"mode": "smoke"' in report.read_text(encoding="utf-8")


def test_cli_proof_dispatches_real_workflow(monkeypatch, tmp_path):
    calls = []

    def report(root, *, proof):
        calls.append((root, proof))
        return {"mode": "proof", "passed": True, "updates": 8}

    monkeypatch.setattr(fixture, "_real_workflow_report", report)
    result = fixture.main([
        "proof", "--device", "cpu", "--arc-root", str(tmp_path),
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert calls == [(tmp_path, True)]
    report = tmp_path / "example21-proof.json"
    assert report.exists()
    assert '"updates": 8' in report.read_text(encoding="utf-8")


def test_cli_proof_rejects_elapsed_deadline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fixture,
        "_real_workflow_report",
        lambda _root, *, proof: {"mode": "proof", "passed": True},
    )
    ticks = iter((10.0, 190.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))

    result = fixture.main([
        "proof", "--device", "cpu", "--arc-root", str(tmp_path),
        "--output-dir", str(tmp_path),
    ])

    assert result == 1
    report = json.loads((tmp_path / "example21-proof.json").read_text())
    assert report["elapsed_seconds"] == pytest.approx(180.0)
    assert report["deadline_seconds"] == fixture.PROOF_DEADLINE_SECONDS
    assert report["deadline_exceeded"]
    assert not report["passed"]


def test_cli_run_dispatches_fixed_real_workflow(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        fixture,
        "_real_workflow_report",
        lambda root, *, proof: calls.append((root, proof)) or {
            "mode": "run", "passed": True, "updates": 64,
        },
    )
    assert fixture.main(["run", "--arc-root", str(tmp_path)]) == 0
    assert calls == [(tmp_path, False)]


def test_proof_source_has_no_synthetic_schedule_probe():
    source = _PATH.read_text(encoding="utf-8")
    assert "_ScheduleProbe" not in source
    for name in ("load_task", "BrainCellArcModel", "PPPropEpisodeTrainer", "run_fixed_schedule"):
        assert name in source


def test_supervised_request_loss_changes_when_target_changes():
    logits = jnp.zeros((fixture.N_READOUT,)).at[0].set(4.0)
    target_shape = jnp.asarray([0, 0], dtype=jnp.int32)
    target_rows = jnp.zeros((30,), dtype=jnp.int32)
    valid_mask = jnp.zeros((30,), dtype=jnp.float32).at[0].set(1.0)

    first = fixture._supervised_request_loss(
        logits, target_shape, target_rows, valid_mask, jnp.asarray(1)
    )
    changed = fixture._supervised_request_loss(
        logits, jnp.asarray([1, 0], dtype=jnp.int32), target_rows,
        valid_mask, jnp.asarray(1)
    )
    assert float(first) != pytest.approx(float(changed))
    assert fixture._supervised_request_loss(
        logits, target_shape, target_rows, valid_mask, jnp.asarray(0)
    ) == 0.0
    row_logits = jnp.zeros((fixture.N_READOUT,)).at[60].set(4.0)
    row_first = fixture._supervised_request_loss(
        row_logits, target_shape, target_rows, valid_mask, jnp.asarray(2)
    )
    row_changed = fixture._supervised_request_loss(
        row_logits, target_shape, target_rows.at[0].set(1), valid_mask, jnp.asarray(2)
    )
    assert float(row_first) != pytest.approx(float(row_changed))


def test_supervised_episodes_separate_request_targets_from_event_inputs(monkeypatch):
    target = np.asarray([[2, 3], [4, 5]], dtype=np.uint8)
    task = SimpleNamespace(targets=[None, target])

    monkeypatch.setattr(fixture, "load_task", lambda *_args: task)
    monkeypatch.setattr(
        fixture,
        "encode_episode",
        lambda _task, query_index: (
            np.zeros((705, fixture.N_INPUTS), dtype=bool),
            np.ones((705,), dtype=bool),
        ),
    )
    episode = fixture._supervised_episodes("root", ("task",))[0]

    assert int(jnp.sum(episode["loss_mask"])) == 31
    assert int(episode["request_kind"][674]) == 1
    assert jnp.all(episode["request_kind"][675:] == 2)
    assert jnp.array_equal(episode["target_shape"][0], jnp.asarray([1, 1]))
    assert jnp.array_equal(episode["target_rows"][675][:2], jnp.asarray([2, 3]))
    assert jnp.array_equal(episode["target_rows"][676][:2], jnp.asarray([4, 5]))
    assert jnp.array_equal(
        episode["target_valid_mask"][675],
        jnp.asarray([1, 1] + [0] * 28, dtype=jnp.float32),
    )
    assert jnp.all(episode["advance_mask"])


def test_supervised_episodes_reject_an_unsupervised_task(monkeypatch):
    monkeypatch.setattr(
        fixture, "load_task", lambda *_args: SimpleNamespace(targets=[None])
    )
    with pytest.raises(ValueError, match="no supervised query"):
        fixture._supervised_episodes("root", ("task",))


def test_screen_predictions_uses_advance_mask_and_direct_decoder(monkeypatch):
    class State:
        value = jnp.zeros((1, fixture.N_READOUT))

    class Model:
        readout_weight = State()
        readout_bias = State()
        readout_bias.value = jnp.zeros((fixture.N_READOUT,))

        def reset_episode(self, _learner):
            pass

    monkeypatch.setattr(
        fixture,
        "run_event_sequence",
        lambda _model, events, advances: jnp.zeros((31, 1)),
    )
    records = fixture._screen_predictions(
        Model(), object(), [{
            "task_id": "task",
            "query_index": 0,
            "events": jnp.zeros((705, fixture.N_INPUTS)),
            "loss_mask": jnp.ones((705,), dtype=bool),
            "target": np.zeros((1, 1), dtype=np.uint8),
        }],
    )
    assert records[0]["prediction"] == [[0]]
    assert records[0]["exact"]


def test_direct_readout_gradient_changes_when_target_changes():
    features = jnp.ones((1, 4), dtype=jnp.float32)
    weight = jnp.zeros((4, fixture.N_READOUT), dtype=jnp.float32)
    bias = jnp.arange(fixture.N_READOUT, dtype=jnp.float32) / 10.0
    common = {
        "features": features,
        "target_rows": jnp.zeros((1, 30), dtype=jnp.int32),
        "target_valid_mask": jnp.zeros((1, 30), dtype=jnp.float32),
        "request_kind": jnp.ones((1,), dtype=jnp.int32),
        "request_mask": jnp.ones((1,), dtype=bool),
        "readout_weight": weight,
        "readout_bias": bias,
    }
    first = fixture._direct_readout_gradients(
        target_shape=jnp.asarray([[0, 0]], dtype=jnp.int32), **common
    )
    changed = fixture._direct_readout_gradients(
        target_shape=jnp.asarray([[1, 0]], dtype=jnp.int32), **common
    )
    assert not jnp.array_equal(first["readout_bias"], changed["readout_bias"])


def test_real_proof_requires_data_model_updates_and_observed_changes(monkeypatch, tmp_path):
    calls = []
    episode = {
        "task_id": "d631b094",
        "events": jnp.zeros((705, fixture.N_INPUTS)),
        "loss_mask": jnp.ones((705,), dtype=bool),
        "request_kind": jnp.zeros((705,), dtype=jnp.int32),
        "target_shape": jnp.zeros((705, 2), dtype=jnp.int32),
        "target_rows": jnp.zeros((705, 30), dtype=jnp.int32),
        "target_valid_mask": jnp.zeros((705, 30), dtype=jnp.float32),
        "target": [[0]],
        "query_index": 0,
    }
    validation = {**episode, "task_id": "46f33fce"}

    def load_episodes(root, task_ids):
        calls.append((root, tuple(task_ids)))
        return [validation] if task_ids == ("46f33fce",) else [episode]

    class State:
        def __init__(self, value):
            self.value = value

    class Model:
        input_weight = State(jnp.zeros((1,)))
        recurrent_weight = State(jnp.zeros((1,)))
        readout_weight = State(jnp.zeros((1, fixture.N_READOUT)))
        readout_bias = State(jnp.zeros((fixture.N_READOUT,)))

        def readout_features(self):
            return jnp.ones((1,))

    model = Model()

    class Trainer:
        def __init__(self):
            self.parameters = {"input": jnp.zeros((1,)), "recurrent": jnp.zeros((1,))}
            self.updates = 0

        def optimizer_is_finite(self):
            return True

    trainer = Trainer()
    screen_count = {"value": 0}

    def screen(_model, _learner, episodes):
        screen_count["value"] += 1
        value = 0 if screen_count["value"] != 2 else 1
        return [{
            "task_id": episodes[0]["task_id"],
            "query_index": 0,
            "prediction": [[value]],
            "target": [[0]],
            "exact": value == 0,
        }]

    def schedule(_trainer, episodes, *, proof):
        assert proof is True
        assert len(episodes) == fixture.PROOF_UPDATES
        assert all(item["task_id"] == "d631b094" for item in episodes)
        assert all("step_fn" in item for item in episodes)
        assert all("direct_grad_fn" in item for item in episodes)
        assert all("request_kind" in item for item in episodes)
        _, features = episodes[0]["step_fn"](
            jnp.zeros((fixture.N_INPUTS,)),
            jnp.asarray(True),
            jnp.asarray(1),
            jnp.asarray([0, 0]),
            jnp.zeros((30,), dtype=jnp.int32),
            jnp.zeros((30,), dtype=jnp.float32),
        )
        episodes[0]["direct_grad_fn"](
            aux=features[None, :],
            request_kind=jnp.ones((1,), dtype=jnp.int32),
            target_shape=jnp.zeros((1, 2), dtype=jnp.int32),
            target_rows=jnp.zeros((1, 30), dtype=jnp.int32),
            target_valid_mask=jnp.zeros((1, 30), dtype=jnp.float32),
            mask=jnp.ones((1,), dtype=bool),
        )
        model.recurrent_weight.value = jnp.ones((1,))
        _trainer.updates = fixture.PROOF_UPDATES

    monkeypatch.setattr(fixture, "_supervised_episodes", load_episodes)
    monkeypatch.setattr(fixture, "BrainCellArcModel", lambda: model)
    monkeypatch.setattr(fixture, "compile_pp_prop_model", lambda _model: (lambda _event: None))
    monkeypatch.setattr(fixture, "PPPropEpisodeTrainer", lambda *_args: trainer)
    monkeypatch.setattr(fixture, "_screen_predictions", screen)
    monkeypatch.setattr(fixture, "run_fixed_schedule", schedule)

    report = fixture._real_workflow_report(tmp_path, proof=True)

    assert report["passed"]
    assert report["recurrent_weight_changed"]
    assert report["prediction_changed"]
    assert report["validation_parameter_state_unchanged"]
    assert calls == [
        (tmp_path, ("d631b094",)),
        (tmp_path, ("46f33fce",)),
    ]


def test_cli_rejects_combined_modes():
    with pytest.raises(SystemExit) as error:
        fixture.main(["proof", "--smoke"])
    assert error.value.code == 2


def test_cli_rejects_unavailable_device(monkeypatch):
    def unavailable(_name):
        raise RuntimeError("missing device")

    monkeypatch.setattr(fixture.jax, "devices", unavailable)
    with pytest.raises(SystemExit) as error:
        fixture.main(["--smoke", "--device", "gpu"])
    assert error.value.code == 2


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
    outputs, spikes = fixture.run_event_sequence(
        model, events, [False, False], return_spikes=True
    )
    assert outputs.shape == (2, fixture.N_NEURONS)
    assert spikes.shape == (2, fixture.N_NEURONS)
    assert jnp.array_equal(model.cell.V.value.to_decimal(u.mV), before)
    assert jnp.array_equal(outputs, jnp.zeros_like(outputs))
    assert jnp.array_equal(spikes, jnp.zeros_like(spikes))


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


def test_two_half_step_fallback_does_not_replay_event():
    event = jnp.asarray([1.0, 2.0])
    assert jnp.array_equal(
        fixture.integration_substep_events(event, 2),
        jnp.asarray([[1.0, 2.0], [0.0, 0.0]]),
    )


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


def test_grouped_muon_updates_matrix_and_vector_parameters():
    parameters = {
        "input": jnp.zeros((2, 2)),
        "readout_bias": jnp.zeros((2,)),
    }
    gradients = {name: jnp.ones_like(value) for name, value in parameters.items()}
    updated, states = fixture.grouped_muon_update(parameters, gradients)
    assert set(states) == set(parameters)
    assert all(jnp.any(value != parameters[name]) for name, value in updated.items())
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves((updated, states))
    )


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


def test_event_sequence_uses_candidate_neuron_count_for_false_events():
    topology = type("Topology", (), {
        "neuron_count": 3,
        "input_source": jnp.asarray([], dtype=jnp.int32),
        "input_target": jnp.asarray([], dtype=jnp.int32),
        "input_value": jnp.asarray([], dtype=jnp.float32),
        "recurrent_source": jnp.asarray([], dtype=jnp.int32),
        "recurrent_target": jnp.asarray([], dtype=jnp.int32),
        "recurrent_value": jnp.asarray([], dtype=jnp.float32),
        "readout": jnp.zeros((3, fixture.N_READOUT), dtype=jnp.float32),
    })()
    model = fixture.BrainCellArcModel(topology)
    outputs = fixture.run_event_sequence(
        model, jnp.zeros((1, fixture.N_INPUTS)), [False]
    )
    assert outputs.shape == (1, 3)


def test_pp_prop_sequence_preserves_eligibility_across_interspersed_padding():
    events = jnp.zeros((2, fixture.N_INPUTS), dtype=jnp.float32)
    direct_model = fixture.BrainCellArcModel()
    padded_model = fixture.BrainCellArcModel()
    direct = fixture.run_pp_prop_sequence(
        fixture.compile_pp_prop_model(direct_model), events
    )
    padded = fixture.run_pp_prop_sequence(
        fixture.compile_pp_prop_model(padded_model),
        jnp.concatenate((events[:1], events[:1], events[1:])),
        [True, False, True],
    )
    assert jnp.array_equal(direct[1], padded[2])
    assert jnp.array_equal(
        direct_model.cell.V.value.to_decimal(u.mV),
        padded_model.cell.V.value.to_decimal(u.mV),
    )


def test_trainer_passes_request_mask_into_gradient_objective():
    class Learner:
        def __init__(self):
            self.mask = None

        def etrace_grad(self, events, *, step_fn, mask, **kwargs):
            self.mask = mask
            return {"input": jnp.ones((1,))}, jnp.asarray([2.0, 3.0]) * mask

    learner = Learner()
    trainer = fixture.PPPropEpisodeTrainer(learner, {"input": jnp.zeros((1,))})
    mask = jnp.asarray([1.0, 0.0])
    loss, _ = trainer.update_episode(jnp.zeros((2, 1)), lambda _: 0.0, loss_mask=mask)
    assert jnp.array_equal(learner.mask, mask)
    assert float(loss) == pytest.approx(2.0)
    assert trainer.updates == 1
    assert trainer.optimizer_is_finite()


def test_trainer_masks_false_advances_from_loss_and_gradient():
    class Learner:
        def etrace_grad(self, events, advances, *, step_fn, mask, **kwargs):
            self.advances = advances
            self.mask = mask
            return {"input": jnp.ones((1,))}, jnp.ones((events.shape[0],))

    learner = Learner()
    trainer = fixture.PPPropEpisodeTrainer(learner, {"input": jnp.zeros((1,))})
    trainer.update_episode(
        jnp.zeros((2, 1)),
        lambda event, advance: jnp.asarray(1.0),
        loss_mask=jnp.ones((2,), dtype=bool),
        advance_mask=jnp.asarray([True, False]),
    )

    assert jnp.array_equal(learner.advances, jnp.asarray([True, False]))
    assert jnp.array_equal(learner.mask, jnp.asarray([1.0, 0.0]))


def test_trainer_updates_direct_readout_parameters_and_shared_schedule():
    class Learner:
        def etrace_grad(self, events, *, step_fn, mask, **kwargs):
            return {"input": jnp.ones((1,))}, jnp.asarray([1.0])

    parameters = {
        "input": jnp.zeros((1,)),
        "readout_weight": jnp.zeros((1, 2)),
        "readout_bias": jnp.zeros((2,)),
    }
    trainer = fixture.PPPropEpisodeTrainer(Learner(), parameters)
    trainer.update_episode(
        jnp.zeros((1, 1)), lambda _: 0.0,
        direct_grad_fn=lambda **_: {
            "readout_weight": jnp.ones((1, 2)),
            "readout_bias": jnp.ones((2,)),
        },
    )
    assert jnp.all(parameters["readout_weight"] == 0.0)
    assert jnp.all(trainer.parameters["readout_weight"] < 0.0)
    assert jnp.all(trainer.parameters["readout_bias"] < 0.0)
    assert set(trainer.muon_groups) == set(parameters)


def test_schedule_rejects_validation_and_wrong_ordinary_task_order():
    class Trainer:
        def update_episode(self, **kwargs):
            raise AssertionError("gate must reject before update")

    proof = [{"task_id": "d631b094", "validation": True}] * 8
    with pytest.raises(ValueError, match="forward-only"):
        fixture.run_fixed_schedule(Trainer(), proof, proof=True)
    wrong = [{"task_id": "dc433765"}] * 64
    with pytest.raises(ValueError, match="task order"):
        fixture.run_fixed_schedule(Trainer(), wrong)


def test_valid_schedule_does_not_forward_schedule_metadata():
    class Trainer:
        def reset_episode(self):
            pass

        def update_episode(self, **kwargs):
            return kwargs["events"]

    trainer = Trainer()
    episodes = [{
        "task_id": fixture.TRAINING_TASK_IDS[index % len(fixture.TRAINING_TASK_IDS)],
        "validation": False,
        "events": jnp.asarray([index]),
    } for index in range(fixture.ORDINARY_UPDATES)]
    result = fixture.run_fixed_schedule(trainer, episodes)
    assert jnp.array_equal(result[-1], jnp.asarray([fixture.ORDINARY_UPDATES - 1]))


def test_real_schedule_passes_advance_mask_resets_and_freezes_padding(monkeypatch):
    target = np.asarray([[1]], dtype=np.uint8)
    task = SimpleNamespace(targets=[target])
    monkeypatch.setattr(fixture, "load_task", lambda *_args: task)
    monkeypatch.setattr(
        fixture,
        "encode_episode",
        lambda *_args: (
            np.zeros((705, fixture.N_INPUTS), dtype=bool),
            np.asarray([True, False] + [True] * 703),
        ),
    )
    episode = fixture._supervised_episodes("root", ("d631b094",))[0]
    accepted = set(inspect.signature(fixture.PPPropEpisodeTrainer.update_episode).parameters)
    payload = {
        key for key in episode
        if key not in {"task_id", "target", "query_index"}
    }
    assert payload <= accepted

    class Trainer:
        def __init__(self):
            self.reset_count = brainstate.State(jnp.asarray(0, dtype=jnp.int32))
            self.update_count = brainstate.State(jnp.asarray(0, dtype=jnp.int32))
            self.false_count = brainstate.State(jnp.asarray(0, dtype=jnp.int32))
            self.loss_total = brainstate.State(jnp.asarray(0.0))

        def reset_episode(self):
            self.reset_count.value += 1

        def update_episode(
            self,
            events,
            step_fn,
            advance_mask,
            loss_mask,
            request_kind,
            target_shape,
            target_rows,
            target_valid_mask,
        ):
            del loss_mask
            self.update_count.value += 1
            self.false_count.value += jnp.sum(~advance_mask)
            losses = brainstate.transform.for_loop(
                lambda event, advance, kind, shape, rows, valid: step_fn(
                    event, advance, kind, shape, rows, valid
                ),
                events,
                advance_mask,
                request_kind,
                target_shape,
                target_rows,
                target_valid_mask,
            )
            self.loss_total.value += jnp.sum(losses)
            return losses

    trainer = Trainer()
    true_calls = brainstate.State(jnp.asarray(0, dtype=jnp.int32))

    def step_fn(event, advance, *_args):
        del event

        def advancing():
            true_calls.value += 1
            return jnp.asarray(1.0)

        return brainstate.transform.cond(advance, advancing, lambda: jnp.asarray(0.0))

    episodes = [{**episode, "step_fn": step_fn}] * fixture.PROOF_UPDATES
    fixture.run_fixed_schedule(trainer, episodes, proof=True)

    assert int(trainer.reset_count.value) == fixture.PROOF_UPDATES
    assert int(trainer.update_count.value) == fixture.PROOF_UPDATES
    assert int(trainer.false_count.value) == fixture.PROOF_UPDATES
    assert int(true_calls.value) == 704 * fixture.PROOF_UPDATES
    assert float(trainer.loss_total.value) == pytest.approx(704 * fixture.PROOF_UPDATES)


def test_matched_check_reports_selected_fallback_and_forward_validation_isolated():
    check = fixture.matched_integration_check(jnp.zeros((1, fixture.N_INPUTS)))
    assert check["selected_substeps"] in (1, 2)
    trainer = fixture.PPPropEpisodeTrainer(
        object(), {"input": jnp.zeros((1,))}
    )
    before = trainer.parameters["input"].copy()
    result = trainer.evaluate_forward(lambda: "validation")
    assert result == "validation"
    assert jnp.array_equal(trainer.parameters["input"], before)


def test_forward_validation_rejects_biological_and_eligibility_state_changes():
    class Learner:
        def __init__(self):
            self.biological = brainstate.HiddenState(jnp.zeros((1,)))
            self.eligibility = fixture.braintrace.EligibilityTrace(jnp.zeros((1,)))

        def states(self):
            return {"biological": self.biological, "eligibility": self.eligibility}

    learner = Learner()
    trainer = fixture.PPPropEpisodeTrainer(learner, {"input": jnp.zeros((1,))})

    def mutate_state():
        learner.biological.value = jnp.ones((1,))
        return "validation"

    with pytest.raises(RuntimeError, match="biological or eligibility"):
        trainer.evaluate_forward(mutate_state)


def test_real_compiled_episode_updates_grouped_parameters_and_param_states():
    model = fixture.BrainCellArcModel()
    learner = fixture.compile_pp_prop_model(model)
    trainer = fixture.PPPropEpisodeTrainer(
        learner,
        {"input": model.input_weight.value, "recurrent": model.recurrent_weight.value},
    )
    trainer.update_episode(
        jnp.zeros((1, fixture.N_INPUTS)),
        lambda event: jnp.sum(
            learner.etrace_evolve(event[None, :], return_outputs=True)[0]
        ),
    )
    assert trainer.updates == 1
    assert trainer.optimizer_is_finite()
    assert jnp.array_equal(
        model.input_weight.value, trainer.parameters["input"]
    )


def test_inferred_readout_parameters_have_muon_groups_and_update():
    model = fixture.BrainCellArcModel()
    class Learner:
        model4compile = model

        def etrace_grad(self, events, *, step_fn, mask, **kwargs):
            return {"input": jnp.zeros_like(model.input_weight.value)}, jnp.asarray([0.0])

    trainer = fixture.PPPropEpisodeTrainer(
        Learner(),
        {"input": model.input_weight.value, "recurrent": model.recurrent_weight.value},
    )
    before_weight = trainer.parameters["readout_weight"].copy()
    before_bias = trainer.parameters["readout_bias"].copy()

    trainer.update_episode(
        jnp.zeros((1, fixture.N_INPUTS)),
        lambda event: jnp.asarray(0.0),
        direct_grad_fn=lambda **_: {
            "readout_weight": jnp.ones_like(before_weight),
            "readout_bias": jnp.ones_like(before_bias),
        },
    )

    assert "readout_weight" in trainer.muon_groups
    assert "readout_bias" in trainer.muon_groups
    assert not jnp.array_equal(trainer.parameters["readout_weight"], before_weight)
    assert not jnp.array_equal(trainer.parameters["readout_bias"], before_bias)


def test_muon_state_remains_concrete_after_transformed_updates():
    class Learner:
        def etrace_grad(self, events, *, step_fn, mask, **kwargs):
            return {"input": jnp.ones((2,))}, jnp.zeros((1,))

    trainer = fixture.PPPropEpisodeTrainer(
        Learner(), {"input": jnp.zeros((2,))}
    )

    def update(_index):
        return trainer.update_episode(
            jnp.zeros((1, fixture.N_INPUTS)),
            lambda event: jnp.asarray(0.0),
            loss_mask=jnp.ones((1,), dtype=bool),
        )

    brainstate.transform.jit(
        lambda indices: brainstate.transform.for_loop(update, indices)
    )(jnp.arange(2, dtype=jnp.int32))
    leaves = jax.tree_util.tree_leaves(trainer.muon_groups)
    assert leaves
    assert all(type(leaf).__name__ != "DynamicJaxprTracer" for leaf in leaves)
    counts = [int(leaf) for leaf in leaves if getattr(leaf, "shape", None) == ()]
    assert 2 in counts
    assert int(trainer.updates) == 2


def test_compacted_model_reset_uses_candidate_neuron_count():
    structural_path = Path(__file__).with_name("example21_structural.py")
    structural_spec = importlib.util.spec_from_file_location("example21_structural", structural_path)
    structural = importlib.util.module_from_spec(structural_spec)
    structural_spec.loader.exec_module(structural)
    model = fixture.BrainCellArcModel()
    topology = structural.topology_from_model(model)
    alive = jnp.ones((topology.neuron_count,), dtype=bool)
    alive = alive.at[:structural.mutation_count(topology.neuron_count)].set(False)
    zeros = [jnp.zeros_like(value) for value in (
        topology.readout, topology.readout, topology.input_value,
        topology.input_value, topology.recurrent_value, topology.recurrent_value,
    )]
    compacted = structural.compact(
        topology, alive, structural.StructuralAdam(*zeros)
    )[0]
    candidate = fixture.BrainCellArcModel(compacted)
    candidate.reset_episode()
    assert candidate.previous_spikes.value.shape == (compacted.neuron_count,)
