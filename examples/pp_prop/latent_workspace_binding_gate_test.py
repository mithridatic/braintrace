"""Tests for the post-architecture Example 21 binding gate."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.pp_prop import latent_workspace_binding_gate as gate


def _accuracy(accuracy: float, lower: float) -> dict[str, float | int]:
    return {
        "accuracy": accuracy,
        "wilson_95_lower": lower,
        "count": 512,
        "correct": round(accuracy * 512),
    }


def _passing_evaluation() -> dict[str, object]:
    intact = _accuracy(0.90, 0.86)
    shuffled = _accuracy(0.20, 0.17)
    no_context = _accuracy(0.10, 0.08)
    return {
        "intact": intact,
        "shuffled": shuffled,
        "no_context": no_context,
        "intact_minus_shuffled": 0.70,
        "pairing_chance": 0.25,
        "depths": {
            "0": {"intact": intact, "shuffled": shuffled, "no_context": no_context},
            "1": {"intact": intact, "shuffled": shuffled, "no_context": no_context},
        },
    }


def _passing_diagnostics() -> dict[str, object]:
    return {
        "memory": {
            "intact_shuffled_different_count": 512,
            "applicable_count": 512,
            "every_intact_shuffled_pair_differs": True,
            "no_context_exact_zero": True,
        }
    }


def _passing_compiler() -> dict[str, object]:
    paths = [
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    ]
    return {
        "available": True,
        "all_required_direct": True,
        "diagnostics": [],
        "compiled_parameter_paths": paths,
        "required_direct_paths": paths,
        "direct_path_status": {path: True for path in paths},
        "direct_path_evidence": {
            path: [{"relation_key": "weight", "classification": "all_direct"}]
            for path in paths
        },
    }


def _passing_training() -> dict[str, object]:
    return {
        "required_direct_parameter_movement": {
            path: {"l2_delta": 0.1, "parameter_count": 1, "changed": True}
            for path in _passing_compiler()["required_direct_paths"]
        }
    }


def _passing_architecture() -> dict[str, object]:
    return {
        "mode": "associative_workspace",
        "memory_width": 32,
        "key_map": "fixed_rff_cosine",
        "value_map": "fixed_tanh_projection",
        "rff_gamma": 2.0,
        "key_basis_seed": 2209,
        "key_bias_seed": 2210,
        "value_basis_seed": 2211,
        "key_basis_sha256": "a" * 64,
        "key_bias_sha256": "b" * 64,
        "value_basis_sha256": "c" * 64,
        "write_component_type": "braintrace.element_wise",
        "query_component_type": "braintrace.nn.Linear",
        "read_component_type": "braintrace.nn.Linear",
    }


def _passing_separation(
    architecture: dict[str, object], *, portable: bool = False
) -> dict[str, object]:
    reported_architecture = dict(architecture)
    if portable:
        reported_architecture.update(
            key_basis_sha256="d" * 64,
            key_bias_sha256="e" * 64,
            value_basis_sha256="f" * 64,
        )
    return {
        "memory_width": 32,
        "model_seed": 2108,
        "separation_margin": 0.503,
        "required_margin": 0.25,
        "margin_passed": True,
        "zero_event_key_max_abs": 0.0,
        "zero_event_key_exact_zero": True,
        "architecture": reported_architecture,
    }


def test_config_preregisters_production_pp_prop_memory_gate() -> None:
    config = gate.BindingGateConfig()

    assert config.training_updates == 10_000
    assert config.batch_size == 64
    assert config.validation_episodes == 512
    assert config.gap_steps == 1
    assert config.neuron_count == 2048
    assert config.recurrent_edges == 16_384
    assert config.readout_width == 128
    assert config.color_rank == 16
    assert config.trace_decay == 0.9
    assert config.context_memory_width == 32
    assert config.memory_decay == 1.0
    assert config.configuration_scale == "production_topology"
    assert config.qualification_regime == "preregistered_full"

    assert (
        dataclasses.replace(config, training_updates=9_999).qualification_regime
        == "nonqualifying_abbreviated"
    )
    assert (
        dataclasses.replace(config, context_memory_width=64).qualification_regime
        == "nonqualifying_abbreviated"
    )
    assert (
        dataclasses.replace(config, memory_decay=0.9).qualification_regime
        == "nonqualifying_abbreviated"
    )
    with pytest.raises(ValueError, match="context_memory_width"):
        dataclasses.replace(config, context_memory_width=0)
    with pytest.raises(ValueError, match="context_memory_width"):
        dataclasses.replace(config, context_memory_width=129)
    with pytest.raises(ValueError, match="memory_decay"):
        dataclasses.replace(config, memory_decay=-0.1)
    with pytest.raises(ValueError, match="memory_decay"):
        dataclasses.replace(config, memory_decay=1.1)
    with pytest.raises(ValueError, match="batch_size"):
        dataclasses.replace(config, batch_size=3)


def test_model_config_wires_exact_binding_features_and_memory() -> None:
    config = gate.BindingGateConfig.smoke_config()
    model_config = gate._model_config(config, batch_size=3)
    indices = gate.associative_memory_feature_indices(config.row_config)
    rows = config.row_config

    assert model_config.batch_size == 3
    assert model_config.context_memory_width == 32
    assert model_config.memory_decay == 1.0
    assert model_config.memory_key_indices == indices.key_indices
    assert model_config.memory_value_indices == indices.value_indices
    assert model_config.demonstration_phase_index == rows.phase_slice.start
    assert model_config.query_phase_index == rows.phase_slice.start + 1
    assert model_config.input_side_valid_index == rows.side_valid_slice.start
    assert model_config.output_side_valid_index == rows.side_valid_slice.start + 1


def test_deep_supervision_mask_covers_query_and_every_latent_depth() -> None:
    config = gate.BindingGateConfig.smoke_config()
    mask = np.asarray(gate._deep_supervision_mask(config))

    assert mask.shape == (config.sequence_length,)
    np.testing.assert_array_equal(mask[: gate.SYMBOL_COUNT], 0.0)
    np.testing.assert_allclose(
        mask[gate.SYMBOL_COUNT :],
        np.full((config.gap_steps + 1,), 1.0 / (config.gap_steps + 1)),
        rtol=0.0,
        atol=0.0,
    )
    assert float(mask.sum()) == pytest.approx(1.0)

    deeper = dataclasses.replace(config, gap_steps=3)
    deeper_mask = np.asarray(gate._deep_supervision_mask(deeper))
    np.testing.assert_array_equal(deeper_mask[: gate.SYMBOL_COUNT], 0.0)
    np.testing.assert_allclose(deeper_mask[gate.SYMBOL_COUNT :], 0.25)
    assert float(deeper_mask.sum()) == pytest.approx(1.0)


def test_configuration_digest_is_deterministic_and_setting_sensitive() -> None:
    config = gate.BindingGateConfig.smoke_config()
    model = gate._model_config(config, batch_size=config.batch_size)
    repeated = gate._model_config(config, batch_size=config.batch_size)
    changed = dataclasses.replace(config, context_memory_width=64)

    assert gate._configuration_digest(config, model) == gate._configuration_digest(
        config, repeated
    )
    assert gate._configuration_digest(config, model) != gate._configuration_digest(
        changed, gate._model_config(changed, batch_size=changed.batch_size)
    )


def test_shared_data_preserves_exact_marginals_and_changes_pairings() -> None:
    config = gate.BindingGateConfig.smoke_config()
    data = gate.build_binding_data(config)
    report = gate._marginal_identity_report(data, config)

    assert report["exact_input_marginal_equality"] is True
    assert report["exact_output_marginal_equality"] is True
    assert report["exact_marginal_equality"] is True
    assert report["different_pairing_count"] == config.validation_episodes
    assert report["every_pairing_differs"] is True


def test_live_width32_standard_arc_key_gate_reproduces_preregistered_margin() -> None:
    report = gate._standard_arc_key_separation_report()

    assert report["protocol"] == "standard_arc_k4_query_key_gram_424_features"
    assert report["row_event_input_width"] == 830
    assert report["raw_key_feature_width"] == 424
    assert report["memory_width"] == 32
    np.testing.assert_allclose(
        report["diagonal"],
        [0.7478269935, 0.8287837505, 0.9190308452, 0.8519610763],
        rtol=0.0,
        atol=5e-4,
    )
    assert report["off_diagonal_maximum"] == pytest.approx(0.2448071092, abs=5e-4)
    assert report["separation_margin"] == pytest.approx(0.5030198693, abs=5e-4)
    assert report["margin_passed"] is True
    assert report["zero_event_key_max_abs"] == 0.0
    assert report["zero_event_key_exact_zero"] is True
    architecture = report["architecture"]
    assert architecture["key_basis_seed"] == 2209
    assert architecture["key_bias_seed"] == 2210
    assert architecture["value_basis_seed"] == 2211
    assert architecture["write_component_type"] == "braintrace.element_wise"


def test_diagnostic_report_checks_each_memory_and_every_depth() -> None:
    intact_memory = np.asarray(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
        dtype=np.float32,
    )
    shuffled_memory = intact_memory[:, ::-1, :].copy()
    no_context_memory = np.zeros_like(intact_memory)
    intact_read = np.asarray([[[1.0, 0.0], [2.0, 0.0]], [[1.5, 0.0], [2.5, 0.0]]])
    shuffled_read = intact_read + 1.0
    no_context_read = np.zeros_like(intact_read)
    intact_workspace = np.asarray([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]])
    shuffled_workspace = intact_workspace - 0.5
    no_context_workspace = np.zeros_like(intact_workspace)

    report = gate._diagnostic_report(
        {
            "intact": (intact_memory, intact_read, intact_workspace),
            "shuffled": (shuffled_memory, shuffled_read, shuffled_workspace),
            "no_context": (no_context_memory, no_context_read, no_context_workspace),
        }
    )

    assert report["memory"]["applicable_count"] == 2
    assert report["memory"]["intact_shuffled_different_count"] == 2
    assert report["memory"]["every_intact_shuffled_pair_differs"] is True
    assert report["memory"]["no_context_exact_zero"] is True
    assert len(report["read_by_depth"]) == 2
    assert len(report["workspace_by_depth"]) == 2
    assert report["read_by_depth"]["0"]["different_count"] == 2
    assert report["workspace_by_depth"]["1"]["different_count"] == 2

    with pytest.raises(ValueError, match="memory snapshots"):
        gate._diagnostic_report(
            {
                "intact": (intact_memory, intact_read, intact_workspace),
                "shuffled": (shuffled_memory, shuffled_read, shuffled_workspace),
                "no_context": (
                    no_context_memory[:1],
                    no_context_read,
                    no_context_workspace,
                ),
            }
        )


def test_compiler_report_requires_all_three_direct_paths() -> None:
    required = {
        ("memory_write_scale",),
        ("workspace_query_projection", "weight"),
        ("memory_read_projection", "weight"),
    }

    def relation(path: tuple[str, ...], classification: str = "all_direct"):
        return SimpleNamespace(
            trainable_paths={"weight": path},
            path_classification={"weight": classification},
        )

    learner = SimpleNamespace(
        report=SimpleNamespace(diagnostics=[], etrace_weights=[]),
        graph=SimpleNamespace(
            hidden_param_op_relations=[relation(path) for path in required]
        ),
        param_states={path: object() for path in required},
    )
    complete = gate._compiler_report(learner)

    assert complete["all_required_direct"] is True
    assert set(complete["required_direct_paths"]) == {
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    }

    learner.graph.hidden_param_op_relations[-1] = relation(
        ("memory_read_projection", "weight"), "indirect"
    )
    incomplete = gate._compiler_report(learner)
    assert incomplete["all_required_direct"] is False

    learner.graph.hidden_param_op_relations[-1] = relation(
        ("memory_read_projection", "weight")
    )
    learner.graph.hidden_param_op_relations.append(
        relation(("memory_read_projection", "weight"), "indirect")
    )
    mixed = gate._compiler_report(learner)
    assert mixed["direct_path_status"]["memory_read_projection/weight"] is False
    assert mixed["all_required_direct"] is False


def test_required_parameter_movement_is_reported_per_direct_path() -> None:
    before = {
        "memory_write_scale": np.asarray([1.0, 2.0]),
        "workspace_query_projection/weight": np.asarray([[3.0]]),
        "memory_read_projection/weight": np.asarray([[4.0, 5.0]]),
    }
    after = {
        "memory_write_scale": np.asarray([1.0, 2.5]),
        "workspace_query_projection/weight": np.asarray([[3.0]]),
        "memory_read_projection/weight": np.asarray([[3.0, 5.0]]),
    }

    report = gate._required_parameter_movement(before, after)

    assert report["memory_write_scale"] == {
        "l2_delta": 0.5,
        "parameter_count": 2,
        "changed": True,
    }
    assert report["workspace_query_projection/weight"]["changed"] is False
    assert report["memory_read_projection/weight"]["l2_delta"] == 1.0
    with pytest.raises(ValueError, match="memory_read_projection/weight"):
        gate._required_parameter_movement(
            before, {key: value for key, value in after.items() if "read" not in key}
        )


@pytest.mark.parametrize(
    ("mutation", "criterion"),
    [
        ("validation_count", "held_out_count_at_least_256"),
        ("intact_accuracy", "intact_accuracy_at_least_0_80"),
        ("intact_wilson", "intact_wilson_lower_above_pairing_chance"),
        ("pairing_gap", "intact_minus_shuffled_at_least_0_25"),
        ("shuffled_chance", "shuffled_not_demonstrably_above_pairing_chance"),
        ("marginals", "exact_marginal_equality"),
        ("memory_pairing", "every_intact_shuffled_memory_differs"),
        ("no_context", "no_context_memory_exact_zero"),
        ("compiler", "compiler_required_paths_all_direct"),
        ("compiler_error", "compiler_required_paths_all_direct"),
        ("movement", "required_direct_parameters_moved"),
        ("architecture", "architecture_matches_preregistered_components"),
        ("native_separation", "gate_native_key_separation_margin_passed"),
        ("native_zero_key", "gate_native_zero_event_key_exact_zero"),
        ("native_basis", "gate_native_basis_matches_training_model"),
        ("standard_separation", "standard_arc_key_separation_margin_passed"),
        ("standard_zero_key", "standard_arc_zero_event_key_exact_zero"),
        ("standard_shared", "standard_arc_shared_encoder_invariants_match"),
        ("source", "source_clean"),
        ("source_commit", "source_commit_available"),
        ("config", "preregistered_full_configuration"),
    ],
)
def test_qualification_fails_closed_for_every_required_criterion(
    mutation: str, criterion: str
) -> None:
    config = gate.BindingGateConfig()
    evaluation = _passing_evaluation()
    diagnostics = _passing_diagnostics()
    compiler = _passing_compiler()
    training = _passing_training()
    architecture = _passing_architecture()
    gate_native = _passing_separation(architecture)
    standard_arc = _passing_separation(architecture, portable=True)
    marginal = {"exact_marginal_equality": True}
    source = {"dirty": False, "commit": "abc"}

    if mutation == "validation_count":
        evaluation["intact"] = _accuracy(0.90, 0.86) | {"count": 255}
    elif mutation == "intact_accuracy":
        evaluation["intact"] = _accuracy(0.79, 0.75)
    elif mutation == "intact_wilson":
        evaluation["intact"] = _accuracy(0.90, 0.25)
    elif mutation == "pairing_gap":
        evaluation["intact_minus_shuffled"] = 0.24
    elif mutation == "shuffled_chance":
        evaluation["shuffled"] = _accuracy(0.40, 0.31)
    elif mutation == "marginals":
        marginal["exact_marginal_equality"] = False
    elif mutation == "memory_pairing":
        diagnostics["memory"] = dict(diagnostics["memory"])
        diagnostics["memory"]["every_intact_shuffled_pair_differs"] = False
    elif mutation == "no_context":
        diagnostics["memory"] = dict(diagnostics["memory"])
        diagnostics["memory"]["no_context_exact_zero"] = False
    elif mutation == "compiler":
        compiler["direct_path_evidence"]["memory_read_projection/weight"] = []
    elif mutation == "compiler_error":
        compiler["diagnostics"] = [{"level": "error", "message": "bad route"}]
    elif mutation == "movement":
        training["required_direct_parameter_movement"]["memory_write_scale"] = {
            "l2_delta": 0.0,
            "parameter_count": 1024,
            "changed": False,
        }
    elif mutation == "architecture":
        architecture["read_component_type"] = "unknown"
    elif mutation == "native_separation":
        gate_native["margin_passed"] = False
    elif mutation == "native_zero_key":
        gate_native["zero_event_key_exact_zero"] = False
    elif mutation == "native_basis":
        gate_native["architecture"] = dict(gate_native["architecture"])
        gate_native["architecture"]["key_basis_sha256"] = "0" * 64
    elif mutation == "standard_separation":
        standard_arc["margin_passed"] = False
    elif mutation == "standard_zero_key":
        standard_arc["zero_event_key_exact_zero"] = False
    elif mutation == "standard_shared":
        standard_arc["architecture"] = dict(standard_arc["architecture"])
        standard_arc["architecture"]["rff_gamma"] = 3.0
    elif mutation == "source":
        source["dirty"] = True
    elif mutation == "source_commit":
        source["commit"] = "unavailable"
    elif mutation == "config":
        config = dataclasses.replace(config, training_updates=9_999)

    report = gate._qualification_report(
        evaluation=evaluation,
        diagnostics=diagnostics,
        compiler=compiler,
        training=training,
        architecture=architecture,
        gate_native_separation=gate_native,
        standard_arc_separation=standard_arc,
        marginals=marginal,
        source=source,
        config=config,
    )

    assert report["passed"] is False
    assert report["criteria"][criterion] is False
    assert report["interpretation"].endswith("no_capability_conclusion")


def test_qualification_passes_only_complete_preregistered_evidence() -> None:
    architecture = _passing_architecture()
    report = gate._qualification_report(
        evaluation=_passing_evaluation(),
        diagnostics=_passing_diagnostics(),
        compiler=_passing_compiler(),
        training=_passing_training(),
        architecture=architecture,
        gate_native_separation=_passing_separation(architecture),
        standard_arc_separation=_passing_separation(architecture, portable=True),
        marginals={"exact_marginal_equality": True},
        source={"dirty": False, "commit": "abc"},
        config=gate.BindingGateConfig(),
    )

    assert report["passed"] is True
    assert all(report["criteria"].values())
    assert report["interpretation"] == "gate_a_passed_associative_binding"

    reduced_architecture = _passing_architecture()
    reduced = gate._qualification_report(
        evaluation=_passing_evaluation(),
        diagnostics=_passing_diagnostics(),
        compiler=_passing_compiler(),
        training=_passing_training(),
        architecture=reduced_architecture,
        gate_native_separation=_passing_separation(reduced_architecture),
        standard_arc_separation=_passing_separation(
            reduced_architecture, portable=True
        ),
        marginals={"exact_marginal_equality": True},
        source={"dirty": False, "commit": "abc"},
        config=gate.BindingGateConfig.smoke_config(),
    )
    assert reduced["passed"] is False
    assert reduced["interpretation"] == (
        "nonqualifying_abbreviated_no_capability_conclusion"
    )


def test_qualification_malformed_evidence_fails_closed() -> None:
    report = gate._qualification_report(
        evaluation={},
        diagnostics={},
        compiler={},
        training={},
        architecture={},
        gate_native_separation={},
        standard_arc_separation={},
        marginals={},
        source={},
        config=gate.BindingGateConfig(),
    )

    assert report["passed"] is False
    assert not any(report["criteria"].values())
    assert report["interpretation"] == "gate_a_failed_stop_no_capability_conclusion"


def test_model_drivers_use_brainstate_transforms_and_no_jax_random() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("train_all", "train_one", "evaluate", "step"):
        nodes = [
            node for function_name, node in functions.items() if function_name == name
        ]
        assert nodes
        for node in nodes:
            assert not any(
                isinstance(child, (ast.For, ast.While)) for child in ast.walk(node)
            )
    assert "brainstate.transform.for_loop" in source
    assert "learner.etrace_grad" in source
    assert "bptt_param_gradients" not in source
    assert "brainstate.transform.grad" not in source
    assert "jax.random" not in source


def test_source_report_fails_closed_when_git_and_env_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAINTRACE_SOURCE_COMMIT", raising=False)
    monkeypatch.delenv("BRAINTRACE_SOURCE_DIRTY", raising=False)

    def unavailable(*args: object, **kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr(gate.subprocess, "run", unavailable)

    assert gate._source_report() == {"commit": "unavailable", "dirty": True}


def test_actual_model_pp_prop_smoke_records_routes_depths_and_state_evidence() -> None:
    config = gate.BindingGateConfig.smoke_config()
    result = gate.run_binding_gate(config)

    assert result["learner"] == "pp_prop_only"
    assert result["config"]["configuration_scale"] == "reduced_smoke"
    assert result["config"]["qualification_regime"] == ("nonqualifying_abbreviated")
    training = result["training"]
    assert len(training["losses"]) == config.training_updates
    assert np.isfinite(training["losses"]).all()
    np.testing.assert_allclose(
        training["supervision_mask"], [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    )
    assert training["supervision_weight_sum"] == pytest.approx(1.0)
    compiler = training["compiler"]
    assert compiler["all_required_direct"] is True
    assert all(compiler["direct_path_status"].values())
    assert set(training["required_direct_parameter_movement"]) == set(
        compiler["required_direct_paths"]
    )
    assert result["architecture"]["memory_width"] == 32
    assert result["architecture"]["key_map"] == "fixed_rff_cosine"
    assert result["gate_native_key_separation"]["margin_passed"] is True
    assert result["gate_native_key_separation"]["zero_event_key_exact_zero"] is True
    assert (
        result["gate_native_key_separation"]["architecture"] == result["architecture"]
    )
    assert result["standard_arc_key_separation"]["margin_passed"] is True
    assert result["standard_arc_key_separation"]["zero_event_key_exact_zero"] is True
    assert result["data"]["marginals"]["exact_marginal_equality"] is True
    assert set(result["evaluation"]["depths"]) == {"0", "1"}
    for depth in result["evaluation"]["depths"].values():
        for arm in ("intact", "shuffled", "no_context"):
            assert depth[arm]["count"] == config.validation_episodes
    memory = result["diagnostics"]["memory"]
    assert memory["every_intact_shuffled_pair_differs"] is True
    assert memory["no_context_exact_zero"] is True
    assert result["qualification"]["passed"] is False
    assert result["interpretation"] == (
        "nonqualifying_abbreviated_no_capability_conclusion"
    )


def test_artifact_writer_emits_strict_json(tmp_path: Path) -> None:
    destination = gate.write_artifact(
        {"finite": 1.0, "nonfinite": float("nan"), "array": np.asarray([1, 2])},
        tmp_path / "gate.json",
    )
    parsed = json.loads(destination.read_text(encoding="utf-8"))

    assert parsed == {"array": [1, 2], "finite": 1.0, "nonfinite": None}
    assert not (tmp_path / "gate.json.tmp").exists()


def test_cli_defaults_are_production_and_smoke_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    defaults = gate._parser().parse_args([])
    assert defaults.training_updates == 10_000
    assert defaults.neuron_count == 2048
    assert defaults.recurrent_edges == 16_384
    assert defaults.context_memory_width == 32
    assert defaults.memory_decay == 1.0
    assert defaults.smoke is False

    captured: list[gate.BindingGateConfig] = []

    def run(config: gate.BindingGateConfig) -> dict[str, object]:
        captured.append(config)
        return {"qualification": {"passed": False}}

    destination = tmp_path / "smoke.json"
    monkeypatch.setattr(gate, "run_binding_gate", run)
    monkeypatch.setattr(gate, "write_artifact", lambda result, path: Path(path))

    assert gate.main(["--smoke", "--output", str(destination)]) == 0
    assert captured == [gate.BindingGateConfig.smoke_config()]
