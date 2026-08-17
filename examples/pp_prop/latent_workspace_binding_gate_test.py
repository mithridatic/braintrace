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


_VALID_HEAD = "a" * 40


def _accuracy(accuracy: float) -> dict[str, object]:
    correct = round(accuracy * 512)
    measured = correct / 512
    lower, upper = gate.legacy._wilson_interval(correct, 512)
    return {
        "accuracy": measured,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "count": 512,
        "correct": correct,
        "prediction_histogram": [52, 52, 51, 51, 51, 51, 51, 51, 51, 51],
        "prediction_sha256": "d" * 64,
    }


def _passing_evaluation() -> dict[str, object]:
    intact = _accuracy(0.90)
    shuffled = _accuracy(0.20)
    no_context = _accuracy(0.10)
    return {
        "intact": intact,
        "shuffled": shuffled,
        "no_context": no_context,
        "intact_minus_shuffled": intact["accuracy"] - shuffled["accuracy"],
        "intact_minus_no_context": intact["accuracy"] - no_context["accuracy"],
        "pairing_chance": 0.25,
        "unconditional_color_chance": 0.10,
        "reported_checkpoint": 1,
        "supervised_depths": [0, 1],
        "all_compact_logits_finite": True,
        "depths": {
            "0": {"intact": intact, "shuffled": shuffled, "no_context": no_context},
            "1": {"intact": intact, "shuffled": shuffled, "no_context": no_context},
        },
    }


def _passing_diagnostics() -> dict[str, object]:
    return {
        "all_state_tensors_finite": True,
        "memory": {
            "intact_shuffled_different_count": 512,
            "different_count": 512,
            "applicable_count": 512,
            "every_intact_shuffled_pair_differs": True,
            "every_pair_differs": True,
            "no_context_exact_zero": True,
            "intact_l2_norm": 3.0,
            "shuffled_l2_norm": 3.0,
            "no_context_l2_norm": 0.0,
        },
        "read_by_depth": {
            "0": {
                "applicable_count": 512,
                "mean_l2_difference": 1.0,
                "different_count": 512,
                "every_pair_differs": True,
                "no_context_l2_norm": 0.0,
            },
            "1": {
                "applicable_count": 512,
                "mean_l2_difference": 1.2,
                "different_count": 512,
                "every_pair_differs": True,
                "no_context_l2_norm": 0.0,
            },
        },
        "workspace_by_depth": {
            "0": {
                "applicable_count": 512,
                "mean_l2_difference": 0.5,
                "different_count": 512,
                "every_pair_differs": True,
                "no_context_l2_norm": 0.0,
            },
            "1": {
                "applicable_count": 512,
                "mean_l2_difference": 0.7,
                "different_count": 512,
                "every_pair_differs": True,
                "no_context_l2_norm": 0.0,
            },
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
            "memory_write_scale": [
                {
                    "relation_key": "weight",
                    "classification": "all_direct",
                    "hidden_groups": [
                        {"index": 1, "hidden_paths": ["context_memory"]}
                    ],
                }
            ],
            "workspace_query_projection/weight": [
                {
                    "relation_key": "weight",
                    "classification": "all_direct",
                    "hidden_groups": [
                        {"index": 3, "hidden_paths": ["reasoning_query"]}
                    ],
                }
            ],
            "memory_read_projection/weight": [
                {
                    "relation_key": "weight",
                    "classification": "all_direct",
                    "hidden_groups": [
                        {
                            "index": 0,
                            "hidden_paths": ["ff_syn/post/V", "workspace_carrier"],
                        }
                    ],
                }
            ],
        },
        "hidden_groups": [
            {"index": 0, "hidden_paths": ["ff_syn/post/V", "workspace_carrier"]},
            {"index": 1, "hidden_paths": ["context_memory"]},
            {"index": 2, "hidden_paths": ["query_encoding"]},
            {"index": 3, "hidden_paths": ["reasoning_query"]},
        ],
        "context_memory_isolated_from_workspace_lif": True,
    }


def _passing_training() -> dict[str, object]:
    losses = [1.0] * 9_936 + [0.5] * 64
    return {
        "algorithm": "production_pp_prop",
        "losses": losses,
        "initial_loss": 1.0,
        "final_loss": 0.5,
        "tail_64_mean_loss": 0.5,
        "supervision_weight_sum": 1.0,
        "supervision_mask": [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        "supervised_depths": [0, 1],
        "required_direct_parameter_movement": {
            path: {"l2_delta": 0.1, "parameter_count": 1, "changed": True}
            for path in _passing_compiler()["required_direct_paths"]
        }
    }


def _passing_environment() -> dict[str, object]:
    return {
        "backend": "gpu",
        "image_digest": "sha256:" + "a" * 64,
        "devices": [
            {"id": 0, "platform": "gpu", "device_kind": "test GPU"}
        ],
    }


def _passing_source() -> dict[str, object]:
    return {
        "commit": _VALID_HEAD,
        "asserted_commit": _VALID_HEAD,
        "asserted_commit_matches_head": True,
        "commit_is_valid_40_hex": True,
        "verified": True,
        "dirty": False,
        "asserted_dirty": False,
        "asserted_dirty_matches_worktree": True,
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
    gram = np.full((10, 10), 0.497, dtype=np.float64)
    np.fill_diagonal(gram, 1.0)
    return {
        "protocol": (
            "standard_arc_k10_query_key_gram_424_features"
            if portable
            else "gate_native_k10_query_key_gram_18_features"
        ),
        "row_event_input_width": 830 if portable else 41,
        "raw_key_feature_width": 424 if portable else 18,
        "memory_width": 32,
        "model_seed": 2108,
        "candidate_colors": list(range(10)),
        "color_count": 10,
        "gram_shape": [10, 10],
        "gram": gram.tolist(),
        "diagonal": np.diag(gram).tolist(),
        "diagonal_minimum": 1.0,
        "off_diagonal_maximum": 0.497,
        "separation_margin": 0.503,
        "worst_global_margin": 0.503,
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
        dataclasses.replace(config, batch_size=9)


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


def test_live_width32_standard_arc_key_gate_covers_all_catalog_colors() -> None:
    report = gate._standard_arc_key_separation_report()

    assert report["protocol"] == "standard_arc_k10_query_key_gram_424_features"
    assert report["row_event_input_width"] == 830
    assert report["raw_key_feature_width"] == 424
    assert report["memory_width"] == 32
    assert report["candidate_colors"] == list(range(10))
    assert report["color_count"] == 10
    assert report["gram_shape"] == [10, 10]
    assert len(report["diagonal"]) == 10
    np.testing.assert_allclose(
        report["diagonal"][:4],
        [0.7478269935, 0.8287837505, 0.9190308452, 0.8519610763],
        rtol=0.0,
        atol=5e-4,
    )
    assert report["worst_global_margin"] == report["separation_margin"]
    assert report["separation_margin"] > report["required_margin"]
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

    context_group = SimpleNamespace(index=1, hidden_paths=[("context_memory",)])
    workspace_group = SimpleNamespace(
        index=0,
        hidden_paths=[("ff_syn", "post", "V"), ("workspace_carrier",)],
    )
    reasoning_group = SimpleNamespace(index=3, hidden_paths=[("reasoning_query",)])

    def relation(path: tuple[str, ...], classification: str = "all_direct"):
        hidden_groups = (
            [context_group]
            if path == ("memory_write_scale",)
            else [reasoning_group]
            if path == ("workspace_query_projection", "weight")
            else [workspace_group]
        )
        return SimpleNamespace(
            trainable_paths={"weight": path},
            path_classification={"weight": classification},
            hidden_groups=hidden_groups,
        )

    learner = SimpleNamespace(
        report=SimpleNamespace(diagnostics=[], etrace_weights=[]),
        graph=SimpleNamespace(
            hidden_param_op_relations=[relation(path) for path in required],
            hidden_groups=[workspace_group, context_group, reasoning_group],
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
    assert complete["context_memory_isolated_from_workspace_lif"] is True
    assert complete["hidden_groups"][1] == {
        "index": 1,
        "hidden_paths": ["context_memory"],
    }
    write_evidence = complete["direct_path_evidence"]["memory_write_scale"]
    assert write_evidence[0]["hidden_groups"] == [
        {"index": 1, "hidden_paths": ["context_memory"]}
    ]

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

    merged_context = SimpleNamespace(
        index=1,
        hidden_paths=[("context_memory",), ("workspace_carrier",)],
    )
    learner.graph.hidden_groups[1] = merged_context
    grouped = gate._compiler_report(learner)
    assert grouped["context_memory_isolated_from_workspace_lif"] is False


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
        ("compiler_relation", "compiler_required_paths_all_direct"),
        ("compiler_group", "context_memory_hidden_group_isolated"),
        ("movement", "required_direct_parameters_moved"),
        ("loss_count", "training_losses_complete_and_finite"),
        ("loss_nan", "training_losses_complete_and_finite"),
        ("loss_summary", "training_losses_complete_and_finite"),
        ("supervision_mask", "training_losses_complete_and_finite"),
        ("evaluation_depth", "evaluation_complete_and_finite"),
        ("evaluation_nan", "evaluation_complete_and_finite"),
        ("evaluation_mismatch", "evaluation_complete_and_finite"),
        ("fabricated_wilson", "evaluation_complete_and_finite"),
        ("diagnostic_nan", "diagnostic_state_tensors_complete_and_finite"),
        ("diagnostic_flag", "diagnostic_state_tensors_complete_and_finite"),
        ("read_pairing", "diagnostic_state_tensors_complete_and_finite"),
        ("workspace_pairing", "diagnostic_state_tensors_complete_and_finite"),
        ("backend", "gpu_backend_verified"),
        ("fake_gpu", "gpu_backend_verified"),
        ("image_digest", "gpu_backend_verified"),
        ("architecture", "architecture_matches_preregistered_components"),
        ("architecture_digest", "architecture_matches_preregistered_components"),
        ("native_colors", "gate_native_all_colors_covered"),
        ("native_protocol", "gate_native_all_colors_covered"),
        ("native_separation", "gate_native_key_separation_margin_passed"),
        ("native_zero_key", "gate_native_zero_event_key_exact_zero"),
        ("native_basis", "gate_native_basis_matches_training_model"),
        ("standard_colors", "standard_arc_all_colors_covered"),
        ("standard_protocol", "standard_arc_all_colors_covered"),
        ("standard_separation", "standard_arc_key_separation_margin_passed"),
        ("standard_zero_key", "standard_arc_zero_event_key_exact_zero"),
        ("standard_shared", "standard_arc_shared_encoder_invariants_match"),
        ("source_start", "source_start_verified_clean"),
        ("source_end", "source_end_verified_clean"),
        ("source_assertion", "source_start_verified_clean"),
        ("source_missing_assertion", "source_start_verified_clean"),
        ("source_missing_dirty_assertion", "source_start_verified_clean"),
        ("head_drift", "source_head_stable"),
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
    environment = _passing_environment()
    marginal = {"exact_marginal_equality": True}
    source_start = _passing_source()
    source_end = _passing_source()

    if mutation == "validation_count":
        evaluation["intact"] = _accuracy(0.90) | {"count": 255}
    elif mutation == "intact_accuracy":
        evaluation["intact"] = _accuracy(0.79)
    elif mutation == "intact_wilson":
        evaluation["intact"] = _accuracy(0.90)
        evaluation["intact"]["wilson_95_lower"] = 0.25
    elif mutation == "pairing_gap":
        evaluation["intact_minus_shuffled"] = 0.24
    elif mutation == "shuffled_chance":
        evaluation["shuffled"] = _accuracy(0.40)
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
    elif mutation == "compiler_relation":
        compiler["direct_path_evidence"]["memory_write_scale"][0][
            "hidden_groups"
        ] = [{"index": 3, "hidden_paths": ["reasoning_query"]}]
    elif mutation == "compiler_group":
        compiler["context_memory_isolated_from_workspace_lif"] = False
    elif mutation == "movement":
        training["required_direct_parameter_movement"]["memory_write_scale"] = {
            "l2_delta": 0.0,
            "parameter_count": 1024,
            "changed": False,
        }
    elif mutation == "loss_count":
        training["losses"].pop()
    elif mutation == "loss_nan":
        training["losses"][0] = float("nan")
    elif mutation == "loss_summary":
        training["final_loss"] = 0.25
    elif mutation == "supervision_mask":
        training["supervision_mask"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    elif mutation == "evaluation_depth":
        del evaluation["depths"]["1"]
    elif mutation == "evaluation_nan":
        evaluation["depths"]["0"]["intact"] = dict(
            evaluation["depths"]["0"]["intact"]
        )
        evaluation["depths"]["0"]["intact"]["accuracy"] = float("nan")
    elif mutation == "evaluation_mismatch":
        evaluation["depths"]["1"]["shuffled"] = dict(
            evaluation["depths"]["1"]["shuffled"]
        )
        evaluation["depths"]["1"]["shuffled"]["accuracy"] = 0.5
    elif mutation == "fabricated_wilson":
        fabricated = _accuracy(0.40)
        fabricated["wilson_95_lower"] = 0.20
        evaluation["shuffled"] = fabricated
        for depth in evaluation["depths"].values():
            depth["shuffled"] = fabricated
        evaluation["intact_minus_shuffled"] = (
            evaluation["intact"]["accuracy"] - fabricated["accuracy"]
        )
    elif mutation == "diagnostic_nan":
        diagnostics["read_by_depth"]["0"]["mean_l2_difference"] = float("nan")
    elif mutation == "diagnostic_flag":
        diagnostics["all_state_tensors_finite"] = False
    elif mutation == "read_pairing":
        diagnostics["read_by_depth"]["0"].update(
            different_count=0,
            every_pair_differs=False,
            mean_l2_difference=0.0,
        )
    elif mutation == "workspace_pairing":
        diagnostics["workspace_by_depth"]["1"].update(
            different_count=0,
            every_pair_differs=False,
            mean_l2_difference=0.0,
        )
    elif mutation == "backend":
        environment["backend"] = "cpu"
    elif mutation == "fake_gpu":
        environment["devices"] = [{"id": 0, "platform": "cpu"}]
    elif mutation == "image_digest":
        environment["image_digest"] = "unverified"
    elif mutation == "architecture":
        architecture["read_component_type"] = "unknown"
    elif mutation == "architecture_digest":
        architecture["key_basis_sha256"] = "G" * 64
    elif mutation == "native_colors":
        gate_native["candidate_colors"] = list(range(4))
        gate_native["color_count"] = 4
        gate_native["gram_shape"] = [4, 4]
    elif mutation == "native_protocol":
        gate_native["protocol"] = "standard_arc_k10_query_key_gram_424_features"
    elif mutation == "native_separation":
        gate_native["margin_passed"] = False
    elif mutation == "native_zero_key":
        gate_native["zero_event_key_exact_zero"] = False
    elif mutation == "native_basis":
        gate_native["architecture"] = dict(gate_native["architecture"])
        gate_native["architecture"]["key_basis_sha256"] = "0" * 64
    elif mutation == "standard_colors":
        standard_arc["candidate_colors"] = list(range(4))
        standard_arc["color_count"] = 4
        standard_arc["gram_shape"] = [4, 4]
    elif mutation == "standard_protocol":
        standard_arc["raw_key_feature_width"] = 18
    elif mutation == "standard_separation":
        standard_arc["margin_passed"] = False
    elif mutation == "standard_zero_key":
        standard_arc["zero_event_key_exact_zero"] = False
    elif mutation == "standard_shared":
        standard_arc["architecture"] = dict(standard_arc["architecture"])
        standard_arc["architecture"]["rff_gamma"] = 3.0
    elif mutation == "source_start":
        source_start["dirty"] = True
    elif mutation == "source_end":
        source_end["dirty"] = True
    elif mutation == "source_assertion":
        source_start["asserted_commit_matches_head"] = False
        source_start["verified"] = False
    elif mutation == "source_missing_assertion":
        source_start["asserted_commit"] = None
    elif mutation == "source_missing_dirty_assertion":
        source_start["asserted_dirty"] = None
    elif mutation == "head_drift":
        source_end["commit"] = "b" * 40
        source_end["asserted_commit"] = "b" * 40
    elif mutation == "config":
        config = dataclasses.replace(config, training_updates=9_999)

    report = gate._qualification_report(
        evaluation=evaluation,
        diagnostics=diagnostics,
        compiler=compiler,
        training=training,
        environment=environment,
        architecture=architecture,
        gate_native_separation=gate_native,
        standard_arc_separation=standard_arc,
        marginals=marginal,
        source_start=source_start,
        source_end=source_end,
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
        environment=_passing_environment(),
        architecture=architecture,
        gate_native_separation=_passing_separation(architecture),
        standard_arc_separation=_passing_separation(architecture, portable=True),
        marginals={"exact_marginal_equality": True},
        source_start=_passing_source(),
        source_end=_passing_source(),
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
        environment=_passing_environment(),
        architecture=reduced_architecture,
        gate_native_separation=_passing_separation(reduced_architecture),
        standard_arc_separation=_passing_separation(
            reduced_architecture, portable=True
        ),
        marginals={"exact_marginal_equality": True},
        source_start=_passing_source(),
        source_end=_passing_source(),
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
        environment={},
        architecture={},
        gate_native_separation={},
        standard_arc_separation={},
        marginals={},
        source_start={},
        source_end={},
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

    report = gate._source_report()
    assert report["commit"] == "unavailable"
    assert report["commit_is_valid_40_hex"] is False
    assert report["verified"] is False
    assert report["dirty"] is True


def test_source_report_verifies_actual_head_and_rejects_arbitrary_env_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=_VALID_HEAD + "\n")
        if command[1:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr(gate.subprocess, "run", git)
    monkeypatch.delenv("BRAINTRACE_SOURCE_COMMIT", raising=False)
    monkeypatch.delenv("BRAINTRACE_SOURCE_DIRTY", raising=False)

    unasserted = gate._source_report()
    assert unasserted["commit"] == _VALID_HEAD
    assert unasserted["commit_is_valid_40_hex"] is True
    assert unasserted["verified"] is False

    monkeypatch.setenv("BRAINTRACE_SOURCE_COMMIT", "not-a-commit")
    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "false")

    mismatched = gate._source_report()
    assert mismatched["commit"] == _VALID_HEAD
    assert mismatched["asserted_commit"] == "not-a-commit"
    assert mismatched["asserted_commit_matches_head"] is False
    assert mismatched["commit_is_valid_40_hex"] is True
    assert mismatched["verified"] is False
    assert mismatched["dirty"] is False

    monkeypatch.setenv("BRAINTRACE_SOURCE_COMMIT", _VALID_HEAD)
    matched = gate._source_report()
    assert matched["asserted_commit_matches_head"] is True
    assert matched["verified"] is True

    monkeypatch.setenv("BRAINTRACE_SOURCE_DIRTY", "true")
    dirty_assertion_mismatch = gate._source_report()
    assert dirty_assertion_mismatch["dirty"] is False
    assert dirty_assertion_mismatch["asserted_dirty"] is True
    assert dirty_assertion_mismatch["asserted_dirty_matches_worktree"] is False
    assert dirty_assertion_mismatch["verified"] is False


def test_actual_model_pp_prop_smoke_records_routes_depths_and_state_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINTRACE_IMAGE_DIGEST", "sha256:" + "f" * 64)
    config = gate.BindingGateConfig.smoke_config()
    result = gate.run_binding_gate(config)

    assert result["schema_version"] == 2
    assert result["learner"] == "pp_prop_only"
    assert result["config"]["configuration_scale"] == "reduced_smoke"
    assert result["config"]["qualification_regime"] == ("nonqualifying_abbreviated")
    assert result["environment"]["backend"] == "gpu"
    assert any(
        device["platform"] == "gpu" for device in result["environment"]["devices"]
    )
    assert result["source_end"]["commit"] == result["source"]["commit"]
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
    assert compiler["context_memory_isolated_from_workspace_lif"] is True
    assert set(training["required_direct_parameter_movement"]) == set(
        compiler["required_direct_paths"]
    )
    assert result["architecture"]["memory_width"] == 32
    assert result["architecture"]["key_map"] == "fixed_rff_cosine"
    assert result["gate_native_key_separation"]["candidate_colors"] == list(
        range(10)
    )
    assert result["gate_native_key_separation"]["margin_passed"] is True
    assert result["gate_native_key_separation"]["zero_event_key_exact_zero"] is True
    assert (
        result["gate_native_key_separation"]["architecture"] == result["architecture"]
    )
    assert result["standard_arc_key_separation"]["margin_passed"] is True
    assert result["standard_arc_key_separation"]["candidate_colors"] == list(
        range(10)
    )
    assert result["standard_arc_key_separation"]["zero_event_key_exact_zero"] is True
    assert result["data"]["marginals"]["exact_marginal_equality"] is True
    assert set(result["evaluation"]["depths"]) == {"0", "1"}
    for depth in result["evaluation"]["depths"].values():
        for arm in ("intact", "shuffled", "no_context"):
            assert depth[arm]["count"] == config.validation_episodes
    memory = result["diagnostics"]["memory"]
    assert result["diagnostics"]["all_state_tensors_finite"] is True
    assert memory["every_intact_shuffled_pair_differs"] is True
    assert memory["no_context_exact_zero"] is True
    assert result["qualification"]["passed"] is False
    for criterion in (
        "compiler_required_paths_all_direct",
        "context_memory_hidden_group_isolated",
        "required_direct_parameters_moved",
        "training_losses_complete_and_finite",
        "evaluation_complete_and_finite",
        "diagnostic_state_tensors_complete_and_finite",
        "gpu_backend_verified",
        "architecture_matches_preregistered_components",
        "gate_native_all_colors_covered",
        "standard_arc_all_colors_covered",
    ):
        assert result["qualification"]["criteria"][criterion] is True
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
