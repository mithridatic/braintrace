"""Tests for the post-architecture Example 21 binding gate."""

from __future__ import annotations

import ast
import copy
import dataclasses
import msgspec_json
from collections import namedtuple
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
        "head_command_succeeded": True,
        "verified": True,
        "dirty": False,
        "asserted_dirty": False,
        "asserted_dirty_matches_worktree": True,
        "status_command_succeeded": True,
    }


def _passing_formal_initialization() -> dict[str, object]:
    return {
        "fresh_model": True,
        "model_seed": 2108,
        "parameter_sha256": gate.PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256,
        "parameter_count": gate.PREREGISTERED_PARAMETER_COUNT,
    }


def _passing_formal_data() -> dict[str, object]:
    return {
        "training_schedule_sha256": gate.PREREGISTERED_TRAINING_SCHEDULE_SHA256,
        "validation_schedule_sha256": gate.PREREGISTERED_STABILITY_DIGESTS[
            "validation_schedule_sha256"
        ],
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
        "carrier_stabilizer": "per_example_stopped_unit_l2_cap",
        "carrier_radius": 1.0,
        "carrier_consumers": [
            "readout_projection",
            "workspace_query_projection",
        ],
    }


def _passing_projection_measurement() -> dict[str, object]:
    return {
        "projection_telemetry": {
            name: [
                {"rms": 0.1, "max_abs": 0.2, "nonzero_fraction": 1.0},
                {"rms": 0.1, "max_abs": 0.2, "nonzero_fraction": 1.0},
            ]
            for name in (
                "readout_preactivation",
                "readout_post_gelu",
                "row_factors",
                "column_factors",
                "color_factors",
            )
        },
        "compact_reconciliation_max_abs": [0.0, 0.0],
        "consumer_witnesses": {
            "readout_capped_residual_max_abs": 0.0,
            "readout_uncapped_delta_min_l2": 0.1,
            "query_capped_residual_max_l2": 0.0,
            "query_uncapped_delta_min_l2": 0.1,
            "sample_count": 192,
        },
    }


def _passing_one_update_admission() -> dict[str, object]:
    return {
        "schema_version": 1,
        "control": "example21_stage21_one_update_admission",
        "target": "one_update",
        "executed_updates": 1,
        "source_training_updates": 10_000,
        "batch_size": 64,
        "configuration_scale": "production_topology",
        "learner": "pp_prop_only",
        "optimizer": "Adam",
        "config": {
            **dataclasses.asdict(gate.BindingGateConfig.stage21_one_update_config()),
            "configuration_scale": "production_topology",
        },
        "data": {
            "training_schedule_sha256": (
                "25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b"
            ),
            **gate.PREREGISTERED_UPDATE_ZERO_DIGESTS,
            "update_zero_episode_count": 64,
            "rng_source_episode_count": 640_000,
        },
        "architecture": _passing_architecture(),
        "initialization": {
            "fresh_model": True,
            "model_seed": 2108,
            "parameter_sha256": gate.PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256,
            "parameter_count": gate.PREREGISTERED_PARAMETER_COUNT,
        },
        "depths": {
            "0": {
                "pre_cross_entropy": 2.3,
                "post_cross_entropy": 2.5,
                "pre_max_abs_color_logit": 0.8,
                "post_max_abs_color_logit": 1.2,
            },
            "1": {
                "pre_cross_entropy": 2.4,
                "post_cross_entropy": 2.6,
                "pre_max_abs_color_logit": 1.1,
                "post_max_abs_color_logit": 1.4,
            },
        },
        "carrier": {
            "pre": {
                "sample_count": 128,
                "raw_max_l2_norm": 5.0,
                "capped_max_l2_norm": 1.0,
                "capped_count": 64,
                "capped_fraction": 0.5,
            },
            "post": {
                "sample_count": 128,
                "raw_max_l2_norm": 4.0,
                "capped_max_l2_norm": 1.0,
                "capped_count": 64,
                "capped_fraction": 0.5,
            },
        },
        "pre_measurement": _passing_projection_measurement(),
        "post_measurement": _passing_projection_measurement(),
        "gradient_group_norms": {
            "memory_write_scale": {"l2_norm": 0.1, "parameter_count": 1_024},
            "workspace_query_projection/weight": {
                "l2_norm": 0.2,
                "parameter_count": 65_536,
            },
            "memory_read_projection/weight": {
                "l2_norm": 0.3,
                "parameter_count": 65_536,
            },
            "readout_projection/weight": {
                "l2_norm": 0.4,
                "parameter_count": 262_272,
            },
            "color_factor_head/weight": {
                "l2_norm": 0.5,
                "parameter_count": 144_480,
            },
        },
        "pp_prop_factor_group_norms": {
            "memory_write_scale": {"l2_norm": 0.1, "factor_count": 65_536},
            "workspace_query_projection/weight": {
                "l2_norm": 0.2,
                "factor_count": 133_120,
            },
            "memory_read_projection/weight": {
                "l2_norm": 0.3,
                "factor_count": 526_336,
            },
        },
        "adam_factor_group_norms": {
            path: {
                "first_moment_l2_norm": 0.1,
                "second_moment_l2_norm": 0.01,
                "first_moment_count": count,
                "second_moment_count": count,
                "adam_step": 1,
                "schedule_step": 1,
                "optimizer_step": 1,
            }
            for path, count in {
                "memory_write_scale": 1_024,
                "workspace_query_projection/weight": 65_536,
                "memory_read_projection/weight": 65_536,
                "readout_projection/weight": 262_272,
                "color_factor_head/weight": 144_480,
            }.items()
        },
        "parameter_update_group_norms": {
            path: {"l2_norm": 0.1, "parameter_count": count}
            for path, count in gate._STAGE21_PARAMETER_COUNTS.items()
        },
        "finite_telemetry": {
            "cross_entropies": True,
            "color_logits": True,
            "raw_carriers": True,
            "capped_carriers": True,
            "gradients": True,
            "pp_prop_factors": True,
            "adam_factors": True,
            "parameter_updates": True,
            "decoder_factors": True,
        },
    }


def _passing_stability_admission() -> dict[str, object]:
    losses = [2.4] * 192 + [2.0] * 64
    return {
        "schema_version": 1,
        "control": "example21_stage21_stability_256_admission",
        "target": "stability_256",
        "training_updates": 256,
        "batch_size": 64,
        "validation_episodes": 512,
        "configuration_scale": "production_topology",
        "qualification_regime": "nonqualifying_abbreviated",
        "learner": "pp_prop_only",
        "optimizer": "Adam",
        "config": {
            **dataclasses.asdict(gate.BindingGateConfig.stage21_stability_config()),
            "configuration_scale": "production_topology",
            "qualification_regime": "nonqualifying_abbreviated",
        },
        "data": dict(gate.PREREGISTERED_STABILITY_DIGESTS),
        "architecture": _passing_architecture(),
        "initialization": {
            "fresh_model": True,
            "model_seed": 2108,
            "parameter_sha256": gate.PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256,
            "parameter_count": gate.PREREGISTERED_PARAMETER_COUNT,
            "update_zero_event_sha256": gate.PREREGISTERED_UPDATE_ZERO_DIGESTS[
                "update_zero_event_sha256"
            ],
            "update_zero_target_sha256": gate.PREREGISTERED_UPDATE_ZERO_DIGESTS[
                "update_zero_target_sha256"
            ],
        },
        "losses": losses,
        "initial_depth_cross_entropy": {"0": 2.3, "1": 2.5},
        "tail_64_mean_loss": 2.0,
        "held_out_intact_by_depth": {
            "0": {
                "count": 512,
                "prediction_histogram": [256, 256, 0, 0, 0, 0, 0, 0, 0, 0],
                "unique_predicted_colors": 2,
            },
            "1": {
                "count": 512,
                "prediction_histogram": [128, 384, 0, 0, 0, 0, 0, 0, 0, 0],
                "unique_predicted_colors": 2,
            },
        },
        "finite_telemetry": {
            "losses": True,
            "states": True,
            "logits": True,
            "gradients": True,
            "pp_prop_factors": True,
            "adam_factors": True,
            "parameters": True,
        },
        "telemetry_summaries": {
            name: {
                "observed_count": count,
                "max_abs": 1.0,
                "finite": True,
            }
            for name, count in {
                "losses": 256,
                "states": 256,
                "logits": 256,
                "gradients": 256 * 5,
                "pp_prop_factors": 256 * 3,
                "adam_factors": 256,
                "parameters": 256,
            }.items()
        },
        "evaluation_all_compact_logits_finite": True,
        "evaluation_all_state_tensors_finite": True,
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
        dataclasses.replace(config, context_memory_width=513)
    assert (
        dataclasses.replace(config, context_memory_width=512).context_memory_width
        == 512
    )
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


@pytest.mark.parametrize("correct", [0, 512])
def test_accuracy_evidence_accepts_exact_binomial_boundaries(correct: int) -> None:
    metric = _accuracy(correct / 512)

    assert gate._accuracy_evidence_complete(metric, 512) is True


def test_accuracy_evidence_rejects_fabricated_boundary_intervals() -> None:
    perfect = _accuracy(1.0)
    perfect["wilson_95_upper"] = float(
        np.nextafter(np.float64(perfect["wilson_95_upper"]), -np.inf)
    )
    assert gate._accuracy_evidence_complete(perfect, 512) is False

    zero = _accuracy(0.0)
    zero["wilson_95_lower"] = 0.1
    assert gate._accuracy_evidence_complete(zero, 512) is False


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
        ("movement_bool", "required_direct_parameters_moved"),
        ("loss_count", "training_losses_complete_and_finite"),
        ("loss_nan", "training_losses_complete_and_finite"),
        ("loss_bool_array", "training_losses_complete_and_finite"),
        ("loss_summary", "training_losses_complete_and_finite"),
        ("supervision_mask", "training_losses_complete_and_finite"),
        ("supervision_weight_bool", "training_losses_complete_and_finite"),
        ("evaluation_depth", "evaluation_complete_and_finite"),
        ("reported_checkpoint_bool", "evaluation_complete_and_finite"),
        ("accuracy_bool", "evaluation_complete_and_finite"),
        ("evaluation_nan", "evaluation_complete_and_finite"),
        ("evaluation_mismatch", "evaluation_complete_and_finite"),
        ("fabricated_wilson", "evaluation_complete_and_finite"),
        ("diagnostic_nan", "diagnostic_state_tensors_complete_and_finite"),
        ("diagnostic_flag", "diagnostic_state_tensors_complete_and_finite"),
        ("diagnostic_mean_bool", "diagnostic_state_tensors_complete_and_finite"),
        ("diagnostic_norm_bool", "diagnostic_state_tensors_complete_and_finite"),
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
        ("native_bool_gram", "gate_native_all_colors_covered"),
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
            ("source_head_command", "source_start_verified_clean"),
            ("source_status_command", "source_end_verified_clean"),
            ("head_drift", "source_head_stable"),
            ("formal_initialization", "formal_initialization_matches_admissions"),
            ("formal_schedule", "formal_schedule_matches_admissions"),
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
    initialization = _passing_formal_initialization()
    data = _passing_formal_data()

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
    elif mutation == "movement_bool":
        training["required_direct_parameter_movement"]["memory_write_scale"].update(
            l2_delta=True,
            parameter_count=True,
        )
    elif mutation == "loss_count":
        training["losses"].pop()
    elif mutation == "loss_nan":
        training["losses"][0] = float("nan")
    elif mutation == "loss_bool_array":
        training["losses"] = [True] * 9_936 + [False] * 64
        training["initial_loss"] = 1.0
        training["final_loss"] = 0.0
        training["tail_64_mean_loss"] = 0.0
    elif mutation == "loss_summary":
        training["final_loss"] = 0.25
    elif mutation == "supervision_mask":
        training["supervision_mask"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    elif mutation == "supervision_weight_bool":
        training["supervision_weight_sum"] = True
    elif mutation == "evaluation_depth":
        del evaluation["depths"]["1"]
    elif mutation == "reported_checkpoint_bool":
        evaluation["reported_checkpoint"] = True
    elif mutation == "accuracy_bool":
        perfect = _accuracy(1.0)
        perfect["accuracy"] = True
        perfect["wilson_95_upper"] = True
        evaluation["intact"] = perfect
        for depth in evaluation["depths"].values():
            depth["intact"] = perfect
        evaluation["intact_minus_shuffled"] = (
            1.0 - float(evaluation["shuffled"]["accuracy"])
        )
        evaluation["intact_minus_no_context"] = (
            1.0 - float(evaluation["no_context"]["accuracy"])
        )
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
    elif mutation == "diagnostic_mean_bool":
        diagnostics["read_by_depth"]["0"]["mean_l2_difference"] = True
    elif mutation == "diagnostic_norm_bool":
        diagnostics["memory"]["intact_l2_norm"] = True
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
    elif mutation == "native_bool_gram":
        gate_native.update(
            gram=np.eye(10, dtype=bool).tolist(),
            diagonal=[True] * 10,
            diagonal_minimum=True,
            off_diagonal_maximum=False,
            separation_margin=True,
            worst_global_margin=True,
        )
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
    elif mutation == "source_head_command":
        source_start["head_command_succeeded"] = False
    elif mutation == "source_status_command":
        source_end["status_command_succeeded"] = False
    elif mutation == "head_drift":
        source_end["commit"] = "b" * 40
        source_end["asserted_commit"] = "b" * 40
    elif mutation == "formal_initialization":
        initialization["parameter_sha256"] = "0" * 64
    elif mutation == "formal_schedule":
        data["validation_schedule_sha256"] = "0" * 64
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
        one_update_admission=_passing_one_update_admission(),
        stability_admission=_passing_stability_admission(),
        initialization=initialization,
        data=data,
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
        one_update_admission=_passing_one_update_admission(),
        stability_admission=_passing_stability_admission(),
        initialization=_passing_formal_initialization(),
        data=_passing_formal_data(),
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
        one_update_admission=_passing_one_update_admission(),
        stability_admission=_passing_stability_admission(),
        initialization=_passing_formal_initialization(),
        data=_passing_formal_data(),
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
        one_update_admission={},
        stability_admission={},
        initialization={},
        data={},
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
        raise OSError("Git unavailable. Update the fixture or expected result to satisfy this assertion.")

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

    assert result["schema_version"] == 3
    assert result["learner"] == "pp_prop_only"
    assert result["config"]["configuration_scale"] == "reduced_smoke"
    assert result["config"]["qualification_regime"] == ("nonqualifying_abbreviated")
    assert result["environment"]["backend"] == "cpu"
    assert all(
        device["platform"] == "cpu" for device in result["environment"]["devices"]
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
        "architecture_matches_preregistered_components",
        "gate_native_all_colors_covered",
        "standard_arc_all_colors_covered",
    ):
        assert result["qualification"]["criteria"][criterion] is True, criterion
    assert result["qualification"]["criteria"]["gpu_backend_verified"] is False
    assert result["interpretation"] == (
        "nonqualifying_abbreviated_no_capability_conclusion"
    )


def test_artifact_writer_emits_strict_json(tmp_path: Path) -> None:
    destination = gate.write_artifact(
        {"finite": 1.0, "nonfinite": float("nan"), "array": np.asarray([1, 2])},
        tmp_path / "gate.json",
    )
    parsed = msgspec_json.loads(destination.read_text(encoding="utf-8"))

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


def test_stage21_admission_configs_are_fixed_and_nonqualifying() -> None:
    one_update = gate.BindingGateConfig.stage21_one_update_config()
    stability = gate.BindingGateConfig.stage21_stability_config()

    assert one_update.training_updates == 10_000
    assert one_update.batch_size == 64
    assert one_update.validation_episodes == 512
    assert one_update.configuration_scale == "production_topology"
    assert one_update.qualification_regime == "preregistered_full"

    assert stability.training_updates == 256
    assert stability.batch_size == 64
    assert stability.validation_episodes == 512
    assert stability.configuration_scale == "production_topology"
    assert stability.context_memory_width == 32
    assert stability.memory_decay == 1.0
    assert stability.qualification_regime == "nonqualifying_abbreviated"


@pytest.mark.parametrize(
    "mutation,criterion",
    [
        ("schema", "schema_and_target"),
        ("schema_bool", "schema_and_target"),
        ("executed_bool", "fixed_production_configuration"),
        ("config_gap_bool", "fixed_production_configuration"),
        ("config_gradient_chunk_bool", "fixed_production_configuration"),
        ("config_memory_decay_bool", "fixed_production_configuration"),
        ("schedule", "exact_production_rng_prefix"),
        ("rng_count", "exact_production_rng_prefix"),
        ("architecture", "carrier_architecture_identity"),
        ("carrier_radius_bool", "carrier_architecture_identity"),
        ("consumer", "carrier_architecture_identity"),
        ("pre_nan", "finite_telemetry"),
        ("post_ce_bool", "finite_telemetry"),
        ("post_ce", "post_cross_entropy_envelope"),
        ("post_logit_bool", "finite_telemetry"),
        ("post_logit", "post_logit_envelope"),
        ("carrier_count", "carrier_cap_nonvacuous"),
        ("carrier_fraction_bool", "finite_telemetry"),
        ("carrier_raw_bool", "finite_telemetry"),
        ("carrier_capped_bool", "finite_telemetry"),
        ("carrier_not_engaged", "carrier_cap_nonvacuous"),
        ("carrier_above_cap", "carrier_norm_envelope"),
        ("gradient_zero", "required_gradient_groups_nonzero"),
        ("gradient_norm_bool", "required_gradient_groups_nonzero"),
        ("gradient_count", "required_gradient_groups_nonzero"),
        ("factor_missing", "required_pp_prop_factors_nonzero"),
        ("factor_count", "required_pp_prop_factors_nonzero"),
        ("adam_zero", "required_adam_factors_nonzero"),
        ("adam_first_norm_bool", "required_adam_factors_nonzero"),
        ("adam_second_norm_bool", "required_adam_factors_nonzero"),
        ("adam_key", "required_adam_factors_nonzero"),
        ("adam_count", "required_adam_factors_nonzero"),
        ("adam_step", "required_adam_factors_nonzero"),
        ("adam_step_bool", "required_adam_factors_nonzero"),
        ("schedule_step", "required_adam_factors_nonzero"),
        ("schedule_step_bool", "required_adam_factors_nonzero"),
        ("optimizer_step", "required_adam_factors_nonzero"),
        ("optimizer_step_bool", "required_adam_factors_nonzero"),
        ("update_zero", "required_parameter_updates_nonzero"),
        ("update_count", "required_parameter_updates_nonzero"),
        ("decoder_zero", "decoder_carrier_signal_finite_nonzero"),
        ("decoder_reconciliation_bool", "decoder_carrier_signal_finite_nonzero"),
        ("decoder_residual_bool", "decoder_carrier_signal_finite_nonzero"),
        ("decoder_delta_bool", "decoder_carrier_signal_finite_nonzero"),
        ("decoder_rms_bool", "decoder_carrier_signal_finite_nonzero"),
        ("decoder_max_bool", "decoder_carrier_signal_finite_nonzero"),
        ("decoder_fraction_bool", "decoder_carrier_signal_finite_nonzero"),
    ],
)
def test_one_update_admission_fails_closed(
    mutation: str, criterion: str
) -> None:
    report = _passing_one_update_admission()
    if mutation == "schema":
        report["schema_version"] = 0
    elif mutation == "schema_bool":
        report["schema_version"] = True
    elif mutation == "executed_bool":
        report["executed_updates"] = True
    elif mutation == "config_gap_bool":
        report["config"]["gap_steps"] = True
    elif mutation == "config_gradient_chunk_bool":
        report["config"]["gradient_chunk_size"] = True
    elif mutation == "config_memory_decay_bool":
        report["config"]["memory_decay"] = True
    elif mutation == "schedule":
        report["data"]["training_schedule_sha256"] = "0" * 64
    elif mutation == "rng_count":
        report["data"]["rng_source_episode_count"] = 64
    elif mutation == "architecture":
        report["architecture"]["carrier_radius"] = 2.0
    elif mutation == "carrier_radius_bool":
        report["architecture"]["carrier_radius"] = True
    elif mutation == "consumer":
        report["architecture"]["carrier_consumers"] = ["readout_projection"]
    elif mutation == "pre_nan":
        report["depths"]["0"]["pre_cross_entropy"] = float("nan")
    elif mutation == "post_ce_bool":
        report["depths"]["0"]["post_cross_entropy"] = True
    elif mutation == "post_ce":
        report["depths"]["1"]["post_cross_entropy"] = 3.6
    elif mutation == "post_logit":
        report["depths"]["0"]["post_max_abs_color_logit"] = 10.0
    elif mutation == "post_logit_bool":
        report["depths"]["0"]["post_max_abs_color_logit"] = True
    elif mutation == "carrier_count":
        report["carrier"]["pre"]["sample_count"] = 0
    elif mutation == "carrier_fraction_bool":
        report["carrier"]["pre"]["capped_count"] = 128
        report["carrier"]["pre"]["capped_fraction"] = True
    elif mutation == "carrier_raw_bool":
        report["carrier"]["pre"]["raw_max_l2_norm"] = True
    elif mutation == "carrier_capped_bool":
        report["carrier"]["pre"]["capped_max_l2_norm"] = True
    elif mutation == "carrier_not_engaged":
        report["carrier"]["pre"]["raw_max_l2_norm"] = 1.0
        report["carrier"]["post"]["raw_max_l2_norm"] = 1.0
        report["carrier"]["pre"]["capped_count"] = 0
        report["carrier"]["post"]["capped_count"] = 0
    elif mutation == "carrier_above_cap":
        report["carrier"]["post"]["capped_max_l2_norm"] = 1.0001
    elif mutation == "gradient_zero":
        report["gradient_group_norms"]["readout_projection/weight"][
            "l2_norm"
        ] = 0.0
    elif mutation == "gradient_norm_bool":
        report["gradient_group_norms"]["memory_write_scale"]["l2_norm"] = True
    elif mutation == "gradient_count":
        report["gradient_group_norms"]["memory_write_scale"][
            "parameter_count"
        ] = 1_023
    elif mutation == "factor_missing":
        del report["pp_prop_factor_group_norms"]["memory_write_scale"]
    elif mutation == "factor_count":
        report["pp_prop_factor_group_norms"]["memory_read_projection/weight"][
            "factor_count"
        ] = 526_335
    elif mutation == "adam_zero":
        report["adam_factor_group_norms"]["color_factor_head/weight"][
            "second_moment_l2_norm"
        ] = 0.0
    elif mutation == "adam_first_norm_bool":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "first_moment_l2_norm"
        ] = True
    elif mutation == "adam_second_norm_bool":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "second_moment_l2_norm"
        ] = True
    elif mutation == "adam_key":
        report["adam_factor_group_norms"]["unexpected"] = (
            report["adam_factor_group_norms"].pop("memory_write_scale")
        )
    elif mutation == "adam_count":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "first_moment_count"
        ] = 1_023
    elif mutation == "adam_step":
        report["adam_factor_group_norms"]["memory_write_scale"]["adam_step"] = 0
    elif mutation == "adam_step_bool":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "adam_step"
        ] = True
    elif mutation == "schedule_step":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "schedule_step"
        ] = 0
    elif mutation == "schedule_step_bool":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "schedule_step"
        ] = True
    elif mutation == "optimizer_step":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "optimizer_step"
        ] = 0
    elif mutation == "optimizer_step_bool":
        report["adam_factor_group_norms"]["memory_write_scale"][
            "optimizer_step"
        ] = True
    elif mutation == "update_zero":
        report["parameter_update_group_norms"]["memory_write_scale"][
            "l2_norm"
        ] = 0.0
    elif mutation == "update_count":
        report["parameter_update_group_norms"]["color_factor_head/weight"][
            "parameter_count"
        ] = 144_479
    elif mutation == "decoder_zero":
        report["post_measurement"]["projection_telemetry"]["color_factors"][0][
            "rms"
        ] = 0.0
    elif mutation == "decoder_reconciliation_bool":
        report["post_measurement"]["compact_reconciliation_max_abs"] = [
            False,
            False,
        ]
    elif mutation == "decoder_residual_bool":
        report["post_measurement"]["consumer_witnesses"][
            "readout_capped_residual_max_abs"
        ] = False
    elif mutation == "decoder_delta_bool":
        report["post_measurement"]["consumer_witnesses"][
            "readout_uncapped_delta_min_l2"
        ] = True
    elif mutation == "decoder_rms_bool":
        report["post_measurement"]["projection_telemetry"]["color_factors"][0][
            "rms"
        ] = True
    elif mutation == "decoder_max_bool":
        report["post_measurement"]["projection_telemetry"]["color_factors"][0][
            "max_abs"
        ] = True
    elif mutation == "decoder_fraction_bool":
        report["post_measurement"]["projection_telemetry"]["color_factors"][0][
            "nonzero_fraction"
        ] = True

    qualification = gate._one_update_admission_qualification(report)

    assert qualification["passed"] is False
    assert qualification["criteria"][criterion] is False
    assert qualification["interpretation"] == (
        "stage21_one_update_failed_stop_no_gate_a"
    )


def test_one_update_admission_accepts_complete_production_evidence() -> None:
    qualification = gate._one_update_admission_qualification(
        _passing_one_update_admission()
    )

    assert qualification["passed"] is True
    assert all(qualification["criteria"].values())
    assert qualification["interpretation"] == "stage21_one_update_passed"


def test_one_update_carrier_norm_uses_only_fixed_float32_remeasurement_tolerance() -> None:
    report = _passing_one_update_admission()
    report["carrier"]["post"]["capped_max_l2_norm"] = (
        1.0 + gate.STAGE21_CARRIER_NORM_TOLERANCE
    )
    assert gate._one_update_admission_qualification(report)["passed"] is True

    report["carrier"]["post"]["capped_max_l2_norm"] = (
        1.0 + gate.STAGE21_CARRIER_NORM_TOLERANCE + 1e-9
    )
    qualification = gate._one_update_admission_qualification(report)
    assert qualification["passed"] is False
    assert qualification["criteria"]["carrier_norm_envelope"] is False


def _decoder_replay_boundary_report() -> dict[str, object]:
    report = _passing_one_update_admission()
    for phase in ("pre_measurement", "post_measurement"):
        report[phase]["compact_reconciliation_max_abs"] = [3e-5, 3e-5]
        report[phase]["consumer_witnesses"][
            "readout_capped_residual_max_abs"
        ] = 3e-5
    return report


def test_decoder_replay_tolerance_accepts_preregistered_float32_bound() -> None:
    assert gate.STAGE21_DECODER_REPLAY_ATOL == 3e-5

    qualification = gate._one_update_admission_qualification(
        _decoder_replay_boundary_report()
    )

    assert qualification["passed"] is True
    assert qualification["criteria"][
        "decoder_carrier_signal_finite_nonzero"
    ] is True


@pytest.mark.parametrize("phase", ["pre_measurement", "post_measurement"])
@pytest.mark.parametrize("field", ["compact", "readout"])
def test_decoder_replay_tolerance_rejects_one_ulp_over_bound(
    phase: str, field: str
) -> None:
    report = _decoder_replay_boundary_report()
    assert gate._one_update_admission_qualification(report)["passed"] is True
    over = float(np.nextafter(np.float64(3e-5), np.inf))
    if field == "compact":
        report[phase]["compact_reconciliation_max_abs"][0] = over
    else:
        report[phase]["consumer_witnesses"][
            "readout_capped_residual_max_abs"
        ] = over

    qualification = gate._one_update_admission_qualification(report)

    assert qualification["passed"] is False
    assert qualification["criteria"][
        "decoder_carrier_signal_finite_nonzero"
    ] is False


def test_query_capped_residual_retains_tighter_tolerance() -> None:
    report = _decoder_replay_boundary_report()
    assert gate._one_update_admission_qualification(report)["passed"] is True
    report["post_measurement"]["consumer_witnesses"][
        "query_capped_residual_max_l2"
    ] = float(np.nextafter(np.float64(1e-6), np.inf))

    qualification = gate._one_update_admission_qualification(report)

    assert qualification["passed"] is False
    assert qualification["criteria"][
        "decoder_carrier_signal_finite_nonzero"
    ] is False


@pytest.mark.parametrize(
    "mutation,criterion",
    [
        ("schema", "schema_and_target"),
        ("updates", "fixed_production_configuration"),
        ("qualifying", "explicitly_nonqualifying"),
        ("loss_count", "complete_finite_telemetry"),
        ("loss_bool_array", "complete_finite_telemetry"),
        ("initial_ce_bool", "complete_finite_telemetry"),
        ("tail_bool", "tail_64_descends_from_initial_depth_mean"),
        ("telemetry_max_bool", "complete_finite_telemetry"),
        ("state_nonfinite", "complete_finite_telemetry"),
        ("stale_tail", "tail_64_descends_from_initial_depth_mean"),
        ("no_descent", "tail_64_descends_from_initial_depth_mean"),
        ("collapsed_h0", "held_out_predictions_do_not_collapse"),
        ("bad_histogram", "held_out_predictions_do_not_collapse"),
        ("architecture", "carrier_architecture_identity"),
    ],
)
def test_stability_256_admission_fails_closed(
    mutation: str, criterion: str
) -> None:
    report = _passing_stability_admission()
    if mutation == "schema":
        report["schema_version"] = 0
    elif mutation == "updates":
        report["training_updates"] = 255
    elif mutation == "qualifying":
        report["qualification_regime"] = "preregistered_full"
    elif mutation == "loss_count":
        report["losses"].pop()
    elif mutation == "loss_bool_array":
        report["losses"] = [True] * 192 + [False] * 64
        report["initial_depth_cross_entropy"] = {"0": 1.0, "1": 1.0}
        report["tail_64_mean_loss"] = 0.0
    elif mutation == "initial_ce_bool":
        report["initial_depth_cross_entropy"]["0"] = True
        report["initial_depth_cross_entropy"]["1"] = 3.8
    elif mutation == "tail_bool":
        report["losses"][-64:] = [1.0] * 64
        report["tail_64_mean_loss"] = True
    elif mutation == "telemetry_max_bool":
        report["telemetry_summaries"]["gradients"]["max_abs"] = True
    elif mutation == "state_nonfinite":
        report["finite_telemetry"]["states"] = False
    elif mutation == "stale_tail":
        report["tail_64_mean_loss"] = 1.5
    elif mutation == "no_descent":
        report["losses"][-64:] = [2.5] * 64
        report["tail_64_mean_loss"] = 2.5
    elif mutation == "collapsed_h0":
        report["held_out_intact_by_depth"]["0"].update(
            prediction_histogram=[512, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            unique_predicted_colors=1,
        )
    elif mutation == "bad_histogram":
        report["held_out_intact_by_depth"]["1"]["prediction_histogram"][0] = 0
    elif mutation == "architecture":
        report["architecture"]["carrier_stabilizer"] = "unknown"

    qualification = gate._stability_admission_qualification(report)

    assert qualification["passed"] is False
    assert qualification["criteria"][criterion] is False
    assert qualification["interpretation"] == (
        "stage21_stability_256_failed_stop_no_gate_a"
    )


def test_stability_256_accepts_descent_without_claiming_gate_a() -> None:
    qualification = gate._stability_admission_qualification(
        _passing_stability_admission()
    )

    assert qualification["passed"] is True
    assert all(qualification["criteria"].values())
    assert qualification["interpretation"] == (
        "stage21_stability_256_passed_nonqualifying"
    )


def test_gate_a_qualification_requires_both_stage21_admissions() -> None:
    architecture = _passing_architecture()
    kwargs = dict(
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
        initialization=_passing_formal_initialization(),
        data=_passing_formal_data(),
        config=gate.BindingGateConfig(),
    )

    passed = gate._qualification_report(
        **kwargs,
        one_update_admission=_passing_one_update_admission(),
        stability_admission=_passing_stability_admission(),
    )
    assert passed["passed"] is True
    assert passed["criteria"]["stage21_one_update_admitted"] is True
    assert passed["criteria"]["stage21_stability_256_admitted"] is True

    failed = gate._qualification_report(
        **kwargs,
        one_update_admission={},
        stability_admission=_passing_stability_admission(),
    )
    assert failed["passed"] is False
    assert failed["criteria"]["stage21_one_update_admitted"] is False


def test_stage21_drivers_share_the_production_transformed_train_step() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_make_pp_prop_trainer" in functions
    for name in (
        "_train_pp_prop",
        "_run_one_update_admission",
        "_run_stability_admission",
    ):
        assert name in functions
        assert not any(
            isinstance(child, (ast.For, ast.While))
            for child in ast.walk(functions[name])
        )
    assert source.count("learner.etrace_grad(") == 1


def test_reduced_stage21_data_builder_preserves_layout_controls_and_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = gate.BindingGateConfig.smoke_config()
    full = dataclasses.replace(smoke, training_updates=4)
    stability = dataclasses.replace(smoke, training_updates=2)
    monkeypatch.setattr(
        gate.BindingGateConfig,
        "stage21_one_update_config",
        classmethod(lambda cls: full),
    )
    monkeypatch.setattr(
        gate.BindingGateConfig,
        "stage21_stability_config",
        classmethod(lambda cls: stability),
    )

    data = gate._build_stage21_admission_data()

    assert data.training_events.shape == (
        stability.training_updates,
        stability.sequence_length,
        stability.batch_size,
        stability.row_config.input_width,
    )
    assert data.validation_intact.shape[1] == stability.validation_episodes
    assert np.any(data.validation_intact[: gate.SYMBOL_COUNT])
    assert not np.any(data.validation_no_context[: gate.SYMBOL_COUNT])
    assert np.any(data.validation_no_context[gate.SYMBOL_COUNT :])
    assert data.training_events.flags.writeable is False
    prefix = gate._production_prefix_data(data, 1)
    assert prefix.training_events.shape[0] == 1
    assert prefix.validation_intact is data.validation_intact
    schedule = gate._stability_schedule_report(data)
    assert set(schedule) == set(gate.PREREGISTERED_STABILITY_DIGESTS)
    assert all(len(value) == 64 for value in schedule.values())


def test_actual_smoke_stage21_drivers_emit_finite_nonqualifying_evidence() -> None:
    config = gate.BindingGateConfig.smoke_config()
    data = gate.build_binding_data(config)

    one_update = gate._run_one_update_admission(data, config)
    stability = gate._run_stability_admission(data, config)

    assert one_update["control"] == "example21_stage21_one_update_admission"
    assert one_update["executed_updates"] == 1
    assert all(one_update["finite_telemetry"].values())
    assert set(one_update["gradient_group_norms"]) == set(
        gate._STAGE21_OPTIMIZATION_PATHS
    )
    assert set(one_update["adam_factor_group_norms"]) == set(
        gate._STAGE21_OPTIMIZATION_PATHS
    )
    assert all(
        value["adam_step"] == value["schedule_step"] == value["optimizer_step"] == 1
        for value in one_update["adam_factor_group_norms"].values()
    )
    assert one_update["qualification"]["passed"] is False

    assert stability["control"] == "example21_stage21_stability_256_admission"
    assert len(stability["losses"]) == config.training_updates
    assert all(stability["finite_telemetry"].values())
    assert set(stability["held_out_intact_by_depth"]) == {"0", "1"}
    assert stability["qualification"]["passed"] is False


def test_telemetry_helpers_reject_shape_and_adam_state_drift() -> None:
    paths = {
        label: tuple(label.split("/")) for label in gate._STAGE21_OPTIMIZATION_PATHS
    }
    moment = {key: np.asarray([1.0, 2.0]) for key in paths.values()}
    adam_state = namedtuple("ScaleByAdamState", ("count", "mu", "nu"))
    schedule_state = namedtuple("ScaleByScheduleState", ("count",))
    optimizer = SimpleNamespace(
        opt_state=SimpleNamespace(
            value=(
                adam_state(
                    np.asarray(1, dtype=np.int32),
                    moment,
                    {key: value / 10.0 for key, value in moment.items()},
                ),
                schedule_state(np.asarray(1, dtype=np.int32)),
            )
        ),
        step_count=SimpleNamespace(value=np.asarray(1, dtype=np.int32)),
    )
    trainer = SimpleNamespace(
        optimizer=optimizer,
        learner=SimpleNamespace(
            param_states={
                key: SimpleNamespace(value=np.zeros((2,), dtype=np.float32))
                for key in paths.values()
            }
        ),
        parameter_keys=paths,
    )

    reports = gate._adam_factor_reports(trainer)

    assert set(reports) == set(paths)
    assert all(report["first_moment_count"] == 2 for report in reports.values())
    assert all(report["second_moment_count"] == 2 for report in reports.values())
    assert gate._scalar_step(np.asarray(1, dtype=np.int32), "step") == 1
    with pytest.raises(RuntimeError, match="scalar integer"):
        gate._scalar_step(np.asarray([1], dtype=np.int32), "step")
    with pytest.raises(RuntimeError, match="first/second moments"):
        gate._adam_factor_reports(
            SimpleNamespace(optimizer=SimpleNamespace(opt_state=SimpleNamespace(value=())))
        )

    mismatched = copy.deepcopy(trainer)
    mismatched.optimizer.opt_state.value[0].mu.pop(paths["memory_write_scale"])
    with pytest.raises(RuntimeError, match="exactly match"):
        gate._adam_factor_reports(mismatched)

    counts = {"a": 1, "b": 2}
    assert gate._reports_from_vector(
        [[0.0, 0.0], [1.0, 2.0]], ("a", "b"), counts, count_name="count"
    ) == {
        "a": {"l2_norm": 1.0, "count": 1},
        "b": {"l2_norm": 2.0, "count": 2},
    }
    with pytest.raises(ValueError, match="required paths"):
        gate._reports_from_vector([1.0], ("a", "b"), counts, count_name="count")


def test_formal_admission_evidence_recomputes_both_inner_qualifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher

    admissions = {
        "one_update": _passing_one_update_admission(),
        "stability_256": _passing_stability_admission(),
    }

    def load(_path: object, *, target: str, **_kwargs: object) -> dict[str, object]:
        return {
            "admission": copy.deepcopy(admissions[target]),
            "manifest": {"bundle_sha256": target + "-bundle"},
            "manifest_sha256": target + "-manifest",
            "preflight_sha256": target + "-preflight",
            "result_sha256": target + "-result",
        }

    monkeypatch.setattr(launcher, "load_authenticated_admission", load)
    evidence = gate._formal_admission_evidence(
        {"one_update": "one.json", "stability_256": "stability.json"},
        source=_passing_source(),
        environment=_passing_environment(),
    )

    assert set(evidence) == {"one_update", "stability_256"}
    assert evidence["one_update"]["admission"]["target"] == "one_update"
    assert evidence["stability_256"]["manifest_sha256"].endswith("-manifest")

    admissions["one_update"]["schema_version"] = 0
    with pytest.raises(ValueError, match="fails recomputation"):
        gate._formal_admission_evidence(
            {"one_update": "one.json", "stability_256": "stability.json"},
            source=_passing_source(),
            environment=_passing_environment(),
        )
    with pytest.raises(ValueError, match="both fixed"):
        gate._formal_admission_evidence(
            {"one_update": "one.json"},
            source=_passing_source(),
            environment=_passing_environment(),
        )


def test_authenticated_gpu_launch_rejects_source_or_device_substitution() -> None:
    gate._require_authenticated_gpu_launch(_passing_source(), _passing_environment())
    dirty = _passing_source()
    dirty["dirty"] = True
    with pytest.raises(RuntimeError, match="verified clean"):
        gate._require_authenticated_gpu_launch(dirty, _passing_environment())
    cpu = _passing_environment()
    cpu["backend"] = "cpu"
    with pytest.raises(RuntimeError, match="GPU image/device"):
        gate._require_authenticated_gpu_launch(_passing_source(), cpu)


@pytest.mark.parametrize(
    ("target", "control"),
    [
        ("one_update", "example21_stage21_one_update_admission"),
        ("stability_256", "example21_stage21_stability_256_admission"),
    ],
)
def test_stage21_envelope_uses_only_fixed_authenticated_target(
    monkeypatch: pytest.MonkeyPatch, target: str, control: str
) -> None:
    admission = {
        "control": control,
        "config": {"training_updates": 1},
        "qualification": {"passed": False, "interpretation": "diagnostic"},
    }
    monkeypatch.setattr(gate, "_source_report", _passing_source)
    monkeypatch.setattr(gate, "_environment_report", _passing_environment)
    monkeypatch.setattr(gate, "_build_stage21_admission_data", lambda: object())
    monkeypatch.setattr(
        gate,
        "_run_one_update_admission",
        lambda data, config: copy.deepcopy(admission),
    )
    monkeypatch.setattr(
        gate,
        "_run_stability_admission",
        lambda data, config: copy.deepcopy(admission),
    )
    monkeypatch.setattr(gate.legacy, "_device_memory_report", lambda: {"peak": 1})

    result = gate.run_stage21_admission(target)

    assert result["schema_version"] == 3
    assert result["target"] == target
    assert result["control"] == control
    assert result["qualification"]["passed"] is False
    assert result["environment"]["device_memory_after_run"] == {"peak": 1}


def test_stage21_envelope_rejects_unknown_target_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "_source_report",
        lambda: pytest.fail("source must not be read for an unknown target"),
    )
    with pytest.raises(ValueError, match="one_update.*stability_256"):
        gate.run_stage21_admission("other")


def test_formal_gate_invokes_authenticated_manifest_validator_before_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ValidatedStop(RuntimeError):
        pass

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(gate, "_source_report", _passing_source)
    monkeypatch.setattr(gate, "_environment_report", _passing_environment)

    def validate(
        manifests: dict[str, str], **kwargs: object
    ) -> dict[str, object]:
        calls.append({"manifests": manifests, **kwargs})
        raise ValidatedStop

    monkeypatch.setattr(gate, "_formal_admission_evidence", validate)
    monkeypatch.setattr(
        gate,
        "build_binding_data",
        lambda config: pytest.fail("formal data ran before manifest authentication"),
    )
    manifests = {"one_update": "one.json", "stability_256": "stability.json"}

    with pytest.raises(ValidatedStop):
        gate.run_binding_gate(
            gate.BindingGateConfig(), admission_manifests=manifests
        )

    assert calls == [
        {
            "manifests": manifests,
            "source": _passing_source(),
            "environment": _passing_environment(),
        }
    ]


def test_formal_gate_rejects_missing_manifests_before_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_source_report", _passing_source)
    monkeypatch.setattr(gate, "_environment_report", _passing_environment)
    monkeypatch.setattr(
        gate,
        "build_binding_data",
        lambda config: pytest.fail("formal data ran without admission manifests"),
    )

    with pytest.raises(ValueError, match="authenticated Stage 2.1"):
        gate.run_binding_gate(gate.BindingGateConfig())


def test_admission_cli_returns_success_after_writing_failed_scientific_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = {
        "qualification": {"passed": False, "interpretation": "stop"},
    }
    destination = tmp_path / "admission.json"
    monkeypatch.setattr(gate, "run_stage21_admission", lambda target: result)
    monkeypatch.setattr(gate, "write_artifact", lambda value, path: destination)

    exit_code = gate.main(
        ["--target", "one_update", "--output", str(destination)]
    )

    assert exit_code == 0


@pytest.mark.parametrize(
    "option",
    [
        ("--training-updates", "9999"),
        ("--batch-size", "63"),
        ("--validation-episodes", "256"),
        ("--gap-steps", "2"),
        ("--neuron-count", "1024"),
        ("--recurrent-edges", "8192"),
        ("--readout-width", "64"),
        ("--color-rank", "8"),
        ("--learning-rate", "0.001"),
        ("--context-memory-width", "16"),
        ("--memory-decay", "0.9"),
        ("--sparse-backend", "jax_raw"),
    ],
)
def test_fixed_admission_cli_rejects_every_topology_or_budget_override(
    monkeypatch: pytest.MonkeyPatch, option: tuple[str, str]
) -> None:
    monkeypatch.setattr(
        gate,
        "run_stage21_admission",
        lambda target: pytest.fail("invalid fixed target reached execution"),
    )
    with pytest.raises(ValueError, match="(?i)fixed admission"):
        gate.main(["--target", "one_update", *option])


def test_smoke_and_nondefault_gate_a_reject_authenticated_manifest_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "run_binding_gate",
        lambda *args, **kwargs: pytest.fail("invalid Gate A reached execution"),
    )
    manifests = [
        "--one-update-manifest",
        "one.json",
        "--stability-manifest",
        "stability.json",
    ]
    with pytest.raises(ValueError, match="formal.*manifests"):
        gate.main(["--smoke", *manifests])
    with pytest.raises(ValueError, match="preregistered"):
        gate.main(["--training-updates", "9999", *manifests])
