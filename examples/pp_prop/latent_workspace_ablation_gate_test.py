"""Tests for the preregistered Example 21 Gate C causal ablations."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
import pytest

from examples.pp_prop import latent_workspace_binding_gate as gate_a
from examples.pp_prop import latent_workspace_depth_gate as gate_b


_MODULE_NAME = "examples.pp_prop.latent_workspace_ablation_gate"
try:
    gate_c = importlib.import_module(_MODULE_NAME)
    _IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as error:
    if error.name != _MODULE_NAME:
        raise
    gate_c = None
    _IMPORT_ERROR = error


requires_gate_c = pytest.mark.skipif(
    gate_c is None,
    reason="latent_workspace_ablation_gate is not implemented",
)


_ARMS = ("full", "query_only", "terminal_only", "legacy", "frozen_write")
_REGIMES = ("gate_a", "gate_b")
_SHARED_PATHS = (
    "color_factor_head/weight",
    "ff_syn/comm/weight",
    "height_head/weight",
    "readout_projection/weight",
    "rec_syn/comm/weight",
    "width_head/weight",
)
_MEMORY_PATHS = (
    "memory_read_projection/weight",
    "memory_write_scale",
    "workspace_query_projection/weight",
)
_FULL_PATHS = tuple(sorted((*_SHARED_PATHS, *_MEMORY_PATHS)))

_GATE_A_SCHEDULE_SHA256 = {
    "training_schedule_sha256": (
        "25cae0684c3a0cb1a0d0ae1a12b7db8bdf37a1f15d687cdf79362c9c6163ef9b"
    ),
    "validation_schedule_sha256": (
        "80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd"
    ),
    "training_mapping_ids_sha256": (
        "fbd48ad9a8d3ecb0dd0812abbbda35953def52862785ce048e17b2eb9fdd3499"
    ),
    "validation_mapping_ids_sha256": (
        "a75b3b2ab05110e21fef1ea44ae3fb701d557f45351e5a17cf89a80e80f689f3"
    ),
}
_GATE_B_TRAINING_SHA256 = {
    "events": "a1937b7f8d5d4da5f30216847cc63d022d9ec46d5cf152b25f5a30a59a1eb84f",
    "targets": "4082d2fd1440e9d14b0c81c754158f05b8056137a9116aee667f8d112312184c",
    "loss_weights": (
        "044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31"
    ),
    "advance_masks": (
        "2fc1b2acd9f73e567684d2a85f44c4009c5941ce262a527589066117ec27a4cc"
    ),
    "mapping_ids": (
        "78c2d8aaa9e874dbcc1c25363875ff8aec0356a711d2426e09f2e79c76c72cb7"
    ),
    "efforts": "c7ca75132501bda8e6b5695a48a1ae5cde22da587f4658f7721bd4e3adcd58e6",
    "query_colors": (
        "38b4cecef323dce16b0478fdd3874c9383804c913c39aaf017ce34554dcd37cb"
    ),
    "presentation_orders": (
        "0650be382b381d7ab14b642c6fcdb16ae410e70a4c5821b10643bce41e3f7ca5"
    ),
}
_GATE_B_VALIDATION_SHA256 = {
    "mapping_ids": (
        "b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8"
    ),
    "query_colors": (
        "c7e70f56cca66d920d5d690a902b9943f2fcfdff7003fa4bbb3580070738d67e"
    ),
    "presentation_orders": (
        "0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0"
    ),
    "shuffled_shifts": (
        "15af1f04589cc523d89b66d2f07027158d69068901d786eecfd259a156f2f2d0"
    ),
    "intact": "5683aa84aa2ef8a1ff623e5e0b60afb3451e617728f0363d3ad84f2ea52dacde",
    "shuffled": (
        "abd5eb4ab2e2a685faeb8f6bf785ad2deb97721b00e8d194b4f65d4995516be3"
    ),
    "no_context": (
        "45fd14d3faefad83b0ce6d908456320afa67944b361159cfe503fdfab591162d"
    ),
    "targets_by_depth": (
        "a438d64347dc4ec5cfc639342d8b142c785e497ddf06728eb03f8ccfb42d3cd6"
    ),
    "advance_masks": (
        "b88b3593d9df51260fbafa4a937159c3da3f56fc33335a30993c0ff8a7462ac8"
    ),
}


def test_gate_c_module_is_importable() -> None:
    assert _IMPORT_ERROR is None, (
        "missing production module examples.pp_prop.latent_workspace_ablation_gate"
    )


def _spec_dict(spec: Any) -> dict[str, Any]:
    assert dataclasses.is_dataclass(spec)
    return dataclasses.asdict(spec)


def _manual_path_digest(path: str, value: Any) -> str:
    fields = [
        b"example21-gate-c-gradient-path-v1",
        path.encode("utf-8"),
    ]
    for index, leaf in enumerate(jax.tree.leaves(value)):
        array = np.ascontiguousarray(np.asarray(leaf))
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _manual_global_digest(gradients: Mapping[str, Any]) -> str:
    fields = [b"example21-gate-c-gradient-global-v1"]
    for path in sorted(gradients):
        fields.extend(
            (
                path.encode("utf-8"),
                _manual_path_digest(path, gradients[path]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


@requires_gate_c
class TestGateCContracts:
    def test_api_and_fixed_artifact_literals(self) -> None:
        required = {
            "GateCArmSpec",
            "GateCRegimeSpec",
            "GateCConfig",
            "ARM_ORDER",
            "REGIME_ORDER",
            "ARM_SPECS",
            "REGIME_SPECS",
            "SHARED_PARAMETER_PATHS",
            "FULL_PARAMETER_PATHS",
            "QUALIFICATION_CRITERIA",
            "_loss_weights",
            "_model_config_for_arm",
            "_optimizer_parameter_paths",
            "_schedule_identity_report",
            "_metric_summary",
            "_blocking_margin_report",
            "_gradient_path_sha256",
            "_gradient_global_sha256",
            "_gradient_comparison",
            "_oracle_contract",
            "_qualification_report",
        }

        assert not (required - set(vars(gate_c)))
        assert gate_c.GATE_C_SCHEMA_VERSION == 1
        assert (
            gate_c.GATE_C_INITIALIZATION_CONTROL
            == "example21_gate_c_initialization_admission"
        )
        assert gate_c.GATE_C_CONTROL == "example21_pp_prop_learnability_gate_c"
        assert gate_c.ARM_ORDER == _ARMS
        assert gate_c.REGIME_ORDER == _REGIMES

    def test_arm_specs_are_exact_and_frozen(self) -> None:
        assert tuple(gate_c.ARM_SPECS) == _ARMS
        assert {
            name: _spec_dict(gate_c.ARM_SPECS[name]) for name in _ARMS
        } == {
            "full": {
                "name": "full",
                "memory_mode": "full",
                "supervision": "per_checkpoint",
                "context_memory_width": 32,
                "optimizer_excluded_paths": (),
            },
            "query_only": {
                "name": "query_only",
                "memory_mode": "query_only",
                "supervision": "per_checkpoint",
                "context_memory_width": 32,
                "optimizer_excluded_paths": (),
            },
            "terminal_only": {
                "name": "terminal_only",
                "memory_mode": "full",
                "supervision": "terminal_only",
                "context_memory_width": 32,
                "optimizer_excluded_paths": (),
            },
            "legacy": {
                "name": "legacy",
                "memory_mode": "legacy",
                "supervision": "per_checkpoint",
                "context_memory_width": 0,
                "optimizer_excluded_paths": (),
            },
            "frozen_write": {
                "name": "frozen_write",
                "memory_mode": "full",
                "supervision": "per_checkpoint",
                "context_memory_width": 32,
                "optimizer_excluded_paths": ("memory_write_scale",),
            },
        }

    def test_config_pins_both_canonical_regimes(self) -> None:
        config = gate_c.GateCConfig()

        assert config.gate_a_config == gate_a.BindingGateConfig()
        assert config.gate_b_config == gate_b.DepthGateConfig()
        assert config.oracle_validation_index == 0
        assert config.oracle_effort == 8
        assert config.gradient_chunk_size == 1
        assert config.qualification_regime == "preregistered_full"
        assert tuple(gate_c.REGIME_SPECS) == _REGIMES
        assert _spec_dict(gate_c.REGIME_SPECS["gate_a"]) == {
            "name": "gate_a",
            "sequence_length": 6,
            "input_width": 41,
            "training_updates": 10_000,
            "batch_size": 64,
            "validation_episodes": 512,
            "full_parameter_count": 646_940,
            "full_parameter_sha256": (
                "b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8"
            ),
            "legacy_parameter_count": 514_844,
        }
        assert _spec_dict(gate_c.REGIME_SPECS["gate_b"]) == {
            "name": "gate_b",
            "sequence_length": 19,
            "input_width": 47,
            "training_updates": 4_096,
            "batch_size": 64,
            "validation_episodes": 512,
            "full_parameter_count": 659_228,
            "full_parameter_sha256": (
                "aa463549a8c3c1dbc24c9f727944eada035b776df666a83139214078d0f83d6d"
            ),
            "legacy_parameter_count": 527_132,
        }

    @pytest.mark.parametrize(
        "mutation",
        ["gate_a_budget", "gate_b_budget", "oracle_index", "oracle_effort", "chunk"],
    )
    def test_any_valid_config_mutation_is_nonqualifying(self, mutation: str) -> None:
        config = gate_c.GateCConfig()
        if mutation == "gate_a_budget":
            config = dataclasses.replace(
                config,
                gate_a_config=dataclasses.replace(
                    config.gate_a_config, training_updates=9_999
                ),
            )
        elif mutation == "gate_b_budget":
            config = dataclasses.replace(
                config,
                gate_b_config=dataclasses.replace(
                    config.gate_b_config, training_updates=1
                ),
            )
        elif mutation == "oracle_index":
            config = dataclasses.replace(config, oracle_validation_index=1)
        elif mutation == "oracle_effort":
            config = dataclasses.replace(config, oracle_effort=4)
        else:
            config = dataclasses.replace(config, gradient_chunk_size=2)

        assert config.qualification_regime == "nonqualifying_abbreviated"

    @pytest.mark.parametrize("regime", _REGIMES)
    def test_loss_weights_are_normalized_and_only_terminal_arm_changes_schedule(
        self,
        regime: str,
    ) -> None:
        efforts = np.asarray([1, 2, 4, 8], dtype=np.int32)
        full = gate_c._loss_weights(regime, "full", efforts=efforts)
        terminal = gate_c._loss_weights(regime, "terminal_only", efforts=efforts)

        if regime == "gate_a":
            assert full.shape == (6,)
            np.testing.assert_array_equal(full[:4], 0.0)
            np.testing.assert_array_equal(full[4:], 0.5)
            np.testing.assert_array_equal(terminal[:-1], 0.0)
            assert terminal[-1] == 1.0
        else:
            assert full.shape == terminal.shape == (4, 19)
            for row, effort in enumerate(efforts):
                np.testing.assert_array_equal(full[row, :10], 0.0)
                np.testing.assert_array_equal(
                    full[row, 10 : 11 + effort],
                    np.float32(1.0 / (effort + 1)),
                )
                np.testing.assert_array_equal(full[row, 11 + effort :], 0.0)
                np.testing.assert_array_equal(terminal[row, : 10 + effort], 0.0)
                assert terminal[row, 10 + effort] == 1.0
                np.testing.assert_array_equal(terminal[row, 11 + effort :], 0.0)
        assert full.dtype == terminal.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(np.sum(full, axis=-1), 1.0, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            np.sum(terminal, axis=-1), 1.0, rtol=0.0, atol=1e-7
        )
        for unchanged in ("query_only", "legacy", "frozen_write"):
            np.testing.assert_array_equal(
                gate_c._loss_weights(regime, unchanged, efforts=efforts), full
            )

    @pytest.mark.parametrize("regime", _REGIMES)
    def test_model_configs_differ_only_for_the_declared_legacy_memory(
        self,
        regime: str,
    ) -> None:
        config = gate_c.GateCConfig()
        batch_size = 64
        by_arm = {
            arm: gate_c._model_config_for_arm(
                config, regime, arm, batch_size=batch_size
            )
            for arm in _ARMS
        }

        for arm in ("query_only", "terminal_only", "frozen_write"):
            assert by_arm[arm] == by_arm["full"]
        legacy = by_arm["legacy"]
        assert legacy.context_memory_width == 0
        assert legacy.demonstration_phase_index is None
        assert legacy.query_phase_index is None
        assert legacy.input_side_valid_index is None
        assert legacy.output_side_valid_index is None
        assert legacy.memory_key_indices == ()
        assert legacy.memory_value_indices == ()
        memory_fields = {
            "context_memory_width",
            "demonstration_phase_index",
            "query_phase_index",
            "input_side_valid_index",
            "output_side_valid_index",
            "memory_key_indices",
            "memory_value_indices",
        }
        full_values = dataclasses.asdict(by_arm["full"])
        legacy_values = dataclasses.asdict(legacy)
        assert {
            key: value for key, value in full_values.items() if key not in memory_fields
        } == {
            key: value for key, value in legacy_values.items() if key not in memory_fields
        }

    def test_shared_and_optimizer_paths_are_exact(self) -> None:
        assert gate_c.SHARED_PARAMETER_PATHS == _SHARED_PATHS
        assert gate_c.FULL_PARAMETER_PATHS == _FULL_PATHS

        for arm in ("full", "query_only", "terminal_only"):
            assert gate_c._optimizer_parameter_paths(_FULL_PATHS, arm) == _FULL_PATHS
        assert gate_c._optimizer_parameter_paths(_SHARED_PATHS, "legacy") == (
            _SHARED_PATHS
        )
        assert gate_c._optimizer_parameter_paths(
            _FULL_PATHS, "frozen_write"
        ) == tuple(path for path in _FULL_PATHS if path != "memory_write_scale")
        with pytest.raises(ValueError, match="memory_write_scale"):
            gate_c._optimizer_parameter_paths(_SHARED_PATHS, "frozen_write")

    def test_schedule_identity_contract_matches_both_production_generators(self) -> None:
        report = gate_c._schedule_identity_report(gate_c.GateCConfig())

        assert report == {
            "gate_a": _GATE_A_SCHEDULE_SHA256,
            "gate_b": {
                "training_global_sha256": _GATE_B_TRAINING_SHA256,
                "validation_sha256": _GATE_B_VALIDATION_SHA256,
            },
        }
        assert gate_b._PRODUCTION_ENCODED_GLOBAL_SHA256 == _GATE_B_TRAINING_SHA256
        assert gate_b._PRODUCTION_VALIDATION_SHA256 == _GATE_B_VALIDATION_SHA256
        assert (
            gate_a.PREREGISTERED_TRAINING_SCHEDULE_SHA256
            == _GATE_A_SCHEDULE_SHA256["training_schedule_sha256"]
        )
        assert (
            gate_a.PREREGISTERED_STABILITY_DIGESTS["validation_schedule_sha256"]
            == _GATE_A_SCHEDULE_SHA256["validation_schedule_sha256"]
        )

    def test_metric_summary_uses_terminal_gate_a_and_mean_matching_gate_b(self) -> None:
        gate_a_evaluation = {
            "depths": {
                "1": {
                    "intact": {"accuracy": 0.75},
                    "shuffled": {"accuracy": 0.25},
                }
            }
        }
        gate_b_evaluation = {
            "efforts": {
                str(effort): {"intact": {"accuracy": accuracy}}
                for effort, accuracy in zip((1, 2, 4, 8), (0.9, 0.7, 0.5, 0.3), strict=True)
            }
        }

        assert gate_c._metric_summary(gate_a_evaluation, gate_b_evaluation) == {
            "binding_gap": 0.5,
            "depth_accuracy": 0.6,
        }

    @pytest.mark.parametrize("invalid", [True, math.nan, math.inf, "0.5"])
    def test_metric_summary_rejects_nonfinite_or_type_confused_accuracy(
        self,
        invalid: Any,
    ) -> None:
        gate_a_evaluation = {
            "depths": {
                "1": {
                    "intact": {"accuracy": invalid},
                    "shuffled": {"accuracy": 0.0},
                }
            }
        }
        gate_b_evaluation = {
            "efforts": {
                str(effort): {"intact": {"accuracy": 0.5}}
                for effort in (1, 2, 4, 8)
            }
        }

        with pytest.raises((TypeError, ValueError)):
            gate_c._metric_summary(gate_a_evaluation, gate_b_evaluation)

    def test_blocking_margins_pass_at_every_exact_boundary(self) -> None:
        metrics = {
            "full": {"binding_gap": 0.80, "depth_accuracy": 0.60},
            "query_only": {"binding_gap": 0.82, "depth_accuracy": 0.45},
            "terminal_only": {"binding_gap": 0.82, "depth_accuracy": 0.50},
            "legacy": {"binding_gap": 0.55, "depth_accuracy": 0.45},
            "frozen_write": {"binding_gap": 0.75, "depth_accuracy": 0.55},
        }

        report = gate_c._blocking_margin_report(metrics)

        assert report["query_only"]["passed"] is True
        assert report["terminal_only"]["passed"] is True
        assert report["legacy"]["passed"] is True
        assert report["blocking_passed"] is True
        assert report["frozen_write"]["blocking"] is False
        assert report["frozen_write"]["write_modulation_necessary"] is True

    @pytest.mark.parametrize(
        ("arm", "field", "direction"),
        [
            ("query_only", "depth_accuracy", 1.0),
            ("query_only", "binding_gap", 1.0),
            ("terminal_only", "depth_accuracy", 1.0),
            ("terminal_only", "binding_gap", 1.0),
            ("legacy", "depth_accuracy", 1.0),
            ("legacy", "binding_gap", 1.0),
        ],
    )
    def test_each_blocking_margin_fails_immediately_beyond_boundary(
        self,
        arm: str,
        field: str,
        direction: float,
    ) -> None:
        metrics = {
            "full": {"binding_gap": 0.80, "depth_accuracy": 0.60},
            "query_only": {"binding_gap": 0.82, "depth_accuracy": 0.45},
            "terminal_only": {"binding_gap": 0.82, "depth_accuracy": 0.50},
            "legacy": {"binding_gap": 0.55, "depth_accuracy": 0.45},
            "frozen_write": {"binding_gap": 0.80, "depth_accuracy": 0.60},
        }
        value = metrics[arm][field]
        metrics[arm][field] = math.nextafter(value, value + direction)

        report = gate_c._blocking_margin_report(metrics)

        assert report[arm]["passed"] is False
        assert report["blocking_passed"] is False

    def test_qualification_fails_closed_on_missing_or_fabricated_evidence(self) -> None:
        config = gate_c.GateCConfig()
        for report in ({}, {"qualification": {"passed": True}}, {"schema_version": True}):
            qualification = gate_c._qualification_report(report, config=config)
            assert qualification["passed"] is False
            assert set(qualification["criteria"]) == set(
                gate_c.QUALIFICATION_CRITERIA
            )
            assert not all(qualification["criteria"].values())
            assert qualification["interpretation"] == (
                "gate_c_failed_stop_no_causal_mechanism_conclusion"
            )

    def test_oracle_contract_is_the_exact_first_gate_b_validation_episode(self) -> None:
        contract = gate_c._oracle_contract(gate_c.GateCConfig())

        assert contract == {
            "regime": "gate_b",
            "validation_episode_index": 0,
            "arm": "intact",
            "effort": 8,
            "batch_size": 1,
            "mapping_id": 232_423,
            "mapping": [6, 7, 5, 2, 0, 4, 8, 9, 1, 3],
            "query_color": 4,
            "presentation_order": [6, 2, 5, 3, 8, 7, 4, 9, 0, 1],
            "shuffled_shift": 1,
            "targets": [0, 6, 8, 1, 7, 9, 3, 2, 5],
            "advance_mask": [True] * 19,
            "events_shape": [19, 47],
            "events_dtype": "<f4",
            "events_sha256": (
                "36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec"
            ),
            "gradient_chunk_size": 1,
        }

    def test_gradient_digest_framing_is_exact_and_order_independent(self) -> None:
        gradients = {
            "alpha/path": {
                "z": np.asarray([1.5, -2.0], dtype=np.float32),
                "a": np.asarray([[3.0]], dtype=np.float16),
            },
            "beta": (np.asarray(0.25, dtype=np.float32),),
        }

        assert gate_c._gradient_path_sha256(
            "alpha/path", gradients["alpha/path"]
        ) == "c3282de0c034378cf4e8773f92c78f0e13b83187b576474f8678f3f0d05fc1b9"
        assert gate_c._gradient_path_sha256(
            "beta", gradients["beta"]
        ) == "acab34c99de63ff01b021452ef957de7d8db396bd42e71e9c9f400b99f3f5a6a"
        expected = "8d78936f1c703719d88ddc2b64d0c4166ed1f79243b2eb657c3c884fe91d1089"
        assert gate_c._gradient_global_sha256(gradients) == expected
        assert gate_c._gradient_global_sha256(dict(reversed(gradients.items()))) == expected
        assert gate_c._gradient_global_sha256(gradients) == _manual_global_digest(
            gradients
        )

    def test_gradient_digest_rejects_boolean_or_nonfinite_leaves(self) -> None:
        for invalid in (
            np.asarray([True]),
            np.asarray([math.nan], dtype=np.float32),
            np.asarray([math.inf], dtype=np.float32),
        ):
            with pytest.raises((TypeError, ValueError)):
                gate_c._gradient_path_sha256("invalid", invalid)

    def test_gradient_comparison_uses_full_denominator_and_strict_null_rules(
        self,
    ) -> None:
        comparison = gate_c._gradient_comparison(
            np.asarray([3.0, 4.0]), np.asarray([0.0, 4.0])
        )
        assert {
            key: comparison[key]
            for key in (
                "full_norm",
                "arm_norm",
                "l2_difference",
                "relative_deviation",
                "cosine",
            )
        } == pytest.approx(
            {
                "full_norm": 5.0,
                "arm_norm": 4.0,
                "l2_difference": 3.0,
                "relative_deviation": 0.6,
                "cosine": 0.8,
            }
        )
        assert {
            "relative_deviation_defined": comparison[
                "relative_deviation_defined"
            ],
            "cosine_defined": comparison["cosine_defined"],
        } == {
            "relative_deviation_defined": True,
            "cosine_defined": True,
        }

        zero_full = gate_c._gradient_comparison(
            np.zeros((2,), dtype=np.float32), np.ones((2,), dtype=np.float32)
        )
        assert zero_full["relative_deviation"] is None
        assert zero_full["relative_deviation_defined"] is False
        assert zero_full["cosine"] is None
        assert zero_full["cosine_defined"] is False

        zero_arm = gate_c._gradient_comparison(
            np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)
        )
        assert zero_arm["relative_deviation"] == pytest.approx(1.0)
        assert zero_arm["relative_deviation_defined"] is True
        assert zero_arm["cosine"] is None
        assert zero_arm["cosine_defined"] is False
