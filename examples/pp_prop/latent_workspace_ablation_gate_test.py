"""Tests for the preregistered Example 21 Gate C causal ablations."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import math
from collections.abc import Mapping
from typing import Any

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop import latent_workspace_binding_gate as gate_a
from examples.pp_prop import latent_workspace_depth_gate as gate_b
from examples.pp_prop.latent_workspace_model import LatentWorkspaceModel


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
_INITIALIZATION_CRITERIA = (
    "schema_and_control",
    "preregistered_regimes",
    "gate_a_prerequisite_authenticated",
    "gate_b_prerequisite_authenticated",
    "source_and_gpu_authenticated",
    "source_files_exact",
    "canonical_full_initializations_exact",
    "legacy_initializations_complete",
    "shared_paths_byte_identical",
    "arm_initialization_refs_exact",
    "optimizer_paths_exact",
    "fresh_optimizer_states_zero_and_finite",
    "compiler_topologies_complete",
    "no_behavioral_updates",
)
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


def _manual_shared_path_digest(path: str, value: Any) -> str:
    fields = [
        b"example21-gate-c-shared-path-v1",
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


def _manual_shared_global_digest(values: Mapping[str, Any]) -> str:
    fields = [b"example21-gate-c-shared-global-v1"]
    for path in sorted(values):
        fields.extend(
            (
                path.encode("utf-8"),
                _manual_shared_path_digest(path, values[path]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _manual_optimizer_state_digest(
    trainer: Any,
    *,
    regime: str,
    arm: str,
) -> str:
    fields = [
        b"example21-gate-c-optimizer-state-v1",
        regime.encode("utf-8"),
        arm.encode("utf-8"),
        *(path.encode("utf-8") for path in sorted(trainer.optimizer_parameter_paths)),
    ]
    for index, leaf in enumerate(jax.tree.leaves(trainer.optimizer.opt_state.value)):
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


def _reduced_gate_c_config() -> Any:
    return gate_c.GateCConfig(
        gate_a_config=gate_a.BindingGateConfig.smoke_config(),
        gate_b_config=gate_b.DepthGateConfig(
            training_updates=1,
            batch_size=2,
            validation_episodes=4,
            neuron_count=64,
            recurrent_edges=64,
            readout_width=8,
            color_rank=1,
            staging_chunk_updates=1,
        ),
    )


def _parameter_states(model: LatentWorkspaceModel) -> dict[str, Any]:
    return {
        gate_a._path(path): state
        for path, state in model.states(brainstate.ParamState).items()
    }


def _parameter_digest(model: LatentWorkspaceModel) -> str:
    return legacy._array_digest(legacy._parameter_values(model))


def _assert_parameter_values_equal(
    left: LatentWorkspaceModel,
    right: LatentWorkspaceModel,
    paths: tuple[str, ...],
) -> None:
    left_values = legacy._parameter_values(left)
    right_values = legacy._parameter_values(right)
    for path in paths:
        assert jax.tree.structure(left_values[path]) == jax.tree.structure(
            right_values[path]
        )
        for left_leaf, right_leaf in zip(
            jax.tree.leaves(left_values[path]),
            jax.tree.leaves(right_values[path]),
            strict=True,
        ):
            np.testing.assert_array_equal(left_leaf, right_leaf, err_msg=path)


def _assert_dataclass_arrays_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    assert dataclasses.is_dataclass(left)
    for field in dataclasses.fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value, err_msg=field.name)
        else:
            assert left_value == right_value


def _tree_is_zero(value: Any) -> bool:
    leaves = jax.tree.leaves(value)
    return bool(leaves) and all(
        np.count_nonzero(np.asarray(leaf)) == 0 for leaf in leaves
    )


def _telemetry_is_finite(telemetry: Mapping[str, Any]) -> bool:
    return bool(
        np.isfinite(np.asarray(telemetry["loss"])).all()
        and all(
            bool(np.asarray(values).all())
            for values in telemetry["finite"].values()
        )
        and all(
            bool(np.isfinite(np.asarray(values)).all())
            for values in telemetry["max_abs"].values()
        )
    )


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
            "GateCTrainer",
            "_new_model_for_arm",
            "_copy_shared_initialization",
            "_make_arm_trainer",
            "_regenerate_gate_a_data",
            "_regenerate_gate_b_data",
            "_evaluate_arm",
            "GATE_C_INITIALIZATION_QUALIFICATION_CRITERIA",
            "_shared_path_sha256",
            "_shared_global_sha256",
            "_optimizer_initial_state_report",
            "_initialization_topology_report",
            "_normalized_prerequisites",
            "_gate_c_initialization_qualification",
            "run_gate_c_initialization",
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
        assert gate_c.GATE_C_INITIALIZATION_QUALIFICATION_CRITERIA == (
            _INITIALIZATION_CRITERIA
        )

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

    def test_arm_model_factory_preserves_paired_initialization_and_topology(
        self,
    ) -> None:
        config = _reduced_gate_c_config()
        models = {
            arm: gate_c._new_model_for_arm(
                config,
                "gate_b",
                arm,
                batch_size=config.gate_b_config.batch_size,
            )
            for arm in _ARMS
        }

        assert all(isinstance(model, LatentWorkspaceModel) for model in models.values())
        assert models["query_only"].memory_read_policy == "query_only"
        for arm in ("full", "terminal_only", "legacy", "frozen_write"):
            assert models[arm].memory_read_policy == "full"
        for arm in ("query_only", "terminal_only", "frozen_write"):
            assert _parameter_digest(models[arm]) == _parameter_digest(models["full"])
            _assert_parameter_values_equal(models["full"], models[arm], _FULL_PATHS)
        assert tuple(sorted(_parameter_states(models["legacy"]))) == _SHARED_PATHS
        assert tuple(sorted(_parameter_states(models["full"]))) == _FULL_PATHS
        for left_index, left in enumerate(_ARMS):
            for right in _ARMS[left_index + 1 :]:
                shared = set(_parameter_states(models[left])) & set(
                    _parameter_states(models[right])
                )
                assert all(
                    _parameter_states(models[left])[path]
                    is not _parameter_states(models[right])[path]
                    for path in shared
                )
        np.testing.assert_array_equal(
            np.asarray(_parameter_states(models["frozen_write"])["memory_write_scale"].value),
            1.0,
        )

    def test_shared_initialization_is_copied_into_legacy_without_mutating_full(
        self,
    ) -> None:
        config = _reduced_gate_c_config()
        full = gate_c._new_model_for_arm(
            config,
            "gate_b",
            "full",
            batch_size=config.gate_b_config.batch_size,
        )
        legacy_model = gate_c._new_model_for_arm(
            config,
            "gate_b",
            "legacy",
            batch_size=config.gate_b_config.batch_size,
        )
        legacy_readout = _parameter_states(legacy_model)["readout_projection/weight"]
        legacy_readout.value = jax.tree.map(jnp.zeros_like, legacy_readout.value)
        canonical_sha = _parameter_digest(full)

        evidence = gate_c._copy_shared_initialization(full, legacy_model)

        assert canonical_sha == _parameter_digest(full)
        _assert_parameter_values_equal(full, legacy_model, _SHARED_PATHS)
        assert tuple(evidence["paths"]) == _SHARED_PATHS
        assert evidence["all_equal"] is True
        assert isinstance(evidence["sha256"], str)
        assert len(evidence["sha256"]) == 64
        assert set(evidence["sha256"]) <= set("0123456789abcdef")
        assert not (set(_MEMORY_PATHS) & set(_parameter_states(legacy_model)))

    def test_data_regeneration_matches_each_canonical_generator_exactly(self) -> None:
        config = _reduced_gate_c_config()

        gate_a_first = gate_c._regenerate_gate_a_data(config)
        gate_a_second = gate_c._regenerate_gate_a_data(config)
        _assert_dataclass_arrays_equal(gate_a_first, gate_a_second)
        _assert_dataclass_arrays_equal(
            gate_a_first,
            legacy.build_binding_data(config.gate_a_config),
        )
        assert not np.intersect1d(
            gate_a_first.training_mapping_ids,
            gate_a_first.validation_mapping_ids,
        ).size

        schedule, validation = gate_c._regenerate_gate_b_data(config)
        repeated_schedule, repeated_validation = gate_c._regenerate_gate_b_data(config)
        _assert_dataclass_arrays_equal(schedule, repeated_schedule)
        _assert_dataclass_arrays_equal(validation, repeated_validation)
        expected_schedule = gate_b._build_schedule(config.gate_b_config)
        expected_validation = gate_b._encode_validation_data(
            expected_schedule,
            config.gate_b_config,
        )
        _assert_dataclass_arrays_equal(schedule, expected_schedule)
        _assert_dataclass_arrays_equal(validation, expected_validation)
        assert not np.intersect1d(
            schedule.training_mapping_ids,
            schedule.validation_mapping_ids,
        ).size

    def test_arm_training_driver_is_one_jit_with_internal_for_loop(self) -> None:
        source = inspect.getsource(gate_c._make_arm_trainer)
        function = ast.parse(source).body[0]
        assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        nested = {
            node.name: node
            for node in ast.walk(function)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        train_chunk = nested["train_chunk"]

        assert len(train_chunk.decorator_list) == 1
        assert ast.unparse(train_chunk.decorator_list[0]) == "brainstate.transform.jit"
        assert not any(
            isinstance(node, (ast.For, ast.While)) for node in ast.walk(train_chunk)
        )
        for_loop_calls = [
            node
            for node in ast.walk(train_chunk)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "brainstate.transform.for_loop"
        ]
        assert len(for_loop_calls) == 1


@pytest.fixture(scope="module")
def reduced_gate_b_arm_run() -> dict[str, Any]:
    config = _reduced_gate_c_config()
    schedule, validation = gate_c._regenerate_gate_b_data(config)
    schedule_chunk = next(
        gate_b._iter_schedule_chunks(schedule, config.gate_b_config)
    )
    chunk = gate_b._encode_training_chunk(schedule_chunk, config.gate_b_config)
    models = {
        arm: gate_c._new_model_for_arm(
            config,
            "gate_b",
            arm,
            batch_size=config.gate_b_config.batch_size,
        )
        for arm in _ARMS
    }
    shared_copy = gate_c._copy_shared_initialization(models["full"], models["legacy"])
    trainers = {
        arm: gate_c._make_arm_trainer(models[arm], config, "gate_b", arm)
        for arm in _ARMS
    }
    initial_sha = {arm: _parameter_digest(model) for arm, model in models.items()}
    initial_adam_zero = {
        arm: _tree_is_zero(trainer.optimizer.opt_state.value)
        for arm, trainer in trainers.items()
    }
    frozen_write_before = np.array(
        _parameter_states(models["frozen_write"])["memory_write_scale"].value,
        copy=True,
    )
    telemetry: dict[str, Mapping[str, Any]] = {}
    evaluation: dict[str, Mapping[str, Any]] = {}
    for arm in _ARMS:
        weights = gate_c._loss_weights("gate_b", arm, efforts=chunk.efforts)
        telemetry[arm] = jax.device_get(
            jax.block_until_ready(
                trainers[arm].train_chunk(
                    chunk.events,
                    chunk.targets,
                    weights,
                    chunk.advance_masks,
                )
            )
        )
        evaluation[arm] = gate_c._evaluate_arm(
            models[arm],
            validation,
            config,
            "gate_b",
            arm,
        )
    return {
        "config": config,
        "models": models,
        "trainers": trainers,
        "shared_copy": shared_copy,
        "initial_sha": initial_sha,
        "initial_adam_zero": initial_adam_zero,
        "final_sha": {arm: _parameter_digest(model) for arm, model in models.items()},
        "frozen_write_before": frozen_write_before,
        "frozen_write_after": np.asarray(
            _parameter_states(models["frozen_write"])["memory_write_scale"].value
        ),
        "telemetry": telemetry,
        "evaluation": evaluation,
    }


@pytest.fixture(scope="module")
def reduced_gate_a_full_legacy_run() -> dict[str, Any]:
    config = _reduced_gate_c_config()
    data = gate_c._regenerate_gate_a_data(config)
    arms = ("full", "legacy")
    models = {
        arm: gate_c._new_model_for_arm(
            config,
            "gate_a",
            arm,
            batch_size=config.gate_a_config.batch_size,
        )
        for arm in arms
    }
    gate_c._copy_shared_initialization(models["full"], models["legacy"])
    trainers = {
        arm: gate_c._make_arm_trainer(models[arm], config, "gate_a", arm)
        for arm in arms
    }
    advances = np.ones(data.training_events.shape[:3], dtype=np.bool_)
    telemetry: dict[str, Mapping[str, Any]] = {}
    evaluation: dict[str, Mapping[str, Any]] = {}
    initial_sha = {arm: _parameter_digest(model) for arm, model in models.items()}
    for arm in arms:
        weights = gate_c._loss_weights(
            "gate_a", arm, efforts=np.asarray([1], dtype=np.int32)
        )
        telemetry[arm] = jax.device_get(
            jax.block_until_ready(
                trainers[arm].train_chunk(
                    data.training_events,
                    data.training_targets,
                    weights,
                    advances,
                )
            )
        )
        evaluation[arm] = gate_c._evaluate_arm(
            models[arm],
            data,
            config,
            "gate_a",
            arm,
        )
    return {
        "models": models,
        "trainers": trainers,
        "initial_sha": initial_sha,
        "final_sha": {arm: _parameter_digest(model) for arm, model in models.items()},
        "telemetry": telemetry,
        "evaluation": evaluation,
    }


@requires_gate_c
def test_reduced_gate_b_runs_all_five_real_pp_prop_arms(
    reduced_gate_b_arm_run: dict[str, Any],
) -> None:
    run = reduced_gate_b_arm_run
    trainers = run["trainers"]
    models = run["models"]
    assert run["shared_copy"]["all_equal"] is True
    assert len({id(trainer) for trainer in trainers.values()}) == len(_ARMS)
    assert len({id(trainer.learner) for trainer in trainers.values()}) == len(_ARMS)
    assert len({id(trainer.optimizer) for trainer in trainers.values()}) == len(_ARMS)
    assert all(run["initial_adam_zero"].values())

    for arm in _ARMS:
        trainer = trainers[arm]
        available = tuple(sorted(_parameter_states(models[arm])))
        expected_optimizer_paths = gate_c._optimizer_parameter_paths(available, arm)
        assert trainer.algorithm == "production_pp_prop"
        assert tuple(trainer.optimizer_parameter_paths) == expected_optimizer_paths
        assert tuple(trainer.excluded_optimizer_paths) == gate_c.ARM_SPECS[
            arm
        ].optimizer_excluded_paths
        assert trainer.compiler["available"] is True
        assert _telemetry_is_finite(run["telemetry"][arm])
        assert run["initial_sha"][arm] != run["final_sha"][arm]
        evaluation = run["evaluation"][arm]
        assert evaluation["finite"] is True
        assert set(evaluation["efforts"]) == {"1", "2", "4", "8"}

    np.testing.assert_array_equal(run["frozen_write_before"], 1.0)
    np.testing.assert_array_equal(
        run["frozen_write_after"], run["frozen_write_before"]
    )
    assert "memory_write_scale" not in trainers[
        "frozen_write"
    ].optimizer_parameter_paths
    assert "memory_write_scale" in trainers[
        "frozen_write"
    ].compiler["compiled_parameter_paths"]


@requires_gate_c
def test_frozen_write_first_update_changes_only_the_excluded_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _reduced_gate_c_config()
    config = dataclasses.replace(
        base,
        gate_b_config=dataclasses.replace(
            base.gate_b_config,
            clip_norm=1e-6,
        ),
    )
    schedule, _ = gate_c._regenerate_gate_b_data(config)
    chunk = gate_b._encode_training_chunk(
        next(gate_b._iter_schedule_chunks(schedule, config.gate_b_config)),
        config.gate_b_config,
    )
    full = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "full",
        batch_size=config.gate_b_config.batch_size,
    )
    frozen = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "frozen_write",
        batch_size=config.gate_b_config.batch_size,
    )
    _assert_parameter_values_equal(full, frozen, _FULL_PATHS)
    clipped_path_sets: list[tuple[str, ...]] = []
    real_clip_grad_norm = brainstate.nn.clip_grad_norm

    def record_clip_paths(gradients: Mapping[Any, Any], max_norm: float) -> Any:
        clipped_path_sets.append(
            tuple(sorted(gate_a._path(path) for path in gradients))
        )
        return real_clip_grad_norm(gradients, max_norm)

    monkeypatch.setattr(brainstate.nn, "clip_grad_norm", record_clip_paths)
    full_trainer = gate_c._make_arm_trainer(full, config, "gate_b", "full")
    frozen_trainer = gate_c._make_arm_trainer(
        frozen,
        config,
        "gate_b",
        "frozen_write",
    )
    weights = gate_c._loss_weights("gate_b", "full", efforts=chunk.efforts)

    jax.block_until_ready(
        full_trainer.train_chunk(
            chunk.events,
            chunk.targets,
            weights,
            chunk.advance_masks,
        )
    )
    jax.block_until_ready(
        frozen_trainer.train_chunk(
            chunk.events,
            chunk.targets,
            weights,
            chunk.advance_masks,
        )
    )

    retained_paths = tuple(
        path for path in _FULL_PATHS if path != "memory_write_scale"
    )
    assert clipped_path_sets == [_FULL_PATHS, _FULL_PATHS]
    _assert_parameter_values_equal(full, frozen, retained_paths)
    assert not np.array_equal(
        np.asarray(_parameter_states(full)["memory_write_scale"].value),
        np.asarray(_parameter_states(frozen)["memory_write_scale"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(_parameter_states(frozen)["memory_write_scale"].value),
        1.0,
    )


@requires_gate_c
def test_reduced_gate_a_runs_real_full_and_legacy_pp_prop_arms(
    reduced_gate_a_full_legacy_run: dict[str, Any],
) -> None:
    run = reduced_gate_a_full_legacy_run
    for arm in ("full", "legacy"):
        assert run["trainers"][arm].algorithm == "production_pp_prop"
        assert _telemetry_is_finite(run["telemetry"][arm])
        assert run["initial_sha"][arm] != run["final_sha"][arm]
        evaluation = run["evaluation"][arm]
        assert evaluation["all_compact_logits_finite"] is True
        assert set(evaluation["depths"]) == {"0", "1"}


@requires_gate_c
def test_core_contract_inputs_fail_closed_before_execution() -> None:
    with pytest.raises(TypeError, match="gate_a_config"):
        gate_c.GateCConfig(gate_a_config=object())
    with pytest.raises(TypeError, match="gate_b_config"):
        gate_c.GateCConfig(gate_b_config=object())
    for field, invalid, error in (
        ("oracle_validation_index", True, TypeError),
        ("oracle_effort", 1.5, TypeError),
        ("oracle_effort", -1, ValueError),
        ("gradient_chunk_size", 0, ValueError),
    ):
        with pytest.raises(error):
            gate_c.GateCConfig(**{field: invalid})

    for invalid in ("unknown", None):
        with pytest.raises(ValueError, match="unknown Gate C arm"):
            gate_c._loss_weights(
                "gate_a",
                invalid,
                efforts=np.asarray([1], dtype=np.int32),
            )
        with pytest.raises(ValueError, match="unknown Gate C regime"):
            gate_c._loss_weights(
                invalid,
                "full",
                efforts=np.asarray([1], dtype=np.int32),
            )
    for efforts, error in (
        (np.asarray(1, dtype=np.int32), ValueError),
        (np.asarray([True]), ValueError),
        (np.asarray([1.0], dtype=np.float32), TypeError),
        (np.asarray([], dtype=np.int32), ValueError),
        (np.asarray([3], dtype=np.int32), ValueError),
    ):
        with pytest.raises(error):
            gate_c._loss_weights("gate_b", "full", efforts=efforts)

    for helper in (
        gate_c._regenerate_gate_a_data,
        gate_c._regenerate_gate_b_data,
        gate_c._schedule_identity_report,
        gate_c._oracle_contract,
    ):
        with pytest.raises(TypeError, match="GateCConfig"):
            helper(object())
    with pytest.raises(RuntimeError, match="no array leaves"):
        gate_c._tree_telemetry(())
    with pytest.raises(TypeError, match="LatentWorkspaceModel"):
        gate_c._make_arm_trainer(
            object(),
            gate_c.GateCConfig(),
            "gate_a",
            "full",
        )

    out_of_range_gate_a = {
        "depths": {
            "1": {
                "intact": {"accuracy": 1.1},
                "shuffled": {"accuracy": 0.0},
            }
        }
    }
    valid_gate_b = {
        "efforts": {
            str(effort): {"intact": {"accuracy": 0.5}}
            for effort in (1, 2, 4, 8)
        }
    }
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gate_c._metric_summary(out_of_range_gate_a, valid_gate_b)
    with pytest.raises(ValueError, match="exactly the five"):
        gate_c._blocking_margin_report({})
    qualification = gate_c._qualification_report({}, config=object())
    assert qualification["passed"] is False
    assert not any(qualification["criteria"].values())


@requires_gate_c
def test_shared_initialization_rejects_type_topology_and_geometry_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_gate_c_config()
    full = gate_c._new_model_for_arm(config, "gate_b", "full", batch_size=2)
    legacy_model = gate_c._new_model_for_arm(
        config, "gate_b", "legacy", batch_size=2
    )
    with pytest.raises(TypeError, match="workspace models"):
        gate_c._copy_shared_initialization(object(), legacy_model)
    with pytest.raises(ValueError, match="exactly the shared paths"):
        gate_c._copy_shared_initialization(full, full)

    real_states = gate_c._parameter_states_by_path

    def missing_canonical_path(model: LatentWorkspaceModel) -> dict[str, Any]:
        states = real_states(model)
        if model is full:
            states.pop(_SHARED_PATHS[0])
        return states

    with monkeypatch.context() as context:
        context.setattr(
            gate_c,
            "_parameter_states_by_path",
            missing_canonical_path,
        )
        with pytest.raises(ValueError, match="canonical model is missing"):
            gate_c._copy_shared_initialization(full, legacy_model)

    structure_drift = gate_c._new_model_for_arm(
        config, "gate_b", "legacy", batch_size=2
    )
    structure_state = _parameter_states(structure_drift)[
        "readout_projection/weight"
    ]
    structure_state.value = (structure_state.value,)
    with pytest.raises(ValueError, match="structure differs"):
        gate_c._copy_shared_initialization(full, structure_drift)

    geometry_drift = gate_c._new_model_for_arm(
        config, "gate_b", "legacy", batch_size=2
    )
    geometry_state = _parameter_states(geometry_drift)["readout_projection/weight"]
    geometry_state.value = jax.tree.map(
        lambda leaf: jnp.zeros(
            (np.asarray(leaf).size + 1,),
            dtype=np.asarray(leaf).dtype,
        ),
        geometry_state.value,
    )
    with pytest.raises(ValueError, match="geometry differs"):
        gate_c._copy_shared_initialization(full, geometry_drift)


@requires_gate_c
def test_evaluation_rejects_cross_regime_data_before_model_execution() -> None:
    config = _reduced_gate_c_config()
    gate_a_data = gate_c._regenerate_gate_a_data(config)
    _, gate_b_data = gate_c._regenerate_gate_b_data(config)
    gate_a_model = gate_c._new_model_for_arm(
        config,
        "gate_a",
        "full",
        batch_size=config.gate_a_config.batch_size,
    )
    gate_b_model = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "full",
        batch_size=config.gate_b_config.batch_size,
    )

    with pytest.raises(TypeError, match="Gate A evaluation requires BindingData"):
        gate_c._evaluate_arm(
            gate_a_model,
            gate_b_data,
            config,
            "gate_a",
            "full",
        )
    with pytest.raises(TypeError, match="Gate B evaluation requires DepthValidationData"):
        gate_c._evaluate_arm(
            gate_b_model,
            gate_a_data,
            config,
            "gate_b",
            "full",
        )


@requires_gate_c
def test_gradient_evidence_rejects_empty_nonnumeric_and_shape_drift() -> None:
    for invalid in ((), {"value": "not numeric"}):
        with pytest.raises((TypeError, ValueError)):
            gate_c._gradient_path_sha256("path", invalid)
    for invalid_path in ("", None):
        with pytest.raises(TypeError, match="nonempty string"):
            gate_c._gradient_path_sha256(
                invalid_path,
                np.asarray([1.0], dtype=np.float32),
            )
    for invalid_gradients in ({}, []):
        with pytest.raises(TypeError, match="nonempty mapping"):
            gate_c._gradient_global_sha256(invalid_gradients)
    with pytest.raises(ValueError, match="same shape"):
        gate_c._gradient_comparison(
            np.zeros((2,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
        )


@requires_gate_c
def test_shared_parameter_digest_framing_is_exact_and_order_independent() -> None:
    values = {
        "alpha/path": {
            "z": np.asarray([1.5, -2.0], dtype=np.float32),
            "a": np.asarray([[3.0]], dtype=np.float16),
        },
        "beta": (np.asarray(0.25, dtype=np.float32),),
    }

    assert gate_c._shared_path_sha256(
        "alpha/path", values["alpha/path"]
    ) == "c0c0b4c8132ed2f5a742712350397b6f9986c453b122dc7c0e81fdf4651e88ea"
    assert gate_c._shared_path_sha256(
        "beta", values["beta"]
    ) == "fd457b69655788bb537a79c5cca2ad9942681e5300a19d90126ab431730c5247"
    expected = "a01cb823743564571f5bc82e44eeb9adb14b61f710915a4ab9cc5bc8e1867cf6"
    assert gate_c._shared_global_sha256(values) == expected
    assert gate_c._shared_global_sha256(dict(reversed(values.items()))) == expected
    assert gate_c._shared_global_sha256(values) == _manual_shared_global_digest(
        values
    )


@requires_gate_c
def test_shared_parameter_digest_rejects_invalid_paths_and_leaves() -> None:
    for invalid in (
        (),
        np.asarray([True]),
        np.asarray([math.nan], dtype=np.float32),
        np.asarray([math.inf], dtype=np.float32),
    ):
        with pytest.raises((TypeError, ValueError)):
            gate_c._shared_path_sha256("path", invalid)
    for invalid_path in ("", None):
        with pytest.raises(TypeError, match="nonempty string"):
            gate_c._shared_path_sha256(
                invalid_path,
                np.asarray([1.0], dtype=np.float32),
            )
    for invalid_values in ({}, []):
        with pytest.raises(TypeError, match="nonempty mapping"):
            gate_c._shared_global_sha256(invalid_values)


@pytest.fixture(scope="module")
def reduced_gate_c_initialization_subject() -> dict[str, Any]:
    config = _reduced_gate_c_config()
    model = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "full",
        batch_size=config.gate_b_config.batch_size,
    )
    trainer = gate_c._make_arm_trainer(model, config, "gate_b", "full")
    return {"config": config, "model": model, "trainer": trainer}


@requires_gate_c
def test_optimizer_initial_state_report_is_fresh_zero_finite_and_exactly_framed(
    reduced_gate_c_initialization_subject: dict[str, Any],
) -> None:
    trainer = reduced_gate_c_initialization_subject["trainer"]

    report = gate_c._optimizer_initial_state_report(
        trainer,
        regime="gate_b",
        arm="full",
    )

    leaves = jax.tree.leaves(trainer.optimizer.opt_state.value)
    assert set(report) == {
        "included",
        "excluded",
        "fresh_state_finite",
        "fresh_state_all_zero",
        "state_leaf_count",
        "value_count",
        "state_sha256",
        "executed_updates",
    }
    assert report["included"] == list(trainer.optimizer_parameter_paths)
    assert report["excluded"] == []
    assert report["fresh_state_finite"] is True
    assert report["fresh_state_all_zero"] is True
    assert report["state_leaf_count"] == len(leaves)
    assert report["value_count"] == sum(np.asarray(leaf).size for leaf in leaves)
    assert report["state_sha256"] == _manual_optimizer_state_digest(
        trainer,
        regime="gate_b",
        arm="full",
    )
    assert report["executed_updates"] == 0


@requires_gate_c
def test_initialization_topology_report_binds_model_tree_and_compiler(
    reduced_gate_c_initialization_subject: dict[str, Any],
) -> None:
    model = reduced_gate_c_initialization_subject["model"]
    trainer = reduced_gate_c_initialization_subject["trainer"]

    report = gate_c._initialization_topology_report(
        model,
        trainer,
        regime="gate_b",
        tree="canonical_full",
    )

    parameter_values = legacy._parameter_values(model)
    assert set(report) == {
        "fresh_model",
        "model_seed",
        "memory_read_policy",
        "model_config",
        "parameter_paths",
        "parameter_count",
        "parameter_sha256",
        "parameters_finite",
        "compiler",
    }
    assert report["fresh_model"] is True
    assert report["model_seed"] == model.config.seed
    assert report["memory_read_policy"] == "full"
    assert report["model_config"] == dataclasses.asdict(model.config)
    assert report["parameter_paths"] == list(_FULL_PATHS)
    assert report["parameter_count"] == sum(
        np.asarray(leaf).size
        for value in parameter_values.values()
        for leaf in jax.tree.leaves(value)
    )
    assert report["parameter_sha256"] == _parameter_digest(model)
    assert report["parameters_finite"] is True
    assert report["compiler"] == trainer.compiler


@requires_gate_c
def test_reduced_gate_c_initialization_is_isolated_and_has_no_behavioral_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_gate_c_config()
    prerequisites = {
        "gate_a": {"target": "formal_gate_a", "authenticated": True},
        "gate_b": {"target": "formal_gate_b", "authenticated": True},
    }
    source_start = {"head": "a" * 40, "clean": True}
    source_end = {"head": "a" * 40, "clean": True, "phase": "end"}
    source_files = {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in (
            "examples/pp_prop/latent_workspace_model.py",
            "examples/pp_prop/latent_workspace_task.py",
            "examples/pp_prop/latent_workspace_binding_control.py",
            "examples/pp_prop/latent_workspace_binding_gate.py",
            "examples/pp_prop/latent_workspace_depth_gate.py",
            "examples/pp_prop/latent_workspace_ablation_gate.py",
        )
    }
    environment = {
        "gpu_authenticated": True,
        "image_digest": "sha256:" + "b" * 64,
    }
    events: list[tuple[Any, ...]] = []
    models: dict[tuple[str, str], LatentWorkspaceModel] = {}
    trainers: dict[tuple[str, str], Any] = {}
    trainer_parameter_sha: dict[tuple[str, str], str] = {}
    topology_reports: dict[tuple[str, str], Mapping[str, Any]] = {}
    optimizer_reports: dict[tuple[str, str], Mapping[str, Any]] = {}
    real_new_model = gate_c._new_model_for_arm
    real_copy_shared = gate_c._copy_shared_initialization
    real_make_trainer = gate_c._make_arm_trainer
    real_topology_report = gate_c._initialization_topology_report
    real_optimizer_report = gate_c._optimizer_initial_state_report

    def normalize(*args: Any, **kwargs: Any) -> dict[str, Any]:
        supplied = args[0] if args else kwargs["prerequisites"]
        assert supplied == prerequisites
        events.append(("normalize",))
        return {
            "gate_a": dict(prerequisites["gate_a"]),
            "gate_b": dict(prerequisites["gate_b"]),
        }

    def new_model(
        runner_config: Any,
        regime: str,
        arm: str,
        *,
        batch_size: int,
    ) -> LatentWorkspaceModel:
        events.append(("model", regime, arm))
        model = real_new_model(
            runner_config,
            regime,
            arm,
            batch_size=batch_size,
        )
        assert (regime, arm) not in models
        models[(regime, arm)] = model
        return model

    def copy_shared(
        canonical: LatentWorkspaceModel,
        legacy_model: LatentWorkspaceModel,
    ) -> dict[str, Any]:
        regime = next(
            name
            for (name, arm), model in models.items()
            if arm == "legacy" and model is legacy_model
        )
        events.append(("copy_shared", regime))
        return real_copy_shared(canonical, legacy_model)

    def no_behavior(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("initialization admission must not execute behavior")

    def make_trainer(
        model: LatentWorkspaceModel,
        runner_config: Any,
        regime: str,
        arm: str,
    ) -> Any:
        events.append(("trainer", regime, arm))
        trainer = real_make_trainer(model, runner_config, regime, arm)
        trainer.train_chunk = no_behavior
        trainers[(regime, arm)] = trainer
        trainer_parameter_sha[(regime, arm)] = _parameter_digest(model)
        return trainer

    def topology_report(
        model: LatentWorkspaceModel,
        trainer: Any,
        *,
        regime: str,
        tree: str,
    ) -> dict[str, Any]:
        events.append(("topology", regime, tree))
        result = real_topology_report(
            model,
            trainer,
            regime=regime,
            tree=tree,
        )
        topology_reports[(regime, tree)] = result
        return result

    def optimizer_report(
        trainer: Any,
        *,
        regime: str,
        arm: str,
    ) -> dict[str, Any]:
        events.append(("optimizer", regime, arm))
        result = real_optimizer_report(trainer, regime=regime, arm=arm)
        optimizer_reports[(regime, arm)] = result
        return result

    def source_end_reporter() -> dict[str, Any]:
        events.append(("source_end",))
        return dict(source_end)

    monkeypatch.setattr(gate_c, "_normalized_prerequisites", normalize)
    monkeypatch.setattr(gate_c, "_new_model_for_arm", new_model)
    monkeypatch.setattr(gate_c, "_copy_shared_initialization", copy_shared)
    monkeypatch.setattr(gate_c, "_make_arm_trainer", make_trainer)
    monkeypatch.setattr(
        gate_c,
        "_initialization_topology_report",
        topology_report,
    )
    monkeypatch.setattr(
        gate_c,
        "_optimizer_initial_state_report",
        optimizer_report,
    )
    monkeypatch.setattr(gate_c, "_regenerate_gate_a_data", no_behavior)
    monkeypatch.setattr(gate_c, "_regenerate_gate_b_data", no_behavior)
    monkeypatch.setattr(gate_c, "_evaluate_arm", no_behavior)

    report = gate_c.run_gate_c_initialization(
        config,
        prerequisites=prerequisites,
        source_start=source_start,
        source_end_reporter=source_end_reporter,
        source_files=source_files,
        environment=environment,
    )

    assert set(report) == {
        "schema_version",
        "control",
        "qualification_regime",
        "prerequisites",
        "regimes",
        "initialization",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
    }
    assert report["schema_version"] == 1
    assert report["control"] == "example21_gate_c_initialization_admission"
    assert report["qualification_regime"] == "nonqualifying_abbreviated"
    assert report["prerequisites"] == prerequisites
    assert report["source_start"] == source_start
    assert report["source_end"] == source_end
    assert report["source_files"] == source_files
    assert report["environment"] == environment
    assert report["regimes"] == {
        regime: {
            "spec": dataclasses.asdict(gate_c.REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
        }
        for regime in _REGIMES
    }
    assert report["qualification"] == {
        "criteria": {name: False for name in _INITIALIZATION_CRITERIA},
        "passed": False,
        "interpretation": "gate_c_initialization_admission_failed_stop",
    }

    assert events[0] == ("normalize",)
    assert events[-1] == ("source_end",)
    assert sum(event[0] == "model" for event in events) == 10
    assert sum(event[0] == "trainer" for event in events) == 10
    assert sum(event[0] == "copy_shared" for event in events) == 2
    assert sum(event[0] == "topology" for event in events) == 4
    assert sum(event[0] == "optimizer" for event in events) == 10
    assert len(models) == len(trainers) == len(trainer_parameter_sha) == 10
    assert len({id(model) for model in models.values()}) == 10
    assert len({id(trainer) for trainer in trainers.values()}) == 10
    assert len({id(trainer.learner) for trainer in trainers.values()}) == 10
    assert len({id(trainer.optimizer) for trainer in trainers.values()}) == 10
    assert {
        (
            regime,
            "legacy" if model.config.context_memory_width == 0 else "canonical_full",
        )
        for (regime, _), model in models.items()
    } == {
        ("gate_a", "canonical_full"),
        ("gate_a", "legacy"),
        ("gate_b", "canonical_full"),
        ("gate_b", "legacy"),
    }

    for regime in _REGIMES:
        initialization = report["initialization"][regime]
        assert set(initialization) == {
            "canonical_full",
            "legacy",
            "shared_paths",
            "arm_initialization_refs",
            "optimizer_paths",
        }
        assert initialization["canonical_full"] == topology_reports[
            (regime, "canonical_full")
        ]
        assert initialization["legacy"] == topology_reports[(regime, "legacy")]
        shared = initialization["shared_paths"]
        assert set(shared) == {
            "paths",
            "framing",
            "canonical_path_sha256",
            "legacy_path_sha256",
            "canonical_sha256",
            "legacy_sha256",
            "all_equal",
        }
        assert shared["paths"] == list(_SHARED_PATHS)
        assert shared["framing"] == {
            "path": "example21-gate-c-shared-path-v1",
            "global": "example21-gate-c-shared-global-v1",
        }
        canonical_values = legacy._parameter_values(models[(regime, "full")])
        legacy_values = legacy._parameter_values(models[(regime, "legacy")])
        expected_canonical_paths = {
            path: gate_c._shared_path_sha256(path, canonical_values[path])
            for path in _SHARED_PATHS
        }
        expected_legacy_paths = {
            path: gate_c._shared_path_sha256(path, legacy_values[path])
            for path in _SHARED_PATHS
        }
        assert shared["canonical_path_sha256"] == expected_canonical_paths
        assert shared["legacy_path_sha256"] == expected_legacy_paths
        assert shared["canonical_sha256"] == gate_c._shared_global_sha256(
            {path: canonical_values[path] for path in _SHARED_PATHS}
        )
        assert shared["legacy_sha256"] == gate_c._shared_global_sha256(
            {path: legacy_values[path] for path in _SHARED_PATHS}
        )
        assert shared["canonical_sha256"] == shared["legacy_sha256"]
        assert shared["all_equal"] is True

        refs = initialization["arm_initialization_refs"]
        optimizer_paths = initialization["optimizer_paths"]
        assert set(refs) == set(optimizer_paths) == set(_ARMS)
        for arm in _ARMS:
            tree = "legacy" if arm == "legacy" else "canonical_full"
            assert refs[arm] == {
                "tree": tree,
                "parameter_sha256": trainer_parameter_sha[(regime, arm)],
            }
            assert optimizer_paths[arm] == optimizer_reports[(regime, arm)]
            assert optimizer_paths[arm]["executed_updates"] == 0
            assert optimizer_paths[arm]["fresh_state_finite"] is True
            assert optimizer_paths[arm]["fresh_state_all_zero"] is True
            assert _parameter_digest(models[(regime, arm)]) == (
                trainer_parameter_sha[(regime, arm)]
            )
