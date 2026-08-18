"""Tests for the preregistered Example 21 Gate C causal ablations."""

from __future__ import annotations

import ast
import copy
import dataclasses
import gc
import hashlib
import importlib
import inspect
import json
import math
import re
import weakref
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import brainunit as u
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop import latent_workspace_binding_gate as gate_a
from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher
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
_GATE_C_SOURCE_FILES = (
    "examples/pp_prop/latent_workspace_model.py",
    "examples/pp_prop/latent_workspace_task.py",
    "examples/pp_prop/latent_workspace_binding_control.py",
    "examples/pp_prop/latent_workspace_binding_gate.py",
    "examples/pp_prop/latent_workspace_depth_gate.py",
    "examples/pp_prop/latent_workspace_ablation_gate.py",
)
_GATE_A_REFERENCE = {
    "qualification_passed": True,
    "result_sha256": "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632",
    "manifest_sha256": "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf",
    "source_commit": "4737e9172b1c6ca99347af5b2c83fc795a294a16",
    "bundle_sha256": "ba850a205c4691d573facef7b8e90cabd4824905c73fcd4f6add29293cd95875",
    "preflight_sha256": "d1d54406d0972d52ac10cddec7e6d1ed38c55481d51e21989e444fe7c3f03d08",
    "result_path": (
        "var/example21-binding-gate/"
        "4737e9172b1c6ca99347af5b2c83fc795a294a16-formal-gate-a.json"
    ),
    "manifest_path": (
        "var/example21-binding-gate/"
        "4737e9172b1c6ca99347af5b2c83fc795a294a16-formal-gate-a.manifest.json"
    ),
}
_GATE_B_REFERENCE = {
    "qualification_passed": True,
    "result_sha256": "6456537ea108cea8892d00c8a71c1f647217e074b525bc9ed01b64aef9001766",
    "manifest_sha256": "99c42985e203413eb0600a5dabe321188776eff8058500dc86f4a1618b413eab",
    "source_commit": "dafa64a8b4c3848241baa117affa55b632518a8e",
    "bundle_sha256": "be07e8c92d8deaa94508f34dcee45f5feb09740cb2804778d6280a2fa3c64851",
    "preflight_sha256": "91e86d92670cd33d3f4206ff3d5096e3721104996a9506223a9e34c082dd052f",
    "result_path": (
        "var/example21-depth-gate/"
        "dafa64a8b4c3848241baa117affa55b632518a8e-formal-gate-b.json"
    ),
    "manifest_path": (
        "var/example21-depth-gate/"
        "dafa64a8b4c3848241baa117affa55b632518a8e-formal-gate-b.manifest.json"
    ),
}
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


def _passing_gate_c_source() -> dict[str, Any]:
    head = "c" * 40
    return {
        "asserted_commit": head,
        "asserted_commit_matches_head": True,
        "asserted_dirty": False,
        "asserted_dirty_matches_worktree": True,
        "commit": head,
        "commit_is_valid_40_hex": True,
        "dirty": False,
        "head_command_succeeded": True,
        "status_command_succeeded": True,
        "verified": True,
    }


def _passing_gate_c_environment() -> dict[str, Any]:
    return {
        "backend": "gpu",
        "devices": [
            {
                "device_kind": "synthetic qualifying GPU",
                "id": 0,
                "platform": "gpu",
                "process_index": 0,
            }
        ],
        "image_digest": "sha256:" + "d" * 64,
        "jax": "0.7.2",
        "python": "3.14.6",
    }


def _current_gate_c_source_files() -> dict[str, str]:
    repo_root = Path(gate_c.__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        for relative in _GATE_C_SOURCE_FILES
    }


def _passing_gate_c_compiler(*, legacy_tree: bool) -> dict[str, Any]:
    required = [
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    ]
    if legacy_tree:
        return {
            "available": True,
            "diagnostics": [],
            "compiled_parameter_paths": list(_SHARED_PATHS),
            "required_direct_paths": required,
            "direct_path_status": {path: False for path in required},
            "direct_path_evidence": {path: [] for path in required},
            "hidden_groups": [
                {
                    "index": 0,
                    "hidden_paths": [
                        "ff_syn/post/V",
                        "ff_syn/syn/g",
                        "rec_syn/syn/g",
                    ],
                }
            ],
            "all_required_direct": False,
            "context_memory_isolated_from_workspace_lif": False,
        }
    hidden_groups = [
        {
            "index": 0,
            "hidden_paths": [
                "ff_syn/post/V",
                "ff_syn/syn/g",
                "rec_syn/syn/g",
                "workspace_carrier",
            ],
        },
        {"index": 1, "hidden_paths": ["context_memory"]},
        {"index": 2, "hidden_paths": ["query_encoding"]},
        {"index": 3, "hidden_paths": ["reasoning_query"]},
        {"index": 4, "hidden_paths": ["memory_read"]},
    ]
    return {
        "available": True,
        "diagnostics": [],
        "compiled_parameter_paths": list(_FULL_PATHS),
        "required_direct_paths": required,
        "direct_path_status": {path: True for path in required},
        "direct_path_evidence": {
            "memory_write_scale": [
                {
                    "relation_key": "('context_memory',)",
                    "classification": "all_direct",
                    "hidden_groups": [hidden_groups[1]],
                }
            ],
            "workspace_query_projection/weight": [
                {
                    "relation_key": "('reasoning_query',)",
                    "classification": "all_direct",
                    "hidden_groups": [hidden_groups[3]],
                }
            ],
            "memory_read_projection/weight": [
                {
                    "relation_key": "('ff_syn', 'post', 'V')",
                    "classification": "all_direct",
                    "hidden_groups": [hidden_groups[0]],
                },
                {
                    "relation_key": "('workspace_carrier',)",
                    "classification": "all_direct",
                    "hidden_groups": [hidden_groups[0]],
                },
            ],
        },
        "hidden_groups": hidden_groups,
        "all_required_direct": True,
        "context_memory_isolated_from_workspace_lif": True,
    }


def _shared_digest_from_path_digests(path_digests: Mapping[str, str]) -> str:
    fields = [b"example21-gate-c-shared-global-v1"]
    for path in sorted(path_digests):
        fields.extend(
            (
                path.encode("utf-8"),
                path_digests[path].encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _synthetic_optimizer_state_sha(
    *,
    regime: str,
    arm: str,
    included: tuple[str, ...],
) -> str:
    value = np.asarray(0, dtype=np.int32)
    fields = [
        b"example21-gate-c-optimizer-state-v1",
        regime.encode("utf-8"),
        arm.encode("utf-8"),
        *(path.encode("utf-8") for path in sorted(included)),
        b"0",
        value.dtype.str.encode("ascii"),
        b"",
        value.tobytes(),
    ]
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _passing_optimizer_initialization(
    regime: str,
    arm: str,
) -> dict[str, Any]:
    available = _SHARED_PATHS if arm == "legacy" else _FULL_PATHS
    included = gate_c._optimizer_parameter_paths(available, arm)
    excluded = gate_c.ARM_SPECS[arm].optimizer_excluded_paths
    return {
        "included": list(included),
        "excluded": list(excluded),
        "fresh_state_finite": True,
        "fresh_state_all_zero": True,
        "state_leaf_count": 1,
        "value_count": 1,
        "state_sha256": _synthetic_optimizer_state_sha(
            regime=regime,
            arm=arm,
            included=included,
        ),
        "executed_updates": 0,
    }


def _passing_gate_c_initialization_base(
    config: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    if config is None:
        config = gate_c.GateCConfig()
    initialization: dict[str, Any] = {}
    regimes: dict[str, Any] = {}
    for regime in _REGIMES:
        regime_config = (
            config.gate_a_config if regime == "gate_a" else config.gate_b_config
        )
        spec = gate_c.REGIME_SPECS[regime]
        full_model_config = gate_c._model_config_for_arm(
            config,
            regime,
            "full",
            batch_size=regime_config.batch_size,
        )
        legacy_model_config = gate_c._model_config_for_arm(
            config,
            regime,
            "legacy",
            batch_size=regime_config.batch_size,
        )
        legacy_sha = hashlib.sha256(f"{regime}-legacy-gpu".encode()).hexdigest()
        canonical = {
            "fresh_model": True,
            "model_seed": 2108,
            "memory_read_policy": "full",
            "model_config": dataclasses.asdict(full_model_config),
            "parameter_paths": list(_FULL_PATHS),
            "parameter_count": spec.full_parameter_count,
            "parameter_sha256": spec.full_parameter_sha256,
            "parameters_finite": True,
            "compiler": _passing_gate_c_compiler(legacy_tree=False),
        }
        legacy_topology = {
            "fresh_model": True,
            "model_seed": 2108,
            "memory_read_policy": "full",
            "model_config": dataclasses.asdict(legacy_model_config),
            "parameter_paths": list(_SHARED_PATHS),
            "parameter_count": spec.legacy_parameter_count,
            "parameter_sha256": legacy_sha,
            "parameters_finite": True,
            "compiler": _passing_gate_c_compiler(legacy_tree=True),
        }
        shared_values = {
            path: np.asarray(
                [list(_REGIMES).index(regime), index],
                dtype=np.float32,
            )
            for index, path in enumerate(_SHARED_PATHS)
        }
        path_digests = {
            path: _manual_shared_path_digest(path, value)
            for path, value in shared_values.items()
        }
        shared_sha = _shared_digest_from_path_digests(path_digests)
        initialization[regime] = {
            "canonical_full": canonical,
            "legacy": legacy_topology,
            "shared_paths": {
                "paths": list(_SHARED_PATHS),
                "framing": {
                    "path": "example21-gate-c-shared-path-v1",
                    "global": "example21-gate-c-shared-global-v1",
                },
                "canonical_path_sha256": dict(path_digests),
                "legacy_path_sha256": dict(path_digests),
                "canonical_sha256": shared_sha,
                "legacy_sha256": shared_sha,
                "all_equal": True,
            },
            "arm_initialization_refs": {
                arm: {
                    "tree": "legacy" if arm == "legacy" else "canonical_full",
                    "parameter_sha256": (
                        legacy_sha if arm == "legacy" else spec.full_parameter_sha256
                    ),
                }
                for arm in _ARMS
            },
            "optimizer_paths": {
                arm: _passing_optimizer_initialization(regime, arm)
                for arm in _ARMS
            },
        }
        regimes[regime] = {
            "spec": dataclasses.asdict(spec),
            "config": dataclasses.asdict(regime_config),
        }
    source_start = _passing_gate_c_source()
    report = {
        "schema_version": 1,
        "control": "example21_gate_c_initialization_admission",
        "qualification_regime": config.qualification_regime,
        "prerequisites": {
            "gate_a": dict(_GATE_A_REFERENCE),
            "gate_b": dict(_GATE_B_REFERENCE),
        },
        "regimes": regimes,
        "initialization": initialization,
        "source_start": source_start,
        "source_end": dict(source_start),
        "source_files": _current_gate_c_source_files(),
        "environment": _passing_gate_c_environment(),
    }
    return report, config


def _passing_gate_c_initialization_report(
    config: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    report, config = _passing_gate_c_initialization_base(config)
    report["qualification"] = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )
    return report, config


def _gate_c_initialization_wrapper(report: Mapping[str, Any]) -> dict[str, Any]:
    source_head = str(report["source_start"]["commit"])
    preflight_sha256 = hashlib.sha256(b"gate-c-preflight").hexdigest()
    result_sha256 = gate_b._strict_json_sha256(report)
    bundle_sha256 = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_c_init\0"
            f"{source_head}\0{preflight_sha256}\0{result_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "target": "gate_c_init",
        "source_head": source_head,
        "image_digest": report["environment"]["image_digest"],
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": hashlib.sha256(b"gate-c-manifest").hexdigest(),
        "preflight_sha256": preflight_sha256,
        "result_sha256": result_sha256,
        "admission": copy.deepcopy(report),
    }


def _mutate_gate_c_initialization_report(
    report: dict[str, Any],
    criterion: str,
) -> None:
    if criterion == "schema_and_control":
        report["control"] = "wrong_control"
    elif criterion == "preregistered_regimes":
        report["regimes"]["gate_a"]["spec"]["training_updates"] -= 1
    elif criterion == "gate_a_prerequisite_authenticated":
        report["prerequisites"]["gate_a"]["result_sha256"] = "0" * 64
    elif criterion == "gate_b_prerequisite_authenticated":
        report["prerequisites"]["gate_b"]["result_sha256"] = "0" * 64
    elif criterion == "source_and_gpu_authenticated":
        report["source_end"]["dirty"] = True
    elif criterion == "source_files_exact":
        report["source_files"].pop(_GATE_C_SOURCE_FILES[0])
    elif criterion == "canonical_full_initializations_exact":
        report["initialization"]["gate_a"]["canonical_full"][
            "parameter_count"
        ] += 1
    elif criterion == "legacy_initializations_complete":
        report["initialization"]["gate_b"]["legacy"]["parameter_count"] += 1
    elif criterion == "shared_paths_byte_identical":
        report["initialization"]["gate_a"]["shared_paths"]["all_equal"] = False
    elif criterion == "arm_initialization_refs_exact":
        report["initialization"]["gate_b"]["arm_initialization_refs"][
            "query_only"
        ]["parameter_sha256"] = "0" * 64
    elif criterion == "optimizer_paths_exact":
        report["initialization"]["gate_a"]["optimizer_paths"]["frozen_write"][
            "included"
        ].append("memory_write_scale")
    elif criterion == "fresh_optimizer_states_zero_and_finite":
        report["initialization"]["gate_b"]["optimizer_paths"]["full"][
            "fresh_state_all_zero"
        ] = False
    elif criterion == "compiler_topologies_complete":
        report["initialization"]["gate_a"]["canonical_full"]["compiler"][
            "all_required_direct"
        ] = False
    elif criterion == "no_behavioral_updates":
        report["initialization"]["gate_b"]["optimizer_paths"]["legacy"][
            "executed_updates"
        ] = 1
    else:
        raise AssertionError(f"unhandled criterion {criterion}")


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
            "_actual_schedule_identity_report",
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
            "_validated_gate_c_initialization_admission",
            "run_gate_c_initialization",
            "_source_files_report",
            "write_artifact",
            "_parser",
            "main",
            "_run_gate_c_arm",
            "_formal_arm_initialization_report",
            "_arm_initialization_reproduced",
            "_paired_h0_identity_report",
            "_mechanism_oracle",
            "chunked_online_param_gradients",
            "run_gate_c",
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

    def test_actual_schedule_identity_is_derived_from_generated_reduced_data(
        self,
    ) -> None:
        config = _reduced_gate_c_config()
        gate_a_data = gate_c._regenerate_gate_a_data(config)
        gate_b_schedule, gate_b_validation = gate_c._regenerate_gate_b_data(config)
        gate_b_training = gate_b._encoded_schedule_report(
            gate_b_schedule,
            config.gate_b_config,
        )
        expected = {
            "gate_a": {
                "training_schedule_sha256": legacy._digest_arrays(
                    gate_a_data.training_events,
                    gate_a_data.training_targets,
                    gate_a_data.training_mapping_ids,
                ),
                "validation_schedule_sha256": legacy._digest_arrays(
                    gate_a_data.validation_intact,
                    gate_a_data.validation_targets,
                    gate_a_data.validation_mapping_ids,
                ),
                "training_mapping_ids_sha256": legacy._digest_arrays(
                    gate_a_data.training_mapping_ids.reshape(-1)
                ),
                "validation_mapping_ids_sha256": legacy._digest_arrays(
                    gate_a_data.validation_mapping_ids
                ),
            },
            "gate_b": {
                "training_global_sha256": gate_b_training["global_sha256"],
                "validation_sha256": gate_b._validation_data_report(
                    gate_b_validation
                )["sha256"],
            },
        }

        report = gate_c._actual_schedule_identity_report(
            config,
            gate_a_data,
            (gate_b_schedule, gate_b_validation),
        )

        assert report == expected
        assert report != gate_c._schedule_identity_report(config)

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
        assert report["frozen_write"]["interpretation"] == (
            "learned_memory_write_modulation_necessary"
        )

    def test_frozen_write_interpretation_is_not_shown_if_either_margin_fails(
        self,
    ) -> None:
        metrics = {
            "full": {"binding_gap": 0.80, "depth_accuracy": 0.60},
            "query_only": {"binding_gap": 0.82, "depth_accuracy": 0.45},
            "terminal_only": {"binding_gap": 0.82, "depth_accuracy": 0.50},
            "legacy": {"binding_gap": 0.55, "depth_accuracy": 0.45},
            "frozen_write": {"binding_gap": 0.75, "depth_accuracy": 0.551},
        }

        report = gate_c._blocking_margin_report(metrics)

        assert report["blocking_passed"] is True
        assert report["frozen_write"]["write_modulation_necessary"] is False
        assert report["frozen_write"]["interpretation"] == (
            "learned_memory_write_modulation_not_shown_necessary"
        )

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
        "config": config,
        "data": data,
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
def test_reduced_gate_a_full_retains_binding_state_evidence(
    reduced_gate_a_full_legacy_run: dict[str, Any],
) -> None:
    run = reduced_gate_a_full_legacy_run
    evidence = run["evaluation"]["full"]["binding_state"]

    assert set(evidence) == {
        "applicable_count",
        "intact_shuffled_different_count",
        "every_intact_shuffled_pair_differs",
        "no_context_exact_zero",
        "intact_sha256",
        "shuffled_sha256",
        "no_context_sha256",
        "all_finite",
    }
    assert evidence["applicable_count"] == (
        run["config"].gate_a_config.validation_episodes
    )
    assert 0 <= evidence["intact_shuffled_different_count"] <= evidence[
        "applicable_count"
    ]
    assert evidence["every_intact_shuffled_pair_differs"] is (
        evidence["intact_shuffled_different_count"]
        == evidence["applicable_count"]
    )
    assert evidence["no_context_exact_zero"] is True
    assert evidence["all_finite"] is True
    for name in ("intact_sha256", "shuffled_sha256", "no_context_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", evidence[name])


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
def test_initialization_topology_rejects_trainer_from_another_model(
    reduced_gate_c_initialization_subject: dict[str, Any],
) -> None:
    config = reduced_gate_c_initialization_subject["config"]
    trainer = reduced_gate_c_initialization_subject["trainer"]
    different_model = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "full",
        batch_size=config.gate_b_config.batch_size,
    )

    with pytest.raises(ValueError, match="same model"):
        gate_c._initialization_topology_report(
            different_model,
            trainer,
            regime="gate_b",
            tree="canonical_full",
        )


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


@requires_gate_c
def test_normalized_prerequisites_require_exact_frozen_compact_references() -> None:
    prerequisites = {
        "gate_a": dict(_GATE_A_REFERENCE),
        "gate_b": dict(_GATE_B_REFERENCE),
    }

    normalized = gate_c._normalized_prerequisites(prerequisites)

    assert normalized == prerequisites
    assert normalized is not prerequisites
    assert normalized["gate_a"] is not prerequisites["gate_a"]
    assert normalized["gate_b"] is not prerequisites["gate_b"]
    for gate_name in ("gate_a", "gate_b"):
        for field in prerequisites[gate_name]:
            mutated = copy.deepcopy(prerequisites)
            if field == "qualification_passed":
                mutated[gate_name][field] = 1
            elif field == "source_commit":
                mutated[gate_name][field] = "0" * 40
            elif field.endswith("_path"):
                mutated[gate_name][field] = "wrong/path.json"
            else:
                mutated[gate_name][field] = "0" * 64
            with pytest.raises((TypeError, ValueError)):
                gate_c._normalized_prerequisites(mutated)
    for malformed in (
        {"gate_a": dict(_GATE_A_REFERENCE)},
        {
            "gate_a": dict(_GATE_A_REFERENCE),
            "gate_b": dict(_GATE_B_REFERENCE),
            "extra": {},
        },
        {"gate_a": [], "gate_b": dict(_GATE_B_REFERENCE)},
    ):
        with pytest.raises((TypeError, ValueError)):
            gate_c._normalized_prerequisites(malformed)


@requires_gate_c
def test_full_synthetic_gate_c_initialization_genuinely_passes() -> None:
    report, config = _passing_gate_c_initialization_report()

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert report["qualification"] == qualification
    assert qualification == {
        "criteria": {name: True for name in _INITIALIZATION_CRITERIA},
        "passed": True,
        "interpretation": "gate_c_initialization_admission_passed",
    }


@pytest.mark.parametrize("criterion", _INITIALIZATION_CRITERIA)
@requires_gate_c
def test_each_initialization_criterion_fails_on_one_field_mutation(
    criterion: str,
) -> None:
    report, config = _passing_gate_c_initialization_report()
    _mutate_gate_c_initialization_report(report, criterion)

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["criteria"][criterion] is False
    assert {
        name for name, passed in qualification["criteria"].items() if not passed
    } == {criterion}
    assert qualification["interpretation"] == (
        "gate_c_initialization_admission_failed_stop"
    )


@requires_gate_c
def test_initialization_qualification_rejects_extra_prerequisite() -> None:
    report, config = _passing_gate_c_initialization_report()
    report["prerequisites"]["untrusted_extra"] = {
        "qualification_passed": True,
    }

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["criteria"]["gate_a_prerequisite_authenticated"] is False
    assert qualification["criteria"]["gate_b_prerequisite_authenticated"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "source_extra_key",
        "environment_extra_key",
        "device_extra_key",
        "device_boolean_id",
        "device_boolean_process_index",
        "empty_jax_version",
    ),
)
@requires_gate_c
def test_source_and_gpu_evidence_rejects_schema_and_boolean_confusion(
    mutation: str,
) -> None:
    report, config = _passing_gate_c_initialization_report()
    if mutation == "source_extra_key":
        report["source_start"]["untrusted"] = True
    elif mutation == "environment_extra_key":
        report["environment"]["untrusted"] = True
    elif mutation == "device_extra_key":
        report["environment"]["devices"][0]["untrusted"] = True
    elif mutation == "device_boolean_id":
        report["environment"]["devices"][0]["id"] = True
    elif mutation == "device_boolean_process_index":
        report["environment"]["devices"][0]["process_index"] = False
    else:
        report["environment"]["jax"] = ""

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["criteria"]["source_and_gpu_authenticated"] is False


@requires_gate_c
def test_legacy_compiler_rejects_fabricated_context_memory_group() -> None:
    report, config = _passing_gate_c_initialization_report()
    report["initialization"]["gate_a"]["legacy"]["compiler"][
        "hidden_groups"
    ].append({"index": 1, "hidden_paths": ["context_memory"]})

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["criteria"]["compiler_topologies_complete"] is False
    assert qualification["criteria"]["legacy_initializations_complete"] is True


@pytest.mark.parametrize(
    "diagnostic",
    (
        "warning",
        {"kind": "compile", "level": "warning"},
        {
            "kind": "compile",
            "level": "warning",
            "message": "synthetic",
            "extra": True,
        },
        {"kind": "compile", "level": 1, "message": "synthetic"},
        {
            "kind": "compile",
            "level": "warning",
            "message": "synthetic",
            "weight_path": 1,
        },
    ),
)
@requires_gate_c
def test_compiler_diagnostics_require_exact_string_mapping(
    diagnostic: Any,
) -> None:
    report, config = _passing_gate_c_initialization_report()
    report["initialization"]["gate_a"]["canonical_full"]["compiler"][
        "diagnostics"
    ] = [diagnostic]

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["criteria"]["compiler_topologies_complete"] is False
    assert qualification["criteria"]["canonical_full_initializations_exact"] is True


@requires_gate_c
def test_nested_behavioral_evidence_invalidates_initialization_schema() -> None:
    report, config = _passing_gate_c_initialization_report()
    report["initialization"]["gate_a"]["behavioral_metrics"] = {
        "accuracy": 1.0
    }

    qualification = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["criteria"]["no_behavioral_updates"] is False


@requires_gate_c
def test_authenticated_initialization_wrapper_recomputes_hashes_and_qualification() -> None:
    report, config = _passing_gate_c_initialization_report()
    wrapper = _gate_c_initialization_wrapper(report)

    admission = gate_c._validated_gate_c_initialization_admission(
        wrapper,
        config,
        source_start=report["source_start"],
        environment=report["environment"],
        source_files=report["source_files"],
        require_pass=True,
    )

    assert admission == report
    assert set(wrapper) == {
        "target",
        "source_head",
        "image_digest",
        "bundle_sha256",
        "manifest_sha256",
        "preflight_sha256",
        "result_sha256",
        "admission",
    }
    assert wrapper["result_sha256"] == gate_b._strict_json_sha256(report)
    assert wrapper["bundle_sha256"] == hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_c_init\0"
            f"{wrapper['source_head']}\0{wrapper['preflight_sha256']}\0"
            f"{wrapper['result_sha256']}"
        ).encode("utf-8")
    ).hexdigest()


@requires_gate_c
def test_authenticated_initialization_wrapper_rejects_tampering_and_staleness() -> None:
    report, config = _passing_gate_c_initialization_report()
    wrapper = _gate_c_initialization_wrapper(report)
    call_kwargs = {
        "source_start": report["source_start"],
        "environment": report["environment"],
        "source_files": report["source_files"],
        "require_pass": True,
    }

    extra = copy.deepcopy(wrapper)
    extra["extra"] = True
    with pytest.raises(ValueError, match="authenticated"):
        gate_c._validated_gate_c_initialization_admission(
            extra,
            config,
            **call_kwargs,
        )

    stale_result = copy.deepcopy(wrapper)
    stale_result["admission"]["control"] = "tampered"
    with pytest.raises(ValueError, match="result digest"):
        gate_c._validated_gate_c_initialization_admission(
            stale_result,
            config,
            **call_kwargs,
        )

    stale_bundle = copy.deepcopy(wrapper)
    stale_bundle["bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bundle digest"):
        gate_c._validated_gate_c_initialization_admission(
            stale_bundle,
            config,
            **call_kwargs,
        )

    stale_qualification_report = copy.deepcopy(report)
    stale_qualification_report["qualification"]["passed"] = False
    stale_qualification = _gate_c_initialization_wrapper(
        stale_qualification_report
    )
    with pytest.raises(ValueError, match="qualification is stale"):
        gate_c._validated_gate_c_initialization_admission(
            stale_qualification,
            config,
            **call_kwargs,
        )

    wrong_head = copy.deepcopy(wrapper)
    wrong_head["source_head"] = "e" * 40
    wrong_head["bundle_sha256"] = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_c_init\0"
            f"{wrong_head['source_head']}\0{wrong_head['preflight_sha256']}\0"
            f"{wrong_head['result_sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="source or image"):
        gate_c._validated_gate_c_initialization_admission(
            wrong_head,
            config,
            **call_kwargs,
        )

    wrong_image = copy.deepcopy(wrapper)
    wrong_image["image_digest"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="source or image"):
        gate_c._validated_gate_c_initialization_admission(
            wrong_image,
            config,
            **call_kwargs,
        )

    wrong_source_files = dict(report["source_files"])
    wrong_source_files[_GATE_C_SOURCE_FILES[0]] = "0" * 64
    with pytest.raises(ValueError, match="source files"):
        gate_c._validated_gate_c_initialization_admission(
            wrapper,
            config,
            source_start=report["source_start"],
            environment=report["environment"],
            source_files=wrong_source_files,
            require_pass=True,
        )

    dirty_formal_source = copy.deepcopy(report["source_start"])
    dirty_formal_source["dirty"] = True
    with pytest.raises(ValueError):
        gate_c._validated_gate_c_initialization_admission(
            wrapper,
            config,
            source_start=dirty_formal_source,
            environment=report["environment"],
            source_files=report["source_files"],
            require_pass=True,
        )

    cpu_formal_environment = copy.deepcopy(report["environment"])
    cpu_formal_environment["backend"] = "cpu"
    with pytest.raises(ValueError):
        gate_c._validated_gate_c_initialization_admission(
            wrapper,
            config,
            source_start=report["source_start"],
            environment=cpu_formal_environment,
            source_files=report["source_files"],
            require_pass=True,
        )


@requires_gate_c
def test_authenticated_initialization_wrapper_require_pass_is_strict() -> None:
    report, config = _passing_gate_c_initialization_report()
    _mutate_gate_c_initialization_report(
        report,
        "canonical_full_initializations_exact",
    )
    report["qualification"] = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )
    wrapper = _gate_c_initialization_wrapper(report)
    kwargs = {
        "source_start": report["source_start"],
        "environment": report["environment"],
        "source_files": report["source_files"],
    }

    assert gate_c._validated_gate_c_initialization_admission(
        wrapper,
        config,
        require_pass=False,
        **kwargs,
    ) == report
    with pytest.raises(ValueError, match="did not pass"):
        gate_c._validated_gate_c_initialization_admission(
            wrapper,
            config,
            require_pass=True,
            **kwargs,
        )


@requires_gate_c
def test_wrapper_require_pass_false_does_not_relax_inner_schema() -> None:
    report, config = _passing_gate_c_initialization_report()
    report["untrusted_extra"] = True
    report["qualification"] = gate_c._gate_c_initialization_qualification(
        report,
        config=config,
    )
    wrapper = _gate_c_initialization_wrapper(report)

    with pytest.raises(ValueError, match="schema"):
        gate_c._validated_gate_c_initialization_admission(
            wrapper,
            config,
            source_start=report["source_start"],
            environment=report["environment"],
            source_files=report["source_files"],
            require_pass=False,
        )


@requires_gate_c
def test_initialization_authentication_precedes_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_gate_c_config()
    valid_prerequisites = {
        "gate_a": dict(_GATE_A_REFERENCE),
        "gate_b": dict(_GATE_B_REFERENCE),
    }
    source = _passing_gate_c_source()
    environment = _passing_gate_c_environment()
    model_calls = 0

    def forbidden_model(*args: Any, **kwargs: Any) -> Any:
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("model construction preceded authentication")

    monkeypatch.setattr(gate_c, "_new_model_for_arm", forbidden_model)
    invalid_prerequisites = copy.deepcopy(valid_prerequisites)
    invalid_prerequisites["gate_b"]["result_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        gate_c.run_gate_c_initialization(
            config,
            prerequisites=invalid_prerequisites,
            source_start=source,
            source_end_reporter=lambda: source,
            source_files=_current_gate_c_source_files(),
            environment=environment,
        )
    assert model_calls == 0

    invalid_source = dict(source)
    invalid_source["dirty"] = True
    with pytest.raises(RuntimeError, match="source"):
        gate_c.run_gate_c_initialization(
            config,
            prerequisites=valid_prerequisites,
            source_start=invalid_source,
            source_end_reporter=lambda: source,
            source_files=_current_gate_c_source_files(),
            environment=environment,
        )
    assert model_calls == 0


@requires_gate_c
def test_initialization_rejects_nonmapping_source_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_gate_c_config()
    values = {
        path: np.asarray([index], dtype=np.float32)
        for index, path in enumerate(_FULL_PATHS)
    }

    def fake_model(
        runner_config: Any,
        regime: str,
        arm: str,
        *,
        batch_size: int,
    ) -> dict[str, Any]:
        del runner_config, batch_size
        return {"regime": regime, "arm": arm}

    def parameter_values(model: Mapping[str, str]) -> dict[str, Any]:
        paths = _SHARED_PATHS if model["arm"] == "legacy" else _FULL_PATHS
        return {path: values[path] for path in paths}

    monkeypatch.setattr(gate_c, "_new_model_for_arm", fake_model)
    monkeypatch.setattr(gate_c, "_copy_shared_initialization", lambda *args: {})
    monkeypatch.setattr(gate_c, "_make_arm_trainer", lambda *args: object())
    monkeypatch.setattr(
        gate_c,
        "_initialization_topology_report",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        gate_c,
        "_optimizer_initial_state_report",
        lambda trainer, *, regime, arm: _passing_optimizer_initialization(
            regime,
            arm,
        ),
    )
    monkeypatch.setattr(gate_c.legacy, "_parameter_values", parameter_values)

    with pytest.raises(TypeError, match="source_end.*mapping"):
        gate_c.run_gate_c_initialization(
            config,
            prerequisites={
                "gate_a": dict(_GATE_A_REFERENCE),
                "gate_b": dict(_GATE_B_REFERENCE),
            },
            source_start=_passing_gate_c_source(),
            source_end_reporter=lambda: [],
            source_files=_current_gate_c_source_files(),
            environment=_passing_gate_c_environment(),
        )


def _gate_c_cli_host_argv(
    *,
    head: str = "c" * 40,
) -> tuple[launcher.LaunchConfig, launcher.TargetPaths, list[str]]:
    repo_root = Path(gate_c.__file__).resolve().parents[2]
    config = launcher.LaunchConfig(
        target="gate_c_init",
        repo_root=repo_root,
        output_dir=repo_root / "var" / "example21-causal-gate",
    )
    paths = launcher.target_paths(config, head, "gate_c_init")
    gate_a_paths = launcher._gate_a_artifact_paths(config)
    gate_b_paths = launcher._formal_gate_b_artifact_paths(config)
    argv = [
        "--target",
        "gate_c_init",
        "--gate-a-result",
        str(gate_a_paths.result),
        "--gate-a-manifest",
        str(gate_a_paths.manifest),
        "--gate-b-manifest",
        str(gate_b_paths.manifest),
        "--output",
        str(paths.result),
    ]
    return config, paths, argv


def _formal_gate_c_cli_host_argv(
    *,
    head: str = "c" * 40,
) -> tuple[launcher.LaunchConfig, Path, Path, list[str]]:
    config, _, _ = _gate_c_cli_host_argv(head=head)
    gate_a_paths = launcher._gate_a_artifact_paths(config)
    gate_b_paths = launcher._formal_gate_b_artifact_paths(config)
    initialization_manifest = (
        config.output_dir / f"{head}-gate-c-init.manifest.json"
    )
    output = config.output_dir / f"{head}-formal-gate-c.json"
    argv = [
        "--target",
        "formal_gate_c",
        "--gate-a-result",
        str(gate_a_paths.result),
        "--gate-a-manifest",
        str(gate_a_paths.manifest),
        "--gate-b-manifest",
        str(gate_b_paths.manifest),
        "--gate-c-init-manifest",
        str(initialization_manifest),
        "--output",
        str(output),
    ]
    return config, initialization_manifest, output, argv


@requires_gate_c
def test_gate_c_source_file_report_hashes_exact_live_six_file_set() -> None:
    report = gate_c._source_files_report()

    assert report == _current_gate_c_source_files()
    assert tuple(report) == _GATE_C_SOURCE_FILES


@requires_gate_c
def test_gate_c_artifact_writer_is_atomic_deterministic_and_strict(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "gate-c.json"
    value = {"z": 1, "a": [True, None]}

    written = gate_c.write_artifact(value, destination)

    expected = (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    )
    assert written == destination
    assert destination.read_text(encoding="utf-8") == expected
    assert not destination.with_suffix(".json.tmp").exists()

    invalid = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="JSON|range|compliant"):
        gate_c.write_artifact({"loss": math.nan}, invalid)
    assert not invalid.exists()
    assert not invalid.with_suffix(".json.tmp").exists()


def _gate_c2_controls_writer_value() -> dict[str, Any]:
    return {
        "schema_version": gate_c.GATE_C2_CONTROLS_SCHEMA_VERSION,
        "control": gate_c.GATE_C2_CONTROLS_CONTROL,
        "qualification_regime": "preregistered_full_pretraining_control",
        "a": {"finite": 1.25, "flags": [True, False, None]},
        "z": 7,
    }


@requires_gate_c
def test_gate_c2_controls_writer_streams_exact_compact_strict_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _gate_c2_controls_writer_value()
    expected = (
        "".join(
            json.JSONEncoder(
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).iterencode(value)
        ).encode("utf-8")
        + b"\n"
    )

    def forbidden_dumps(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise AssertionError("Gate C2 controls must stream JSON chunks")

    monkeypatch.setattr(gate_c.json, "dumps", forbidden_dumps)
    destination = tmp_path / "gate-c2-controls.json"

    written = gate_c.write_artifact(value, destination)

    payload = destination.read_bytes()
    assert written == destination
    assert payload == expected
    assert payload.endswith(b"\n")
    assert b"\n" not in payload[:-1]
    assert json.loads(payload) == value
    assert not destination.with_suffix(".json.tmp").exists()


@requires_gate_c
def test_gate_c2_controls_writer_enforces_exact_size_including_newline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert gate_c.GATE_C2_CONTROLS_MAX_JSON_BYTES == 201_326_592
    value = _gate_c2_controls_writer_value()
    compact = "".join(
        json.JSONEncoder(
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).iterencode(value)
    ).encode("utf-8")

    exact = tmp_path / "exact.json"
    monkeypatch.setattr(
        gate_c,
        "GATE_C2_CONTROLS_MAX_JSON_BYTES",
        len(compact) + 1,
    )
    gate_c.write_artifact(value, exact)
    assert exact.stat().st_size == len(compact) + 1

    oversized = tmp_path / "oversized.json"
    previous = b"authenticated-previous-artifact\n"
    oversized.write_bytes(previous)
    monkeypatch.setattr(
        gate_c,
        "GATE_C2_CONTROLS_MAX_JSON_BYTES",
        len(compact),
    )

    with pytest.raises(ValueError, match="JSON|size|limit|bytes|MiB"):
        gate_c.write_artifact(value, oversized)

    assert oversized.read_bytes() == previous
    assert not oversized.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    "control",
    (
        "example21_gate_c_initialization_admission",
        "example21_pp_prop_learnability_gate_c",
        "example21_pp_prop_learnability_gate_c2",
    ),
)
@requires_gate_c
def test_non_control_gate_c_writer_retains_v1_indented_bytes(
    control: str,
    tmp_path: Path,
) -> None:
    value = {"schema_version": 1, "control": control, "z": 1, "a": [True]}
    destination = tmp_path / f"{control}.json"
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")

    gate_c.write_artifact(value, destination)

    assert destination.read_bytes() == expected


@requires_gate_c
def test_large_gate_c2_controls_streamed_payload_stays_below_cap(
    tmp_path: Path,
) -> None:
    read_hex = "00" * (512 * 32 * np.dtype(np.float32).itemsize)
    memory_hex = "00" * (512 * 32 * 32 * np.dtype(np.float32).itemsize)
    read_record = {
        "dtype": "<f4",
        "shape": [512, 32],
        "data_hex": read_hex,
        "sha256": "0" * 64,
        "finite_count": 512 * 32,
        "nonfinite_count": 0,
    }
    memory_record = {
        "dtype": "<f4",
        "shape": [512, 32, 32],
        "data_hex": memory_hex,
        "sha256": "1" * 64,
        "finite_count": 512 * 32 * 32,
        "nonfinite_count": 0,
    }
    ticks = {
        f"H{index}": {
            "selected_read": read_record,
            "selected_drive": read_record,
        }
        for index in range(1, 9)
    }
    value = {
        "schema_version": gate_c.GATE_C2_CONTROLS_SCHEMA_VERSION,
        "control": gate_c.GATE_C2_CONTROLS_CONTROL,
        "regimes": {
            "gate_b": {
                "query_only_latent_no_read": {
                    "streams": {
                        name: ticks
                        for name in ("intact", "shuffled", "no_context")
                    },
                    "perturbations": {
                        name: {"replacement": memory_record}
                        for name in ("plus_7", "minus_7")
                    },
                }
            }
        },
    }
    destination = tmp_path / "large-streamed-controls.json"

    gate_c.write_artifact(value, destination)

    assert 10 * 1024 * 1024 < destination.stat().st_size
    assert destination.stat().st_size < gate_c.GATE_C2_CONTROLS_MAX_JSON_BYTES
    assert json.loads(destination.read_bytes()) == value


@requires_gate_c
def test_complete_gate_c2_controls_fixture_streams_below_cap_and_roundtrips(
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    tmp_path: Path,
) -> None:
    admission, no_read_reports = passing_gate_c2_no_read_reports
    report = _passing_gate_c2_controls_report(admission, no_read_reports)
    destination = tmp_path / "complete-gate-c2-controls.json"

    written = gate_c.write_artifact(report, destination)

    payload = destination.read_bytes()
    assert written == destination
    assert payload.endswith(b"\n")
    assert b"\n" not in payload[:-1]
    assert len(payload) < gate_c.GATE_C2_CONTROLS_MAX_JSON_BYTES
    decoded = json.loads(payload)
    assert gate_a._json_exact(decoded, report)
    assert not destination.with_suffix(".json.tmp").exists()


@requires_gate_c
def test_gate_c_parser_accepts_exact_launcher_module_argv() -> None:
    config, paths, _ = _gate_c_cli_host_argv()
    command = launcher.gate_command(
        config,
        image_id="sha256:" + "d" * 64,
        head="c" * 40,
        paths=paths,
        git_dir_in_container="/git-common/worktrees/gate-c",
        admission_manifests=None,
    )
    python_index = command.index("python")
    assert command[python_index : python_index + 3] == [
        "python",
        "-m",
        "examples.pp_prop.latent_workspace_ablation_gate",
    ]

    parsed = gate_c._parser().parse_args(command[python_index + 3 :])

    assert set(vars(parsed)) == {
        "target",
        "gate_a_result",
        "gate_a_manifest",
        "gate_b_manifest",
        "gate_c_init_manifest",
        "output",
    }
    assert parsed.target == "gate_c_init"
    assert parsed.gate_c_init_manifest is None
    assert parsed.output == Path(str(paths.container_result))


@requires_gate_c
def test_gate_c_parser_exposes_no_run_overrides() -> None:
    _, _, argv = _gate_c_cli_host_argv()

    with pytest.raises(SystemExit) as override_error:
        gate_c._parser().parse_args([*argv, "--training-updates", "2"])
    assert override_error.value.code == 2


@requires_gate_c
def test_gate_c_parser_accepts_exact_formal_launcher_argv() -> None:
    _, initialization_manifest, output, argv = _formal_gate_c_cli_host_argv()

    parsed = gate_c._parser().parse_args(argv)

    assert set(vars(parsed)) == {
        "target",
        "gate_a_result",
        "gate_a_manifest",
        "gate_b_manifest",
        "gate_c_init_manifest",
        "output",
    }
    assert parsed.target == "formal_gate_c"
    assert parsed.gate_c_init_manifest == initialization_manifest
    assert parsed.output == output


@pytest.mark.parametrize(
    "mutation",
    ("gate_a_result", "gate_a_manifest", "gate_b_manifest", "output"),
)
@requires_gate_c
def test_gate_c_cli_rejects_nonfixed_artifact_paths(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths, argv = _gate_c_cli_host_argv()
    flag = {
        "gate_a_result": "--gate-a-result",
        "gate_a_manifest": "--gate-a-manifest",
        "gate_b_manifest": "--gate-b-manifest",
        "output": "--output",
    }[mutation]
    argv[argv.index(flag) + 1] = str(
        paths.result.with_name(f"wrong-{mutation}.json")
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_source_report",
        lambda: _passing_gate_c_source(),
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_environment_report",
        lambda: _passing_gate_c_environment(),
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_require_authenticated_gpu_launch",
        lambda *args: None,
    )

    with pytest.raises(ValueError, match="fixed|path|output"):
        gate_c.main(argv)


@requires_gate_c
def test_gate_c_init_cli_authenticates_and_emits_full_admission(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, paths, argv = _gate_c_cli_host_argv()
    passing, _ = _passing_gate_c_initialization_report()
    prerequisites = copy.deepcopy(passing["prerequisites"])
    source_reports = iter([passing["source_start"], passing["source_end"]])
    events: list[str] = []
    captured: dict[str, Any] = {}

    def source_report() -> dict[str, Any]:
        value = copy.deepcopy(next(source_reports))
        events.append("source_start" if not events else "source_end")
        return value

    def environment_report() -> dict[str, Any]:
        events.append("environment")
        return copy.deepcopy(passing["environment"])

    def require_launch(
        source: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> None:
        events.append("gpu_authenticated")
        assert source == passing["source_start"]
        assert environment == passing["environment"]

    def load_prerequisites(
        actual: launcher.LaunchConfig,
    ) -> dict[str, Any]:
        events.append("prerequisites_loaded")
        assert actual.target == "gate_c_init"
        assert actual.repo_root == config.repo_root
        assert actual.output_dir == config.output_dir
        return copy.deepcopy(prerequisites)

    def source_files_report() -> dict[str, str]:
        events.append("source_files")
        return copy.deepcopy(passing["source_files"])

    def run_initialization(
        actual_config: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("core_run")
        captured.update(config=actual_config, **kwargs)
        reporter = kwargs["source_end_reporter"]
        assert callable(reporter)
        result = copy.deepcopy(passing)
        result["source_end"] = reporter()
        result["qualification"] = {
            "criteria": {
                name: False for name in _INITIALIZATION_CRITERIA
            },
            "passed": False,
            "interpretation": "gate_c_initialization_admission_failed_stop",
        }
        return result

    def write_artifact(value: dict[str, Any], output: Path) -> Path:
        events.append("artifact_written")
        captured.update(value=value, output=output)
        return output

    monkeypatch.setattr(gate_c.gate_a, "_source_report", source_report)
    monkeypatch.setattr(
        gate_c.gate_a,
        "_environment_report",
        environment_report,
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_require_authenticated_gpu_launch",
        require_launch,
    )
    monkeypatch.setattr(
        launcher,
        "_load_gate_c_prerequisites",
        load_prerequisites,
    )
    monkeypatch.setattr(
        gate_c,
        "_source_files_report",
        source_files_report,
        raising=False,
    )
    monkeypatch.setattr(
        gate_c,
        "run_gate_c_initialization",
        run_initialization,
    )
    monkeypatch.setattr(
        gate_c,
        "write_artifact",
        write_artifact,
        raising=False,
    )

    assert gate_c.main(argv) == 0

    assert events == [
        "source_start",
        "environment",
        "gpu_authenticated",
        "prerequisites_loaded",
        "source_files",
        "core_run",
        "source_end",
        "artifact_written",
    ]
    assert captured["config"] == gate_c.GateCConfig()
    assert captured["prerequisites"] == prerequisites
    assert captured["source_start"] == passing["source_start"]
    assert callable(captured["source_end_reporter"])
    assert captured["source_files"] == passing["source_files"]
    assert captured["environment"] == passing["environment"]
    assert captured["output"] == paths.result
    assert captured["value"]["qualification"]["passed"] is False
    stdout = capsys.readouterr().out
    assert str(paths.result) in stdout
    assert '"passed": false' in stdout


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_gate_a_result",
        "wrong_gate_a_manifest",
        "wrong_gate_b_manifest",
        "wrong_gate_c_init_manifest",
        "wrong_output",
        "init_with_init_manifest",
        "formal_without_init_manifest",
    ),
)
@requires_gate_c
def test_gate_c_cli_rejects_nonfixed_or_target_incompatible_formal_paths(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if mutation == "init_with_init_manifest":
        _, paths, argv = _gate_c_cli_host_argv()
        argv.extend(["--gate-c-init-manifest", str(paths.manifest)])
    else:
        _, initialization_manifest, output, argv = _formal_gate_c_cli_host_argv()
        if mutation == "formal_without_init_manifest":
            index = argv.index("--gate-c-init-manifest")
            del argv[index : index + 2]
        else:
            flag = {
                "wrong_gate_a_result": "--gate-a-result",
                "wrong_gate_a_manifest": "--gate-a-manifest",
                "wrong_gate_b_manifest": "--gate-b-manifest",
                "wrong_gate_c_init_manifest": "--gate-c-init-manifest",
                "wrong_output": "--output",
            }[mutation]
            argv[argv.index(flag) + 1] = str(
                output.with_name(f"wrong-{mutation}.json")
            )
            assert initialization_manifest.name.endswith(
                "-gate-c-init.manifest.json"
            )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_source_report",
        lambda: _passing_gate_c_source(),
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_environment_report",
        lambda: _passing_gate_c_environment(),
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_require_authenticated_gpu_launch",
        lambda *args: None,
    )

    with pytest.raises(ValueError, match="fixed|manifest|target|output"):
        gate_c.main(argv)


@requires_gate_c
def test_formal_gate_c_cli_authenticates_before_core_and_writes_scientific_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, initialization_manifest, output, argv = _formal_gate_c_cli_host_argv()
    passing, _ = _passing_formal_gate_c_report()
    prerequisites = copy.deepcopy(passing["prerequisites"])
    source_reports = iter([passing["source_start"], passing["source_end"]])
    events: list[str] = []
    captured: dict[str, Any] = {}

    def source_report() -> dict[str, Any]:
        value = copy.deepcopy(next(source_reports))
        events.append("source_start" if not events else "source_end")
        return value

    def environment_report() -> dict[str, Any]:
        events.append("environment")
        return copy.deepcopy(passing["environment"])

    def require_launch(
        source: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> None:
        events.append("gpu_authenticated")
        assert source == passing["source_start"]
        assert environment == passing["environment"]

    def load_prerequisites(
        actual: launcher.LaunchConfig,
        *,
        head: str,
        image_id: str,
    ) -> dict[str, Any]:
        events.append("prerequisites_authenticated")
        assert actual.target == "formal_gate_c"
        assert actual.repo_root == config.repo_root
        assert actual.output_dir == config.output_dir
        assert head == passing["source_start"]["commit"]
        assert image_id == passing["environment"]["image_digest"]
        assert Path(argv[argv.index("--gate-c-init-manifest") + 1]) == (
            initialization_manifest
        )
        return copy.deepcopy(prerequisites)

    def source_files_report() -> dict[str, str]:
        events.append("source_files")
        return copy.deepcopy(passing["source_files"])

    def run_gate(
        actual_config: gate_c.GateCConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("core_run")
        captured.update(config=actual_config, **kwargs)
        assert events[:4] == [
            "source_start",
            "environment",
            "gpu_authenticated",
            "prerequisites_authenticated",
        ]
        reporter = kwargs["source_end_reporter"]
        assert callable(reporter)
        result = copy.deepcopy(passing)
        result["prerequisites"] = copy.deepcopy(kwargs["prerequisites"])
        result["source_start"] = copy.deepcopy(kwargs["source_start"])
        result["source_end"] = reporter()
        result["qualification"] = {
            "criteria": {
                name: False for name in gate_c.QUALIFICATION_CRITERIA
            },
            "passed": False,
            "interpretation": (
                "gate_c_failed_stop_no_causal_mechanism_conclusion"
            ),
        }
        return result

    def write_artifact(value: dict[str, Any], destination: Path) -> Path:
        events.append("artifact_written")
        captured.update(value=value, output=destination)
        return destination

    monkeypatch.setattr(gate_c.gate_a, "_source_report", source_report)
    monkeypatch.setattr(
        gate_c.gate_a,
        "_environment_report",
        environment_report,
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_require_authenticated_gpu_launch",
        require_launch,
    )
    monkeypatch.setattr(
        launcher,
        "_load_formal_gate_c_prerequisites",
        load_prerequisites,
        raising=False,
    )
    monkeypatch.setattr(gate_c, "_source_files_report", source_files_report)
    monkeypatch.setattr(gate_c, "run_gate_c", run_gate)
    monkeypatch.setattr(gate_c, "write_artifact", write_artifact)

    assert gate_c.main(argv) == 0

    assert events == [
        "source_start",
        "environment",
        "gpu_authenticated",
        "prerequisites_authenticated",
        "source_files",
        "core_run",
        "source_end",
        "artifact_written",
    ]
    assert captured["config"] == gate_c.GateCConfig()
    assert captured["prerequisites"] == prerequisites
    assert captured["source_start"] == passing["source_start"]
    assert callable(captured["source_end_reporter"])
    assert captured["source_files"] == passing["source_files"]
    assert captured["environment"] == passing["environment"]
    assert captured["output"] == output
    assert captured["value"]["qualification"]["passed"] is False
    stdout = capsys.readouterr().out
    assert str(output) in stdout
    assert '"passed": false' in stdout


@requires_gate_c
def test_formal_gate_c_cli_stops_before_core_when_prerequisite_authentication_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, argv = _formal_gate_c_cli_host_argv()
    passing, _ = _passing_formal_gate_c_report()
    events: list[str] = []

    monkeypatch.setattr(
        gate_c.gate_a,
        "_source_report",
        lambda: copy.deepcopy(passing["source_start"]),
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_environment_report",
        lambda: copy.deepcopy(passing["environment"]),
    )
    monkeypatch.setattr(
        gate_c.gate_a,
        "_require_authenticated_gpu_launch",
        lambda *args: events.append("gpu_authenticated"),
    )

    def reject_prerequisites(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("prerequisites_rejected")
        raise launcher.ProvenanceError("Gate C initialization authentication failed")

    monkeypatch.setattr(
        launcher,
        "_load_formal_gate_c_prerequisites",
        reject_prerequisites,
        raising=False,
    )
    monkeypatch.setattr(
        gate_c,
        "run_gate_c",
        lambda *args, **kwargs: pytest.fail("core ran before authentication"),
    )
    monkeypatch.setattr(
        gate_c,
        "write_artifact",
        lambda *args, **kwargs: pytest.fail("artifact written after auth failure"),
    )

    with pytest.raises(launcher.ProvenanceError, match="authentication failed"):
        gate_c.main(argv)

    assert events == ["gpu_authenticated", "prerequisites_rejected"]


def _minimal_formal_arm_initialization_report(
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> dict[str, Any]:
    reference = admission["initialization"][regime][
        "arm_initialization_refs"
    ][arm]
    paths = _SHARED_PATHS if arm == "legacy" else _FULL_PATHS
    topology = admission["initialization"][regime][reference["tree"]]
    return {
        "initialization": {
            "tree": reference["tree"],
            "parameter_sha256": reference["parameter_sha256"],
            "parameter_count": topology["parameter_count"],
            "parameter_paths": list(paths),
            "shared_paths": copy.deepcopy(
                admission["initialization"][regime]["shared_paths"]
            ),
        },
        "optimizer": _passing_optimizer_initialization(regime, arm),
        "compiler": copy.deepcopy(topology["compiler"]),
    }


def _minimal_formal_arm_report(
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
    *,
    initialization_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        "full": {"binding_gap": 0.7, "depth_accuracy": 0.8},
        "query_only": {"binding_gap": 0.7, "depth_accuracy": 0.6},
        "terminal_only": {"binding_gap": 0.7, "depth_accuracy": 0.65},
        "legacy": {"binding_gap": 0.4, "depth_accuracy": 0.6},
        "frozen_write": {"binding_gap": 0.65, "depth_accuracy": 0.75},
    }
    initial = (
        _minimal_formal_arm_initialization_report(admission, regime, arm)
        if initialization_report is None
        else copy.deepcopy(dict(initialization_report))
    )
    arm_metrics = metrics[arm]
    if regime == "gate_a":
        evaluation = {
            "finite": True,
            "depths": {
                "1": {
                    "intact": {
                        "accuracy": arm_metrics["binding_gap"] + 0.1,
                    },
                    "shuffled": {"accuracy": 0.1},
                }
            },
        }
    else:
        evaluation = {
            "finite": True,
            "efforts": {
                str(effort): {
                    "intact": {
                        "accuracy": arm_metrics["depth_accuracy"],
                    }
                }
                for effort in gate_b.QUALIFYING_EFFORTS
            },
        }
    return {
        **initial,
        "training": {
            "algorithm": "production_pp_prop",
            "executed_updates": 1,
            "finite": True,
        },
        "parameter_movement": {
            "parameter_count": 1,
            "l2_delta": 1.0,
        },
        "evaluation": evaluation,
        "metrics": {},
    }


def _formal_accuracy_metric(
    correct: int,
    *,
    checkpoint: int,
    identity: str,
) -> dict[str, Any]:
    count = 512
    lower, upper = legacy._wilson_interval(correct, count)
    return {
        "correct": correct,
        "count": count,
        "accuracy": correct / count,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "prediction_histogram": [52, 52, *([51] * 8)],
        "prediction_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "checkpoint": checkpoint,
    }


def _formal_paired_diagnostic(
    *,
    identity: str,
    different: int,
) -> dict[str, Any]:
    left_sha256 = hashlib.sha256(f"{identity}:left".encode()).hexdigest()
    return {
        "applicable_count": 512,
        "different_count": different,
        "every_pair_differs": different == 512,
        "mean_l2_difference": 1.0 if different else 0.0,
        "left_sha256": left_sha256,
        "right_sha256": (
            hashlib.sha256(f"{identity}:right".encode()).hexdigest()
            if different
            else left_sha256
        ),
        "no_context_l2_norm": 0.0,
    }


def _formal_binding_diagnostic(arm: str) -> dict[str, Any]:
    different = 0 if arm == "legacy" else 512
    memory = _formal_paired_diagnostic(
        identity=f"{arm}:memory",
        different=different,
    )
    memory.update(
        {
            "intact_shuffled_different_count": different,
            "every_intact_shuffled_pair_differs": different == 512,
            "no_context_exact_zero": True,
            "no_context_sha256": hashlib.sha256(
                f"{arm}:memory:no-context".encode()
            ).hexdigest(),
            "intact_l2_norm": 1.0 if arm != "legacy" else 0.0,
            "shuffled_l2_norm": 1.0 if arm != "legacy" else 0.0,
            "no_context_l2_norm": 0.0,
            "storage_contract": (
                "one final S_K snapshot per arm; S_K is not stacked"
            ),
        }
    )
    read_by_depth = {
        str(depth): _formal_paired_diagnostic(
            identity=f"{arm}:read:{depth}",
            different=(0 if arm == "query_only" and depth == 1 else different),
        )
        for depth in range(2)
    }
    return {
        "all_state_tensors_finite": True,
        "memory": memory,
        "read_by_depth": read_by_depth,
        "workspace_by_depth": {
            str(depth): _formal_paired_diagnostic(
                identity=f"{arm}:workspace:{depth}",
                different=different,
            )
            for depth in range(2)
        },
    }


def _formal_gate_a_evaluation(arm: str, intact_correct: int) -> dict[str, Any]:
    depths: dict[str, Any] = {}
    for depth in range(2):
        depth_intact = 300 if depth == 0 else intact_correct
        depths[str(depth)] = {
            stream: _formal_accuracy_metric(
                depth_intact if stream == "intact" else 50,
                checkpoint=depth,
                identity=f"{arm}:gate-a:{depth}:{stream}",
            )
            for stream in ("intact", "shuffled", "no_context")
        }
    diagnostic = _formal_binding_diagnostic(arm)
    memory = diagnostic["memory"]
    binding_state = {
        "applicable_count": memory["applicable_count"],
        "intact_shuffled_different_count": memory[
            "intact_shuffled_different_count"
        ],
        "every_intact_shuffled_pair_differs": memory[
            "every_intact_shuffled_pair_differs"
        ],
        "no_context_exact_zero": memory["no_context_exact_zero"],
        "intact_sha256": memory["left_sha256"],
        "shuffled_sha256": memory["right_sha256"],
        "no_context_sha256": memory["no_context_sha256"],
        "all_finite": diagnostic["all_state_tensors_finite"],
    }
    final = depths["1"]
    return {
        "finite": True,
        "all_compact_logits_finite": True,
        "all_state_tensors_finite": True,
        "depths": depths,
        "intact": copy.deepcopy(final["intact"]),
        "shuffled": copy.deepcopy(final["shuffled"]),
        "no_context": copy.deepcopy(final["no_context"]),
        "intact_minus_shuffled": (
            final["intact"]["accuracy"] - final["shuffled"]["accuracy"]
        ),
        "binding_state": binding_state,
        "binding_diagnostic": diagnostic,
    }


def _formal_gate_b_evaluation(arm: str, matching_correct: int) -> dict[str, Any]:
    depths: dict[str, Any] = {}
    h0_identity = f"{arm}:gate-b:h0:intact"
    for depth in range(9):
        depths[str(depth)] = {
            stream: _formal_accuracy_metric(
                (205 if depth == 0 else matching_correct)
                if stream == "intact"
                else 50,
                checkpoint=depth,
                identity=(
                    h0_identity
                    if depth == 0 and stream == "intact"
                    else f"{arm}:gate-b:{depth}:{stream}"
                ),
            )
            for stream in ("intact", "shuffled", "no_context")
        }
    efforts: dict[str, Any] = {}
    for effort in gate_b.QUALIFYING_EFFORTS:
        matching = depths[str(effort)]
        h0_final = _formal_accuracy_metric(
            205,
            checkpoint=0,
            identity=h0_identity,
        )
        efforts[str(effort)] = {
            "intact": copy.deepcopy(matching["intact"]),
            "shuffled": copy.deepcopy(matching["shuffled"]),
            "no_context": copy.deepcopy(matching["no_context"]),
            "h0_final_target": h0_final,
            "intact_minus_h0": (
                matching["intact"]["accuracy"] - h0_final["accuracy"]
            ),
            "intact_minus_shuffled": (
                matching["intact"]["accuracy"]
                - matching["shuffled"]["accuracy"]
            ),
        }
    return {
        "finite": True,
        "h0_proper": copy.deepcopy(depths["0"]["intact"]),
        "depths": depths,
        "efforts": efforts,
    }


def _formal_parameter_movement(
    parameter_count: int,
    paths: tuple[str, ...],
    *,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    path_counts = {
        "color_factor_head/weight": 144_480,
        "ff_syn/comm/weight": 83_968 if regime == "gate_a" else 96_256,
        "height_head/weight": 3_870,
        "memory_read_projection/weight": 65_536,
        "memory_write_scale": 1_024,
        "readout_projection/weight": 262_272,
        "rec_syn/comm/weight": 16_384,
        "width_head/weight": 3_870,
        "workspace_query_projection/weight": 65_536,
    }
    assert sum(path_counts[path] for path in paths) == parameter_count
    path_reports: dict[str, Any] = {}
    squared = 0.0
    for path in paths:
        zero_movement = (
            path in {"height_head/weight", "width_head/weight"}
            or (arm == "frozen_write" and path == "memory_write_scale")
            or (
                arm == "query_only"
                and path == "workspace_query_projection/weight"
            )
        )
        l2_delta = 0.0 if zero_movement else 1.0
        squared += l2_delta * l2_delta
        path_reports[path] = {
            "l2_delta": l2_delta,
            "parameter_count": path_counts[path],
        }
    return {
        "l2_delta": math.sqrt(squared),
        "parameter_count": parameter_count,
        "paths": path_reports,
    }


def _formal_training_report(
    config: Any,
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
    execution_index: int,
    data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    reference = admission["initialization"][regime][
        "arm_initialization_refs"
    ][arm]
    if regime == "gate_a":
        weights = gate_c._loss_weights(
            regime,
            arm,
            efforts=np.ones((1,), dtype=np.int32),
        )
        chunk_count = 1
    else:
        efforts = np.resize(
            np.asarray(gate_b.QUALIFYING_EFFORTS, dtype=np.int32),
            regime_config.training_updates,
        )
        weights = gate_c._loss_weights(regime, arm, efforts=efforts)
        chunk_count = regime_config.staging_chunk_count
    losses = [0.5] * regime_config.training_updates
    categories = (
        "logits",
        "model_states",
        "gradients",
        "pp_prop_traces",
        "adam",
        "parameters",
    )
    return {
        "algorithm": "production_pp_prop",
        "execution_index": execution_index,
        "intervention": dataclasses.asdict(gate_c.ARM_SPECS[arm]),
        "data_identity": copy.deepcopy(data_identity),
        "executed_updates": regime_config.training_updates,
        "batch_size": regime_config.batch_size,
        "chunk_count": chunk_count,
        "cold_compile_and_train_seconds": 1.0,
        "initial_parameter_sha256": reference["parameter_sha256"],
        "final_parameter_sha256": hashlib.sha256(
            f"{regime}:{arm}:final".encode()
        ).hexdigest(),
        "optimizer_final_step": regime_config.training_updates,
        "loss_weights": {
            "dtype": weights.dtype.str,
            "shape": list(weights.shape),
            "sha256": legacy._digest_arrays(weights),
        },
        "compile_warnings": [],
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail_64_mean_loss": 0.5,
        "finite": {name: True for name in categories},
        "max_abs": {name: 1.0 for name in categories},
        "value_count": {name: 1 for name in categories},
        "frozen_write": {
            "applicable": arm == "frozen_write",
            "all_ones_before": arm != "legacy",
            "all_ones_after": arm == "frozen_write",
            "excluded_from_optimizer": arm == "frozen_write",
        },
    }


def _formal_mechanism_oracle() -> dict[str, Any]:
    full_gradients = {
        path: np.asarray([index + 1.0], dtype=np.float32)
        for index, path in enumerate(_FULL_PATHS)
    }
    comparisons: dict[str, Any] = {}
    for arm in ("query_only", "terminal_only"):
        required_paths = (
            [
                "memory_read_projection/weight",
                "workspace_query_projection/weight",
            ]
            if arm == "query_only"
            else []
        )
        arm_gradients = {
            path: (
                np.zeros_like(full_gradients[path])
                if path in required_paths
                else full_gradients[path]
                * np.float32(0.8 if arm == "query_only" else 0.7)
            )
            for path in _FULL_PATHS
        }
        paths: dict[str, Any] = {}
        for path in _FULL_PATHS:
            paths[path] = {
                **gate_c._gradient_comparison(
                    full_gradients[path],
                    arm_gradients[path],
                ),
                "full_sha256": gate_c._gradient_path_sha256(
                    path,
                    full_gradients[path],
                ),
                "arm_sha256": gate_c._gradient_path_sha256(
                    path,
                    arm_gradients[path],
                ),
            }
        comparisons[arm] = {
            "global": {
                **gate_c._gradient_comparison(full_gradients, arm_gradients),
                "full_sha256": gate_c._gradient_global_sha256(
                    full_gradients
                ),
                "arm_sha256": gate_c._gradient_global_sha256(arm_gradients),
            },
            "paths": paths,
            "required_paths": required_paths,
            "required_paths_passed": True,
            "passed": True,
        }
    return {
        "contract": gate_c._oracle_contract(gate_c.GateCConfig()),
        "objective": {
            "wrapper": "sqrt_weight_times_sqrt_cross_entropy",
            "unsupervised_output_exact_zero": True,
        },
        "gradient_chunk_size": 1,
        "comparisons": comparisons,
        "complete": True,
    }


def _formal_paired_h0_identity(
    admission: Mapping[str, Any],
    regime: str,
) -> dict[str, Any]:
    parameter_sha256 = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]
    return {
        "checkpoint": 0,
        "initialization_parameter_sha256": {
            "full": parameter_sha256,
            "query_only": parameter_sha256,
        },
        "streams": {
            stream: {
                "full_compact_sha256": hashlib.sha256(
                    f"{regime}:{stream}:compact".encode()
                ).hexdigest(),
                "query_only_compact_sha256": hashlib.sha256(
                    f"{regime}:{stream}:compact".encode()
                ).hexdigest(),
                "full_state_sha256": hashlib.sha256(
                    f"{regime}:{stream}:state".encode()
                ).hexdigest(),
                "query_only_state_sha256": hashlib.sha256(
                    f"{regime}:{stream}:state".encode()
                ).hexdigest(),
                "compact_byte_identical": True,
                "state_byte_identical": True,
            }
            for stream in ("intact", "shuffled", "no_context")
        },
        "passed": True,
    }


def _passing_formal_gate_c_report() -> tuple[dict[str, Any], Any]:
    admission, config = _passing_gate_c_initialization_report()
    schedule = {
        "gate_a": copy.deepcopy(_GATE_A_SCHEDULE_SHA256),
        "gate_b": {
            "training_global_sha256": copy.deepcopy(_GATE_B_TRAINING_SHA256),
            "validation_sha256": copy.deepcopy(_GATE_B_VALIDATION_SHA256),
        },
    }
    gate_a_correct = {
        "full": 460,
        "query_only": 450,
        "terminal_only": 450,
        "legacy": 306,
        "frozen_write": 409,
    }
    gate_b_correct = {
        "full": 410,
        "query_only": 307,
        "terminal_only": 333,
        "legacy": 307,
        "frozen_write": 358,
    }
    arms: dict[str, dict[str, Any]] = {arm: {} for arm in _ARMS}
    for regime_index, regime in enumerate(_REGIMES):
        for arm_index, arm in enumerate(_ARMS):
            initialization = _minimal_formal_arm_initialization_report(
                admission,
                regime,
                arm,
            )
            reference = admission["initialization"][regime][
                "arm_initialization_refs"
            ][arm]
            topology = admission["initialization"][regime][reference["tree"]]
            paths = _SHARED_PATHS if arm == "legacy" else _FULL_PATHS
            evaluation = (
                _formal_gate_a_evaluation(arm, gate_a_correct[arm])
                if regime == "gate_a"
                else _formal_gate_b_evaluation(arm, gate_b_correct[arm])
            )
            arms[arm][regime] = {
                **initialization,
                "training": _formal_training_report(
                    config,
                    admission,
                    regime,
                    arm,
                    regime_index * len(_ARMS) + arm_index,
                    schedule[regime],
                ),
                "parameter_movement": _formal_parameter_movement(
                    topology["parameter_count"],
                    paths,
                    regime=regime,
                    arm=arm,
                ),
                "evaluation": evaluation,
                "metrics": {},
            }
    metrics = {
        arm: gate_c._metric_summary(
            arms[arm]["gate_a"]["evaluation"],
            arms[arm]["gate_b"]["evaluation"],
        )
        for arm in _ARMS
    }
    for arm in _ARMS:
        for regime in _REGIMES:
            arms[arm][regime]["metrics"] = copy.deepcopy(metrics[arm])
    margins = gate_c._blocking_margin_report(metrics)
    source_start = copy.deepcopy(admission["source_start"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "control": "example21_pp_prop_learnability_gate_c",
        "qualification_regime": "preregistered_full",
        "learner": "pp_prop_only",
        "prerequisites": {
            "gate_a": copy.deepcopy(_GATE_A_REFERENCE),
            "gate_b": copy.deepcopy(_GATE_B_REFERENCE),
            "gate_c_initialization": _gate_c_initialization_wrapper(admission),
        },
        "regimes": {
            regime: {
                "spec": dataclasses.asdict(gate_c.REGIME_SPECS[regime]),
                "config": dataclasses.asdict(
                    config.gate_a_config
                    if regime == "gate_a"
                    else config.gate_b_config
                ),
                "schedule": copy.deepcopy(schedule[regime]),
                "metrics": copy.deepcopy(metrics),
                "margins": copy.deepcopy(margins),
                "paired_h0_identity": _formal_paired_h0_identity(
                    admission,
                    regime,
                ),
            }
            for regime in _REGIMES
        },
        "arms": arms,
        "mechanism_oracle": _formal_mechanism_oracle(),
        "source_start": source_start,
        "source_end": copy.deepcopy(source_start),
        "source_files": copy.deepcopy(admission["source_files"]),
        "environment": copy.deepcopy(admission["environment"]),
        "total_wall_seconds": 1.0,
    }
    report["qualification"] = {
        "criteria": {name: True for name in gate_c.QUALIFICATION_CRITERIA},
        "passed": True,
        "interpretation": "gate_c_passed_pp_prop_learnability_mechanism",
    }
    return report, config


def _mutate_formal_gate_c_report(
    report: dict[str, Any],
    criterion: str,
) -> None:
    if criterion == "schema_and_control":
        report["control"] = "wrong-control"
    elif criterion == "exact_configuration":
        report["arms"]["query_only"]["gate_a"]["training"][
            "intervention"
        ]["memory_mode"] = "full"
    elif criterion == "prerequisites_authenticated":
        report["prerequisites"]["gate_a"]["result_sha256"] = "0" * 64
    elif criterion == "initialization_authenticated":
        report["arms"]["full"]["gate_a"]["initialization"][
            "parameter_sha256"
        ] = "0" * 64
    elif criterion == "canonical_schedules_complete":
        report["arms"]["full"]["gate_b"]["training"]["data_identity"][
            "training_global_sha256"
        ]["events"] = "0" * 64
    elif criterion == "fresh_isolated_optimizers":
        report["arms"]["full"]["gate_b"]["optimizer"][
            "fresh_state_all_zero"
        ] = False
    elif criterion == "compiler_and_training_complete":
        report["arms"]["query_only"]["gate_a"]["training"]["finite"][
            "gradients"
        ] = False
    elif criterion == "full_gate_a_passed":
        report["arms"]["full"]["gate_a"]["evaluation"][
            "binding_diagnostic"
        ]["read_by_depth"].pop("1")
    elif criterion == "full_gate_b_passed":
        evaluation = report["arms"]["full"]["gate_b"]["evaluation"]
        h0 = _formal_accuracy_metric(
            50,
            checkpoint=0,
            identity="full:gate-b:h0:intact",
        )
        evaluation["depths"]["0"]["intact"] = copy.deepcopy(h0)
        evaluation["h0_proper"] = copy.deepcopy(h0)
        for effort in gate_b.QUALIFYING_EFFORTS:
            evidence = evaluation["efforts"][str(effort)]
            evidence["h0_final_target"] = copy.deepcopy(h0)
            evidence["intact_minus_h0"] = (
                evidence["intact"]["accuracy"] - h0["accuracy"]
            )
    elif criterion == "blocking_behavioral_margins":
        report["regimes"]["gate_a"]["margins"]["query_only"][
            "passed"
        ] = False
    elif criterion == "paired_h0_identity":
        report["regimes"]["gate_b"]["paired_h0_identity"]["streams"][
            "intact"
        ]["query_only_state_sha256"] = "0" * 64
    elif criterion == "frozen_write_complete":
        report["arms"]["frozen_write"]["gate_a"]["training"][
            "frozen_write"
        ]["all_ones_after"] = False
    elif criterion == "mechanism_oracle_complete":
        report["mechanism_oracle"]["comparisons"]["query_only"]["global"][
            "full_norm"
        ] += 1.0
    elif criterion == "source_and_gpu_authenticated":
        report["source_end"]["dirty"] = True
    else:
        raise AssertionError(f"unhandled Gate C criterion {criterion}")


@pytest.fixture(scope="module")
def passing_formal_gate_c_report() -> tuple[dict[str, Any], Any]:
    return _passing_formal_gate_c_report()


@requires_gate_c
def test_complete_formal_gate_c_report_recomputes_all_fourteen_criteria(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    report, config = passing_formal_gate_c_report

    recomputed = gate_c._qualification_report(report, config=config)

    assert set(report) == {
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "arms",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
        "total_wall_seconds",
    }
    assert set(report["prerequisites"]) == {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }
    assert all(
        set(report["arms"][arm][regime])
        == {
            "initialization",
            "optimizer",
            "compiler",
            "training",
            "parameter_movement",
            "evaluation",
            "metrics",
        }
        for arm in _ARMS
        for regime in _REGIMES
    )
    assert all(
        set(report["regimes"][regime])
        == {
            "spec",
            "config",
            "schedule",
            "metrics",
            "margins",
            "paired_h0_identity",
        }
        for regime in _REGIMES
    )
    full_gate_a = report["arms"]["full"]["gate_a"]["evaluation"]
    assert set(full_gate_a) == {
        "finite",
        "all_compact_logits_finite",
        "all_state_tensors_finite",
        "depths",
        "intact",
        "shuffled",
        "no_context",
        "intact_minus_shuffled",
        "binding_state",
        "binding_diagnostic",
    }
    assert gate_a._diagnostic_evidence_complete(
        full_gate_a["binding_diagnostic"],
        config.gate_a_config,
    )
    oracle = report["mechanism_oracle"]
    assert oracle["contract"] == gate_c._oracle_contract(config)
    query_paths = oracle["comparisons"]["query_only"]["paths"]
    terminal_paths = oracle["comparisons"]["terminal_only"]["paths"]
    assert {
        path: evidence["full_sha256"]
        for path, evidence in query_paths.items()
    } == {
        path: evidence["full_sha256"]
        for path, evidence in terminal_paths.items()
    }
    for comparison in oracle["comparisons"].values():
        path_evidence = comparison["paths"]
        global_evidence = comparison["global"]
        expected_full_norm = math.sqrt(
            math.fsum(item["full_norm"] ** 2 for item in path_evidence.values())
        )
        expected_arm_norm = math.sqrt(
            math.fsum(item["arm_norm"] ** 2 for item in path_evidence.values())
        )
        expected_difference = math.sqrt(
            math.fsum(
                item["l2_difference"] ** 2 for item in path_evidence.values()
            )
        )
        expected_dot = math.fsum(
            0.0
            if item["cosine"] is None
            else item["cosine"] * item["full_norm"] * item["arm_norm"]
            for item in path_evidence.values()
        )
        expected_cosine = expected_dot / (expected_full_norm * expected_arm_norm)
        assert math.isclose(
            global_evidence["full_norm"], expected_full_norm, abs_tol=1e-12
        )
        assert math.isclose(
            global_evidence["arm_norm"], expected_arm_norm, abs_tol=1e-12
        )
        assert math.isclose(
            global_evidence["l2_difference"],
            expected_difference,
            abs_tol=1e-12,
        )
        assert math.isclose(
            global_evidence["cosine"], expected_cosine, abs_tol=1e-12
        )
        for side in ("full", "arm"):
            fields = [b"example21-gate-c-gradient-global-v1"]
            for path in sorted(path_evidence):
                fields.extend(
                    (
                        path.encode(),
                        path_evidence[path][f"{side}_sha256"].encode(),
                    )
                )
            assert global_evidence[f"{side}_sha256"] == hashlib.sha256(
                b"\0".join(fields)
            ).hexdigest()
    assert recomputed == report["qualification"]
    assert recomputed["passed"] is True
    assert all(recomputed["criteria"].values())
    assert recomputed["interpretation"] == (
        "gate_c_passed_pp_prop_learnability_mechanism"
    )


@pytest.mark.parametrize("criterion", tuple(gate_c.QUALIFICATION_CRITERIA))
@requires_gate_c
def test_each_formal_gate_c_criterion_fails_on_its_owned_evidence_mutation(
    criterion: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    baseline = gate_c._qualification_report(passing, config=config)
    assert baseline["passed"] is True
    assert all(baseline["criteria"].values())
    report = copy.deepcopy(passing)
    _mutate_formal_gate_c_report(report, criterion)

    recomputed = gate_c._qualification_report(report, config=config)

    assert recomputed["criteria"][criterion] is False
    assert recomputed["passed"] is False
    assert recomputed["interpretation"] == (
        "gate_c_failed_stop_no_causal_mechanism_conclusion"
    )
    assert recomputed != report["qualification"]


@pytest.mark.parametrize("mutation", ("global_norm", "global_digest"))
@requires_gate_c
def test_mechanism_oracle_rejects_colluding_pass_flags_with_inconsistent_evidence(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    comparison = report["mechanism_oracle"]["comparisons"]["terminal_only"]
    if mutation == "global_norm":
        comparison["global"]["l2_difference"] += 1.0
    else:
        comparison["global"]["full_sha256"] = "0" * 64
    assert comparison["passed"] is True
    assert report["mechanism_oracle"]["complete"] is True

    recomputed = gate_c._qualification_report(report, config=config)

    assert recomputed["criteria"]["mechanism_oracle_complete"] is False
    assert recomputed["passed"] is False


@requires_gate_c
def test_mechanism_oracle_rejects_impossible_zero_arm_norm_geometry(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    oracle = report["mechanism_oracle"]
    record = oracle["comparisons"]["query_only"]["paths"][
        "memory_read_projection/weight"
    ]
    assert record["full_norm"] > 0.0
    assert record["arm_norm"] == 0.0
    record["l2_difference"] = record["full_norm"] * 1e-3
    record["relative_deviation"] = 1e-3
    record["cosine"] = None
    record["cosine_defined"] = False
    assert oracle["comparisons"]["query_only"]["passed"] is True
    assert oracle["complete"] is True

    recomputed = gate_c._qualification_report(report, config=config)

    assert recomputed["criteria"]["mechanism_oracle_complete"] is False
    assert recomputed["passed"] is False


@requires_gate_c
def test_query_only_h1_read_must_remain_exactly_zero(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    read = report["arms"]["query_only"]["gate_a"]["evaluation"][
        "binding_diagnostic"
    ]["read_by_depth"]["1"]
    assert read["different_count"] == 0
    assert read["every_pair_differs"] is False
    assert read["mean_l2_difference"] == 0.0
    assert read["left_sha256"] == read["right_sha256"]
    read["different_count"] = 512
    read["every_pair_differs"] = True
    read["mean_l2_difference"] = 1.0
    read["right_sha256"] = "0" * 64

    recomputed = gate_c._qualification_report(report, config=config)

    assert recomputed["criteria"]["compiler_and_training_complete"] is False
    assert recomputed["passed"] is False


@requires_gate_c
def test_full_gate_a_requires_every_final_memory_pair_to_differ(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    evaluation = report["arms"]["full"]["gate_a"]["evaluation"]
    memory = evaluation["binding_diagnostic"]["memory"]
    memory["different_count"] = 0
    memory["intact_shuffled_different_count"] = 0
    memory["every_pair_differs"] = False
    memory["every_intact_shuffled_pair_differs"] = False
    memory["mean_l2_difference"] = 0.0
    memory["right_sha256"] = memory["left_sha256"]
    binding_state = evaluation["binding_state"]
    binding_state["intact_shuffled_different_count"] = 0
    binding_state["every_intact_shuffled_pair_differs"] = False
    binding_state["shuffled_sha256"] = binding_state["intact_sha256"]

    recomputed = gate_c._qualification_report(report, config=config)

    assert recomputed["criteria"]["full_gate_a_passed"] is False
    assert recomputed["passed"] is False


@pytest.mark.parametrize(
    ("regime", "arm", "path"),
    (
        ("gate_a", "full", "height_head/weight"),
        ("gate_b", "query_only", "workspace_query_projection/weight"),
    ),
)
@requires_gate_c
def test_formal_training_rejects_nonzero_causally_dead_parameter_movement(
    regime: str,
    arm: str,
    path: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    movement = report["arms"][arm][regime]["parameter_movement"]
    assert movement["paths"][path]["l2_delta"] == 0.0
    movement["paths"][path]["l2_delta"] = 1.0
    movement["l2_delta"] = math.hypot(movement["l2_delta"], 1.0)
    squared = math.fsum(
        evidence["l2_delta"] ** 2 for evidence in movement["paths"].values()
    )
    assert math.isclose(movement["l2_delta"], math.sqrt(squared), abs_tol=1e-12)

    recomputed = gate_c._qualification_report(report, config=config)

    assert recomputed["criteria"]["compiler_and_training_complete"] is False
    assert recomputed["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "arm_schema",
        "training_schema",
        "loss_count",
        "loss_type",
        "telemetry_schema",
        "telemetry_nonfinite",
        "telemetry_negative_max",
        "telemetry_bool_count",
        "algorithm",
        "executed_updates",
        "batch_size",
        "chunk_count",
        "training_time",
        "initial_digest",
        "final_digest",
        "loss_weight_digest",
        "compile_warning_type",
        "initial_loss",
        "final_loss",
        "tail_loss",
        "frozen_schema",
        "frozen_bool_type",
        "frozen_applicable",
        "frozen_excluded",
        "full_before_not_one",
        "legacy_before_one",
        "frozen_after_not_one",
        "compiler_mismatch",
    ),
)
@requires_gate_c
def test_training_report_fails_closed_on_malformed_retained_evidence(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    admission = passing["prerequisites"]["gate_c_initialization"]["admission"]
    arm = (
        "legacy"
        if mutation == "legacy_before_one"
        else "frozen_write"
        if mutation in {"frozen_excluded", "frozen_after_not_one"}
        else "full"
    )
    arm_report = copy.deepcopy(passing["arms"][arm]["gate_a"])
    assert gate_c._training_report_complete(
        arm_report,
        admission,
        config,
        regime="gate_a",
        arm=arm,
    )
    training = arm_report["training"]
    if mutation == "arm_schema":
        arm_report["unexpected"] = None
    elif mutation == "training_schema":
        training.pop("algorithm")
    elif mutation == "loss_count":
        training["losses"].pop()
    elif mutation == "loss_type":
        training["losses"][0] = True
    elif mutation == "telemetry_schema":
        training["finite"].pop("logits")
    elif mutation == "telemetry_nonfinite":
        training["finite"]["model_states"] = False
    elif mutation == "telemetry_negative_max":
        training["max_abs"]["pp_prop_traces"] = -1.0
    elif mutation == "telemetry_bool_count":
        training["value_count"]["adam"] = True
    elif mutation == "algorithm":
        training["algorithm"] = "bptt"
    elif mutation == "executed_updates":
        training["executed_updates"] = True
    elif mutation == "batch_size":
        training["batch_size"] += 1
    elif mutation == "chunk_count":
        training["chunk_count"] = False
    elif mutation == "training_time":
        training["cold_compile_and_train_seconds"] = -1.0
    elif mutation == "initial_digest":
        training["initial_parameter_sha256"] = "0" * 64
    elif mutation == "final_digest":
        training["final_parameter_sha256"] = training[
            "initial_parameter_sha256"
        ]
    elif mutation == "loss_weight_digest":
        training["loss_weights"]["sha256"] = "0" * 64
    elif mutation == "compile_warning_type":
        training["compile_warnings"] = [False]
    elif mutation == "initial_loss":
        training["initial_loss"] += 1.0
    elif mutation == "final_loss":
        training["final_loss"] += 1.0
    elif mutation == "tail_loss":
        training["tail_64_mean_loss"] += 1.0
    elif mutation == "frozen_schema":
        training["frozen_write"].pop("applicable")
    elif mutation == "frozen_bool_type":
        training["frozen_write"]["all_ones_before"] = 1
    elif mutation == "frozen_applicable":
        training["frozen_write"]["applicable"] = True
    elif mutation == "frozen_excluded":
        training["frozen_write"]["excluded_from_optimizer"] = False
    elif mutation == "full_before_not_one":
        training["frozen_write"]["all_ones_before"] = False
    elif mutation == "legacy_before_one":
        training["frozen_write"]["all_ones_before"] = True
    elif mutation == "frozen_after_not_one":
        training["frozen_write"]["all_ones_after"] = False
    elif mutation == "compiler_mismatch":
        arm_report["compiler"]["all_required_direct"] = False
    else:
        raise AssertionError(f"unhandled training mutation {mutation}")

    assert not gate_c._training_report_complete(
        arm_report,
        admission,
        config,
        regime="gate_a",
        arm=arm,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "top_schema",
        "path_schema",
        "path_record_schema",
        "negative_delta",
        "wrong_path_count",
        "nonzero_dead_path",
        "wrong_total_norm",
        "wrong_total_count",
    ),
)
@requires_gate_c
def test_parameter_movement_validator_rejects_malformed_or_dead_paths(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, _ = passing_formal_gate_c_report
    movement = copy.deepcopy(
        passing["arms"]["full"]["gate_a"]["parameter_movement"]
    )
    assert gate_c._parameter_movement_complete(
        movement,
        regime="gate_a",
        arm="full",
    )
    path = "color_factor_head/weight"
    if mutation == "top_schema":
        movement["unexpected"] = None
    elif mutation == "path_schema":
        movement["paths"].pop(path)
    elif mutation == "path_record_schema":
        movement["paths"][path]["unexpected"] = None
    elif mutation == "negative_delta":
        movement["paths"][path]["l2_delta"] = -1.0
    elif mutation == "wrong_path_count":
        movement["paths"][path]["parameter_count"] += 1
    elif mutation == "nonzero_dead_path":
        movement["paths"]["height_head/weight"]["l2_delta"] = 1.0
        movement["l2_delta"] = math.hypot(movement["l2_delta"], 1.0)
    elif mutation == "wrong_total_norm":
        movement["l2_delta"] += 1.0
    elif mutation == "wrong_total_count":
        movement["parameter_count"] += 1
    else:
        raise AssertionError(f"unhandled movement mutation {mutation}")

    assert not gate_c._parameter_movement_complete(
        movement,
        regime="gate_a",
        arm="full",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "top_schema",
        "finite",
        "depth_schema",
        "stream_schema",
        "metric_schema",
        "metric_checkpoint_type",
        "metric_digest",
        "h0_mismatch",
        "effort_set",
        "effort_schema",
        "effort_depth_mismatch",
        "h0_final_digest",
        "h0_final_histogram",
        "h0_gap",
        "shuffled_gap",
    ),
)
@requires_gate_c
def test_gate_b_evaluation_validator_rejects_malformed_or_stale_evidence(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    evaluation = copy.deepcopy(
        passing["arms"]["full"]["gate_b"]["evaluation"]
    )
    assert gate_c._gate_b_evaluation_complete(
        evaluation,
        config,
        require_no_collapse=True,
    )
    if mutation == "top_schema":
        evaluation["unexpected"] = None
    elif mutation == "finite":
        evaluation["finite"] = False
    elif mutation == "depth_schema":
        evaluation["depths"].pop("8")
    elif mutation == "stream_schema":
        evaluation["depths"]["0"].pop("no_context")
    elif mutation == "metric_schema":
        evaluation["depths"]["1"]["intact"].pop("checkpoint")
    elif mutation == "metric_checkpoint_type":
        evaluation["depths"]["1"]["intact"]["checkpoint"] = True
    elif mutation == "metric_digest":
        evaluation["depths"]["1"]["intact"]["prediction_sha256"] = "bad"
    elif mutation == "h0_mismatch":
        evaluation["h0_proper"]["prediction_sha256"] = "0" * 64
    elif mutation == "effort_set":
        evaluation["efforts"].pop("8")
    elif mutation == "effort_schema":
        evaluation["efforts"]["1"].pop("intact_minus_h0")
    elif mutation == "effort_depth_mismatch":
        evaluation["efforts"]["2"]["intact"]["prediction_sha256"] = "0" * 64
    elif mutation == "h0_final_digest":
        evaluation["efforts"]["4"]["h0_final_target"][
            "prediction_sha256"
        ] = "0" * 64
    elif mutation == "h0_final_histogram":
        evaluation["efforts"]["4"]["h0_final_target"][
            "prediction_histogram"
        ][0] += 1
        evaluation["efforts"]["4"]["h0_final_target"][
            "prediction_histogram"
        ][1] -= 1
    elif mutation == "h0_gap":
        evaluation["efforts"]["8"]["intact_minus_h0"] += 0.1
    elif mutation == "shuffled_gap":
        evaluation["efforts"]["8"]["intact_minus_shuffled"] += 0.1
    else:
        raise AssertionError(f"unhandled Gate B evaluation mutation {mutation}")

    assert not gate_c._gate_b_evaluation_complete(
        evaluation,
        config,
        require_no_collapse=True,
    )


@requires_gate_c
def test_gate_b_collapse_guard_applies_only_to_the_full_success_control(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    evaluation = copy.deepcopy(
        passing["arms"]["legacy"]["gate_b"]["evaluation"]
    )
    collapsed = [config.gate_b_config.validation_episodes, *([0] * 9)]
    for depth in evaluation["depths"].values():
        for metric in depth.values():
            metric["prediction_histogram"] = collapsed.copy()
    evaluation["h0_proper"] = copy.deepcopy(evaluation["depths"]["0"]["intact"])
    for effort in gate_b.QUALIFYING_EFFORTS:
        evidence = evaluation["efforts"][str(effort)]
        for stream in ("intact", "shuffled", "no_context"):
            evidence[stream] = copy.deepcopy(
                evaluation["depths"][str(effort)][stream]
            )
        evidence["h0_final_target"]["prediction_histogram"] = collapsed.copy()

    assert gate_c._gate_b_evaluation_complete(
        evaluation,
        config,
        require_no_collapse=False,
    )
    assert not gate_c._gate_b_evaluation_complete(
        evaluation,
        config,
        require_no_collapse=True,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "numeric_type",
        "negative_norm",
        "triangle",
        "defined_flag",
        "digest",
        "relative_type",
        "relative_value",
        "cosine_type",
        "cosine_value",
        "zero_full_relative",
        "zero_arm_cosine",
    ),
)
@requires_gate_c
def test_gradient_record_validator_rejects_type_confusion_and_impossible_geometry(
    mutation: str,
) -> None:
    record = copy.deepcopy(
        _formal_mechanism_oracle()["comparisons"]["terminal_only"]["paths"][
            "color_factor_head/weight"
        ]
    )
    assert gate_c._gradient_record_complete(record)
    if mutation == "schema":
        record["unexpected"] = None
    elif mutation == "numeric_type":
        record["full_norm"] = True
    elif mutation == "negative_norm":
        record["l2_difference"] = -1.0
    elif mutation == "triangle":
        record["l2_difference"] = record["full_norm"] + record["arm_norm"] + 1.0
    elif mutation == "defined_flag":
        record["relative_deviation_defined"] = False
    elif mutation == "digest":
        record["arm_sha256"] = "bad"
    elif mutation == "relative_type":
        record["relative_deviation"] = True
    elif mutation == "relative_value":
        record["relative_deviation"] += 0.1
    elif mutation == "cosine_type":
        record["cosine"] = True
    elif mutation == "cosine_value":
        record["cosine"] = -1.0
    elif mutation == "zero_full_relative":
        record.update(
            {
                "full_norm": 0.0,
                "arm_norm": 1.0,
                "l2_difference": 1.0,
                "relative_deviation": 0.0,
                "relative_deviation_defined": False,
                "cosine": None,
                "cosine_defined": False,
            }
        )
    elif mutation == "zero_arm_cosine":
        record.update(
            {
                "full_norm": 1.0,
                "arm_norm": 0.0,
                "l2_difference": 1.0,
                "relative_deviation": 1.0,
                "relative_deviation_defined": True,
                "cosine": 0.0,
                "cosine_defined": False,
            }
        )
    else:
        raise AssertionError(f"unhandled gradient mutation {mutation}")

    assert not gate_c._gradient_record_complete(record)


@pytest.mark.parametrize(
    "mutation",
    (
        "top_schema",
        "contract",
        "objective",
        "chunk_type",
        "comparison_set",
        "comparison_schema",
        "path_set",
        "path_record",
        "global_record",
        "required_paths",
        "stored_pass",
        "cross_full_snapshot",
        "complete_flag",
    ),
)
@requires_gate_c
def test_mechanism_oracle_validator_rejects_malformed_or_colluding_evidence(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    _, config = passing_formal_gate_c_report
    oracle = _formal_mechanism_oracle()
    assert gate_c._mechanism_oracle_complete(oracle, config)
    query = oracle["comparisons"]["query_only"]
    terminal = oracle["comparisons"]["terminal_only"]
    path = "color_factor_head/weight"
    if mutation == "top_schema":
        oracle["unexpected"] = None
    elif mutation == "contract":
        oracle["contract"]["events_sha256"] = "0" * 64
    elif mutation == "objective":
        oracle["objective"]["unsupervised_output_exact_zero"] = False
    elif mutation == "chunk_type":
        oracle["gradient_chunk_size"] = True
    elif mutation == "comparison_set":
        oracle["comparisons"].pop("terminal_only")
    elif mutation == "comparison_schema":
        query["unexpected"] = None
    elif mutation == "path_set":
        query["paths"].pop(path)
    elif mutation == "path_record":
        query["paths"][path]["unexpected"] = None
    elif mutation == "global_record":
        query["global"]["l2_difference"] = -1.0
    elif mutation == "required_paths":
        query["required_paths"] = []
    elif mutation == "stored_pass":
        query["required_paths_passed"] = False
    elif mutation == "cross_full_snapshot":
        terminal["paths"][path]["full_sha256"] = "0" * 64
        terminal["global"]["full_sha256"] = gate_c._gradient_digest_from_records(
            terminal["paths"],
            side="full",
        )
    elif mutation == "complete_flag":
        oracle["complete"] = False
    else:
        raise AssertionError(f"unhandled oracle mutation {mutation}")

    assert not gate_c._mechanism_oracle_complete(oracle, config)


@pytest.mark.parametrize(
    "mutation",
    (
        "top_schema",
        "finite_flag",
        "depth_schema",
        "stream_schema",
        "metric_schema",
        "final_stream_mismatch",
        "gap",
        "diagnostic_schema",
        "memory_schema",
        "memory_pair_count",
        "diagnostic_depth_schema",
        "binding_state",
    ),
)
@requires_gate_c
def test_gate_a_evaluation_validator_rejects_malformed_binding_evidence(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    evaluation = copy.deepcopy(
        passing["arms"]["full"]["gate_a"]["evaluation"]
    )
    assert gate_c._gate_a_evaluation_complete(evaluation, config)
    if mutation == "top_schema":
        evaluation["unexpected"] = None
    elif mutation == "finite_flag":
        evaluation["all_state_tensors_finite"] = False
    elif mutation == "depth_schema":
        evaluation["depths"].pop("1")
    elif mutation == "stream_schema":
        evaluation["depths"]["0"].pop("no_context")
    elif mutation == "metric_schema":
        evaluation["depths"]["0"]["intact"].pop("checkpoint")
    elif mutation == "final_stream_mismatch":
        evaluation["intact"]["prediction_sha256"] = "0" * 64
    elif mutation == "gap":
        evaluation["intact_minus_shuffled"] += 0.1
    elif mutation == "diagnostic_schema":
        evaluation["binding_diagnostic"]["unexpected"] = None
    elif mutation == "memory_schema":
        evaluation["binding_diagnostic"]["memory"].pop("storage_contract")
    elif mutation == "memory_pair_count":
        evaluation["binding_diagnostic"]["memory"]["applicable_count"] = True
    elif mutation == "diagnostic_depth_schema":
        evaluation["binding_diagnostic"]["read_by_depth"].pop("0")
    elif mutation == "binding_state":
        evaluation["binding_state"]["all_finite"] = False
    else:
        raise AssertionError(f"unhandled Gate A evaluation mutation {mutation}")

    assert not gate_c._gate_a_evaluation_complete(evaluation, config)


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "zero_with_positive_mean",
        "zero_with_distinct_hashes",
        "different_with_zero_mean",
        "different_with_equal_hashes",
        "different_count_out_of_range",
        "negative_no_context_norm",
    ),
)
@requires_gate_c
def test_paired_diagnostic_validator_rejects_incoherent_count_norm_and_hashes(
    mutation: str,
) -> None:
    record = copy.deepcopy(
        _formal_paired_diagnostic(identity="coverage:legacy", different=0)
    )
    assert gate_c._paired_diagnostic_record_complete(record, count=512)
    if mutation == "schema":
        record["unexpected"] = None
    elif mutation == "zero_with_positive_mean":
        record["mean_l2_difference"] = 1.0
    elif mutation == "zero_with_distinct_hashes":
        record["right_sha256"] = "0" * 64
    elif mutation == "different_with_zero_mean":
        record["different_count"] = 1
        record["mean_l2_difference"] = 0.0
        record["right_sha256"] = "0" * 64
    elif mutation == "different_with_equal_hashes":
        record["different_count"] = 1
        record["mean_l2_difference"] = 1.0
    elif mutation == "different_count_out_of_range":
        record["different_count"] = 513
    elif mutation == "negative_no_context_norm":
        record["no_context_l2_norm"] = -1.0
    else:
        raise AssertionError(f"unhandled paired diagnostic mutation {mutation}")

    assert not gate_c._paired_diagnostic_record_complete(record, count=512)


@pytest.mark.parametrize("arm_norm", (0.0, 1.0))
@requires_gate_c
def test_gradient_record_accepts_well_defined_zero_full_norm_geometry(
    arm_norm: float,
) -> None:
    digest = hashlib.sha256(f"zero-full:{arm_norm}".encode()).hexdigest()
    record = {
        "full_norm": 0.0,
        "arm_norm": arm_norm,
        "l2_difference": arm_norm,
        "relative_deviation": None,
        "relative_deviation_defined": False,
        "cosine": None,
        "cosine_defined": False,
        "full_sha256": digest,
        "arm_sha256": digest,
    }

    assert gate_c._gradient_record_complete(record)


@requires_gate_c
def test_legacy_intervention_requires_all_memory_and_read_paths_exact_zero(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    evaluation = copy.deepcopy(
        passing["arms"]["legacy"]["gate_a"]["evaluation"]
    )
    assert gate_c._gate_a_evaluation_complete(evaluation, config)
    assert gate_c._gate_a_intervention_diagnostic_complete(
        evaluation,
        arm="legacy",
    )
    evaluation["binding_diagnostic"]["memory"]["intact_l2_norm"] = 1.0

    assert not gate_c._gate_a_intervention_diagnostic_complete(
        evaluation,
        arm="legacy",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "nonqualifying_config",
        "qualification_regime",
        "regime_schema",
        "regime_spec",
        "arm_regime_schema",
        "arm_report_schema",
        "execution_index_type",
        "intervention",
    ),
)
@requires_gate_c
def test_formal_configuration_validator_rejects_stale_or_type_confused_identity(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    candidate_config = config
    if mutation == "nonqualifying_config":
        candidate_config = dataclasses.replace(config, oracle_effort=4)
    elif mutation == "qualification_regime":
        report["qualification_regime"] = "nonqualifying_abbreviated"
    elif mutation == "regime_schema":
        report["regimes"]["gate_a"]["unexpected"] = None
    elif mutation == "regime_spec":
        report["regimes"]["gate_a"]["spec"]["training_updates"] += 1
    elif mutation == "arm_regime_schema":
        report["arms"]["full"].pop("gate_b")
    elif mutation == "arm_report_schema":
        report["arms"]["full"]["gate_a"]["unexpected"] = None
    elif mutation == "execution_index_type":
        report["arms"]["full"]["gate_a"]["training"]["execution_index"] = True
    elif mutation == "intervention":
        report["arms"]["full"]["gate_a"]["training"]["intervention"][
            "memory_mode"
        ] = "none"
    else:
        raise AssertionError(f"unhandled formal configuration mutation {mutation}")

    assert not gate_c._exact_formal_configuration_complete(
        report,
        candidate_config,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "initialization_report_type",
        "initialization_schema",
        "initialization_count_type",
        "schedule",
        "data_identity",
        "loss_weights",
        "paired_schema",
        "paired_checkpoint_type",
        "paired_initialization",
        "paired_stream_schema",
        "paired_digest",
        "paired_boolean",
        "paired_pass",
    ),
)
@requires_gate_c
def test_formal_identity_validators_reject_malformed_or_stale_evidence(
    mutation: str,
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    report = copy.deepcopy(passing)
    admission = report["prerequisites"]["gate_c_initialization"]["admission"]
    arm_report: Any = report["arms"]["full"]["gate_a"]
    if mutation == "initialization_report_type":
        assert not gate_c._formal_initialization_complete(
            [],
            admission,
            regime="gate_a",
            arm="full",
        )
        return
    if mutation == "initialization_schema":
        arm_report["initialization"]["unexpected"] = None
        assert not gate_c._formal_initialization_complete(
            arm_report,
            admission,
            regime="gate_a",
            arm="full",
        )
        return
    if mutation == "initialization_count_type":
        arm_report["initialization"]["parameter_count"] = True
        assert not gate_c._formal_initialization_complete(
            arm_report,
            admission,
            regime="gate_a",
            arm="full",
        )
        return
    if mutation == "schedule":
        report["regimes"]["gate_a"]["schedule"]["events"] = "0" * 64
        assert not gate_c._canonical_schedules_complete(report, config)
        return
    if mutation == "data_identity":
        report["arms"]["full"]["gate_a"]["training"]["data_identity"][
            "events"
        ] = "0" * 64
        assert not gate_c._canonical_schedules_complete(report, config)
        return
    if mutation == "loss_weights":
        report["arms"]["full"]["gate_a"]["training"]["loss_weights"][
            "sha256"
        ] = "0" * 64
        assert not gate_c._canonical_schedules_complete(report, config)
        return
    paired = report["regimes"]["gate_a"]["paired_h0_identity"]
    if mutation == "paired_schema":
        paired["unexpected"] = None
    elif mutation == "paired_checkpoint_type":
        paired["checkpoint"] = True
    elif mutation == "paired_initialization":
        paired["initialization_parameter_sha256"]["full"] = "0" * 64
    elif mutation == "paired_stream_schema":
        paired["streams"]["intact"]["unexpected"] = None
    elif mutation == "paired_digest":
        paired["streams"]["intact"]["full_compact_sha256"] = "bad"
    elif mutation == "paired_boolean":
        paired["streams"]["intact"]["compact_byte_identical"] = 1
    elif mutation == "paired_pass":
        paired["passed"] = False
    else:
        raise AssertionError(f"unhandled formal identity mutation {mutation}")
    assert not gate_c._paired_h0_identity_complete(
        paired,
        admission,
        regime="gate_a",
    )


@requires_gate_c
def test_full_gate_b_and_behavioral_margins_reject_incomplete_duplicate_evidence(
    passing_formal_gate_c_report: tuple[dict[str, Any], Any],
) -> None:
    passing, config = passing_formal_gate_c_report
    malformed_full = copy.deepcopy(passing)
    malformed_full["arms"]["full"]["gate_b"]["evaluation"]["unexpected"] = None
    assert not gate_c._full_gate_b_complete(malformed_full, config)

    stale_metrics = copy.deepcopy(passing)
    stale_metrics["arms"]["legacy"]["gate_a"]["metrics"]["binding_gap"] += 0.1
    assert not gate_c._behavioral_margins_complete(stale_metrics, config)


@requires_gate_c
def test_arm_initialization_reproduction_binds_authenticated_evidence() -> None:
    admission, _ = _passing_gate_c_initialization_report()
    report = _minimal_formal_arm_initialization_report(
        admission,
        "gate_b",
        "frozen_write",
    )

    assert gate_c._arm_initialization_reproduced(
        report,
        admission,
        "gate_b",
        "frozen_write",
    ) is True

    colluding = copy.deepcopy(report)
    colluding["initialization"]["parameter_sha256"] = "e" * 64
    colluding["optimizer"]["state_sha256"] = "e" * 64
    assert gate_c._arm_initialization_reproduced(
        colluding,
        admission,
        "gate_b",
        "frozen_write",
    ) is False


@requires_gate_c
def test_formal_arm_initialization_builder_raises_before_training_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    config = _reduced_gate_c_config()
    model = gate_c._new_model_for_arm(
        config,
        "gate_a",
        "full",
        batch_size=config.gate_a_config.batch_size,
    )
    trainer = gate_c._make_arm_trainer(
        model,
        config,
        "gate_a",
        "full",
    )
    checker_calls = 0

    def reject_reproduction(*args: Any, **kwargs: Any) -> bool:
        nonlocal checker_calls
        checker_calls += 1
        return False

    monkeypatch.setattr(
        gate_c,
        "_arm_initialization_reproduced",
        reject_reproduction,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="initialization|reproduc"):
        gate_c._formal_arm_initialization_report(
            model,
            trainer,
            admission,
            "gate_a",
            "full",
        )
    assert checker_calls == 1


@requires_gate_c
def test_formal_runner_authenticates_initialization_before_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _ = _passing_gate_c_initialization_report()
    prerequisites = {
        "gate_a": dict(_GATE_A_REFERENCE),
        "gate_b": dict(_GATE_B_REFERENCE),
        "gate_c_initialization": _gate_c_initialization_wrapper(report),
    }
    model_calls = 0

    def reject_initialization(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("Gate C initialization is not authenticated")

    def forbidden_model(*args: Any, **kwargs: Any) -> Any:
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("model constructed before initialization auth")

    monkeypatch.setattr(
        gate_c,
        "_validated_gate_c_initialization_admission",
        reject_initialization,
    )
    monkeypatch.setattr(gate_c, "_new_model_for_arm", forbidden_model)

    with pytest.raises(ValueError, match="initialization.*authenticated"):
        gate_c.run_gate_c(
            _reduced_gate_c_config(),
            prerequisites=prerequisites,
            source_start=_passing_gate_c_source(),
            source_end_reporter=_passing_gate_c_source,
            source_files=_current_gate_c_source_files(),
            environment=_passing_gate_c_environment(),
        )
    assert model_calls == 0


@requires_gate_c
def test_formal_runner_executes_ten_fresh_isolated_arms_in_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    wrapper = _gate_c_initialization_wrapper(admission)
    prerequisites = {
        "gate_a": dict(_GATE_A_REFERENCE),
        "gate_b": dict(_GATE_B_REFERENCE),
        "gate_c_initialization": wrapper,
    }
    config = _reduced_gate_c_config()
    source_start = _passing_gate_c_source()
    source_end = copy.deepcopy(source_start)
    environment = _passing_gate_c_environment()
    source_files = _current_gate_c_source_files()
    gate_a_data = object()
    gate_b_data = (object(), object())
    schedule_reports = {
        "gate_a": {
            "training_schedule_sha256": "a" * 64,
            "validation_schedule_sha256": "b" * 64,
            "training_mapping_ids_sha256": "c" * 64,
            "validation_mapping_ids_sha256": "d" * 64,
        },
        "gate_b": {
            "training_global_sha256": {
                name: "e" * 64
                for name in (
                    "events",
                    "targets",
                    "loss_weights",
                    "advance_masks",
                    "mapping_ids",
                    "efforts",
                    "query_colors",
                    "presentation_orders",
                )
            },
            "validation_sha256": {
                name: "f" * 64
                for name in (
                    "mapping_ids",
                    "query_colors",
                    "presentation_orders",
                    "shuffled_shifts",
                    "intact",
                    "shuffled",
                    "no_context",
                    "targets_by_depth",
                    "advance_masks",
                )
            },
        },
    }
    paired_h0_reports = {
        regime: {
            "checkpoint": 0,
            "initialization_parameter_sha256": {
                "full": admission["initialization"][regime][
                    "canonical_full"
                ]["parameter_sha256"],
                "query_only": admission["initialization"][regime][
                    "canonical_full"
                ]["parameter_sha256"],
            },
            "streams": {
                stream: {
                    "full_compact_sha256": "1" * 64,
                    "query_only_compact_sha256": "1" * 64,
                    "full_state_sha256": "2" * 64,
                    "query_only_state_sha256": "2" * 64,
                    "compact_byte_identical": True,
                    "state_byte_identical": True,
                }
                for stream in ("intact", "shuffled", "no_context")
            },
            "passed": True,
        }
        for regime in _REGIMES
    }
    events: list[tuple[Any, ...]] = []
    models: dict[tuple[str, str], Any] = {}
    trainers: dict[tuple[str, str], Any] = {}
    initialization_reports: dict[tuple[str, str], dict[str, Any]] = {}
    run_calls: list[tuple[str, str, object, object, object]] = []

    def validate_initialization(
        prerequisite: Mapping[str, Any],
        actual_config: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append(("initialization_authenticated",))
        assert prerequisite is wrapper
        assert actual_config is config
        assert kwargs == {
            "source_start": source_start,
            "environment": environment,
            "source_files": source_files,
            "require_pass": True,
        }
        return admission

    def regenerate_gate_a(actual_config: Any) -> object:
        assert actual_config is config
        events.append(("data", "gate_a"))
        return gate_a_data

    def regenerate_gate_b(actual_config: Any) -> tuple[object, object]:
        assert actual_config is config
        events.append(("data", "gate_b"))
        return gate_b_data

    def new_model(
        actual_config: Any,
        regime: str,
        arm: str,
        *,
        batch_size: int,
    ) -> Any:
        assert actual_config is config
        assert batch_size > 0
        model = SimpleNamespace(regime=regime, arm=arm, instance=object())
        models[(regime, arm)] = model
        events.append(("model", regime, arm))
        return model

    def copy_shared(canonical: Any, legacy_model: Any) -> dict[str, Any]:
        assert canonical is models[(canonical.regime, "full")]
        assert legacy_model is models[(canonical.regime, "legacy")]
        events.append(("shared_copy", canonical.regime))
        return {"all_equal": True}

    def make_trainer(
        model: Any,
        actual_config: Any,
        regime: str,
        arm: str,
    ) -> Any:
        assert actual_config is config
        assert model is models[(regime, arm)]
        trainer = SimpleNamespace(
            optimizer=object(),
            regime=regime,
            arm=arm,
            instance=object(),
        )
        trainers[(regime, arm)] = trainer
        events.append(("trainer", regime, arm))
        return trainer

    def formal_initialization_report(
        model: Any,
        trainer: Any,
        actual_admission: Mapping[str, Any],
        regime: str,
        arm: str,
    ) -> dict[str, Any]:
        assert model is models[(regime, arm)]
        assert trainer is trainers[(regime, arm)]
        assert actual_admission is admission
        report = _minimal_formal_arm_initialization_report(
            admission,
            regime,
            arm,
        )
        initialization_reports[(regime, arm)] = report
        events.append(("arm_initialization", regime, arm))
        return report

    def run_arm(
        model: Any,
        trainer: Any,
        data: Any,
        actual_config: Any,
        regime: str,
        arm: str,
        *,
        initialization_report: Mapping[str, Any],
        execution_index: int,
        data_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert actual_config is config
        assert model is models[(regime, arm)]
        assert trainer is trainers[(regime, arm)]
        assert initialization_report == initialization_reports[(regime, arm)]
        expected_data = gate_a_data if regime == "gate_a" else gate_b_data
        assert data is expected_data
        expected_index = _REGIMES.index(regime) * len(_ARMS) + _ARMS.index(arm)
        assert execution_index == expected_index
        assert data_identity == schedule_reports[regime]
        events.append(("run", regime, arm))
        run_calls.append(
            (
                regime,
                arm,
                model.instance,
                trainer.instance,
                trainer.optimizer,
            )
        )
        report = _minimal_formal_arm_report(
            admission,
            regime,
            arm,
            initialization_report=initialization_report,
        )
        report["training"].update(
            {
                "execution_index": execution_index,
                "intervention": dataclasses.asdict(gate_c.ARM_SPECS[arm]),
                "data_identity": copy.deepcopy(data_identity),
            }
        )
        return report

    def actual_schedule_identity(
        actual_config: Any,
        actual_gate_a_data: Any,
        actual_gate_b_data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert actual_gate_a_data is gate_a_data
        assert actual_gate_b_data is gate_b_data
        events.append(("actual_schedule_identity",))
        return copy.deepcopy(schedule_reports)

    def paired_h0_identity(
        actual_config: Any,
        *,
        initialization: Mapping[str, Any],
        regime: str,
        data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert initialization is admission
        assert data is (gate_a_data if regime == "gate_a" else gate_b_data)
        events.append(("paired_h0_identity", regime))
        return copy.deepcopy(paired_h0_reports[regime])

    def mechanism_oracle(
        actual_config: Any,
        *,
        initialization: Mapping[str, Any],
        gate_b_data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert initialization is admission
        assert gate_b_data is gate_b_data_value
        events.append(("mechanism_oracle",))
        return {"complete": False, "gradient_chunk_size": 1}

    gate_b_data_value = gate_b_data

    def source_end_reporter() -> dict[str, Any]:
        events.append(("source_end",))
        return source_end

    def qualification(
        formal_report: Mapping[str, Any],
        *,
        config: Any,
    ) -> dict[str, Any]:
        assert config is not None
        assert formal_report["source_end"] == source_end
        assert isinstance(formal_report["total_wall_seconds"], float)
        assert math.isfinite(formal_report["total_wall_seconds"])
        assert formal_report["total_wall_seconds"] >= 0.0
        for regime in _REGIMES:
            assert formal_report["regimes"][regime][
                "paired_h0_identity"
            ] == paired_h0_reports[regime]
        for execution_index, (regime, arm) in enumerate(
            (pair for pair in ((r, a) for r in _REGIMES for a in _ARMS))
        ):
            training = formal_report["arms"][arm][regime]["training"]
            assert training["execution_index"] == execution_index
            assert training["intervention"] == dataclasses.asdict(
                gate_c.ARM_SPECS[arm]
            )
            assert training["data_identity"] == schedule_reports[regime]
        events.append(("qualification",))
        return {
            "criteria": {
                name: False for name in gate_c.QUALIFICATION_CRITERIA
            },
            "passed": False,
            "interpretation": "gate_c_failed_stop_no_causal_mechanism_conclusion",
        }

    monkeypatch.setattr(
        gate_c,
        "_validated_gate_c_initialization_admission",
        validate_initialization,
    )
    monkeypatch.setattr(gate_c, "_regenerate_gate_a_data", regenerate_gate_a)
    monkeypatch.setattr(gate_c, "_regenerate_gate_b_data", regenerate_gate_b)
    monkeypatch.setattr(
        gate_c,
        "_schedule_identity_report",
        lambda actual_config: (_ for _ in ()).throw(
            AssertionError("constants-only schedule reporter is inadmissible")
        ),
    )
    monkeypatch.setattr(
        gate_c,
        "_actual_schedule_identity_report",
        actual_schedule_identity,
        raising=False,
    )
    monkeypatch.setattr(gate_c, "_new_model_for_arm", new_model)
    monkeypatch.setattr(gate_c, "_copy_shared_initialization", copy_shared)
    monkeypatch.setattr(gate_c, "_make_arm_trainer", make_trainer)
    monkeypatch.setattr(
        gate_c,
        "_formal_arm_initialization_report",
        formal_initialization_report,
        raising=False,
    )
    monkeypatch.setattr(gate_c, "_run_gate_c_arm", run_arm, raising=False)
    monkeypatch.setattr(
        gate_c,
        "_mechanism_oracle",
        mechanism_oracle,
        raising=False,
    )
    monkeypatch.setattr(
        gate_c,
        "_paired_h0_identity_report",
        paired_h0_identity,
        raising=False,
    )
    monkeypatch.setattr(gate_c, "_qualification_report", qualification)

    result = gate_c.run_gate_c(
        config,
        prerequisites=prerequisites,
        source_start=source_start,
        source_end_reporter=source_end_reporter,
        source_files=source_files,
        environment=environment,
    )

    expected_order = [
        (regime, arm) for regime in _REGIMES for arm in _ARMS
    ]
    assert [(regime, arm) for regime, arm, *_ in run_calls] == expected_order
    assert len({model_id for *_, model_id, _, _ in run_calls}) == 10
    assert len({trainer_id for *_, trainer_id, _ in run_calls}) == 10
    assert len({optimizer_id for *_, optimizer_id in run_calls}) == 10
    first_run = min(
        events.index(("run", regime, arm))
        for regime, arm in expected_order
    )
    assert all(
        events.index(("arm_initialization", regime, arm)) < first_run
        for regime, arm in expected_order
    )
    for regime in _REGIMES:
        copy_position = events.index(("shared_copy", regime))
        first_regime_run = min(
            events.index(("run", regime, arm)) for arm in _ARMS
        )
        assert (
            events.index(("model", regime, "legacy"))
            < copy_position
            < first_regime_run
        )
    assert events[0] == ("initialization_authenticated",)
    assert events.index(("actual_schedule_identity",)) < first_run
    assert all(
        events.index(("paired_h0_identity", regime))
        < events.index(("source_end",))
        for regime in _REGIMES
    )
    assert events.index(("mechanism_oracle",)) < events.index(("source_end",))
    assert events[-1] == ("qualification",)
    assert set(result) == {
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "arms",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
        "total_wall_seconds",
    }
    assert result["schema_version"] == 1
    assert result["control"] == "example21_pp_prop_learnability_gate_c"
    assert result["learner"] == "pp_prop_only"
    assert set(result["arms"]) == set(_ARMS)
    assert all(set(result["arms"][arm]) == set(_REGIMES) for arm in _ARMS)
    assert all(
        set(result["arms"][arm][regime])
        == {
            "initialization",
            "optimizer",
            "compiler",
            "training",
            "parameter_movement",
            "evaluation",
            "metrics",
        }
        for arm in _ARMS
        for regime in _REGIMES
    )
    for regime in _REGIMES:
        assert result["regimes"][regime]["schedule"] == schedule_reports[regime]
        assert result["regimes"][regime]["paired_h0_identity"] == (
            paired_h0_reports[regime]
        )
    assert result["source_end"] == source_end
    assert result["qualification"]["passed"] is False
    assert isinstance(result["total_wall_seconds"], float)
    assert math.isfinite(result["total_wall_seconds"])
    assert result["total_wall_seconds"] >= 0.0


@requires_gate_c
def test_formal_runner_audits_then_rebuilds_each_arm_with_bounded_live_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    wrapper = _gate_c_initialization_wrapper(admission)
    config = _reduced_gate_c_config()
    gate_a_data = object()
    gate_b_data = (object(), object())
    schedule_reports = {
        "gate_a": {name: "a" * 64 for name in _GATE_A_SCHEDULE_SHA256},
        "gate_b": {
            "training_global_sha256": {
                name: "b" * 64 for name in _GATE_B_TRAINING_SHA256
            },
            "validation_sha256": {
                name: "c" * 64 for name in _GATE_B_VALIDATION_SHA256
            },
        },
    }
    expected_order = [
        (regime, arm) for regime in _REGIMES for arm in _ARMS
    ]

    class TrackedModel:
        __slots__ = ("regime", "arm", "__weakref__")

        def __init__(self, regime: str, arm: str) -> None:
            self.regime = regime
            self.arm = arm

    class TrackedTrainer:
        __slots__ = ("regime", "arm", "optimizer", "__weakref__")

        def __init__(self, regime: str, arm: str) -> None:
            self.regime = regime
            self.arm = arm
            self.optimizer = object()

    model_refs: list[weakref.ReferenceType[Any]] = []
    trainer_refs: list[weakref.ReferenceType[Any]] = []
    max_live_models = 0
    max_live_trainers = 0
    initialization_calls = {pair: 0 for pair in expected_order}
    audit_model_ids: dict[tuple[str, str], int] = {}
    audit_trainer_ids: dict[tuple[str, str], int] = {}
    audit_reports: dict[tuple[str, str], dict[str, Any]] = {}
    latest_model_ids: dict[tuple[str, str], int] = {}
    latest_trainer_ids: dict[tuple[str, str], int] = {}
    run_order: list[tuple[str, str]] = []
    reproduced_before_update: list[bool] = []

    def alive(refs: list[weakref.ReferenceType[Any]]) -> int:
        gc.collect()
        return sum(reference() is not None for reference in refs)

    def new_model(
        actual_config: Any,
        regime: str,
        arm: str,
        *,
        batch_size: int,
    ) -> TrackedModel:
        nonlocal max_live_models
        assert actual_config is config
        assert batch_size > 0
        model = TrackedModel(regime, arm)
        model_refs.append(weakref.ref(model))
        max_live_models = max(max_live_models, alive(model_refs))
        return model

    def make_trainer(
        model: TrackedModel,
        actual_config: Any,
        regime: str,
        arm: str,
    ) -> TrackedTrainer:
        nonlocal max_live_trainers
        assert actual_config is config
        assert (model.regime, model.arm) == (regime, arm)
        trainer = TrackedTrainer(regime, arm)
        trainer_refs.append(weakref.ref(trainer))
        max_live_trainers = max(max_live_trainers, alive(trainer_refs))
        return trainer

    def initialization_report(
        model: TrackedModel,
        trainer: TrackedTrainer,
        actual_admission: Mapping[str, Any],
        regime: str,
        arm: str,
    ) -> dict[str, Any]:
        pair = (regime, arm)
        assert actual_admission is admission
        assert (model.regime, model.arm) == pair
        assert (trainer.regime, trainer.arm) == pair
        initialization_calls[pair] += 1
        report = _minimal_formal_arm_initialization_report(
            admission,
            regime,
            arm,
        )
        if initialization_calls[pair] == 1:
            audit_model_ids[pair] = id(model)
            audit_trainer_ids[pair] = id(trainer)
            audit_reports[pair] = copy.deepcopy(report)
        else:
            assert initialization_calls[pair] == 2
            assert id(model) != audit_model_ids[pair]
            assert id(trainer) != audit_trainer_ids[pair]
            assert report == audit_reports[pair]
        latest_model_ids[pair] = id(model)
        latest_trainer_ids[pair] = id(trainer)
        return report

    def run_arm(
        model: TrackedModel,
        trainer: TrackedTrainer,
        data: Any,
        actual_config: Any,
        regime: str,
        arm: str,
        *,
        initialization_report: Mapping[str, Any],
        execution_index: int,
        data_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        pair = (regime, arm)
        assert actual_config is config
        assert execution_index == expected_order.index(pair)
        assert data_identity == schedule_reports[regime]
        assert data is (gate_a_data if regime == "gate_a" else gate_b_data)
        assert all(initialization_calls[item] >= 1 for item in expected_order)
        reproduced_before_update.append(
            initialization_calls[pair] == 2
            and id(model) == latest_model_ids[pair]
            and id(trainer) == latest_trainer_ids[pair]
            and initialization_report == audit_reports[pair]
        )
        run_order.append(pair)
        report = _minimal_formal_arm_report(
            admission,
            regime,
            arm,
            initialization_report=initialization_report,
        )
        report["training"].update(
            {
                "execution_index": execution_index,
                "intervention": dataclasses.asdict(gate_c.ARM_SPECS[arm]),
                "data_identity": copy.deepcopy(data_identity),
            }
        )
        return report

    monkeypatch.setattr(
        gate_c,
        "_validated_gate_c_initialization_admission",
        lambda *args, **kwargs: admission,
    )
    monkeypatch.setattr(
        gate_c,
        "_normalized_prerequisites",
        lambda value: copy.deepcopy(value),
    )
    monkeypatch.setattr(gate_c, "_regenerate_gate_a_data", lambda value: gate_a_data)
    monkeypatch.setattr(gate_c, "_regenerate_gate_b_data", lambda value: gate_b_data)
    monkeypatch.setattr(
        gate_c,
        "_actual_schedule_identity_report",
        lambda *args: copy.deepcopy(schedule_reports),
    )
    monkeypatch.setattr(gate_c, "_new_model_for_arm", new_model)
    monkeypatch.setattr(
        gate_c,
        "_copy_shared_initialization",
        lambda canonical, legacy_model: {"all_equal": True},
    )
    monkeypatch.setattr(gate_c, "_make_arm_trainer", make_trainer)
    monkeypatch.setattr(
        gate_c,
        "_formal_arm_initialization_report",
        initialization_report,
    )
    monkeypatch.setattr(gate_c, "_run_gate_c_arm", run_arm)
    monkeypatch.setattr(
        gate_c,
        "_mechanism_oracle",
        lambda *args, **kwargs: {"complete": False},
    )
    monkeypatch.setattr(
        gate_c,
        "_paired_h0_identity_report",
        lambda *args, regime, **kwargs: _formal_paired_h0_identity(
            admission,
            regime,
        ),
    )
    monkeypatch.setattr(
        gate_c,
        "_qualification_report",
        lambda *args, **kwargs: {
            "criteria": {name: False for name in gate_c.QUALIFICATION_CRITERIA},
            "passed": False,
            "interpretation": "gate_c_failed_stop_no_causal_mechanism_conclusion",
        },
    )

    gate_c.run_gate_c(
        config,
        prerequisites={
            "gate_a": dict(_GATE_A_REFERENCE),
            "gate_b": dict(_GATE_B_REFERENCE),
            "gate_c_initialization": wrapper,
        },
        source_start=_passing_gate_c_source(),
        source_end_reporter=_passing_gate_c_source,
        source_files=_current_gate_c_source_files(),
        environment=_passing_gate_c_environment(),
    )

    assert run_order == expected_order
    assert all(initialization_calls[pair] == 2 for pair in expected_order)
    assert all(reproduced_before_update)
    assert max_live_models <= 2
    assert max_live_trainers <= 1


def _reduced_formal_arm_initialization_report(
    model: LatentWorkspaceModel,
    trainer: Any,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    values = legacy._parameter_values(model)
    tree = "legacy" if arm == "legacy" else "canonical_full"
    return {
        "initialization": {
            "tree": tree,
            "parameter_sha256": legacy._array_digest(values),
            "parameter_count": sum(
                np.asarray(leaf).size
                for value in values.values()
                for leaf in jax.tree.leaves(value)
            ),
            "parameter_paths": sorted(values),
            "shared_paths": {
                "paths": list(_SHARED_PATHS),
                "all_equal": True,
            },
        },
        "optimizer": gate_c._optimizer_initial_state_report(
            trainer,
            regime=regime,
            arm=arm,
        ),
        "compiler": copy.deepcopy(trainer.compiler),
    }


@pytest.fixture(scope="module")
def reduced_formal_terminal_and_frozen_reports() -> dict[str, Any]:
    config = _reduced_gate_c_config()
    gate_a_data = gate_c._regenerate_gate_a_data(config)
    gate_b_data = gate_c._regenerate_gate_b_data(config)
    data_by_regime = {
        "gate_a": gate_a_data,
        "gate_b": gate_b_data,
    }
    expected_weights: dict[tuple[str, str], np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {
        arm: {} for arm in ("terminal_only", "frozen_write")
    }
    frozen_values: dict[str, dict[str, np.ndarray]] = {}

    for regime in _REGIMES:
        regime_config = (
            config.gate_a_config
            if regime == "gate_a"
            else config.gate_b_config
        )
        for arm in ("terminal_only", "frozen_write"):
            model = gate_c._new_model_for_arm(
                config,
                regime,
                arm,
                batch_size=regime_config.batch_size,
            )
            trainer = gate_c._make_arm_trainer(
                model,
                config,
                regime,
                arm,
            )
            initialization = _reduced_formal_arm_initialization_report(
                model,
                trainer,
                regime,
                arm,
            )
            if regime == "gate_a":
                weights = gate_c._loss_weights(
                    regime,
                    arm,
                    efforts=np.ones(
                        (regime_config.training_updates,),
                        dtype=np.int32,
                    ),
                )
            else:
                schedule = gate_b_data[0]
                weights = np.concatenate(
                    [
                        gate_c._loss_weights(
                            regime,
                            arm,
                            efforts=np.asarray(
                                gate_b._encode_training_chunk(
                                    schedule_chunk,
                                    config.gate_b_config,
                                ).efforts
                            ),
                        )
                        for schedule_chunk in gate_b._iter_schedule_chunks(
                            schedule,
                            config.gate_b_config,
                        )
                    ],
                    axis=0,
                )
            expected_weights[(regime, arm)] = np.asarray(weights)
            before_write = np.array(
                _parameter_states(model)["memory_write_scale"].value,
                copy=True,
            )
            reports[arm][regime] = gate_c._run_gate_c_arm(
                model,
                trainer,
                data_by_regime[regime],
                config,
                regime,
                arm,
                initialization_report=initialization,
            )
            frozen_values[f"{regime}:{arm}"] = {
                "before": before_write,
                "after": np.asarray(
                    _parameter_states(model)["memory_write_scale"].value
                ),
            }
    return {
        "config": config,
        "reports": reports,
        "weights": expected_weights,
        "frozen_values": frozen_values,
    }


@requires_gate_c
def test_reduced_formal_arms_retain_exact_updates_losses_and_masks(
    reduced_formal_terminal_and_frozen_reports: dict[str, Any],
) -> None:
    run = reduced_formal_terminal_and_frozen_reports
    config = run["config"]
    for arm in ("terminal_only", "frozen_write"):
        for regime in _REGIMES:
            report = run["reports"][arm][regime]
            training = report["training"]
            regime_config = (
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            )
            weights = run["weights"][(regime, arm)]
            assert training["algorithm"] == "production_pp_prop"
            assert training["executed_updates"] == regime_config.training_updates
            assert training["optimizer_final_step"] == regime_config.training_updates
            assert len(training["losses"]) == regime_config.training_updates
            assert np.isfinite(np.asarray(training["losses"])).all()
            assert training["initial_parameter_sha256"] == report[
                "initialization"
            ]["parameter_sha256"]
            assert training["final_parameter_sha256"] != training[
                "initial_parameter_sha256"
            ]
            assert training["loss_weights"] == {
                "dtype": weights.dtype.str,
                "shape": list(weights.shape),
                "sha256": legacy._digest_arrays(weights),
            }
            assert set(training["finite"]) == {
                "logits",
                "model_states",
                "gradients",
                "pp_prop_traces",
                "adam",
                "parameters",
            }
            assert all(training["finite"].values())
            assert all(
                math.isfinite(value) and value >= 0.0
                for value in training["max_abs"].values()
            )
            assert all(value > 0 for value in training["value_count"].values())
            assert report["parameter_movement"]["l2_delta"] > 0.0
            assert report["evaluation"]["finite"] is True

            if arm == "terminal_only":
                if regime == "gate_a":
                    np.testing.assert_array_equal(
                        np.flatnonzero(weights),
                        [weights.shape[0] - 1],
                    )
                else:
                    assert weights.ndim == 2
                    assert all(np.count_nonzero(row) == 1 for row in weights)
                np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
            elif regime == "gate_a":
                np.testing.assert_allclose(weights[-2:], [0.5, 0.5])
                assert np.count_nonzero(weights) == 2
            else:
                np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


@requires_gate_c
def test_reduced_formal_frozen_write_retains_literal_write_and_zero_movement(
    reduced_formal_terminal_and_frozen_reports: dict[str, Any],
) -> None:
    run = reduced_formal_terminal_and_frozen_reports
    for regime in _REGIMES:
        report = run["reports"]["frozen_write"][regime]
        evidence = report["training"]["frozen_write"]
        values = run["frozen_values"][f"{regime}:frozen_write"]
        assert evidence == {
            "applicable": True,
            "all_ones_before": True,
            "all_ones_after": True,
            "excluded_from_optimizer": True,
        }
        np.testing.assert_array_equal(values["before"], 1.0)
        np.testing.assert_array_equal(values["after"], values["before"])
        assert report["parameter_movement"]["paths"][
            "memory_write_scale"
        ] == {
            "l2_delta": 0.0,
            "parameter_count": values["before"].size,
        }


@requires_gate_c
def test_reduced_formal_paired_evaluations_derive_finite_metrics(
    reduced_formal_terminal_and_frozen_reports: dict[str, Any],
) -> None:
    reports = reduced_formal_terminal_and_frozen_reports["reports"]
    for arm in ("terminal_only", "frozen_write"):
        metrics = gate_c._metric_summary(
            reports[arm]["gate_a"]["evaluation"],
            reports[arm]["gate_b"]["evaluation"],
        )
        assert set(metrics) == {"binding_gap", "depth_accuracy"}
        assert all(math.isfinite(value) for value in metrics.values())
        assert -1.0 <= metrics["binding_gap"] <= 1.0
        assert 0.0 <= metrics["depth_accuracy"] <= 1.0


@pytest.fixture(scope="module")
def reduced_finite_window_oracle_inputs() -> dict[str, Any]:
    config = _reduced_gate_c_config()
    production_gate_b_config = gate_b.DepthGateConfig()
    schedule = gate_b._build_schedule(production_gate_b_config)
    validation = gate_b._encode_validation_data(
        schedule,
        production_gate_b_config,
    )
    full = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "full",
        batch_size=1,
    )
    initial_values = legacy._parameter_values(full)
    initial_sha256 = legacy._array_digest(initial_values)
    initialization = {
        "initialization": {
            "gate_b": {
                "canonical_full": {
                    "parameter_sha256": initial_sha256,
                    "parameter_count": sum(
                        np.asarray(leaf).size
                        for value in initial_values.values()
                        for leaf in jax.tree.leaves(value)
                    ),
                    "parameter_paths": list(_FULL_PATHS),
                },
                "arm_initialization_refs": {
                    arm: {
                        "tree": "canonical_full",
                        "parameter_sha256": initial_sha256,
                    }
                    for arm in ("full", "query_only", "terminal_only")
                },
            }
        }
    }
    return {
        "config": config,
        "initialization": initialization,
        "data": (schedule, validation),
    }


def _assert_oracle_numeric_record(record: Mapping[str, Any]) -> None:
    assert set(record) == {
        "full_norm",
        "arm_norm",
        "l2_difference",
        "relative_deviation",
        "relative_deviation_defined",
        "cosine",
        "cosine_defined",
        "full_sha256",
        "arm_sha256",
    }
    for name in ("full_norm", "arm_norm", "l2_difference"):
        assert not isinstance(record[name], (bool, np.bool_))
        assert math.isfinite(record[name])
        assert record[name] >= 0.0
    assert record["relative_deviation_defined"] is (
        record["relative_deviation"] is not None
    )
    assert record["cosine_defined"] is (record["cosine"] is not None)
    for name in ("relative_deviation", "cosine"):
        if record[name] is not None:
            assert not isinstance(record[name], (bool, np.bool_))
            assert math.isfinite(record[name])
    assert re.fullmatch(r"[0-9a-f]{64}", record["full_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", record["arm_sha256"])


def _oracle_threshold_passed(record: Mapping[str, Any]) -> bool:
    return bool(
        record["full_norm"] > 0.0
        and record["relative_deviation_defined"] is True
        and record["relative_deviation"] >= 1e-3
        and record["l2_difference"]
        > max(1e-8, 1e-4 * record["full_norm"])
    )


@requires_gate_c
def test_reduced_oracle_executes_real_finite_window_pp_prop_and_thresholds(
    reduced_finite_window_oracle_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from braintrace._testing.oracle import (
        chunked_online_param_gradients as real_chunked_gradients,
    )

    calls: list[int] = []

    def recording_chunked_gradients(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["chunk_size"])
        return real_chunked_gradients(*args, **kwargs)

    monkeypatch.setattr(
        gate_c,
        "chunked_online_param_gradients",
        recording_chunked_gradients,
        raising=False,
    )
    report = gate_c._mechanism_oracle(
        reduced_finite_window_oracle_inputs["config"],
        initialization=reduced_finite_window_oracle_inputs[
            "initialization"
        ],
        gate_b_data=reduced_finite_window_oracle_inputs["data"],
    )

    assert calls == [1, 1, 1]
    assert set(report) == {
        "contract",
        "objective",
        "gradient_chunk_size",
        "comparisons",
        "complete",
    }
    assert report["contract"] == gate_c._oracle_contract(
        reduced_finite_window_oracle_inputs["config"]
    )
    assert report["objective"] == {
        "wrapper": "sqrt_weight_times_sqrt_cross_entropy",
        "unsupervised_output_exact_zero": True,
    }
    assert report["gradient_chunk_size"] == 1
    assert set(report["comparisons"]) == {
        "query_only",
        "terminal_only",
    }
    comparison_passes: list[bool] = []
    for arm, comparison in report["comparisons"].items():
        assert set(comparison) == {
            "global",
            "paths",
            "required_paths",
            "required_paths_passed",
            "passed",
        }
        _assert_oracle_numeric_record(comparison["global"])
        assert set(comparison["paths"]) == set(_FULL_PATHS)
        for path_record in comparison["paths"].values():
            _assert_oracle_numeric_record(path_record)
        required = (
            [
                "memory_read_projection/weight",
                "workspace_query_projection/weight",
            ]
            if arm == "query_only"
            else []
        )
        assert comparison["required_paths"] == required
        expected_required = all(
            _oracle_threshold_passed(comparison["paths"][path])
            for path in required
        )
        assert comparison["required_paths_passed"] is expected_required
        expected_pass = bool(
            _oracle_threshold_passed(comparison["global"])
            and expected_required
        )
        assert comparison["passed"] is expected_pass
        comparison_passes.append(expected_pass)
    assert report["complete"] is all(comparison_passes)


@requires_gate_c
def test_reduced_mechanism_oracle_audits_every_model_construction(
    reduced_finite_window_oracle_inputs: dict[str, Any],
) -> None:
    initialization = reduced_finite_window_oracle_inputs["initialization"]
    audit = gate_c._GateC2ControlsAudit(initialization)
    previous = gate_c._ACTIVE_GATE_C2_CONTROLS_AUDIT
    try:
        gate_c._ACTIVE_GATE_C2_CONTROLS_AUDIT = audit
        with audit:
            gate_c._mechanism_oracle(
                reduced_finite_window_oracle_inputs["config"],
                initialization=initialization,
                gate_b_data=reduced_finite_window_oracle_inputs["data"],
            )
    finally:
        gate_c._ACTIVE_GATE_C2_CONTROLS_AUDIT = previous

    evidence = audit.report()
    expected_roles = [
        role
        for role in _GATE_C2_CONTROL_MODEL_ROLES
        if role.startswith("gate_b:mechanism_oracle:")
    ]
    assert evidence["model_factory_calls"] == expected_roles
    assert evidence["model_constructor_calls"] == expected_roles
    assert list(evidence["materialized_roles"]) == expected_roles
    for role in expected_roles:
        regime, probe, policy = _GATE_C2_CONTROL_MODEL_ROLES[role]
        expected = initialization["initialization"][regime]["canonical_full"][
            "parameter_sha256"
        ]
        assert evidence["materialized_roles"][role] == {
            "regime": regime,
            "probe": probe,
            "policy": policy,
            "initialization_tree": "canonical_full",
            "expected_parameter_sha256": expected,
            "before_parameter_sha256": expected,
            "after_parameter_sha256": expected,
            "parameters_equal": True,
        }
    assert evidence["trainer_factory_calls"] == []
    assert evidence["training_step_calls"] == []
    assert evidence["optimizer_constructor_calls"] == []
    assert evidence["optimizer_update_calls"] == []
    assert evidence["complete"] is False


@requires_gate_c
def test_oracle_rejects_event_digest_mismatch_before_gradients(
    reduced_finite_window_oracle_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, validation = reduced_finite_window_oracle_inputs["data"]
    tampered_events = np.array(validation.intact, copy=True)
    tampered_events[0, 0, 0] = np.nextafter(
        tampered_events[0, 0, 0],
        np.float32(np.inf),
    )
    tampered = dataclasses.replace(validation, intact=tampered_events)
    calls = 0

    def forbidden_gradients(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("gradient execution preceded oracle authentication")

    monkeypatch.setattr(
        gate_c,
        "chunked_online_param_gradients",
        forbidden_gradients,
        raising=False,
    )

    with pytest.raises(ValueError, match="event|digest|oracle"):
        gate_c._mechanism_oracle(
            reduced_finite_window_oracle_inputs["config"],
            initialization=reduced_finite_window_oracle_inputs[
                "initialization"
            ],
            gate_b_data=(schedule, tampered),
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("oracle_validation_index", 1),
        ("oracle_effort", 4),
        ("gradient_chunk_size", 2),
    ),
)
@requires_gate_c
def test_oracle_rejects_nonpreregistered_coordinates_before_gradients(
    field: str,
    value: int,
    reduced_finite_window_oracle_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dataclasses.replace(
        reduced_finite_window_oracle_inputs["config"],
        **{field: value},
    )
    calls = 0

    def forbidden_gradients(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("nonpreregistered oracle reached gradient execution")

    monkeypatch.setattr(
        gate_c,
        "chunked_online_param_gradients",
        forbidden_gradients,
    )

    with pytest.raises(ValueError, match="oracle|preregister|coordinate|config"):
        gate_c._mechanism_oracle(
            config,
            initialization=reduced_finite_window_oracle_inputs[
                "initialization"
            ],
            gate_b_data=reduced_finite_window_oracle_inputs["data"],
        )
    assert calls == 0


@pytest.mark.parametrize("mutation", ("wrong_geometry", "path_collision"))
@requires_gate_c
def test_oracle_rejects_gradient_geometry_and_normalization_collisions(
    mutation: str,
    reduced_finite_window_oracle_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong = {
        path: np.zeros((1,), dtype=np.float32) for path in _FULL_PATHS
    }
    if mutation == "path_collision":
        wrong[("memory_read_projection", "weight")] = np.ones(
            (1,),
            dtype=np.float32,
        )
    calls = 0

    def fake_gradients(*args: Any, **kwargs: Any) -> dict[Any, Any]:
        nonlocal calls
        calls += 1
        return copy.deepcopy(wrong)

    monkeypatch.setattr(
        gate_c,
        "chunked_online_param_gradients",
        fake_gradients,
    )

    with pytest.raises(ValueError, match="gradient|geometry|shape|collision|path"):
        gate_c._mechanism_oracle(
            reduced_finite_window_oracle_inputs["config"],
            initialization=reduced_finite_window_oracle_inputs[
                "initialization"
            ],
            gate_b_data=reduced_finite_window_oracle_inputs["data"],
        )
    assert 1 <= calls <= 3


@requires_gate_c
def test_oracle_rejects_validation_metadata_disagreeing_with_schedule(
    reduced_finite_window_oracle_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule, validation = reduced_finite_window_oracle_inputs["data"]
    mapping_ids = np.array(validation.mapping_ids, copy=True)
    mapping_ids[0] = mapping_ids[0] + 1
    tampered = dataclasses.replace(validation, mapping_ids=mapping_ids)
    calls = 0

    def forbidden_gradients(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("metadata mismatch reached gradient execution")

    monkeypatch.setattr(
        gate_c,
        "chunked_online_param_gradients",
        forbidden_gradients,
    )

    with pytest.raises(ValueError, match="metadata|schedule|oracle|mapping"):
        gate_c._mechanism_oracle(
            reduced_finite_window_oracle_inputs["config"],
            initialization=reduced_finite_window_oracle_inputs[
                "initialization"
            ],
            gate_b_data=(schedule, tampered),
        )
    assert calls == 0


@requires_gate_c
def test_oracle_rejects_fresh_hidden_state_snapshot_mismatch_before_gradients(
    reduced_finite_window_oracle_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_init_all_states = brainstate.nn.init_all_states
    tampered_initializations = 0
    gradient_calls = 0

    def tampering_init_all_states(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal tampered_initializations
        result = real_init_all_states(model, *args, **kwargs)
        if (
            isinstance(model, LatentWorkspaceModel)
            and model.config.batch_size == 1
            and model.memory_read_policy == "query_only"
        ):
            tampered_initializations += 1
            model.context_memory.value = jnp.ones_like(
                model.context_memory.value
            )
        return result

    def forbidden_gradients(*args: Any, **kwargs: Any) -> Any:
        nonlocal gradient_calls
        gradient_calls += 1
        raise AssertionError("state mismatch reached gradient execution")

    monkeypatch.setattr(
        brainstate.nn,
        "init_all_states",
        tampering_init_all_states,
    )
    monkeypatch.setattr(
        gate_c,
        "chunked_online_param_gradients",
        forbidden_gradients,
    )

    with pytest.raises(ValueError, match="state|snapshot|initialization|oracle"):
        gate_c._mechanism_oracle(
            reduced_finite_window_oracle_inputs["config"],
            initialization=reduced_finite_window_oracle_inputs[
                "initialization"
            ],
            gate_b_data=reduced_finite_window_oracle_inputs["data"],
        )
    assert tampered_initializations >= 1
    assert gradient_calls == 0


_GATE_C2_CRITERIA = (
    "schema_and_control",
    "exact_configuration",
    "prerequisites_authenticated",
    "initialization_authenticated",
    "canonical_schedules_complete",
    "consumed_gate_b_loss_weights_exact",
    "fresh_isolated_optimizers",
    "compiler_and_training_complete",
    "full_gate_a_passed",
    "full_gate_b_passed",
    "blocking_behavioral_margins",
    "paired_h0_operational_equivalence",
    "frozen_write_complete",
    "mechanism_oracle_complete",
    "source_and_gpu_authenticated",
)
_GATE_C2_CONTROLS_CRITERIA = (
    "schema_and_control",
    "exact_configuration",
    "prerequisites_authenticated",
    "initialization_authenticated",
    "canonical_schedules_complete",
    "no_behavioral_or_optimizer_updates",
    "paired_h0_operational_equivalence",
    "mechanism_oracle_complete",
    "source_and_gpu_authenticated",
)
_GATE_C2_CONTROL_AUDIT_LABELS = (
    "braintools.optim.Adam.__init__",
    "braintools.optim.Adam.update",
    (
        "examples.pp_prop.latent_workspace_ablation_gate."
        "GateCTrainer.train_chunk"
    ),
    "examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer",
    (
        "examples.pp_prop.latent_workspace_binding_gate."
        "_PPPropTrainer.train"
    ),
    "examples.pp_prop.latent_workspace_binding_gate._make_pp_prop_trainer",
    (
        "examples.pp_prop.latent_workspace_depth_gate."
        "_DepthPPPropTrainer.train_chunk"
    ),
    "examples.pp_prop.latent_workspace_depth_gate._make_pp_prop_trainer",
)
_GATE_C2_CONTROL_MODEL_ROLES = {
    "gate_a:paired_h0:copied_full": (
        "gate_a",
        "paired_h0_operational_equivalence",
        "full",
    ),
    "gate_a:paired_h0:full_reference": (
        "gate_a",
        "paired_h0_operational_equivalence",
        "full",
    ),
    "gate_a:paired_h0:query_only": (
        "gate_a",
        "paired_h0_operational_equivalence",
        "query_only",
    ),
    "gate_a:query_only_latent_no_read:full_positive_control": (
        "gate_a",
        "full_positive_control",
        "full",
    ),
    "gate_a:query_only_latent_no_read:query_only": (
        "gate_a",
        "query_only_latent_no_read",
        "query_only",
    ),
    "gate_a:removed_path_finite_window:gate_a_h1": (
        "gate_a",
        "removed_path_finite_window_influence",
        "query_only",
    ),
    "gate_b:mechanism_oracle:full:finite_window": (
        "gate_b",
        "mechanism_oracle:full:finite_window",
        "full",
    ),
    "gate_b:mechanism_oracle:full:reference": (
        "gate_b",
        "mechanism_oracle:full:reference",
        "full",
    ),
    "gate_b:mechanism_oracle:query_only:finite_window": (
        "gate_b",
        "mechanism_oracle:query_only:finite_window",
        "query_only",
    ),
    "gate_b:mechanism_oracle:query_only:reference": (
        "gate_b",
        "mechanism_oracle:query_only:reference",
        "query_only",
    ),
    "gate_b:mechanism_oracle:terminal_only:finite_window": (
        "gate_b",
        "mechanism_oracle:terminal_only:finite_window",
        "full",
    ),
    "gate_b:mechanism_oracle:terminal_only:reference": (
        "gate_b",
        "mechanism_oracle:terminal_only:reference",
        "full",
    ),
    "gate_b:paired_h0:copied_full": (
        "gate_b",
        "paired_h0_operational_equivalence",
        "full",
    ),
    "gate_b:paired_h0:full_reference": (
        "gate_b",
        "paired_h0_operational_equivalence",
        "full",
    ),
    "gate_b:paired_h0:query_only": (
        "gate_b",
        "paired_h0_operational_equivalence",
        "query_only",
    ),
    "gate_b:query_only_latent_no_read:full_positive_control": (
        "gate_b",
        "full_positive_control",
        "full",
    ),
    "gate_b:query_only_latent_no_read:query_only": (
        "gate_b",
        "query_only_latent_no_read",
        "query_only",
    ),
    "gate_b:removed_path_finite_window:gate_b_index0_r8_h8": (
        "gate_b",
        "removed_path_finite_window_influence",
        "query_only",
    ),
}
_GATE_C2_NONTERMINAL_WEIGHT_REPORT = {
    "dtype": "<f8",
    "shape": [4_096, 19],
    "sha256": "044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31",
}
_GATE_C2_TERMINAL_WEIGHT_REPORT = {
    "dtype": "<f8",
    "shape": [4_096, 19],
    "sha256": "f381a6b856be26071898fc7427ee1f098bbb333b3305dc3f833c5e80750e1970",
}
_GATE_C2_HIDDEN_GEOMETRY = {
    "context_memory#0": (512, 32, 32),
    "ff_syn/post/V#0": (512, 2_048),
    "ff_syn/syn/g#0": (512, 2_048),
    "memory_read#0": (512, 32),
    "query_encoding#0": (512, 32),
    "reasoning_query#0": (512, 32),
    "rec_syn/syn/g#0": (512, 2_048),
    "workspace_carrier#0": (512, 2_048),
}
_GATE_C2_CACHED_READ_REPLACEMENTS = {
    "plus_11": {
        "fill_value": 11.0,
        "dtype": "<f4",
        "shape": [512, 32],
        "sha256": "156517ec70f2d721974202ac8581ca7f15594db382051fafbac40fb9057c81bc",
    },
    "minus_11": {
        "fill_value": -11.0,
        "dtype": "<f4",
        "shape": [512, 32],
        "sha256": "b5725644875e21d4fce1fe5116695c12d18af3d9b8f243cbdd6878c3404201f6",
    },
}


@requires_gate_c
def test_gate_c2_constants_keep_v1_identity_separate_and_exact() -> None:
    assert gate_c.GATE_C_SCHEMA_VERSION == 1
    assert gate_c.GATE_C_CONTROL == "example21_pp_prop_learnability_gate_c"
    assert gate_c.QUALIFICATION_CRITERIA != _GATE_C2_CRITERIA

    assert gate_c.GATE_C2_SCHEMA_VERSION == 2
    assert gate_c.GATE_C2_CONTROL == "example21_pp_prop_learnability_gate_c2"
    assert gate_c.GATE_C2_QUALIFICATION_REGIME == "preregistered_gate_c2_full"
    assert gate_c.GATE_C2_QUALIFICATION_CRITERIA == _GATE_C2_CRITERIA
    assert gate_c.GATE_C2_PASSING_INTERPRETATION == (
        "gate_c2_passed_pp_prop_learnability_mechanism"
    )
    assert gate_c.GATE_C2_FAILING_INTERPRETATION == (
        "gate_c2_failed_stop_no_causal_mechanism_conclusion"
    )
    assert gate_c.GATE_C2_TOP_LEVEL_KEYS == (
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "arms",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
        "total_wall_seconds",
    )

    assert gate_c.GATE_C2_CONTROLS_SCHEMA_VERSION == 1
    assert gate_c.GATE_C2_CONTROLS_CONTROL == (
        "example21_gate_c2_pretraining_control_admission"
    )
    assert gate_c.GATE_C2_CONTROLS_QUALIFICATION_REGIME == (
        "preregistered_gate_c2_pretraining_controls"
    )
    assert gate_c.GATE_C2_CONTROLS_QUALIFICATION_CRITERIA == (
        _GATE_C2_CONTROLS_CRITERIA
    )
    assert gate_c.GATE_C2_CONTROLS_AUDIT_LABELS == (
        _GATE_C2_CONTROL_AUDIT_LABELS
    )
    assert list(gate_c.GATE_C2_CONTROLS_MODEL_ROLES) == list(
        _GATE_C2_CONTROL_MODEL_ROLES
    )
    assert {
        role: (
            value["regime"],
            value["probe"],
            value["policy"],
        )
        for role, value in gate_c.GATE_C2_CONTROLS_MODEL_ROLES.items()
    } == _GATE_C2_CONTROL_MODEL_ROLES
    assert gate_c.GATE_C2_CONTROLS_TOP_LEVEL_KEYS == (
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
        "total_wall_seconds",
    )
    assert gate_c.GATE_C2_CACHED_READ_REPLACEMENTS == (
        _GATE_C2_CACHED_READ_REPLACEMENTS
    )


@requires_gate_c
def test_gate_c2_qualifiers_fail_closed_with_exact_criterion_sets() -> None:
    config = gate_c.GateCConfig()

    formal = gate_c._gate_c2_qualification_report({}, config=config)
    controls = gate_c._gate_c2_controls_qualification_report({}, config=config)

    assert formal == {
        "criteria": {name: False for name in _GATE_C2_CRITERIA},
        "passed": False,
        "interpretation": (
            "gate_c2_failed_stop_no_causal_mechanism_conclusion"
        ),
    }
    assert controls == {
        "criteria": {name: False for name in _GATE_C2_CONTROLS_CRITERIA},
        "passed": False,
        "interpretation": "gate_c2_pretraining_controls_failed_stop",
    }


def _canonical_gate_b_loss_weight_inputs() -> tuple[np.ndarray, np.ndarray]:
    schedule = gate_b._build_schedule(gate_b.DepthGateConfig())
    efforts = np.asarray(schedule.training_efforts)
    weights = np.zeros((efforts.size, 19), dtype=np.float64)
    for effort in gate_b.QUALIFYING_EFFORTS:
        selected = efforts == effort
        weights[selected, 10 : 11 + effort] = 1.0 / (effort + 1)
    weights.setflags(write=False)
    return weights, efforts


@requires_gate_c
def test_gate_c2_gate_b_weights_preserve_canonical_host_arrays_and_dtype() -> None:
    canonical, efforts = _canonical_gate_b_loss_weight_inputs()
    assert {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": legacy._digest_arrays(canonical),
    } == _GATE_C2_NONTERMINAL_WEIGHT_REPORT

    reports: dict[str, dict[str, Any]] = {}
    for arm in ("full", "query_only", "legacy", "frozen_write"):
        consumed = gate_c._gate_c2_gate_b_loss_weights(
            canonical,
            arm=arm,
            efforts=efforts,
        )
        assert consumed is canonical
        reports[arm] = {
            "dtype": consumed.dtype.str,
            "shape": list(consumed.shape),
            "sha256": legacy._digest_arrays(consumed),
        }

    terminal = gate_c._gate_c2_gate_b_loss_weights(
        canonical,
        arm="terminal_only",
        efforts=efforts,
    )
    assert terminal is not canonical
    assert not np.shares_memory(terminal, canonical)
    assert terminal.dtype == canonical.dtype == np.dtype(np.float64)
    for row, effort in enumerate(efforts):
        nonzero = np.flatnonzero(terminal[row])
        np.testing.assert_array_equal(nonzero, np.asarray([10 + effort]))
        assert terminal[row, 10 + effort] == 1.0
    reports["terminal_only"] = {
        "dtype": terminal.dtype.str,
        "shape": list(terminal.shape),
        "sha256": legacy._digest_arrays(terminal),
    }
    assert reports["terminal_only"] == _GATE_C2_TERMINAL_WEIGHT_REPORT
    assert gate_c._gate_c2_consumed_gate_b_loss_weights_complete(reports)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_arm",
        "nonterminal_dtype",
        "nonterminal_shape",
        "nonterminal_digest",
        "terminal_digest",
        "extra_arm",
    ),
)
@requires_gate_c
def test_gate_c2_consumed_weight_validator_rejects_all_five_arm_failures(
    mutation: str,
) -> None:
    reports = {
        arm: dict(_GATE_C2_NONTERMINAL_WEIGHT_REPORT)
        for arm in ("full", "query_only", "legacy", "frozen_write")
    }
    reports["terminal_only"] = dict(_GATE_C2_TERMINAL_WEIGHT_REPORT)
    assert gate_c._gate_c2_consumed_gate_b_loss_weights_complete(reports)

    if mutation == "missing_arm":
        reports.pop("frozen_write")
    elif mutation == "nonterminal_dtype":
        reports["full"]["dtype"] = "<f4"
    elif mutation == "nonterminal_shape":
        reports["query_only"]["shape"] = [4_095, 19]
    elif mutation == "nonterminal_digest":
        reports["legacy"]["sha256"] = "0" * 64
    elif mutation == "terminal_digest":
        reports["terminal_only"] = dict(_GATE_C2_NONTERMINAL_WEIGHT_REPORT)
    elif mutation == "extra_arm":
        reports["extra"] = dict(_GATE_C2_NONTERMINAL_WEIGHT_REPORT)
    else:
        raise AssertionError(f"unhandled consumed-weight mutation {mutation}")

    assert not gate_c._gate_c2_consumed_gate_b_loss_weights_complete(reports)


def _gate_c2_control_execution_evidence(
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role, (regime, probe, policy) in _GATE_C2_CONTROL_MODEL_ROLES.items():
        expected = admission["initialization"][regime]["canonical_full"][
            "parameter_sha256"
        ]
        roles[role] = {
            "regime": regime,
            "probe": probe,
            "policy": policy,
            "initialization_tree": "canonical_full",
            "expected_parameter_sha256": expected,
            "before_parameter_sha256": expected,
            "after_parameter_sha256": expected,
            "parameters_equal": True,
        }
    role_names = list(_GATE_C2_CONTROL_MODEL_ROLES)
    return {
        "instrumented_training_entry_points": list(
            _GATE_C2_CONTROL_AUDIT_LABELS
        ),
        "trainer_factory_calls": [],
        "trainer_factory_call_count": 0,
        "training_step_calls": [],
        "training_step_call_count": 0,
        "optimizer_constructor_calls": [],
        "optimizer_instance_count": 0,
        "optimizer_update_calls": [],
        "optimizer_update_call_count": 0,
        "model_factory_calls": role_names,
        "model_constructor_calls": list(role_names),
        "materialized_roles": roles,
        "complete": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "empty_roles",
        "trainer_call",
        "optimizer_instance",
        "optimizer_update",
        "constructor_order",
        "duplicate_role",
        "unexpected_role_field",
        "wrong_tree",
        "wrong_expected_digest",
        "changed_parameter",
        "stored_complete",
    ),
)
@requires_gate_c
def test_gate_c2_control_update_evidence_recomputes_zero_calls_and_parameters(
    mutation: str,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    evidence = _gate_c2_control_execution_evidence(admission)
    assert gate_c._gate_c2_no_update_evidence_complete(evidence, admission)

    role = evidence["model_factory_calls"][0]
    if mutation == "empty_roles":
        evidence["model_factory_calls"] = []
        evidence["model_constructor_calls"] = []
        evidence["materialized_roles"] = {}
    elif mutation == "trainer_call":
        evidence["trainer_factory_calls"] = [
            "examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer"
        ]
        evidence["trainer_factory_call_count"] = 1
    elif mutation == "optimizer_instance":
        evidence["optimizer_constructor_calls"] = [
            "braintools.optim.Adam.__init__"
        ]
        evidence["optimizer_instance_count"] = 1
    elif mutation == "optimizer_update":
        evidence["optimizer_update_calls"] = [
            "braintools.optim.Adam.update"
        ]
        evidence["optimizer_update_call_count"] = 1
    elif mutation == "constructor_order":
        evidence["model_constructor_calls"] = ["wrong-role"]
    elif mutation == "duplicate_role":
        evidence["model_factory_calls"].append(role)
        evidence["model_constructor_calls"].append(role)
    elif mutation == "unexpected_role_field":
        evidence["materialized_roles"][role]["unexpected"] = None
    elif mutation == "wrong_tree":
        evidence["materialized_roles"][role]["initialization_tree"] = "legacy"
    elif mutation == "wrong_expected_digest":
        evidence["materialized_roles"][role][
            "expected_parameter_sha256"
        ] = "0" * 64
    elif mutation == "changed_parameter":
        evidence["materialized_roles"][role][
            "after_parameter_sha256"
        ] = "0" * 64
        evidence["materialized_roles"][role]["parameters_equal"] = False
    elif mutation == "stored_complete":
        evidence["complete"] = False
    else:
        raise AssertionError(f"unhandled no-update mutation {mutation}")

    assert not gate_c._gate_c2_no_update_evidence_complete(
        evidence,
        admission,
    )


@requires_gate_c
def test_gate_c2_no_update_evidence_accepts_reordered_exact_role_mapping() -> None:
    admission, _ = _passing_gate_c_initialization_report()
    evidence = _gate_c2_control_execution_evidence(admission)
    roles = evidence["materialized_roles"]
    reordered_roles = dict(reversed(tuple(roles.items())))
    assert reordered_roles == roles
    assert list(reordered_roles) != list(roles)
    evidence["materialized_roles"] = reordered_roles

    missing = dict(evidence)
    missing_roles = dict(reordered_roles)
    missing_roles.pop(next(iter(missing_roles)))
    missing["materialized_roles"] = missing_roles
    assert not gate_c._gate_c2_no_update_evidence_complete(missing, admission)

    extra = dict(evidence)
    extra_roles = dict(reordered_roles)
    extra_roles["unexpected"] = next(iter(reordered_roles.values()))
    extra["materialized_roles"] = extra_roles
    assert not gate_c._gate_c2_no_update_evidence_complete(extra, admission)

    assert gate_c._gate_c2_no_update_evidence_complete(evidence, admission)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_role",
        "extra_role",
        "factory_order",
        "constructor_order",
        "wrong_regime",
        "wrong_probe",
        "wrong_policy",
        "missing_training_label",
        "reordered_training_labels",
        "direct_training_step",
    ),
)
@requires_gate_c
def test_gate_c2_control_audit_requires_exact_role_and_boundary_matrix(
    mutation: str,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    evidence = _gate_c2_control_execution_evidence(admission)
    assert gate_c._gate_c2_no_update_evidence_complete(evidence, admission)

    first = evidence["model_factory_calls"][0]
    if mutation == "missing_role":
        evidence["model_factory_calls"].pop(0)
        evidence["model_constructor_calls"].pop(0)
        evidence["materialized_roles"].pop(first)
    elif mutation == "extra_role":
        extra = "gate_b:unexpected:model"
        evidence["model_factory_calls"].append(extra)
        evidence["model_constructor_calls"].append(extra)
        evidence["materialized_roles"][extra] = copy.deepcopy(
            evidence["materialized_roles"][first]
        )
    elif mutation == "factory_order":
        evidence["model_factory_calls"][0:2] = reversed(
            evidence["model_factory_calls"][0:2]
        )
    elif mutation == "constructor_order":
        evidence["model_constructor_calls"][0:2] = reversed(
            evidence["model_constructor_calls"][0:2]
        )
    elif mutation == "wrong_regime":
        evidence["materialized_roles"][first]["regime"] = "gate_b"
    elif mutation == "wrong_probe":
        evidence["materialized_roles"][first]["probe"] = "colluding-probe"
    elif mutation == "wrong_policy":
        evidence["materialized_roles"][first]["policy"] = "query_only"
    elif mutation == "missing_training_label":
        evidence["instrumented_training_entry_points"].pop()
    elif mutation == "reordered_training_labels":
        evidence["instrumented_training_entry_points"][0:2] = reversed(
            evidence["instrumented_training_entry_points"][0:2]
        )
    elif mutation == "direct_training_step":
        evidence["training_step_calls"] = [
            "examples.pp_prop.latent_workspace_ablation_gate."
            "GateCTrainer.train_chunk"
        ]
        evidence["training_step_call_count"] = 1
    else:
        raise AssertionError(f"unhandled audit matrix mutation {mutation}")

    assert not gate_c._gate_c2_no_update_evidence_complete(
        evidence,
        admission,
    )


@requires_gate_c
def test_gate_c2_controls_audit_records_all_direct_forbidden_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()

    class FakeTrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.train = lambda *args, **kwargs: None
            self.train_chunk = lambda *args, **kwargs: None

    monkeypatch.setattr(gate_c, "GateCTrainer", FakeTrainer)
    monkeypatch.setattr(gate_a, "_PPPropTrainer", FakeTrainer)
    monkeypatch.setattr(gate_b, "_DepthPPPropTrainer", FakeTrainer)
    original_adam = gate_c.braintools.optim.Adam
    audit = gate_c._GateC2ControlsAudit(admission)

    with audit:
        for factory in (
            gate_c._make_arm_trainer,
            gate_a._make_pp_prop_trainer,
            gate_b._make_pp_prop_trainer,
        ):
            with pytest.raises(RuntimeError, match="forbid trainer"):
                factory()

        gate_c_trainer = gate_c.GateCTrainer()
        binding_trainer = gate_a._PPPropTrainer()
        depth_trainer = gate_b._DepthPPPropTrainer()
        for training_call in (
            gate_c_trainer.train_chunk,
            binding_trainer.train,
            depth_trainer.train_chunk,
        ):
            with pytest.raises(RuntimeError, match="forbid training"):
                training_call()

        with pytest.raises(RuntimeError, match="forbid Adam"):
            gate_c.braintools.optim.Adam()
        with pytest.raises(RuntimeError, match="forbid optimizer"):
            original_adam.update(object())

    evidence = audit.report()
    assert evidence["trainer_factory_calls"] == [
        "examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer",
        "examples.pp_prop.latent_workspace_binding_gate._make_pp_prop_trainer",
        "examples.pp_prop.latent_workspace_depth_gate._make_pp_prop_trainer",
    ]
    assert evidence["training_step_calls"] == [
        "examples.pp_prop.latent_workspace_ablation_gate.GateCTrainer.train_chunk",
        "examples.pp_prop.latent_workspace_binding_gate._PPPropTrainer.train",
        (
            "examples.pp_prop.latent_workspace_depth_gate."
            "_DepthPPPropTrainer.train_chunk"
        ),
    ]
    assert evidence["optimizer_constructor_calls"] == [
        "braintools.optim.Adam.__init__"
    ]
    assert evidence["optimizer_update_calls"] == [
        "braintools.optim.Adam.update"
    ]
    assert evidence["complete"] is False


@pytest.mark.parametrize("fail_at", range(1, 9))
@requires_gate_c
def test_gate_c2_controls_audit_rolls_back_partial_enter_installation(
    fail_at: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    original_adam = gate_c.braintools.optim.Adam
    boundary_locations = (
        (gate_c, "_make_arm_trainer"),
        (gate_a, "_make_pp_prop_trainer"),
        (gate_b, "_make_pp_prop_trainer"),
        (gate_c, "GateCTrainer"),
        (gate_a, "_PPPropTrainer"),
        (gate_b, "_DepthPPPropTrainer"),
        (gate_c.braintools.optim, "Adam"),
        (original_adam, "update"),
    )
    boundaries = tuple(
        (owner, name, inspect.getattr_static(owner, name))
        for owner, name in boundary_locations
    )
    audit = gate_c._GateC2ControlsAudit(admission)
    real_replace = audit._replace
    replace_calls = 0

    def fail_after_install(owner: Any, name: str, replacement: Any) -> None:
        nonlocal replace_calls
        replace_calls += 1
        real_replace(owner, name, replacement)
        if replace_calls == fail_at:
            raise RuntimeError("injected late audit install failure")

    monkeypatch.setattr(audit, "_replace", fail_after_install)
    try:
        with pytest.raises(RuntimeError, match="late audit install failure"):
            audit.__enter__()

        assert replace_calls == fail_at
        assert audit._restorations == []
        for owner, name, original in boundaries:
            assert inspect.getattr_static(owner, name) is original
    finally:
        audit.__exit__(None, None, None)


@requires_gate_c
def test_gate_c2_floating_difference_records_are_raw_and_recomputed() -> None:
    left = np.zeros((512, 4), dtype=np.float32)
    right = left.copy()
    right[0, 0] = np.float32(1e-7)
    record = gate_c._gate_c2_floating_difference_record(
        left,
        right,
        rms_tolerance=1e-6,
    )

    assert set(record) == {
        "left",
        "right",
        "per_example_sum_squared_difference",
        "per_example_compared_value_count",
        "per_example_rms_difference",
        "per_example_max_abs_difference",
        "sum_squared_difference",
        "rms_difference",
        "max_per_example_rms_difference",
        "max_abs_difference",
        "within_tolerance",
    }
    assert record["left"]["dtype"] == record["right"]["dtype"] == "<f4"
    assert record["left"]["shape"] == record["right"]["shape"] == [512, 4]
    assert len(record["per_example_sum_squared_difference"]) == 512
    assert len(record["per_example_compared_value_count"]) == 512
    assert len(record["per_example_rms_difference"]) == 512
    assert len(record["per_example_max_abs_difference"]) == 512
    assert math.isclose(
        record["sum_squared_difference"],
        float(np.square(right.astype(np.float64)).sum()),
        rel_tol=0.0,
        abs_tol=1e-30,
    )
    assert gate_c._gate_c2_floating_difference_record_complete(
        record,
        expected_dtype="<f4",
        expected_shape=(512, 4),
        rms_tolerance=1e-6,
    )

    mutations = []
    for field in (
        "sum_squared_difference",
        "rms_difference",
        "max_per_example_rms_difference",
        "max_abs_difference",
    ):
        malformed = copy.deepcopy(record)
        malformed[field] += 1.0
        mutations.append(malformed)
    malformed = copy.deepcopy(record)
    malformed["per_example_compared_value_count"][0] = True
    mutations.append(malformed)
    malformed = copy.deepcopy(record)
    malformed["right"]["shape"] = [512, 5]
    mutations.append(malformed)
    malformed = copy.deepcopy(record)
    malformed["right"]["nonfinite_count"] = 1
    mutations.append(malformed)
    malformed = copy.deepcopy(record)
    malformed["within_tolerance"] = False
    mutations.append(malformed)

    assert all(
        not gate_c._gate_c2_floating_difference_record_complete(
            malformed,
            expected_dtype="<f4",
            expected_shape=(512, 4),
            rms_tolerance=1e-6,
        )
        for malformed in mutations
    )


@requires_gate_c
def test_gate_c2_prediction_difference_recomputes_histograms_and_hamming() -> None:
    left = np.arange(512, dtype=np.int32) % 10
    right = left.copy()
    record = gate_c._gate_c2_prediction_difference_record(left, right)

    assert set(record) == {
        "left",
        "right",
        "per_example_hamming_count",
        "hamming_count",
        "equal",
    }
    assert record["left"]["dtype"] == record["right"]["dtype"] == "<i4"
    assert record["left"]["shape"] == record["right"]["shape"] == [512]
    assert sum(record["left"]["histogram"]) == 512
    assert record["hamming_count"] == 0
    assert gate_c._gate_c2_prediction_difference_record_complete(
        record,
        count=512,
    )

    for mutate in ("histogram", "hamming", "digest", "stored_equal"):
        malformed = copy.deepcopy(record)
        if mutate == "histogram":
            malformed["left"]["histogram"][0] += 1
        elif mutate == "hamming":
            malformed["per_example_hamming_count"][0] = 1
        elif mutate == "digest":
            malformed["right"]["sha256"] = "0" * 64
        else:
            malformed["equal"] = False
        assert not gate_c._gate_c2_prediction_difference_record_complete(
            malformed,
            count=512,
        )


def _gate_c2_zero_difference_record(shape: tuple[int, ...]) -> dict[str, Any]:
    values = np.zeros(shape, dtype=np.float32)
    return gate_c._gate_c2_floating_difference_record(
        values,
        values.copy(),
        rms_tolerance=1e-6,
    )


def _gate_c2_cached_boundary_tree_sha256(
    hidden_paths: Mapping[str, Mapping[str, Any]],
) -> str:
    fields = [b"example21-gate-c2-cached-read-boundary-state-v1"]
    for path in sorted(hidden_paths):
        endpoint = hidden_paths[path]
        fields.extend(
            (
                path.encode(),
                b"0",
                str(endpoint["dtype"]).encode("ascii"),
                ",".join(map(str, endpoint["shape"])).encode("ascii"),
                str(endpoint["sha256"]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gate_c2_passing_comparison(
    difference_templates: Mapping[str, Mapping[str, Any]],
    prediction_template: Mapping[str, Any],
    *,
    include_context_memory: bool,
) -> dict[str, Any]:
    hidden_paths = {
        path: copy.deepcopy(difference_templates[path])
        for path in _GATE_C2_HIDDEN_GEOMETRY
        if include_context_memory or path != "context_memory#0"
    }
    return {
        "compact": copy.deepcopy(difference_templates["compact"]),
        "hidden_paths": hidden_paths,
        "predictions": copy.deepcopy(prediction_template),
        "passed": True,
    }


def _gate_c2_passing_cached_read_probe(
    *,
    regime: str,
    start_tick: str,
    canonical_parameter_sha256: str,
    endpoint_templates: Mapping[str, Mapping[str, Any]],
    difference_templates: Mapping[str, Mapping[str, Any]],
    prediction_template: Mapping[str, Any],
    selected_read: Mapping[str, Any],
    selected_drive: Mapping[str, Any],
) -> dict[str, Any]:
    tick_order = gate_c.GATE_C2_LATENT_TICKS[regime]
    start = tick_order.index(start_tick)
    continuation_ticks = tick_order[start:]
    source = copy.deepcopy(endpoint_templates["memory_read#0"])
    sentinels: dict[str, Any] = {}
    for name, replacement in _GATE_C2_CACHED_READ_REPLACEMENTS.items():
        before_hidden = {
            path: copy.deepcopy(endpoint_templates[path])
            for path in _GATE_C2_HIDDEN_GEOMETRY
        }
        after_hidden = copy.deepcopy(before_hidden)
        replacement_array = np.full(
            (512, 32),
            replacement["fill_value"],
            dtype=np.float32,
        )
        replacement_endpoint = gate_c._gate_c2_array_endpoint(
            replacement_array
        )
        assert replacement_endpoint["sha256"] == replacement["sha256"]
        after_hidden["memory_read#0"] = replacement_endpoint
        before = {
            "hidden_paths": before_hidden,
            "hidden_state_tree_sha256": (
                _gate_c2_cached_boundary_tree_sha256(before_hidden)
            ),
            "parameter_tree_sha256": canonical_parameter_sha256,
        }
        after = {
            "hidden_paths": after_hidden,
            "hidden_state_tree_sha256": (
                _gate_c2_cached_boundary_tree_sha256(after_hidden)
            ),
            "parameter_tree_sha256": canonical_parameter_sha256,
        }
        unchanged = sorted(
            set(_GATE_C2_HIDDEN_GEOMETRY) - {"memory_read#0"}
        )
        sentinels[name] = {
            "replacement": copy.deepcopy(replacement),
            "boundary": {
                "before_replacement": before,
                "after_replacement": after,
                "changed_paths": ["memory_read#0"],
                "unchanged_paths": unchanged,
                "parameters_equal": True,
                "only_memory_read_replaced": True,
                "passed": True,
            },
            "selected_read": copy.deepcopy(selected_read),
            "selected_drive": copy.deepcopy(selected_drive),
            "continuation": {
                "ticks": {
                    tick: _gate_c2_passing_comparison(
                        difference_templates,
                        prediction_template,
                        include_context_memory=True,
                    )
                    for tick in continuation_ticks
                },
                "passed": True,
            },
            "passed": True,
        }
    return {
        "source_cached_memory_read": source,
        "plus_11": sentinels["plus_11"],
        "minus_11": sentinels["minus_11"],
        "passed": True,
    }


def _gate_c2_passing_no_read_report(
    admission: Mapping[str, Any],
    *,
    regime: str,
) -> dict[str, Any]:
    difference_templates = {
        "compact": _gate_c2_zero_difference_record((512, 1_180)),
        **{
            path: _gate_c2_zero_difference_record(shape)
            for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
        },
    }
    endpoint_templates = {
        path: gate_c._gate_c2_array_endpoint(
            np.zeros(shape, dtype=np.float32)
        )
        for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
    }
    prediction_template = gate_c._gate_c2_prediction_difference_record(
        np.zeros((512,), dtype=np.int32),
        np.zeros((512,), dtype=np.int32),
    )
    selected_read = gate_c._gate_c2_zero_array_record(
        np.zeros((512, 32), dtype=np.float32)
    )
    selected_drive = gate_c._gate_c2_zero_array_record(
        np.zeros((512, 2_048), dtype=np.float32)
    )
    non_s_k_values = {
        path: np.zeros(shape, dtype=np.float32)
        for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
        if path != "context_memory#0"
    }
    non_s_k_state = gate_c._gate_c2_tree_equality_record(
        non_s_k_values,
        {path: np.array(value, copy=True) for path, value in non_s_k_values.items()},
        framing="nul_joined_gate_c2_non_s_k_state_v1",
        parameter_values=False,
    )
    model = gate_c._new_model_for_arm(
        gate_c.GateCConfig(),
        regime,
        "full",
        batch_size=1,
    )
    parameter_values = legacy._parameter_values(model)
    parameter_copies = jax.tree.map(
        lambda value: np.array(value, copy=True),
        parameter_values,
    )
    parameters = gate_c._gate_c2_tree_equality_record(
        parameter_values,
        parameter_copies,
        framing="authenticated_gate_c_parameter_array_digest_v1",
        parameter_values=True,
    )
    canonical = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]
    parameters["left_value_sha256"] = canonical
    parameters["right_value_sha256"] = canonical
    parameters["values_equal"] = True
    tick_order = gate_c.GATE_C2_LATENT_TICKS[regime]
    streams = {
        stream: {
            tick: {
                "selected_read": copy.deepcopy(selected_read),
                "selected_drive": copy.deepcopy(selected_drive),
                "cached_read_probe": _gate_c2_passing_cached_read_probe(
                    regime=regime,
                    start_tick=tick,
                    canonical_parameter_sha256=canonical,
                    endpoint_templates=endpoint_templates,
                    difference_templates=difference_templates,
                    prediction_template=prediction_template,
                    selected_read=selected_read,
                    selected_drive=selected_drive,
                ),
                "cached_h0_read_reused": False,
            }
            for tick in tick_order
        }
        for stream in ("intact", "shuffled", "no_context")
    }
    source_s_k = legacy._digest_arrays(
        np.zeros((512, 32, 32), dtype=np.float32)
    )
    perturbations: dict[str, Any] = {}
    positive_controls: dict[str, Any] = {}
    nonzero_read = gate_c._gate_c2_floating_difference_record(
        np.zeros((512, 32), dtype=np.float32),
        np.ones((512, 32), dtype=np.float32),
        rms_tolerance=1e-6,
    )
    zero_read_difference = _gate_c2_zero_difference_record((512, 32))
    zero_drive_difference = _gate_c2_zero_difference_record((512, 2_048))
    for replacement_name, replacement in (
        gate_c.GATE_C2_CONTEXT_MEMORY_REPLACEMENTS.items()
    ):
        query_streams: dict[str, Any] = {}
        positive_streams: dict[str, Any] = {}
        for stream_index, stream_name in enumerate(
            ("intact", "shuffled", "no_context")
        ):
            query_ticks: dict[str, Any] = {}
            positive_ticks: dict[str, Any] = {}
            for tick_index, tick in enumerate(tick_order):
                shared_boundary = {
                    "source_s_k_sha256": source_s_k,
                    "replacement_s_k_sha256": replacement["sha256"],
                    "source_replacement_differ": True,
                    "non_s_k_state": copy.deepcopy(non_s_k_state),
                    "parameters": copy.deepcopy(parameters),
                }
                query_ticks[tick] = {
                    **copy.deepcopy(shared_boundary),
                    "selected_read": copy.deepcopy(selected_read),
                    "selected_drive": copy.deepcopy(selected_drive),
                    "continuation": _gate_c2_passing_comparison(
                        difference_templates,
                        prediction_template,
                        include_context_memory=False,
                    ),
                    "passed": True,
                }
                positive_ticks[tick] = {
                    **copy.deepcopy(shared_boundary),
                    "selected_read_difference": copy.deepcopy(
                        nonzero_read
                        if stream_index == 0 and tick_index == 0
                        else zero_read_difference
                    ),
                    "selected_drive_difference": copy.deepcopy(
                        zero_drive_difference
                    ),
                    "continuation": _gate_c2_passing_comparison(
                        difference_templates,
                        prediction_template,
                        include_context_memory=False,
                    ),
                    "passed": True,
                }
            query_streams[stream_name] = query_ticks
            positive_streams[stream_name] = positive_ticks
        perturbations[replacement_name] = {
            "replacement": copy.deepcopy(replacement),
            "streams": query_streams,
            "passed": True,
        }
        positive_controls[replacement_name] = {
            "replacement": copy.deepcopy(replacement),
            "streams": positive_streams,
            "passed": True,
        }
    return {
        "streams": streams,
        "perturbations": perturbations,
        "full_positive_control": {
            **positive_controls,
            "passed": True,
        },
        "removed_path_finite_window_influence": {
            "fixture_regime": regime,
            "complete": True,
        },
        "passed": True,
    }


@pytest.fixture(scope="module")
def passing_gate_c2_no_read_reports() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    admission, _ = _passing_gate_c_initialization_report()
    return admission, {
        regime: _gate_c2_passing_no_read_report(admission, regime=regime)
        for regime in _REGIMES
    }


def _gate_c2_strict_canonical_json_roundtrip(value: Any) -> Any:
    def reject_nonfinite_constant(token: str) -> Any:
        raise ValueError(f"nonfinite JSON constant {token}")

    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json.loads(
        payload,
        parse_constant=reject_nonfinite_constant,
        strict=True,
    )


def _patch_gate_c2_fixture_removed_path_validator(
    monkeypatch: pytest.MonkeyPatch,
    admission: Mapping[str, Any],
) -> None:
    def removed_complete(
        value: Any,
        *,
        regime: str,
        canonical_parameter_sha256: str,
    ) -> bool:
        canonical = admission["initialization"][regime]["canonical_full"][
            "parameter_sha256"
        ]
        return bool(
            value == {"fixture_regime": regime, "complete": True}
            and canonical_parameter_sha256 == canonical
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_influence_complete",
        removed_complete,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_no_read_validator_accepts_complete_independent_raw_evidence(
    regime: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    canonical = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]

    def removed_complete(
        value: Any,
        *,
        regime: str,
        canonical_parameter_sha256: str,
    ) -> bool:
        return bool(
            value == {"fixture_regime": regime, "complete": True}
            and canonical_parameter_sha256 == canonical
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_influence_complete",
        removed_complete,
    )
    report = reports[regime]
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )
    cached_boundaries = [
        sentinel["boundary"]
        for stream in report["streams"].values()
        for tick in stream.values()
        for sentinel in (
            tick["cached_read_probe"]["plus_11"],
            tick["cached_read_probe"]["minus_11"],
        )
    ]
    assert len({id(item) for item in cached_boundaries}) == len(
        cached_boundaries
    )
    assert all(
        boundary["before_replacement"]
        is not boundary["after_replacement"]
        for boundary in cached_boundaries
    )


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_no_read_validator_accepts_canonical_json_mapping_order(
    regime: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    _patch_gate_c2_fixture_removed_path_validator(monkeypatch, admission)
    report = reports[regime]
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )

    decoded_admission = _gate_c2_strict_canonical_json_roundtrip(admission)
    decoded_report = _gate_c2_strict_canonical_json_roundtrip(report)
    assert tuple(decoded_report["streams"]) == (
        "intact",
        "no_context",
        "shuffled",
    )
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        decoded_report,
        decoded_admission,
        regime=regime,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("mutation", ("missing_key", "extra_key"))
@requires_gate_c
def test_gate_c2_no_read_validator_keeps_exact_mapping_schemas(
    regime: str,
    mutation: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    _patch_gate_c2_fixture_removed_path_validator(monkeypatch, admission)
    report = dict(reports[regime])
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )

    if mutation == "missing_key":
        report.pop("passed")
    else:
        report["unexpected"] = None
    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize(
    "sequence_path",
    ("cached_unchanged_paths", "non_s_k_paths"),
)
@requires_gate_c
def test_gate_c2_no_read_validator_keeps_semantic_sequence_order_strict(
    regime: str,
    sequence_path: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    _patch_gate_c2_fixture_removed_path_validator(monkeypatch, admission)
    report = reports[regime]
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )
    first_tick = gate_c.GATE_C2_LATENT_TICKS[regime][0]
    if sequence_path == "cached_unchanged_paths":
        container = report["streams"]["intact"][first_tick][
            "cached_read_probe"
        ]["plus_11"]["boundary"]
        field = "unchanged_paths"
    else:
        container = report["perturbations"]["plus_7"]["streams"][
            "intact"
        ][first_tick]["non_s_k_state"]
        field = "paths"
    original = container[field]
    reversed_order = list(reversed(original))
    assert reversed_order != original
    container[field] = reversed_order
    try:
        assert not gate_c._gate_c2_query_only_latent_no_read_complete(
            report,
            admission,
            regime=regime,
        )
    finally:
        container[field] = original


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize(
    "mutation",
    (
        "missing_stream",
        "missing_tick",
        "cached_replacement_digest",
        "cached_source_equals_replacement",
        "cached_boundary_alias",
        "cached_changed_paths",
        "cached_nonmemory_leaf",
        "cached_parameter_digest",
        "cached_selected_read",
        "cached_missing_suffix_tick",
        "cached_continuation_aggregate",
        "cached_prediction_hamming",
        "cached_colluding_pass",
        "cached_reused_boolean",
        "s_k_replacement_digest",
        "s_k_source_equals_replacement",
        "s_k_nonstate_collusion",
        "s_k_parameter_collusion",
        "s_k_selected_drive",
        "s_k_continuation_path",
        "full_positive_zero",
        "full_positive_source_collision",
        "full_positive_parameter_collusion",
        "removed_gradient",
        "stored_pass",
    ),
)
@requires_gate_c
def test_gate_c2_no_read_validator_rejects_each_nested_raw_mutation(
    regime: str,
    mutation: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    canonical = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]

    def removed_complete(
        value: Any,
        *,
        regime: str,
        canonical_parameter_sha256: str,
    ) -> bool:
        return bool(
            value == {"fixture_regime": regime, "complete": True}
            and canonical_parameter_sha256 == canonical
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_influence_complete",
        removed_complete,
    )
    report = reports[regime]
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )
    first_tick = gate_c.GATE_C2_LATENT_TICKS[regime][0]
    tick = report["streams"]["intact"][first_tick]
    cached = tick["cached_read_probe"]
    sentinel = cached["plus_11"]
    boundary = sentinel["boundary"]
    perturbation = report["perturbations"]["plus_7"]
    perturbation_tick = perturbation["streams"]["intact"][first_tick]
    positive = report["full_positive_control"]["plus_7"]
    positive_tick = positive["streams"]["intact"][first_tick]
    container: dict[str, Any]
    key: str

    if mutation == "missing_stream":
        container, key = report, "streams"
        replacement = {
            name: value
            for name, value in report["streams"].items()
            if name != "no_context"
        }
    elif mutation == "missing_tick":
        container, key = report["streams"], "intact"
        replacement = {
            name: value
            for name, value in report["streams"]["intact"].items()
            if name != first_tick
        }
    elif mutation == "cached_replacement_digest":
        container, key = sentinel, "replacement"
        replacement = {
            **sentinel["replacement"],
            "sha256": "0" * 64,
        }
    elif mutation == "cached_source_equals_replacement":
        container, key = cached, "source_cached_memory_read"
        replacement = copy.deepcopy(
            boundary["after_replacement"]["hidden_paths"]["memory_read#0"]
        )
    elif mutation == "cached_boundary_alias":
        container, key = boundary, "after_replacement"
        replacement = boundary["before_replacement"]
    elif mutation == "cached_changed_paths":
        container, key = boundary, "changed_paths"
        replacement = ["context_memory#0"]
    elif mutation == "cached_nonmemory_leaf":
        container, key = boundary, "after_replacement"
        replacement = copy.deepcopy(boundary["after_replacement"])
        replacement["hidden_paths"]["query_encoding#0"]["sha256"] = "0" * 64
    elif mutation == "cached_parameter_digest":
        container, key = boundary, "after_replacement"
        replacement = copy.deepcopy(boundary["after_replacement"])
        replacement["parameter_tree_sha256"] = "0" * 64
    elif mutation == "cached_selected_read":
        container, key = sentinel, "selected_read"
        replacement = gate_c._gate_c2_zero_array_record(
            np.ones((512, 32), dtype=np.float32)
        )
    elif mutation == "cached_missing_suffix_tick":
        container, key = sentinel["continuation"], "ticks"
        replacement = dict(sentinel["continuation"]["ticks"])
        replacement.pop(next(reversed(replacement)))
    elif mutation == "cached_continuation_aggregate":
        container, key = sentinel["continuation"], "ticks"
        replacement = copy.deepcopy(sentinel["continuation"]["ticks"])
        replacement[first_tick]["compact"]["rms_difference"] = 1.0
    elif mutation == "cached_prediction_hamming":
        container, key = sentinel["continuation"], "ticks"
        replacement = copy.deepcopy(sentinel["continuation"]["ticks"])
        replacement[first_tick]["predictions"][
            "per_example_hamming_count"
        ][0] = 1
    elif mutation == "cached_colluding_pass":
        container, key = sentinel, "passed"
        replacement = False
    elif mutation == "cached_reused_boolean":
        container, key = tick, "cached_h0_read_reused"
        replacement = True
    elif mutation == "s_k_replacement_digest":
        container, key = perturbation, "replacement"
        replacement = {
            **perturbation["replacement"],
            "sha256": "0" * 64,
        }
    elif mutation == "s_k_source_equals_replacement":
        container, key = perturbation_tick, "source_s_k_sha256"
        replacement = perturbation_tick["replacement_s_k_sha256"]
    elif mutation == "s_k_nonstate_collusion":
        container, key = perturbation_tick, "non_s_k_state"
        replacement = copy.deepcopy(perturbation_tick["non_s_k_state"])
        replacement["right_value_sha256"] = "0" * 64
    elif mutation == "s_k_parameter_collusion":
        container, key = perturbation_tick, "parameters"
        replacement = copy.deepcopy(perturbation_tick["parameters"])
        replacement["right_value_sha256"] = "0" * 64
    elif mutation == "s_k_selected_drive":
        container, key = perturbation_tick, "selected_drive"
        replacement = gate_c._gate_c2_zero_array_record(
            np.ones((512, 2_048), dtype=np.float32)
        )
    elif mutation == "s_k_continuation_path":
        container, key = perturbation_tick, "continuation"
        replacement = copy.deepcopy(perturbation_tick["continuation"])
        replacement["hidden_paths"].pop("memory_read#0")
    elif mutation == "full_positive_zero":
        container, key = positive_tick, "selected_read_difference"
        replacement = _gate_c2_zero_difference_record((512, 32))
    elif mutation == "full_positive_source_collision":
        container, key = positive_tick, "source_s_k_sha256"
        replacement = positive_tick["replacement_s_k_sha256"]
    elif mutation == "full_positive_parameter_collusion":
        container, key = positive_tick, "parameters"
        replacement = copy.deepcopy(positive_tick["parameters"])
        replacement["right_value_sha256"] = "0" * 64
    elif mutation == "removed_gradient":
        container, key = report, "removed_path_finite_window_influence"
        replacement = {"fixture_regime": "wrong", "complete": True}
    elif mutation == "stored_pass":
        container, key = report, "passed"
        replacement = False
    else:
        raise AssertionError(f"unhandled no-read mutation {mutation}")

    original = container[key]
    container[key] = replacement
    try:
        assert not gate_c._gate_c2_query_only_latent_no_read_complete(
            report,
            admission,
            regime=regime,
        )
    finally:
        container[key] = original


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_no_read_validator_rejects_colluding_noncanonical_parameter_hashes(
    regime: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    report = copy.deepcopy(reports[regime])
    canonical = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]

    def removed_complete(
        value: Any,
        *,
        regime: str,
        canonical_parameter_sha256: str,
    ) -> bool:
        return bool(
            value == {"fixture_regime": regime, "complete": True}
            and canonical_parameter_sha256 == canonical
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_influence_complete",
        removed_complete,
    )
    assert gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )

    colluding = "e" * 64
    assert colluding != canonical
    for stream in report["streams"].values():
        for tick in stream.values():
            for sentinel_name in gate_c.GATE_C2_CACHED_READ_REPLACEMENTS:
                boundary = tick["cached_read_probe"][sentinel_name]["boundary"]
                boundary["before_replacement"][
                    "parameter_tree_sha256"
                ] = colluding
                boundary["after_replacement"][
                    "parameter_tree_sha256"
                ] = colluding
    for family in ("perturbations", "full_positive_control"):
        for sentinel_name in gate_c.GATE_C2_CONTEXT_MEMORY_REPLACEMENTS:
            for stream in report[family][sentinel_name]["streams"].values():
                for tick in stream.values():
                    tick["parameters"]["left_value_sha256"] = colluding
                    tick["parameters"]["right_value_sha256"] = colluding

    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        report,
        admission,
        regime=regime,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize(
    "mutation",
    ("missing_initialization", "missing_regime", "missing_canonical", "boolean_sha"),
)
@requires_gate_c
def test_gate_c2_no_read_validator_fails_closed_on_malformed_admission(
    regime: str,
    mutation: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    malformed = copy.deepcopy(admission)
    canonical = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]

    def removed_complete(
        value: Any,
        *,
        regime: str,
        canonical_parameter_sha256: str,
    ) -> bool:
        return bool(
            value == {"fixture_regime": regime, "complete": True}
            and canonical_parameter_sha256 == canonical
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_influence_complete",
        removed_complete,
    )
    if mutation == "missing_initialization":
        malformed.pop("initialization")
    elif mutation == "missing_regime":
        malformed["initialization"].pop(regime)
    elif mutation == "missing_canonical":
        malformed["initialization"][regime].pop("canonical_full")
    elif mutation == "boolean_sha":
        malformed["initialization"][regime]["canonical_full"][
            "parameter_sha256"
        ] = True
    else:
        raise AssertionError(f"unhandled admission mutation {mutation}")

    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        reports[regime],
        malformed,
        regime=regime,
    )


def _reduced_gate_c2_gate_a_no_read_inputs() -> tuple[Any, Any, dict[str, Any]]:
    gate_a_config = dataclasses.replace(
        gate_a.BindingGateConfig.smoke_config(),
        batch_size=10,
        validation_episodes=2,
    )
    config = gate_c.GateCConfig(
        gate_a_config=gate_a_config,
        gate_b_config=_reduced_gate_c_config().gate_b_config,
    )
    data = gate_c._regenerate_gate_a_data(config)
    model = gate_c._new_model_for_arm(
        config,
        "gate_a",
        "full",
        batch_size=gate_a_config.validation_episodes,
    )
    initialization = {
        "initialization": {
            "gate_a": {
                "canonical_full": {
                    "parameter_sha256": legacy._array_digest(
                        legacy._parameter_values(model)
                    )
                }
            }
        }
    }
    return config, data, initialization


def _reduced_gate_c2_gate_b_no_read_inputs() -> tuple[Any, Any, dict[str, Any]]:
    config = _reduced_gate_c_config()
    data = gate_c._regenerate_gate_b_data(config)
    model = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "full",
        batch_size=config.gate_b_config.validation_episodes,
    )
    initialization = {
        "initialization": {
            "gate_b": {
                "canonical_full": {
                    "parameter_sha256": legacy._array_digest(
                        legacy._parameter_values(model)
                    )
                }
            }
        }
    }
    return config, data, initialization


@requires_gate_c
def test_reduced_no_read_report_derives_cached_read_sentinels_from_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data, initialization = _reduced_gate_c2_gate_a_no_read_inputs()
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_finite_window_influence",
        lambda *args, **kwargs: {"complete": True},
    )
    report = gate_c._query_only_latent_no_read_report(
        config,
        initialization=initialization,
        regime="gate_a",
        data=data,
    )

    for stream in ("intact", "shuffled", "no_context"):
        tick = report["streams"][stream]["H1"]
        assert set(tick) == {
            "selected_read",
            "selected_drive",
            "cached_read_probe",
            "cached_h0_read_reused",
        }
        assert set(tick["cached_read_probe"]) == {
            "source_cached_memory_read",
            "plus_11",
            "minus_11",
            "passed",
        }
        assert tick["cached_read_probe"]["passed"] is True
        assert tick["cached_h0_read_reused"] is False


@requires_gate_c
def test_reduced_no_read_report_uses_distinct_pre_post_boundary_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data, initialization = _reduced_gate_c2_gate_a_no_read_inputs()
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_finite_window_influence",
        lambda *args, **kwargs: {"complete": True},
    )
    original = gate_c._gate_c2_tree_equality_record
    calls: list[tuple[Any, Any, str]] = []

    def distinct_record(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        *,
        framing: str,
        parameter_values: bool,
    ) -> dict[str, Any]:
        calls.append((left, right, framing))
        assert left is not right
        assert set(left) == set(right)
        for path in left:
            for left_leaf, right_leaf in zip(
                jax.tree.leaves(left[path]),
                jax.tree.leaves(right[path]),
                strict=True,
            ):
                assert left_leaf is not right_leaf
                assert not np.shares_memory(
                    np.asarray(left_leaf),
                    np.asarray(right_leaf),
                )
        return original(
            left,
            right,
            framing=framing,
            parameter_values=parameter_values,
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_tree_equality_record",
        distinct_record,
    )
    gate_c._query_only_latent_no_read_report(
        config,
        initialization=initialization,
        regime="gate_a",
        data=data,
    )
    assert calls
    assert {
        framing for _, _, framing in calls
    } == {
        "nul_joined_gate_c2_non_s_k_state_v1",
        "authenticated_gate_c_parameter_array_digest_v1",
    }


@requires_gate_c
def test_gate_b_no_read_report_has_exact_bounded_driver_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data, initialization = _reduced_gate_c2_gate_b_no_read_inputs()
    latent_lengths: dict[str, list[int]] = {"query_only": [], "full": []}
    prefix_lengths: dict[str, list[int]] = {"query_only": [], "full": []}
    boundary_calls = 0
    real_latent_driver = gate_c._gate_c2_latent_driver
    real_prefix_driver = gate_c._gate_c2_h0_driver
    real_boundary_snapshot = gate_c._gate_c2_boundary_snapshot

    def recording_latent_driver(model: LatentWorkspaceModel) -> Any:
        driver, paths = real_latent_driver(model)
        policy = model.memory_read_policy

        def recorded(events: Any, advances: Any) -> Any:
            latent_lengths[policy].append(int(events.shape[0]))
            return driver(events, advances)

        return recorded, paths

    def recording_prefix_driver(model: LatentWorkspaceModel) -> Any:
        driver = real_prefix_driver(model)
        policy = model.memory_read_policy

        def recorded(events: Any, advances: Any) -> Any:
            prefix_lengths[policy].append(int(events.shape[0]))
            return driver(events, advances)

        return recorded

    def recording_boundary_snapshot(*args: Any, **kwargs: Any) -> Any:
        nonlocal boundary_calls
        boundary_calls += 1
        return real_boundary_snapshot(*args, **kwargs)

    monkeypatch.setattr(gate_c, "_gate_c2_latent_driver", recording_latent_driver)
    monkeypatch.setattr(gate_c, "_gate_c2_h0_driver", recording_prefix_driver)
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_boundary_snapshot",
        recording_boundary_snapshot,
    )
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_finite_window_influence",
        lambda *args, **kwargs: {"complete": True},
    )

    gate_c._query_only_latent_no_read_report(
        config,
        initialization=initialization,
        regime="gate_b",
        data=data,
    )

    h0_length = config.gate_b_config.sequence_length - config.gate_b_config.gap_steps
    workload = {
        "query": Counter(latent_lengths["query_only"]),
        "full": Counter(latent_lengths["full"]),
        "prefix_query": Counter(prefix_lengths["query_only"]),
        "prefix_full": Counter(prefix_lengths["full"]),
        "query_call_count": len(latent_lengths["query_only"]),
        "full_call_count": len(latent_lengths["full"]),
        "prefix_call_count": sum(map(len, prefix_lengths.values())),
        "total_call_count": sum(map(len, latent_lengths.values()))
        + sum(map(len, prefix_lengths.values())),
        "boundary_rerun_count": boundary_calls,
    }
    assert workload == {
        "query": Counter({1: 54, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6, 8: 9}),
        "full": Counter({1: 48, 8: 3}),
        "prefix_query": Counter({h0_length: 3}),
        "prefix_full": Counter({h0_length: 3}),
        "query_call_count": 99,
        "full_call_count": 51,
        "prefix_call_count": 6,
        "total_call_count": 156,
        "boundary_rerun_count": 0,
    }


@requires_gate_c
def test_gate_c2_host_boundaries_are_exact_independent_and_replayable() -> None:
    host_boundary_snapshots = gate_c._gate_c2_host_boundary_snapshots
    config, data, _ = _reduced_gate_c2_gate_b_no_read_inputs()
    validation = data[1]
    count = config.gate_b_config.validation_episodes
    h0_end = config.gate_b_config.sequence_length - config.gate_b_config.gap_steps
    events = np.asarray(validation.intact)[h0_end:]
    advances = np.asarray(validation.advance_masks)[h0_end:]
    model = gate_c._new_model_for_arm(
        config,
        "gate_b",
        "query_only",
        batch_size=count,
    )
    h0_snapshot = gate_c._gate_c2_stop_gradient_snapshot(model.snapshot_state())
    driver, hidden_paths = gate_c._gate_c2_latent_driver(model)
    baseline = gate_c._gate_c2_latent_capture(
        model,
        driver,
        hidden_paths,
        h0_snapshot,
        events,
        advances,
    )
    source_hidden = {
        path: np.array(value, copy=True, order="C")
        for path, value in baseline["hidden_paths"].items()
    }

    boundaries = host_boundary_snapshots(
        h0_snapshot,
        baseline["hidden_paths"],
    )

    assert isinstance(boundaries, tuple)
    assert len(boundaries) == 8
    assert all(type(boundary) is type(h0_snapshot) for boundary in boundaries)
    h0_entry_paths = tuple(path for path, _ in h0_snapshot.entries)
    assert all(
        tuple(path for path, _ in boundary.entries) == h0_entry_paths
        for boundary in boundaries
    )

    def direct_arrays(snapshot: Any) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for path, value in snapshot.entries:
            name = gate_a._path(path)
            for index, leaf in enumerate(jax.tree.leaves(value)):
                mantissa = u.get_mantissa(leaf)
                assert isinstance(mantissa, np.ndarray)
                assert mantissa.flags.c_contiguous
                result[f"{name}#{index}"] = mantissa
        return result

    def value_structures(snapshot: Any) -> dict[str, Any]:
        return {
            gate_a._path(path): jax.tree.structure(value)
            for path, value in snapshot.entries
        }

    def value_units(snapshot: Any) -> dict[str, Any]:
        return {
            gate_a._path(path): u.get_unit(value)
            for path, value in snapshot.entries
        }

    def framed_digest(values: Mapping[str, np.ndarray]) -> str:
        endpoints = {
            path: gate_c._gate_c2_array_endpoint(value)
            for path, value in values.items()
        }
        return gate_c._gate_c2_cached_boundary_tree_sha256(endpoints)

    h0_arrays = gate_c._gate_c2_snapshot_arrays(h0_snapshot)
    actual_arrays = [direct_arrays(boundary) for boundary in boundaries]
    expected_arrays = [
        h0_arrays,
        *(
            {
                path: source_hidden[path][tick_index]
                for path in sorted(source_hidden)
            }
            for tick_index in range(7)
        ),
    ]
    expected_structures = value_structures(h0_snapshot)
    expected_units = value_units(h0_snapshot)
    for boundary, actual, expected in zip(
        boundaries,
        actual_arrays,
        expected_arrays,
        strict=True,
    ):
        assert value_structures(boundary) == expected_structures
        assert value_units(boundary) == expected_units
        assert tuple(actual) == tuple(expected)
        assert legacy._array_digest(actual) == legacy._array_digest(expected)
        assert framed_digest(actual) == framed_digest(expected)
        for path in actual:
            assert actual[path].dtype == expected[path].dtype
            assert actual[path].shape == expected[path].shape
            assert actual[path].tobytes() == np.ascontiguousarray(
                expected[path]
            ).tobytes()
            assert not np.shares_memory(actual[path], expected[path])

    for left_index, left in enumerate(actual_arrays):
        for right in actual_arrays[left_index + 1 :]:
            for path in left:
                assert not np.shares_memory(left[path], right[path])

    missing_path = dict(baseline["hidden_paths"])
    missing_path.pop(next(iter(missing_path)))
    with pytest.raises(ValueError, match="path"):
        host_boundary_snapshots(h0_snapshot, missing_path)
    wrong_length = dict(baseline["hidden_paths"])
    length_path = next(iter(wrong_length))
    wrong_length[length_path] = wrong_length[length_path][:-1]
    with pytest.raises(ValueError, match="leading|length|tick"):
        host_boundary_snapshots(h0_snapshot, wrong_length)
    wrong_shape = dict(baseline["hidden_paths"])
    shape_path = next(iter(wrong_shape))
    wrong_shape[shape_path] = wrong_shape[shape_path][:, :-1]
    with pytest.raises(ValueError, match="shape|geometry"):
        host_boundary_snapshots(h0_snapshot, wrong_shape)

    def slice_capture(capture: Mapping[str, Any], start: int) -> dict[str, Any]:
        return {
            "compact": capture["compact"][start:],
            "selected_read": capture["selected_read"][start:],
            "selected_drive": capture["selected_drive"][start:],
            "hidden_paths": {
                path: value[start:]
                for path, value in capture["hidden_paths"].items()
            },
            "predictions": capture["predictions"][start:],
        }

    replays = (
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[0],
            events,
            advances,
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[1],
            events[1:],
            advances[1:],
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[2],
            events[2:],
            advances[2:],
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[3],
            events[3:],
            advances[3:],
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[4],
            events[4:],
            advances[4:],
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[5],
            events[5:],
            advances[5:],
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[6],
            events[6:],
            advances[6:],
        ),
        gate_c._gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            boundaries[7],
            events[7:],
            advances[7:],
        ),
    )
    tick_names = gate_c.GATE_C2_LATENT_TICKS["gate_b"]
    expected_replays = tuple(
        slice_capture(baseline, boundary_index)
        for boundary_index in range(len(boundaries))
    )
    for boundary_index, (expected, replay) in enumerate(
        zip(expected_replays, replays, strict=True)
    ):
        comparison = gate_c._gate_c2_suffix_comparison(
            expected,
            replay,
            tick_names=tick_names[boundary_index:],
            include_context_memory=True,
        )
        assert comparison["passed"] is True
        np.testing.assert_array_equal(
            replay["predictions"],
            expected["predictions"],
        )
        for tick_index in range(expected["selected_read"].shape[0]):
            for path in ("selected_read", "selected_drive"):
                difference = gate_c._gate_c2_floating_difference_record(
                    expected[path][tick_index],
                    replay[path][tick_index],
                    rms_tolerance=1e-6,
                )
                assert difference["max_per_example_rms_difference"] <= 1e-6
                assert difference["within_tolerance"] is True

    boundary_digests = [legacy._array_digest(value) for value in actual_arrays]
    source_digest = legacy._array_digest(source_hidden)
    mutation = actual_arrays[2]["context_memory#0"]
    mutation.flat[0] = np.nextafter(mutation.flat[0], np.float32(np.inf))
    mutated_digests = [
        legacy._array_digest(direct_arrays(boundary)) for boundary in boundaries
    ]
    assert mutated_digests[2] != boundary_digests[2]
    assert mutated_digests[:2] == boundary_digests[:2]
    assert mutated_digests[3:] == boundary_digests[3:]
    assert legacy._array_digest(baseline["hidden_paths"]) == source_digest
    for path in source_hidden:
        np.testing.assert_array_equal(
            baseline["hidden_paths"][path],
            source_hidden[path],
        )


def _passing_gate_c2_operational_h0(
    admission: Mapping[str, Any],
    *,
    regime: str,
) -> dict[str, Any]:
    compact = _gate_c2_zero_difference_record((512, 1_180))
    hidden = {
        path: _gate_c2_zero_difference_record(shape)
        for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
    }
    predictions = gate_c._gate_c2_prediction_difference_record(
        np.zeros((512,), dtype=np.int32),
        np.zeros((512,), dtype=np.int32),
    )
    comparison = {
        "compact": compact,
        "hidden_paths": hidden,
        "predictions": predictions,
        "passed": True,
    }
    state_sha256 = hashlib.sha256(f"{regime}:h0-state".encode()).hexdigest()
    stream = {
        "initial_state_sha256": {
            role: state_sha256
            for role in (
                "full_reference",
                "full_replay",
                "copied_full",
                "query_only",
            )
        },
        "comparisons": {
            name: copy.deepcopy(comparison)
            for name in (
                "same_full_replay",
                "copied_full_separate_jit",
                "copied_full_vs_query",
            )
        },
        "passed": True,
    }
    parameter_sha256 = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]
    return {
        "backend": "canonical_production_sparse",
        "checkpoint": 0,
        "intervention_boundary": (
            "after_ordinary_query_h0_before_first_latent_tick_h1"
        ),
        "rms_tolerance": 1e-6,
        "initialization_parameter_sha256": {
            role: parameter_sha256
            for role in (
                "full_reference",
                "full_replay",
                "copied_full",
                "query_only",
            )
        },
        "streams": {
            name: copy.deepcopy(stream)
            for name in ("intact", "shuffled", "no_context")
        },
        "passed": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "backend",
        "checkpoint_type",
        "parameter_digest",
        "initial_state_digest",
        "stream_schema",
        "comparison_schema",
        "missing_hidden_path",
        "stale_aggregate",
        "prediction_hamming",
        "stored_pass",
    ),
)
@requires_gate_c
def test_gate_c2_operational_h0_validator_recomputes_all_three_controls(
    mutation: str,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    report = _passing_gate_c2_operational_h0(admission, regime="gate_a")
    assert gate_c._gate_c2_paired_h0_operational_equivalence_complete(
        report,
        admission,
        regime="gate_a",
    )

    if mutation == "backend":
        report["backend"] = "jax_raw"
    elif mutation == "checkpoint_type":
        report["checkpoint"] = False
    elif mutation == "parameter_digest":
        report["initialization_parameter_sha256"]["query_only"] = "0" * 64
    elif mutation == "initial_state_digest":
        report["streams"]["intact"]["initial_state_sha256"][
            "full_replay"
        ] = "0" * 64
    elif mutation == "stream_schema":
        report["streams"].pop("no_context")
    elif mutation == "comparison_schema":
        report["streams"]["shuffled"]["comparisons"].pop(
            "same_full_replay"
        )
    elif mutation == "missing_hidden_path":
        report["streams"]["intact"]["comparisons"][
            "copied_full_vs_query"
        ]["hidden_paths"].pop("memory_read#0")
    elif mutation == "stale_aggregate":
        report["streams"]["intact"]["comparisons"][
            "same_full_replay"
        ]["compact"]["rms_difference"] = 1.0
    elif mutation == "prediction_hamming":
        report["streams"]["no_context"]["comparisons"][
            "copied_full_separate_jit"
        ]["predictions"]["per_example_hamming_count"][0] = 1
    elif mutation == "stored_pass":
        report["passed"] = False
    else:
        raise AssertionError(f"unhandled H0 mutation {mutation}")

    assert not gate_c._gate_c2_paired_h0_operational_equivalence_complete(
        report,
        admission,
        regime="gate_a",
    )


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_operational_h0_records_are_independent_and_enforce_per_example_bound(
    regime: str,
) -> None:
    admission, _ = _passing_gate_c_initialization_report()
    report = _passing_gate_c2_operational_h0(admission, regime=regime)
    comparisons = [
        comparison
        for stream in report["streams"].values()
        for comparison in stream["comparisons"].values()
    ]
    assert len({id(comparison) for comparison in comparisons}) == 9
    assert len({id(stream) for stream in report["streams"].values()}) == 3
    assert gate_c._gate_c2_paired_h0_operational_equivalence_complete(
        report,
        admission,
        regime=regime,
    )

    left = np.zeros((512, 1_180), dtype=np.float32)
    at_boundary = left.copy()
    at_boundary[0] = np.float32(1e-6)
    boundary_record = gate_c._gate_c2_floating_difference_record(
        left,
        at_boundary,
        rms_tolerance=1e-6,
    )
    assert boundary_record["max_per_example_rms_difference"] <= 1e-6
    report["streams"]["intact"]["comparisons"]["same_full_replay"][
        "compact"
    ] = boundary_record
    assert gate_c._gate_c2_paired_h0_operational_equivalence_complete(
        report,
        admission,
        regime=regime,
    )

    concentrated = left.copy()
    concentrated[0] = np.nextafter(
        np.float32(1e-6),
        np.float32(np.inf),
    )
    concentrated_record = gate_c._gate_c2_floating_difference_record(
        left,
        concentrated,
        rms_tolerance=1e-6,
    )
    assert concentrated_record["rms_difference"] < 1e-6
    assert concentrated_record["max_per_example_rms_difference"] > 1e-6
    report["streams"]["intact"]["comparisons"]["same_full_replay"][
        "compact"
    ] = concentrated_record
    report["streams"]["intact"]["comparisons"]["same_full_replay"][
        "passed"
    ] = True
    report["streams"]["intact"]["passed"] = True
    report["passed"] = True
    assert report["streams"]["shuffled"]["comparisons"][
        "same_full_replay"
    ]["compact"]["max_abs_difference"] == 0.0
    assert not gate_c._gate_c2_paired_h0_operational_equivalence_complete(
        report,
        admission,
        regime=regime,
    )


@pytest.mark.parametrize(
    "shape",
    ((512, 32), (512, 2_048)),
)
@requires_gate_c
def test_gate_c2_no_read_zero_array_records_reconstruct_raw_bytes(
    shape: tuple[int, int],
) -> None:
    values = np.zeros(shape, dtype=np.float32)
    report = gate_c._gate_c2_zero_array_record(values)

    assert set(report) == {
        "dtype",
        "shape",
        "sha256",
        "value_count",
        "finite_count",
        "nonfinite_count",
        "zero_count",
        "sum_of_squares",
        "max_abs",
        "exact_zero",
    }
    assert report == {
        "dtype": "<f4",
        "shape": list(shape),
        "sha256": legacy._digest_arrays(values),
        "value_count": int(np.prod(shape)),
        "finite_count": int(np.prod(shape)),
        "nonfinite_count": 0,
        "zero_count": int(np.prod(shape)),
        "sum_of_squares": 0.0,
        "max_abs": 0.0,
        "exact_zero": True,
    }
    assert gate_c._gate_c2_zero_array_record_complete(
        report,
        expected_dtype="<f4",
        expected_shape=shape,
    )

    for mutation in ("digest", "count", "norm", "stored_zero", "schema"):
        malformed = copy.deepcopy(report)
        if mutation == "digest":
            malformed["sha256"] = "0" * 64
        elif mutation == "count":
            malformed["zero_count"] -= 1
        elif mutation == "norm":
            malformed["sum_of_squares"] = 1.0
        elif mutation == "stored_zero":
            malformed["exact_zero"] = False
        else:
            malformed["unexpected"] = None
        assert not gate_c._gate_c2_zero_array_record_complete(
            malformed,
            expected_dtype="<f4",
            expected_shape=shape,
        )


@requires_gate_c
def test_gate_c2_removed_path_objectives_are_finite_window_and_pinned() -> None:
    assert gate_c.GATE_C2_REMOVED_PATH_GRADIENT_CHUNK_SIZE == 1
    assert gate_c.GATE_C2_REMOVED_PATH_START_STATE == (
        "materialized_h0_stop_gradient"
    )
    assert gate_c.GATE_C2_REMOVED_PATHS == (
        "memory_read_projection/weight",
        "workspace_query_projection/weight",
    )
    assert gate_c.GATE_C2_LIVE_PATHS == (
        "color_factor_head/weight",
        "readout_projection/weight",
        "rec_syn/comm/weight",
    )
    assert gate_c.GATE_C2_LATENT_TICKS == {
        "gate_a": ("H1",),
        "gate_b": ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"),
    }
    assert gate_c.GATE_C2_CONTEXT_MEMORY_REPLACEMENTS == {
        "plus_7": {
            "fill_value": 7.0,
            "dtype": "<f4",
            "shape": [512, 32, 32],
            "sha256": "b7b1338c1b2b0124633638a1823ec4e7a4ba8be321eb7306153c0ca8db8c696e",
        },
        "minus_7": {
            "fill_value": -7.0,
            "dtype": "<f4",
            "shape": [512, 32, 32],
            "sha256": "815cda0e5c57f2387a6c645d372de7ed2df8e9b9be232aeaef6534da35194572",
        },
    }
    assert gate_c.GATE_C2_REMOVED_PATH_OBJECTIVES == {
        "gate_a_h1": {
            "regime": "gate_a",
            "stream": "intact",
            "validation_episode_index": 0,
            "batch_size": 1,
            "checkpoint": "H1",
            "source_metadata": {
                "mapping_id": 850050,
                "input_colors": [2, 5, 7, 8],
                "output_colors": [6, 5, 3, 8],
                "presentation_order_indices": [0, 2, 1, 3],
                "query_index": 3,
                "query_color": 8,
                "target": 8,
                "demonstration_indices": [0, 1, 2, 3],
                "h0_index": 4,
                "h1_index": 5,
            },
            "schedule_sha256": {
                "validation_schedule_sha256": (
                    "80057e092a130e2c78e8f8397b3978bc13a0ff2a5b64bb5207abe238e08feddd"
                ),
                "validation_mapping_ids_sha256": (
                    "a75b3b2ab05110e21fef1ea44ae3fb701d557f45351e5a17cf89a80e80f689f3"
                ),
            },
            "source_arrays": {
                "events": (
                    "<f4",
                    [6, 41],
                    "213fa1ede3635169cba47db69ad36cfab86e759e6f7e35e02e2d07687f71d36b",
                ),
                "advances": (
                    "|b1",
                    [6],
                    "42817343a401805d2af9b07c45738f71274aab865d20efb8fb1980e1ed7dc450",
                ),
                "targets": (
                    "<i4",
                    [1],
                    "88c7413927e162658f4518fd4a62598fe8b0ea6e2ba5fa334940fdfc49ac845a",
                ),
                "canonical_loss_weights": (
                    "<f4",
                    [6],
                    "a13746d7d9b7bc9b071cfccfc55a0e8c54ad8454f4bcbf808f5235334c5a6c45",
                ),
                "h0_prefix": (
                    "<f4",
                    [5, 41],
                    "401cb20096483b305ef7e4383f03377b354fe167aaeffbd06df03f8251b021b2",
                ),
            },
            "continuation": {
                "source_indices": [5],
                "source_events_sha256": (
                    "e9c01c22b9b1bfa0f9bc74cde1820fab5ad99037f581e27df1625db565d6c239"
                ),
                "batched_events_sha256": (
                    "7e0242875d49aef8e5b0c716cd7993f29895e26cf1dcc67cb2f198c1c351f5df"
                ),
                "selection_mask_values": [True],
                "base_checkpoint_weights": [0.5],
                "effective_loss_weights": [0.5],
                "packed_inputs_sha256": (
                    "aee2f5f2f2a672f091c4d02e24ade55262e8ebe50ddd32317c4ead5f8e5b84c5"
                ),
            },
        },
        "gate_b_index0_r8_h8": {
            "regime": "gate_b",
            "stream": "intact",
            "validation_episode_index": 0,
            "batch_size": 1,
            "effort": 8,
            "checkpoint": "H8",
            "source_metadata": {
                "mapping_id": 232423,
                "mapping": [6, 7, 5, 2, 0, 4, 8, 9, 1, 3],
                "query_color": 4,
                "presentation_order": [6, 2, 5, 3, 8, 7, 4, 9, 0, 1],
                "shuffled_shift": 1,
                "h0_through_h8_targets": [0, 6, 8, 1, 7, 9, 3, 2, 5],
            },
            "schedule_sha256": dict(_GATE_B_VALIDATION_SHA256),
            "source_arrays": {
                "events": (
                    "<f4",
                    [19, 47],
                    "36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec",
                ),
                "advances": (
                    "|b1",
                    [19],
                    "c45890e2f9f99fa66ffa09db8f685dc4d138c5f0e1ca0346a044f2dbbf1290a9",
                ),
                "targets": (
                    "<i4",
                    [19],
                    "c4af41cac4f5eb682df15e7d6cf92b0c134b943fae1abfe99b0bfc4c2ddb27e0",
                ),
                "canonical_loss_weights": (
                    "<f8",
                    [19],
                    "205496cf3f437986dc5b65bec81d423848179306a7ce9e1a391dcc22c7340197",
                ),
                "h0_prefix": (
                    "<f4",
                    [11, 47],
                    "a445ffd2a62e56808e15b6205cb7825fef7ed9a63a78b74d3942d89d7b6409a8",
                ),
            },
            "continuation": {
                "source_indices": [11, 12, 13, 14, 15, 16, 17, 18],
                "source_events_sha256": (
                    "87460e7b0e6ea0b632c89c84afa56b7c85c759bc8828b02e275fe4ac3a6be57a"
                ),
                "batched_events_sha256": (
                    "3d2da82783d3194730d1a4671d06df1254ef298fbe2260ee5e7c86474e111a32"
                ),
                "selection_mask_values": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                "base_checkpoint_weight_sha256": (
                    "c587060a4599f096433183ee7bc88de3234021291f1815e332950b80025d93b7"
                ),
                "effective_loss_weights_sha256": (
                    "b15eb5519fa66d2d01c020c1c3f5f93a62a6b015bcfa7e1c171e71770a729b61"
                ),
                "packed_inputs_sha256": (
                    "5061ac4deaaf0e6bc153f0766aac8ba630d9d29c7a235a0998fc8121072eb910"
                ),
            },
        },
    }


@pytest.fixture(scope="module")
def reduced_gate_c2_removed_path_reports() -> dict[str, Any]:
    reduced = _reduced_gate_c_config()
    config = gate_c.GateCConfig(
        gate_a_config=dataclasses.replace(
            reduced.gate_a_config,
            neuron_count=2_048,
        ),
        gate_b_config=dataclasses.replace(
            reduced.gate_b_config,
            neuron_count=2_048,
        ),
    )
    gate_a_data = gate_c._regenerate_gate_a_data(config)
    production_gate_b_config = gate_b.DepthGateConfig()
    gate_b_schedule = gate_b._build_schedule(production_gate_b_config)
    gate_b_validation = gate_b._encode_validation_data(
        gate_b_schedule,
        production_gate_b_config,
    )
    data = {
        "gate_a": gate_a_data,
        "gate_b": (gate_b_schedule, gate_b_validation),
    }
    initialization: dict[str, Any] = {"initialization": {}}
    for regime in _REGIMES:
        model = gate_c._new_model_for_arm(
            config,
            regime,
            "full",
            batch_size=1,
        )
        initialization["initialization"][regime] = {
            "canonical_full": {
                "parameter_sha256": legacy._array_digest(
                    legacy._parameter_values(model)
                )
            }
        }
    captures: dict[str, Any] = {}
    current_regime = ""

    def fake_chunked(
        model_factory: Any,
        inputs: Any,
        *,
        algo_factory: Any,
        chunk_size: int,
        compiled_scan: bool = False,
        after_init: Any | None = None,
    ) -> dict[str, Any]:
        model = model_factory()
        lifecycle = ["model_factory"]
        brainstate.nn.init_all_states(model, batch_size=1)
        lifecycle.append("init_all_states")
        algorithm = SimpleNamespace(boundary="initialized_etrace")
        lifecycle.extend(("compile_graph", "init_etrace_state"))
        callback_count = 0
        if after_init is not None:
            after_init(model, algorithm)
            callback_count += 1
            lifecycle.append("after_init")
        gradient_start_snapshot = original_stop(model.snapshot_state())
        lifecycle.append("first_gradient_chunk")
        captures[current_regime] = {
            "inputs": np.ascontiguousarray(np.asarray(inputs)),
            "chunk_size": chunk_size,
            "model": model,
            "algo_factory": algo_factory,
            "compiled_scan": compiled_scan,
            "after_init_callback_count": callback_count,
            "gradient_start_snapshot": gradient_start_snapshot,
            "lifecycle": lifecycle,
        }
        gradients = {
            path: jax.tree.map(
                lambda leaf: jnp.ones_like(leaf),
                value,
            )
            for path, value in legacy._parameter_values(model).items()
        }
        for path in gate_c.GATE_C2_REMOVED_PATHS:
            gradients[path] = jax.tree.map(
                lambda leaf: jnp.zeros_like(leaf),
                gradients[path],
            )
        return gradients

    original = gate_c.chunked_online_param_gradients
    original_stop = gate_c._gate_c2_stop_gradient_snapshot
    stop_gradient_calls: list[str] = []

    def recording_stop(snapshot: Any) -> Any:
        stop_gradient_calls.append(current_regime)
        return original_stop(snapshot)

    reports: dict[str, Any] = {}
    try:
        gate_c.chunked_online_param_gradients = fake_chunked
        gate_c._gate_c2_stop_gradient_snapshot = recording_stop
        for regime in _REGIMES:
            current_regime = regime
            reports[regime] = (
                gate_c._gate_c2_removed_path_finite_window_influence(
                    config,
                    initialization=initialization,
                    regime=regime,
                    data=data[regime],
                )
            )
    finally:
        gate_c.chunked_online_param_gradients = original
        gate_c._gate_c2_stop_gradient_snapshot = original_stop
    full_gradient_records: dict[str, Any] = {}
    for regime in _REGIMES:
        full_model = gate_c._new_model_for_arm(
            gate_c.GateCConfig(),
            regime,
            "query_only",
            batch_size=1,
        )
        gradients = {
            path: jax.tree.map(
                lambda leaf: jnp.ones_like(leaf),
                value,
            )
            for path, value in legacy._parameter_values(full_model).items()
        }
        for path in gate_c.GATE_C2_REMOVED_PATHS:
            gradients[path] = jax.tree.map(
                lambda leaf: jnp.zeros_like(leaf),
                gradients[path],
            )
        full_gradient_records[regime] = {
            "global": gate_c._gate_c2_global_gradient_record(gradients),
            "live_paths": {
                path: gate_c._gate_c2_gradient_path_record(
                    path,
                    gradients[path],
                    require_zero=False,
                )
                for path in gate_c.GATE_C2_LIVE_PATHS
            },
            "removed_paths": {
                path: gate_c._gate_c2_gradient_path_record(
                    path,
                    gradients[path],
                    require_zero=True,
                )
                for path in gate_c.GATE_C2_REMOVED_PATHS
            },
        }
    return {
        "config": config,
        "data": data,
        "initialization": initialization,
        "reports": reports,
        "captures": captures,
        "full_gradient_records": full_gradient_records,
        "stop_gradient_calls": stop_gradient_calls,
    }


_GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY = {
    path: (1, *shape[1:])
    for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
}


def _gate_c2_test_raw_h0_snapshot(
    snapshot: Any,
    *,
    parameter_sha256: str,
) -> dict[str, Any]:
    arrays = gate_c._gate_c2_snapshot_arrays(snapshot)
    assert set(arrays) == set(_GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY)
    hidden_paths: dict[str, Any] = {}
    for path, expected_shape in _GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY.items():
        array = np.ascontiguousarray(arrays[path])
        assert array.dtype.str == "<f4"
        assert array.shape == expected_shape
        finite = np.isfinite(array)
        hidden_paths[path] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_hex": array.tobytes(order="C").hex(),
            "sha256": legacy._digest_arrays(array),
            "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()),
        }

    def path_key(path: str) -> tuple[str, int]:
        base_path, leaf_index = path.rsplit("#", 1)
        return base_path, int(leaf_index)

    fields: list[bytes] = [b"example21-gate-c-hidden-state-v1"]
    for path in sorted(hidden_paths, key=path_key):
        base_path, leaf_index = path.rsplit("#", 1)
        leaf = hidden_paths[path]
        fields.extend(
            (
                base_path.encode("utf-8"),
                leaf_index.encode("ascii"),
                leaf["dtype"].encode("ascii"),
                ",".join(map(str, leaf["shape"])).encode("ascii"),
                bytes.fromhex(leaf["data_hex"]),
            )
        )
    return {
        "hidden_paths": hidden_paths,
        "hidden_state_tree_sha256": hashlib.sha256(b"\0".join(fields)).hexdigest(),
        "parameter_sha256": parameter_sha256,
    }


def _assert_gate_c2_actual_gradient_boundary(
    fixture: Mapping[str, Any],
    *,
    regime: str,
) -> None:
    capture = fixture["captures"][regime]
    objective_name = (
        "gate_a_h1" if regime == "gate_a" else "gate_b_index0_r8_h8"
    )
    boundary = fixture["reports"][regime]["objectives"][objective_name][
        "continuation"
    ]["h0_gradient_boundary"]
    canonical = fixture["initialization"]["initialization"][regime][
        "canonical_full"
    ]["parameter_sha256"]
    assert _parameter_digest(capture["model"]) == canonical
    materialized = _gate_c2_test_raw_h0_snapshot(
        capture["model"]._gate_c2_h0_snapshot,
        parameter_sha256=canonical,
    )
    actual = _gate_c2_test_raw_h0_snapshot(
        capture["gradient_start_snapshot"],
        parameter_sha256=canonical,
    )

    assert capture["compiled_scan"] is True
    assert capture["after_init_callback_count"] == 1
    assert capture["lifecycle"] == [
        "model_factory",
        "init_all_states",
        "compile_graph",
        "init_etrace_state",
        "after_init",
        "first_gradient_chunk",
    ]
    assert set(boundary) == {
        "capture_point",
        "capture_count",
        "materialized_prefix",
        "actual_gradient_start",
        "canonical_parameter_sha256",
        "all_hidden_leaves_equal",
        "passed",
    }
    assert boundary["capture_point"] == (
        "after_init_etrace_state_before_first_gradient_chunk"
    )
    assert boundary["capture_count"] == 1
    assert boundary["materialized_prefix"] == materialized
    assert boundary["actual_gradient_start"] == actual
    assert boundary["canonical_parameter_sha256"] == canonical
    assert materialized == actual
    assert boundary["all_hidden_leaves_equal"] is True
    assert boundary["passed"] is True


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_removed_path_captures_raw_h0_at_actual_after_init_hook(
    regime: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    _assert_gate_c2_actual_gradient_boundary(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )


@requires_gate_c
def test_gate_c2_gate_a_objective_executes_only_h1_with_canonical_half_weight(
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = reduced_gate_c2_removed_path_reports["reports"]["gate_a"]
    capture = reduced_gate_c2_removed_path_reports["captures"]["gate_a"]
    objective = report["objectives"]["gate_a_h1"]
    packed = capture["inputs"]

    assert capture["chunk_size"] == 1
    assert reduced_gate_c2_removed_path_reports["stop_gradient_calls"].count(
        "gate_a"
    ) == 1
    assert capture["model"]._gate_c2_h0_snapshot is not None
    assert packed.dtype == np.dtype(np.float32)
    assert packed.shape == (1, 1, 44)
    assert legacy._digest_arrays(packed) == (
        "aee2f5f2f2a672f091c4d02e24ade55262e8ebe50ddd32317c4ead5f8e5b84c5"
    )
    np.testing.assert_array_equal(packed[:, 0, -1], np.asarray([0.5], np.float32))
    assert objective["source_contract"]["metadata"] == (
        gate_c.GATE_C2_REMOVED_PATH_OBJECTIVES["gate_a_h1"]["source_metadata"]
    )
    assert objective["continuation"]["source_indices"] == [5]
    assert objective["continuation"]["selection_mask"]["values"] == [True]
    assert objective["continuation"]["effective_loss_weights"]["values"] == [0.5]
    _assert_gate_c2_actual_gradient_boundary(
        reduced_gate_c2_removed_path_reports,
        regime="gate_a",
    )


@requires_gate_c
def test_gate_c2_gate_b_objective_carries_full_h1_h8_suffix_and_selects_only_h8(
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = reduced_gate_c2_removed_path_reports["reports"]["gate_b"]
    capture = reduced_gate_c2_removed_path_reports["captures"]["gate_b"]
    objective = report["objectives"]["gate_b_index0_r8_h8"]
    packed = capture["inputs"]
    expected_weight = np.float32(1.0 / 9.0)

    assert capture["chunk_size"] == 1
    assert reduced_gate_c2_removed_path_reports["stop_gradient_calls"].count(
        "gate_b"
    ) == 1
    assert capture["model"]._gate_c2_h0_snapshot is not None
    assert packed.dtype == np.dtype(np.float32)
    assert packed.shape == (8, 1, 50)
    assert legacy._digest_arrays(packed) == (
        "5061ac4deaaf0e6bc153f0766aac8ba630d9d29c7a235a0998fc8121072eb910"
    )
    np.testing.assert_array_equal(packed[:7, 0, -1], np.zeros(7, np.float32))
    np.testing.assert_array_equal(
        packed[7:, 0, -1],
        np.asarray([expected_weight], np.float32),
    )
    assert objective["continuation"]["source_indices"] == list(range(11, 19))
    assert objective["continuation"]["selection_mask"]["values"] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert objective["continuation"]["base_checkpoint_weights"]["values"] == [
        float(expected_weight)
    ] * 8
    assert objective["continuation"]["effective_loss_weights"]["values"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        float(expected_weight),
    ]
    assert objective["base_checkpoint_weight"]["values"] == [
        float(expected_weight)
    ]
    _assert_gate_c2_actual_gradient_boundary(
        reduced_gate_c2_removed_path_reports,
        regime="gate_b",
    )


def _gate_c2_validator_ready_removed_path_report(
    fixture: Mapping[str, Any],
    *,
    regime: str,
) -> dict[str, Any]:
    report = copy.deepcopy(fixture["reports"][regime])
    report.update(copy.deepcopy(fixture["full_gradient_records"][regime]))
    report["complete"] = True
    return report


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_removed_path_validator_accepts_complete_raw_objective(
    regime: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    assert gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize(
    "mutation",
    (
        "source_metadata",
        "source_events_dtype",
        "source_events_shape",
        "source_events_digest",
        "source_advances_values",
        "source_targets_digest",
        "canonical_weights_dtype",
        "canonical_weights_digest",
        "h0_prefix_indices",
        "h0_prefix_digest",
        "schedule_cross_bound",
        "continuation_indices",
        "continuation_events_digest",
        "batched_events_shape",
        "continuation_advances",
        "continuation_targets",
        "selection_mask",
        "base_weights",
        "effective_weights",
        "packed_inputs_digest",
        "h0_boundary_capture_point",
        "h0_boundary_tree_digest",
        "source_slice_exact",
        "continuation_passed",
        "raw_loss_digest",
        "raw_loss_value",
        "base_weight_digest",
        "weighted_loss_digest",
        "weighted_loss_value",
        "objective_passed",
        "global_digest",
        "live_path_zero",
        "removed_path_nonzero",
    ),
)
@requires_gate_c
def test_gate_c2_removed_path_validator_cross_binds_every_raw_field(
    regime: str,
    mutation: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    assert gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )
    canonical_report = copy.deepcopy(report)
    objective_name = (
        "gate_a_h1" if regime == "gate_a" else "gate_b_index0_r8_h8"
    )
    objective = report["objectives"][objective_name]
    source = objective["source_contract"]
    continuation = objective["continuation"]

    if mutation == "source_metadata":
        source["metadata"]["mapping_id"] += 1
    elif mutation == "source_events_dtype":
        source["events"]["dtype"] = "<f8"
    elif mutation == "source_events_shape":
        source["events"]["shape"][0] -= 1
    elif mutation == "source_events_digest":
        source["events"]["sha256"] = "0" * 64
    elif mutation == "source_advances_values":
        source["advances"]["values"][0] = False
    elif mutation == "source_targets_digest":
        source["targets"]["sha256"] = "0" * 64
    elif mutation == "canonical_weights_dtype":
        source["canonical_loss_weights"]["dtype"] = (
            "<f8" if regime == "gate_a" else "<f4"
        )
    elif mutation == "canonical_weights_digest":
        source["canonical_loss_weights"]["sha256"] = "0" * 64
    elif mutation == "h0_prefix_indices":
        source["h0_prefix"]["source_indices"] = [0]
    elif mutation == "h0_prefix_digest":
        source["h0_prefix"]["sha256"] = "0" * 64
    elif mutation == "schedule_cross_bound":
        first = next(iter(source["schedule_cross_bound"]))
        source["schedule_cross_bound"][first] = "0" * 64
    elif mutation == "continuation_indices":
        continuation["source_indices"] = [4]
    elif mutation == "continuation_events_digest":
        continuation["source_events"]["sha256"] = "0" * 64
    elif mutation == "batched_events_shape":
        continuation["batched_events"]["shape"][-1] += 1
    elif mutation == "continuation_advances":
        continuation["advances"]["values"][0] = False
    elif mutation == "continuation_targets":
        continuation["targets"]["values"][-1] = 0
    elif mutation == "selection_mask":
        continuation["selection_mask"]["values"] = [False] * len(
            continuation["selection_mask"]["values"]
        )
    elif mutation == "base_weights":
        continuation["base_checkpoint_weights"]["values"] = [1.0] * len(
            continuation["base_checkpoint_weights"]["values"]
        )
    elif mutation == "effective_weights":
        continuation["effective_loss_weights"]["values"] = [1.0] * len(
            continuation["effective_loss_weights"]["values"]
        )
    elif mutation == "packed_inputs_digest":
        continuation["packed_inputs"]["sha256"] = "0" * 64
    elif mutation == "h0_boundary_capture_point":
        continuation["h0_gradient_boundary"]["capture_point"] = "after_gradients"
    elif mutation == "h0_boundary_tree_digest":
        continuation["h0_gradient_boundary"]["actual_gradient_start"][
            "hidden_state_tree_sha256"
        ] = "0" * 64
    elif mutation == "source_slice_exact":
        continuation["source_slice_exact"] = False
    elif mutation == "continuation_passed":
        continuation["passed"] = False
    elif mutation == "raw_loss_digest":
        objective["raw_cross_entropy"]["sha256"] = "0" * 64
    elif mutation == "raw_loss_value":
        objective["raw_cross_entropy"]["value"] *= 2.0
    elif mutation == "base_weight_digest":
        objective["base_checkpoint_weight"]["sha256"] = "0" * 64
    elif mutation == "weighted_loss_digest":
        objective["weighted_cross_entropy"]["sha256"] = "0" * 64
    elif mutation == "weighted_loss_value":
        objective["weighted_cross_entropy"]["value"] *= 2.0
    elif mutation == "objective_passed":
        objective["passed"] = False
    elif mutation == "global_digest":
        report["global"]["sha256"] = "0" * 64
    elif mutation == "live_path_zero":
        path = gate_c.GATE_C2_LIVE_PATHS[0]
        report["live_paths"][path]["nonzero"] = False
    elif mutation == "removed_path_nonzero":
        path = gate_c.GATE_C2_REMOVED_PATHS[0]
        report["removed_paths"][path]["exact_zero"] = False
    else:
        raise AssertionError(f"unhandled removed-path mutation {mutation}")

    assert not gate_a._json_exact(report, canonical_report)
    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize("path", tuple(_GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY))
@requires_gate_c
def test_gate_c2_removed_path_validator_recomputes_raw_bytes_for_every_h0_leaf(
    path: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    regime = "gate_a"
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    objective = report["objectives"]["gate_a_h1"]
    leaf = objective["continuation"]["h0_gradient_boundary"][
        "actual_gradient_start"
    ]["hidden_paths"][path]
    raw = leaf["data_hex"]
    leaf["data_hex"] = ("01" if raw[:2] != "01" else "02") + raw[2:]

    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize(
    ("snapshot_name", "mutation"),
    (
        ("materialized_prefix", "dtype"),
        ("materialized_prefix", "shape"),
        ("materialized_prefix", "leaf_digest"),
        ("materialized_prefix", "finite_count"),
        ("materialized_prefix", "tree_digest"),
        ("actual_gradient_start", "dtype"),
        ("actual_gradient_start", "shape"),
        ("actual_gradient_start", "leaf_digest"),
        ("actual_gradient_start", "finite_count"),
        ("actual_gradient_start", "tree_digest"),
    ),
)
@requires_gate_c
def test_gate_c2_removed_path_validator_recomputes_h0_geometry_and_digests(
    regime: str,
    snapshot_name: str,
    mutation: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    objective_name = (
        "gate_a_h1" if regime == "gate_a" else "gate_b_index0_r8_h8"
    )
    snapshot = report["objectives"][objective_name]["continuation"][
        "h0_gradient_boundary"
    ][snapshot_name]
    leaf = snapshot["hidden_paths"]["context_memory#0"]
    if mutation == "dtype":
        leaf["dtype"] = "<f8"
    elif mutation == "shape":
        leaf["shape"][-1] += 1
    elif mutation == "leaf_digest":
        leaf["sha256"] = "0" * 64
    elif mutation == "finite_count":
        leaf["finite_count"] -= 1
        leaf["nonfinite_count"] += 1
    elif mutation == "tree_digest":
        snapshot["hidden_state_tree_sha256"] = "0" * 64
    else:
        raise AssertionError(f"unhandled H0 snapshot mutation {mutation}")

    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize(
    "mutation",
    (
        "colluding_hidden_tree_digests",
        "colluding_parameter_digests",
        "capture_count",
        "stored_hidden_equality",
        "stored_boundary_pass",
    ),
)
@requires_gate_c
def test_gate_c2_removed_path_validator_rejects_colluding_h0_boundary_evidence(
    regime: str,
    mutation: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    objective_name = (
        "gate_a_h1" if regime == "gate_a" else "gate_b_index0_r8_h8"
    )
    boundary = report["objectives"][objective_name]["continuation"][
        "h0_gradient_boundary"
    ]
    if mutation == "colluding_hidden_tree_digests":
        colluding = "d" * 64
        boundary["materialized_prefix"]["hidden_state_tree_sha256"] = colluding
        boundary["actual_gradient_start"][
            "hidden_state_tree_sha256"
        ] = colluding
    elif mutation == "colluding_parameter_digests":
        colluding = "e" * 64
        boundary["materialized_prefix"]["parameter_sha256"] = colluding
        boundary["actual_gradient_start"]["parameter_sha256"] = colluding
        boundary["canonical_parameter_sha256"] = colluding
    elif mutation == "capture_count":
        boundary["capture_count"] = 2
    elif mutation == "stored_hidden_equality":
        boundary["all_hidden_leaves_equal"] = False
    elif mutation == "stored_boundary_pass":
        boundary["passed"] = False
    else:
        raise AssertionError(f"unhandled H0 boundary mutation {mutation}")

    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("path", gate_c.GATE_C2_LIVE_PATHS)
@requires_gate_c
def test_gate_c2_removed_path_validator_rejects_all_zero_live_leaf_inventory(
    regime: str,
    path: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    live = report["live_paths"][path]
    assert live["l2_norm"] > 0.0
    assert live["nonzero"] is True
    for leaf in live["leaves"]:
        leaf["zero_count"] = leaf["value_count"]

    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("index", False),
        ("finite_count", "float"),
        ("nonfinite_count", False),
    ),
)
@requires_gate_c
def test_gate_c2_live_gradient_leaf_schema_rejects_noninteger_scalars(
    field: str,
    replacement: Any,
) -> None:
    path = gate_c.GATE_C2_LIVE_PATHS[0]
    record = gate_c._gate_c2_gradient_path_record(
        path,
        np.ones((2, 3), dtype=np.float32),
        require_zero=False,
    )
    assert gate_c._gate_c2_live_gradient_record_complete(record, path=path)
    leaf = record["leaves"][0]
    leaf[field] = (
        float(leaf["value_count"])
        if replacement == "float"
        else replacement
    )

    assert not gate_c._gate_c2_live_gradient_record_complete(record, path=path)


def _passing_gate_c2_controls_report(
    admission: Mapping[str, Any],
    no_read_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    config = gate_c.GateCConfig()
    source = copy.deepcopy(admission["source_start"])
    environment = copy.deepcopy(admission["environment"])
    environment["execution_and_update_evidence"] = (
        _gate_c2_control_execution_evidence(admission)
    )
    report = {
        "schema_version": 1,
        "control": "example21_gate_c2_pretraining_control_admission",
        "qualification_regime": (
            "preregistered_gate_c2_pretraining_controls"
        ),
        "learner": "pp_prop_only",
        "prerequisites": {
            "gate_a": dict(_GATE_A_REFERENCE),
            "gate_b": dict(_GATE_B_REFERENCE),
            "gate_c_initialization": _gate_c_initialization_wrapper(admission),
        },
        "regimes": {
            regime: {
                "spec": dataclasses.asdict(gate_c.REGIME_SPECS[regime]),
                "config": dataclasses.asdict(
                    config.gate_a_config
                    if regime == "gate_a"
                    else config.gate_b_config
                ),
                "schedule_identity": copy.deepcopy(
                    gate_c._schedule_identity_report(config)[regime]
                ),
                "paired_h0_operational_equivalence": (
                    _passing_gate_c2_operational_h0(
                        admission,
                        regime=regime,
                    )
                ),
                "query_only_latent_no_read": no_read_reports[regime],
            }
            for regime in _REGIMES
        },
        "mechanism_oracle": _formal_mechanism_oracle(),
        "source_start": source,
        "source_end": copy.deepcopy(source),
        "source_files": copy.deepcopy(admission["source_files"]),
        "environment": environment,
        "total_wall_seconds": 1.0,
        "qualification": {
            "criteria": {
                name: False for name in _GATE_C2_CONTROLS_CRITERIA
            },
            "passed": False,
            "interpretation": "colluding-embedded-value-is-not-trusted",
        },
    }
    return report


@requires_gate_c
def test_gate_c2_controls_qualifier_recomputes_all_nine_true_from_raw_evidence(
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, no_read_reports = passing_gate_c2_no_read_reports
    report = _passing_gate_c2_controls_report(admission, no_read_reports)

    def no_read_complete(
        value: Any,
        got_admission: Any,
        *,
        regime: str,
    ) -> bool:
        return bool(
            value is no_read_reports[regime]
            and got_admission is not None
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_query_only_latent_no_read_complete",
        no_read_complete,
    )
    qualification = gate_c._gate_c2_controls_qualification(
        report,
        config=gate_c.GateCConfig(),
    )
    assert qualification == {
        "criteria": {
            name: True for name in _GATE_C2_CONTROLS_CRITERIA
        },
        "passed": True,
        "interpretation": "gate_c2_pretraining_controls_passed",
    }
    assert gate_c._gate_c2_controls_qualification_report(
        report,
        config=gate_c.GateCConfig(),
    ) == qualification
    assert report["qualification"]["passed"] is False


@requires_gate_c
def test_gate_c2_controls_qualifier_accepts_reordered_exact_regime_mapping(
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, no_read_reports = passing_gate_c2_no_read_reports
    report = _passing_gate_c2_controls_report(admission, no_read_reports)

    def no_read_complete(
        value: Any,
        got_admission: Any,
        *,
        regime: str,
    ) -> bool:
        return bool(
            value is no_read_reports[regime]
            and got_admission is not None
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_query_only_latent_no_read_complete",
        no_read_complete,
    )
    raw_qualification = gate_c._gate_c2_controls_qualification(
        report,
        config=gate_c.GateCConfig(),
    )
    assert raw_qualification["passed"] is True

    regimes = report["regimes"]
    reordered_regimes = dict(reversed(tuple(regimes.items())))
    assert reordered_regimes == regimes
    assert list(reordered_regimes) != list(regimes)
    report["regimes"] = reordered_regimes
    reordered_qualification = gate_c._gate_c2_controls_qualification(
        report,
        config=gate_c.GateCConfig(),
    )

    for mutation in ("missing", "extra"):
        malformed = dict(report)
        malformed_regimes = dict(reordered_regimes)
        if mutation == "missing":
            malformed_regimes.pop("gate_a")
        else:
            malformed_regimes["unexpected"] = reordered_regimes["gate_a"]
        malformed["regimes"] = malformed_regimes
        qualification = gate_c._gate_c2_controls_qualification(
            malformed,
            config=gate_c.GateCConfig(),
        )
        assert qualification["criteria"]["exact_configuration"] is False
        assert qualification["passed"] is False

    assert reordered_qualification == raw_qualification
    assert reordered_qualification["passed"] is True


@requires_gate_c
def test_gate_c2_controls_qualification_survives_canonical_json_roundtrip(
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, no_read_reports = passing_gate_c2_no_read_reports
    _patch_gate_c2_fixture_removed_path_validator(monkeypatch, admission)
    report = _passing_gate_c2_controls_report(admission, no_read_reports)
    raw_qualification = gate_c._gate_c2_controls_qualification(
        report,
        config=gate_c.GateCConfig(),
    )
    assert raw_qualification["passed"] is True

    decoded = _gate_c2_strict_canonical_json_roundtrip(report)
    decoded_qualification = gate_c._gate_c2_controls_qualification(
        decoded,
        config=gate_c.GateCConfig(),
    )
    assert decoded_qualification == raw_qualification
    assert decoded_qualification["passed"] is True


@pytest.mark.parametrize(
    ("criterion", "mutation"),
    (
        ("schema_and_control", "control"),
        ("exact_configuration", "config"),
        ("prerequisites_authenticated", "prerequisite"),
        ("initialization_authenticated", "initialization"),
        ("canonical_schedules_complete", "schedule"),
        ("no_behavioral_or_optimizer_updates", "training_call"),
        ("paired_h0_operational_equivalence", "h0"),
        ("mechanism_oracle_complete", "oracle"),
        ("source_and_gpu_authenticated", "source"),
    ),
)
@requires_gate_c
def test_gate_c2_controls_qualifier_independently_rejects_each_criterion(
    criterion: str,
    mutation: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, no_read_reports = passing_gate_c2_no_read_reports
    report = _passing_gate_c2_controls_report(admission, no_read_reports)

    def no_read_complete(
        value: Any,
        got_admission: Any,
        *,
        regime: str,
    ) -> bool:
        return bool(
            value is no_read_reports[regime]
            and got_admission is not None
        )

    monkeypatch.setattr(
        gate_c,
        "_gate_c2_query_only_latent_no_read_complete",
        no_read_complete,
    )
    if mutation == "control":
        report["control"] = "wrong-control"
    elif mutation == "config":
        report["regimes"]["gate_a"]["spec"]["training_updates"] -= 1
    elif mutation == "prerequisite":
        report["prerequisites"]["gate_a"]["result_sha256"] = "0" * 64
    elif mutation == "initialization":
        report["prerequisites"]["gate_c_initialization"][
            "bundle_sha256"
        ] = "0" * 64
    elif mutation == "schedule":
        report["regimes"]["gate_b"]["schedule_identity"][
            "schedule_sha256"
        ] = "0" * 64
    elif mutation == "training_call":
        execution = report["environment"]["execution_and_update_evidence"]
        execution["trainer_factory_calls"] = [
            "examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer"
        ]
        execution["trainer_factory_call_count"] = 1
    elif mutation == "h0":
        report["regimes"]["gate_a"][
            "paired_h0_operational_equivalence"
        ]["backend"] = "jax_raw"
    elif mutation == "oracle":
        report["mechanism_oracle"]["complete"] = False
    elif mutation == "source":
        report["source_end"]["dirty"] = True
    else:
        raise AssertionError(f"unhandled controls mutation {mutation}")

    qualification = gate_c._gate_c2_controls_qualification(
        report,
        config=gate_c.GateCConfig(),
    )
    assert qualification["criteria"][criterion] is False
    assert qualification["passed"] is False
    assert qualification["interpretation"] == (
        "gate_c2_pretraining_controls_failed_stop"
    )


@requires_gate_c
def test_gate_c2_controls_parser_accepts_only_fixed_target_inputs() -> None:
    config, _, _ = _gate_c_cli_host_argv()
    head = "c" * 40
    gate_a_paths = launcher._gate_a_artifact_paths(config)
    gate_b_paths = launcher._formal_gate_b_artifact_paths(config)
    initialization_manifest = (
        config.output_dir / f"{head}-gate-c-init.manifest.json"
    )
    output = config.output_dir / f"{head}-gate-c2-controls.json"
    argv = [
        "--target",
        "gate_c2_controls",
        "--gate-a-result",
        str(gate_a_paths.result),
        "--gate-a-manifest",
        str(gate_a_paths.manifest),
        "--gate-b-manifest",
        str(gate_b_paths.manifest),
        "--gate-c-init-manifest",
        str(initialization_manifest),
        "--output",
        str(output),
    ]

    parsed = gate_c._parser().parse_args(argv)

    assert set(vars(parsed)) == {
        "target",
        "gate_a_result",
        "gate_a_manifest",
        "gate_b_manifest",
        "gate_c_init_manifest",
        "output",
    }
    assert parsed.target == "gate_c2_controls"
    assert parsed.gate_c_init_manifest == initialization_manifest
    assert parsed.output == output


@requires_gate_c
def test_gate_c2_controls_runner_is_read_only_and_orders_all_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialization, _ = _passing_gate_c_initialization_report()
    wrapper = _gate_c_initialization_wrapper(initialization)
    prerequisites = {
        "gate_a": dict(_GATE_A_REFERENCE),
        "gate_b": dict(_GATE_B_REFERENCE),
        "gate_c_initialization": wrapper,
    }
    config = gate_c.GateCConfig()
    gate_a_data = object()
    gate_b_data = (object(), object())
    source_start = _passing_gate_c_source()
    source_end = copy.deepcopy(source_start)
    source_files = _current_gate_c_source_files()
    environment = _passing_gate_c_environment()
    schedules = {
        "gate_a": {"identity": "gate-a-schedule"},
        "gate_b": {"identity": "gate-b-schedule"},
    }
    h0_reports = {
        regime: {"probe": f"{regime}:operational-h0"}
        for regime in _REGIMES
    }
    no_read_reports = {
        regime: {"probe": f"{regime}:query-only-no-read"}
        for regime in _REGIMES
    }
    mechanism_oracle = {"probe": "fresh-initialization-oracle"}
    events: list[tuple[Any, ...]] = []
    forbidden_calls: list[str] = []

    def validate_initialization(
        prerequisite: Mapping[str, Any],
        actual_config: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append(("initialization_authenticated",))
        assert prerequisite is wrapper
        assert actual_config is config
        assert kwargs == {
            "source_start": source_start,
            "environment": environment,
            "source_files": source_files,
            "require_pass": True,
        }
        return initialization

    def regenerate_gate_a(actual_config: Any) -> object:
        assert actual_config is config
        events.append(("data", "gate_a"))
        return gate_a_data

    def regenerate_gate_b(actual_config: Any) -> tuple[object, object]:
        assert actual_config is config
        events.append(("data", "gate_b"))
        return gate_b_data

    def actual_schedule(
        actual_config: Any,
        actual_gate_a_data: Any,
        actual_gate_b_data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert actual_gate_a_data is gate_a_data
        assert actual_gate_b_data is gate_b_data
        events.append(("schedule_identity",))
        return copy.deepcopy(schedules)

    def operational_h0(
        actual_config: Any,
        *,
        initialization: Mapping[str, Any],
        regime: str,
        data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert initialization is initialization_report
        assert data is (gate_a_data if regime == "gate_a" else gate_b_data)
        events.append(("paired_h0_operational_equivalence", regime))
        return copy.deepcopy(h0_reports[regime])

    def no_read(
        actual_config: Any,
        *,
        initialization: Mapping[str, Any],
        regime: str,
        data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert initialization is initialization_report
        assert data is (gate_a_data if regime == "gate_a" else gate_b_data)
        events.append(("query_only_latent_no_read", regime))
        return copy.deepcopy(no_read_reports[regime])

    def oracle(
        actual_config: Any,
        *,
        initialization: Mapping[str, Any],
        gate_b_data: Any,
    ) -> dict[str, Any]:
        assert actual_config is config
        assert initialization is initialization_report
        assert gate_b_data is gate_b_data_value
        events.append(("mechanism_oracle",))
        return copy.deepcopy(mechanism_oracle)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("control admission reached training or optimizer code")

    def source_end_reporter() -> dict[str, Any]:
        events.append(("source_end",))
        return copy.deepcopy(source_end)

    def qualification(
        report: Mapping[str, Any],
        *,
        config: Any,
    ) -> dict[str, Any]:
        assert config is control_config
        events.append(("qualification",))
        assert set(report) == {
            "schema_version",
            "control",
            "qualification_regime",
            "learner",
            "prerequisites",
            "regimes",
            "mechanism_oracle",
            "source_start",
            "source_end",
            "source_files",
            "environment",
            "total_wall_seconds",
        }
        return {
            "criteria": {
                name: False for name in _GATE_C2_CONTROLS_CRITERIA
            },
            "passed": False,
            "interpretation": "gate_c2_pretraining_controls_failed_stop",
        }

    initialization_report = initialization
    gate_b_data_value = gate_b_data
    control_config = config
    monkeypatch.setattr(
        gate_c,
        "_validated_gate_c_initialization_admission",
        validate_initialization,
    )
    monkeypatch.setattr(gate_c, "_regenerate_gate_a_data", regenerate_gate_a)
    monkeypatch.setattr(gate_c, "_regenerate_gate_b_data", regenerate_gate_b)
    monkeypatch.setattr(
        gate_c,
        "_actual_schedule_identity_report",
        actual_schedule,
        raising=False,
    )
    monkeypatch.setattr(
        gate_c,
        "_schedule_identity_report",
        lambda actual_config: (
            copy.deepcopy(schedules)
            if actual_config is config
            else (_ for _ in ()).throw(AssertionError("unexpected config"))
        ),
    )
    monkeypatch.setattr(
        gate_c,
        "_paired_h0_operational_equivalence_report",
        operational_h0,
        raising=False,
    )
    monkeypatch.setattr(
        gate_c,
        "_query_only_latent_no_read_report",
        no_read,
        raising=False,
    )
    monkeypatch.setattr(gate_c, "_mechanism_oracle", oracle)
    monkeypatch.setattr(gate_c, "_make_arm_trainer", forbidden)
    monkeypatch.setattr(gate_a, "_make_pp_prop_trainer", forbidden)
    monkeypatch.setattr(gate_b, "_make_pp_prop_trainer", forbidden)
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_controls_qualification_report",
        qualification,
        raising=False,
    )

    result = gate_c.run_gate_c2_controls(
        config,
        prerequisites=prerequisites,
        source_start=source_start,
        source_end_reporter=source_end_reporter,
        source_files=source_files,
        environment=environment,
    )

    assert forbidden_calls == []
    assert events == [
        ("initialization_authenticated",),
        ("data", "gate_a"),
        ("data", "gate_b"),
        ("schedule_identity",),
        ("paired_h0_operational_equivalence", "gate_a"),
        ("query_only_latent_no_read", "gate_a"),
        ("paired_h0_operational_equivalence", "gate_b"),
        ("query_only_latent_no_read", "gate_b"),
        ("mechanism_oracle",),
        ("source_end",),
        ("qualification",),
    ]
    assert set(result) == {
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
        "total_wall_seconds",
    }
    assert result["schema_version"] == 1
    assert result["control"] == (
        "example21_gate_c2_pretraining_control_admission"
    )
    assert result["qualification_regime"] == (
        "preregistered_gate_c2_pretraining_controls"
    )
    assert result["learner"] == "pp_prop_only"
    assert set(result["prerequisites"]) == {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }
    for regime in _REGIMES:
        assert set(result["regimes"][regime]) == {
            "spec",
            "config",
            "schedule_identity",
            "paired_h0_operational_equivalence",
            "query_only_latent_no_read",
        }
        assert result["regimes"][regime]["schedule_identity"] == (
            schedules[regime]
        )
        assert result["regimes"][regime][
            "paired_h0_operational_equivalence"
        ] == h0_reports[regime]
        assert result["regimes"][regime]["query_only_latent_no_read"] == (
            no_read_reports[regime]
        )
    assert result["mechanism_oracle"] == mechanism_oracle
    assert set(result["environment"]) == {
        "backend",
        "devices",
        "image_digest",
        "jax",
        "python",
        "execution_and_update_evidence",
    }
    evidence = result["environment"]["execution_and_update_evidence"]
    assert set(evidence) == {
        "instrumented_training_entry_points",
        "trainer_factory_calls",
        "trainer_factory_call_count",
        "training_step_calls",
        "training_step_call_count",
        "optimizer_constructor_calls",
        "optimizer_instance_count",
        "optimizer_update_calls",
        "optimizer_update_call_count",
        "model_factory_calls",
        "model_constructor_calls",
        "materialized_roles",
        "complete",
    }
    assert evidence["instrumented_training_entry_points"] == list(
        _GATE_C2_CONTROL_AUDIT_LABELS
    )
    assert evidence["trainer_factory_calls"] == []
    assert evidence["trainer_factory_call_count"] == 0
    assert evidence["training_step_calls"] == []
    assert evidence["training_step_call_count"] == 0
    assert evidence["optimizer_constructor_calls"] == []
    assert evidence["optimizer_instance_count"] == 0
    assert evidence["optimizer_update_calls"] == []
    assert evidence["optimizer_update_call_count"] == 0
    assert result["qualification"]["passed"] is False


@requires_gate_c
def test_gate_c2_controls_writer_preserves_destination_on_stream_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "gate-c2-controls.json"
    temporary = destination.with_suffix(".json.tmp")
    previous = b"authenticated-previous-artifact\n"
    destination.write_bytes(previous)

    nonfinite = _gate_c2_controls_writer_value()
    nonfinite["a"]["finite"] = math.nan
    with pytest.raises(ValueError, match="JSON|range|compliant"):
        gate_c.write_artifact(nonfinite, destination)
    assert destination.read_bytes() == previous
    assert not temporary.exists()

    def failed_fsync(file_descriptor: int) -> None:
        del file_descriptor
        raise OSError("simulated fsync failure")

    with monkeypatch.context() as context:
        context.setattr(gate_c.os, "fsync", failed_fsync)
        with pytest.raises(OSError, match="fsync"):
            gate_c.write_artifact(_gate_c2_controls_writer_value(), destination)
    assert destination.read_bytes() == previous
    assert not temporary.exists()

    original_replace = Path.replace

    def failed_replace(path: Path, target: Path) -> Path:
        if path == temporary:
            raise OSError("simulated atomic replace failure")
        return original_replace(path, target)

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", failed_replace)
        with pytest.raises(OSError, match="replace"):
            gate_c.write_artifact(_gate_c2_controls_writer_value(), destination)
    assert destination.read_bytes() == previous
    assert not temporary.exists()


@requires_gate_c
def test_gate_c2_evidence_helpers_reject_invalid_and_colluding_records(
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
) -> None:
    with pytest.raises(ValueError, match="batch"):
        gate_c._gate_c2_array_endpoint(np.asarray(1.0, dtype=np.float32))
    with pytest.raises(ValueError, match="nonnegative"):
        gate_c._gate_c2_floating_difference_record(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            rms_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="geometry"):
        gate_c._gate_c2_floating_difference_record(
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            rms_tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="finite"):
        gate_c._gate_c2_floating_difference_record(
            np.asarray([[math.nan]], dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            rms_tolerance=1e-6,
        )
    assert not gate_c._gate_c2_endpoint_complete(
        None,
        expected_dtype="<f4",
        expected_shape=(1, 1),
    )

    difference = gate_c._gate_c2_floating_difference_record(
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        rms_tolerance=1e-6,
    )
    mutations = (
        ("per_example_sum_squared_difference", [0.0]),
        ("per_example_compared_value_count", [1, 1]),
        ("per_example_sum_squared_difference", [-1.0, 0.0]),
        ("per_example_rms_difference", [1.0, 0.0]),
        ("per_example_max_abs_difference", [-1.0, 0.0]),
    )
    for field, replacement in mutations:
        malformed = copy.deepcopy(difference)
        malformed[field] = replacement
        assert not gate_c._gate_c2_floating_difference_record_complete(
            malformed,
            expected_dtype="<f4",
            expected_shape=(2, 2),
            rms_tolerance=1e-6,
        )
    assert not gate_c._gate_c2_floating_difference_record_complete(
        difference,
        expected_dtype="<f4",
        expected_shape=(2, 2),
        rms_tolerance=object(),
    )

    with pytest.raises(ValueError, match="vector"):
        gate_c._gate_c2_prediction_endpoint(np.zeros((1, 1), dtype=np.int32))
    with pytest.raises(ValueError, match="0..9"):
        gate_c._gate_c2_prediction_endpoint(np.asarray([10], dtype=np.int32))
    with pytest.raises(ValueError, match="equal vectors"):
        gate_c._gate_c2_prediction_difference_record(
            np.asarray([0], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
        )
    predictions = gate_c._gate_c2_prediction_difference_record(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )
    assert not gate_c._gate_c2_prediction_difference_record_complete(
        None,
        count=2,
    )
    malformed_predictions = copy.deepcopy(predictions)
    malformed_predictions["per_example_hamming_count"] = [0]
    assert not gate_c._gate_c2_prediction_difference_record_complete(
        malformed_predictions,
        count=2,
    )

    with pytest.raises(ValueError, match="nonempty floating"):
        gate_c._gate_c2_zero_array_record(np.asarray([], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        gate_c._gate_c2_zero_array_record(
            np.asarray([math.inf], dtype=np.float32)
        )
    zero = gate_c._gate_c2_zero_array_record(np.zeros((1, 1), dtype=np.float32))
    zero["sum_of_squares"] = object()
    assert not gate_c._gate_c2_zero_array_record_complete(
        zero,
        expected_dtype="<f4",
        expected_shape=(1, 1),
    )

    admission, _ = passing_gate_c2_no_read_reports
    evidence = _gate_c2_control_execution_evidence(admission)
    assert not gate_c._gate_c2_no_update_evidence_complete(None, admission)
    assert not gate_c._gate_c2_no_update_evidence_complete(evidence, {})


@requires_gate_c
def test_gate_c2_hidden_digest_capture_and_comparison_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        config = SimpleNamespace(color_rank=1)

        def __init__(self) -> None:
            self.restored: list[Any] = []

        def states(self, state_type: Any) -> dict[tuple[str, ...], Any]:
            assert state_type is brainstate.HiddenState
            return {
                ("z",): SimpleNamespace(
                    value=(
                        np.zeros((2, 2), dtype=np.float32),
                        np.ones((2, 1), dtype=np.float32),
                    )
                ),
                ("a",): SimpleNamespace(
                    value=np.full((2, 1), 2.0, dtype=np.float32)
                ),
            }

        def restore_state(self, snapshot: Any) -> None:
            self.restored.append(snapshot)

    model = FakeModel()
    first_digest = gate_c._hidden_state_sha256(model)
    second_digest = gate_c._hidden_state_sha256(model)
    assert first_digest == second_digest
    assert gate_c._sha256_complete(first_digest)
    hidden = gate_c._gate_c2_hidden_arrays(model)
    assert tuple(hidden) == ("a#0", "z#0", "z#1")

    monkeypatch.setattr(
        gate_c.legacy,
        "_color_logits",
        lambda compact, color_rank: compact[:, :10] + color_rank,
    )
    compact = np.zeros((2, 10), dtype=np.float32)
    compact[:, 3] = 1.0
    capture = gate_c._gate_c2_h0_capture(
        model,
        lambda events, advances: jnp.asarray(compact)
        + jnp.asarray(events).sum() * 0.0
        + jnp.asarray(advances).sum() * 0.0,
        "snapshot",
        np.zeros((1, 2, 1), dtype=np.float32),
        np.ones((1, 2), dtype=np.bool_),
    )
    assert model.restored == ["snapshot"]
    assert capture["predictions"].tolist() == [3, 3]
    comparison = gate_c._gate_c2_h0_comparison(capture, copy.deepcopy(capture))
    assert comparison["passed"] is True

    wrong = copy.deepcopy(capture)
    wrong["hidden_paths"].pop("z#1")
    with pytest.raises(ValueError, match="paths"):
        gate_c._gate_c2_h0_comparison(capture, wrong)


@requires_gate_c
def test_gate_c2_operational_h0_report_runs_all_replay_controls_without_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_gate_c_config()
    regime_config = config.gate_a_config
    count = regime_config.validation_episodes
    length = regime_config.sequence_length
    validation = np.zeros((length, count, 1), dtype=np.float32)
    data = legacy.BindingData(
        training_events=np.zeros((1, length, 1, 1), dtype=np.float32),
        training_targets=np.zeros((1, 1), dtype=np.int32),
        training_mapping_ids=np.zeros((1, 1), dtype=np.int64),
        training_query_indices=np.zeros((1, 1), dtype=np.int32),
        validation_intact=validation,
        validation_shuffled=np.array(validation, copy=True),
        validation_no_context=np.array(validation, copy=True),
        validation_targets=np.zeros((count,), dtype=np.int32),
        validation_mapping_ids=np.zeros((count,), dtype=np.int64),
        validation_query_indices=np.zeros((count,), dtype=np.int32),
    )
    finished: list[str] = []
    captures: list[tuple[str, int]] = []

    class FakeModel:
        def __init__(self, role: str) -> None:
            self.role = role
            self.restored: list[Any] = []

        def snapshot_state(self) -> str:
            return "initial-snapshot"

        def restore_state(self, snapshot: Any) -> None:
            self.restored.append(snapshot)

    def control_model(*args: Any, role: str, **kwargs: Any) -> FakeModel:
        del args, kwargs
        return FakeModel(role)

    def capture(
        model: FakeModel,
        driver: Any,
        snapshot: Any,
        events: np.ndarray,
        advances: np.ndarray,
    ) -> dict[str, Any]:
        assert driver == model.role
        assert snapshot == "initial-snapshot"
        assert events.shape[0] == advances.shape[0]
        captures.append((model.role, int(events.shape[0])))
        return {"role": model.role, "length": int(events.shape[0])}

    monkeypatch.setattr(gate_c, "_gate_c2_control_model", control_model)
    monkeypatch.setattr(gate_c.legacy, "_copy_parameters", lambda *args: None)
    monkeypatch.setattr(
        gate_c.legacy,
        "_parameter_values",
        lambda model: {"weight": np.ones((1,), dtype=np.float32)},
    )
    monkeypatch.setattr(gate_c, "_gate_c2_h0_driver", lambda model: model.role)
    monkeypatch.setattr(gate_c, "_gate_c2_h0_capture", capture)
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_h0_comparison",
        lambda left, right: {"left": left, "right": right, "passed": True},
    )
    monkeypatch.setattr(gate_c, "_hidden_state_sha256", lambda model: "a" * 64)
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_finish_control_model",
        lambda role, model: finished.append(role),
    )

    report = gate_c._paired_h0_operational_equivalence_report(
        config,
        initialization={"initialization": {}},
        regime="gate_a",
        data=data,
    )

    assert report["passed"] is True
    assert tuple(report["streams"]) == ("intact", "shuffled", "no_context")
    assert all(stream["passed"] is True for stream in report["streams"].values())
    assert len(captures) == 12
    assert finished == [
        "gate_a:paired_h0:full_reference",
        "gate_a:paired_h0:copied_full",
        "gate_a:paired_h0:query_only",
    ]
    with pytest.raises(TypeError, match="BindingData"):
        gate_c._paired_h0_operational_equivalence_report(
            config,
            initialization={},
            regime="gate_a",
            data=object(),
        )
    with pytest.raises(TypeError, match="DepthValidationData"):
        gate_c._paired_h0_operational_equivalence_report(
            config,
            initialization={},
            regime="gate_b",
            data=object(),
        )


def _install_gate_c2_runner_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    schedules_match: bool = True,
) -> tuple[
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    config = gate_c.GateCConfig()
    initialization = {"initialization": {}}
    prerequisites = {
        "gate_a": dict(_GATE_A_REFERENCE),
        "gate_b": dict(_GATE_B_REFERENCE),
        "gate_c_initialization": {"authenticated": True},
    }
    source_start = _passing_gate_c_source()
    source_files = _current_gate_c_source_files()
    environment = _passing_gate_c_environment()
    schedules = {
        "gate_a": {"schedule": "gate-a"},
        "gate_b": {"schedule": "gate-b"},
    }
    expected = copy.deepcopy(schedules)
    if not schedules_match:
        expected["gate_b"]["schedule"] = "different"
    monkeypatch.setattr(
        gate_c,
        "_validated_gate_c_initialization_admission",
        lambda *args, **kwargs: initialization,
    )
    monkeypatch.setattr(gate_c, "_regenerate_gate_a_data", lambda config: object())
    monkeypatch.setattr(gate_c, "_regenerate_gate_b_data", lambda config: object())
    monkeypatch.setattr(
        gate_c,
        "_actual_schedule_identity_report",
        lambda *args: copy.deepcopy(schedules),
    )
    monkeypatch.setattr(
        gate_c,
        "_schedule_identity_report",
        lambda config: copy.deepcopy(expected),
    )
    return config, prerequisites, source_start, source_files, environment


@requires_gate_c
def test_gate_c2_controls_runner_rejects_invalid_contract_inputs() -> None:
    kwargs = {
        "prerequisites": {},
        "source_start": {},
        "source_end_reporter": lambda: {},
        "source_files": {},
        "environment": {},
    }
    with pytest.raises(TypeError, match="GateCConfig"):
        gate_c.run_gate_c2_controls(None, **kwargs)
    with pytest.raises(ValueError, match="prerequisites"):
        gate_c.run_gate_c2_controls(gate_c.GateCConfig(), **kwargs)
    kwargs["prerequisites"] = {
        "gate_a": {},
        "gate_b": {},
        "gate_c_initialization": {},
    }
    kwargs["source_end_reporter"] = None
    with pytest.raises(TypeError, match="callable"):
        gate_c.run_gate_c2_controls(gate_c.GateCConfig(), **kwargs)


@requires_gate_c
def test_gate_c2_controls_runner_restores_audit_when_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, prerequisites, source_start, source_files, environment = (
        _install_gate_c2_runner_stubs(monkeypatch)
    )
    original_entries = (
        gate_c._make_arm_trainer,
        gate_a._make_pp_prop_trainer,
        gate_b._make_pp_prop_trainer,
        gate_c.GateCTrainer,
        gate_a._PPPropTrainer,
        gate_b._DepthPPPropTrainer,
        gate_c.braintools.optim.Adam,
    )

    def failed_probe(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("simulated control probe failure")

    monkeypatch.setattr(
        gate_c,
        "_paired_h0_operational_equivalence_report",
        failed_probe,
    )
    assert gate_c._ACTIVE_GATE_C2_CONTROLS_AUDIT is None
    with pytest.raises(RuntimeError, match="probe failure"):
        gate_c.run_gate_c2_controls(
            config,
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=lambda: source_start,
            source_files=source_files,
            environment=environment,
        )
    assert gate_c._ACTIVE_GATE_C2_CONTROLS_AUDIT is None
    assert original_entries == (
        gate_c._make_arm_trainer,
        gate_a._make_pp_prop_trainer,
        gate_b._make_pp_prop_trainer,
        gate_c.GateCTrainer,
        gate_a._PPPropTrainer,
        gate_b._DepthPPPropTrainer,
        gate_c.braintools.optim.Adam,
    )


@requires_gate_c
def test_gate_c2_controls_runner_rejects_schedule_active_audit_and_source_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, prerequisites, source_start, source_files, environment = (
        _install_gate_c2_runner_stubs(monkeypatch, schedules_match=False)
    )
    kwargs = {
        "prerequisites": prerequisites,
        "source_start": source_start,
        "source_end_reporter": lambda: source_start,
        "source_files": source_files,
        "environment": environment,
    }
    with pytest.raises(RuntimeError, match="schedules"):
        gate_c.run_gate_c2_controls(config, **kwargs)

    config, prerequisites, source_start, source_files, environment = (
        _install_gate_c2_runner_stubs(monkeypatch)
    )
    kwargs = {
        "prerequisites": prerequisites,
        "source_start": source_start,
        "source_end_reporter": lambda: source_start,
        "source_files": source_files,
        "environment": environment,
    }
    monkeypatch.setattr(gate_c, "_ACTIVE_GATE_C2_CONTROLS_AUDIT", object())
    with pytest.raises(RuntimeError, match="already active"):
        gate_c.run_gate_c2_controls(config, **kwargs)
    monkeypatch.setattr(gate_c, "_ACTIVE_GATE_C2_CONTROLS_AUDIT", None)
    monkeypatch.setattr(
        gate_c,
        "_paired_h0_operational_equivalence_report",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        gate_c,
        "_query_only_latent_no_read_report",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        gate_c,
        "_mechanism_oracle",
        lambda *args, **kwargs: {"complete": True},
    )
    kwargs["source_end_reporter"] = lambda: None
    with pytest.raises(TypeError, match="mapping"):
        gate_c.run_gate_c2_controls(config, **kwargs)
    assert gate_c._ACTIVE_GATE_C2_CONTROLS_AUDIT is None


@pytest.mark.parametrize(
    "mutation",
    (
        "perturbations_schema",
        "perturbation_stream_schema",
        "perturbation_tick_digest",
        "positive_schema",
        "positive_replacement_schema",
        "positive_stream_schema",
        "positive_tick_digest",
        "positive_stored_pass",
    ),
)
@requires_gate_c
def test_gate_c2_no_read_validator_rejects_top_level_and_family_schema_damage(
    mutation: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    report = reports["gate_a"]
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_removed_path_influence_complete",
        lambda *args, **kwargs: True,
    )
    tick_name = gate_c.GATE_C2_LATENT_TICKS["gate_a"][0]
    perturbation = report["perturbations"]["plus_7"]
    positive = report["full_positive_control"]
    container: dict[str, Any]
    key: str
    if mutation == "perturbations_schema":
        container, key, replacement = report, "perturbations", None
    elif mutation == "perturbation_stream_schema":
        container, key, replacement = perturbation["streams"], "intact", None
    elif mutation == "perturbation_tick_digest":
        container = perturbation["streams"]["intact"][tick_name]
        key, replacement = "replacement_s_k_sha256", "0" * 64
    elif mutation == "positive_schema":
        container, key, replacement = report, "full_positive_control", None
    elif mutation == "positive_replacement_schema":
        container, key, replacement = positive, "plus_7", None
    elif mutation == "positive_stream_schema":
        container = positive["plus_7"]["streams"]
        key, replacement = "intact", None
    elif mutation == "positive_tick_digest":
        container = positive["plus_7"]["streams"]["intact"][tick_name]
        key, replacement = "replacement_s_k_sha256", "0" * 64
    elif mutation == "positive_stored_pass":
        container, key, replacement = positive, "passed", False
    else:
        raise AssertionError(f"unhandled no-read schema mutation {mutation}")
    original = container[key]
    container[key] = replacement
    try:
        assert not gate_c._gate_c2_query_only_latent_no_read_complete(
            report,
            admission,
            regime="gate_a",
        )
    finally:
        container[key] = original


@requires_gate_c
def test_gate_c2_no_read_validator_rejects_nonmapping_and_invalid_canonical(
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
) -> None:
    admission, reports = passing_gate_c2_no_read_reports
    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        reports["gate_a"],
        None,
        regime="gate_a",
    )
    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        None,
        admission,
        regime="gate_a",
    )
    malformed = copy.deepcopy(admission)
    malformed["initialization"]["gate_a"]["canonical_full"][
        "parameter_sha256"
    ] = "0" * 64
    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        reports["gate_a"],
        malformed,
        regime="gate_a",
    )
    assert not gate_c._gate_c2_query_only_latent_no_read_complete(
        reports["gate_a"],
        admission,
        regime="invalid",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "top_schema",
        "gradient_chunk",
        "objective_schema",
        "coordinate",
        "source_schema",
        "continuation_schema",
        "global_schema",
        "global_semantics",
        "live_schema",
        "removed_schema",
        "removed_record_schema",
        "removed_leaf_schema",
        "numeric_exception",
    ),
)
@requires_gate_c
def test_gate_c2_removed_path_validator_rejects_structural_failures(
    mutation: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
) -> None:
    regime = "gate_a"
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    report: Any = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    objective = report["objectives"]["gate_a_h1"]
    if mutation == "top_schema":
        report = None
    elif mutation == "gradient_chunk":
        report["gradient_chunk_size"] = 0
    elif mutation == "objective_schema":
        report["objectives"]["gate_a_h1"] = None
    elif mutation == "coordinate":
        objective["checkpoint"] = 8
    elif mutation == "source_schema":
        objective["source_contract"] = None
    elif mutation == "continuation_schema":
        objective["continuation"] = None
    elif mutation == "global_schema":
        report["global"] = None
    elif mutation == "global_semantics":
        report["global"]["sha256"] = "0" * 64
    elif mutation == "live_schema":
        report["live_paths"] = None
    elif mutation == "removed_schema":
        report["removed_paths"] = None
    elif mutation == "removed_record_schema":
        path = gate_c.GATE_C2_REMOVED_PATHS[0]
        report["removed_paths"][path] = None
    elif mutation == "removed_leaf_schema":
        path = gate_c.GATE_C2_REMOVED_PATHS[0]
        report["removed_paths"][path]["leaves"][0] = None
    elif mutation == "numeric_exception":
        report["global"]["l2_norm"] = object()
    else:
        raise AssertionError(f"unhandled removed-path structure {mutation}")
    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@pytest.mark.parametrize(
    "mutation",
    ("source_array_contract", "continuation_contract", "gate_a_weight_contract"),
)
@requires_gate_c
def test_gate_c2_removed_path_validator_cross_checks_preregistered_constants(
    mutation: str,
    reduced_gate_c2_removed_path_reports: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regime = "gate_a"
    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime=regime,
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ][regime]["canonical_full"]["parameter_sha256"]
    contracts = copy.deepcopy(gate_c.GATE_C2_REMOVED_PATH_OBJECTIVES)
    expected = contracts["gate_a_h1"]
    if mutation == "source_array_contract":
        dtype, shape, _ = expected["source_arrays"]["events"]
        expected["source_arrays"]["events"] = (dtype, shape, "0" * 64)
    elif mutation == "continuation_contract":
        expected["continuation"]["source_events_sha256"] = "0" * 64
    elif mutation == "gate_a_weight_contract":
        expected["continuation"]["base_checkpoint_weights"] = [1.0]
    else:
        raise AssertionError(f"unhandled removed-path contract {mutation}")
    monkeypatch.setattr(gate_c, "GATE_C2_REMOVED_PATH_OBJECTIVES", contracts)
    assert not gate_c._gate_c2_removed_path_influence_complete(
        report,
        regime=regime,
        canonical_parameter_sha256=canonical,
    )


@requires_gate_c
def test_gate_c2_h0_raw_evidence_rejects_malformed_paths_geometry_and_schema(
    reduced_gate_c2_removed_path_reports: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="path"):
        gate_c._gate_c2_hidden_leaf_path_key("missing-index")
    with pytest.raises(ValueError, match="digest"):
        gate_c._gate_c2_raw_h0_snapshot_record(
            SimpleNamespace(entries=()),
            parameter_sha256="0" * 64,
        )
    with monkeypatch.context() as context:
        context.setattr(gate_c, "_gate_c2_snapshot_arrays", lambda snapshot: {})
        with pytest.raises(ValueError, match="paths"):
            gate_c._gate_c2_raw_h0_snapshot_record(
                object(),
                parameter_sha256="a" * 64,
            )
    wrong_geometry = {
        path: np.zeros(shape, dtype=np.float32)
        for path, shape in gate_c._GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY.items()
    }
    first_path = next(iter(wrong_geometry))
    wrong_geometry[first_path] = wrong_geometry[first_path].astype(np.float64)
    with monkeypatch.context() as context:
        context.setattr(
            gate_c,
            "_gate_c2_snapshot_arrays",
            lambda snapshot: wrong_geometry,
        )
        with pytest.raises(ValueError, match="geometry"):
            gate_c._gate_c2_raw_h0_snapshot_record(
                object(),
                parameter_sha256="a" * 64,
            )

    report = _gate_c2_validator_ready_removed_path_report(
        reduced_gate_c2_removed_path_reports,
        regime="gate_a",
    )
    canonical = reduced_gate_c2_removed_path_reports["initialization"][
        "initialization"
    ]["gate_a"]["canonical_full"]["parameter_sha256"]
    boundary = report["objectives"]["gate_a_h1"]["continuation"][
        "h0_gradient_boundary"
    ]
    assert not gate_c._gate_c2_raw_h0_snapshot_complete(
        None,
        canonical_parameter_sha256=canonical,
    )
    malformed_snapshot = copy.deepcopy(boundary["actual_gradient_start"])
    malformed_snapshot["hidden_paths"][first_path] = None
    assert not gate_c._gate_c2_raw_h0_snapshot_complete(
        malformed_snapshot,
        canonical_parameter_sha256=canonical,
    )
    assert not gate_c._gate_c2_h0_gradient_boundary_complete(
        None,
        canonical_parameter_sha256=canonical,
    )
    assert not gate_c._gate_c2_h0_gradient_boundary_complete(
        boundary,
        canonical_parameter_sha256="0" * 64,
    )


class _GateC2RaisingMapping(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)


class _GateC2ExecutionRaisingMapping(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key == "execution_and_update_evidence":
            raise KeyError(key)
        return super().__getitem__(key)


@pytest.mark.parametrize(
    ("criterion", "mutation"),
    (
        ("schema_and_control", "schema"),
        ("exact_configuration", "configuration"),
        ("prerequisites_authenticated", "prerequisites"),
        ("initialization_authenticated", "initialization"),
        ("canonical_schedules_complete", "schedule"),
        ("no_behavioral_or_optimizer_updates", "execution"),
        ("paired_h0_operational_equivalence", "h0"),
        ("mechanism_oracle_complete", "oracle"),
        ("source_and_gpu_authenticated", "source"),
    ),
)
@requires_gate_c
def test_gate_c2_controls_qualifier_fails_closed_on_internal_evidence_errors(
    criterion: str,
    mutation: str,
    passing_gate_c2_no_read_reports: tuple[
        dict[str, Any],
        dict[str, dict[str, Any]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, no_read_reports = passing_gate_c2_no_read_reports
    report = _passing_gate_c2_controls_report(admission, no_read_reports)
    report = dict(report)
    monkeypatch.setattr(
        gate_c,
        "_validated_gate_c_initialization_admission",
        lambda *args, **kwargs: admission,
    )
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_no_update_evidence_complete",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_paired_h0_operational_equivalence_complete",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_query_only_latent_no_read_complete",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(gate_c, "_mechanism_oracle_complete", lambda *args: True)
    monkeypatch.setattr(gate_c, "_source_and_gpu_complete", lambda *args: True)
    monkeypatch.setattr(gate_c, "_source_files_complete", lambda *args: True)
    if mutation == "schema":
        report.pop("schema_version")
    elif mutation == "configuration":
        report["regimes"] = _GateC2RaisingMapping(report["regimes"])
    elif mutation == "prerequisites":
        report["prerequisites"] = _GateC2RaisingMapping(report["prerequisites"])
    elif mutation == "initialization":
        monkeypatch.setattr(
            gate_c,
            "_validated_gate_c_initialization_admission",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid")),
        )
    elif mutation == "schedule":
        report["regimes"] = _GateC2RaisingMapping(report["regimes"])
    elif mutation == "execution":
        report["environment"] = _GateC2ExecutionRaisingMapping(
            report["environment"]
        )
    elif mutation == "h0":
        regimes = dict(report["regimes"])
        gate_a_regime = dict(regimes["gate_a"])
        gate_a_regime.pop("paired_h0_operational_equivalence")
        regimes["gate_a"] = gate_a_regime
        report["regimes"] = regimes
    elif mutation == "oracle":
        report.pop("mechanism_oracle")
    elif mutation == "source":
        report.pop("source_start")
    else:
        raise AssertionError(f"unhandled qualifier error mutation {mutation}")
    qualification = gate_c._gate_c2_controls_qualification(
        report,
        config=gate_c.GateCConfig(),
    )
    assert qualification["criteria"][criterion] is False
    assert qualification["passed"] is False
    assert qualification["interpretation"] == (
        "gate_c2_pretraining_controls_failed_stop"
    )


@requires_gate_c
def test_gate_c2_controls_qualifier_rejects_nonmapping_inputs() -> None:
    qualification = gate_c._gate_c2_controls_qualification(
        None,
        config=gate_c.GateCConfig(),
    )
    assert qualification == {
        "criteria": {name: False for name in _GATE_C2_CONTROLS_CRITERIA},
        "passed": False,
        "interpretation": "gate_c2_pretraining_controls_failed_stop",
    }


@requires_gate_c
def test_gate_c2_controls_main_runs_fixed_host_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    head = "c" * 40
    source = _passing_gate_c_source()
    source["commit"] = head
    environment = _passing_gate_c_environment()
    repo_root = Path(gate_c.__file__).resolve().parents[2]
    config = launcher.LaunchConfig(
        target="gate_c2_controls",
        repo_root=repo_root,
        output_dir=repo_root / "var" / "example21-causal-gate",
    )
    gate_a_paths = launcher._gate_a_artifact_paths(config)
    gate_b_paths = launcher._formal_gate_b_artifact_paths(config)
    initialization = launcher.target_paths(config, head, "gate_c_init")
    output = launcher.target_paths(config, head, "gate_c2_controls")
    argv = [
        "--target",
        "gate_c2_controls",
        "--gate-a-result",
        str(gate_a_paths.result),
        "--gate-a-manifest",
        str(gate_a_paths.manifest),
        "--gate-b-manifest",
        str(gate_b_paths.manifest),
        "--gate-c-init-manifest",
        str(initialization.manifest),
        "--output",
        str(output.result),
    ]
    prerequisites = {"authenticated": True}
    result = {
        "qualification": {
            "passed": False,
            "interpretation": "gate_c2_pretraining_controls_failed_stop",
        }
    }
    events: list[str] = []
    monkeypatch.setattr(gate_a, "_source_report", lambda: copy.deepcopy(source))
    monkeypatch.setattr(
        gate_a,
        "_environment_report",
        lambda: copy.deepcopy(environment),
    )
    monkeypatch.setattr(
        gate_a,
        "_require_authenticated_gpu_launch",
        lambda *args: events.append("gpu"),
    )
    monkeypatch.setattr(
        launcher,
        "_load_gate_c2_controls_prerequisites",
        lambda *args, **kwargs: prerequisites,
    )
    monkeypatch.setattr(
        launcher,
        "_load_formal_gate_c_prerequisites",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal loader must not run")
        ),
    )
    monkeypatch.setattr(gate_c, "_source_files_report", lambda: {"source": "hash"})

    def run_controls(actual_config: Any, **kwargs: Any) -> dict[str, Any]:
        assert actual_config == gate_c.GateCConfig()
        assert kwargs["prerequisites"] is prerequisites
        assert kwargs["source_start"] == source
        assert kwargs["environment"] == environment
        assert kwargs["source_files"] == {"source": "hash"}
        events.append("run")
        return result

    monkeypatch.setattr(gate_c, "run_gate_c2_controls", run_controls)
    monkeypatch.setattr(
        gate_c,
        "write_artifact",
        lambda value, path: Path(path),
    )

    assert gate_c.main(argv) == 0
    assert events == ["gpu", "run"]
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0] == str(output.result)
    assert json.loads(output_lines[1]) == result["qualification"]


@pytest.mark.parametrize("regime", _REGIMES)
@requires_gate_c
def test_gate_c2_operational_h0_restores_independent_live_state_leaves(
    regime: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_gate_c_config()
    data = (
        gate_c._regenerate_gate_a_data(config)
        if regime == "gate_a"
        else gate_c._regenerate_gate_b_data(config)
    )
    models: dict[str, LatentWorkspaceModel] = {}
    source_snapshot: dict[str, Any] = {}
    captured: list[tuple[str, dict[str, np.ndarray]]] = []
    original_snapshot_state = LatentWorkspaceModel.snapshot_state

    def recording_snapshot(model: LatentWorkspaceModel) -> Any:
        snapshot = original_snapshot_state(model)
        source_snapshot.setdefault("value", snapshot)
        return snapshot

    def control_model(
        actual_config: Any,
        initialization: Mapping[str, Any],
        *,
        policy: str,
        batch_size: int,
        role: str,
        **kwargs: Any,
    ) -> LatentWorkspaceModel:
        del initialization, kwargs
        arm = "query_only" if policy == "query_only" else "full"
        model = gate_c._new_model_for_arm(
            actual_config,
            regime,
            arm,
            batch_size=batch_size,
        )
        models[role] = model
        return model

    def live_arrays(model: LatentWorkspaceModel) -> dict[str, np.ndarray]:
        return {
            f"{gate_a._path(path)}#{index}": np.asarray(
                u.get_mantissa(jax.device_get(leaf))
            )
            for path, state in model.states(brainstate.HiddenState).items()
            for index, leaf in enumerate(jax.tree.leaves(state.value))
        }

    def snapshot_arrays(snapshot: Any) -> dict[str, np.ndarray]:
        return {
            f"{gate_a._path(path)}#{index}": np.asarray(
                u.get_mantissa(jax.device_get(leaf))
            )
            for path, value in snapshot.entries
            for index, leaf in enumerate(jax.tree.leaves(value))
        }

    def capture(
        model: LatentWorkspaceModel,
        driver: str,
        snapshot: Any,
        events: np.ndarray,
        advances: np.ndarray,
    ) -> dict[str, Any]:
        del events, advances
        model.restore_state(snapshot)
        current = live_arrays(model)
        source = snapshot_arrays(snapshot)
        assert tuple(current) == tuple(source)
        for path in current:
            np.testing.assert_array_equal(current[path], source[path])
            assert current[path] is not source[path]
            assert not np.shares_memory(current[path], source[path])
            for _, earlier in captured:
                assert current[path] is not earlier[path]
                assert not np.shares_memory(current[path], earlier[path])
        captured.append((driver, current))
        return {"role": driver}

    monkeypatch.setattr(LatentWorkspaceModel, "snapshot_state", recording_snapshot)
    monkeypatch.setattr(gate_c, "_gate_c2_control_model", control_model)
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_h0_driver",
        lambda model: next(role for role, value in models.items() if value is model),
    )
    monkeypatch.setattr(gate_c, "_gate_c2_h0_capture", capture)
    monkeypatch.setattr(
        gate_c,
        "_gate_c2_h0_comparison",
        lambda left, right: {"left": left, "right": right, "passed": True},
    )
    monkeypatch.setattr(gate_c, "_hidden_state_sha256", lambda model: "a" * 64)
    monkeypatch.setattr(gate_c, "_gate_c2_finish_control_model", lambda *args: None)

    report = gate_c._paired_h0_operational_equivalence_report(
        config,
        initialization={"initialization": {}},
        regime=regime,
        data=data,
    )

    assert report["passed"] is True
    assert len(captured) == 12
    assert source_snapshot
    immutable_source = {
        path: np.array(value, copy=True)
        for path, value in snapshot_arrays(source_snapshot["value"]).items()
    }
    other_before = {
        role: {
            path: np.array(value, copy=True)
            for path, value in live_arrays(model).items()
        }
        for role, model in models.items()
    }
    mutated_role = f"{regime}:paired_h0:copied_full"
    mutated = models[mutated_role]
    first_state = next(iter(mutated.states(brainstate.HiddenState).values()))
    first_state.value = jax.tree.map(
        lambda leaf: leaf + jnp.ones_like(leaf),
        first_state.value,
    )
    assert any(
        not np.array_equal(live_arrays(mutated)[path], value)
        for path, value in other_before[mutated_role].items()
    )
    for role, model in models.items():
        if role == mutated_role:
            continue
        for path, value in other_before[role].items():
            np.testing.assert_array_equal(live_arrays(model)[path], value)
    for path, value in immutable_source.items():
        np.testing.assert_array_equal(snapshot_arrays(source_snapshot["value"])[path], value)
