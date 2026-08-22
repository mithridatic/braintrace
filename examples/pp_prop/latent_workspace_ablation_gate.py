"""Preregistered pp-prop mechanism ablations for Example 21 Gate C."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._testing.oracle import chunked_online_param_gradients

from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop import latent_workspace_binding_gate as gate_a
from examples.pp_prop import latent_workspace_depth_gate as gate_b
from examples.pp_prop.latent_workspace_model import (
    LatentWorkspaceModel,
    ModelConfig,
    compile_pp_prop,
)


GATE_C_SCHEMA_VERSION = 1
GATE_C_INITIALIZATION_CONTROL = "example21_gate_c_initialization_admission"
GATE_C_CONTROL = "example21_pp_prop_learnability_gate_c"

# Gate C v1 is retained above without reinterpretation.  Every Gate C2 identity
# has a distinct name so loading a schema-1 formal artifact cannot silently use
# the amended protocol.
GATE_C2_SCHEMA_VERSION = 2
GATE_C2_CONTROL = "example21_pp_prop_learnability_gate_c2"
GATE_C2_QUALIFICATION_REGIME = "preregistered_gate_c2_full"
GATE_C2_PASSING_INTERPRETATION = (
    "gate_c2_passed_pp_prop_learnability_mechanism"
)
GATE_C2_FAILING_INTERPRETATION = (
    "gate_c2_failed_stop_no_causal_mechanism_conclusion"
)
GATE_C2_TOP_LEVEL_KEYS = (
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
GATE_C2_QUALIFICATION_CRITERIA = (
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

GATE_C2_CONTROLS_SCHEMA_VERSION = 1
GATE_C2_CONTROLS_CONTROL = "example21_gate_c2_pretraining_control_admission"
GATE_C2_CONTROLS_QUALIFICATION_REGIME = (
    "preregistered_gate_c2_pretraining_controls"
)
GATE_C2_CONTROLS_MAX_JSON_BYTES = 201_326_592
GATE_C2_CONTROLS_TOP_LEVEL_KEYS = (
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
GATE_C2_CONTROLS_QUALIFICATION_CRITERIA = (
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
GATE_C2_CONTROLS_AUDIT_LABELS = (
    "braintools.optim.Adam.__init__",
    "braintools.optim.Adam.update",
    "examples.pp_prop.latent_workspace_ablation_gate.GateCTrainer.train_chunk",
    "examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer",
    "examples.pp_prop.latent_workspace_binding_gate._PPPropTrainer.train",
    "examples.pp_prop.latent_workspace_binding_gate._make_pp_prop_trainer",
    "examples.pp_prop.latent_workspace_depth_gate._DepthPPPropTrainer.train_chunk",
    "examples.pp_prop.latent_workspace_depth_gate._make_pp_prop_trainer",
)
GATE_C2_CONTROLS_MODEL_ROLES = {
    "gate_a:paired_h0:copied_full": {
        "regime": "gate_a",
        "probe": "paired_h0_operational_equivalence",
        "policy": "full",
    },
    "gate_a:paired_h0:full_reference": {
        "regime": "gate_a",
        "probe": "paired_h0_operational_equivalence",
        "policy": "full",
    },
    "gate_a:paired_h0:query_only": {
        "regime": "gate_a",
        "probe": "paired_h0_operational_equivalence",
        "policy": "query_only",
    },
    "gate_a:query_only_latent_no_read:full_positive_control": {
        "regime": "gate_a",
        "probe": "full_positive_control",
        "policy": "full",
    },
    "gate_a:query_only_latent_no_read:query_only": {
        "regime": "gate_a",
        "probe": "query_only_latent_no_read",
        "policy": "query_only",
    },
    "gate_a:removed_path_finite_window:gate_a_h1": {
        "regime": "gate_a",
        "probe": "removed_path_finite_window_influence",
        "policy": "query_only",
    },
    "gate_b:mechanism_oracle:full:finite_window": {
        "regime": "gate_b",
        "probe": "mechanism_oracle:full:finite_window",
        "policy": "full",
    },
    "gate_b:mechanism_oracle:full:reference": {
        "regime": "gate_b",
        "probe": "mechanism_oracle:full:reference",
        "policy": "full",
    },
    "gate_b:mechanism_oracle:query_only:finite_window": {
        "regime": "gate_b",
        "probe": "mechanism_oracle:query_only:finite_window",
        "policy": "query_only",
    },
    "gate_b:mechanism_oracle:query_only:reference": {
        "regime": "gate_b",
        "probe": "mechanism_oracle:query_only:reference",
        "policy": "query_only",
    },
    "gate_b:mechanism_oracle:terminal_only:finite_window": {
        "regime": "gate_b",
        "probe": "mechanism_oracle:terminal_only:finite_window",
        "policy": "full",
    },
    "gate_b:mechanism_oracle:terminal_only:reference": {
        "regime": "gate_b",
        "probe": "mechanism_oracle:terminal_only:reference",
        "policy": "full",
    },
    "gate_b:paired_h0:copied_full": {
        "regime": "gate_b",
        "probe": "paired_h0_operational_equivalence",
        "policy": "full",
    },
    "gate_b:paired_h0:full_reference": {
        "regime": "gate_b",
        "probe": "paired_h0_operational_equivalence",
        "policy": "full",
    },
    "gate_b:paired_h0:query_only": {
        "regime": "gate_b",
        "probe": "paired_h0_operational_equivalence",
        "policy": "query_only",
    },
    "gate_b:query_only_latent_no_read:full_positive_control": {
        "regime": "gate_b",
        "probe": "full_positive_control",
        "policy": "full",
    },
    "gate_b:query_only_latent_no_read:query_only": {
        "regime": "gate_b",
        "probe": "query_only_latent_no_read",
        "policy": "query_only",
    },
    "gate_b:removed_path_finite_window:gate_b_index0_r8_h8": {
        "regime": "gate_b",
        "probe": "removed_path_finite_window_influence",
        "policy": "query_only",
    },
}

# Gate C3 is a controls-only amendment. It reuses the C2 control probes and
# changes only the authenticated GPU environment and the blocking mechanism
# objective.
GATE_C3_CONTROLS_SCHEMA_VERSION = 1
GATE_C3_CONTROLS_CONTROL = "example21_gate_c3_pretraining_control_admission"
GATE_C3_CONTROLS_QUALIFICATION_REGIME = (
    "preregistered_gate_c3_pretraining_controls"
)
GATE_C3_CONTROLS_PASSING_INTERPRETATION = (
    "gate_c3_pretraining_controls_passed"
)
GATE_C3_CONTROLS_FAILING_INTERPRETATION = (
    "gate_c3_pretraining_controls_failed_stop"
)
GATE_C3_CONTROLS_INVALID_INTERPRETATION = (
    "gate_c3_pretraining_controls_invalid_stop"
)
GATE_C3_CONTROLS_QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "exact_configuration",
    "prerequisites_authenticated",
    "initialization_authenticated",
    "deterministic_environment_authenticated",
    "canonical_schedules_complete",
    "no_behavioral_or_optimizer_updates",
    "paired_h0_operational_equivalence",
    "no_read_and_removed_path_complete",
    "mechanism_oracle_complete",
    "source_and_gpu_authenticated",
)
GATE_C3_CONTROLS_TOP_LEVEL_KEYS = GATE_C2_CONTROLS_TOP_LEVEL_KEYS
GATE_C3_CONTROLS_MAX_JSON_BYTES = GATE_C2_CONTROLS_MAX_JSON_BYTES
GATE_C3_DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
}
GATE_C3_TERMINAL_H8_OBJECTIVE = {
    "regime": "gate_b",
    "validation_episode_index": 0,
    "stream": "intact",
    "effort": 8,
    "batch_size": 1,
    "mapping_id": 232_423,
    "events_sha256": (
        "36838c2ecd8d00e3b470bf5dc85538539fdc8afac7ce724c6451f0d72a5612ec"
    ),
    "checkpoint": "H_8",
    "sequence_index": 18,
    "gradient_chunk_size": 1,
    "compiled_scan": True,
    "technical_replays": 2,
    "required_paths": [
        "memory_read_projection/weight",
        "workspace_query_projection/weight",
    ],
    "relative_deviation_minimum": 1e-3,
    "l2_difference_absolute_floor": 1e-8,
    "l2_difference_relative_floor": 1e-4,
}
_GATE_C3_TERMINAL_MODEL_ROLES = {
    (
        f"gate_b:mechanism_oracle:terminal_h8:replay_{replay}:"
        f"{policy}:{stage}"
    ): {
        "regime": "gate_b",
        "probe": (
            f"mechanism_oracle:terminal_h8:replay_{replay}:"
            f"{policy}:{stage}"
        ),
        "policy": "full" if policy == "full_read_h8" else "query_only",
    }
    for replay in (1, 2)
    for policy in ("full_read_h8", "query_only_h8")
    for stage in ("reference", "finite_window")
}
GATE_C3_CONTROLS_MODEL_ROLES = dict(
    sorted(
        {
            **{
                role: contract
                for role, contract in GATE_C2_CONTROLS_MODEL_ROLES.items()
                if "mechanism_oracle" not in role
            },
            **_GATE_C3_TERMINAL_MODEL_ROLES,
        }.items()
    )
)

GATE_C2_REMOVED_PATH_GRADIENT_CHUNK_SIZE = 1
GATE_C2_REMOVED_PATH_START_STATE = "materialized_h0_stop_gradient"
GATE_C2_REMOVED_PATHS = (
    "memory_read_projection/weight",
    "workspace_query_projection/weight",
)
GATE_C2_LIVE_PATHS = (
    "color_factor_head/weight",
    "readout_projection/weight",
    "rec_syn/comm/weight",
)
GATE_C2_LATENT_TICKS = {
    "gate_a": ("H1",),
    "gate_b": ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"),
}
GATE_C2_CONTEXT_MEMORY_REPLACEMENTS = {
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
GATE_C2_CACHED_READ_REPLACEMENTS = {
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
GATE_C2_REMOVED_PATH_OBJECTIVES = {
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
        "schedule_sha256": {
            "mapping_ids": "b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8",
            "query_colors": "c7e70f56cca66d920d5d690a902b9943f2fcfdff7003fa4bbb3580070738d67e",
            "presentation_orders": "0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0",
            "shuffled_shifts": "15af1f04589cc523d89b66d2f07027158d69068901d786eecfd259a156f2f2d0",
            "intact": "5683aa84aa2ef8a1ff623e5e0b60afb3451e617728f0363d3ad84f2ea52dacde",
            "shuffled": "abd5eb4ab2e2a685faeb8f6bf785ad2deb97721b00e8d194b4f65d4995516be3",
            "no_context": "45fd14d3faefad83b0ce6d908456320afa67944b361159cfe503fdfab591162d",
            "targets_by_depth": "a438d64347dc4ec5cfc639342d8b142c785e497ddf06728eb03f8ccfb42d3cd6",
            "advance_masks": "b88b3593d9df51260fbafa4a937159c3da3f56fc33335a30993c0ff8a7462ac8",
        },
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
            "selection_mask_values": [False] * 7 + [True],
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

ARM_ORDER = ("full", "query_only", "terminal_only", "legacy", "frozen_write")
REGIME_ORDER = ("gate_a", "gate_b")

SHARED_PARAMETER_PATHS = (
    "color_factor_head/weight",
    "ff_syn/comm/weight",
    "height_head/weight",
    "readout_projection/weight",
    "rec_syn/comm/weight",
    "width_head/weight",
)
MEMORY_PARAMETER_PATHS = (
    "memory_read_projection/weight",
    "memory_write_scale",
    "workspace_query_projection/weight",
)
FULL_PARAMETER_PATHS = tuple(sorted((*SHARED_PARAMETER_PATHS, *MEMORY_PARAMETER_PATHS)))

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

QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "exact_configuration",
    "prerequisites_authenticated",
    "initialization_authenticated",
    "canonical_schedules_complete",
    "fresh_isolated_optimizers",
    "compiler_and_training_complete",
    "full_gate_a_passed",
    "full_gate_b_passed",
    "blocking_behavioral_margins",
    "paired_h0_identity",
    "frozen_write_complete",
    "mechanism_oracle_complete",
    "source_and_gpu_authenticated",
)

GATE_C_INITIALIZATION_QUALIFICATION_CRITERIA = (
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


@dataclass(frozen=True, slots=True)
class GateCArmSpec:
    """Describe one fixed Gate C intervention.

    Parameters
    ----------
    name
        Stable arm identifier.
    memory_mode
        Full, query-only, or legacy contextual-memory policy.
    supervision
        Per-checkpoint or terminal-only loss policy.
    context_memory_width
        Fast-weight width; zero selects the legacy reservoir.
    optimizer_excluded_paths
        Exact parameter paths withheld from optimizer updates.
    """

    name: str
    memory_mode: str
    supervision: str
    context_memory_width: int
    optimizer_excluded_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateCRegimeSpec:
    """Describe one canonical Gate C data and initialization regime.

    Parameters
    ----------
    name
        Stable regime identifier.
    sequence_length, input_width
        Static encoded event geometry.
    training_updates, batch_size, validation_episodes
        Exact production data budget.
    full_parameter_count, legacy_parameter_count
        Backend-independent parameter counts.
    full_parameter_sha256
        Authenticated GPU full-memory initialization digest.
    """

    name: str
    sequence_length: int
    input_width: int
    training_updates: int
    batch_size: int
    validation_episodes: int
    full_parameter_count: int
    full_parameter_sha256: str
    legacy_parameter_count: int


ARM_SPECS: dict[str, GateCArmSpec] = {
    "full": GateCArmSpec("full", "full", "per_checkpoint", 32),
    "query_only": GateCArmSpec(
        "query_only", "query_only", "per_checkpoint", 32
    ),
    "terminal_only": GateCArmSpec(
        "terminal_only", "full", "terminal_only", 32
    ),
    "legacy": GateCArmSpec("legacy", "legacy", "per_checkpoint", 0),
    "frozen_write": GateCArmSpec(
        "frozen_write",
        "full",
        "per_checkpoint",
        32,
        ("memory_write_scale",),
    ),
}

REGIME_SPECS: dict[str, GateCRegimeSpec] = {
    "gate_a": GateCRegimeSpec(
        name="gate_a",
        sequence_length=6,
        input_width=41,
        training_updates=10_000,
        batch_size=64,
        validation_episodes=512,
        full_parameter_count=646_940,
        full_parameter_sha256=(
            "b8ecb04f9c481118afa46651ead411abaccc338ad387f29a1f113d455788a5c8"
        ),
        legacy_parameter_count=514_844,
    ),
    "gate_b": GateCRegimeSpec(
        name="gate_b",
        sequence_length=19,
        input_width=47,
        training_updates=4_096,
        batch_size=64,
        validation_episodes=512,
        full_parameter_count=659_228,
        full_parameter_sha256=(
            "aa463549a8c3c1dbc24c9f727944eada035b776df666a83139214078d0f83d6d"
        ),
        legacy_parameter_count=527_132,
    ),
}


@dataclass(frozen=True, slots=True)
class GateCConfig:
    """Configure the fixed Gate C paired mechanism experiment.

    Parameters
    ----------
    gate_a_config, gate_b_config
        Canonical binding and demonstrated-depth regimes.
    oracle_validation_index
        Fixed Gate B held-out episode used by the mechanism oracle.
    oracle_effort
        Fixed demonstrated depth used by the mechanism oracle.
    gradient_chunk_size
        Strictly finite pp-prop gradient window.
    """

    gate_a_config: gate_a.BindingGateConfig = field(
        default_factory=gate_a.BindingGateConfig
    )
    gate_b_config: gate_b.DepthGateConfig = field(default_factory=gate_b.DepthGateConfig)
    oracle_validation_index: int = 0
    oracle_effort: int = 8
    gradient_chunk_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.gate_a_config, gate_a.BindingGateConfig):
            raise TypeError("gate_a_config must be a BindingGateConfig")
        if not isinstance(self.gate_b_config, gate_b.DepthGateConfig):
            raise TypeError("gate_b_config must be a DepthGateConfig")
        for name in (
            "oracle_validation_index",
            "oracle_effort",
            "gradient_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.gradient_chunk_size == 0:
            raise ValueError("gradient_chunk_size must be positive")

    @property
    def qualification_regime(self) -> str:
        """Return whether every preregistered production coordinate is exact."""

        exact = (
            self.gate_a_config == gate_a.BindingGateConfig()
            and self.gate_b_config == gate_b.DepthGateConfig()
            and self.oracle_validation_index == 0
            and self.oracle_effort == 8
            and self.gradient_chunk_size == 1
        )
        return "preregistered_full" if exact else "nonqualifying_abbreviated"


def _arm_spec(arm: str) -> GateCArmSpec:
    try:
        return ARM_SPECS[arm]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown Gate C arm: {arm!r}") from error


def _regime_spec(regime: str) -> GateCRegimeSpec:
    try:
        return REGIME_SPECS[regime]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown Gate C regime: {regime!r}") from error


def _loss_weights(
    regime: str,
    arm: str,
    *,
    efforts: np.ndarray,
) -> np.ndarray:
    """Return the exact normalized temporal loss mask for an arm."""

    _regime_spec(regime)
    arm_spec = _arm_spec(arm)
    raw_efforts = np.asarray(efforts)
    if raw_efforts.ndim != 1 or raw_efforts.dtype == np.bool_:
        raise ValueError("efforts must be a one-dimensional integer array")
    if not np.issubdtype(raw_efforts.dtype, np.integer):
        raise TypeError("efforts must contain integers")

    if regime == "gate_a":
        weights = np.zeros((REGIME_SPECS[regime].sequence_length,), dtype=np.float32)
        if arm_spec.supervision == "terminal_only":
            weights[-1] = 1.0
        else:
            weights[-2:] = 0.5
        return weights

    if raw_efforts.size == 0 or not np.isin(raw_efforts, gate_b.QUALIFYING_EFFORTS).all():
        raise ValueError("Gate B efforts must lie in {1, 2, 4, 8}")
    weights = np.zeros(
        (raw_efforts.size, REGIME_SPECS[regime].sequence_length), dtype=np.float32
    )
    for row, effort_value in enumerate(raw_efforts):
        effort = int(effort_value)
        if arm_spec.supervision == "terminal_only":
            weights[row, 10 + effort] = 1.0
        else:
            weights[row, 10 : 11 + effort] = np.float32(1.0 / (effort + 1))
    return weights


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


def _gate_c2_gate_b_loss_weights(
    encoded_loss_weights: np.ndarray,
    *,
    arm: str,
    efforts: np.ndarray,
) -> np.ndarray:
    """Return the exact Gate C2 tensor supplied at the trainer boundary."""

    _arm_spec(arm)
    weights = np.asarray(encoded_loss_weights)
    effort_values = np.asarray(efforts)
    if weights.ndim != 2 or weights.shape[1] != REGIME_SPECS["gate_b"].sequence_length:
        raise ValueError("encoded_loss_weights must have shape (updates, 19)")
    if weights.dtype != np.dtype(np.float64):
        raise TypeError("Gate C2 Gate B loss weights must be canonical float64")
    if effort_values.shape != (weights.shape[0],):
        raise ValueError("efforts must match the Gate B update count")
    if effort_values.dtype == np.bool_ or not np.issubdtype(
        effort_values.dtype, np.integer
    ):
        raise TypeError("efforts must contain integers")
    if not np.isin(effort_values, gate_b.QUALIFYING_EFFORTS).all():
        raise ValueError("Gate B efforts must lie in {1, 2, 4, 8}")
    if arm != "terminal_only":
        return encoded_loss_weights
    terminal = np.zeros_like(weights)
    rows = np.arange(weights.shape[0], dtype=np.intp)
    terminal[rows, 10 + effort_values.astype(np.intp, copy=False)] = 1.0
    return terminal


def _gate_c2_consumed_gate_b_loss_weights_complete(value: Any) -> bool:
    """Validate all five independently pinned Gate C2 weight reports."""

    if not isinstance(value, Mapping) or set(value) != set(ARM_ORDER):
        return False
    for arm in ARM_ORDER:
        expected = (
            _GATE_C2_TERMINAL_WEIGHT_REPORT
            if arm == "terminal_only"
            else _GATE_C2_NONTERMINAL_WEIGHT_REPORT
        )
        if not gate_a._json_exact(value[arm], expected):
            return False
    return True


def _model_config_for_arm(
    config: GateCConfig,
    regime: str,
    arm: str,
    *,
    batch_size: int,
) -> ModelConfig:
    """Return one fixed model configuration for a regime and arm."""

    _regime_spec(regime)
    arm_spec = _arm_spec(arm)
    if regime == "gate_a":
        result = gate_a._model_config(config.gate_a_config, batch_size=batch_size)
    else:
        result = gate_b._model_config(config.gate_b_config, batch_size=batch_size)
    if arm_spec.context_memory_width:
        return result
    return dataclasses.replace(
        result,
        context_memory_width=0,
        demonstration_phase_index=None,
        query_phase_index=None,
        input_side_valid_index=None,
        output_side_valid_index=None,
        memory_key_indices=(),
        memory_value_indices=(),
    )


def _optimizer_parameter_paths(
    available_paths: tuple[str, ...],
    arm: str,
) -> tuple[str, ...]:
    """Return the exact optimizer path set after the declared intervention."""

    spec = _arm_spec(arm)
    paths = tuple(available_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("optimizer parameter paths must be unique")
    missing = set(spec.optimizer_excluded_paths) - set(paths)
    if missing:
        raise ValueError(
            "memory_write_scale is required by the frozen-write intervention"
        )
    return tuple(path for path in paths if path not in spec.optimizer_excluded_paths)


def _new_model_for_arm(
    config: GateCConfig,
    regime: str,
    arm: str,
    *,
    batch_size: int,
) -> LatentWorkspaceModel:
    """Construct one fresh, statically intervened Gate C model."""

    arm_spec = _arm_spec(arm)
    model_config = _model_config_for_arm(
        config,
        regime,
        arm,
        batch_size=batch_size,
    )
    policy = "query_only" if arm_spec.memory_mode == "query_only" else "full"
    return LatentWorkspaceModel(model_config, memory_read_policy=policy)


def _parameter_states_by_path(
    model: LatentWorkspaceModel,
) -> dict[str, brainstate.ParamState]:
    return {
        gate_a._path(path): state
        for path, state in model.states(brainstate.ParamState).items()
    }


def _copy_shared_initialization(
    canonical: LatentWorkspaceModel,
    legacy_model: LatentWorkspaceModel,
) -> dict[str, Any]:
    """Copy and verify the six same-shaped initialization paths."""

    if not isinstance(canonical, LatentWorkspaceModel) or not isinstance(
        legacy_model, LatentWorkspaceModel
    ):
        raise TypeError("shared initialization subjects must be workspace models")
    canonical_states = _parameter_states_by_path(canonical)
    legacy_states = _parameter_states_by_path(legacy_model)
    if not set(SHARED_PARAMETER_PATHS).issubset(canonical_states):
        raise ValueError("canonical model is missing a shared parameter path")
    if tuple(sorted(legacy_states)) != SHARED_PARAMETER_PATHS:
        raise ValueError("legacy model must contain exactly the shared paths")
    for path in SHARED_PARAMETER_PATHS:
        source = canonical_states[path].value
        target = legacy_states[path].value
        if jax.tree.structure(source) != jax.tree.structure(target):
            raise ValueError(f"shared parameter structure differs: {path}")
        for source_leaf, target_leaf in zip(
            jax.tree.leaves(source),
            jax.tree.leaves(target),
            strict=True,
        ):
            if source_leaf.shape != target_leaf.shape or source_leaf.dtype != target_leaf.dtype:
                raise ValueError(f"shared parameter geometry differs: {path}")
        legacy_states[path].value = jax.tree.map(
            lambda leaf: jnp.array(leaf, copy=True), source
        )

    canonical_values = legacy._parameter_values(canonical)
    copied_values = legacy._parameter_values(legacy_model)
    all_equal = all(
        jax.tree.structure(canonical_values[path])
        == jax.tree.structure(copied_values[path])
        and all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree.leaves(canonical_values[path]),
                jax.tree.leaves(copied_values[path]),
                strict=True,
            )
        )
        for path in SHARED_PARAMETER_PATHS
    )
    shared_values = {path: copied_values[path] for path in SHARED_PARAMETER_PATHS}
    return {
        "paths": list(SHARED_PARAMETER_PATHS),
        "all_equal": bool(all_equal),
        "sha256": legacy._array_digest(shared_values),
    }


def _regenerate_gate_a_data(config: GateCConfig) -> legacy.BindingData:
    """Regenerate the canonical Gate A schedule from its frozen config."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    return legacy.build_binding_data(config.gate_a_config)


def _regenerate_gate_b_data(
    config: GateCConfig,
) -> tuple[gate_b.DepthSchedule, gate_b.DepthValidationData]:
    """Regenerate the canonical Gate B schedule and held-out controls."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    schedule = gate_b._build_schedule(config.gate_b_config)
    return schedule, gate_b._encode_validation_data(schedule, config.gate_b_config)


@dataclass(slots=True)
class GateCTrainer:
    """Hold one isolated production pp-prop arm trainer.

    Parameters
    ----------
    learner
        Compiled pp-prop sequence learner.
    optimizer
        Fresh Adam optimizer registered only on the arm's update paths.
    compiler, compile_warnings
        Retained compiler topology and warning evidence.
    train_chunk
        One JIT-compiled chunk driver with an internal BrainState loop.
    algorithm
        Stable learning-rule identifier.
    optimizer_parameter_paths, excluded_optimizer_paths
        Exact updated and deliberately frozen parameter paths.
    """

    learner: Any
    optimizer: Any
    compiler: dict[str, Any]
    compile_warnings: list[str]
    train_chunk: Any
    algorithm: str
    optimizer_parameter_paths: tuple[str, ...]
    excluded_optimizer_paths: tuple[str, ...]


def _tree_telemetry(value: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(value))
    if not leaves:
        raise RuntimeError("telemetry subject has no array leaves")
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    maximum = jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(leaf.astype(jnp.float32)), initial=0.0)
                for leaf in leaves
            ]
        )
    )
    count = jnp.asarray(sum(leaf.size for leaf in leaves), dtype=jnp.int32)
    return finite, maximum, count


def _make_arm_trainer(
    model: LatentWorkspaceModel,
    config: GateCConfig,
    regime: str,
    arm: str,
) -> GateCTrainer:
    """Compile one fresh pp-prop trainer for a fixed Gate C arm."""

    if not isinstance(model, LatentWorkspaceModel):
        raise TypeError("model must be a LatentWorkspaceModel")
    _regime_spec(regime)
    arm_spec = _arm_spec(arm)
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    compiler = gate_a._compiler_report(learner)
    parameter_keys = {
        gate_a._path(path): path for path in learner.param_states.keys()
    }
    available_paths = tuple(sorted(parameter_keys))
    optimizer_parameter_paths = _optimizer_parameter_paths(available_paths, arm)
    optimizer_keys = tuple(parameter_keys[path] for path in optimizer_parameter_paths)
    optimizer_states = {
        key: learner.param_states[key] for key in optimizer_keys
    }
    optimizer = braintools.optim.Adam(lr=regime_config.learning_rate)
    optimizer.register_trainable_weights(optimizer_states)
    trace_labels = tuple(
        path
        for path in (
            "ff_syn/comm/weight",
            "rec_syn/comm/weight",
            "memory_write_scale",
            "workspace_query_projection/weight",
            "memory_read_projection/weight",
        )
        if path in parameter_keys
    )
    if not trace_labels:
        raise RuntimeError("trainer has no pp-prop trace parameter")
    model_states = tuple(
        state
        for state in model.states().values()
        if not isinstance(state, brainstate.ParamState)
    )
    rank = regime_config.color_rank
    batch_size = model.config.batch_size

    @brainstate.transform.jit
    def train_chunk(
        events: jax.Array,
        targets: jax.Array,
        loss_weights: jax.Array,
        advance_masks: jax.Array,
    ) -> dict[str, jax.Array]:
        if targets.ndim == 2:
            target_sequences = jnp.broadcast_to(
                targets[:, None, :], events.shape[:3]
            )
        elif targets.ndim == 3:
            target_sequences = targets
        else:
            raise ValueError("targets must have shape (updates,batch) or (updates,time,batch)")
        if loss_weights.ndim == 1:
            weight_sequences = jnp.broadcast_to(
                loss_weights[None, :], events.shape[:2]
            )
        elif loss_weights.ndim == 2:
            weight_sequences = loss_weights
        else:
            raise ValueError("loss weights must have shape (time,) or (updates,time)")

        def train_one(
            inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> dict[str, jax.Array]:
            sequence, target_sequence, weights, advances = inputs
            model.reset_state()
            learner.reset_state(batch_size=batch_size)

            def step_loss(
                event: jax.Array,
                advance: jax.Array,
                target: jax.Array,
            ) -> jax.Array:
                return legacy._classification_loss(
                    learner(event, advance), target, rank
                )

            gradients, loss = learner.etrace_grad(
                sequence,
                advances,
                target_sequence,
                step_fn=step_loss,
                mask=weights.astype(jnp.float32),
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            compact = model.compact_readout()
            trace_factors = tuple(
                learner.get_etrace_of(parameter_keys[label])
                for label in trace_labels
            )
            clipped_gradients = brainstate.nn.clip_grad_norm(
                gradients, regime_config.clip_norm
            )
            optimizer.update(
                {key: clipped_gradients[key] for key in optimizer_keys}
            )
            measurements = {
                "logits": _tree_telemetry(legacy._color_logits(compact, rank)),
                "model_states": _tree_telemetry(
                    tuple(state.value for state in model_states)
                ),
                "gradients": _tree_telemetry(gradients),
                "pp_prop_traces": _tree_telemetry(trace_factors),
                "adam": _tree_telemetry(optimizer.opt_state.value),
                "parameters": _tree_telemetry(
                    tuple(state.value for state in learner.param_states.values())
                ),
            }
            return {
                "loss": loss,
                "finite": {key: value[0] for key, value in measurements.items()},
                "max_abs": {key: value[1] for key, value in measurements.items()},
                "value_count": {
                    key: value[2] for key, value in measurements.items()
                },
            }

        return brainstate.transform.for_loop(
            train_one,
            (events, target_sequences, weight_sequences, advance_masks),
        )

    return GateCTrainer(
        learner=learner,
        optimizer=optimizer,
        compiler=compiler,
        compile_warnings=[str(item.message) for item in caught],
        train_chunk=train_chunk,
        algorithm="production_pp_prop",
        optimizer_parameter_paths=optimizer_parameter_paths,
        excluded_optimizer_paths=arm_spec.optimizer_excluded_paths,
    )


def _optimizer_initial_state_report(
    trainer: GateCTrainer,
    *,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    """Report one fresh arm optimizer before any update."""

    _regime_spec(regime)
    _arm_spec(arm)
    if not isinstance(trainer, GateCTrainer):
        raise TypeError("trainer must be a GateCTrainer")
    leaves = [
        np.ascontiguousarray(np.asarray(leaf))
        for leaf in jax.tree.leaves(trainer.optimizer.opt_state.value)
    ]
    if not leaves:
        raise RuntimeError("optimizer state has no array leaves")
    if any(
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        for array in leaves
    ):
        raise TypeError("optimizer state leaves must be numeric")
    finite = all(np.isfinite(array).all() for array in leaves)
    all_zero = all(np.count_nonzero(array) == 0 for array in leaves)
    fields: list[bytes] = [
        b"example21-gate-c-optimizer-state-v1",
        regime.encode("utf-8"),
        arm.encode("utf-8"),
        *(
            path.encode("utf-8")
            for path in sorted(trainer.optimizer_parameter_paths)
        ),
    ]
    for index, array in enumerate(leaves):
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return {
        "included": list(trainer.optimizer_parameter_paths),
        "excluded": list(trainer.excluded_optimizer_paths),
        "fresh_state_finite": bool(finite),
        "fresh_state_all_zero": bool(all_zero),
        "state_leaf_count": len(leaves),
        "value_count": int(sum(array.size for array in leaves)),
        "state_sha256": hashlib.sha256(b"\0".join(fields)).hexdigest(),
        "executed_updates": int(
            np.asarray(trainer.optimizer.step_count.value).item()
        ),
    }


def _initialization_topology_report(
    model: LatentWorkspaceModel,
    trainer: GateCTrainer,
    *,
    regime: str,
    tree: str,
) -> dict[str, Any]:
    """Bind one fresh model tree to its pp-prop compiler evidence."""

    _regime_spec(regime)
    if tree not in ("canonical_full", "legacy"):
        raise ValueError("initialization tree must be canonical_full or legacy")
    if not isinstance(model, LatentWorkspaceModel):
        raise TypeError("model must be a LatentWorkspaceModel")
    if not isinstance(trainer, GateCTrainer):
        raise TypeError("trainer must be a GateCTrainer")
    values = legacy._parameter_values(model)
    leaves = [
        np.asarray(leaf)
        for value in values.values()
        for leaf in jax.tree.leaves(value)
    ]
    expected_paths = (
        FULL_PARAMETER_PATHS if tree == "canonical_full" else SHARED_PARAMETER_PATHS
    )
    if tuple(sorted(values)) != expected_paths:
        raise ValueError("initialization parameter paths differ from the tree")
    model_states = _parameter_states_by_path(model)
    learner_states = {
        gate_a._path(path): state
        for path, state in trainer.learner.param_states.items()
    }
    if tuple(sorted(learner_states)) != expected_paths or any(
        learner_states[path] is not model_states[path] for path in expected_paths
    ):
        raise ValueError("trainer must be compiled from the same model")
    return {
        "fresh_model": True,
        "model_seed": model.config.seed,
        "memory_read_policy": model.memory_read_policy,
        "model_config": dataclasses.asdict(model.config),
        "parameter_paths": list(expected_paths),
        "parameter_count": int(sum(array.size for array in leaves)),
        "parameter_sha256": legacy._array_digest(values),
        "parameters_finite": bool(
            leaves and all(np.isfinite(array).all() for array in leaves)
        ),
        "compiler": trainer.compiler,
    }


def _normalized_prerequisites(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy the two frozen authenticated prerequisite refs."""

    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "gate_a",
        "gate_b",
    }:
        raise ValueError("Gate C initialization requires Gate A and Gate B")
    if not all(isinstance(prerequisites[name], Mapping) for name in prerequisites):
        raise TypeError("Gate C prerequisite references must be mappings")
    expected = {"gate_a": _GATE_A_REFERENCE, "gate_b": _GATE_B_REFERENCE}
    for name in ("gate_a", "gate_b"):
        if not gate_a._json_exact(prerequisites[name], expected[name]):
            raise ValueError(f"Gate C {name} prerequisite is not authenticated")
    return {
        name: dict(prerequisites[name]) for name in ("gate_a", "gate_b")
    }


def _sha256_complete(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _strict_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _source_files_complete(source_files: Any) -> bool:
    if not isinstance(source_files, Mapping) or set(source_files) != set(
        _GATE_C_SOURCE_FILES
    ):
        return False
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in _GATE_C_SOURCE_FILES
    }
    return gate_a._json_exact(source_files, expected)


def _source_and_gpu_complete(report: Mapping[str, Any]) -> bool:
    start = report["source_start"]
    end = report["source_end"]
    environment = report["environment"]
    source_keys = {
        "asserted_commit",
        "asserted_commit_matches_head",
        "asserted_dirty",
        "asserted_dirty_matches_worktree",
        "commit",
        "commit_is_valid_40_hex",
        "dirty",
        "head_command_succeeded",
        "status_command_succeeded",
        "verified",
    }
    environment_keys = {"backend", "devices", "image_digest", "jax", "python"}
    if (
        not isinstance(start, Mapping)
        or not isinstance(end, Mapping)
        or set(start) != source_keys
        or set(end) != source_keys
        or not isinstance(environment, Mapping)
        or set(environment) != environment_keys
        or not isinstance(environment["jax"], str)
        or not environment["jax"]
        or not isinstance(environment["python"], str)
        or not environment["python"]
        or not isinstance(environment["devices"], list)
        or not environment["devices"]
    ):
        return False
    device_keys = {"device_kind", "id", "platform", "process_index"}
    for device in environment["devices"]:
        if (
            not isinstance(device, Mapping)
            or set(device) != device_keys
            or not isinstance(device["device_kind"], str)
            or not device["device_kind"]
            or not isinstance(device["platform"], str)
            or device["platform"] != "gpu"
            or not _strict_integer(device["id"])
            or not _strict_integer(device["process_index"])
        ):
            return False
    return bool(
        gate_a._source_evidence_clean(start)
        and gate_a._source_evidence_clean(end)
        and start.get("commit") == end.get("commit")
        and gate_a._gpu_environment_verified(environment)
    )


def _compiler_common_complete(
    compiler: Any,
    *,
    expected_paths: tuple[str, ...],
) -> bool:
    if not isinstance(compiler, Mapping):
        return False
    diagnostics = compiler.get("diagnostics")
    diagnostic_keys = {"kind", "level", "message"}
    diagnostics_complete = isinstance(diagnostics, list) and all(
        isinstance(item, Mapping)
        and set(item) in (diagnostic_keys, diagnostic_keys | {"weight_path"})
        and all(isinstance(item[key], str) for key in diagnostic_keys)
        and (
            "weight_path" not in item
            or isinstance(item["weight_path"], str)
        )
        and item["level"].lower() != "error"
        for item in diagnostics
    )
    return bool(
        compiler.get("available") is True
        and diagnostics_complete
        and compiler.get("compiled_parameter_paths") == list(expected_paths)
    )


def _full_compiler_complete(compiler: Any) -> bool:
    try:
        return bool(
            _compiler_common_complete(
                compiler, expected_paths=FULL_PARAMETER_PATHS
            )
            and gate_b._compiler_evidence_complete(compiler)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _legacy_compiler_complete(compiler: Any) -> bool:
    required = {
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    }
    try:
        hidden_groups = compiler["hidden_groups"]
        groups_complete = bool(hidden_groups) and all(
            isinstance(group, Mapping)
            and set(group) == {"index", "hidden_paths"}
            and _strict_integer(group["index"])
            and isinstance(group["hidden_paths"], list)
            and bool(group["hidden_paths"])
            and all(isinstance(path, str) and path for path in group["hidden_paths"])
            and not {
                "context_memory",
                "reasoning_query",
                "memory_read",
            }.intersection(group["hidden_paths"])
            for group in hidden_groups
        )
        return bool(
            _compiler_common_complete(
                compiler, expected_paths=SHARED_PARAMETER_PATHS
            )
            and set(compiler["required_direct_paths"]) == required
            and set(compiler["direct_path_status"]) == required
            and set(compiler["direct_path_evidence"]) == required
            and all(compiler["direct_path_status"][path] is False for path in required)
            and all(compiler["direct_path_evidence"][path] == [] for path in required)
            and compiler["all_required_direct"] is False
            and compiler["context_memory_isolated_from_workspace_lif"] is False
            and isinstance(hidden_groups, list)
            and groups_complete
        )
    except (KeyError, TypeError, ValueError):
        return False


def _regimes_complete(report: Mapping[str, Any], config: GateCConfig) -> bool:
    expected = {
        regime: {
            "spec": dataclasses.asdict(REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
        }
        for regime in REGIME_ORDER
    }
    return bool(
        config.qualification_regime == "preregistered_full"
        and report.get("qualification_regime") == "preregistered_full"
        and gate_a._json_exact(report.get("regimes"), expected)
    )


def _topology_report_complete(
    topology: Any,
    *,
    config: GateCConfig,
    regime: str,
    tree: str,
) -> bool:
    if not isinstance(topology, Mapping) or set(topology) != {
        "fresh_model",
        "model_seed",
        "memory_read_policy",
        "model_config",
        "parameter_paths",
        "parameter_count",
        "parameter_sha256",
        "parameters_finite",
        "compiler",
    }:
        return False
    spec = REGIME_SPECS[regime]
    arm = "legacy" if tree == "legacy" else "full"
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    expected_model = dataclasses.asdict(
        _model_config_for_arm(
            config,
            regime,
            arm,
            batch_size=regime_config.batch_size,
        )
    )
    expected_paths = SHARED_PARAMETER_PATHS if tree == "legacy" else FULL_PARAMETER_PATHS
    expected_count = (
        spec.legacy_parameter_count if tree == "legacy" else spec.full_parameter_count
    )
    sha = topology["parameter_sha256"]
    sha_valid = _sha256_complete(sha) and (
        tree == "legacy" or sha == spec.full_parameter_sha256
    )
    return bool(
        topology["fresh_model"] is True
        and _strict_integer(topology["model_seed"])
        and int(topology["model_seed"]) == 2108
        and topology["memory_read_policy"] == "full"
        and gate_a._json_exact(topology["model_config"], expected_model)
        and topology["parameter_paths"] == list(expected_paths)
        and _strict_integer(topology["parameter_count"])
        and int(topology["parameter_count"]) == expected_count
        and sha_valid
        and topology["parameters_finite"] is True
    )


def _shared_digest_from_path_digests(path_digests: Mapping[str, str]) -> str:
    fields: list[bytes] = [b"example21-gate-c-shared-global-v1"]
    for path in sorted(path_digests):
        fields.extend(
            (path.encode("utf-8"), path_digests[path].encode("ascii"))
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _shared_report_complete(shared: Any) -> bool:
    if not isinstance(shared, Mapping) or set(shared) != {
        "paths",
        "framing",
        "canonical_path_sha256",
        "legacy_path_sha256",
        "canonical_sha256",
        "legacy_sha256",
        "all_equal",
    }:
        return False
    expected_framing = {
        "path": "example21-gate-c-shared-path-v1",
        "global": "example21-gate-c-shared-global-v1",
    }
    canonical = shared["canonical_path_sha256"]
    legacy_paths = shared["legacy_path_sha256"]
    if not (
        shared["paths"] == list(SHARED_PARAMETER_PATHS)
        and gate_a._json_exact(shared["framing"], expected_framing)
        and isinstance(canonical, Mapping)
        and isinstance(legacy_paths, Mapping)
        and set(canonical) == set(legacy_paths) == set(SHARED_PARAMETER_PATHS)
        and all(_sha256_complete(canonical[path]) for path in SHARED_PARAMETER_PATHS)
        and gate_a._json_exact(canonical, legacy_paths)
    ):
        return False
    expected_global = _shared_digest_from_path_digests(canonical)
    return bool(
        shared["canonical_sha256"] == expected_global
        and shared["legacy_sha256"] == expected_global
        and shared["all_equal"] is True
    )


def _arm_refs_complete(initialization: Mapping[str, Any]) -> bool:
    refs = initialization["arm_initialization_refs"]
    if not isinstance(refs, Mapping) or set(refs) != set(ARM_ORDER):
        return False
    canonical_sha = initialization["canonical_full"]["parameter_sha256"]
    legacy_sha = initialization["legacy"]["parameter_sha256"]
    for arm in ARM_ORDER:
        expected = {
            "tree": "legacy" if arm == "legacy" else "canonical_full",
            "parameter_sha256": legacy_sha if arm == "legacy" else canonical_sha,
        }
        if not gate_a._json_exact(refs[arm], expected):
            return False
    return True


def _optimizer_report_complete(
    value: Any,
    *,
    arm: str,
    expected_paths: tuple[str, ...],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "included",
        "excluded",
        "fresh_state_finite",
        "fresh_state_all_zero",
        "state_leaf_count",
        "value_count",
        "state_sha256",
        "executed_updates",
    }:
        return False
    return bool(
        value["included"] == list(_optimizer_parameter_paths(expected_paths, arm))
        and value["excluded"] == list(ARM_SPECS[arm].optimizer_excluded_paths)
        and _strict_integer(value["state_leaf_count"])
        and int(value["state_leaf_count"]) > 0
        and _strict_integer(value["value_count"])
        and int(value["value_count"]) > 0
        and _sha256_complete(value["state_sha256"])
    )


def _optimizer_paths_complete(initialization: Mapping[str, Any]) -> bool:
    reports = initialization["optimizer_paths"]
    if not isinstance(reports, Mapping) or set(reports) != set(ARM_ORDER):
        return False
    return all(
        _optimizer_report_complete(
            reports[arm],
            arm=arm,
            expected_paths=(
                SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
            ),
        )
        for arm in ARM_ORDER
    )


def _optimizer_states_complete(initialization: Mapping[str, Any]) -> bool:
    return all(
        initialization["optimizer_paths"][arm]["fresh_state_finite"] is True
        and initialization["optimizer_paths"][arm]["fresh_state_all_zero"] is True
        for arm in ARM_ORDER
    )


def _no_behavioral_updates(report: Mapping[str, Any]) -> bool:
    allowed = {
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
    expected_initialization_keys = {
        "canonical_full",
        "legacy",
        "shared_paths",
        "arm_initialization_refs",
        "optimizer_paths",
    }
    return bool(
        set(report).issubset(allowed)
        and isinstance(report.get("initialization"), Mapping)
        and set(report["initialization"]) == set(REGIME_ORDER)
        and all(
            isinstance(report["initialization"][regime], Mapping)
            and set(report["initialization"][regime])
            == expected_initialization_keys
            for regime in REGIME_ORDER
        )
        and all(
            report["initialization"][regime]["optimizer_paths"][arm][
                "executed_updates"
            ]
            == 0
            and _strict_integer(
                report["initialization"][regime]["optimizer_paths"][arm][
                    "executed_updates"
                ]
            )
            for regime in REGIME_ORDER
            for arm in ARM_ORDER
        )
    )


def _gate_c_initialization_qualification(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Recompute every Gate C initialization admission criterion."""

    criteria = {
        name: False for name in GATE_C_INITIALIZATION_QUALIFICATION_CRITERIA
    }
    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        return {
            "criteria": criteria,
            "passed": False,
            "interpretation": "gate_c_initialization_admission_failed_stop",
        }
    if config.qualification_regime != "preregistered_full":
        return {
            "criteria": criteria,
            "passed": False,
            "interpretation": "gate_c_initialization_admission_failed_stop",
        }
    try:
        base_keys = {
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
        }
        criteria["schema_and_control"] = bool(
            set(report) in (base_keys, base_keys | {"qualification"})
            and _strict_integer(report["schema_version"])
            and int(report["schema_version"]) == GATE_C_SCHEMA_VERSION
            and report["control"] == GATE_C_INITIALIZATION_CONTROL
        )
        criteria["preregistered_regimes"] = _regimes_complete(report, config)
        prerequisites = report["prerequisites"]
        exact_prerequisites = isinstance(prerequisites, Mapping) and set(
            prerequisites
        ) == {"gate_a", "gate_b"}
        criteria["gate_a_prerequisite_authenticated"] = bool(
            exact_prerequisites
            and "gate_a" in prerequisites
            and gate_a._json_exact(prerequisites["gate_a"], _GATE_A_REFERENCE)
        )
        criteria["gate_b_prerequisite_authenticated"] = bool(
            exact_prerequisites
            and "gate_b" in prerequisites
            and gate_a._json_exact(prerequisites["gate_b"], _GATE_B_REFERENCE)
        )
        criteria["source_and_gpu_authenticated"] = _source_and_gpu_complete(report)
        criteria["source_files_exact"] = _source_files_complete(
            report["source_files"]
        )
        initialization = report["initialization"]
        exact_regimes = isinstance(initialization, Mapping) and set(
            initialization
        ) == set(REGIME_ORDER) and all(
            isinstance(initialization[regime], Mapping)
            and set(initialization[regime])
            == {
                "canonical_full",
                "legacy",
                "shared_paths",
                "arm_initialization_refs",
                "optimizer_paths",
            }
            for regime in REGIME_ORDER
        )
        criteria["schema_and_control"] = bool(
            criteria["schema_and_control"] and exact_regimes
        )
        if exact_regimes:
            criteria["canonical_full_initializations_exact"] = all(
                _topology_report_complete(
                    initialization[regime]["canonical_full"],
                    config=config,
                    regime=regime,
                    tree="canonical_full",
                )
                for regime in REGIME_ORDER
            )
            criteria["legacy_initializations_complete"] = all(
                _topology_report_complete(
                    initialization[regime]["legacy"],
                    config=config,
                    regime=regime,
                    tree="legacy",
                )
                for regime in REGIME_ORDER
            )
            criteria["shared_paths_byte_identical"] = all(
                _shared_report_complete(initialization[regime]["shared_paths"])
                for regime in REGIME_ORDER
            )
            criteria["arm_initialization_refs_exact"] = all(
                _arm_refs_complete(initialization[regime])
                for regime in REGIME_ORDER
            )
            criteria["optimizer_paths_exact"] = all(
                _optimizer_paths_complete(initialization[regime])
                for regime in REGIME_ORDER
            )
            criteria["fresh_optimizer_states_zero_and_finite"] = all(
                _optimizer_states_complete(initialization[regime])
                for regime in REGIME_ORDER
            )
            criteria["compiler_topologies_complete"] = all(
                _full_compiler_complete(
                    initialization[regime]["canonical_full"]["compiler"]
                )
                and _legacy_compiler_complete(
                    initialization[regime]["legacy"]["compiler"]
                )
                for regime in REGIME_ORDER
            )
            criteria["no_behavioral_updates"] = _no_behavioral_updates(report)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        pass
    passed = bool(criteria and all(criteria.values()))
    return {
        "criteria": criteria,
        "passed": passed,
        "interpretation": (
            "gate_c_initialization_admission_passed"
            if passed
            else "gate_c_initialization_admission_failed_stop"
        ),
    }


def _validated_gate_c_initialization_admission(
    prerequisite: Mapping[str, Any],
    config: GateCConfig,
    *,
    source_start: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_files: Mapping[str, str],
    require_pass: bool,
) -> Mapping[str, Any]:
    """Validate the complete authenticated Gate C initialization wrapper."""

    expected_keys = {
        "target",
        "source_head",
        "image_digest",
        "bundle_sha256",
        "manifest_sha256",
        "preflight_sha256",
        "result_sha256",
        "admission",
    }
    if not isinstance(prerequisite, Mapping) or set(prerequisite) != expected_keys:
        raise ValueError("Gate C initialization prerequisite is not authenticated")
    source_head = prerequisite["source_head"]
    image_digest = prerequisite["image_digest"]
    if (
        prerequisite["target"] != "gate_c_init"
        or not isinstance(source_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
        or not all(
            _sha256_complete(prerequisite[name])
            for name in (
                "bundle_sha256",
                "manifest_sha256",
                "preflight_sha256",
                "result_sha256",
            )
        )
    ):
        raise ValueError("Gate C initialization provenance fields are invalid")
    admission = prerequisite["admission"]
    if not isinstance(admission, Mapping):
        raise ValueError("Gate C initialization admission is missing")
    admission_keys = {
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
    if set(admission) != admission_keys:
        raise ValueError("Gate C initialization admission schema is invalid")
    if gate_b._strict_json_sha256(admission) != prerequisite["result_sha256"]:
        raise ValueError("Gate C initialization result digest is invalid")
    expected_bundle = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_c_init\0"
            f"{source_head}\0{prerequisite['preflight_sha256']}\0"
            f"{prerequisite['result_sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    if prerequisite["bundle_sha256"] != expected_bundle:
        raise ValueError("Gate C initialization bundle digest is invalid")
    qualification = _gate_c_initialization_qualification(admission, config=config)
    if not gate_a._json_exact(admission.get("qualification"), qualification):
        raise ValueError("Gate C initialization qualification is stale")
    if require_pass and qualification["passed"] is not True:
        raise ValueError("Gate C initialization admission did not pass")
    try:
        source_matches = bool(
            admission["source_start"]["commit"] == source_head
            and admission["source_end"]["commit"] == source_head
            and admission["environment"]["image_digest"] == image_digest
            and source_start["commit"] == source_head
            and environment["image_digest"] == image_digest
        )
    except (KeyError, TypeError):
        source_matches = False
    if not source_matches:
        raise ValueError("Gate C initialization source or image differs")
    if not _source_and_gpu_complete(
        {
            "source_start": source_start,
            "source_end": source_start,
            "environment": environment,
        }
    ):
        raise ValueError("Gate C formal source or GPU evidence is invalid")
    if not (
        gate_a._json_exact(admission.get("source_files"), source_files)
        and _source_files_complete(source_files)
    ):
        raise ValueError("Gate C initialization source files differ")
    return admission


def _arm_initialization_reproduced(
    report: Mapping[str, Any],
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> bool:
    """Check one formal arm against its authenticated initialization evidence."""

    try:
        _regime_spec(regime)
        _arm_spec(arm)
        if not isinstance(report, Mapping) or set(report) != {
            "initialization",
            "optimizer",
            "compiler",
        }:
            return False
        initialization = report["initialization"]
        if not isinstance(initialization, Mapping) or set(initialization) != {
            "tree",
            "parameter_sha256",
            "parameter_count",
            "parameter_paths",
            "shared_paths",
        }:
            return False
        regime_admission = admission["initialization"][regime]
        reference = regime_admission["arm_initialization_refs"][arm]
        topology = regime_admission[reference["tree"]]
        expected_paths = (
            SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
        )
        return bool(
            initialization["tree"] == reference["tree"]
            and initialization["parameter_sha256"]
            == reference["parameter_sha256"]
            and _strict_integer(initialization["parameter_count"])
            and initialization["parameter_count"] == topology["parameter_count"]
            and initialization["parameter_paths"] == list(expected_paths)
            and gate_a._json_exact(
                initialization["shared_paths"],
                regime_admission["shared_paths"],
            )
            and gate_a._json_exact(
                report["optimizer"],
                regime_admission["optimizer_paths"][arm],
            )
            and gate_a._json_exact(report["compiler"], topology["compiler"])
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _formal_arm_initialization_report(
    model: LatentWorkspaceModel,
    trainer: GateCTrainer,
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> dict[str, Any]:
    """Rebuild and bind one arm's complete pre-update initialization report."""

    reference = admission["initialization"][regime]["arm_initialization_refs"][arm]
    tree = str(reference["tree"])
    topology = _initialization_topology_report(
        model,
        trainer,
        regime=regime,
        tree=tree,
    )
    values = legacy._parameter_values(model)
    shared_values = {path: values[path] for path in SHARED_PARAMETER_PATHS}
    shared = admission["initialization"][regime]["shared_paths"]
    digest_key = "legacy" if arm == "legacy" else "canonical"
    actual_path_digests = {
        path: _shared_path_sha256(path, shared_values[path])
        for path in SHARED_PARAMETER_PATHS
    }
    actual_shared = {
        **shared,
        f"{digest_key}_path_sha256": actual_path_digests,
        f"{digest_key}_sha256": _shared_global_sha256(shared_values),
    }
    report = {
        "initialization": {
            "tree": tree,
            "parameter_sha256": topology["parameter_sha256"],
            "parameter_count": topology["parameter_count"],
            "parameter_paths": topology["parameter_paths"],
            "shared_paths": actual_shared,
        },
        "optimizer": _optimizer_initial_state_report(
            trainer,
            regime=regime,
            arm=arm,
        ),
        "compiler": topology["compiler"],
    }
    if not _arm_initialization_reproduced(
        report,
        admission,
        regime,
        arm,
    ):
        raise RuntimeError("formal arm initialization was not reproduced")
    return report


def _fresh_formal_arm(
    config: GateCConfig,
    admission: Mapping[str, Any],
    regime: str,
    arm: str,
) -> tuple[LatentWorkspaceModel, GateCTrainer, dict[str, Any]]:
    """Construct and authenticate one fresh formal arm before any update."""

    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    if arm == "legacy":
        canonical = _new_model_for_arm(
            config,
            regime,
            "full",
            batch_size=regime_config.batch_size,
        )
        model = _new_model_for_arm(
            config,
            regime,
            arm,
            batch_size=regime_config.batch_size,
        )
        _copy_shared_initialization(canonical, model)
        del canonical
    else:
        model = _new_model_for_arm(
            config,
            regime,
            arm,
            batch_size=regime_config.batch_size,
        )
    trainer = _make_arm_trainer(model, config, regime, arm)
    report = _formal_arm_initialization_report(
        model,
        trainer,
        admission,
        regime,
        arm,
    )
    return model, trainer, report


def _source_files_report() -> dict[str, str]:
    """Hash the exact six scientific source files for Gate C."""

    repo_root = Path(__file__).resolve().parents[2]
    return {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in _GATE_C_SOURCE_FILES
    }


def write_artifact(value: Mapping[str, Any], path: str | Path) -> Path:
    """Write one deterministic, strict Gate C artifact atomically.

    Parameters
    ----------
    value
        JSON-compatible top-level mapping. NaN and infinity are rejected.
    path
        Final artifact path.

    Returns
    -------
    pathlib.Path
        Final artifact path after atomic replacement.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        if value.get("control") in {
            GATE_C2_CONTROLS_CONTROL,
            GATE_C3_CONTROLS_CONTROL,
        }:
            controls_name = (
                "Gate C2"
                if value.get("control") == GATE_C2_CONTROLS_CONTROL
                else "Gate C3"
            )
            encoder = json.JSONEncoder(
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            byte_count = 0
            with temporary.open("wb") as stream:
                for chunk in encoder.iterencode(value):
                    encoded = chunk.encode("utf-8")
                    if (
                        byte_count + len(encoded) + 1
                        > GATE_C2_CONTROLS_MAX_JSON_BYTES
                    ):
                        raise ValueError(
                            f"{controls_name} controls JSON exceeds the 192 MiB size limit"
                        )
                    stream.write(encoded)
                    byte_count += len(encoded)
                stream.write(b"\n")
                byte_count += 1
                if byte_count > GATE_C2_CONTROLS_MAX_JSON_BYTES:
                    raise ValueError(
                        f"{controls_name} controls JSON exceeds the 192 MiB size limit"
                    )
                stream.flush()
                os.fsync(stream.fileno())
        else:
            payload = (
                json.dumps(
                    value,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                    separators=(",", ": "),
                )
                + "\n"
            )
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def run_gate_c_initialization(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and inspect all Gate C initial states without behavior.

    Parameters
    ----------
    config
        Fixed paired Gate C configuration.
    prerequisites
        Launcher-authenticated compact Gate A and Gate B references.
    source_start
        Live clean-source evidence captured before construction.
    source_end_reporter
        Zero-argument callback that captures live source evidence after every
        topology and optimizer report is complete.
    source_files
        Exact six-file scientific source digest mapping.
    environment
        Authenticated GPU image and device evidence.

    Returns
    -------
    dict
        Strict initialization-only artifact payload.
    """

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    normalized_prerequisites = _normalized_prerequisites(prerequisites)
    source_keys = {
        "asserted_commit",
        "asserted_commit_matches_head",
        "asserted_dirty",
        "asserted_dirty_matches_worktree",
        "commit",
        "commit_is_valid_40_hex",
        "dirty",
        "head_command_succeeded",
        "status_command_succeeded",
        "verified",
    }
    authenticate_inputs = bool(
        config.qualification_regime == "preregistered_full"
        or (isinstance(source_start, Mapping) and set(source_start) == source_keys)
    )
    if authenticate_inputs:
        if not isinstance(source_start, Mapping) or not gate_a._source_evidence_clean(
            source_start
        ):
            raise RuntimeError("Gate C initialization source is not authenticated")
        if not isinstance(environment, Mapping) or not gate_a._gpu_environment_verified(
            environment
        ):
            raise RuntimeError("Gate C initialization GPU is not authenticated")
        if not _source_files_complete(source_files):
            raise RuntimeError("Gate C initialization source files are not exact")
    initialization: dict[str, Any] = {}
    regime_reports = {
        regime: {
            "spec": dataclasses.asdict(REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
        }
        for regime in REGIME_ORDER
    }

    for regime in REGIME_ORDER:
        regime_config = (
            config.gate_a_config if regime == "gate_a" else config.gate_b_config
        )
        models = {
            arm: _new_model_for_arm(
                config,
                regime,
                arm,
                batch_size=regime_config.batch_size,
            )
            for arm in ARM_ORDER
        }
        _copy_shared_initialization(models["full"], models["legacy"])
        trainers = {
            arm: _make_arm_trainer(models[arm], config, regime, arm)
            for arm in ARM_ORDER
        }
        canonical = _initialization_topology_report(
            models["full"],
            trainers["full"],
            regime=regime,
            tree="canonical_full",
        )
        legacy_report = _initialization_topology_report(
            models["legacy"],
            trainers["legacy"],
            regime=regime,
            tree="legacy",
        )
        canonical_values = legacy._parameter_values(models["full"])
        legacy_values = legacy._parameter_values(models["legacy"])
        canonical_shared = {
            path: canonical_values[path] for path in SHARED_PARAMETER_PATHS
        }
        legacy_shared = {
            path: legacy_values[path] for path in SHARED_PARAMETER_PATHS
        }
        canonical_path_sha256 = {
            path: _shared_path_sha256(path, canonical_shared[path])
            for path in SHARED_PARAMETER_PATHS
        }
        legacy_path_sha256 = {
            path: _shared_path_sha256(path, legacy_shared[path])
            for path in SHARED_PARAMETER_PATHS
        }
        canonical_shared_sha256 = _shared_global_sha256(canonical_shared)
        legacy_shared_sha256 = _shared_global_sha256(legacy_shared)
        arm_refs = {
            arm: {
                "tree": "legacy" if arm == "legacy" else "canonical_full",
                "parameter_sha256": legacy._array_digest(
                    legacy._parameter_values(models[arm])
                ),
            }
            for arm in ARM_ORDER
        }
        optimizer_paths = {
            arm: _optimizer_initial_state_report(
                trainers[arm], regime=regime, arm=arm
            )
            for arm in ARM_ORDER
        }
        initialization[regime] = {
            "canonical_full": canonical,
            "legacy": legacy_report,
            "shared_paths": {
                "paths": list(SHARED_PARAMETER_PATHS),
                "framing": {
                    "path": "example21-gate-c-shared-path-v1",
                    "global": "example21-gate-c-shared-global-v1",
                },
                "canonical_path_sha256": canonical_path_sha256,
                "legacy_path_sha256": legacy_path_sha256,
                "canonical_sha256": canonical_shared_sha256,
                "legacy_sha256": legacy_shared_sha256,
                "all_equal": bool(
                    canonical_path_sha256 == legacy_path_sha256
                    and canonical_shared_sha256 == legacy_shared_sha256
                ),
            },
            "arm_initialization_refs": arm_refs,
            "optimizer_paths": optimizer_paths,
        }

    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    report: dict[str, Any] = {
        "schema_version": GATE_C_SCHEMA_VERSION,
        "control": GATE_C_INITIALIZATION_CONTROL,
        "qualification_regime": config.qualification_regime,
        "prerequisites": normalized_prerequisites,
        "regimes": regime_reports,
        "initialization": initialization,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": dict(environment),
    }
    report["qualification"] = _gate_c_initialization_qualification(
        report, config=config
    )
    return report


def _evaluate_arm(
    trained_model: LatentWorkspaceModel,
    data: legacy.BindingData | gate_b.DepthValidationData,
    config: GateCConfig,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    """Evaluate one trained arm on its three canonical held-out streams."""

    _regime_spec(regime)
    _arm_spec(arm)
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    validation_episodes = regime_config.validation_episodes
    model = _new_model_for_arm(
        config,
        regime,
        arm,
        batch_size=validation_episodes,
    )
    legacy._copy_parameters(trained_model, model)
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A evaluation requires BindingData")
        event_streams = {
            "intact": data.validation_intact,
            "shuffled": data.validation_shuffled,
            "no_context": data.validation_no_context,
        }
        advances = jnp.ones(
            (regime_config.sequence_length, validation_episodes), dtype=jnp.bool_
        )
        targets_by_depth = np.broadcast_to(
            np.asarray(data.validation_targets)[None, :],
            (regime_config.gap_steps + 1, validation_episodes),
        )
    else:
        if not isinstance(data, gate_b.DepthValidationData):
            raise TypeError("Gate B evaluation requires DepthValidationData")
        event_streams = {
            "intact": data.intact,
            "shuffled": data.shuffled,
            "no_context": data.no_context,
        }
        advances = jnp.asarray(data.advance_masks)
        targets_by_depth = np.asarray(data.targets_by_depth)
    model_states = tuple(model.states().values())
    checkpoint_start = regime_config.sequence_length - regime_config.gap_steps - 1

    @brainstate.transform.jit
    def evaluate_stream(
        events: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        model.reset_state()

        def step(
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            event, advance = inputs
            compact = model.update(event, advance)
            state_finite = jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
                        for state in model_states
                        for leaf in jax.tree.leaves(state.value)
                    ]
                )
            )
            if model.config.memory_enabled:
                memory_read = jnp.asarray(model.memory_read.value)
                workspace = jnp.asarray(model.workspace_carrier.value)
            else:
                memory_read = jnp.zeros(
                    (validation_episodes, 0), dtype=jnp.float32
                )
                workspace = jnp.zeros(
                    (validation_episodes, 0), dtype=jnp.float32
                )
            return compact, state_finite, memory_read, workspace

        compact, state_finite, reads, workspaces = brainstate.transform.for_loop(
            step, (events, advances)
        )
        final_memory = (
            jnp.asarray(model.context_memory.value)
            if model.config.memory_enabled
            else jnp.zeros((validation_episodes, 0, 0), dtype=jnp.float32)
        )
        return (
            compact[checkpoint_start:],
            state_finite,
            reads[checkpoint_start:],
            workspaces[checkpoint_start:],
            final_memory,
        )

    raw: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for name, events in event_streams.items():
        values = jax.block_until_ready(evaluate_stream(jnp.asarray(events)))
        raw[name] = tuple(np.asarray(value) for value in values)  # type: ignore[assignment]
    finite = all(
        bool(np.isfinite(value).all())
        for values in raw.values()
        for value in values
    ) and all(bool(values[1].all()) for values in raw.values())
    predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in event_streams
    }
    depth_reports: dict[str, dict[str, Any]] = {}
    for depth in range(regime_config.gap_steps + 1):
        depth_metrics: dict[str, Any] = {}
        for name in event_streams:
            color_logits = np.asarray(
                legacy._color_logits(jnp.asarray(raw[name][0][depth]), regime_config.color_rank)
            )
            finite = finite and bool(np.isfinite(color_logits).all())
            prediction = np.argmax(color_logits, axis=-1)
            predictions[name].append(prediction)
            depth_metrics[name] = {
                **legacy._accuracy(prediction, targets_by_depth[depth]),
                "checkpoint": depth,
            }
        depth_reports[str(depth)] = depth_metrics

    if regime == "gate_a":
        final = depth_reports[str(regime_config.gap_steps)]
        binding_diagnostic = gate_a._diagnostic_report(
            {
                name: (values[4], values[2], values[3])
                for name, values in raw.items()
            }
        )
        memory = binding_diagnostic["memory"]
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
            "all_finite": binding_diagnostic["all_state_tensors_finite"],
        }
        return {
            "finite": finite,
            "all_compact_logits_finite": finite,
            "all_state_tensors_finite": bool(
                all(values[1].all() for values in raw.values())
            ),
            "depths": depth_reports,
            "intact": final["intact"],
            "shuffled": final["shuffled"],
            "no_context": final["no_context"],
            "intact_minus_shuffled": (
                final["intact"]["accuracy"] - final["shuffled"]["accuracy"]
            ),
            "binding_state": binding_state,
            "binding_diagnostic": binding_diagnostic,
        }

    h0_predictions = predictions["intact"][0]
    efforts: dict[str, dict[str, Any]] = {}
    for effort in gate_b.QUALIFYING_EFFORTS:
        intact = depth_reports[str(effort)]["intact"]
        shuffled = depth_reports[str(effort)]["shuffled"]
        no_context = depth_reports[str(effort)]["no_context"]
        h0_final = {
            **legacy._accuracy(h0_predictions, targets_by_depth[effort]),
            "checkpoint": 0,
        }
        efforts[str(effort)] = {
            "intact": intact,
            "shuffled": shuffled,
            "no_context": no_context,
            "h0_final_target": h0_final,
            "intact_minus_h0": intact["accuracy"] - h0_final["accuracy"],
            "intact_minus_shuffled": (
                intact["accuracy"] - shuffled["accuracy"]
            ),
        }
    return {
        "finite": finite,
        "h0_proper": depth_reports["0"]["intact"],
        "depths": depth_reports,
        "efforts": efforts,
    }


def _parameter_movement_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    path_reports: dict[str, dict[str, float | int]] = {}
    total_squared = 0.0
    total_count = 0
    if set(before) != set(after):
        raise RuntimeError("formal arm parameter paths changed during training")
    for path in sorted(before):
        squared = 0.0
        count = 0
        for old, new in zip(
            jax.tree.leaves(before[path]),
            jax.tree.leaves(after[path]),
            strict=True,
        ):
            delta = np.asarray(new, dtype=np.float64) - np.asarray(
                old, dtype=np.float64
            )
            squared += float(np.sum(delta * delta, dtype=np.float64))
            count += int(delta.size)
        path_reports[path] = {
            "l2_delta": math.sqrt(squared),
            "parameter_count": count,
        }
        total_squared += squared
        total_count += count
    return {
        "l2_delta": math.sqrt(total_squared),
        "parameter_count": total_count,
        "paths": path_reports,
    }


def _aggregate_training_telemetry(
    chunks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("formal arm executed no training chunks")
    losses = np.concatenate(
        [np.asarray(chunk["loss"], dtype=np.float64).reshape(-1) for chunk in chunks]
    )
    categories = tuple(chunks[0]["finite"])
    finite = {
        category: bool(
            all(
                np.asarray(chunk["finite"][category], dtype=np.bool_).all()
                for chunk in chunks
            )
        )
        for category in categories
    }
    maxima = {
        category: float(
            max(
                np.max(
                    np.asarray(chunk["max_abs"][category], dtype=np.float64),
                    initial=0.0,
                )
                for chunk in chunks
            )
        )
        for category in categories
    }
    value_counts = {
        category: int(
            sum(
                int(
                    np.sum(
                        np.asarray(
                            chunk["value_count"][category],
                            dtype=np.int64,
                        )
                    )
                )
                for chunk in chunks
            )
        )
        for category in categories
    }
    if not np.isfinite(losses).all():
        raise RuntimeError("formal arm produced a non-finite training loss")
    return {
        "losses": losses.tolist(),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "tail_64_mean_loss": float(losses[-min(64, losses.size) :].mean()),
        "finite": finite,
        "max_abs": maxima,
        "value_count": value_counts,
    }


def _run_gate_c_arm(
    model: LatentWorkspaceModel,
    trainer: GateCTrainer,
    data: legacy.BindingData | tuple[gate_b.DepthSchedule, gate_b.DepthValidationData],
    config: GateCConfig,
    regime: str,
    arm: str,
    *,
    initialization_report: Mapping[str, Any],
    execution_index: int | None = None,
    data_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate one isolated formal Gate C arm."""

    _regime_spec(regime)
    _arm_spec(arm)
    expected_execution_index = (
        REGIME_ORDER.index(regime) * len(ARM_ORDER) + ARM_ORDER.index(arm)
    )
    if execution_index is None:
        execution_index = expected_execution_index
    if (
        not _strict_integer(execution_index)
        or int(execution_index) != expected_execution_index
    ):
        raise ValueError("formal arm execution index differs from the fixed order")
    before = legacy._parameter_values(model)
    before_sha256 = legacy._array_digest(before)
    telemetry: list[Mapping[str, Any]] = []
    consumed_weight_chunks: list[np.ndarray] = []
    start = time.perf_counter()
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A arm requires BindingData")
        updates = config.gate_a_config.training_updates
        efforts = np.ones((updates,), dtype=np.int32)
        advances = np.ones(
            (
                updates,
                config.gate_a_config.sequence_length,
                config.gate_a_config.batch_size,
            ),
            dtype=np.bool_,
        )
        weights = _loss_weights("gate_a", arm, efforts=efforts)
        consumed_weight_chunks.append(np.asarray(weights))
        telemetry.append(
            jax.device_get(
                jax.block_until_ready(
                    trainer.train_chunk(
                        data.training_events,
                        data.training_targets,
                        weights,
                        advances,
                    )
                )
            )
        )
        evaluation_data: legacy.BindingData | gate_b.DepthValidationData = data
        chunk_count = 1
        actual_data_identity = {
            "training_schedule_sha256": legacy._digest_arrays(
                data.training_events,
                data.training_targets,
                data.training_mapping_ids,
            ),
            "validation_schedule_sha256": legacy._digest_arrays(
                data.validation_intact,
                data.validation_targets,
                data.validation_mapping_ids,
            ),
            "training_mapping_ids_sha256": legacy._digest_arrays(
                data.training_mapping_ids.reshape(-1)
            ),
            "validation_mapping_ids_sha256": legacy._digest_arrays(
                data.validation_mapping_ids
            ),
        }
    else:
        if not isinstance(data, tuple) or len(data) != 2:
            raise TypeError("Gate B arm requires its schedule and validation data")
        schedule, validation = data
        if not isinstance(schedule, gate_b.DepthSchedule) or not isinstance(
            validation, gate_b.DepthValidationData
        ):
            raise TypeError("Gate B arm data has the wrong type")
        chunk_count = 0
        hash_state = gate_b._new_encoded_schedule_hash_state()
        for schedule_chunk in gate_b._iter_schedule_chunks(
            schedule,
            config.gate_b_config,
        ):
            encoded = gate_b._encode_training_chunk(
                schedule_chunk,
                config.gate_b_config,
            )
            gate_b._update_encoded_schedule_hash_state(
                hash_state,
                encoded,
                config.gate_b_config,
            )
            weights = _loss_weights(
                "gate_b",
                arm,
                efforts=np.asarray(encoded.efforts),
            )
            consumed_weight_chunks.append(np.asarray(weights))
            telemetry.append(
                jax.device_get(
                    jax.block_until_ready(
                        trainer.train_chunk(
                            encoded.events,
                            encoded.targets,
                            weights,
                            encoded.advance_masks,
                        )
                    )
                )
            )
            chunk_count += 1
        evaluation_data = validation
        encoded_identity = gate_b._finish_encoded_schedule_report(
            hash_state,
            config.gate_b_config,
        )
        actual_data_identity = {
            "training_global_sha256": dict(encoded_identity["global_sha256"]),
            "validation_sha256": dict(
                gate_b._validation_data_report(validation)["sha256"]
            ),
        }
    if data_identity is not None and not gate_a._json_exact(
        data_identity, actual_data_identity
    ):
        raise ValueError("formal arm data identity differs from consumed bytes")
    training_seconds = time.perf_counter() - start
    after = legacy._parameter_values(model)
    aggregate = _aggregate_training_telemetry(telemetry)
    consumed_weights = (
        consumed_weight_chunks[0]
        if regime == "gate_a"
        else np.concatenate(consumed_weight_chunks, axis=0)
    )
    executed_updates = len(aggregate["losses"])
    training = {
        "algorithm": trainer.algorithm,
        "execution_index": int(execution_index),
        "intervention": dataclasses.asdict(ARM_SPECS[arm]),
        "data_identity": actual_data_identity,
        "executed_updates": executed_updates,
        "batch_size": (
            config.gate_a_config.batch_size
            if regime == "gate_a"
            else config.gate_b_config.batch_size
        ),
        "chunk_count": chunk_count,
        "cold_compile_and_train_seconds": training_seconds,
        "initial_parameter_sha256": before_sha256,
        "final_parameter_sha256": legacy._array_digest(after),
        "optimizer_final_step": int(
            np.asarray(jax.device_get(trainer.optimizer.step_count.value)).item()
        ),
        "loss_weights": {
            "dtype": consumed_weights.dtype.str,
            "shape": list(consumed_weights.shape),
            "sha256": legacy._digest_arrays(consumed_weights),
        },
        "compile_warnings": list(trainer.compile_warnings),
        **aggregate,
    }
    if training["optimizer_final_step"] != executed_updates:
        raise RuntimeError("formal arm optimizer step count differs from updates")
    if not all(training["finite"].values()):
        raise RuntimeError("formal arm telemetry is not finite")
    movement = _parameter_movement_report(before, after)
    write_before = before.get("memory_write_scale")
    write_after = after.get("memory_write_scale")
    training["frozen_write"] = {
        "applicable": arm == "frozen_write",
        "all_ones_before": bool(
            write_before is not None
            and all(
                np.equal(np.asarray(leaf), 1.0).all()
                for leaf in jax.tree.leaves(write_before)
            )
        ),
        "all_ones_after": bool(
            write_after is not None
            and all(
                np.equal(np.asarray(leaf), 1.0).all()
                for leaf in jax.tree.leaves(write_after)
            )
        ),
        "excluded_from_optimizer": (
            "memory_write_scale" in trainer.excluded_optimizer_paths
        ),
    }
    evaluation = _evaluate_arm(
        model,
        evaluation_data,
        config,
        regime,
        arm,
    )
    return {
        "initialization": dict(initialization_report["initialization"]),
        "optimizer": dict(initialization_report["optimizer"]),
        "compiler": dict(initialization_report["compiler"]),
        "training": training,
        "parameter_movement": movement,
        "evaluation": evaluation,
        "metrics": {},
    }


def _schedule_identity_report(config: GateCConfig) -> dict[str, Any]:
    """Return the preregistered schedule identities for both regimes."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    return {
        "gate_a": dict(_GATE_A_SCHEDULE_SHA256),
        "gate_b": {
            "training_global_sha256": dict(
                gate_b._PRODUCTION_ENCODED_GLOBAL_SHA256
            ),
            "validation_sha256": dict(gate_b._PRODUCTION_VALIDATION_SHA256),
        },
    }


def _actual_schedule_identity_report(
    config: GateCConfig,
    gate_a_data: legacy.BindingData,
    gate_b_data: tuple[gate_b.DepthSchedule, gate_b.DepthValidationData],
) -> dict[str, Any]:
    """Hash the generated Gate A and Gate B schedule bytes."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if not isinstance(gate_a_data, legacy.BindingData):
        raise TypeError("Gate C schedule evidence requires BindingData")
    if (
        not isinstance(gate_b_data, tuple)
        or len(gate_b_data) != 2
        or not isinstance(gate_b_data[0], gate_b.DepthSchedule)
        or not isinstance(gate_b_data[1], gate_b.DepthValidationData)
    ):
        raise TypeError("Gate C schedule evidence requires Gate B data")
    schedule, validation = gate_b_data
    gate_b_training = gate_b._encoded_schedule_report(
        schedule,
        config.gate_b_config,
    )
    return {
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
            "training_global_sha256": dict(gate_b_training["global_sha256"]),
            "validation_sha256": dict(
                gate_b._validation_data_report(validation)["sha256"]
            ),
        },
    }


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _metric_summary(
    gate_a_evaluation: Mapping[str, Any],
    gate_b_evaluation: Mapping[str, Any],
) -> dict[str, float]:
    """Compute Gate C binding and demonstrated-depth metrics."""

    try:
        binding = gate_a_evaluation["depths"]["1"]
        intact = _finite_real(binding["intact"]["accuracy"], "Gate A intact accuracy")
        shuffled = _finite_real(
            binding["shuffled"]["accuracy"], "Gate A shuffled accuracy"
        )
        depth_values = [
            _finite_real(
                gate_b_evaluation["efforts"][str(effort)]["intact"]["accuracy"],
                f"Gate B effort {effort} intact accuracy",
            )
            for effort in gate_b.QUALIFYING_EFFORTS
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("evaluation evidence is incomplete") from error
    for value in (intact, shuffled, *depth_values):
        if not 0.0 <= value <= 1.0:
            raise ValueError("accuracy evidence must lie in [0, 1]")
    return {
        "binding_gap": intact - shuffled,
        "depth_accuracy": math.fsum(depth_values) / len(depth_values),
    }


def _decimal_margin(full: float, arm: float) -> Decimal:
    return Decimal(str(full)) - Decimal(str(arm))


def _blocking_margin_report(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute all blocking and characterization-only Gate C margins."""

    if set(metrics) != set(ARM_ORDER):
        raise ValueError("metrics must contain exactly the five Gate C arms")
    values: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        item = metrics[arm]
        values[arm] = {
            name: _finite_real(item[name], f"{arm} {name}")
            for name in ("binding_gap", "depth_accuracy")
        }
    full = values["full"]

    def comparison(
        arm: str,
        *,
        depth_threshold: float,
        binding_threshold: float,
        blocking: bool = True,
    ) -> dict[str, Any]:
        depth_margin = full["depth_accuracy"] - values[arm]["depth_accuracy"]
        binding_margin = full["binding_gap"] - values[arm]["binding_gap"]
        depth_passed = _decimal_margin(
            full["depth_accuracy"], values[arm]["depth_accuracy"]
        ) >= Decimal(str(depth_threshold))
        binding_passed = _decimal_margin(
            full["binding_gap"], values[arm]["binding_gap"]
        ) >= Decimal(str(binding_threshold))
        return {
            "binding_gap_difference": binding_margin,
            "depth_accuracy_difference": depth_margin,
            "binding_threshold": binding_threshold,
            "depth_threshold": depth_threshold,
            "binding_passed": bool(binding_passed),
            "depth_passed": bool(depth_passed),
            "blocking": blocking,
            "passed": bool(binding_passed and depth_passed),
        }

    query = comparison("query_only", depth_threshold=0.15, binding_threshold=-0.02)
    terminal = comparison(
        "terminal_only", depth_threshold=0.10, binding_threshold=-0.02
    )
    legacy = comparison("legacy", depth_threshold=0.15, binding_threshold=0.25)
    frozen = comparison(
        "frozen_write", depth_threshold=0.05, binding_threshold=0.05, blocking=False
    )
    frozen["write_modulation_necessary"] = frozen["passed"]
    frozen["interpretation"] = (
        "learned_memory_write_modulation_necessary"
        if frozen["passed"]
        else "learned_memory_write_modulation_not_shown_necessary"
    )
    return {
        "query_only": query,
        "terminal_only": terminal,
        "legacy": legacy,
        "frozen_write": frozen,
        "blocking_passed": bool(
            query["passed"] and terminal["passed"] and legacy["passed"]
        ),
    }


def _hidden_state_sha256(model: LatentWorkspaceModel) -> str:
    fields: list[bytes] = [b"example21-gate-c-hidden-state-v1"]
    states = sorted(
        model.states(brainstate.HiddenState).items(),
        key=lambda item: gate_a._path(item[0]),
    )
    for path, state in states:
        fields.append(gate_a._path(path).encode("utf-8"))
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            array = np.ascontiguousarray(np.asarray(u.get_mantissa(leaf)))
            fields.extend(
                (
                    str(index).encode("ascii"),
                    array.dtype.str.encode("ascii"),
                    ",".join(map(str, array.shape)).encode("ascii"),
                    array.tobytes(),
                )
            )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


_GATE_C2_FLOAT_DIFFERENCE_KEYS = {
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
_GATE_C2_ENDPOINT_KEYS = {
    "dtype",
    "shape",
    "sha256",
    "value_count",
    "per_example_finite_count",
    "per_example_nonfinite_count",
    "finite_count",
    "nonfinite_count",
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
_GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY = {
    path: (1, *shape[1:])
    for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
}
_GATE_C2_SHORT_TERM_DIAGNOSTIC_PATHS = {
    "memory_read_active#0",
    "memory_read_count#0",
    "memory_read_step#0",
}
_GATE_C2_H0_AUXILIARY_HIDDEN_PATHS = {"memory_drive#0"}


def _gate_c2_array_endpoint(value: Any) -> dict[str, Any]:
    array = np.asarray(u.get_mantissa(value))
    if array.ndim < 1 or array.shape[0] <= 0:
        raise ValueError("Gate C2 evidence arrays require a nonempty batch axis")
    array = np.ascontiguousarray(array)
    flat = array.reshape(array.shape[0], -1)
    finite = np.isfinite(flat)
    per_finite = np.sum(finite, axis=1, dtype=np.int64)
    per_nonfinite = flat.shape[1] - per_finite
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": legacy._digest_arrays(array),
        "value_count": int(array.size),
        "per_example_finite_count": per_finite.tolist(),
        "per_example_nonfinite_count": per_nonfinite.tolist(),
        "finite_count": int(per_finite.sum()),
        "nonfinite_count": int(per_nonfinite.sum()),
    }


def _gate_c2_floating_difference_record(
    left: Any,
    right: Any,
    *,
    rms_tolerance: float,
) -> dict[str, Any]:
    """Build independently recomputable per-example floating evidence."""

    tolerance = _finite_real(rms_tolerance, "Gate C2 RMS tolerance")
    if tolerance < 0.0:
        raise ValueError("rms_tolerance must be nonnegative")
    left_array = np.ascontiguousarray(np.asarray(u.get_mantissa(left)))
    right_array = np.ascontiguousarray(np.asarray(u.get_mantissa(right)))
    if (
        left_array.shape != right_array.shape
        or left_array.dtype != right_array.dtype
        or left_array.ndim < 1
        or not np.issubdtype(left_array.dtype, np.floating)
    ):
        raise ValueError("Gate C2 floating endpoints need equal floating geometry")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("Gate C2 floating endpoints must be finite")
    difference = left_array.astype(np.float64) - right_array.astype(np.float64)
    flat = difference.reshape(difference.shape[0], -1)
    squared = np.square(flat)
    per_sum = np.sum(squared, axis=1, dtype=np.float64)
    per_count = np.full(flat.shape[0], flat.shape[1], dtype=np.int64)
    per_rms = np.sqrt(per_sum / per_count)
    per_max = np.max(np.abs(flat), axis=1)
    total_sum = float(np.sum(per_sum, dtype=np.float64))
    rms = math.sqrt(total_sum / int(per_count.sum()))
    maximum_rms = float(np.max(per_rms))
    maximum_abs = float(np.max(per_max))
    return {
        "left": _gate_c2_array_endpoint(left_array),
        "right": _gate_c2_array_endpoint(right_array),
        "per_example_sum_squared_difference": per_sum.tolist(),
        "per_example_compared_value_count": per_count.tolist(),
        "per_example_rms_difference": per_rms.tolist(),
        "per_example_max_abs_difference": per_max.tolist(),
        "sum_squared_difference": total_sum,
        "rms_difference": rms,
        "max_per_example_rms_difference": maximum_rms,
        "max_abs_difference": maximum_abs,
        "within_tolerance": bool(maximum_rms <= tolerance),
    }


def _gate_c2_endpoint_complete(
    value: Any,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != _GATE_C2_ENDPOINT_KEYS:
        return False
    batch = expected_shape[0]
    per_count = math.prod(expected_shape[1:])
    finite = value["per_example_finite_count"]
    nonfinite = value["per_example_nonfinite_count"]
    return bool(
        value["dtype"] == expected_dtype
        and value["shape"] == list(expected_shape)
        and _sha256_complete(value["sha256"])
        and _strict_integer(value["value_count"])
        and int(value["value_count"]) == math.prod(expected_shape)
        and isinstance(finite, list)
        and len(finite) == batch
        and all(
            _strict_integer(item) and int(item) == per_count for item in finite
        )
        and isinstance(nonfinite, list)
        and len(nonfinite) == batch
        and all(_strict_integer(item) and int(item) == 0 for item in nonfinite)
        and _strict_integer(value["finite_count"])
        and int(value["finite_count"]) == math.prod(expected_shape)
        and _strict_integer(value["nonfinite_count"])
        and int(value["nonfinite_count"]) == 0
    )


def _gate_c2_floating_difference_record_complete(
    value: Any,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    rms_tolerance: float,
    require_within: bool = True,
) -> bool:
    """Recompute every retained floating-difference aggregate."""

    try:
        tolerance = _finite_real(rms_tolerance, "Gate C2 RMS tolerance")
        if (
            tolerance < 0.0
            or not isinstance(value, Mapping)
            or set(value) != _GATE_C2_FLOAT_DIFFERENCE_KEYS
            or not _gate_c2_endpoint_complete(
                value["left"],
                expected_dtype=expected_dtype,
                expected_shape=expected_shape,
            )
            or not _gate_c2_endpoint_complete(
                value["right"],
                expected_dtype=expected_dtype,
                expected_shape=expected_shape,
            )
        ):
            return False
        batch = expected_shape[0]
        compared_count = math.prod(expected_shape[1:])
        per_sum = value["per_example_sum_squared_difference"]
        per_counts = value["per_example_compared_value_count"]
        per_rms = value["per_example_rms_difference"]
        per_max = value["per_example_max_abs_difference"]
        if not all(
            isinstance(section, list) and len(section) == batch
            for section in (per_sum, per_counts, per_rms, per_max)
        ):
            return False
        if not all(
            _strict_integer(item) and int(item) == compared_count
            for item in per_counts
        ):
            return False
        sums = [
            _finite_real(item, "per-example squared difference")
            for item in per_sum
        ]
        rms_values = [
            _finite_real(item, "per-example RMS difference")
            for item in per_rms
        ]
        maxima = [
            _finite_real(item, "per-example maximum difference")
            for item in per_max
        ]
        if any(item < 0.0 for item in (*sums, *rms_values, *maxima)):
            return False
        if any(
            not math.isclose(
                actual,
                math.sqrt(squared / compared_count),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for squared, actual in zip(sums, rms_values, strict=True)
        ):
            return False
        if any(maximum + 1e-15 < rms for maximum, rms in zip(maxima, rms_values)):
            return False
        total_sum = math.fsum(sums)
        total_rms = math.sqrt(total_sum / (batch * compared_count))
        maximum_rms = max(rms_values)
        maximum_abs = max(maxima)
        if not all(
            math.isclose(
                _finite_real(value[field], field),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for field, expected in (
                ("sum_squared_difference", total_sum),
                ("rms_difference", total_rms),
                ("max_per_example_rms_difference", maximum_rms),
                ("max_abs_difference", maximum_abs),
            )
        ):
            return False
        expected_within = maximum_rms <= tolerance
        return bool(
            isinstance(value["within_tolerance"], bool)
            and value["within_tolerance"] is expected_within
            and (expected_within or not require_within)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _gate_c2_prediction_endpoint(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    if array.ndim != 1 or array.shape[0] <= 0:
        raise ValueError("Gate C2 predictions must be one nonempty vector")
    if not np.logical_and(array >= 0, array < 10).all():
        raise ValueError("Gate C2 predictions must lie in 0..9")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": legacy._digest_arrays(array),
        "histogram": np.bincount(array, minlength=10).tolist(),
        "count": int(array.size),
    }


def _gate_c2_prediction_difference_record(
    left: Any,
    right: Any,
) -> dict[str, Any]:
    """Build exact decoded-prediction equality evidence."""

    left_array = np.ascontiguousarray(np.asarray(left, dtype=np.int32))
    right_array = np.ascontiguousarray(np.asarray(right, dtype=np.int32))
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("Gate C2 prediction endpoints need equal vectors")
    hamming = np.not_equal(left_array, right_array).astype(np.int64)
    count = int(hamming.sum())
    return {
        "left": _gate_c2_prediction_endpoint(left_array),
        "right": _gate_c2_prediction_endpoint(right_array),
        "per_example_hamming_count": hamming.tolist(),
        "hamming_count": count,
        "equal": bool(count == 0),
    }


def _gate_c2_prediction_difference_record_complete(
    value: Any,
    *,
    count: int,
    require_equal: bool = True,
) -> bool:
    """Validate decoded predictions and recompute Hamming equality."""

    endpoint_keys = {"dtype", "shape", "sha256", "histogram", "count"}
    if (
        not _strict_integer(count)
        or int(count) <= 0
        or not isinstance(value, Mapping)
        or set(value)
        != {
            "left",
            "right",
            "per_example_hamming_count",
            "hamming_count",
            "equal",
        }
    ):
        return False
    expected_count = int(count)
    endpoints = []
    for side in ("left", "right"):
        endpoint = value[side]
        if (
            not isinstance(endpoint, Mapping)
            or set(endpoint) != endpoint_keys
            or endpoint["dtype"] != "<i4"
            or endpoint["shape"] != [expected_count]
            or not _sha256_complete(endpoint["sha256"])
            or not _strict_integer(endpoint["count"])
            or int(endpoint["count"]) != expected_count
            or not isinstance(endpoint["histogram"], list)
            or len(endpoint["histogram"]) != 10
            or not all(
                _strict_integer(item) and int(item) >= 0
                for item in endpoint["histogram"]
            )
            or sum(int(item) for item in endpoint["histogram"])
            != expected_count
        ):
            return False
        endpoints.append(endpoint)
    per_example = value["per_example_hamming_count"]
    if (
        not isinstance(per_example, list)
        or len(per_example) != expected_count
        or not all(_strict_integer(item) and int(item) in (0, 1) for item in per_example)
        or not _strict_integer(value["hamming_count"])
    ):
        return False
    hamming_count = sum(int(item) for item in per_example)
    expected_equal = bool(
        hamming_count == 0
        and endpoints[0]["sha256"] == endpoints[1]["sha256"]
        and endpoints[0]["histogram"] == endpoints[1]["histogram"]
    )
    return bool(
        int(value["hamming_count"]) == hamming_count
        and isinstance(value["equal"], bool)
        and value["equal"] is expected_equal
        and (expected_equal or not require_equal)
    )


def _gate_c2_zero_array_record(value: Any) -> dict[str, Any]:
    """Retain raw geometry and reconstructable digest for one zero array."""

    array = np.ascontiguousarray(np.asarray(u.get_mantissa(value)))
    if array.size == 0 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Gate C2 selected arrays must be nonempty floating arrays")
    finite = np.isfinite(array)
    zero = np.equal(array, 0.0)
    if finite.all():
        squared = float(np.sum(np.square(array.astype(np.float64))))
        maximum = float(np.max(np.abs(array.astype(np.float64))))
    else:
        raise ValueError("Gate C2 selected arrays must be finite")
    exact_zero = bool(zero.all())
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": legacy._digest_arrays(array),
        "value_count": int(array.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(array.size - finite.sum()),
        "zero_count": int(zero.sum()),
        "sum_of_squares": squared,
        "max_abs": maximum,
        "exact_zero": exact_zero,
    }


def _gate_c2_zero_array_record_complete(
    value: Any,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    require_zero: bool = True,
) -> bool:
    """Validate a retained array summary and optionally require exact zero."""

    keys = {
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
    if not isinstance(value, Mapping) or set(value) != keys:
        return False
    try:
        count = math.prod(expected_shape)
        zero = np.zeros(expected_shape, dtype=np.dtype(expected_dtype))
        zero_count = int(value["zero_count"])
        squared = _finite_real(value["sum_of_squares"], "zero sum")
        maximum = _finite_real(value["max_abs"], "zero maximum")
        exact_zero = bool(
            zero_count == count and squared == 0.0 and maximum == 0.0
        )
        return bool(
            value["dtype"] == expected_dtype
            and value["shape"] == list(expected_shape)
            and _sha256_complete(value["sha256"])
            and _strict_integer(value["value_count"])
            and int(value["value_count"]) == count
            and _strict_integer(value["finite_count"])
            and int(value["finite_count"]) == count
            and _strict_integer(value["nonfinite_count"])
            and int(value["nonfinite_count"]) == 0
            and _strict_integer(value["zero_count"])
            and 0 <= zero_count <= count
            and squared >= 0.0
            and maximum >= 0.0
            and (maximum * maximum <= squared or count == 0)
            and (exact_zero or (zero_count < count and squared > 0.0 and maximum > 0.0))
            and exact_zero is (zero_count == count)
            and isinstance(value["exact_zero"], bool)
            and value["exact_zero"] is exact_zero
            and (
                (exact_zero and value["sha256"] == legacy._digest_arrays(zero))
                or (not exact_zero and value["sha256"] != legacy._digest_arrays(zero))
            )
            and (exact_zero or not require_zero)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _gate_c2_no_update_evidence_complete(
    value: Any,
    initialization: Mapping[str, Any],
    *,
    model_roles: Mapping[str, Mapping[str, str]] | None = None,
) -> bool:
    """Recompute the zero-training and parameter-identity control evidence."""

    keys = {
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
    role_keys = {
        "regime",
        "probe",
        "policy",
        "initialization_tree",
        "expected_parameter_sha256",
        "before_parameter_sha256",
        "after_parameter_sha256",
        "parameters_equal",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        return False
    if value["instrumented_training_entry_points"] != list(
        GATE_C2_CONTROLS_AUDIT_LABELS
    ):
        return False
    for calls, count_name in (
        ("trainer_factory_calls", "trainer_factory_call_count"),
        ("training_step_calls", "training_step_call_count"),
        ("optimizer_update_calls", "optimizer_update_call_count"),
    ):
        if (
            value[calls] != []
            or not _strict_integer(value[count_name])
            or int(value[count_name]) != 0
        ):
            return False
    if (
        value["optimizer_constructor_calls"] != []
        or not _strict_integer(value["optimizer_instance_count"])
        or int(value["optimizer_instance_count"]) != 0
    ):
        return False
    factories = value["model_factory_calls"]
    constructors = value["model_constructor_calls"]
    roles = value["materialized_roles"]
    expected_roles = model_roles or GATE_C2_CONTROLS_MODEL_ROLES
    expected_role_names = list(expected_roles)
    if (
        not isinstance(factories, list)
        or factories != expected_role_names
        or factories != constructors
        or len(factories) != len(set(factories))
        or not isinstance(roles, Mapping)
        or set(roles) != set(expected_role_names)
    ):
        return False
    for role_name in factories:
        role = roles[role_name]
        expected_role = expected_roles[role_name]
        if (
            not isinstance(role_name, str)
            or not role_name
            or not isinstance(role, Mapping)
            or set(role) != role_keys
            or role["regime"] != expected_role["regime"]
            or role["probe"] != expected_role["probe"]
            or role["policy"] != expected_role["policy"]
            or role["initialization_tree"] != "canonical_full"
        ):
            return False
        try:
            expected = initialization["initialization"][role["regime"]][
                "canonical_full"
            ]["parameter_sha256"]
        except (KeyError, TypeError):
            return False
        if not (
            _sha256_complete(expected)
            and role["expected_parameter_sha256"] == expected
            and role["before_parameter_sha256"] == expected
            and role["after_parameter_sha256"] == expected
            and role["parameters_equal"] is True
        ):
            return False
    return value["complete"] is True


def _gate_c2_paired_h0_operational_equivalence_complete(
    value: Any,
    initialization: Mapping[str, Any],
    *,
    regime: str,
    require_pass: bool = True,
) -> bool:
    """Validate all same-replay, full/full, and full/query H0 controls."""

    try:
        _regime_spec(regime)
        expected_keys = {
            "backend",
            "checkpoint",
            "intervention_boundary",
            "rms_tolerance",
            "initialization_parameter_sha256",
            "streams",
            "passed",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            return False
        if (
            value["backend"] != "canonical_production_sparse"
            or not _strict_integer(value["checkpoint"])
            or int(value["checkpoint"]) != 0
            or value["intervention_boundary"]
            != "after_ordinary_query_h0_before_first_latent_tick_h1"
            or _finite_real(value["rms_tolerance"], "H0 tolerance") != 1e-6
        ):
            return False
        roles = {"full_reference", "full_replay", "copied_full", "query_only"}
        canonical = initialization["initialization"][regime]["canonical_full"][
            "parameter_sha256"
        ]
        parameter_sha = value["initialization_parameter_sha256"]
        if (
            not isinstance(parameter_sha, Mapping)
            or set(parameter_sha) != roles
            or not _sha256_complete(canonical)
            or any(digest != canonical for digest in parameter_sha.values())
        ):
            return False
        streams = value["streams"]
        if not isinstance(streams, Mapping) or set(streams) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        comparison_names = {
            "same_full_replay",
            "copied_full_separate_jit",
            "copied_full_vs_query",
        }
        comparison_keys = {"compact", "hidden_paths", "predictions", "passed"}
        stream_passes = []
        for stream in streams.values():
            if not isinstance(stream, Mapping) or set(stream) != {
                "initial_state_sha256",
                "comparisons",
                "passed",
            }:
                return False
            initial_sha = stream["initial_state_sha256"]
            if (
                not isinstance(initial_sha, Mapping)
                or set(initial_sha) != roles
                or not all(_sha256_complete(item) for item in initial_sha.values())
                or len(set(initial_sha.values())) != 1
            ):
                return False
            comparisons = stream["comparisons"]
            if not isinstance(comparisons, Mapping) or set(comparisons) != comparison_names:
                return False
            comparison_passes = []
            for comparison in comparisons.values():
                if (
                    not isinstance(comparison, Mapping)
                    or set(comparison) != comparison_keys
                    or not _gate_c2_floating_difference_record_complete(
                        comparison["compact"],
                        expected_dtype="<f4",
                        expected_shape=(512, 1_180),
                        rms_tolerance=1e-6,
                        require_within=False,
                    )
                    or not isinstance(comparison["hidden_paths"], Mapping)
                    or set(comparison["hidden_paths"])
                    != set(_GATE_C2_HIDDEN_GEOMETRY)
                    or any(
                        not _gate_c2_floating_difference_record_complete(
                            comparison["hidden_paths"][path],
                            expected_dtype="<f4",
                            expected_shape=shape,
                            rms_tolerance=1e-6,
                            require_within=False,
                        )
                        for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items()
                    )
                    or not _gate_c2_prediction_difference_record_complete(
                        comparison["predictions"],
                        count=512,
                        require_equal=False,
                    )
                ):
                    return False
                comparison_pass = bool(
                    comparison["compact"]["within_tolerance"] is True
                    and all(
                        record["within_tolerance"] is True
                        for record in comparison["hidden_paths"].values()
                    )
                    and comparison["predictions"]["equal"] is True
                )
                if comparison["passed"] is not comparison_pass:
                    return False
                comparison_passes.append(comparison_pass)
            stream_pass = all(comparison_passes)
            if stream["passed"] is not stream_pass:
                return False
            stream_passes.append(stream_pass)
        passed = all(stream_passes)
        return bool(value["passed"] is passed and (passed or not require_pass))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class _GateC2ControlsAudit:
    """Collect model identity and prove that no training boundary was entered."""

    def __init__(
        self,
        initialization: Mapping[str, Any],
        *,
        model_roles: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.initialization = initialization
        self.model_roles = model_roles or GATE_C2_CONTROLS_MODEL_ROLES
        self.model_factory_calls: list[str] = []
        self.model_constructor_calls: list[str] = []
        self.materialized_roles: dict[str, dict[str, Any]] = {}
        self.trainer_factory_calls: list[str] = []
        self.training_step_calls: list[str] = []
        self.optimizer_constructor_calls: list[str] = []
        self.optimizer_update_calls: list[str] = []
        self._restorations: list[tuple[Any, str, Any]] = []

    def _replace(self, owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name)
        self._restorations.append((owner, name, original))
        setattr(owner, name, replacement)

    def _restore(self) -> None:
        while self._restorations:
            owner, name, original = self._restorations.pop()
            setattr(owner, name, original)

    def __enter__(self) -> _GateC2ControlsAudit:
        try:
            return self._install()
        except BaseException:
            self._restore()
            raise

    def _install(self) -> _GateC2ControlsAudit:
        local_module = sys.modules[__name__]
        factory_entries = (
            (
                local_module,
                "_make_arm_trainer",
                "examples.pp_prop.latent_workspace_ablation_gate._make_arm_trainer",
            ),
            (
                gate_a,
                "_make_pp_prop_trainer",
                "examples.pp_prop.latent_workspace_binding_gate._make_pp_prop_trainer",
            ),
            (
                gate_b,
                "_make_pp_prop_trainer",
                "examples.pp_prop.latent_workspace_depth_gate._make_pp_prop_trainer",
            ),
        )
        for owner, name, label in factory_entries:
            original = getattr(owner, name)

            def blocked_factory(
                *args: Any,
                _label: str = label,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                del args, kwargs, _original
                self.trainer_factory_calls.append(_label)
                raise RuntimeError(
                    "Gate C2 pretraining controls forbid trainer construction"
                )

            self._replace(owner, name, blocked_factory)

        trainer_classes = (
            (
                local_module,
                "GateCTrainer",
                "train_chunk",
                (
                    "examples.pp_prop.latent_workspace_ablation_gate."
                    "GateCTrainer.train_chunk"
                ),
            ),
            (
                gate_a,
                "_PPPropTrainer",
                "train",
                (
                    "examples.pp_prop.latent_workspace_binding_gate."
                    "_PPPropTrainer.train"
                ),
            ),
            (
                gate_b,
                "_DepthPPPropTrainer",
                "train_chunk",
                (
                    "examples.pp_prop.latent_workspace_depth_gate."
                    "_DepthPPPropTrainer.train_chunk"
                ),
            ),
        )
        for owner, name, callable_name, label in trainer_classes:
            original_class = getattr(owner, name)

            def audited_constructor(
                *args: Any,
                _class: Any = original_class,
                _callable_name: str = callable_name,
                _label: str = label,
                **kwargs: Any,
            ) -> Any:
                instance = _class(*args, **kwargs)

                def blocked_step(*step_args: Any, **step_kwargs: Any) -> Any:
                    del step_args, step_kwargs
                    self.training_step_calls.append(_label)
                    raise RuntimeError(
                        "Gate C2 pretraining controls forbid training steps"
                    )

                setattr(instance, _callable_name, blocked_step)
                return instance

            self._replace(owner, name, audited_constructor)

        try:
            original_adam = braintools.optim.Adam

            def blocked_adam(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                self.optimizer_constructor_calls.append(
                    "braintools.optim.Adam.__init__"
                )
                raise RuntimeError(
                    "Gate C2 pretraining controls forbid Adam construction"
                )

            self._replace(braintools.optim, "Adam", blocked_adam)
            original_update = getattr(original_adam, "update", None)
            if original_update is not None:

                def blocked_update(instance: Any, *args: Any, **kwargs: Any) -> Any:
                    del instance, args, kwargs
                    self.optimizer_update_calls.append(
                        "braintools.optim.Adam.update"
                    )
                    raise RuntimeError(
                        "Gate C2 pretraining controls forbid optimizer updates"
                    )

                self._replace(original_adam, "update", blocked_update)
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self._restore()

    def register(
        self,
        role: str,
        model: LatentWorkspaceModel,
        *,
        regime: str,
        probe: str,
        policy: str,
    ) -> None:
        if role in self.materialized_roles:
            raise RuntimeError(f"duplicate Gate C2 control model role {role!r}")
        expected = self.initialization["initialization"][regime][
            "canonical_full"
        ]["parameter_sha256"]
        before = legacy._array_digest(legacy._parameter_values(model))
        if before != expected:
            raise ValueError("Gate C2 control model did not reproduce initialization")
        self.model_factory_calls.append(role)
        self.model_constructor_calls.append(role)
        self.materialized_roles[role] = {
            "regime": regime,
            "probe": probe,
            "policy": policy,
            "initialization_tree": "canonical_full",
            "expected_parameter_sha256": expected,
            "before_parameter_sha256": before,
            "after_parameter_sha256": None,
            "parameters_equal": False,
        }

    def finish(self, role: str, model: LatentWorkspaceModel) -> None:
        record = self.materialized_roles[role]
        after = legacy._array_digest(legacy._parameter_values(model))
        record["after_parameter_sha256"] = after
        record["parameters_equal"] = bool(
            after
            == record["before_parameter_sha256"]
            == record["expected_parameter_sha256"]
        )

    def report(self) -> dict[str, Any]:
        expected_names = sorted(self.model_roles)
        factories = sorted(self.model_factory_calls)
        constructors = sorted(self.model_constructor_calls)
        roles = dict(sorted(self.materialized_roles.items()))
        finished = bool(
            list(roles) == expected_names
            and factories == expected_names
            and constructors == expected_names
            and len(self.model_factory_calls) == len(set(self.model_factory_calls))
            and len(self.model_constructor_calls)
            == len(set(self.model_constructor_calls))
            and all(
                role["parameters_equal"] is True
                and _sha256_complete(role["after_parameter_sha256"])
                for role in roles.values()
            )
        )
        no_calls = not any(
            (
                self.trainer_factory_calls,
                self.training_step_calls,
                self.optimizer_constructor_calls,
                self.optimizer_update_calls,
            )
        )
        return {
            "instrumented_training_entry_points": list(
                GATE_C2_CONTROLS_AUDIT_LABELS
            ),
            "trainer_factory_calls": list(self.trainer_factory_calls),
            "trainer_factory_call_count": len(self.trainer_factory_calls),
            "training_step_calls": list(self.training_step_calls),
            "training_step_call_count": len(self.training_step_calls),
            "optimizer_constructor_calls": list(
                self.optimizer_constructor_calls
            ),
            "optimizer_instance_count": len(self.optimizer_constructor_calls),
            "optimizer_update_calls": list(self.optimizer_update_calls),
            "optimizer_update_call_count": len(self.optimizer_update_calls),
            "model_factory_calls": factories,
            "model_constructor_calls": constructors,
            "materialized_roles": roles,
            "complete": bool(finished and no_calls),
        }


_ACTIVE_GATE_C2_CONTROLS_AUDIT: _GateC2ControlsAudit | None = None


def _gate_c2_control_model(
    config: GateCConfig,
    initialization: Mapping[str, Any],
    *,
    regime: str,
    policy: str,
    batch_size: int,
    role: str,
    probe: str,
    arm: str | None = None,
    constructor: Any = None,
) -> LatentWorkspaceModel:
    selected_arm = arm or ("query_only" if policy == "query_only" else "full")
    if constructor is None:
        model = _new_model_for_arm(
            config,
            regime,
            selected_arm,
            batch_size=batch_size,
        )
    else:
        model_config = _model_config_for_arm(
            config,
            regime,
            selected_arm,
            batch_size=batch_size,
        )
        model = constructor(model_config, policy)
        if not isinstance(model, LatentWorkspaceModel):
            raise TypeError("Gate C2 control constructor must return a model")
    expected = initialization["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]
    actual = legacy._array_digest(legacy._parameter_values(model))
    if actual != expected:
        raise ValueError("Gate C2 control model initialization differs")
    if _ACTIVE_GATE_C2_CONTROLS_AUDIT is not None:
        _ACTIVE_GATE_C2_CONTROLS_AUDIT.register(
            role,
            model,
            regime=regime,
            probe=probe,
            policy=policy,
        )
    return model


def _gate_c2_finish_control_model(
    role: str,
    model: LatentWorkspaceModel,
) -> None:
    if _ACTIVE_GATE_C2_CONTROLS_AUDIT is not None:
        _ACTIVE_GATE_C2_CONTROLS_AUDIT.finish(role, model)


def _gate_c2_hidden_arrays(model: LatentWorkspaceModel) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    states = sorted(
        model.states(brainstate.HiddenState).items(),
        key=lambda item: gate_a._path(item[0]),
    )
    for path, state in states:
        name = gate_a._path(path)
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            values[f"{name}#{index}"] = np.ascontiguousarray(
                np.asarray(u.get_mantissa(jax.device_get(leaf)))
            )
    return values


def _gate_c2_h0_driver(model: LatentWorkspaceModel) -> Any:
    @brainstate.transform.jit
    def run_h0(events: jax.Array, advances: jax.Array) -> jax.Array:
        def step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
            event, advance = inputs
            return model.update(event, advance)

        compact = brainstate.transform.for_loop(step, (events, advances))
        return compact[-1]

    return run_h0


def _gate_c2_h0_capture(
    model: LatentWorkspaceModel,
    driver: Any,
    snapshot: Any,
    events: np.ndarray,
    advances: np.ndarray,
) -> dict[str, Any]:
    model.restore_state(snapshot)
    compact = np.ascontiguousarray(
        np.asarray(
            jax.device_get(
                jax.block_until_ready(
                    driver(jnp.asarray(events), jnp.asarray(advances))
                )
            )
        )
    )
    color_logits = np.asarray(
        legacy._color_logits(jnp.asarray(compact), model.config.color_rank)
    )
    predictions = np.argmax(color_logits, axis=-1).astype(np.int32, copy=False)
    return {
        "compact": compact,
        "hidden_paths": _gate_c2_hidden_arrays(model),
        "predictions": np.ascontiguousarray(predictions),
    }


def _gate_c2_h0_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    compact = _gate_c2_floating_difference_record(
        left["compact"],
        right["compact"],
        rms_tolerance=1e-6,
    )
    left_hidden = left["hidden_paths"]
    right_hidden = right["hidden_paths"]
    if set(left_hidden) != set(right_hidden):
        raise ValueError("Gate C2 H0 hidden-state paths differ")
    hidden = {
        path: _gate_c2_floating_difference_record(
            left_hidden[path],
            right_hidden[path],
            rms_tolerance=1e-6,
        )
        for path in sorted(left_hidden)
    }
    predictions = _gate_c2_prediction_difference_record(
        left["predictions"],
        right["predictions"],
    )
    passed = bool(
        compact["within_tolerance"] is True
        and all(record["within_tolerance"] is True for record in hidden.values())
        and predictions["equal"] is True
    )
    return {
        "compact": compact,
        "hidden_paths": hidden,
        "predictions": predictions,
        "passed": passed,
    }


def _paired_h0_operational_equivalence_report(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    regime: str,
    data: Any,
) -> dict[str, Any]:
    """Measure H0 replay, full/full, and full/query numerical controls."""

    _regime_spec(regime)
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    count = regime_config.validation_episodes
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A H0 controls require BindingData")
        streams = {
            "intact": np.asarray(data.validation_intact),
            "shuffled": np.asarray(data.validation_shuffled),
            "no_context": np.asarray(data.validation_no_context),
        }
        advances = np.ones(
            (regime_config.sequence_length, count),
            dtype=np.bool_,
        )
    else:
        if (
            not isinstance(data, tuple)
            or len(data) != 2
            or not isinstance(data[1], gate_b.DepthValidationData)
        ):
            raise TypeError("Gate B H0 controls require DepthValidationData")
        validation = data[1]
        streams = {
            "intact": np.asarray(validation.intact),
            "shuffled": np.asarray(validation.shuffled),
            "no_context": np.asarray(validation.no_context),
        }
        advances = np.asarray(validation.advance_masks)
    checkpoint_index = regime_config.sequence_length - regime_config.gap_steps - 1
    roles = {
        "full_reference": f"{regime}:paired_h0:full_reference",
        "copied_full": f"{regime}:paired_h0:copied_full",
        "query_only": f"{regime}:paired_h0:query_only",
    }
    models = {
        key: _gate_c2_control_model(
            config,
            initialization,
            regime=regime,
            policy="query_only" if key == "query_only" else "full",
            batch_size=count,
            role=role,
            probe="paired_h0_operational_equivalence",
        )
        for key, role in roles.items()
    }
    legacy._copy_parameters(models["full_reference"], models["copied_full"])
    legacy._copy_parameters(models["full_reference"], models["query_only"])
    initial_snapshot = models["full_reference"].snapshot_state()
    for model in (models["copied_full"], models["query_only"]):
        model.restore_state(initial_snapshot)
    drivers = {name: _gate_c2_h0_driver(model) for name, model in models.items()}
    parameter_sha = {
        "full_reference": legacy._array_digest(
            legacy._parameter_values(models["full_reference"])
        ),
        "full_replay": legacy._array_digest(
            legacy._parameter_values(models["full_reference"])
        ),
        "copied_full": legacy._array_digest(
            legacy._parameter_values(models["copied_full"])
        ),
        "query_only": legacy._array_digest(
            legacy._parameter_values(models["query_only"])
        ),
    }
    stream_reports: dict[str, Any] = {}
    prefix_advances = advances[: checkpoint_index + 1]
    for stream_name, values in streams.items():
        prefix = values[: checkpoint_index + 1]
        for model in models.values():
            model.restore_state(initial_snapshot)
        initial_state_sha = {
            "full_reference": _hidden_state_sha256(models["full_reference"]),
            "full_replay": _hidden_state_sha256(models["full_reference"]),
            "copied_full": _hidden_state_sha256(models["copied_full"]),
            "query_only": _hidden_state_sha256(models["query_only"]),
        }
        first = _gate_c2_h0_capture(
            models["full_reference"],
            drivers["full_reference"],
            initial_snapshot,
            prefix,
            prefix_advances,
        )
        replay = _gate_c2_h0_capture(
            models["full_reference"],
            drivers["full_reference"],
            initial_snapshot,
            prefix,
            prefix_advances,
        )
        copied = _gate_c2_h0_capture(
            models["copied_full"],
            drivers["copied_full"],
            initial_snapshot,
            prefix,
            prefix_advances,
        )
        query = _gate_c2_h0_capture(
            models["query_only"],
            drivers["query_only"],
            initial_snapshot,
            prefix,
            prefix_advances,
        )
        comparisons = {
            "same_full_replay": _gate_c2_h0_comparison(first, replay),
            "copied_full_separate_jit": _gate_c2_h0_comparison(first, copied),
            "copied_full_vs_query": _gate_c2_h0_comparison(first, query),
        }
        stream_reports[stream_name] = {
            "initial_state_sha256": initial_state_sha,
            "comparisons": comparisons,
            "passed": all(item["passed"] is True for item in comparisons.values()),
        }
    for key, role in roles.items():
        _gate_c2_finish_control_model(role, models[key])
    return {
        "backend": "canonical_production_sparse",
        "checkpoint": 0,
        "intervention_boundary": (
            "after_ordinary_query_h0_before_first_latent_tick_h1"
        ),
        "rms_tolerance": 1e-6,
        "initialization_parameter_sha256": parameter_sha,
        "streams": stream_reports,
        "passed": all(item["passed"] is True for item in stream_reports.values()),
    }


def _gate_c2_raw_array_record(
    value: Any,
    *,
    include_values: bool = False,
    fill_value: float | None = None,
) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(u.get_mantissa(value)))
    report: dict[str, Any] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": legacy._digest_arrays(array),
    }
    if include_values:
        report["values"] = array.tolist()
    if fill_value is not None:
        report["fill_value"] = float(fill_value)
    return report


def _gate_c2_gradient_path_record(
    path: str,
    value: Any,
    *,
    require_zero: bool,
) -> dict[str, Any]:
    leaves = _numeric_gradient_leaves(value)
    leaf_reports = []
    squared = 0.0
    value_count = 0
    zero_count = 0
    for index, array in enumerate(leaves):
        numeric = array.astype(np.float64)
        count = int(array.size)
        zeros = int(np.count_nonzero(array == 0))
        squared += float(np.sum(np.square(numeric)))
        value_count += count
        zero_count += zeros
        leaf_reports.append(
            {
                "index": index,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "value_count": count,
                "finite_count": count,
                "nonfinite_count": 0,
                "zero_count": zeros,
                "sha256": legacy._digest_arrays(array),
            }
        )
    norm = math.sqrt(squared)
    report = {
        "tree_paths": [path],
        "leaf_count": len(leaves),
        "leaves": leaf_reports,
        "value_count": value_count,
        "l2_norm": norm,
        "sha256": _gradient_path_sha256(path, value),
        "finite": True,
        ("exact_zero" if require_zero else "nonzero"): (
            bool(zero_count == value_count and norm == 0.0)
            if require_zero
            else bool(norm > 0.0)
        ),
    }
    return report


def _gate_c2_global_gradient_record(
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    leaves = [
        array
        for path in sorted(gradients)
        for array in _numeric_gradient_leaves(gradients[path])
    ]
    squared = math.fsum(
        float(np.sum(np.square(array.astype(np.float64)))) for array in leaves
    )
    count = sum(int(array.size) for array in leaves)
    return {
        "tree_paths": sorted(gradients),
        "leaf_count": len(leaves),
        "value_count": count,
        "l2_norm": math.sqrt(squared),
        "sha256": _gradient_global_sha256(gradients),
        "finite": True,
        "nonzero": bool(squared > 0.0),
    }


def _gate_c2_stop_gradient_snapshot(snapshot: Any) -> Any:
    return dataclasses.replace(
        snapshot,
        entries=tuple(
            (
                path,
                jax.tree.map(
                    lambda leaf: jax.lax.stop_gradient(leaf),
                    value,
                ),
            )
            for path, value in snapshot.entries
        ),
    )


def _gate_c2_removed_path_finite_window_influence(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    regime: str,
    data: Any,
) -> dict[str, Any]:
    """Measure removed latent-read paths from a materialized H0 boundary."""

    _regime_spec(regime)
    try:
        canonical_parameter_sha256 = initialization["initialization"][regime][
            "canonical_full"
        ]["parameter_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Gate C2 removed-path initialization is incomplete") from error
    if (
        not _sha256_complete(canonical_parameter_sha256)
        or canonical_parameter_sha256 == "0" * 64
    ):
        raise ValueError("Gate C2 removed-path canonical parameters are invalid")
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A removed-path objective requires BindingData")
        pinned = GATE_C2_REMOVED_PATH_OBJECTIVES["gate_a_h1"]
        mapping_id = int(pinned["source_metadata"]["mapping_id"])
        encoded, encoded_targets, _, _, encoded_queries = (
            legacy._encode_mapping_episodes(
                np.asarray([mapping_id], dtype=np.int64),
                seed=config.gate_a_config.validation_episode_seed,
                config=config.gate_a_config,
                controls=True,
            )
        )
        source_events = np.ascontiguousarray(encoded[0])
        source_advances = np.ones(source_events.shape[0], dtype=np.bool_)
        target = int(np.asarray(encoded_targets[0]).item())
        source_targets = np.asarray([target], dtype=np.int32)
        canonical_weights = np.zeros(source_events.shape[0], dtype=np.float32)
        canonical_weights[-2:] = np.float32(0.5)
        h0_end = source_events.shape[0] - config.gate_a_config.gap_steps
        continuation_events = source_events[h0_end:]
        continuation_advances = source_advances[h0_end:]
        continuation_targets = np.full(
            continuation_events.shape[0], target, dtype=np.int32
        )
        selection = np.zeros(continuation_events.shape[0], dtype=np.bool_)
        selection[-1] = True
        base_weight = np.float32(0.5)
        objective_name = "gate_a_h1"
        coordinates = {
            "regime": "gate_a",
            "stream": "intact",
            "validation_episode_index": 0,
            "batch_size": 1,
            "checkpoint": "H1",
        }
        schedule_cross_bound = dict(pinned["schedule_sha256"])
        input_colors, output_colors = legacy._decode_mapping(mapping_id)
        presentation_orders, query_indices = legacy._episode_choices(
            1,
            config.gate_a_config.validation_episode_seed,
        )
        query_index = int(np.asarray(encoded_queries[0]).item())
        if query_index != int(query_indices[0]) or target != output_colors[query_index]:
            raise ValueError("Gate A query metadata differs from its schedule")
        source_metadata = {
            "mapping_id": mapping_id,
            "input_colors": list(input_colors),
            "output_colors": list(output_colors),
            "presentation_order_indices": presentation_orders[0].tolist(),
            "query_index": query_index,
            "query_color": int(input_colors[query_index]),
            "target": target,
            "demonstration_indices": list(range(len(input_colors))),
            "h0_index": h0_end - 1,
            "h1_index": h0_end,
        }
        if not gate_a._json_exact(source_metadata, pinned["source_metadata"]):
            raise ValueError("Gate A objective metadata differs from its pin")
    else:
        if (
            not isinstance(data, tuple)
            or len(data) != 2
            or not isinstance(data[0], gate_b.DepthSchedule)
            or not isinstance(data[1], gate_b.DepthValidationData)
        ):
            raise TypeError("Gate B removed-path objective requires canonical data")
        schedule, validation = data
        source_events = np.ascontiguousarray(validation.intact[:, 0, :])
        source_advances = np.ascontiguousarray(validation.advance_masks[:, 0])
        source_targets = np.zeros(source_events.shape[0], dtype=np.int32)
        source_targets[10:] = np.asarray(validation.targets_by_depth[:, 0])
        effort = 8
        canonical_weights = np.asarray(
            gate_b._checkpoint_contract(
                gate_b.unrank_ten_cycle(
                    int(np.asarray(schedule.validation_mapping_ids[0]).item())
                ),
                int(np.asarray(schedule.validation_query_colors[0]).item()),
                effort,
            ).loss_weights
        )
        h0_end = 11
        continuation_events = source_events[h0_end:]
        continuation_advances = source_advances[h0_end:]
        continuation_targets = source_targets[h0_end:]
        selection = np.zeros(continuation_events.shape[0], dtype=np.bool_)
        selection[-1] = True
        base_weight = np.float32(1.0 / 9.0)
        objective_name = "gate_b_index0_r8_h8"
        coordinates = {
            "regime": "gate_b",
            "stream": "intact",
            "validation_episode_index": 0,
            "batch_size": 1,
            "effort": 8,
            "checkpoint": "H8",
        }
        schedule_cross_bound = dict(
            gate_b._validation_data_report(validation)["sha256"]
        )
        source_metadata = {
            "mapping_id": int(np.asarray(schedule.validation_mapping_ids[0]).item()),
            "mapping": gate_b.unrank_ten_cycle(
                int(np.asarray(schedule.validation_mapping_ids[0]).item())
            ).tolist(),
            "query_color": int(np.asarray(schedule.validation_query_colors[0]).item()),
            "presentation_order": np.asarray(
                schedule.validation_presentation_orders[0]
            ).tolist(),
            "shuffled_shift": int(np.asarray(validation.shuffled_shifts[0]).item()),
            "h0_through_h8_targets": np.asarray(
                validation.targets_by_depth[:, 0]
            ).tolist(),
        }
    prefix_events = source_events[:h0_end]
    prefix_advances = source_advances[:h0_end]
    effective_weights = np.where(selection, base_weight, np.float32(0.0)).astype(
        np.float32
    )
    packed = np.concatenate(
        (
            continuation_events[:, None, :],
            continuation_advances[:, None, None].astype(np.float32),
            continuation_targets[:, None, None].astype(np.float32),
            effective_weights[:, None, None],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    created_models: list[LatentWorkspaceModel] = []
    snapshots: list[Any] = []
    materialized_prefixes: list[dict[str, Any]] = []
    actual_gradient_starts: list[dict[str, Any]] = []
    gradient_boundary_capture_count = 0
    role = f"{regime}:removed_path_finite_window:{objective_name}"

    class _MaterializedObjective(LatentWorkspaceModel):
        def __init__(self, model_config: ModelConfig) -> None:
            self._gate_c2_h0_snapshot = None
            super().__init__(model_config, memory_read_policy="query_only")

        def reset_state(self, batch_size: int | None = None, **kwargs: object) -> None:
            super().reset_state(batch_size=batch_size, **kwargs)
            if self._gate_c2_h0_snapshot is not None:
                self.restore_state(self._gate_c2_h0_snapshot)

        def update(self, packed_input: jax.Array) -> jax.Array:
            event = packed_input[:, : self.config.input_width]
            advance = packed_input[:, self.config.input_width] > 0.5
            targets = packed_input[:, self.config.input_width + 1].astype(jnp.int32)
            weights = packed_input[:, self.config.input_width + 2]
            compact = LatentWorkspaceModel.update(self, event, advance)
            loss = legacy._classification_loss(
                compact,
                targets,
                self.config.color_rank,
            )
            wrapped = jnp.sqrt(weights) * jnp.sqrt(jnp.maximum(loss, 0.0))
            return jnp.where(weights == 0.0, jnp.zeros_like(wrapped), wrapped)

    def model_factory() -> _MaterializedObjective:
        def construct(
            model_config: ModelConfig,
            policy: str,
        ) -> _MaterializedObjective:
            if policy != "query_only":
                raise ValueError("removed-path objective requires query-only policy")
            return _MaterializedObjective(model_config)

        model = _gate_c2_control_model(
            config,
            initialization,
            regime=regime,
            policy="query_only",
            batch_size=1,
            role=role,
            probe="removed_path_finite_window_influence",
            arm="query_only",
            constructor=construct,
        )
        if not isinstance(model, _MaterializedObjective):
            raise TypeError("Gate C2 objective factory returned the wrong model")

        @brainstate.transform.jit
        def materialize(events: jax.Array, advances: jax.Array) -> jax.Array:
            def step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
                event, advance = inputs
                return LatentWorkspaceModel.update(model, event, advance)

            return brainstate.transform.for_loop(step, (events, advances))

        model.reset_state()
        jax.block_until_ready(
            materialize(jnp.asarray(prefix_events[:, None, :]), jnp.asarray(prefix_advances[:, None]))
        )
        live_snapshot = model.snapshot_state()
        materialized_prefixes.append(
            _gate_c2_raw_h0_snapshot_record(
                live_snapshot,
                parameter_sha256=legacy._array_digest(
                    legacy._parameter_values(model)
                ),
            )
        )
        snapshot = _gate_c2_stop_gradient_snapshot(live_snapshot)
        model._gate_c2_h0_snapshot = snapshot
        snapshots.append(snapshot)
        created_models.append(model)
        return model

    def algorithm_factory(model: brainstate.nn.Module) -> braintrace.ETraceAlgorithm:
        return braintrace.pp_prop(
            model,
            decay_or_rank=model.config.trace_decay,
            vjp_method="multi-step",
        )

    def capture_actual_gradient_start(
        model: brainstate.nn.Module,
        _algorithm: braintrace.ETraceAlgorithm,
    ) -> None:
        nonlocal gradient_boundary_capture_count
        if (
            not isinstance(model, _MaterializedObjective)
            or not snapshots
            or model is not created_models[-1]
        ):
            raise TypeError("Gate C2 gradient-boundary callback model differs")
        model.restore_state(snapshots[-1])
        actual_gradient_starts.append(
            _gate_c2_raw_h0_snapshot_record(
                model.snapshot_state(),
                parameter_sha256=legacy._array_digest(
                    legacy._parameter_values(model)
                ),
            )
        )
        gradient_boundary_capture_count += 1

    raw_gradients = chunked_online_param_gradients(
        model_factory,
        jnp.asarray(packed),
        algo_factory=algorithm_factory,
        chunk_size=GATE_C2_REMOVED_PATH_GRADIENT_CHUNK_SIZE,
        compiled_scan=True,
        after_init=capture_actual_gradient_start,
    )
    if (
        gradient_boundary_capture_count != 1
        or len(materialized_prefixes) != 1
        or len(actual_gradient_starts) != 1
    ):
        raise RuntimeError("Gate C2 gradient boundary was not captured exactly once")
    if not isinstance(raw_gradients, Mapping):
        raise TypeError("Gate C2 removed-path gradients must be a mapping")
    gradients = {
        key if isinstance(key, str) else gate_a._path(key): value
        for key, value in raw_gradients.items()
    }
    if tuple(sorted(gradients)) != FULL_PARAMETER_PATHS:
        raise ValueError("Gate C2 removed-path gradient paths differ")
    model = created_models[-1]
    snapshot = snapshots[-1]
    model.restore_state(snapshot)
    materialized_prefix = materialized_prefixes[-1]
    actual_gradient_start = actual_gradient_starts[-1]
    all_hidden_leaves_equal = bool(
        gate_a._json_exact(
            materialized_prefix["hidden_paths"],
            actual_gradient_start["hidden_paths"],
        )
        and materialized_prefix["hidden_state_tree_sha256"]
        == actual_gradient_start["hidden_state_tree_sha256"]
    )
    canonical_parameters_equal = bool(
        materialized_prefix["parameter_sha256"]
        == actual_gradient_start["parameter_sha256"]
        == canonical_parameter_sha256
    )
    h0_gradient_boundary = {
        "capture_point": "after_init_etrace_state_before_first_gradient_chunk",
        "capture_count": gradient_boundary_capture_count,
        "materialized_prefix": materialized_prefix,
        "actual_gradient_start": actual_gradient_start,
        "canonical_parameter_sha256": canonical_parameter_sha256,
        "all_hidden_leaves_equal": all_hidden_leaves_equal,
        "passed": bool(all_hidden_leaves_equal and canonical_parameters_equal),
    }

    @brainstate.transform.jit
    def measure(events: jax.Array, advances: jax.Array) -> jax.Array:
        def step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
            return LatentWorkspaceModel.update(model, *inputs)

        return brainstate.transform.for_loop(step, (events, advances))

    compact = jax.block_until_ready(
        measure(
            jnp.asarray(continuation_events[:, None, :]),
            jnp.asarray(continuation_advances[:, None]),
        )
    )
    raw_loss = np.asarray(
        legacy._classification_loss(
            compact[-1],
            jnp.asarray([continuation_targets[-1]], dtype=jnp.int32),
            model.config.color_rank,
        ),
        dtype=np.float32,
    ).reshape(1)
    weighted_loss = (raw_loss * base_weight).astype(np.float32)
    global_record = _gate_c2_global_gradient_record(gradients)
    live_records = {
        path: _gate_c2_gradient_path_record(
            path,
            gradients[path],
            require_zero=False,
        )
        for path in GATE_C2_LIVE_PATHS
    }
    removed_records = {
        path: _gate_c2_gradient_path_record(
            path,
            gradients[path],
            require_zero=True,
        )
        for path in GATE_C2_REMOVED_PATHS
    }
    if _ACTIVE_GATE_C2_CONTROLS_AUDIT is not None:
        _ACTIVE_GATE_C2_CONTROLS_AUDIT.finish(role, model)
    objective = {
        **coordinates,
        "source_contract": {
            "metadata": source_metadata,
            "events": _gate_c2_raw_array_record(source_events),
            "advances": _gate_c2_raw_array_record(
                source_advances,
                include_values=True,
            ),
            "targets": _gate_c2_raw_array_record(
                source_targets,
                include_values=True,
            ),
            "canonical_loss_weights": _gate_c2_raw_array_record(
                canonical_weights,
                include_values=True,
            ),
            "h0_prefix": {
                **_gate_c2_raw_array_record(prefix_events),
                "source_indices": list(range(h0_end)),
            },
            "schedule_cross_bound": schedule_cross_bound,
        },
        "continuation": {
            "source_indices": list(range(h0_end, source_events.shape[0])),
            "source_events": _gate_c2_raw_array_record(
                continuation_events,
                fill_value=0.0,
            ),
            "batched_events": _gate_c2_raw_array_record(
                continuation_events[:, None, :],
                fill_value=0.0,
            ),
            "advances": _gate_c2_raw_array_record(
                continuation_advances,
                include_values=True,
            ),
            "targets": _gate_c2_raw_array_record(
                continuation_targets,
                include_values=True,
            ),
            "selection_mask": _gate_c2_raw_array_record(
                selection,
                include_values=True,
            ),
            "base_checkpoint_weights": _gate_c2_raw_array_record(
                np.full(selection.shape, base_weight, dtype=np.float32),
                include_values=True,
            ),
            "effective_loss_weights": _gate_c2_raw_array_record(
                effective_weights,
                include_values=True,
            ),
            "packed_inputs": _gate_c2_raw_array_record(packed),
            "h0_gradient_boundary": h0_gradient_boundary,
            "source_slice_exact": True,
            "passed": h0_gradient_boundary["passed"],
        },
        "raw_cross_entropy": {
            **_gate_c2_raw_array_record(raw_loss),
            "value": float(raw_loss[0]),
            "finite": bool(np.isfinite(raw_loss).all()),
            "nonzero": bool(raw_loss[0] > 0.0),
        },
        "base_checkpoint_weight": _gate_c2_raw_array_record(
            np.asarray([base_weight], dtype=np.float32),
            include_values=True,
        ),
        "weighted_cross_entropy": {
            **_gate_c2_raw_array_record(weighted_loss),
            "value": float(weighted_loss[0]),
            "finite": bool(np.isfinite(weighted_loss).all()),
            "nonzero": bool(weighted_loss[0] > 0.0),
        },
        "passed": bool(raw_loss[0] > 0.0 and weighted_loss[0] > 0.0),
    }
    complete = bool(
        objective["passed"]
        and h0_gradient_boundary["passed"]
        and global_record["nonzero"]
        and all(record["nonzero"] for record in live_records.values())
        and all(record["exact_zero"] for record in removed_records.values())
    )
    return {
        "gradient_chunk_size": GATE_C2_REMOVED_PATH_GRADIENT_CHUNK_SIZE,
        "start_state": GATE_C2_REMOVED_PATH_START_STATE,
        "objectives": {objective_name: objective},
        "global": global_record,
        "live_paths": live_records,
        "removed_paths": removed_records,
        "complete": complete,
    }


def _gate_c2_snapshot_arrays(snapshot: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for path, value in snapshot.entries:
        name = gate_a._path(path)
        for index, leaf in enumerate(jax.tree.leaves(value)):
            leaf_path = f"{name}#{index}"
            if leaf_path in _GATE_C2_SHORT_TERM_DIAGNOSTIC_PATHS:
                continue
            result[leaf_path] = np.array(
                u.get_mantissa(jax.device_get(leaf)),
                copy=True,
                order="C",
            )
    return result


def _gate_c2_hidden_leaf_path_key(path: str) -> tuple[str, int]:
    base_path, separator, leaf_index = path.rpartition("#")
    if (
        separator != "#"
        or not base_path
        or not leaf_index
        or not leaf_index.isdecimal()
    ):
        raise ValueError("Gate C2 hidden leaf path is malformed")
    return base_path, int(leaf_index)


def _gate_c2_raw_h0_tree_sha256(
    hidden_paths: Mapping[str, Mapping[str, Any]],
) -> str:
    fields: list[bytes] = [b"example21-gate-c-hidden-state-v1"]
    for path in sorted(hidden_paths, key=_gate_c2_hidden_leaf_path_key):
        base_path, leaf_index = _gate_c2_hidden_leaf_path_key(path)
        leaf = hidden_paths[path]
        fields.extend(
            (
                base_path.encode("utf-8"),
                str(leaf_index).encode("ascii"),
                str(leaf["dtype"]).encode("ascii"),
                ",".join(map(str, leaf["shape"])).encode("ascii"),
                bytes.fromhex(str(leaf["data_hex"])),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gate_c2_raw_h0_snapshot_record(
    snapshot: Any,
    *,
    parameter_sha256: str,
) -> dict[str, Any]:
    if not _sha256_complete(parameter_sha256) or parameter_sha256 == "0" * 64:
        raise ValueError("Gate C2 H0 snapshot parameter digest is invalid")
    arrays = _gate_c2_snapshot_arrays(snapshot)
    if set(arrays) != (
        set(_GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY)
        | _GATE_C2_H0_AUXILIARY_HIDDEN_PATHS
    ):
        raise ValueError("Gate C2 H0 snapshot hidden paths differ")
    hidden_paths: dict[str, dict[str, Any]] = {}
    for path, expected_shape in _GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY.items():
        array = np.ascontiguousarray(arrays[path])
        if array.dtype.str != "<f4" or array.shape != expected_shape:
            raise ValueError("Gate C2 H0 snapshot hidden geometry differs")
        finite = np.isfinite(array)
        hidden_paths[path] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_hex": array.tobytes(order="C").hex(),
            "sha256": legacy._digest_arrays(array),
            "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()),
        }
    return {
        "hidden_paths": hidden_paths,
        "hidden_state_tree_sha256": _gate_c2_raw_h0_tree_sha256(hidden_paths),
        "parameter_sha256": parameter_sha256,
    }


def _gate_c2_raw_h0_snapshot_complete(
    value: Any,
    *,
    canonical_parameter_sha256: str,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "hidden_paths",
        "hidden_state_tree_sha256",
        "parameter_sha256",
    }:
        return False
    hidden_paths = value["hidden_paths"]
    if (
        not isinstance(hidden_paths, Mapping)
        or set(hidden_paths) != set(_GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY)
        or value["parameter_sha256"] != canonical_parameter_sha256
    ):
        return False
    try:
        for path, expected_shape in _GATE_C2_BATCH_ONE_HIDDEN_GEOMETRY.items():
            leaf = hidden_paths[path]
            if not isinstance(leaf, Mapping) or set(leaf) != {
                "dtype",
                "shape",
                "data_hex",
                "sha256",
                "finite_count",
                "nonfinite_count",
            }:
                return False
            value_count = math.prod(expected_shape)
            data_hex = leaf["data_hex"]
            if (
                leaf["dtype"] != "<f4"
                or leaf["shape"] != list(expected_shape)
                or not isinstance(data_hex, str)
                or re.fullmatch(r"[0-9a-f]+", data_hex) is None
                or len(data_hex) != value_count * np.dtype("<f4").itemsize * 2
                or not _strict_integer(leaf["finite_count"])
                or not _strict_integer(leaf["nonfinite_count"])
            ):
                return False
            raw = bytes.fromhex(data_hex)
            array = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(
                expected_shape
            )
            finite_count = int(np.isfinite(array).sum())
            nonfinite_count = value_count - finite_count
            if (
                leaf["sha256"] != legacy._digest_arrays(array)
                or int(leaf["finite_count"]) != finite_count
                or int(leaf["nonfinite_count"]) != nonfinite_count
                or finite_count != value_count
                or nonfinite_count != 0
            ):
                return False
        return bool(
            _sha256_complete(value["hidden_state_tree_sha256"])
            and value["hidden_state_tree_sha256"]
            == _gate_c2_raw_h0_tree_sha256(hidden_paths)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _gate_c2_h0_gradient_boundary_complete(
    value: Any,
    *,
    canonical_parameter_sha256: str,
) -> bool:
    if (
        not _sha256_complete(canonical_parameter_sha256)
        or canonical_parameter_sha256 == "0" * 64
        or not isinstance(value, Mapping)
        or set(value)
        != {
            "capture_point",
            "capture_count",
            "materialized_prefix",
            "actual_gradient_start",
            "canonical_parameter_sha256",
            "all_hidden_leaves_equal",
            "passed",
        }
    ):
        return False
    materialized = value["materialized_prefix"]
    actual = value["actual_gradient_start"]
    snapshots_complete = bool(
        _gate_c2_raw_h0_snapshot_complete(
            materialized,
            canonical_parameter_sha256=canonical_parameter_sha256,
        )
        and _gate_c2_raw_h0_snapshot_complete(
            actual,
            canonical_parameter_sha256=canonical_parameter_sha256,
        )
    )
    hidden_equal = bool(
        snapshots_complete
        and gate_a._json_exact(
            materialized["hidden_paths"],
            actual["hidden_paths"],
        )
        and materialized["hidden_state_tree_sha256"]
        == actual["hidden_state_tree_sha256"]
    )
    recomputed = bool(
        value["capture_point"]
        == "after_init_etrace_state_before_first_gradient_chunk"
        and _strict_integer(value["capture_count"])
        and int(value["capture_count"]) == 1
        and value["canonical_parameter_sha256"]
        == canonical_parameter_sha256
        and hidden_equal
    )
    return bool(
        isinstance(value["all_hidden_leaves_equal"], bool)
        and value["all_hidden_leaves_equal"] is hidden_equal
        and isinstance(value["passed"], bool)
        and value["passed"] is recomputed
        and recomputed
    )


def _gate_c2_parameter_arrays(
    model: LatentWorkspaceModel,
) -> dict[str, Any]:
    return {
        path: jax.tree.map(
            lambda leaf: np.array(
                u.get_mantissa(jax.device_get(leaf)),
                copy=True,
                order="C",
            ),
            value,
        )
        for path, value in legacy._parameter_values(model).items()
    }


def _gate_c2_cached_boundary_tree_sha256(
    hidden_paths: Mapping[str, Mapping[str, Any]],
) -> str:
    fields: list[bytes] = [
        b"example21-gate-c2-cached-read-boundary-state-v1"
    ]
    for path in sorted(hidden_paths):
        endpoint = hidden_paths[path]
        fields.extend(
            (
                path.encode("utf-8"),
                b"0",
                str(endpoint["dtype"]).encode("ascii"),
                ",".join(map(str, endpoint["shape"])).encode("ascii"),
                str(endpoint["sha256"]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gate_c2_boundary_snapshot(
    model: LatentWorkspaceModel,
    driver: Any,
    h0_snapshot: Any,
    events: np.ndarray,
    advances: np.ndarray,
    *,
    tick_index: int,
) -> Any:
    model.restore_state(h0_snapshot)
    if tick_index:
        jax.block_until_ready(
            driver(
                jnp.asarray(events[:tick_index]),
                jnp.asarray(advances[:tick_index]),
            )
        )
    return _gate_c2_stop_gradient_snapshot(model.snapshot_state())


def _gate_c2_cache_boundary_report(
    before_snapshot: Any,
    after_snapshot: Any,
    *,
    before_parameters: Mapping[str, Any],
    after_parameters: Mapping[str, Any],
    canonical_parameter_sha256: str,
) -> dict[str, Any]:
    before_arrays = _gate_c2_snapshot_arrays(before_snapshot)
    after_arrays = _gate_c2_snapshot_arrays(after_snapshot)
    before_hidden = {
        path: _gate_c2_array_endpoint(value)
        for path, value in before_arrays.items()
    }
    after_hidden = {
        path: _gate_c2_array_endpoint(value)
        for path, value in after_arrays.items()
    }
    changed = [
        path
        for path in sorted(before_hidden)
        if before_hidden[path]["sha256"] != after_hidden[path]["sha256"]
    ]
    unchanged = [
        path
        for path in sorted(before_hidden)
        if before_hidden[path]["sha256"] == after_hidden[path]["sha256"]
    ]
    before_parameter_sha256 = legacy._array_digest(before_parameters)
    after_parameter_sha256 = legacy._array_digest(after_parameters)
    parameters_equal = bool(
        _sha256_complete(canonical_parameter_sha256)
        and before_parameter_sha256 == canonical_parameter_sha256
        and after_parameter_sha256 == canonical_parameter_sha256
    )
    only_memory_read = bool(
        set(before_hidden) == set(after_hidden)
        and changed == ["memory_read#0"]
        and unchanged
        == sorted(set(before_hidden) - {"memory_read#0"})
        and parameters_equal
    )
    before = {
        "hidden_paths": before_hidden,
        "hidden_state_tree_sha256": _gate_c2_cached_boundary_tree_sha256(
            before_hidden
        ),
        "parameter_tree_sha256": before_parameter_sha256,
    }
    after = {
        "hidden_paths": after_hidden,
        "hidden_state_tree_sha256": _gate_c2_cached_boundary_tree_sha256(
            after_hidden
        ),
        "parameter_tree_sha256": after_parameter_sha256,
    }
    passed = bool(
        parameters_equal
        and only_memory_read
        and before["hidden_state_tree_sha256"]
        != after["hidden_state_tree_sha256"]
    )
    return {
        "before_replacement": before,
        "after_replacement": after,
        "changed_paths": changed,
        "unchanged_paths": unchanged,
        "parameters_equal": parameters_equal,
        "only_memory_read_replaced": only_memory_read,
        "passed": passed,
    }


def _gate_c2_suffix_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tick_names: tuple[str, ...],
    include_context_memory: bool,
) -> dict[str, Any]:
    ticks = {
        tick: _gate_c2_continuation_comparison(
            left,
            right,
            tick_index=index,
            exclude_context_memory=not include_context_memory,
        )
        for index, tick in enumerate(tick_names)
    }
    return {
        "ticks": ticks,
        "passed": all(item["passed"] is True for item in ticks.values()),
    }


def _gate_c2_tree_geometry_sha256(
    values: Mapping[str, Any],
    *,
    domain: str,
) -> str:
    fields = [domain.encode("utf-8")]
    for path in sorted(values):
        for index, leaf in enumerate(jax.tree.leaves(values[path])):
            array = np.ascontiguousarray(np.asarray(u.get_mantissa(leaf)))
            fields.extend(
                (
                    path.encode("utf-8"),
                    str(index).encode("ascii"),
                    array.dtype.str.encode("ascii"),
                    ",".join(map(str, array.shape)).encode("ascii"),
                )
            )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gate_c2_tree_value_sha256(
    values: Mapping[str, Any],
    *,
    domain: str,
) -> str:
    fields = [domain.encode("utf-8")]
    for path in sorted(values):
        for index, leaf in enumerate(jax.tree.leaves(values[path])):
            array = np.ascontiguousarray(np.asarray(u.get_mantissa(leaf)))
            fields.extend(
                (
                    path.encode("utf-8"),
                    str(index).encode("ascii"),
                    array.dtype.str.encode("ascii"),
                    ",".join(map(str, array.shape)).encode("ascii"),
                    array.tobytes(),
                )
            )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gate_c2_tree_equality_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    framing: str,
    parameter_values: bool,
) -> dict[str, Any]:
    paths = sorted(left)
    if paths != sorted(right):
        raise ValueError("Gate C2 equality path sets differ")
    tree_domain = (
        "example21-gate-c2-parameter-tree-v1"
        if parameter_values
        else "example21-gate-c2-non-s-k-tree-v1"
    )
    value_domain = "example21-gate-c2-non-s-k-state-v1"
    left_tree = _gate_c2_tree_geometry_sha256(left, domain=tree_domain)
    right_tree = _gate_c2_tree_geometry_sha256(right, domain=tree_domain)
    if parameter_values:
        left_value = legacy._array_digest(left)
        right_value = legacy._array_digest(right)
    else:
        left_value = _gate_c2_tree_value_sha256(left, domain=value_domain)
        right_value = _gate_c2_tree_value_sha256(right, domain=value_domain)
    return {
        "paths": paths,
        "framing": framing,
        "left_tree_sha256": left_tree,
        "right_tree_sha256": right_tree,
        "left_value_sha256": left_value,
        "right_value_sha256": right_value,
        "tree_equal": bool(left_tree == right_tree),
        "values_equal": bool(left_value == right_value),
    }


def _gate_c2_latent_driver(model: LatentWorkspaceModel) -> Any:
    hidden_states = sorted(
        model.states(brainstate.HiddenState).items(),
        key=lambda item: gate_a._path(item[0]),
    )

    @brainstate.transform.jit
    def run_latent(events: jax.Array, advances: jax.Array) -> Any:
        def step(inputs: tuple[jax.Array, jax.Array]) -> Any:
            event, advance = inputs
            compact = model.update(event, advance)
            read = model.memory_read.value
            drive = model.memory_read_projection(read)
            hidden = tuple(
                u.get_mantissa(leaf)
                for _, state in hidden_states
                for leaf in jax.tree.leaves(state.value)
            )
            return compact, read, drive, hidden

        return brainstate.transform.for_loop(step, (events, advances))

    paths = tuple(
        f"{gate_a._path(path)}#{index}"
        for path, state in hidden_states
        for index, _ in enumerate(jax.tree.leaves(state.value))
    )
    return run_latent, paths


def _gate_c2_latent_capture(
    model: LatentWorkspaceModel,
    driver: Any,
    hidden_paths: tuple[str, ...],
    snapshot: Any,
    events: np.ndarray,
    advances: np.ndarray,
    *,
    replacement: float | None = None,
) -> dict[str, Any]:
    model.restore_state(snapshot)
    if replacement is not None:
        model.context_memory.value = jnp.full_like(
            model.context_memory.value,
            np.float32(replacement),
        )
    compact, read, drive, hidden = jax.block_until_ready(
        driver(jnp.asarray(events), jnp.asarray(advances))
    )
    compact_array = np.ascontiguousarray(np.asarray(jax.device_get(compact)))
    flat_compact = compact_array.reshape((-1, compact_array.shape[-1]))
    color_logits = np.asarray(
        legacy._color_logits(jnp.asarray(flat_compact), model.config.color_rank)
    ).reshape((*compact_array.shape[:-1], -1))
    predictions = np.argmax(color_logits, axis=-1).astype(np.int32, copy=False)
    return {
        "compact": compact_array,
        "selected_read": np.ascontiguousarray(np.asarray(jax.device_get(read))),
        "selected_drive": np.ascontiguousarray(np.asarray(jax.device_get(drive))),
        "hidden_paths": {
            path: np.ascontiguousarray(np.asarray(jax.device_get(value)))
            for path, value in zip(hidden_paths, hidden, strict=True)
        },
        "predictions": np.ascontiguousarray(predictions),
    }


def _gate_c2_host_boundary_snapshots(
    h0_snapshot: Any,
    hidden_paths: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Copy every pre-tick latent boundary into independent host storage."""

    h0_arrays = _gate_c2_snapshot_arrays(h0_snapshot)
    if set(hidden_paths) != set(h0_arrays):
        raise ValueError("Gate C2 host boundary hidden path set differs")
    tick_count: int | None = None
    stacked: dict[str, np.ndarray] = {}
    for path, expected in h0_arrays.items():
        value = np.asarray(hidden_paths[path])
        if value.ndim < 1:
            raise ValueError("Gate C2 host boundary leading tick axis is missing")
        if tick_count is None:
            tick_count = int(value.shape[0])
        elif int(value.shape[0]) != tick_count:
            raise ValueError("Gate C2 host boundary tick lengths differ")
        if value.shape[1:] != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Gate C2 host boundary hidden geometry differs")
        stacked[path] = value
    if tick_count is None or tick_count < 1:
        raise ValueError("Gate C2 host boundary tick length is empty")

    boundaries: list[Any] = []
    for boundary_index in range(tick_count):
        entries: list[tuple[Any, Any]] = []
        for path, template in h0_snapshot.entries:
            name = gate_a._path(path)
            leaves: list[np.ndarray] = []
            for leaf_index, template_leaf in enumerate(jax.tree.leaves(template)):
                leaf_path = f"{name}#{leaf_index}"
                if leaf_path in h0_arrays:
                    source = (
                        h0_arrays[leaf_path]
                        if boundary_index == 0
                        else stacked[leaf_path][boundary_index - 1]
                    )
                else:
                    source = u.get_mantissa(jax.device_get(template_leaf))
                leaves.append(np.array(source, copy=True, order="C"))
            value = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(template),
                leaves,
            )
            entries.append((path, value))
        boundaries.append(
            dataclasses.replace(h0_snapshot, entries=tuple(entries))
        )
    return tuple(boundaries)


def _gate_c2_latent_capture_slice(
    capture: Mapping[str, Any],
    start: int,
    stop: int | None = None,
) -> dict[str, Any]:
    index = slice(start, stop)
    return {
        "compact": capture["compact"][index],
        "selected_read": capture["selected_read"][index],
        "selected_drive": capture["selected_drive"][index],
        "hidden_paths": {
            path: value[index]
            for path, value in capture["hidden_paths"].items()
        },
        "predictions": capture["predictions"][index],
    }


def _gate_c2_continuation_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tick_index: int,
    exclude_context_memory: bool,
) -> dict[str, Any]:
    compact = _gate_c2_floating_difference_record(
        left["compact"][tick_index],
        right["compact"][tick_index],
        rms_tolerance=1e-6,
    )
    paths = sorted(left["hidden_paths"])
    if exclude_context_memory:
        paths = [path for path in paths if path != "context_memory#0"]
    hidden = {
        path: _gate_c2_floating_difference_record(
            left["hidden_paths"][path][tick_index],
            right["hidden_paths"][path][tick_index],
            rms_tolerance=1e-6,
        )
        for path in paths
    }
    predictions = _gate_c2_prediction_difference_record(
        left["predictions"][tick_index],
        right["predictions"][tick_index],
    )
    passed = bool(
        compact["within_tolerance"]
        and all(record["within_tolerance"] for record in hidden.values())
        and predictions["equal"]
    )
    return {
        "compact": compact,
        "hidden_paths": hidden,
        "predictions": predictions,
        "passed": passed,
    }


def _gate_c2_no_read_stream_data(
    config: GateCConfig,
    regime: str,
    data: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A no-read probe requires BindingData")
        streams = {
            "intact": np.asarray(data.validation_intact),
            "shuffled": np.asarray(data.validation_shuffled),
            "no_context": np.asarray(data.validation_no_context),
        }
        advances = np.ones(
            (regime_config.sequence_length, regime_config.validation_episodes),
            dtype=np.bool_,
        )
    else:
        if (
            not isinstance(data, tuple)
            or len(data) != 2
            or not isinstance(data[1], gate_b.DepthValidationData)
        ):
            raise TypeError("Gate B no-read probe requires DepthValidationData")
        validation = data[1]
        streams = {
            "intact": np.asarray(validation.intact),
            "shuffled": np.asarray(validation.shuffled),
            "no_context": np.asarray(validation.no_context),
        }
        advances = np.asarray(validation.advance_masks)
    h0_end = regime_config.sequence_length - regime_config.gap_steps
    return streams, advances, h0_end


def _query_only_latent_no_read_report(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    regime: str,
    data: Any,
) -> dict[str, Any]:
    """Probe selected zero reads, cached reads, and context replacements."""

    _regime_spec(regime)
    streams, advances, h0_end = _gate_c2_no_read_stream_data(
        config,
        regime,
        data,
    )
    count = next(iter(streams.values())).shape[1]
    canonical_parameter_sha256 = initialization["initialization"][regime][
        "canonical_full"
    ]["parameter_sha256"]
    roles = {
        "query_only": f"{regime}:query_only_latent_no_read:query_only",
        "full": f"{regime}:query_only_latent_no_read:full_positive_control",
    }
    query_model = _gate_c2_control_model(
        config,
        initialization,
        regime=regime,
        policy="query_only",
        batch_size=count,
        role=roles["query_only"],
        probe="query_only_latent_no_read",
    )
    full_model = _gate_c2_control_model(
        config,
        initialization,
        regime=regime,
        policy="full",
        batch_size=count,
        role=roles["full"],
        probe="full_positive_control",
    )
    legacy._copy_parameters(query_model, full_model)
    initial_snapshot = _gate_c2_stop_gradient_snapshot(query_model.snapshot_state())
    full_model.restore_state(initial_snapshot)
    prefix_drivers = {
        "query_only": _gate_c2_h0_driver(query_model),
        "full": _gate_c2_h0_driver(full_model),
    }
    query_driver, query_hidden_paths = _gate_c2_latent_driver(query_model)
    full_driver, full_hidden_paths = _gate_c2_latent_driver(full_model)
    if query_hidden_paths != full_hidden_paths:
        raise ValueError("Gate C2 no-read hidden-state paths differ")
    tick_names = GATE_C2_LATENT_TICKS[regime]
    selected_streams: dict[str, Any] = {}
    perturbation_streams = {
        replacement: {} for replacement in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
    }
    positive_streams = {
        replacement: {} for replacement in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
    }
    positive_nonzero = {
        replacement: False for replacement in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
    }
    replacement_reports = {
        name: {
            "fill_value": float(spec["fill_value"]),
            **_gate_c2_raw_array_record(
                np.full(
                    (count, 32, 32),
                    float(spec["fill_value"]),
                    dtype=np.float32,
                )
            ),
        }
        for name, spec in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS.items()
    }

    def cached_read_probe(
        *,
        boundary: Any,
        suffix_events: np.ndarray,
        suffix_advances: np.ndarray,
        baseline: Mapping[str, Any],
        suffix_ticks: tuple[str, ...],
    ) -> dict[str, Any]:
        before_arrays = _gate_c2_snapshot_arrays(boundary)
        source = _gate_c2_array_endpoint(before_arrays["memory_read#0"])
        sentinels: dict[str, Any] = {}
        for name, spec in GATE_C2_CACHED_READ_REPLACEMENTS.items():
            query_model.restore_state(boundary)
            before_parameters = _gate_c2_parameter_arrays(query_model)
            replacement = np.full_like(
                before_arrays["memory_read#0"],
                float(spec["fill_value"]),
            )
            query_model.memory_read.value = jnp.asarray(replacement)
            after_boundary = _gate_c2_stop_gradient_snapshot(
                query_model.snapshot_state()
            )
            after_parameters = _gate_c2_parameter_arrays(query_model)
            perturbed = _gate_c2_latent_capture(
                query_model,
                query_driver,
                query_hidden_paths,
                after_boundary,
                suffix_events,
                suffix_advances,
            )
            selected_read = _gate_c2_zero_array_record(
                perturbed["selected_read"][0]
            )
            selected_drive = _gate_c2_zero_array_record(
                perturbed["selected_drive"][0]
            )
            continuation = _gate_c2_suffix_comparison(
                baseline,
                perturbed,
                tick_names=suffix_ticks,
                include_context_memory=True,
            )
            boundary_report = _gate_c2_cache_boundary_report(
                boundary,
                after_boundary,
                before_parameters=before_parameters,
                after_parameters=after_parameters,
                canonical_parameter_sha256=canonical_parameter_sha256,
            )
            replacement_report = {
                "fill_value": float(spec["fill_value"]),
                **_gate_c2_raw_array_record(replacement),
            }
            passed = bool(
                source["sha256"] != replacement_report["sha256"]
                and boundary_report["passed"]
                and selected_read["exact_zero"]
                and selected_drive["exact_zero"]
                and continuation["passed"]
            )
            sentinels[name] = {
                "replacement": replacement_report,
                "boundary": boundary_report,
                "selected_read": selected_read,
                "selected_drive": selected_drive,
                "continuation": continuation,
                "passed": passed,
            }
        probe_passed = all(
            sentinels[name]["passed"] is True
            for name in GATE_C2_CACHED_READ_REPLACEMENTS
        )
        return {
            "source_cached_memory_read": source,
            **sentinels,
            "passed": probe_passed,
        }

    def context_intervention(
        *,
        model: LatentWorkspaceModel,
        driver: Any,
        hidden_paths: tuple[str, ...],
        boundary: Any,
        suffix_events: np.ndarray,
        suffix_advances: np.ndarray,
        baseline: Mapping[str, Any],
        fill: float,
        positive: bool,
    ) -> dict[str, Any]:
        model.restore_state(boundary)
        before_arrays = _gate_c2_snapshot_arrays(boundary)
        before_parameters = _gate_c2_parameter_arrays(model)
        source_memory = before_arrays["context_memory#0"]
        replacement = np.full_like(source_memory, fill)
        model.context_memory.value = jnp.asarray(replacement)
        after_boundary = _gate_c2_stop_gradient_snapshot(model.snapshot_state())
        after_arrays = _gate_c2_snapshot_arrays(after_boundary)
        after_parameters = _gate_c2_parameter_arrays(model)
        before_non_memory = {
            path: value
            for path, value in before_arrays.items()
            if path != "context_memory#0"
        }
        after_non_memory = {
            path: value
            for path, value in after_arrays.items()
            if path != "context_memory#0"
        }
        non_s_k_state = _gate_c2_tree_equality_record(
            before_non_memory,
            after_non_memory,
            framing="nul_joined_gate_c2_non_s_k_state_v1",
            parameter_values=False,
        )
        parameters = _gate_c2_tree_equality_record(
            before_parameters,
            after_parameters,
            framing="authenticated_gate_c_parameter_array_digest_v1",
            parameter_values=True,
        )
        perturbed = _gate_c2_latent_capture(
            model,
            driver,
            hidden_paths,
            after_boundary,
            suffix_events,
            suffix_advances,
        )
        continuation = _gate_c2_continuation_comparison(
            baseline,
            perturbed,
            tick_index=0,
            exclude_context_memory=True,
        )
        shared = {
            "source_s_k_sha256": legacy._digest_arrays(source_memory),
            "replacement_s_k_sha256": legacy._digest_arrays(replacement),
            "source_replacement_differ": bool(
                not np.array_equal(source_memory, replacement)
            ),
            "non_s_k_state": non_s_k_state,
            "parameters": parameters,
        }
        boundary_passed = bool(
            shared["source_replacement_differ"]
            and non_s_k_state["tree_equal"]
            and non_s_k_state["values_equal"]
            and parameters["tree_equal"]
            and parameters["values_equal"]
        )
        if not positive:
            selected_read = _gate_c2_zero_array_record(
                perturbed["selected_read"][0]
            )
            selected_drive = _gate_c2_zero_array_record(
                perturbed["selected_drive"][0]
            )
            return {
                **shared,
                "selected_read": selected_read,
                "selected_drive": selected_drive,
                "continuation": continuation,
                "passed": bool(
                    boundary_passed
                    and selected_read["exact_zero"]
                    and selected_drive["exact_zero"]
                    and continuation["passed"]
                ),
            }
        read_difference = _gate_c2_floating_difference_record(
            baseline["selected_read"][0],
            perturbed["selected_read"][0],
            rms_tolerance=1e-6,
        )
        drive_difference = _gate_c2_floating_difference_record(
            baseline["selected_drive"][0],
            perturbed["selected_drive"][0],
            rms_tolerance=1e-6,
        )
        return {
            **shared,
            "selected_read_difference": read_difference,
            "selected_drive_difference": drive_difference,
            "continuation": continuation,
            "passed": bool(
                boundary_passed
                and read_difference["left"]["nonfinite_count"] == 0
                and read_difference["right"]["nonfinite_count"] == 0
                and drive_difference["left"]["nonfinite_count"] == 0
                and drive_difference["right"]["nonfinite_count"] == 0
            ),
        }

    for stream_name, events in streams.items():
        prefix_events = events[:h0_end]
        prefix_advances = advances[:h0_end]
        latent_events = events[h0_end:]
        latent_advances = advances[h0_end:]
        query_model.restore_state(initial_snapshot)
        jax.block_until_ready(
            prefix_drivers["query_only"](
                jnp.asarray(prefix_events),
                jnp.asarray(prefix_advances),
            )
        )
        query_h0 = _gate_c2_stop_gradient_snapshot(query_model.snapshot_state())
        full_model.restore_state(initial_snapshot)
        jax.block_until_ready(
            prefix_drivers["full"](
                jnp.asarray(prefix_events),
                jnp.asarray(prefix_advances),
            )
        )
        full_h0 = _gate_c2_stop_gradient_snapshot(full_model.snapshot_state())
        query_complete_baseline = _gate_c2_latent_capture(
            query_model,
            query_driver,
            query_hidden_paths,
            query_h0,
            latent_events,
            latent_advances,
        )
        full_complete_baseline = _gate_c2_latent_capture(
            full_model,
            full_driver,
            full_hidden_paths,
            full_h0,
            latent_events,
            latent_advances,
        )
        query_boundaries = _gate_c2_host_boundary_snapshots(
            query_h0,
            query_complete_baseline["hidden_paths"],
        )
        full_boundaries = _gate_c2_host_boundary_snapshots(
            full_h0,
            full_complete_baseline["hidden_paths"],
        )
        if (
            len(query_boundaries) != len(tick_names)
            or len(full_boundaries) != len(tick_names)
        ):
            raise ValueError("Gate C2 host boundary tick count differs")
        selected_ticks: dict[str, Any] = {}
        query_ticks_by_replacement = {
            name: {} for name in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
        }
        full_ticks_by_replacement = {
            name: {} for name in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
        }
        for tick_index, tick in enumerate(tick_names):
            suffix_events = latent_events[tick_index:]
            suffix_advances = latent_advances[tick_index:]
            suffix_ticks = tick_names[tick_index:]
            tick_events = latent_events[tick_index : tick_index + 1]
            tick_advances = latent_advances[tick_index : tick_index + 1]
            query_boundary = query_boundaries[tick_index]
            full_boundary = full_boundaries[tick_index]
            query_suffix_baseline = _gate_c2_latent_capture_slice(
                query_complete_baseline,
                tick_index,
            )
            query_tick_baseline = _gate_c2_latent_capture_slice(
                query_complete_baseline,
                tick_index,
                tick_index + 1,
            )
            full_tick_baseline = _gate_c2_latent_capture_slice(
                full_complete_baseline,
                tick_index,
                tick_index + 1,
            )
            cached_probe = cached_read_probe(
                boundary=query_boundary,
                suffix_events=suffix_events,
                suffix_advances=suffix_advances,
                baseline=query_suffix_baseline,
                suffix_ticks=suffix_ticks,
            )
            selected_ticks[tick] = {
                "selected_read": _gate_c2_zero_array_record(
                    query_tick_baseline["selected_read"][0]
                ),
                "selected_drive": _gate_c2_zero_array_record(
                    query_tick_baseline["selected_drive"][0]
                ),
                "cached_read_probe": cached_probe,
                "cached_h0_read_reused": bool(not cached_probe["passed"]),
            }
            for replacement_name, replacement_spec in (
                GATE_C2_CONTEXT_MEMORY_REPLACEMENTS.items()
            ):
                fill = float(replacement_spec["fill_value"])
                query_tick = context_intervention(
                    model=query_model,
                    driver=query_driver,
                    hidden_paths=query_hidden_paths,
                    boundary=query_boundary,
                    suffix_events=tick_events,
                    suffix_advances=tick_advances,
                    baseline=query_tick_baseline,
                    fill=fill,
                    positive=False,
                )
                full_tick = context_intervention(
                    model=full_model,
                    driver=full_driver,
                    hidden_paths=full_hidden_paths,
                    boundary=full_boundary,
                    suffix_events=tick_events,
                    suffix_advances=tick_advances,
                    baseline=full_tick_baseline,
                    fill=fill,
                    positive=True,
                )
                query_ticks_by_replacement[replacement_name][tick] = query_tick
                full_ticks_by_replacement[replacement_name][tick] = full_tick
                positive_nonzero[replacement_name] = bool(
                    positive_nonzero[replacement_name]
                    or full_tick["selected_read_difference"][
                        "max_abs_difference"
                    ]
                    > 0.0
                    or full_tick["selected_drive_difference"][
                        "max_abs_difference"
                    ]
                    > 0.0
                )
        selected_streams[stream_name] = selected_ticks
        for replacement_name in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS:
            perturbation_streams[replacement_name][stream_name] = (
                query_ticks_by_replacement[replacement_name]
            )
            positive_streams[replacement_name][stream_name] = (
                full_ticks_by_replacement[replacement_name]
            )

    perturbations = {
        replacement: {
            "replacement": replacement_reports[replacement],
            "streams": perturbation_streams[replacement],
            "passed": all(
                tick["passed"] is True
                for stream in perturbation_streams[replacement].values()
                for tick in stream.values()
            ),
        }
        for replacement in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
    }
    positive = {
        replacement: {
            "replacement": replacement_reports[replacement],
            "streams": positive_streams[replacement],
            "passed": bool(
                positive_nonzero[replacement]
                and all(
                    tick["passed"] is True
                    for stream in positive_streams[replacement].values()
                    for tick in stream.values()
                )
            ),
        }
        for replacement in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS
    }
    full_positive_control = {
        **positive,
        "passed": all(item["passed"] is True for item in positive.values()),
    }
    influence = _gate_c2_removed_path_finite_window_influence(
        config,
        initialization=initialization,
        regime=regime,
        data=data,
    )
    _gate_c2_finish_control_model(roles["query_only"], query_model)
    _gate_c2_finish_control_model(roles["full"], full_model)
    streams_passed = all(
        tick["selected_read"]["exact_zero"] is True
        and tick["selected_drive"]["exact_zero"] is True
        and tick["cached_read_probe"]["passed"] is True
        and tick["cached_h0_read_reused"] is False
        for stream in selected_streams.values()
        for tick in stream.values()
    )
    return {
        "streams": selected_streams,
        "perturbations": perturbations,
        "full_positive_control": full_positive_control,
        "removed_path_finite_window_influence": influence,
        "passed": bool(
            streams_passed
            and all(item["passed"] is True for item in perturbations.values())
            and full_positive_control["passed"] is True
            and influence["complete"] is True
        ),
    }


def _gate_c2_equality_record_complete(
    value: Any,
    *,
    expected_paths: tuple[str, ...],
    framing: str,
    require_equal: bool = True,
) -> bool:
    keys = {
        "paths",
        "framing",
        "left_tree_sha256",
        "right_tree_sha256",
        "left_value_sha256",
        "right_value_sha256",
        "tree_equal",
        "values_equal",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != keys
        or value["paths"] != list(expected_paths)
        or value["framing"] != framing
    ):
        return False
    hashes = tuple(
        value[name]
        for name in (
            "left_tree_sha256",
            "right_tree_sha256",
            "left_value_sha256",
            "right_value_sha256",
        )
    )
    if not all(_sha256_complete(item) for item in hashes):
        return False
    tree_equal = hashes[0] == hashes[1]
    values_equal = hashes[2] == hashes[3]
    return bool(
        isinstance(value["tree_equal"], bool)
        and value["tree_equal"] is tree_equal
        and isinstance(value["values_equal"], bool)
        and value["values_equal"] is values_equal
        and ((tree_equal and values_equal) or not require_equal)
    )


def _gate_c2_difference_endpoints_coherent(value: Mapping[str, Any]) -> bool:
    maximum = _finite_real(value["max_abs_difference"], "maximum difference")
    same_digest = value["left"]["sha256"] == value["right"]["sha256"]
    return bool((maximum == 0.0) is same_digest)


def _gate_c2_continuation_record_complete(
    value: Any,
    *,
    count: int,
    include_context_memory: bool,
    require_pass: bool,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "compact",
        "hidden_paths",
        "predictions",
        "passed",
    }:
        return False
    compact = value["compact"]
    if (
        not _gate_c2_floating_difference_record_complete(
            compact,
            expected_dtype="<f4",
            expected_shape=(count, 1_180),
            rms_tolerance=1e-6,
            require_within=False,
        )
        or not _gate_c2_difference_endpoints_coherent(compact)
    ):
        return False
    expected_hidden = tuple(
        path
        for path in _GATE_C2_HIDDEN_GEOMETRY
        if include_context_memory or path != "context_memory#0"
    )
    hidden = value["hidden_paths"]
    if not isinstance(hidden, Mapping) or set(hidden) != set(expected_hidden):
        return False
    hidden_within = True
    for path in expected_hidden:
        shape = (count, *_GATE_C2_HIDDEN_GEOMETRY[path][1:])
        record = hidden[path]
        if (
            not _gate_c2_floating_difference_record_complete(
                record,
                expected_dtype="<f4",
                expected_shape=shape,
                rms_tolerance=1e-6,
                require_within=False,
            )
            or not _gate_c2_difference_endpoints_coherent(record)
        ):
            return False
        hidden_within = bool(hidden_within and record["within_tolerance"])
    predictions = value["predictions"]
    if not _gate_c2_prediction_difference_record_complete(
        predictions,
        count=count,
        require_equal=False,
    ):
        return False
    recomputed_pass = bool(
        compact["within_tolerance"]
        and hidden_within
        and predictions["equal"]
    )
    return bool(
        isinstance(value["passed"], bool)
        and value["passed"] is recomputed_pass
        and (recomputed_pass or not require_pass)
    )


def _gate_c2_replacement_record_complete(
    value: Any,
    *,
    spec: Mapping[str, Any],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "fill_value",
        "dtype",
        "shape",
        "sha256",
    }:
        return False
    try:
        expected = np.full(
            tuple(int(item) for item in spec["shape"]),
            float(spec["fill_value"]),
            dtype=np.dtype(str(spec["dtype"])),
        )
        return bool(
            value["fill_value"] == float(spec["fill_value"])
            and value["dtype"] == expected.dtype.str
            and value["shape"] == list(expected.shape)
            and value["sha256"] == legacy._digest_arrays(expected)
            and value["sha256"] == spec["sha256"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _gate_c2_cached_boundary_complete(
    value: Any,
    *,
    source: Mapping[str, Any],
    replacement: Mapping[str, Any],
    require_pass: bool = True,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "before_replacement",
        "after_replacement",
        "changed_paths",
        "unchanged_paths",
        "parameters_equal",
        "only_memory_read_replaced",
        "passed",
    }:
        return False
    before = value["before_replacement"]
    after = value["after_replacement"]
    side_keys = {
        "hidden_paths",
        "hidden_state_tree_sha256",
        "parameter_tree_sha256",
    }
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or before is after
        or set(before) != side_keys
        or set(after) != side_keys
        or not isinstance(before["hidden_paths"], Mapping)
        or not isinstance(after["hidden_paths"], Mapping)
        or set(before["hidden_paths"]) != set(_GATE_C2_HIDDEN_GEOMETRY)
        or set(after["hidden_paths"]) != set(_GATE_C2_HIDDEN_GEOMETRY)
    ):
        return False
    for path, shape in _GATE_C2_HIDDEN_GEOMETRY.items():
        if not _gate_c2_endpoint_complete(
            before["hidden_paths"][path],
            expected_dtype="<f4",
            expected_shape=shape,
        ) or not _gate_c2_endpoint_complete(
            after["hidden_paths"][path],
            expected_dtype="<f4",
            expected_shape=shape,
        ):
            return False
    before_memory = before["hidden_paths"]["memory_read#0"]
    after_memory = after["hidden_paths"]["memory_read#0"]
    if (
        not gate_a._json_exact(before_memory, source)
        or source["sha256"] == replacement["sha256"]
        or after_memory["dtype"] != replacement["dtype"]
        or after_memory["shape"] != replacement["shape"]
        or after_memory["sha256"] != replacement["sha256"]
    ):
        return False
    changed = sorted(
        path
        for path in _GATE_C2_HIDDEN_GEOMETRY
        if not gate_a._json_exact(
            before["hidden_paths"][path], after["hidden_paths"][path]
        )
    )
    unchanged = sorted(set(_GATE_C2_HIDDEN_GEOMETRY) - set(changed))
    before_tree = _gate_c2_cached_boundary_tree_sha256(before["hidden_paths"])
    after_tree = _gate_c2_cached_boundary_tree_sha256(after["hidden_paths"])
    parameters_equal = bool(
        _sha256_complete(before["parameter_tree_sha256"])
        and _sha256_complete(after["parameter_tree_sha256"])
        and before["parameter_tree_sha256"]
        == after["parameter_tree_sha256"]
    )
    only_memory_read = bool(
        changed == ["memory_read#0"]
        and parameters_equal
    )
    passed = bool(
        only_memory_read
        and before_tree != after_tree
    )
    return bool(
        before["hidden_state_tree_sha256"] == before_tree
        and after["hidden_state_tree_sha256"] == after_tree
        and value["changed_paths"] == changed
        and value["unchanged_paths"] == unchanged
        and isinstance(value["parameters_equal"], bool)
        and value["parameters_equal"] is parameters_equal
        and isinstance(value["only_memory_read_replaced"], bool)
        and value["only_memory_read_replaced"] is only_memory_read
        and isinstance(value["passed"], bool)
        and value["passed"] is passed
        and (passed or not require_pass)
    )


def _gate_c2_cached_read_probe_complete(
    value: Any,
    *,
    regime: str,
    start_tick: str,
    require_pass: bool = True,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "source_cached_memory_read",
        "plus_11",
        "minus_11",
        "passed",
    }:
        return False
    source = value["source_cached_memory_read"]
    if not _gate_c2_endpoint_complete(
        source,
        expected_dtype="<f4",
        expected_shape=(512, 32),
    ):
        return False
    ticks = GATE_C2_LATENT_TICKS[regime]
    expected_suffix = ticks[ticks.index(start_tick) :]
    sentinel_passes = []
    for name, spec in GATE_C2_CACHED_READ_REPLACEMENTS.items():
        sentinel = value[name]
        if not isinstance(sentinel, Mapping) or set(sentinel) != {
            "replacement",
            "boundary",
            "selected_read",
            "selected_drive",
            "continuation",
            "passed",
        }:
            return False
        replacement = sentinel["replacement"]
        continuation = sentinel["continuation"]
        if (
            not _gate_c2_replacement_record_complete(replacement, spec=spec)
            or not _gate_c2_cached_boundary_complete(
                sentinel["boundary"],
                source=source,
                replacement=replacement,
                require_pass=False,
            )
            or not _gate_c2_zero_array_record_complete(
                sentinel["selected_read"],
                expected_dtype="<f4",
                expected_shape=(512, 32),
                require_zero=False,
            )
            or not _gate_c2_zero_array_record_complete(
                sentinel["selected_drive"],
                expected_dtype="<f4",
                expected_shape=(512, 2_048),
                require_zero=False,
            )
            or not isinstance(continuation, Mapping)
            or set(continuation) != {"ticks", "passed"}
            or not isinstance(continuation["ticks"], Mapping)
            or set(continuation["ticks"]) != set(expected_suffix)
        ):
            return False
        comparisons_valid = all(
            _gate_c2_continuation_record_complete(
                continuation["ticks"][tick],
                count=512,
                include_context_memory=True,
                require_pass=False,
            )
            for tick in expected_suffix
        )
        if not comparisons_valid:
            return False
        comparisons_pass = all(
            continuation["ticks"][tick]["passed"] is True
            for tick in expected_suffix
        )
        if continuation["passed"] is not comparisons_pass:
            return False
        sentinel_pass = bool(
            comparisons_pass
            and continuation["passed"] is comparisons_pass
            and sentinel["boundary"]["passed"] is True
            and sentinel["selected_read"]["exact_zero"] is True
            and sentinel["selected_drive"]["exact_zero"] is True
        )
        if not isinstance(sentinel["passed"], bool) or sentinel["passed"] is not sentinel_pass:
            return False
        sentinel_passes.append(sentinel_pass)
    probe_pass = all(sentinel_passes)
    return bool(
        isinstance(value["passed"], bool)
        and value["passed"] is probe_pass
        and (probe_pass or not require_pass)
    )


def _gate_c2_raw_array_record_complete(
    value: Any,
    expected: np.ndarray,
    *,
    include_values: bool = False,
    fill_value: float | None = None,
    extra_keys: tuple[str, ...] = (),
) -> bool:
    array = np.ascontiguousarray(np.asarray(expected))
    keys = {"dtype", "shape", "sha256", *extra_keys}
    if include_values:
        keys.add("values")
    if fill_value is not None:
        keys.add("fill_value")
    if not isinstance(value, Mapping) or set(value) != keys:
        return False
    try:
        return bool(
            value["dtype"] == array.dtype.str
            and value["shape"] == list(array.shape)
            and value["sha256"] == legacy._digest_arrays(array)
            and (
                not include_values
                or gate_a._json_exact(value["values"], array.tolist())
            )
            and (
                fill_value is None
                or _finite_real(value["fill_value"], "array fill")
                == float(fill_value)
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _gate_c2_removed_path_expected_arrays(regime: str) -> dict[str, np.ndarray]:
    if regime == "gate_a":
        config = gate_a.BindingGateConfig()
        metadata = GATE_C2_REMOVED_PATH_OBJECTIVES["gate_a_h1"][
            "source_metadata"
        ]
        events, targets, _, _, _ = legacy._encode_mapping_episodes(
            np.asarray([metadata["mapping_id"]], dtype=np.int64),
            seed=config.validation_episode_seed,
            config=config,
            controls=True,
        )
        source_events = np.ascontiguousarray(events[0])
        source_targets = np.ascontiguousarray(
            np.asarray([targets[0]], dtype=np.int32)
        )
        source_advances = np.ones((6,), dtype=np.bool_)
        canonical_weights = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
            dtype=np.float32,
        )
        h0_end = 5
        continuation_targets = np.asarray([8], dtype=np.int32)
        base_weight = np.float32(0.5)
    else:
        config = gate_b.DepthGateConfig()
        metadata = GATE_C2_REMOVED_PATH_OBJECTIVES[
            "gate_b_index0_r8_h8"
        ]["source_metadata"]
        mapping = np.asarray(metadata["mapping"], dtype=np.int32)
        encoded = gate_b._encode_cycle_episode(
            mapping,
            int(metadata["query_color"]),
            np.asarray(metadata["presentation_order"], dtype=np.int32),
            config.row_config,
        )
        source_events = np.zeros(
            (config.sequence_length, config.row_config.input_width),
            dtype=np.float32,
        )
        source_events[: encoded.shape[0]] = encoded
        source_advances = np.ones((config.sequence_length,), dtype=np.bool_)
        source_targets = np.zeros((config.sequence_length,), dtype=np.int32)
        source_targets[10:] = np.asarray(
            metadata["h0_through_h8_targets"],
            dtype=np.int32,
        )
        canonical_weights = np.ascontiguousarray(
            np.asarray(
                gate_b._checkpoint_contract(
                    mapping,
                    int(metadata["query_color"]),
                    8,
                ).loss_weights
            )
        )
        h0_end = 11
        continuation_targets = source_targets[h0_end:]
        base_weight = np.float32(1.0 / 9.0)
    continuation_events = np.ascontiguousarray(source_events[h0_end:])
    continuation_advances = np.ascontiguousarray(source_advances[h0_end:])
    selection = np.zeros((continuation_events.shape[0],), dtype=np.bool_)
    selection[-1] = True
    base_weights = np.full(selection.shape, base_weight, dtype=np.float32)
    effective_weights = np.where(
        selection,
        base_weight,
        np.float32(0.0),
    ).astype(np.float32)
    packed = np.concatenate(
        (
            continuation_events[:, None, :],
            continuation_advances[:, None, None].astype(np.float32),
            continuation_targets[:, None, None].astype(np.float32),
            effective_weights[:, None, None],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    return {
        "events": source_events,
        "advances": source_advances,
        "targets": source_targets,
        "canonical_loss_weights": canonical_weights,
        "h0_prefix": np.ascontiguousarray(source_events[:h0_end]),
        "continuation_events": continuation_events,
        "batched_events": np.ascontiguousarray(continuation_events[:, None, :]),
        "continuation_advances": continuation_advances,
        "continuation_targets": np.ascontiguousarray(continuation_targets),
        "selection_mask": selection,
        "base_checkpoint_weights": base_weights,
        "effective_loss_weights": effective_weights,
        "packed_inputs": np.ascontiguousarray(packed),
        "base_checkpoint_weight": np.asarray([base_weight], dtype=np.float32),
    }


def _gate_c2_loss_record_complete(
    value: Any,
    *,
    require_nonzero: bool = True,
) -> tuple[bool, np.float32]:
    if not isinstance(value, Mapping) or set(value) != {
        "dtype",
        "shape",
        "sha256",
        "value",
        "finite",
        "nonzero",
    }:
        return False, np.float32(0.0)
    try:
        retained = np.float32(_finite_real(value["value"], "retained loss"))
        array = np.asarray([retained], dtype=np.float32)
        nonzero = bool(retained > np.float32(0.0))
        complete = bool(
            value["dtype"] == "<f4"
            and value["shape"] == [1]
            and value["sha256"] == legacy._digest_arrays(array)
            and float(retained) == float(value["value"])
            and retained >= np.float32(0.0)
            and value["finite"] is True
            and isinstance(value["nonzero"], bool)
            and value["nonzero"] is nonzero
            and (nonzero or not require_nonzero)
        )
        return complete, retained
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, np.float32(0.0)


def _gate_c2_live_gradient_record_complete(
    value: Any,
    *,
    path: str,
    require_nonzero: bool = True,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "tree_paths",
        "leaf_count",
        "leaves",
        "value_count",
        "l2_norm",
        "sha256",
        "finite",
        "nonzero",
    }:
        return False
    leaves = value["leaves"]
    if (
        value["tree_paths"] != [path]
        or not isinstance(leaves, list)
        or not leaves
        or not _strict_integer(value["leaf_count"])
        or int(value["leaf_count"]) != len(leaves)
        or not _strict_integer(value["value_count"])
        or int(value["value_count"]) <= 0
        or not _sha256_complete(value["sha256"])
        or value["sha256"] == "0" * 64
        or value["finite"] is not True
        or not isinstance(value["nonzero"], bool)
    ):
        return False
    norm = _finite_real(value["l2_norm"], "live gradient norm")
    if norm < 0.0:
        return False
    count = 0
    zero_count = 0
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, Mapping) or set(leaf) != {
            "index",
            "dtype",
            "shape",
            "value_count",
            "finite_count",
            "nonfinite_count",
            "zero_count",
            "sha256",
        }:
            return False
        leaf_count = leaf["value_count"]
        if (
            not _strict_integer(leaf["index"])
            or int(leaf["index"]) != index
            or leaf["dtype"] != "<f4"
            or not isinstance(leaf["shape"], list)
            or not leaf["shape"]
            or not all(_strict_integer(item) and int(item) > 0 for item in leaf["shape"])
            or not _strict_integer(leaf_count)
            or int(leaf_count) != math.prod(int(item) for item in leaf["shape"])
            or not _strict_integer(leaf["finite_count"])
            or int(leaf["finite_count"]) != int(leaf_count)
            or not _strict_integer(leaf["nonfinite_count"])
            or int(leaf["nonfinite_count"]) != 0
            or not _strict_integer(leaf["zero_count"])
            or not 0 <= int(leaf["zero_count"]) <= int(leaf_count)
            or not _sha256_complete(leaf["sha256"])
        ):
            return False
        count += int(leaf_count)
        zero_count += int(leaf["zero_count"])
    nonzero = bool(norm > 0.0)
    return bool(
        count == int(value["value_count"])
        and (zero_count < count) is nonzero
        and value["nonzero"] is nonzero
        and (nonzero or not require_nonzero)
    )


def _gate_c2_removed_path_influence_complete(
    value: Any,
    *,
    regime: str,
    canonical_parameter_sha256: str,
    require_pass: bool = True,
) -> bool:
    try:
        _regime_spec(regime)
        if not isinstance(value, Mapping) or set(value) != {
            "gradient_chunk_size",
            "start_state",
            "objectives",
            "global",
            "live_paths",
            "removed_paths",
            "complete",
        }:
            return False
        objective_name = (
            "gate_a_h1" if regime == "gate_a" else "gate_b_index0_r8_h8"
        )
        expected = GATE_C2_REMOVED_PATH_OBJECTIVES[objective_name]
        arrays = _gate_c2_removed_path_expected_arrays(regime)
        if (
            not _strict_integer(value["gradient_chunk_size"])
            or int(value["gradient_chunk_size"])
            != GATE_C2_REMOVED_PATH_GRADIENT_CHUNK_SIZE
            or value["start_state"] != GATE_C2_REMOVED_PATH_START_STATE
            or not isinstance(value["objectives"], Mapping)
            or set(value["objectives"]) != {objective_name}
        ):
            return False
        objective = value["objectives"][objective_name]
        coordinate_keys = {
            "regime",
            "stream",
            "validation_episode_index",
            "batch_size",
            "checkpoint",
        }
        if regime == "gate_b":
            coordinate_keys.add("effort")
        if not isinstance(objective, Mapping) or set(objective) != coordinate_keys | {
            "source_contract",
            "continuation",
            "raw_cross_entropy",
            "base_checkpoint_weight",
            "weighted_cross_entropy",
            "passed",
        }:
            return False
        if any(objective[name] != expected[name] for name in coordinate_keys):
            return False
        source = objective["source_contract"]
        if not isinstance(source, Mapping) or set(source) != {
            "metadata",
            "events",
            "advances",
            "targets",
            "canonical_loss_weights",
            "h0_prefix",
            "schedule_cross_bound",
        }:
            return False
        if (
            not gate_a._json_exact(source["metadata"], expected["source_metadata"])
            or not gate_a._json_exact(
                source["schedule_cross_bound"],
                expected["schedule_sha256"],
            )
            or not _gate_c2_raw_array_record_complete(
                source["events"], arrays["events"]
            )
            or not _gate_c2_raw_array_record_complete(
                source["advances"],
                arrays["advances"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                source["targets"],
                arrays["targets"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                source["canonical_loss_weights"],
                arrays["canonical_loss_weights"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                source["h0_prefix"],
                arrays["h0_prefix"],
                extra_keys=("source_indices",),
            )
            or source["h0_prefix"]["source_indices"]
            != list(range(arrays["h0_prefix"].shape[0]))
        ):
            return False
        for name, array_name in (
            ("events", "events"),
            ("advances", "advances"),
            ("targets", "targets"),
            ("canonical_loss_weights", "canonical_loss_weights"),
            ("h0_prefix", "h0_prefix"),
        ):
            dtype, shape, digest = expected["source_arrays"][name]
            array = arrays[array_name]
            if (
                array.dtype.str != dtype
                or list(array.shape) != shape
                or legacy._digest_arrays(array) != digest
            ):
                return False
        continuation = objective["continuation"]
        if not isinstance(continuation, Mapping) or set(continuation) != {
            "source_indices",
            "source_events",
            "batched_events",
            "advances",
            "targets",
            "selection_mask",
            "base_checkpoint_weights",
            "effective_loss_weights",
            "packed_inputs",
            "h0_gradient_boundary",
            "source_slice_exact",
            "passed",
        }:
            return False
        expected_indices = expected["continuation"]["source_indices"]
        if (
            continuation["source_indices"] != expected_indices
            or not _gate_c2_raw_array_record_complete(
                continuation["source_events"],
                arrays["continuation_events"],
                fill_value=0.0,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["batched_events"],
                arrays["batched_events"],
                fill_value=0.0,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["advances"],
                arrays["continuation_advances"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["targets"],
                arrays["continuation_targets"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["selection_mask"],
                arrays["selection_mask"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["base_checkpoint_weights"],
                arrays["base_checkpoint_weights"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["effective_loss_weights"],
                arrays["effective_loss_weights"],
                include_values=True,
            )
            or not _gate_c2_raw_array_record_complete(
                continuation["packed_inputs"],
                arrays["packed_inputs"],
            )
        ):
            return False
        expected_continuation = expected["continuation"]
        if (
            continuation["source_events"]["sha256"]
            != expected_continuation["source_events_sha256"]
            or continuation["batched_events"]["sha256"]
            != expected_continuation["batched_events_sha256"]
            or continuation["selection_mask"]["values"]
            != expected_continuation["selection_mask_values"]
            or continuation["packed_inputs"]["sha256"]
            != expected_continuation["packed_inputs_sha256"]
        ):
            return False
        if regime == "gate_a":
            if (
                continuation["base_checkpoint_weights"]["values"]
                != expected_continuation["base_checkpoint_weights"]
                or continuation["effective_loss_weights"]["values"]
                != expected_continuation["effective_loss_weights"]
            ):
                return False
        elif (
            objective["base_checkpoint_weight"]["sha256"]
            != expected_continuation["base_checkpoint_weight_sha256"]
            or continuation["effective_loss_weights"]["sha256"]
            != expected_continuation["effective_loss_weights_sha256"]
        ):
            return False
        if (
            not _gate_c2_h0_gradient_boundary_complete(
                continuation["h0_gradient_boundary"],
                canonical_parameter_sha256=canonical_parameter_sha256,
            )
            or continuation["source_slice_exact"] is not True
            or continuation["passed"] is not True
        ):
            return False
        raw_complete, raw_loss = _gate_c2_loss_record_complete(
            objective["raw_cross_entropy"],
            require_nonzero=False,
        )
        weighted_complete, weighted_loss = _gate_c2_loss_record_complete(
            objective["weighted_cross_entropy"],
            require_nonzero=False,
        )
        if (
            not raw_complete
            or not weighted_complete
            or not _gate_c2_raw_array_record_complete(
                objective["base_checkpoint_weight"],
                arrays["base_checkpoint_weight"],
                include_values=True,
            )
        ):
            return False
        recomputed_weighted = np.float32(
            raw_loss * arrays["base_checkpoint_weight"][0]
        )
        objective_pass = bool(
            raw_loss > np.float32(0.0)
            and weighted_loss > np.float32(0.0)
            and weighted_loss == recomputed_weighted
        )
        if (
            not isinstance(objective["passed"], bool)
            or objective["passed"] is not objective_pass
        ):
            return False
        global_record = value["global"]
        if not isinstance(global_record, Mapping) or set(global_record) != {
            "tree_paths",
            "leaf_count",
            "value_count",
            "l2_norm",
            "sha256",
            "finite",
            "nonzero",
        }:
            return False
        global_norm = _finite_real(
            global_record["l2_norm"], "global gradient norm"
        )
        global_nonzero = bool(global_norm > 0.0)
        if (
            global_record["tree_paths"] != list(FULL_PARAMETER_PATHS)
            or not _strict_integer(global_record["leaf_count"])
            or int(global_record["leaf_count"]) <= 0
            or not _strict_integer(global_record["value_count"])
            or int(global_record["value_count"]) <= 0
            or global_norm < 0.0
            or not _sha256_complete(global_record["sha256"])
            or global_record["sha256"] == "0" * 64
            or global_record["finite"] is not True
            or not isinstance(global_record["nonzero"], bool)
            or global_record["nonzero"] is not global_nonzero
        ):
            return False
        live = value["live_paths"]
        if (
            not isinstance(live, Mapping)
            or set(live) != set(GATE_C2_LIVE_PATHS)
        ):
            return False
        if not all(
            _gate_c2_live_gradient_record_complete(
                live[path], path=path, require_nonzero=False
            )
            for path in GATE_C2_LIVE_PATHS
        ):
            return False
        removed = value["removed_paths"]
        if (
            not isinstance(removed, Mapping)
            or set(removed) != set(GATE_C2_REMOVED_PATHS)
        ):
            return False
        removed_geometry = {
            "memory_read_projection/weight": (32, 2_048),
            "workspace_query_projection/weight": (2_048, 32),
        }
        removed_zero: dict[str, bool] = {}
        for path, shape in removed_geometry.items():
            record = removed[path]
            zero = np.zeros(shape, dtype=np.float32)
            expected_count = int(zero.size)
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {
                    "tree_paths",
                    "leaf_count",
                    "leaves",
                    "value_count",
                    "l2_norm",
                    "sha256",
                    "finite",
                    "exact_zero",
                }
                or record["tree_paths"] != [path]
                or not _strict_integer(record["leaf_count"])
                or int(record["leaf_count"]) != 1
                or not _strict_integer(record["value_count"])
                or int(record["value_count"]) != expected_count
                or not _sha256_complete(record["sha256"])
                or record["finite"] is not True
                or not isinstance(record["exact_zero"], bool)
                or not isinstance(record["leaves"], list)
                or len(record["leaves"]) != 1
            ):
                return False
            leaf = record["leaves"][0]
            if (
                not isinstance(leaf, Mapping)
                or set(leaf)
                != {
                    "index",
                    "dtype",
                    "shape",
                    "value_count",
                    "finite_count",
                    "nonfinite_count",
                    "zero_count",
                    "sha256",
                }
                or not _strict_integer(leaf["index"])
                or int(leaf["index"]) != 0
                or leaf["dtype"] != "<f4"
                or leaf["shape"] != list(shape)
                or not _strict_integer(leaf["value_count"])
                or int(leaf["value_count"]) != expected_count
                or not _strict_integer(leaf["finite_count"])
                or int(leaf["finite_count"]) != expected_count
                or not _strict_integer(leaf["nonfinite_count"])
                or int(leaf["nonfinite_count"]) != 0
                or not _strict_integer(leaf["zero_count"])
                or not 0 <= int(leaf["zero_count"]) <= expected_count
                or not _sha256_complete(leaf["sha256"])
            ):
                return False
            norm = _finite_real(record["l2_norm"], "removed gradient norm")
            exact_zero = bool(
                int(leaf["zero_count"]) == expected_count and norm == 0.0
            )
            if (
                norm < 0.0
                or (int(leaf["zero_count"]) == expected_count) is not (norm == 0.0)
                or record["exact_zero"] is not exact_zero
                or (
                    exact_zero
                    and (
                        record["sha256"] != _gradient_path_sha256(path, zero)
                        or leaf["sha256"] != legacy._digest_arrays(zero)
                    )
                )
                or (
                    not exact_zero
                    and (
                        record["sha256"] == _gradient_path_sha256(path, zero)
                        or leaf["sha256"] == legacy._digest_arrays(zero)
                    )
                )
            ):
                return False
            removed_zero[path] = exact_zero
        recomputed_complete = bool(
            objective_pass
            and global_nonzero
            and all(live[path]["nonzero"] for path in GATE_C2_LIVE_PATHS)
            and all(removed_zero[path] for path in GATE_C2_REMOVED_PATHS)
        )
        return bool(
            isinstance(value["complete"], bool)
            and value["complete"] is recomputed_complete
            and (recomputed_complete or not require_pass)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _gate_c2_query_only_latent_no_read_complete(
    value: Any,
    admission: Any,
    *,
    regime: str,
    require_pass: bool = True,
) -> bool:
    """Validate every retained no-read intervention and nested aggregate."""

    try:
        _regime_spec(regime)
        if not isinstance(admission, Mapping):
            return False
        canonical_parameter_sha256 = admission["initialization"][regime][
            "canonical_full"
        ]["parameter_sha256"]
        if (
            not _sha256_complete(canonical_parameter_sha256)
            or canonical_parameter_sha256 == "0" * 64
        ):
            return False
        if not isinstance(value, Mapping) or set(value) != {
            "streams",
            "perturbations",
            "full_positive_control",
            "removed_path_finite_window_influence",
            "passed",
        }:
            return False
        ticks = GATE_C2_LATENT_TICKS[regime]
        stream_names = ("intact", "shuffled", "no_context")
        streams = value["streams"]
        if not isinstance(streams, Mapping) or set(streams) != set(stream_names):
            return False
        parameter_sha256s: set[str] = set()
        selected_passes = []
        for stream_name in stream_names:
            stream = streams[stream_name]
            if not isinstance(stream, Mapping) or set(stream) != set(ticks):
                return False
            for tick_name in ticks:
                tick = stream[tick_name]
                if (
                    not isinstance(tick, Mapping)
                    or set(tick)
                    != {
                        "selected_read",
                        "selected_drive",
                        "cached_read_probe",
                        "cached_h0_read_reused",
                    }
                    or not _gate_c2_zero_array_record_complete(
                        tick["selected_read"],
                        expected_dtype="<f4",
                        expected_shape=(512, 32),
                        require_zero=False,
                    )
                    or not _gate_c2_zero_array_record_complete(
                        tick["selected_drive"],
                        expected_dtype="<f4",
                        expected_shape=(512, 2_048),
                        require_zero=False,
                    )
                    or not _gate_c2_cached_read_probe_complete(
                        tick["cached_read_probe"],
                        regime=regime,
                        start_tick=tick_name,
                        require_pass=False,
                    )
                ):
                    return False
                probe = tick["cached_read_probe"]
                expected_reused = bool(not probe["passed"])
                if (
                    not isinstance(tick["cached_h0_read_reused"], bool)
                    or tick["cached_h0_read_reused"] is not expected_reused
                ):
                    return False
                for name in GATE_C2_CACHED_READ_REPLACEMENTS:
                    boundary = probe[name]["boundary"]
                    parameter_sha256s.update(
                        (
                            boundary["before_replacement"][
                                "parameter_tree_sha256"
                            ],
                            boundary["after_replacement"][
                                "parameter_tree_sha256"
                            ],
                        )
                    )
                selected_passes.append(
                    bool(
                        tick["selected_read"]["exact_zero"]
                        and tick["selected_drive"]["exact_zero"]
                        and probe["passed"]
                        and not tick["cached_h0_read_reused"]
                    )
                )

        non_s_k_paths = tuple(
            path
            for path in _GATE_C2_HIDDEN_GEOMETRY
            if path != "context_memory#0"
        )

        def intervention_tick_complete(tick: Any, *, positive: bool) -> bool:
            common_keys = {
                "source_s_k_sha256",
                "replacement_s_k_sha256",
                "source_replacement_differ",
                "non_s_k_state",
                "parameters",
                "continuation",
                "passed",
            }
            selected_keys = (
                {"selected_read_difference", "selected_drive_difference"}
                if positive
                else {"selected_read", "selected_drive"}
            )
            if (
                not isinstance(tick, Mapping)
                or set(tick) != common_keys | selected_keys
                or not _sha256_complete(tick["source_s_k_sha256"])
                or not _sha256_complete(tick["replacement_s_k_sha256"])
                or not isinstance(tick["source_replacement_differ"], bool)
                or not _gate_c2_equality_record_complete(
                    tick["non_s_k_state"],
                    expected_paths=non_s_k_paths,
                    framing="nul_joined_gate_c2_non_s_k_state_v1",
                    require_equal=False,
                )
                or not _gate_c2_equality_record_complete(
                    tick["parameters"],
                    expected_paths=FULL_PARAMETER_PATHS,
                    framing="authenticated_gate_c_parameter_array_digest_v1",
                    require_equal=False,
                )
                or not _gate_c2_continuation_record_complete(
                    tick["continuation"],
                    count=512,
                    include_context_memory=False,
                    require_pass=False,
                )
            ):
                return False
            source_differ = bool(
                tick["source_s_k_sha256"] != tick["replacement_s_k_sha256"]
            )
            if tick["source_replacement_differ"] is not source_differ:
                return False
            parameter_sha256s.update(
                (
                    tick["parameters"]["left_value_sha256"],
                    tick["parameters"]["right_value_sha256"],
                )
            )
            if positive:
                selected_valid = all(
                    _gate_c2_floating_difference_record_complete(
                        tick[field],
                        expected_dtype="<f4",
                        expected_shape=shape,
                        rms_tolerance=1e-6,
                        require_within=False,
                    )
                    and _gate_c2_difference_endpoints_coherent(tick[field])
                    for field, shape in (
                        ("selected_read_difference", (512, 32)),
                        ("selected_drive_difference", (512, 2_048)),
                    )
                )
            else:
                selected_valid = bool(
                    _gate_c2_zero_array_record_complete(
                        tick["selected_read"],
                        expected_dtype="<f4",
                        expected_shape=(512, 32),
                        require_zero=False,
                    )
                    and _gate_c2_zero_array_record_complete(
                        tick["selected_drive"],
                        expected_dtype="<f4",
                        expected_shape=(512, 2_048),
                        require_zero=False,
                    )
                )
            boundary_pass = bool(
                source_differ
                and tick["non_s_k_state"]["tree_equal"]
                and tick["non_s_k_state"]["values_equal"]
                and tick["parameters"]["tree_equal"]
                and tick["parameters"]["values_equal"]
            )
            recomputed_pass = bool(
                boundary_pass
                and selected_valid
                and (
                    positive
                    or (
                        tick["selected_read"]["exact_zero"]
                        and tick["selected_drive"]["exact_zero"]
                        and tick["continuation"]["passed"] is True
                    )
                )
            )
            return bool(
                isinstance(tick["passed"], bool)
                and tick["passed"] is recomputed_pass
            )

        perturbations = value["perturbations"]
        if (
            not isinstance(perturbations, Mapping)
            or set(perturbations) != set(GATE_C2_CONTEXT_MEMORY_REPLACEMENTS)
        ):
            return False
        perturbation_passes = []
        for name in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS:
            replacement_value = perturbations[name]
            spec = GATE_C2_CONTEXT_MEMORY_REPLACEMENTS[name]
            if (
                not isinstance(replacement_value, Mapping)
                or set(replacement_value) != {"replacement", "streams", "passed"}
                or not _gate_c2_replacement_record_complete(
                    replacement_value["replacement"],
                    spec=spec,
                )
                or not isinstance(replacement_value["streams"], Mapping)
                or set(replacement_value["streams"]) != set(stream_names)
            ):
                return False
            tick_passes = []
            for stream_name in stream_names:
                stream = replacement_value["streams"][stream_name]
                if not isinstance(stream, Mapping) or set(stream) != set(ticks):
                    return False
                for tick_name in ticks:
                    tick = stream[tick_name]
                    if tick.get("replacement_s_k_sha256") != spec["sha256"]:
                        return False
                    if not intervention_tick_complete(tick, positive=False):
                        return False
                    tick_passes.append(tick["passed"])
            replacement_pass = all(tick_passes)
            if (
                not isinstance(replacement_value["passed"], bool)
                or replacement_value["passed"] is not replacement_pass
                or (not replacement_pass and require_pass)
            ):
                return False
            perturbation_passes.append(replacement_pass)

        positive = value["full_positive_control"]
        if not isinstance(positive, Mapping) or set(positive) != {
            *GATE_C2_CONTEXT_MEMORY_REPLACEMENTS,
            "passed",
        }:
            return False
        positive_passes = []
        for name, spec in GATE_C2_CONTEXT_MEMORY_REPLACEMENTS.items():
            replacement_value = positive[name]
            if (
                not isinstance(replacement_value, Mapping)
                or set(replacement_value) != {"replacement", "streams", "passed"}
                or not _gate_c2_replacement_record_complete(
                    replacement_value["replacement"],
                    spec=spec,
                )
                or not isinstance(replacement_value["streams"], Mapping)
                or set(replacement_value["streams"]) != set(stream_names)
            ):
                return False
            tick_passes = []
            nonzero = False
            for stream_name in stream_names:
                stream = replacement_value["streams"][stream_name]
                if not isinstance(stream, Mapping) or set(stream) != set(ticks):
                    return False
                for tick_name in ticks:
                    tick = stream[tick_name]
                    if tick.get("replacement_s_k_sha256") != spec["sha256"]:
                        return False
                    if not intervention_tick_complete(tick, positive=True):
                        return False
                    tick_passes.append(tick["passed"])
                    nonzero = bool(
                        nonzero
                        or tick["selected_read_difference"][
                            "max_abs_difference"
                        ]
                        > 0.0
                        or tick["selected_drive_difference"][
                            "max_abs_difference"
                        ]
                        > 0.0
                    )
            replacement_pass = bool(nonzero and all(tick_passes))
            if (
                not isinstance(replacement_value["passed"], bool)
                or replacement_value["passed"] is not replacement_pass
                or (not replacement_pass and require_pass)
            ):
                return False
            positive_passes.append(replacement_pass)
        positive_pass = all(positive_passes)
        if (
            not isinstance(positive["passed"], bool)
            or positive["passed"] is not positive_pass
            or (not positive_pass and require_pass)
        ):
            return False
        removed_kwargs: dict[str, Any] = {
            "regime": regime,
            "canonical_parameter_sha256": canonical_parameter_sha256,
        }
        if not require_pass:
            removed_kwargs["require_pass"] = False
        removed_pass = _gate_c2_removed_path_influence_complete(
            value["removed_path_finite_window_influence"],
            **removed_kwargs,
        )
        if not removed_pass:
            return False
        removed_complete = value["removed_path_finite_window_influence"][
            "complete"
        ]
        if parameter_sha256s != {canonical_parameter_sha256}:
            return False
        recomputed = bool(
            all(selected_passes)
            and all(perturbation_passes)
            and positive_pass
            and removed_complete
        )
        return bool(
            isinstance(value["passed"], bool)
            and value["passed"] is recomputed
            and (recomputed or not require_pass)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _paired_h0_identity_report(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    regime: str,
    data: Any,
) -> dict[str, Any]:
    """Compare fresh full and query-only models at checkpoint H0."""

    _regime_spec(regime)
    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    try:
        canonical = initialization["initialization"][regime]["canonical_full"]
    except (KeyError, TypeError) as error:
        raise ValueError("Gate C H0 initialization evidence is incomplete") from error
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    count = regime_config.validation_episodes
    if regime == "gate_a":
        if not isinstance(data, legacy.BindingData):
            raise TypeError("Gate A H0 identity requires BindingData")
        streams = {
            "intact": data.validation_intact,
            "shuffled": data.validation_shuffled,
            "no_context": data.validation_no_context,
        }
        advances = np.ones(
            (regime_config.sequence_length, count), dtype=np.bool_
        )
    else:
        if (
            not isinstance(data, tuple)
            or len(data) != 2
            or not isinstance(data[1], gate_b.DepthValidationData)
        ):
            raise TypeError("Gate B H0 identity requires DepthValidationData")
        validation = data[1]
        streams = {
            "intact": validation.intact,
            "shuffled": validation.shuffled,
            "no_context": validation.no_context,
        }
        advances = np.asarray(validation.advance_masks)
    checkpoint_index = regime_config.sequence_length - regime_config.gap_steps - 1
    models = {
        arm: _new_model_for_arm(
            config,
            regime,
            arm,
            batch_size=count,
        )
        for arm in ("full", "query_only")
    }
    legacy._copy_parameters(models["full"], models["query_only"])
    parameter_sha256 = {
        arm: legacy._array_digest(legacy._parameter_values(model))
        for arm, model in models.items()
    }
    if (
        not isinstance(canonical, Mapping)
        or any(
            digest != canonical.get("parameter_sha256")
            for digest in parameter_sha256.values()
        )
    ):
        raise ValueError("Gate C H0 models did not reproduce initialization")
    initial_snapshot = models["full"].snapshot_state()
    models["query_only"].restore_state(initial_snapshot)

    def driver(model: LatentWorkspaceModel) -> Any:
        @brainstate.transform.jit
        def run_h0(events: jax.Array, advance_values: jax.Array) -> jax.Array:
            def step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
                event, advance = inputs
                return model.update(event, advance)

            compact = brainstate.transform.for_loop(
                step,
                (events, advance_values),
            )
            return compact[-1]

        return run_h0

    drivers = {arm: driver(model) for arm, model in models.items()}
    stream_reports: dict[str, Any] = {}
    for name, event_values in streams.items():
        compact_sha256: dict[str, str] = {}
        state_sha256: dict[str, str] = {}
        for arm in ("full", "query_only"):
            models[arm].restore_state(initial_snapshot)
            compact = jax.block_until_ready(
                drivers[arm](
                    jnp.asarray(event_values[: checkpoint_index + 1]),
                    jnp.asarray(advances[: checkpoint_index + 1]),
                )
            )
            compact_sha256[arm] = legacy._digest_arrays(np.asarray(compact))
            state_sha256[arm] = _hidden_state_sha256(models[arm])
        stream_reports[name] = {
            "full_compact_sha256": compact_sha256["full"],
            "query_only_compact_sha256": compact_sha256["query_only"],
            "full_state_sha256": state_sha256["full"],
            "query_only_state_sha256": state_sha256["query_only"],
            "compact_byte_identical": (
                compact_sha256["full"] == compact_sha256["query_only"]
            ),
            "state_byte_identical": (
                state_sha256["full"] == state_sha256["query_only"]
            ),
        }
    passed = all(
        evidence["compact_byte_identical"] is True
        and evidence["state_byte_identical"] is True
        for evidence in stream_reports.values()
    )
    return {
        "checkpoint": 0,
        "initialization_parameter_sha256": parameter_sha256,
        "streams": stream_reports,
        "passed": bool(passed),
    }


def _numeric_gradient_leaves(value: Any) -> list[np.ndarray]:
    leaves: list[np.ndarray] = []
    for leaf in jax.tree.leaves(value):
        array = np.ascontiguousarray(np.asarray(u.get_mantissa(leaf)))
        if np.issubdtype(array.dtype, np.bool_):
            raise TypeError("gradient leaves must be numeric, not boolean")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("gradient leaves must be numeric")
        if not np.isfinite(array).all():
            raise ValueError("gradient leaves must be finite")
        leaves.append(array)
    if not leaves:
        raise ValueError("gradient tree has no leaves")
    return leaves


def _shared_path_sha256(path: str, value: Any) -> str:
    """Hash one shared initialization subtree with Gate C framing."""

    if not isinstance(path, str) or not path:
        raise TypeError("shared parameter path must be a nonempty string")
    fields: list[bytes] = [
        b"example21-gate-c-shared-path-v1",
        path.encode("utf-8"),
    ]
    for index, array in enumerate(_numeric_gradient_leaves(value)):
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _shared_global_sha256(values: Mapping[str, Any]) -> str:
    """Hash all shared initialization paths in canonical name order."""

    if not isinstance(values, Mapping) or not values:
        raise TypeError("shared values must be a nonempty mapping")
    fields: list[bytes] = [b"example21-gate-c-shared-global-v1"]
    for path in sorted(values):
        fields.extend(
            (
                path.encode("utf-8"),
                _shared_path_sha256(path, values[path]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_path_sha256(path: str, value: Any) -> str:
    """Hash one gradient subtree with the frozen Gate C framing."""

    if not isinstance(path, str) or not path:
        raise TypeError("gradient path must be a nonempty string")
    fields: list[bytes] = [
        b"example21-gate-c-gradient-path-v1",
        path.encode("utf-8"),
    ]
    for index, array in enumerate(_numeric_gradient_leaves(value)):
        fields.extend(
            (
                str(index).encode("ascii"),
                array.dtype.str.encode("ascii"),
                ",".join(map(str, array.shape)).encode("ascii"),
                array.tobytes(),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_global_sha256(gradients: Mapping[str, Any]) -> str:
    """Hash all gradient paths in canonical name order."""

    if not isinstance(gradients, Mapping) or not gradients:
        raise TypeError("gradients must be a nonempty mapping")
    fields: list[bytes] = [b"example21-gate-c-gradient-global-v1"]
    for path in sorted(gradients):
        fields.extend(
            (
                path.encode("utf-8"),
                _gradient_path_sha256(path, gradients[path]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_comparison(full: Any, arm: Any) -> dict[str, Any]:
    """Compare two flattened gradients using the full norm denominator."""

    if jax.tree.structure(full) != jax.tree.structure(arm):
        raise ValueError("gradient trees must have the same structure")
    full_leaves = _numeric_gradient_leaves(full)
    arm_leaves = _numeric_gradient_leaves(arm)
    if any(
        full_leaf.shape != arm_leaf.shape or full_leaf.dtype != arm_leaf.dtype
        for full_leaf, arm_leaf in zip(full_leaves, arm_leaves, strict=True)
    ):
        raise ValueError("gradient leaves must have the same shape and dtype")
    full_vector = np.concatenate([leaf.astype(np.float64).reshape(-1) for leaf in full_leaves])
    arm_vector = np.concatenate([leaf.astype(np.float64).reshape(-1) for leaf in arm_leaves])
    if full_vector.shape != arm_vector.shape:
        raise ValueError("gradient vectors must have the same shape")
    full_norm = float(np.linalg.norm(full_vector))
    arm_norm = float(np.linalg.norm(arm_vector))
    difference = float(np.linalg.norm(arm_vector - full_vector))
    relative = difference / full_norm if full_norm > 0.0 else None
    cosine = (
        float(np.dot(arm_vector, full_vector) / (arm_norm * full_norm))
        if full_norm > 0.0 and arm_norm > 0.0
        else None
    )
    return {
        "full_norm": full_norm,
        "arm_norm": arm_norm,
        "l2_difference": difference,
        "relative_deviation": relative,
        "relative_deviation_defined": relative is not None,
        "cosine": cosine,
        "cosine_defined": cosine is not None,
    }


def _oracle_contract(config: GateCConfig) -> dict[str, Any]:
    """Return the exact preregistered finite-window mechanism episode."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    return {
        "regime": "gate_b",
        "validation_episode_index": config.oracle_validation_index,
        "arm": "intact",
        "effort": config.oracle_effort,
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
        "gradient_chunk_size": config.gradient_chunk_size,
    }


def _gate_c3_deterministic_environment_complete(value: Any) -> bool:
    """Return whether the exact Gate C3 deterministic environment is bound."""

    return bool(
        isinstance(value, Mapping)
        and set(value) == set(GATE_C3_DETERMINISTIC_ENVIRONMENT)
        and all(
            value[name] == expected
            for name, expected in GATE_C3_DETERMINISTIC_ENVIRONMENT.items()
        )
    )


def _gate_c3_terminal_h8_inputs(
    config: GateCConfig,
    *,
    gate_b_data: Any,
) -> dict[str, np.ndarray]:
    """Materialize the fixed terminal-H8 oracle arrays."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if (
        config.oracle_validation_index != 0
        or config.oracle_effort != 8
        or config.gradient_chunk_size != 1
    ):
        raise ValueError("Gate C3 oracle coordinates differ from preregistration")
    if (
        not isinstance(gate_b_data, tuple)
        or len(gate_b_data) != 2
        or not isinstance(gate_b_data[0], gate_b.DepthSchedule)
        or not isinstance(gate_b_data[1], gate_b.DepthValidationData)
    ):
        raise TypeError("Gate C3 oracle requires canonical Gate B data")
    schedule, validation = gate_b_data
    index = GATE_C3_TERMINAL_H8_OBJECTIVE["validation_episode_index"]
    if not (
        np.array_equal(
            np.asarray(validation.mapping_ids[index]),
            np.asarray(schedule.validation_mapping_ids[index]),
        )
        and np.array_equal(
            np.asarray(validation.query_colors[index]),
            np.asarray(schedule.validation_query_colors[index]),
        )
        and np.array_equal(
            np.asarray(validation.presentation_orders[index]),
            np.asarray(schedule.validation_presentation_orders[index]),
        )
    ):
        raise ValueError("Gate C3 oracle validation metadata differs from schedule")

    events = np.ascontiguousarray(np.asarray(validation.intact[:, index, :]))
    targets = np.zeros((events.shape[0],), dtype=np.int32)
    targets[10:] = np.asarray(
        validation.targets_by_depth[:, index],
        dtype=np.int32,
    )
    advances = np.ascontiguousarray(
        np.asarray(validation.advance_masks[:, index], dtype=np.float32)
    )
    loss_weights = np.zeros((events.shape[0],), dtype=np.float32)
    loss_weights[18] = np.float32(1.0)
    packed_inputs = np.ascontiguousarray(
        np.concatenate(
            (
                events[:, None, :],
                advances[:, None, None],
                targets.astype(np.float32)[:, None, None],
                loss_weights[:, None, None],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
    )
    actual = {
        "mapping_id": int(np.asarray(schedule.validation_mapping_ids[index]).item()),
        "events": _gate_c2_raw_array_record(events),
        "targets": _gate_c2_raw_array_record(targets),
        "advances": _gate_c2_raw_array_record(advances),
        "loss_weights": _gate_c2_raw_array_record(loss_weights),
        "packed_inputs": _gate_c2_raw_array_record(packed_inputs),
    }
    expected = {
        "mapping_id": 232_423,
        "events": {
            "dtype": "<f4",
            "shape": [19, 47],
            "sha256": GATE_C3_TERMINAL_H8_OBJECTIVE["events_sha256"],
        },
        "targets": {
            "dtype": "<i4",
            "shape": [19],
            "sha256": (
                "c4af41cac4f5eb682df15e7d6cf92b0c134b943fae1abfe99b0bfc4c2ddb27e0"
            ),
        },
        "advances": {
            "dtype": "<f4",
            "shape": [19],
            "sha256": (
                "d69cc2400af318c684ba7c8ba0d66204f25264b3bcbba9d8d96d999bdefc4a07"
            ),
        },
        "loss_weights": {
            "dtype": "<f4",
            "shape": [19],
            "sha256": (
                "07fecad3bfcbd816df57ab71c500db391cbf3b581a99376678d0e5f9da8e6693"
            ),
        },
        "packed_inputs": {
            "dtype": "<f4",
            "shape": [19, 1, 50],
            "sha256": (
                "ef1c75296133458d90de3d5d9c204890127f83238148bd11bc2736bae6a205e1"
            ),
        },
    }
    if not gate_a._json_exact(actual, expected):
        raise ValueError("Gate C3 terminal-H8 input contract differs")
    return {
        "events": events,
        "targets": targets,
        "advances": advances,
        "loss_weights": loss_weights,
        "packed_inputs": packed_inputs,
    }


def _mechanism_oracle(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    gate_b_data: Any,
) -> dict[str, Any]:
    """Measure the preregistered Gate C finite-window pp-prop mechanism."""

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if (
        config.oracle_validation_index != 0
        or config.oracle_effort != 8
        or config.gradient_chunk_size != 1
        or config.gradient_chunk_size >= REGIME_SPECS["gate_b"].sequence_length
    ):
        raise ValueError("Gate C oracle coordinates differ from preregistration")
    contract = _oracle_contract(config)
    if (
        not isinstance(gate_b_data, tuple)
        or len(gate_b_data) != 2
        or not isinstance(gate_b_data[0], gate_b.DepthSchedule)
        or not isinstance(gate_b_data[1], gate_b.DepthValidationData)
    ):
        raise TypeError("Gate C oracle requires canonical Gate B data")
    schedule, validation = gate_b_data
    index = contract["validation_episode_index"]
    if not (
        np.array_equal(
            np.asarray(validation.mapping_ids[index]),
            np.asarray(schedule.validation_mapping_ids[index]),
        )
        and np.array_equal(
            np.asarray(validation.query_colors[index]),
            np.asarray(schedule.validation_query_colors[index]),
        )
        and np.array_equal(
            np.asarray(validation.presentation_orders[index]),
            np.asarray(schedule.validation_presentation_orders[index]),
        )
    ):
        raise ValueError("Gate C oracle validation metadata differs from schedule")
    events = np.ascontiguousarray(np.asarray(validation.intact[:, index, :]))
    mapping_id = int(np.asarray(schedule.validation_mapping_ids[index]).item())
    actual_contract = {
        "mapping_id": mapping_id,
        "mapping": gate_b.unrank_ten_cycle(mapping_id).tolist(),
        "query_color": int(np.asarray(schedule.validation_query_colors[index]).item()),
        "presentation_order": np.asarray(
            schedule.validation_presentation_orders[index]
        ).tolist(),
        "shuffled_shift": int(np.asarray(validation.shuffled_shifts[index]).item()),
        "targets": np.asarray(validation.targets_by_depth[:, index]).tolist(),
        "advance_mask": np.asarray(validation.advance_masks[:, index]).tolist(),
        "events_shape": list(events.shape),
        "events_dtype": events.dtype.str,
        "events_sha256": legacy._digest_arrays(events),
    }
    if any(
        not gate_a._json_exact(actual_contract[name], contract[name])
        for name in actual_contract
    ):
        raise ValueError("Gate C oracle event contract or digest differs")

    try:
        gate_b_initialization = initialization["initialization"]["gate_b"]
        canonical = gate_b_initialization["canonical_full"]
        arm_refs = gate_b_initialization["arm_initialization_refs"]
    except (KeyError, TypeError) as error:
        raise ValueError("Gate C oracle initialization evidence is incomplete") from error
    if (
        not isinstance(canonical, Mapping)
        or canonical.get("parameter_paths") != list(FULL_PARAMETER_PATHS)
        or not _strict_integer(canonical.get("parameter_count"))
        or not _sha256_complete(canonical.get("parameter_sha256"))
        or not isinstance(arm_refs, Mapping)
        or any(
            not isinstance(arm_refs.get(arm), Mapping)
            or arm_refs[arm].get("tree") != "canonical_full"
            or arm_refs[arm].get("parameter_sha256")
            != canonical["parameter_sha256"]
            for arm in ("full", "query_only", "terminal_only")
        )
    ):
        raise ValueError("Gate C oracle initialization identity differs")

    targets = np.zeros((events.shape[0],), dtype=np.int32)
    targets[10:] = np.asarray(validation.targets_by_depth[:, index], dtype=np.int32)
    advances = np.asarray(validation.advance_masks[:, index], dtype=np.float32)
    full_weights = _loss_weights(
        "gate_b",
        "full",
        efforts=np.asarray([contract["effort"]], dtype=np.int32),
    )[0]
    terminal_weights = _loss_weights(
        "gate_b",
        "terminal_only",
        efforts=np.asarray([contract["effort"]], dtype=np.int32),
    )[0]

    class _OracleObjective(LatentWorkspaceModel):
        def update(self, packed: jax.Array) -> jax.Array:
            expected_width = self.config.input_width + 3
            if packed.ndim != 2 or packed.shape[-1] != expected_width:
                raise ValueError(
                    "Gate C oracle input must contain event, advance, target, and weight"
                )
            event = packed[:, : self.config.input_width]
            advance = packed[:, self.config.input_width] > 0.5
            target = packed[:, self.config.input_width + 1].astype(jnp.int32)
            weight = packed[:, self.config.input_width + 2]
            loss = legacy._classification_loss(
                super().update(event, advance),
                target,
                self.config.color_rank,
            )
            weighted = jnp.sqrt(weight) * jnp.sqrt(jnp.maximum(loss, 0.0))
            return jnp.where(weight == 0.0, jnp.zeros_like(weight), weighted)

    expected_sha256 = canonical["parameter_sha256"]
    expected_count = canonical["parameter_count"]
    oracle_models: dict[str, _OracleObjective] = {}

    def model_factory(arm: str, stage: str) -> Any:
        if stage not in ("reference", "finite_window"):
            raise ValueError("Gate C2 oracle model stage is invalid")

        def factory() -> _OracleObjective:
            policy = "query_only" if arm == "query_only" else "full"
            role = f"gate_b:mechanism_oracle:{arm}:{stage}"

            def construct(
                model_config: ModelConfig,
                actual_policy: str,
            ) -> _OracleObjective:
                return _OracleObjective(
                    model_config,
                    memory_read_policy=actual_policy,
                )

            model = _gate_c2_control_model(
                config,
                initialization,
                regime="gate_b",
                policy=policy,
                batch_size=1,
                role=role,
                probe=f"mechanism_oracle:{arm}:{stage}",
                arm=arm,
                constructor=construct,
            )
            if not isinstance(model, _OracleObjective):
                raise TypeError("Gate C2 oracle factory returned the wrong model")
            values = legacy._parameter_values(model)
            count = sum(
                np.asarray(u.get_mantissa(leaf)).size
                for value in values.values()
                for leaf in jax.tree.leaves(value)
            )
            if (
                tuple(sorted(values)) != FULL_PARAMETER_PATHS
                or count != expected_count
                or legacy._array_digest(values) != expected_sha256
            ):
                raise ValueError("Gate C oracle did not reproduce initialization")
            if role in oracle_models:
                raise RuntimeError(f"duplicate Gate C2 oracle role {role!r}")
            oracle_models[role] = model
            return model

        return factory

    def packed_inputs(weights: np.ndarray) -> jax.Array:
        return jnp.asarray(
            np.concatenate(
                (
                    events[:, None, :],
                    advances[:, None, None],
                    targets[:, None, None].astype(np.float32),
                    np.asarray(weights, dtype=np.float32)[:, None, None],
                ),
                axis=-1,
            ),
            dtype=jnp.float32,
        )

    def algorithm_factory(model: brainstate.nn.Module) -> braintrace.ETraceAlgorithm:
        return braintrace.pp_prop(
            model,
            decay_or_rank=model.config.trace_decay,
            vjp_method="multi-step",
        )

    reference_models = {
        arm: model_factory(arm, "reference")()
        for arm in ("full", "query_only", "terminal_only")
    }
    for model in reference_models.values():
        brainstate.nn.init_all_states(model, batch_size=1)
    reference_parameters = {
        arm: legacy._parameter_values(model)
        for arm, model in reference_models.items()
    }

    def snapshots_equal(left: Any, right: Any) -> bool:
        if (
            left.batch_size != right.batch_size
            or left.neuron_count != right.neuron_count
            or tuple(path for path, _ in left.entries)
            != tuple(path for path, _ in right.entries)
        ):
            return False
        for (_, left_value), (_, right_value) in zip(
            left.entries, right.entries, strict=True
        ):
            if jax.tree.structure(left_value) != jax.tree.structure(right_value):
                return False
            left_leaves = jax.tree.leaves(left_value)
            right_leaves = jax.tree.leaves(right_value)
            for left_leaf, right_leaf in zip(
                left_leaves, right_leaves, strict=True
            ):
                left_array = np.ascontiguousarray(
                    np.asarray(u.get_mantissa(left_leaf))
                )
                right_array = np.ascontiguousarray(
                    np.asarray(u.get_mantissa(right_leaf))
                )
                if (
                    left_array.shape != right_array.shape
                    or left_array.dtype != right_array.dtype
                    or left_array.tobytes() != right_array.tobytes()
                ):
                    return False
        return True

    full_snapshot = reference_models["full"].snapshot_state()
    if any(
        not snapshots_equal(full_snapshot, reference_models[arm].snapshot_state())
        for arm in ("query_only", "terminal_only")
    ):
        raise ValueError("Gate C oracle hidden-state snapshots differ")

    raw_gradients = {
        "full": chunked_online_param_gradients(
            model_factory("full", "finite_window"),
            packed_inputs(full_weights),
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        ),
        "query_only": chunked_online_param_gradients(
            model_factory("query_only", "finite_window"),
            packed_inputs(full_weights),
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        ),
        "terminal_only": chunked_online_param_gradients(
            model_factory("terminal_only", "finite_window"),
            packed_inputs(terminal_weights),
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        ),
    }

    gradients: dict[str, dict[str, Any]] = {}
    for arm, raw in raw_gradients.items():
        if not isinstance(raw, Mapping):
            raise TypeError("Gate C oracle gradients must be a path mapping")
        normalized = {
            key if isinstance(key, str) else gate_a._path(key): value
            for key, value in raw.items()
        }
        if (
            len(normalized) != len(raw)
            or tuple(sorted(normalized)) != FULL_PARAMETER_PATHS
        ):
            raise ValueError("Gate C oracle gradient paths differ")
        for path in FULL_PARAMETER_PATHS:
            gradient = normalized[path]
            parameter = reference_parameters[arm][path]
            if jax.tree.structure(gradient) != jax.tree.structure(parameter):
                raise ValueError("Gate C oracle gradient tree differs from parameter")
            gradient_leaves = _numeric_gradient_leaves(gradient)
            parameter_leaves = _numeric_gradient_leaves(parameter)
            if any(
                gradient_leaf.shape != parameter_leaf.shape
                or gradient_leaf.dtype != parameter_leaf.dtype
                for gradient_leaf, parameter_leaf in zip(
                    gradient_leaves, parameter_leaves, strict=True
                )
            ):
                raise ValueError("Gate C oracle gradient geometry differs")
        gradients[arm] = normalized

    def numeric_record(full: Any, arm: Any, *, path: str | None) -> dict[str, Any]:
        record = _gradient_comparison(full, arm)
        if path is None:
            record["full_sha256"] = _gradient_global_sha256(full)
            record["arm_sha256"] = _gradient_global_sha256(arm)
        else:
            record["full_sha256"] = _gradient_path_sha256(path, full)
            record["arm_sha256"] = _gradient_path_sha256(path, arm)
        return record

    def threshold_passed(record: Mapping[str, Any]) -> bool:
        return bool(
            record["full_norm"] > 0.0
            and record["relative_deviation_defined"] is True
            and record["relative_deviation"] >= 1e-3
            and record["l2_difference"]
            > max(1e-8, 1e-4 * record["full_norm"])
        )

    comparisons: dict[str, Any] = {}
    for arm in ("query_only", "terminal_only"):
        global_record = numeric_record(
            gradients["full"], gradients[arm], path=None
        )
        path_records = {
            path: numeric_record(
                gradients["full"][path],
                gradients[arm][path],
                path=path,
            )
            for path in FULL_PARAMETER_PATHS
        }
        required_paths = (
            [
                "memory_read_projection/weight",
                "workspace_query_projection/weight",
            ]
            if arm == "query_only"
            else []
        )
        required_paths_passed = all(
            threshold_passed(path_records[path]) for path in required_paths
        )
        global_passed = bool(
            global_record["arm_norm"] > 0.0
            and threshold_passed(global_record)
        )
        comparisons[arm] = {
            "global": global_record,
            "paths": path_records,
            "required_paths": required_paths,
            "required_paths_passed": bool(required_paths_passed),
            "passed": bool(global_passed and required_paths_passed),
        }
    for role, model in oracle_models.items():
        _gate_c2_finish_control_model(role, model)
    return {
        "contract": contract,
        "objective": {
            "wrapper": "sqrt_weight_times_sqrt_cross_entropy",
            "unsupervised_output_exact_zero": True,
        },
        "gradient_chunk_size": config.gradient_chunk_size,
        "comparisons": comparisons,
        "complete": all(
            comparison["passed"] is True for comparison in comparisons.values()
        ),
    }


def _gate_c3_hidden_state_sha256(model: LatentWorkspaceModel) -> str:
    snapshot = model.snapshot_state()
    values = {
        gate_a._path(path): value for path, value in snapshot.entries
    }
    return _gate_c2_tree_value_sha256(
        values,
        domain="example21-gate-c3-terminal-h8-hidden-state-v1",
    )


def _gate_c3_gradient_record(path: str, full: Any, arm: Any) -> dict[str, Any]:
    def leaf_records(value: Any) -> list[dict[str, Any]]:
        records = []
        for index, array in enumerate(_numeric_gradient_leaves(value)):
            records.append(
                {
                    "index": index,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "value_count": int(array.size),
                    "finite_count": int(np.isfinite(array).sum()),
                    "sha256": legacy._digest_arrays(array),
                }
            )
        return records

    full_leaves = leaf_records(full)
    arm_leaves = leaf_records(arm)
    return {
        **_gradient_comparison(full, arm),
        "full_sha256": _gradient_path_sha256(path, full),
        "arm_sha256": _gradient_path_sha256(path, arm),
        "geometry": {
            "path": path,
            "leaf_order": list(range(len(full_leaves))),
            "leaf_count": len(full_leaves),
            "value_count": sum(item["value_count"] for item in full_leaves),
            "full_leaves": full_leaves,
            "arm_leaves": arm_leaves,
        },
    }


def _gate_c3_expected_gradient_geometry(
    config: GateCConfig,
) -> dict[str, list[tuple[str, list[int]]]]:
    depth = config.gate_b_config
    neurons = depth.neuron_count
    readout = depth.readout_width
    memory = depth.context_memory_width
    color_width = 70 * depth.color_rank
    return {
        "color_factor_head/weight": [
            ("<f4", [color_width]),
            ("<f4", [readout, color_width]),
        ],
        "ff_syn/comm/weight": [("<f4", [depth.row_config.input_width, neurons])],
        "height_head/weight": [("<f4", [30]), ("<f4", [readout, 30])],
        "memory_read_projection/weight": [("<f4", [memory, neurons])],
        "memory_write_scale": [("<f4", [memory, memory])],
        "readout_projection/weight": [
            ("<f4", [readout]),
            ("<f4", [neurons, readout]),
        ],
        "rec_syn/comm/weight": [("<f4", [depth.recurrent_edges])],
        "width_head/weight": [("<f4", [30]), ("<f4", [readout, 30])],
        "workspace_query_projection/weight": [("<f4", [neurons, memory])],
    }


def _gate_c3_gradient_record_complete(
    value: Any,
    *,
    path: str,
    expected_geometry: list[tuple[str, list[int]]],
) -> bool:
    numeric_keys = {
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
    if not isinstance(value, Mapping) or set(value) != numeric_keys | {"geometry"}:
        return False
    numeric = {name: value[name] for name in numeric_keys}
    geometry = value["geometry"]
    if not _gradient_record_complete(numeric) or not isinstance(geometry, Mapping):
        return False
    if set(geometry) != {
        "path",
        "leaf_order",
        "leaf_count",
        "value_count",
        "full_leaves",
        "arm_leaves",
    } or geometry["path"] != path:
        return False
    full_leaves = geometry["full_leaves"]
    arm_leaves = geometry["arm_leaves"]
    if not isinstance(full_leaves, list) or not isinstance(arm_leaves, list):
        return False
    if not (
        full_leaves
        and len(full_leaves) == len(arm_leaves)
        and len(full_leaves) == len(expected_geometry)
        and _strict_integer(geometry["leaf_count"])
        and int(geometry["leaf_count"]) == len(full_leaves)
        and geometry["leaf_order"] == list(range(len(full_leaves)))
    ):
        return False
    total = 0
    leaf_keys = {
        "index",
        "dtype",
        "shape",
        "value_count",
        "finite_count",
        "sha256",
    }
    for index, (full_leaf, arm_leaf) in enumerate(
        zip(full_leaves, arm_leaves, strict=True)
    ):
        expected_dtype, expected_shape = expected_geometry[index]
        if not (
            isinstance(full_leaf, Mapping)
            and isinstance(arm_leaf, Mapping)
            and set(full_leaf) == leaf_keys
            and set(arm_leaf) == leaf_keys
            and _strict_integer(full_leaf["index"])
            and _strict_integer(arm_leaf["index"])
            and int(full_leaf["index"]) == int(arm_leaf["index"]) == index
            and full_leaf["dtype"] == arm_leaf["dtype"]
            and full_leaf["dtype"] == expected_dtype
            and full_leaf["shape"] == arm_leaf["shape"]
            and full_leaf["shape"] == expected_shape
            and _strict_integer(full_leaf["value_count"])
            and _strict_integer(arm_leaf["value_count"])
            and int(full_leaf["value_count"])
            == int(arm_leaf["value_count"])
            == math.prod(full_leaf["shape"])
            and _strict_integer(full_leaf["finite_count"])
            and _strict_integer(arm_leaf["finite_count"])
            and int(full_leaf["finite_count"])
            == int(arm_leaf["finite_count"])
            == int(full_leaf["value_count"])
            and _sha256_complete(full_leaf["sha256"])
            and _sha256_complete(arm_leaf["sha256"])
        ):
            return False
        total += int(full_leaf["value_count"])
    return bool(
        _strict_integer(geometry["value_count"])
        and int(geometry["value_count"]) == total
    )


def _gate_c3_terminal_h8_mechanism_oracle(
    config: GateCConfig,
    *,
    initialization: Mapping[str, Any],
    gate_b_data: Any,
) -> dict[str, Any]:
    """Measure the fixed terminal-H8 mechanism twice with compiled scans."""

    inputs = _gate_c3_terminal_h8_inputs(config, gate_b_data=gate_b_data)
    try:
        canonical = initialization["initialization"]["gate_b"][
            "canonical_full"
        ]
        arm_refs = initialization["initialization"]["gate_b"][
            "arm_initialization_refs"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("Gate C3 oracle initialization evidence is incomplete") from error
    if (
        not isinstance(canonical, Mapping)
        or canonical.get("parameter_paths") != list(FULL_PARAMETER_PATHS)
        or not _strict_integer(canonical.get("parameter_count"))
        or not _sha256_complete(canonical.get("parameter_sha256"))
        or not isinstance(arm_refs, Mapping)
        or any(
            not isinstance(arm_refs.get(arm), Mapping)
            or arm_refs[arm].get("tree") != "canonical_full"
            or arm_refs[arm].get("parameter_sha256")
            != canonical["parameter_sha256"]
            for arm in ("full", "query_only")
        )
    ):
        raise ValueError("Gate C3 oracle initialization identity differs")

    class _TerminalH8Objective(LatentWorkspaceModel):
        def update(self, packed: jax.Array) -> jax.Array:
            expected_width = self.config.input_width + 3
            if packed.ndim != 2 or packed.shape[-1] != expected_width:
                raise ValueError("Gate C3 oracle packed input geometry differs")
            event = packed[:, : self.config.input_width]
            advance = packed[:, self.config.input_width] > 0.5
            target = packed[:, self.config.input_width + 1].astype(jnp.int32)
            weight = packed[:, self.config.input_width + 2]
            loss = legacy._classification_loss(
                super().update(event, advance),
                target,
                self.config.color_rank,
            )
            wrapped = jnp.sqrt(jnp.maximum(loss, 0.0))
            return jnp.where(weight == 0.0, jnp.zeros_like(wrapped), wrapped)

    def algorithm_factory(model: brainstate.nn.Module) -> braintrace.ETraceAlgorithm:
        return braintrace.pp_prop(
            model,
            decay_or_rank=model.config.trace_decay,
            vjp_method="multi-step",
        )

    expected_parameter_sha256 = str(canonical["parameter_sha256"])
    expected_parameter_count = int(canonical["parameter_count"])
    packed = jnp.asarray(inputs["packed_inputs"], dtype=jnp.float32)
    replay_reports: list[dict[str, Any]] = []

    for replay in (1, 2):
        parameter_sha256: dict[str, str] = {}
        hidden_state_sha256: dict[str, str] = {}
        models: dict[str, _TerminalH8Objective] = {}

        def create_model(policy_name: str, stage: str) -> _TerminalH8Objective:
            policy = "full" if policy_name == "full_read_h8" else "query_only"
            role = (
                f"gate_b:mechanism_oracle:terminal_h8:replay_{replay}:"
                f"{policy_name}:{stage}"
            )

            def construct(
                model_config: ModelConfig,
                actual_policy: str,
            ) -> _TerminalH8Objective:
                return _TerminalH8Objective(
                    model_config,
                    memory_read_policy=actual_policy,
                )

            model = _gate_c2_control_model(
                config,
                initialization,
                regime="gate_b",
                policy=policy,
                batch_size=1,
                role=role,
                probe=(
                    f"mechanism_oracle:terminal_h8:replay_{replay}:"
                    f"{policy_name}:{stage}"
                ),
                arm="full" if policy == "full" else "query_only",
                constructor=construct,
            )
            if not isinstance(model, _TerminalH8Objective):
                raise TypeError("Gate C3 oracle factory returned the wrong model")
            values = legacy._parameter_values(model)
            count = sum(
                np.asarray(u.get_mantissa(leaf)).size
                for value in values.values()
                for leaf in jax.tree.leaves(value)
            )
            if (
                tuple(sorted(values)) != FULL_PARAMETER_PATHS
                or count != expected_parameter_count
                or legacy._array_digest(values) != expected_parameter_sha256
            ):
                raise ValueError("Gate C3 oracle did not reproduce initialization")
            if role in models:
                raise RuntimeError(f"duplicate Gate C3 oracle role {role!r}")
            models[role] = model
            return model

        full_reference_role = (
            f"gate_b:mechanism_oracle:terminal_h8:replay_{replay}:"
            "full_read_h8:reference"
        )
        full_reference = create_model("full_read_h8", "reference")
        brainstate.nn.init_all_states(full_reference, batch_size=1)
        common_snapshot = full_reference.snapshot_state()
        parameter_sha256[full_reference_role] = legacy._array_digest(
            legacy._parameter_values(full_reference)
        )
        hidden_state_sha256[full_reference_role] = (
            _gate_c3_hidden_state_sha256(full_reference)
        )

        raw_gradients: dict[str, Mapping[str, Any]] = {}
        for policy_name in ("full_read_h8", "query_only_h8"):
            reference_role = (
                f"gate_b:mechanism_oracle:terminal_h8:replay_{replay}:"
                f"{policy_name}:reference"
            )
            if policy_name == "query_only_h8":
                reference = create_model(policy_name, "reference")
                brainstate.nn.init_all_states(reference, batch_size=1)
                reference.restore_state(common_snapshot)
                parameter_sha256[reference_role] = legacy._array_digest(
                    legacy._parameter_values(reference)
                )
                hidden_state_sha256[reference_role] = (
                    _gate_c3_hidden_state_sha256(reference)
                )
            finite_role = (
                f"gate_b:mechanism_oracle:terminal_h8:replay_{replay}:"
                f"{policy_name}:finite_window"
            )
            finite_models: list[_TerminalH8Objective] = []

            def model_factory(
                selected_policy: str = policy_name,
                selected_role: str = finite_role,
            ) -> _TerminalH8Objective:
                model = create_model(selected_policy, "finite_window")
                brainstate.nn.init_all_states(model, batch_size=1)
                model.restore_state(common_snapshot)
                parameter_sha256[selected_role] = legacy._array_digest(
                    legacy._parameter_values(model)
                )
                hidden_state_sha256[selected_role] = (
                    _gate_c3_hidden_state_sha256(model)
                )
                finite_models.append(model)
                return model

            def restore_common_start(
                model: brainstate.nn.Module,
                _algorithm: braintrace.ETraceAlgorithm,
                selected_role: str = finite_role,
            ) -> None:
                if not isinstance(model, _TerminalH8Objective):
                    raise TypeError("Gate C3 finite-window model type differs")
                model.restore_state(common_snapshot)
                actual_parameter = legacy._array_digest(
                    legacy._parameter_values(model)
                )
                actual_hidden = _gate_c3_hidden_state_sha256(model)
                if (
                    parameter_sha256.get(selected_role) != actual_parameter
                    or hidden_state_sha256.get(selected_role) != actual_hidden
                ):
                    raise ValueError("Gate C3 gradient start identity differs")

            raw = chunked_online_param_gradients(
                model_factory,
                packed,
                algo_factory=algorithm_factory,
                chunk_size=1,
                compiled_scan=True,
                after_init=restore_common_start,
            )
            if not finite_models:
                raise RuntimeError("Gate C3 oracle did not materialize its model")
            if not isinstance(raw, Mapping):
                raise TypeError("Gate C3 oracle gradients must be a path mapping")
            normalized = {
                key if isinstance(key, str) else gate_a._path(key): value
                for key, value in raw.items()
            }
            if len(normalized) != len(raw) or tuple(sorted(normalized)) != (
                FULL_PARAMETER_PATHS
            ):
                raise ValueError("Gate C3 oracle gradient paths differ")
            reference_values = legacy._parameter_values(
                full_reference
                if policy_name == "full_read_h8"
                else models[reference_role]
            )
            for path in FULL_PARAMETER_PATHS:
                if jax.tree.structure(normalized[path]) != jax.tree.structure(
                    reference_values[path]
                ):
                    raise ValueError("Gate C3 oracle gradient tree differs")
                gradient_leaves = _numeric_gradient_leaves(normalized[path])
                parameter_leaves = _numeric_gradient_leaves(reference_values[path])
                if any(
                    gradient.shape != parameter.shape
                    or gradient.dtype != parameter.dtype
                    for gradient, parameter in zip(
                        gradient_leaves,
                        parameter_leaves,
                        strict=True,
                    )
                ):
                    raise ValueError("Gate C3 oracle gradient geometry differs")
            raw_gradients[policy_name] = normalized

        full_gradients = raw_gradients["full_read_h8"]
        query_gradients = raw_gradients["query_only_h8"]
        paths = {
            path: _gate_c3_gradient_record(
                path,
                full_gradients[path],
                query_gradients[path],
            )
            for path in FULL_PARAMETER_PATHS
        }
        global_record = {
            **_gradient_comparison(full_gradients, query_gradients),
            "full_sha256": _gradient_global_sha256(full_gradients),
            "arm_sha256": _gradient_global_sha256(query_gradients),
        }
        path_order = list(FULL_PARAMETER_PATHS)
        leaf_order = [
            f"{path}#{index}"
            for path in path_order
            for index in paths[path]["geometry"]["leaf_order"]
        ]
        value_count = sum(
            int(paths[path]["geometry"]["value_count"])
            for path in path_order
        )
        global_record["geometry"] = {
            "path_order": path_order,
            "leaf_order": leaf_order,
            "leaf_count": len(leaf_order),
            "value_count": value_count,
            "full_finite_count": value_count,
            "arm_finite_count": value_count,
        }
        required_paths = list(
            GATE_C3_TERMINAL_H8_OBJECTIVE["required_paths"]
        )
        required_passed = all(
            _gate_c3_terminal_h8_threshold_passed(paths[path])
            for path in required_paths
        )
        comparison_passed = bool(
            _gate_c3_terminal_h8_threshold_passed(global_record)
            and required_passed
        )
        expected_roles = [
            role
            for role in GATE_C3_CONTROLS_MODEL_ROLES
            if f"replay_{replay}:" in role
        ]
        identity = {
            "parameter_sha256": {
                role: parameter_sha256[role] for role in expected_roles
            },
            "hidden_state_sha256": {
                role: hidden_state_sha256[role] for role in expected_roles
            },
            "parameters_byte_identical": bool(
                len(set(parameter_sha256.values())) == 1
                and next(iter(parameter_sha256.values()))
                == expected_parameter_sha256
            ),
            "hidden_states_byte_identical": bool(
                len(set(hidden_state_sha256.values())) == 1
            ),
        }
        replay_reports.append(
            {
                "replay": replay,
                "pre_execution_identity": identity,
                "comparison": {
                    "global": global_record,
                    "paths": paths,
                    "required_paths": required_paths,
                    "required_paths_passed": bool(required_passed),
                    "passed": comparison_passed,
                },
            }
        )
        for role in expected_roles:
            _gate_c2_finish_control_model(role, models[role])

    raw_records = {
        name: _gate_c2_raw_array_record(
            inputs[name],
            include_values=name in {"targets", "advances", "loss_weights"},
        )
        for name in ("events", "targets", "advances", "loss_weights", "packed_inputs")
    }
    raw_records["consumed_loss_weights"] = _gate_c2_raw_array_record(
        np.asarray(inputs["packed_inputs"])[:, 0, -1],
        include_values=True,
    )
    return {
        "contract": dict(GATE_C3_TERMINAL_H8_OBJECTIVE),
        "objective": {
            "wrapper": "terminal_h8_sqrt_cross_entropy",
            "unsupervised_output_exact_zero": True,
            "gradient_chunk_size": 1,
            "compiled_scan": True,
            "arrays": raw_records,
        },
        "replays": replay_reports,
        "complete": all(
            replay["comparison"]["passed"] is True
            for replay in replay_reports
        ),
    }


def _gate_c3_terminal_h8_threshold_passed(value: Mapping[str, Any]) -> bool:
    try:
        full_norm = _finite_real(value["full_norm"], "full gradient norm")
        relative = _finite_real(
            value["relative_deviation"],
            "relative gradient deviation",
        )
        difference = _finite_real(
            value["l2_difference"],
            "gradient difference",
        )
        return bool(
            full_norm > 0.0
            and value["relative_deviation_defined"] is True
            and relative
            >= GATE_C3_TERMINAL_H8_OBJECTIVE[
                "relative_deviation_minimum"
            ]
            and difference
            > max(
                GATE_C3_TERMINAL_H8_OBJECTIVE[
                    "l2_difference_absolute_floor"
                ],
                GATE_C3_TERMINAL_H8_OBJECTIVE[
                    "l2_difference_relative_floor"
                ]
                * full_norm,
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _gate_c3_terminal_h8_mechanism_oracle_status(
    value: Any,
    config: GateCConfig,
    *,
    canonical_parameter_sha256: str | None = None,
    audit_evidence: Mapping[str, Any] | None = None,
) -> tuple[bool, bool]:
    if (
        not isinstance(config, GateCConfig)
        or not isinstance(value, Mapping)
        or set(value) != {"contract", "objective", "replays", "complete"}
        or not gate_a._json_exact(
            value["contract"],
            GATE_C3_TERMINAL_H8_OBJECTIVE,
        )
    ):
        return False, False
    objective = value["objective"]
    if not isinstance(objective, Mapping) or set(objective) != {
        "wrapper",
        "unsupervised_output_exact_zero",
        "gradient_chunk_size",
        "compiled_scan",
        "arrays",
    }:
        return False, False
    if not (
        objective["wrapper"] == "terminal_h8_sqrt_cross_entropy"
        and objective["unsupervised_output_exact_zero"] is True
        and _strict_integer(objective["gradient_chunk_size"])
        and int(objective["gradient_chunk_size"]) == 1
        and objective["compiled_scan"] is True
    ):
        return False, False
    arrays = objective["arrays"]
    if not isinstance(arrays, Mapping) or set(arrays) != {
        "events",
        "targets",
        "advances",
        "loss_weights",
        "packed_inputs",
        "consumed_loss_weights",
    }:
        return False, False
    targets = np.asarray(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 8, 1, 7, 9, 3, 2, 5],
        dtype=np.int32,
    )
    advances = np.ones((19,), dtype=np.float32)
    weights = np.asarray([0.0] * 18 + [1.0], dtype=np.float32)
    fixed_records = {
        "events": {
            "dtype": "<f4",
            "shape": [19, 47],
            "sha256": GATE_C3_TERMINAL_H8_OBJECTIVE["events_sha256"],
        },
        "packed_inputs": {
            "dtype": "<f4",
            "shape": [19, 1, 50],
            "sha256": (
                "ef1c75296133458d90de3d5d9c204890127f83238148bd11bc2736bae6a205e1"
            ),
        },
    }
    if not (
        gate_a._json_exact(arrays["events"], fixed_records["events"])
        and gate_a._json_exact(
            arrays["packed_inputs"],
            fixed_records["packed_inputs"],
        )
        and _gate_c2_raw_array_record_complete(
            arrays["targets"],
            targets,
            include_values=True,
        )
        and _gate_c2_raw_array_record_complete(
            arrays["advances"],
            advances,
            include_values=True,
        )
        and _gate_c2_raw_array_record_complete(
            arrays["loss_weights"],
            weights,
            include_values=True,
        )
        and _gate_c2_raw_array_record_complete(
            arrays["consumed_loss_weights"],
            weights,
            include_values=True,
        )
    ):
        return False, False

    replays = value["replays"]
    if not isinstance(replays, list) or len(replays) != 2:
        return False, False
    replay_passes: list[bool] = []
    all_hidden_digests: set[str] = set()
    for replay_index, replay in enumerate(replays, start=1):
        if not isinstance(replay, Mapping) or set(replay) != {
            "replay",
            "pre_execution_identity",
            "comparison",
        }:
            return False, False
        if not _strict_integer(replay["replay"]) or int(replay["replay"]) != (
            replay_index
        ):
            return False, False
        expected_roles = [
            role
            for role in GATE_C3_CONTROLS_MODEL_ROLES
            if f"replay_{replay_index}:" in role
        ]
        identity = replay["pre_execution_identity"]
        if not isinstance(identity, Mapping) or set(identity) != {
            "parameter_sha256",
            "hidden_state_sha256",
            "parameters_byte_identical",
            "hidden_states_byte_identical",
        }:
            return False, False
        parameter_sha = identity["parameter_sha256"]
        hidden_sha = identity["hidden_state_sha256"]
        if not (
            isinstance(parameter_sha, Mapping)
            and isinstance(hidden_sha, Mapping)
            and set(parameter_sha) == set(expected_roles)
            and set(hidden_sha) == set(expected_roles)
            and all(_sha256_complete(item) for item in parameter_sha.values())
            and all(_sha256_complete(item) for item in hidden_sha.values())
            and len(set(parameter_sha.values())) == 1
            and len(set(hidden_sha.values())) == 1
            and (
                canonical_parameter_sha256 is None
                or all(
                    item == canonical_parameter_sha256
                    for item in parameter_sha.values()
                )
            )
            and identity["parameters_byte_identical"] is True
            and identity["hidden_states_byte_identical"] is True
        ):
            return False, False
        all_hidden_digests.update(str(item) for item in hidden_sha.values())

        comparison = replay["comparison"]
        if not isinstance(comparison, Mapping) or set(comparison) != {
            "global",
            "paths",
            "required_paths",
            "required_paths_passed",
            "passed",
        }:
            return False, False
        paths = comparison["paths"]
        expected_geometry = _gate_c3_expected_gradient_geometry(config)
        global_record = comparison["global"]
        if (
            not isinstance(paths, Mapping)
            or tuple(sorted(paths)) != FULL_PARAMETER_PATHS
            or not all(
                _gate_c3_gradient_record_complete(
                    paths[path],
                    path=path,
                    expected_geometry=expected_geometry[path],
                )
                for path in FULL_PARAMETER_PATHS
            )
            or not isinstance(global_record, Mapping)
            or set(global_record) != _GRADIENT_RECORD_KEYS | {"geometry"}
            or not _gradient_record_complete(
                {name: global_record[name] for name in _GRADIENT_RECORD_KEYS}
            )
        ):
            return False, False
        path_order = list(FULL_PARAMETER_PATHS)
        leaf_order = [
            f"{path}#{index}"
            for path in path_order
            for index in paths[path]["geometry"]["leaf_order"]
        ]
        value_count = sum(
            int(paths[path]["geometry"]["value_count"])
            for path in path_order
        )
        if not gate_a._json_exact(
            global_record["geometry"],
            {
                "path_order": path_order,
                "leaf_order": leaf_order,
                "leaf_count": len(leaf_order),
                "value_count": value_count,
                "full_finite_count": value_count,
                "arm_finite_count": value_count,
            },
        ):
            return False, False
        if not _gradient_global_algebra_complete(
            global_record,
            paths,
            allow_null_cosine=True,
        ):
            return False, False
        required_paths = list(
            GATE_C3_TERMINAL_H8_OBJECTIVE["required_paths"]
        )
        if comparison["required_paths"] != required_paths:
            return False, False
        required_passed = all(
            _gate_c3_terminal_h8_threshold_passed(paths[path])
            for path in required_paths
        )
        passed = bool(
            _gate_c3_terminal_h8_threshold_passed(global_record)
            and required_passed
        )
        if not (
            comparison["required_paths_passed"] is required_passed
            and comparison["passed"] is passed
        ):
            return False, False
        replay_passes.append(passed)
    complete = all(replay_passes)
    if value["complete"] is not complete or len(all_hidden_digests) != 1:
        return False, False
    if audit_evidence is not None:
        if not _sha256_complete(canonical_parameter_sha256):
            return False, False
        materialized = audit_evidence.get("materialized_roles")
        expected_roles = list(_GATE_C3_TERMINAL_MODEL_ROLES)
        if not isinstance(materialized, Mapping) or not all(
            role in materialized for role in expected_roles
        ):
            return False, False
        for role in expected_roles:
            record = materialized[role]
            if not isinstance(record, Mapping) or not (
                record.get("expected_parameter_sha256")
                == canonical_parameter_sha256
                and record.get("before_parameter_sha256")
                == canonical_parameter_sha256
                and record.get("after_parameter_sha256")
                == canonical_parameter_sha256
                and record.get("parameters_equal") is True
            ):
                return False, False
    return True, complete


def _gate_c3_terminal_h8_mechanism_oracle_complete(
    value: Any,
    config: GateCConfig,
    *,
    canonical_parameter_sha256: str | None = None,
    audit_evidence: Mapping[str, Any] | None = None,
) -> bool:
    """Recompute the strict terminal-H8 two-replay mechanism decision."""

    try:
        valid, complete = _gate_c3_terminal_h8_mechanism_oracle_status(
            value,
            config,
            canonical_parameter_sha256=canonical_parameter_sha256,
            audit_evidence=audit_evidence,
        )
        return bool(valid and complete)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _run_gate_c_controls_engine(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
    c3: bool,
) -> dict[str, Any]:
    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    gate_number = 3 if c3 else 2
    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }:
        raise ValueError(f"Gate C{gate_number} controls require exact authenticated prerequisites")
    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    authentication_environment = (
        _gate_c3_controls_base_environment(environment) if c3 else environment
    )
    start = time.perf_counter()
    initialization = _validated_gate_c_initialization_admission(
        prerequisites["gate_c_initialization"],
        config,
        source_start=source_start,
        environment=authentication_environment,
        source_files=source_files,
        require_pass=True,
    )
    normalized = _normalized_prerequisites(
        {
            "gate_a": prerequisites["gate_a"],
            "gate_b": prerequisites["gate_b"],
        }
    )
    gate_a_data = _regenerate_gate_a_data(config)
    gate_b_data = _regenerate_gate_b_data(config)
    schedules = _actual_schedule_identity_report(
        config,
        gate_a_data,
        gate_b_data,
    )
    if (
        config == GateCConfig()
        and not gate_a._json_exact(schedules, _schedule_identity_report(config))
    ):
        raise RuntimeError(f"generated Gate C{gate_number} schedules differ from preregistration")
    audit = (
        _GateC2ControlsAudit(
            initialization,
            model_roles=GATE_C3_CONTROLS_MODEL_ROLES,
        )
        if c3
        else _GateC2ControlsAudit(initialization)
    )
    global _ACTIVE_GATE_C2_CONTROLS_AUDIT
    previous_audit = _ACTIVE_GATE_C2_CONTROLS_AUDIT
    if previous_audit is not None:
        raise RuntimeError(f"a {'Gate C' if c3 else 'Gate C2'} controls audit is already active")
    try:
        _ACTIVE_GATE_C2_CONTROLS_AUDIT = audit
        with audit:
            regime_reports: dict[str, Any] = {}
            for regime in REGIME_ORDER:
                regime_data = gate_a_data if regime == "gate_a" else gate_b_data
                regime_config = (
                    config.gate_a_config
                    if regime == "gate_a"
                    else config.gate_b_config
                )
                regime_reports[regime] = {
                    "spec": dataclasses.asdict(REGIME_SPECS[regime]),
                    "config": dataclasses.asdict(regime_config),
                    "schedule_identity": dict(schedules[regime]),
                    "paired_h0_operational_equivalence": (
                        _paired_h0_operational_equivalence_report(
                            config,
                            initialization=initialization,
                            regime=regime,
                            data=regime_data,
                        )
                    ),
                    "query_only_latent_no_read": (
                        _query_only_latent_no_read_report(
                            config,
                            initialization=initialization,
                            regime=regime,
                            data=regime_data,
                        )
                    ),
                }
                gc.collect()
            mechanism_oracle = (
                _gate_c3_terminal_h8_mechanism_oracle
                if c3
                else _mechanism_oracle
            )(
                config,
                initialization=initialization,
                gate_b_data=gate_b_data,
            )
    finally:
        _ACTIVE_GATE_C2_CONTROLS_AUDIT = previous_audit
    execution_evidence = None if c3 else audit.report()
    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    if c3:
        execution_evidence = audit.report()
    report: dict[str, Any] = {
        "schema_version": (
            GATE_C3_CONTROLS_SCHEMA_VERSION
            if c3
            else GATE_C2_CONTROLS_SCHEMA_VERSION
        ),
        "control": GATE_C3_CONTROLS_CONTROL if c3 else GATE_C2_CONTROLS_CONTROL,
        "qualification_regime": (
            GATE_C3_CONTROLS_QUALIFICATION_REGIME
            if c3
            else GATE_C2_CONTROLS_QUALIFICATION_REGIME
        ),
        "learner": "pp_prop_only",
        "prerequisites": {
            **normalized,
            "gate_c_initialization": dict(
                prerequisites["gate_c_initialization"]
            ),
        },
        "regimes": regime_reports,
        "mechanism_oracle": mechanism_oracle,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": {
            **dict(environment),
            "execution_and_update_evidence": execution_evidence,
        },
        "total_wall_seconds": time.perf_counter() - start,
    }
    qualification = (
        _gate_c3_controls_qualification_report
        if c3
        else _gate_c2_controls_qualification_report
    )
    report["qualification"] = qualification(report, config=config)
    return report


def run_gate_c2_controls(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the authenticated Gate C2 probes without training or Adam."""

    return _run_gate_c_controls_engine(
        config,
        prerequisites=prerequisites,
        source_start=source_start,
        source_end_reporter=source_end_reporter,
        source_files=source_files,
        environment=environment,
        c3=False,
    )


def run_gate_c3_controls(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the deterministic Gate C3 controls without training or Adam.

    Parameters
    ----------
    config
        Fixed paired Gate C configuration.
    prerequisites
        Authenticated Gate A, Gate B, and Gate C initialization evidence.
    source_start, source_end_reporter, source_files, environment
        Live source, immutable-file, GPU, and deterministic-environment evidence.

    Returns
    -------
    dict
        Strict controls artifact with a recomputed tri-state qualification.
    """

    return _run_gate_c_controls_engine(
        config,
        prerequisites=prerequisites,
        source_start=source_start,
        source_end_reporter=source_end_reporter,
        source_files=source_files,
        environment=environment,
        c3=True,
    )


def run_gate_c(
    config: GateCConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Any,
    source_files: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all ten isolated formal Gate C pp-prop trainings.

    Parameters
    ----------
    config
        Fixed paired Gate C configuration.
    prerequisites
        Exact Gate A, Gate B, and authenticated Gate C initialization evidence.
    source_start, source_end_reporter, source_files, environment
        Live provenance and exact GPU evidence.

    Returns
    -------
    dict
        Strict formal Gate C artifact payload.
    """

    if not isinstance(config, GateCConfig):
        raise TypeError("config must be a GateCConfig")
    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }:
        raise ValueError("formal Gate C requires exact authenticated prerequisites")
    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    start = time.perf_counter()
    initialization = _validated_gate_c_initialization_admission(
        prerequisites["gate_c_initialization"],
        config,
        source_start=source_start,
        environment=environment,
        source_files=source_files,
        require_pass=True,
    )
    normalized_prerequisites = _normalized_prerequisites(
        {"gate_a": prerequisites["gate_a"], "gate_b": prerequisites["gate_b"]}
    )
    gate_a_data = _regenerate_gate_a_data(config)
    gate_b_data = _regenerate_gate_b_data(config)
    schedule_reports = _actual_schedule_identity_report(
        config,
        gate_a_data,
        gate_b_data,
    )
    if (
        config.qualification_regime == "preregistered_full"
        and not gate_a._json_exact(schedule_reports, _schedule_identity_report(config))
    ):
        raise RuntimeError("generated Gate C schedules differ from preregistration")

    initialization_reports: dict[str, dict[str, dict[str, Any]]] = {
        regime: {} for regime in REGIME_ORDER
    }
    for regime in REGIME_ORDER:
        for arm in ARM_ORDER:
            model, trainer, initialization_report = _fresh_formal_arm(
                config,
                initialization,
                regime,
                arm,
            )
            initialization_reports[regime][arm] = initialization_report
            del model, trainer
            gc.collect()

    paired_h0_reports = {
        "gate_a": _paired_h0_identity_report(
            config,
            initialization=initialization,
            regime="gate_a",
            data=gate_a_data,
        ),
        "gate_b": _paired_h0_identity_report(
            config,
            initialization=initialization,
            regime="gate_b",
            data=gate_b_data,
        ),
    }
    gc.collect()

    arms: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in ARM_ORDER
    }
    for regime in REGIME_ORDER:
        data = gate_a_data if regime == "gate_a" else gate_b_data
        for arm in ARM_ORDER:
            execution_index = (
                REGIME_ORDER.index(regime) * len(ARM_ORDER)
                + ARM_ORDER.index(arm)
            )
            model, trainer, reproduced = _fresh_formal_arm(
                config,
                initialization,
                regime,
                arm,
            )
            audit = initialization_reports[regime][arm]
            if not gate_a._json_exact(reproduced, audit):
                raise RuntimeError(
                    "formal arm initialization changed between audit and training"
                )
            arms[arm][regime] = _run_gate_c_arm(
                model,
                trainer,
                data,
                config,
                regime,
                arm,
                initialization_report=audit,
                execution_index=execution_index,
                data_identity=schedule_reports[regime],
            )
            del model, trainer
            gc.collect()
    metrics: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        gate_a_metrics = arms[arm]["gate_a"].get("metrics")
        gate_b_metrics = arms[arm]["gate_b"].get("metrics")
        if (
            isinstance(gate_a_metrics, Mapping)
            and set(gate_a_metrics) == {"binding_gap", "depth_accuracy"}
            and gate_a._json_exact(gate_a_metrics, gate_b_metrics)
        ):
            metrics[arm] = {
                name: _finite_real(gate_a_metrics[name], f"{arm} {name}")
                for name in ("binding_gap", "depth_accuracy")
            }
        else:
            metrics[arm] = _metric_summary(
                arms[arm]["gate_a"]["evaluation"],
                arms[arm]["gate_b"]["evaluation"],
            )
    for arm in ARM_ORDER:
        for regime in REGIME_ORDER:
            arms[arm][regime]["metrics"] = dict(metrics[arm])
    margins = _blocking_margin_report(metrics)
    mechanism_oracle = _mechanism_oracle(
        config,
        initialization=initialization,
        gate_b_data=gate_b_data,
    )
    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    regimes = {
        regime: {
            "spec": dataclasses.asdict(REGIME_SPECS[regime]),
            "config": dataclasses.asdict(
                config.gate_a_config
                if regime == "gate_a"
                else config.gate_b_config
            ),
            "schedule": dict(schedule_reports[regime]),
            "metrics": {
                arm: dict(metrics[arm]) for arm in ARM_ORDER
            },
            "margins": dict(margins),
            "paired_h0_identity": paired_h0_reports[regime],
        }
        for regime in REGIME_ORDER
    }
    report: dict[str, Any] = {
        "schema_version": GATE_C_SCHEMA_VERSION,
        "control": GATE_C_CONTROL,
        "qualification_regime": config.qualification_regime,
        "learner": "pp_prop_only",
        "prerequisites": {
            **normalized_prerequisites,
            "gate_c_initialization": dict(prerequisites["gate_c_initialization"]),
        },
        "regimes": regimes,
        "arms": arms,
        "mechanism_oracle": mechanism_oracle,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": dict(environment),
        "total_wall_seconds": time.perf_counter() - start,
    }
    report["qualification"] = _qualification_report(report, config=config)
    return report


_ACCURACY_KEYS = {
    "correct",
    "count",
    "accuracy",
    "wilson_95_lower",
    "wilson_95_upper",
    "prediction_histogram",
    "prediction_sha256",
    "checkpoint",
}
_TELEMETRY_CATEGORIES = {
    "logits",
    "model_states",
    "gradients",
    "pp_prop_traces",
    "adam",
    "parameters",
}


def _accuracy_record_complete(
    metric: Any,
    *,
    count: int,
    checkpoint: int,
) -> bool:
    if not isinstance(metric, Mapping) or set(metric) != _ACCURACY_KEYS:
        return False
    return bool(
        _strict_integer(metric["checkpoint"])
        and int(metric["checkpoint"]) == checkpoint
        and _sha256_complete(metric["prediction_sha256"])
        and gate_a._accuracy_evidence_complete(metric, count)
    )


def _paired_diagnostic_record_complete(value: Any, *, count: int) -> bool:
    expected = {
        "applicable_count",
        "different_count",
        "every_pair_differs",
        "mean_l2_difference",
        "left_sha256",
        "right_sha256",
        "no_context_l2_norm",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    applicable = value["applicable_count"]
    different = value["different_count"]
    if (
        not _strict_integer(applicable)
        or int(applicable) != count
        or not _strict_integer(different)
        or not 0 <= int(different) <= count
    ):
        return False
    mean_difference = _finite_real(
        value["mean_l2_difference"], "mean diagnostic difference"
    )
    no_context_norm = _finite_real(
        value["no_context_l2_norm"], "no-context diagnostic norm"
    )
    return bool(
        value["every_pair_differs"] is (int(different) == count)
        and mean_difference >= 0.0
        and no_context_norm >= 0.0
        and _sha256_complete(value["left_sha256"])
        and _sha256_complete(value["right_sha256"])
        and (
            (
                int(different) == 0
                and mean_difference == 0.0
                and value["left_sha256"] == value["right_sha256"]
            )
            or (
                int(different) > 0
                and mean_difference > 0.0
                and value["left_sha256"] != value["right_sha256"]
            )
        )
    )


def _binding_diagnostic_complete(value: Any, *, count: int, depths: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "all_state_tensors_finite",
        "memory",
        "read_by_depth",
        "workspace_by_depth",
    }:
        return False
    memory = value["memory"]
    memory_keys = {
        "applicable_count",
        "different_count",
        "every_pair_differs",
        "mean_l2_difference",
        "left_sha256",
        "right_sha256",
        "intact_shuffled_different_count",
        "every_intact_shuffled_pair_differs",
        "no_context_exact_zero",
        "no_context_sha256",
        "intact_l2_norm",
        "shuffled_l2_norm",
        "no_context_l2_norm",
        "storage_contract",
    }
    if not isinstance(memory, Mapping) or set(memory) != memory_keys:
        return False
    memory_pair = {
        key: memory[key]
        for key in (
            "applicable_count",
            "different_count",
            "every_pair_differs",
            "mean_l2_difference",
            "left_sha256",
            "right_sha256",
            "no_context_l2_norm",
        )
    }
    if not _paired_diagnostic_record_complete(memory_pair, count=count):
        return False
    expected_depths = {str(index) for index in range(depths)}
    read_by_depth = value["read_by_depth"]
    workspace_by_depth = value["workspace_by_depth"]
    if (
        value["all_state_tensors_finite"] is not True
        or not isinstance(read_by_depth, Mapping)
        or not isinstance(workspace_by_depth, Mapping)
        or set(read_by_depth) != expected_depths
        or set(workspace_by_depth) != expected_depths
        or not all(
            _paired_diagnostic_record_complete(records[depth], count=count)
            for records in (read_by_depth, workspace_by_depth)
            for depth in expected_depths
        )
    ):
        return False
    different = int(memory["different_count"])
    return bool(
        _strict_integer(memory["intact_shuffled_different_count"])
        and int(memory["intact_shuffled_different_count"]) == different
        and memory["every_intact_shuffled_pair_differs"]
        is memory["every_pair_differs"]
        and memory["no_context_exact_zero"] is True
        and _finite_real(
            memory["no_context_l2_norm"], "no-context memory norm"
        )
        == 0.0
        and _sha256_complete(memory["no_context_sha256"])
        and all(
            _finite_real(memory[name], name) >= 0.0
            for name in (
                "intact_l2_norm",
                "shuffled_l2_norm",
                "no_context_l2_norm",
            )
        )
        and memory["storage_contract"]
        == "one final S_K snapshot per arm; S_K is not stacked"
    )


def _gate_a_evaluation_complete(
    evaluation: Any,
    config: GateCConfig,
) -> bool:
    expected_keys = {
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
    if not isinstance(evaluation, Mapping) or set(evaluation) != expected_keys:
        return False
    count = config.gate_a_config.validation_episodes
    depth_count = config.gate_a_config.gap_steps + 1
    depths = evaluation["depths"]
    if (
        evaluation["finite"] is not True
        or evaluation["all_compact_logits_finite"] is not True
        or evaluation["all_state_tensors_finite"] is not True
        or not isinstance(depths, Mapping)
        or set(depths) != {str(index) for index in range(depth_count)}
    ):
        return False
    for depth in range(depth_count):
        streams = depths[str(depth)]
        if not isinstance(streams, Mapping) or set(streams) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        if not all(
            _accuracy_record_complete(
                streams[name],
                count=count,
                checkpoint=depth,
            )
            for name in ("intact", "shuffled", "no_context")
        ):
            return False
    final = depths[str(config.gate_a_config.gap_steps)]
    if not all(
        gate_a._json_exact(evaluation[name], final[name])
        for name in ("intact", "shuffled", "no_context")
    ):
        return False
    gap = _finite_real(evaluation["intact_minus_shuffled"], "Gate A gap")
    if not math.isclose(
        gap,
        float(final["intact"]["accuracy"])
        - float(final["shuffled"]["accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    diagnostic = evaluation["binding_diagnostic"]
    if not _binding_diagnostic_complete(
        diagnostic,
        count=count,
        depths=depth_count,
    ):
        return False
    memory = diagnostic["memory"]
    binding_state = evaluation["binding_state"]
    expected_binding_state = {
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
    return gate_a._json_exact(binding_state, expected_binding_state)


def _gate_a_intervention_diagnostic_complete(
    evaluation: Mapping[str, Any],
    *,
    arm: str,
) -> bool:
    diagnostic = evaluation["binding_diagnostic"]

    def exact_zero_pair(value: Mapping[str, Any]) -> bool:
        return bool(
            _strict_integer(value["different_count"])
            and int(value["different_count"]) == 0
            and value["every_pair_differs"] is False
            and _finite_real(value["mean_l2_difference"], "diagnostic difference")
            == 0.0
            and value["left_sha256"] == value["right_sha256"]
            and _finite_real(
                value["no_context_l2_norm"], "no-context diagnostic norm"
            )
            == 0.0
        )

    if arm == "query_only":
        return exact_zero_pair(diagnostic["read_by_depth"]["1"])
    if arm != "legacy":
        return True
    memory = diagnostic["memory"]
    if not (
        exact_zero_pair(memory)
        and _finite_real(memory["intact_l2_norm"], "legacy intact memory norm")
        == 0.0
        and _finite_real(memory["shuffled_l2_norm"], "legacy shuffled memory norm")
        == 0.0
    ):
        return False
    return all(
        exact_zero_pair(value)
        for section in ("read_by_depth", "workspace_by_depth")
        for value in diagnostic[section].values()
    )


def _gate_b_evaluation_complete(
    evaluation: Any,
    config: GateCConfig,
    *,
    require_no_collapse: bool,
) -> bool:
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "finite",
        "h0_proper",
        "depths",
        "efforts",
    }:
        return False
    count = config.gate_b_config.validation_episodes
    depths = evaluation["depths"]
    expected_depths = {
        str(index) for index in range(config.gate_b_config.gap_steps + 1)
    }
    if (
        evaluation["finite"] is not True
        or not isinstance(depths, Mapping)
        or set(depths) != expected_depths
    ):
        return False
    for depth in range(config.gate_b_config.gap_steps + 1):
        streams = depths[str(depth)]
        if not isinstance(streams, Mapping) or set(streams) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        for name in ("intact", "shuffled", "no_context"):
            metric = streams[name]
            if not _accuracy_record_complete(
                metric,
                count=count,
                checkpoint=depth,
            ):
                return False
            if require_no_collapse and max(map(int, metric["prediction_histogram"])) >= count:
                return False
    if not gate_a._json_exact(evaluation["h0_proper"], depths["0"]["intact"]):
        return False
    efforts = evaluation["efforts"]
    if not isinstance(efforts, Mapping) or set(efforts) != {
        str(effort) for effort in gate_b.QUALIFYING_EFFORTS
    }:
        return False
    h0 = evaluation["h0_proper"]
    for effort in gate_b.QUALIFYING_EFFORTS:
        evidence = efforts[str(effort)]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "intact",
            "shuffled",
            "no_context",
            "h0_final_target",
            "intact_minus_h0",
            "intact_minus_shuffled",
        }:
            return False
        matching = depths[str(effort)]
        if not all(
            gate_a._json_exact(evidence[name], matching[name])
            for name in ("intact", "shuffled", "no_context")
        ):
            return False
        h0_final = evidence["h0_final_target"]
        if (
            not _accuracy_record_complete(h0_final, count=count, checkpoint=0)
            or h0_final["prediction_sha256"] != h0["prediction_sha256"]
            or h0_final["prediction_histogram"] != h0["prediction_histogram"]
        ):
            return False
        expected_h0_gap = (
            float(evidence["intact"]["accuracy"])
            - float(h0_final["accuracy"])
        )
        expected_shuffled_gap = (
            float(evidence["intact"]["accuracy"])
            - float(evidence["shuffled"]["accuracy"])
        )
        if not (
            math.isclose(
                _finite_real(evidence["intact_minus_h0"], "Gate B H0 gap"),
                expected_h0_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _finite_real(
                    evidence["intact_minus_shuffled"], "Gate B shuffled gap"
                ),
                expected_shuffled_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def _formal_initialization_complete(
    arm_report: Any,
    admission: Mapping[str, Any],
    *,
    regime: str,
    arm: str,
) -> bool:
    if not isinstance(arm_report, Mapping):
        return False
    initialization = arm_report.get("initialization")
    if not isinstance(initialization, Mapping) or set(initialization) != {
        "tree",
        "parameter_sha256",
        "parameter_count",
        "parameter_paths",
        "shared_paths",
    }:
        return False
    regime_admission = admission["initialization"][regime]
    reference = regime_admission["arm_initialization_refs"][arm]
    topology = regime_admission[reference["tree"]]
    expected_paths = (
        SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
    )
    return bool(
        initialization["tree"] == reference["tree"]
        and initialization["parameter_sha256"] == reference["parameter_sha256"]
        and _strict_integer(initialization["parameter_count"])
        and int(initialization["parameter_count"]) == topology["parameter_count"]
        and initialization["parameter_paths"] == list(expected_paths)
        and gate_a._json_exact(
            initialization["shared_paths"],
            regime_admission["shared_paths"],
        )
    )


def _formal_optimizer_complete(
    arm_report: Any,
    admission: Mapping[str, Any],
    *,
    regime: str,
    arm: str,
    updates: int,
) -> bool:
    optimizer = arm_report["optimizer"]
    expected = admission["initialization"][regime]["optimizer_paths"][arm]
    training = arm_report["training"]
    return bool(
        gate_a._json_exact(optimizer, expected)
        and optimizer["fresh_state_finite"] is True
        and optimizer["fresh_state_all_zero"] is True
        and _strict_integer(optimizer["executed_updates"])
        and int(optimizer["executed_updates"]) == 0
        and _strict_integer(training["optimizer_final_step"])
        and int(training["optimizer_final_step"]) == updates
    )


def _expected_parameter_counts(regime: str, arm: str) -> dict[str, int]:
    counts = {
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
    paths = SHARED_PARAMETER_PATHS if arm == "legacy" else FULL_PARAMETER_PATHS
    return {path: counts[path] for path in paths}


def _parameter_movement_complete(
    movement: Any,
    *,
    regime: str,
    arm: str,
) -> bool:
    if not isinstance(movement, Mapping) or set(movement) != {
        "l2_delta",
        "parameter_count",
        "paths",
    }:
        return False
    expected_counts = _expected_parameter_counts(regime, arm)
    paths = movement["paths"]
    if not isinstance(paths, Mapping) or set(paths) != set(expected_counts):
        return False
    squared = 0.0
    count = 0
    for path, expected_count in expected_counts.items():
        value = paths[path]
        if not isinstance(value, Mapping) or set(value) != {
            "l2_delta",
            "parameter_count",
        }:
            return False
        delta = _finite_real(value["l2_delta"], f"{path} movement")
        if (
            delta < 0.0
            or not _strict_integer(value["parameter_count"])
            or int(value["parameter_count"]) != expected_count
        ):
            return False
        expected_zero = path in {
            "height_head/weight",
            "width_head/weight",
        } or (arm == "frozen_write" and path == "memory_write_scale") or (
            arm == "query_only"
            and path == "workspace_query_projection/weight"
        )
        if expected_zero:
            if delta != 0.0:
                return False
        squared += delta * delta
        count += expected_count
    total = _finite_real(movement["l2_delta"], "total parameter movement")
    return bool(
        total > 0.0
        and math.isclose(total, math.sqrt(squared), rel_tol=1e-12, abs_tol=1e-12)
        and _strict_integer(movement["parameter_count"])
        and int(movement["parameter_count"]) == count
    )


def _expected_loss_weight_report(
    config: GateCConfig,
    *,
    regime: str,
    arm: str,
) -> dict[str, Any]:
    if regime == "gate_a":
        efforts = np.ones((1,), dtype=np.int32)
    else:
        efforts = np.resize(
            np.asarray(gate_b.QUALIFYING_EFFORTS, dtype=np.int32),
            config.gate_b_config.training_updates,
        )
    weights = _loss_weights(regime, arm, efforts=efforts)
    return {
        "dtype": weights.dtype.str,
        "shape": list(weights.shape),
        "sha256": legacy._digest_arrays(weights),
    }


def _training_report_complete(
    arm_report: Any,
    admission: Mapping[str, Any],
    config: GateCConfig,
    *,
    regime: str,
    arm: str,
) -> bool:
    expected_arm_keys = {
        "initialization",
        "optimizer",
        "compiler",
        "training",
        "parameter_movement",
        "evaluation",
        "metrics",
    }
    if not isinstance(arm_report, Mapping) or set(arm_report) != expected_arm_keys:
        return False
    training = arm_report["training"]
    expected_training_keys = {
        "algorithm",
        "execution_index",
        "intervention",
        "data_identity",
        "executed_updates",
        "batch_size",
        "chunk_count",
        "cold_compile_and_train_seconds",
        "initial_parameter_sha256",
        "final_parameter_sha256",
        "optimizer_final_step",
        "loss_weights",
        "compile_warnings",
        "losses",
        "initial_loss",
        "final_loss",
        "tail_64_mean_loss",
        "finite",
        "max_abs",
        "value_count",
        "frozen_write",
    }
    if not isinstance(training, Mapping) or set(training) != expected_training_keys:
        return False
    regime_config = (
        config.gate_a_config if regime == "gate_a" else config.gate_b_config
    )
    updates = regime_config.training_updates
    losses = training["losses"]
    if not isinstance(losses, list) or len(losses) != updates:
        return False
    try:
        loss_values = np.asarray(
            [_finite_real(value, "training loss") for value in losses],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return False
    if not np.isfinite(loss_values).all():
        return False
    finite = training["finite"]
    maxima = training["max_abs"]
    value_counts = training["value_count"]
    if not all(
        isinstance(section, Mapping) and set(section) == _TELEMETRY_CATEGORIES
        for section in (finite, maxima, value_counts)
    ):
        return False
    if not all(finite[name] is True for name in _TELEMETRY_CATEGORIES):
        return False
    if not all(
        _finite_real(maxima[name], f"{name} maximum") >= 0.0
        for name in _TELEMETRY_CATEGORIES
    ):
        return False
    if not all(
        _strict_integer(value_counts[name]) and int(value_counts[name]) > 0
        for name in _TELEMETRY_CATEGORIES
    ):
        return False
    reference = admission["initialization"][regime]["arm_initialization_refs"][arm]
    chunk_count = 1 if regime == "gate_a" else config.gate_b_config.staging_chunk_count
    tail = float(loss_values[-min(64, updates) :].mean())
    if not (
        training["algorithm"] == "production_pp_prop"
        and _strict_integer(training["executed_updates"])
        and int(training["executed_updates"]) == updates
        and _strict_integer(training["batch_size"])
        and int(training["batch_size"]) == regime_config.batch_size
        and _strict_integer(training["chunk_count"])
        and int(training["chunk_count"]) == chunk_count
        and _finite_real(
            training["cold_compile_and_train_seconds"], "training time"
        )
        >= 0.0
        and training["initial_parameter_sha256"] == reference["parameter_sha256"]
        and _sha256_complete(training["final_parameter_sha256"])
        and training["final_parameter_sha256"]
        != training["initial_parameter_sha256"]
        and gate_a._json_exact(
            training["loss_weights"],
            _expected_loss_weight_report(config, regime=regime, arm=arm),
        )
        and isinstance(training["compile_warnings"], list)
        and all(isinstance(item, str) for item in training["compile_warnings"])
        and math.isclose(
            _finite_real(training["initial_loss"], "initial loss"),
            float(loss_values[0]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            _finite_real(training["final_loss"], "final loss"),
            float(loss_values[-1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            _finite_real(training["tail_64_mean_loss"], "tail loss"),
            tail,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False
    frozen = training["frozen_write"]
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "applicable",
        "all_ones_before",
        "all_ones_after",
        "excluded_from_optimizer",
    }:
        return False
    if not all(isinstance(frozen[name], bool) for name in frozen):
        return False
    if frozen["applicable"] is not (arm == "frozen_write"):
        return False
    if frozen["excluded_from_optimizer"] is not (arm == "frozen_write"):
        return False
    if arm == "legacy":
        if frozen["all_ones_before"] is not False or frozen["all_ones_after"] is not False:
            return False
    elif frozen["all_ones_before"] is not True:
        return False
    if arm == "frozen_write" and frozen["all_ones_after"] is not True:
        return False
    compiler = arm_report["compiler"]
    reference_tree = reference["tree"]
    expected_compiler = admission["initialization"][regime][reference_tree]["compiler"]
    return bool(
        gate_a._json_exact(compiler, expected_compiler)
        and (
            _legacy_compiler_complete(compiler)
            if arm == "legacy"
            else _full_compiler_complete(compiler)
        )
        and _parameter_movement_complete(
            arm_report["parameter_movement"],
            regime=regime,
            arm=arm,
        )
    )


def _exact_formal_configuration_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    if config != GateCConfig() or config.qualification_regime != "preregistered_full":
        return False
    regimes = report["regimes"]
    arms = report["arms"]
    if (
        report["qualification_regime"] != "preregistered_full"
        or not isinstance(regimes, Mapping)
        or set(regimes) != set(REGIME_ORDER)
        or not isinstance(arms, Mapping)
        or set(arms) != set(ARM_ORDER)
    ):
        return False
    for regime in REGIME_ORDER:
        value = regimes[regime]
        if not isinstance(value, Mapping) or set(value) != {
            "spec",
            "config",
            "schedule",
            "metrics",
            "margins",
            "paired_h0_identity",
        }:
            return False
        regime_config = (
            config.gate_a_config if regime == "gate_a" else config.gate_b_config
        )
        if not (
            gate_a._json_exact(value["spec"], dataclasses.asdict(REGIME_SPECS[regime]))
            and gate_a._json_exact(value["config"], dataclasses.asdict(regime_config))
        ):
            return False
    for regime_index, regime in enumerate(REGIME_ORDER):
        for arm_index, arm in enumerate(ARM_ORDER):
            if not isinstance(arms[arm], Mapping) or set(arms[arm]) != set(
                REGIME_ORDER
            ):
                return False
            arm_report = arms[arm][regime]
            if not isinstance(arm_report, Mapping) or set(arm_report) != {
                "initialization",
                "optimizer",
                "compiler",
                "training",
                "parameter_movement",
                "evaluation",
                "metrics",
            }:
                return False
            training = arm_report["training"]
            expected_index = regime_index * len(ARM_ORDER) + arm_index
            if not (
                isinstance(training, Mapping)
                and _strict_integer(training.get("execution_index"))
                and int(training["execution_index"]) == expected_index
                and gate_a._json_exact(
                    training.get("intervention"),
                    dataclasses.asdict(ARM_SPECS[arm]),
                )
            ):
                return False
    return True


def _canonical_schedules_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    expected = _schedule_identity_report(config)
    for regime in REGIME_ORDER:
        if not gate_a._json_exact(report["regimes"][regime]["schedule"], expected[regime]):
            return False
        for arm in ARM_ORDER:
            training = report["arms"][arm][regime]["training"]
            if not (
                gate_a._json_exact(training["data_identity"], expected[regime])
                and gate_a._json_exact(
                    training["loss_weights"],
                    _expected_loss_weight_report(config, regime=regime, arm=arm),
                )
            ):
                return False
    return True


def _paired_h0_identity_complete(
    value: Any,
    admission: Mapping[str, Any],
    *,
    regime: str,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "checkpoint",
        "initialization_parameter_sha256",
        "streams",
        "passed",
    }:
        return False
    canonical_sha = admission["initialization"][regime]["canonical_full"][
        "parameter_sha256"
    ]
    if not (
        _strict_integer(value["checkpoint"])
        and int(value["checkpoint"]) == 0
        and gate_a._json_exact(
            value["initialization_parameter_sha256"],
            {"full": canonical_sha, "query_only": canonical_sha},
        )
        and isinstance(value["streams"], Mapping)
        and set(value["streams"]) == {"intact", "shuffled", "no_context"}
    ):
        return False
    for stream in value["streams"].values():
        if not isinstance(stream, Mapping) or set(stream) != {
            "full_compact_sha256",
            "query_only_compact_sha256",
            "full_state_sha256",
            "query_only_state_sha256",
            "compact_byte_identical",
            "state_byte_identical",
        }:
            return False
        if not (
            all(
                _sha256_complete(stream[name])
                for name in (
                    "full_compact_sha256",
                    "query_only_compact_sha256",
                    "full_state_sha256",
                    "query_only_state_sha256",
                )
            )
            and stream["full_compact_sha256"]
            == stream["query_only_compact_sha256"]
            and stream["full_state_sha256"] == stream["query_only_state_sha256"]
            and stream["compact_byte_identical"] is True
            and stream["state_byte_identical"] is True
        ):
            return False
    return value["passed"] is True


def _full_gate_a_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    arm_report = report["arms"]["full"]["gate_a"]
    evaluation = arm_report["evaluation"]
    if not _gate_a_evaluation_complete(evaluation, config):
        return False
    final = evaluation["depths"][str(config.gate_a_config.gap_steps)]
    intact = final["intact"]
    shuffled = final["shuffled"]
    memory = evaluation["binding_diagnostic"]["memory"]
    pairing_chance = 1.0 / legacy.SYMBOL_COUNT
    return bool(
        float(intact["accuracy"]) >= 0.80
        and float(intact["wilson_95_lower"]) > pairing_chance
        and float(evaluation["intact_minus_shuffled"]) >= 0.25
        and float(shuffled["wilson_95_lower"]) <= pairing_chance
        and gate_a._diagnostic_evidence_complete(
            evaluation["binding_diagnostic"],
            config.gate_a_config,
        )
        and _strict_integer(memory["applicable_count"])
        and int(memory["applicable_count"])
        == config.gate_a_config.validation_episodes
        and _strict_integer(memory["intact_shuffled_different_count"])
        and int(memory["intact_shuffled_different_count"])
        == int(memory["applicable_count"])
        and memory["every_intact_shuffled_pair_differs"] is True
        and memory["no_context_exact_zero"] is True
        and _full_compiler_complete(arm_report["compiler"])
        and arm_report["compiler"].get(
            "context_memory_isolated_from_workspace_lif"
        )
        is True
    )


def _full_gate_b_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    evaluation = report["arms"]["full"]["gate_b"]["evaluation"]
    if not _gate_b_evaluation_complete(
        evaluation,
        config,
        require_no_collapse=True,
    ):
        return False
    efforts = evaluation["efforts"]
    improvements = [
        float(efforts[str(effort)]["intact_minus_h0"])
        for effort in gate_b.QUALIFYING_EFFORTS
    ]
    return bool(
        all(
            float(efforts[str(effort)]["intact"]["wilson_95_lower"])
            > 1.0 / 8.0
            for effort in gate_b.QUALIFYING_EFFORTS
        )
        and sum(value >= 0.15 for value in improvements) >= 2
        and all(
            float(efforts[str(effort)]["intact_minus_shuffled"]) >= 0.15
            for effort in gate_b.QUALIFYING_EFFORTS
        )
        and all(
            float(efforts[str(effort)][stream]["wilson_95_lower"])
            <= 1.0 / 8.0
            for effort in gate_b.QUALIFYING_EFFORTS
            for stream in ("shuffled", "no_context")
        )
        and float(evaluation["h0_proper"]["wilson_95_lower"]) > 1.0 / 8.0
    )


def _behavioral_margins_complete(
    report: Mapping[str, Any],
    config: GateCConfig,
) -> bool:
    metrics: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        gate_a_evaluation = report["arms"][arm]["gate_a"]["evaluation"]
        gate_b_evaluation = report["arms"][arm]["gate_b"]["evaluation"]
        if not (
            _gate_a_evaluation_complete(gate_a_evaluation, config)
            and _gate_b_evaluation_complete(
                gate_b_evaluation,
                config,
                require_no_collapse=arm == "full",
            )
        ):
            return False
        metrics[arm] = _metric_summary(gate_a_evaluation, gate_b_evaluation)
        for regime in REGIME_ORDER:
            if not gate_a._json_exact(
                report["arms"][arm][regime]["metrics"],
                metrics[arm],
            ):
                return False
    expected_margins = _blocking_margin_report(metrics)
    for regime in REGIME_ORDER:
        regime_report = report["regimes"][regime]
        if not (
            gate_a._json_exact(regime_report["metrics"], metrics)
            and gate_a._json_exact(regime_report["margins"], expected_margins)
        ):
            return False
    return expected_margins["blocking_passed"] is True


def _frozen_write_complete(report: Mapping[str, Any]) -> bool:
    for regime in REGIME_ORDER:
        arm_report = report["arms"]["frozen_write"][regime]
        frozen = arm_report["training"]["frozen_write"]
        movement = arm_report["parameter_movement"]["paths"][
            "memory_write_scale"
        ]
        if not (
            gate_a._json_exact(
                frozen,
                {
                    "applicable": True,
                    "all_ones_before": True,
                    "all_ones_after": True,
                    "excluded_from_optimizer": True,
                },
            )
            and _strict_integer(movement["parameter_count"])
            and int(movement["parameter_count"]) == 1_024
            and _finite_real(movement["l2_delta"], "frozen write movement") == 0.0
        ):
            return False
    metrics = {
        arm: report["regimes"]["gate_a"]["metrics"][arm]
        for arm in ARM_ORDER
    }
    expected = _blocking_margin_report(metrics)["frozen_write"]
    return all(
        gate_a._json_exact(
            report["regimes"][regime]["margins"]["frozen_write"],
            expected,
        )
        for regime in REGIME_ORDER
    )


def _source_and_gpu_formal_complete(
    report: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> bool:
    return bool(
        _source_and_gpu_complete(report)
        and _source_files_complete(report["source_files"])
        and gate_a._json_exact(report["source_files"], admission["source_files"])
        and report["source_start"]["commit"]
        == admission["source_start"]["commit"]
        and report["environment"]["image_digest"]
        == admission["environment"]["image_digest"]
    )


_GRADIENT_RECORD_KEYS = {
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


def _gradient_record_complete(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != _GRADIENT_RECORD_KEYS:
        return False
    try:
        full_norm = _finite_real(value["full_norm"], "full gradient norm")
        arm_norm = _finite_real(value["arm_norm"], "arm gradient norm")
        difference = _finite_real(value["l2_difference"], "gradient difference")
    except (TypeError, ValueError):
        return False
    if min(full_norm, arm_norm, difference) < 0.0:
        return False
    if full_norm == 0.0 or arm_norm == 0.0:
        if not math.isclose(
            difference,
            max(full_norm, arm_norm),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return False
    else:
        tolerance = max(1e-12, 1e-9 * (full_norm + arm_norm))
        if not (
            abs(full_norm - arm_norm) - tolerance
            <= difference
            <= full_norm + arm_norm + tolerance
        ):
            return False
    relative_defined = full_norm > 0.0
    cosine_defined = full_norm > 0.0 and arm_norm > 0.0
    if (
        value["relative_deviation_defined"] is not relative_defined
        or value["cosine_defined"] is not cosine_defined
        or not _sha256_complete(value["full_sha256"])
        or not _sha256_complete(value["arm_sha256"])
    ):
        return False
    if relative_defined:
        try:
            relative = _finite_real(
                value["relative_deviation"], "relative gradient deviation"
            )
        except (TypeError, ValueError):
            return False
        if not math.isclose(
            relative,
            difference / full_norm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return False
    elif value["relative_deviation"] is not None:
        return False
    if cosine_defined:
        try:
            cosine = _finite_real(value["cosine"], "gradient cosine")
        except (TypeError, ValueError):
            return False
        expected_cosine = (
            full_norm * full_norm + arm_norm * arm_norm - difference * difference
        ) / (2.0 * full_norm * arm_norm)
        if not (
            -1.0 - 1e-9 <= cosine <= 1.0 + 1e-9
            and math.isclose(
                cosine,
                expected_cosine,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            return False
    elif value["cosine"] is not None:
        return False
    return True


def _gradient_digest_from_records(
    paths: Mapping[str, Mapping[str, Any]],
    *,
    side: str,
) -> str:
    fields: list[bytes] = [b"example21-gate-c-gradient-global-v1"]
    for path in sorted(paths):
        fields.extend(
            (
                path.encode("utf-8"),
                str(paths[path][f"{side}_sha256"]).encode("ascii"),
            )
        )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()


def _gradient_global_algebra_complete(
    value: Mapping[str, Any],
    paths: Mapping[str, Mapping[str, Any]],
    *,
    allow_null_cosine: bool,
) -> bool:
    full_norm = math.sqrt(
        math.fsum(float(record["full_norm"]) ** 2 for record in paths.values())
    )
    arm_norm = math.sqrt(
        math.fsum(float(record["arm_norm"]) ** 2 for record in paths.values())
    )
    difference = math.sqrt(
        math.fsum(
            float(record["l2_difference"]) ** 2 for record in paths.values()
        )
    )
    dot = math.fsum(
        0.0
        if record["cosine"] is None
        else float(record["cosine"])
        * float(record["full_norm"])
        * float(record["arm_norm"])
        for record in paths.values()
    )
    cosine = dot / (full_norm * arm_norm) if full_norm > 0.0 and arm_norm > 0.0 else None
    return bool(
        math.isclose(
            float(value["full_norm"]), full_norm, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            float(value["arm_norm"]), arm_norm, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            float(value["l2_difference"]),
            difference,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and (
            (allow_null_cosine and cosine is None and value["cosine"] is None)
            or (
                cosine is not None
                and value["cosine"] is not None
                and math.isclose(
                    float(value["cosine"]), cosine, rel_tol=1e-9, abs_tol=1e-12
                )
            )
        )
        and value["full_sha256"]
        == _gradient_digest_from_records(paths, side="full")
        and value["arm_sha256"] == _gradient_digest_from_records(paths, side="arm")
    )


def _mechanism_oracle_complete(
    value: Any,
    config: GateCConfig,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "contract",
        "objective",
        "gradient_chunk_size",
        "comparisons",
        "complete",
    }:
        return False
    if not (
        gate_a._json_exact(value["contract"], _oracle_contract(config))
        and gate_a._json_exact(
            value["objective"],
            {
                "wrapper": "sqrt_weight_times_sqrt_cross_entropy",
                "unsupervised_output_exact_zero": True,
            },
        )
        and _strict_integer(value["gradient_chunk_size"])
        and int(value["gradient_chunk_size"]) == 1
    ):
        return False
    comparisons = value["comparisons"]
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "query_only",
        "terminal_only",
    }:
        return False
    expected_required = {
        "query_only": [
            "memory_read_projection/weight",
            "workspace_query_projection/weight",
        ],
        "terminal_only": [],
    }
    recomputed_pass: dict[str, bool] = {}
    full_records: dict[str, dict[str, Any]] = {}
    for arm in ("query_only", "terminal_only"):
        comparison = comparisons[arm]
        if not isinstance(comparison, Mapping) or set(comparison) != {
            "global",
            "paths",
            "required_paths",
            "required_paths_passed",
            "passed",
        }:
            return False
        paths = comparison["paths"]
        if not isinstance(paths, Mapping) or tuple(sorted(paths)) != FULL_PARAMETER_PATHS:
            return False
        if not all(_gradient_record_complete(paths[path]) for path in FULL_PARAMETER_PATHS):
            return False
        global_record = comparison["global"]
        if not _gradient_record_complete(global_record):
            return False
        if not _gradient_global_algebra_complete(
            global_record,
            paths,
            allow_null_cosine=False,
        ):
            return False
        if comparison["required_paths"] != expected_required[arm]:
            return False

        def threshold(record: Mapping[str, Any]) -> bool:
            return bool(
                float(record["full_norm"]) > 0.0
                and record["relative_deviation_defined"] is True
                and float(record["relative_deviation"]) >= 1e-3
                and float(record["l2_difference"])
                > max(1e-8, 1e-4 * float(record["full_norm"]))
            )

        required_passed = all(threshold(paths[path]) for path in expected_required[arm])
        global_passed = bool(float(global_record["arm_norm"]) > 0.0 and threshold(global_record))
        passed = bool(global_passed and required_passed)
        if not (
            comparison["required_paths_passed"] is required_passed
            and comparison["passed"] is passed
        ):
            return False
        recomputed_pass[arm] = passed
        full_records[arm] = {
            path: {
                "norm": paths[path]["full_norm"],
                "sha256": paths[path]["full_sha256"],
            }
            for path in FULL_PARAMETER_PATHS
        }
    if not gate_a._json_exact(
        full_records["query_only"],
        full_records["terminal_only"],
    ):
        return False
    complete = all(recomputed_pass.values())
    return value["complete"] is complete and complete


def _gate_c2_controls_base_environment(value: Any) -> dict[str, Any]:
    expected = {
        "backend",
        "devices",
        "image_digest",
        "jax",
        "python",
        "execution_and_update_evidence",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Gate C2 controls environment schema differs")
    return {
        name: value[name]
        for name in ("backend", "devices", "image_digest", "jax", "python")
    }


def _gate_c2_controls_qualification(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Recompute the Gate C2 pretraining-control admission."""

    criteria = {
        name: False for name in GATE_C2_CONTROLS_QUALIFICATION_CRITERIA
    }
    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        return {
            "criteria": criteria,
            "passed": False,
            "interpretation": "gate_c2_pretraining_controls_failed_stop",
        }
    base_keys = set(GATE_C2_CONTROLS_TOP_LEVEL_KEYS) - {"qualification"}
    try:
        qualification_shape = "qualification" not in report or bool(
            isinstance(report["qualification"], Mapping)
            and set(report["qualification"])
            == {"criteria", "passed", "interpretation"}
        )
        criteria["schema_and_control"] = bool(
            set(report) in (base_keys, set(GATE_C2_CONTROLS_TOP_LEVEL_KEYS))
            and qualification_shape
            and _strict_integer(report["schema_version"])
            and int(report["schema_version"])
            == GATE_C2_CONTROLS_SCHEMA_VERSION
            and report["control"] == GATE_C2_CONTROLS_CONTROL
            and report["qualification_regime"]
            == GATE_C2_CONTROLS_QUALIFICATION_REGIME
            and report["learner"] == "pp_prop_only"
            and _finite_real(report["total_wall_seconds"], "total wall time")
            >= 0.0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    try:
        regimes = report["regimes"]
        criteria["exact_configuration"] = bool(
            config == GateCConfig()
            and isinstance(regimes, Mapping)
            and set(regimes) == set(REGIME_ORDER)
            and all(
                isinstance(regimes[regime], Mapping)
                and set(regimes[regime])
                == {
                    "spec",
                    "config",
                    "schedule_identity",
                    "paired_h0_operational_equivalence",
                    "query_only_latent_no_read",
                }
                and gate_a._json_exact(
                    regimes[regime]["spec"],
                    dataclasses.asdict(REGIME_SPECS[regime]),
                )
                and gate_a._json_exact(
                    regimes[regime]["config"],
                    dataclasses.asdict(
                        config.gate_a_config
                        if regime == "gate_a"
                        else config.gate_b_config
                    ),
                )
                for regime in REGIME_ORDER
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    prerequisites: Any = report.get("prerequisites")
    try:
        criteria["prerequisites_authenticated"] = bool(
            isinstance(prerequisites, Mapping)
            and set(prerequisites)
            == {"gate_a", "gate_b", "gate_c_initialization"}
            and gate_a._json_exact(
                _normalized_prerequisites(
                    {
                        "gate_a": prerequisites["gate_a"],
                        "gate_b": prerequisites["gate_b"],
                    }
                ),
                {"gate_a": _GATE_A_REFERENCE, "gate_b": _GATE_B_REFERENCE},
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    admission: Mapping[str, Any] | None = None
    try:
        base_environment = _gate_c2_controls_base_environment(
            report["environment"]
        )
        admission = _validated_gate_c_initialization_admission(
            prerequisites["gate_c_initialization"],
            config,
            source_start=report["source_start"],
            environment=base_environment,
            source_files=report["source_files"],
            require_pass=True,
        )
        criteria["initialization_authenticated"] = True
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        admission = None

    try:
        schedules = _schedule_identity_report(config)
        criteria["canonical_schedules_complete"] = all(
            gate_a._json_exact(
                report["regimes"][regime]["schedule_identity"],
                schedules[regime],
            )
            for regime in REGIME_ORDER
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    if admission is not None:
        try:
            criteria["no_behavioral_or_optimizer_updates"] = (
                _gate_c2_no_update_evidence_complete(
                    report["environment"]["execution_and_update_evidence"],
                    admission,
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        try:
            criteria["paired_h0_operational_equivalence"] = all(
                _gate_c2_paired_h0_operational_equivalence_complete(
                    report["regimes"][regime][
                        "paired_h0_operational_equivalence"
                    ],
                    admission,
                    regime=regime,
                )
                and _gate_c2_query_only_latent_no_read_complete(
                    report["regimes"][regime]["query_only_latent_no_read"],
                    admission,
                    regime=regime,
                )
                for regime in REGIME_ORDER
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass

    try:
        criteria["mechanism_oracle_complete"] = _mechanism_oracle_complete(
            report["mechanism_oracle"],
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    if admission is not None:
        try:
            base_report = {
                **dict(report),
                "environment": _gate_c2_controls_base_environment(
                    report["environment"]
                ),
            }
            criteria["source_and_gpu_authenticated"] = bool(
                _source_and_gpu_complete(base_report)
                and _source_files_complete(report["source_files"])
                and gate_a._json_exact(
                    report["source_files"], admission["source_files"]
                )
                and report["source_start"]["commit"]
                == admission["source_start"]["commit"]
                and report["environment"]["image_digest"]
                == admission["environment"]["image_digest"]
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass

    passed = bool(criteria and all(criteria.values()))
    return {
        "criteria": criteria,
        "passed": passed,
        "interpretation": (
            "gate_c2_pretraining_controls_passed"
            if passed
            else "gate_c2_pretraining_controls_failed_stop"
        ),
    }


def _gate_c2_controls_qualification_report(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Compatibility name used by the in-process controls runner."""

    return _gate_c2_controls_qualification(report, config=config)


def _gate_c3_controls_base_environment(value: Any) -> dict[str, Any]:
    base_names = {"backend", "devices", "image_digest", "jax", "python"}
    allowed = base_names | {"deterministic_environment"}
    if isinstance(value, Mapping) and "execution_and_update_evidence" in value:
        allowed.add("execution_and_update_evidence")
    if (
        not isinstance(value, Mapping)
        or set(value) != allowed
        or not _gate_c3_deterministic_environment_complete(
            value.get("deterministic_environment")
        )
    ):
        raise ValueError("Gate C3 controls environment schema differs")
    return {name: value[name] for name in base_names}


def _gate_c3_controls_qualification(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Recompute the tri-state Gate C3 controls admission."""

    criteria = {
        name: False for name in GATE_C3_CONTROLS_QUALIFICATION_CRITERIA
    }
    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        return {
            "valid": False,
            "passed": False,
            "criteria": criteria,
            "failures": sorted(criteria),
            "interpretation": GATE_C3_CONTROLS_INVALID_INTERPRETATION,
        }
    base_keys = set(GATE_C3_CONTROLS_TOP_LEVEL_KEYS) - {"qualification"}
    qualification_shape = "qualification" not in report or bool(
        isinstance(report.get("qualification"), Mapping)
        and set(report["qualification"])
        == {"valid", "passed", "criteria", "failures", "interpretation"}
    )
    structural_valid = False
    try:
        structural_valid = bool(
            set(report) in (base_keys, set(GATE_C3_CONTROLS_TOP_LEVEL_KEYS))
            and qualification_shape
            and isinstance(report["prerequisites"], Mapping)
            and isinstance(report["regimes"], Mapping)
            and isinstance(report["mechanism_oracle"], Mapping)
            and isinstance(report["source_start"], Mapping)
            and isinstance(report["source_end"], Mapping)
            and isinstance(report["source_files"], Mapping)
            and isinstance(report["environment"], Mapping)
            and _finite_real(report["total_wall_seconds"], "total wall time")
            >= 0.0
        )
        criteria["schema_and_control"] = bool(
            structural_valid
            and _strict_integer(report["schema_version"])
            and int(report["schema_version"])
            == GATE_C3_CONTROLS_SCHEMA_VERSION
            and report["control"] == GATE_C3_CONTROLS_CONTROL
            and report["qualification_regime"]
            == GATE_C3_CONTROLS_QUALIFICATION_REGIME
            and report["learner"] == "pp_prop_only"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        structural_valid = False

    try:
        regimes = report["regimes"]
        criteria["exact_configuration"] = bool(
            config == GateCConfig()
            and isinstance(regimes, Mapping)
            and set(regimes) == set(REGIME_ORDER)
            and all(
                isinstance(regimes[regime], Mapping)
                and set(regimes[regime])
                == {
                    "spec",
                    "config",
                    "schedule_identity",
                    "paired_h0_operational_equivalence",
                    "query_only_latent_no_read",
                }
                and gate_a._json_exact(
                    regimes[regime]["spec"],
                    dataclasses.asdict(REGIME_SPECS[regime]),
                )
                and gate_a._json_exact(
                    regimes[regime]["config"],
                    dataclasses.asdict(
                        config.gate_a_config
                        if regime == "gate_a"
                        else config.gate_b_config
                    ),
                )
                for regime in REGIME_ORDER
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    prerequisites: Any = report.get("prerequisites")
    try:
        criteria["prerequisites_authenticated"] = bool(
            isinstance(prerequisites, Mapping)
            and set(prerequisites)
            == {"gate_a", "gate_b", "gate_c_initialization"}
            and gate_a._json_exact(
                _normalized_prerequisites(
                    {
                        "gate_a": prerequisites["gate_a"],
                        "gate_b": prerequisites["gate_b"],
                    }
                ),
                {"gate_a": _GATE_A_REFERENCE, "gate_b": _GATE_B_REFERENCE},
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    admission: Mapping[str, Any] | None = None
    base_environment: dict[str, Any] | None = None
    evidence_valid = True
    try:
        base_environment = _gate_c3_controls_base_environment(
            report["environment"]
        )
        criteria["deterministic_environment_authenticated"] = True
        admission = _validated_gate_c_initialization_admission(
            prerequisites["gate_c_initialization"],
            config,
            source_start=report["source_start"],
            environment=base_environment,
            source_files=report["source_files"],
            require_pass=True,
        )
        criteria["initialization_authenticated"] = True
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        admission = None

    try:
        schedules = _schedule_identity_report(config)
        criteria["canonical_schedules_complete"] = all(
            gate_a._json_exact(
                report["regimes"][regime]["schedule_identity"],
                schedules[regime],
            )
            for regime in REGIME_ORDER
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    if admission is not None:
        try:
            criteria["no_behavioral_or_optimizer_updates"] = (
                _gate_c2_no_update_evidence_complete(
                    report["environment"]["execution_and_update_evidence"],
                    admission,
                    model_roles=GATE_C3_CONTROLS_MODEL_ROLES,
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        try:
            criteria["paired_h0_operational_equivalence"] = all(
                _gate_c2_paired_h0_operational_equivalence_complete(
                    report["regimes"][regime][
                        "paired_h0_operational_equivalence"
                    ],
                    admission,
                    regime=regime,
                )
                for regime in REGIME_ORDER
            )
            evidence_valid = bool(
                evidence_valid
                and all(
                    _gate_c2_paired_h0_operational_equivalence_complete(
                        report["regimes"][regime][
                            "paired_h0_operational_equivalence"
                        ],
                        admission,
                        regime=regime,
                        require_pass=False,
                    )
                    for regime in REGIME_ORDER
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            evidence_valid = False
        try:
            criteria["no_read_and_removed_path_complete"] = all(
                _gate_c2_query_only_latent_no_read_complete(
                    report["regimes"][regime]["query_only_latent_no_read"],
                    admission,
                    regime=regime,
                )
                for regime in REGIME_ORDER
            )
            evidence_valid = bool(
                evidence_valid
                and all(
                    _gate_c2_query_only_latent_no_read_complete(
                        report["regimes"][regime][
                            "query_only_latent_no_read"
                        ],
                        admission,
                        regime=regime,
                        require_pass=False,
                    )
                    for regime in REGIME_ORDER
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            evidence_valid = False

    try:
        mechanism_valid, mechanism_complete = (
            _gate_c3_terminal_h8_mechanism_oracle_status(
                report["mechanism_oracle"],
                config,
                canonical_parameter_sha256=(
                    None
                    if admission is None
                    else admission["initialization"]["gate_b"][
                        "canonical_full"
                    ]["parameter_sha256"]
                ),
                audit_evidence=(
                    None
                    if admission is None
                    else report["environment"][
                        "execution_and_update_evidence"
                    ]
                ),
            )
        )
        criteria["mechanism_oracle_complete"] = mechanism_complete
        evidence_valid = bool(evidence_valid and mechanism_valid)
    except (KeyError, TypeError, ValueError, OverflowError):
        evidence_valid = False

    if admission is not None and base_environment is not None:
        try:
            base_report = {**dict(report), "environment": base_environment}
            criteria["source_and_gpu_authenticated"] = bool(
                _source_and_gpu_complete(base_report)
                and _source_files_complete(report["source_files"])
                and gate_a._json_exact(
                    report["source_files"],
                    admission["source_files"],
                )
                and report["source_start"]["commit"]
                == admission["source_start"]["commit"]
                and report["environment"]["image_digest"]
                == admission["environment"]["image_digest"]
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass

    infrastructure_criteria = (
        "schema_and_control",
        "exact_configuration",
        "prerequisites_authenticated",
        "initialization_authenticated",
        "deterministic_environment_authenticated",
        "canonical_schedules_complete",
        "no_behavioral_or_optimizer_updates",
        "source_and_gpu_authenticated",
    )
    valid = bool(
        structural_valid
        and all(criteria[name] for name in infrastructure_criteria)
        and evidence_valid
    )
    failures = sorted(name for name, passed in criteria.items() if not passed)
    passed = bool(valid and not failures)
    return {
        "valid": valid,
        "passed": passed,
        "criteria": criteria,
        "failures": failures,
        "interpretation": (
            GATE_C3_CONTROLS_INVALID_INTERPRETATION
            if not valid
            else (
                GATE_C3_CONTROLS_PASSING_INTERPRETATION
                if passed
                else GATE_C3_CONTROLS_FAILING_INTERPRETATION
            )
        ),
    }


def _gate_c3_controls_qualification_report(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    return _gate_c3_controls_qualification(report, config=config)


def _gate_c2_qualification_report(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Fail-closed schema-2 qualifier for the staged formal target."""

    criteria = {name: False for name in GATE_C2_QUALIFICATION_CRITERIA}
    if isinstance(report, Mapping) and isinstance(config, GateCConfig):
        base_keys = set(GATE_C2_TOP_LEVEL_KEYS) - {"qualification"}
        try:
            criteria["schema_and_control"] = bool(
                set(report) in (base_keys, set(GATE_C2_TOP_LEVEL_KEYS))
                and _strict_integer(report["schema_version"])
                and int(report["schema_version"]) == GATE_C2_SCHEMA_VERSION
                and report["control"] == GATE_C2_CONTROL
                and report["qualification_regime"]
                == GATE_C2_QUALIFICATION_REGIME
                and report["learner"] == "pp_prop_only"
                and _finite_real(
                    report["total_wall_seconds"], "total wall time"
                )
                >= 0.0
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    passed = bool(criteria and all(criteria.values()))
    return {
        "criteria": criteria,
        "passed": passed,
        "interpretation": (
            GATE_C2_PASSING_INTERPRETATION
            if passed
            else GATE_C2_FAILING_INTERPRETATION
        ),
    }


def _qualification_report(
    report: Mapping[str, Any],
    *,
    config: GateCConfig,
) -> dict[str, Any]:
    """Recompute every formal Gate C criterion from retained raw evidence."""

    criteria = {name: False for name in QUALIFICATION_CRITERIA}
    if not isinstance(report, Mapping) or not isinstance(config, GateCConfig):
        passed = False
        return {
            "criteria": criteria,
            "passed": passed,
            "interpretation": "gate_c_failed_stop_no_causal_mechanism_conclusion",
        }
    base_keys = {
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
        "total_wall_seconds",
    }
    try:
        keys = set(report)
        qualification_shape = True
        if "qualification" in report:
            qualification_shape = bool(
                isinstance(report["qualification"], Mapping)
                and set(report["qualification"])
                == {"criteria", "passed", "interpretation"}
            )
        criteria["schema_and_control"] = bool(
            keys in (base_keys, base_keys | {"qualification"})
            and qualification_shape
            and _strict_integer(report["schema_version"])
            and int(report["schema_version"]) == GATE_C_SCHEMA_VERSION
            and report["control"] == GATE_C_CONTROL
            and report["learner"] == "pp_prop_only"
            and _finite_real(report["total_wall_seconds"], "total wall time") >= 0.0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    try:
        criteria["exact_configuration"] = _exact_formal_configuration_complete(
            report,
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    prerequisites: Any = report.get("prerequisites")
    try:
        criteria["prerequisites_authenticated"] = bool(
            isinstance(prerequisites, Mapping)
            and set(prerequisites)
            == {"gate_a", "gate_b", "gate_c_initialization"}
            and gate_a._json_exact(
                _normalized_prerequisites(
                    {
                        "gate_a": prerequisites["gate_a"],
                        "gate_b": prerequisites["gate_b"],
                    }
                ),
                {"gate_a": _GATE_A_REFERENCE, "gate_b": _GATE_B_REFERENCE},
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    admission: Mapping[str, Any] | None = None
    try:
        admission = _validated_gate_c_initialization_admission(
            prerequisites["gate_c_initialization"],
            config,
            source_start=report["source_start"],
            environment=report["environment"],
            source_files=report["source_files"],
            require_pass=True,
        )
        criteria["initialization_authenticated"] = all(
            _formal_initialization_complete(
                report["arms"][arm][regime],
                admission,
                regime=regime,
                arm=arm,
            )
            for regime in REGIME_ORDER
            for arm in ARM_ORDER
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        admission = None

    try:
        criteria["canonical_schedules_complete"] = (
            _canonical_schedules_complete(report, config)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass

    if admission is not None:
        try:
            optimizer_digests: set[str] = set()
            optimizers_complete = True
            for regime in REGIME_ORDER:
                regime_config = (
                    config.gate_a_config
                    if regime == "gate_a"
                    else config.gate_b_config
                )
                for arm in ARM_ORDER:
                    arm_report = report["arms"][arm][regime]
                    optimizers_complete = bool(
                        optimizers_complete
                        and _formal_optimizer_complete(
                            arm_report,
                            admission,
                            regime=regime,
                            arm=arm,
                            updates=regime_config.training_updates,
                        )
                    )
                    optimizer_digests.add(arm_report["optimizer"]["state_sha256"])
            criteria["fresh_isolated_optimizers"] = bool(
                optimizers_complete and len(optimizer_digests) == 10
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass

        try:
            criteria["compiler_and_training_complete"] = all(
                _training_report_complete(
                    report["arms"][arm][regime],
                    admission,
                    config,
                    regime=regime,
                    arm=arm,
                )
                and (
                    _gate_a_evaluation_complete(
                        report["arms"][arm][regime]["evaluation"],
                        config,
                    )
                    and _gate_a_intervention_diagnostic_complete(
                        report["arms"][arm][regime]["evaluation"],
                        arm=arm,
                    )
                    if regime == "gate_a"
                    else _gate_b_evaluation_complete(
                        report["arms"][arm][regime]["evaluation"],
                        config,
                        require_no_collapse=arm == "full",
                    )
                )
                for regime in REGIME_ORDER
                for arm in ARM_ORDER
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass

    try:
        criteria["full_gate_a_passed"] = _full_gate_a_complete(report, config)
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    try:
        criteria["full_gate_b_passed"] = _full_gate_b_complete(report, config)
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    try:
        criteria["blocking_behavioral_margins"] = _behavioral_margins_complete(
            report,
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    if admission is not None:
        try:
            criteria["paired_h0_identity"] = all(
                _paired_h0_identity_complete(
                    report["regimes"][regime]["paired_h0_identity"],
                    admission,
                    regime=regime,
                )
                for regime in REGIME_ORDER
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    try:
        criteria["frozen_write_complete"] = _frozen_write_complete(report)
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    try:
        criteria["mechanism_oracle_complete"] = _mechanism_oracle_complete(
            report["mechanism_oracle"],
            config,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    if admission is not None:
        try:
            criteria["source_and_gpu_authenticated"] = (
                _source_and_gpu_formal_complete(report, admission)
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass
    passed = bool(criteria and all(criteria.values()))
    return {
        "criteria": criteria,
        "passed": passed,
        "interpretation": (
            "gate_c_passed_pp_prop_learnability_mechanism"
            if passed
            else "gate_c_failed_stop_no_causal_mechanism_conclusion"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=(
            "gate_c_init",
            "formal_gate_c",
            "gate_c2_controls",
            "gate_c3_controls",
        ),
        required=True,
    )
    parser.add_argument("--gate-a-result", type=Path, required=True)
    parser.add_argument("--gate-a-manifest", type=Path, required=True)
    parser.add_argument("--gate-b-manifest", type=Path, required=True)
    parser.add_argument("--gate-c-init-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a fixed authenticated Gate C target.

    Parameters
    ----------
    argv
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a complete artifact is written. Scientific failure remains
        encoded in the artifact for the authenticated launcher to sign.
    """

    from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher

    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    launch_config = launcher.LaunchConfig(
        target=args.target,
        repo_root=repo_root,
        output_dir=args.output.resolve().parent,
    )
    gate_a_paths = launcher._gate_a_artifact_paths(launch_config)
    gate_b_paths = launcher._formal_gate_b_artifact_paths(launch_config)
    if (
        args.gate_a_result.resolve() != gate_a_paths.result.resolve()
        or args.gate_a_manifest.resolve() != gate_a_paths.manifest.resolve()
    ):
        raise ValueError("Gate C target requires the fixed Gate A artifact paths")
    if args.gate_b_manifest.resolve() != gate_b_paths.manifest.resolve():
        raise ValueError("Gate C target requires the fixed Gate B manifest path")

    if args.target == "gate_c_init":
        if args.gate_c_init_manifest is not None:
            raise ValueError(
                "gate_c_init does not accept a Gate C initialization manifest"
            )
    elif args.gate_c_init_manifest is None:
        raise ValueError(
            f"{args.target} requires the fixed initialization manifest"
        )

    source_start = gate_a._source_report()
    environment = gate_a._environment_report()
    gate_a._require_authenticated_gpu_launch(source_start, environment)
    head = str(source_start["commit"])
    expected_paths = launcher.target_paths(
        launch_config,
        head,
        args.target,
    )
    if args.output.resolve() != expected_paths.result.resolve():
        raise ValueError("Gate C target requires the fixed output path")

    if args.target in {
        "formal_gate_c",
        "gate_c2_controls",
        "gate_c3_controls",
    }:
        initialization_paths = launcher.target_paths(
            launch_config,
            head,
            "gate_c_init",
        )
        if (
            args.gate_c_init_manifest is None
            or args.gate_c_init_manifest.resolve()
            != initialization_paths.manifest.resolve()
        ):
            raise ValueError(
                f"{args.target} requires the fixed initialization manifest path"
            )
        if args.target == "formal_gate_c":
            prerequisites = launcher._load_formal_gate_c_prerequisites(
                launch_config,
                head=head,
                image_id=str(environment["image_digest"]),
            )
        elif args.target == "gate_c2_controls":
            prerequisites = launcher._load_gate_c2_controls_prerequisites(
                launch_config,
                head=head,
                image_id=str(environment["image_digest"]),
            )
        else:
            prerequisites = launcher._load_gate_c3_controls_prerequisites(
                launch_config,
                head=head,
                image_id=str(environment["image_digest"]),
            )
    else:
        prerequisites = launcher._load_gate_c_prerequisites(launch_config)
    source_files = _source_files_report()
    if args.target == "formal_gate_c":
        result = run_gate_c(
            GateCConfig(),
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=gate_a._source_report,
            source_files=source_files,
            environment=environment,
        )
    elif args.target == "gate_c2_controls":
        result = run_gate_c2_controls(
            GateCConfig(),
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=gate_a._source_report,
            source_files=source_files,
            environment=environment,
        )
    elif args.target == "gate_c3_controls":
        c3_environment = {
            **dict(environment),
            "deterministic_environment": {
                name: os.environ.get(name)
                for name in GATE_C3_DETERMINISTIC_ENVIRONMENT
            },
        }
        result = run_gate_c3_controls(
            GateCConfig(),
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=gate_a._source_report,
            source_files=source_files,
            environment=c3_environment,
        )
    else:
        result = run_gate_c_initialization(
            GateCConfig(),
            prerequisites=prerequisites,
            source_start=source_start,
            source_end_reporter=gate_a._source_report,
            source_files=source_files,
            environment=environment,
        )
    destination = write_artifact(result, args.output)
    print(destination)
    print(json.dumps(result["qualification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
