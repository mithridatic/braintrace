"""21 - Protocol-v2 ARC with pp-prop and recurrent latent spiking effort.

The default protocol compares 0, 30, and 60 recurrent reasoning ticks with an
equal 30-row frozen-state decoder sweep at every effort. Exact global top-two
factorized candidates are submitted only from the final checkpoint. The
paper's private architecture, data, and training recipe are unavailable; this
is a repository-native interface and evaluation audit, not a reproduction or
paper-scale compute claim.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import importlib.metadata
import math
import os
import pathlib
import platform
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Literal

import brainstate
import braintools
import jax
import jax.numpy as jnp
import msgspec
import numpy as np
import optax

try:
    from examples.pp_prop.latent_workspace_adaptation import snapshot_parameters
    from examples.pp_prop.latent_workspace_analysis import (
        OutputLogits,
        adam_parameter_travel_budget,
        aggregate_arc_metrics,
        analyze_latent_trajectory,
        assess_model_only_completion,
        input_echo_fraction,
        compare_control_trajectories,
        decode_candidates,
        score_query_candidates,
        select_checkpoint_candidates,
    )
    from examples.pp_prop.latent_workspace_rules import (
        clear_rule_cache,
        verified_rule_candidates,
    )
    from examples.pp_prop.latent_workspace_arc_adaptation import (
        ArcTargetFreeTaskBank,
        ArcTaskBankAdaptationResult,
        build_arc_target_free_task_bank,
        compile_arc_task_local_adaptation_runner,
    )
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        arc_loss_per_example,
        compile_pp_prop,
        expand_decoder_logits,
        parameter_snapshot,
        refinement_head_width,
        refinement_parameter_paths,
        run_selected_packed_stream,
    )
    from examples.pp_prop.latent_workspace_protocol import (
        PROTOCOL_VERSION,
        StepGates,
        build_batched_protocol_v2_arm,
    )
    from examples.pp_prop.latent_workspace_refinement import (
        RowRefinementLayout,
        row_refinement_loss_per_example,
    )
    from examples.pp_prop.latent_workspace_resource_safety import (
        assess_gpu_runtime_safety,
        require_pre_device_gpu_environment,
        require_recurrent_edge_budget,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcPair,
        ArcTask,
        DatasetSource,
        EncodedQueryEpisode,
        LoadedDataset,
        RowEventConfig,
        _encode_arc_query_episodes_batched,
        assert_no_evaluation_leakage,
        associative_memory_feature_indices,
        augment_training_task,
        canonical_task_fingerprint,
        encode_arc_query_episode,
        encode_query_episode,
        learned_update_feature_indices,
        leave_one_demonstration_out_episodes,
        load_dataset_source,
        smoke_loaded_dataset,
    )
except ModuleNotFoundError:
    from latent_workspace_adaptation import snapshot_parameters
    from latent_workspace_analysis import (
        OutputLogits,
        adam_parameter_travel_budget,
        aggregate_arc_metrics,
        analyze_latent_trajectory,
        assess_model_only_completion,
        input_echo_fraction,
        compare_control_trajectories,
        decode_candidates,
        score_query_candidates,
        select_checkpoint_candidates,
    )
    from latent_workspace_rules import (
        clear_rule_cache,
        verified_rule_candidates,
    )
    from latent_workspace_arc_adaptation import (
        ArcTargetFreeTaskBank,
        ArcTaskBankAdaptationResult,
        build_arc_target_free_task_bank,
        compile_arc_task_local_adaptation_runner,
    )
    from latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        arc_loss_per_example,
        compile_pp_prop,
        expand_decoder_logits,
        parameter_snapshot,
        refinement_head_width,
        refinement_parameter_paths,
        run_selected_packed_stream,
    )
    from latent_workspace_protocol import (
        PROTOCOL_VERSION,
        StepGates,
        build_batched_protocol_v2_arm,
    )
    from latent_workspace_refinement import (
        RowRefinementLayout,
        row_refinement_loss_per_example,
    )
    from latent_workspace_resource_safety import (
        assess_gpu_runtime_safety,
        require_pre_device_gpu_environment,
        require_recurrent_edge_budget,
    )
    from latent_workspace_task import (
        ArcPair,
        ArcTask,
        DatasetSource,
        EncodedQueryEpisode,
        LoadedDataset,
        RowEventConfig,
        _encode_arc_query_episodes_batched,
        assert_no_evaluation_leakage,
        associative_memory_feature_indices,
        augment_training_task,
        canonical_task_fingerprint,
        encode_arc_query_episode,
        encode_query_episode,
        learned_update_feature_indices,
        leave_one_demonstration_out_episodes,
        load_dataset_source,
        smoke_loaded_dataset,
    )


DeviceName = Literal["cpu", "gpu"]
PrimaryCandidateMode = Literal["model_only", "rule_then_model"]
AdaptationSchedule = Literal["per_episode", "per_tick"]
DecoderMode = Literal["legacy_cp", "row_refinement", "latent_row_decode"]
SparseBackend = Literal["default", "jax_raw"]
MemoryCoding = Literal[
    "frozen",
    "learned_keys",
    "learned_write",
    "learned_update",
    "delta_write",
    "situ_glu_update",
]
MemoryReadTransform = Literal["linear", "gated", "gated_rms"]
LatentResidualMixer = Literal["none", "attention_residual"]
OptimizerName = Literal["adam", "adamw", "muon"]

LrScheduleName = Literal["constant", "cosine"]
EffortScheduleName = Literal["uniform", "progressive"]

FULL_SCALE_NEURON_COUNT = 4096

FULL_SCALE_RECURRENT_EDGES = 4_194_304

AUTO_TRAINING_CHUNK_LIMIT = 5

TraceEngine = Literal["pp_prop", "d_rtrl"]
RefinementMixer = Literal["linear", "carrier_gate", "attention_residual"]
NeuronTyping = Literal["none", "ei_dale"]
CHECKPOINT_INTERVAL = 30
CHECKPOINTS = (0, 30, 60)
TRAINING_EFFORTS = (30, 60)
SUBMISSION_CHECKPOINT = 60
SUBMISSION_POLICY = "latest_checkpoint_factorized_global_top2_v2"
RULE_SUBMISSION_POLICY = (
    "latest_checkpoint_demonstration_verified_rule_then_model_v1"
)
RULE_ARM_DEMONSTRATIONS = ("intact", "shuffled")
EVALUATION_ARM_ORDER = (
    "intact",
    "repeat_intact",
    "no_context",
    "shuffled_demonstrations",
    "state_hold",
    "recurrent_lesion",
    "slot_ablation",
)
STATE_RMS_TOLERANCE = 1e-6
APPROVED_TRAINING_SOURCES = frozenset(
    {
        "arc-agi-1 training",
        "re-arc",
        "conceptarc",
        "arc-heavy",
        "arc-gen100k",
    }
)
APPROVED_EVALUATION_SOURCES = frozenset({"arc-agi-1 evaluation", "arc-task-gen"})
CLAIM_BOUNDARY = (
    "Claim boundary: Example 21 instantiates the paper's public ARC task, "
    "ranked-candidate, and variable-effort contract with BrainPy LIF neurons, "
    "BrainTrace sparse synapses, and pp-prop. The paper's private data, model "
    "dimensions, internal update rules, and training recipe were unavailable. "
    "This is not a reproduction, makes no paper-score or inference-cost claim, "
    "and asserts no agreement between pp-prop and a BPTT gradient oracle."
)


def _progress_evidence(
    *,
    stage: str,
    completed: int,
    total: int,
    started_at: float,
    now: float | None = None,
) -> dict[str, object]:
    """Return one machine-readable progress observation."""
    observed_at = time.perf_counter() if now is None else now
    elapsed = max(0.0, observed_at - started_at)
    eta = None
    if 0 < completed < total:
        eta = elapsed * (total - completed) / completed
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
    }


def _emit_progress(
    stage: str, completed: int, total: int, started_at: float
) -> None:
    evidence = _progress_evidence(
        stage=stage,
        completed=completed,
        total=total,
        started_at=started_at,
    )
    print(
        f"[example21-progress] {msgspec.json.encode(evidence).decode()}",
        file=sys.stderr,
        flush=True,
    )


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _checkpoint_schedule(latent_steps: int) -> tuple[int, ...]:
    """Return complete 30-tick checkpoints through the configured horizon."""
    if latent_steps % CHECKPOINT_INTERVAL:
        raise ValueError("latent_steps must be divisible by 30")
    return tuple(range(0, latent_steps + 1, CHECKPOINT_INTERVAL))


@dataclass(frozen=True)
class ExperimentConfig:
    """Configure data, one-model training, frozen evaluation, and artifacts.

    Parameters
    ----------
    source_manifest : pathlib.Path or None
        JSON declaration of public local ARC sources. Full scientific runs
        require at least one training and one evaluation source.
    output_dir : pathlib.Path
        Directory for ``result.json``, text report, plot, and resolved manifest.
    device : {"cpu", "gpu"}
        Requested fail-closed JAX backend.
    seed : int
        BrainState random seed for parameters, topology, scheduling, and
        training-only augmentation.
    neuron_count, recurrent_edges : int
        Physical LIF population and exact directed sparse-edge count.
    readout_width, color_rank : int
        Shared readout bottleneck and CP rank of the full-grid color head.
    context_memory_width : int
        Associative workspace width. Zero selects the byte-compatible legacy
        reservoir; positive values up to 512 opt into ``S_K/H_r``.
    memory_decay : float
        Associative memory self-decay in the closed interval ``[0, 1]``.
    memory_read_transform : {"linear", "gated", "gated_rms"}
        Associative read projection. The default preserves the historical
        linear path; gated modes use the previous workspace as gate input.
    memory_read_interval : int
        Positive one-based cadence for latent associative reads. Query rows
        always read; the default ``1`` reads on every latent tick.
    latent_residual_mixer : {"none", "attention_residual"}
        Optional attention residual across latent reasoning blocks.
    latent_residual_block_size : int
        Positive latent ticks per residual summary block.
    memory_coding : {"frozen", "learned_keys", "learned_write", "learned_update", "delta_write", "situ_glu_update"}
        Storage-coding trainability. ``"frozen"`` keeps the fixed random
        Fourier keys and fixed value bases bit-exactly; ``"learned_keys"``
        makes the key projection a trainable ETP linear layer that learns
        through the retrieval path (the write-side key is gradient-detached);
        ``"learned_write"`` additionally routes the write itself through the
        fused ``braintrace.outer_write`` primitive, so the stored key and value
        codings carry gradient too.
    trace_engine : {"pp_prop", "d_rtrl"}
        Eligibility-trace engine. ``"pp_prop"`` keeps the IO-factorized
        coordinate; ``"d_rtrl"`` compiles the per-parameter exact-trace
        coordinate, which carries the memory write's pairing gradient exactly
        at a much higher memory cost.
    neuron_typing : {"none", "ei_dale"}
        Recurrent neuron-type structure. ``"none"`` keeps the untyped legacy
        substrate bit-exactly. ``"ei_dale"`` assigns a seeded binary E/I
        split and enforces Dale's law on recurrent weight signs at
        initialization and by projection after every optimizer step.
    excitatory_fraction : float
        Fraction of excitatory neurons under ``ei_dale``; rejected with
        ``neuron_typing="none"`` unless left at its 0.8 default.
    max_demonstrations, max_grid_size : int
        Static lossless ARC row-event capacities.
    latent_steps : int
        Maximum zero-input recurrent effort. Must be at least 60 so evaluation
        contains two complete 30-row answer sweeps.
    training_updates : int
        Number of pp-prop optimizer updates shared across effort lengths.
    training_chunk_size : int
        Number of updates staged on device at once. ``0`` selects the largest
        divisor of ``training_updates`` no greater than
        ``AUTO_TRAINING_CHUNK_LIMIT`` so the default run remains complete while
        bounding device staging. Any other value must divide
        ``training_updates`` so that every chunk compiles to the same scan
        length.
    runtime_profile : bool
        Emit diagnostic phase and pipeline timing. Profiling synchronizes
        device work for attribution and is therefore not a throughput mode.
    learning_rate, clip_norm : float
        Base optimizer rate and global gradient clipping norm.
    lr_schedule : {"constant", "cosine"}
        Shared-training learning-rate schedule. ``"constant"`` holds the base
        rate for every update; ``"cosine"`` decays it from the base rate to
        zero over ``training_updates`` optimizer updates (optax cosine decay
        with ``alpha=0``, no warmup). The per-tick adaptation path keeps its
        own constant ``adaptation_learning_rate`` either way.
    lr_warmup_fraction : float
        Fraction of shared-training updates used for linear warmup before
        cosine decay. Zero preserves the historical cosine schedule exactly.
        Nonzero warmup is incompatible with the constant-rate control.
    effort_schedule : {"uniform", "progressive"}
        Training-depth sampling schedule. Progressive introduces checkpoints
        in contiguous phases; uniform preserves the historical global shuffle.
    effort_distillation_weight : float
        Nonnegative deeper-effort self-distillation weight. The default zero
        preserves the supervised objective. Positive values remain unavailable
        until the preregistered R60 teacher gate passes.
    optimizer : {"adam", "adamw", "muon"}
        Shared-training optimizer. Muon applies to rank-two leaves and uses
        its built-in AdamW fallback for other leaves.
    weight_decay : float or None
        Decoupled weight decay for AdamW and Muon. ``None`` resolves to zero
        for Adam, 0.01 for AdamW, and 0.1 for Muon. Plain Adam applies no
        decay, so a nonzero explicit value with ``optimizer='adam'`` is
        rejected rather than silently ignored.
    copy_residual_gain : float
        Fixed identity-residual logit magnitude added to the answer row
        head's output at the query's own colour for every occupied column,
        so training learns deviations from copy instead of the copy map
        itself. Zero (the default) keeps the bare head bit-exactly.
    row_head_carrier_scale : float
        Constant multiplier on the carrier block of the answer row head's
        input only. Zero starves the row head of the task-identifying
        carrier so training cannot displace the copy path.
    row_copy_gate : bool
        Scale the row head's logits by a learned per-cell gate that leaves the
        copy residual ungated. Closed at initialisation, so the decoded row is
        the query row exactly and the model starts as a strict copy machine.
    row_copy_gate_bias : float
        Pre-sigmoid bias for that gate. More negative starts it further closed.
    row_head_modulation : {"none", "bilinear"}
        Multiplicative carrier-by-query-row term for the row head. The head is
        a bias-free linear map over ``concat(carrier, event blocks)``, so it can
        add a task carrier to a query row but cannot apply a carrier-selected
        transformation to it. ``"bilinear"`` adds that missing product term.
    row_head_modulation_rank : int
        Rank of the bilinear modulation.
    row_head_carrier_gate : bool
        Compatibility spelling for ``refinement_mixer="carrier_gate"``.
        The carrier ablation uses an event-only head plus a carrier head
        gated by a zero-initialised trainable 300-coordinate ``tanh`` vector, so the row
        answer starts carrier-free and training must buy carrier access.
        Incompatible with a non-default ``row_head_carrier_scale``.
    shape_head_carrier_scale : float
        Constant multiplier on the carrier block of the answer shape head's
        input only. Zero makes the shape answer a pure function of the row
        events, testing whether the shape head suffers the same carrier
        displacement as the row head.
    refinement_mixer : {"linear", "carrier_gate", "attention_residual"}
        Explicit refinement proposal mixer. Only ``"attention_residual"``
        implements the paper's learned source-axis softmax; the other choices
        are retained ablations.
    memory_value_softcap_beta : float
        Softcap magnitude ``beta`` of the memory value coding,
        ``softcap(x, beta) = beta * tanh(x / beta)``, bounding the stored
        value code to ``(-beta, beta)``. ``1.0`` reproduces the legacy
        ``tanh`` value map bit-exactly.
    reasoning_query_softcap_beta : float
        Softcap magnitude of the iterative reasoning query on latent steps.
        ``1.0`` reproduces the legacy ``tanh`` cap bit-exactly.
    balanced_color_loss : bool
        Whether each target color contributes equal total valid-cell weight.
        This option is valid only with the explicit legacy CP decoder.
    decoder_mode : {"legacy_cp", "row_refinement"}
        Explicit output representation. Production and CLI defaults use learned
        row-wise refinement; legacy CP remains available only when named.
    sparse_backend : {"default", "jax_raw"}
        CSR execution backend. ``"default"`` preserves the brainevent default;
        ``"jax_raw"`` is an explicit benchmarkable backend selection.
    adaptation_learning_rate : float
        Learning rate for the optional task-local adaptation diagnostic. It is
        ignored by the default shared-model evaluation.
    adaptation_epochs : int
        Number of epochs for the optional task-local adaptation diagnostic.
    task_local_adaptation : bool
        Enable the optional leave-one-demonstration-out adaptation diagnostic.
        The default keeps one trained parameter set frozen for evaluation.
    evaluation_controls : bool
        Enable the optional repeat, no-context, shuffled-demonstration, and
        slot-ablation evaluation arms. The default runs only the intact arm.
    adaptation_update_schedule : {"per_episode", "per_tick"}
        Whether a fold accumulates one gradient and takes one optimizer step,
        or applies the eligibility-trace term as an update at every supervised
        tick. ``per_tick`` is the online schedule and the default: it led on
        shape, pixel, exact answers, and per-task helped-versus-hurt, and
        trailed on none. The margin over a tuned ``per_episode`` arm is within
        noise at 86 tasks and it costs about 2.3x the wall clock, so the
        cheaper schedule remains selectable.
    adaptation_task_group : int
        Number of tasks per compiled adaptation dispatch when the optional
        diagnostic is enabled.
    ablation_slot : int
        Deterministic 64-neuron slot used by the frozen ablation control.
    evaluation_task_limit : int or None
        Development-only task cap. Any cap disqualifies a full scientific run.
    smoke : bool
        Whether results use embedded fixtures and are plumbing-only.
    structural_only : bool
        Instantiate and run without optimization; never scientific evidence.
    primary_candidate_mode : {"model_only", "rule_then_model"}
        Fail-closed primary ARC scoring mode. Only candidates decoded from the
        model may occupy submitted pass@2 slots.
    """

    source_manifest: pathlib.Path | None = None
    output_dir: pathlib.Path = pathlib.Path("var/example21")
    device: DeviceName = "gpu"
    seed: int = 9999
    neuron_count: int = FULL_SCALE_NEURON_COUNT
    recurrent_edges: int = FULL_SCALE_RECURRENT_EDGES
    readout_width: int = 128
    color_rank: int = 16
    context_memory_width: int = 32
    memory_decay: float = 1.0
    memory_read_transform: MemoryReadTransform = "linear"
    memory_read_interval: int = 1
    latent_residual_mixer: LatentResidualMixer = "none"
    latent_residual_block_size: int = 10
    memory_coding: MemoryCoding = "learned_update"
    trace_engine: TraceEngine = "pp_prop"
    neuron_typing: NeuronTyping = "none"
    excitatory_fraction: float = 0.8
    max_demonstrations: int = 10
    max_grid_size: int = 30
    latent_steps: int = 60
    submission_effort: int | None = None
    training_updates: int = 260
    training_chunk_size: int = 0
    training_batch_size: int = 32
    training_bank_size: int = 0
    training_workers: int = 8
    runtime_profile: bool = False
    learning_rate: float = 1e-3
    lr_schedule: LrScheduleName = "cosine"
    lr_warmup_fraction: float = 0.0
    effort_schedule: EffortScheduleName = "uniform"
    effort_distillation_weight: float = 0.0
    optimizer: OptimizerName = "muon"
    weight_decay: float | None = None
    adaptation_learning_rate: float = 5e-5
    adaptation_epochs: int = 1
    task_local_adaptation: bool = False
    evaluation_controls: bool = True
    clip_norm: float = 1.0
    copy_residual_gain: float = 0.0
    row_head_carrier_scale: float = 1.0
    row_head_carrier_gate: bool = False
    row_head_modulation: Literal["none", "bilinear"] = "none"
    row_head_modulation_rank: int = 64
    row_copy_gate: bool = False
    row_copy_gate_bias: float = -4.0
    shape_head_carrier_scale: float = 1.0
    refinement_mixer: RefinementMixer = "linear"
    memory_value_softcap_beta: float = 4.0
    reasoning_query_softcap_beta: float = 25.0
    balanced_color_loss: bool = False
    ablation_slot: int = 0
    adaptation_task_group: int = 20
    parameter_checkpoint: pathlib.Path | None = None
    adaptation_update_schedule: AdaptationSchedule = "per_tick"
    initial_checkpoint: pathlib.Path | None = None
    checkpoint_every: int = 0
    training_holdout_tasks: int = 0
    evaluation_task_limit: int | None = None
    smoke: bool = False
    structural_only: bool = False
    primary_candidate_mode: PrimaryCandidateMode = "model_only"
    decoder_mode: DecoderMode = "latent_row_decode"
    sparse_backend: SparseBackend = "default"

    def __post_init__(self) -> None:
        for name, minimum in (
            ("seed", 0),
            ("neuron_count", 64),
            ("recurrent_edges", 1),
            ("readout_width", 1),
            ("color_rank", 1),
            ("context_memory_width", 0),
            ("memory_read_interval", 1),
            ("latent_residual_block_size", 1),
            ("max_demonstrations", 1),
            ("max_grid_size", 1),
            ("latent_steps", 30),
            ("training_updates", 0),
            ("training_chunk_size", 0),
            ("training_batch_size", 1),
            ("training_bank_size", 0),
            ("training_workers", 1),
            ("adaptation_epochs", 1),
            ("adaptation_task_group", 0),
            ("checkpoint_every", 0),
            ("training_holdout_tasks", 0),
            ("ablation_slot", 0),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=minimum)
            )
        if self.submission_effort is not None:
            object.__setattr__(
                self,
                "submission_effort",
                _integer(self.submission_effort, "submission_effort", minimum=0),
            )
            if self.submission_effort not in self.checkpoints:
                raise ValueError(
                    "submission_effort must be one of the scored checkpoints"
                )
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.primary_candidate_mode not in ("model_only", "rule_then_model"):
            raise ValueError(
                "primary_candidate_mode must be 'model_only' or 'rule_then_model'"
            )
        if self.adaptation_update_schedule not in ("per_episode", "per_tick"):
            raise ValueError(
                "adaptation_update_schedule must be 'per_episode' or 'per_tick'"
            )
        if self.decoder_mode not in (
            "legacy_cp",
            "row_refinement",
            "latent_row_decode",
        ):
            raise ValueError(
                "decoder_mode must be 'legacy_cp', 'row_refinement' or "
                "'latent_row_decode'"
            )
        if self.sparse_backend not in ("default", "jax_raw"):
            raise ValueError("sparse_backend must be 'default' or 'jax_raw'")
        if self.optimizer not in ("adam", "adamw", "muon"):
            raise ValueError("optimizer must be 'adam', 'adamw', or 'muon'")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if self.effort_schedule not in ("uniform", "progressive"):
            raise ValueError("effort_schedule must be 'uniform' or 'progressive'")
        distillation_weight = _nonnegative_real(
            self.effort_distillation_weight, "effort_distillation_weight"
        )
        if distillation_weight > 0.0:
            raise ValueError(
                "effort_distillation_weight requires a qualified R60 teacher; "
                "the Stage 6 teacher gate has not been satisfied"
            )
        object.__setattr__(
            self, "effort_distillation_weight", distillation_weight
        )
        warmup = self.lr_warmup_fraction
        if isinstance(warmup, (bool, np.bool_)) or not isinstance(warmup, Real):
            raise ValueError("lr_warmup_fraction must be finite and in [0, 1)")
        warmup = float(warmup)
        if not math.isfinite(warmup) or not 0.0 <= warmup < 1.0:
            raise ValueError("lr_warmup_fraction must be finite and in [0, 1)")
        if self.lr_schedule == "constant" and warmup != 0.0:
            raise ValueError(
                "lr_warmup_fraction must be zero when lr_schedule='constant'"
            )
        object.__setattr__(self, "lr_warmup_fraction", warmup)
        if self.memory_coding not in (
            "frozen",
            "learned_keys",
            "learned_write",
            "learned_update",
            "delta_write",
            "situ_glu_update",
        ):
            raise ValueError(
                "memory_coding must be 'frozen', 'learned_keys', "
                "'learned_write', 'learned_update', 'delta_write' or "
                "'situ_glu_update'"
            )
        if not isinstance(self.memory_read_transform, str):
            raise TypeError(
                "memory_read_transform must be 'linear', 'gated' or 'gated_rms'"
            )
        if self.memory_read_transform not in ("linear", "gated", "gated_rms"):
            raise ValueError(
                "memory_read_transform must be 'linear', 'gated' or 'gated_rms'"
            )
        if self.memory_read_transform != "linear" and self.context_memory_width == 0:
            raise ValueError(
                "memory_read_transform requires a positive context_memory_width"
            )
        if self.latent_residual_mixer not in ("none", "attention_residual"):
            raise ValueError(
                "latent_residual_mixer must be 'none' or 'attention_residual'"
            )
        if (
            self.latent_residual_mixer == "attention_residual"
            and self.context_memory_width == 0
        ):
            raise ValueError(
                "latent_residual_mixer requires a positive context_memory_width"
            )
        if self.memory_coding != "frozen" and self.context_memory_width == 0:
            raise ValueError("memory_coding requires a positive context_memory_width")
        if self.trace_engine not in ("pp_prop", "d_rtrl"):
            raise ValueError("trace_engine must be 'pp_prop' or 'd_rtrl'")
        if self.refinement_mixer not in (
            "linear",
            "carrier_gate",
            "attention_residual",
        ):
            raise ValueError(
                "refinement_mixer must be 'linear', 'carrier_gate' or "
                "'attention_residual'"
            )
        if self.row_head_carrier_gate:
            if self.refinement_mixer == "attention_residual":
                raise ValueError(
                    "--row-head-carrier-gate conflicts with "
                    "--refinement-mixer attention_residual"
                )
            if self.refinement_mixer == "linear":
                object.__setattr__(self, "refinement_mixer", "carrier_gate")
        if self.refinement_mixer == "carrier_gate":
            object.__setattr__(self, "row_head_carrier_gate", True)
        if self.neuron_typing not in ("none", "ei_dale"):
            raise ValueError("neuron_typing must be 'none' or 'ei_dale'")
        if self.neuron_typing == "ei_dale" and self.task_local_adaptation:
            raise ValueError(
                "neuron_typing='ei_dale' does not support task_local_adaptation "
                "yet: the adaptation path takes optimizer steps without the "
                "Dale sign projection"
            )
        object.__setattr__(
            self,
            "excitatory_fraction",
            _unit_interval(self.excitatory_fraction, "excitatory_fraction"),
        )
        if self.neuron_typing == "none" and self.excitatory_fraction != 0.8:
            raise ValueError(
                "excitatory_fraction requires neuron_typing='ei_dale'"
            )
        if self.decoder_mode in ("row_refinement", "latent_row_decode") and self.context_memory_width == 0:
            raise ValueError(
                "decoder_mode='row_refinement' requires positive context_memory_width"
            )
        if self.decoder_mode in ("row_refinement", "latent_row_decode") and self.balanced_color_loss:
            raise ValueError(
                "balanced_color_loss is supported only by decoder_mode='legacy_cp'"
            )
        if (
            self.decoder_mode == "latent_row_decode"
            and self.refinement_mixer == "attention_residual"
        ):
            raise ValueError(
                "attention_residual bypasses the latent-binding test under "
                "latent_row_decode"
            )
        if self.neuron_count % 64:
            raise ValueError("neuron_count must be divisible by 64")
        if self.context_memory_width > 512:
            raise ValueError("context_memory_width must be at most 512")
        if self.recurrent_edges > self.neuron_count * (self.neuron_count - 1):
            raise ValueError("recurrent_edges exceeds directed no-self capacity")
        require_recurrent_edge_budget(self.neuron_count, self.recurrent_edges)
        if self.max_grid_size != 30:
            raise ValueError("max_grid_size must be 30 for standard ARC")
        _checkpoint_schedule(self.latent_steps)
        if self.ablation_slot >= self.neuron_count // 64:
            raise ValueError("ablation_slot exceeds the configured 64-neuron slots")
        if self.evaluation_task_limit is not None:
            object.__setattr__(
                self,
                "evaluation_task_limit",
                _integer(
                    self.evaluation_task_limit, "evaluation_task_limit", minimum=1
                ),
            )
        object.__setattr__(
            self, "learning_rate", _positive_real(self.learning_rate, "learning_rate")
        )
        decay = {"adam": 0.0, "adamw": 0.01, "muon": 0.1}[self.optimizer]
        if self.weight_decay is not None:
            decay = _nonnegative_real(self.weight_decay, "weight_decay")
        if self.optimizer == "adam" and decay:
            raise ValueError(
                "weight_decay must be zero for optimizer='adam': plain Adam "
                "applies no decoupled decay, so a nonzero value would be "
                "reported in the optimizer policy but never applied"
            )
        object.__setattr__(self, "weight_decay", decay)
        object.__setattr__(
            self, "clip_norm", _positive_real(self.clip_norm, "clip_norm")
        )
        for name in (
            "copy_residual_gain",
            "row_head_carrier_scale",
            "shape_head_carrier_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative real scalar")
            object.__setattr__(self, name, value)
        if self.refinement_mixer == "attention_residual" and (
            self.copy_residual_gain != 0.0
            or self.row_head_carrier_scale != 1.0
            or self.shape_head_carrier_scale != 1.0
        ):
            raise ValueError(
                "refinement_mixer='attention_residual' cannot be combined with "
                "copy_residual_gain or non-default row/shape carrier scales"
            )
        for name in ("memory_value_softcap_beta", "reasoning_query_softcap_beta"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        object.__setattr__(
            self, "memory_decay", _unit_interval(self.memory_decay, "memory_decay")
        )
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.source_manifest is not None:
            object.__setattr__(
                self, "source_manifest", pathlib.Path(self.source_manifest)
            )
        if self.parameter_checkpoint is not None:
            object.__setattr__(
                self, "parameter_checkpoint", pathlib.Path(self.parameter_checkpoint)
            )
        if self.initial_checkpoint is not None:
            object.__setattr__(
                self, "initial_checkpoint", pathlib.Path(self.initial_checkpoint)
            )
        if self.checkpoint_every and self.parameter_checkpoint is None:
            raise ValueError("checkpoint_every requires parameter_checkpoint")
        restored = (
            self.parameter_checkpoint is not None and self.parameter_checkpoint.exists()
        )
        if (
            not self.structural_only
            and not restored
            and self.training_updates < len(self.training_efforts)
        ):
            raise ValueError(
                "training_updates must cover every configured training effort "
                "including effort zero under protocol v2"
            )
        if (
            self.training_chunk_size
            and self.training_updates
            and self.training_updates % self.training_chunk_size
        ):
            raise ValueError("training_chunk_size must divide training_updates")
        if (
            self.training_bank_size
            and self.training_bank_size < self.training_batch_size
        ):
            raise ValueError("training_bank_size must cover one training batch")
        if self.adaptation_learning_rate:
            object.__setattr__(
                self,
                "adaptation_learning_rate",
                _positive_real(
                    self.adaptation_learning_rate, "adaptation_learning_rate"
                ),
            )

    @property
    def checkpoints(self) -> tuple[int, ...]:
        """Scored recurrent checkpoints from zero through ``latent_steps``."""
        return _checkpoint_schedule(self.latent_steps)

    @property
    def training_efforts(self) -> tuple[int, ...]:
        """Recurrent depths distributed across optimizer updates."""
        if self.decoder_mode == "latent_row_decode":
            return self.checkpoints
        return self.checkpoints[1:]

    @property
    def submission_checkpoint(self) -> int:
        """Primary model-only checkpoint at the configured latent horizon."""
        if self.submission_effort is not None:
            return self.submission_effort
        return self.checkpoints[-1]

    @classmethod
    def smoke_config(
        cls,
        *,
        output_dir: pathlib.Path = pathlib.Path("var/example21-smoke"),
        device: DeviceName = "cpu",
        seed: int = 9999,
        context_memory_width: int | None = None,
        memory_decay: float = 1.0,
        memory_read_transform: MemoryReadTransform = "linear",
        memory_read_interval: int = 1,
        latent_residual_mixer: LatentResidualMixer = "none",
        latent_residual_block_size: int = 10,
        memory_coding: MemoryCoding | None = None,
        trace_engine: TraceEngine = "pp_prop",
        neuron_typing: NeuronTyping = "none",
        excitatory_fraction: float = 0.8,
        optimizer: OptimizerName = "muon",
        weight_decay: float | None = None,
        refinement_mixer: RefinementMixer = "linear",
        lr_schedule: LrScheduleName = "cosine",
        lr_warmup_fraction: float = 0.0,
        effort_schedule: EffortScheduleName = "uniform",
        effort_distillation_weight: float = 0.0,
        balanced_color_loss: bool = False,
        decoder_mode: DecoderMode = "latent_row_decode",
        runtime_profile: bool = False,
        sparse_backend: SparseBackend = "default",
    ) -> ExperimentConfig:
        """Return a reduced complete-pipeline configuration.

        Parameters
        ----------
        output_dir : pathlib.Path
            Artifact directory for the smoke run.
        device : {"cpu", "gpu"}
            Requested JAX backend.
        seed : int
            Deterministic model, schedule, and augmentation seed.
        context_memory_width : int, optional
            Associative workspace width. Defaults to 2 for row refinement and
            zero for explicitly selected legacy CP mode.
        memory_decay : float
            Associative memory decay in ``[0, 1]``.
        memory_read_transform : {"linear", "gated", "gated_rms"}
            Associative read projection forwarded to the model.
        memory_read_interval : int
            Positive one-based latent associative-read cadence.
        latent_residual_mixer : {"none", "attention_residual"}
            Optional latent-depth residual mixer.
        latent_residual_block_size : int
            Positive latent ticks per summary block.
        memory_coding : {"frozen", "learned_keys", "learned_write", "learned_update", "delta_write", "situ_glu_update"}
            Storage-coding trainability forwarded to the model.
        trace_engine : {"pp_prop", "d_rtrl"}
            Eligibility-trace engine forwarded to the model.
        neuron_typing : {"none", "ei_dale"}
            Recurrent neuron-type structure forwarded to the model.
        excitatory_fraction : float
            Excitatory fraction under ``ei_dale``.
        optimizer : {"adam", "adamw", "muon"}
            Shared-training optimizer.
        weight_decay : float or None
            Explicit optimizer weight decay, or the optimizer-specific default.
            Nonzero values are rejected for plain Adam, which applies no decay.
        refinement_mixer : {"linear", "carrier_gate", "attention_residual"}
            Row-refinement proposal mixer.
        lr_schedule : {"constant", "cosine"}
            Shared-training learning-rate schedule forwarded unchanged.
        lr_warmup_fraction : float
            Leading fraction of updates used for linear cosine warmup.
        effort_schedule : {"uniform", "progressive"}
            Training-depth sampling schedule.
        effort_distillation_weight : float
            Deeper-effort self-distillation weight. Only zero is accepted
            while the Stage 6 teacher gate is closed.
        balanced_color_loss : bool
            Whether to balance valid-cell color loss by present target class.
        decoder_mode : {"legacy_cp", "row_refinement"}
            Explicit decoder selected by the bounded smoke run.
        runtime_profile : bool
            Emit diagnostic phase and pipeline timing.
        sparse_backend : {"default", "jax_raw"}
            CSR execution backend.

        Returns
        -------
        ExperimentConfig
            A 128-neuron, 1,024-edge, three-update plumbing-only run.
        """
        if context_memory_width is None:
            context_memory_width = (
                2 if decoder_mode in ("row_refinement", "latent_row_decode") else 0
            )
        if memory_coding is None:
            memory_coding = (
                "learned_update" if decoder_mode == "latent_row_decode" else "frozen"
            )
        return cls(
            output_dir=output_dir,
            device=device,
            seed=seed,
            neuron_count=128,
            recurrent_edges=1024,
            readout_width=32,
            color_rank=4,
            context_memory_width=context_memory_width,
            memory_decay=memory_decay,
            memory_read_transform=memory_read_transform,
            memory_read_interval=memory_read_interval,
            latent_residual_mixer=latent_residual_mixer,
            latent_residual_block_size=latent_residual_block_size,
            memory_coding=memory_coding,
            trace_engine=trace_engine,
            neuron_typing=neuron_typing,
            excitatory_fraction=excitatory_fraction,
            optimizer=optimizer,
            weight_decay=weight_decay,
            refinement_mixer=refinement_mixer,
            lr_schedule=lr_schedule,
            lr_warmup_fraction=lr_warmup_fraction,
            effort_schedule=effort_schedule,
            effort_distillation_weight=effort_distillation_weight,
            balanced_color_loss=balanced_color_loss,
            decoder_mode=decoder_mode,
            runtime_profile=runtime_profile,
            sparse_backend=sparse_backend,
            max_demonstrations=4,
            latent_steps=60,
            training_updates=3,
            training_batch_size=1,
            learning_rate=5e-4,
            smoke=True,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe configuration mapping."""
        result = asdict(self)
        result["source_manifest"] = (
            None if self.source_manifest is None else str(self.source_manifest)
        )
        result["output_dir"] = str(self.output_dir)
        result["parameter_checkpoint"] = (
            None
            if self.parameter_checkpoint is None
            else str(self.parameter_checkpoint)
        )
        result["initial_checkpoint"] = (
            None if self.initial_checkpoint is None else str(self.initial_checkpoint)
        )
        result["checkpoints"] = list(self.checkpoints)
        result["training_efforts"] = list(self.training_efforts)
        result["submission_checkpoint"] = self.submission_checkpoint
        result["primary_evaluation_mode"] = (
            "compiled_task_local_pp_prop_leave_one_out"
            if (
                self.task_local_adaptation
                and self.decoder_mode in ("row_refinement", "latent_row_decode")
                and not self.structural_only
            )
            else "shared_model_frozen"
        )
        result["recurrent_edge_budget"] = require_recurrent_edge_budget(
            self.neuron_count,
            self.recurrent_edges,
        ).to_dict()
        return result


@dataclass(frozen=True)
class _OriginTask:
    source_name: str
    role: str
    task: ArcTask


@dataclass(frozen=True)
class _ExperimentData:
    training: tuple[_OriginTask, ...]
    evaluation: tuple[_OriginTask, ...]
    loaded: tuple[LoadedDataset, ...]
    plumbing_only: bool


@dataclass(frozen=True)
class _TrainingTensors:
    events: np.ndarray
    advances: np.ndarray
    heights: np.ndarray
    widths: np.ndarray
    colors: np.ndarray
    masks: np.ndarray
    efforts: np.ndarray
    task_fingerprints: tuple[str, ...]
    base_task_fingerprints: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    held_out_demonstration_indices: tuple[int, ...] = ()


@dataclass
class _TrainingProfile:
    """Diagnostic timing accumulated at the producer/consumer boundary."""

    chunk_count: int = 0
    producer_encoding_seconds: float = 0.0
    consumer_wait_seconds: float = 0.0
    host_to_device_staging_seconds: float = 0.0
    first_call_compilation_seconds: float = 0.0
    first_call_device_compute_seconds: float = 0.0
    steady_state_device_compute_seconds: float = 0.0
    host_result_copy_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe timing evidence."""

        return {
            "chunk_count": self.chunk_count,
            "producer_encoding_seconds": self.producer_encoding_seconds,
            "consumer_wait_seconds": self.consumer_wait_seconds,
            "host_to_device_staging_seconds": self.host_to_device_staging_seconds,
            "first_call_compilation_seconds": self.first_call_compilation_seconds,
            "first_call_device_compute_seconds": self.first_call_device_compute_seconds,
            "steady_state_device_compute_seconds": self.steady_state_device_compute_seconds,
            "host_result_copy_seconds": self.host_result_copy_seconds,
        }


@dataclass(frozen=True)
class _EvaluationRecord:
    source_name: str
    task_key: str
    encoded: EncodedQueryEpisode
    query_input: np.ndarray


def _devices_for(platform: DeviceName) -> list[jax.Device]:
    return list(jax.devices(platform))


def _device_memory_stats(device: jax.Device) -> dict[str, int]:
    try:
        return {
            str(key): int(value)
            for key, value in (device.memory_stats() or {}).items()
            if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        }
    except (AttributeError, RuntimeError):
        return {}


def _nvidia_smi(
    arguments: Sequence[str],
) -> tuple[tuple[str, ...], str | None]:
    """Run one bounded ``nvidia-smi`` query without invoking a shell."""

    command = ["nvidia-smi", *arguments]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return (), f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return (), detail
    return tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ), None


def _mib_bytes(value: str) -> int:
    """Parse one integral MiB value from ``nvidia-smi`` output."""

    stripped = value.strip()
    if not stripped.isdigit():
        raise ValueError("nvidia-smi memory value must be an integer MiB count")
    return int(stripped) * 1024 * 1024


def _sample_nvidia_smi(
    *,
    device_index: int,
    process_id: int,
) -> dict[str, object]:
    """Sample physical capacity and current-process use for one NVIDIA GPU."""

    device_argument = f"--id={int(device_index)}"
    device_rows, device_error = _nvidia_smi(
        (
            device_argument,
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    process_rows, process_error = _nvidia_smi(
        (
            device_argument,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    errors = [error for error in (device_error, process_error) if error]
    physical_bytes: int | None = None
    current_device_bytes: int | None = None
    current_process_bytes: int | None = None
    try:
        if len(device_rows) != 1:
            raise ValueError("nvidia-smi returned no unique device memory row")
        device_fields = tuple(field.strip() for field in device_rows[0].split(","))
        if len(device_fields) != 2:
            raise ValueError("nvidia-smi returned malformed device memory evidence")
        current_device_bytes = _mib_bytes(device_fields[0])
        physical_bytes = _mib_bytes(device_fields[1])
        matching_bytes = 0
        process_observed = False
        process_unavailable = False
        for row in process_rows:
            fields = tuple(field.strip() for field in row.split(","))
            if len(fields) != 2 or not fields[0].isdigit():
                raise ValueError("nvidia-smi returned malformed process evidence")
            if int(fields[0]) == int(process_id):
                if fields[1].casefold() in {"n/a", "[n/a]"}:
                    process_unavailable = True
                else:
                    matching_bytes += _mib_bytes(fields[1])
                    process_observed = True
        if process_observed:
            current_process_bytes = matching_bytes
        elif process_unavailable:
            current_process_bytes = None
        else:
            current_process_bytes = 0
    except ValueError as error:
        errors.append(str(error))
    return {
        "physical_device_bytes": physical_bytes,
        "current_device_bytes": current_device_bytes,
        "current_process_bytes": current_process_bytes,
        "error": "; ".join(errors) if errors else None,
    }


class _NvidiaSmiGpuMonitor:
    """Sample current-process NVIDIA memory on a background monitoring thread."""

    def __init__(
        self,
        *,
        device_index: int,
        process_id: int | None = None,
        interval_seconds: float = 0.25,
    ) -> None:
        self.device_index = int(device_index)
        self.process_id = os.getpid() if process_id is None else int(process_id)
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._physical_device_bytes: int | None = None
        self._peak_device_bytes: int | None = None
        self._peak_process_bytes: int | None = None
        self._sample_count = 0
        self._errors: list[str] = []

    def _record(self, sample: Mapping[str, object]) -> None:
        """Merge one sampler result into the peak report."""

        physical = sample.get("physical_device_bytes")
        current_device = sample.get("current_device_bytes")
        current = sample.get("current_process_bytes")
        error = sample.get("error")
        with self._lock:
            self._sample_count += 1
            if isinstance(physical, Integral) and not isinstance(physical, bool):
                physical_value = int(physical)
                if self._physical_device_bytes is None:
                    self._physical_device_bytes = physical_value
                elif self._physical_device_bytes != physical_value:
                    message = "nvidia-smi physical capacity changed during the run"
                    if message not in self._errors:
                        self._errors.append(message)
            if (
                isinstance(current_device, Integral)
                and not isinstance(current_device, bool)
                and int(current_device) > 0
            ):
                device_value = int(current_device)
                self._peak_device_bytes = max(
                    self._peak_device_bytes or device_value,
                    device_value,
                )
            if (
                isinstance(current, Integral)
                and not isinstance(current, bool)
                and int(current) > 0
            ):
                current_value = int(current)
                self._peak_process_bytes = max(
                    self._peak_process_bytes or current_value,
                    current_value,
                )
            if isinstance(error, str) and error and error not in self._errors:
                self._errors.append(error)

    def _sample_once(self) -> None:
        self._record(
            _sample_nvidia_smi(
                device_index=self.device_index,
                process_id=self.process_id,
            )
        )

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def start(self) -> None:
        """Take an initial sample and start periodic background sampling."""

        if self._thread is not None:
            raise RuntimeError("GPU monitor is already started")
        self._sample_once()
        self._thread = threading.Thread(
            target=self._monitor,
            name="example21-nvidia-smi-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        """Stop monitoring, take a final sample, and return peak evidence."""

        thread = self._thread
        if thread is None:
            raise RuntimeError("GPU monitor was not started")
        self._stop_event.set()
        thread.join(timeout=5.0)
        if thread.is_alive():
            with self._lock:
                self._errors.append("nvidia-smi monitor thread did not stop")
        self._sample_once()
        self._thread = None
        return self.report()

    def report(self) -> dict[str, object]:
        """Return the accumulated JSON-safe monitor evidence."""

        with self._lock:
            return {
                "sampler": "nvidia-smi",
                "device_index": self.device_index,
                "process_id": self.process_id,
                "sample_count": self._sample_count,
                "physical_device_bytes": self._physical_device_bytes,
                "peak_device_bytes": self._peak_device_bytes,
                "peak_process_bytes": self._peak_process_bytes,
                "evidence_complete": bool(
                    self._physical_device_bytes is not None
                    and self._peak_device_bytes is not None
                ),
                "errors": list(self._errors),
            }


def _make_gpu_monitor(device: jax.Device) -> _NvidiaSmiGpuMonitor:
    """Construct the runtime monitor for a resolved JAX GPU device."""

    return _NvidiaSmiGpuMonitor(device_index=int(device.id))


def _gpu_runtime_safety_report(
    config: ExperimentConfig,
    environment: Mapping[str, str],
    memory_stats: Mapping[str, object],
    monitor_report: Mapping[str, object],
) -> dict[str, object]:
    """Normalize JAX allocator and NVIDIA process evidence through policy."""

    device_peak = monitor_report.get("peak_device_bytes")
    process_peak = monitor_report.get("peak_process_bytes")
    sampled_peaks = [
        int(value)
        for value in (device_peak, process_peak)
        if isinstance(value, Integral)
        and not isinstance(value, bool)
        and int(value) > 0
    ]
    conservative_peak = max(sampled_peaks) if sampled_peaks else None
    if not (
        isinstance(device_peak, Integral)
        and not isinstance(device_peak, bool)
        and int(device_peak) > 0
    ):
        conservative_peak = None
    assessment = assess_gpu_runtime_safety(
        run_scope="smoke" if config.smoke else "full",
        environment=environment,
        allocator_peak_bytes=memory_stats.get("peak_bytes_in_use"),
        allocator_limit_bytes=memory_stats.get("bytes_limit"),
        physical_device_bytes=monitor_report.get("physical_device_bytes"),
        process_peak_bytes=conservative_peak,
    )
    return {
        "applicable": True,
        **assessment.to_dict(),
        "nvidia_smi_peak_device_bytes": device_peak,
        "nvidia_smi_peak_process_bytes": process_peak,
        "nvidia_smi_conservative_peak_bytes": conservative_peak,
        "nvidia_smi_transient_errors": list(monitor_report.get("errors", ())),
    }


def _resolve_device(platform: DeviceName) -> tuple[jax.Device, dict[str, object]]:
    try:
        devices = _devices_for(platform)
    except RuntimeError as error:
        devices = []
        detail = f": {error}"
    else:
        detail = ""
    if not devices:
        raise RuntimeError(
            f"requested JAX {platform} backend is unavailable{detail}; "
            "choose --device cpu explicitly only for a reduced run"
        )
    device = devices[0]
    return device, {
        "requested": platform,
        "platform": str(device.platform),
        "id": int(device.id),
        "kind": str(getattr(device, "device_kind", device)),
        "memory_stats": _device_memory_stats(device),
    }


def _source_declarations(path: pathlib.Path) -> tuple[DatasetSource, ...]:
    try:
        payload = msgspec.json.decode(path.read_bytes())
    except (OSError, msgspec.DecodeError) as error:
        raise ValueError(f"cannot read source manifest {path}: {error}") from error
    values = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("source manifest must contain a nonempty 'sources' list")
    declarations: list[DatasetSource] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"sources[{index}] must be an object")
        required = {"name", "role", "version", "path", "license_reference"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"sources[{index}] is missing {missing}")
        source_path = pathlib.Path(str(value["path"]))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        declarations.append(
            DatasetSource(
                name=value["name"],
                role=value["role"],
                version=value["version"],
                path=str(source_path),
                license_reference=value["license_reference"],
                format=value.get("format", "auto"),
                exclude_fingerprints=tuple(value.get("exclude_fingerprints", ())),
            )
        )
    return tuple(declarations)


def _load_data(config: ExperimentConfig) -> _ExperimentData:
    if config.smoke or (config.structural_only and config.source_manifest is None):
        fixture = smoke_loaded_dataset()
        origins = tuple(
            _OriginTask(fixture.manifest.source.name, "fixture", task)
            for task in fixture.tasks
        )
        return _ExperimentData(origins, origins, (fixture,), True)
    source_manifest = config.source_manifest
    if source_manifest is None:
        declared_manifest = os.environ.get("EXAMPLE21_SOURCE_MANIFEST", "").strip()
        if declared_manifest:
            source_manifest = pathlib.Path(declared_manifest)
    if source_manifest is None:
        raise ValueError("full runs require --source-manifest")
    declarations = _source_declarations(source_manifest)
    loaded = tuple(load_dataset_source(source) for source in declarations)
    assert_no_evaluation_leakage(item.manifest for item in loaded)
    training = tuple(
        _OriginTask(item.manifest.source.name, item.manifest.source.role, task)
        for item in loaded
        if item.manifest.source.role == "train"
        for task in item.tasks
    )
    evaluation = tuple(
        _OriginTask(item.manifest.source.name, item.manifest.source.role, task)
        for item in loaded
        if item.manifest.source.role == "evaluation"
        for task in item.tasks
    )
    if not config.structural_only and not training:
        raise ValueError("scientific training requires at least one train-role source")
    if not evaluation:
        raise ValueError("evaluation requires at least one evaluation-role source")
    return _ExperimentData(training, evaluation, loaded, False)


def _row_config(config: ExperimentConfig) -> RowEventConfig:
    return RowEventConfig(
        max_demonstrations=config.max_demonstrations,
        max_grid_size=config.max_grid_size,
    )


def _packed_events(
    encoded: EncodedQueryEpisode, config: ExperimentConfig
) -> np.ndarray:
    total = (
        encoded.events.shape[0]
        + config.latent_steps
        + (
            CHECKPOINT_INTERVAL
            if config.decoder_mode == "latent_row_decode"
            else 0
        )
    )
    result = np.zeros((total, encoded.events.shape[1]), dtype=np.float32)
    result[: encoded.events.shape[0]] = encoded.events
    return result


def _demonstration_advance_width(
    encoded: EncodedQueryEpisode, row_config: RowEventConfig
) -> int:
    """Return the advancing row count shared by every demonstration block.

    Demonstration blocks are a fixed ``max_grid_size`` rows wide whatever the
    grid heights are, so advancing a whole block spends the unused rows as
    all-zero membrane-leak steps.  This is the per-episode maximum occupied
    height instead: it never drops an encoded row, and because
    :func:`_derange_task` rotates the outputs it is identical for the intact and
    deranged encodings, which keeps the ``shuffled_demonstrations`` control on a
    byte-identical schedule.
    """
    valid = encoded.events[:, row_config.valid_slice.start] > 0.0
    return max(
        (int(valid[start:stop].sum()) for start, stop in encoded.demonstration_spans),
        default=0,
    )


def _packed_advances(
    encoded: EncodedQueryEpisode,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> np.ndarray:
    """Build a matched context/padding/latent state-advance schedule."""
    total = encoded.events.shape[0] + config.latent_steps
    advances = np.zeros((total,), dtype=np.bool_)
    width = _demonstration_advance_width(encoded, row_config)
    for start, _stop in encoded.demonstration_spans:
        advances[start : start + width] = True
    advances[encoded.query_start : encoded.query_stop] = True
    advances[encoded.query_stop : encoded.query_stop + config.latent_steps] = True
    return advances


def _effort_schedule(
    updates: int,
    rng: brainstate.random.RandomState,
    efforts: Sequence[int] = TRAINING_EFFORTS,
    schedule: EffortScheduleName = "uniform",
) -> np.ndarray:
    effort_values = np.asarray(efforts, dtype=np.int32)
    if schedule == "uniform":
        base = np.resize(effort_values, updates)
        order = np.asarray(rng.permutation(updates), dtype=np.int32)
        return base[order]
    if schedule != "progressive":
        raise ValueError("schedule must be 'uniform' or 'progressive'")
    if updates == 0:
        return np.zeros((0,), dtype=np.int32)
    if effort_values.size == 0:
        raise ValueError("efforts must contain at least one checkpoint")
    phase_sizes = np.full(effort_values.size, updates // effort_values.size)
    phase_sizes[: updates % effort_values.size] += 1
    phases: list[np.ndarray] = []
    for phase_index, phase_size in enumerate(phase_sizes):
        if phase_size == 0:
            continue
        introduced = effort_values[: phase_index + 1]
        phase = np.resize(introduced, int(phase_size))
        order = np.asarray(rng.permutation(int(phase_size)), dtype=np.int32)
        phases.append(phase[order])
    return np.concatenate(phases).astype(np.int32, copy=False)


def _empty_training_tensors() -> _TrainingTensors:
    empty = np.zeros((0,), dtype=np.float32)
    return _TrainingTensors(empty, empty, empty, empty, empty, empty, empty, ())


def _training_sequence_length(data: _ExperimentData, config: ExperimentConfig) -> int:
    """Return the smallest safe static training horizon for admitted data.

    Parameters
    ----------
    data
        Admitted training and evaluation data.
    config
        Resolved experiment configuration.

    Returns
    -------
    int
        Maximum semantic advance count over every fold and orientation.
    """
    if not data.training:
        raise ValueError("training data is empty")
    maximum = 0
    for origin in data.training:
        demonstrations = origin.task.train
        if not demonstrations:
            raise ValueError("training task has no demonstrations")
        orientations = (False,) if data.plumbing_only else (False, True)
        for transposed in orientations:
            for held_out_index, held_out in enumerate(demonstrations):

                def extent(grid: Any) -> int:
                    return int(grid.width if transposed else grid.height)

                context_width = max(
                    (
                        max(extent(pair.input), extent(pair.output))
                        for index, pair in enumerate(demonstrations)
                        if index != held_out_index and pair.output is not None
                    ),
                    default=0,
                )
                maximum = max(
                    maximum,
                    (len(demonstrations) - 1) * context_width
                    + extent(held_out.input)
                    + config.latent_steps
                    + (
                        CHECKPOINT_INTERVAL
                        if config.decoder_mode == "latent_row_decode"
                        else 0
                    ),
                )
    if maximum <= config.latent_steps:
        raise ValueError("training horizon contains no observed query rows")
    return maximum


def _compact_training_stream(
    encoded: EncodedQueryEpisode,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    sequence_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Move every semantic training advance into one static-shape prefix.

    Returns the compact event tensor, its prefix-only advance schedule, and the
    compact index of the query-terminal checkpoint.  The gathered rows retain
    the order and physical semantics of :func:`_packed_advances`; only frozen
    layout positions move behind the final latent tick, where no later loss can
    consume their eligibility trace.
    """
    padded = _packed_events(encoded, config)
    padded_advances = _packed_advances(encoded, config, row_config)
    active_indices = np.flatnonzero(padded_advances)
    query_terminal = encoded.query_stop - 1
    compact_query = np.flatnonzero(active_indices == query_terminal)
    if compact_query.size != 1:
        raise ValueError("query terminal must be one semantic training advance")

    required_length = int(active_indices.size)
    if sequence_length is None:
        sequence_length = int(padded.shape[0])
    elif (
        isinstance(sequence_length, (bool, np.bool_))
        or not isinstance(sequence_length, Integral)
        or int(sequence_length) < required_length
    ):
        raise ValueError(
            "training sequence length is shorter than the semantic advance prefix"
        )
    sequence_length = int(sequence_length)
    if sequence_length > int(padded.shape[0]):
        raise ValueError("training sequence length exceeds packed event capacity")
    compact = np.zeros((sequence_length, padded.shape[1]), dtype=padded.dtype)
    compact[: active_indices.size] = padded[active_indices]
    advances = np.zeros((sequence_length,), dtype=padded_advances.dtype)
    advances[: active_indices.size] = True
    query_checkpoint = int(compact_query[0])
    latent_count = int(active_indices.size) - query_checkpoint - 1
    if latent_count != config.latent_steps:
        raise ValueError("compact training prefix has the wrong latent length")
    return compact, advances, query_checkpoint


def _without_official_test_targets(task: ArcTask) -> ArcTask:
    """Return a task whose official queries cannot leak labels into training."""

    return ArcTask(
        train=task.train,
        test=tuple(ArcPair(pair.input, None) for pair in task.test),
        task_id=task.task_id,
    )


@lru_cache(maxsize=4096)
def _cached_base_training_task(task: ArcTask) -> tuple[ArcTask, str]:
    """Return the target-free task and stable fingerprint for one source task."""

    base_task = _without_official_test_targets(task)
    return base_task, canonical_task_fingerprint(base_task)


def _protocol_v2_training_schedule(
    sequence: np.ndarray,
    advances: np.ndarray,
    query_checkpoint: int,
    effort: int,
    row_config: RowEventConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align one episode's recurrent effort and fixed decoder sweep.

    Decoder windows occupy the final 30 static ticks for every episode in a
    batch. Padding between context and reasoning advances nothing, so differing
    context lengths cannot change per-episode objective weight.
    """

    if effort not in CHECKPOINTS:
        raise ValueError("protocol-v2 effort must be one of 0, 30, or 60")
    context_stop = query_checkpoint + 1
    decoder_start = sequence.shape[0] - CHECKPOINT_INTERVAL
    reasoning_start = decoder_start - effort
    if reasoning_start < context_stop:
        raise ValueError("protocol-v2 training sequence is too short")
    packed = np.zeros_like(sequence)
    packed[:context_stop] = sequence[:context_stop]
    physical = np.zeros_like(advances)
    physical[:context_stop] = advances[:context_stop]
    physical[reasoning_start:decoder_start] = True
    packed[decoder_start:, row_config.phase_slice.start + 1] = 1.0
    mask = np.zeros((sequence.shape[0],), dtype=np.float32)
    mask[decoder_start:] = np.float32(1.0 / CHECKPOINT_INTERVAL)
    return packed, physical, mask


def _training_row(
    origin: _OriginTask,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    rng: brainstate.random.RandomState,
    *,
    effort: int,
    plumbing_only: bool,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Encode one augmented leave-one-demonstration-out training update."""

    base_task, base_fingerprint = _cached_base_training_task(origin.task)
    task = (
        base_task
        if plumbing_only
        else augment_training_task(base_task, rng, role="train")
    )
    episodes = leave_one_demonstration_out_episodes(task)
    held_out_index = int(np.asarray(rng.randint(0, len(episodes))))
    encoded = encode_arc_query_episode(episodes[held_out_index], row_config)
    if encoded.target is None:
        raise ValueError(
            f"training fold {task.task_id or encoded.task_fingerprint}:"
            f"{held_out_index} lacks a target"
        )
    sequence, advances, query_checkpoint = _compact_training_stream(
        encoded, config, row_config, sequence_length=sequence_length
    )
    mask = np.zeros((sequence.shape[0],), dtype=np.float32)
    if config.decoder_mode == "latent_row_decode":
        sequence, advances, mask = _protocol_v2_training_schedule(
            sequence, advances, query_checkpoint, effort, row_config
        )
    else:
        terminal = query_checkpoint + effort
        if effort > config.latent_steps or terminal >= int(np.count_nonzero(advances)):
            raise ValueError("terminal effort exceeds packed sequence capacity")
    if config.decoder_mode == "row_refinement":
        mask[query_checkpoint + 1 : terminal + 1] = np.float32(1.0 / effort)
    elif config.decoder_mode == "legacy_cp":
        depth_count = effort + 1
        mask[query_checkpoint : terminal + 1] = np.float32(1.0 / depth_count)
    target = encoded.target
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[: target.height, : target.width] = target.as_array()
    return {
        "events": sequence[:, None, :],
        "advances": advances[:, None],
        "heights": target.height - 1,
        "widths": target.width - 1,
        "colors": padded[None],
        "masks": mask,
        "task_fingerprints": canonical_task_fingerprint(task),
        "base_task_fingerprints": base_fingerprint,
        "source_names": origin.source_name,
        "held_out_demonstration_index": held_out_index,
    }


def _training_row_from_encoded(
    descriptor: Mapping[str, Any],
    encoded: EncodedQueryEpisode,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> dict[str, Any]:
    """Build one training row from a pre-encoded batched episode."""

    sequence, advances, query_checkpoint = _compact_training_stream(
        encoded,
        config,
        row_config,
        sequence_length=int(descriptor["sequence_length"]),
    )
    effort = int(descriptor["effort"])
    mask = np.zeros((sequence.shape[0],), dtype=np.float32)
    if config.decoder_mode == "latent_row_decode":
        sequence, advances, mask = _protocol_v2_training_schedule(
            sequence, advances, query_checkpoint, effort, row_config
        )
    else:
        terminal = query_checkpoint + effort
        if effort > config.latent_steps or terminal >= int(np.count_nonzero(advances)):
            raise ValueError("terminal effort exceeds packed sequence capacity")
    if config.decoder_mode == "row_refinement":
        mask[query_checkpoint + 1 : terminal + 1] = np.float32(1.0 / effort)
    elif config.decoder_mode == "legacy_cp":
        depth_count = effort + 1
        mask[query_checkpoint : terminal + 1] = np.float32(1.0 / depth_count)
    target = encoded.target
    if target is None:
        raise ValueError("batched training episode lacks a target")
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[: target.height, : target.width] = target.as_array()
    return {
        "events": sequence[:, None, :],
        "advances": advances[:, None],
        "heights": target.height - 1,
        "widths": target.width - 1,
        "colors": padded[None],
        "masks": mask,
        "task_fingerprints": descriptor["task_fingerprint"],
        "base_task_fingerprints": descriptor["base_task_fingerprint"],
        "source_names": descriptor["source_name"],
        "held_out_demonstration_index": int(
            descriptor["held_out_demonstration_index"]
        ),
    }


def _merge_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine independent episodes into one batched optimizer update.

    Episodes carry different demonstration counts, so their latent windows sit
    at different physical ticks. The batched tick mask therefore selects every
    tick on which at least one episode is latent, and the step function weights
    each episode by its own advance gate. A single-episode batch reproduces the
    unbatched tensors exactly.
    """
    masks = np.stack([row["masks"] for row in rows])
    active = np.any(masks > 0.0, axis=0)
    mask = np.zeros_like(rows[0]["masks"])
    mask[active] = np.float32(1.0 / np.count_nonzero(active))
    return {
        "events": np.concatenate([row["events"] for row in rows], axis=1),
        "advances": np.concatenate([row["advances"] for row in rows], axis=1),
        "colors": np.concatenate([row["colors"] for row in rows], axis=0),
        "heights": np.asarray([row["heights"] for row in rows], dtype=np.int32),
        "widths": np.asarray([row["widths"] for row in rows], dtype=np.int32),
        "masks": mask,
        "task_fingerprints": tuple(row["task_fingerprints"] for row in rows),
        "base_task_fingerprints": tuple(row["base_task_fingerprints"] for row in rows),
        "source_names": tuple(row["source_names"] for row in rows),
        "held_out_demonstration_indices": tuple(
            row["held_out_demonstration_index"] for row in rows
        ),
    }


def _stacked_chunk(rows: list[dict[str, Any]], efforts: np.ndarray) -> _TrainingTensors:
    def column(name: str) -> list[Any]:
        return [row[name] for row in rows]

    def flattened(name: str) -> tuple[Any, ...]:
        return tuple(value for row in rows for value in row[name])

    return _TrainingTensors(
        events=np.stack(column("events")),
        advances=np.stack(column("advances")),
        heights=np.stack(column("heights")),
        widths=np.stack(column("widths")),
        colors=np.stack(column("colors")),
        masks=np.stack(column("masks")),
        efforts=efforts,
        task_fingerprints=flattened("task_fingerprints"),
        base_task_fingerprints=flattened("base_task_fingerprints"),
        source_names=flattened("source_names"),
        held_out_demonstration_indices=flattened("held_out_demonstration_indices"),
    )


def _training_bank(
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    rng: brainstate.random.RandomState,
    *,
    sequence_length: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Encode a reusable episode bank, one list per supervised effort.

    Encoding one fresh fold per episode slot dominates wall clock once updates
    are batched. A bank of ``training_bank_size`` episodes per effort is encoded
    once and drawn from with replacement instead. Size zero keeps the original
    behaviour of encoding every episode slot independently.
    """
    if not config.training_bank_size:
        return {}
    return {
        int(effort): [
            _training_row(
                data.training[int(index)],
                config,
                row_config,
                rng,
                effort=int(effort),
                plumbing_only=data.plumbing_only,
                sequence_length=sequence_length,
            )
            for index in rng.randint(
                0, len(data.training), size=config.training_bank_size
            )
        ]
        for effort in config.training_efforts
    }


def _banked_training_row(
    bank: dict[int, list[dict[str, Any]]],
    origin: _OriginTask,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    rng: brainstate.random.RandomState,
    *,
    effort: int,
    plumbing_only: bool,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Draw one episode from the bank, or encode a fresh one when it is empty."""
    if not bank:
        return _training_row(
            origin,
            config,
            row_config,
            rng,
            effort=effort,
            plumbing_only=plumbing_only,
            sequence_length=sequence_length,
        )
    episodes = bank[int(effort)]
    return episodes[int(np.asarray(rng.randint(0, len(episodes))))]


@dataclass(frozen=True)
class _TrainingRowJob:
    """Immutable work descriptor for one independently encoded episode."""

    ordinal: int
    origin: _OriginTask
    config: ExperimentConfig
    row_config: RowEventConfig
    seed: int
    effort: int
    plumbing_only: bool
    sequence_length: int


def _materialize_training_episode(job: _TrainingRowJob) -> dict[str, Any]:
    """Prepare one deterministic episode descriptor for batched encoding."""

    base_task, base_fingerprint = _cached_base_training_task(job.origin.task)
    rng = brainstate.random.RandomState(job.seed)
    task = (
        base_task
        if job.plumbing_only
        else augment_training_task(base_task, rng, role="train")
    )
    episodes = leave_one_demonstration_out_episodes(task)
    held_out_index = int(np.asarray(rng.randint(0, len(episodes))))
    return {
        "episode": episodes[held_out_index],
        "task_fingerprint": canonical_task_fingerprint(task),
        "base_task_fingerprint": base_fingerprint,
        "source_name": job.origin.source_name,
        "held_out_demonstration_index": held_out_index,
        "effort": job.effort,
        "config": job.config,
        "row_config": job.row_config,
        "sequence_length": job.sequence_length,
    }


def _ordered_training_episodes(
    jobs: Iterable[_TrainingRowJob], workers: int
) -> Iterator[dict[str, Any]]:
    """Materialize deterministic episode descriptors in ordinal order."""

    if workers == 1:
        for job in jobs:
            yield _materialize_training_episode(job)
        return
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="example21-training-row"
    )
    pending: dict[int, concurrent.futures.Future[dict[str, Any]]] = {}
    source = iter(jobs)
    next_ordinal = 0

    def fill() -> None:
        while len(pending) < 2 * workers:
            try:
                job = next(source)
            except StopIteration:
                return
            pending[job.ordinal] = executor.submit(_materialize_training_episode, job)

    try:
        fill()
        while pending:
            future = pending.pop(next_ordinal)
            yield future.result()
            next_ordinal += 1
            fill()
    except BaseException:
        for future in pending.values():
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _training_pool(
    data: _ExperimentData, config: ExperimentConfig
) -> Sequence[Any]:
    """Return the tasks admitted to optimization, less any reserved holdout.

    Reserving the tail of the training split gives an episode-scaling curve a
    measurement surface that the shared training stream never saw, so checkpoints
    can be compared without spending — or selecting on — the ARC evaluation split.
    """
    if not data.training:
        raise ValueError("training data is empty")
    reserved = config.training_holdout_tasks
    if reserved >= len(data.training):
        raise ValueError("training_holdout_tasks leaves no task to train on")
    return data.training[: len(data.training) - reserved] if reserved else data.training


def _resolved_training_chunk_size(config: ExperimentConfig) -> int:
    """Resolve the device staging size without dropping a partial schedule.

    Parameters
    ----------
    config
        Experiment configuration whose explicit chunk size may be zero for
        automatic bounded staging.

    Returns
    -------
    int
        A positive divisor of ``training_updates``. Zero-update structural
        runs return one because they still emit an empty training chunk.
    """
    updates = int(config.training_updates)
    if updates <= 0:
        return 1
    if config.training_chunk_size:
        return int(config.training_chunk_size)
    for candidate in range(min(AUTO_TRAINING_CHUNK_LIMIT, updates), 0, -1):
        if updates % candidate == 0:
            return candidate
    return 1


def _training_chunks(
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> Iterator[_TrainingTensors]:
    """Yield the training schedule in fixed-size, device-sized chunks.

    The effort and task draws stay up front at full ``training_updates`` size
    and the per-update walk visits updates in schedule order, so the random
    stream — and therefore every produced tensor — is independent of how the
    schedule is chunked.
    """
    if config.structural_only:
        yield _empty_training_tensors()
        return
    pool = _training_pool(data, config)
    rng = brainstate.random.RandomState(config.seed + 1000)
    efforts = _effort_schedule(
        config.training_updates,
        rng,
        config.training_efforts,
        config.effort_schedule,
    )
    batch = config.training_batch_size
    task_indices = np.asarray(
        rng.randint(0, len(pool), size=config.training_updates * batch),
        dtype=np.int32,
    )
    sequence_length = _training_sequence_length(data, config)
    bank = _training_bank(
        data,
        config,
        row_config,
        rng,
        sequence_length=sequence_length,
    )
    size = _resolved_training_chunk_size(config)

    rows: list[dict[str, Any]] = []
    for update_index, effort in enumerate(efforts):
        picks = task_indices[update_index * batch : (update_index + 1) * batch]
        if bank:
            episode_rows = [
                _banked_training_row(
                    bank,
                    pool[int(task_index)],
                    config,
                    row_config,
                    rng,
                    effort=int(effort),
                    plumbing_only=data.plumbing_only,
                    sequence_length=sequence_length,
                )
                for task_index in picks
            ]
        else:
            seeds = np.asarray(
                rng.randint(0, np.iinfo(np.int32).max, size=batch), dtype=np.int64
            )
            jobs = [
                _TrainingRowJob(
                    ordinal=slot,
                    origin=pool[int(task_index)],
                    config=config,
                    row_config=row_config,
                    seed=int(seed),
                    effort=int(effort),
                    plumbing_only=data.plumbing_only,
                    sequence_length=sequence_length,
                )
                for slot, (task_index, seed) in enumerate(zip(picks, seeds, strict=True))
            ]
            descriptors = list(
                _ordered_training_episodes(jobs, config.training_workers)
            )
            encoded_episodes = _encode_arc_query_episodes_batched(
                tuple(descriptor["episode"] for descriptor in descriptors), row_config
            )
            episode_rows = [
                _training_row_from_encoded(descriptor, encoded, config, row_config)
                for descriptor, encoded in zip(descriptors, encoded_episodes, strict=True)
            ]
        rows.append(_merge_training_rows(episode_rows))
        if len(rows) == size:
            start = update_index + 1 - size
            chunk = _stacked_chunk(rows, efforts[start : update_index + 1])
            rows = []
            yield chunk


@dataclass(frozen=True)
class _ProducerFailure:
    """Exception and traceback captured by the training producer."""

    error: BaseException
    traceback: Any


_PREFETCH_COMPLETE = object()


def _prefetched_training_chunks(
    chunks: Iterable[Any], profile: _TrainingProfile | None = None
) -> Iterator[Any]:
    """Yield items from a one-ahead asynchronous CPU producer.

    The producer may advance the source exactly once beyond the item currently
    consumed. Producer failures retain their traceback, and early consumer exit
    stops and joins the worker.

    Parameters
    ----------
    chunks
        Ordered items to prepare.
    profile
        Optional diagnostic accumulator. When provided, producer construction
        and consumer queue wait are timed without changing item order.

    Yields
    ------
    Any
        Items in unchanged source order.
    """
    source = iter(chunks)
    pending: queue.Queue[object] = queue.Queue(maxsize=1)
    advance_permit = threading.Semaphore(1)
    stopped = threading.Event()

    def await_permit() -> bool:
        while not stopped.is_set():
            if advance_permit.acquire(timeout=0.05):
                return True
        return False

    def publish(item: object) -> bool:
        while not stopped.is_set():
            try:
                pending.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        failure: _ProducerFailure | None = None
        completed = False
        try:
            while await_permit():
                encoding_started = time.perf_counter()
                try:
                    item = next(source)
                except StopIteration:
                    completed = True
                    break
                finally:
                    if profile is not None:
                        profile.producer_encoding_seconds += (
                            time.perf_counter() - encoding_started
                        )
                if not publish(item):
                    return
        except BaseException as error:
            failure = _ProducerFailure(error, error.__traceback__)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    if failure is None:
                        failure = _ProducerFailure(error, error.__traceback__)
            if failure is not None:
                publish(failure)
            elif completed:
                publish(_PREFETCH_COMPLETE)

    worker = threading.Thread(
        target=produce,
        name="example21-training-chunk-producer",
        daemon=True,
    )
    worker.start()
    try:
        while True:
            wait_started = time.perf_counter()
            item = pending.get()
            if profile is not None:
                profile.consumer_wait_seconds += time.perf_counter() - wait_started
            advance_permit.release()
            if item is _PREFETCH_COMPLETE:
                return
            if isinstance(item, _ProducerFailure):
                raise item.error.with_traceback(item.traceback)
            yield item
    finally:
        stopped.set()
        worker.join()


def _row_refinement_layout(row_config: RowEventConfig) -> RowRefinementLayout:
    """Map the ARC row-event schema into the learned feedback layout."""

    return RowRefinementLayout(
        input_width=row_config.input_width,
        event_valid_index=row_config.valid_slice.start,
        demonstration_phase_index=row_config.phase_slice.start,
        query_phase_index=row_config.phase_slice.start + 1,
        input_side_valid_index=row_config.side_valid_slice.start,
        output_side_valid_index=row_config.side_valid_slice.start + 1,
        normalized_start=row_config.normalized_slice.start,
        row_index_start=row_config.row_index_slice.start,
        input_height_start=row_config.input_height_slice.start,
        input_width_start=row_config.input_width_slice.start,
        output_height_start=row_config.output_height_slice.start,
        output_width_start=row_config.output_width_slice.start,
        input_mask_start=row_config.input_mask_slice.start,
        output_mask_start=row_config.output_mask_slice.start,
        input_color_start=row_config.input_color_slice.start,
        output_color_start=row_config.output_color_slice.start,
    )


def _model_config(
    config: ExperimentConfig, row_config: RowEventConfig, *, batch_size: int
) -> ModelConfig:
    arguments: dict[str, object] = {
        "input_width": row_config.input_width,
        "batch_size": batch_size,
        "neuron_count": config.neuron_count,
        "recurrent_edges": config.recurrent_edges,
        "max_latent_steps": config.latent_steps,
        "readout_width": config.readout_width,
        "color_rank": config.color_rank,
        "seed": config.seed,
        "event_valid_index": row_config.valid_slice.start,
        "decoder_mode": config.decoder_mode,
        "refinement_steps": config.latent_steps,
        "sparse_backend": (
            None if config.sparse_backend == "default" else config.sparse_backend
        ),
        "trace_engine": config.trace_engine,
        "memory_value_softcap_beta": config.memory_value_softcap_beta,
        "reasoning_query_softcap_beta": config.reasoning_query_softcap_beta,
        "neuron_typing": config.neuron_typing,
        "excitatory_fraction": config.excitatory_fraction,
    }
    if config.decoder_mode in ("row_refinement", "latent_row_decode"):
        arguments["refinement_layout"] = _row_refinement_layout(row_config)
        arguments["copy_residual_gain"] = config.copy_residual_gain
        arguments["row_head_carrier_scale"] = config.row_head_carrier_scale
        arguments["row_head_carrier_gate"] = config.row_head_carrier_gate
        arguments["row_head_modulation"] = config.row_head_modulation
        arguments["row_head_modulation_rank"] = config.row_head_modulation_rank
        arguments["row_copy_gate"] = config.row_copy_gate
        arguments["row_copy_gate_bias"] = config.row_copy_gate_bias
        arguments["shape_head_carrier_scale"] = config.shape_head_carrier_scale
        arguments["refinement_mixer"] = config.refinement_mixer
    if config.context_memory_width > 0:
        features = associative_memory_feature_indices(row_config)
        update_features = learned_update_feature_indices(row_config)
        arguments.update(
            {
                "context_memory_width": config.context_memory_width,
                "memory_decay": config.memory_decay,
                "memory_read_transform": config.memory_read_transform,
                "memory_read_interval": config.memory_read_interval,
                "latent_residual_mixer": config.latent_residual_mixer,
                "latent_residual_block_size": config.latent_residual_block_size,
                "demonstration_phase_index": row_config.phase_slice.start,
                "query_phase_index": row_config.phase_slice.start + 1,
                "input_side_valid_index": row_config.side_valid_slice.start,
                "output_side_valid_index": row_config.side_valid_slice.start + 1,
                "memory_key_indices": features.key_indices,
                "memory_value_indices": features.value_indices,
                "memory_coding": config.memory_coding,
                "memory_update_indices": update_features.indices,
                "memory_update_feature_order": update_features.order,
            }
        )
    return ModelConfig(**arguments)


def _memory_architecture_report(
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    training_batch_size: int,
    evaluation_batch_size: int,
) -> dict[str, object]:
    """Describe the selected reasoning mode and dense context-state cost."""
    training_batch_size = _integer(
        training_batch_size, "training_batch_size", minimum=1
    )
    evaluation_batch_size = _integer(
        evaluation_batch_size, "evaluation_batch_size", minimum=1
    )
    enabled = config.context_memory_width > 0
    if enabled:
        features = associative_memory_feature_indices(row_config)
        key_width = len(features.key_indices)
        value_width = len(features.value_indices)
    else:
        key_width = 0
        value_width = 0
    bytes_per_example = config.context_memory_width**2 * np.dtype(np.float32).itemsize
    report = {
        "reasoning_mode": ("associative_workspace" if enabled else "legacy_reservoir"),
        "context_memory_width": config.context_memory_width,
        "memory_decay": config.memory_decay,
        "effort_schedule": config.effort_schedule,
        "effort_distillation_weight": config.effort_distillation_weight,
        "raw_key_feature_width": key_width,
        "raw_value_feature_width": value_width,
        "context_memory_bytes_per_example": bytes_per_example,
        "context_memory_bytes_training_batch": (
            bytes_per_example * training_batch_size
        ),
        "context_memory_bytes_evaluation_batch": (
            bytes_per_example * evaluation_batch_size
        ),
    }
    if enabled:
        report["memory_read_transform"] = config.memory_read_transform
        report["memory_read_interval"] = config.memory_read_interval
        report["latent_residual_mixer"] = config.latent_residual_mixer
        report["latent_residual_block_size"] = config.latent_residual_block_size
    return report


def _model_memory_report(model: LatentWorkspaceModel) -> dict[str, object]:
    """Return the model-owned associative representation provenance.

    Under ``memory_coding="learned_write"`` this also carries the write-versus-
    retrieval key divergence, without which a pinned-at-zero pairing readout
    cannot be told apart from a read that simply drifted out of the code the
    memory was written in.
    """
    report = model.associative_memory_report().to_dict()
    report.update(model.memory_coding_divergence())
    if model.config.memory_enabled:
        report.update(model.memory_read_diagnostics())
    return report


def _make_model(
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    batch_size: int,
    device: jax.Device,
) -> LatentWorkspaceModel:
    with jax.default_device(device), brainstate.random.seed_context(config.seed):
        return LatentWorkspaceModel(
            _model_config(config, row_config, batch_size=batch_size)
        )


def _copy_parameters(
    source: LatentWorkspaceModel, target: LatentWorkspaceModel
) -> None:
    source_states = source.states(brainstate.ParamState)
    target_states = target.states(brainstate.ParamState)
    if tuple(source_states.keys()) != tuple(target_states.keys()):
        raise ValueError("training and evaluation parameter paths differ")
    for source_state, target_state in zip(
        source_states.values(), target_states.values(), strict=True
    ):
        target_state.value = jax.tree.map(jnp.array, source_state.value)


def _tree_digest(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        for leaf in jax.tree.leaves(values[key]):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _compiler_evidence(learner: Any) -> dict[str, object]:
    report = getattr(learner, "report", None)
    if report is None:
        return {
            "available": False,
            "counts": {},
            "etrace_weights": [],
            "excluded_weights": [],
            "diagnostics": [],
        }

    def path_text(path: object) -> str:
        if isinstance(path, (tuple, list)):
            return ".".join(str(part) for part in path)
        return str(path)

    def enum_text(value: object) -> str:
        return str(getattr(value, "value", value))

    diagnostics = []
    for record in report.diagnostics:
        item: dict[str, object] = {
            "kind": enum_text(record.kind),
            "level": enum_text(record.level),
            "message": str(record.message),
        }
        if hasattr(record, "weight_path"):
            item["weight_path"] = path_text(record.weight_path)
        if hasattr(record, "hidden_paths"):
            item["hidden_paths"] = [path_text(path) for path in record.hidden_paths]
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            path_classification = context.get("path_classification")
            if isinstance(path_classification, dict):
                item["path_classification_by_hidden_state"] = {
                    path_text(path): enum_text(classification)
                    for path, classification in path_classification.items()
                }
        diagnostics.append(item)
    warning_count = sum(item["level"] == "warning" for item in diagnostics)
    error_count = sum(item["level"] == "error" for item in diagnostics)
    return {
        "available": True,
        "counts": {
            "hidden_groups": len(report.hidden_groups),
            "etrace_weights": len(report.etrace_weights),
            "excluded_weights": len(report.excluded_weights),
            "warnings": warning_count,
            "errors": error_count,
        },
        "etrace_weights": [
            {
                "parameter": path_text(path),
                "hidden_group_indices": [int(index) for index in groups],
            }
            for path, groups in report.etrace_weights
        ],
        "excluded_weights": [
            {"parameter": path_text(path), "reason": str(reason)}
            for path, reason in report.excluded_weights
        ],
        "diagnostics": diagnostics,
    }


def _parameter_travel_budget(config: ExperimentConfig) -> dict[str, object]:
    """Return how far the configured budget can move an answer-head weight.

    Adam cannot displace a coordinate further than the summed per-update
    rates, so an operating point below one answer-head initialisation sigma
    trains a head that stays a perturbation of its random initialisation
    whatever the loss says. Under a flat schedule that sum is
    ``learning_rate * updates``; under cosine decay to zero the schedule
    integral is half the flat value, so the base rate is multiplied by the
    schedule integral factor (0.5) before the bound is applied. Recording the
    width the sigma was taken against lets a later reader redo the arithmetic
    without knowing which head revision ran.
    """
    if config.optimizer != "adam":
        return {
            "applicable": False,
            "optimizer": config.optimizer,
            "reason": "the Adam coordinate-displacement bound does not apply",
        }
    width = (
        refinement_head_width(config.neuron_count)
        if config.decoder_mode in ("row_refinement", "latent_row_decode")
        else config.readout_width
    )
    integral_factor = 1.0 if config.lr_schedule == "constant" else 0.5
    budget = adam_parameter_travel_budget(
        config.learning_rate * integral_factor, config.training_updates, width
    )
    budget["lr_schedule"] = config.lr_schedule
    budget["schedule_integral_factor"] = integral_factor
    return budget


def _optimizer_policy(config: ExperimentConfig) -> dict[str, object]:
    """Return the resolved shared-training optimizer policy."""
    policy: dict[str, object] = {
        "name": config.optimizer,
        "learning_rate": config.learning_rate,
        "lr_schedule": config.lr_schedule,
        "lr_warmup_fraction": config.lr_warmup_fraction,
        "effort_distillation_weight": config.effort_distillation_weight,
        "weight_decay": config.weight_decay,
    }
    if config.optimizer == "muon":
        policy.update(
            {
                "matrix_optimizer": "muon",
                "matrix_rank": 2,
                "nonmatrix_optimizer": "adamw",
                "muon_ns_steps": 5,
                "muon_momentum_beta": 0.95,
            }
        )
    return policy


def _training_learning_rate(config: ExperimentConfig) -> float | optax.Schedule:
    """Return the shared-training rate, flat or as a cosine decay schedule.

    Parameters
    ----------
    config : ExperimentConfig
        Resolved experiment configuration.

    Returns
    -------
    float or optax.Schedule
        The flat base rate for ``lr_schedule="constant"``, otherwise an optax
        cosine decay schedule from the base rate to zero (``alpha=0``) over
        ``training_updates`` optimizer updates.
    """
    if config.lr_schedule == "constant":
        return config.learning_rate
    if config.lr_warmup_fraction:
        warmup_steps = max(
            1,
            math.ceil(config.lr_warmup_fraction * config.training_updates),
        )
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max(int(config.training_updates), 1),
            end_value=0.0,
        )
    return optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(int(config.training_updates), 1),
    )


def _make_training_optimizer(
    config: ExperimentConfig, param_states: Mapping[object, brainstate.ParamState]
) -> braintools.optim.Optimizer:
    """Construct and register the configured shared-training optimizer.

    All three optimizers are built as optax transformations so the
    shared-training learning-rate schedule threads through identically;
    ``braintools.optim.Adam``/``AdamW`` accept only flat rates or braintools
    schedulers, so the adam and adamw branches use the equivalent
    ``optax.adam``/``optax.adamw`` (same moments, same decoupled decay
    placement for AdamW) wrapped in ``braintools.optim.OptaxOptimizer``.
    """
    rate = _training_learning_rate(config)
    if config.optimizer == "adam":
        transformation = optax.adam(learning_rate=rate)
    elif config.optimizer == "adamw":
        transformation = optax.adamw(
            learning_rate=rate, weight_decay=config.weight_decay
        )
    else:
        transformation = optax.contrib.muon(
            learning_rate=rate,
            weight_decay=config.weight_decay,
            adam_learning_rate=rate,
            adam_weight_decay=config.weight_decay,
        )
    optimizer = braintools.optim.OptaxOptimizer(
        tx=transformation, lr=config.learning_rate
    )
    optimizer.register_trainable_weights(param_states)
    return optimizer


def _parameter_change_evidence(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, object]]:
    if before.keys() != after.keys():
        raise ValueError("parameter paths changed during optimization")
    result: dict[str, dict[str, object]] = {}
    for path in before:
        before_leaves = jax.tree.leaves(before[path])
        after_leaves = jax.tree.leaves(after[path])
        if len(before_leaves) != len(after_leaves):
            raise ValueError(f"parameter structure changed during optimization: {path}")
        squared = 0.0
        for before_leaf, after_leaf in zip(before_leaves, after_leaves, strict=True):
            delta = np.asarray(after_leaf, dtype=np.float64) - np.asarray(
                before_leaf, dtype=np.float64
            )
            squared += float(np.sum(delta * delta))
        before_sha = _tree_digest({path: before[path]})
        after_sha = _tree_digest({path: after[path]})
        result[path] = {
            "l2_delta": math.sqrt(squared),
            "changed": before_sha != after_sha,
            "sha256_before": before_sha,
            "sha256_after": after_sha,
        }
    return result


@dataclass(frozen=True)
class _TrainingSchedule:
    """Per-update training metadata accumulated across chunks, in order."""

    training_sequence_length: int | None = None
    efforts: tuple[int, ...] = ()
    task_fingerprints: tuple[str, ...] = ()
    base_task_fingerprints: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    held_out_demonstration_indices: tuple[int, ...] = ()

    def extended(self, chunk: _TrainingTensors) -> _TrainingSchedule:
        sequence_length = int(chunk.events.shape[1])
        if (
            self.training_sequence_length is not None
            and self.training_sequence_length != sequence_length
        ):
            raise ValueError("training chunks disagree on packed sequence length")
        return _TrainingSchedule(
            training_sequence_length=sequence_length,
            efforts=self.efforts + tuple(int(value) for value in chunk.efforts),
            task_fingerprints=self.task_fingerprints + tuple(chunk.task_fingerprints),
            base_task_fingerprints=(
                self.base_task_fingerprints + tuple(chunk.base_task_fingerprints)
            ),
            source_names=self.source_names + tuple(chunk.source_names),
            held_out_demonstration_indices=(
                self.held_out_demonstration_indices
                + tuple(chunk.held_out_demonstration_indices)
            ),
        )


def _train_chunks(
    chunks: Iterable[_TrainingTensors],
    train_all: Any,
    on_chunk: Callable[[int], None] | None = None,
    profile: _TrainingProfile | None = None,
) -> tuple[list[float], _TrainingSchedule]:
    """Stage each chunk on device in turn and accumulate the whole schedule.

    The per-update work stays inside the single compiled ``train_all`` scan;
    this loop only walks a handful of data-staging steps so that peak device
    memory tracks the chunk size rather than ``training_updates``.
    """
    losses: list[float] = []
    schedule = _TrainingSchedule()
    sequence_length: int | None = None
    for index, chunk in enumerate(chunks):
        if sequence_length is None:
            sequence_length = int(chunk.events.shape[1])
        elif int(chunk.events.shape[1]) != sequence_length:
            raise ValueError("training chunks disagree on packed sequence length")
        if profile is None:
            output = train_all(
                jnp.asarray(chunk.events),
                jnp.asarray(chunk.advances),
                jnp.asarray(chunk.heights),
                jnp.asarray(chunk.widths),
                jnp.asarray(chunk.colors),
                jnp.asarray(chunk.masks),
            )
            values = np.asarray(output)
        else:
            staging_started = time.perf_counter()
            staged = (
                jnp.asarray(chunk.events),
                jnp.asarray(chunk.advances),
                jnp.asarray(chunk.heights),
                jnp.asarray(chunk.widths),
                jnp.asarray(chunk.colors),
                jnp.asarray(chunk.masks),
            )
            profile.host_to_device_staging_seconds += (
                time.perf_counter() - staging_started
            )
            call_started = time.perf_counter()
            output = train_all(*staged)
            call_seconds = time.perf_counter() - call_started
            if index == 0:
                profile.first_call_compilation_seconds = call_seconds
            compute_started = time.perf_counter()
            jax.block_until_ready(output)
            compute_seconds = time.perf_counter() - compute_started
            if index == 0:
                profile.first_call_device_compute_seconds = compute_seconds
            else:
                profile.steady_state_device_compute_seconds += compute_seconds
            copy_started = time.perf_counter()
            values = np.asarray(output)
            profile.host_result_copy_seconds += time.perf_counter() - copy_started
        losses.extend(float(value) for value in values)
        if profile is not None:
            profile.chunk_count += 1
        schedule = schedule.extended(chunk)
        if on_chunk is not None:
            on_chunk(index)
    if len(losses) != len(schedule.efforts):
        raise ValueError("training losses and effort schedule disagree in length")
    return losses, schedule


def _write_parameter_checkpoint(
    model: LatentWorkspaceModel,
    path: pathlib.Path,
    *,
    effort_schedule: EffortScheduleName = "uniform",
    effort_distillation_weight: float = 0.0,
) -> str:
    """Write trainable parameter leaves and return the file digest.

    The shared training stage is expensive and optional diagnostic variants can
    reuse its parameters. Leaves are stored in deterministic tree order so a
    model of the same shape restores them exactly.
    """
    states = model.states(brainstate.ParamState)
    leaves = jax.tree.leaves({key: state.value for key, state in states.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".partial")
    architecture = np.frombuffer(
        msgspec.json.encode(
            {
                "schema_version": 1,
                "memory_coding": model.config.memory_coding,
                "memory_read_interval": model.config.memory_read_interval,
                "latent_residual_mixer": model.config.latent_residual_mixer,
                "latent_residual_block_size": (
                    model.config.latent_residual_block_size
                ),
                "effort_schedule": effort_schedule,
                "effort_distillation_weight": effort_distillation_weight,
            }
        ),
        dtype=np.uint8,
    )
    with staged.open("wb") as handle:
        np.savez(
            handle,
            *[np.asarray(leaf) for leaf in leaves],
            __architecture__=architecture,
        )
    os.replace(staged, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_parameter_checkpoint(
    model: LatentWorkspaceModel,
    path: pathlib.Path,
    *,
    effort_schedule: EffortScheduleName = "uniform",
    effort_distillation_weight: float = 0.0,
) -> str:
    """Restore parameter leaves written by :func:`_write_parameter_checkpoint`.

    A checkpoint from a different neuron count, edge count, or decoder mode has
    a different parameter tree and fails closed here rather than being reshaped.
    """
    states = model.states(brainstate.ParamState)
    leaves, structure = jax.tree.flatten(
        {key: state.value for key, state in states.items()}
    )
    stored = np.load(path)
    names = [f"arr_{index}" for index in range(len(leaves))]
    stored_names = set(stored.files)
    if stored_names not in (set(names), set(names) | {"__architecture__"}):
        raise ValueError("parameter checkpoint does not match the model tree")
    if "__architecture__" in stored_names:
        architecture = msgspec.json.decode(
            np.asarray(stored["__architecture__"], dtype=np.uint8).tobytes()
        )
        if isinstance(architecture, dict) and architecture.get("schema_version") == 1:
            architecture.setdefault("latent_residual_mixer", "none")
            architecture.setdefault("latent_residual_block_size", 10)
            architecture.setdefault("effort_schedule", "uniform")
            architecture.setdefault("memory_coding", model.config.memory_coding)
            architecture.setdefault("effort_distillation_weight", 0.0)
        if architecture != {
            "schema_version": 1,
            "memory_coding": model.config.memory_coding,
            "memory_read_interval": model.config.memory_read_interval,
            "latent_residual_mixer": model.config.latent_residual_mixer,
            "latent_residual_block_size": model.config.latent_residual_block_size,
            "effort_schedule": effort_schedule,
            "effort_distillation_weight": effort_distillation_weight,
        }:
            raise ValueError("parameter checkpoint does not match model architecture")
    restored = [jnp.asarray(stored[name]) for name in names]
    for target, value in zip(leaves, restored, strict=True):
        if np.shape(target) != np.shape(value):
            raise ValueError("parameter checkpoint does not match the model shapes")
    for key, value in jax.tree.unflatten(structure, restored).items():
        states[key].value = value
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restore_initial_parameters(
    model: LatentWorkspaceModel, config: ExperimentConfig
) -> str | None:
    """Seed a training segment from a previous segment's parameters.

    Returns the restored file's digest, or ``None`` when the run starts from a
    fresh initialization. The restore fails closed on a parameter tree from a
    different neuron count, edge count, or decoder mode.
    """
    initial = config.initial_checkpoint
    if initial is None or not initial.exists():
        return None
    return _read_parameter_checkpoint(
        model,
        initial,
        effort_schedule=config.effort_schedule,
        effort_distillation_weight=config.effort_distillation_weight,
    )


def _checkpoint_writer(
    model: LatentWorkspaceModel, config: ExperimentConfig
) -> Callable[[int], None] | None:
    """Return a per-chunk callback that periodically persists parameters.

    A device fault during a multi-hour segment then costs at most one interval
    of training rather than the whole segment. ``None`` reproduces the single
    write after training.
    """
    every = config.checkpoint_every
    path = config.parameter_checkpoint
    if not every or path is None:
        return None

    def write(index: int) -> None:
        if (index + 1) % every == 0:
            _write_parameter_checkpoint(
                model,
                path,
                effort_schedule=config.effort_schedule,
                effort_distillation_weight=config.effort_distillation_weight,
            )

    return write


def _restored_training_report(
    model: LatentWorkspaceModel, config: ExperimentConfig, digest: str
) -> dict[str, object]:
    """Describe a run that restored parameters instead of optimizing them."""
    return {
        "performed": False,
        "reason": "restored_parameter_checkpoint",
        "parameter_checkpoint": str(config.parameter_checkpoint),
        "parameter_checkpoint_sha256": digest,
        "one_shared_model": True,
        "supervised_depths": (
            "latent_row_ticks_1..effort"
            if config.decoder_mode in ("row_refinement", "latent_row_decode")
            else "0..effort"
        ),
        "depth_weighting": "uniform_unit_sum_per_update",
        "balanced_color_loss": config.balanced_color_loss,
        "optimizer_updates_by_effort": {
            str(value): 0 for value in config.training_efforts
        },
        "effort_schedule_policy": config.effort_schedule,
        "effort_schedule": [],
        "effort_self_distillation": {
            "enabled": False,
            "weight": config.effort_distillation_weight,
            "teacher_gate": "not_satisfied",
        },
        "losses": [],
        "parameter_sha256_before": _tree_digest(parameter_snapshot(model)),
        "parameter_sha256_after": _tree_digest(parameter_snapshot(model)),
    }


def _per_episode_efforts(efforts: tuple[int, ...], batch_size: int) -> tuple[int, ...]:
    """Expand one supervised depth per update into one per batched episode."""
    return tuple(int(effort) for effort in efforts for _ in range(batch_size))


def _train_model(
    model: LatentWorkspaceModel,
    chunks: Iterable[_TrainingTensors],
    config: ExperimentConfig,
    on_chunk: Callable[[int], None] | None = None,
    profile: _TrainingProfile | None = None,
) -> dict[str, object]:
    learner = compile_pp_prop(model)
    compiler_report = _compiler_evidence(learner)
    compiler = {
        "pp_prop_compiled": True,
        "learner_type": type(learner).__name__,
        "event_and_advance_arguments": True,
        "compiler_report": compiler_report,
        "compiled_parameter_paths": [
            ".".join(str(part) for part in path)
            if isinstance(path, tuple)
            else str(path)
            for path in getattr(learner, "param_states", {}).keys()
        ],
    }
    supervised_depths = (
        "latent_row_ticks_1..effort"
        if config.decoder_mode in ("row_refinement", "latent_row_decode")
        else "0..effort"
    )
    if config.structural_only:
        model.reset_state()
        learner.reset_state(batch_size=model.config.batch_size)
        return {
            "performed": False,
            "reason": "structural_only",
            "one_shared_model": True,
            "supervised_depths": supervised_depths,
            "depth_weighting": "uniform_unit_sum_per_update",
            "balanced_color_loss": config.balanced_color_loss,
            "optimizer": _optimizer_policy(config),
            "effort_self_distillation": {
                "enabled": False,
                "weight": config.effort_distillation_weight,
                "teacher_gate": "not_satisfied",
            },
            **compiler,
            "optimizer_updates_by_effort": {
                str(value): 0 for value in config.training_efforts
            },
            "losses": [],
            "runtime_profile": None if profile is None else profile.to_dict(),
        }
    optimizer = _make_training_optimizer(config, learner.param_states)
    before_snapshot = parameter_snapshot(model)
    before = _tree_digest(before_snapshot)
    rank = model.config.color_rank

    @brainstate.transform.jit
    def train_all(events, advances, heights, widths, colors, masks):
        def train_one(inputs):
            sequence, advance, target_height, target_width, target_colors, mask = inputs
            model.reset_state()
            learner.reset_state(batch_size=model.config.batch_size)

            trace_control: jax.Array | StepGates = advance
            if config.decoder_mode == "latent_row_decode":
                assert model.config.query_phase_index is not None
                decoder_gate = (~advance) & (
                    sequence[:, :, model.config.query_phase_index] > 0.5
                )
                latent_gate = advance & ~(
                    sequence[:, :, model.config.event_valid_index] > 0.5
                )
                trace_control = StepGates(
                    advance_physics=advance,
                    latent_update=latent_gate,
                    decode_row=decoder_gate,
                    answer_feedback=jnp.zeros_like(advance),
                    recurrent_enabled=advance,
                )

            def step_loss(event, advance_gate):
                compact = learner(event, advance_gate)
                if config.decoder_mode in ("row_refinement", "latent_row_decode"):
                    current_rows = jnp.mod(
                        jnp.asarray(model.reasoning_index.value, dtype=jnp.int32) - 1,
                        30,
                    )
                    losses = row_refinement_loss_per_example(
                        compact,
                        target_height,
                        target_width,
                        target_colors,
                        current_rows,
                    )
                    if isinstance(advance_gate, StepGates):
                        supervised = advance_gate.decode_row
                    else:
                        supervised = advance_gate & ~(
                            event[:, model.config.event_valid_index] > 0.5
                        )
                else:
                    losses = arc_loss_per_example(
                        compact,
                        target_height + 1,
                        target_width + 1,
                        target_colors,
                        color_rank=rank,
                        class_balanced_colors=config.balanced_color_loss,
                    )
                    supervised = advance_gate
                return jnp.sum(jnp.where(supervised, losses, 0.0)) / jnp.maximum(
                    jnp.sum(supervised), 1
                )

            gradients, objective = learner.etrace_grad(
                sequence,
                trace_control,
                step_fn=step_loss,
                mask=mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            # Dale's law is a hard constraint, not a preference: re-project
            # the recurrent signs after every optimizer step (no-op under
            # neuron_typing="none").
            model.project_recurrent_dale_weights()
            return objective

        return brainstate.transform.for_loop(
            train_one, (events, advances, heights, widths, colors, masks)
        )

    losses, schedule = _train_chunks(
        _prefetched_training_chunks(chunks, profile), train_all, on_chunk, profile
    )
    after_snapshot = parameter_snapshot(model)
    after = _tree_digest(after_snapshot)
    counts = Counter(schedule.efforts)
    sample_records = [
        {
            "source": source,
            "base_task_fingerprint": base_fingerprint,
            "augmented_task_fingerprint": augmented_fingerprint,
            "episode_kind": "leave_one_demonstration_out",
            "held_out_demonstration_index": int(held_out_index),
            "maximum_supervised_depth": int(effort),
        }
        for source, base_fingerprint, augmented_fingerprint, held_out_index, effort in zip(
            schedule.source_names,
            schedule.base_task_fingerprints,
            schedule.task_fingerprints,
            schedule.held_out_demonstration_indices,
            _per_episode_efforts(schedule.efforts, config.training_batch_size),
            strict=True,
        )
    ]
    return {
        "performed": True,
        "one_shared_model": True,
        "one_shared_optimizer_state": True,
        **compiler,
        "supervised_depths": supervised_depths,
        "depth_weighting": "uniform_unit_sum_per_update",
        "per_update_depth_weight_sum": 1.0,
        "training_sequence_length": schedule.training_sequence_length,
        "training_chunk_prefetch": {
            "enabled": True,
            "max_buffered_chunks": 1,
        },
        "training_workers": config.training_workers,
        "training_encoder": "bounded_batched_numpy_with_scalar_oracle",
        "checkpoint_every_chunks": config.checkpoint_every,
        "training_holdout_tasks": config.training_holdout_tasks,
        "training_row_max_inflight": 2 * config.training_workers,
        "runtime_profile": None if profile is None else profile.to_dict(),
        "balanced_color_loss": config.balanced_color_loss,
        "optimizer": _optimizer_policy(config),
        "parameter_travel_budget": _parameter_travel_budget(config),
        "loss_weights": {"height": 1.0, "width": 1.0, "valid_cell_color": 1.0},
        "optimizer_updates_by_effort": {
            str(value): int(counts[value]) for value in config.training_efforts
        },
        "losses": np.asarray(losses, dtype=np.float64).tolist(),
        "effort_schedule_policy": config.effort_schedule,
        "effort_schedule": list(schedule.efforts),
        "effort_self_distillation": {
            "enabled": False,
            "weight": config.effort_distillation_weight,
            "teacher_gate": "not_satisfied",
        },
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameters_moved": before != after,
        "parameter_changes": _parameter_change_evidence(
            before_snapshot, after_snapshot
        ),
        "training_task_fingerprints": list(schedule.task_fingerprints),
        "training_episode_kind": "leave_one_demonstration_out",
        "sampled_base_task_count": len(set(schedule.base_task_fingerprints)),
        "sampled_base_fold_count": len(
            set(
                zip(
                    schedule.base_task_fingerprints,
                    schedule.held_out_demonstration_indices,
                    strict=True,
                )
            )
        ),
        "sampling_with_replacement": True,
        "training_samples": sample_records,
    }


def _origin_task_key(origin: _OriginTask, task_index: int) -> str:
    fingerprint = canonical_task_fingerprint(origin.task, include_test_outputs=False)
    task_name = origin.task.task_id or fingerprint[:12]
    return f"{origin.source_name}:{task_index:04d}:{task_name}:{fingerprint}"


def _selected_evaluation_origins(
    data: _ExperimentData, config: ExperimentConfig
) -> tuple[_OriginTask, ...]:
    origins = data.evaluation
    if config.evaluation_task_limit is not None:
        origins = origins[: config.evaluation_task_limit]
    return origins


def _evaluation_records(
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
) -> tuple[_EvaluationRecord, ...]:
    origins = _selected_evaluation_origins(data, config)
    records: list[_EvaluationRecord] = []
    for task_index, origin in enumerate(origins):
        task_key = _origin_task_key(origin, task_index)
        for query_index in range(len(origin.task.test)):
            encoded = encode_query_episode(
                origin.task,
                query_index,
                row_config,
                task_index=task_index,
            )
            if encoded.target is None:
                raise ValueError(
                    f"evaluation query {task_key}:{query_index} lacks target"
                )
            records.append(
                _EvaluationRecord(
                    origin.source_name,
                    task_key,
                    encoded,
                    np.asarray(origin.task.test[query_index].input.as_array()),
                )
            )
    if not records:
        raise ValueError("evaluation produced no scored queries")
    return tuple(records)


def _submission_policy_name(mode: PrimaryCandidateMode) -> str:
    """Return the policy identifier that describes ``mode`` honestly."""

    return SUBMISSION_POLICY if mode == "model_only" else RULE_SUBMISSION_POLICY


def _rule_proposals(
    records: Sequence["_EvaluationRecord"],
    source_tasks: Sequence["_OriginTask"],
    *,
    arm: Literal["intact", "no_context", "shuffled"],
) -> tuple[tuple[str, np.ndarray] | None, ...]:
    """Propose one demonstration-verified candidate per query for one arm.

    The channel is fitted on the demonstrations *that arm actually has*, never
    the original task's. Without this the proposals would be arm-invariant and
    every control would report the same solves as ``intact`` -- the exact
    signature this experiment treats as evidence of a non-causal result. No
    query target is read anywhere in this path.

    Parameters
    ----------
    records
        Evaluation records in scoring order.
    source_tasks
        Origin tasks the records were encoded from.
    arm
        Evaluation arm whose demonstrations should be fitted.

    Returns
    -------
    tuple
        One ``(rule_name, grid)`` proposal or ``None`` per record, aligned with
        ``records``.
    """

    if arm == "no_context":
        return tuple(None for _ in records)
    lookup = {
        _origin_task_key(origin, task_index): origin.task
        for task_index, origin in enumerate(source_tasks)
    }
    proposals: list[tuple[str, np.ndarray] | None] = []
    for record in records:
        task = lookup[record.task_key]
        if arm == "shuffled":
            task = _derange_task(task)
        if task is None:
            proposals.append(None)
            continue
        demonstrations = [
            (pair.input.as_array(), pair.output.as_array())
            for pair in task.train
            if pair.output is not None
        ]
        query = task.test[record.encoded.query_index].input.as_array()
        found = verified_rule_candidates(demonstrations, query)
        proposals.append(found[0] if found else None)
    return tuple(proposals)


def _derange_task(task: ArcTask) -> ArcTask | None:
    if len(task.train) < 2:
        return None
    outputs = tuple(pair.output for pair in task.train)
    return ArcTask(
        train=tuple(
            ArcPair(pair.input, outputs[(index + 1) % len(outputs)])
            for index, pair in enumerate(task.train)
        ),
        test=task.test,
        task_id=task.task_id,
    )


def _arm_sequences(
    records: Sequence[_EvaluationRecord],
    config: ExperimentConfig,
    row_config: RowEventConfig,
    *,
    arm: Literal["intact", "no_context", "shuffled"],
    source_tasks: Sequence[_OriginTask],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    sequences: list[np.ndarray] = []
    advance_rows: list[np.ndarray] = []
    query_stops: list[int] = []
    metadata: list[dict[str, object]] = []
    task_lookup = {
        _origin_task_key(origin, task_index): origin.task
        for task_index, origin in enumerate(source_tasks)
    }
    for record in records:
        encoded = record.encoded
        arm_encoded = encoded
        detail: dict[str, object] = {"available": True, "timing_matched": True}
        if arm == "no_context":
            events = np.array(encoded.events, copy=True)
            events[: encoded.query_start] = 0.0
            packed = _packed_events(encoded, config)
            packed[: encoded.events.shape[0]] = events
        elif arm == "shuffled":
            changed = _derange_task(task_lookup[record.task_key])
            if changed is None:
                packed = _packed_events(encoded, config)
                detail = {
                    "available": False,
                    "reason": "fewer than two demonstrations",
                    "timing_matched": True,
                }
            else:
                arm_encoded = encode_query_episode(
                    changed,
                    encoded.query_index,
                    row_config,
                    task_index=encoded.task_index,
                )
                packed = _packed_events(arm_encoded, config)
                detail["timing_matched"] = (
                    arm_encoded.query_start == encoded.query_start
                    and arm_encoded.query_stop == encoded.query_stop
                )
                if np.array_equal(
                    arm_encoded.events[: arm_encoded.query_start],
                    encoded.events[: encoded.query_start],
                ):
                    detail = {
                        "available": False,
                        "reason": (
                            "rotation leaves demonstration associations unchanged"
                        ),
                        "timing_matched": bool(detail["timing_matched"]),
                    }
        else:
            packed = _packed_events(encoded, config)
        sequences.append(packed)
        advance_rows.append(_packed_advances(encoded, config, row_config))
        query_stops.append(encoded.query_stop)
        metadata.append(detail)
    stacked = np.stack(sequences, axis=1)
    advances = np.stack(advance_rows, axis=1)
    return stacked, advances, np.asarray(query_stops, dtype=np.int32), metadata


def _score_windows(
    compact: np.ndarray,
    records: Sequence[_EvaluationRecord],
    color_rank: int,
    decoder_mode: DecoderMode,
    checkpoints: Sequence[int] = CHECKPOINTS,
    submission_checkpoint: int = SUBMISSION_CHECKPOINT,
    rule_proposals: Sequence[tuple[str, np.ndarray] | None] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Score submitted candidates at every frozen checkpoint.

    With ``rule_proposals`` omitted the submission is model-only. Supplying it
    selects the ``rule_then_model`` merge described in the submission policy.
    """

    checkpoint_indices = np.asarray(checkpoints, dtype=np.int32)
    checkpoint_compact = (
        compact
        if compact.shape[0] == len(checkpoints)
        else compact[checkpoint_indices]
    )
    return _score_checkpoint_logits(
        checkpoint_compact,
        records,
        color_rank,
        decoder_mode,
        checkpoints,
        submission_checkpoint,
        rule_proposals,
    )



def _dump_checkpoint_logits(
    height: np.ndarray,
    width: np.ndarray,
    colors: np.ndarray,
    records: Sequence["_EvaluationRecord"],
    checkpoints: Sequence[int],
) -> None:
    """Write raw decoder logits when ``EXAMPLE21_LOGITS_DUMP`` names a path."""

    destination = os.environ.get("EXAMPLE21_LOGITS_DUMP")
    if not destination:
        return
    path = pathlib.Path(destination)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = np.full((len(records), 30, 30), -1, dtype=np.int8)
    target_shapes = np.zeros((len(records), 2), dtype=np.int32)
    for index, record in enumerate(records):
        target = record.encoded.target
        assert target is not None
        grid = np.asarray(target.as_array(), dtype=np.int8)
        target_shapes[index] = grid.shape
        targets[index, : grid.shape[0], : grid.shape[1]] = grid
    np.savez_compressed(
        path,
        height=np.asarray(height, dtype=np.float32),
        width=np.asarray(width, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.float32),
        checkpoints=np.asarray(checkpoints, dtype=np.int32),
        task_ids=np.asarray([record.task_key for record in records]),
        query_indices=np.asarray(
            [record.encoded.query_index for record in records], dtype=np.int32
        ),
        targets=targets,
        target_shapes=target_shapes,
    )


def _merge_rule_candidate(
    proposal: tuple[str, np.ndarray] | None,
    candidates: Sequence[Any],
    candidate_payloads: Sequence[Mapping[str, object]],
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Place an admitted demonstration-verified rule in submission slot one.

    Candidate two becomes the model's own first decoded grid, so an admitted
    rule that is wrong costs at most the model's runner-up slot and never
    displaces the model's best grid.

    Parameters
    ----------
    proposal
        ``(rule_name, grid)`` from the verified rule channel, or ``None`` when
        the channel admitted no rule for this query.
    candidates
        Decoded model candidates in rank order.
    candidate_payloads
        Serialized model candidates aligned with ``candidates``.

    Returns
    -------
    tuple
        Submitted grids in rank order and their serialized payloads.
    """

    model_grids = [np.asarray(candidate.grid) for candidate in candidates]
    payloads = [dict(payload) for payload in candidate_payloads]
    if proposal is None:
        return model_grids, payloads
    rule_name, rule_grid = proposal
    rule_payload = {
        "height": int(np.asarray(rule_grid).shape[0]),
        "width": int(np.asarray(rule_grid).shape[1]),
        "grid": np.asarray(rule_grid).astype(int).tolist(),
        "changed_decision": None,
        "log_probability": None,
        "provenance": "rule",
        "rule_name": str(rule_name),
        "source_checkpoint": payloads[0].get("source_checkpoint"),
        "selection_role": "demonstration_verified_rule",
        "rank": 1,
    }
    hedge_index = 0
    if np.asarray(rule_grid).shape == model_grids[0].shape and np.array_equal(
        np.asarray(rule_grid), model_grids[0]
    ):
        if len(model_grids) < 2:
            return model_grids, payloads
        hedge_index = 1
    runner_up = dict(payloads[hedge_index])
    runner_up["rank"] = 2
    return (
        [np.asarray(rule_grid), model_grids[hedge_index]],
        [rule_payload, runner_up],
    )


def _score_checkpoint_logits(
    checkpoint_compact: np.ndarray,
    records: Sequence[_EvaluationRecord],
    color_rank: int,
    decoder_mode: DecoderMode,
    checkpoints: Sequence[int] = CHECKPOINTS,
    submission_checkpoint: int = SUBMISSION_CHECKPOINT,
    rule_proposals: Sequence[tuple[str, np.ndarray] | None] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Score model logits supplied at the configured semantic checkpoints.

    Parameters
    ----------
    checkpoint_compact
        Model logits in ``checkpoints`` order. No intermediate trajectory is
        required.
    records
        Ordered official-query metadata and out-of-band targets.
    color_rank
        Legacy CP decoder rank. Ignored by explicit row-refinement logits.
    decoder_mode
        Explicit decoder representation.

    Returns
    -------
    tuple
        Strict metrics and per-query candidate/score records by checkpoint.
    """

    checkpoint_compact = np.asarray(checkpoint_compact)
    if checkpoint_compact.ndim != 3:
        raise ValueError("checkpoint_compact must have checkpoint, query, width axes")
    checkpoints = tuple(int(value) for value in checkpoints)
    if checkpoint_compact.shape[0] != len(checkpoints):
        raise ValueError("checkpoint_compact must match the checkpoint schedule")
    if checkpoint_compact.shape[1] != len(records):
        raise ValueError("checkpoint query count must match evaluation records")
    expanded = expand_decoder_logits(
        jnp.asarray(checkpoint_compact), color_rank, decoder_mode
    )
    height = np.asarray(expanded.height)
    width = np.asarray(expanded.width)
    colors = np.asarray(expanded.colors)
    _dump_checkpoint_logits(height, width, colors, records, checkpoints)
    logits_by_checkpoint: dict[int, list[OutputLogits]] = {
        effort: [
            OutputLogits(
                height[effort_index, query_index],
                width[effort_index, query_index],
                colors[effort_index, query_index],
            )
            for query_index in range(len(records))
        ]
        for effort_index, effort in enumerate(checkpoints)
    }
    selected_by_checkpoint: dict[
        int, list[tuple[list[Any], list[dict[str, object]]]]
    ] = {effort: [] for effort in checkpoints}
    for effort in checkpoints:
        for query_index in range(len(records)):
            logits = logits_by_checkpoint[effort][query_index]
            if effort == 0:
                decoded = list(decode_candidates(logits))
                candidate_payloads = []
                diagnostic_roles = (
                    "diagnostic_checkpoint_joint_argmax",
                    "diagnostic_checkpoint_logit_runner_up",
                )
                for rank, (candidate, role) in enumerate(
                    zip(decoded, diagnostic_roles, strict=True), start=1
                ):
                    payload = dict(candidate.to_dict())
                    provenance = payload.get("provenance", "model")
                    if provenance != "model":
                        raise ValueError(
                            "primary scoring rejected non-model candidate provenance "
                            f"{provenance!r}"
                        )
                    payload.update(
                        {
                            "rank": rank,
                            "provenance": "model",
                            "source_checkpoint": 0,
                            "selection_role": role,
                        }
                    )
                    candidate_payloads.append(payload)
            else:
                selected = select_checkpoint_candidates(
                    {
                        checkpoint: logits_by_checkpoint[checkpoint][query_index]
                        for checkpoint in checkpoints
                        if 0 < checkpoint <= effort
                    },
                    latest_checkpoint=effort,
                )
                decoded = [item.candidate for item in selected]
                candidate_payloads = []
                for rank, candidate in enumerate(selected, start=1):
                    payload = candidate.to_dict()
                    if payload.get("provenance") != "model":
                        raise ValueError(
                            "primary scoring rejected non-model candidate provenance"
                        )
                    payload["rank"] = rank
                    candidate_payloads.append(payload)
            if len(candidate_payloads) != 2:
                raise ValueError(
                    "primary scoring requires exactly two model candidates"
                )
            selected_by_checkpoint[effort].append((decoded, candidate_payloads))

    if rule_proposals is not None and len(rule_proposals) != len(records):
        raise ValueError("rule proposals must match the evaluation records")
    candidate_mode: PrimaryCandidateMode = (
        "model_only" if rule_proposals is None else "rule_then_model"
    )
    metrics: dict[str, dict[str, object]] = {}
    query_details: dict[str, list[dict[str, object]]] = {}
    for effort in checkpoints:
        scores = []
        details = []
        for query_index, record in enumerate(records):
            candidates, candidate_payloads = selected_by_checkpoint[effort][query_index]
            target = record.encoded.target
            assert target is not None
            proposal = None if rule_proposals is None else rule_proposals[query_index]
            submitted_grids, candidate_payloads = _merge_rule_candidate(
                proposal, candidates, candidate_payloads
            )
            score = score_query_candidates(
                submitted_grids,
                target.as_array(),
                task_id=record.task_key,
                query_index=record.encoded.query_index,
            )
            scores.append(score)
            details.append(
                {
                    "task_id": record.task_key,
                    "query_index": record.encoded.query_index,
                    "input_echo": input_echo_fraction(
                        submitted_grids[0], record.query_input
                    ),
                    "primary_candidate_mode": candidate_mode,
                    "submission_role": (
                        "primary_submission"
                        if effort == submission_checkpoint
                        else "diagnostic_only"
                    ),
                    "candidates": candidate_payloads,
                    "score": score.to_dict(),
                }
            )
        metrics[str(effort)] = aggregate_arc_metrics(scores)
        query_details[str(effort)] = details
    return metrics, query_details


def _trajectory_reports(
    compact: np.ndarray,
    spikes: np.ndarray,
    voltage: np.ndarray,
    feedforward_current: np.ndarray,
    recurrent_current: np.ndarray,
    records: Sequence[_EvaluationRecord],
    color_rank: int,
    decoder_mode: DecoderMode,
    step_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if step_indices is None:
        steps = np.arange(compact.shape[0], dtype=np.int32)
    else:
        steps = np.asarray(step_indices, dtype=np.int32)
        if steps.shape != (compact.shape[0],):
            raise ValueError("step_indices must match the gathered trajectory")
    expanded = expand_decoder_logits(jnp.asarray(compact), color_rank, decoder_mode)
    height = np.asarray(expanded.height)
    width = np.asarray(expanded.width)
    colors = np.asarray(expanded.colors)
    reports: list[dict[str, object]] = []
    for query_index, record in enumerate(records):
        target = record.encoded.target
        assert target is not None
        report = analyze_latent_trajectory(
            height[:, query_index],
            width[:, query_index],
            colors[:, query_index],
            spikes[:, query_index],
            voltage[:, query_index],
            feedforward_current=feedforward_current[:, query_index],
            recurrent_current=recurrent_current[:, query_index],
            target=target.as_array(),
            task_id=record.task_key,
            query_index=record.encoded.query_index,
            step_indices=steps,
        )
        report["task_id"] = record.task_key
        report["query_index"] = record.encoded.query_index
        reports.append(report)

    pair_count = min(256, len(records)) if len(records) > 1 else 0
    pair_left = np.arange(pair_count, dtype=np.int32)
    pair_right = (pair_left * 131 + 17) % len(records) if pair_count else pair_left
    if pair_count:
        pair_right = np.where(
            pair_right == pair_left, (pair_right + 1) % len(records), pair_right
        )

    def distribution(values: np.ndarray) -> dict[str, float] | None:
        if values.size == 0:
            return None
        return {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
        }

    aggregate: list[dict[str, object]] = []
    for step_index, semantic_step in enumerate(steps):
        rows = [report["steps"][step_index] for report in reports]
        if pair_count:
            pair_spike = np.mean(
                spikes[step_index, pair_left] != spikes[step_index, pair_right], axis=1
            )
            scale = math.sqrt(spikes.shape[2])
            pair_voltage = (
                np.linalg.norm(
                    voltage[step_index, pair_left]
                    - voltage[step_index, pair_right],
                    axis=1,
                )
                / scale
            )
            pair_feedforward = (
                np.linalg.norm(
                    feedforward_current[step_index, pair_left]
                    - feedforward_current[step_index, pair_right],
                    axis=1,
                )
                / scale
            )
            pair_recurrent = (
                np.linalg.norm(
                    recurrent_current[step_index, pair_left]
                    - recurrent_current[step_index, pair_right],
                    axis=1,
                )
                / scale
            )
        else:
            pair_spike = np.asarray([], dtype=np.float64)
            pair_voltage = np.asarray([], dtype=np.float64)
            pair_feedforward = np.asarray([], dtype=np.float64)
            pair_recurrent = np.asarray([], dtype=np.float64)
        aggregate.append(
            {
                "step": int(semantic_step),
                "mean_firing_rate": float(
                    np.mean([row["firing_rate"] for row in rows])
                ),
                "mean_spike_count": float(
                    np.mean([row["spike_count"] for row in rows])
                ),
                "mean_voltage_l2": float(np.mean([row["voltage_l2"] for row in rows])),
                "mean_feedforward_current_l2": float(
                    np.mean([row["feedforward_current_l2"] for row in rows])
                ),
                "mean_recurrent_current_l2": float(
                    np.mean([row["recurrent_current_l2"] for row in rows])
                ),
                "mean_predictive_entropy": float(
                    np.mean([row["predictive_entropy"] for row in rows])
                ),
                "mean_changed_cell_fraction": (
                    None
                    if step_index == 0
                    else float(np.mean([row["changed_cell_fraction"] for row in rows]))
                ),
                "converged_fraction": float(
                    np.mean([row["converged"] for row in rows])
                ),
                "near_silence_fraction": float(
                    np.mean([row["near_silence"] for row in rows])
                ),
                "near_saturation_fraction": float(
                    np.mean([row["near_saturation"] for row in rows])
                ),
                "unique_state_hashes": len({row["state_sha256"] for row in rows}),
                "pair_sample_count": pair_count,
                "pair_sampling": "deterministic modular query pairs",
                "pairwise_spike_hamming_fraction": distribution(pair_spike),
                "pairwise_voltage_rms_distance": distribution(pair_voltage),
                "pairwise_feedforward_current_rms_distance": distribution(
                    pair_feedforward
                ),
                "pairwise_recurrent_current_rms_distance": distribution(pair_recurrent),
            }
        )
    return reports, aggregate


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _control_summary(
    name: str,
    intact: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    control: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    records: Sequence[_EvaluationRecord],
    color_rank: int,
    decoder_mode: DecoderMode,
    intact_metrics: dict[str, dict[str, object]],
    metadata: Sequence[dict[str, object]] | None = None,
    checkpoints: Sequence[int] = CHECKPOINTS,
    submission_checkpoint: int = SUBMISSION_CHECKPOINT,
    rule_proposals: Sequence[tuple[str, np.ndarray] | None] | None = None,
    intact_rule_proposals: Sequence[tuple[str, np.ndarray] | None] | None = None,
) -> dict[str, object]:
    if metadata is None:
        metadata = tuple({"available": True, "timing_matched": True} for _ in records)
    if len(metadata) != len(records):
        raise ValueError("control metadata must match the evaluation records")
    applicable = np.asarray(
        [bool(item.get("available", False)) for item in metadata], dtype=np.bool_
    )
    applicable_indices = np.flatnonzero(applicable)
    applicable_records = tuple(records[index] for index in applicable_indices)

    def subset(window: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return tuple(value[:, applicable_indices] for value in window)

    applicable_intact = subset(intact)
    applicable_control = subset(control)

    def subset_rules(
        proposals: Sequence[tuple[str, np.ndarray] | None] | None,
    ) -> tuple[tuple[str, np.ndarray] | None, ...] | None:
        if proposals is None:
            return None
        if len(proposals) != len(records):
            raise ValueError("control rule proposals must match the records")
        return tuple(proposals[index] for index in applicable_indices)

    if applicable_records:
        metrics, control_checkpoint_queries = _score_windows(
            applicable_control[0],
            applicable_records,
            color_rank,
            decoder_mode,
            checkpoints,
            submission_checkpoint,
            subset_rules(rule_proposals),
        )
        matched_intact_metrics, intact_checkpoint_queries = _score_windows(
            applicable_intact[0],
            applicable_records,
            color_rank,
            decoder_mode,
            checkpoints,
            submission_checkpoint,
            subset_rules(intact_rule_proposals),
        )
        if (
            len(applicable_records) == len(records)
            and matched_intact_metrics != intact_metrics
        ):
            raise ValueError("recomputed intact control metrics are inconsistent")
        candidate_match_count_by_effort: dict[str, int] = {}
        candidates_match_by_effort: dict[str, bool] = {}
        for effort in checkpoints:
            key = str(effort)
            intact_rows = intact_checkpoint_queries[key]
            control_rows = control_checkpoint_queries[key]
            if len(intact_rows) != len(control_rows):
                raise ValueError("matched control candidate rows differ in length")
            match_count = int(
                sum(
                    intact_row["candidates"] == control_row["candidates"]
                    for intact_row, control_row in zip(
                        intact_rows, control_rows, strict=True
                    )
                )
            )
            candidate_match_count_by_effort[key] = match_count
            candidates_match_by_effort[key] = match_count == len(intact_rows)
        decoded_candidates_match = bool(all(candidates_match_by_effort.values()))
    else:
        metrics = {}
        matched_intact_metrics = {}
        candidate_match_count_by_effort = {}
        candidates_match_by_effort = {}
        decoded_candidates_match = None

    if applicable_records:
        intact_spikes = applicable_intact[1]
        intact_voltage = applicable_intact[2]
        control_spikes = applicable_control[1]
        control_voltage = applicable_control[2]
        comparison = compare_control_trajectories(
            intact_spikes.transpose(0, 2, 1).reshape(intact_spikes.shape[0], -1),
            intact_voltage.transpose(0, 2, 1).reshape(intact_voltage.shape[0], -1),
            control_spikes.transpose(0, 2, 1).reshape(control_spikes.shape[0], -1),
            control_voltage.transpose(0, 2, 1).reshape(control_voltage.shape[0], -1),
            control_name=name,
            intact_scores=matched_intact_metrics,
            control_scores=metrics,
            intact_synaptic_currents={
                "feedforward": applicable_intact[3]
                .transpose(0, 2, 1)
                .reshape(applicable_intact[3].shape[0], -1),
                "recurrent": applicable_intact[4]
                .transpose(0, 2, 1)
                .reshape(applicable_intact[4].shape[0], -1),
            },
            control_synaptic_currents={
                "feedforward": applicable_control[3]
                .transpose(0, 2, 1)
                .reshape(applicable_control[3].shape[0], -1),
                "recurrent": applicable_control[4]
                .transpose(0, 2, 1)
                .reshape(applicable_control[4].shape[0], -1),
            },
        )
        comparison["state_byte_identical_by_step"] = [
            all(
                _array_bytes_equal(
                    applicable_intact[state_index][step_index],
                    applicable_control[state_index][step_index],
                )
                for state_index in range(1, 5)
            )
            for step_index in range(applicable_intact[0].shape[0])
        ]
    else:
        comparison = {
            "control_name": name,
            "available": False,
            "causally_null_at_measured_precision": None,
            "interpretation": f"{name} had no applicable evaluation queries.",
        }
    per_query_null = [
        bool(
            all(
                _array_bytes_equal(
                    intact[state_index][:, index],
                    control[state_index][:, index],
                )
                for state_index in range(1, 5)
            )
        )
        for index in applicable_indices
    ]
    timing_matched_applicable = int(
        sum(
            bool(item.get("timing_matched", False))
            for item, is_applicable in zip(metadata, applicable, strict=True)
            if is_applicable
        )
    )
    result: dict[str, object] = {
        "metrics_by_effort": metrics,
        "trajectory_comparison": comparison,
        "decoded_candidates_match_intact": decoded_candidates_match,
        "decoded_candidates_match_intact_by_effort": candidates_match_by_effort,
        "decoded_candidate_match_query_count_by_effort": (
            candidate_match_count_by_effort
        ),
        "causally_null_query_count": int(sum(per_query_null)),
        "byte_identical_query_count": int(sum(per_query_null)),
        "query_count": len(records),
        "applicable_query_count": int(applicable_indices.size),
        "available_query_count": int(applicable_indices.size),
        "unavailable_query_count": int(len(records) - applicable_indices.size),
        "timing_matched_applicable_query_count": timing_matched_applicable,
        "intervention_metadata": list(metadata),
    }
    result["timing_matched_query_count"] = int(
        sum(bool(item.get("timing_matched", False)) for item in metadata)
    )
    return result


def _state_tolerance_summary(
    intact: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    step_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    leading_shape = intact[0].shape[:2]
    for name, left, right in zip(
        ("compact", "spikes", "voltage", "feedforward_current", "recurrent_current"),
        intact,
        candidate,
        strict=True,
    ):
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError(f"matched {name} windows must have equal rank-3 shapes")
        if left.shape[:2] != leading_shape:
            raise ValueError(
                f"matched {name} windows must share step and query dimensions"
            )
    if leading_shape[1] < 1:
        raise ValueError("matched state windows must contain at least one query")
    if step_indices is None:
        indices = np.arange(intact[0].shape[0], dtype=np.int32)
    else:
        indices = np.asarray(step_indices, dtype=np.int32)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("step_indices must be a nonempty rank-1 sequence")
        if np.any(indices < 0) or np.any(indices >= intact[0].shape[0]):
            raise ValueError("step_indices exceed the matched windows")

    selected_spikes = intact[1][indices]
    candidate_spikes = candidate[1][indices]
    spike_difference = selected_spikes != candidate_spikes
    spike_hamming_by_query = np.count_nonzero(spike_difference, axis=(0, 2)).astype(
        np.int64
    )
    spike_hamming = int(np.sum(spike_hamming_by_query))
    numeric_names = (
        "compact_logits",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    )
    per_step_query_rms: dict[str, list[list[float]]] = {}
    per_query_maximum_rms: dict[str, list[float]] = {}
    per_query_maximum_absolute: dict[str, list[float]] = {}
    maximum_absolute: dict[str, float] = {}
    for name, state_index in zip(numeric_names, (0, 2, 3, 4), strict=True):
        delta = np.asarray(
            candidate[state_index][indices], dtype=np.float64
        ) - np.asarray(intact[state_index][indices], dtype=np.float64)
        rms = np.sqrt(np.mean(delta * delta, axis=2))
        query_maximum = np.max(rms, axis=0)
        absolute_query_maximum = np.max(np.abs(delta), axis=(0, 2))
        per_step_query_rms[name] = rms.tolist()
        per_query_maximum_rms[name] = query_maximum.tolist()
        per_query_maximum_absolute[name] = absolute_query_maximum.tolist()
        maximum_absolute[name] = float(np.max(np.abs(delta)))
    maximum_rms = {
        name: float(max(values)) for name, values in per_query_maximum_rms.items()
    }
    intact_dtype_by_state = {
        name: str(intact[state_index].dtype)
        for name, state_index in zip(numeric_names, (0, 2, 3, 4), strict=True)
    }
    candidate_dtype_by_state = {
        name: str(candidate[state_index].dtype)
        for name, state_index in zip(numeric_names, (0, 2, 3, 4), strict=True)
    }
    required_float32_dtypes = bool(
        all(value == "float32" for value in intact_dtype_by_state.values())
        and candidate_dtype_by_state == intact_dtype_by_state
    )
    state_byte_identical_by_query = np.asarray(
        [
            all(
                _array_bytes_equal(
                    intact[state_index][indices, query_index],
                    candidate[state_index][indices, query_index],
                )
                for state_index in range(1, 5)
            )
            for query_index in range(leading_shape[1])
        ],
        dtype=np.bool_,
    )
    compact_byte_identical_by_query = np.asarray(
        [
            _array_bytes_equal(
                intact[0][indices, query_index],
                candidate[0][indices, query_index],
            )
            for query_index in range(leading_shape[1])
        ],
        dtype=np.bool_,
    )
    within_tolerance_by_query = spike_hamming_by_query == 0
    for values in per_query_maximum_rms.values():
        within_tolerance_by_query &= np.asarray(values) <= STATE_RMS_TOLERANCE
    within_tolerance_by_query &= required_float32_dtypes
    within_tolerance_query_count = int(np.count_nonzero(within_tolerance_by_query))
    query_count = int(leading_shape[1])
    return {
        "evaluated_steps": indices.astype(int).tolist(),
        "query_count": query_count,
        "state_byte_identical": bool(np.all(state_byte_identical_by_query)),
        "state_byte_identical_by_query": state_byte_identical_by_query.tolist(),
        "state_byte_identical_by_step": [
            all(
                _array_bytes_equal(
                    intact[state_index][step_index],
                    candidate[state_index][step_index],
                )
                for state_index in range(1, 5)
            )
            for step_index in indices
        ],
        "compact_logits_byte_identical": bool(np.all(compact_byte_identical_by_query)),
        "compact_logits_byte_identical_by_query": (
            compact_byte_identical_by_query.tolist()
        ),
        "within_declared_tolerance": within_tolerance_query_count == query_count,
        "within_tolerance_by_query": within_tolerance_by_query.tolist(),
        "within_tolerance_query_count": within_tolerance_query_count,
        "declared_per_query_axis_rms_tolerance": STATE_RMS_TOLERANCE,
        "spike_hamming_count": spike_hamming,
        "spike_hamming_count_by_query": spike_hamming_by_query.tolist(),
        "per_step_query_rms": per_step_query_rms,
        "per_query_maximum_rms": per_query_maximum_rms,
        "maximum_rms": maximum_rms,
        "per_query_maximum_absolute": per_query_maximum_absolute,
        "maximum_absolute": maximum_absolute,
        "intact_dtype_by_state": intact_dtype_by_state,
        "candidate_dtype_by_state": candidate_dtype_by_state,
        "required_float32_dtypes": required_float32_dtypes,
    }


def _checkpoint_zero_gate_summary(
    intact_metrics: dict[str, dict[str, object]],
    control: dict[str, object],
    numeric_summary: dict[str, object],
) -> dict[str, bool]:
    state_within_tolerance = numeric_summary.get("within_declared_tolerance") is True
    candidate_matches = control.get("decoded_candidates_match_intact_by_effort")
    decoded_candidates_exact = bool(
        isinstance(candidate_matches, dict) and candidate_matches.get("0") is True
    )
    control_metrics = control.get("metrics_by_effort")
    metrics_exact = bool(
        isinstance(control_metrics, dict)
        and "0" in intact_metrics
        and "0" in control_metrics
        and control_metrics["0"] == intact_metrics["0"]
    )
    return {
        "state_within_tolerance": state_within_tolerance,
        "decoded_candidates_exact": decoded_candidates_exact,
        "metrics_exact": metrics_exact,
        "matched": bool(
            state_within_tolerance and decoded_candidates_exact and metrics_exact
        ),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _associative_evaluation_diagnostics(
    enabled: bool,
    intact: tuple[np.ndarray, np.ndarray, np.ndarray],
    controls: dict[
        str,
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray],
            Sequence[dict[str, object]],
        ],
    ],
) -> dict[str, object]:
    """Summarize bounded pairing-sensitive ``S_K``/read/workspace evidence."""
    if not enabled:
        return {
            "available": False,
            "complete": True,
            "reason": "legacy_reservoir_has_no_associative_state",
        }

    def validate(
        name: str, window: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        workspace, memory_read, context_memory = map(np.asarray, window)
        if workspace.ndim != 3 or memory_read.ndim != 3:
            raise ValueError(f"{name} workspace/read diagnostics must be rank three")
        if workspace.shape[:2] != memory_read.shape[:2]:
            raise ValueError(f"{name} workspace/read checkpoints must align")
        if context_memory.ndim != 3 or context_memory.shape[0] != workspace.shape[1]:
            raise ValueError(f"{name} context memory batch must align")
        if context_memory.shape[1] != context_memory.shape[2]:
            raise ValueError(f"{name} context memory must be square")
        if memory_read.shape[2] != context_memory.shape[1]:
            raise ValueError(f"{name} memory read width must match context memory")
        if context_memory.shape[1] < 1:
            raise ValueError(f"{name} associative diagnostics must have positive width")
        for array in (workspace, memory_read, context_memory):
            if (
                not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} associative diagnostics must be finite")
        return workspace, memory_read, context_memory

    intact_workspace, intact_read, intact_memory = validate("intact", intact)
    depth_count, query_count = intact_workspace.shape[:2]

    def l2_by_depth(value: np.ndarray) -> list[float]:
        norms = np.linalg.norm(value.astype(np.float64), axis=2)
        return np.mean(norms, axis=1).tolist()

    def comparison(
        name: str,
        window: tuple[np.ndarray, np.ndarray, np.ndarray],
        metadata: Sequence[dict[str, object]],
    ) -> dict[str, object]:
        workspace, memory_read, context_memory = validate(name, window)
        if workspace.shape != intact_workspace.shape:
            raise ValueError(f"{name} workspace shape must match intact")
        if memory_read.shape != intact_read.shape:
            raise ValueError(f"{name} memory read shape must match intact")
        if context_memory.shape != intact_memory.shape:
            raise ValueError(f"{name} context memory shape must match intact")
        if len(metadata) != query_count:
            raise ValueError(f"{name} metadata must match the query count")
        applicable = np.asarray(
            [bool(item.get("available", True)) for item in metadata],
            dtype=np.bool_,
        )
        applicable_count = int(np.count_nonzero(applicable))
        memory_delta = context_memory.astype(np.float64) - intact_memory.astype(
            np.float64
        )
        memory_l2 = np.linalg.norm(memory_delta.reshape(query_count, -1), axis=1)
        memory_rms = np.sqrt(np.mean(memory_delta * memory_delta, axis=(1, 2)))
        memory_changed = np.asarray(
            [
                not _array_bytes_equal(intact_memory[index], context_memory[index])
                for index in range(query_count)
            ],
            dtype=np.bool_,
        )

        def trajectory_delta(
            left: np.ndarray, right: np.ndarray
        ) -> tuple[list[float], list[int], int]:
            delta = right.astype(np.float64) - left.astype(np.float64)
            l2 = np.linalg.norm(delta, axis=2)
            changed = np.any(left != right, axis=2) & applicable[None, :]
            if applicable_count:
                mean_l2 = np.mean(l2[:, applicable], axis=1).tolist()
            else:
                mean_l2 = [0.0] * depth_count
            return (
                mean_l2,
                np.count_nonzero(changed, axis=1).astype(int).tolist(),
                int(np.count_nonzero(np.any(changed, axis=0))),
            )

        read_l2, read_changed, read_changed_any = trajectory_delta(
            intact_read, memory_read
        )
        workspace_l2, workspace_changed, workspace_changed_any = trajectory_delta(
            intact_workspace, workspace
        )
        zero_memory = np.asarray(
            [
                np.count_nonzero(context_memory[index]) == 0
                for index in range(query_count)
            ]
        )
        context_memory_exact = _array_bytes_equal(intact_memory, context_memory)
        memory_read_exact = _array_bytes_equal(intact_read, memory_read)
        workspace_exact = _array_bytes_equal(intact_workspace, workspace)
        return {
            "applicable_query_count": applicable_count,
            "context_memory_changed_applicable_query_count": int(
                np.count_nonzero(memory_changed & applicable)
            ),
            "context_memory_l2_by_query": memory_l2.tolist(),
            "context_memory_rms_by_query": memory_rms.tolist(),
            "context_memory_sha256_by_query": [
                _array_sha256(context_memory[index]) for index in range(query_count)
            ],
            "context_memory_zero_query_count": int(np.count_nonzero(zero_memory)),
            "memory_read_mean_l2_by_depth": read_l2,
            "memory_read_changed_query_count_by_depth": read_changed,
            "memory_read_changed_at_any_depth_applicable_query_count": (
                read_changed_any
            ),
            "workspace_carrier_mean_l2_by_depth": workspace_l2,
            "workspace_carrier_changed_query_count_by_depth": workspace_changed,
            "workspace_carrier_changed_at_any_depth_applicable_query_count": (
                workspace_changed_any
            ),
            "context_memory_byte_identical_to_intact": context_memory_exact,
            "memory_read_byte_identical_to_intact": memory_read_exact,
            "workspace_carrier_byte_identical_to_intact": workspace_exact,
            "byte_identical_to_intact": bool(
                context_memory_exact and memory_read_exact and workspace_exact
            ),
        }

    expected_controls = {
        "repeat_intact",
        "no_context",
        "shuffled_demonstrations",
        "slot_ablation",
    }
    if set(controls) != expected_controls:
        raise ValueError("associative controls are incomplete")
    control_reports = {
        name: comparison(name, *controls[name]) for name in sorted(expected_controls)
    }
    repeat_report = control_reports["repeat_intact"]
    repeat_exact = bool(
        repeat_report["context_memory_byte_identical_to_intact"]
        and repeat_report["memory_read_byte_identical_to_intact"]
    )
    no_context_zero = (
        control_reports["no_context"]["context_memory_zero_query_count"] == query_count
    )
    shuffled_report = control_reports["shuffled_demonstrations"]
    shuffled_applicable = int(shuffled_report["applicable_query_count"])
    shuffled_pairing_sensitive = bool(
        shuffled_applicable > 0
        and int(shuffled_report["context_memory_changed_applicable_query_count"])
        == shuffled_applicable
        and int(
            shuffled_report["memory_read_changed_at_any_depth_applicable_query_count"]
        )
        == shuffled_applicable
    )
    return {
        "available": True,
        "complete": bool(
            repeat_exact and no_context_zero and shuffled_pairing_sensitive
        ),
        "depth_count": depth_count,
        "query_count": query_count,
        "intact_context_memory_sha256_by_query": [
            _array_sha256(intact_memory[index]) for index in range(query_count)
        ],
        "intact_context_memory_frobenius_norm_by_query": np.linalg.norm(
            intact_memory.astype(np.float64).reshape(query_count, -1), axis=1
        ).tolist(),
        "intact_memory_read_mean_l2_by_depth": l2_by_depth(intact_read),
        "intact_workspace_carrier_mean_l2_by_depth": l2_by_depth(intact_workspace),
        "repeat_intact_exact": repeat_exact,
        "no_context_memory_exactly_zero": no_context_zero,
        "shuffled_pairing_sensitive_for_every_applicable_query": (
            shuffled_pairing_sensitive
        ),
        "controls": control_reports,
    }


def _compile_evaluation_arm(
    model: LatentWorkspaceModel,
    selected_indices: jax.Array,
    slots: jax.Array,
):
    """Compile the common device-side driver for one selected evaluation arm."""
    selected_indices = jnp.asarray(selected_indices)
    slots = jnp.asarray(slots)

    @brainstate.transform.jit(
        inline=False,
        name="example21_evaluation_arm",
    )
    def run_arm(events, advances, gates):
        packed = run_selected_packed_stream(
            model,
            events,
            selected_indices,
            reset=True,
            advance_gates=advances,
            ablation_slots=slots,
            ablation_gates=gates,
        )
        selected_context_memory = getattr(
            packed,
            "context_memory",
            jnp.broadcast_to(
                packed.final_context_memory,
                (packed.compact_logits.shape[0],) + packed.final_context_memory.shape,
            ),
        )
        return (
            packed.compact_logits,
            packed.spikes,
            packed.voltage,
            packed.feedforward_current,
            packed.recurrent_current,
            packed.memory_read,
            packed.final_context_memory,
            selected_context_memory,
        )

    return run_arm


def _expected_queries_by_task(data: _ExperimentData) -> dict[str, int]:
    expected = {
        _origin_task_key(origin, task_index): len(origin.task.test)
        for task_index, origin in enumerate(data.evaluation)
    }
    if len(expected) != len(data.evaluation):
        raise ValueError("target-free evaluation task identities must be unique")
    return expected


def _bank_task_slice(
    bank: ArcTargetFreeTaskBank, start: int, stop: int
) -> ArcTargetFreeTaskBank:
    """Return the sub-bank holding one contiguous group of tasks.

    Every bank leaf carries a leading task axis and padding widths are global to
    the bank, so a slice keeps every trailing shape and the compiled runner is
    reused across groups without recompiling.
    """
    return jax.tree.map(lambda leaf: leaf[start:stop], bank)


def _run_adaptation_in_task_groups(
    runner: Any, bank: ArcTargetFreeTaskBank, group_size: int
) -> ArcTaskBankAdaptationResult:
    """Adapt and infer one bounded group of tasks at a time.

    A single compiled call spanning all 400 tasks dispatched more GPU work than
    the local WDDM device tolerates: a complete run died with
    ``CUDA_ERROR_UNKNOWN`` at the first host synchronisation after the whole-bank
    call. The runner restores parameters and optimizer state per task, so
    grouping tasks is arithmetically identical and bounds each dispatch.
    """
    task_count = int(np.asarray(bank.query_valid).shape[0])
    if group_size <= 0 or group_size >= task_count:
        return runner(bank)
    groups = []
    for start in range(0, task_count, group_size):
        stop = min(start + group_size, task_count)
        result = runner(_bank_task_slice(bank, start, stop))
        groups.append(jax.tree.map(np.asarray, result))
    return ArcTaskBankAdaptationResult(
        *(
            np.concatenate([getattr(group, name) for group in groups])
            for name in ArcTaskBankAdaptationResult._fields
        )
    )


def _task_local_adaptation_evaluation(
    trained_model: LatentWorkspaceModel,
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    device: jax.Device,
    records: Sequence[_EvaluationRecord],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    """Adapt one isolated model per ARC task and score target-free checkpoints.

    Parameters
    ----------
    trained_model
        Shared post-training batch-one model.
    data, config, row_config, device
        Resolved experiment inputs and execution device.
    records
        Ordered official-query metadata. Targets remain here and are joined
        only after the compiled runner returns model logits.

    Returns
    -------
    tuple
        Metrics, per-query scoring records, and JSON-safe adaptation evidence.
    """

    origins = _selected_evaluation_origins(data, config)
    bank = build_arc_target_free_task_bank(
        tuple(_without_official_test_targets(origin.task) for origin in origins),
        row_config,
        latent_steps=config.latent_steps,
    )
    model = _make_model(config, row_config, batch_size=1, device=device)
    _copy_parameters(trained_model, model)
    base_parameters = snapshot_parameters(model)
    before = _tree_digest(parameter_snapshot(model))
    learner = compile_pp_prop(model)
    adaptation_rate = config.adaptation_learning_rate or config.learning_rate
    optimizer = braintools.optim.Adam(lr=adaptation_rate)
    optimizer.register_trainable_weights(learner.param_states)
    runner = compile_arc_task_local_adaptation_runner(
        model,
        learner,
        optimizer,
        base_parameters=base_parameters,
        row_config=row_config,
        latent_steps=config.latent_steps,
        clip_norm=config.clip_norm,
        epochs=config.adaptation_epochs,
        update_schedule=config.adaptation_update_schedule,
    )
    started = time.perf_counter()
    result = _run_adaptation_in_task_groups(runner, bank, config.adaptation_task_group)
    wall_seconds = time.perf_counter() - started
    after = _tree_digest(parameter_snapshot(model))
    valid = np.asarray(result.query_valid, dtype=np.bool_)
    recorded = np.asarray(result.checkpoint_recorded, dtype=np.bool_)
    expected_recorded = np.broadcast_to(valid[..., None], recorded.shape)
    if not np.array_equal(recorded, expected_recorded):
        raise ValueError("task-local adaptation checkpoint validity is inconsistent")
    task_ordinals = np.broadcast_to(
        np.asarray(bank.task_ordinals)[:, None], valid.shape
    )[valid]
    query_ordinals = np.asarray(bank.query_ordinals)[valid]
    expected_ordinals = np.asarray(
        [(record.encoded.task_index, record.encoded.query_index) for record in records],
        dtype=np.int32,
    )
    actual_ordinals = np.stack((task_ordinals, query_ordinals), axis=-1)
    if not np.array_equal(actual_ordinals, expected_ordinals):
        raise ValueError("adapted task/query order does not match evaluation records")
    flattened = np.asarray(result.checkpoint_outputs)[valid]
    if flattened.shape[0] != len(records):
        raise ValueError("adapted query order does not match evaluation records")
    checkpoint_logits = np.transpose(flattened, (1, 0, 2))
    metrics, checkpoint_queries = _score_checkpoint_logits(
        checkpoint_logits,
        records,
        config.color_rank,
        config.decoder_mode,
    )
    fold_applied = np.asarray(result.fold_applied, dtype=np.bool_)
    fold_losses = np.asarray(result.fold_losses, dtype=np.float64)
    fold_count = (
        int(np.count_nonzero(np.asarray(bank.fold_inputs.fold_valid)))
        * config.adaptation_epochs
    )
    applied_fold_count = int(np.count_nonzero(fold_applied))
    evidence = {
        "performed": True,
        "mode": "compiled_task_local_pp_prop_leave_one_out",
        "learning_rate": float(adaptation_rate),
        "epochs": int(config.adaptation_epochs),
        "task_group_size": int(config.adaptation_task_group),
        "target_free_query_bank": True,
        "target_free_official_query_bank": True,
        "task_count": int(valid.shape[0]),
        "query_count": int(np.count_nonzero(valid)),
        "fold_capacity": int(fold_applied.shape[1]),
        "fold_count": fold_count,
        "distinct_fold_count": int(
            np.count_nonzero(np.asarray(bank.fold_inputs.fold_valid))
        ),
        "applied_fold_count": applied_fold_count,
        "fold_applied": fold_applied.tolist(),
        "fold_losses": fold_losses.tolist(),
        "bank_bytes": bank.projected_bytes,
        "semantic_checkpoints": list(config.checkpoints),
        "checkpoint_output_shape": list(result.checkpoint_outputs.shape),
        "all_valid_checkpoints_recorded": bool(
            np.array_equal(recorded, expected_recorded)
        ),
        "base_parameter_sha256": before,
        "restored_parameter_sha256": after,
        "base_parameters_restored": before == after,
        "optimizer_step_count_after_cleanup": int(optimizer.step_count.value),
        "compiler": _compiler_evidence(learner),
        "wall_seconds": wall_seconds,
    }
    if (
        before != after
        or int(optimizer.step_count.value) != 0
        or applied_fold_count != fold_count
    ):
        raise ValueError("task-local adaptation did not restore shared state")
    return metrics, checkpoint_queries, evidence


def _model_only_completion_report(
    submission_rows: Sequence[Mapping[str, object]],
    submission_metrics: Mapping[str, object],
    data: _ExperimentData,
    config: ExperimentConfig,
) -> dict[str, object]:
    expected = _expected_queries_by_task(data)
    eligible = bool(
        not config.smoke
        and not config.structural_only
        and not data.plumbing_only
        and config.evaluation_task_limit is None
        and config.decoder_mode in ("row_refinement", "latent_row_decode")
        and len(expected) == 400
    )
    if eligible:
        report = dict(assess_model_only_completion(submission_rows, expected))
        report.update(
            {
                "eligible_for_completion": True,
                "eligibility_reason": None,
                "submission_checkpoint": config.submission_checkpoint,
                "submission_policy": SUBMISSION_POLICY,
            }
        )
        return report

    task_values = submission_metrics.get("tasks", {})
    tasks = task_values if isinstance(task_values, Mapping) else {}
    exact_task_count = sum(
        1
        for value in tasks.values()
        if isinstance(value, Mapping) and value.get("pass_at_2") is True
    )
    observed_rate = submission_metrics.get("strict_task_pass_at_2", 0.0)
    return {
        "primary_candidate_mode": "model_only",
        "eligible_for_completion": False,
        "eligibility_reason": (
            "completion requires a complete 400-task non-smoke row-refinement run"
        ),
        "submission_checkpoint": config.submission_checkpoint,
        "submission_policy": SUBMISSION_POLICY,
        "required_task_count": 400,
        "evaluated_task_count": int(submission_metrics.get("task_count", 0)),
        "evaluated_query_count": int(submission_metrics.get("query_count", 0)),
        "required_exact_task_count": 160,
        "exact_task_count": exact_task_count,
        "strict_task_pass_at_2": float(observed_rate),
        "passed": False,
        "tasks": dict(tasks),
    }


def _submitted_completion_report(
    submission_metrics: Mapping[str, object], config: ExperimentConfig
) -> dict[str, object]:
    """Summarize the submitted channel's exact task count without purity gates.

    ``model_only_completion`` stays a strict model-only measurement. This report
    describes whatever the run actually submitted, which under
    ``rule_then_model`` includes demonstration-verified rule candidates.

    Parameters
    ----------
    submission_metrics
        Aggregate metrics at the submission checkpoint.
    config
        Run configuration, read for the active candidate mode.

    Returns
    -------
    dict
        JSON-safe submitted-channel counts and the policy that produced them.
    """

    task_values = submission_metrics.get("tasks", {})
    tasks = task_values if isinstance(task_values, Mapping) else {}
    exact_at_1 = sum(
        1
        for value in tasks.values()
        if isinstance(value, Mapping) and value.get("pass_at_1") is True
    )
    exact_at_2 = sum(
        1
        for value in tasks.values()
        if isinstance(value, Mapping) and value.get("pass_at_2") is True
    )
    query_count = int(submission_metrics.get("query_count", 0))
    return {
        "primary_candidate_mode": config.primary_candidate_mode,
        "submission_policy": _submission_policy_name(config.primary_candidate_mode),
        "submission_checkpoint": config.submission_checkpoint,
        "evaluated_task_count": int(submission_metrics.get("task_count", 0)),
        "evaluated_query_count": query_count,
        "exact_query_count_at_1": round(
            float(submission_metrics.get("query_pass_at_1", 0.0)) * query_count
        ),
        "exact_query_count_at_2": round(
            float(submission_metrics.get("query_pass_at_2", 0.0)) * query_count
        ),
        "exact_task_count_at_1": exact_at_1,
        "exact_task_count_at_2": exact_at_2,
        "strict_task_pass_at_1": float(
            submission_metrics.get("strict_task_pass_at_1", 0.0)
        ),
        "strict_task_pass_at_2": float(
            submission_metrics.get("strict_task_pass_at_2", 0.0)
        ),
    }


def _evaluation_offsets(config: ExperimentConfig) -> np.ndarray:
    """Return sparse primary checkpoints or dense diagnostic trajectory steps."""
    if config.evaluation_controls:
        return np.arange(config.latent_steps + 1, dtype=np.int32)
    return np.asarray(config.checkpoints, dtype=np.int32)


def _evaluate(
    trained_model: LatentWorkspaceModel,
    data: _ExperimentData,
    config: ExperimentConfig,
    row_config: RowEventConfig,
    device: jax.Device,
) -> dict[str, object]:
    checkpoints = config.checkpoints
    training_efforts = config.training_efforts
    submission_checkpoint = config.submission_checkpoint
    records = _evaluation_records(data, config, row_config)
    adaptation_enabled = (
        config.task_local_adaptation
        and config.decoder_mode in ("row_refinement", "latent_row_decode")
        and not config.structural_only
    )
    if adaptation_enabled:
        primary_metrics, primary_checkpoint_queries, adaptation_evidence = (
            _task_local_adaptation_evaluation(
                trained_model,
                data,
                config,
                row_config,
                device,
                records,
            )
        )
    else:
        primary_metrics = None
        primary_checkpoint_queries = None
        adaptation_evidence = {
            "performed": False,
            "mode": "disabled",
            "reason": (
                "structural_only"
                if config.structural_only
                else "disabled_by_configuration"
                if not config.task_local_adaptation
                else "decoder_mode_is_not_row_refinement"
            ),
            "target_free_query_bank": True,
            "target_free_official_query_bank": True,
        }
    batch_size = len(records)
    model = _make_model(config, row_config, batch_size=batch_size, device=device)
    _copy_parameters(trained_model, model)
    before = _tree_digest(parameter_snapshot(model))
    intact_events, intact_advances, query_stops, intact_meta = _arm_sequences(
        records, config, row_config, arm="intact", source_tasks=data.evaluation
    )
    if config.evaluation_controls:
        no_context_events, no_context_advances, no_context_stops, no_context_meta = (
            _arm_sequences(
                records,
                config,
                row_config,
                arm="no_context",
                source_tasks=data.evaluation,
            )
        )
        shuffled_events, shuffled_advances, shuffled_stops, shuffled_meta = (
            _arm_sequences(
                records,
                config,
                row_config,
                arm="shuffled",
                source_tasks=data.evaluation,
            )
        )
        if not np.array_equal(query_stops, no_context_stops) or not np.array_equal(
            query_stops, shuffled_stops
        ):
            raise ValueError("control query boundaries are not matched")
    else:
        no_context_events = np.empty((0,), dtype=np.float32)
        no_context_advances = np.empty((0,), dtype=np.bool_)
        no_context_stops = query_stops
        no_context_meta = []
        shuffled_events = np.empty((0,), dtype=np.float32)
        shuffled_advances = np.empty((0,), dtype=np.bool_)
        shuffled_stops = query_stops
        shuffled_meta = []

    protocol_v2 = config.decoder_mode == "latent_row_decode"
    base_intact_events = intact_events
    base_intact_advances = intact_advances
    if protocol_v2:
        intact_protocol = build_batched_protocol_v2_arm(
            base_intact_events, base_intact_advances, query_stops
        )
        intact_events = np.asarray(intact_protocol.events)
        intact_advances = intact_protocol.gates
        selected_indices = np.asarray(
            intact_protocol.metadata["checkpoint_indices"], dtype=np.int32
        )
        if config.evaluation_controls:
            no_context_protocol = build_batched_protocol_v2_arm(
                no_context_events, no_context_advances, no_context_stops
            )
            shuffled_protocol = build_batched_protocol_v2_arm(
                shuffled_events, shuffled_advances, shuffled_stops
            )
            no_context_events = np.asarray(no_context_protocol.events)
            no_context_advances = no_context_protocol.gates
            shuffled_events = np.asarray(shuffled_protocol.events)
            shuffled_advances = shuffled_protocol.gates
    slots = np.full((batch_size,), config.ablation_slot, dtype=np.int32)
    inactive_gates = np.zeros((intact_events.shape[0], batch_size), dtype=np.bool_)
    evaluation_offsets = _evaluation_offsets(config)
    if not protocol_v2:
        selected_indices = query_stops[None, :] - 1 + evaluation_offsets[:, None]
    run_device_arm = _compile_evaluation_arm(
        model,
        jnp.asarray(selected_indices),
        jnp.asarray(slots),
    )
    arm_wall_seconds: dict[str, float] = {}

    def run_arm(
        name: str,
        events: np.ndarray,
        advances: np.ndarray | StepGates,
        gates: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        arm_started = time.perf_counter()
        packed = run_device_arm(
            jnp.asarray(events),
            advances if isinstance(advances, StepGates) else jnp.asarray(advances),
            jnp.asarray(gates),
        )
        window = tuple(np.asarray(value) for value in packed)
        arm_wall_seconds[name] = time.perf_counter() - arm_started
        del packed
        physical = window[:5]
        associative = (physical[2], window[5], window[6])
        return physical, associative

    intact, intact_associative = run_arm(
        "intact", intact_events, intact_advances, inactive_gates
    )
    protocol_evidence: dict[str, object] | None = None
    if protocol_v2:
        boundaries = intact_protocol.metadata["per_example_boundaries"]
        audit_rows: list[list[int]] = []
        for effort in checkpoints:
            audit_rows.append(
                [
                    int(item[f"decode_r{effort}_start"]) - 1
                    for item in boundaries
                ]
            )
            audit_rows.append(
                [int(item[f"decode_r{effort}_stop"]) - 1 for item in boundaries]
            )
        audit_driver = _compile_evaluation_arm(
            model,
            jnp.asarray(audit_rows, dtype=jnp.int32),
            jnp.asarray(slots),
        )
        audit_window = tuple(
            np.asarray(value)
            for value in audit_driver(
                jnp.asarray(intact_events), intact_advances, jnp.asarray(inactive_gates)
            )
        )
        audit_voltage = audit_window[2]
        audit_memory = audit_window[7]
        decoder_hashes: dict[str, dict[str, object]] = {}
        for index, effort in enumerate(checkpoints):
            before_index = 2 * index
            after_index = before_index + 1
            h_before = _array_sha256(audit_voltage[before_index])
            h_after = _array_sha256(audit_voltage[after_index])
            s_before = _array_sha256(audit_memory[before_index])
            s_after = _array_sha256(audit_memory[after_index])
            decoder_hashes[str(effort)] = {
                "physical_state_before_sha256": h_before,
                "physical_state_after_sha256": h_after,
                "associative_memory_before_sha256": s_before,
                "associative_memory_after_sha256": s_after,
                "physical_state_unchanged": h_before == h_after,
                "associative_memory_unchanged": s_before == s_after,
            }

        def gate_count(gates: StepGates) -> dict[str, int]:
            return {
                "advance_physics": int(np.count_nonzero(gates.advance_physics)),
                "latent_update": int(np.count_nonzero(gates.latent_update)),
                "decode_row": int(np.count_nonzero(gates.decode_row)),
                "answer_feedback": int(np.count_nonzero(gates.answer_feedback)),
                "recurrent_enabled": int(np.count_nonzero(gates.recurrent_enabled)),
            }

        reasoning_or_decode = np.asarray(intact_advances.latent_update) | np.asarray(
            intact_advances.decode_row
        )
        no_context_prequery = (
            {
                name: int(
                    sum(
                        np.count_nonzero(np.asarray(value)[: int(stop), batch])
                        for batch, stop in enumerate(no_context_stops.tolist())
                    )
                )
                for name, value in {
                    "latent_update": no_context_protocol.gates.latent_update,
                    "decode_row": no_context_protocol.gates.decode_row,
                    "answer_feedback": no_context_protocol.gates.answer_feedback,
                }.items()
            }
            if config.evaluation_controls
            else None
        )
        protocol_evidence = {
            "version": PROTOCOL_VERSION,
            "efforts": list(checkpoints),
            "recurrent_reasoning_ticks": [0, 30, 60],
            "decoder_rows_per_effort": 30,
            "gate_counts": gate_count(intact_advances),
            "decoder_state_hashes": decoder_hashes,
            "decoder_state_immutable": all(
                item["physical_state_unchanged"]
                and item["associative_memory_unchanged"]
                for item in decoder_hashes.values()
            ),
            "primary_reasoning_and_decoder_external_input_l2": float(
                np.linalg.norm(intact_events[reasoning_or_decode].astype(np.float64))
            ),
            "primary_answer_feedback_norm": 0.0,
            "no_context_prequery_gate_counts": no_context_prequery,
            "no_context_prequery_memory_norm": (
                0.0 if config.evaluation_controls else None
            ),
            "candidate_policy": _submission_policy_name(
                config.primary_candidate_mode
            ),
        }
        del audit_window, audit_voltage, audit_memory
    model_only_metrics, model_only_queries = _score_windows(
        intact[0], records, config.color_rank, config.decoder_mode,
        checkpoints, submission_checkpoint,
    )
    intact_rules: tuple[tuple[str, np.ndarray] | None, ...] | None = None
    shuffled_rules: tuple[tuple[str, np.ndarray] | None, ...] | None = None
    no_context_rules: tuple[tuple[str, np.ndarray] | None, ...] | None = None
    if config.primary_candidate_mode == "rule_then_model":
        clear_rule_cache()
        intact_rules = _rule_proposals(records, data.evaluation, arm="intact")
        if config.evaluation_controls:
            shuffled_rules = _rule_proposals(
                records, data.evaluation, arm="shuffled"
            )
            no_context_rules = _rule_proposals(
                records, data.evaluation, arm="no_context"
            )
        frozen_metrics, frozen_checkpoint_queries = _score_windows(
            intact[0], records, config.color_rank, config.decoder_mode,
            checkpoints, submission_checkpoint, intact_rules,
        )
    else:
        frozen_metrics, frozen_checkpoint_queries = (
            model_only_metrics,
            model_only_queries,
        )
    if primary_metrics is None or primary_checkpoint_queries is None:
        primary_metrics = frozen_metrics
        primary_checkpoint_queries = frozen_checkpoint_queries
    completion_report = _model_only_completion_report(
        model_only_queries[str(submission_checkpoint)],
        model_only_metrics[str(submission_checkpoint)],
        data,
        config,
    )
    submitted_completion = _submitted_completion_report(
        primary_metrics[str(submission_checkpoint)], config
    )
    channel_attribution = _channel_attribution(primary_checkpoint_queries)
    model_only_attribution = _channel_attribution(model_only_queries)
    model_only_input_echo = _input_echo_summary(model_only_queries)
    trajectory_steps = (
        np.asarray(checkpoints, dtype=np.int32)
        if protocol_v2
        else evaluation_offsets
        if intact[0].shape[0] == evaluation_offsets.size
        else None
    )
    trajectories, aggregate_trajectory = _trajectory_reports(
        *intact,
        records,
        config.color_rank,
        config.decoder_mode,
        trajectory_steps,
    )

    if not config.evaluation_controls:
        after = _tree_digest(parameter_snapshot(model))
        frozen_diagnostic = {
            "role": "primary_shared_model",
            "trajectory_role": "primary_shared_model",
            "metrics_by_effort": frozen_metrics,
            "checkpoint_queries": frozen_checkpoint_queries,
            "query_trajectories": trajectories,
            "aggregate_trajectory": aggregate_trajectory,
            "associative_memory_diagnostics": {
                "available": False,
                "complete": True,
                "reason": "evaluation_controls_disabled",
            },
            "determinism": {
                "enabled": False,
                "reason": "evaluation_controls_disabled",
            },
            "controls": {
                "enabled": False,
                "reason": "disabled_by_configuration",
            },
            "execution": {
                "arm_order": ["intact"],
                "selected_arm_driver": "brainstate.transform.jit",
                "jit_name": "example21_evaluation_arm",
                "sequential_separate_arms": False,
                "repeat_intact_cached": False,
                "wall_seconds_by_arm": arm_wall_seconds,
                "gathered_steps": (
                    list(checkpoints) if protocol_v2 else evaluation_offsets.tolist()
                ),
                "gathered_step_count": (
                    len(checkpoints) if protocol_v2 else int(evaluation_offsets.size)
                ),
            },
        }
        return {
            "query_count": batch_size,
            "task_count": len({record.task_key for record in records}),
            "same_frozen_parameter_bytes": before == after,
            "parameter_sha256_before": before,
            "parameter_sha256_after": after,
            "primary_candidate_mode": config.primary_candidate_mode,
            "primary_evaluation_mode": "shared_model_frozen",
            "metrics_by_effort": primary_metrics,
            "submission_policy": {
                "name": _submission_policy_name(config.primary_candidate_mode),
                "submission_checkpoint": config.submission_checkpoint,
                "completed_sweep_checkpoints": list(training_efforts),
                "candidate_budget": 2,
                "fallback": "latest_checkpoint_factorized_global_runner_up",
                "target_free_selection": True,
                "rule_channel_enabled": (
                    config.primary_candidate_mode == "rule_then_model"
                ),
            },
            "protocol": {
                **(protocol_evidence or {}),
                "output_complete_at_every_effort": all(
                    int(primary_metrics[str(effort)]["query_count"]) == batch_size
                    for effort in checkpoints
                ),
            },
            "model_only_completion": completion_report,
            "submitted_completion": submitted_completion,
            "channel_attribution": channel_attribution,
            "model_only_metrics_by_effort": model_only_metrics,
            "model_only_channel_attribution": model_only_attribution,
            "model_only_input_echo_by_effort": model_only_input_echo,
            "checkpoint_queries": primary_checkpoint_queries,
            "task_local_adaptation": adaptation_evidence,
            "physical_diagnostic_role": "primary_shared_model",
            "frozen_no_adaptation": frozen_diagnostic,
            "query_trajectories": trajectories,
            "aggregate_trajectory": aggregate_trajectory,
            "associative_memory_diagnostics": frozen_diagnostic[
                "associative_memory_diagnostics"
            ],
            "determinism": frozen_diagnostic["determinism"],
            "controls": {
                "enabled": False,
                "reason": "disabled_by_configuration",
                "truncation": {
                    "enabled": False,
                    "checkpoints": list(checkpoints),
                    "uses_one_continuous_intact_trajectory": True,
                },
            },
            "execution": frozen_diagnostic["execution"],
        }

    repeat_intact, repeat_associative = run_arm(
        "repeat_intact", intact_events, intact_advances, inactive_gates
    )
    repeat_result = _control_summary(
        "repeat_intact",
        intact,
        repeat_intact,
        records,
        config.color_rank,
        config.decoder_mode,
        frozen_metrics,
        intact_meta,
        checkpoints,
        submission_checkpoint,
        intact_rules,
        intact_rules,
    )
    repeat_match = _state_tolerance_summary(intact, repeat_intact)
    if protocol_v2:
        repeat_match["evaluated_steps"] = list(checkpoints)
    repeat_metrics_exact = repeat_result["metrics_by_effort"] == frozen_metrics
    repeat_predictions_exact = bool(repeat_result["decoded_candidates_match_intact"])
    repeat_reproducible = bool(
        repeat_match["within_declared_tolerance"]
        and repeat_metrics_exact
        and repeat_predictions_exact
    )
    repeat_comparison = repeat_result["trajectory_comparison"]
    repeat_comparison["state_byte_identical_all_steps"] = repeat_match[
        "state_byte_identical"
    ]
    repeat_comparison["state_byte_identical_by_step"] = repeat_match[
        "state_byte_identical_by_step"
    ]
    repeat_comparison["within_declared_reproducibility_tolerance"] = repeat_reproducible
    repeat_comparison["causally_null_at_measured_precision"] = repeat_reproducible
    repeat_comparison["interpretation"] = (
        "repeat_intact is reproducible within the declared tolerance; spikes, "
        "decoded candidates, and metrics are exact while logit/state/current RMS "
        f"differences are at most {STATE_RMS_TOLERANCE:.1e} on the feature axis "
        "for logits and neuron axis for physical state at every checkpoint/query."
        if repeat_reproducible
        else "repeat_intact exceeded the declared reproducibility tolerance."
    )
    repeat_result["causally_null_query_count"] = repeat_match[
        "within_tolerance_query_count"
    ]
    repeat_result["within_tolerance_query_count"] = repeat_match[
        "within_tolerance_query_count"
    ]
    del repeat_intact

    no_context, no_context_associative = run_arm(
        "no_context", no_context_events, no_context_advances, inactive_gates
    )
    no_context_result = _control_summary(
        "no_context",
        intact,
        no_context,
        records,
        config.color_rank,
        config.decoder_mode,
        frozen_metrics,
        no_context_meta,
        checkpoints,
        submission_checkpoint,
        no_context_rules,
        intact_rules,
    )
    del no_context

    shuffled, shuffled_associative = run_arm(
        "shuffled_demonstrations",
        shuffled_events,
        shuffled_advances,
        inactive_gates,
    )
    shuffled_result = _control_summary(
        "shuffled_demonstrations",
        intact,
        shuffled,
        records,
        config.color_rank,
        config.decoder_mode,
        frozen_metrics,
        shuffled_meta,
        checkpoints,
        submission_checkpoint,
        shuffled_rules,
        intact_rules,
    )
    del shuffled

    if protocol_v2:
        state_hold_protocol = build_batched_protocol_v2_arm(
            base_intact_events,
            base_intact_advances,
            query_stops,
            control="state_hold",
        )
        recurrent_lesion_protocol = build_batched_protocol_v2_arm(
            base_intact_events,
            base_intact_advances,
            query_stops,
            control="recurrent_lesion",
        )
        state_hold, _ = run_arm(
            "state_hold",
            np.asarray(state_hold_protocol.events),
            state_hold_protocol.gates,
            inactive_gates,
        )
        recurrent_lesion, _ = run_arm(
            "recurrent_lesion",
            np.asarray(recurrent_lesion_protocol.events),
            recurrent_lesion_protocol.gates,
            inactive_gates,
        )
        state_hold_result = _control_summary(
            "state_hold",
            intact,
            state_hold,
            records,
            config.color_rank,
            config.decoder_mode,
            frozen_metrics,
            intact_meta,
            checkpoints,
            submission_checkpoint,
            intact_rules,
            intact_rules,
        )
        state_hold_result["r30_r60_equal_r0"] = bool(
            np.array_equal(state_hold[0][0], state_hold[0][1])
            and np.array_equal(state_hold[0][0], state_hold[0][2])
        )
        recurrent_lesion_result = _control_summary(
            "recurrent_lesion",
            intact,
            recurrent_lesion,
            records,
            config.color_rank,
            config.decoder_mode,
            frozen_metrics,
            intact_meta,
            checkpoints,
            submission_checkpoint,
            intact_rules,
            intact_rules,
        )
        del state_hold, recurrent_lesion
    else:
        state_hold_result = {
            "status": "not_run",
            "required": False,
            "reason": "legacy decoder",
        }
        recurrent_lesion_result = dict(state_hold_result)

    gates = inactive_gates.copy()
    intervention_steps = query_stops
    if protocol_v2:
        intervention_steps = selected_indices[0] + 1
    valid_intervention = intervention_steps < gates.shape[0]
    gates[
        intervention_steps[valid_intervention],
        np.arange(batch_size)[valid_intervention],
    ] = True
    ablated, ablated_associative = run_arm(
        "slot_ablation", intact_events, intact_advances, gates
    )
    ablation_result = _control_summary(
        f"slot_ablation_{config.ablation_slot}",
        intact,
        ablated,
        records,
        config.color_rank,
        config.decoder_mode,
        frozen_metrics,
        intact_meta,
        checkpoints,
        submission_checkpoint,
        intact_rules,
        intact_rules,
    )
    pre_intervention_match = _state_tolerance_summary(
        intact, ablated, step_indices=(0,)
    )
    ablation_checkpoint_zero = _checkpoint_zero_gate_summary(
        frozen_metrics, ablation_result, pre_intervention_match
    )
    associative_diagnostics = _associative_evaluation_diagnostics(
        config.context_memory_width > 0,
        intact_associative,
        {
            "repeat_intact": (repeat_associative, intact_meta),
            "no_context": (no_context_associative, no_context_meta),
            "shuffled_demonstrations": (shuffled_associative, shuffled_meta),
            "slot_ablation": (ablated_associative, intact_meta),
        },
    )
    del (
        intact_associative,
        repeat_associative,
        no_context_associative,
        shuffled_associative,
        ablated_associative,
    )
    del ablated
    after = _tree_digest(parameter_snapshot(model))
    result = {
        "query_count": batch_size,
        "task_count": len({record.task_key for record in records}),
        "same_frozen_parameter_bytes": before == after,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "primary_candidate_mode": config.primary_candidate_mode,
        "primary_evaluation_mode": (
            adaptation_evidence["mode"]
            if adaptation_evidence["performed"]
            else "frozen_no_adaptation"
        ),
        "metrics_by_effort": primary_metrics,
        "submission_policy": {
            "name": _submission_policy_name(config.primary_candidate_mode),
            "submission_checkpoint": submission_checkpoint,
            "completed_sweep_checkpoints": list(training_efforts),
            "candidate_budget": 2,
            "fallback": "latest_checkpoint_factorized_global_runner_up",
            "target_free_selection": True,
            "rule_channel_enabled": (
                config.primary_candidate_mode == "rule_then_model"
            ),
        },
        "protocol": {
            **(protocol_evidence or {}),
            "output_complete_at_every_effort": all(
                int(primary_metrics[str(effort)]["query_count"]) == batch_size
                for effort in checkpoints
            ),
        },
        "model_only_completion": completion_report,
        "submitted_completion": submitted_completion,
        "channel_attribution": channel_attribution,
        "model_only_metrics_by_effort": model_only_metrics,
        "model_only_channel_attribution": model_only_attribution,
        "model_only_input_echo_by_effort": model_only_input_echo,
        "checkpoint_queries": primary_checkpoint_queries,
        "task_local_adaptation": adaptation_evidence,
        "physical_diagnostic_role": "frozen_no_adaptation_diagnostic",
        "frozen_no_adaptation": {
            "role": "diagnostic_control_not_primary_submission",
            "trajectory_role": "frozen_no_adaptation_diagnostic",
            "metrics_by_effort": frozen_metrics,
            "checkpoint_queries": frozen_checkpoint_queries,
        },
        "query_trajectories": trajectories,
        "aggregate_trajectory": aggregate_trajectory,
        "associative_memory_diagnostics": associative_diagnostics,
        "determinism": {
            "same_control_capable_execution_path": True,
            "state_rms_tolerance": STATE_RMS_TOLERANCE,
            "spike_tolerance": "exact identity",
            "metric_absolute_tolerance": 0.0,
            "repeat_intact_state_byte_identical": repeat_match["state_byte_identical"],
            "repeat_intact_compact_logits_byte_identical": repeat_match[
                "compact_logits_byte_identical"
            ],
            "repeat_intact_within_tolerance": repeat_match["within_declared_tolerance"],
            "repeat_intact_metrics_exact": repeat_metrics_exact,
            "repeat_intact_decoded_candidates_exact": repeat_predictions_exact,
            "repeat_intact_numeric_evidence": repeat_match,
            "slot_ablation_checkpoint_zero_byte_identical": pre_intervention_match[
                "state_byte_identical"
            ],
            "slot_ablation_checkpoint_zero_state_within_tolerance": (
                ablation_checkpoint_zero["state_within_tolerance"]
            ),
            "slot_ablation_checkpoint_zero_decoded_candidates_exact": (
                ablation_checkpoint_zero["decoded_candidates_exact"]
            ),
            "slot_ablation_checkpoint_zero_metrics_exact": ablation_checkpoint_zero[
                "metrics_exact"
            ],
            "slot_ablation_checkpoint_zero_within_tolerance": (
                ablation_checkpoint_zero["matched"]
            ),
            "slot_ablation_checkpoint_zero_numeric_evidence": pre_intervention_match,
        },
        "controls": {
            "repeat_intact": repeat_result,
            "no_context": no_context_result,
            "shuffled_demonstrations": shuffled_result,
            "slot_ablation": ablation_result,
            "state_hold": state_hold_result,
            "recurrent_lesion": recurrent_lesion_result,
            "truncation": {
                "checkpoints": list(checkpoints),
                "uses_one_continuous_intact_trajectory": True,
            },
        },
        "execution": {
            "arm_order": list(arm_wall_seconds),
            "selected_arm_driver": "brainstate.transform.jit",
            "jit_name": "example21_evaluation_arm",
            "jit_inline": False,
            "sequential_separate_arms": True,
            "repeat_intact_cached": False,
            "gathered_steps": list(checkpoints),
            "gathered_step_count": len(checkpoints),
            "wall_seconds_by_arm": arm_wall_seconds,
            "cold_intact_to_warm_repeat_ratio": (
                arm_wall_seconds["intact"] / arm_wall_seconds["repeat_intact"]
                if arm_wall_seconds["repeat_intact"] > 0.0
                else None
            ),
        },
    }
    frozen_diagnostic = result["frozen_no_adaptation"]
    frozen_diagnostic["query_trajectories"] = result["query_trajectories"]
    frozen_diagnostic["aggregate_trajectory"] = result["aggregate_trajectory"]
    frozen_diagnostic["associative_memory_diagnostics"] = result[
        "associative_memory_diagnostics"
    ]
    frozen_diagnostic["determinism"] = result["determinism"]
    frozen_diagnostic["controls"] = result["controls"]
    frozen_diagnostic["execution"] = result["execution"]
    return result


def _qualification(
    config: ExperimentConfig,
    data: _ExperimentData,
    training: dict[str, object],
    evaluation: dict[str, object],
    device_report: dict[str, object],
    model_report: dict[str, object],
) -> dict[str, object]:
    checkpoints = config.checkpoints
    training_efforts = config.training_efforts
    submission_checkpoint = config.submission_checkpoint
    execution_evidence = evaluation.get("execution", {})
    reported_steps = (
        execution_evidence.get("gathered_steps")
        if isinstance(execution_evidence, Mapping)
        else None
    )
    expected_trajectory_steps = (
        tuple(int(value) for value in reported_steps)
        if isinstance(reported_steps, list) and reported_steps
        else tuple(range(submission_checkpoint + 1))
    )

    def finite_tree(value: object) -> bool:
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(item) for item in value)
        if isinstance(value, (bool, np.bool_)) or value is None:
            return True
        if isinstance(value, Real):
            return math.isfinite(float(value))
        return True

    required_metric_names = {
        "query_count",
        "task_count",
        "query_pass_at_1",
        "query_pass_at_2",
        "strict_task_pass_at_1",
        "strict_task_pass_at_2",
        "shape_accuracy_diagnostic",
        "valid_cell_pixel_accuracy_diagnostic",
    }

    def metrics_complete(value: object) -> bool:
        if not isinstance(value, dict) or set(value) != {
            str(checkpoint) for checkpoint in checkpoints
        }:
            return False
        return all(
            isinstance(row, dict)
            and required_metric_names <= row.keys()
            and int(row["query_count"]) == expected_query_count
            and int(row["task_count"]) == expected_task_count
            and finite_tree(row)
            for row in value.values()
        )

    expected_origins = data.evaluation
    if config.evaluation_task_limit is not None:
        expected_origins = expected_origins[: config.evaluation_task_limit]
    expected_task_count = len(expected_origins)
    expected_query_count = sum(len(origin.task.test) for origin in expected_origins)

    required_trajectory_names = {
        "step",
        "mean_firing_rate",
        "mean_spike_count",
        "mean_voltage_l2",
        "mean_feedforward_current_l2",
        "mean_recurrent_current_l2",
        "mean_predictive_entropy",
        "mean_changed_cell_fraction",
        "converged_fraction",
        "near_silence_fraction",
        "near_saturation_fraction",
        "unique_state_hashes",
        "pair_sample_count",
        "pairwise_spike_hamming_fraction",
        "pairwise_voltage_rms_distance",
        "pairwise_feedforward_current_rms_distance",
        "pairwise_recurrent_current_rms_distance",
    }
    aggregate = evaluation.get("aggregate_trajectory")
    aggregate_complete = bool(
        isinstance(aggregate, list)
        and len(aggregate) == len(expected_trajectory_steps)
        and all(
            isinstance(row, dict)
            and required_trajectory_names <= row.keys()
            and row.get("step") == expected_step
            and finite_tree(row)
            for expected_step, row in zip(
                expected_trajectory_steps, aggregate, strict=True
            )
        )
    )
    query_trajectories = evaluation.get("query_trajectories")
    query_count = int(evaluation.get("query_count", 0))
    required_query_step_names = {
        "step",
        "candidates",
        "changed_cell_count",
        "changed_cell_fraction",
        "predictive_entropy",
        "top_two_logit_margin",
        "spike_count",
        "firing_rate",
        "raster_active_indices",
        "voltage_mean",
        "voltage_std",
        "voltage_mean_absolute",
        "voltage_l2",
        "spike_hamming_displacement",
        "spike_hamming_fraction",
        "voltage_l2_displacement",
        "feedforward_current_mean_absolute",
        "feedforward_current_l2",
        "feedforward_current_l2_displacement",
        "recurrent_current_mean_absolute",
        "recurrent_current_l2",
        "recurrent_current_l2_displacement",
        "converged",
        "near_silence",
        "near_saturation",
        "state_sha256",
        "score",
    }

    def query_trajectory_complete(report: object) -> bool:
        if not isinstance(report, dict) or not isinstance(report.get("steps"), list):
            return False
        steps = report["steps"]
        return bool(
            report.get("step_count") == len(expected_trajectory_steps)
            and int(report.get("neuron_count", 0))
            == int(model_report.get("neuron_count", -1))
            and len(steps) == len(expected_trajectory_steps)
            and all(
                isinstance(row, dict)
                and required_query_step_names <= row.keys()
                and row.get("step") == expected_step
                and isinstance(row.get("candidates"), list)
                and bool(row["candidates"])
                and finite_tree(row)
                for expected_step, row in zip(
                    expected_trajectory_steps, steps, strict=True
                )
            )
            and finite_tree(report)
        )

    query_trajectories_complete = bool(
        isinstance(query_trajectories, list)
        and len(query_trajectories) == query_count
        and all(query_trajectory_complete(report) for report in query_trajectories)
    )
    checkpoint_queries = evaluation.get("checkpoint_queries")

    def candidate_provenance_complete(effort: int, row: Mapping[str, object]) -> bool:
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            return False
        allowed = (
            ("model",)
            if config.primary_candidate_mode == "model_only"
            else ("model", "rule")
        )
        if not all(
            isinstance(candidate, dict)
            and candidate.get("provenance") in allowed
            and candidate.get("rank") == rank
            for rank, candidate in enumerate(candidates, start=1)
        ):
            return False
        first, second = candidates
        if first.get("provenance") == "rule":
            return bool(
                first.get("selection_role") == "demonstration_verified_rule"
                and isinstance(first.get("rule_name"), str)
                and second.get("provenance") == "model"
                and second.get("grid") != first.get("grid")
                and second.get("selection_role")
                in (
                    "latest_sweep_joint_argmax",
                    "diagnostic_checkpoint_joint_argmax",
                    "latest_sweep_logit_runner_up",
                    "diagnostic_checkpoint_logit_runner_up",
                )
                and row.get("submission_role")
                == (
                    "primary_submission"
                    if effort == submission_checkpoint
                    else "diagnostic_only"
                )
            )
        if effort == 0:
            return bool(
                row.get("submission_role") == "diagnostic_only"
                and first.get("source_checkpoint") == 0
                and second.get("source_checkpoint") == 0
                and first.get("selection_role") == "diagnostic_checkpoint_joint_argmax"
                and second.get("selection_role")
                == "diagnostic_checkpoint_logit_runner_up"
            )
        if row.get("submission_role") != (
            "primary_submission"
            if effort == submission_checkpoint
            else "diagnostic_only"
        ):
            return False
        if (
            first.get("source_checkpoint") != effort
            or first.get("selection_role") != "latest_sweep_joint_argmax"
        ):
            return False
        return bool(
            second.get("selection_role") == "latest_sweep_logit_runner_up"
            and second.get("source_checkpoint") == effort
        )

    checkpoint_queries_complete = bool(
        isinstance(checkpoint_queries, dict)
        and set(checkpoint_queries) == {str(checkpoint) for checkpoint in checkpoints}
        and all(
            isinstance(rows, list)
            and len(rows) == expected_query_count
            and all(
                isinstance(row, dict)
                and {
                    "task_id",
                    "query_index",
                    "primary_candidate_mode",
                    "candidates",
                    "score",
                }
                <= row.keys()
                and row["primary_candidate_mode"] == config.primary_candidate_mode
                and candidate_provenance_complete(checkpoint, row)
                and finite_tree(row)
                for row in rows
            )
            for checkpoint, rows in (
                (checkpoint, checkpoint_queries[str(checkpoint)])
                for checkpoint in checkpoints
            )
        )
    )
    submission_policy = evaluation.get("submission_policy")
    submission_policy_complete = bool(
        isinstance(submission_policy, dict)
        and submission_policy.get("name")
        == _submission_policy_name(config.primary_candidate_mode)
        and submission_policy.get("submission_checkpoint") == submission_checkpoint
        and submission_policy.get("completed_sweep_checkpoints")
        == list(training_efforts)
        and submission_policy.get("candidate_budget") == 2
        and submission_policy.get("fallback")
        == "latest_checkpoint_factorized_global_runner_up"
        and submission_policy.get("target_free_selection") is True
        and submission_policy.get("rule_channel_enabled")
        is (config.primary_candidate_mode == "rule_then_model")
    )
    completion = evaluation.get("model_only_completion")
    completion_expected_eligible = bool(
        not config.smoke
        and not config.structural_only
        and not data.plumbing_only
        and config.evaluation_task_limit is None
        and config.decoder_mode in ("row_refinement", "latent_row_decode")
        and len(data.evaluation) == 400
    )
    completion_complete = bool(
        isinstance(completion, dict)
        and completion.get("primary_candidate_mode") == "model_only"
        and completion.get("eligible_for_completion") is completion_expected_eligible
        and completion.get("submission_checkpoint") == submission_checkpoint
        and completion.get("submission_policy") == SUBMISSION_POLICY
        and completion.get("required_task_count") == 400
        and completion.get("evaluated_task_count") == expected_task_count
        and completion.get("evaluated_query_count") == expected_query_count
        and completion.get("required_exact_task_count") == 160
        and isinstance(completion.get("exact_task_count"), Integral)
        and not isinstance(completion.get("exact_task_count"), bool)
        and isinstance(completion.get("passed"), bool)
        and (completion.get("passed") is False or completion_expected_eligible)
        and finite_tree(completion)
    )
    if completion_complete and isinstance(completion, dict):
        exact_task_count = int(completion["exact_task_count"])
        if completion_expected_eligible:
            completion_tasks = completion.get("tasks")
            completion_complete = bool(
                isinstance(completion_tasks, dict)
                and len(completion_tasks) == 400
                and completion.get("strict_task_pass_at_2") == exact_task_count / 400
                and completion.get("passed") == (exact_task_count >= 160)
            )
        else:
            completion_complete = bool(
                completion.get("passed") is False
                and isinstance(completion.get("eligibility_reason"), str)
                and bool(completion["eligibility_reason"])
            )

    def comparison_complete(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        current_distance = value.get("synaptic_current_l2_by_step")
        score_deltas = value.get("score_deltas_control_minus_intact")
        expected_control_steps = (
            len(checkpoints)
            if config.decoder_mode == "latent_row_decode"
            else max(checkpoints) + 1
        )
        return bool(
            isinstance(value.get("causally_null_at_measured_precision"), bool)
            and len(value.get("state_byte_identical_by_step", ()))
            == expected_control_steps
            and len(value.get("spike_hamming_by_step", ())) == expected_control_steps
            and len(value.get("spike_hamming_fraction_by_step", ()))
            == expected_control_steps
            and len(value.get("voltage_l2_by_step", ())) == expected_control_steps
            and isinstance(current_distance, dict)
            and set(current_distance) == {"feedforward", "recurrent"}
            and all(
                len(current_distance[name]) == expected_control_steps
                for name in current_distance
            )
            and isinstance(score_deltas, dict)
            and {
                f"{submission_checkpoint}.query_pass_at_2",
                f"{submission_checkpoint}.valid_cell_pixel_accuracy_diagnostic",
            }
            <= score_deltas.keys()
            and finite_tree(value)
        )

    controls = evaluation.get("controls")
    controls_required = bool(
        config.evaluation_controls
        or (isinstance(controls, dict) and controls.get("enabled") is not False)
    )

    def control_complete(name: str) -> bool:
        if not isinstance(controls, dict) or not isinstance(controls.get(name), dict):
            return False
        control = controls[name]
        total = int(control.get("query_count", -1))
        applicable = int(control.get("applicable_query_count", -1))
        unavailable = int(control.get("unavailable_query_count", -1))
        timing_matched = int(control.get("timing_matched_applicable_query_count", -1))
        control_metrics = control.get("metrics_by_effort")
        metrics_ok = (
            bool(
                isinstance(control_metrics, dict)
                and set(control_metrics)
                == {str(checkpoint) for checkpoint in checkpoints}
                and all(
                    isinstance(row, dict)
                    and required_metric_names <= row.keys()
                    and int(row["query_count"]) == applicable
                    and 1 <= int(row["task_count"]) <= expected_task_count
                    and finite_tree(row)
                    for row in control_metrics.values()
                )
            )
            if applicable > 0
            else control_metrics == {}
        )
        comparison_ok = (
            comparison_complete(control.get("trajectory_comparison"))
            if applicable > 0
            else isinstance(control.get("trajectory_comparison"), dict)
            and control["trajectory_comparison"].get("available") is False
        )
        candidate_matches = control.get("decoded_candidates_match_intact_by_effort")
        candidate_match_counts = control.get(
            "decoded_candidate_match_query_count_by_effort"
        )
        decoded_candidates_match = control.get("decoded_candidates_match_intact")
        candidate_summary_ok = (
            bool(
                isinstance(candidate_matches, dict)
                and set(candidate_matches)
                == {str(checkpoint) for checkpoint in checkpoints}
                and all(isinstance(value, bool) for value in candidate_matches.values())
                and isinstance(candidate_match_counts, dict)
                and set(candidate_match_counts) == set(candidate_matches)
                and all(
                    isinstance(value, Integral)
                    and not isinstance(value, bool)
                    and 0 <= int(value) <= applicable
                    for value in candidate_match_counts.values()
                )
                and all(
                    candidate_matches[key]
                    == (int(candidate_match_counts[key]) == applicable)
                    for key in candidate_matches
                )
                and isinstance(decoded_candidates_match, bool)
                and decoded_candidates_match == all(candidate_matches.values())
            )
            if applicable > 0
            else candidate_matches == {}
            and candidate_match_counts == {}
            and decoded_candidates_match is None
        )
        applicability_ok = (
            applicable == expected_query_count
            if name in ("repeat_intact", "no_context", "slot_ablation")
            else applicable > 0
        )
        return bool(
            total == query_count
            and total == expected_query_count
            and applicable >= 0
            and unavailable >= 0
            and applicable + unavailable == total
            and timing_matched == applicable
            and applicability_ok
            and metrics_ok
            and comparison_ok
            and candidate_summary_ok
        )

    required_control_names = [
        "repeat_intact",
        "no_context",
        "shuffled_demonstrations",
        "slot_ablation",
    ]
    if config.decoder_mode == "latent_row_decode":
        required_control_names.extend(("state_hold", "recurrent_lesion"))
    required_controls_complete = (
        all(
            control_complete(name) for name in required_control_names
        )
        and (
            config.decoder_mode != "latent_row_decode"
            or bool(controls["state_hold"].get("r30_r60_equal_r0"))
        )
        if controls_required
        else isinstance(controls, dict) and controls.get("enabled") is False
    )
    truncation = controls.get("truncation", {}) if isinstance(controls, dict) else {}
    truncation_complete = (
        bool(
            truncation.get("checkpoints") == list(checkpoints)
            and truncation.get("uses_one_continuous_intact_trajectory") is True
        )
        if controls_required
        else isinstance(truncation, dict) and truncation.get("enabled") is False
    )
    determinism = evaluation.get("determinism", {})
    required_numeric_states = {
        "compact_logits",
        "voltage",
        "feedforward_current",
        "recurrent_current",
    }

    def numeric_evidence_complete(value: object, steps: Sequence[int]) -> bool:
        if not isinstance(value, dict):
            return False
        maximum_rms = value.get("maximum_rms")
        per_query_rms = value.get("per_query_maximum_rms")
        per_step_query_rms = value.get("per_step_query_rms")
        intact_dtypes = value.get("intact_dtype_by_state")
        candidate_dtypes = value.get("candidate_dtype_by_state")
        if not (
            isinstance(maximum_rms, dict)
            and set(maximum_rms) == required_numeric_states
            and isinstance(per_query_rms, dict)
            and set(per_query_rms) == required_numeric_states
            and isinstance(per_step_query_rms, dict)
            and set(per_step_query_rms) == required_numeric_states
            and isinstance(intact_dtypes, dict)
            and isinstance(candidate_dtypes, dict)
            and intact_dtypes == {name: "float32" for name in required_numeric_states}
            and candidate_dtypes == intact_dtypes
            and value.get("required_float32_dtypes") is True
        ):
            return False

        def tolerated(value: object) -> bool:
            return bool(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= STATE_RMS_TOLERANCE
            )

        def count_is(name: str, expected: int) -> bool:
            item = value.get(name)
            return bool(
                isinstance(item, Integral)
                and not isinstance(item, bool)
                and int(item) == expected
            )

        expected_steps = list(steps)
        per_query_ok = all(
            isinstance(values, list)
            and len(values) == expected_query_count
            and all(tolerated(item) for item in values)
            for values in per_query_rms.values()
        )
        per_step_query_ok = all(
            isinstance(rows, list)
            and len(rows) == len(expected_steps)
            and all(
                isinstance(row, list)
                and len(row) == expected_query_count
                and all(tolerated(item) for item in row)
                for row in rows
            )
            for rows in per_step_query_rms.values()
        )
        return bool(
            value.get("evaluated_steps") == expected_steps
            and count_is("query_count", expected_query_count)
            and value.get("within_declared_tolerance") is True
            and value.get("within_tolerance_by_query") == [True] * expected_query_count
            and count_is("within_tolerance_query_count", expected_query_count)
            and count_is("spike_hamming_count", 0)
            and value.get("spike_hamming_count_by_query") == [0] * expected_query_count
            and value.get("declared_per_query_axis_rms_tolerance")
            == STATE_RMS_TOLERANCE
            and all(tolerated(item) for item in maximum_rms.values())
            and per_query_ok
            and per_step_query_ok
        )

    repeat_control = (
        controls.get("repeat_intact", {}) if isinstance(controls, dict) else {}
    )
    repeat_candidate_counts = (
        repeat_control.get("decoded_candidate_match_query_count_by_effort", {})
        if isinstance(repeat_control, dict)
        else {}
    )
    repeat_candidate_flags = (
        repeat_control.get("decoded_candidates_match_intact_by_effort", {})
        if isinstance(repeat_control, dict)
        else {}
    )
    repeat_candidates_exact = bool(
        isinstance(repeat_candidate_counts, dict)
        and isinstance(repeat_candidate_flags, dict)
        and set(repeat_candidate_counts)
        == {str(checkpoint) for checkpoint in checkpoints}
        and set(repeat_candidate_flags) == set(repeat_candidate_counts)
        and all(
            isinstance(repeat_candidate_counts[key], Integral)
            and not isinstance(repeat_candidate_counts[key], bool)
            and int(repeat_candidate_counts[key]) == expected_query_count
            and repeat_candidate_flags[key] is True
            for key in repeat_candidate_counts
        )
        and repeat_control.get("decoded_candidates_match_intact") is True
    )
    frozen_no_adaptation = evaluation.get("frozen_no_adaptation", {})
    frozen_metrics = (
        frozen_no_adaptation.get("metrics_by_effort")
        if isinstance(frozen_no_adaptation, dict)
        else None
    )
    repeat_metrics_exact = bool(
        isinstance(repeat_control, dict)
        and repeat_control.get("metrics_by_effort") == frozen_metrics
    )
    repeat_numeric_exact = bool(
        isinstance(determinism, dict)
        and determinism.get("state_rms_tolerance") == STATE_RMS_TOLERANCE
        and determinism.get("spike_tolerance") == "exact identity"
        and isinstance(determinism.get("metric_absolute_tolerance"), Real)
        and not isinstance(determinism.get("metric_absolute_tolerance"), bool)
        and float(determinism["metric_absolute_tolerance"]) == 0.0
        and numeric_evidence_complete(
            determinism.get("repeat_intact_numeric_evidence"),
            expected_trajectory_steps,
        )
    )
    repeatable = (
        bool(
            isinstance(determinism, dict)
            and determinism.get("same_control_capable_execution_path") is True
            and determinism.get("repeat_intact_within_tolerance") is True
            and determinism.get("repeat_intact_metrics_exact") is True
            and determinism.get("repeat_intact_decoded_candidates_exact") is True
            and repeat_candidates_exact
            and repeat_metrics_exact
            and repeat_numeric_exact
        )
        if controls_required
        else True
    )
    slot_control = (
        controls.get("slot_ablation", {}) if isinstance(controls, dict) else {}
    )
    slot_candidate_counts = (
        slot_control.get("decoded_candidate_match_query_count_by_effort", {})
        if isinstance(slot_control, dict)
        else {}
    )
    slot_candidate_flags = (
        slot_control.get("decoded_candidates_match_intact_by_effort", {})
        if isinstance(slot_control, dict)
        else {}
    )
    slot_metrics = (
        slot_control.get("metrics_by_effort", {})
        if isinstance(slot_control, dict)
        else {}
    )
    slot_checkpoint_zero_exact = bool(
        isinstance(slot_candidate_counts, dict)
        and isinstance(slot_candidate_flags, dict)
        and isinstance(slot_candidate_counts.get("0"), Integral)
        and not isinstance(slot_candidate_counts.get("0"), bool)
        and int(slot_candidate_counts["0"]) == expected_query_count
        and slot_candidate_flags.get("0") is True
        and isinstance(slot_metrics, dict)
        and isinstance(frozen_metrics, dict)
        and slot_metrics.get("0") == frozen_metrics.get("0")
    )
    slot_numeric_exact = bool(
        isinstance(determinism, dict)
        and numeric_evidence_complete(
            determinism.get("slot_ablation_checkpoint_zero_numeric_evidence"), (0,)
        )
    )
    ablation_matched = (
        bool(
            isinstance(determinism, dict)
            and determinism.get(
                "slot_ablation_checkpoint_zero_state_within_tolerance"
            )
            is True
            and determinism.get(
                "slot_ablation_checkpoint_zero_decoded_candidates_exact"
            )
            is True
            and determinism.get("slot_ablation_checkpoint_zero_metrics_exact")
            is True
            and determinism.get("slot_ablation_checkpoint_zero_within_tolerance")
            is True
            and slot_checkpoint_zero_exact
            and slot_numeric_exact
        )
        if controls_required
        else True
    )
    adaptation = evaluation.get("task_local_adaptation")
    adaptation_expected = bool(
        config.task_local_adaptation
        and config.decoder_mode in ("row_refinement", "latent_row_decode")
        and not config.structural_only
    )
    if adaptation_expected:
        adaptation_complete = bool(
            isinstance(adaptation, dict)
            and adaptation.get("performed") is True
            and adaptation.get("mode") == "compiled_task_local_pp_prop_leave_one_out"
            and adaptation.get("target_free_query_bank") is True
            and int(adaptation.get("task_count", -1)) == expected_task_count
            and int(adaptation.get("query_count", -1)) == expected_query_count
            and int(adaptation.get("applied_fold_count", 0)) > 0
            and adaptation.get("fold_count") == adaptation.get("applied_fold_count")
            and adaptation.get("semantic_checkpoints") == list(checkpoints)
            and adaptation.get("all_valid_checkpoints_recorded") is True
            and adaptation.get("base_parameters_restored") is True
            and adaptation.get("base_parameter_sha256")
            == adaptation.get("restored_parameter_sha256")
            and int(adaptation.get("optimizer_step_count_after_cleanup", -1)) == 0
            and finite_tree(adaptation)
        )
    else:
        adaptation_complete = bool(
            isinstance(adaptation, dict) and adaptation.get("performed") is False
        )
    frozen_no_adaptation_complete = bool(
        isinstance(frozen_no_adaptation, dict)
        and frozen_no_adaptation.get("role")
        in {"diagnostic_control_not_primary_submission", "primary_shared_model"}
        and metrics_complete(frozen_no_adaptation.get("metrics_by_effort"))
        and isinstance(frozen_no_adaptation.get("checkpoint_queries"), dict)
    )
    primary_evaluation_complete = bool(
        evaluation.get("primary_candidate_mode") == config.primary_candidate_mode
        and query_count > 0
        and query_count == expected_query_count
        and int(evaluation.get("task_count", 0)) == expected_task_count
        and metrics_complete(evaluation.get("metrics_by_effort"))
        and checkpoint_queries_complete
        and submission_policy_complete
        and completion_complete
        and adaptation_complete
    )
    frozen_diagnostics_complete = bool(
        aggregate_complete
        and query_trajectories_complete
        and required_controls_complete
        and truncation_complete
        and frozen_no_adaptation_complete
    )
    evaluation_complete = bool(
        primary_evaluation_complete
        and frozen_diagnostics_complete
        and finite_tree(evaluation)
    )

    compiler_report = training.get("compiler_report", {})
    compiler_counts = (
        compiler_report.get("counts", {}) if isinstance(compiler_report, dict) else {}
    )
    routed_paths = (
        {
            item.get("parameter")
            for item in compiler_report.get("etrace_weights", ())
            if isinstance(item, dict)
        }
        if isinstance(compiler_report, dict)
        else set()
    )
    plain_paths = (
        {
            item.get("parameter")
            for item in compiler_report.get("excluded_weights", ())
            if isinstance(item, dict)
        }
        if isinstance(compiler_report, dict)
        else set()
    )
    legacy_temporal_paths = {
        "ff_syn.comm.weight",
        "rec_syn.comm.weight",
    }
    row_refinement_paths = set(refinement_parameter_paths(config))
    legacy_plain_paths = {
        "color_factor_head.weight",
        "height_head.weight",
        "readout_projection.weight",
        "width_head.weight",
    }
    plain_paths_expected = (
        legacy_plain_paths if config.decoder_mode == "legacy_cp" else set()
    )
    associative_paths = {
        "memory_write_scale",
        "workspace_query_projection.weight",
        "memory_read_projection.weight",
    }
    if config.memory_coding == "learned_update":
        associative_paths.update(
            {
                "memory_key_projection.weight",
                "memory_update_projection.weight",
            }
        )
    memory_enabled = config.context_memory_width > 0
    routed_paths_expected = (
        legacy_temporal_paths
        | (associative_paths if memory_enabled else set())
        | (
            row_refinement_paths
            if config.decoder_mode in ("row_refinement", "latent_row_decode")
            else set()
        )
    )
    expected_parameter_paths = routed_paths_expected | plain_paths_expected
    route_classifications: dict[object, set[object]] = {}
    for item in (
        compiler_report.get("diagnostics", ())
        if isinstance(compiler_report, dict)
        else ()
    ):
        if not isinstance(item, dict) or item.get("kind") != "relation_included":
            continue
        classifications = item.get("path_classification_by_hidden_state")
        if not isinstance(classifications, dict) or not classifications:
            continue
        route_classifications.setdefault(item.get("weight_path"), set()).update(
            classifications.values()
        )
    associative_routes_direct = bool(
        not memory_enabled
        or all(
            route_classifications.get(path) == {"all_direct"}
            for path in associative_paths
        )
    )
    row_routes_direct = bool(
        config.decoder_mode not in ("row_refinement", "latent_row_decode")
        or all(
            route_classifications.get(path) == {"all_direct"}
            for path in row_refinement_paths
        )
    )
    associative_diagnostics = evaluation.get("associative_memory_diagnostics")
    associative_diagnostics_complete = bool(
        not memory_enabled
        or not controls_required
        or (
            isinstance(associative_diagnostics, dict)
            and associative_diagnostics.get("available") is True
            and associative_diagnostics.get("complete") is True
            and associative_diagnostics.get("repeat_intact_exact") is True
            and associative_diagnostics.get("no_context_memory_exactly_zero") is True
            and associative_diagnostics.get(
                "shuffled_pairing_sensitive_for_every_applicable_query"
            )
            is True
            and int(associative_diagnostics.get("query_count", 0)) == query_count
            and int(associative_diagnostics.get("depth_count", 0))
            == (
                len(checkpoints)
                if config.decoder_mode == "latent_row_decode"
                else max(checkpoints) + 1
            )
        )
    )
    compiler_complete = bool(
        training.get("pp_prop_compiled") is True
        and isinstance(compiler_report, dict)
        and compiler_report.get("available") is True
        and int(compiler_counts.get("hidden_groups", 0)) >= 1
        and int(compiler_counts.get("errors", -1)) == 0
        and routed_paths == routed_paths_expected
        and plain_paths == plain_paths_expected
        and routed_paths | plain_paths == expected_parameter_paths
        and associative_routes_direct
        and row_routes_direct
    )
    full_scale = bool(
        model_report.get("neuron_count") == FULL_SCALE_NEURON_COUNT
        and model_report.get("recurrent_edge_count") == FULL_SCALE_RECURRENT_EDGES
        and model_report.get("slot_count") == 64
        and int(model_report.get("parameter_count", 0)) > 0
    )
    component_types = model_report.get("component_types", {})
    component_contract = bool(
        isinstance(component_types, dict)
        and component_types
        == {
            "neuron": "LIF",
            "feedforward_projection_wrapper": "AlignPostProj",
            "feedforward_projection": "Linear",
            "feedforward_synapse": "Expon",
            "feedforward_output": "CUBA",
            "recurrent_projection_wrapper": "AlignPostProj",
            "recurrent_projection": "SparseLinear",
            "recurrent_synapse": "Expon",
            "recurrent_output": "CUBA",
        }
    )
    gpu_complete = str(device_report.get("platform", "")).casefold() == "gpu"
    gpu_runtime_report = device_report.get("gpu_runtime_safety", {})
    gpu_runtime_resource_safe = bool(
        gpu_complete
        and isinstance(gpu_runtime_report, dict)
        and gpu_runtime_report.get("full_qualification_safe") is True
    )
    frozen = evaluation.get("same_frozen_parameter_bytes") is True
    structural_checks = {
        "actual_full_scale": full_scale,
        "physical_component_contract": component_contract,
        "actual_gpu_backend": gpu_complete,
        "gpu_runtime_resource_safe": gpu_runtime_resource_safe,
        "pp_prop_compiler_routes": compiler_complete,
        "associative_routes_all_direct": associative_routes_direct,
        "row_routes_all_direct": row_routes_direct,
        "associative_diagnostics_complete": associative_diagnostics_complete,
        "complete_primary_evaluation": primary_evaluation_complete,
        "complete_frozen_diagnostics": frozen_diagnostics_complete,
        "complete_frozen_evaluation": evaluation_complete,
        "frozen_parameters_unchanged": frozen,
        "repeat_intact_deterministic": repeatable,
        "slot_ablation_pre_intervention_matched": ablation_matched,
        "required_controls_executed": bool(
            config.evaluation_controls and required_controls_complete
        ),
    }
    structural = all(structural_checks.values())

    training_counts = training.get("optimizer_updates_by_effort", {})
    mixed = bool(
        isinstance(training_counts, dict)
        and all(
            int(training_counts.get(str(effort), 0)) > 0 for effort in training_efforts
        )
        and sum(int(training_counts.get(str(effort), 0)) for effort in training_efforts)
        == config.training_updates
    )
    losses = training.get("losses")
    losses_complete = bool(
        isinstance(losses, list)
        and len(losses) == config.training_updates
        and finite_tree(losses)
    )
    parameter_changes = training.get("parameter_changes")
    temporal_paths_moved = bool(
        isinstance(parameter_changes, dict)
        and all(
            isinstance(parameter_changes.get(path), dict)
            and parameter_changes[path].get("changed") is True
            and float(parameter_changes[path].get("l2_delta", 0.0)) > 0.0
            and math.isfinite(float(parameter_changes[path]["l2_delta"]))
            for path in ("ff_syn.comm.weight", "rec_syn.comm.weight")
        )
    )
    all_active_parameter_changes_finite = bool(
        isinstance(parameter_changes, dict)
        and expected_parameter_paths <= set(parameter_changes)
        and all(
            isinstance(parameter_changes.get(path), dict)
            and parameter_changes[path].get("changed") is True
            and float(parameter_changes[path].get("l2_delta", 0.0)) > 0.0
            and math.isfinite(float(parameter_changes[path]["l2_delta"]))
            for path in expected_parameter_paths
        )
    )
    sources = [item.manifest.source for item in data.loaded]
    training_names = {
        str(source.name).casefold() for source in sources if source.role == "train"
    }
    evaluation_names = {
        str(source.name).casefold() for source in sources if source.role == "evaluation"
    }
    approved_sources = bool(
        training_names
        and evaluation_names
        and training_names <= APPROVED_TRAINING_SOURCES
        and evaluation_names <= APPROVED_EVALUATION_SOURCES
    )
    no_rejected_sources = all(
        len(getattr(item.manifest, "rejected", ())) == 0 for item in data.loaded
    )
    expected_supervision = (
        "latent_row_ticks_1..effort"
        if config.decoder_mode in ("row_refinement", "latent_row_decode")
        else "0..effort"
    )
    depth_supervision = bool(
        training.get("supervised_depths") == expected_supervision
        and training.get("depth_weighting") == "uniform_unit_sum_per_update"
        and training.get("per_update_depth_weight_sum", 1.0) == 1.0
    )
    if "supervised_depths" not in training:
        depth_supervision = training.get("terminal_supervision_only") is True
    associative_capability_status = (
        "associative_capability_gates_pending"
        if memory_enabled
        else "not_applicable_legacy"
    )
    scientific_checks = {
        "structural_qualification": structural,
        "not_smoke_or_structural_only": not config.smoke and not config.structural_only,
        "complete_evaluation_split": config.evaluation_task_limit is None,
        "approved_train_and_evaluation_sources": approved_sources,
        "no_rejected_source_records": no_rejected_sources,
        "not_plumbing_only": not data.plumbing_only,
        "one_model_one_optimizer_depth_supervision": bool(
            training.get("performed") is True
            and training.get("one_shared_model") is True
            and training.get("one_shared_optimizer_state") is True
            and depth_supervision
        ),
        "mixed_effort_update_schedule": mixed,
        "finite_loss_per_update": losses_complete,
        "parameters_moved": training.get("parameters_moved") is True,
        "temporal_synapses_moved": temporal_paths_moved,
        "all_active_parameter_groups_moved_with_finite_delta": (
            all_active_parameter_changes_finite
        ),
        "associative_capability_gates_complete": not memory_enabled,
    }
    scientific = all(scientific_checks.values())
    score_gate_passed = bool(
        completion_complete
        and isinstance(completion, dict)
        and completion.get("passed") is True
    )
    approved_completion_target = bool(scientific and score_gate_passed)
    structural_messages = {
        "actual_full_scale": (
            f"actual model is not the required {FULL_SCALE_NEURON_COUNT}-neuron/"
            f"{FULL_SCALE_RECURRENT_EDGES}-edge scale"
        ),
        "physical_component_contract": "actual neuron, projection, synapse, or current-output component types do not match the declared substrate",
        "actual_gpu_backend": "actual evaluation backend is not GPU",
        "gpu_runtime_resource_safe": "full GPU runtime resource-safety evidence is missing, incomplete, or over policy limits",
        "pp_prop_compiler_routes": "pp-prop compilation or feedforward/recurrent eligibility routing evidence is incomplete",
        "associative_routes_all_direct": "associative pp-prop routes are not all_direct",
        "row_routes_all_direct": "row and shape pp-prop routes are not all_direct",
        "associative_diagnostics_complete": "pairing-sensitive S_K, memory-read, or continuous-workspace diagnostics are incomplete",
        "complete_primary_evaluation": "adapted primary metrics, candidates, or task-local evidence are incomplete",
        "complete_frozen_diagnostics": "frozen no-adaptation trajectories or controls are incomplete",
        "complete_frozen_evaluation": "exact metrics, trajectories, or controls are incomplete or non-finite",
        "frozen_parameters_unchanged": "evaluation mutated frozen parameter bytes",
        "repeat_intact_deterministic": "same-run intact repeat exceeded the declared state/logit tolerance or changed exact candidates or metrics",
        "slot_ablation_pre_intervention_matched": "slot-ablation checkpoint zero exceeded the declared state/logit tolerance or changed exact candidates or effort-0 metrics",
        "required_controls_executed": "one or more required protocol-v2 controls were disabled, incomplete, or failed its invariant",
    }
    scientific_messages = {
        "not_smoke_or_structural_only": "smoke fixtures or disabled optimization cannot be scientific evidence",
        "complete_evaluation_split": "evaluation_task_limit makes this a development subset",
        "approved_train_and_evaluation_sources": "approved train/evaluation source roles were not both present",
        "no_rejected_source_records": "source rejections were present",
        "not_plumbing_only": "embedded fixtures are plumbing-only",
        "one_model_one_optimizer_depth_supervision": "training did not retain one shared model, optimizer state, and normalized decoder-appropriate depth supervision",
        "mixed_effort_update_schedule": "one shared model did not receive the complete 30/60 sweep schedule",
        "finite_loss_per_update": "one finite loss was not retained for every optimizer update",
        "parameters_moved": "training did not change parameter bytes",
        "temporal_synapses_moved": "feedforward and recurrent eligibility-routed synapses did not both move",
        "all_active_parameter_groups_moved_with_finite_delta": "not every active parameter group moved with a finite delta",
        "associative_capability_gates_complete": "associative_capability_gates_pending",
    }
    reasons_not_structural = [
        structural_messages[name]
        for name, passed in structural_checks.items()
        if not passed
    ]
    reasons_not_scientific = list(reasons_not_structural)
    reasons_not_scientific.extend(
        scientific_messages[name]
        for name, passed in scientific_checks.items()
        if name != "structural_qualification" and not passed
    )
    structural_check_results = {
        name: {
            "status": "passed" if passed else "failed",
            "required": True,
            "reason": None if passed else structural_messages[name],
        }
        for name, passed in structural_checks.items()
    }
    scientific_check_results = {
        name: {
            "status": "passed" if passed else "failed",
            "required": True,
            "reason": None if passed else (
                "structural qualification failed"
                if name == "structural_qualification"
                else scientific_messages[name]
            ),
        }
        for name, passed in scientific_checks.items()
    }
    controls_executed = bool(config.evaluation_controls and required_controls_complete)
    control_check = {
        "status": (
            "passed"
            if controls_executed
            else "failed"
            if config.evaluation_controls
            else "not_run"
        ),
        "required": True,
        "reason": (
            None
            if controls_executed
            else "evaluation controls were incomplete"
            if config.evaluation_controls
            else "evaluation controls were disabled"
        ),
    }
    return {
        "full_structural_qualification": structural,
        "full_scientific_qualification": scientific,
        "model_only_score_gate_passed": score_gate_passed,
        "approved_completion_target_passed": approved_completion_target,
        "model_only_completion": completion,
        "plumbing_only": data.plumbing_only,
        "associative_capability_status": associative_capability_status,
        "structural_checks": structural_checks,
        "scientific_checks": scientific_checks,
        "structural_check_results": structural_check_results,
        "scientific_check_results": scientific_check_results,
        "control_execution_check": control_check,
        "reasons_not_structural": reasons_not_structural,
        "reasons_not_scientific": reasons_not_scientific,
    }


def _parameter_count(values: dict[str, Any]) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(values)))


def _software_report(
    pre_device_gpu_environment: Mapping[str, object],
    environment: Mapping[str, str],
) -> dict[str, object]:
    distributions = (
        "braintrace",
        "brainstate",
        "brainpy",
        "braintools",
        "optax",
        "jax",
        "jaxlib",
        "numpy",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "xla_python_client_preallocate": environment.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
        "xla_python_client_mem_fraction": environment.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION"
        ),
        "pre_device_gpu_environment": dict(pre_device_gpu_environment),
        "provenance": {
            "image_digest": environment.get("EXAMPLE21_IMAGE_DIGEST")
            or environment.get("BRAINTRACE_IMAGE_DIGEST"),
            "source_revision": environment.get("EXAMPLE21_SOURCE_REVISION"),
            "source_dirty": environment.get("EXAMPLE21_SOURCE_DIRTY"),
            "arc_revision": environment.get("EXAMPLE21_ARC_REVISION")
            or environment.get("ARC_AGI_1_COMMIT"),
        },
    }


def _git_source_provenance(directory: pathlib.Path) -> dict[str, object]:
    """Resolve live Git provenance without trusting launcher declarations.

    Parameters
    ----------
    directory : pathlib.Path
        File or directory inside the source checkout.

    Returns
    -------
    dict
        Actual revision and dirty state plus any declared-revision mismatch.
    """
    root = directory if directory.is_dir() else directory.parent
    try:
        revision = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    else:
        dirty = bool(status.strip())
    declared = os.environ.get("EXAMPLE21_SOURCE_REVISION")
    return {
        "source_revision": revision,
        "source_dirty": dirty,
        "declared_source_revision": declared,
        "declared_revision_mismatch": bool(
            declared is not None and revision is not None and declared != revision
        ),
    }


def _implementation_report() -> dict[str, object]:
    directory = pathlib.Path(__file__).resolve().parent
    names = (
        pathlib.Path(__file__).name,
        "latent_workspace_task.py",
        "latent_workspace_analysis.py",
        "latent_workspace_model.py",
        "latent_workspace_protocol.py",
        "latent_workspace_refinement.py",
        "latent_workspace_resource_safety.py",
        "21-arc-agi-latent-reasoning.py",
    )
    combined = hashlib.sha256()
    files: dict[str, str] = {}
    for name in names:
        payload = (directory / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[name] = digest
        combined.update(name.encode("utf-8"))
        combined.update(payload)
    return {
        "source_tree_sha256": combined.hexdigest(),
        "file_sha256": files,
        **_git_source_provenance(directory),
        "image_digest": os.environ.get("EXAMPLE21_IMAGE_DIGEST")
        or os.environ.get("BRAINTRACE_IMAGE_DIGEST"),
        "arc_revision": os.environ.get("EXAMPLE21_ARC_REVISION")
        or os.environ.get("ARC_AGI_1_COMMIT"),
    }


def _artifact_manifest(paths: Mapping[str, pathlib.Path]) -> dict[str, object]:
    """Build a checksum sidecar for materialized run artifacts.

    Parameters
    ----------
    paths : mapping of str to pathlib.Path
        Artifact names and paths after every file has been written.

    Returns
    -------
    dict
        Schema-v2 size and SHA-256 records ordered by artifact name.
    """
    artifacts: dict[str, dict[str, object]] = {}
    for name, path in sorted(paths.items()):
        payload = path.read_bytes()
        artifacts[name] = {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {"schema_version": 2, "artifacts": artifacts}


def _data_summary(
    data: _ExperimentData,
    manifests: Sequence[dict[str, object]],
    evaluation: dict[str, object],
) -> dict[str, object]:
    task_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    source_names: dict[str, list[str]] = {}
    for item in data.loaded:
        role = str(item.manifest.source.role)
        task_counts[role] += len(item.tasks)
        query_counts[role] += sum(len(task.test) for task in item.tasks)
        source_names.setdefault(role, []).append(str(item.manifest.source.name))
    canonical = msgspec.json.encode(list(manifests), order="sorted")
    return {
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_count": len(manifests),
        "source_names_by_role": {
            role: sorted(names) for role, names in source_names.items()
        },
        "task_counts_by_role": dict(sorted(task_counts.items())),
        "query_counts_by_role": dict(sorted(query_counts.items())),
        "training_task_pool_count": len(data.training),
        "evaluated_task_count": int(evaluation["task_count"]),
        "evaluated_query_count": int(evaluation["query_count"]),
        "parsed_task_count": int(
            sum(int(manifest["parsed_task_count"]) for manifest in manifests)
        ),
        "valid_task_count": int(
            sum(int(manifest["valid_task_count"]) for manifest in manifests)
        ),
        "rejected_task_count": int(
            sum(int(manifest["rejected_task_count"]) for manifest in manifests)
        ),
        "duplicate_task_count": int(
            sum(int(manifest["duplicate_task_count"]) for manifest in manifests)
        ),
        "excluded_task_count": int(
            sum(int(manifest["excluded_task_count"]) for manifest in manifests)
        ),
        "split_overlap_check": "passed",
        "private_paper_data_available": False,
        "private_training_recipe_available": False,
    }


def _input_echo_summary(
    checkpoint_queries: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, float]:
    """Mean input-echo fraction of the rank-one candidate at every effort.

    A value near 1.0 says the submitted grids are the query input cropped to
    the predicted shape. Such a run scores the copy baseline on the pixel
    diagnostic while carrying no rule content, so the number belongs beside
    the exact-match metrics rather than behind them.

    Parameters
    ----------
    checkpoint_queries
        Per-effort query records as written to ``result.json``.

    Returns
    -------
    dict
        One mean per effort, keyed by the effort as a string.
    """

    summary: dict[str, float] = {}
    for effort, details in checkpoint_queries.items():
        echoes = [
            float(item["input_echo"]) for item in details if "input_echo" in item
        ]
        summary[str(effort)] = float(np.mean(echoes)) if echoes else 0.0
    return summary


def _channel_attribution(
    checkpoint_queries: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, object]]:
    """Split exact solves between the model and the verified rule channel.

    Parameters
    ----------
    checkpoint_queries
        Per-effort query records as written to ``result.json``.

    Returns
    -------
    dict
        One summary per effort, keyed by the effort as a string.
    """

    summary: dict[str, dict[str, object]] = {}
    for effort, details in checkpoint_queries.items():
        channel_roles = {item.get("submission_role") for item in details}
        if len(channel_roles) != 1 or channel_roles - {
            "diagnostic_only",
            "primary_submission",
        }:
            raise ValueError("primary attribution found invalid submission roles")
        channel_role = next(iter(channel_roles))
        candidates = [
            candidate
            for item in details
            for candidate in item.get("candidates", ())
            if isinstance(candidate, Mapping)
        ]
        if any(
            candidate.get("provenance") not in ("model", "rule")
            for candidate in candidates
        ):
            raise ValueError("primary attribution found unknown candidate provenance")
        modes = {item.get("primary_candidate_mode") for item in details}
        if len(modes) != 1:
            raise ValueError("primary attribution found mixed candidate modes")
        mode = next(iter(modes))
        model_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("provenance") == "model"
        ]
        admitted = [
            item
            for item in details
            if any(
                candidate.get("provenance") == "rule"
                for candidate in item.get("candidates", ())
                if isinstance(candidate, Mapping)
            )
        ]
        exact_by_rule = sum(
            1 for item in admitted if item["score"]["pass_at_1"]
        )
        pass_at_2 = sum(1 for item in details if item["score"]["pass_at_2"])
        rule_names = collections.Counter(
            str(candidate.get("rule_name"))
            for item in admitted
            for candidate in item.get("candidates", ())
            if isinstance(candidate, Mapping)
            and candidate.get("provenance") == "rule"
            and item["score"]["pass_at_1"]
        )
        summary[effort] = {
            "primary_candidate_mode": mode,
            "submission_role": channel_role,
            "query_count": len(details),
            "model_candidate_count": len(model_candidates),
            "submitted_model_candidate_count": (
                len(model_candidates) if channel_role == "primary_submission" else 0
            ),
            "rule_admitted_query_count": len(admitted),
            "rule_admitted_not_exact_count": len(admitted) - exact_by_rule,
            "exact_by_rule_channel": exact_by_rule,
            "exact_by_model_candidates": pass_at_2 - exact_by_rule,
            "exact_total": pass_at_2,
            "solving_rules": dict(rule_names.most_common()),
        }
    return summary


def _render_report(result: dict[str, object]) -> str:
    configuration = result.get("configuration", {})
    checkpoints = tuple(configuration.get("checkpoints", CHECKPOINTS))
    submission_checkpoint = int(
        configuration.get("submission_checkpoint", checkpoints[-1])
    )
    device = result.get("device", {})
    model = result.get("model", {})
    training = result.get("training", {})
    evaluation = result.get("evaluation", {})
    task_local_adaptation = evaluation.get("task_local_adaptation", {})
    frozen_no_adaptation = evaluation.get("frozen_no_adaptation", {})
    associative_diagnostics = frozen_no_adaptation.get(
        "associative_memory_diagnostics",
        evaluation.get("associative_memory_diagnostics", {}),
    )
    qualification = result.get("qualification", {})
    data_summary = result.get("data_summary", {})
    software = result.get("software", {})
    implementation = result.get("implementation", {})
    compiler_report = training.get("compiler_report", {"counts": {}, "diagnostics": []})
    compiler_counts = compiler_report.get("counts", {})
    runtime = float(result.get("runtime_seconds", 0.0))
    frozen_role = frozen_no_adaptation.get("role", "unreported")
    frozen_label = (
        "Frozen shared-model evaluation: primary submission; role="
        if frozen_role == "primary_shared_model"
        else "Frozen no-adaptation diagnostic: diagnostic control, not primary "
        "submission; role="
    )
    if training.get("performed") is True:
        optimizer_name = str(training.get("optimizer", {}).get("name", "unreported"))
        training_line = (
            "Training: one parameter set and one "
            f"{optimizer_name.capitalize()} state; normalized uniform "
            f"{training.get('supervised_depths', 'unreported')} supervision; "
            "updates by complete 30-row sweep depth "
            f"{training.get('optimizer_updates_by_effort', {})}."
        )
    else:
        training_line = (
            "Training: optimization was not performed; reason="
            f"{training.get('reason', 'unreported')}; updates="
            f"{training.get('optimizer_updates_by_effort', {})}."
        )
    if training.get("performed") is True:
        plain_route_line = (
            "The plain routes received exact current-window gradients in this run; "
            "they do not carry temporal eligibility."
        )
    else:
        plain_route_line = (
            "Optimization was disabled, so no routes were trained in this run; when "
            "enabled, the plain routes receive exact current-window gradients and do "
            "not carry temporal eligibility."
        )
    lines = [
        "Example 21 - ARC latent reasoning with pp-prop",
        "",
        str(result.get("claim_boundary", CLAIM_BOUNDARY)),
        "",
        f"Seed: {configuration.get('seed', 'unreported')}",
        f"Runtime: {runtime:.3f} seconds",
        f"Runtime profile: {result.get('runtime_profile', 'disabled')}.",
        (
            f"Device: requested={device.get('requested', 'unreported')}, "
            f"actual={device.get('platform', 'unreported')} "
            f"({device.get('kind', 'unreported')})"
        ),
        (
            f"Device memory after training/evaluation: {device.get('memory_stats', {})}; "
            f"capture={device.get('memory_stats_capture', 'unreported')}."
        ),
        (
            "Implementation: source-tree SHA-256="
            f"{implementation.get('source_tree_sha256', 'unreported')}; "
            f"revision={implementation.get('source_revision') or 'unreported'}; "
            f"source_dirty={implementation.get('source_dirty') or 'unreported'}."
        ),
        (
            f"Software: Python {software.get('python', 'unreported')}; packages "
            f"{software.get('packages', {})}."
        ),
        f"Configuration: {configuration}",
        "",
        (
            f"Physical model: {model.get('neuron_count', 'unreported')} LIF neurons, "
            f"{model.get('recurrent_edge_count', 'unreported')} directed sparse edges, "
            f"{model.get('slot_count', 'unreported')} x 64-neuron analysis slots, "
            f"{model.get('parameter_count', 'unreported')} scalar parameters."
        ),
        (
            "Reasoning memory: mode="
            f"{model.get('reasoning_mode', 'unreported')}; width="
            f"{model.get('context_memory_width', 'unreported')}; decay="
            f"{model.get('memory_decay', 'unreported')}; raw key/value widths="
            f"{model.get('raw_key_feature_width', 'unreported')}/"
            f"{model.get('raw_value_feature_width', 'unreported')}; dense S bytes "
            "per-example/training-batch/evaluation-batch="
            f"{model.get('context_memory_bytes_per_example', 'unreported')}/"
            f"{model.get('context_memory_bytes_training_batch', 'unreported')}/"
            f"{model.get('context_memory_bytes_evaluation_batch', 'unreported')} bytes."
        ),
        (
            "Associative memory implementation: "
            f"{model.get('associative_memory_implementation', {})}."
        ),
        f"Physical component types: {model.get('component_types', {})}.",
        f"Neuron typing: {model.get('neuron_typing', {})}.",
        (
            f"Data manifest SHA-256: {data_summary.get('manifest_sha256', 'unreported')}; "
            f"sources={data_summary.get('source_names_by_role', {})}."
        ),
        (
            f"Splits: tasks={data_summary.get('task_counts_by_role', {})}; "
            f"queries={data_summary.get('query_counts_by_role', {})}; "
            f"evaluated={data_summary.get('evaluated_task_count', 'unreported')} tasks/"
            f"{data_summary.get('evaluated_query_count', 'unreported')} queries; "
            f"rejected={data_summary.get('rejected_task_count', 'unreported')}; "
            f"duplicates={data_summary.get('duplicate_task_count', 'unreported')}; "
            f"explicit exclusions={data_summary.get('excluded_task_count', 'unreported')}."
        ),
        training_line,
        (
            "Effort self-distillation: "
            f"{training.get('effort_self_distillation', {})}."
        ),
        (
            f"Training exposure: {training.get('sampled_base_task_count', 'unreported')} "
            "unique base tasks and "
            f"{training.get('sampled_base_fold_count', 'unreported')} unique "
            "leave-one-demonstration-out folds sampled with replacement="
            f"{training.get('sampling_with_replacement', 'unreported')} "
            f"from a {data_summary.get('training_task_pool_count', 'unreported')}-task pool."
        ),
        (
            f"Training movement: parameter bytes changed={training.get('parameters_moved', False)}; "
            f"per-group evidence={training.get('parameter_changes', {})}."
        ),
        (
            "Compiler: eligibility-trace temporal routes="
            f"{compiler_counts.get('etrace_weights', 0)} "
            f"({[item.get('parameter') for item in compiler_report.get('etrace_weights', [])]}), "
            "plain exact current-window reverse-mode routes="
            f"{compiler_counts.get('excluded_weights', 0)} "
            f"({[item.get('parameter') for item in compiler_report.get('excluded_weights', [])]}), "
            f"{compiler_counts.get('warnings', 0)} "
            "warnings, "
            f"{compiler_counts.get('errors', 0)} errors. {plain_route_line}"
        ),
        f"Evaluation execution: {evaluation.get('execution', {})}.",
        f"Protocol evidence: {evaluation.get('protocol', {})}.",
        (
            "Task-local adaptation: performed="
            f"{task_local_adaptation.get('performed')}; mode="
            f"{task_local_adaptation.get('mode')}; folds applied="
            f"{task_local_adaptation.get('applied_fold_count', 0)}/"
            f"{task_local_adaptation.get('fold_count', 0)}; queries="
            f"{task_local_adaptation.get('query_count', 0)}; bank bytes="
            f"{task_local_adaptation.get('bank_bytes', 'unreported')}; "
            "target-free query bank="
            f"{task_local_adaptation.get('target_free_query_bank')}; restored="
            f"{task_local_adaptation.get('base_parameters_restored')}."
        ),
        (
            frozen_label
            + f"{frozen_no_adaptation.get('role', 'unreported')}; metrics="
            f"{frozen_no_adaptation.get('metrics_by_effort', {})}."
        ),
        (
            "Associative evaluation diagnostics: "
            f"available={associative_diagnostics.get('available')}; "
            f"complete={associative_diagnostics.get('complete')}; "
            "repeat exact="
            f"{associative_diagnostics.get('repeat_intact_exact')}; "
            "no-context S zero="
            f"{associative_diagnostics.get('no_context_memory_exactly_zero')}; "
            "shuffled pairing-sensitive for every applicable query="
            f"{associative_diagnostics.get('shuffled_pairing_sensitive_for_every_applicable_query')}."
        ),
        "",
        "Primary exact ARC results:",
    ]
    for effort in checkpoints:
        metrics = evaluation.get("metrics_by_effort", {}).get(str(effort))
        if metrics is None:
            lines.append(f"  effort {effort:>2}: unavailable")
            continue
        role = "primary submission" if effort == submission_checkpoint else "diagnostic"
        lines.append(
            f"  effort {effort:>2} ({role}): query pass@1={metrics['query_pass_at_1']:.4f}, "
            f"pass@2={metrics['query_pass_at_2']:.4f}; strict task pass@1="
            f"{metrics['strict_task_pass_at_1']:.4f}, pass@2="
            f"{metrics['strict_task_pass_at_2']:.4f}; shape diagnostic="
            f"{metrics['shape_accuracy_diagnostic']:.4f}, pixel diagnostic="
            f"{metrics['valid_cell_pixel_accuracy_diagnostic']:.4f}"
        )
        echo = evaluation.get("model_only_input_echo_by_effort", {}).get(str(effort))
        if echo is not None:
            lines.append(
                f"    model-only input echo (rank-one cells copying the query "
                f"input): {float(echo):.4f}"
            )
    attribution = evaluation.get("channel_attribution", {})
    for effort in checkpoints:
        split = attribution.get(str(effort))
        if split is None:
            continue
        lines.append(
            f"  effort {effort:>2} {split.get('submission_role', 'unreported')} "
            f"channel: {split.get('primary_candidate_mode', 'model_only')}; exact "
            f"total={split.get('exact_total', 'unreported')} of "
            f"{split['query_count']} queries "
            f"(rule={split.get('exact_by_rule_channel', 0)}, "
            f"model={split['exact_by_model_candidates']}); rules admitted="
            f"{split.get('rule_admitted_query_count', 0)}, admitted-not-exact="
            f"{split.get('rule_admitted_not_exact_count', 0)}; solving rules="
            f"{split.get('solving_rules', {})}."
        )
    model_only_attribution = evaluation.get("model_only_channel_attribution", {})
    model_only_efforts = evaluation.get("model_only_metrics_by_effort", {})
    for effort in checkpoints:
        split = model_only_attribution.get(str(effort))
        row = model_only_efforts.get(str(effort))
        if split is None or row is None:
            continue
        lines.append(
            f"  effort {effort:>2} model-only channel (not the submitted score): "
            f"exact={split['exact_by_model_candidates']} of "
            f"{split['query_count']} queries; query pass@1="
            f"{row['query_pass_at_1']:.4f}, pass@2={row['query_pass_at_2']:.4f}; "
            f"strict task pass@1={row['strict_task_pass_at_1']:.4f}, pass@2="
            f"{row['strict_task_pass_at_2']:.4f}."
        )
    submitted = evaluation.get("submitted_completion")
    if submitted:
        lines.append(
            "  Submitted channel: policy="
            f"{submitted.get('submission_policy')}; exact queries pass@1="
            f"{submitted.get('exact_query_count_at_1')}, pass@2="
            f"{submitted.get('exact_query_count_at_2')}; exact tasks pass@1="
            f"{submitted.get('exact_task_count_at_1')}, pass@2="
            f"{submitted.get('exact_task_count_at_2')} of "
            f"{submitted.get('evaluated_task_count')}."
        )
    completion = evaluation.get("model_only_completion", {})
    lines.append(
        "  Model-only 40% completion gate: eligible="
        f"{completion.get('eligible_for_completion')}; exact tasks="
        f"{completion.get('exact_task_count', 'unreported')}/"
        f"{completion.get('required_task_count', 400)}; required="
        f"{completion.get('required_exact_task_count', 160)}; strict task pass@2="
        f"{completion.get('strict_task_pass_at_2', 'unreported')}; passed="
        f"{completion.get('passed')}; reason="
        f"{completion.get('eligibility_reason') or 'eligible full evaluation'}."
    )
    intact_metrics = evaluation.get("metrics_by_effort", {})
    submission_key = str(submission_checkpoint)
    if "0" in intact_metrics and submission_key in intact_metrics:
        effort_zero = intact_metrics["0"]["query_pass_at_2"]
        effort_final = intact_metrics[submission_key]["query_pass_at_2"]
        exact_count_final = round(
            effort_final * int(intact_metrics[submission_key]["query_count"])
        )
        direction = (
            "improved"
            if effort_final > effort_zero
            else "worsened"
            if effort_final < effort_zero
            else "tied"
        )
        lines.append(
            f"  Empirical outcome: effort {submission_checkpoint} {direction} effort 0 "
            f"on exact pass@2 ({effort_final:.4f} versus {effort_zero:.4f}); "
            f"{exact_count_final} effort-{submission_checkpoint} queries were exact "
            "within the scored set."
        )
    lines.extend(["", "Frozen no-adaptation aggregate latent trajectory:"])
    trajectory = frozen_no_adaptation.get(
        "aggregate_trajectory", evaluation.get("aggregate_trajectory", [])
    )
    trajectory_by_step = {
        int(row["step"]): row
        for row in trajectory
        if isinstance(row, Mapping) and isinstance(row.get("step"), Integral)
    }
    for effort in checkpoints:
        if effort not in trajectory_by_step:
            lines.append(f"  step {effort:>2}: unavailable")
            continue
        row = trajectory_by_step[effort]
        lines.append(
            f"  step {effort:>2}: firing={row.get('mean_firing_rate', math.nan):.6f}; "
            f"Voltage L2={row.get('mean_voltage_l2', math.nan):.6f}; "
            f"feedforward-current L2={row.get('mean_feedforward_current_l2', math.nan):.6f}; "
            f"recurrent-current L2={row.get('mean_recurrent_current_l2', math.nan):.6f}; "
            f"entropy={row.get('mean_predictive_entropy', math.nan):.6f}; "
            f"changed-cell fraction={row.get('mean_changed_cell_fraction')}; "
            f"converged/silent/saturated={row.get('converged_fraction')}/"
            f"{row.get('near_silence_fraction')}/{row.get('near_saturation_fraction')}; "
            f"raw-byte state hashes={row.get('unique_state_hashes')}."
        )
        lines.append(
            f"           deterministic pair sample n={row.get('pair_sample_count', 0)}; "
            f"spike-Hamming={row.get('pairwise_spike_hamming_fraction')}; "
            f"voltage RMS={row.get('pairwise_voltage_rms_distance')}; "
            f"feedforward/recurrent current RMS="
            f"{row.get('pairwise_feedforward_current_rms_distance')}/"
            f"{row.get('pairwise_recurrent_current_rms_distance')}."
        )
    lines.extend(
        [
            "  Raw-byte hash counts report collisions only; pairwise distances, not hash uniqueness, test geometry.",
            "",
            "Frozen controls and deterministic repeat:",
        ]
    )
    controls = evaluation.get("controls", {})
    for name in (
        "repeat_intact",
        "no_context",
        "shuffled_demonstrations",
        "slot_ablation",
    ):
        control = controls.get(name)
        if not isinstance(control, dict):
            lines.append(f"  {name}: unavailable")
            continue
        applicable = int(control.get("applicable_query_count", 0))
        unavailable = int(control.get("unavailable_query_count", 0))
        comparison = control.get("trajectory_comparison", {})
        lines.append(
            f"  {name}: applicable={applicable}/{control.get('query_count', 0)}, "
            f"unavailable={unavailable}, timing-matched="
            f"{control.get('timing_matched_applicable_query_count', 0)}/{applicable}; "
            f"causally_null={comparison.get('causally_null_at_measured_precision')}; "
            f"null_queries={control.get('causally_null_query_count', 0)}/{applicable}; "
            f"byte-identical queries={control.get('byte_identical_query_count', 0)}/{applicable}."
        )
        if applicable:
            control_metrics = control.get("metrics_by_effort", {})
            for effort in checkpoints:
                row = control_metrics.get(str(effort))
                if row is None:
                    lines.append(f"    effort {effort:>2}: unavailable")
                    continue
                lines.append(
                    f"    effort {effort:>2}: pass@1={row['query_pass_at_1']:.4f}; "
                    f"pass@2={row['query_pass_at_2']:.4f}; shape="
                    f"{row['shape_accuracy_diagnostic']:.4f}; pixels="
                    f"{row['valid_cell_pixel_accuracy_diagnostic']:.4f}."
                )
            lines.append(
                "    state comparison: "
                f"{comparison.get('interpretation', 'unreported')}; "
                "aggregate score deltas control-minus-intact="
                f"{dict((key, value) for key, value in comparison.get('score_deltas_control_minus_intact', {}).items() if '.tasks.' not in key)}."
            )
            current_l2 = comparison.get("synaptic_current_l2_by_step", {})
            spike_fraction = comparison.get("spike_hamming_fraction_by_step", [])
            voltage_l2 = comparison.get("voltage_l2_by_step", [])
            feedforward_l2 = current_l2.get("feedforward", [])
            recurrent_l2 = current_l2.get("recurrent", [])
            gathered_steps = evaluation.get("execution", {}).get(
                "gathered_steps", list(range(submission_checkpoint + 1))
            )
            submission_index = (
                gathered_steps.index(submission_checkpoint)
                if submission_checkpoint in gathered_steps
                else -1
            )
            if all(
                submission_index >= 0 and len(values) > submission_index
                for values in (
                    spike_fraction,
                    voltage_l2,
                    feedforward_l2,
                    recurrent_l2,
                )
            ):
                lines.append(
                    f"    step-{submission_checkpoint} state deltas: "
                    f"spike-Hamming fraction={spike_fraction[submission_index]:.6f}; "
                    f"voltage L2={voltage_l2[submission_index]:.6f}; "
                    "feedforward/recurrent current L2="
                    f"{feedforward_l2[submission_index]:.6f}/"
                    f"{recurrent_l2[submission_index]:.6f}."
                )
    determinism = evaluation.get("determinism", {})
    repeat_numeric = determinism.get("repeat_intact_numeric_evidence", {})
    ablation_numeric = determinism.get(
        "slot_ablation_checkpoint_zero_numeric_evidence", {}
    )

    def numeric_noise_line(label: str, evidence: object) -> str:
        if not isinstance(evidence, dict):
            return f"{label} numeric noise: unavailable."

        def formatted_values(value: object) -> dict[str, str]:
            if not isinstance(value, dict):
                return {}
            return {
                str(name): f"{float(number):.3e}"
                for name, number in value.items()
                if isinstance(number, Real) and not isinstance(number, bool)
            }

        steps = evidence.get("evaluated_steps")
        step_count = len(steps) if isinstance(steps, list) else "unreported"
        return (
            f"{label} numeric noise: queries={evidence.get('query_count', 'unreported')}; "
            f"steps={step_count}; spike mismatches="
            f"{evidence.get('spike_hamming_count', 'unreported')}; maximum RMS="
            f"{formatted_values(evidence.get('maximum_rms'))}; maximum absolute="
            f"{formatted_values(evidence.get('maximum_absolute'))}; dtypes="
            f"{evidence.get('intact_dtype_by_state', 'unreported')}; within tolerance="
            f"{evidence.get('within_declared_tolerance', 'unreported')}."
        )

    lines.extend(
        [
            "",
            (
                "Determinism gate: repeat intact byte-identical="
                f"{determinism.get('repeat_intact_state_byte_identical')}; "
                "compact logits byte-identical="
                f"{determinism.get('repeat_intact_compact_logits_byte_identical')}; "
                f"within tolerance={determinism.get('repeat_intact_within_tolerance')}; "
                f"decoded candidates exact={determinism.get('repeat_intact_decoded_candidates_exact')}; "
                f"metrics exact={determinism.get('repeat_intact_metrics_exact')}; "
                "slot-ablation checkpoint 0 matched="
                f"{determinism.get('slot_ablation_checkpoint_zero_within_tolerance')} "
                f"(byte-identical={determinism.get('slot_ablation_checkpoint_zero_byte_identical')}; "
                "state/logits within tolerance="
                f"{determinism.get('slot_ablation_checkpoint_zero_state_within_tolerance')}; "
                "decoded candidates exact="
                f"{determinism.get('slot_ablation_checkpoint_zero_decoded_candidates_exact')}; "
                "effort-0 metrics exact="
                f"{determinism.get('slot_ablation_checkpoint_zero_metrics_exact')}); "
                "per-query RMS tolerance (feature axis for logits; neuron axis "
                "for physical state)="
                f"{determinism.get('state_rms_tolerance', 'unreported')}; "
                f"metric absolute tolerance={determinism.get('metric_absolute_tolerance', 'unreported')}."
            ),
            numeric_noise_line("Repeat", repeat_numeric),
            numeric_noise_line("Ablation checkpoint-0", ablation_numeric),
        ]
    )
    compiler_warnings = [
        item
        for item in compiler_report.get("diagnostics", [])
        if item.get("level") == "warning"
    ]
    if compiler_warnings:
        lines.extend(["", "Compiler warnings (retained, not hidden):"])
        for item in compiler_warnings:
            lines.append(f"  - {item['message']}")
    lines.extend(
        [
            "",
            (
                "Qualification: structural="
                f"{qualification.get('full_structural_qualification', False)}, "
                "scientific="
                f"{qualification.get('full_scientific_qualification', False)}, "
                "approved 40% completion="
                f"{qualification.get('approved_completion_target_passed', False)}."
            ),
            f"Structural checks: {qualification.get('structural_checks', {})}.",
            f"Scientific checks: {qualification.get('scientific_checks', {})}.",
        ]
    )
    for reason in qualification.get("reasons_not_scientific", []):
        lines.append(f"  - {reason}")
    if qualification.get("approved_completion_target_passed", False):
        interpretation = (
            "This run satisfies the declared scientific protocol and the strict "
            "model-only 160-of-400 ARC completion target."
        )
    elif qualification.get("full_scientific_qualification", False):
        interpretation = (
            "This run satisfies the declared full scientific protocol gates. It is "
            "not evidence that the strict model-only 160-of-400 target was reached."
        )
    elif qualification.get("full_structural_qualification", False):
        interpretation = (
            "This run satisfies the full structural protocol gates only; it is not "
            "scientific model-quality evidence."
        )
    else:
        interpretation = (
            "This artifact does not satisfy the full structural or scientific "
            "qualification gates."
        )
    lines.extend(["", f"Interpretation boundary: {interpretation}"])
    return "\n".join(lines) + "\n"


def _plot(result: dict[str, object], path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configuration = result.get("configuration", {})
    checkpoints = tuple(configuration.get("checkpoints", CHECKPOINTS))
    submission_checkpoint = int(
        configuration.get("submission_checkpoint", checkpoints[-1])
    )
    metrics = result["evaluation"]["metrics_by_effort"]
    frozen_diagnostic = result["evaluation"].get("frozen_no_adaptation", {})
    trajectory = frozen_diagnostic.get(
        "aggregate_trajectory", result["evaluation"]["aggregate_trajectory"]
    )
    efforts = np.asarray(checkpoints)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(
        efforts,
        [metrics[str(value)]["strict_task_pass_at_1"] for value in efforts],
        marker="o",
        label="strict pass@1",
    )
    axes[0, 0].plot(
        efforts,
        [metrics[str(value)]["strict_task_pass_at_2"] for value in efforts],
        marker="o",
        label="strict pass@2",
    )
    axes[0, 0].set(title="Exact ARC quality", xlabel="latent steps", ylabel="rate")
    axes[0, 0].legend()
    steps = [row["step"] for row in trajectory]
    changed = [
        np.nan
        if row["mean_changed_cell_fraction"] is None
        else row["mean_changed_cell_fraction"]
        for row in trajectory
    ]
    axes[0, 1].plot(steps, changed, color="tab:blue", label="changed cells")
    axes[0, 1].set(
        title="Frozen no-adaptation output dynamics",
        xlabel="latent step",
        ylabel="changed-cell fraction",
    )
    entropy_axis = axes[0, 1].twinx()
    entropy_axis.plot(
        steps,
        [row["mean_predictive_entropy"] for row in trajectory],
        color="tab:orange",
        label="predictive entropy",
    )
    entropy_axis.set_ylabel("predictive entropy")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    entropy_handles, entropy_labels = entropy_axis.get_legend_handles_labels()
    axes[0, 1].legend(handles + entropy_handles, labels + entropy_labels)
    axes[1, 0].plot(
        steps,
        [row["mean_firing_rate"] for row in trajectory],
        color="tab:blue",
        label="firing rate",
    )
    axes[1, 0].plot(
        steps,
        [row["near_saturation_fraction"] for row in trajectory],
        color="tab:green",
        linestyle="--",
        label="saturated queries",
    )
    axes[1, 0].set(
        title="Spike and voltage dynamics",
        xlabel="latent step",
        ylabel="spike-derived fraction",
    )
    voltage_axis = axes[1, 0].twinx()
    voltage_axis.plot(
        steps,
        [row["mean_voltage_l2"] for row in trajectory],
        color="tab:red",
        label="voltage L2",
    )
    voltage_axis.set_ylabel("voltage L2")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    voltage_handles, voltage_labels = voltage_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles + voltage_handles, labels + voltage_labels)
    controls = result["evaluation"]["controls"]
    names = ["no_context", "shuffled_demonstrations", "slot_ablation"]
    exact_deltas = []
    diagnostic_deltas = []
    state_effects = []
    gathered_steps = result["evaluation"].get("execution", {}).get(
        "gathered_steps", list(range(submission_checkpoint + 1))
    )
    submission_index = gathered_steps.index(submission_checkpoint)
    for name in names:
        comparison = controls.get(name, {}).get("trajectory_comparison", {})
        score_deltas = comparison.get("score_deltas_control_minus_intact", {})
        exact_key = f"{submission_checkpoint}.query_pass_at_2"
        pixel_key = (
            f"{submission_checkpoint}.valid_cell_pixel_accuracy_diagnostic"
        )
        if exact_key not in score_deltas:
            exact_deltas.append(np.nan)
            diagnostic_deltas.append(np.nan)
            state_effects.append(np.nan)
            continue
        exact_deltas.append(score_deltas[exact_key])
        diagnostic_deltas.append(score_deltas[pixel_key])
        state_effects.append(
            comparison["spike_hamming_fraction_by_step"][submission_index]
        )
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.36
    axes[1, 1].bar(
        positions - width / 2, exact_deltas, width=width, label="pass@2 delta"
    )
    axes[1, 1].bar(
        positions + width / 2,
        diagnostic_deltas,
        width=width,
        label="pixel diagnostic delta",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        title=f"Control deltas at effort {submission_checkpoint}",
        ylabel="control minus intact",
        xticks=positions,
        xticklabels=names,
    )
    axes[1, 1].tick_params(axis="x", rotation=15)
    state_axis = axes[1, 1].twinx()
    state_axis.plot(
        positions,
        state_effects,
        color="black",
        marker="D",
        linestyle="none",
        label="spike-Hamming fraction",
    )
    state_axis.set_ylabel(f"state effect at step {submission_checkpoint}")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    state_handles, state_labels = state_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + state_handles, labels + state_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run training, frozen interventions, and evidence generation.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.

    Returns
    -------
    dict
        JSON-safe complete experiment evidence.
    """
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    training_profile = _TrainingProfile() if config.runtime_profile else None
    _emit_progress("run", 0, 1, started)
    environment_snapshot = dict(os.environ)
    if config.device == "gpu":
        pre_device_assessment = require_pre_device_gpu_environment(os.environ)
        pre_device_report: dict[str, object] = {
            "applicable": True,
            **pre_device_assessment.to_dict(),
        }
    else:
        pre_device_report = {
            "applicable": False,
            "status": "not_applicable_cpu",
        }
    device, device_report = _resolve_device(config.device)
    monitor = _make_gpu_monitor(device) if config.device == "gpu" else None
    if monitor is not None:
        monitor.start()
    try:
        data_started = time.perf_counter()
        _emit_progress("data", 0, 1, data_started)
        data = _load_data(config)
        _emit_progress("data", 1, 1, data_started)
        phase_seconds["data"] = time.perf_counter() - data_started
        rows = _row_config(config)
        model_started = time.perf_counter()
        _emit_progress("model", 0, 1, model_started)
        model = _make_model(config, rows, batch_size=1, device=device)
        _emit_progress("model", 1, 1, model_started)
        phase_seconds["model"] = time.perf_counter() - model_started
        checkpoint = config.parameter_checkpoint
        training_started = time.perf_counter()
        if checkpoint is not None and checkpoint.exists():
            _emit_progress("training", 0, 1, training_started)
            training = _restored_training_report(
                model,
                config,
                _read_parameter_checkpoint(
                    model,
                    checkpoint,
                    effort_schedule=config.effort_schedule,
                    effort_distillation_weight=config.effort_distillation_weight,
                ),
            )
            _emit_progress("training", 1, 1, training_started)
        else:
            fitted = (
                model
                if config.training_batch_size == 1
                else _make_model(
                    config, rows, batch_size=config.training_batch_size, device=device
                )
            )
            initial_digest = _restore_initial_parameters(fitted, config)
            chunk_size = _resolved_training_chunk_size(config)
            total_chunks = max(1, math.ceil(config.training_updates / chunk_size))
            _emit_progress("training", 0, total_chunks, training_started)
            checkpoint_callback = _checkpoint_writer(fitted, config)

            def on_chunk(index: int) -> None:
                if checkpoint_callback is not None:
                    checkpoint_callback(index)
                _emit_progress("training", index + 1, total_chunks, training_started)

            training = _train_model(
                fitted,
                _training_chunks(data, config, rows),
                config,
                on_chunk,
                training_profile,
            )
            if initial_digest is not None:
                training["initial_checkpoint"] = str(config.initial_checkpoint)
                training["initial_checkpoint_sha256"] = initial_digest
            if fitted is not model:
                _copy_parameters(fitted, model)
            if checkpoint is not None:
                training["parameter_checkpoint"] = str(checkpoint)
                training["parameter_checkpoint_sha256"] = _write_parameter_checkpoint(
                    model,
                    checkpoint,
                    effort_schedule=config.effort_schedule,
                    effort_distillation_weight=config.effort_distillation_weight,
                )
        phase_seconds["training"] = time.perf_counter() - training_started
        evaluation_started = time.perf_counter()
        _emit_progress("evaluation", 0, 1, evaluation_started)
        evaluation = _evaluate(model, data, config, rows, device)
        _emit_progress("evaluation", 1, 1, evaluation_started)
        phase_seconds["evaluation"] = time.perf_counter() - evaluation_started
    finally:
        monitor_report = monitor.stop() if monitor is not None else None
    memory_stats = _device_memory_stats(device)
    device_report["memory_stats"] = memory_stats
    device_report["memory_stats_capture"] = "after training and evaluation"
    device_report["pre_device_gpu_environment"] = pre_device_report
    if monitor_report is None:
        device_report["gpu_monitor"] = {
            "applicable": False,
            "status": "not_applicable_cpu",
        }
        device_report["gpu_runtime_safety"] = {
            "applicable": False,
            "status": "not_applicable_cpu",
            "full_qualification_safe": False,
        }
    else:
        device_report["gpu_monitor"] = {
            "applicable": True,
            **monitor_report,
        }
        device_report["gpu_runtime_safety"] = _gpu_runtime_safety_report(
            config,
            environment_snapshot,
            memory_stats,
            monitor_report,
        )
    manifests = [item.manifest.to_dict() for item in data.loaded]
    memory_architecture = _memory_architecture_report(
        config,
        rows,
        training_batch_size=config.training_batch_size,
        evaluation_batch_size=int(evaluation["query_count"]),
    )
    memory_implementation = _model_memory_report(model)
    memory_contract = {
        "mode": "reasoning_mode",
        "memory_width": "context_memory_width",
        "key_feature_width": "raw_key_feature_width",
        "value_feature_width": "raw_value_feature_width",
    }
    for implementation_name, architecture_name in memory_contract.items():
        if memory_implementation.get(implementation_name) != memory_architecture.get(
            architecture_name
        ):
            raise ValueError(
                "model and experiment associative-memory reports disagree on "
                f"{implementation_name}"
            )
    model_report = {
        "neuron_count": model.neuron_count,
        "recurrent_edge_count": model.recurrent_edge_count,
        "sparse_backend": config.sparse_backend,
        "slot_count": model.slot_count,
        "neurons_per_slot": 64,
        "input_width": rows.input_width,
        "decoder_mode": model.config.decoder_mode,
        "refinement_steps": model.config.refinement_steps,
        "refinement_mixer": getattr(
            model.config, "refinement_mixer", config.refinement_mixer
        ),
        "memory_value_softcap_beta": config.memory_value_softcap_beta,
        "reasoning_query_softcap_beta": config.reasoning_query_softcap_beta,
        "training_output_width": model.config.training_output_width,
        "checkpoint_output_width": model.config.checkpoint_output_width,
        "compact_output_width": model.config.compact_output_width,
        "color_rank": model.config.color_rank,
        "parameter_count": _parameter_count(parameter_snapshot(model)),
        "component_types": {
            "neuron": type(model.neu).__name__,
            "feedforward_projection_wrapper": type(model.ff_syn).__name__,
            "feedforward_projection": type(model.ff_syn.comm).__name__,
            "feedforward_synapse": type(model.ff_syn.syn).__name__,
            "feedforward_output": type(model.ff_syn.out).__name__,
            "recurrent_projection_wrapper": type(model.rec_syn).__name__,
            "recurrent_projection": type(model.rec_syn.comm).__name__,
            "recurrent_synapse": type(model.rec_syn.syn).__name__,
            "recurrent_output": type(model.rec_syn.out).__name__,
        },
        **memory_architecture,
        "associative_memory_implementation": memory_implementation,
        "neuron_typing": model.neuron_typing_report(),
    }
    configuration = config.to_dict()
    configuration_sha256 = hashlib.sha256(
        msgspec.json.encode(configuration, order="sorted")
    ).hexdigest()
    result: dict[str, object] = {
        "schema_version": 2,
        "protocol_version": 2,
        "claim_boundary": CLAIM_BOUNDARY,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "device": device_report,
        "model": model_report,
        "software": _software_report(pre_device_report, environment_snapshot),
        "implementation": _implementation_report(),
        "data_manifests": manifests,
        "data_summary": _data_summary(data, manifests, evaluation),
        "training": training,
        "evaluation": evaluation,
        "runtime_seconds": time.perf_counter() - started,
    }
    if config.runtime_profile:
        result["runtime_profile"] = {
            "enabled": True,
            "phase_seconds": phase_seconds,
            "training": (
                {} if training_profile is None else training_profile.to_dict()
            ),
            "synchronization": {
                "diagnostic_device_barriers": True,
                "final_throughput_measurement": "run without --profile",
            },
        }
    result["qualification"] = _qualification(
        config, data, training, evaluation, device_report, model_report
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "data_manifest.json"
    result_path = config.output_dir / "result.json"
    report_path = config.output_dir / "report.txt"
    figure_path = config.output_dir / "latent_reasoning.png"
    artifact_manifest_path = config.output_dir / "artifact_manifest.json"
    result["artifacts"] = {
        "data_manifest": str(manifest_path),
        "result": str(result_path),
        "report": str(report_path),
        "figure": str(figure_path),
        "manifest": str(artifact_manifest_path),
    }
    artifacts_started = time.perf_counter()
    _emit_progress("artifacts", 0, 1, artifacts_started)
    manifest_path.write_bytes(msgspec.json.encode(manifests))
    result_path.write_bytes(msgspec.json.encode(result))
    report_path.write_text(_render_report(result), encoding="utf-8")
    _plot(result, figure_path)
    _emit_progress("artifacts", 1, 1, artifacts_started)
    if config.runtime_profile:
        phase_seconds["artifacts"] = time.perf_counter() - artifacts_started
        result["runtime_profile"]["phase_seconds"] = phase_seconds
        report_path.write_text(_render_report(result), encoding="utf-8")
        result_path.write_bytes(msgspec.json.encode(result))
    artifact_manifest_path.write_bytes(
        msgspec.json.encode(
            _artifact_manifest(
                {
                    "data_manifest": manifest_path,
                    "result": result_path,
                    "report": report_path,
                    "figure": figure_path,
                }
            ),
            order="sorted",
        )
    )
    _emit_progress("run", 1, 1, started)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=pathlib.Path)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("var/example21")
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--neurons", type=int, default=4096)
    parser.add_argument(
        "--recurrent-edges", type=int, default=FULL_SCALE_RECURRENT_EDGES
    )
    parser.add_argument("--context-memory-width", type=int, default=32)
    parser.add_argument("--memory-decay", type=float, default=1.0)
    parser.add_argument(
        "--memory-read-transform",
        choices=("linear", "gated", "gated_rms"),
        default="linear",
    )
    parser.add_argument("--memory-read-interval", type=int, default=1)
    parser.add_argument(
        "--latent-residual-mixer",
        choices=("none", "attention_residual"),
        default="none",
    )
    parser.add_argument("--latent-residual-block-size", type=int, default=10)
    parser.add_argument(
        "--memory-coding",
        choices=(
            "frozen",
            "learned_keys",
            "learned_write",
            "learned_update",
            "delta_write",
            "situ_glu_update",
        ),
        default=None,
    )
    parser.add_argument(
        "--trace-engine",
        choices=("pp_prop", "d_rtrl"),
        default="pp_prop",
    )
    parser.add_argument(
        "--neuron-typing",
        choices=("none", "ei_dale"),
        default="none",
    )
    parser.add_argument("--excitatory-fraction", type=float, default=0.8)
    parser.add_argument("--max-demonstrations", type=int, default=10)
    parser.add_argument("--latent-steps", type=int, default=60)
    parser.add_argument("--submission-checkpoint", type=int, default=None)
    parser.add_argument("--training-updates", type=int, default=260)
    parser.add_argument("--training-chunk-size", type=int, default=0)
    parser.add_argument("--training-batch-size", type=int, default=32)
    parser.add_argument("--training-bank-size", type=int, default=0)
    parser.add_argument("--training-workers", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--sparse-backend", choices=("default", "jax_raw"), default="default"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="cosine"
    )
    parser.add_argument("--lr-warmup-fraction", type=float, default=0.0)
    parser.add_argument(
        "--effort-schedule",
        choices=("uniform", "progressive"),
        default="uniform",
    )
    parser.add_argument("--effort-distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", choices=("adam", "adamw", "muon"), default="muon"
    )
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--copy-residual-gain", type=float, default=0.0)
    parser.add_argument("--row-head-carrier-scale", type=float, default=1.0)
    parser.add_argument("--row-copy-gate", action="store_true")
    parser.add_argument("--row-copy-gate-bias", type=float, default=-4.0)
    parser.add_argument("--shape-head-carrier-scale", type=float, default=1.0)
    parser.add_argument("--memory-value-softcap-beta", type=float, default=4.0)
    parser.add_argument("--reasoning-query-softcap-beta", type=float, default=25.0)
    parser.add_argument("--row-head-carrier-gate", action="store_true")
    parser.add_argument(
        "--row-head-modulation", choices=("none", "bilinear"), default="none"
    )
    parser.add_argument("--row-head-modulation-rank", type=int, default=64)
    parser.add_argument(
        "--refinement-mixer",
        choices=("linear", "carrier_gate", "attention_residual"),
        default="linear",
    )
    parser.add_argument("--adaptation-learning-rate", type=float, default=5e-5)
    parser.add_argument("--adaptation-epochs", type=int, default=1)
    parser.add_argument("--task-local-adaptation", action="store_true")
    parser.add_argument(
        "--evaluation-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptation-update-schedule",
        choices=("per_episode", "per_tick"),
        default="per_tick",
    )
    parser.add_argument("--parameter-checkpoint", type=pathlib.Path)
    parser.add_argument("--initial-checkpoint", type=pathlib.Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--training-holdout-tasks", type=int, default=0)
    parser.add_argument("--adaptation-task-group", type=int, default=20)
    parser.add_argument("--balanced-color-loss", action="store_true")
    parser.add_argument(
        "--decoder-mode",
        choices=("legacy_cp", "row_refinement", "latent_row_decode"),
        default="latent_row_decode",
    )
    parser.add_argument("--evaluation-task-limit", type=int)
    parser.add_argument("--ablation-slot", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument(
        "--primary-candidate-mode",
        choices=("model_only", "rule_then_model"),
        default="model_only",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        if (
            args.neurons != FULL_SCALE_NEURON_COUNT
            or args.recurrent_edges != FULL_SCALE_RECURRENT_EDGES
        ):
            raise ValueError("--smoke owns its reduced neuron and edge scale")
        return ExperimentConfig.smoke_config(
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            context_memory_width=args.context_memory_width,
            memory_decay=args.memory_decay,
            memory_read_transform=args.memory_read_transform,
            memory_read_interval=args.memory_read_interval,
            latent_residual_mixer=args.latent_residual_mixer,
            latent_residual_block_size=args.latent_residual_block_size,
            memory_coding=args.memory_coding,
            trace_engine=args.trace_engine,
            neuron_typing=args.neuron_typing,
            excitatory_fraction=args.excitatory_fraction,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            refinement_mixer=args.refinement_mixer,
            lr_schedule=args.lr_schedule,
            lr_warmup_fraction=args.lr_warmup_fraction,
            effort_schedule=args.effort_schedule,
            effort_distillation_weight=args.effort_distillation_weight,
            balanced_color_loss=args.balanced_color_loss,
            decoder_mode=args.decoder_mode,
            runtime_profile=args.profile,
            sparse_backend=args.sparse_backend,
        )
    return ExperimentConfig(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        neuron_count=args.neurons,
        recurrent_edges=args.recurrent_edges,
        context_memory_width=args.context_memory_width,
        memory_decay=args.memory_decay,
        memory_read_transform=args.memory_read_transform,
        memory_read_interval=args.memory_read_interval,
        latent_residual_mixer=args.latent_residual_mixer,
        latent_residual_block_size=args.latent_residual_block_size,
        memory_coding=args.memory_coding or "learned_update",
        trace_engine=args.trace_engine,
        neuron_typing=args.neuron_typing,
        excitatory_fraction=args.excitatory_fraction,
        max_demonstrations=args.max_demonstrations,
        latent_steps=args.latent_steps,
        submission_effort=args.submission_checkpoint,
        training_updates=args.training_updates,
        training_chunk_size=args.training_chunk_size,
        training_batch_size=args.training_batch_size,
        training_bank_size=args.training_bank_size,
        training_workers=args.training_workers,
        runtime_profile=args.profile,
        sparse_backend=args.sparse_backend,
        adaptation_learning_rate=args.adaptation_learning_rate,
        adaptation_epochs=args.adaptation_epochs,
        task_local_adaptation=args.task_local_adaptation,
        evaluation_controls=args.evaluation_controls,
        adaptation_update_schedule=args.adaptation_update_schedule,
        parameter_checkpoint=args.parameter_checkpoint,
        initial_checkpoint=args.initial_checkpoint,
        checkpoint_every=args.checkpoint_every,
        training_holdout_tasks=args.training_holdout_tasks,
        adaptation_task_group=args.adaptation_task_group,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_warmup_fraction=args.lr_warmup_fraction,
        effort_schedule=args.effort_schedule,
        effort_distillation_weight=args.effort_distillation_weight,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        copy_residual_gain=args.copy_residual_gain,
        row_head_carrier_scale=args.row_head_carrier_scale,
        row_head_carrier_gate=args.row_head_carrier_gate,
        row_head_modulation=args.row_head_modulation,
        row_head_modulation_rank=args.row_head_modulation_rank,
        row_copy_gate=args.row_copy_gate,
        row_copy_gate_bias=args.row_copy_gate_bias,
        shape_head_carrier_scale=args.shape_head_carrier_scale,
        refinement_mixer=args.refinement_mixer,
        memory_value_softcap_beta=args.memory_value_softcap_beta,
        reasoning_query_softcap_beta=args.reasoning_query_softcap_beta,
        balanced_color_loss=args.balanced_color_loss,
        decoder_mode=args.decoder_mode,
        evaluation_task_limit=args.evaluation_task_limit,
        ablation_slot=args.ablation_slot,
        structural_only=args.structural_only,
        primary_candidate_mode=args.primary_candidate_mode,
    )


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run Example 21 from command-line arguments.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments excluding the executable name. ``None`` uses ``sys.argv``.

    Returns
    -------
    dict
        Structured result also written under ``--output-dir``.
    """
    config = _config_from_args(_parser().parse_args(argv))
    result = run_experiment(config)
    print(_render_report(result), end="")
    return result


if __name__ == "__main__":
    main()
