"""Recurrent spiking and associative ARC workspace for Example 21.

The opt-in memory architecture adds a dense fast-weight ``S_K`` with a
diagonal-friendly self-transition and a separate continuous reasoning carrier
``H_r``.  Demonstration rows write fixed nonlinear key/value codes.  The full
read policy re-reads frozen ``S_K`` at the query and every exactly-zero latent
tick; the query-only intervention reads it only at the ordinary query.  All
trainable memory operations use BrainTrace ETP primitives and pp-prop remains
the production learning rule.  A memory width of zero retains the original
reservoir-only architecture and state and parameter paths.

The color head is a compact CP factorization.  It emits row, column, and color
factors while the network is running and expands them to independent
``30 x 30 x 10`` logits only at a requested checkpoint.  This avoids a dense
``readout_width x 9000`` parameter matrix and avoids materializing 9,000 logits
at every context row.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Literal, NamedTuple

import brainpy.state as bpstate
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import braintrace

try:
    from examples.pp_prop.latent_workspace_refinement import (
        RowRefinementLayout,
        build_refinement_feedback_event,
        capture_query_rows,
        next_reasoning_index,
        refinement_output_logits,
        refinement_training_logits,
        scatter_answer_rows,
        split_refinement_output_logits,
    )
    from examples.pp_prop.latent_workspace_resource_safety import (
        DEFAULT_MAX_EDGES_PER_NEURON,
        assess_recurrent_edge_budget,
    )
except ImportError:
    from latent_workspace_refinement import (
        RowRefinementLayout,
        build_refinement_feedback_event,
        capture_query_rows,
        next_reasoning_index,
        refinement_output_logits,
        refinement_training_logits,
        scatter_answer_rows,
        split_refinement_output_logits,
    )
    from latent_workspace_resource_safety import (
        DEFAULT_MAX_EDGES_PER_NEURON,
        assess_recurrent_edge_budget,
    )

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
NEURONS_PER_SLOT = 64
MEMORY_KEY_RFF_GAMMA = 2.0

#: Storage-coding trainability levels, in increasing order of what learns.
MEMORY_CODINGS = ("frozen", "learned_keys", "learned_write")

#: Codings whose *retrieval* key projection is a trainable ETP layer.
LEARNED_RETRIEVAL_KEY_CODINGS = ("learned_keys", "learned_write")

#: Eligibility-trace engines the model can compile under.
TRACE_ENGINES = ("pp_prop", "d_rtrl")

#: Row-refinement proposal mixers. Only ``attention_residual`` implements the
#: paper mechanism; the other two are retained benchmark ablations.
REFINEMENT_MIXERS = ("linear", "carrier_gate", "attention_residual")

#: Neuron-typing modes for the recurrent population (spec
#: 2026-08-21-example21-neuron-types.md).  ``"none"`` is the untyped legacy
#: substrate; ``"ei_dale"`` assigns a seeded binary E/I split and constrains
#: recurrent weight signs by presynaptic type (Dale's law).
NEURON_TYPINGS = ("none", "ei_dale")

#: Seed offset of the dedicated neuron-type random stream, so enabling typing
#: never perturbs the existing topology (`seed`), parameter (`seed + 1`), or
#: memory (`seed + 101..105`) streams.
_NEURON_TYPE_SEED_OFFSET = 7

#: Reported key-map name per coding; provenance hashes stay on the frozen bases.
_MEMORY_KEY_MAP_NAMES = {
    "frozen": "fixed_rff_cosine",
    "learned_keys": "learned_rff_cosine_retrieval_path",
    "learned_write": "learned_rff_cosine_write_and_retrieval",
}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive non-boolean integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative non-boolean integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative non-boolean integer")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return result


def _unit_interval_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite real scalar in [0, 1]")
    return result


def _optional_index(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _index_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of nonnegative integers")
    indices = tuple(_nonnegative_integer(index, name) for index in value)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicate indices")
    return indices


@dataclass(frozen=True)
class ModelConfig:
    """Configure the ARC recurrent spiking workspace.

    Parameters
    ----------
    input_width : int
        Width of one encoded ARC row event.
    batch_size : int, default=1
        Native batch dimension used by the BrainState model.
    neuron_count : int, default=2048
        Number of physical LIF neurons.  It must be divisible by 64 so slot
        controls always refer to exact 64-neuron ranges.
    recurrent_edges : int, default=16384
        Exact number of directed non-self recurrent edges.
    max_latent_steps : int, default=32
        Maximum allowed zero-input recurrent steps.
    readout_width : int, default=128
        Shared low-rank readout bottleneck.
    color_rank : int, default=16
        CP rank of the spatial color-logit tensor.
    membrane_tau_ms, feedforward_tau_ms, recurrent_tau_ms : float
        Physical time constants for neurons and both synapses.
    time_step_ms : float, default=1.0
        Duration represented by one event or latent step.
    input_gain, recurrent_gain : float
        Initialization gains for the feed-forward and recurrent projections.
    trace_decay : float, default=0.9
        pp-prop eligibility-trace decay in ``[0, 1)``.
    context_memory_width : int, default=0
        Width of the opt-in square contextual fast-weight memory.  Zero keeps
        the legacy reservoir architecture byte-compatible.
    memory_decay : float, default=1.0
        Self-decay of contextual memory on valid demonstration ticks.
    demonstration_phase_index, query_phase_index : int, optional
        Event channels identifying demonstration and query rows in memory mode.
    input_side_valid_index, output_side_valid_index : int, optional
        Event channels gating complete input/output association writes.
    memory_key_indices, memory_value_indices : tuple of int
        Input-event features projected by fixed deterministic key/value bases.
    memory_coding : {"frozen", "learned_keys", "learned_write"}
        Storage-coding trainability. ``"frozen"`` (default) keeps the fixed
        random Fourier key map and fixed tanh value map bit-exactly.
        ``"learned_keys"`` replaces the key projection with a trainable ETP
        linear layer initialized to the frozen basis, function-identical to
        ``"frozen"`` at initialization. The learned key trains through the
        retrieval path only (query encoding and read); the write-side key is
        gradient-detached because pp-prop's position-preserving requirement
        excludes gradients through the *unfused* outer-product memory write,
        and the value map stays fixed for the same structural reason.
        ``"learned_write"`` additionally routes the write through
        ``braintrace.outer_write``, a fused ETP primitive that owns both
        projections, so *what gets stored* carries gradient too; it keeps the
        ``"learned_keys"`` retrieval path, and its write-side projections are
        separate parameters initialized from the same frozen bases (a single
        parameter cannot be owned by two ETP primitives).
    trace_engine : {"pp_prop", "d_rtrl"}, default="pp_prop"
        Eligibility-trace engine the model compiles under. ``"pp_prop"``
        (default) keeps the IO-factorized ES-D-RTRL coordinate with the
        configured ``trace_decay``. ``"d_rtrl"`` selects the per-parameter
        exact-trace coordinate (diagonal recurrence scope, true hidden
        Jacobian, no decay knob) — much heavier in memory, but it carries the
        write projections' pairing gradient exactly where pp-prop's rank-1
        collapse loses it (spec ``2026-08-21-etp-outer-write-drtrl-trace.md``).
    decoder_mode : {"legacy_cp", "row_refinement"}, default="legacy_cp"
        Output architecture. Row refinement captures the query and constructs
        an explicit answer with one learned recurrent row per latent tick.
    refinement_steps : int, default=30
        Number of latent row ticks used by the opt-in decoder. It must be a
        positive multiple of 30 and no larger than ``max_latent_steps``.
    refinement_layout : RowRefinementLayout, optional
        Complete row-event layout required by ``row_refinement`` mode.
    copy_residual_gain : float, default=0.0
        Fixed logit magnitude added to the answer row head's output at the
        query's own colour for every occupied column — an identity residual
        that lets training learn deviations from copy instead of the copy map
        itself.  Zero keeps the bare head bit-exactly.
    row_head_carrier_scale : float, default=1.0
        Constant multiplier on the carrier block of the answer row head's
        input only.  Zero starves the row head of the task-identifying
        carrier so training cannot displace the copy path with
        task-conditional fits.
    row_head_carrier_gate : bool, default=False
        Compatibility spelling for ``refinement_mixer="carrier_gate"``.
        Replace the single answer row head with an event-only head plus a
        carrier head whose contribution is multiplied elementwise by
        ``tanh(w)``, where ``w`` is a zero-initialised trainable per-logit
        gate vector (a bias-free 1×300 linear fed with ones, so the gate is
        ETP-tracked and matches the hidden group's width, which the
        eligibility-trace VJP requires).  At initialisation the
        row answer is exactly carrier-free; training must buy carrier access
        through the gate.  Incompatible with a non-default
        ``row_head_carrier_scale``.
    shape_head_carrier_scale : float, default=1.0
        Constant multiplier on the carrier block of the answer shape head's
        input only.  Zero makes the shape answer a pure function of the row
        events, testing whether the shape head suffers the same carrier
        displacement as the row head.
    refinement_mixer : {"linear", "carrier_gate", "attention_residual"}
        Refinement proposal mixer. ``"linear"`` preserves the direct heads,
        ``"carrier_gate"`` selects the retained carrier-gating ablation, and
        ``"attention_residual"`` applies learned source-axis attention across
        completed refinement sweeps.
    memory_value_softcap_beta : float, default=1.0
        Softcap magnitude of the memory value coding,
        ``softcap(x, beta) = beta * tanh(x / beta)``, bounding the stored
        value code to ``(-beta, beta)``.  The default 1.0 reproduces the
        legacy ``tanh`` value map bit-exactly.
    reasoning_query_softcap_beta : float, default=1.0
        Softcap magnitude of the iterative reasoning query on latent steps.
        The default 1.0 reproduces the legacy ``tanh`` cap bit-exactly.
    event_valid_index : int, default=0
        Row-event channel whose one means a context row advances state.  Latent
        steps use a separate advance gate, keeping their external vector
        exactly zero.
    neuron_typing : {"none", "ei_dale"}, default="none"
        Recurrent neuron-type structure.  ``"none"`` keeps the untyped legacy
        substrate bit-exactly.  ``"ei_dale"`` assigns every neuron a binary
        excitatory/inhibitory type deterministically from the seed and
        constrains every recurrent edge's sign to its *presynaptic* neuron's
        type (Dale's law) at initialization; training must re-project after
        each optimizer step via
        :meth:`LatentWorkspaceModel.project_recurrent_dale_weights`.
    excitatory_fraction : float, default=0.8
        Fraction of neurons assigned the excitatory type under ``ei_dale``.
        The realized excitatory count is ``round(fraction * neuron_count)``
        and both types must be non-empty.  Supplying a non-default fraction
        with ``neuron_typing="none"`` is rejected (fail closed).
    seed : int, default=2108
        Seed for all topology and parameter randomness.  Random values are
        drawn exclusively through :mod:`brainstate.random`.
    sparse_backend : str, optional
        Optional ``brainevent.CSR`` execution backend.

    Examples
    --------
    .. code-block:: python

        >>> config = ModelConfig(input_width=828)
        >>> (config.neuron_count, config.recurrent_edges, config.slot_count)
        (2048, 16384, 32)
    """

    input_width: int
    batch_size: int = 1
    neuron_count: int = 2048
    recurrent_edges: int = 16384
    max_latent_steps: int = 32
    readout_width: int = 128
    color_rank: int = 16
    membrane_tau_ms: float = 20.0
    feedforward_tau_ms: float = 40.0
    recurrent_tau_ms: float = 10.0
    time_step_ms: float = 1.0
    input_gain: float = 4.0
    recurrent_gain: float = 0.8
    trace_decay: float = 0.9
    event_valid_index: int = 0
    neuron_typing: Literal["none", "ei_dale"] = "none"
    excitatory_fraction: float = 0.8
    context_memory_width: int = 0
    memory_decay: float = 1.0
    demonstration_phase_index: int | None = None
    query_phase_index: int | None = None
    input_side_valid_index: int | None = None
    output_side_valid_index: int | None = None
    memory_key_indices: tuple[int, ...] = ()
    memory_value_indices: tuple[int, ...] = ()
    memory_coding: Literal["frozen", "learned_keys", "learned_write"] = "frozen"
    trace_engine: Literal["pp_prop", "d_rtrl"] = "pp_prop"
    decoder_mode: Literal["legacy_cp", "row_refinement"] = "legacy_cp"
    refinement_steps: int = MAX_GRID_SIZE
    refinement_layout: RowRefinementLayout | None = None
    copy_residual_gain: float = 0.0
    row_head_carrier_scale: float = 1.0
    row_head_carrier_gate: bool = False
    shape_head_carrier_scale: float = 1.0
    refinement_mixer: Literal[
        "linear", "carrier_gate", "attention_residual"
    ] = "linear"
    memory_value_softcap_beta: float = 1.0
    reasoning_query_softcap_beta: float = 1.0
    seed: int = 2108
    sparse_backend: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "batch_size",
            "neuron_count",
            "recurrent_edges",
            "max_latent_steps",
            "readout_width",
            "color_rank",
            "refinement_steps",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "context_memory_width",
            _nonnegative_integer(self.context_memory_width, "context_memory_width"),
        )
        object.__setattr__(
            self,
            "event_valid_index",
            _nonnegative_integer(self.event_valid_index, "event_valid_index"),
        )
        if self.event_valid_index >= self.input_width:
            raise ValueError("event_valid_index must be smaller than input_width")
        for name in (
            "membrane_tau_ms",
            "feedforward_tau_ms",
            "recurrent_tau_ms",
            "time_step_ms",
            "input_gain",
            "recurrent_gain",
            "memory_value_softcap_beta",
            "reasoning_query_softcap_beta",
        ):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if self.neuron_count % NEURONS_PER_SLOT:
            raise ValueError(
                f"neuron_count must be divisible by {NEURONS_PER_SLOT} for exact "
                "slot ablation"
            )
        edge_budget = assess_recurrent_edge_budget(
            self.neuron_count,
            self.recurrent_edges,
            max_edges_per_neuron=DEFAULT_MAX_EDGES_PER_NEURON,
        )
        if self.recurrent_edges > edge_budget.no_self_edge_cap:
            raise ValueError(
                f"recurrent_edges {self.recurrent_edges} exceeds no-self capacity "
                f"{edge_budget.no_self_edge_cap}"
            )
        if self.recurrent_edges > edge_budget.policy_edge_cap:
            raise ValueError(
                f"recurrent_edges {self.recurrent_edges} exceeds "
                f"{DEFAULT_MAX_EDGES_PER_NEURON} edges per neuron policy cap "
                f"{edge_budget.policy_edge_cap}"
            )
        if isinstance(self.trace_decay, (bool, np.bool_)) or not isinstance(
            self.trace_decay, Real
        ):
            raise TypeError("trace_decay must be a finite real scalar in [0, 1)")
        trace_decay = float(self.trace_decay)
        if not math.isfinite(trace_decay) or not 0.0 <= trace_decay < 1.0:
            raise ValueError("trace_decay must be a finite real scalar in [0, 1)")
        object.__setattr__(self, "trace_decay", trace_decay)
        object.__setattr__(
            self,
            "memory_decay",
            _unit_interval_real(self.memory_decay, "memory_decay"),
        )
        object.__setattr__(
            self,
            "copy_residual_gain",
            _nonnegative_real(self.copy_residual_gain, "copy_residual_gain"),
        )
        object.__setattr__(
            self,
            "row_head_carrier_scale",
            _nonnegative_real(self.row_head_carrier_scale, "row_head_carrier_scale"),
        )
        if self.row_head_carrier_gate and self.row_head_carrier_scale != 1.0:
            raise ValueError(
                "row_head_carrier_gate replaces row_head_carrier_scale; "
                "leave the scale at its default of 1.0"
            )
        if not isinstance(self.refinement_mixer, str):
            raise TypeError(
                "refinement_mixer must be 'linear', 'carrier_gate' or "
                "'attention_residual'"
            )
        if self.refinement_mixer not in REFINEMENT_MIXERS:
            raise ValueError(
                "refinement_mixer must be 'linear', 'carrier_gate' or "
                "'attention_residual'"
            )
        if self.row_head_carrier_gate:
            if self.refinement_mixer == "attention_residual":
                raise ValueError(
                    "row_head_carrier_gate conflicts with "
                    "refinement_mixer='attention_residual'; run the carrier "
                    "ablation separately"
                )
            if self.refinement_mixer == "linear":
                object.__setattr__(self, "refinement_mixer", "carrier_gate")
        if self.refinement_mixer == "carrier_gate":
            object.__setattr__(self, "row_head_carrier_gate", True)
        object.__setattr__(
            self,
            "shape_head_carrier_scale",
            _nonnegative_real(
                self.shape_head_carrier_scale, "shape_head_carrier_scale"
            ),
        )
        if self.refinement_mixer == "attention_residual" and (
            self.copy_residual_gain != 0.0
            or self.row_head_carrier_scale != 1.0
            or self.shape_head_carrier_scale != 1.0
        ):
            raise ValueError(
                "refinement_mixer='attention_residual' cannot be combined with "
                "copy_residual_gain or non-default row/shape carrier scales; "
                "run those legacy ablations separately"
            )
        for name in (
            "demonstration_phase_index",
            "query_phase_index",
            "input_side_valid_index",
            "output_side_valid_index",
        ):
            index = _optional_index(getattr(self, name), name)
            if index is not None and index >= self.input_width:
                raise ValueError(f"{name} must be smaller than input_width")
            object.__setattr__(self, name, index)
        for name in ("memory_key_indices", "memory_value_indices"):
            indices = _index_tuple(getattr(self, name), name)
            if any(index >= self.input_width for index in indices):
                raise ValueError(f"{name} entries must be smaller than input_width")
            object.__setattr__(self, name, indices)
        memory_fields = (
            self.demonstration_phase_index,
            self.query_phase_index,
            self.input_side_valid_index,
            self.output_side_valid_index,
        )
        if self.context_memory_width == 0:
            if any(index is not None for index in memory_fields) or any(
                (self.memory_key_indices, self.memory_value_indices)
            ):
                raise ValueError(
                    "context_memory_width must be positive when memory event "
                    "configuration is supplied"
                )
        else:
            required_names = (
                "demonstration_phase_index",
                "query_phase_index",
                "input_side_valid_index",
                "output_side_valid_index",
            )
            for name, index in zip(required_names, memory_fields, strict=True):
                if index is None:
                    raise ValueError(
                        f"{name} is required when context_memory_width is positive"
                    )
            if self.demonstration_phase_index == self.query_phase_index:
                raise ValueError(
                    "demonstration_phase_index and query_phase_index must differ"
                )
            if self.input_side_valid_index == self.output_side_valid_index:
                raise ValueError(
                    "input_side_valid_index and output_side_valid_index must differ"
                )
            if not self.memory_key_indices:
                raise ValueError(
                    "memory_key_indices is required when context_memory_width is positive"
                )
            if not self.memory_value_indices:
                raise ValueError(
                    "memory_value_indices is required when context_memory_width is positive"
                )
        if not isinstance(self.memory_coding, str):
            raise TypeError(
                "memory_coding must be 'frozen', 'learned_keys' or 'learned_write'"
            )
        if self.memory_coding not in MEMORY_CODINGS:
            raise ValueError(
                "memory_coding must be 'frozen', 'learned_keys' or 'learned_write'"
            )
        if self.memory_coding != "frozen" and self.context_memory_width == 0:
            raise ValueError(
                "memory_coding requires a positive context_memory_width"
            )
        if not isinstance(self.trace_engine, str):
            raise TypeError("trace_engine must be 'pp_prop' or 'd_rtrl'")
        if self.trace_engine not in TRACE_ENGINES:
            raise ValueError("trace_engine must be 'pp_prop' or 'd_rtrl'")
        if not isinstance(self.neuron_typing, str):
            raise TypeError("neuron_typing must be 'none' or 'ei_dale'")
        if self.neuron_typing not in NEURON_TYPINGS:
            raise ValueError("neuron_typing must be 'none' or 'ei_dale'")
        object.__setattr__(
            self,
            "excitatory_fraction",
            _unit_interval_real(self.excitatory_fraction, "excitatory_fraction"),
        )
        if self.neuron_typing == "none":
            if self.excitatory_fraction != 0.8:
                raise ValueError(
                    "excitatory_fraction requires neuron_typing='ei_dale'"
                )
        else:
            excitatory = round(self.excitatory_fraction * self.neuron_count)
            if not 1 <= excitatory <= self.neuron_count - 1:
                raise ValueError(
                    "excitatory_fraction must leave at least one neuron of "
                    "each type"
                )
        if self.sparse_backend is not None and not isinstance(self.sparse_backend, str):
            raise TypeError("sparse_backend must be a string or None")
        if not isinstance(self.decoder_mode, str):
            raise TypeError("decoder_mode must be 'legacy_cp' or 'row_refinement'")
        if self.decoder_mode not in ("legacy_cp", "row_refinement"):
            raise ValueError("decoder_mode must be 'legacy_cp' or 'row_refinement'")
        if self.decoder_mode == "legacy_cp":
            if self.refinement_layout is not None:
                raise ValueError(
                    "refinement_layout is valid only when decoder_mode is "
                    "'row_refinement'"
                )
        else:
            if not isinstance(self.refinement_layout, RowRefinementLayout):
                raise ValueError(
                    "refinement_layout is required when decoder_mode is "
                    "'row_refinement'"
                )
            if self.refinement_layout.input_width != self.input_width:
                raise ValueError("refinement_layout input_width must match input_width")
            if self.refinement_layout.event_valid_index != self.event_valid_index:
                raise ValueError(
                    "refinement_layout event_valid_index must match event_valid_index"
                )
            if self.refinement_steps % MAX_GRID_SIZE:
                raise ValueError("refinement_steps must be a multiple of 30")
            if self.refinement_steps > self.max_latent_steps:
                raise ValueError("refinement_steps must not exceed max_latent_steps")
            if self.memory_enabled:
                matched_indices = (
                    (
                        self.demonstration_phase_index,
                        self.refinement_layout.demonstration_phase_index,
                    ),
                    (
                        self.query_phase_index,
                        self.refinement_layout.query_phase_index,
                    ),
                    (
                        self.input_side_valid_index,
                        self.refinement_layout.input_side_valid_index,
                    ),
                    (
                        self.output_side_valid_index,
                        self.refinement_layout.output_side_valid_index,
                    ),
                )
                if any(left != right for left, right in matched_indices):
                    raise ValueError(
                        "refinement_layout phase and side indices must match "
                        "context-memory indices"
                    )

    @property
    def memory_enabled(self) -> bool:
        """Return whether the associative contextual-memory path is enabled."""

        return self.context_memory_width > 0

    @property
    def row_refinement_enabled(self) -> bool:
        """Return whether learned row-wise answer refinement is enabled."""

        return self.decoder_mode == "row_refinement"

    @property
    def refinement_sweeps(self) -> int:
        """Return the number of complete 30-row refinement sweeps."""

        return self.refinement_steps // MAX_GRID_SIZE

    @property
    def slot_count(self) -> int:
        """Return the number of exact 64-neuron analysis slots."""
        return self.neuron_count // NEURONS_PER_SLOT

    @property
    def compact_output_width(self) -> int:
        """Return the width of the factorized ARC output vector."""
        return compact_output_width(self.color_rank)

    @property
    def training_output_width(self) -> int:
        """Return per-tick BrainTrace training-output width."""

        return 360 if self.row_refinement_enabled else self.compact_output_width

    @property
    def checkpoint_output_width(self) -> int:
        """Return the width retained at inference checkpoints."""

        return 9060 if self.row_refinement_enabled else self.compact_output_width


@dataclass(frozen=True)
class AssociativeMemoryReport:
    """Describe the fixed associative-memory representation and operators.

    Parameters
    ----------
    mode : str
        ``"legacy_reservoir"`` or ``"associative_workspace"``.
    memory_width, key_feature_width, value_feature_width : int
        Logical memory and raw selected row-feature widths.
    key_map, value_map : str, optional
        Stable names for the frozen deterministic encoders.
    rff_gamma : float, optional
        Random Fourier frequency scale used by the key encoder.
    key_basis_seed, key_bias_seed, value_basis_seed : int, optional
        Dedicated :mod:`brainstate.random` stream seeds.
    key_basis_sha256, key_bias_sha256, value_basis_sha256 : str, optional
        Shape-, dtype-, and byte-sensitive fixed-array digests.
    write_component_type, query_component_type, read_component_type : str, optional
        Stable public names of the three pp-prop-visible trainable operations.

    Attributes
    ----------
    carrier_stabilizer : str, optional
        Stable identity of the memory-mode carrier stabilizer.
    carrier_radius : float, optional
        Fixed radius enforced at the dense carrier consumers.
    carrier_consumers : tuple of str, optional
        Stable names of the projections that consume stabilized carriers.
    """

    mode: str
    memory_width: int
    key_feature_width: int
    value_feature_width: int
    key_map: str | None
    value_map: str | None
    rff_gamma: float | None
    key_basis_seed: int | None
    key_bias_seed: int | None
    value_basis_seed: int | None
    key_basis_sha256: str | None
    key_bias_sha256: str | None
    value_basis_sha256: str | None
    write_component_type: str | None
    query_component_type: str | None
    read_component_type: str | None

    @property
    def carrier_stabilizer(self) -> str | None:
        """Return the fixed memory-mode carrier stabilizer identity.

        Returns
        -------
        str or None
            Stabilizer identity in memory mode; otherwise ``None``.
        """
        if self.mode == "associative_workspace":
            return "per_example_stopped_unit_l2_cap"
        return None

    @property
    def carrier_radius(self) -> float | None:
        """Return the fixed memory-mode carrier radius.

        Returns
        -------
        float or None
            Unit radius in memory mode; otherwise ``None``.
        """
        if self.mode == "associative_workspace":
            return 1.0
        return None

    @property
    def carrier_consumers(self) -> tuple[str, str] | None:
        """Return the projections that consume stabilized carriers.

        Returns
        -------
        tuple of str or None
            Stable projection names in memory mode; otherwise ``None``.
        """
        if self.mode == "associative_workspace":
            return ("readout_projection", "workspace_query_projection")
        return None

    def to_dict(self) -> dict[str, object]:
        """Return the serialized architecture with memory-only stabilization.

        Returns
        -------
        dict of str to object
            The legacy dataclass schema, extended with carrier stabilization
            metadata only in associative-workspace mode.
        """
        report = asdict(self)
        if self.mode == "associative_workspace":
            report.update(
                carrier_stabilizer=self.carrier_stabilizer,
                carrier_radius=self.carrier_radius,
                carrier_consumers=self.carrier_consumers,
            )
        return report


@dataclass(frozen=True)
class SparseTopology:
    """Hold one deterministic directed sparse topology.

    Parameters
    ----------
    rows, columns : numpy.ndarray
        Int32 edge endpoint arrays.  The recurrent projection computes
        ``y = spikes @ CSR`` (:func:`braintrace.sparse_matmul`), so ``rows``
        is contracted with the spike vector and is therefore the
        *presynaptic* endpoint; ``columns`` is the postsynaptic endpoint.
    values : numpy.ndarray
        Float32 initial edge values before physical current units are attached.
    neuron_count : int
        Square adjacency dimension.

    Examples
    --------
    .. code-block:: python

        >>> topology = build_sparse_topology(64, 128, seed=1)
        >>> (topology.edge_count, bool((topology.rows == topology.columns).any()))
        (128, False)
    """

    rows: NDArray[np.int32]
    columns: NDArray[np.int32]
    values: NDArray[np.float32]
    neuron_count: int

    @property
    def edge_count(self) -> int:
        """Return the exact number of stored directed edges."""
        return int(self.values.size)


@dataclass(frozen=True)
class ArcLogits:
    """Hold expanded ARC shape and cell-color logits.

    Parameters
    ----------
    height, width : jax.Array
        Logits whose final dimension has length 30 and indexes sizes 1--30.
    colors : jax.Array
        Color logits with trailing shape ``(30, 30, 10)``.
    """

    height: jax.Array
    width: jax.Array
    colors: jax.Array


@dataclass(frozen=True)
class ModelStateSnapshot:
    """Hold an exact copy of every non-parameter model state.

    Parameters
    ----------
    entries : tuple
        Pairs of BrainState graph paths and copied state pytrees.
    batch_size, neuron_count : int
        Shape identity used to reject restoration into another configuration.
    """

    entries: tuple[tuple[tuple[Any, ...], Any], ...]
    batch_size: int
    neuron_count: int


@dataclass(frozen=True)
class ContextCheckpoint:
    """Describe the query-terminal state before latent computation.

    Parameters
    ----------
    compact_logits : jax.Array
        Compact output shaped ``(batch, compact_output_width)``.
    spikes, voltage : jax.Array
        Query-terminal spikes and voltage shaped ``(batch, neurons)``.  Voltage
        is stored as the numeric value in millivolts.
    feedforward_current, recurrent_current : jax.Array
        Separate Expon synaptic-current states shaped ``(batch, neurons)`` and
        stored as numeric values in milliamps.
    snapshot : ModelStateSnapshot
        Restorable state at checkpoint zero.
    context_steps : int
        Number of valid ARC row events executed.
    """

    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    snapshot: ModelStateSnapshot
    context_steps: int


@dataclass(frozen=True)
class ModelTrajectory:
    """Hold checkpoint zero and a continuous zero-input latent trajectory.

    Parameters
    ----------
    compact_logits : jax.Array
        Factorized outputs shaped ``(steps + 1, batch, compact_width)``.
    spikes, voltage : jax.Array
        Physical trajectories shaped ``(steps + 1, batch, neurons)``.
    feedforward_current, recurrent_current : jax.Array
        Separate Expon current trajectories shaped
        ``(steps + 1, batch, neurons)`` in milliamps.
    zero_inputs : jax.Array
        Exactly-zero external inputs used for latent updates, shaped
        ``(steps, batch, input_width)``.
    color_rank : int
        Rank needed to expand compact color factors.
    decoder_mode : {"legacy_cp", "row_refinement"}, default="legacy_cp"
        Representation stored in ``compact_logits``.
    """

    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    zero_inputs: jax.Array
    color_rank: int
    decoder_mode: Literal["legacy_cp", "row_refinement"] = "legacy_cp"

    @property
    def latent_steps(self) -> int:
        """Return the number of recurrent updates after checkpoint zero."""
        return int(self.compact_logits.shape[0] - 1)

    @property
    def expanded(self) -> ArcLogits:
        """Expand every checkpoint to full ARC logits."""
        return expand_decoder_logits(
            self.compact_logits, self.color_rank, self.decoder_mode
        )

    def at_effort(self, effort: int) -> ArcLogits:
        """Return expanded logits at one latent-effort checkpoint.

        Parameters
        ----------
        effort : int
            Number of recurrent latent updates, from zero through
            :attr:`latent_steps`.

        Returns
        -------
        ArcLogits
            Batched logits at the selected checkpoint.
        """
        effort = _nonnegative_integer(effort, "effort")
        if effort > self.latent_steps:
            raise ValueError(
                f"effort {effort} exceeds trajectory length {self.latent_steps}"
            )
        return expand_decoder_logits(
            self.compact_logits[effort], self.color_rank, self.decoder_mode
        )


@dataclass(frozen=True)
class PackedTrajectory:
    """Hold outputs recorded after every tick of one fixed packed stream.

    Parameters
    ----------
    compact_logits : jax.Array
        Compact outputs shaped ``(time, batch, compact_width)``.
    spikes, voltage : jax.Array
        Physical states shaped ``(time, batch, neurons)``.
    feedforward_current, recurrent_current : jax.Array
        Separate Expon current states shaped ``(time, batch, neurons)`` and
        represented numerically in milliamps.
    workspace_carrier : jax.Array
        Continuous workspace values.  This aliases ``voltage`` in the current
        implementation and therefore adds no duplicate storage.
    memory_read : jax.Array
        Associative reads shaped ``(time, batch, memory_width)``.  Legacy mode
        has a zero-width final dimension.
    final_context_memory : jax.Array
        One final memory snapshot shaped ``(batch, memory_width, memory_width)``;
        legacy mode has two zero-width trailing dimensions.
    color_rank : int
        Rank needed to expand compact color factors.
    decoder_mode : {"legacy_cp", "row_refinement"}, default="legacy_cp"
        Representation stored in ``compact_logits``.
    """

    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    workspace_carrier: jax.Array
    memory_read: jax.Array
    final_context_memory: jax.Array
    color_rank: int
    decoder_mode: Literal["legacy_cp", "row_refinement"] = "legacy_cp"

    @property
    def expanded(self) -> ArcLogits:
        """Expand all packed ticks to full ARC logits."""
        return expand_decoder_logits(
            self.compact_logits, self.color_rank, self.decoder_mode
        )


@dataclass(frozen=True)
class SelectedPackedTrajectory:
    """Hold only requested per-example checkpoints from a packed stream.

    Parameters
    ----------
    selected_indices : jax.Array
        Strictly increasing stream indices shaped ``(checkpoints, batch)``.
    compact_logits : jax.Array
        Compact outputs shaped ``(checkpoints, batch, compact_width)``.
    spikes, voltage : jax.Array
        Selected physical states shaped ``(checkpoints, batch, neurons)``.
    feedforward_current, recurrent_current : jax.Array
        Selected Expon current states shaped
        ``(checkpoints, batch, neurons)`` in milliamps.
    workspace_carrier : jax.Array
        Selected continuous workspace values shaped
        ``(checkpoints, batch, neurons)``.  The current carrier is numeric LIF
        voltage, so this field aliases ``voltage`` instead of allocating a
        duplicate selected buffer.  Legacy mode reports the same voltage.
    memory_read : jax.Array
        Selected associative reads shaped ``(checkpoints, batch, memory_width)``.
        Legacy mode has a zero-width final dimension.
    final_context_memory : jax.Array
        One final memory snapshot shaped ``(batch, memory_width, memory_width)``.
        This preserves pairing-sensitive evidence without stacking ``S`` over
        time.  Legacy mode has two zero-width trailing dimensions.
    color_rank : int
        Rank needed to expand compact color factors.
    decoder_mode : {"legacy_cp", "row_refinement"}, default="legacy_cp"
        Representation stored in ``compact_logits``.
    """

    selected_indices: jax.Array
    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    workspace_carrier: jax.Array
    memory_read: jax.Array
    final_context_memory: jax.Array
    color_rank: int
    decoder_mode: Literal["legacy_cp", "row_refinement"] = "legacy_cp"

    @property
    def expanded(self) -> ArcLogits:
        """Expand selected compact checkpoints to full ARC logits."""
        return expand_decoder_logits(
            self.compact_logits, self.color_rank, self.decoder_mode
        )


@dataclass(frozen=True)
class SequenceResult:
    """Pair a context checkpoint with its continuous latent trajectory.

    Parameters
    ----------
    context : ContextCheckpoint
        Query-terminal checkpoint before recurrent latent updates.
    trajectory : ModelTrajectory
        Checkpoint zero followed by all zero-input updates.
    """

    context: ContextCheckpoint
    trajectory: ModelTrajectory

    @property
    def feedforward_current(self) -> jax.Array:
        """Return checkpoint-zero-through-latent feed-forward current."""
        return self.trajectory.feedforward_current

    @property
    def recurrent_current(self) -> jax.Array:
        """Return checkpoint-zero-through-latent recurrent current."""
        return self.trajectory.recurrent_current


class ArcLossComponents(NamedTuple):
    """Hold scalar total, height, width, and valid-cell color losses.

    Parameters
    ----------
    total, height, width, colors : jax.Array
        Scalar mean losses.  ``total`` contains the configured weighted sum.
    """

    total: jax.Array
    height: jax.Array
    width: jax.Array
    colors: jax.Array


def compact_output_width(color_rank: int) -> int:
    """Return compact output width for a CP color rank.

    Parameters
    ----------
    color_rank : int
        Number of color-tensor factors.

    Returns
    -------
    int
        Two 30-way shape heads plus rank times 30 rows, 30 columns, and 10
        colors.
    """
    color_rank = _positive_integer(color_rank, "color_rank")
    return 2 * MAX_GRID_SIZE + color_rank * (2 * MAX_GRID_SIZE + COLOR_COUNT)


def build_sparse_topology(
    neuron_count: int,
    edge_count: int,
    *,
    seed: int,
    recurrent_gain: float = 0.8,
) -> SparseTopology:
    """Build a deterministic exact-edge, no-self sparse topology.

    Each row receives either ``floor(E/N)`` or ``ceil(E/N)`` edges.  A random
    affine permutation of the ``N - 1`` legal offsets selects distinct
    presynaptic endpoints in that row.  Starts, multipliers, and weights are
    sampled with :class:`brainstate.random.RandomState`; NumPy is used only for
    static host-side indexing and validation.

    Parameters
    ----------
    neuron_count : int
        Number of recurrent neurons, at least two.
    edge_count : int
        Exact directed edge count, no larger than ``N * (N - 1)``.
    seed : int
        Nonnegative BrainState random seed.
    recurrent_gain : float, default=0.8
        Standard-deviation gain relative to square-root mean degree.

    Returns
    -------
    SparseTopology
        Sorted CSR-compatible endpoints and float32 values.
    """
    neuron_count = _positive_integer(neuron_count, "neuron_count")
    edge_count = _positive_integer(edge_count, "edge_count")
    seed = _nonnegative_integer(seed, "seed")
    recurrent_gain = _positive_real(recurrent_gain, "recurrent_gain")
    if neuron_count < 2:
        raise ValueError("neuron_count must be at least 2 for no-self topology")
    capacity = neuron_count * (neuron_count - 1)
    if edge_count > capacity:
        raise ValueError(f"edge_count {edge_count} exceeds no-self capacity {capacity}")

    random = brainstate.random.RandomState(seed)
    modulus = neuron_count - 1
    starts = np.asarray(
        random.randint(0, modulus, size=(neuron_count,), dtype=jnp.int32),
        dtype=np.int64,
    )
    if modulus == 1:
        multipliers = np.ones(neuron_count, dtype=np.int64)
    else:
        multipliers = np.asarray(
            random.randint(1, modulus, size=(neuron_count,), dtype=jnp.int32),
            dtype=np.int64,
        )
        # Project every draw onto the finite set of units modulo N-1.  This is
        # deterministic post-processing of BrainState randomness, not a second
        # random source.
        while_indices = np.flatnonzero(np.gcd(multipliers, modulus) != 1)
        while while_indices.size:
            multipliers[while_indices] = (
                multipliers[while_indices] % (modulus - 1)
            ) + 1
            while_indices = np.flatnonzero(np.gcd(multipliers, modulus) != 1)

    base_degree, remainder = divmod(edge_count, neuron_count)
    degrees = np.full(neuron_count, base_degree, dtype=np.int64)
    degrees[:remainder] += 1
    maximum_degree = int(degrees.max())
    positions = np.arange(maximum_degree, dtype=np.int64)[None, :]
    offsets = (starts[:, None] + multipliers[:, None] * positions) % modulus + 1
    rows_matrix = np.broadcast_to(
        np.arange(neuron_count, dtype=np.int64)[:, None], offsets.shape
    )
    valid = positions < degrees[:, None]
    rows = rows_matrix[valid]
    columns = (rows_matrix + offsets)[valid] % neuron_count
    order = np.lexsort((columns, rows))
    rows = rows[order].astype(np.int32, copy=False)
    columns = columns[order].astype(np.int32, copy=False)
    value_scale = recurrent_gain / math.sqrt(edge_count / neuron_count)
    values = np.asarray(random.randn(edge_count), dtype=np.float32) * value_scale
    values = values[order].astype(np.float32, copy=False)

    if rows.size != edge_count or np.any(rows == columns):
        raise RuntimeError("sparse topology construction violated its exact contract")
    flat = rows.astype(np.int64) * neuron_count + columns.astype(np.int64)
    if np.unique(flat).size != edge_count:
        raise RuntimeError("sparse topology construction produced duplicate edges")
    for array in (rows, columns, values):
        array.setflags(write=False)
    return SparseTopology(rows, columns, values, neuron_count)


def assign_neuron_type_signs(
    neuron_count: int,
    excitatory_fraction: float,
    *,
    seed: int,
) -> NDArray[np.int8]:
    """Assign deterministic binary E/I type signs to a neuron population.

    A seeded permutation drawn through :mod:`brainstate.random` marks
    ``round(excitatory_fraction * neuron_count)`` neurons as excitatory
    (``+1``) and the remainder as inhibitory (``-1``).  The stream uses a
    dedicated seed offset so enabling typing never perturbs the existing
    topology, parameter, or memory random streams.

    Parameters
    ----------
    neuron_count : int
        Number of neurons to type, at least two.
    excitatory_fraction : float
        Fraction of excitatory neurons in ``[0, 1]``.  The realized split
        must leave at least one neuron of each type.
    seed : int
        Nonnegative base model seed; the stream uses ``seed + 7``.

    Returns
    -------
    numpy.ndarray
        Read-only int8 vector of ``+1`` (excitatory) and ``-1`` (inhibitory)
        entries, one per neuron.

    Examples
    --------
    .. code-block:: python

        >>> signs = assign_neuron_type_signs(64, 0.75, seed=2108)
        >>> (int((signs == 1).sum()), int((signs == -1).sum()))
        (48, 16)
    """
    neuron_count = _positive_integer(neuron_count, "neuron_count")
    excitatory_fraction = _unit_interval_real(
        excitatory_fraction, "excitatory_fraction"
    )
    seed = _nonnegative_integer(seed, "seed")
    excitatory = round(excitatory_fraction * neuron_count)
    if not 1 <= excitatory <= neuron_count - 1:
        raise ValueError(
            "excitatory_fraction must leave at least one neuron of each type"
        )
    random = brainstate.random.RandomState(seed + _NEURON_TYPE_SEED_OFFSET)
    permutation = np.asarray(random.permutation(neuron_count), dtype=np.int64)
    signs = np.full(neuron_count, -1, dtype=np.int8)
    signs[permutation[:excitatory]] = 1
    signs.setflags(write=False)
    return signs


def apply_dale_signs(
    topology: SparseTopology,
    type_signs: NDArray[np.int8],
) -> SparseTopology:
    """Constrain a topology's edge signs to their presynaptic neuron types.

    Every edge value becomes ``type_signs[presynaptic] * |value|``, where the
    presynaptic endpoint is ``topology.rows`` (the axis contracted with the
    spike vector in ``y = spikes @ CSR``).  Magnitudes, endpoints, and edge
    order are preserved exactly.

    Parameters
    ----------
    topology : SparseTopology
        Untyped source topology.
    type_signs : numpy.ndarray
        Per-neuron ``+1``/``-1`` vector from
        :func:`assign_neuron_type_signs`.

    Returns
    -------
    SparseTopology
        New topology whose values obey Dale's law.

    Examples
    --------
    .. code-block:: python

        >>> topology = build_sparse_topology(64, 128, seed=1)
        >>> signs = assign_neuron_type_signs(64, 0.8, seed=1)
        >>> typed = apply_dale_signs(topology, signs)
        >>> bool((np.sign(typed.values) == signs[typed.rows]).all())
        True
    """
    signs = np.asarray(type_signs)
    if signs.shape != (topology.neuron_count,):
        raise ValueError(
            "type_signs length must equal the topology neuron count"
        )
    if not np.all(np.abs(signs.astype(np.int64)) == 1):
        raise ValueError("type_signs entries must be +1 or -1")
    values = (
        np.abs(topology.values) * signs[topology.rows].astype(np.float32)
    ).astype(np.float32, copy=False)
    values.setflags(write=False)
    return SparseTopology(
        topology.rows, topology.columns, values, topology.neuron_count
    )


def project_dale_weights(weights: Any, edge_signs: Any) -> Any:
    """Project edge weights onto their Dale-legal half-line.

    Positive-typed edges are clamped to ``max(w, 0)`` and negative-typed
    edges to ``min(w, 0)``; legal weights pass through unchanged, so the
    projection is idempotent.  Works on plain arrays and on
    :class:`brainunit.Quantity` values alike.

    Parameters
    ----------
    weights : ArrayLike or brainunit.Quantity
        Per-edge weight vector.
    edge_signs : ArrayLike
        Per-edge ``+1``/``-1`` presynaptic type signs.

    Returns
    -------
    ArrayLike or brainunit.Quantity
        Projected weights with the input's dtype and unit.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> weights = jnp.asarray([1.0, -2.0]) * u.mA
        >>> project_dale_weights(weights, jnp.asarray([1, 1]))
        ArrayImpl([1., 0.], dtype=float32) * mamp
    """
    zero = u.math.zeros_like(weights)
    return u.math.where(
        jnp.asarray(edge_signs) > 0,
        u.math.maximum(weights, zero),
        u.math.minimum(weights, zero),
    )


def _indptr_from_rows(rows: NDArray[np.int32], neuron_count: int) -> NDArray[np.int32]:
    indptr = np.zeros(neuron_count + 1, dtype=np.int64)
    np.add.at(indptr, rows.astype(np.int64) + 1, 1)
    return np.cumsum(indptr).astype(np.int32)


def _topology_to_csr(topology: SparseTopology, backend: str | None) -> Any:
    import brainevent

    return brainevent.CSR(
        jnp.asarray(topology.values),
        jnp.asarray(topology.columns),
        jnp.asarray(_indptr_from_rows(topology.rows, topology.neuron_count)),
        shape=(topology.neuron_count, topology.neuron_count),
        backend=backend,
    )


def _split_compact_logits(
    compact: jax.Array, color_rank: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    expected = compact_output_width(color_rank)
    if compact.ndim < 1 or compact.shape[-1] != expected:
        raise ValueError(
            f"compact logits must have final width {expected}, got {compact.shape}"
        )
    cursor = 0
    height = compact[..., cursor : cursor + MAX_GRID_SIZE]
    cursor += MAX_GRID_SIZE
    width = compact[..., cursor : cursor + MAX_GRID_SIZE]
    cursor += MAX_GRID_SIZE
    row_count = color_rank * MAX_GRID_SIZE
    row = compact[..., cursor : cursor + row_count].reshape(
        *compact.shape[:-1], color_rank, MAX_GRID_SIZE
    )
    cursor += row_count
    column = compact[..., cursor : cursor + row_count].reshape(
        *compact.shape[:-1], color_rank, MAX_GRID_SIZE
    )
    cursor += row_count
    color = compact[..., cursor:].reshape(*compact.shape[:-1], color_rank, COLOR_COUNT)
    return height, width, row, column, color


def expand_compact_logits(compact: jax.Array, color_rank: int) -> ArcLogits:
    """Expand factorized outputs to full ARC logits.

    Parameters
    ----------
    compact : jax.Array
        Array with any leading dimensions and final width given by
        :func:`compact_output_width`.
    color_rank : int
        CP rank encoded by ``compact``.

    Returns
    -------
    ArcLogits
        Height and width logits plus a color array with trailing shape
        ``(30, 30, 10)``.
    """
    color_rank = _positive_integer(color_rank, "color_rank")
    compact = jnp.asarray(compact)
    height, width, row, column, color = _split_compact_logits(compact, color_rank)
    color_logits = jnp.einsum(
        "...ri,...rj,...rc->...ijc", row, column, color
    ) / math.sqrt(color_rank)
    return ArcLogits(height=height, width=width, colors=color_logits)


def expand_decoder_logits(
    logits: jax.Array,
    color_rank: int,
    decoder_mode: Literal["legacy_cp", "row_refinement"],
) -> ArcLogits:
    """Expand checkpoint logits from either supported decoder.

    Parameters
    ----------
    logits : jax.Array
        Decoder output with arbitrary leading dimensions.
    color_rank : int
        CP rank used by the legacy decoder.
    decoder_mode : {"legacy_cp", "row_refinement"}
        Static decoder representation encoded by ``logits``.

    Returns
    -------
    ArcLogits
        Explicit height, width, and color logits.
    """

    if decoder_mode == "legacy_cp":
        return expand_compact_logits(logits, color_rank)
    if decoder_mode != "row_refinement":
        raise ValueError("decoder_mode must be 'legacy_cp' or 'row_refinement'")
    height, width, colors = split_refinement_output_logits(logits)
    return ArcLogits(height=height, width=width, colors=colors)


def arc_loss_components(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float = 1.0,
    color_weight: float = 1.0,
    class_balanced_colors: bool = False,
) -> ArcLossComponents:
    """Compute terminal ARC height, width, and valid-cell cross entropy.

    Parameters
    ----------
    compact_logits : jax.Array
        Batched compact model output shaped ``(batch, compact_width)``.
    target_height, target_width : jax.Array
        Integer sizes in 1--30, shaped ``(batch,)``.
    target_colors : jax.Array
        Integer padded color grid shaped ``(batch, 30, 30)``.  Only cells
        inside each target shape contribute.
    color_rank : int
        CP rank of ``compact_logits``.
    shape_weight, color_weight : float, default=1.0
        Nonnegative component weights.
    class_balanced_colors : bool, default=False
        If true, each color present inside a target's valid shape contributes
        equal total color-loss weight.  Otherwise, every valid cell receives
        equal weight.

    Returns
    -------
    ArcLossComponents
        Scalar mean losses suitable for a terminal pp-prop step loss.
    """
    total, height_loss, width_loss, color_loss = _arc_loss_vectors(
        compact_logits,
        target_height,
        target_width,
        target_colors,
        color_rank=color_rank,
        shape_weight=shape_weight,
        color_weight=color_weight,
        class_balanced_colors=class_balanced_colors,
    )
    return ArcLossComponents(
        total.mean(), height_loss.mean(), width_loss.mean(), color_loss.mean()
    )


def arc_loss_per_example(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float = 1.0,
    color_weight: float = 1.0,
    class_balanced_colors: bool = False,
) -> jax.Array:
    """Return one terminal ARC loss for each batch element.

    This form lets a packed pp-prop stream multiply losses by a per-time,
    per-example terminal gate before reducing.  Each example's cell loss is
    normalized within its own target.  By default, normalization is by valid
    target area so large grids do not receive more weight merely because they
    contain more cells.  Optional class balancing instead gives every color
    present in a target equal total weight.

    Parameters
    ----------
    compact_logits : jax.Array
        Batched compact model output shaped ``(batch, compact_width)``.
    target_height, target_width : jax.Array
        Integer sizes in 1--30, shaped ``(batch,)``.
    target_colors : jax.Array
        Integer padded colors shaped ``(batch, 30, 30)``.
    color_rank : int
        CP rank encoded by ``compact_logits``.
    shape_weight, color_weight : float, default=1.0
        Nonnegative component weights.
    class_balanced_colors : bool, default=False
        If true, average the valid-cell losses within each present color, then
        average those color losses.  If false, average all valid cells.

    Returns
    -------
    jax.Array
        Loss vector shaped ``(batch,)``.
    """
    total, _, _, _ = _arc_loss_vectors(
        compact_logits,
        target_height,
        target_width,
        target_colors,
        color_rank=color_rank,
        shape_weight=shape_weight,
        color_weight=color_weight,
        class_balanced_colors=class_balanced_colors,
    )
    return total


def _arc_loss_vectors(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float,
    color_weight: float,
    class_balanced_colors: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    compact_logits = jnp.asarray(compact_logits)
    if compact_logits.ndim != 2:
        raise ValueError(
            "compact_logits must have shape (batch, compact_width), got "
            f"{compact_logits.shape}"
        )
    batch_size = compact_logits.shape[0]
    target_height = jnp.asarray(target_height)
    target_width = jnp.asarray(target_width)
    target_colors = jnp.asarray(target_colors)
    if target_height.shape != (batch_size,) or target_width.shape != (batch_size,):
        raise ValueError("target_height and target_width must each have shape (batch,)")
    if target_colors.shape != (batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE):
        raise ValueError(
            "target_colors must have shape "
            f"({batch_size}, {MAX_GRID_SIZE}, {MAX_GRID_SIZE})"
        )
    shape_weight = float(shape_weight)
    color_weight = float(color_weight)
    if not math.isfinite(shape_weight) or shape_weight < 0.0:
        raise ValueError("shape_weight must be finite and nonnegative")
    if not math.isfinite(color_weight) or color_weight < 0.0:
        raise ValueError("color_weight must be finite and nonnegative")

    logits = expand_compact_logits(compact_logits, color_rank)
    height_indices = target_height.astype(jnp.int32) - 1
    width_indices = target_width.astype(jnp.int32) - 1
    height_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.height, axis=-1),
        height_indices[:, None],
        axis=-1,
    )[:, 0]
    width_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.width, axis=-1),
        width_indices[:, None],
        axis=-1,
    )[:, 0]
    color_nll = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.colors, axis=-1),
        target_colors.astype(jnp.int32)[..., None],
        axis=-1,
    )[..., 0]
    rows = jnp.arange(MAX_GRID_SIZE)[None, :, None]
    columns = jnp.arange(MAX_GRID_SIZE)[None, None, :]
    valid = (rows < target_height[:, None, None]) & (
        columns < target_width[:, None, None]
    )
    if class_balanced_colors:
        valid_colors = (
            jax.nn.one_hot(
                target_colors.astype(jnp.int32), COLOR_COUNT, dtype=color_nll.dtype
            )
            * valid[..., None]
        )
        class_counts = jnp.sum(valid_colors, axis=(1, 2))
        target_class_counts = jnp.sum(
            valid_colors * class_counts[:, None, None, :], axis=-1
        )
        present_class_count = jnp.sum(class_counts > 0, axis=-1)
        cell_weights = jnp.where(
            valid,
            1.0
            / jnp.maximum(target_class_counts, 1.0)
            / jnp.maximum(present_class_count[:, None, None], 1),
            0.0,
        )
        color_loss = jnp.sum(color_nll * cell_weights, axis=(1, 2))
    else:
        color_loss = jnp.sum(
            jnp.where(valid, color_nll, 0.0), axis=(1, 2)
        ) / jnp.maximum(jnp.sum(valid, axis=(1, 2)), 1)
    total = shape_weight * (height_loss + width_loss) + color_weight * color_loss
    return total, height_loss, width_loss, color_loss


def terminal_arc_loss(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float = 1.0,
    color_weight: float = 1.0,
    class_balanced_colors: bool = False,
) -> jax.Array:
    """Return scalar terminal ARC loss for pp-prop supervision.

    Parameters
    ----------
    compact_logits : jax.Array
        Batched compact model output shaped ``(batch, compact_width)``.
    target_height, target_width : jax.Array
        Integer sizes in 1--30, shaped ``(batch,)``.
    target_colors : jax.Array
        Integer padded colors shaped ``(batch, 30, 30)``.
    color_rank : int
        CP rank encoded by ``compact_logits``.
    shape_weight, color_weight : float, default=1.0
        Nonnegative component weights.
    class_balanced_colors : bool, default=False
        If true, each color present inside a target's valid shape contributes
        equal total color-loss weight.  If false, valid cells are weighted
        uniformly.

    Returns
    -------
    jax.Array
        Scalar weighted cross entropy.
    """
    return arc_loss_components(
        compact_logits,
        target_height,
        target_width,
        target_colors,
        color_rank=color_rank,
        shape_weight=shape_weight,
        color_weight=color_weight,
        class_balanced_colors=class_balanced_colors,
    ).total


def _copy_tree(value: Any) -> Any:
    return jax.tree.map(lambda leaf: jnp.array(leaf, copy=True), value)


def _array_sha256(value: jax.Array) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def update_context_memory(
    memory: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    write_gate: jax.Array,
    decay: float,
    write_scale: jax.Array | None = None,
) -> jax.Array:
    """Apply one gated diagonal-friendly associative-memory update.

    Parameters
    ----------
    memory : jax.Array
        Current memory shaped ``(batch, key_width, value_width)``.
    key, value : jax.Array
        Per-example write vectors shaped ``(batch, key_width)`` and
        ``(batch, value_width)``.
    write_gate : jax.Array
        Boolean gate shaped ``(batch,)``.  False lanes remain byte-identical.
    decay : float
        Finite self-decay in ``[0, 1]`` for lanes whose gate is true.
    write_scale : jax.Array, optional
        Element-wise scale shaped ``(key_width, value_width)``.  It is applied
        to the literal outer-product write, not to the recurrent memory term.

    Returns
    -------
    jax.Array
        Updated memory with the same shape and dtype as ``memory``.
    """
    memory = jnp.asarray(memory)
    key = jnp.asarray(key)
    value = jnp.asarray(value)
    write_gate = jnp.asarray(write_gate, dtype=jnp.bool_)
    if memory.ndim != 3:
        raise ValueError("memory must have shape (batch, key_width, value_width)")
    batch_size, key_width, value_width = memory.shape
    if key.shape != (batch_size, key_width):
        raise ValueError(
            f"key must have shape ({batch_size}, {key_width}), got {key.shape}"
        )
    if value.shape != (batch_size, value_width):
        raise ValueError(
            f"value must have shape ({batch_size}, {value_width}), got {value.shape}"
        )
    if write_gate.shape != (batch_size,):
        raise ValueError(
            f"write_gate must have shape ({batch_size},), got {write_gate.shape}"
        )
    decay = _unit_interval_real(decay, "decay")
    write = jnp.einsum("bi,bj->bij", key, value)
    if write_scale is not None:
        write_scale = jnp.asarray(write_scale)
        if write_scale.shape != (key_width, value_width):
            raise ValueError(
                "write_scale must have shape "
                f"({key_width}, {value_width}), got {write_scale.shape}"
            )
        write = write * write_scale[None, :, :]
    return apply_context_memory_write(
        memory, write, write_gate=write_gate, decay=decay
    )


def apply_context_memory_write(
    memory: jax.Array,
    write: jax.Array,
    *,
    write_gate: jax.Array,
    decay: float,
) -> jax.Array:
    """Commit an already-scaled write matrix into the gated decaying memory.

    Separated from :func:`update_context_memory` so that a write produced by the
    fused ``braintrace.outer_write`` primitive -- which forms the outer product
    *inside* an ETP primitive rather than here -- commits through exactly the
    same recurrence, leaving one place where the gate and the decay are decided.

    Parameters
    ----------
    memory : jax.Array
        Current memory shaped ``(batch, key_width, value_width)``.
    write : jax.Array
        Write matrix shaped like ``memory``, already carrying any write scale
        and side-validity masking.
    write_gate : jax.Array
        Boolean gate shaped ``(batch,)``.  False lanes remain byte-identical.
    decay : float
        Finite self-decay in ``[0, 1]`` for lanes whose gate is true.

    Returns
    -------
    jax.Array
        Updated memory with the same shape and dtype as ``memory``.
    """
    decay = _unit_interval_real(decay, "decay")
    candidate = decay * memory + write
    return jnp.where(write_gate[:, None, None], candidate, memory)


def softcap(value: jax.Array, beta: float) -> jax.Array:
    """Smoothly cap ``value`` to ``(-beta, beta)`` with unit slope at zero.

    Computes ``beta * tanh(value / beta)`` (the softcap of SiTU-GLU Eq. 12,
    used here without the GLU structure). ``beta = 1.0`` reproduces
    ``tanh(value)`` bit-exactly because division and multiplication by one
    are exact float operations, so the legacy hard caps stay recoverable.

    Parameters
    ----------
    value : jax.Array
        Pre-activation to cap.
    beta : float
        Positive finite cap magnitude.

    Returns
    -------
    jax.Array
        The capped activation, bounded in magnitude by ``beta`` (the
        mathematical bound is the open interval ``(-beta, beta)``; float
        rounding can reach the endpoints for very large inputs).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from examples.pp_prop.latent_workspace_model import softcap
        >>> softcap(jnp.asarray(0.5), 1.0) == jnp.tanh(jnp.asarray(0.5))
        Array(True, dtype=bool)
        >>>
        >>> float(softcap(jnp.asarray(6.0), 4.0)) < 4.0
        True
    """
    return beta * jnp.tanh(value / beta)


def _unit_l2_cap(value: jax.Array) -> jax.Array:
    """Cap each final-axis vector at unit L2 norm with a stopped divisor."""
    value = jnp.asarray(value)
    accumulator_dtype = jnp.promote_types(value.dtype, jnp.float32)
    accumulator = value.astype(accumulator_dtype)
    norm = jnp.linalg.vector_norm(
        jax.lax.stop_gradient(accumulator), axis=-1, keepdims=True
    )
    denominator = jax.lax.stop_gradient(
        jnp.maximum(jnp.asarray(1.0, dtype=norm.dtype), norm)
    )
    return (accumulator / denominator).astype(value.dtype)


def _unit_rms_carrier(value: jax.Array) -> jax.Array:
    """Cap each final-axis vector at unit root-mean-square with a stopped divisor.

    ``_unit_l2_cap`` normalises to unit *total* L2 across ``n`` coordinates, so
    every coordinate carries ``1/sqrt(n)``.  Feeding that to a bias-free head
    initialised at ``1/sqrt(n)`` yields initialisation logits of standard
    deviation ``1/sqrt(n)`` -- 0.031 at 1024 neurons -- and a softmax
    indistinguishable from uniform.  Capping the root-mean-square instead keeps
    each coordinate at O(1) and the resulting logits at O(1).
    """
    value = jnp.asarray(value)
    accumulator_dtype = jnp.promote_types(value.dtype, jnp.float32)
    accumulator = value.astype(accumulator_dtype)
    mean_square = jnp.mean(
        jax.lax.stop_gradient(accumulator) ** 2, axis=-1, keepdims=True
    )
    root_mean_square = jnp.sqrt(mean_square)
    denominator = jax.lax.stop_gradient(
        jnp.maximum(jnp.asarray(1.0, dtype=root_mean_square.dtype), root_mean_square)
    )
    return (accumulator / denominator).astype(value.dtype)


def _refinement_head_input(
    carrier: jax.Array, event: jax.Array, layout: RowRefinementLayout
) -> jax.Array:
    """Concatenate the workspace carrier with the row being written.

    The answer heads are bias-free linear maps, so anything absent from this
    vector is unreachable by any amount of training.  The membrane carrier
    identifies the task but not the refinement row: across one 30-row sweep its
    row-to-row cosine is 0.99 and a linear probe recovers the row index at
    0.058 against 0.033 chance, while recovering query identity at 1.000.  The
    row-position one-hot and the query colours of the row being transcribed are
    both already present in the refinement feedback event, but reach the heads
    only after the membrane has integrated them away.

    The query's own grid dimensions are appended for the same reason. The shape
    head is supervised on one tick of a 30-row sweep, the completed sweep at row
    29, and the colour block is written gated by ``input_row_valid``, which is
    exactly zero beyond the input height -- so at that tick the colour block is
    identically zero for every query shorter than 30 rows and the row one-hot is
    the same for every query. Without the dimension one-hots the carrier is the
    only query-varying signal the shape head sees where it is supervised, and
    ``output_shape == input_shape`` -- true for 66.1% of ARC-AGI-1 evaluation
    queries -- is not representable. With them it is a 30x30 identity block.

    Every appended block is scaled to unit root-mean-square on its occupied
    coordinates -- exactly ``sqrt(30)`` for a 30-way one-hot and ``sqrt(10)``
    for a per-column colour one-hot -- so no block dominates the head input by
    an accident of encoding sparsity.
    """
    carrier = jnp.asarray(carrier, dtype=jnp.float32)
    event = jnp.asarray(event, dtype=jnp.float32)
    row_scale = math.sqrt(MAX_GRID_SIZE)
    row_position = event[:, layout.row_index_slice] * row_scale
    row_colors = event[:, layout.input_color_slice] * math.sqrt(COLOR_COUNT)
    input_height = event[:, layout.input_height_slice] * row_scale
    input_width = event[:, layout.input_width_slice] * row_scale
    return jnp.concatenate(
        (carrier, row_position, row_colors, input_height, input_width), axis=-1
    )


def _carrier_scaled_head_input(
    unit_carrier: jax.Array,
    event: jax.Array,
    layout: RowRefinementLayout,
    scale: float,
    unscaled: jax.Array,
) -> jax.Array:
    """Return the head input with the carrier block multiplied by ``scale``.

    ``scale == 1.0`` returns ``unscaled`` untouched so the default path stays
    bit-exact; any other value rebuilds the concatenation from the scaled
    carrier, leaving the event-derived blocks untouched.
    """
    if scale == 1.0:
        return unscaled
    return _refinement_head_input(unit_carrier * scale, event, layout)


def _copy_residual_logits(
    row_logits: jax.Array,
    event: jax.Array,
    layout: RowRefinementLayout,
    gain: float,
) -> jax.Array:
    """Add a fixed identity residual from the query's colours to the row logits.

    The colour block of the refinement event is indexed
    ``column * COLOR_COUNT + colour`` and the row logits reshape
    ``(MAX_GRID_SIZE, COLOR_COUNT)`` on the same index, so adding the raw
    one-hot block scaled by ``gain`` raises the logit of exactly the query's
    colour at each occupied column.  Coordinates the query does not occupy --
    including every column at or beyond the input's width and, through
    ``input_row_valid`` gating, every row beyond the input's height -- receive
    exactly zero, so the learned head keeps sole custody of rule deviations
    and out-of-range cells.

    This fixed additive copy bias is retained as an ablation. It is not the
    learned source-axis softmax defined by Attention Residuals.
    """
    if gain == 0.0:
        return row_logits
    residual = event[:, layout.input_color_slice].astype(row_logits.dtype)
    return row_logits + jnp.asarray(gain, dtype=row_logits.dtype) * residual


def _rms_balanced_identity_source(
    identity: jax.Array, proposal: jax.Array
) -> jax.Array:
    """Scale an identity source to the proposal RMS with a stopped multiplier."""
    identity = jnp.asarray(identity, dtype=proposal.dtype)
    accumulator_dtype = jnp.promote_types(proposal.dtype, jnp.float32)
    identity_rms = jnp.sqrt(
        jnp.mean(jnp.square(identity.astype(accumulator_dtype)), axis=-1, keepdims=True)
    )
    proposal_rms = jnp.sqrt(
        jnp.mean(
            jnp.square(jax.lax.stop_gradient(proposal).astype(accumulator_dtype)),
            axis=-1,
            keepdims=True,
        )
    )
    epsilon = jnp.asarray(jnp.finfo(accumulator_dtype).eps, dtype=accumulator_dtype)
    multiplier = jnp.where(
        identity_rms > epsilon,
        proposal_rms / jnp.maximum(identity_rms, epsilon),
        jnp.asarray(1.0, dtype=accumulator_dtype),
    )
    return identity * jax.lax.stop_gradient(multiplier).astype(identity.dtype)


def refinement_parameter_paths(config: Any) -> tuple[str, ...]:
    """Return architecture-aware required refinement parameter paths.

    Parameters
    ----------
    config : object
        Architecture descriptor exposing ``decoder_mode`` and
        ``refinement_mixer``. Both ``ModelConfig`` and the Example 21
        experiment configuration satisfy this contract.

    Returns
    -------
    tuple of str
        Sorted parameter paths. Legacy CP has no refinement paths.
    """
    if config.decoder_mode != "row_refinement":
        return ()
    if config.refinement_mixer == "carrier_gate":
        paths = (
            "answer_row_event_head.weight",
            "answer_row_carrier_head.weight",
            "row_carrier_gate_head.weight",
            "answer_shape_head.weight",
        )
    elif config.refinement_mixer == "attention_residual":
        paths = (
            "answer_row_proposal_head.weight",
            "answer_shape_proposal_head.weight",
            "row_attention_residual.query",
            "shape_attention_residual.query",
        )
    else:
        paths = ("answer_row_head.weight", "answer_shape_head.weight")
    return tuple(sorted(paths))


def refinement_head_width(neuron_count: int) -> int:
    """Return the answer-head input width for a row-conditioned decoder.

    Parameters
    ----------
    neuron_count : int
        Physical LIF population size supplying the workspace carrier.

    Returns
    -------
    int
        ``neuron_count`` plus the row-position one-hot, the row colour block,
        and the query input height and width one-hots.

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_model import refinement_head_width
        >>> refinement_head_width(1024)
        1414
    """
    return (
        int(neuron_count)
        + MAX_GRID_SIZE
        + MAX_GRID_SIZE * COLOR_COUNT
        + 2 * MAX_GRID_SIZE
    )


class LatentWorkspaceModel(brainstate.nn.Module):
    """BrainPy LIF network with sparse recurrent ARC computation.

    Parameters
    ----------
    config : ModelConfig
        Physical, sparse, readout, and batching configuration.
    memory_read_policy : {"full", "query_only"}, default="full"
        Static constructor policy for contextual-memory reads.  ``full``
        re-reads memory on query and latent ticks.  ``query_only`` preserves
        the ordinary query read but supplies an exactly zero memory read and
        drive on latent ticks.  Width-zero legacy mode is unaffected.

    Notes
    -----
    ``cell_step`` advances physical state and returns spikes.  ``update`` adds
    the compact ARC head and is therefore the callable compiled by BrainTrace.
    Inference context loops call ``cell_step`` and run the head only at the
    query-terminal checkpoint.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        memory_read_policy: Literal["full", "query_only"] = "full",
    ):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        if not isinstance(memory_read_policy, str):
            raise TypeError("memory_read_policy must be 'full' or 'query_only'")
        if memory_read_policy not in ("full", "query_only"):
            raise ValueError("memory_read_policy must be 'full' or 'query_only'")
        self.config = config
        self._memory_read_policy = memory_read_policy
        self.topology = build_sparse_topology(
            config.neuron_count,
            config.recurrent_edges,
            seed=config.seed,
            recurrent_gain=config.recurrent_gain,
        )
        if config.neuron_typing == "ei_dale":
            type_signs = assign_neuron_type_signs(
                config.neuron_count,
                config.excitatory_fraction,
                seed=config.seed,
            )
            edge_signs = type_signs[self.topology.rows]
            self._dale_init_flip_count = int(
                np.sum(self.topology.values * edge_signs < 0)
            )
            self.topology = apply_dale_signs(self.topology, type_signs)
            self.neuron_type_signs = type_signs
            self._dale_edge_signs = jnp.asarray(edge_signs, dtype=jnp.int8)
        else:
            self.neuron_type_signs = None
            self._dale_edge_signs = None
            self._dale_init_flip_count = 0
        random = brainstate.random.RandomState(config.seed + 1)

        self.neu = bpstate.LIF(
            config.neuron_count,
            R=1.0 * u.ohm,
            tau=config.membrane_tau_ms * u.ms,
            V_th=1.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_rest=0.0 * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        input_weights = random.randn(config.input_width, config.neuron_count)
        input_weights = (
            input_weights * (config.input_gain / math.sqrt(config.input_width)) * u.mA
        )
        recurrent_linear = braintrace.nn.SparseLinear(
            _topology_to_csr(self.topology, config.sparse_backend), b_init=None
        )
        recurrent_parameters = dict(recurrent_linear.weight.value)
        recurrent_parameters["weight"] = recurrent_parameters["weight"] * u.mA
        recurrent_linear.weight.value = recurrent_parameters

        self.ff_syn = bpstate.AlignPostProj(
            comm=braintrace.nn.Linear(
                config.input_width,
                config.neuron_count,
                w_init=input_weights,
                b_init=None,
            ),
            syn=bpstate.Expon(
                config.neuron_count,
                tau=config.feedforward_tau_ms * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=bpstate.CUBA(scale=1.0),
            post=self.neu,
        )
        self.rec_syn = bpstate.AlignPostProj(
            comm=recurrent_linear,
            syn=bpstate.Expon(
                config.neuron_count,
                tau=config.recurrent_tau_ms * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=bpstate.CUBA(scale=1.0),
            post=self.neu,
        )

        bottleneck_weights = random.randn(config.neuron_count, config.readout_width)
        bottleneck_weights = bottleneck_weights / math.sqrt(config.neuron_count)
        shape_weights = random.randn(config.readout_width, MAX_GRID_SIZE)
        shape_weights = shape_weights / math.sqrt(config.readout_width)
        factor_width = config.color_rank * (2 * MAX_GRID_SIZE + COLOR_COUNT)
        factor_weights = random.randn(config.readout_width, factor_width)
        factor_weights = factor_weights / math.sqrt(config.readout_width)
        self.readout_projection = braintrace.nn.Linear(
            config.neuron_count,
            config.readout_width,
            w_init=bottleneck_weights,
            b_init=braintools.init.ZeroInit(),
        )
        self.height_head = braintrace.nn.Linear(
            config.readout_width,
            MAX_GRID_SIZE,
            w_init=shape_weights,
            b_init=braintools.init.ZeroInit(),
        )
        self.width_head = braintrace.nn.Linear(
            config.readout_width,
            MAX_GRID_SIZE,
            w_init=random.randn(config.readout_width, MAX_GRID_SIZE)
            / math.sqrt(config.readout_width),
            b_init=braintools.init.ZeroInit(),
        )
        self.color_factor_head = braintrace.nn.Linear(
            config.readout_width,
            factor_width,
            w_init=factor_weights,
            b_init=braintools.init.ZeroInit(),
        )
        if config.memory_enabled:
            memory_width = config.context_memory_width
            key_width = len(config.memory_key_indices)
            value_width = len(config.memory_value_indices)
            key_random = brainstate.random.RandomState(config.seed + 101)
            key_bias_random = brainstate.random.RandomState(config.seed + 102)
            value_random = brainstate.random.RandomState(config.seed + 103)
            query_random = brainstate.random.RandomState(config.seed + 104)
            read_random = brainstate.random.RandomState(config.seed + 105)
            self._memory_key_basis = jnp.asarray(
                key_random.randn(key_width, memory_width), dtype=jnp.float32
            )
            self._memory_key_bias = jnp.asarray(
                2.0 * math.pi * key_bias_random.rand(memory_width),
                dtype=jnp.float32,
            )
            self._memory_value_basis = jnp.asarray(
                value_random.randn(value_width, memory_width) / math.sqrt(value_width),
                dtype=jnp.float32,
            )
            self.memory_write_scale = brainstate.ParamState(
                jnp.ones((memory_width, memory_width), dtype=jnp.float32)
            )
            if config.memory_coding in LEARNED_RETRIEVAL_KEY_CODINGS:
                self.memory_key_projection = braintrace.nn.Linear(
                    key_width,
                    memory_width,
                    w_init=MEMORY_KEY_RFF_GAMMA * self._memory_key_basis,
                    b_init=self._memory_key_bias,
                )
            if config.memory_coding == "learned_write":
                # Write-side projections owned by the fused `outer_write`
                # primitive. They are *separate* parameters from the retrieval
                # projection above -- the compiler rejects a parameter shared
                # between two ETP primitives -- but start from the same frozen
                # bases, so step 0 is function-identical to `"frozen"`.
                self.write_key_weight = brainstate.ParamState(
                    MEMORY_KEY_RFF_GAMMA * self._memory_key_basis
                )
                self.write_key_bias = brainstate.ParamState(self._memory_key_bias)
                self.write_value_weight = brainstate.ParamState(
                    self._memory_value_basis
                )
            self.workspace_query_projection = braintrace.nn.Linear(
                config.neuron_count,
                memory_width,
                w_init=(
                    query_random.randn(config.neuron_count, memory_width)
                    / math.sqrt(config.neuron_count)
                ),
                b_init=None,
            )
            self.memory_read_projection = braintrace.nn.Linear(
                memory_width,
                config.neuron_count,
                w_init=(
                    read_random.randn(memory_width, config.neuron_count)
                    / math.sqrt(memory_width)
                ),
                b_init=None,
            )
            self.context_memory = brainstate.HiddenState(
                jnp.zeros(
                    (config.batch_size, memory_width, memory_width),
                    dtype=jnp.float32,
                )
            )
            self.query_encoding = brainstate.HiddenState(
                jnp.zeros((config.batch_size, memory_width), dtype=jnp.float32)
            )
            self.reasoning_query = brainstate.HiddenState(
                jnp.zeros((config.batch_size, memory_width), dtype=jnp.float32)
            )
            self.memory_read = brainstate.HiddenState(
                jnp.zeros((config.batch_size, memory_width), dtype=jnp.float32)
            )
            self.workspace_carrier = brainstate.HiddenState(
                jnp.zeros((config.batch_size, config.neuron_count), dtype=jnp.float32)
            )
        if config.row_refinement_enabled:
            head_width = refinement_head_width(config.neuron_count)
            row_weights = random.randn(head_width, MAX_GRID_SIZE * COLOR_COUNT)
            row_weights = row_weights / math.sqrt(head_width)
            shape_weights = random.randn(head_width, 2 * MAX_GRID_SIZE)
            shape_weights = shape_weights / math.sqrt(head_width)
            if config.refinement_mixer == "carrier_gate":
                event_width = head_width - config.neuron_count
                self.answer_row_event_head = braintrace.nn.Linear(
                    event_width,
                    MAX_GRID_SIZE * COLOR_COUNT,
                    w_init=row_weights[config.neuron_count :],
                    b_init=None,
                )
                self.answer_row_carrier_head = braintrace.nn.Linear(
                    config.neuron_count,
                    MAX_GRID_SIZE * COLOR_COUNT,
                    w_init=row_weights[: config.neuron_count],
                    b_init=None,
                )
                self.row_carrier_gate_head = braintrace.nn.Linear(
                    1,
                    MAX_GRID_SIZE * COLOR_COUNT,
                    w_init=jnp.zeros((1, MAX_GRID_SIZE * COLOR_COUNT)),
                    b_init=None,
                )
            elif config.refinement_mixer == "attention_residual":
                self.answer_row_proposal_head = braintrace.nn.Linear(
                    head_width,
                    MAX_GRID_SIZE * COLOR_COUNT,
                    w_init=row_weights,
                    b_init=None,
                )
                self.answer_shape_proposal_head = braintrace.nn.Linear(
                    head_width,
                    2 * MAX_GRID_SIZE,
                    w_init=shape_weights,
                    b_init=None,
                )
                self.row_attention_residual = braintrace.nn.AttentionResidual(
                    MAX_GRID_SIZE * COLOR_COUNT,
                    query_count=config.refinement_sweeps,
                )
                self.shape_attention_residual = braintrace.nn.AttentionResidual(
                    2 * MAX_GRID_SIZE,
                    query_count=config.refinement_sweeps,
                )
            else:
                self.answer_row_head = braintrace.nn.Linear(
                    head_width,
                    MAX_GRID_SIZE * COLOR_COUNT,
                    w_init=row_weights,
                    b_init=None,
                )
            if config.refinement_mixer != "attention_residual":
                self.answer_shape_head = braintrace.nn.Linear(
                    head_width,
                    2 * MAX_GRID_SIZE,
                    w_init=shape_weights,
                    b_init=None,
                )
            self.answer_row = brainstate.HiddenState(
                jnp.zeros(
                    (config.batch_size, MAX_GRID_SIZE * COLOR_COUNT),
                    dtype=jnp.float32,
                )
            )
            self.answer_shape = brainstate.HiddenState(
                jnp.zeros((config.batch_size, 2 * MAX_GRID_SIZE), dtype=jnp.float32)
            )
            self.query_grid = brainstate.ShortTermState(
                jnp.zeros(
                    (
                        config.batch_size,
                        MAX_GRID_SIZE,
                        MAX_GRID_SIZE,
                        COLOR_COUNT,
                    ),
                    dtype=jnp.float32,
                )
            )
            self.query_shape = brainstate.ShortTermState(
                jnp.zeros((config.batch_size, 2 * MAX_GRID_SIZE), dtype=jnp.float32)
            )
            self.answer_grid = brainstate.ShortTermState(
                jnp.zeros(
                    (
                        config.batch_size,
                        MAX_GRID_SIZE,
                        MAX_GRID_SIZE,
                        COLOR_COUNT,
                    ),
                    dtype=jnp.float32,
                )
            )
            self.reasoning_index = brainstate.ShortTermState(
                jnp.zeros((config.batch_size,), dtype=jnp.int32)
            )
            if config.refinement_mixer == "attention_residual":
                self.row_proposal = brainstate.HiddenState(
                    jnp.zeros(
                        (config.batch_size, MAX_GRID_SIZE * COLOR_COUNT),
                        dtype=jnp.float32,
                    )
                )
                self.shape_proposal = brainstate.HiddenState(
                    jnp.zeros(
                        (config.batch_size, 2 * MAX_GRID_SIZE), dtype=jnp.float32
                    )
                )
                self.row_proposal_history = brainstate.ShortTermState(
                    jnp.zeros(
                        (
                            config.batch_size,
                            config.refinement_sweeps,
                            MAX_GRID_SIZE,
                            MAX_GRID_SIZE * COLOR_COUNT,
                        ),
                        dtype=jnp.float32,
                    )
                )
                self.shape_proposal_history = brainstate.ShortTermState(
                    jnp.zeros(
                        (
                            config.batch_size,
                            config.refinement_sweeps,
                            2 * MAX_GRID_SIZE,
                        ),
                        dtype=jnp.float32,
                    )
                )
                self.reasoning_sweep = brainstate.ShortTermState(
                    jnp.zeros((config.batch_size,), dtype=jnp.int32)
                )
        brainstate.nn.init_all_states(self, batch_size=config.batch_size)

    @property
    def memory_read_policy(self) -> Literal["full", "query_only"]:
        """Return the immutable contextual-memory read intervention.

        Returns
        -------
        {"full", "query_only"}
            Static policy selected when the model was constructed.
        """
        return self._memory_read_policy

    @property
    def neuron_count(self) -> int:
        """Return the physical LIF population size."""
        return self.config.neuron_count

    @property
    def recurrent_edge_count(self) -> int:
        """Return the instantiated recurrent edge count."""
        return self.topology.edge_count

    @property
    def slot_count(self) -> int:
        """Return the number of exact 64-neuron slots."""
        return self.config.slot_count

    @property
    def spikes(self) -> jax.Array:
        """Return current binary LIF spikes shaped ``(batch, neurons)``."""
        return jnp.asarray(self.neu.get_spike())

    @property
    def voltage(self) -> jax.Array:
        """Return current membrane voltage numeric values in millivolts."""
        return jnp.asarray(self.neu.V.value.to_decimal(u.mV))

    @property
    def feedforward_current(self) -> jax.Array:
        """Return the feed-forward Expon state numerically in milliamps."""
        return jnp.asarray(self.ff_syn.syn.g.value.to_decimal(u.mA))

    @property
    def recurrent_current(self) -> jax.Array:
        """Return the recurrent Expon state numerically in milliamps."""
        return jnp.asarray(self.rec_syn.syn.g.value.to_decimal(u.mA))

    def project_recurrent_dale_weights(self) -> None:
        """Clamp recurrent weights onto their presynaptic Dale sign in place.

        Under ``neuron_typing="ei_dale"`` this rewrites the recurrent sparse
        weight so excitatory rows stay nonnegative and inhibitory rows stay
        nonpositive; call it immediately after every optimizer step.  It is a
        no-op under ``neuron_typing="none"`` and inside a trace it lowers to
        a single elementwise ``where``.

        Examples
        --------
        .. code-block:: python

            >>> config = ModelConfig(input_width=8, neuron_count=128,
            ...                      recurrent_edges=1024,
            ...                      neuron_typing="ei_dale")
            >>> model = LatentWorkspaceModel(config)
            >>> model.project_recurrent_dale_weights()
            >>> model.neuron_typing_report()["recurrent_sign_violation_count"]
            0
        """
        if self._dale_edge_signs is None:
            return
        parameters = dict(self.rec_syn.comm.weight.value)
        parameters["weight"] = project_dale_weights(
            parameters["weight"], self._dale_edge_signs
        )
        self.rec_syn.comm.weight.value = parameters

    def neuron_typing_report(self) -> dict[str, object]:
        """Describe the neuron-type structure and current Dale compliance.

        Returns
        -------
        dict
            JSON-safe mapping.  Under ``"none"`` only ``mode`` and null
            counts are reported.  Under ``"ei_dale"`` it carries the E/I
            counts, the realized excitatory fraction, the number of edges
            whose initial random sign was flipped by the Dale constraint,
            and the number of *current* recurrent weights violating their
            presynaptic sign (zero after init and after every projection).

        Examples
        --------
        .. code-block:: python

            >>> config = ModelConfig(input_width=8, neuron_count=128,
            ...                      recurrent_edges=1024)
            >>> LatentWorkspaceModel(config).neuron_typing_report()["mode"]
            'none'
        """
        if self.neuron_type_signs is None:
            return {
                "mode": "none",
                "excitatory_count": None,
                "inhibitory_count": None,
                "configured_excitatory_fraction": None,
                "realized_excitatory_fraction": None,
                "initial_sign_flip_count": None,
                "recurrent_sign_violation_count": None,
            }
        signs = np.asarray(self.neuron_type_signs)
        excitatory = int(np.sum(signs == 1))
        weight = np.asarray(
            u.get_mantissa(self.rec_syn.comm.weight.value["weight"])
        )
        edge_signs = np.asarray(self._dale_edge_signs)
        return {
            "mode": "ei_dale",
            "excitatory_count": excitatory,
            "inhibitory_count": int(signs.size - excitatory),
            "configured_excitatory_fraction": float(
                self.config.excitatory_fraction
            ),
            "realized_excitatory_fraction": excitatory / signs.size,
            "initial_sign_flip_count": self._dale_init_flip_count,
            "recurrent_sign_violation_count": int(
                np.sum(weight * edge_signs < 0)
            ),
        }

    def encode_memory_key(self, event: jax.Array) -> jax.Array:
        """Encode input-side row features with a fixed nonlinear key map.

        Parameters
        ----------
        event : jax.Array
            Native batched row event shaped ``(batch, input_width)``.

        Returns
        -------
        jax.Array
            Random Fourier key code shaped ``(batch, memory_width)``.  Rows
            without a valid input side are exactly zero.
        """
        if not self.config.memory_enabled:
            raise RuntimeError("context memory is disabled")
        event = jnp.asarray(event, dtype=jnp.float32)
        expected = (self.config.batch_size, self.config.input_width)
        if event.shape != expected:
            raise ValueError(f"event must have shape {expected}, got {event.shape}")
        features = event[..., jnp.asarray(self.config.memory_key_indices)]
        if self.config.memory_coding in LEARNED_RETRIEVAL_KEY_CODINGS:
            phase = self.memory_key_projection(features)
        else:
            phase = (
                MEMORY_KEY_RFF_GAMMA * (features @ self._memory_key_basis)
                + self._memory_key_bias
            )
        scale = math.sqrt(2.0 / self.config.context_memory_width)
        code = scale * jnp.cos(phase)
        assert self.config.input_side_valid_index is not None
        side_valid = event[..., self.config.input_side_valid_index] > 0.5
        return jnp.where(side_valid[:, None], code, jnp.zeros_like(code))

    def encode_memory_value(self, event: jax.Array) -> jax.Array:
        """Encode output-side row features with a separate fixed value map.

        Parameters
        ----------
        event : jax.Array
            Native batched row event shaped ``(batch, input_width)``.

        Returns
        -------
        jax.Array
            Value code shaped ``(batch, memory_width)``, softcap-bounded to
            ``(-memory_value_softcap_beta, memory_value_softcap_beta)``.
            Rows without a valid output side are exactly zero.
        """
        if not self.config.memory_enabled:
            raise RuntimeError("context memory is disabled")
        event = jnp.asarray(event, dtype=jnp.float32)
        expected = (self.config.batch_size, self.config.input_width)
        if event.shape != expected:
            raise ValueError(f"event must have shape {expected}, got {event.shape}")
        features = event[..., jnp.asarray(self.config.memory_value_indices)]
        code = softcap(
            features @ self._memory_value_basis,
            self.config.memory_value_softcap_beta,
        )
        assert self.config.output_side_valid_index is not None
        side_valid = event[..., self.config.output_side_valid_index] > 0.5
        return jnp.where(side_valid[:, None], code, jnp.zeros_like(code))

    def memory_coding_divergence(self) -> dict[str, float]:
        """Report how far the write key encoder has drifted from the retrieval one.

        Under ``"learned_write"`` the memory is written in
        ``write_key_weight`` space and queried in ``memory_key_projection``
        space. The two start identical and then train independently, so they
        can drift apart far enough that retrieval degrades for reasons
        unrelated to binding. Without this measurement a pinned-at-zero pairing
        result is unattributable: a broken read and an absent binder look the
        same downstream, and a gradient check cannot separate them because the
        gradient is fine — it is the forward retrieval that decayed.

        ``write_key_row_norm_mean`` covers the companion risk: the folded-gamma
        initialization pins the key scale only at step 0, and an unnormalized
        key whose norm grows lets large-scale components dominate the read.

        Returns
        -------
        dict
            Cosine and relative-L2 divergence of the two key projections, the
            relative-L2 divergence of their phase offsets, and the mean L2 norm
            of the write projection's rows. Empty for every coding other than
            ``"learned_write"``, which has only one key encoder.
        """
        if self.config.memory_coding != "learned_write":
            return {}
        write_weight = jnp.asarray(self.write_key_weight.value).reshape(-1)
        retrieval_weight = jnp.asarray(
            self.memory_key_projection.weight.value["weight"]
        ).reshape(-1)
        write_bias = jnp.asarray(self.write_key_bias.value).reshape(-1)
        retrieval_bias = jnp.asarray(
            self.memory_key_projection.weight.value["bias"]
        ).reshape(-1)
        return {
            "write_retrieval_key_cosine": float(
                write_weight
                @ retrieval_weight
                / (
                    jnp.linalg.norm(write_weight)
                    * jnp.linalg.norm(retrieval_weight)
                )
            ),
            "write_retrieval_key_relative_l2": float(
                jnp.linalg.norm(write_weight - retrieval_weight)
                / jnp.linalg.norm(retrieval_weight)
            ),
            "write_retrieval_key_bias_relative_l2": float(
                jnp.linalg.norm(write_bias - retrieval_bias)
                / jnp.linalg.norm(retrieval_bias)
            ),
            "write_key_row_norm_mean": float(
                jnp.mean(
                    jnp.linalg.norm(
                        jnp.asarray(self.write_key_weight.value), axis=0)
                )
            ),
        }

    def encode_memory_write(self, event: jax.Array) -> jax.Array:
        """Encode one row event directly into its rank-one write matrix.

        The fused counterpart of :meth:`encode_memory_key` followed by
        :meth:`encode_memory_value` and an outer product. Forming the product
        inside ``braintrace.outer_write`` is what lets the write projections
        carry eligibility-trace gradient at all: an outer product assembled
        outside an ETP primitive mixes hidden positions, which pp-prop rejects.

        Parameters
        ----------
        event : jax.Array
            Native batched row event shaped ``(batch, input_width)``.

        Returns
        -------
        jax.Array
            Write matrix shaped ``(batch, memory_width, memory_width)``. Rows
            missing either side are exactly zero, matching the frozen path
            where a zeroed key or value annihilates the product.

        Raises
        ------
        RuntimeError
            If context memory is disabled, or the coding is not
            ``"learned_write"``.
        """
        if not self.config.memory_enabled:
            raise RuntimeError("context memory is disabled")
        if self.config.memory_coding != "learned_write":
            raise RuntimeError(
                "encode_memory_write requires memory_coding='learned_write'; "
                f"got {self.config.memory_coding!r}"
            )
        event = jnp.asarray(event, dtype=jnp.float32)
        expected = (self.config.batch_size, self.config.input_width)
        if event.shape != expected:
            raise ValueError(f"event must have shape {expected}, got {event.shape}")
        write = braintrace.outer_write(
            event[..., jnp.asarray(self.config.memory_key_indices)],
            event[..., jnp.asarray(self.config.memory_value_indices)],
            key_weight=self.write_key_weight.value,
            key_bias=self.write_key_bias.value,
            value_weight=self.write_value_weight.value,
            key_scale=math.sqrt(2.0 / self.config.context_memory_width),
        )
        assert self.config.input_side_valid_index is not None
        assert self.config.output_side_valid_index is not None
        both_sides_valid = (
            (event[..., self.config.input_side_valid_index] > 0.5)
            & (event[..., self.config.output_side_valid_index] > 0.5)
        )
        return write * both_sides_valid[:, None, None]

    def read_context_memory(self, query: jax.Array | None = None) -> jax.Array:
        """Read the frozen contextual memory with a key-space query.

        Parameters
        ----------
        query : jax.Array, optional
            Query shaped ``(batch, memory_width)``.  The current explicit
            reasoning query is used when omitted.

        Returns
        -------
        jax.Array
            Associative read shaped ``(batch, memory_width)``.
        """
        if not self.config.memory_enabled:
            raise RuntimeError("context memory is disabled")
        if query is None:
            query = self.reasoning_query.value
        query = jnp.asarray(query)
        expected = (
            self.config.batch_size,
            self.config.context_memory_width,
        )
        if query.shape != expected:
            raise ValueError(f"query must have shape {expected}, got {query.shape}")
        return jnp.einsum("bkv,bk->bv", self.context_memory.value, query)

    def associative_memory_report(self) -> AssociativeMemoryReport:
        """Return a stable read-only description of the memory architecture.

        Returns
        -------
        AssociativeMemoryReport
            Fixed widths, encoder identity, basis provenance, and trainable
            component types.  Legacy mode reports no memory arrays or paths.
        """
        if not self.config.memory_enabled:
            return AssociativeMemoryReport(
                mode="legacy_reservoir",
                memory_width=0,
                key_feature_width=0,
                value_feature_width=0,
                key_map=None,
                value_map=None,
                rff_gamma=None,
                key_basis_seed=None,
                key_bias_seed=None,
                value_basis_seed=None,
                key_basis_sha256=None,
                key_bias_sha256=None,
                value_basis_sha256=None,
                write_component_type=None,
                query_component_type=None,
                read_component_type=None,
            )
        return AssociativeMemoryReport(
            mode="associative_workspace",
            memory_width=self.config.context_memory_width,
            key_feature_width=len(self.config.memory_key_indices),
            value_feature_width=len(self.config.memory_value_indices),
            key_map=_MEMORY_KEY_MAP_NAMES[self.config.memory_coding],
            value_map=(
                "learned_tanh_projection"
                if self.config.memory_coding == "learned_write"
                else "fixed_tanh_projection"
            ),
            rff_gamma=MEMORY_KEY_RFF_GAMMA,
            key_basis_seed=self.config.seed + 101,
            key_bias_seed=self.config.seed + 102,
            value_basis_seed=self.config.seed + 103,
            key_basis_sha256=_array_sha256(self._memory_key_basis),
            key_bias_sha256=_array_sha256(self._memory_key_bias),
            value_basis_sha256=_array_sha256(self._memory_value_basis),
            write_component_type=(
                "braintrace.outer_write"
                if self.config.memory_coding == "learned_write"
                else "braintrace.element_wise"
            ),
            query_component_type="braintrace.nn.Linear",
            read_component_type="braintrace.nn.Linear",
        )

    def reset_state(self, batch_size: int | None = None, **_: object) -> None:
        """Reset every inference state without modifying any parameter.

        Parameters
        ----------
        batch_size : int, optional
            If supplied, it must match the configured native batch size.
        **_ : object
            Extra framework reset options, accepted and ignored.
        """
        if batch_size is not None and batch_size != self.config.batch_size:
            raise ValueError(
                f"batch_size {batch_size} does not match configured "
                f"{self.config.batch_size}"
            )
        self.neu.reset_state(batch_size=self.config.batch_size)
        self.ff_syn.syn.reset_state(batch_size=self.config.batch_size)
        self.rec_syn.syn.reset_state(batch_size=self.config.batch_size)
        if self.config.memory_enabled:
            memory_width = self.config.context_memory_width
            self.context_memory.value = jnp.zeros(
                (self.config.batch_size, memory_width, memory_width),
                dtype=jnp.float32,
            )
            zero_query = jnp.zeros(
                (self.config.batch_size, memory_width), dtype=jnp.float32
            )
            self.query_encoding.value = zero_query
            self.reasoning_query.value = zero_query
            self.memory_read.value = zero_query
            self.workspace_carrier.value = jnp.zeros(
                (self.config.batch_size, self.config.neuron_count),
                dtype=jnp.float32,
            )
        if self.config.row_refinement_enabled:
            self.answer_row.value = jnp.zeros(
                (self.config.batch_size, MAX_GRID_SIZE * COLOR_COUNT),
                dtype=jnp.float32,
            )
            self.answer_shape.value = jnp.zeros(
                (self.config.batch_size, 2 * MAX_GRID_SIZE), dtype=jnp.float32
            )
            self.query_grid.value = jnp.zeros(
                (
                    self.config.batch_size,
                    MAX_GRID_SIZE,
                    MAX_GRID_SIZE,
                    COLOR_COUNT,
                ),
                dtype=jnp.float32,
            )
            self.query_shape.value = jnp.zeros(
                (self.config.batch_size, 2 * MAX_GRID_SIZE), dtype=jnp.float32
            )
            self.answer_grid.value = jnp.zeros(
                (
                    self.config.batch_size,
                    MAX_GRID_SIZE,
                    MAX_GRID_SIZE,
                    COLOR_COUNT,
                ),
                dtype=jnp.float32,
            )
            self.reasoning_index.value = jnp.zeros(
                (self.config.batch_size,), dtype=jnp.int32
            )
            if self.config.refinement_mixer == "attention_residual":
                self.row_proposal.value = jnp.zeros(
                    (self.config.batch_size, MAX_GRID_SIZE * COLOR_COUNT),
                    dtype=jnp.float32,
                )
                self.shape_proposal.value = jnp.zeros(
                    (self.config.batch_size, 2 * MAX_GRID_SIZE), dtype=jnp.float32
                )
                self.row_proposal_history.value = jnp.zeros(
                    (
                        self.config.batch_size,
                        self.config.refinement_sweeps,
                        MAX_GRID_SIZE,
                        MAX_GRID_SIZE * COLOR_COUNT,
                    ),
                    dtype=jnp.float32,
                )
                self.shape_proposal_history.value = jnp.zeros(
                    (
                        self.config.batch_size,
                        self.config.refinement_sweeps,
                        2 * MAX_GRID_SIZE,
                    ),
                    dtype=jnp.float32,
                )
                self.reasoning_sweep.value = jnp.zeros(
                    (self.config.batch_size,), dtype=jnp.int32
                )

    def _snapshot_state_items(self) -> tuple[tuple[tuple[Any, ...], Any], ...]:
        """Return every state required for exact inference restoration."""

        items = tuple(
            (tuple(path), state)
            for path, state in self.states(brainstate.HiddenState).items()
        )
        if not self.config.row_refinement_enabled:
            return items
        refinement_items = (
            (("query_grid",), self.query_grid),
            (("query_shape",), self.query_shape),
            (("answer_grid",), self.answer_grid),
            (("reasoning_index",), self.reasoning_index),
        )
        if self.config.refinement_mixer == "attention_residual":
            refinement_items += (
                (("row_proposal_history",), self.row_proposal_history),
                (("shape_proposal_history",), self.shape_proposal_history),
                (("reasoning_sweep",), self.reasoning_sweep),
            )
        hidden_paths = {path for path, _ in items}
        return items + tuple(
            item for item in refinement_items if item[0] not in hidden_paths
        )

    def snapshot_state(self) -> ModelStateSnapshot:
        """Copy every inference state while excluding all parameters.

        Returns
        -------
        ModelStateSnapshot
            Exact restorable state keyed by BrainState graph path.
        """
        entries = tuple(
            (tuple(path), _copy_tree(state.value))
            for path, state in self._snapshot_state_items()
        )
        return ModelStateSnapshot(
            entries=entries,
            batch_size=self.config.batch_size,
            neuron_count=self.config.neuron_count,
        )

    def restore_state(self, snapshot: ModelStateSnapshot) -> None:
        """Restore a compatible hidden-state snapshot exactly.

        Parameters
        ----------
        snapshot : ModelStateSnapshot
            Snapshot produced by this model configuration.
        """
        if not isinstance(snapshot, ModelStateSnapshot):
            raise TypeError("snapshot must be a ModelStateSnapshot")
        if (
            snapshot.batch_size != self.config.batch_size
            or snapshot.neuron_count != self.config.neuron_count
        ):
            raise ValueError("snapshot configuration does not match this model")
        state_items = self._snapshot_state_items()
        expected_paths = tuple(path for path, _ in state_items)
        actual_paths = tuple(path for path, _ in snapshot.entries)
        if actual_paths != expected_paths:
            raise ValueError("snapshot state paths do not match this model")
        validated: list[tuple[Any, Any]] = []
        for (path, value), (_, state) in zip(
            snapshot.entries, state_items, strict=True
        ):
            if tuple(path) not in expected_paths:
                raise ValueError("snapshot contains an unknown state path")
            value_structure = jax.tree.structure(value)
            state_structure = jax.tree.structure(state.value)
            value_leaves = jax.tree.leaves(value)
            state_leaves = jax.tree.leaves(state.value)
            if value_structure != state_structure or len(value_leaves) != len(
                state_leaves
            ):
                raise ValueError("snapshot state structure does not match model")
            if any(
                np.shape(value_leaf) != np.shape(state_leaf)
                for value_leaf, state_leaf in zip(
                    value_leaves, state_leaves, strict=True
                )
            ):
                raise ValueError("snapshot state shape does not match model")
            validated.append((state, value))
        for state, value in validated:
            state.value = _copy_tree(value)

    def ablate_slot(self, slot_index: int) -> tuple[jax.Array, jax.Array]:
        """Zero one exact 64-neuron voltage/spike slice without changing weights.

        Parameters
        ----------
        slot_index : int
            Zero-based slot index in ``[0, slot_count)``.
        Returns
        -------
        tuple of jax.Array
            Post-ablation voltage and derived spikes.  The selected 64-neuron
            slice is exactly zero in both arrays because BrainPy LIF computes
            spikes directly from its voltage state.
        """
        slot_index = _nonnegative_integer(slot_index, "slot_index")
        if slot_index >= self.slot_count:
            raise ValueError(f"slot_index {slot_index} outside [0, {self.slot_count})")
        slots = jnp.full((self.config.batch_size,), slot_index, dtype=jnp.int32)
        enabled = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)
        return self.mask_slots(slots, enabled)

    def mask_slots(
        self, slot_indices: jax.Array, enabled: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Apply a jittable per-example 64-neuron state-ablation gate.

        Parameters
        ----------
        slot_indices : jax.Array
            Integer slot index for each batch element, shaped ``(batch,)``.
        enabled : jax.Array
            Boolean gate per batch element.  False elements remain unchanged.

        Returns
        -------
        tuple of jax.Array
            Post-mask numeric voltage and derived spikes, each shaped
            ``(batch, neurons)``.

        Notes
        -----
        :class:`brainpy.state.LIF` has no separate spike state:
        ``get_spike()`` derives spikes from ``V`` on demand.  Masking ``V``
        therefore masks the exact spike vector consumed by ``rec_syn`` on the
        next step.  Callers that accept traced indices must validate their
        range before entering a compiled loop; :meth:`ablate_slot` provides the
        checked scalar interface.
        """
        slot_indices = jnp.asarray(slot_indices, dtype=jnp.int32)
        enabled = jnp.asarray(enabled, dtype=jnp.bool_)
        expected = (self.config.batch_size,)
        if slot_indices.shape != expected or enabled.shape != expected:
            raise ValueError(
                f"slot_indices and enabled must each have shape {expected}"
            )
        neuron_slots = jnp.arange(self.config.neuron_count) // NEURONS_PER_SLOT
        selected = neuron_slots[None, :] == slot_indices[:, None]
        keep = ~(selected & enabled[:, None])
        self.neu.V.value = self.neu.V.value * keep
        if self.config.memory_enabled:
            self.workspace_carrier.value = self.workspace_carrier.value * keep
        return self.voltage, self.spikes

    def etrace_config(self) -> braintrace.ETraceConfig:
        """Return the configured eligibility-trace coordinate.

        Returns
        -------
        braintrace.ETraceConfig
            Diagonal recurrent pp-prop with the configured scalar trace decay
            under ``trace_engine="pp_prop"``; the per-parameter exact-trace
            (D-RTRL) coordinate under ``trace_engine="d_rtrl"``, which has no
            decay knob because the trace follows the true recurrence.
        """
        if self.config.trace_engine == "d_rtrl":
            return braintrace.ETraceConfig(
                trace_factorization="per_param",
                recurrence_scope="diagonal",
            )
        return braintrace.ETraceConfig(
            trace_factorization="io_factorized",
            recurrence_scope="diagonal",
            decay=self.config.trace_decay,
        )

    def compact_readout(self, carrier: jax.Array | None = None) -> jax.Array:
        """Return checkpoint logits for the configured decoder.

        Parameters
        ----------
        carrier : jax.Array, optional
            Workspace values shaped ``(batch, neurons)``.  When omitted,
            memory mode uses its continuous workspace state and legacy mode
            uses current LIF spikes.

        Returns
        -------
        jax.Array
            Legacy compact factors or explicit row-refinement logits shaped
            ``(batch, checkpoint_output_width)``.
        """
        if self.config.row_refinement_enabled:
            return refinement_output_logits(
                self.answer_shape.value, self.answer_grid.value
            )
        if carrier is None:
            carrier = (
                self.workspace_carrier.value
                if self.config.memory_enabled
                else self.spikes
            )
        carrier = jnp.asarray(carrier)
        if carrier.shape != (self.config.batch_size, self.config.neuron_count):
            raise ValueError(
                "carrier must have shape "
                f"({self.config.batch_size}, {self.config.neuron_count}), got "
                f"{carrier.shape}"
            )
        if self.config.memory_enabled:
            carrier = _unit_l2_cap(carrier)
        hidden = jax.nn.gelu(self.readout_projection(carrier))
        return jnp.concatenate(
            (
                self.height_head(hidden),
                self.width_head(hidden),
                self.color_factor_head(hidden),
            ),
            axis=-1,
        )

    def training_readout(self, carrier: jax.Array | None = None) -> jax.Array:
        """Return the bounded per-tick output consumed by BrainTrace.

        Parameters
        ----------
        carrier : jax.Array, optional
            Legacy workspace carrier. Row-refinement mode reads its current
            shape and row states directly.

        Returns
        -------
        jax.Array
            Per-tick logits shaped ``(batch, training_output_width)``.
        """

        if self.config.row_refinement_enabled:
            return refinement_training_logits(
                self.answer_shape.value, self.answer_row.value
            )
        return self.compact_readout(carrier)

    def _row_head_logits(
        self, unit_carrier: jax.Array, event: jax.Array, head_input: jax.Array
    ) -> jax.Array:
        """Row logits before the copy residual, under scale or gate access.

        Without the gate this is the single row head over the (optionally
        carrier-scaled) concatenated input.  With the gate, the event-only
        head fires on the event blocks of ``head_input`` and the carrier
        head's contribution is multiplied by ``tanh`` of the gate weight,
        which is zero at initialisation — the §9.1 carrier-free start.
        """
        assert self.config.refinement_layout is not None
        if self.config.refinement_mixer == "linear":
            row_head_input = _carrier_scaled_head_input(
                unit_carrier,
                event,
                self.config.refinement_layout,
                self.config.row_head_carrier_scale,
                head_input,
            )
            return self.answer_row_head(row_head_input)
        if self.config.refinement_mixer == "attention_residual":
            return self.answer_row_proposal_head(head_input)
        event_blocks = head_input[:, self.config.neuron_count :]
        gate_input = jnp.ones((head_input.shape[0], 1), dtype=head_input.dtype)
        gate = jnp.tanh(self.row_carrier_gate_head(gate_input))
        carrier_logits = self.answer_row_carrier_head(unit_carrier)
        return self.answer_row_event_head(event_blocks) + gate * carrier_logits

    def _shape_head_logits(self, head_input: jax.Array) -> jax.Array:
        """Return the current shape proposal under the configured mixer."""
        if self.config.refinement_mixer == "attention_residual":
            return self.answer_shape_proposal_head(head_input)
        return self.answer_shape_head(head_input)

    def _attention_refinement_outputs(
        self,
        row_proposal: jax.Array,
        shape_proposal: jax.Array,
        refinement_rows: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Mix query identity, completed sweeps, and current proposals."""
        batch_indices = jnp.arange(self.config.batch_size)
        sweeps = jnp.asarray(self.reasoning_sweep.value)
        row_identity = self.query_grid.value[
            batch_indices, refinement_rows
        ].reshape(self.config.batch_size, -1)
        row_identity = _rms_balanced_identity_source(row_identity, row_proposal)
        shape_identity = _rms_balanced_identity_source(
            self.query_shape.value, shape_proposal
        )
        row_history = jax.vmap(
            lambda history, row: history[:, row, :]
        )(jax.lax.stop_gradient(self.row_proposal_history.value), refinement_rows)
        shape_history = jax.lax.stop_gradient(self.shape_proposal_history.value)
        row_sources = jnp.concatenate(
            (row_identity[:, None, :], row_history, row_proposal[:, None, :]),
            axis=1,
        )
        shape_sources = jnp.concatenate(
            (
                shape_identity[:, None, :],
                shape_history,
                shape_proposal[:, None, :],
            ),
            axis=1,
        )
        history_valid = (
            jnp.arange(self.config.refinement_sweeps)[None, :] < sweeps[:, None]
        )
        source_mask = jnp.concatenate(
            (
                jnp.ones((self.config.batch_size, 1), dtype=jnp.bool_),
                history_valid,
                jnp.ones((self.config.batch_size, 1), dtype=jnp.bool_),
            ),
            axis=1,
        )
        return (
            self.row_attention_residual(
                row_sources, source_mask=source_mask, query_index=sweeps
            ),
            self.shape_attention_residual(
                shape_sources, source_mask=source_mask, query_index=sweeps
            ),
        )

    def _cache_attention_proposals(
        self,
        row_proposal: jax.Array,
        shape_proposal: jax.Array,
        refinement_rows: jax.Array,
        refinement_latent: jax.Array,
    ) -> None:
        """Store current proposals and advance the completed-sweep index."""
        sweeps = jnp.asarray(self.reasoning_sweep.value)
        updated_rows = jax.vmap(
            lambda history, sweep, row, value: history.at[sweep, row].set(value)
        )(
            self.row_proposal_history.value,
            sweeps,
            refinement_rows,
            row_proposal,
        )
        self.row_proposal_history.value = jnp.where(
            refinement_latent[:, None, None, None],
            updated_rows,
            self.row_proposal_history.value,
        )
        sweep_complete = refinement_latent & (refinement_rows == MAX_GRID_SIZE - 1)
        updated_shapes = jax.vmap(
            lambda history, sweep, value: history.at[sweep].set(value)
        )(self.shape_proposal_history.value, sweeps, shape_proposal)
        self.shape_proposal_history.value = jnp.where(
            sweep_complete[:, None, None],
            updated_shapes,
            self.shape_proposal_history.value,
        )
        next_sweep = jnp.minimum(sweeps + 1, self.config.refinement_sweeps - 1)
        self.reasoning_sweep.value = jnp.where(
            sweep_complete, next_sweep, sweeps
        )

    def cell_step(
        self, event: jax.Array, advance: jax.Array | None = None
    ) -> jax.Array:
        """Advance one physical row-event or zero-input latent step.

        Parameters
        ----------
        event : jax.Array
            Native batched external input shaped ``(batch, input_width)``.
        advance : jax.Array, optional
            Boolean state-advance gate shaped ``(batch,)``.  When omitted it
            is read from ``event[:, event_valid_index]``.  False rows leave
            voltage and both synaptic currents byte-identical to their prior
            values.  Latent callers pass true while keeping ``event`` zero.

        Returns
        -------
        jax.Array
            Current binary spikes shaped ``(batch, neurons)``.
        """
        event = jnp.asarray(event, dtype=jnp.float32)
        expected = (self.config.batch_size, self.config.input_width)
        if event.shape != expected:
            raise ValueError(f"event must have shape {expected}, got {event.shape}")
        if advance is None:
            advance = event[:, self.config.event_valid_index] > 0.5
        else:
            advance = jnp.asarray(advance, dtype=jnp.bool_)
        if advance.shape != (self.config.batch_size,):
            raise ValueError(
                f"advance must have shape ({self.config.batch_size},), got "
                f"{advance.shape}"
            )
        refinement_latent = jnp.zeros((self.config.batch_size,), dtype=jnp.bool_)
        refinement_rows = jnp.zeros((self.config.batch_size,), dtype=jnp.int32)
        if self.config.row_refinement_enabled:
            assert self.config.refinement_layout is not None
            refinement_rows = jnp.asarray(self.reasoning_index.value)
            self.query_grid.value, self.query_shape.value = capture_query_rows(
                self.query_grid.value,
                self.query_shape.value,
                event,
                advance,
                self.config.refinement_layout,
            )
            refinement_latent = advance & ~(
                event[:, self.config.refinement_layout.event_valid_index] > 0.5
            )
            feedback_grid = scatter_answer_rows(
                self.answer_grid.value,
                self.answer_row.value,
                refinement_rows,
            )
            feedback_event = build_refinement_feedback_event(
                self.query_grid.value,
                self.query_shape.value,
                feedback_grid,
                self.answer_shape.value,
                refinement_rows,
                self.config.refinement_layout,
            )
            event = jnp.where(refinement_latent[:, None], feedback_event, event)
        previous_voltage = self.neu.V.value
        previous_feedforward = self.ff_syn.syn.g.value
        previous_recurrent = self.rec_syn.syn.g.value
        memory_drive = jnp.zeros(
            (self.config.batch_size, self.config.neuron_count), dtype=jnp.float32
        )
        if self.config.memory_enabled:
            previous_workspace = self.workspace_carrier.value
            event_valid = event[:, self.config.event_valid_index] > 0.5
            assert self.config.demonstration_phase_index is not None
            assert self.config.query_phase_index is not None
            assert self.config.input_side_valid_index is not None
            assert self.config.output_side_valid_index is not None
            demonstration = (
                advance
                & event_valid
                & (event[:, self.config.demonstration_phase_index] > 0.5)
            )
            query = (
                advance
                & event_valid
                & (event[:, self.config.query_phase_index] > 0.5)
                & (event[:, self.config.input_side_valid_index] > 0.5)
            )
            input_side_valid = event[:, self.config.input_side_valid_index] > 0.5
            output_side_valid = event[:, self.config.output_side_valid_index] > 0.5
            write_gate = demonstration & (input_side_valid | output_side_valid)
            key = self.encode_memory_key(event)
            value = self.encode_memory_value(event)
            write_scale = braintrace.element_wise(self.memory_write_scale.value)
            if self.config.memory_coding == "learned_write":
                self.context_memory.value = apply_context_memory_write(
                    self.context_memory.value,
                    self.encode_memory_write(event) * write_scale[None, :, :],
                    write_gate=write_gate,
                    decay=self.config.memory_decay,
                )
            else:
                write_key = key
                if self.config.memory_coding == "learned_keys":
                    # The unfused outer product is outside the differentiable
                    # path under pp-prop; the learned key trains via retrieval.
                    write_key = jax.lax.stop_gradient(key)
                self.context_memory.value = update_context_memory(
                    self.context_memory.value,
                    write_key,
                    value,
                    write_gate=write_gate,
                    decay=self.config.memory_decay,
                    write_scale=write_scale,
                )

            self.query_encoding.value = self.query_encoding.value + jnp.where(
                query[:, None], key, jnp.zeros_like(key)
            )
            latent = advance & ~event_valid
            projected_query = self.workspace_query_projection(
                _unit_l2_cap(previous_workspace)
            )
            iterative_query = softcap(
                self.query_encoding.value + projected_query,
                self.config.reasoning_query_softcap_beta,
            )
            next_reasoning_query = jnp.where(
                query[:, None],
                self.query_encoding.value,
                jnp.where(
                    latent[:, None],
                    iterative_query,
                    self.reasoning_query.value,
                ),
            )
            self.reasoning_query.value = next_reasoning_query
            reasoning_gate = query | latent
            if self.memory_read_policy == "full":
                raw_read = self.read_context_memory(next_reasoning_query)
                raw_read = jnp.where(
                    reasoning_gate[:, None], raw_read, jnp.zeros_like(raw_read)
                )
            else:
                query_read = jnp.where(
                    query[:, None],
                    next_reasoning_query,
                    jnp.zeros_like(next_reasoning_query),
                )
                raw_read = self.read_context_memory(query_read)
                raw_read = jnp.where(query[:, None], raw_read, jnp.zeros_like(raw_read))
            self.memory_read.value = jnp.where(
                reasoning_gate[:, None],
                jax.lax.stop_gradient(raw_read),
                self.memory_read.value,
            )
            memory_drive = self.memory_read_projection(raw_read)
        with brainstate.environ.context(dt=self.config.time_step_ms * u.ms):
            self.ff_syn(event)
            self.rec_syn(self.neu.get_spike())
            self.neu(memory_drive * u.mA)
        gate = advance[:, None]
        self.neu.V.value = u.math.where(gate, self.neu.V.value, previous_voltage)
        self.ff_syn.syn.g.value = u.math.where(
            gate, self.ff_syn.syn.g.value, previous_feedforward
        )
        self.rec_syn.syn.g.value = u.math.where(
            gate, self.rec_syn.syn.g.value, previous_recurrent
        )
        if self.config.memory_enabled:
            self.workspace_carrier.value = jnp.where(
                gate, self.voltage, previous_workspace
            )
        if self.config.row_refinement_enabled:
            assert self.config.refinement_layout is not None
            carrier = (
                self.workspace_carrier.value
                if self.config.memory_enabled
                else self.voltage
            )
            unit_carrier = _unit_rms_carrier(carrier)
            head_input = _refinement_head_input(
                unit_carrier, event, self.config.refinement_layout
            )
            shape_head_input = _carrier_scaled_head_input(
                unit_carrier,
                event,
                self.config.refinement_layout,
                self.config.shape_head_carrier_scale,
                head_input,
            )
            row_proposal = _copy_residual_logits(
                self._row_head_logits(unit_carrier, event, head_input),
                event,
                self.config.refinement_layout,
                self.config.copy_residual_gain,
            )
            shape_proposal = self._shape_head_logits(shape_head_input)
            refinement_gate = refinement_latent[:, None]
            if self.config.refinement_mixer == "attention_residual":
                self.row_proposal.value = jnp.where(
                    refinement_gate, row_proposal, self.row_proposal.value
                )
                self.shape_proposal.value = jnp.where(
                    refinement_gate, shape_proposal, self.shape_proposal.value
                )
                mixed_row, mixed_shape = self._attention_refinement_outputs(
                    self.row_proposal.value,
                    self.shape_proposal.value,
                    refinement_rows,
                )
                next_row = jnp.where(
                    refinement_gate, mixed_row, self.answer_row.value
                )
                next_shape = jnp.where(
                    refinement_gate, mixed_shape, self.answer_shape.value
                )
                self._cache_attention_proposals(
                    self.row_proposal.value,
                    self.shape_proposal.value,
                    refinement_rows,
                    refinement_latent,
                )
            else:
                next_row = row_proposal
                next_shape = shape_proposal
            self.answer_row.value = jnp.where(
                refinement_gate, next_row, self.answer_row.value
            )
            self.answer_shape.value = jnp.where(
                refinement_gate, next_shape, self.answer_shape.value
            )
            scattered = scatter_answer_rows(
                self.answer_grid.value, next_row, refinement_rows
            )
            self.answer_grid.value = jnp.where(
                refinement_latent[:, None, None, None],
                scattered,
                self.answer_grid.value,
            )
            next_rows = next_reasoning_index(refinement_rows)
            self.reasoning_index.value = jnp.where(
                refinement_latent, next_rows, refinement_rows
            )
        return self.spikes

    def update(self, event: jax.Array, advance: jax.Array | None = None) -> jax.Array:
        """Advance one step and return the bounded BrainTrace training output.

        Parameters
        ----------
        event : jax.Array
            Native batched external input shaped ``(batch, input_width)``.
        advance : jax.Array, optional
            Per-example state-advance gate.  See :meth:`cell_step`.

        Returns
        -------
        jax.Array
            Logits shaped ``(batch, training_output_width)``.
        """
        spikes = self.cell_step(event, advance)
        if self.config.memory_enabled:
            return self.training_readout()
        return self.training_readout(spikes)


def _batched_events(model: LatentWorkspaceModel, events: jax.Array) -> jax.Array:
    events = jnp.asarray(events, dtype=jnp.float32)
    if events.ndim == 2:
        if model.config.batch_size != 1:
            raise ValueError(
                "rank-two events are supported only when configured batch_size is 1"
            )
        events = events[:, None, :]
    expected_tail = (model.config.batch_size, model.config.input_width)
    if events.ndim != 3 or events.shape[1:] != expected_tail:
        raise ValueError(
            f"events must have shape (time, {expected_tail[0]}, "
            f"{expected_tail[1]}), got {events.shape}"
        )
    if events.shape[0] < 1:
        raise ValueError("context events must contain at least one valid row")
    return events


def run_context(
    model: LatentWorkspaceModel,
    events: jax.Array,
    *,
    reset: bool = True,
) -> ContextCheckpoint:
    """Execute valid ARC row events through one BrainState compiled loop.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose physical state is advanced.
    events : jax.Array
        Valid, padding-trimmed events shaped ``(time, input_width)`` for batch
        one or ``(time, batch, input_width)`` generally.
    reset : bool, default=True
        Reset all inference state before the context loop.

    Returns
    -------
    ContextCheckpoint
        Query-terminal state, compact output, and exact restorable snapshot.
    """
    packed = _batched_events(model, events)
    if reset:
        model.reset_state()

    advance = packed[..., model.config.event_valid_index] > 0.5

    def context_step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
        event, gate = inputs
        model.cell_step(event, gate)
        return jnp.asarray(0, dtype=jnp.int8)

    brainstate.transform.for_loop(context_step, (packed, advance))
    spikes = model.spikes
    return ContextCheckpoint(
        compact_logits=model.compact_readout(),
        spikes=spikes,
        voltage=model.voltage,
        feedforward_current=model.feedforward_current,
        recurrent_current=model.recurrent_current,
        snapshot=model.snapshot_state(),
        context_steps=int(packed.shape[0]),
    )


def run_packed_stream(
    model: LatentWorkspaceModel,
    events: jax.Array,
    *,
    reset: bool = True,
    advance_gates: jax.Array | None = None,
    ablation_slots: jax.Array | None = None,
    ablation_gates: jax.Array | None = None,
) -> PackedTrajectory:
    """Run a fixed packed context-plus-latent stream and record every tick.

    This is the integration path for batches whose valid contexts have
    different lengths.  The caller packs each sample's valid context, zero
    tail, and 32 additional zero rows into one static tensor, then gathers
    checkpoint zero and later efforts using that sample's own ``query_stop``.
    The recurrent driver is one :func:`brainstate.transform.for_loop`.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model to reset and execute.
    events : jax.Array
        Fixed tensor shaped ``(time, batch, input_width)``.  Rank two is
        accepted for native batch one.
    reset : bool, default=True
        Reset all inference state before execution.
    advance_gates : jax.Array, optional
        Boolean gates shaped ``(time, batch)``.  Omitted gates are derived from
        the row-event valid channel, which freezes context padding.  A packed
        stream containing zero-input latent rows must supply true gates for
        those rows so recurrent state advances while external input stays
        exactly zero.
    ablation_slots : jax.Array, optional
        One checked slot index per batch element, shaped ``(batch,)``.
    ablation_gates : jax.Array, optional
        Boolean pre-step ablation gates shaped ``(time, batch)``.  A gate at
        ``query_stop`` zeros the selected voltage/spike slice after checkpoint
        zero and before the first latent update.

    Returns
    -------
    PackedTrajectory
        Compact outputs, spikes, and voltage after every input tick.
    """
    packed = _batched_events(model, events)
    if reset:
        model.reset_state()
    if advance_gates is None:
        advance = packed[..., model.config.event_valid_index] > 0.5
    else:
        advance = jnp.asarray(advance_gates, dtype=jnp.bool_)
        if advance.shape != (packed.shape[0], model.config.batch_size):
            raise ValueError(
                "advance_gates must have shape "
                f"({packed.shape[0]}, {model.config.batch_size})"
            )
    controlled = ablation_slots is not None or ablation_gates is not None
    if controlled:
        if ablation_slots is None or ablation_gates is None:
            raise ValueError(
                "ablation_slots and ablation_gates must be supplied together"
            )
        raw_slots = np.asarray(ablation_slots)
        if raw_slots.shape != (model.config.batch_size,):
            raise ValueError(
                f"ablation_slots must have shape ({model.config.batch_size},)"
            )
        if not np.issubdtype(raw_slots.dtype, np.integer):
            raise ValueError("ablation_slots must contain integers")
        if np.any(raw_slots < 0) or np.any(raw_slots >= model.slot_count):
            raise ValueError(f"ablation_slots must lie in [0, {model.slot_count})")
        slots = jnp.asarray(raw_slots, dtype=jnp.int32)
        gates = jnp.asarray(ablation_gates, dtype=jnp.bool_)
        if gates.shape != (packed.shape[0], model.config.batch_size):
            raise ValueError(
                "ablation_gates must have shape "
                f"({packed.shape[0]}, {model.config.batch_size})"
            )

        def controlled_step(
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ]:
            event, advance_gate, ablation_gate = inputs
            model.mask_slots(slots, ablation_gate)
            spikes = model.cell_step(event, advance_gate)
            memory_read = (
                jnp.asarray(model.memory_read.value)
                if model.config.memory_enabled
                else jnp.zeros((model.config.batch_size, 0), dtype=jnp.float32)
            )
            return (
                model.compact_readout(),
                spikes,
                model.voltage,
                model.feedforward_current,
                model.recurrent_current,
                memory_read,
            )

        (
            compact,
            spikes,
            voltage,
            feedforward_current,
            recurrent_current,
            memory_read,
        ) = brainstate.transform.for_loop(controlled_step, (packed, advance, gates))
    else:

        def packed_step(
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ]:
            event, gate = inputs
            spikes = model.cell_step(event, gate)
            memory_read = (
                jnp.asarray(model.memory_read.value)
                if model.config.memory_enabled
                else jnp.zeros((model.config.batch_size, 0), dtype=jnp.float32)
            )
            return (
                model.compact_readout(),
                spikes,
                model.voltage,
                model.feedforward_current,
                model.recurrent_current,
                memory_read,
            )

        (
            compact,
            spikes,
            voltage,
            feedforward_current,
            recurrent_current,
            memory_read,
        ) = brainstate.transform.for_loop(packed_step, (packed, advance))
    if model.config.memory_enabled:
        final_context_memory = jnp.asarray(model.context_memory.value)
    else:
        final_context_memory = jnp.zeros(
            (model.config.batch_size, 0, 0), dtype=jnp.float32
        )
    return PackedTrajectory(
        compact_logits=compact,
        spikes=spikes,
        voltage=voltage,
        feedforward_current=feedforward_current,
        recurrent_current=recurrent_current,
        workspace_carrier=voltage,
        memory_read=memory_read,
        final_context_memory=final_context_memory,
        color_rank=model.config.color_rank,
        decoder_mode=model.config.decoder_mode,
    )


def run_selected_packed_stream(
    model: LatentWorkspaceModel,
    events: jax.Array,
    selected_indices: jax.Array,
    *,
    reset: bool = True,
    advance_gates: jax.Array | None = None,
    ablation_slots: jax.Array | None = None,
    ablation_gates: jax.Array | None = None,
) -> SelectedPackedTrajectory:
    """Run a packed stream while retaining only selected per-example ticks.

    A BrainState scan carries fixed ``(checkpoints, batch, ...)`` buffers and a
    cursor for each batch element.  Every model step still executes in order,
    including advance and ablation controls, but scan outputs only one byte per
    tick instead of stacking neuron-sized states.  Evaluation memory therefore
    scales with the number of requested checkpoints rather than packed-stream
    length.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model to reset and execute.
    events : jax.Array
        Fixed stream shaped ``(time, batch, input_width)``.  Rank two is
        accepted for native batch one.
    selected_indices : jax.Array
        Integer stream indices shaped ``(checkpoints, batch)``.  Each batch
        column must be strictly increasing and lie in ``[0, time)``.  For the
        full experiment these are ``query_stop - 1 + arange(33)``.
    reset : bool, default=True
        Reset inference state before scanning.
    advance_gates : jax.Array, optional
        Boolean state-advance gates shaped ``(time, batch)``.  Omitted gates
        are derived from the event-valid channel.
    ablation_slots : jax.Array, optional
        One integer 64-neuron slot per batch element.
    ablation_gates : jax.Array, optional
        Boolean pre-step ablation gates shaped ``(time, batch)``.  Both
        ablation arguments must be supplied together.

    Returns
    -------
    SelectedPackedTrajectory
        Only the requested compact outputs and physical states.
    """
    packed = _batched_events(model, events)
    raw_indices = np.asarray(selected_indices)
    if raw_indices.ndim != 2 or raw_indices.shape[1] != model.config.batch_size:
        raise ValueError(
            f"selected_indices must have shape (checkpoints, {model.config.batch_size})"
        )
    if raw_indices.shape[0] < 1:
        raise ValueError("selected_indices must contain at least one checkpoint")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("selected_indices must contain integers")
    if np.any(raw_indices < 0) or np.any(raw_indices >= packed.shape[0]):
        raise ValueError(f"selected_indices must lie in [0, {packed.shape[0]})")
    if np.any(np.diff(raw_indices.astype(np.int64), axis=0) <= 0):
        raise ValueError(
            "selected_indices must be strictly increasing in each batch column"
        )
    indices = jnp.asarray(raw_indices, dtype=jnp.int32)
    checkpoint_count = int(raw_indices.shape[0])
    batch_size = model.config.batch_size

    if advance_gates is None:
        advance = packed[..., model.config.event_valid_index] > 0.5
    else:
        advance = jnp.asarray(advance_gates, dtype=jnp.bool_)
        if advance.shape != (packed.shape[0], batch_size):
            raise ValueError(
                f"advance_gates must have shape ({packed.shape[0]}, {batch_size})"
            )

    controlled = ablation_slots is not None or ablation_gates is not None
    if controlled:
        if ablation_slots is None or ablation_gates is None:
            raise ValueError(
                "ablation_slots and ablation_gates must be supplied together"
            )
        raw_slots = np.asarray(ablation_slots)
        if raw_slots.shape != (batch_size,):
            raise ValueError(f"ablation_slots must have shape ({batch_size},)")
        if not np.issubdtype(raw_slots.dtype, np.integer):
            raise ValueError("ablation_slots must contain integers")
        if np.any(raw_slots < 0) or np.any(raw_slots >= model.slot_count):
            raise ValueError(f"ablation_slots must lie in [0, {model.slot_count})")
        slots = jnp.asarray(raw_slots, dtype=jnp.int32)
        ablations = jnp.asarray(ablation_gates, dtype=jnp.bool_)
        if ablations.shape != (packed.shape[0], batch_size):
            raise ValueError(
                f"ablation_gates must have shape ({packed.shape[0]}, {batch_size})"
            )
    else:
        slots = jnp.zeros((batch_size,), dtype=jnp.int32)
        ablations = jnp.zeros((packed.shape[0], batch_size), dtype=jnp.bool_)

    if reset:
        model.reset_state()
    compact_buffer = jnp.zeros(
        (checkpoint_count, batch_size, model.config.checkpoint_output_width),
        dtype=jnp.float32,
    )
    state_shape = (checkpoint_count, batch_size, model.config.neuron_count)
    spikes_buffer = jnp.zeros(state_shape, dtype=jnp.float32)
    voltage_buffer = jnp.zeros(state_shape, dtype=model.voltage.dtype)
    feedforward_buffer = jnp.zeros(state_shape, dtype=model.feedforward_current.dtype)
    recurrent_buffer = jnp.zeros(state_shape, dtype=model.recurrent_current.dtype)
    memory_buffer = jnp.zeros(
        (
            checkpoint_count,
            batch_size,
            model.config.context_memory_width,
        ),
        dtype=jnp.float32,
    )
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)

    initial_carry = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        compact_buffer,
        spikes_buffer,
        voltage_buffer,
        feedforward_buffer,
        recurrent_buffer,
        memory_buffer,
    )

    def scan_step(
        carry: tuple[jax.Array, ...],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        (
            time_index,
            cursors,
            compact_values,
            spike_values,
            voltage_values,
            feedforward_values,
            recurrent_values,
            memory_values,
        ) = carry
        event, advance_gate, ablation_gate = inputs
        model.mask_slots(slots, ablation_gate)
        spikes = model.cell_step(event, advance_gate)
        voltage = model.voltage
        feedforward = model.feedforward_current
        recurrent = model.recurrent_current
        if model.config.memory_enabled:
            memory_read = jnp.asarray(model.memory_read.value)
        else:
            memory_read = jnp.zeros((batch_size, 0), dtype=jnp.float32)

        safe_cursors = jnp.minimum(cursors, checkpoint_count - 1)
        targets = indices[safe_cursors, batch_indices]
        matched = (cursors < checkpoint_count) & (time_index == targets)

        def record(buffer: jax.Array, value: jax.Array) -> jax.Array:
            previous = buffer[safe_cursors, batch_indices]
            selected = jnp.where(matched[:, None], value, previous)
            return buffer.at[safe_cursors, batch_indices].set(selected)

        def record_checkpoint(buffer: jax.Array) -> jax.Array:
            return record(buffer, model.compact_readout())

        compact_values = jax.lax.cond(
            jnp.any(matched), record_checkpoint, lambda buffer: buffer, compact_values
        )
        spike_values = record(spike_values, spikes)
        voltage_values = record(voltage_values, voltage)
        feedforward_values = record(feedforward_values, feedforward)
        recurrent_values = record(recurrent_values, recurrent)
        memory_values = record(memory_values, memory_read)
        next_carry = (
            time_index + 1,
            cursors + matched.astype(jnp.int32),
            compact_values,
            spike_values,
            voltage_values,
            feedforward_values,
            recurrent_values,
            memory_values,
        )
        return next_carry, jnp.asarray(0, dtype=jnp.int8)

    final_carry, _ = brainstate.transform.scan(
        scan_step, initial_carry, (packed, advance, ablations)
    )
    (
        _,
        _,
        compact_buffer,
        spikes_buffer,
        voltage_buffer,
        feedforward_buffer,
        recurrent_buffer,
        memory_buffer,
    ) = final_carry
    if model.config.memory_enabled:
        final_context_memory = jnp.asarray(model.context_memory.value)
    else:
        final_context_memory = jnp.zeros((batch_size, 0, 0), dtype=jnp.float32)
    return SelectedPackedTrajectory(
        selected_indices=indices,
        compact_logits=compact_buffer,
        spikes=spikes_buffer,
        voltage=voltage_buffer,
        feedforward_current=feedforward_buffer,
        recurrent_current=recurrent_buffer,
        workspace_carrier=voltage_buffer,
        memory_read=memory_buffer,
        final_context_memory=final_context_memory,
        color_rank=model.config.color_rank,
        decoder_mode=model.config.decoder_mode,
    )


def run_latent_trajectory(
    model: LatentWorkspaceModel,
    *,
    steps: int | None = None,
) -> ModelTrajectory:
    """Continue current state through an exactly-zero compiled latent rollout.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Context-conditioned model.  This function does not reset it.
    steps : int, optional
        Number of recurrent updates.  The configured maximum is used when
        omitted.

    Returns
    -------
    ModelTrajectory
        Checkpoint zero followed by every latent update.
    """
    if steps is None:
        steps = model.config.max_latent_steps
    else:
        steps = _nonnegative_integer(steps, "steps")
    if steps > model.config.max_latent_steps:
        raise ValueError(
            f"steps {steps} exceeds configured maximum {model.config.max_latent_steps}"
        )

    initial_spikes = model.spikes
    initial_compact = model.compact_readout()
    initial_voltage = model.voltage
    initial_feedforward_current = model.feedforward_current
    initial_recurrent_current = model.recurrent_current
    zero_inputs = jnp.zeros(
        (steps, model.config.batch_size, model.config.input_width),
        dtype=jnp.float32,
    )

    if steps:
        advance = jnp.ones((steps, model.config.batch_size), dtype=jnp.bool_)

        def latent_step(
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            event, gate = inputs
            spikes = model.cell_step(event, gate)
            return (
                model.compact_readout(),
                spikes,
                model.voltage,
                model.feedforward_current,
                model.recurrent_current,
            )

        compact, spikes, voltage, feedforward_current, recurrent_current = (
            brainstate.transform.for_loop(latent_step, (zero_inputs, advance))
        )
    else:
        compact = jnp.zeros(
            (0, model.config.batch_size, model.config.compact_output_width),
            dtype=initial_compact.dtype,
        )
        spikes = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_spikes.dtype,
        )
        voltage = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_voltage.dtype,
        )
        feedforward_current = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_feedforward_current.dtype,
        )
        recurrent_current = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_recurrent_current.dtype,
        )
    return ModelTrajectory(
        compact_logits=jnp.concatenate((initial_compact[None], compact), axis=0),
        spikes=jnp.concatenate((initial_spikes[None], spikes), axis=0),
        voltage=jnp.concatenate((initial_voltage[None], voltage), axis=0),
        feedforward_current=jnp.concatenate(
            (initial_feedforward_current[None], feedforward_current), axis=0
        ),
        recurrent_current=jnp.concatenate(
            (initial_recurrent_current[None], recurrent_current), axis=0
        ),
        zero_inputs=zero_inputs,
        color_rank=model.config.color_rank,
        decoder_mode=model.config.decoder_mode,
    )


def run_sequence(
    model: LatentWorkspaceModel,
    events: jax.Array,
    *,
    latent_steps: int | None = None,
) -> SequenceResult:
    """Reset, execute ARC context, and collect one continuous latent path.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model to execute.
    events : jax.Array
        Valid ARC row events accepted by :func:`run_context`.
    latent_steps : int, optional
        Recurrent effort after checkpoint zero.

    Returns
    -------
    SequenceResult
        Query-terminal checkpoint and its zero-input latent trajectory.
    """
    context = run_context(model, events, reset=True)
    trajectory = run_latent_trajectory(model, steps=latent_steps)
    return SequenceResult(context=context, trajectory=trajectory)


def compile_pp_prop(
    model: LatentWorkspaceModel,
    *,
    verbose: int = 0,
) -> Any:
    """Compile the model with its configured eligibility coordinate.

    The name is historical: the engine is selected by
    ``model.config.trace_engine`` (``"pp_prop"`` or ``"d_rtrl"``), and the
    coordinate comes from :meth:`LatentWorkspaceModel.etrace_config`.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose parameter objects are shared across all effort lengths.
    verbose : int, default=0
        BrainTrace compiler verbosity.

    Returns
    -------
    object
        BrainTrace learner exposing ``etrace_grad`` and ``etrace_evolve``.
    """
    sample = jnp.zeros(
        (model.config.batch_size, model.config.input_width), dtype=jnp.float32
    )
    sample_advance = jnp.ones((model.config.batch_size,), dtype=jnp.bool_)
    return braintrace.compile(
        model,
        model.etrace_config(),
        sample,
        sample_advance,
        batch_size=model.config.batch_size,
        vmap=False,
        verbose=verbose,
    )


def parameter_snapshot(model: LatentWorkspaceModel) -> dict[str, Any]:
    """Copy every trainable parameter for immutability checks.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose parameters are copied.

    Returns
    -------
    dict of str to object
        Path-keyed copied parameter pytrees.  Sparse weights remain dictionaries
        containing their value arrays.
    """
    return {
        ".".join(map(str, path)): _copy_tree(state.value)
        for path, state in model.states(brainstate.ParamState).items()
    }
