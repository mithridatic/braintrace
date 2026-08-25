"""Query-routing gated memory (V48) for direct ARC generation.

Adds one mechanism to the frozen V44 architecture: a checkpoint-owned
query-routing head. At every decode step, a fixed lossless block carries the
complete query-input colour one-hots and validity mask. For each output
cell, the BrainTrace ETP ``attention_residual`` primitive mixes the 900
query positions — described by their colours, validity, and 65 fixed
geometric match features (identity, shifts, dihedral, upscale, and
mirror-concatenation maps) — with a learned zero-initialised query table.
A fixed seeded gate projection of the recurrent state and coordinates mixes
16 routing programs, and the routed colour votes join the cell logits at a
fixed scale. The query table is the only new trainable leaf and reaches the
logits through no other trainable primitive, satisfying PP-prop's
weight-to-weight invariant. See
``docs/specs/2026-08-24-example21-query-routing-v48.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import braintrace

from examples.pp_prop.latent_workspace_gated_memory_model import (
    PhaseSeparatedGatedMemoryRNN,
)
from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
    online_input_width,
)
from examples.pp_prop.latent_workspace_task import RowEventConfig

ARCHITECTURE_VERSION = "query_routing_gated_memory_v48"
ANSWER_HEAD_VERSION = "query_routing_hierarchical_decoder_v48"
PROPOSAL_SOURCE = "query_routing_model_logits"
BASE_INPUT_WIDTH = online_input_width(10, 30)
SOURCE_COUNT = MAX_GRID_SIZE * MAX_GRID_SIZE
QUERY_GRID_BLOCK_WIDTH = MAX_GRID_SIZE * MAX_GRID_SIZE * (COLOR_COUNT + 1)
MODEL_INPUT_WIDTH = BASE_INPUT_WIDTH + QUERY_GRID_BLOCK_WIDTH
DECODE_ROW_SLICE = slice(BASE_INPUT_WIDTH - 3000, BASE_INPUT_WIDTH - 3000 + 30)
BLOCK_COLOR_SLICE = slice(BASE_INPUT_WIDTH, BASE_INPUT_WIDTH + 9000)
BLOCK_VALID_SLICE = slice(BASE_INPUT_WIDTH + 9000, MODEL_INPUT_WIDTH)
SHIFT_OFFSETS = tuple((dr, dc) for dr in range(-3, 4) for dc in range(-3, 4))
MATCH_FEATURE_COUNT = 1 + len(SHIFT_OFFSETS) + 8 + 3 + 4
SOURCE_FEATURES = COLOR_COUNT + 1 + MATCH_FEATURE_COUNT
GATE_SEED = 777
_ROW_CONFIG = RowEventConfig()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return integer


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real.")
    return number


def extend_events_query_grid(events: np.ndarray) -> np.ndarray:
    """Append the fixed lossless query-grid block to every decode step.

    Parameters
    ----------
    events : numpy.ndarray
        Encoded events shaped ``(..., time, 3831)`` whose final 30 steps are
        the fixed decode instructions carrying the replayed query rows.

    Returns
    -------
    numpy.ndarray
        Events shaped ``(..., time, 13731)``; the appended block repeats the
        complete query colour one-hots and validity mask at every decode
        step and is exactly zero elsewhere. The block repeats only
        query-input information already present earlier in the episode.
    """

    array = np.asarray(events)
    if array.shape[-1] != BASE_INPUT_WIDTH:
        raise ValueError(f"events last dimension must be {BASE_INPUT_WIDTH}.")
    decode = array[..., -MAX_GRID_SIZE:, :]
    colors = decode[..., _ROW_CONFIG.input_color_slice].reshape(
        *array.shape[:-2], 9000
    )
    validity = decode[..., _ROW_CONFIG.input_mask_slice].reshape(
        *array.shape[:-2], 900
    )
    block = np.concatenate([colors, validity], axis=-1).astype(array.dtype)
    blocks = np.zeros(
        (*array.shape[:-1], QUERY_GRID_BLOCK_WIDTH), dtype=array.dtype
    )
    blocks[..., -MAX_GRID_SIZE:, :] = np.broadcast_to(
        block[..., None, :], (*blocks.shape[:-2], MAX_GRID_SIZE, block.shape[-1])
    )
    return np.concatenate([array, blocks], axis=-1)


@dataclass(frozen=True)
class QueryRoutingConfig:
    """Configure the V48 query-routing gated memory.

    Parameters
    ----------
    input_width : int
        Exact target-free online event width including the query block.
    memory_width : int, default=128
        Width of each independent MiniLSTM population.
    expert_count : int, default=12
        Number of parallel checkpoint-owned colour experts, at least two.
    program_count : int, default=16
        Number of learned routing programs, at least two.
    routing_scale : float, default=4.0
        Fixed scale applied to routed colour votes in the cell logits.
    max_demonstrations : int, default=10
        Demonstration capacity bound into the event layout.
    max_grid_size : int, default=30
        Grid capacity bound into the event layout.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    gate_seed : int, default=777
        Declared fixed seed for the non-trainable gate projection.
    architecture_version : str
        Exact checkpoint-schema identifier.
    """

    input_width: int = MODEL_INPUT_WIDTH
    memory_width: int = 128
    expert_count: int = 12
    program_count: int = 16
    routing_scale: float = 4.0
    max_demonstrations: int = 10
    max_grid_size: int = MAX_GRID_SIZE
    seed: int = 2108
    gate_seed: int = GATE_SEED
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "memory_width",
            "expert_count",
            "program_count",
            "max_demonstrations",
            "max_grid_size",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self, "gate_seed", _nonnegative_integer(self.gate_seed, "gate_seed")
        )
        if self.expert_count < 2:
            raise ValueError("expert_count must be at least two.")
        if self.program_count < 2:
            raise ValueError("program_count must be at least two.")
        if self.hidden_width < self.expert_count:
            raise ValueError("hidden_width must be at least expert_count.")
        if self.input_width != MODEL_INPUT_WIDTH:
            raise ValueError(
                f"input_width must match the bound event layout ({MODEL_INPUT_WIDTH})."
            )
        object.__setattr__(
            self,
            "routing_scale",
            _positive_real(self.routing_scale, "routing_scale"),
        )
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )

    @property
    def hidden_width(self) -> int:
        """Return the demo/query/product relation width."""

        return 3 * self.memory_width

    @property
    def demonstration_phase_index(self) -> int:
        """Return the fixed demonstration-phase channel index."""

        return 1

    @property
    def query_phase_index(self) -> int:
        """Return the fixed query/decode-phase channel index."""

        return 2

    @property
    def query_height_slice(self) -> slice:
        """Return the replayed query-height one-hot slice."""

        start = 10 + self.max_demonstrations + self.max_grid_size
        return slice(start, start + self.max_grid_size)

    @property
    def query_width_slice(self) -> slice:
        """Return the replayed query-width one-hot slice."""

        start = self.query_height_slice.stop
        return slice(start, start + self.max_grid_size)

    @property
    def query_color_slice(self) -> slice:
        """Return the position-specific query-colour slice."""

        start = 10 + self.max_demonstrations + 7 * self.max_grid_size
        return slice(start, start + self.max_grid_size * COLOR_COUNT)

    @property
    def query_mask_slice(self) -> slice:
        """Return the query-input cell-validity slice."""

        stop = self.query_color_slice.start
        return slice(stop - self.max_grid_size, stop)


def _eq(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return (jnp.abs(left - right) < 0.5).astype(jnp.float32)


class QueryRoutingGatedMemoryRNN(PhaseSeparatedGatedMemoryRNN):
    """Route query-input content to output coordinates by learned programs.

    Parameters
    ----------
    config : QueryRoutingConfig
        Bound memory width, routing dimensions, event schema, and seeds.

    Notes
    -----
    Routed values come from the lossless query-input block only. No
    demonstration output, target, rule, template, or task-local fitter
    participates. The gate projection is a declared fixed constant, not a
    trainable leaf.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = PROPOSAL_SOURCE

    def __init__(self, config: QueryRoutingConfig):
        brainstate.nn.Module.__init__(self)
        if not isinstance(config, QueryRoutingConfig):
            raise TypeError("config must be a QueryRoutingConfig instance.")
        self.config = config
        with brainstate.random.seed_context(config.seed):
            self.recurrent = (
                braintrace.nn.MiniLSTM(config.input_width, config.memory_width),
                braintrace.nn.MiniLSTM(config.input_width, config.memory_width),
            )
            cell_feature_width = (
                config.hidden_width
                + COLOR_COUNT
                + MAX_GRID_SIZE
                + config.hidden_width * COLOR_COUNT
            )
            self.cell_color_head = braintrace.nn.Linear(
                cell_feature_width, config.expert_count * COLOR_COUNT
            )
            shape_feature_width = (
                config.hidden_width
                + 2 * MAX_GRID_SIZE
                + config.hidden_width * 2 * MAX_GRID_SIZE
            )
            self.height_head = braintrace.nn.Linear(
                shape_feature_width, MAX_GRID_SIZE
            )
            self.width_head = braintrace.nn.Linear(
                shape_feature_width, MAX_GRID_SIZE
            )
            self.routing_query = braintrace.nn.AttentionResidual(
                SOURCE_FEATURES, config.program_count
            )
        self.column_features = jnp.eye(MAX_GRID_SIZE, dtype=jnp.float32)
        gate_width = config.hidden_width + 2 * MAX_GRID_SIZE
        gate_projection = brainstate.random.RandomState(config.gate_seed).normal(
            size=(config.program_count, gate_width), dtype=np.float32
        ) * (gate_width ** -0.5)
        self.gate_projection = jnp.asarray(gate_projection, dtype=jnp.float32)
        self.source_rows = jnp.repeat(
            jnp.arange(MAX_GRID_SIZE, dtype=jnp.float32), MAX_GRID_SIZE
        )
        self.source_columns = jnp.tile(
            jnp.arange(MAX_GRID_SIZE, dtype=jnp.float32), MAX_GRID_SIZE
        )
        self.output_columns = jnp.arange(MAX_GRID_SIZE, dtype=jnp.float32)
        self.relation_features = jnp.eye(MATCH_FEATURE_COUNT, dtype=jnp.float32)

    def _relation_hidden(
        self, demonstration: jnp.ndarray, query: jnp.ndarray
    ) -> jnp.ndarray:
        if demonstration.shape != query.shape:
            raise ValueError("demonstration and query states must have equal shapes.")
        if demonstration.shape[-1] != self.config.memory_width:
            raise ValueError(
                f"state last dimension must be {self.config.memory_width}."
            )
        return jnp.concatenate(
            (demonstration, query, demonstration * query), axis=-1
        )

    @staticmethod
    def _pair(match_r: jnp.ndarray, match_c: jnp.ndarray) -> jnp.ndarray:
        return match_r[:, None, :] * match_c[None, :, :]

    def _source_match(self, target_r, target_c, batch: int) -> jnp.ndarray:
        """Return exact pair matches broadcast to ``(batch, 30, 900)``."""

        match_r = _eq(self.source_rows, target_r[..., None])
        match_c = _eq(self.source_columns, target_c[..., None])
        if target_r.ndim == 1:
            match_r = (
                match_r[:, None, :]
                if target_r.shape[0] == batch
                else match_r[None, :, :]
            )
        if target_c.ndim == 1:
            match_c = (
                match_c[:, None, :]
                if target_c.shape[0] == batch
                else match_c[None, :, :]
            )
        return match_r * match_c

    def _match_features(
        self,
        row: jnp.ndarray,
        height: jnp.ndarray,
        width: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return exact fixed geometric matches ``(batch, 30, 900, 65)``."""

        out_r = row
        out_c = self.output_columns
        batch = row.shape[0]
        features = [self._source_match(out_r, out_c, batch)]
        for dr, dc in SHIFT_OFFSETS:
            features.append(self._source_match(out_r + dr, out_c + dc, batch))
        width_c = width[..., None] - 1.0 - out_c
        dihedral_targets = (
            (out_r, out_c),
            (out_c, out_r),
            (height - 1.0 - out_r, width_c),
            (width_c, out_r),
            (out_r, width_c),
            (height - 1.0 - out_r, out_c),
            (out_c, height - 1.0 - out_r),
            (width_c, height - 1.0 - out_r),
        )
        for target_r, target_c in dihedral_targets:
            features.append(self._source_match(target_r, target_c, batch))
        for factor in (2.0, 3.0, 4.0):
            in_r = (
                (out_r[..., None] >= self.source_rows * factor)
                & (out_r[..., None] < (self.source_rows + 1.0) * factor)
            ).astype(jnp.float32)
            in_c = (
                (out_c[..., None] >= self.source_columns * factor)
                & (out_c[..., None] < (self.source_columns + 1.0) * factor)
            ).astype(jnp.float32)
            features.append(self._pair(in_r, in_c))
        flip_top_r = jnp.where(out_r < height, height - 1.0 - out_r, out_r - height)
        features.append(self._source_match(flip_top_r, out_c, batch))
        flip_bottom_r = jnp.where(
            out_r < height, out_r, 2.0 * height - 1.0 - out_r
        )
        features.append(self._source_match(flip_bottom_r, out_c, batch))
        flip_left_c = jnp.where(
            out_c < width[..., None], width[..., None] - 1.0 - out_c,
            out_c - width[..., None],
        )
        features.append(self._source_match(out_r, flip_left_c, batch))
        flip_right_c = jnp.where(
            out_c < width[..., None], out_c,
            2.0 * width[..., None] - 1.0 - out_c,
        )
        features.append(self._source_match(out_r, flip_right_c, batch))
        return jnp.stack(features, axis=-1)

    def _routing_logits(
        self, hidden: jnp.ndarray, event: jnp.ndarray
    ) -> jnp.ndarray:
        """Return routed colour votes shaped ``(batch, 30, 10)``."""

        batch = event.shape[0]
        colors = event[..., BLOCK_COLOR_SLICE].reshape(
            batch, SOURCE_COUNT, COLOR_COUNT
        )
        validity = event[..., BLOCK_VALID_SLICE]
        row_onehot = event[..., DECODE_ROW_SLICE]
        positions = jnp.arange(MAX_GRID_SIZE, dtype=jnp.float32)
        row = jnp.sum(row_onehot * positions, axis=-1)
        height = jnp.sum(
            event[..., self.config.query_height_slice] * (positions + 1.0), axis=-1
        )
        width = jnp.sum(
            event[..., self.config.query_width_slice] * (positions + 1.0), axis=-1
        )
        matches = self._match_features(row, height, width)
        gathered_colors = jnp.einsum("bosk,bsc->bokc", matches, colors)
        gathered_valid = jnp.einsum("bosk,bs->bok", matches, validity)
        sources = jnp.concatenate(
            (
                gathered_colors,
                gathered_valid[..., None],
                jnp.broadcast_to(
                    self.relation_features[None, None, ...],
                    (batch, MAX_GRID_SIZE, MATCH_FEATURE_COUNT, MATCH_FEATURE_COUNT),
                ),
            ),
            axis=-1,
        )
        mask = gathered_valid > 0.5
        gate_features = jnp.concatenate(
            (
                jnp.broadcast_to(
                    hidden[:, None, :], (batch, MAX_GRID_SIZE, hidden.shape[-1])
                ),
                jnp.broadcast_to(
                    row_onehot[:, None, :],
                    (batch, MAX_GRID_SIZE, MAX_GRID_SIZE),
                ),
                jnp.broadcast_to(
                    self.column_features[None, ...],
                    (batch, MAX_GRID_SIZE, MAX_GRID_SIZE),
                ),
            ),
            axis=-1,
        )
        gate = brainstate.nn.softmax(
            gate_features @ self.gate_projection.T, axis=-1
        )
        program_outputs = [
            self.routing_query(sources, source_mask=mask, query_index=program)
            for program in range(self.config.program_count)
        ]
        mixed = jnp.sum(
            jnp.stack(program_outputs, axis=-2) * gate[..., None], axis=-2
        )
        return mixed[..., :COLOR_COUNT]

    def update(self, event: jnp.ndarray) -> jnp.ndarray:
        """Advance phase-gated memories and emit direct candidate logits.

        Parameters
        ----------
        event : jax.Array
            Batched target-free online event, including the fixed
            query-grid block at decode steps.

        Returns
        -------
        jax.Array
            Fixed row-colour, height, and width logits.
        """

        if event.shape[-1] != self.config.input_width:
            raise ValueError(
                f"event last dimension must be {self.config.input_width}."
            )
        demonstration_old = self.recurrent[0].h.value
        demonstration_candidate = self.recurrent[0](event)
        demonstration_active = event[
            ...,
            self.config.demonstration_phase_index : (
                self.config.demonstration_phase_index + 1
            ),
        ] > 0.5
        demonstration = jnp.where(
            demonstration_active, demonstration_candidate, demonstration_old
        )
        self.recurrent[0].h.value = demonstration

        query_old = self.recurrent[1].h.value
        query_candidate = self.recurrent[1](event)
        query_active = event[
            ...,
            self.config.query_phase_index : self.config.query_phase_index + 1,
        ] > 0.5
        query = jnp.where(query_active, query_candidate, query_old)
        self.recurrent[1].h.value = query

        hidden = self._relation_hidden(demonstration, query)
        row_color = self._cell_logits(hidden, event) + (
            self.config.routing_scale
            * self._routing_logits(hidden, event)
        )
        height, width = self._shape_logits(hidden, event)
        return jnp.concatenate(
            (
                row_color.reshape(
                    *row_color.shape[:-2], MAX_GRID_SIZE * COLOR_COUNT
                ),
                height,
                width,
            ),
            axis=-1,
        )


assert MATCH_FEATURE_COUNT == 65
assert SOURCE_FEATURES == 76
assert DECODE_ROW_SLICE == slice(831, 861)
assert OUTPUT_WIDTH == MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE
