"""Phase-separated gated relation memory for direct ARC generation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace

from examples.pp_prop.latent_workspace_expert_model import TaskGatedOnlineRNN
from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
    online_input_width,
)

ARCHITECTURE_VERSION = "phase_separated_gated_memory_v44"
ANSWER_HEAD_VERSION = ARCHITECTURE_VERSION
PROPOSAL_SOURCE = "gated_memory_model_logits"
MODEL_INPUT_WIDTH = online_input_width(10, 30)


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


@dataclass(frozen=True)
class GatedMemoryConfig:
    """Configure the V44 phase-separated gated relation memory.

    Parameters
    ----------
    input_width : int
        Exact target-free online event width.
    memory_width : int, default=128
        Width of each independent MiniLSTM population.
    expert_count : int, default=12
        Number of parallel checkpoint-owned colour experts, at least two.
    max_demonstrations : int, default=10
        Demonstration capacity bound into the event layout.
    max_grid_size : int, default=30
        Grid capacity bound into the event layout.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    architecture_version : str, default="phase_separated_gated_memory_v44"
        Exact checkpoint-schema identifier.
    """

    input_width: int
    memory_width: int = 128
    expert_count: int = 12
    max_demonstrations: int = 10
    max_grid_size: int = MAX_GRID_SIZE
    seed: int = 2108
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "memory_width",
            "expert_count",
            "max_demonstrations",
            "max_grid_size",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        if self.expert_count < 2:
            raise ValueError("expert_count must be at least two.")
        if self.hidden_width < self.expert_count:
            raise ValueError("hidden_width must be at least expert_count.")
        expected = online_input_width(
            self.max_demonstrations, self.max_grid_size
        )
        if self.input_width != expected:
            raise ValueError(f"input_width must match the bound event layout ({expected}).")
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


class PhaseSeparatedGatedMemoryRNN(TaskGatedOnlineRNN):
    """Keep paired demonstrations separate from query/decode recurrence.

    Parameters
    ----------
    config : GatedMemoryConfig
        Bound memory width, expert count, event schema, and seed.

    Notes
    -----
    Fixed phase channels gate state updates. They never choose an answer,
    expert, transformation, or candidate.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = PROPOSAL_SOURCE

    def __init__(self, config: GatedMemoryConfig):
        brainstate.nn.Module.__init__(self)
        if not isinstance(config, GatedMemoryConfig):
            raise TypeError("config must be a GatedMemoryConfig instance.")
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
        self.column_features = jnp.eye(MAX_GRID_SIZE, dtype=jnp.float32)

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

    def update(self, event: jnp.ndarray) -> jnp.ndarray:
        """Advance phase-gated memories and emit direct candidate logits.

        Parameters
        ----------
        event : jax.Array
            Batched or unbatched target-free online event.

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
        row_color = self._cell_logits(hidden, event)
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


assert OUTPUT_WIDTH == MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE
