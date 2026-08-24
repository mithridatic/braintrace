"""Task-gated recurrent neural operator bank for direct ARC generation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace

from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
    OnlineARCVanillaRNN,
    online_input_width,
)

ARCHITECTURE_VERSION = "task_gated_operator_bank_v42"
ANSWER_HEAD_VERSION = "task_gated_operator_bank_v42"
PROPOSAL_SOURCE = "task_gated_model_logits"
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
class ExpertModelConfig:
    """Configure the V42 task-gated recurrent operator bank.

    Parameters
    ----------
    input_width : int
        Exact target-free online event width.
    encoder_width : int, default=128
        Width of the first recurrent layer.
    hidden_width : int, default=256
        Width of later recurrent layers and task state.
    expert_count : int, default=12
        Number of parallel checkpoint-owned colour experts, at least two.
    recurrent_layers : int, default=2
        Number of direct-state recurrent layers, at least two.
    max_demonstrations : int, default=10
        Demonstration capacity bound into the event layout.
    max_grid_size : int, default=30
        Grid capacity bound into the event layout.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    architecture_version : str, default="task_gated_operator_bank_v42"
        Exact checkpoint-schema identifier.
    """

    input_width: int
    encoder_width: int = 128
    hidden_width: int = 256
    expert_count: int = 12
    recurrent_layers: int = 2
    max_demonstrations: int = 10
    max_grid_size: int = MAX_GRID_SIZE
    seed: int = 2108
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "encoder_width",
            "hidden_width",
            "expert_count",
            "max_demonstrations",
            "max_grid_size",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        layers = _positive_integer(self.recurrent_layers, "recurrent_layers")
        if layers < 2:
            raise ValueError("recurrent_layers must be at least two.")
        object.__setattr__(self, "recurrent_layers", layers)
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


class TaskGatedOnlineRNN(OnlineARCVanillaRNN):
    """Mix parallel neural colour experts from recurrent task activations.

    Parameters
    ----------
    config : ExpertModelConfig
        Bound recurrent widths, expert count, event schema, and seed.

    Notes
    -----
    Expert mixing occurs inside the executed neural answer path. No expert is
    assigned a rule, decoded separately, selected externally, or reranked.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = PROPOSAL_SOURCE

    def __init__(self, config: ExpertModelConfig):
        brainstate.nn.Module.__init__(self)
        if not isinstance(config, ExpertModelConfig):
            raise TypeError("config must be an ExpertModelConfig instance.")
        self.config = config
        with brainstate.random.seed_context(config.seed):
            layers = [
                braintrace.nn.ValinaRNNCell(
                    config.input_width, config.encoder_width, activation="tanh"
                ),
                braintrace.nn.ValinaRNNCell(
                    config.encoder_width, config.hidden_width, activation="tanh"
                ),
            ]
            for _ in range(config.recurrent_layers - 2):
                layers.append(
                    braintrace.nn.ValinaRNNCell(
                        config.hidden_width,
                        config.hidden_width,
                        activation="tanh",
                    )
                )
            self.recurrent = tuple(layers)
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

    def _expert_weights(self, hidden: jnp.ndarray) -> jnp.ndarray:
        return brainstate.nn.softmax(
            hidden[..., : self.config.expert_count], axis=-1
        )

    def _cell_logits(self, hidden: jnp.ndarray, event: jnp.ndarray) -> jnp.ndarray:
        query_colors = event[..., self.config.query_color_slice].reshape(
            *event.shape[:-1], self.config.max_grid_size, COLOR_COUNT
        )
        if self.config.max_grid_size < MAX_GRID_SIZE:
            padding = [(0, 0)] * query_colors.ndim
            padding[-2] = (0, MAX_GRID_SIZE - self.config.max_grid_size)
            query_colors = jnp.pad(query_colors, padding)
        context = jnp.broadcast_to(
            hidden[..., None, :],
            (*hidden.shape[:-1], MAX_GRID_SIZE, hidden.shape[-1]),
        )
        columns = jnp.broadcast_to(
            self.column_features,
            (*hidden.shape[:-1], MAX_GRID_SIZE, MAX_GRID_SIZE),
        )
        interaction = (context[..., :, None] * query_colors[..., None, :]).reshape(
            *hidden.shape[:-1], MAX_GRID_SIZE, hidden.shape[-1] * COLOR_COUNT
        )
        features = jnp.concatenate(
            (context, query_colors, columns, interaction), axis=-1
        )
        experts = self.cell_color_head(features).reshape(
            *hidden.shape[:-1],
            MAX_GRID_SIZE,
            self.config.expert_count,
            COLOR_COUNT,
        )
        weights = self._expert_weights(hidden)[..., None, :, None]
        return jnp.sum(experts * weights, axis=-2)

    def _shape_logits(
        self, hidden: jnp.ndarray, event: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        query_dimensions = jnp.concatenate(
            (
                event[..., self.config.query_height_slice],
                event[..., self.config.query_width_slice],
            ),
            axis=-1,
        )
        if self.config.max_grid_size < MAX_GRID_SIZE:
            height, width = jnp.split(query_dimensions, 2, axis=-1)
            pad_width = MAX_GRID_SIZE - self.config.max_grid_size
            height = jnp.pad(height, [(0, 0)] * (height.ndim - 1) + [(0, pad_width)])
            width = jnp.pad(width, [(0, 0)] * (width.ndim - 1) + [(0, pad_width)])
            query_dimensions = jnp.concatenate((height, width), axis=-1)
        interaction = (hidden[..., :, None] * query_dimensions[..., None, :]).reshape(
            *hidden.shape[:-1], hidden.shape[-1] * 2 * MAX_GRID_SIZE
        )
        features = jnp.concatenate((hidden, query_dimensions, interaction), axis=-1)
        return self.height_head(features), self.width_head(features)

    def update(self, event: jnp.ndarray) -> jnp.ndarray:
        """Advance one event and emit the single mixed neural candidate logits.

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
        hidden = event
        for layer in self.recurrent:
            hidden = layer(hidden)
        row_color = self._cell_logits(hidden, event)
        height, width = self._shape_logits(hidden, event)
        return jnp.concatenate(
            (
                row_color.reshape(*row_color.shape[:-2], MAX_GRID_SIZE * COLOR_COUNT),
                height,
                width,
            ),
            axis=-1,
        )


assert OUTPUT_WIDTH == MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE

