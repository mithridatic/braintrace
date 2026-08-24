"""Checkpoint-owned direct-state recurrent model for online ARC learning."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
ROW_COLOR_WIDTH = MAX_GRID_SIZE * COLOR_COUNT
OUTPUT_WIDTH = ROW_COLOR_WIDTH + 2 * MAX_GRID_SIZE
ARCHITECTURE_VERSION = "online_task_conditioned_cell_decoder_v36"


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
class OnlineModelConfig:
    """Configure the hierarchical row-decoded recurrent model.

    Parameters
    ----------
    input_width : int
        Width of one augmented row or decode-instruction event.
    encoder_width : int, default=128
        Width of the trainable event projection.
    hidden_width : int, default=256
        Width of every BrainTrace recurrent layer after the first.
    recurrent_layers : int, default=2
        Number of stacked recurrent layers, at least two. The first has
        ``encoder_width`` units and later layers have ``hidden_width`` units.
    max_demonstrations : int, default=10
        Demonstration capacity bound into the row-event feature layout.
    max_grid_size : int, default=30
        Grid capacity bound into the row-event feature layout.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    architecture_version : str, default="online_task_conditioned_cell_decoder_v36"
        Exact checkpoint-schema identifier.
    """

    input_width: int
    encoder_width: int = 128
    hidden_width: int = 256
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
            "max_demonstrations",
            "max_grid_size",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        layers = _positive_integer(self.recurrent_layers, "recurrent_layers")
        if layers < 2:
            raise ValueError("recurrent_layers must be at least two.")
        object.__setattr__(self, "recurrent_layers", layers)
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        if self.max_grid_size > MAX_GRID_SIZE:
            raise ValueError(f"max_grid_size must be at most {MAX_GRID_SIZE}.")
        expected_input_width = (
            41 + self.max_demonstrations + 27 * self.max_grid_size
        )
        if self.input_width != expected_input_width:
            raise ValueError(
                "input_width must match the bound row-event and decode layout "
                f"({expected_input_width})."
            )
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )

    @property
    def query_height_slice(self) -> slice:
        """Return the replayed query-input-height feature slice."""

        start = 10 + self.max_demonstrations + self.max_grid_size
        return slice(start, start + self.max_grid_size)

    @property
    def query_width_slice(self) -> slice:
        """Return the replayed query-input-width feature slice."""

        start = self.query_height_slice.stop
        return slice(start, start + self.max_grid_size)

    @property
    def query_color_slice(self) -> slice:
        """Return the replayed position-specific query-colour feature slice."""

        start = 10 + self.max_demonstrations + 7 * self.max_grid_size
        return slice(start, start + self.max_grid_size * COLOR_COUNT)

    @property
    def query_mask_slice(self) -> slice:
        """Return the replayed query-input cell-validity feature slice."""

        stop = self.query_color_slice.start
        return slice(stop - self.max_grid_size, stop)


def split_step_logits(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split one-step output into hierarchical cell and shape logits.

    Parameters
    ----------
    values : jax.Array
        Array whose last dimension is :data:`OUTPUT_WIDTH`.

    Returns
    -------
    row_colors, height, width : tuple of jax.Array
        Gate/conditional-colour logits shaped ``(..., 30, 10)`` and two
        shape-logit arrays shaped ``(..., 30)``.
    """

    if values.shape[-1] != OUTPUT_WIDTH:
        raise ValueError(f"values last dimension must be {OUTPUT_WIDTH}.")
    row = values[..., :ROW_COLOR_WIDTH].reshape(
        *values.shape[:-1], MAX_GRID_SIZE, COLOR_COUNT
    )
    height = values[..., ROW_COLOR_WIDTH : ROW_COLOR_WIDTH + MAX_GRID_SIZE]
    width = values[..., ROW_COLOR_WIDTH + MAX_GRID_SIZE :]
    return row, height, width


class OnlineARCVanillaRNN(brainstate.nn.Module):
    """Consume one lossless event through direct-state recurrent layers.

    Parameters
    ----------
    config : OnlineModelConfig
        Static event and recurrent dimensions.

    Notes
    -----
    Every answer contribution passes through checkpoint-owned BrainTrace
    operators. The model has no raw-grid bypass, retrieval store, rule path, or
    task-local fitter.
    """

    def __init__(self, config: OnlineModelConfig):
        super().__init__()
        if not isinstance(config, OnlineModelConfig):
            raise TypeError("config must be an OnlineModelConfig instance.")
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
                        config.hidden_width, config.hidden_width, activation="tanh"
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
                cell_feature_width, COLOR_COUNT
            )
            shape_feature_width = (
                config.hidden_width
                + 2 * MAX_GRID_SIZE
                + config.hidden_width * 2 * MAX_GRID_SIZE
            )
            self.height_head = braintrace.nn.Linear(shape_feature_width, MAX_GRID_SIZE)
            self.width_head = braintrace.nn.Linear(shape_feature_width, MAX_GRID_SIZE)
        self.column_features = jnp.eye(MAX_GRID_SIZE, dtype=jnp.float32)

    def _cell_logits(self, hidden: jnp.ndarray, event: jnp.ndarray) -> jnp.ndarray:
        query_colors = event[..., self.config.query_color_slice].reshape(
            *event.shape[:-1], self.config.max_grid_size, COLOR_COUNT
        )
        if self.config.max_grid_size < MAX_GRID_SIZE:
            padding = [(0, 0)] * query_colors.ndim
            padding[-2] = (0, MAX_GRID_SIZE - self.config.max_grid_size)
            query_colors = jnp.pad(query_colors, padding)
        context = jnp.broadcast_to(
            hidden[..., None, :], (*hidden.shape[:-1], MAX_GRID_SIZE, hidden.shape[-1])
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
        return self.cell_color_head(features)

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
        """Advance one logical event and return fixed-layout logits.

        Parameters
        ----------
        event : jax.Array
            One augmented row event shaped ``(..., input_width)``.

        Returns
        -------
        jax.Array
            Concatenated hierarchical cell, height, and width logits shaped
            ``(..., OUTPUT_WIDTH)``.
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
                row_color.reshape(*row_color.shape[:-2], ROW_COLOR_WIDTH),
                height,
                width,
            ),
            axis=-1,
        )
