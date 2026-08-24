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
ARCHITECTURE_VERSION = "online_query_replay_hierarchical_decoder_v34"


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
    seed : int, default=2108
        BrainState parameter-initialization seed.
    architecture_version : str, default="online_query_replay_hierarchical_decoder_v34"
        Exact checkpoint-schema identifier.
    """

    input_width: int
    encoder_width: int = 128
    hidden_width: int = 256
    recurrent_layers: int = 2
    seed: int = 2108
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "encoder_width",
            "hidden_width",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        layers = _positive_integer(self.recurrent_layers, "recurrent_layers")
        if layers < 2:
            raise ValueError("recurrent_layers must be at least two.")
        object.__setattr__(self, "recurrent_layers", layers)
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )


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
            self.row_color_head = braintrace.nn.Linear(
                config.hidden_width, ROW_COLOR_WIDTH
            )
            self.height_head = braintrace.nn.Linear(
                config.hidden_width, MAX_GRID_SIZE
            )
            self.width_head = braintrace.nn.Linear(config.hidden_width, MAX_GRID_SIZE)

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
        return jnp.concatenate(
            (
                self.row_color_head(hidden),
                self.height_head(hidden),
                self.width_head(hidden),
            ),
            axis=-1,
        )
