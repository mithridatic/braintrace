"""Continuous spatial recurrent model for checkpoint-owned ARC generation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace

from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
    online_input_width,
)
from examples.pp_prop.latent_workspace_spatial_model import (
    BASE_INPUT_WIDTH,
    DECODE_ROW_CHANNEL as DECODE_ROW_CHANNEL,
    INPUT_COLOR_CHANNEL,
    INPUT_MASK_CHANNEL,
    MODEL_INPUT_WIDTH as LEGACY_SPATIAL_INPUT_WIDTH,
    SPATIAL_INPUT_CHANNELS,
    spatial_event_features,
)
from examples.pp_prop.latent_workspace_task import RowEventConfig

ARCHITECTURE_VERSION = "continuous_spatial_tanh_v41"
ANSWER_HEAD_VERSION = "continuous_spatial_row_decoder_v41"
PROPOSAL_SOURCE = "continuous_spatial_model_logits"
ROW_CONFIG = RowEventConfig(max_demonstrations=10, max_grid_size=30)
MODEL_INPUT_WIDTH = online_input_width(
    ROW_CONFIG.max_demonstrations, ROW_CONFIG.max_grid_size
)


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


def _retention(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("retention must be a finite real in [0, 1).")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number >= 1.0:
        raise ValueError("retention must be a finite real in [0, 1).")
    return number


@dataclass(frozen=True)
class ContinuousSpatialConfig:
    """Configure the V41 continuous spatial recurrent model.

    Parameters
    ----------
    input_width : int
        Exact target-free V39 online-event width.
    spatial_channels : int, default=32
        Positive number of recurrent feature maps.
    retention : float, default=0.8
        Fixed prior-state coefficient in ``[0, 1)``.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    architecture_version : str, default="continuous_spatial_tanh_v41"
        Exact checkpoint-schema identifier.
    """

    input_width: int = MODEL_INPUT_WIDTH
    spatial_channels: int = 32
    retention: float = 0.8
    seed: int = 2108
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        width = _positive_integer(self.input_width, "input_width")
        if width != MODEL_INPUT_WIDTH:
            raise ValueError(f"input_width must be {MODEL_INPUT_WIDTH}.")
        object.__setattr__(self, "input_width", width)
        object.__setattr__(
            self,
            "spatial_channels",
            _positive_integer(self.spatial_channels, "spatial_channels"),
        )
        object.__setattr__(self, "retention", _retention(self.retention))
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )


def validate_continuous_spatial_event(event: np.ndarray) -> None:
    """Fail closed on a malformed host-side V41 event.

    Parameters
    ----------
    event : numpy.ndarray
        Event whose final dimension must match :data:`MODEL_INPUT_WIDTH`.
    """

    values = np.asarray(event)
    if values.shape[-1] != MODEL_INPUT_WIDTH:
        raise ValueError(f"event last dimension must be {MODEL_INPUT_WIDTH}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("event values must be finite.")


def continuous_spatial_event_features(event: jnp.ndarray) -> jnp.ndarray:
    """Map a lossless online event onto the V41 spatial feature canvas.

    Parameters
    ----------
    event : jax.Array
        Batched or unbatched V39 online event.

    Returns
    -------
    jax.Array
        Target-free features shaped ``(..., 30, 30, channels)``.
    """

    if event.shape[-1] != MODEL_INPUT_WIDTH:
        raise ValueError(f"event last dimension must be {MODEL_INPUT_WIDTH}.")
    legacy = event[..., :LEGACY_SPATIAL_INPUT_WIDTH]
    return spatial_event_features(legacy)


class ContinuousSpatialARC(brainstate.nn.Module):
    """Maintain a continuous convolutional canvas and emit ARC logits.

    Parameters
    ----------
    config : ContinuousSpatialConfig
        Bound spatial width, retention, event schema, and seed.

    Notes
    -----
    All emitted colour and shape logits pass through checkpoint-owned
    BrainTrace operators. The model has no target input, rule path, retrieval
    store, task-local parameter update, or raw-grid output bypass.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = PROPOSAL_SOURCE

    def __init__(self, config: ContinuousSpatialConfig):
        super().__init__()
        if not isinstance(config, ContinuousSpatialConfig):
            raise TypeError("config must be a ContinuousSpatialConfig instance.")
        self.config = config
        channels = config.spatial_channels
        with brainstate.random.seed_context(config.seed):
            self.input_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, SPATIAL_INPUT_CHANNELS),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
            )
            self.recurrent_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
            )
            self.color_head = braintrace.nn.Conv2d(
                in_size=(
                    MAX_GRID_SIZE,
                    MAX_GRID_SIZE,
                    channels + COLOR_COUNT + 1,
                ),
                out_channels=COLOR_COUNT,
                kernel_size=1,
                padding="SAME",
            )
            shape_width = channels + 2 * MAX_GRID_SIZE
            self.height_head = braintrace.nn.Linear(shape_width, MAX_GRID_SIZE)
            self.width_head = braintrace.nn.Linear(shape_width, MAX_GRID_SIZE)

    def init_state(self, batch_size: int | None = None, **kwargs: object) -> None:
        """Initialize the continuous recurrent canvas to exact zero.

        Parameters
        ----------
        batch_size : int or None, default=None
            Optional leading batch dimension.
        **kwargs : object
            Unused options accepted by BrainState state initialization.
        """

        del kwargs
        prefix = () if batch_size is None else (batch_size,)
        self.canvas = brainstate.HiddenState(
            jnp.zeros(
                (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.spatial_channels),
                dtype=jnp.float32,
            )
        )

    def reset_state(self, batch_size: int | None = None, **kwargs: object) -> None:
        """Reset the continuous recurrent canvas to exact zero.

        Parameters
        ----------
        batch_size : int or None, default=None
            Optional leading batch dimension.
        **kwargs : object
            Unused options accepted by BrainState state reset.
        """

        del kwargs
        prefix = () if batch_size is None else (batch_size,)
        self.canvas.value = jnp.zeros(
            (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.spatial_channels),
            dtype=jnp.float32,
        )

    def _grid_logits(self, state: jnp.ndarray, event: jnp.ndarray) -> jnp.ndarray:
        features = continuous_spatial_event_features(event)
        readout = jnp.concatenate(
            (
                state,
                features[..., INPUT_COLOR_CHANNEL],
                features[..., INPUT_MASK_CHANNEL : INPUT_MASK_CHANNEL + 1],
            ),
            axis=-1,
        )
        return self.color_head(readout)

    def _shape_logits(
        self, state: jnp.ndarray, event: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pooled = jnp.mean(state, axis=(-3, -2))
        dimensions = jnp.concatenate(
            (
                event[..., ROW_CONFIG.input_height_slice],
                event[..., ROW_CONFIG.input_width_slice],
            ),
            axis=-1,
        )
        features = jnp.concatenate((pooled, dimensions), axis=-1)
        return self.height_head(features), self.width_head(features)

    def update(self, event: jnp.ndarray) -> jnp.ndarray:
        """Advance one event and return row-colour and shape logits.

        Parameters
        ----------
        event : jax.Array
            Batched or unbatched target-free online event.

        Returns
        -------
        jax.Array
            Fixed 360-logit row-colour, height, and width layout.
        """

        if event.shape[-1] != self.config.input_width:
            raise ValueError(
                f"event last dimension must be {self.config.input_width}."
            )
        features = continuous_spatial_event_features(event)
        prior = self.canvas.value
        proposal = jnp.tanh(
            self.input_conv(features) + self.recurrent_conv(prior)
        )
        state = (
            self.config.retention * prior
            + (1.0 - self.config.retention) * proposal
        )
        self.canvas.value = state
        grid_logits = self._grid_logits(state, event)
        height_logits, width_logits = self._shape_logits(state, event)
        decode_rows = event[
            ..., BASE_INPUT_WIDTH + 1 : BASE_INPUT_WIDTH + 1 + MAX_GRID_SIZE
        ]
        row_logits = jnp.sum(
            grid_logits * decode_rows[..., :, None, None], axis=-3
        )
        return jnp.concatenate(
            (
                row_logits.reshape(*event.shape[:-1], MAX_GRID_SIZE * COLOR_COUNT),
                height_logits,
                width_logits,
            ),
            axis=-1,
        )


assert LEGACY_SPATIAL_INPUT_WIDTH == BASE_INPUT_WIDTH + 1 + MAX_GRID_SIZE
assert OUTPUT_WIDTH == MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE
