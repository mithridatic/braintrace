"""Paired spatial recurrent model for checkpoint-owned ARC generation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import braintrace

from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
)

ARCHITECTURE_VERSION = "paired_spatial_conv_tanh_v46"
ANSWER_HEAD_VERSION = "paired_spatial_grid_decoder_v46"
PROPOSAL_SOURCE = "paired_spatial_model_logits"

INPUT_COLOR_CHANNEL = 0
OUTPUT_COLOR_CHANNEL = 1
INPUT_MASK_CHANNEL = 2
OUTPUT_MASK_CHANNEL = 3
DEMO_PHASE_CHANNEL = 4
QUERY_PHASE_CHANNEL = 5
EVENT_CHANNELS = 6
COORDINATE_CHANNELS = 2
OUTPUT_WIDTH = MAX_GRID_SIZE * MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE


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
    if not np.isfinite(number) or not 0.0 <= number < 1.0:
        raise ValueError("retention must be a finite real in [0, 1).")
    return number


@dataclass(frozen=True)
class PairedSpatialConfig:
    """Configure the V46 paired spatial recurrent model.

    Parameters
    ----------
    spatial_channels : int, default=32
        Positive number of channels in each recurrent canvas.
    retention : float, default=0.8
        Fixed prior-state coefficient in ``[0, 1)``.
    refinement_steps : int, default=8
        Positive number of repeated target-free query events.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    architecture_version : str, default="paired_spatial_conv_tanh_v46"
        Exact checkpoint-schema identifier.
    """

    spatial_channels: int = 32
    retention: float = 0.8
    refinement_steps: int = 8
    seed: int = 2108
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spatial_channels",
            _positive_integer(self.spatial_channels, "spatial_channels"),
        )
        object.__setattr__(self, "retention", _retention(self.retention))
        object.__setattr__(
            self,
            "refinement_steps",
            _positive_integer(self.refinement_steps, "refinement_steps"),
        )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )


def validate_paired_spatial_event(event: np.ndarray) -> None:
    """Fail closed on a malformed host-side V46 event.

    Parameters
    ----------
    event : numpy.ndarray
        Event ending in the exact ``(30, 30, 6)`` V46 layout.
    """

    values = np.asarray(event)
    expected = (MAX_GRID_SIZE, MAX_GRID_SIZE, EVENT_CHANNELS)
    if values.shape[-3:] != expected:
        raise ValueError(f"event shape must end in {expected}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("event values must be finite.")
    for channel, name in (
        (INPUT_COLOR_CHANNEL, "input colour"),
        (OUTPUT_COLOR_CHANNEL, "output colour"),
    ):
        colors = values[..., channel]
        if not np.all(colors == np.floor(colors)) or not np.all(
            (0 <= colors) & (colors < COLOR_COUNT)
        ):
            raise ValueError(f"{name} values must be integers in 0..9.")
    for channel, name in (
        (INPUT_MASK_CHANNEL, "input mask"),
        (OUTPUT_MASK_CHANNEL, "output mask"),
        (DEMO_PHASE_CHANNEL, "demonstration phase"),
        (QUERY_PHASE_CHANNEL, "query phase"),
    ):
        plane = values[..., channel]
        if not np.all((plane == 0.0) | (plane == 1.0)):
            raise ValueError(f"{name} values must be binary.")
    demo = values[..., DEMO_PHASE_CHANNEL]
    query = values[..., QUERY_PHASE_CHANNEL]
    if np.any((demo != 0.0) & (query != 0.0)):
        raise ValueError("demonstration and query phases must be exclusive.")


def _coordinate_planes(prefix: tuple[int, ...]) -> jax.Array:
    coordinate = jnp.linspace(-1.0, 1.0, MAX_GRID_SIZE, dtype=jnp.float32)
    rows = jnp.broadcast_to(
        coordinate[:, None], (MAX_GRID_SIZE, MAX_GRID_SIZE)
    )
    columns = jnp.broadcast_to(
        coordinate[None, :], (MAX_GRID_SIZE, MAX_GRID_SIZE)
    )
    planes = jnp.stack((rows, columns), axis=-1)
    return jnp.broadcast_to(
        planes, (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, COORDINATE_CHANNELS)
    )


def _event_planes(event: jax.Array) -> tuple[jax.Array, ...]:
    input_colors = jax.nn.one_hot(
        event[..., INPUT_COLOR_CHANNEL].astype(jnp.int32),
        COLOR_COUNT,
        dtype=jnp.float32,
    )
    output_colors = jax.nn.one_hot(
        event[..., OUTPUT_COLOR_CHANNEL].astype(jnp.int32),
        COLOR_COUNT,
        dtype=jnp.float32,
    )
    input_mask = event[..., INPUT_MASK_CHANNEL : INPUT_MASK_CHANNEL + 1]
    output_mask = event[..., OUTPUT_MASK_CHANNEL : OUTPUT_MASK_CHANNEL + 1]
    coordinates = _coordinate_planes(event.shape[:-3])
    return input_colors, output_colors, input_mask, output_mask, coordinates


class PairedSpatialARC(brainstate.nn.Module):
    """Maintain paired demonstration/query canvases and emit ARC logits.

    Parameters
    ----------
    config : PairedSpatialConfig
        Bound channel width, retention, refinement count, and seed.

    Notes
    -----
    Every eligible colour and shape logit passes through checkpoint-owned
    BrainTrace operators. The model contains no target input, transform bank,
    retrieval path, task-local update, or raw-query output bypass.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = PROPOSAL_SOURCE

    def __init__(self, config: PairedSpatialConfig):
        super().__init__()
        if not isinstance(config, PairedSpatialConfig):
            raise TypeError("config must be a PairedSpatialConfig instance.")
        self.config = config
        channels = config.spatial_channels
        demo_input_channels = 2 * COLOR_COUNT + 2 + COORDINATE_CHANNELS
        query_input_channels = COLOR_COUNT + 1 + COORDINATE_CHANNELS + channels
        readout_channels = 6 * channels + COLOR_COUNT + 1 + COORDINATE_CHANNELS
        shape_width = 3 * channels + 2 * MAX_GRID_SIZE
        with brainstate.random.seed_context(config.seed):
            self.demo_input_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, demo_input_channels),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
            )
            self.demo_recurrent_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
            )
            self.query_input_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, query_input_channels),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
            )
            self.query_recurrent_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
            )
            self.color_head = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, readout_channels),
                out_channels=COLOR_COUNT,
                kernel_size=1,
                padding="SAME",
            )
            self.height_head = braintrace.nn.Linear(shape_width, MAX_GRID_SIZE)
            self.width_head = braintrace.nn.Linear(shape_width, MAX_GRID_SIZE)

    def init_state(self, batch_size: int | None = None, **kwargs: object) -> None:
        """Initialize both recurrent canvases to exact zero.

        Parameters
        ----------
        batch_size : int or None, default=None
            Optional leading batch dimension.
        **kwargs : object
            Unused options accepted by BrainState initialization.
        """

        del kwargs
        prefix = () if batch_size is None else (batch_size,)
        shape = (
            *prefix,
            MAX_GRID_SIZE,
            MAX_GRID_SIZE,
            self.config.spatial_channels,
        )
        self.demo_canvas = brainstate.HiddenState(jnp.zeros(shape, dtype=jnp.float32))
        self.query_canvas = brainstate.HiddenState(jnp.zeros(shape, dtype=jnp.float32))

    def reset_state(self, batch_size: int | None = None, **kwargs: object) -> None:
        """Reset both recurrent canvases to exact zero.

        Parameters
        ----------
        batch_size : int or None, default=None
            Optional leading batch dimension.
        **kwargs : object
            Unused options accepted by BrainState reset.
        """

        del kwargs
        prefix = () if batch_size is None else (batch_size,)
        shape = (
            *prefix,
            MAX_GRID_SIZE,
            MAX_GRID_SIZE,
            self.config.spatial_channels,
        )
        zeros = jnp.zeros(shape, dtype=jnp.float32)
        self.demo_canvas.value = zeros
        self.query_canvas.value = zeros

    def _demo_features(self, event: jax.Array) -> jax.Array:
        input_colors, output_colors, input_mask, output_mask, coordinates = (
            _event_planes(event)
        )
        return jnp.concatenate(
            (input_colors, output_colors, input_mask, output_mask, coordinates),
            axis=-1,
        )

    def _query_features(self, event: jax.Array, demo: jax.Array) -> jax.Array:
        input_colors, _, input_mask, _, coordinates = _event_planes(event)
        return jnp.concatenate(
            (input_colors, input_mask, coordinates, demo), axis=-1
        )

    def _color_logits(
        self, demo: jax.Array, query: jax.Array, event: jax.Array
    ) -> jax.Array:
        input_colors, _, input_mask, _, coordinates = _event_planes(event)
        local_product = demo * query
        global_demo = jnp.mean(demo, axis=(-3, -2), keepdims=True)
        global_query = jnp.mean(query, axis=(-3, -2), keepdims=True)
        global_product = global_demo * global_query
        canvas_shape = demo.shape
        readout = jnp.concatenate(
            (
                demo,
                query,
                local_product,
                jnp.broadcast_to(global_demo, canvas_shape),
                jnp.broadcast_to(global_query, canvas_shape),
                jnp.broadcast_to(global_product, canvas_shape),
                input_colors,
                input_mask,
                coordinates,
            ),
            axis=-1,
        )
        return self.color_head(readout)

    def _shape_logits(
        self, demo: jax.Array, query: jax.Array, event: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        input_mask = event[..., INPUT_MASK_CHANNEL]
        input_heights = jnp.max(input_mask, axis=-1)
        input_widths = jnp.max(input_mask, axis=-2)
        pooled_demo = jnp.mean(demo, axis=(-3, -2))
        pooled_query = jnp.mean(query, axis=(-3, -2))
        shape_features = jnp.concatenate(
            (
                pooled_demo,
                pooled_query,
                pooled_demo * pooled_query,
                input_heights,
                input_widths,
            ),
            axis=-1,
        )
        return self.height_head(shape_features), self.width_head(shape_features)

    def update(self, event: jax.Array) -> jax.Array:
        """Advance one gated canvas event and return whole-grid logits.

        Parameters
        ----------
        event : jax.Array
            Batched or unbatched target-free event ending in ``(30, 30, 6)``.

        Returns
        -------
        jax.Array
            Flattened 9,000 colour logits followed by 30 height and 30 width
            logits.
        """

        expected = (MAX_GRID_SIZE, MAX_GRID_SIZE, EVENT_CHANNELS)
        if event.shape[-3:] != expected:
            raise ValueError(f"event shape must end in {expected}.")
        demo_prior = self.demo_canvas.value
        query_prior = self.query_canvas.value
        demo_proposal = jnp.tanh(
            self.demo_input_conv(self._demo_features(event))
            + self.demo_recurrent_conv(demo_prior)
        )
        demo_candidate = (
            self.config.retention * demo_prior
            + (1.0 - self.config.retention) * demo_proposal
        )
        demo_phase = event[..., :1, :1, DEMO_PHASE_CHANNEL : DEMO_PHASE_CHANNEL + 1]
        demo = jnp.where(demo_phase > 0.5, demo_candidate, demo_prior)
        query_proposal = jnp.tanh(
            self.query_input_conv(self._query_features(event, demo))
            + self.query_recurrent_conv(query_prior)
        )
        query_candidate = (
            self.config.retention * query_prior
            + (1.0 - self.config.retention) * query_proposal
        )
        query_phase = event[
            ..., :1, :1, QUERY_PHASE_CHANNEL : QUERY_PHASE_CHANNEL + 1
        ]
        query = jnp.where(query_phase > 0.5, query_candidate, query_prior)
        self.demo_canvas.value = demo
        self.query_canvas.value = query
        colors = self._color_logits(demo, query, event)
        height, width = self._shape_logits(demo, query, event)
        return jnp.concatenate(
            (
                colors.reshape(*event.shape[:-3], -1),
                height,
                width,
            ),
            axis=-1,
        )
