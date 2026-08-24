"""Spatial Conv-LIF recurrent ARC model for V22 online PP-prop."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

import braintrace

from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
)
from examples.pp_prop.latent_workspace_task import RowEventConfig

ARCHITECTURE_VERSION = "spatial_conv_lif_v26"
ANSWER_HEAD_VERSION = "spatial_conv_lif_row_decoder_v22"
ROW_CONFIG = RowEventConfig(max_demonstrations=10, max_grid_size=30)
BASE_INPUT_WIDTH = ROW_CONFIG.input_width
MODEL_INPUT_WIDTH = BASE_INPUT_WIDTH + 1 + MAX_GRID_SIZE
GLOBAL_FEATURE_WIDTH = ROW_CONFIG.input_color_slice.start
INPUT_COLOR_CHANNEL = slice(GLOBAL_FEATURE_WIDTH, GLOBAL_FEATURE_WIDTH + COLOR_COUNT)
OUTPUT_COLOR_CHANNEL = slice(INPUT_COLOR_CHANNEL.stop, INPUT_COLOR_CHANNEL.stop + COLOR_COUNT)
INPUT_MASK_CHANNEL = OUTPUT_COLOR_CHANNEL.stop
OUTPUT_MASK_CHANNEL = INPUT_MASK_CHANNEL + 1
DECODE_ROW_CHANNEL = OUTPUT_MASK_CHANNEL + 1
SPATIAL_INPUT_CHANNELS = DECODE_ROW_CHANNEL + 1


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real.")
    return number


@dataclass(frozen=True)
class SpatialModelConfig:
    """Configure the spatial Conv-LIF recurrent answer path.

    Parameters
    ----------
    input_width : int
        Must equal the fixed 861-channel augmented event width.
    spatial_channels : int, default=16
        Number of LIF feature maps on the 30-by-30 canvas.
    seed : int, default=2108
        BrainState parameter initialization seed.
    membrane_tau_ms : float, default=100
        LIF membrane time constant in milliseconds.
    synapse_tau_ms : float, default=20
        Exponential-synapse time constant in milliseconds.
    threshold_mv : float, default=0.05
        LIF firing threshold in millivolts.
    architecture_version : str, default="spatial_conv_lif_v22"
        Exact checkpoint schema identifier.
    """

    input_width: int = MODEL_INPUT_WIDTH
    spatial_channels: int = 16
    seed: int = 2108
    membrane_tau_ms: float = 100.0
    synapse_tau_ms: float = 20.0
    threshold_mv: float = 0.05
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
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, Integral
        ):
            raise TypeError("seed must be a nonnegative integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be a nonnegative integer.")
        object.__setattr__(self, "seed", int(self.seed))
        for name in ("membrane_tau_ms", "synapse_tau_ms", "threshold_mv"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )


def validate_spatial_event(event: np.ndarray) -> None:
    """Fail closed on a malformed host-side spatial model event.

    Parameters
    ----------
    event : numpy.ndarray
        Event whose last dimension must be 861 with finite values.
    """

    values = np.asarray(event)
    if values.shape[-1] != MODEL_INPUT_WIDTH:
        raise ValueError(f"event last dimension must be {MODEL_INPUT_WIDTH}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("event values must be finite.")


def spatial_event_features(event: jnp.ndarray) -> jnp.ndarray:
    """Map one augmented row event to a fixed spatial feature canvas.

    Parameters
    ----------
    event : jax.Array
        Event shaped ``(..., 861)``.

    Returns
    -------
    jax.Array
        Target-free features shaped ``(..., 30, 30, channels)``.
    """

    if event.shape[-1] != MODEL_INPUT_WIDTH:
        raise ValueError(f"event last dimension must be {MODEL_INPUT_WIDTH}.")
    prefix = event.shape[:-1]
    metadata = event[..., :GLOBAL_FEATURE_WIDTH]
    metadata = jnp.broadcast_to(
        metadata[..., None, None, :],
        (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, GLOBAL_FEATURE_WIDTH),
    )
    row_selector = event[..., ROW_CONFIG.row_index_slice]
    row_selector = row_selector[..., :, None, None]
    input_colors = event[..., ROW_CONFIG.input_color_slice].reshape(
        *prefix, MAX_GRID_SIZE, COLOR_COUNT
    )
    output_colors = event[..., ROW_CONFIG.output_color_slice].reshape(
        *prefix, MAX_GRID_SIZE, COLOR_COUNT
    )
    input_colors = row_selector * input_colors[..., None, :, :]
    output_colors = row_selector * output_colors[..., None, :, :]
    input_mask = event[..., ROW_CONFIG.input_mask_slice]
    output_mask = event[..., ROW_CONFIG.output_mask_slice]
    input_mask = row_selector * input_mask[..., None, :, None]
    output_mask = row_selector * output_mask[..., None, :, None]
    decode_rows = event[..., BASE_INPUT_WIDTH + 1 : MODEL_INPUT_WIDTH]
    decode_plane = jnp.broadcast_to(
        decode_rows[..., :, None, None],
        (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, 1),
    )
    return jnp.concatenate(
        (
            metadata,
            input_colors,
            output_colors,
            input_mask,
            output_mask,
            decode_plane,
        ),
        axis=-1,
    )


class SpatialARCConvLIF(brainstate.nn.Module):
    """Maintain a 30-by-30 Conv-LIF state and decode checkpoint-owned logits.

    Parameters
    ----------
    config : SpatialModelConfig
        Spatial width, neuron dynamics, and seed.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = "spatial_model_logits"

    def __init__(self, config: SpatialModelConfig):
        super().__init__()
        if not isinstance(config, SpatialModelConfig):
            raise TypeError("config must be a SpatialModelConfig instance.")
        self.config = config
        channels = config.spatial_channels
        with brainstate.random.seed_context(config.seed):
            self.input_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, SPATIAL_INPUT_CHANNELS),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
                w_init=braintools.init.KaimingNormal(1.0, unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA),
            )
            self.recurrent_conv = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                out_channels=channels,
                kernel_size=3,
                padding="SAME",
                w_init=braintools.init.KaimingNormal(0.25, unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA),
            )
            self.color_head = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                out_channels=COLOR_COUNT,
                kernel_size=1,
                padding="SAME",
            )
            self.height_head = braintrace.nn.Linear(channels, MAX_GRID_SIZE)
            self.width_head = braintrace.nn.Linear(channels, MAX_GRID_SIZE)

    @staticmethod
    def _spike(scaled_voltage: jnp.ndarray) -> jnp.ndarray:
        """Return hard spikes with a position-preserving sigmoid gradient."""

        soft = jax.nn.sigmoid(4.0 * scaled_voltage)
        hard = (scaled_voltage >= 0.0).astype(soft.dtype)
        return soft + jax.lax.stop_gradient(hard - soft)

    def init_state(self, batch_size: int | None = None, **kwargs: object) -> None:
        """Initialize same-shape spatial LIF and leaky-logit states.

        Parameters
        ----------
        batch_size : int or None, default=None
            Optional leading batch dimension.
        **kwargs : object
            Unused initialization options accepted by BrainState.
        """

        del kwargs
        prefix = () if batch_size is None else (batch_size,)
        channels = self.config.spatial_channels
        spatial_shape = (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, channels)
        color_shape = (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT)
        shape_head_shape = (*prefix, MAX_GRID_SIZE)
        self.synaptic_current = brainstate.HiddenState(
            jnp.zeros(spatial_shape, dtype=jnp.float32) * u.mA
        )
        self.membrane = brainstate.HiddenState(
            jnp.zeros(spatial_shape, dtype=jnp.float32) * u.mV
        )
        self.last_spikes = brainstate.HiddenState(
            jnp.zeros(spatial_shape, dtype=jnp.float32)
        )
        self.color_trace = brainstate.HiddenState(
            jnp.zeros(color_shape, dtype=jnp.float32)
        )
        self.height_trace = brainstate.HiddenState(
            jnp.zeros(shape_head_shape, dtype=jnp.float32)
        )
        self.width_trace = brainstate.HiddenState(
            jnp.zeros(shape_head_shape, dtype=jnp.float32)
        )

    def reset_state(self, batch_size: int | None = None, **kwargs: object) -> None:
        """Reset every recurrent and answer-logit state to exact zero.

        Parameters
        ----------
        batch_size : int or None, default=None
            Optional leading batch dimension.
        **kwargs : object
            Unused reset options accepted by BrainState.
        """

        del kwargs
        prefix = () if batch_size is None else (batch_size,)
        channels = self.config.spatial_channels
        self.synaptic_current.value = (
            jnp.zeros(
                (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                dtype=jnp.float32,
            )
            * u.mA
        )
        self.membrane.value = (
            jnp.zeros(
                (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
                dtype=jnp.float32,
            )
            * u.mV
        )
        self.last_spikes.value = jnp.zeros(
            (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, channels),
            dtype=jnp.float32,
        )
        self.color_trace.value = jnp.zeros(
            (*prefix, MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT),
            dtype=jnp.float32,
        )
        self.height_trace.value = jnp.zeros(
            (*prefix, MAX_GRID_SIZE), dtype=jnp.float32
        )
        self.width_trace.value = jnp.zeros(
            (*prefix, MAX_GRID_SIZE), dtype=jnp.float32
        )

    def update(self, event: jnp.ndarray) -> jnp.ndarray:
        """Advance one event through spatial synapses and LIF neurons.

        Parameters
        ----------
        event : jax.Array
            Batched or unbatched augmented row event.

        Returns
        -------
        jax.Array
            Row-colour then height and width logits in the V20 fixed layout.
        """

        features = spatial_event_features(event)
        threshold = self.config.threshold_mv * u.mV
        prior_spikes = self.last_spikes.value
        current = self.input_conv(features) + self.recurrent_conv(prior_spikes)
        dt = brainstate.environ.get_dt()
        synapse_steps = u.maybe_decimal(
            self.config.synapse_tau_ms * u.ms / dt
        )
        membrane_steps = u.maybe_decimal(
            self.config.membrane_tau_ms * u.ms / dt
        )
        synapse_decay = u.math.exp(-1.0 / synapse_steps)
        membrane_decay = u.math.exp(-1.0 / membrane_steps)
        synaptic_current = synapse_decay * self.synaptic_current.value + current
        membrane = (
            membrane_decay * self.membrane.value
            + (1.0 - membrane_decay) * (1.0 * u.ohm) * synaptic_current
        )
        scaled_voltage = u.get_magnitude((membrane - threshold) / threshold)
        spikes = self._spike(scaled_voltage)
        readout_activity = jax.nn.sigmoid(4.0 * scaled_voltage)
        membrane = membrane * (1.0 - jax.lax.stop_gradient(spikes))
        self.synaptic_current.value = synaptic_current
        self.membrane.value = membrane
        self.last_spikes.value = spikes

        grid_logits = 0.9 * self.color_trace.value + self.color_head(
            readout_activity
        )
        pooled = jnp.mean(readout_activity, axis=(-3, -2))
        height_logits = 0.9 * self.height_trace.value + self.height_head(pooled)
        width_logits = 0.9 * self.width_trace.value + self.width_head(pooled)
        self.color_trace.value = grid_logits
        self.height_trace.value = height_logits
        self.width_trace.value = width_logits
        decode_rows = event[..., BASE_INPUT_WIDTH + 1 : MODEL_INPUT_WIDTH]
        row_logits = jnp.sum(
            grid_logits * decode_rows[..., :, None, None], axis=-3
        )
        return jnp.concatenate(
            (
                row_logits.reshape(*event.shape[:-1], -1),
                height_logits,
                width_logits,
            ),
            axis=-1,
        )


assert OUTPUT_WIDTH == MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE
