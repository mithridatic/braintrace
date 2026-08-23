"""Trainable recurrent ARC encoder with a direct spatial logit decoder."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
QUERY_FEATURE_WIDTH = COLOR_COUNT + 1


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
class DirectModelConfig:
    """Configure the direct recurrent ARC model.

    Parameters
    ----------
    input_width : int
        Width of one lossless ARC row event.
    encoder_width : int, default=256
        Learned event-projection width.
    hidden_width : int, default=512
        Width of each BrainTrace GRU recurrent layer.
    decoder_width : int, default=256
        Width of the coordinate-conditioned cell decoder.
    recurrent_layers : int, default=2
        Number of stacked recurrent layers.
    seed : int, default=2108
        BrainState parameter-initialization seed.
    """

    input_width: int
    encoder_width: int = 256
    hidden_width: int = 512
    decoder_width: int = 256
    recurrent_layers: int = 2
    seed: int = 2108

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "encoder_width",
            "hidden_width",
            "decoder_width",
            "recurrent_layers",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))


def _coordinate_features() -> jnp.ndarray:
    rows = np.repeat(np.eye(MAX_GRID_SIZE, dtype=np.float32), MAX_GRID_SIZE, axis=0)
    columns = np.tile(np.eye(MAX_GRID_SIZE, dtype=np.float32), (MAX_GRID_SIZE, 1))
    return jnp.asarray(np.concatenate((rows, columns), axis=1))


class DirectARCGRU(brainstate.nn.Module):
    """Encode an ARC episode recurrently and emit every answer logit directly.

    Parameters
    ----------
    config : DirectModelConfig
        Recurrent and spatial-decoder dimensions.

    Notes
    -----
    The query feature tensor is a fixed lossless encoding: ten colour channels
    plus one validity channel. All transformations from that tensor and the
    recurrent task state to output shape and cells are checkpoint-owned
    BrainTrace linear or GRU operations.
    """

    def __init__(self, config: DirectModelConfig):
        super().__init__()
        if not isinstance(config, DirectModelConfig):
            raise TypeError("config must be a DirectModelConfig instance.")
        self.config = config
        with brainstate.random.seed_context(config.seed):
            self.event_projection = braintrace.nn.Linear(
                config.input_width, config.encoder_width
            )
            layers = []
            layer_input_width = config.encoder_width
            for _ in range(config.recurrent_layers):
                layers.append(
                    braintrace.nn.GRUCell(layer_input_width, config.hidden_width)
                )
                layer_input_width = config.hidden_width
            self.recurrent = tuple(layers)
            self.height_head = braintrace.nn.Linear(config.hidden_width, MAX_GRID_SIZE)
            self.width_head = braintrace.nn.Linear(config.hidden_width, MAX_GRID_SIZE)
            self.context_projection = braintrace.nn.Linear(
                config.hidden_width, config.decoder_width
            )
            self.query_projection = braintrace.nn.Linear(
                QUERY_FEATURE_WIDTH, config.decoder_width
            )
            self.coordinate_projection = braintrace.nn.Linear(
                2 * MAX_GRID_SIZE, config.decoder_width
            )
            self.color_head = braintrace.nn.Linear(config.decoder_width, COLOR_COUNT)
        self.coordinate_features = _coordinate_features()

    def update(self, event: jnp.ndarray) -> jnp.ndarray:
        """Advance the recurrent encoder by one row event.

        Parameters
        ----------
        event : jax.Array
            Batched row events shaped ``(batch, input_width)``. Channel zero is
            the fixed event-valid flag.

        Returns
        -------
        jax.Array
            Top recurrent hidden state shaped ``(batch, hidden_width)``.
        """

        valid = jnp.asarray(event[..., 0] > 0.5)
        hidden = brainstate.nn.tanh(self.event_projection(event))
        for layer in self.recurrent:
            previous = layer.h.value
            proposed = layer(hidden)
            layer.h.value = jnp.where(valid[..., None], proposed, previous)
            hidden = layer.h.value
        return hidden

    def decode(
        self, hidden: jnp.ndarray, query_features: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Emit direct height, width, and per-cell colour logits.

        Parameters
        ----------
        hidden : jax.Array
            Final recurrent task state shaped ``(batch, hidden_width)``.
        query_features : jax.Array
            Lossless query colours plus validity shaped ``(batch, 30, 30, 11)``.

        Returns
        -------
        tuple of jax.Array
            Height logits, width logits, and colour logits with shapes
            ``(batch, 30)``, ``(batch, 30)``, and ``(batch, 30, 30, 10)``.
        """

        query_features = jnp.asarray(query_features)
        if query_features.ndim != 4 or query_features.shape[1:] != (
            MAX_GRID_SIZE,
            MAX_GRID_SIZE,
            QUERY_FEATURE_WIDTH,
        ):
            raise ValueError("query_features must have shape (batch, 30, 30, 11).")
        if hidden.ndim != 2 or hidden.shape[0] != query_features.shape[0]:
            raise ValueError("hidden and query_features batch dimensions must match.")
        context = self.context_projection(hidden)[:, None, None, :]
        query = self.query_projection(query_features)
        coordinate = self.coordinate_projection(self.coordinate_features)
        coordinate = coordinate.reshape(
            MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width
        )
        cell_state = brainstate.nn.tanh(context + query + coordinate[None])
        return (
            self.height_head(hidden),
            self.width_head(hidden),
            self.color_head(cell_state),
        )

    def run(
        self, events: jnp.ndarray, query_features: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode all events with one compiled loop and decode one grid.

        Parameters
        ----------
        events : jax.Array
            Time-major row events shaped ``(time, batch, input_width)``.
        query_features : jax.Array
            Lossless query colours plus validity shaped ``(batch, 30, 30, 11)``.

        Returns
        -------
        tuple of jax.Array
            Direct height, width, and cell-colour logits.
        """

        events = jnp.asarray(events)
        if events.ndim != 3 or events.shape[-1] != self.config.input_width:
            raise ValueError(
                "events must have shape (time, batch, config.input_width)."
            )
        hidden_sequence = brainstate.transform.for_loop(self.update, events)
        return self.decode(hidden_sequence[-1], query_features)

