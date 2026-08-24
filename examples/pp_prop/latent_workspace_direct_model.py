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
MAX_DEMONSTRATIONS = 10
DEMONSTRATION_SHAPE_WIDTH = 1 + 4 * MAX_GRID_SIZE
SHAPE_FEATURE_WIDTH = MAX_DEMONSTRATIONS * DEMONSTRATION_SHAPE_WIDTH + 2 * MAX_GRID_SIZE
ARCHITECTURE_VERSION = "direct_spatial_recurrence_v19"
DIRECT_QUERY_COLOR_INITIAL_SCALE = 4.0
SPATIAL_RECURRENCE_STEPS = 4


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
    memory_key_indices, memory_value_indices : tuple of int, default=()
        Matched input-side key and output-side value columns in each lossless
        row event. Both must be supplied together with equal nonzero length.
    memory_key_color_block_width : int, default=0
        Trailing key-feature width containing groups of ten ARC color one-hots.
        One foreground-occupancy feature per group is appended to the full key.
    architecture_version : str, default="direct_spatial_recurrence_v19"
        Fixed checkpoint-schema architecture identifier.
    """

    input_width: int
    encoder_width: int = 256
    hidden_width: int = 512
    decoder_width: int = 256
    recurrent_layers: int = 2
    seed: int = 2108
    memory_key_indices: tuple[int, ...] = ()
    memory_value_indices: tuple[int, ...] = ()
    memory_key_color_block_width: int = 0
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "encoder_width",
            "hidden_width",
            "decoder_width",
            "recurrent_layers",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        normalized_indices = {}
        for name in ("memory_key_indices", "memory_value_indices"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of integers.")
            indices = tuple(raw)
            if any(
                isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                for index in indices
            ):
                raise TypeError(f"{name} must contain only integers.")
            indices = tuple(int(index) for index in indices)
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} must contain unique indices.")
            if any(index < 0 or index >= self.input_width for index in indices):
                raise ValueError(f"{name} indices must be within input_width.")
            normalized_indices[name] = indices
            object.__setattr__(self, name, indices)
        if len(normalized_indices["memory_key_indices"]) != len(
            normalized_indices["memory_value_indices"]
        ):
            raise ValueError("memory key and value indices must have equal length.")
        color_block_width = _nonnegative_integer(
            self.memory_key_color_block_width, "memory_key_color_block_width"
        )
        if color_block_width > len(self.memory_key_indices):
            raise ValueError(
                "memory_key_color_block_width cannot exceed the key width."
            )
        if color_block_width % COLOR_COUNT:
            raise ValueError(
                "memory_key_color_block_width must contain complete color groups."
            )
        object.__setattr__(self, "memory_key_color_block_width", color_block_width)
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(f"architecture_version must be {ARCHITECTURE_VERSION!r}.")


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
            self.query_shape_projection = braintrace.nn.Linear(
                2 * MAX_GRID_SIZE, config.hidden_width
            )
            self.demo_shape_projection = braintrace.nn.Linear(
                DEMONSTRATION_SHAPE_WIDTH, config.hidden_width
            )
            self.query_dimension_projection = braintrace.nn.Linear(
                2 * MAX_GRID_SIZE, config.hidden_width
            )
            self.temporal_summary_projection = braintrace.nn.Linear(
                2 * config.hidden_width, config.hidden_width
            )
            self.context_projection = braintrace.nn.Linear(
                config.hidden_width, config.decoder_width
            )
            self.query_projection = braintrace.nn.Linear(
                QUERY_FEATURE_WIDTH, config.decoder_width
            )
            self.local_query_projection = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, QUERY_FEATURE_WIDTH),
                out_channels=config.decoder_width,
                kernel_size=3,
                padding="SAME",
            )
            self.local_query_refinement = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, config.decoder_width),
                out_channels=config.decoder_width,
                kernel_size=3,
                padding="SAME",
            )
            self.local_query_refinement.weight.value = {
                name: jnp.zeros_like(value)
                for name, value in self.local_query_refinement.weight.value.items()
            }
            self.global_query_pattern_projection = braintrace.nn.Linear(
                MAX_GRID_SIZE * MAX_GRID_SIZE, config.decoder_width
            )
            self.memory_key_projection = braintrace.nn.Linear(
                config.hidden_width, config.decoder_width
            )
            self.memory_value_projection = braintrace.nn.Linear(
                config.hidden_width, config.decoder_width
            )
            self.direct_query_color_head = braintrace.nn.Linear(
                COLOR_COUNT, COLOR_COUNT
            )
            self.direct_query_color_head.weight.value = {
                "weight": jnp.eye(COLOR_COUNT, dtype=jnp.float32)
                * DIRECT_QUERY_COLOR_INITIAL_SCALE,
                "bias": jnp.zeros((COLOR_COUNT,), dtype=jnp.float32),
            }
            self.demonstration_pair_projection = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, 2 * COLOR_COUNT),
                out_channels=config.decoder_width,
                kernel_size=3,
                padding="SAME",
            )
            self.demonstration_pair_refinement = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, config.decoder_width),
                out_channels=config.decoder_width,
                kernel_size=3,
                padding="SAME",
            )
            self.spatial_query_projection = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, QUERY_FEATURE_WIDTH),
                out_channels=config.decoder_width,
                kernel_size=3,
                padding="SAME",
            )
            self.spatial_context_projection = braintrace.nn.Linear(
                config.hidden_width + config.decoder_width,
                2 * config.decoder_width,
            )
            self.spatial_update = braintrace.nn.Conv2d(
                in_size=(MAX_GRID_SIZE, MAX_GRID_SIZE, config.decoder_width),
                out_channels=config.decoder_width,
                kernel_size=3,
                padding="SAME",
            )
            self.spatial_color_head = braintrace.nn.Linear(
                config.decoder_width, COLOR_COUNT
            )
            self.coordinate_projection = braintrace.nn.Linear(
                2 * MAX_GRID_SIZE, config.decoder_width
            )
            self.attention_key_projection = braintrace.nn.Linear(
                QUERY_FEATURE_WIDTH + 2 * MAX_GRID_SIZE,
                config.decoder_width,
            )
            self.attention_value_projection = braintrace.nn.Linear(
                QUERY_FEATURE_WIDTH + 2 * MAX_GRID_SIZE,
                config.decoder_width,
            )
            self.color_head = braintrace.nn.Linear(config.decoder_width, COLOR_COUNT)
        self.coordinate_features = _coordinate_features()

    def _demonstration_pair_summary(
        self,
        demonstration_inputs: jnp.ndarray,
        demonstration_outputs: jnp.ndarray,
        demonstration_grid_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        """Encode demonstration pairs into one pooled trainable latent state."""

        input_valid = jnp.sum(demonstration_inputs, axis=-1) > 0.5
        output_valid = jnp.sum(demonstration_outputs, axis=-1) > 0.5
        pair_valid = jnp.logical_or(input_valid, output_valid)
        pair_valid = jnp.logical_and(
            pair_valid, demonstration_grid_valid[..., None, None]
        )
        batch_size = demonstration_inputs.shape[0]
        pair_features = jnp.concatenate(
            (demonstration_inputs, demonstration_outputs), axis=-1
        ).reshape(
            batch_size * MAX_DEMONSTRATIONS,
            MAX_GRID_SIZE,
            MAX_GRID_SIZE,
            2 * COLOR_COUNT,
        )
        encoded = brainstate.nn.tanh(
            self.demonstration_pair_projection(pair_features)
        )
        encoded = brainstate.nn.tanh(
            self.demonstration_pair_refinement(encoded)
        ).reshape(
            batch_size,
            MAX_DEMONSTRATIONS,
            MAX_GRID_SIZE,
            MAX_GRID_SIZE,
            self.config.decoder_width,
        )
        weights = pair_valid[..., None].astype(encoded.dtype)
        return jnp.sum(encoded * weights, axis=(1, 2, 3)) / jnp.maximum(
            jnp.sum(weights, axis=(1, 2, 3)), 1.0
        )

    def _spatial_recurrence(
        self,
        hidden: jnp.ndarray,
        query_features: jnp.ndarray,
        demonstration_summary: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evolve a query grid with four checkpoint-owned convolutional steps."""

        context = self.spatial_context_projection(
            jnp.concatenate((hidden, demonstration_summary), axis=-1)
        )
        scale, bias = jnp.split(context, 2, axis=-1)
        scale = brainstate.nn.sigmoid(scale)[:, None, None, :]
        bias = 0.25 * brainstate.nn.tanh(bias)[:, None, None, :]
        initial = brainstate.nn.tanh(
            self.spatial_query_projection(query_features)
        )

        def step(state: jnp.ndarray, _: jnp.ndarray):
            update = brainstate.nn.tanh(
                self.spatial_update(state * (1.0 + scale) + bias)
            )
            next_state = brainstate.nn.tanh(state + update)
            return next_state, None

        state, _ = brainstate.transform.scan(
            step, initial, jnp.arange(SPATIAL_RECURRENCE_STEPS)
        )
        return state

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
        self,
        hidden: jnp.ndarray,
        query_features: jnp.ndarray,
        shape_features: jnp.ndarray | None = None,
        temporal_memory: jnp.ndarray | None = None,
        temporal_valid: jnp.ndarray | None = None,
        demonstration_inputs: jnp.ndarray | None = None,
        demonstration_outputs: jnp.ndarray | None = None,
        demonstration_grid_valid: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Emit direct height, width, and per-cell colour logits.

        Parameters
        ----------
        hidden : jax.Array
            Final recurrent task state shaped ``(batch, hidden_width)``.
        query_features : jax.Array
            Lossless query colours plus validity shaped ``(batch, 30, 30, 11)``.
        shape_features : jax.Array, optional
            Lossless demonstration/query dimension encoding shaped
            ``(batch, 1270)``. When omitted, an all-zero diagnostic encoding
            is used.
        temporal_memory : jax.Array, optional
            Recurrent demonstration-event states shaped
            ``(batch, time, hidden_width)``. When omitted with
            ``temporal_valid``, zero memory is used for direct diagnostics.
        temporal_valid : jax.Array, optional
            Boolean event-valid mask shaped ``(batch, time)``.
        demonstration_inputs : jax.Array, optional
            Complete demonstration input colors shaped ``(batch, 10, 30, 30, 10)``.
        demonstration_outputs : jax.Array, optional
            Complete demonstration output colors shaped ``(batch, 10, 30, 30, 10)``.
        demonstration_grid_valid : jax.Array, optional
            Boolean demonstration mask shaped ``(batch, 10)``.

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
        batch_size = query_features.shape[0]
        if (temporal_memory is None) != (temporal_valid is None):
            raise ValueError(
                "temporal_memory and temporal_valid must be provided together."
            )
        if temporal_memory is None:
            temporal_memory = jnp.zeros(
                (batch_size, 1, self.config.hidden_width), dtype=query_features.dtype
            )
            temporal_valid = jnp.zeros((batch_size, 1), dtype=bool)
        temporal_memory = jnp.asarray(temporal_memory)
        temporal_valid = jnp.asarray(temporal_valid, dtype=bool)
        if (
            temporal_memory.ndim != 3
            or temporal_memory.shape[0] != batch_size
            or temporal_memory.shape[2] != self.config.hidden_width
            or temporal_valid.shape != temporal_memory.shape[:2]
        ):
            raise ValueError(
                "temporal memory must have shapes (batch, time, hidden_width) "
                "and (batch, time)."
            )
        demonstration_items = (
            demonstration_inputs,
            demonstration_outputs,
            demonstration_grid_valid,
        )
        if any(item is None for item in demonstration_items) and not all(
            item is None for item in demonstration_items
        ):
            raise ValueError(
                "demonstration inputs, outputs, and valid mask must be provided together."
            )
        if demonstration_inputs is None:
            demonstration_inputs = jnp.zeros(
                (
                    batch_size,
                    MAX_DEMONSTRATIONS,
                    MAX_GRID_SIZE,
                    MAX_GRID_SIZE,
                    COLOR_COUNT,
                ),
                dtype=query_features.dtype,
            )
            demonstration_outputs = jnp.zeros(
                (
                    batch_size,
                    MAX_DEMONSTRATIONS,
                    MAX_GRID_SIZE,
                    MAX_GRID_SIZE,
                    COLOR_COUNT,
                ),
                dtype=query_features.dtype,
            )
            demonstration_grid_valid = jnp.zeros(
                (batch_size, MAX_DEMONSTRATIONS), dtype=bool
            )
        demonstration_outputs = jnp.asarray(demonstration_outputs)
        demonstration_inputs = jnp.asarray(demonstration_inputs)
        demonstration_grid_valid = jnp.asarray(demonstration_grid_valid, dtype=bool)
        expected_demonstration_shape = (
            batch_size,
            MAX_DEMONSTRATIONS,
            MAX_GRID_SIZE,
            MAX_GRID_SIZE,
            COLOR_COUNT,
        )
        if demonstration_inputs.shape != expected_demonstration_shape:
            raise ValueError("demonstration_inputs has an invalid shape.")
        if demonstration_outputs.shape != expected_demonstration_shape:
            raise ValueError("demonstration_outputs has an invalid shape.")
        if demonstration_grid_valid.shape != (
            batch_size,
            MAX_DEMONSTRATIONS,
        ):
            raise ValueError("demonstration_grid_valid has an invalid shape.")
        if shape_features is None:
            shape_features = jnp.zeros(
                (batch_size, SHAPE_FEATURE_WIDTH), dtype=query_features.dtype
            )
        shape_features = jnp.asarray(shape_features)
        if shape_features.shape != (batch_size, SHAPE_FEATURE_WIDTH):
            raise ValueError("shape_features must have shape (batch, 1270).")
        demonstration_features = shape_features[
            :, : MAX_DEMONSTRATIONS * DEMONSTRATION_SHAPE_WIDTH
        ].reshape(batch_size, MAX_DEMONSTRATIONS, DEMONSTRATION_SHAPE_WIDTH)
        demonstration_valid = demonstration_features[..., 0]
        demonstration_states = brainstate.nn.tanh(
            self.demo_shape_projection(demonstration_features)
        )
        demonstration_summary = jnp.sum(
            demonstration_states * demonstration_valid[..., None], axis=1
        ) / jnp.maximum(jnp.sum(demonstration_valid, axis=1)[..., None], 1.0)
        query_dimensions = shape_features[
            :, MAX_DEMONSTRATIONS * DEMONSTRATION_SHAPE_WIDTH :
        ]
        validity = query_features[..., -1]
        query_shape = jnp.concatenate(
            (jnp.max(validity, axis=2), jnp.max(validity, axis=1)), axis=-1
        )
        shape_state = brainstate.nn.tanh(
            hidden
            + self.query_shape_projection(query_shape)
            + demonstration_summary
            + self.query_dimension_projection(query_dimensions)
        )
        context_vector = self.context_projection(hidden)
        context = context_vector[:, None, None, :]
        query = self.query_projection(query_features)
        local_query_keys = brainstate.nn.tanh(
            self.local_query_projection(query_features)
        )
        local_query = self.local_query_refinement(local_query_keys)
        occupancy = jnp.sum(query_features[..., 1:COLOR_COUNT], axis=-1)
        global_pattern = self.global_query_pattern_projection(
            occupancy.reshape(batch_size, -1)
        )[:, None, None, :]
        coordinate = self.coordinate_projection(self.coordinate_features)
        coordinate = coordinate.reshape(
            MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width
        )
        flat_query = query_features.reshape(
            batch_size, MAX_GRID_SIZE * MAX_GRID_SIZE, QUERY_FEATURE_WIDTH
        )
        flat_coordinates = jnp.broadcast_to(
            self.coordinate_features[None],
            (batch_size, MAX_GRID_SIZE * MAX_GRID_SIZE, 2 * MAX_GRID_SIZE),
        )
        source_features = jnp.concatenate((flat_query, flat_coordinates), axis=-1)
        keys = self.attention_key_projection(source_features)
        values = self.attention_value_projection(source_features)
        output_queries = brainstate.nn.tanh(
            context_vector[:, None, :]
            + coordinate.reshape(-1, self.config.decoder_width)[None]
        )
        attention_logits = jnp.einsum("bod,bsd->bos", output_queries, keys) / np.sqrt(
            float(self.config.decoder_width)
        )
        source_valid = flat_query[..., -1] > 0.5
        attention_logits = jnp.where(source_valid[:, None, :], attention_logits, -1.0e9)
        attention = jnp.einsum(
            "bos,bsd->bod", brainstate.nn.softmax(attention_logits, axis=-1), values
        ).reshape(batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width)
        memory_keys = self.memory_key_projection(temporal_memory)
        memory_values = self.memory_value_projection(temporal_memory)
        memory_queries = brainstate.nn.tanh(
            global_pattern.reshape(batch_size, 1, self.config.decoder_width)
            + coordinate.reshape(-1, self.config.decoder_width)[None]
        )
        memory_logits = jnp.einsum(
            "bod,btd->bot", memory_queries, memory_keys
        ) / np.sqrt(float(self.config.decoder_width))
        memory_weights = brainstate.nn.softmax(
            jnp.where(temporal_valid[:, None, :], memory_logits, -1.0e9), axis=-1
        )
        memory_weights = memory_weights * temporal_valid[:, None, :]
        memory_weights = memory_weights / jnp.maximum(
            jnp.sum(memory_weights, axis=-1, keepdims=True), 1.0
        )
        memory_attention = jnp.einsum(
            "bot,btd->bod", memory_weights, memory_values
        ).reshape(batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width)
        demonstration_pair_summary = self._demonstration_pair_summary(
            demonstration_inputs,
            demonstration_outputs,
            demonstration_grid_valid,
        )
        spatial_state = self._spatial_recurrence(
            hidden,
            query_features,
            demonstration_pair_summary,
        )
        cell_state = brainstate.nn.tanh(
            context
            + query
            + local_query
            + global_pattern
            + coordinate[None]
            + attention
            + memory_attention
        )
        return (
            self.height_head(shape_state),
            self.width_head(shape_state),
            self.color_head(cell_state)
            + self.direct_query_color_head(query_features[..., :COLOR_COUNT])
            + self.spatial_color_head(spatial_state),
        )

    def run(
        self,
        events: jnp.ndarray,
        query_features: jnp.ndarray,
        shape_features: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode all events with one compiled loop and decode one grid.

        Parameters
        ----------
        events : jax.Array
            Time-major row events shaped ``(time, batch, input_width)``.
        query_features : jax.Array
            Lossless query colours plus validity shaped ``(batch, 30, 30, 11)``.
        shape_features : jax.Array, optional
            Lossless target-free demonstration/query dimension features.

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
        valid = jnp.asarray(events[..., 0] > 0.5, dtype=hidden_sequence.dtype)
        pooled = jnp.sum(hidden_sequence * valid[..., None], axis=0) / jnp.maximum(
            jnp.sum(valid, axis=0)[..., None], 1.0
        )
        summary = brainstate.nn.tanh(
            self.temporal_summary_projection(
                jnp.concatenate((hidden_sequence[-1], pooled), axis=-1)
            )
        )
        demonstration_inputs = None
        demonstration_outputs = None
        demonstration_grid_valid = None
        if self.config.memory_key_indices:
            value_indices = jnp.asarray(self.config.memory_value_indices)
            demonstration_event_count = MAX_DEMONSTRATIONS * MAX_GRID_SIZE
            if events.shape[0] >= demonstration_event_count:
                raw_demo_keys = events[
                    :demonstration_event_count,
                    ...,
                    jnp.asarray(self.config.memory_key_indices),
                ]
                raw_demo_values = events[
                    :demonstration_event_count,
                    ...,
                    value_indices,
                ]
                input_colors = raw_demo_keys[..., -MAX_GRID_SIZE * COLOR_COUNT :]
                output_colors = raw_demo_values[..., -MAX_GRID_SIZE * COLOR_COUNT :]
                input_colors = input_colors.reshape(
                    MAX_DEMONSTRATIONS,
                    MAX_GRID_SIZE,
                    events.shape[1],
                    MAX_GRID_SIZE,
                    COLOR_COUNT,
                ).transpose(2, 0, 1, 3, 4)
                demonstration_inputs = input_colors
                demonstration_outputs = output_colors.reshape(
                    MAX_DEMONSTRATIONS,
                    MAX_GRID_SIZE,
                    events.shape[1],
                    MAX_GRID_SIZE,
                    COLOR_COUNT,
                ).transpose(2, 0, 1, 3, 4)
                demonstration_grid_valid = jnp.any(
                    events[:demonstration_event_count, ..., 0]
                    .reshape(
                        MAX_DEMONSTRATIONS,
                        MAX_GRID_SIZE,
                        events.shape[1],
                    )
                    .transpose(2, 0, 1)
                    > 0.5,
                    axis=-1,
                )
        return self.decode(
            summary,
            query_features,
            shape_features,
            jnp.swapaxes(hidden_sequence, 0, 1),
            jnp.swapaxes(valid, 0, 1),
            demonstration_inputs=demonstration_inputs,
            demonstration_outputs=demonstration_outputs,
            demonstration_grid_valid=demonstration_grid_valid,
        )
