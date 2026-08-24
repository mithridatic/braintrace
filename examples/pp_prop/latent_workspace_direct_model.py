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
SHAPE_FEATURE_WIDTH = (
    MAX_DEMONSTRATIONS * DEMONSTRATION_SHAPE_WIDTH + 2 * MAX_GRID_SIZE
)
ARCHITECTURE_VERSION = "color_invariant_shared_relation_v11"


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
        Each group is collapsed to one foreground-occupancy feature.
    architecture_version : str, default="color_invariant_shared_relation_v11"
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
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        normalized_indices = {}
        for name in ("memory_key_indices", "memory_value_indices"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of integers.")
            indices = tuple(raw)
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, Integral)
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
        object.__setattr__(
            self, "memory_key_color_block_width", color_block_width
        )
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"architecture_version must be {ARCHITECTURE_VERSION!r}."
            )


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
            self.global_query_pattern_projection = braintrace.nn.Linear(
                MAX_GRID_SIZE * MAX_GRID_SIZE, config.decoder_width
            )
            self.memory_key_projection = braintrace.nn.Linear(
                config.hidden_width, config.decoder_width
            )
            self.memory_value_projection = braintrace.nn.Linear(
                config.hidden_width, config.decoder_width
            )
            associative_key_width = (
                len(config.memory_key_indices)
                - config.memory_key_color_block_width
                + config.memory_key_color_block_width // COLOR_COUNT
            )
            associative_key_width = max(1, associative_key_width)
            associative_value_width = max(1, len(config.memory_value_indices))
            self.associative_key_projection = braintrace.nn.Linear(
                associative_key_width, config.decoder_width
            )
            self.associative_value_projection = braintrace.nn.Linear(
                associative_value_width, config.decoder_width
            )
            self.relation_output_projection = braintrace.nn.Linear(
                config.decoder_width, config.decoder_width
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

    def _associative_key_features(self, events: jnp.ndarray) -> jnp.ndarray:
        selected = events[..., jnp.asarray(self.config.memory_key_indices)]
        color_width = self.config.memory_key_color_block_width
        if color_width == 0:
            return selected
        prefix = selected[..., :-color_width]
        colors = selected[..., -color_width:].reshape(
            *selected.shape[:-1], color_width // COLOR_COUNT, COLOR_COUNT
        )
        foreground = jnp.sum(colors[..., 1:], axis=-1)
        return jnp.concatenate((prefix, foreground), axis=-1)

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
        associative_keys: jnp.ndarray | None = None,
        associative_values: jnp.ndarray | None = None,
        associative_valid: jnp.ndarray | None = None,
        associative_queries: jnp.ndarray | None = None,
        associative_query_valid: jnp.ndarray | None = None,
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
        associative_keys, associative_values : jax.Array, optional
            Projected lossless demonstration-row keys and values shaped
            ``(batch, time, decoder_width)``.
        associative_valid : jax.Array, optional
            Boolean demonstration-row mask shaped ``(batch, time)``.
        associative_queries : jax.Array, optional
            Projected target-free query rows shaped
            ``(batch, time, decoder_width)``.
        associative_query_valid : jax.Array, optional
            Boolean query-row mask shaped ``(batch, time)``.

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
        associative_items = (
            associative_keys,
            associative_values,
            associative_valid,
        )
        if any(item is None for item in associative_items) and not all(
            item is None for item in associative_items
        ):
            raise ValueError(
                "associative keys, values, and valid mask must be provided together."
            )
        if associative_keys is None:
            associative_keys = jnp.zeros(
                (batch_size, 1, self.config.decoder_width),
                dtype=query_features.dtype,
            )
            associative_values = jnp.zeros_like(associative_keys)
            associative_valid = jnp.zeros((batch_size, 1), dtype=bool)
        associative_keys = jnp.asarray(associative_keys)
        associative_values = jnp.asarray(associative_values)
        associative_valid = jnp.asarray(associative_valid, dtype=bool)
        expected_associative_shape = (
            batch_size,
            associative_keys.shape[1],
            self.config.decoder_width,
        )
        if (
            associative_keys.ndim != 3
            or associative_keys.shape != expected_associative_shape
            or associative_values.shape != expected_associative_shape
            or associative_valid.shape != expected_associative_shape[:2]
        ):
            raise ValueError(
                "associative memory must have aligned batch, time, and decoder widths."
            )
        if (associative_queries is None) != (associative_query_valid is None):
            raise ValueError(
                "associative queries and their valid mask must be provided together."
            )
        if associative_queries is None:
            associative_queries = jnp.zeros(
                (batch_size, 1, self.config.decoder_width),
                dtype=query_features.dtype,
            )
            associative_query_valid = jnp.zeros((batch_size, 1), dtype=bool)
        associative_queries = jnp.asarray(associative_queries)
        associative_query_valid = jnp.asarray(
            associative_query_valid, dtype=bool
        )
        if (
            associative_queries.ndim != 3
            or associative_queries.shape[0] != batch_size
            or associative_queries.shape[2] != self.config.decoder_width
            or associative_query_valid.shape != associative_queries.shape[:2]
        ):
            raise ValueError(
                "associative queries must have aligned batch, time, and decoder widths."
            )
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
        ) / jnp.maximum(
            jnp.sum(demonstration_valid, axis=1)[..., None], 1.0
        )
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
        source_features = jnp.concatenate(
            (flat_query, flat_coordinates), axis=-1
        )
        keys = self.attention_key_projection(source_features)
        values = self.attention_value_projection(source_features)
        output_queries = brainstate.nn.tanh(
            context_vector[:, None, :] + coordinate.reshape(-1, self.config.decoder_width)[None]
        )
        attention_logits = jnp.einsum(
            "bod,bsd->bos", output_queries, keys
        ) / np.sqrt(float(self.config.decoder_width))
        source_valid = flat_query[..., -1] > 0.5
        attention_logits = jnp.where(
            source_valid[:, None, :], attention_logits, -1.0e9
        )
        attention = jnp.einsum(
            "bos,bsd->bod", brainstate.nn.softmax(attention_logits, axis=-1), values
        ).reshape(
            batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width
        )
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
        ).reshape(
            batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width
        )
        associative_logits = jnp.einsum(
            "bod,btd->bot", memory_queries, associative_keys
        ) / np.sqrt(float(self.config.decoder_width))
        associative_weights = brainstate.nn.softmax(
            jnp.where(
                associative_valid[:, None, :], associative_logits, -1.0e9
            ),
            axis=-1,
        )
        associative_weights = associative_weights * associative_valid[:, None, :]
        associative_weights = associative_weights / jnp.maximum(
            jnp.sum(associative_weights, axis=-1, keepdims=True), 1.0
        )
        associative_attention = jnp.einsum(
            "bot,btd->bod", associative_weights, associative_values
        ).reshape(
            batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width
        )
        relation_logits = jnp.einsum(
            "bqd,btd->bqt", associative_queries, associative_keys
        ) / np.sqrt(float(self.config.decoder_width))
        relation_weights = brainstate.nn.softmax(
            jnp.where(
                associative_valid[:, None, :], relation_logits, -1.0e9
            ),
            axis=-1,
        )
        relation_weights = relation_weights * associative_valid[:, None, :]
        relation_weights = relation_weights / jnp.maximum(
            jnp.sum(relation_weights, axis=-1, keepdims=True), 1.0
        )
        inferred_query_values = jnp.einsum(
            "bqt,btd->bqd", relation_weights, associative_values
        )
        output_relation_logits = jnp.einsum(
            "bod,bqd->boq", memory_queries, associative_queries
        ) / np.sqrt(float(self.config.decoder_width))
        output_relation_weights = brainstate.nn.softmax(
            jnp.where(
                associative_query_valid[:, None, :],
                output_relation_logits,
                -1.0e9,
            ),
            axis=-1,
        )
        output_relation_weights = (
            output_relation_weights * associative_query_valid[:, None, :]
        )
        output_relation_weights = output_relation_weights / jnp.maximum(
            jnp.sum(output_relation_weights, axis=-1, keepdims=True), 1.0
        )
        relation_attention = self.relation_output_projection(jnp.einsum(
            "boq,bqd->bod", output_relation_weights, inferred_query_values
        )).reshape(
            batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE, self.config.decoder_width
        )
        cell_state = brainstate.nn.tanh(
            context
            + query
            + global_pattern
            + coordinate[None]
            + attention
            + memory_attention
            + associative_attention
            + relation_attention
        )
        return (
            self.height_head(shape_state),
            self.width_head(shape_state),
            self.color_head(cell_state),
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
        associative_keys = None
        associative_values = None
        associative_valid = None
        associative_queries = None
        associative_query_valid = None
        if self.config.memory_key_indices:
            value_indices = jnp.asarray(self.config.memory_value_indices)
            projected_keys = jnp.swapaxes(
                self.associative_key_projection(self._associative_key_features(events)),
                0,
                1,
            )
            associative_keys = projected_keys
            associative_values = jnp.swapaxes(
                self.associative_value_projection(events[..., value_indices]), 0, 1
            )
            associative_valid = jnp.swapaxes(
                jnp.logical_and(valid > 0.5, events[..., 1] > 0.5), 0, 1
            )
            associative_queries = projected_keys
            associative_query_valid = jnp.swapaxes(
                jnp.logical_and(valid > 0.5, events[..., 2] > 0.5), 0, 1
            )
        return self.decode(
            summary,
            query_features,
            shape_features,
            jnp.swapaxes(hidden_sequence, 0, 1),
            jnp.swapaxes(valid, 0, 1),
            associative_keys,
            associative_values,
            associative_valid,
            associative_queries,
            associative_query_valid,
        )
