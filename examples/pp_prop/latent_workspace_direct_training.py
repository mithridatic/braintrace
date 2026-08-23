"""Target-isolated episode encoding and compiled BPTT for direct ARC output."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral, Real

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    from examples.pp_prop.latent_workspace_direct_model import DirectARCGRU
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        encode_query_episode,
        encode_target_grid,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import fallback.
    from latent_workspace_direct_model import (  # pyright: ignore[reportImplicitRelativeImport]
        DirectARCGRU,
    )
    from latent_workspace_task import (  # pyright: ignore[reportImplicitRelativeImport]
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        encode_query_episode,
        encode_target_grid,
    )

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
QUERY_FEATURE_WIDTH = COLOR_COUNT + 1


@dataclass(frozen=True)
class DirectEpisode:
    """Hold one target-free model input and its scorer-side training target.

    Parameters
    ----------
    events : numpy.ndarray
        Lossless row events shaped ``(time, input_width)``.
    query_features : numpy.ndarray
        Query colours and validity shaped ``(30, 30, 11)``.
    target_colors, target_mask : numpy.ndarray
        Padded scorer-side target and valid-cell mask.
    target_height, target_width : int
        Zero-based categorical output dimensions.
    task_id : str
        Host-only task identifier excluded from model inputs.
    """

    events: np.ndarray
    query_features: np.ndarray
    target_colors: np.ndarray
    target_mask: np.ndarray
    target_height: int
    target_width: int
    task_id: str


@dataclass(frozen=True)
class DirectBatch:
    """Hold one time-major batch of direct ARC training episodes.

    Parameters
    ----------
    events : numpy.ndarray
        Time-major tensor shaped ``(time, batch, input_width)``.
    query_features : numpy.ndarray
        Query features shaped ``(batch, 30, 30, 11)``.
    target_colors, target_mask : numpy.ndarray
        Padded targets and valid-cell masks shaped ``(batch, 30, 30)``.
    target_heights, target_widths : numpy.ndarray
        Zero-based dimension labels shaped ``(batch,)``.
    """

    events: np.ndarray
    query_features: np.ndarray
    target_colors: np.ndarray
    target_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray


@dataclass(frozen=True)
class DirectTrainingChunk:
    """Hold update-major batches consumed by one compiled training loop.

    Parameters
    ----------
    events, query_features, target_colors, target_mask, target_heights, target_widths
        Arrays from :class:`DirectBatch` with a leading update dimension.
    """

    events: np.ndarray
    query_features: np.ndarray
    target_colors: np.ndarray
    target_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray


def leave_one_out_tasks(task: ArcTask) -> tuple[ArcTask, ...]:
    """Turn every training pair into a target-isolated query episode.

    Parameters
    ----------
    task : ArcTask
        Source ARC training task with at least two demonstrations.

    Returns
    -------
    tuple of ArcTask
        One task per held-out demonstration; remaining pairs are context.

    Raises
    ------
    ValueError
        If fewer than two demonstrations are available.
    """

    if not isinstance(task, ArcTask):
        raise TypeError("task must be an ArcTask instance.")
    if len(task.train) < 2:
        raise ValueError("leave-one-out training requires at least two demonstrations.")
    episodes = []
    for held_out_index, held_out in enumerate(task.train):
        demonstrations = tuple(
            pair for index, pair in enumerate(task.train) if index != held_out_index
        )
        episodes.append(
            ArcTask(
                train=demonstrations,
                test=(ArcPair(held_out.input, held_out.output),),
                task_id=task.task_id,
            )
        )
    return tuple(episodes)


def _query_features(grid: ArcGrid) -> np.ndarray:
    features = np.zeros(
        (MAX_GRID_SIZE, MAX_GRID_SIZE, QUERY_FEATURE_WIDTH), dtype=np.float32
    )
    array = grid.as_array()
    rows, columns = np.indices(array.shape)
    features[rows, columns, array] = 1.0
    features[: grid.height, : grid.width, COLOR_COUNT] = 1.0
    return features


def encode_direct_episode(
    task: ArcTask, query_index: int, row_config: RowEventConfig
) -> DirectEpisode:
    """Encode one episode while keeping its target outside model inputs.

    Parameters
    ----------
    task : ArcTask
        Task containing demonstrations and the selected query.
    query_index : int
        Index of the query to encode.
    row_config : RowEventConfig
        Static lossless row-event layout.

    Returns
    -------
    DirectEpisode
        Model inputs plus separated supervised targets.
    """

    if not isinstance(row_config, RowEventConfig):
        raise TypeError("row_config must be a RowEventConfig instance.")
    encoded = encode_query_episode(task, query_index, row_config)
    target = encoded.target
    if target is None:
        raise ValueError("Direct training episodes require a scorer-side target.")
    grid_target = encode_target_grid(target, max_grid_size=MAX_GRID_SIZE)
    query = task.test[query_index]
    return DirectEpisode(
        events=np.asarray(encoded.events, dtype=np.float32),
        query_features=_query_features(query.input),
        target_colors=np.asarray(grid_target.colors, dtype=np.int32),
        target_mask=np.asarray(grid_target.valid_mask, dtype=np.bool_),
        target_height=grid_target.height_index,
        target_width=grid_target.width_index,
        task_id=task.task_id or encoded.task_fingerprint,
    )


def stack_direct_episodes(episodes: tuple[DirectEpisode, ...]) -> DirectBatch:
    """Stack equal-layout episodes into one time-major model batch.

    Parameters
    ----------
    episodes : tuple of DirectEpisode
        Nonempty episodes with equal event shapes.

    Returns
    -------
    DirectBatch
        Time-major model inputs and batch-major scorer-side targets.
    """

    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    if any(not isinstance(episode, DirectEpisode) for episode in episodes):
        raise TypeError("Every episode must be a DirectEpisode.")
    event_shapes = {episode.events.shape for episode in episodes}
    if len(event_shapes) != 1:
        raise ValueError("Every episode must have the same event shape.")
    return DirectBatch(
        events=np.stack([episode.events for episode in episodes], axis=1),
        query_features=np.stack(
            [episode.query_features for episode in episodes], axis=0
        ),
        target_colors=np.stack(
            [episode.target_colors for episode in episodes], axis=0
        ),
        target_mask=np.stack([episode.target_mask for episode in episodes], axis=0),
        target_heights=np.asarray(
            [episode.target_height for episode in episodes], dtype=np.int32
        ),
        target_widths=np.asarray(
            [episode.target_width for episode in episodes], dtype=np.int32
        ),
    )


def repeat_batch(batch: DirectBatch, *, updates: int) -> DirectTrainingChunk:
    """Repeat one host batch for a compiled overfit or smoke-test chunk.

    Parameters
    ----------
    batch : DirectBatch
        Batch to repeat without changing its bytes.
    updates : int
        Positive number of compiled optimizer steps.

    Returns
    -------
    DirectTrainingChunk
        Update-major copy of every batch tensor.
    """

    if not isinstance(batch, DirectBatch):
        raise TypeError("batch must be a DirectBatch instance.")
    if isinstance(updates, bool) or not isinstance(updates, Integral):
        raise TypeError("updates must be a positive integer.")
    count = int(updates)
    if count < 1:
        raise ValueError("updates must be a positive integer.")

    def repeated(value: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(value)[None], count, axis=0)

    return DirectTrainingChunk(
        events=repeated(batch.events),
        query_features=repeated(batch.query_features),
        target_colors=repeated(batch.target_colors),
        target_mask=repeated(batch.target_mask),
        target_heights=repeated(batch.target_heights),
        target_widths=repeated(batch.target_widths),
    )


def direct_prediction_loss(
    logits: tuple[jax.Array, jax.Array, jax.Array],
    target_heights: jax.Array,
    target_widths: jax.Array,
    target_colors: jax.Array,
    target_mask: jax.Array,
) -> jax.Array:
    """Return balanced shape and valid-cell cross entropy.

    Parameters
    ----------
    logits : tuple of jax.Array
        Direct height, width, and cell-colour logits.
    target_heights, target_widths : jax.Array
        Zero-based dimension targets.
    target_colors, target_mask : jax.Array
        Padded cell targets and valid-cell mask.

    Returns
    -------
    jax.Array
        Scalar mean loss with equal height, width, and cell terms.
    """

    height_logits, width_logits, color_logits = logits
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_heights
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_widths
    ).mean()
    per_cell = optax.softmax_cross_entropy_with_integer_labels(
        color_logits, target_colors
    )
    mask = jnp.asarray(target_mask, dtype=per_cell.dtype)
    color_loss = jnp.sum(per_cell * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return (height_loss + width_loss + color_loss) / 3.0


def parameter_digest(model: DirectARCGRU) -> str:
    """Hash ordered checkpoint-owned parameter paths, metadata, and bytes.

    Parameters
    ----------
    model : DirectARCGRU
        Model whose current trainable leaves are bound.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """

    if not isinstance(model, DirectARCGRU):
        raise TypeError("model must be a DirectARCGRU instance.")
    digest = hashlib.sha256()
    for path, state in model.states(brainstate.ParamState).items():
        name = ".".join(map(str, path))
        digest.update(name.encode("utf-8"))
        for leaf in jax.tree.leaves(state.value):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class DirectBPTTTrainer:
    """Train a direct BrainTrace GRU with compiled chunked BPTT.

    Parameters
    ----------
    model : DirectARCGRU
        Model updated in place.
    batch_size : int
        Static batch size used by recurrent states.
    learning_rate : float, default=0.001
        Positive finite Adam learning rate.
    """

    def __init__(
        self,
        model: DirectARCGRU,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
    ):
        if not isinstance(model, DirectARCGRU):
            raise TypeError("model must be a DirectARCGRU instance.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
            raise TypeError("learning_rate must be a positive finite real.")
        rate = float(learning_rate)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite real.")
        self.model = model
        brainstate.nn.init_all_states(model, batch_size=self.batch_size)
        self.weights = model.states(brainstate.ParamState)
        self.optimizer = braintools.optim.Adam(lr=rate)
        self.optimizer.register_trainable_weights(self.weights)
        self._train_many = self._make_train_many()

    def _make_train_many(self):
        model = self.model
        weights = self.weights
        optimizer = self.optimizer
        batch_size = self.batch_size

        def update(
            events: jax.Array,
            query_features: jax.Array,
            target_colors: jax.Array,
            target_mask: jax.Array,
            target_heights: jax.Array,
            target_widths: jax.Array,
        ) -> jax.Array:
            brainstate.nn.reset_all_states(model, batch_size=batch_size)

            def objective() -> jax.Array:
                logits = model.run(events, query_features)
                return direct_prediction_loss(
                    logits,
                    target_heights,
                    target_widths,
                    target_colors,
                    target_mask,
                )

            gradients, loss = brainstate.transform.grad(
                objective, weights, return_value=True
            )()
            optimizer.update(gradients)
            return loss

        @brainstate.transform.jit
        def train_many(
            events: jax.Array,
            query_features: jax.Array,
            target_colors: jax.Array,
            target_mask: jax.Array,
            target_heights: jax.Array,
            target_widths: jax.Array,
        ) -> jax.Array:
            return brainstate.transform.for_loop(
                update,
                events,
                query_features,
                target_colors,
                target_mask,
                target_heights,
                target_widths,
            )

        return train_many

    def train_chunk(self, chunk: DirectTrainingChunk) -> jax.Array:
        """Run all optimizer steps in one compiled BrainState loop.

        Parameters
        ----------
        chunk : DirectTrainingChunk
            Update-major batches with the trainer's static batch size.

        Returns
        -------
        jax.Array
            One scalar loss per optimizer update.
        """

        if not isinstance(chunk, DirectTrainingChunk):
            raise TypeError("chunk must be a DirectTrainingChunk instance.")
        if chunk.events.ndim != 4 or chunk.events.shape[2] != self.batch_size:
            raise ValueError(
                "chunk events must have shape (updates, time, batch, input_width)."
            )
        return self._train_many(
            jnp.asarray(chunk.events),
            jnp.asarray(chunk.query_features),
            jnp.asarray(chunk.target_colors),
            jnp.asarray(chunk.target_mask),
            jnp.asarray(chunk.target_heights),
            jnp.asarray(chunk.target_widths),
        )
