"""Target-isolated episode encoding and compiled BPTT for direct ARC output."""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import brainstate
import braintools
import jax
import jax.numpy as jnp
import msgspec
import numpy as np
import optax

try:
    from examples.pp_prop.latent_workspace_direct_model import (
        DirectARCGRU,
        DirectModelConfig,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        RowEventConfig,
        encode_query_episode,
        encode_target_grid,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import fallback.
    from latent_workspace_direct_model import (
        DirectARCGRU,
        DirectModelConfig,
    )
    from latent_workspace_task import (
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
MAX_DEMONSTRATIONS = 10
DEMONSTRATION_SHAPE_WIDTH = 1 + 4 * MAX_GRID_SIZE
SHAPE_FEATURE_WIDTH = (
    MAX_DEMONSTRATIONS * DEMONSTRATION_SHAPE_WIDTH + 2 * MAX_GRID_SIZE
)


@dataclass(frozen=True)
class DirectEpisode:
    """Hold one target-free model input and its scorer-side training target.

    Parameters
    ----------
    events : numpy.ndarray
        Lossless row events shaped ``(time, input_width)``.
    query_features : numpy.ndarray
        Query colours and validity shaped ``(30, 30, 11)``.
    shape_features : numpy.ndarray
        Target-free one-hot demonstration/query dimensions shaped ``(1270,)``.
    target_colors, target_mask : numpy.ndarray
        Padded scorer-side target and valid-cell mask.
    target_height, target_width : int
        Zero-based categorical output dimensions.
    task_id : str
        Host-only task identifier excluded from model inputs.
    """

    events: np.ndarray
    query_features: np.ndarray
    shape_features: np.ndarray
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
    shape_features : numpy.ndarray
        Target-free dimension features shaped ``(batch, 1270)``.
    target_colors, target_mask : numpy.ndarray
        Padded targets and valid-cell masks shaped ``(batch, 30, 30)``.
    target_heights, target_widths : numpy.ndarray
        Zero-based dimension labels shaped ``(batch,)``.
    """

    events: np.ndarray
    query_features: np.ndarray
    shape_features: np.ndarray
    target_colors: np.ndarray
    target_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray


@dataclass(frozen=True)
class DirectTrainingChunk:
    """Hold update-major batches consumed by one compiled training loop.

    Parameters
    ----------
    events, query_features, shape_features, target_colors, target_mask,
    target_heights, target_widths
        Arrays from :class:`DirectBatch` with a leading update dimension.
    """

    events: np.ndarray
    query_features: np.ndarray
    shape_features: np.ndarray
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


def _shape_features(task: ArcTask, query_index: int) -> np.ndarray:
    features = np.zeros((SHAPE_FEATURE_WIDTH,), dtype=np.float32)
    if len(task.train) > MAX_DEMONSTRATIONS:
        raise ValueError("Direct shape encoding supports at most 10 demonstrations.")
    for index, pair in enumerate(task.train):
        if pair.output is None:
            raise ValueError("Demonstration outputs are required for shape encoding.")
        offset = index * DEMONSTRATION_SHAPE_WIDTH
        features[offset] = 1.0
        dimensions = (
            pair.input.height,
            pair.input.width,
            pair.output.height,
            pair.output.width,
        )
        for group, dimension in enumerate(dimensions):
            features[offset + 1 + group * MAX_GRID_SIZE + dimension - 1] = 1.0
    query = task.test[query_index].input
    query_offset = MAX_DEMONSTRATIONS * DEMONSTRATION_SHAPE_WIDTH
    features[query_offset + query.height - 1] = 1.0
    features[query_offset + MAX_GRID_SIZE + query.width - 1] = 1.0
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
        shape_features=_shape_features(task, query_index),
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
        shape_features=np.stack(
            [episode.shape_features for episode in episodes], axis=0
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
        shape_features=repeated(batch.shape_features),
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
        Scalar loss with bounded inverse-frequency cell weighting and equal
        height, width, and aggregate cell contributions.
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
    one_hot = jax.nn.one_hot(target_colors, COLOR_COUNT, dtype=per_cell.dtype)
    class_counts = jnp.sum(one_hot * mask[..., None], axis=(0, 1, 2))
    valid_count = jnp.maximum(jnp.sum(mask), 1.0)
    class_weights = jnp.sqrt(valid_count / jnp.maximum(class_counts, 1.0))
    class_weights = jnp.where(
        class_counts > 0.0, jnp.clip(class_weights, 0.5, 4.0), 0.0
    )
    cell_weights = jnp.take(class_weights, target_colors) * mask
    weighted_cell_loss = per_cell * cell_weights
    color_loss = jnp.sum(weighted_cell_loss) / jnp.maximum(
        jnp.sum(cell_weights), 1.0
    )
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


def save_direct_checkpoint(
    model: DirectARCGRU, path: pathlib.Path
) -> str:
    """Save an exact, schema-bound direct-model checkpoint.

    Parameters
    ----------
    model : DirectARCGRU
        Model whose checkpoint-owned parameters are saved.
    path : pathlib.Path
        Destination ``.npz`` path.

    Returns
    -------
    str
        SHA-256 digest of the ordered parameter schema and bytes.
    """

    if not isinstance(model, DirectARCGRU):
        raise TypeError("model must be a DirectARCGRU instance.")
    path = pathlib.Path(path)
    states = model.states(brainstate.ParamState)
    arrays: dict[str, np.ndarray] = {}
    leaves_metadata = []
    leaf_number = 0
    for state_path, state in states.items():
        path_text = ".".join(map(str, state_path))
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            array = np.ascontiguousarray(np.asarray(leaf))
            key = f"leaf_{leaf_number:04d}"
            arrays[key] = array
            leaves_metadata.append(
                {
                    "key": key,
                    "state_path": path_text,
                    "tree_leaf_index": index,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
            )
            leaf_number += 1
    digest = parameter_digest(model)
    metadata = {
        "schema_version": 1,
        "architecture": asdict(model.config),
        "parameter_sha256": digest,
        "leaves": leaves_metadata,
    }
    arrays["__metadata__"] = np.frombuffer(
        msgspec.json.encode(metadata, order="sorted"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return digest


def load_direct_checkpoint(
    path: pathlib.Path,
) -> tuple[DirectARCGRU, dict[str, object]]:
    """Load a direct-model checkpoint while enforcing its exact leaf schema.

    Parameters
    ----------
    path : pathlib.Path
        Existing checkpoint path.

    Returns
    -------
    model : DirectARCGRU
        Reconstructed model with exact saved parameters.
    metadata : dict
        Validated checkpoint metadata.
    """

    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "__metadata__" not in names:
                raise ValueError("checkpoint metadata is missing.")
            metadata = msgspec.json.decode(bytes(archive["__metadata__"]))
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
                raise ValueError("checkpoint schema_version is unsupported.")
            architecture = metadata.get("architecture")
            leaves_metadata = metadata.get("leaves")
            if not isinstance(architecture, dict) or not isinstance(
                leaves_metadata, list
            ):
                raise TypeError("checkpoint metadata schema is invalid.")
            model = DirectARCGRU(DirectModelConfig(**architecture))
            states = model.states(brainstate.ParamState)
            expected_names = {
                item.get("key") for item in leaves_metadata if isinstance(item, dict)
            }
            if names != expected_names | {"__metadata__"}:
                raise ValueError("checkpoint leaf set does not match metadata.")
            cursor = 0
            for state_path, state in states.items():
                expected_path = ".".join(map(str, state_path))
                structure = jax.tree.structure(state.value)
                original_leaves = jax.tree.leaves(state.value)
                restored = []
                for index, original in enumerate(original_leaves):
                    if cursor >= len(leaves_metadata):
                        raise ValueError("checkpoint is missing parameter leaves.")
                    item = leaves_metadata[cursor]
                    if not isinstance(item, dict):
                        raise TypeError("checkpoint leaf metadata is invalid.")
                    key = item.get("key")
                    array = np.asarray(archive[key])
                    if (
                        item.get("state_path") != expected_path
                        or item.get("tree_leaf_index") != index
                        or item.get("shape") != list(np.asarray(original).shape)
                        or item.get("dtype") != np.asarray(original).dtype.str
                        or list(array.shape) != item.get("shape")
                        or array.dtype.str != item.get("dtype")
                    ):
                        raise ValueError("checkpoint parameter schema does not match model.")
                    restored.append(jnp.asarray(array))
                    cursor += 1
                state.value = jax.tree.unflatten(structure, restored)
            if cursor != len(leaves_metadata):
                raise ValueError("checkpoint has unexpected parameter leaves.")
    except (KeyError, OSError, TypeError, msgspec.DecodeError) as error:
        raise ValueError("checkpoint could not be decoded safely.") from error
    expected_digest = metadata.get("parameter_sha256")
    if not isinstance(expected_digest, str) or parameter_digest(model) != expected_digest:
        raise ValueError("checkpoint parameter digest does not match its contents.")
    return model, metadata


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
            shape_features: jax.Array,
            target_colors: jax.Array,
            target_mask: jax.Array,
            target_heights: jax.Array,
            target_widths: jax.Array,
        ) -> jax.Array:
            brainstate.nn.reset_all_states(model, batch_size=batch_size)

            def objective() -> jax.Array:
                logits = model.run(events, query_features, shape_features)
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
            shape_features: jax.Array,
            target_colors: jax.Array,
            target_mask: jax.Array,
            target_heights: jax.Array,
            target_widths: jax.Array,
        ) -> jax.Array:
            return brainstate.transform.for_loop(
                update,
                events,
                query_features,
                shape_features,
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
            jnp.asarray(chunk.shape_features),
            jnp.asarray(chunk.target_colors),
            jnp.asarray(chunk.target_mask),
            jnp.asarray(chunk.target_heights),
            jnp.asarray(chunk.target_widths),
        )
