"""Target-isolated sequence packing and single-step PP-prop ARC training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral, Real
import pathlib

import brainstate
import braintools
import jax
import jax.numpy as jnp
import msgspec
import numpy as np
import optax

import braintrace

from examples.pp_prop.latent_workspace_direct_training import encode_direct_episode
from examples.pp_prop.latent_workspace_direct_generation import (
    first_prediction_bytes,
    strict_task_pass_at_1,
)
from examples.pp_prop.latent_workspace_online_model import (
    MAX_GRID_SIZE,
    OnlineARCGRU,
    OnlineModelConfig,
    split_step_logits,
)
from examples.pp_prop.latent_workspace_task import (
    ArcTask,
    AugmentationConfig,
    RowEventConfig,
    augment_training_task,
)

DECODE_FEATURE_WIDTH = 1 + MAX_GRID_SIZE


@dataclass(frozen=True)
class OnlineEpisode:
    """Hold one target-free online sequence and scorer-side training signals.

    Parameters
    ----------
    events : numpy.ndarray
        Fixed sequence of left padding, lossless input rows, and decode tokens.
    target_rows, target_cell_mask : numpy.ndarray
        Per-step scorer-side colour labels and cell masks.
    target_heights, target_widths : numpy.ndarray
        Per-step zero-based shape labels.
    decode_mask : numpy.ndarray
        One-dimensional mask selecting the fixed 30 decode steps.
    task_id : str
        Host-only task identifier excluded from the event tensor.
    """

    events: np.ndarray
    target_rows: np.ndarray
    target_cell_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    decode_mask: np.ndarray
    task_id: str


@dataclass(frozen=True)
class OnlineBatch:
    """Hold one time-major batch of online ARC episodes.

    Parameters
    ----------
    events : numpy.ndarray
        Time-major model inputs.
    target_rows, target_cell_mask : numpy.ndarray
        Time-major scorer-side colour labels and cell masks.
    target_heights, target_widths : numpy.ndarray
        Time-major shape labels.
    decode_mask : numpy.ndarray
        Common one-dimensional decode-step mask.
    """

    events: np.ndarray
    target_rows: np.ndarray
    target_cell_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    decode_mask: np.ndarray


@dataclass(frozen=True)
class OnlineTrainingChunk:
    """Hold update-major batches for one compiled optimizer loop.

    Parameters
    ----------
    events, target_rows, target_cell_mask, target_heights, target_widths
        Arrays with leading update and time dimensions.
    decode_mask : numpy.ndarray
        Common one-dimensional decode-step mask.
    """

    events: np.ndarray
    target_rows: np.ndarray
    target_cell_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    decode_mask: np.ndarray


def encode_online_episode(
    task: ArcTask, query_index: int, row_config: RowEventConfig
) -> OnlineEpisode:
    """Encode one ARC query as target-free rows followed by decode tokens.

    Parameters
    ----------
    task : ArcTask
        Task containing demonstrations and the selected query.
    query_index : int
        Query index within ``task.test``.
    row_config : RowEventConfig
        Lossless row-event layout.

    Returns
    -------
    OnlineEpisode
        Fixed model inputs plus separated training/scorer signals.
    """

    if not isinstance(row_config, RowEventConfig):
        raise TypeError("row_config must be a RowEventConfig instance.")
    direct = encode_direct_episode(task, query_index, row_config)
    valid = direct.events[:, row_config.valid_slice.start] == 1.0
    input_rows = direct.events[valid]
    if input_rows.shape[0] > row_config.max_events:
        raise ValueError("valid input rows exceed the fixed event horizon.")
    sequence_length = row_config.max_events + MAX_GRID_SIZE
    events = np.zeros(
        (sequence_length, row_config.input_width + DECODE_FEATURE_WIDTH),
        dtype=np.float32,
    )
    input_start = row_config.max_events - input_rows.shape[0]
    events[input_start : row_config.max_events, : row_config.input_width] = input_rows
    decode = events[row_config.max_events :]
    decode[:, row_config.input_width] = 1.0
    decode[:, row_config.input_width + 1 :] = np.eye(
        MAX_GRID_SIZE, dtype=np.float32
    )

    target_rows = np.zeros((sequence_length, MAX_GRID_SIZE), dtype=np.int32)
    target_cell_mask = np.zeros(
        (sequence_length, MAX_GRID_SIZE), dtype=np.float32
    )
    target_rows[row_config.max_events :] = direct.target_colors
    target_cell_mask[row_config.max_events :] = direct.target_mask
    target_heights = np.full(
        (sequence_length,), direct.target_height, dtype=np.int32
    )
    target_widths = np.full(
        (sequence_length,), direct.target_width, dtype=np.int32
    )
    decode_mask = np.zeros((sequence_length,), dtype=np.float32)
    decode_mask[row_config.max_events :] = 1.0
    return OnlineEpisode(
        events=events,
        target_rows=target_rows,
        target_cell_mask=target_cell_mask,
        target_heights=target_heights,
        target_widths=target_widths,
        decode_mask=decode_mask,
        task_id=direct.task_id,
    )


def stack_online_episodes(episodes: tuple[OnlineEpisode, ...]) -> OnlineBatch:
    """Stack equal-layout online episodes along a batch axis.

    Parameters
    ----------
    episodes : tuple of OnlineEpisode
        Nonempty tuple with identical static layouts.

    Returns
    -------
    OnlineBatch
        Time-major batch with a common decode mask.
    """

    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    if any(not isinstance(episode, OnlineEpisode) for episode in episodes):
        raise TypeError("Every episode must be an OnlineEpisode.")
    shapes = {episode.events.shape for episode in episodes}
    if len(shapes) != 1:
        raise ValueError("Every episode must have the same event shape.")
    reference_mask = episodes[0].decode_mask
    if any(
        episode.decode_mask.tobytes() != reference_mask.tobytes()
        for episode in episodes[1:]
    ):
        raise ValueError("Every episode must use the same decode mask.")
    return OnlineBatch(
        events=np.stack([episode.events for episode in episodes], axis=1),
        target_rows=np.stack([episode.target_rows for episode in episodes], axis=1),
        target_cell_mask=np.stack(
            [episode.target_cell_mask for episode in episodes], axis=1
        ),
        target_heights=np.stack(
            [episode.target_heights for episode in episodes], axis=1
        ),
        target_widths=np.stack(
            [episode.target_widths for episode in episodes], axis=1
        ),
        decode_mask=np.array(reference_mask, copy=True),
    )


def repeat_online_batch(batch: OnlineBatch, updates: int) -> OnlineTrainingChunk:
    """Repeat one batch for a deterministic compiled descent probe.

    Parameters
    ----------
    batch : OnlineBatch
        Batch to repeat without changing any bytes.
    updates : int
        Positive update count.

    Returns
    -------
    OnlineTrainingChunk
        Update-major repeated arrays.
    """

    if not isinstance(batch, OnlineBatch):
        raise TypeError("batch must be an OnlineBatch instance.")
    if isinstance(updates, (bool, np.bool_)) or not isinstance(updates, Integral):
        raise TypeError("updates must be a positive integer.")
    count = int(updates)
    if count < 1:
        raise ValueError("updates must be a positive integer.")

    def repeat(array: np.ndarray) -> np.ndarray:
        return np.repeat(array[None, ...], count, axis=0)

    return OnlineTrainingChunk(
        events=repeat(batch.events),
        target_rows=repeat(batch.target_rows),
        target_cell_mask=repeat(batch.target_cell_mask),
        target_heights=repeat(batch.target_heights),
        target_widths=repeat(batch.target_widths),
        decode_mask=np.array(batch.decode_mask, copy=True),
    )


def sample_online_training_chunk(
    catalog: tuple[ArcTask, ...],
    row_config: RowEventConfig,
    rng: brainstate.random.RandomState,
    *,
    updates: int,
    batch_size: int,
    augment: bool,
) -> OnlineTrainingChunk:
    """Sample and host-encode one update-major online training chunk.

    Parameters
    ----------
    catalog : tuple of ArcTask
        Target-isolated training tasks with one supervised query each.
    row_config : RowEventConfig
        Fixed lossless row-event layout.
    rng : brainstate.random.RandomState
        Sole task-sampling and augmentation random stream.
    updates, batch_size : int
        Positive chunk dimensions.
    augment : bool
        Whether to apply target-preserving training augmentation.

    Returns
    -------
    OnlineTrainingChunk
        Update-major model inputs and separated training signals.
    """

    if not isinstance(catalog, tuple) or not catalog:
        raise ValueError("catalog must be a nonempty tuple.")
    if not isinstance(rng, brainstate.random.RandomState):
        raise TypeError("rng must be a brainstate.random.RandomState.")
    if not isinstance(augment, bool):
        raise TypeError("augment must be boolean.")
    for name, raw in (("updates", updates), ("batch_size", batch_size)):
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Integral):
            raise TypeError(f"{name} must be a positive integer.")
        if int(raw) < 1:
            raise ValueError(f"{name} must be a positive integer.")
    indices = np.asarray(
        rng.randint(0, len(catalog), size=(int(updates), int(batch_size))),
        dtype=np.int32,
    )
    augmentation = AugmentationConfig(
        permute_colors=augment,
        dihedral=augment,
        shuffle_demonstrations=augment,
    )
    batches = []
    for update_indices in indices:
        episodes = []
        for index in update_indices:
            task = catalog[int(index)]
            if augment:
                task = augment_training_task(
                    task, rng, role="train", config=augmentation
                )
            episodes.append(encode_online_episode(task, 0, row_config))
        batches.append(stack_online_episodes(tuple(episodes)))
    return OnlineTrainingChunk(
        events=np.stack([batch.events for batch in batches]),
        target_rows=np.stack([batch.target_rows for batch in batches]),
        target_cell_mask=np.stack(
            [batch.target_cell_mask for batch in batches]
        ),
        target_heights=np.stack([batch.target_heights for batch in batches]),
        target_widths=np.stack([batch.target_widths for batch in batches]),
        decode_mask=np.array(batches[0].decode_mask, copy=True),
    )


def evaluation_online_episodes(
    tasks: tuple[ArcTask, ...], row_config: RowEventConfig
) -> tuple[OnlineEpisode, ...]:
    """Encode every scorer query without augmentation or target input.

    Parameters
    ----------
    tasks : tuple of ArcTask
        Ordered tasks whose query targets remain scorer-side.
    row_config : RowEventConfig
        Fixed lossless row-event layout.

    Returns
    -------
    tuple of OnlineEpisode
        Query-ordered target-free model inputs and scorer targets.
    """

    if not isinstance(tasks, tuple) or not tasks:
        raise ValueError("tasks must be a nonempty tuple.")
    episodes = []
    for task in tasks:
        for query_index in range(len(task.test)):
            episodes.append(encode_online_episode(task, query_index, row_config))
    return tuple(episodes)


def evaluate_online_model(
    model: OnlineARCGRU,
    episodes: tuple[OnlineEpisode, ...],
    *,
    trace_decay: float = 2.0 ** (-1.0 / 40.0),
    batch_size: int = 10,
) -> dict[str, object]:
    """Execute target-free PP-prop evolution and score strict pass-at-one.

    Parameters
    ----------
    model : OnlineARCGRU
        Frozen checkpoint-owned model.
    episodes : tuple of OnlineEpisode
        Ordered model inputs with out-of-band scorer targets.
    trace_decay : float, default=2 ** (-1 / 40)
        Evaluation learner trace decay; traces do not feed model outputs.
    batch_size : int, default=10
        Positive static evaluation batch size.

    Returns
    -------
    dict
        Exact task score, memberships, candidates, and candidate digest.
    """

    if not isinstance(model, OnlineARCGRU):
        raise TypeError("model must be an OnlineARCGRU instance.")
    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be a positive integer.")
    static_batch = min(int(batch_size), len(episodes))
    if static_batch < 1:
        raise ValueError("batch_size must be a positive integer.")
    padding = (-len(episodes)) % static_batch
    padded = episodes + (episodes[-1],) * padding
    batches = tuple(
        stack_online_episodes(padded[index : index + static_batch])
        for index in range(0, len(padded), static_batch)
    )
    brainstate.nn.init_all_states(model, batch_size=static_batch)
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((static_batch, model.config.input_width), dtype=jnp.float32),
        batch_size=static_batch,
        vmap=False,
        decay_or_rank=trace_decay,
        vjp_method="single-step",
    )

    def run_batch(events):
        brainstate.nn.reset_all_states(model, batch_size=static_batch)
        learner.reset_state(batch_size=static_batch)
        return learner.etrace_evolve(events, return_outputs=True)

    @brainstate.transform.jit
    def run_all(events):
        return brainstate.transform.for_loop(run_batch, events)

    raw_outputs = np.asarray(
        run_all(jnp.asarray(np.stack([batch.events for batch in batches])))
    )
    ordered_outputs = raw_outputs.transpose(0, 2, 1, 3).reshape(
        -1, raw_outputs.shape[1], raw_outputs.shape[3]
    )[: len(episodes)]
    dependencies = tuple(
        ".".join(map(str, path)) for path in model.states(brainstate.ParamState)
    )
    candidates = []
    predictions = []
    targets = []
    task_ids = []
    for episode, outputs in zip(episodes, ordered_outputs, strict=True):
        candidate = decode_online_outputs(outputs, episode.decode_mask, dependencies)
        candidates.append(candidate)
        predictions.append(candidate["grid"])
        selected_rows = episode.target_rows[episode.decode_mask.astype(np.bool_)]
        height = int(episode.target_heights[-1]) + 1
        width = int(episode.target_widths[-1]) + 1
        targets.append(selected_rows[:height, :width].tolist())
        task_ids.append(episode.task_id)
    strict_score = strict_task_pass_at_1(predictions, targets, task_ids)
    candidate_bytes = first_prediction_bytes(candidates)
    return {
        **strict_score,
        "query_count": len(episodes),
        "task_count": len(set(task_ids)),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_size": len(candidate_bytes),
        "candidates": candidates,
    }


def online_step_loss(
    output: jnp.ndarray,
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    target_height: jnp.ndarray,
    target_width: jnp.ndarray,
) -> jnp.ndarray:
    """Compute one decode-step colour and shape objective.

    Parameters
    ----------
    output : jax.Array
        Batched fixed-layout model logits.
    target_row : jax.Array
        Batched integer colours for all 30 columns.
    target_cell_mask : jax.Array
        Batched mask selecting true target columns on this row.
    target_height, target_width : jax.Array
        Batched zero-based output dimension labels.

    Returns
    -------
    jax.Array
        Scalar mean of colour, height, and width losses.
    """

    row_logits, height_logits, width_logits = split_step_logits(output)
    per_cell = optax.softmax_cross_entropy_with_integer_labels(
        row_logits, target_row
    )
    mask = jnp.asarray(target_cell_mask, dtype=per_cell.dtype)
    color_loss = jnp.sum(per_cell * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return (color_loss + height_loss + width_loss) / 3.0


def _parameter_group(path: tuple[object, ...]) -> str:
    root = str(path[0])
    if root == "recurrent":
        return "recurrent"
    if root == "row_color_head":
        return "row_color"
    if root == "height_head":
        return "height"
    if root == "width_head":
        return "width"
    raise ValueError(f"Parameter path {path!r} has no online parameter group.")


def parameter_arrays(model: OnlineARCGRU) -> dict[str, np.ndarray]:
    """Return deterministic byte arrays grouped by answer-path role.

    Parameters
    ----------
    model : OnlineARCGRU
        Model whose ordered checkpoint leaves are grouped.

    Returns
    -------
    dict
        One contiguous ``uint8`` array for each parameter group.
    """

    if not isinstance(model, OnlineARCGRU):
        raise TypeError("model must be an OnlineARCGRU instance.")
    chunks: dict[str, list[bytes]] = {
        "recurrent": [],
        "row_color": [],
        "height": [],
        "width": [],
    }
    for path, state in model.states(brainstate.ParamState).items():
        group = _parameter_group(path)
        chunks[group].extend(
            np.ascontiguousarray(np.asarray(leaf)).tobytes()
            for leaf in jax.tree.leaves(state.value)
        )
    return {
        name: np.frombuffer(b"".join(values), dtype=np.uint8).copy()
        for name, values in chunks.items()
    }


def parameter_digest(model: OnlineARCGRU) -> str:
    """Hash ordered parameter paths, metadata, and bytes.

    Parameters
    ----------
    model : OnlineARCGRU
        Model whose current checkpoint-owned parameters are bound.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """

    if not isinstance(model, OnlineARCGRU):
        raise TypeError("model must be an OnlineARCGRU instance.")
    digest = hashlib.sha256()
    for path, state in model.states(brainstate.ParamState).items():
        digest.update(".".join(map(str, path)).encode("utf-8"))
        for leaf in jax.tree.leaves(state.value):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class OnlinePPPropTrainer:
    """Train the row decoder with compiled single-step PP-prop.

    Parameters
    ----------
    model : OnlineARCGRU
        Model updated in place.
    batch_size : int
        Static batch size used by model and eligibility states.
    learning_rate : float, default=0.001
        Positive Adam learning rate.
    trace_decay : float, default=2 ** (-1 / 40)
        PP-prop exponential eligibility-trace decay in ``(0, 1]``.
    """

    algorithm = "pp_prop"
    vjp_method = "single-step"

    def __init__(
        self,
        model: OnlineARCGRU,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
        trace_decay: float = 2.0 ** (-1.0 / 40.0),
    ):
        if not isinstance(model, OnlineARCGRU):
            raise TypeError("model must be an OnlineARCGRU instance.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        for name, raw in (
            ("learning_rate", learning_rate),
            ("trace_decay", trace_decay),
        ):
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"{name} must be a positive finite real.")
            value = float(raw)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite real.")
            if name == "trace_decay" and value > 1.0:
                raise ValueError("trace_decay must be at most 1.0.")
            setattr(self, name, value)
        self.model = model
        self.weights = model.states(brainstate.ParamState)
        brainstate.nn.init_all_states(model, batch_size=self.batch_size)
        self.learner = braintrace.compile(
            model,
            braintrace.pp_prop,
            jnp.zeros(
                (self.batch_size, model.config.input_width), dtype=jnp.float32
            ),
            batch_size=self.batch_size,
            vmap=False,
            decay_or_rank=self.trace_decay,
            vjp_method=self.vjp_method,
        )
        self.optimizer = braintools.optim.Adam(lr=self.learning_rate)
        self.optimizer.register_trainable_weights(self.weights)
        grouped: dict[str, list[tuple[object, ...]]] = {
            "recurrent": [],
            "row_color": [],
            "height": [],
            "width": [],
        }
        for path in self.weights:
            grouped[_parameter_group(path)].append(path)
        self.groups = {name: tuple(paths) for name, paths in grouped.items()}
        if any(not paths for paths in self.groups.values()):
            raise ValueError("Every online parameter group must be nonempty.")
        self._train_many = self._make_train_many()

    def _reset(self) -> None:
        brainstate.nn.reset_all_states(self.model, batch_size=self.batch_size)
        self.learner.reset_state(batch_size=self.batch_size)

    def _make_train_many(self):
        groups = self.groups
        learner = self.learner
        optimizer = self.optimizer

        def gradient_norm(gradients, paths):
            leaves = [
                leaf
                for path in paths
                for leaf in jax.tree.leaves(gradients[path])
            ]
            return jnp.sqrt(
                sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves)
            )

        def train_one(events, rows, cell_mask, heights, widths, decode_mask):
            self._reset()

            def step_loss(event, row, mask, height, width):
                return online_step_loss(learner(event), row, mask, height, width)

            gradients, objective = learner.etrace_grad(
                events,
                rows,
                cell_mask,
                heights,
                widths,
                step_fn=step_loss,
                mask=decode_mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            norms = jnp.stack(
                [gradient_norm(gradients, groups[name]) for name in groups]
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, 1.0))
            return objective, norms

        @brainstate.transform.jit
        def train_many(events, rows, cell_mask, heights, widths, decode_mask):
            def body(update_values):
                return train_one(*update_values, decode_mask)

            return brainstate.transform.for_loop(
                body, (events, rows, cell_mask, heights, widths)
            )

        return train_many

    def train_chunk(
        self, chunk: OnlineTrainingChunk
    ) -> tuple[jax.Array, dict[str, float]]:
        """Run a compiled update chunk and report group gradient maxima.

        Parameters
        ----------
        chunk : OnlineTrainingChunk
            Update-major target-isolated training arrays.

        Returns
        -------
        losses : jax.Array
            One scalar objective per optimizer update.
        gradient_norms : dict
            Maximum observed norm for each parameter group.
        """

        if not isinstance(chunk, OnlineTrainingChunk):
            raise TypeError("chunk must be an OnlineTrainingChunk instance.")
        if chunk.events.shape[2] != self.batch_size:
            raise ValueError(
                f"chunk batch axis must equal trainer batch_size {self.batch_size}."
            )
        losses, norms = self._train_many(
            jnp.asarray(chunk.events),
            jnp.asarray(chunk.target_rows),
            jnp.asarray(chunk.target_cell_mask),
            jnp.asarray(chunk.target_heights),
            jnp.asarray(chunk.target_widths),
            jnp.asarray(chunk.decode_mask),
        )
        observed = np.asarray(norms)
        return losses, {
            name: float(np.max(observed[:, index]))
            for index, name in enumerate(self.groups)
        }


def save_online_checkpoint(model: OnlineARCGRU, path: pathlib.Path) -> str:
    """Save an exact-schema online-model checkpoint.

    Parameters
    ----------
    model : OnlineARCGRU
        Model whose parameters are saved.
    path : pathlib.Path
        Destination ``.npz`` path.

    Returns
    -------
    str
        Ordered parameter digest.
    """

    if not isinstance(model, OnlineARCGRU):
        raise TypeError("model must be an OnlineARCGRU instance.")
    path = pathlib.Path(path)
    arrays: dict[str, np.ndarray] = {}
    leaves_metadata = []
    leaf_number = 0
    for state_path, state in model.states(brainstate.ParamState).items():
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


def load_online_checkpoint(
    path: pathlib.Path,
) -> tuple[OnlineARCGRU, dict[str, object]]:
    """Load an exact-schema online-model checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        Existing checkpoint path.

    Returns
    -------
    model : OnlineARCGRU
        Restored model.
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
                raise ValueError("checkpoint metadata schema is invalid.")
            model = OnlineARCGRU(OnlineModelConfig(**architecture))
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
                restored = []
                for index, original in enumerate(jax.tree.leaves(state.value)):
                    if cursor >= len(leaves_metadata):
                        raise ValueError("checkpoint is missing parameter leaves.")
                    item = leaves_metadata[cursor]
                    if not isinstance(item, dict):
                        raise ValueError("checkpoint leaf metadata is invalid.")
                    key = item.get("key")
                    array = np.asarray(archive[key])
                    if not np.all(np.isfinite(array)):
                        raise ValueError("checkpoint parameter leaves must be finite.")
                    if (
                        item.get("state_path") != expected_path
                        or item.get("tree_leaf_index") != index
                        or item.get("shape") != list(np.asarray(original).shape)
                        or item.get("dtype") != np.asarray(original).dtype.str
                        or list(array.shape) != item.get("shape")
                        or array.dtype.str != item.get("dtype")
                    ):
                        raise ValueError(
                            "checkpoint parameter schema does not match model."
                        )
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


def decode_online_outputs(
    outputs: np.ndarray,
    decode_mask: np.ndarray,
    parameter_dependencies: tuple[str, ...],
) -> dict[str, object]:
    """Decode fixed online row outputs into one greedy ARC candidate.

    Parameters
    ----------
    outputs : numpy.ndarray
        Per-step model logits shaped ``(time, OUTPUT_WIDTH)``.
    decode_mask : numpy.ndarray
        One-dimensional mask selecting exactly 30 decode steps.
    parameter_dependencies : tuple of str
        Ordered checkpoint paths that determine the logits.

    Returns
    -------
    dict
        One fixed greedy first candidate.
    """

    values = np.asarray(outputs)
    mask = np.asarray(decode_mask, dtype=np.bool_)
    if values.ndim != 2 or mask.shape != (values.shape[0],):
        raise ValueError("outputs and decode_mask shapes are incompatible.")
    selected = values[mask]
    if selected.shape[0] != MAX_GRID_SIZE:
        raise ValueError("decode_mask must select exactly 30 steps.")
    if not np.all(np.isfinite(selected)):
        raise ValueError("decode outputs must be finite.")
    row_logits, height_logits, width_logits = split_step_logits(jnp.asarray(selected))
    colors = np.asarray(jnp.argmax(row_logits, axis=-1), dtype=np.int32)
    height = int(np.argmax(np.asarray(height_logits).mean(axis=0))) + 1
    width = int(np.argmax(np.asarray(width_logits).mean(axis=0))) + 1
    return {
        "rank": 1,
        "height": height,
        "width": width,
        "grid": colors[:height, :width].tolist(),
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "parameter_dependencies": list(parameter_dependencies),
        "proposal_source": "online_model_logits",
        "selection_role": "greedy_argmax",
        "ranking_source": "none_single_greedy_candidate",
        "answer_head_version": "online_row_decoder_v20",
    }
