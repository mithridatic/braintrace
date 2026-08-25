"""Target-isolated sequence packing and single-step PP-prop ARC training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
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
    COLOR_COUNT,
    MAX_GRID_SIZE,
    QUERY_PATCH_CHANNELS,
    QUERY_PATCH_SIDE,
    QUERY_PATCH_WIDTH,
    OnlineARCVanillaRNN,
    OnlineModelConfig,
    split_step_logits,
)

from examples.pp_prop.latent_workspace_task import (
    ArcTask,
    AugmentationConfig,
    RowEventConfig,
    augment_training_task,
)

EDIT_CELL_MULTIPLIER = 2.0


def online_decode_feature_width(row_config: RowEventConfig) -> int:
    """Return the V39 decode flag, row index, and query-patch width.

    Parameters
    ----------
    row_config : RowEventConfig
        Bound base-event capacity.

    Returns
    -------
    int
        Number of features appended to every base row event.
    """

    if not isinstance(row_config, RowEventConfig):
        raise TypeError("row_config must be a RowEventConfig instance.")
    return 1 + MAX_GRID_SIZE + row_config.max_grid_size * QUERY_PATCH_WIDTH


@dataclass(frozen=True)
class OnlineEpisode:
    """Hold one target-free online sequence and scorer-side training signals.

    Parameters
    ----------
    events : numpy.ndarray
        Fixed sequence of left padding, lossless input rows, and decode tokens.
    target_rows, target_cell_mask, class_weights : numpy.ndarray
        Per-step scorer-side colour labels, cell masks, and bounded colour
        weights. None are model inputs.
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
    class_weights: np.ndarray
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
    target_rows, target_cell_mask, class_weights : numpy.ndarray
        Time-major scorer-side colour labels, cell masks, and colour weights.
    target_heights, target_widths : numpy.ndarray
        Time-major shape labels.
    decode_mask : numpy.ndarray
        Common one-dimensional decode-step mask.
    """

    events: np.ndarray
    target_rows: np.ndarray
    target_cell_mask: np.ndarray
    class_weights: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    decode_mask: np.ndarray


@dataclass(frozen=True)
class OnlineTrainingChunk:
    """Hold update-major batches for one compiled optimizer loop.

    Parameters
    ----------
    events, target_rows, target_cell_mask, class_weights, target_heights,
    target_widths
        Arrays with leading update and time dimensions.
    decode_mask : numpy.ndarray
        Common one-dimensional decode-step mask.
    """

    events: np.ndarray
    target_rows: np.ndarray
    target_cell_mask: np.ndarray
    class_weights: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    decode_mask: np.ndarray


def target_color_weights(
    target_colors: np.ndarray, target_mask: np.ndarray
) -> np.ndarray:
    """Compute bounded inverse-frequency weights from one training target.

    Parameters
    ----------
    target_colors : numpy.ndarray
        Integer ARC colours in a padded target grid.
    target_mask : numpy.ndarray
        Mask selecting the valid target cells.

    Returns
    -------
    numpy.ndarray
        Ten ``float32`` weights; absent colours have weight zero.
    """

    colors = np.asarray(target_colors)
    mask = np.asarray(target_mask, dtype=np.bool_)
    if colors.shape != mask.shape or colors.ndim != 2:
        raise ValueError("target_colors and target_mask must have the same 2-D shape.")
    if not np.issubdtype(colors.dtype, np.integer):
        raise TypeError("target_colors must contain integers.")
    valid_colors = colors[mask]
    if valid_colors.size < 1 or np.any((valid_colors < 0) | (valid_colors >= 10)):
        raise ValueError("target_mask must select valid ARC colours.")
    counts = np.bincount(valid_colors, minlength=10).astype(np.float32)
    weights = np.zeros((10,), dtype=np.float32)
    present = counts > 0.0
    weights[present] = np.clip(
        np.sqrt(float(valid_colors.size) / counts[present]), 0.5, 4.0
    )
    return weights


def query_replay_rows(
    encoded_events: np.ndarray, row_config: RowEventConfig
) -> np.ndarray:
    """Build 30 target-free decode inputs from encoded query rows.

    Parameters
    ----------
    encoded_events : numpy.ndarray
        Lossless base row events before left-padding or decode augmentation.
    row_config : RowEventConfig
        Static base-event feature layout.

    Returns
    -------
    numpy.ndarray
        Thirty base-width query replay rows. Real query rows are byte-exact;
        later rows retain query dimensions and row position without cells.
    """

    values = np.asarray(encoded_events, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != row_config.input_width:
        raise ValueError("encoded_events must match the row-event width.")
    is_query = (
        (values[:, row_config.valid_slice.start] == 1.0)
        & (values[:, row_config.phase_slice.start + 1] == 1.0)
    )
    query_rows = values[is_query]
    if query_rows.shape[0] < 1 or query_rows.shape[0] > MAX_GRID_SIZE:
        raise ValueError("encoded_events must contain 1..30 query rows.")
    row_ids = np.argmax(query_rows[:, row_config.row_index_slice], axis=-1)
    if not np.array_equal(row_ids, np.arange(query_rows.shape[0])):
        raise ValueError("query rows must use consecutive ordered row indices.")

    replay = np.zeros((MAX_GRID_SIZE, row_config.input_width), dtype=np.float32)
    replay[: query_rows.shape[0]] = query_rows
    template = query_rows[0]
    for row_index in range(query_rows.shape[0], MAX_GRID_SIZE):
        row = replay[row_index]
        row[row_config.valid_slice] = 1.0
        row[row_config.phase_slice.start + 1] = 1.0
        row[row_config.normalized_slice.start] = (
            row_index + 1
        ) / row_config.max_grid_size
        row[row_config.normalized_slice.start + 1 : row_config.normalized_slice.start + 3] = (
            template[
                row_config.normalized_slice.start + 1 : row_config.normalized_slice.start
                + 3
            ]
        )
        if row_index < row_config.max_grid_size:
            row[row_config.row_index_slice.start + row_index] = 1.0
        row[row_config.input_height_slice] = template[row_config.input_height_slice]
        row[row_config.input_width_slice] = template[row_config.input_width_slice]
    return replay


def query_patch_rows(
    encoded_events: np.ndarray, row_config: RowEventConfig
) -> np.ndarray:
    """Encode ordered 3-by-3 query neighbourhoods for every decode row.

    Parameters
    ----------
    encoded_events : numpy.ndarray
        Lossless base row events containing a target-free query grid.
    row_config : RowEventConfig
        Static base-event feature layout.

    Returns
    -------
    numpy.ndarray
        Thirty rows of flattened per-column colour/validity patches.
    """

    replay = query_replay_rows(encoded_events, row_config)
    grid_size = row_config.max_grid_size
    colors = replay[:grid_size, row_config.input_color_slice].reshape(
        grid_size, grid_size, COLOR_COUNT
    )
    valid = replay[:grid_size, row_config.input_mask_slice][..., None]
    grid = np.concatenate((colors, valid), axis=-1)
    padding = QUERY_PATCH_SIDE // 2
    padded = np.pad(
        grid,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="constant",
    )
    offsets = [
        padded[
            row_offset : row_offset + grid_size,
            column_offset : column_offset + grid_size,
        ]
        for row_offset in range(QUERY_PATCH_SIDE)
        for column_offset in range(QUERY_PATCH_SIDE)
    ]
    patches = np.concatenate(offsets, axis=-1)
    if patches.shape != (
        grid_size,
        grid_size,
        QUERY_PATCH_WIDTH,
    ) or patches.shape[-1] != QUERY_PATCH_SIDE**2 * QUERY_PATCH_CHANNELS:
        raise RuntimeError("query patch layout is inconsistent.")
    rows = np.zeros(
        (MAX_GRID_SIZE, grid_size * QUERY_PATCH_WIDTH), dtype=np.float32
    )
    rows[:grid_size] = patches.reshape(grid_size, -1)
    return rows


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
        (
            sequence_length,
            row_config.input_width + online_decode_feature_width(row_config),
        ),
        dtype=np.float32,
    )
    input_start = row_config.max_events - input_rows.shape[0]
    events[input_start : row_config.max_events, : row_config.input_width] = input_rows
    decode = events[row_config.max_events :]
    decode[:, : row_config.input_width] = query_replay_rows(
        direct.events, row_config
    )
    decode[:, row_config.input_width] = 1.0
    row_index_start = row_config.input_width + 1
    row_index_stop = row_index_start + MAX_GRID_SIZE
    decode[:, row_index_start:row_index_stop] = np.eye(
        MAX_GRID_SIZE, dtype=np.float32
    )
    decode[:, row_index_stop:] = query_patch_rows(direct.events, row_config)

    target_rows = np.zeros((sequence_length, MAX_GRID_SIZE), dtype=np.int32)
    target_cell_mask = np.zeros(
        (sequence_length, MAX_GRID_SIZE), dtype=np.float32
    )
    target_rows[row_config.max_events :] = direct.target_colors
    target_cell_mask[row_config.max_events :] = direct.target_mask
    weights = target_color_weights(direct.target_colors, direct.target_mask)
    class_weights = np.repeat(weights[None, :], sequence_length, axis=0)
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
        class_weights=class_weights,
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
        class_weights=np.stack(
            [episode.class_weights for episode in episodes], axis=1
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
        class_weights=repeat(batch.class_weights),
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
        class_weights=np.stack([batch.class_weights for batch in batches]),
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
    model: OnlineARCVanillaRNN,
    episodes: tuple[OnlineEpisode, ...],
    *,
    trace_decay: float = 2.0 ** (-1.0 / 40.0),
    batch_size: int = 10,
) -> dict[str, object]:
    """Execute target-free PP-prop evolution and score strict pass-at-one.

    Parameters
    ----------
    model : OnlineARCVanillaRNN
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

    if not isinstance(model, OnlineARCVanillaRNN):
        raise TypeError("model must be an OnlineARCVanillaRNN instance.")
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
        candidate = decode_hierarchical_online_outputs(
            outputs, episode.decode_mask, dependencies
        )
        candidates.append(candidate)
        predictions.append(candidate["grid"])
        selected_rows = episode.target_rows[episode.decode_mask.astype(np.bool_)]
        height = int(episode.target_heights[-1]) + 1
        width = int(episode.target_widths[-1]) + 1
        targets.append(selected_rows[:height, :width].tolist())
        task_ids.append(episode.task_id)
    strict_score = strict_task_pass_at_1(predictions, targets, task_ids)
    candidate_bytes = first_prediction_bytes(candidates)
    shape_pairs = [
        (np.asarray(prediction), np.asarray(target))
        for prediction, target in zip(predictions, targets, strict=True)
        if np.asarray(prediction).shape == np.asarray(target).shape
    ]
    foreground_total = sum(int(np.count_nonzero(target != 0)) for _, target in shape_pairs)
    background_total = sum(int(np.count_nonzero(target == 0)) for _, target in shape_pairs)
    foreground_correct = sum(
        int(np.count_nonzero((prediction == target) & (target != 0)))
        for prediction, target in shape_pairs
    )
    background_correct = sum(
        int(np.count_nonzero((prediction == target) & (target == 0)))
        for prediction, target in shape_pairs
    )
    predicted_colors = Counter(
        int(color)
        for prediction in predictions
        for color in np.asarray(prediction).reshape(-1)
    )
    return {
        **strict_score,
        "query_count": len(episodes),
        "task_count": len(set(task_ids)),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_size": len(candidate_bytes),
        "diagnostics": {
            "shape_exact_count": len(shape_pairs),
            "foreground_correct": foreground_correct,
            "foreground_total": foreground_total,
            "foreground_accuracy": foreground_correct / max(foreground_total, 1),
            "background_correct": background_correct,
            "background_total": background_total,
            "background_accuracy": background_correct / max(background_total, 1),
            "predicted_color_counts": {
                str(color): count for color, count in sorted(predicted_colors.items())
            },
        },
        "candidates": candidates,
    }


def online_step_loss(
    output: jnp.ndarray,
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    target_height: jnp.ndarray,
    target_width: jnp.ndarray,
    class_weights: jnp.ndarray,
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
    class_weights : jax.Array
        Batched ten-colour weights computed from training targets only.

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
    selected_weights = jnp.take_along_axis(
        class_weights[:, None, :], target_row[..., None], axis=-1
    )[..., 0]
    weighted_mask = mask * selected_weights
    color_loss = jnp.sum(per_cell * weighted_mask) / jnp.maximum(
        jnp.sum(weighted_mask), 1.0
    )
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return (color_loss + height_loss + width_loss) / 3.0


def whole_grid_color_mass(
    target_rows: jnp.ndarray, target_cell_masks: jnp.ndarray
) -> jnp.ndarray:
    """Compute equal whole-grid mass for every present target colour.

    Parameters
    ----------
    target_rows : jax.Array
        Time-major batched integer target rows shaped ``(time, batch, 30)``.
    target_cell_masks : jax.Array
        Matching valid-cell masks.

    Returns
    -------
    jax.Array
        Time-broadcast colour masses shaped ``(time, batch, 10)``.
    """

    if target_rows.shape != target_cell_masks.shape or target_rows.ndim != 3:
        raise ValueError("target_rows and target_cell_masks must match in 3-D.")
    if target_rows.shape[-1] != MAX_GRID_SIZE:
        raise ValueError(f"target row width must be {MAX_GRID_SIZE}.")
    mask = jnp.asarray(target_cell_masks, dtype=jnp.float32)
    color_ids = jnp.arange(COLOR_COUNT, dtype=target_rows.dtype)
    membership = mask[..., None] * (
        target_rows[..., None] == color_ids
    ).astype(mask.dtype)
    counts = jnp.sum(membership, axis=(0, 2))
    present = (counts > 0.0).astype(mask.dtype)
    present_count = jnp.sum(present, axis=-1, keepdims=True)
    per_color = jnp.where(
        counts > 0.0,
        MAX_GRID_SIZE / jnp.maximum(present_count * counts, 1.0),
        0.0,
    )
    return jnp.broadcast_to(per_color[None, ...], (*target_rows.shape[:2], COLOR_COUNT))


def whole_grid_online_step_loss(
    output: jnp.ndarray,
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    target_height: jnp.ndarray,
    target_width: jnp.ndarray,
    color_mass: jnp.ndarray,
) -> jnp.ndarray:
    """Apply fixed whole-grid colour mass at one recurrent decode step.

    Parameters
    ----------
    output : jax.Array
        Batched fixed-layout neural logits.
    target_row : jax.Array
        Batched integer colours for all 30 columns.
    target_cell_mask : jax.Array
        Batched mask selecting true target columns.
    target_height, target_width : jax.Array
        Batched zero-based output dimension labels.
    color_mass : jax.Array
        Batched ten-colour mass computed over the complete target grid.

    Returns
    -------
    jax.Array
        Scalar mean of whole-grid-balanced colour and shape losses.
    """

    if color_mass.shape[-1] != COLOR_COUNT:
        raise ValueError(f"color_mass last dimension must be {COLOR_COUNT}.")
    row_logits, height_logits, width_logits = split_step_logits(output)
    per_cell = optax.softmax_cross_entropy_with_integer_labels(
        row_logits, target_row
    )
    mask = jnp.asarray(target_cell_mask, dtype=per_cell.dtype)
    selected_mass = jnp.take_along_axis(
        color_mass[:, None, :], target_row[..., None], axis=-1
    )[..., 0]
    color_loss = jnp.mean(jnp.sum(per_cell * mask * selected_mass, axis=-1))
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return (color_loss + height_loss + width_loss) / 3.0


def color_dominant_whole_grid_step_loss(
    output: jnp.ndarray,
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    target_height: jnp.ndarray,
    target_width: jnp.ndarray,
    color_mass: jnp.ndarray,
) -> jnp.ndarray:
    """Weight whole-grid colour loss 0.8 and each shape loss 0.1.

    Parameters
    ----------
    output : jax.Array
        Batched fixed-layout neural logits.
    target_row : jax.Array
        Batched integer colours for all 30 columns.
    target_cell_mask : jax.Array
        Batched mask selecting true target columns.
    target_height, target_width : jax.Array
        Batched zero-based output dimension labels.
    color_mass : jax.Array
        Batched ten-colour mass computed over the complete target grid.

    Returns
    -------
    jax.Array
        Scalar fixed-weight colour and shape objective.
    """

    if color_mass.shape[-1] != COLOR_COUNT:
        raise ValueError(f"color_mass last dimension must be {COLOR_COUNT}.")
    row_logits, height_logits, width_logits = split_step_logits(output)
    per_cell = optax.softmax_cross_entropy_with_integer_labels(
        row_logits, target_row
    )
    mask = jnp.asarray(target_cell_mask, dtype=per_cell.dtype)
    selected_mass = jnp.take_along_axis(
        color_mass[:, None, :], target_row[..., None], axis=-1
    )[..., 0]
    color_loss = jnp.mean(jnp.sum(per_cell * mask * selected_mass, axis=-1))
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return 0.8 * color_loss + 0.1 * height_loss + 0.1 * width_loss


def hierarchical_whole_grid_mass(
    target_rows: jnp.ndarray, target_cell_masks: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute balanced whole-grid gate and conditional-colour masses.

    Parameters
    ----------
    target_rows : jax.Array
        Time-major batched integer target rows shaped ``(time, batch, 30)``.
    target_cell_masks : jax.Array
        Matching valid-cell masks.

    Returns
    -------
    gate_mass, color_mass : tuple of jax.Array
        Time-broadcast masses shaped ``(time, batch, 2)`` for background and
        foreground, and ``(time, batch, 9)`` for nonzero colours one to nine.
    """

    if target_rows.shape != target_cell_masks.shape or target_rows.ndim != 3:
        raise ValueError("target_rows and target_cell_masks must match in 3-D.")
    if target_rows.shape[-1] != MAX_GRID_SIZE:
        raise ValueError(f"target row width must be {MAX_GRID_SIZE}.")
    mask = jnp.asarray(target_cell_masks, dtype=jnp.float32)
    gate_targets = (target_rows != 0).astype(target_rows.dtype)
    gate_ids = jnp.arange(2, dtype=target_rows.dtype)
    gate_membership = mask[..., None] * (
        gate_targets[..., None] == gate_ids
    ).astype(mask.dtype)
    gate_counts = jnp.sum(gate_membership, axis=(0, 2))
    gate_present = (gate_counts > 0.0).astype(mask.dtype)
    gate_present_count = jnp.sum(gate_present, axis=-1, keepdims=True)
    per_gate = jnp.where(
        gate_counts > 0.0,
        MAX_GRID_SIZE / jnp.maximum(gate_present_count * gate_counts, 1.0),
        0.0,
    )

    nonzero_ids = jnp.arange(1, COLOR_COUNT, dtype=target_rows.dtype)
    color_membership = mask[..., None] * (
        target_rows[..., None] == nonzero_ids
    ).astype(mask.dtype)
    color_counts = jnp.sum(color_membership, axis=(0, 2))
    color_present = (color_counts > 0.0).astype(mask.dtype)
    color_present_count = jnp.sum(color_present, axis=-1, keepdims=True)
    per_color = jnp.where(
        color_counts > 0.0,
        MAX_GRID_SIZE / jnp.maximum(color_present_count * color_counts, 1.0),
        0.0,
    )
    time = target_rows.shape[0]
    gate_mass = jnp.broadcast_to(per_gate[None, ...], (time, *per_gate.shape))
    color_mass = jnp.broadcast_to(per_color[None, ...], (time, *per_color.shape))
    return gate_mass, color_mass


def sqrt_balanced_hierarchical_mass(
    target_rows: jnp.ndarray, target_cell_masks: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute square-root-balanced gate and equal conditional-colour masses.

    Parameters
    ----------
    target_rows : jax.Array
        Time-major batched integer target rows shaped ``(time, batch, 30)``.
    target_cell_masks : jax.Array
        Matching valid-cell masks.

    Returns
    -------
    gate_mass, color_mass : tuple of jax.Array
        Gate groups receive total mass proportional to the square root of their
        whole-grid counts and summing to 30. Conditional nonzero colours retain
        equal whole-grid mass.
    """

    _, color_mass = hierarchical_whole_grid_mass(target_rows, target_cell_masks)
    mask = jnp.asarray(target_cell_masks, dtype=jnp.float32)
    gate_targets = (target_rows != 0).astype(target_rows.dtype)
    gate_ids = jnp.arange(2, dtype=target_rows.dtype)
    membership = mask[..., None] * (
        gate_targets[..., None] == gate_ids
    ).astype(mask.dtype)
    counts = jnp.sum(membership, axis=(0, 2))
    roots = jnp.sqrt(counts)
    total_roots = jnp.sum(roots, axis=-1, keepdims=True)
    group_totals = jnp.where(
        counts > 0.0,
        MAX_GRID_SIZE * roots / jnp.maximum(total_roots, 1.0),
        0.0,
    )
    per_gate = jnp.where(
        counts > 0.0, group_totals / jnp.maximum(counts, 1.0), 0.0
    )
    gate_mass = jnp.broadcast_to(
        per_gate[None, ...], (target_rows.shape[0], *per_gate.shape)
    )
    return gate_mass, color_mass


def fourth_root_balanced_hierarchical_mass(
    target_rows: jnp.ndarray, target_cell_masks: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute fourth-root gate and equal conditional-colour masses.

    Parameters
    ----------
    target_rows : jax.Array
        Time-major batched integer target rows shaped ``(time, batch, 30)``.
    target_cell_masks : jax.Array
        Matching valid-cell masks.

    Returns
    -------
    gate_mass, color_mass : tuple of jax.Array
        Gate groups receive total mass proportional to the fourth root of their
        whole-grid counts and summing to 30. Conditional nonzero colours retain
        equal whole-grid mass.
    """

    _, color_mass = hierarchical_whole_grid_mass(target_rows, target_cell_masks)
    mask = jnp.asarray(target_cell_masks, dtype=jnp.float32)
    gate_targets = (target_rows != 0).astype(target_rows.dtype)
    gate_ids = jnp.arange(2, dtype=target_rows.dtype)
    membership = mask[..., None] * (
        gate_targets[..., None] == gate_ids
    ).astype(mask.dtype)
    counts = jnp.sum(membership, axis=(0, 2))
    fourth_roots = jnp.sqrt(jnp.sqrt(counts))
    total_fourth_roots = jnp.sum(fourth_roots, axis=-1, keepdims=True)
    group_totals = jnp.where(
        counts > 0.0,
        MAX_GRID_SIZE
        * fourth_roots
        / jnp.maximum(total_fourth_roots, 1.0),
        0.0,
    )
    per_gate = jnp.where(
        counts > 0.0, group_totals / jnp.maximum(counts, 1.0), 0.0
    )
    gate_mass = jnp.broadcast_to(
        per_gate[None, ...], (target_rows.shape[0], *per_gate.shape)
    )
    return gate_mass, color_mass


def hierarchical_whole_grid_step_loss(
    output: jnp.ndarray,
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    target_height: jnp.ndarray,
    target_width: jnp.ndarray,
    gate_mass: jnp.ndarray,
    color_mass: jnp.ndarray,
) -> jnp.ndarray:
    """Score a binary foreground gate and conditional nonzero colour.

    Parameters
    ----------
    output : jax.Array
        Batched fixed-layout neural logits.
    target_row : jax.Array
        Batched integer colours for all 30 columns.
    target_cell_mask : jax.Array
        Batched mask selecting true target columns.
    target_height, target_width : jax.Array
        Batched zero-based output dimension labels.
    gate_mass : jax.Array
        Batched background/foreground mass computed over the complete grid.
    color_mass : jax.Array
        Batched mass for conditional nonzero colours one through nine.

    Returns
    -------
    jax.Array
        Scalar objective with coefficients 0.4, 0.4, 0.1, and 0.1.
    """

    if gate_mass.shape[-1] != 2:
        raise ValueError("gate_mass last dimension must be 2.")
    if color_mass.shape[-1] != COLOR_COUNT - 1:
        raise ValueError(f"color_mass last dimension must be {COLOR_COUNT - 1}.")
    row_logits, height_logits, width_logits = split_step_logits(output)
    mask = jnp.asarray(target_cell_mask, dtype=row_logits.dtype)
    gate_targets = (target_row != 0).astype(jnp.int32)
    gate_per_cell = optax.sigmoid_binary_cross_entropy(
        row_logits[..., 0], gate_targets
    )
    selected_gate_mass = jnp.take_along_axis(
        gate_mass[:, None, :], gate_targets[..., None], axis=-1
    )[..., 0]
    gate_loss = jnp.mean(
        jnp.sum(gate_per_cell * mask * selected_gate_mass, axis=-1)
    )

    conditional_targets = jnp.clip(target_row - 1, 0, COLOR_COUNT - 2)
    color_per_cell = optax.softmax_cross_entropy_with_integer_labels(
        row_logits[..., 1:], conditional_targets
    )
    selected_color_mass = jnp.take_along_axis(
        color_mass[:, None, :], conditional_targets[..., None], axis=-1
    )[..., 0]
    foreground_mask = mask * gate_targets.astype(mask.dtype)
    color_loss = jnp.mean(
        jnp.sum(color_per_cell * foreground_mask * selected_color_mass, axis=-1)
    )
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return (
        0.4 * gate_loss
        + 0.4 * color_loss
        + 0.1 * height_loss
        + 0.1 * width_loss
    )


def edit_cell_weights(
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    query_row: jnp.ndarray,
    query_cell_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return fixed training weights for query-to-target cell edits.

    Parameters
    ----------
    target_row, query_row : jax.Array
        Matching integer target and query colours.
    target_cell_mask, query_cell_mask : jax.Array
        Matching target and query validity masks.

    Returns
    -------
    jax.Array
        Zero outside the target, one for unchanged valid query cells, and
        :data:`EDIT_CELL_MULTIPLIER` for changed or newly created cells.

    Raises
    ------
    ValueError
        If all four inputs do not have matching shapes.
    """

    shapes = {
        target_row.shape,
        target_cell_mask.shape,
        query_row.shape,
        query_cell_mask.shape,
    }
    if len(shapes) != 1:
        raise ValueError("target/query rows and masks must have matching shapes.")
    target_valid = jnp.asarray(target_cell_mask) > 0.5
    query_valid = jnp.asarray(query_cell_mask) > 0.5
    edited = jnp.logical_or(~query_valid, target_row != query_row)
    weights = jnp.where(edited, EDIT_CELL_MULTIPLIER, 1.0)
    return jnp.where(target_valid, weights, 0.0).astype(jnp.float32)


def edit_weighted_hierarchical_step_loss(
    output: jnp.ndarray,
    target_row: jnp.ndarray,
    target_cell_mask: jnp.ndarray,
    query_row: jnp.ndarray,
    query_cell_mask: jnp.ndarray,
    target_height: jnp.ndarray,
    target_width: jnp.ndarray,
    gate_mass: jnp.ndarray,
    color_mass: jnp.ndarray,
) -> jnp.ndarray:
    """Score hierarchical logits with twofold training-only edit emphasis.

    Parameters
    ----------
    output : jax.Array
        Batched fixed-layout neural logits.
    target_row, query_row : jax.Array
        Batched target and corresponding replayed query colours.
    target_cell_mask, query_cell_mask : jax.Array
        Batched target and query validity masks.
    target_height, target_width : jax.Array
        Batched zero-based output dimension labels.
    gate_mass : jax.Array
        Batched fourth-root background/foreground mass.
    color_mass : jax.Array
        Batched conditional nonzero-colour mass.

    Returns
    -------
    jax.Array
        Scalar edit-weighted hierarchical objective.
    """

    if gate_mass.shape[-1] != 2:
        raise ValueError("gate_mass last dimension must be 2.")
    if color_mass.shape[-1] != COLOR_COUNT - 1:
        raise ValueError(f"color_mass last dimension must be {COLOR_COUNT - 1}.")
    row_logits, height_logits, width_logits = split_step_logits(output)
    weights = edit_cell_weights(
        target_row, target_cell_mask, query_row, query_cell_mask
    ).astype(row_logits.dtype)
    gate_targets = (target_row != 0).astype(jnp.int32)
    gate_per_cell = optax.sigmoid_binary_cross_entropy(
        row_logits[..., 0], gate_targets
    )
    selected_gate_mass = jnp.take_along_axis(
        gate_mass[:, None, :], gate_targets[..., None], axis=-1
    )[..., 0]
    gate_loss = jnp.mean(jnp.sum(gate_per_cell * weights * selected_gate_mass, axis=-1))

    conditional_targets = jnp.clip(target_row - 1, 0, COLOR_COUNT - 2)
    color_per_cell = optax.softmax_cross_entropy_with_integer_labels(
        row_logits[..., 1:], conditional_targets
    )
    selected_color_mass = jnp.take_along_axis(
        color_mass[:, None, :], conditional_targets[..., None], axis=-1
    )[..., 0]
    foreground_weights = weights * gate_targets.astype(weights.dtype)
    color_loss = jnp.mean(
        jnp.sum(
            color_per_cell * foreground_weights * selected_color_mass, axis=-1
        )
    )
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return (
        0.4 * gate_loss
        + 0.4 * color_loss
        + 0.1 * height_loss
        + 0.1 * width_loss
    )


def _parameter_group(path: tuple[object, ...]) -> str:
    root = str(path[0])
    if root == "recurrent":
        return "recurrent"
    if root == "cell_color_head":
        return "row_color"
    if root == "routing_query":
        return "row_color"
    if root == "height_head":
        return "height"
    if root == "width_head":
        return "width"
    raise ValueError(f"Parameter path {path!r} has no online parameter group.")


def parameter_arrays(model: OnlineARCVanillaRNN) -> dict[str, np.ndarray]:
    """Return deterministic byte arrays grouped by answer-path role.

    Parameters
    ----------
    model : OnlineARCVanillaRNN
        Model whose ordered checkpoint leaves are grouped.

    Returns
    -------
    dict
        One contiguous ``uint8`` array for each parameter group.
    """

    if not isinstance(model, OnlineARCVanillaRNN):
        raise TypeError("model must be an OnlineARCVanillaRNN instance.")
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


def parameter_digest(model: OnlineARCVanillaRNN) -> str:
    """Hash ordered parameter paths, metadata, and bytes.

    Parameters
    ----------
    model : OnlineARCVanillaRNN
        Model whose current checkpoint-owned parameters are bound.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """

    if not isinstance(model, OnlineARCVanillaRNN):
        raise TypeError("model must be an OnlineARCVanillaRNN instance.")
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
    model : OnlineARCVanillaRNN
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
    loss_version = "twofold_edit_fourth_root_hierarchical_v40"

    def __init__(
        self,
        model: OnlineARCVanillaRNN,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
        trace_decay: float = 2.0 ** (-1.0 / 40.0),
    ):
        if not isinstance(model, OnlineARCVanillaRNN):
            raise TypeError("model must be an OnlineARCVanillaRNN instance.")
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

        def train_one(
            events,
            rows,
            cell_mask,
            class_weights,
            heights,
            widths,
            decode_mask,
        ):
            self._reset()
            gate_mass, color_mass = fourth_root_balanced_hierarchical_mass(
                rows, cell_mask
            )

            def step_loss(
                event,
                row,
                mask,
                step_gate_mass,
                step_color_mass,
                height,
                width,
            ):
                query_width = self.model.config.max_grid_size
                query_one_hot = event[
                    ..., self.model.config.query_color_slice
                ].reshape(*event.shape[:-1], query_width, COLOR_COUNT)
                query_cells = jnp.argmax(query_one_hot, axis=-1)
                query_mask = event[..., self.model.config.query_mask_slice]
                if query_width < MAX_GRID_SIZE:
                    padding = [(0, 0)] * (query_cells.ndim - 1) + [
                        (0, MAX_GRID_SIZE - query_width)
                    ]
                    query_cells = jnp.pad(query_cells, padding)
                    query_mask = jnp.pad(query_mask, padding)
                return edit_weighted_hierarchical_step_loss(
                    learner(event),
                    row,
                    mask,
                    query_cells,
                    query_mask,
                    height,
                    width,
                    step_gate_mass,
                    step_color_mass,
                )

            gradients, objective = learner.etrace_grad(
                events,
                rows,
                cell_mask,
                gate_mass,
                color_mass,
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
        def train_many(
            events,
            rows,
            cell_mask,
            class_weights,
            heights,
            widths,
            decode_mask,
        ):
            def body(update_values):
                return train_one(*update_values, decode_mask)

            return brainstate.transform.for_loop(
                body, (events, rows, cell_mask, class_weights, heights, widths)
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
            jnp.asarray(chunk.class_weights),
            jnp.asarray(chunk.target_heights),
            jnp.asarray(chunk.target_widths),
            jnp.asarray(chunk.decode_mask),
        )
        observed = np.asarray(norms)
        return losses, {
            name: float(np.max(observed[:, index]))
            for index, name in enumerate(self.groups)
        }


def save_online_checkpoint(model: OnlineARCVanillaRNN, path: pathlib.Path) -> str:
    """Save an exact-schema online-model checkpoint.

    Parameters
    ----------
    model : OnlineARCVanillaRNN
        Model whose parameters are saved.
    path : pathlib.Path
        Destination ``.npz`` path.

    Returns
    -------
    str
        Ordered parameter digest.
    """

    if not isinstance(model, OnlineARCVanillaRNN):
        raise TypeError("model must be an OnlineARCVanillaRNN instance.")
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
) -> tuple[OnlineARCVanillaRNN, dict[str, object]]:
    """Load an exact-schema online-model checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        Existing checkpoint path.

    Returns
    -------
    model : OnlineARCVanillaRNN
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
            architecture_version = architecture.get("architecture_version")
            if architecture_version == "task_gated_operator_bank_v42":
                from examples.pp_prop.latent_workspace_expert_model import (
                    ExpertModelConfig,
                    TaskGatedOnlineRNN,
                )

                model = TaskGatedOnlineRNN(ExpertModelConfig(**architecture))
            elif architecture_version == "phase_separated_gated_memory_v44":
                from examples.pp_prop.latent_workspace_gated_memory_model import (
                    GatedMemoryConfig,
                    PhaseSeparatedGatedMemoryRNN,
                )

                model = PhaseSeparatedGatedMemoryRNN(
                    GatedMemoryConfig(**architecture)
                )
            elif architecture_version == "query_routing_gated_memory_v48":
                from examples.pp_prop.latent_workspace_query_routing_model import (
                    QueryRoutingConfig,
                    QueryRoutingGatedMemoryRNN,
                )

                model = QueryRoutingGatedMemoryRNN(
                    QueryRoutingConfig(**architecture)
                )
            else:
                model = OnlineARCVanillaRNN(OnlineModelConfig(**architecture))
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


def decode_hierarchical_online_outputs(
    outputs: np.ndarray,
    decode_mask: np.ndarray,
    parameter_dependencies: tuple[str, ...],
) -> dict[str, object]:
    """Decode a binary foreground gate and conditional colour logits.

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
        One fixed greedy first candidate directly serialized from model logits.
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
    gate_logits = np.asarray(row_logits[..., 0])
    nonzero_colors = np.asarray(jnp.argmax(row_logits[..., 1:], axis=-1)) + 1
    colors = np.where(gate_logits > 0.0, nonzero_colors, 0).astype(np.int32)
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
        "answer_head_version": "task_conditioned_query_patch_decoder_v39",
    }
