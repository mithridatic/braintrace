"""Packing, PP-prop training, and checkpoints for V46 paired spatial ARC."""

from __future__ import annotations

from collections import Counter
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

from examples.pp_prop.latent_workspace_direct_generation import (
    first_prediction_bytes,
    strict_task_pass_at_1,
)
from examples.pp_prop.latent_workspace_paired_spatial_model import (
    ANSWER_HEAD_VERSION,
    COLOR_COUNT,
    DEMO_PHASE_CHANNEL,
    EVENT_CHANNELS,
    INPUT_COLOR_CHANNEL,
    INPUT_MASK_CHANNEL,
    MAX_GRID_SIZE,
    OUTPUT_COLOR_CHANNEL,
    OUTPUT_MASK_CHANNEL,
    OUTPUT_WIDTH,
    PROPOSAL_SOURCE,
    QUERY_PHASE_CHANNEL,
    PairedSpatialARC,
    PairedSpatialConfig,
    validate_paired_spatial_event,
)
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcTask,
    AugmentationConfig,
    augment_training_task,
)

MAX_DEMONSTRATIONS = 10


@dataclass(frozen=True)
class PairedSpatialEpisode:
    """Hold one fixed V46 event sequence and separate scorer signals.

    Parameters
    ----------
    events : numpy.ndarray
        Target-free sequence shaped ``(steps, 30, 30, 6)``.
    target_colors, target_cell_mask : numpy.ndarray
        Repeated whole-grid labels and validity masks, never model inputs.
    target_heights, target_widths : numpy.ndarray
        Repeated zero-based output dimension labels.
    loss_step_mask : numpy.ndarray
        Boolean vector selecting only the final query-refinement step.
    task_id : str
        Host-only task identifier excluded from the model input.
    """

    events: np.ndarray
    target_colors: np.ndarray
    target_cell_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    loss_step_mask: np.ndarray
    task_id: str


@dataclass(frozen=True)
class PairedSpatialBatch:
    """Hold one time-major batch of V46 episodes.

    Parameters
    ----------
    events : numpy.ndarray
        Time-major target-free model inputs.
    target_colors, target_cell_mask : numpy.ndarray
        Time-major scorer-side grid labels and masks.
    target_heights, target_widths : numpy.ndarray
        Time-major zero-based output dimensions.
    loss_step_mask : numpy.ndarray
        Common one-dimensional final-step mask.
    """

    events: np.ndarray
    target_colors: np.ndarray
    target_cell_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    loss_step_mask: np.ndarray


@dataclass(frozen=True)
class PairedSpatialTrainingChunk:
    """Hold update-major arrays for one compiled V46 optimizer chunk.

    Parameters
    ----------
    events : numpy.ndarray
        Update-major, time-major target-free model inputs.
    target_colors, target_cell_mask : numpy.ndarray
        Matching scorer-side grid labels and masks.
    target_heights, target_widths : numpy.ndarray
        Matching zero-based dimension labels.
    loss_step_mask : numpy.ndarray
        Common final-step objective mask.
    """

    events: np.ndarray
    target_colors: np.ndarray
    target_cell_mask: np.ndarray
    target_heights: np.ndarray
    target_widths: np.ndarray
    loss_step_mask: np.ndarray


def _write_grid(event: np.ndarray, grid: ArcGrid, *, output: bool) -> None:
    color_channel = OUTPUT_COLOR_CHANNEL if output else INPUT_COLOR_CHANNEL
    mask_channel = OUTPUT_MASK_CHANNEL if output else INPUT_MASK_CHANNEL
    cells = np.asarray(grid.cells, dtype=np.float32)
    event[: grid.height, : grid.width, color_channel] = cells
    event[: grid.height, : grid.width, mask_channel] = 1.0


def encode_paired_spatial_episode(
    task: ArcTask,
    query_index: int,
    config: PairedSpatialConfig,
) -> PairedSpatialEpisode:
    """Encode one task query without exposing its output to V46.

    Parameters
    ----------
    task : ArcTask
        Task whose demonstrations are allowed model inputs.
    query_index : int
        Zero-based official-query index.
    config : PairedSpatialConfig
        Bound refinement-step count.

    Returns
    -------
    PairedSpatialEpisode
        Fixed ten-demo plus query-refinement sequence and separate labels.
    """

    if not isinstance(task, ArcTask):
        raise TypeError("task must be an ArcTask instance.")
    if isinstance(query_index, (bool, np.bool_)) or not isinstance(
        query_index, Integral
    ):
        raise TypeError("query_index must be a nonnegative integer.")
    index = int(query_index)
    if index < 0 or index >= len(task.test):
        raise ValueError("query_index is outside task.test.")
    if not isinstance(config, PairedSpatialConfig):
        raise TypeError("config must be a PairedSpatialConfig instance.")
    if len(task.train) > MAX_DEMONSTRATIONS:
        raise ValueError(f"task may contain at most {MAX_DEMONSTRATIONS} demonstrations.")
    query = task.test[index]
    if query.output is None:
        raise ValueError("query output is required only for training or scoring.")
    step_count = MAX_DEMONSTRATIONS + config.refinement_steps
    events = np.zeros(
        (step_count, MAX_GRID_SIZE, MAX_GRID_SIZE, EVENT_CHANNELS),
        dtype=np.float32,
    )
    for demo_index, pair in enumerate(task.train):
        event = events[demo_index]
        _write_grid(event, pair.input, output=False)
        if pair.output is None:
            raise ValueError("demonstration output must be present.")
        _write_grid(event, pair.output, output=True)
        event[..., DEMO_PHASE_CHANNEL] = 1.0
    for step in range(MAX_DEMONSTRATIONS, step_count):
        event = events[step]
        _write_grid(event, query.input, output=False)
        event[..., QUERY_PHASE_CHANNEL] = 1.0
    for event in events:
        validate_paired_spatial_event(event)

    target = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int32)
    target_mask = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    target_values = np.asarray(query.output.cells, dtype=np.int32)
    target[: query.output.height, : query.output.width] = target_values
    target_mask[: query.output.height, : query.output.width] = 1.0
    targets = np.broadcast_to(target, (step_count, *target.shape)).copy()
    masks = np.broadcast_to(
        target_mask, (step_count, *target_mask.shape)
    ).copy()
    heights = np.full((step_count,), query.output.height - 1, dtype=np.int32)
    widths = np.full((step_count,), query.output.width - 1, dtype=np.int32)
    loss_step_mask = np.zeros((step_count,), dtype=np.bool_)
    loss_step_mask[-1] = True
    return PairedSpatialEpisode(
        events=events,
        target_colors=targets,
        target_cell_mask=masks,
        target_heights=heights,
        target_widths=widths,
        loss_step_mask=loss_step_mask,
        task_id=task.task_id,
    )


def stack_paired_spatial_episodes(
    episodes: tuple[PairedSpatialEpisode, ...],
) -> PairedSpatialBatch:
    """Stack equally shaped V46 episodes along a batch axis.

    Parameters
    ----------
    episodes : tuple of PairedSpatialEpisode
        Nonempty ordered episode population.

    Returns
    -------
    PairedSpatialBatch
        Time-major arrays with batch axis one.
    """

    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    if not all(isinstance(item, PairedSpatialEpisode) for item in episodes):
        raise TypeError("episodes must contain PairedSpatialEpisode values.")
    reference = episodes[0].loss_step_mask
    if any(
        item.loss_step_mask.tobytes() != reference.tobytes() for item in episodes[1:]
    ):
        raise ValueError("episode loss-step masks must match exactly.")
    return PairedSpatialBatch(
        events=np.stack([item.events for item in episodes], axis=1),
        target_colors=np.stack(
            [item.target_colors for item in episodes], axis=1
        ),
        target_cell_mask=np.stack(
            [item.target_cell_mask for item in episodes], axis=1
        ),
        target_heights=np.stack(
            [item.target_heights for item in episodes], axis=1
        ),
        target_widths=np.stack(
            [item.target_widths for item in episodes], axis=1
        ),
        loss_step_mask=np.array(reference, copy=True),
    )


def repeat_paired_spatial_batch(
    batch: PairedSpatialBatch, updates: int
) -> PairedSpatialTrainingChunk:
    """Repeat one V46 batch for a deterministic compiled descent probe.

    Parameters
    ----------
    batch : PairedSpatialBatch
        Static time-major batch.
    updates : int
        Positive update count.

    Returns
    -------
    PairedSpatialTrainingChunk
        Update-major repeated arrays.
    """

    if not isinstance(batch, PairedSpatialBatch):
        raise TypeError("batch must be a PairedSpatialBatch instance.")
    count = _positive_integer(updates, "updates")

    def repeat(array: np.ndarray) -> np.ndarray:
        return np.repeat(array[None, ...], count, axis=0)

    return PairedSpatialTrainingChunk(
        events=repeat(batch.events),
        target_colors=repeat(batch.target_colors),
        target_cell_mask=repeat(batch.target_cell_mask),
        target_heights=repeat(batch.target_heights),
        target_widths=repeat(batch.target_widths),
        loss_step_mask=np.array(batch.loss_step_mask, copy=True),
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def sample_paired_spatial_training_chunk(
    catalog: tuple[ArcTask, ...],
    config: PairedSpatialConfig,
    rng: brainstate.random.RandomState,
    *,
    updates: int,
    batch_size: int,
    augment: bool,
) -> PairedSpatialTrainingChunk:
    """Sample and host-encode one update-major V46 training chunk.

    Parameters
    ----------
    catalog : tuple of ArcTask
        Target-isolated fitting episodes with one supervised query each.
    config : PairedSpatialConfig
        Bound V46 sequence layout.
    rng : brainstate.random.RandomState
        Sole sampling and augmentation stream.
    updates, batch_size : int
        Positive compiled chunk dimensions.
    augment : bool
        Whether to apply target-preserving colour, dihedral, and order changes.

    Returns
    -------
    PairedSpatialTrainingChunk
        Update-major target-free inputs and separate scorer labels.
    """

    if not isinstance(catalog, tuple) or not catalog:
        raise ValueError("catalog must be a nonempty tuple.")
    if not isinstance(config, PairedSpatialConfig):
        raise TypeError("config must be a PairedSpatialConfig instance.")
    if not isinstance(rng, brainstate.random.RandomState):
        raise TypeError("rng must be a brainstate.random.RandomState.")
    if not isinstance(augment, bool):
        raise TypeError("augment must be boolean.")
    update_count = _positive_integer(updates, "updates")
    static_batch = _positive_integer(batch_size, "batch_size")
    indices = np.asarray(
        rng.randint(0, len(catalog), size=(update_count, static_batch)),
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
        for raw_index in update_indices:
            task = catalog[int(raw_index)]
            if augment:
                task = augment_training_task(
                    task, rng, role="train", config=augmentation
                )
            episodes.append(encode_paired_spatial_episode(task, 0, config))
        batches.append(stack_paired_spatial_episodes(tuple(episodes)))
    return PairedSpatialTrainingChunk(
        events=np.stack([item.events for item in batches]),
        target_colors=np.stack([item.target_colors for item in batches]),
        target_cell_mask=np.stack(
            [item.target_cell_mask for item in batches]
        ),
        target_heights=np.stack([item.target_heights for item in batches]),
        target_widths=np.stack([item.target_widths for item in batches]),
        loss_step_mask=np.array(batches[0].loss_step_mask, copy=True),
    )


def paired_spatial_hierarchical_mass(
    target_colors: jax.Array, target_cell_mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute fourth-root gate and equal foreground-colour masses.

    Parameters
    ----------
    target_colors : jax.Array
        Batched integer whole-grid targets shaped ``(batch, height, width)``.
    target_cell_mask : jax.Array
        Matching valid-cell mask.

    Returns
    -------
    gate_mass, color_mass : tuple of jax.Array
        Per-example mass for background/foreground and colours one through nine.
    """

    if target_colors.shape != target_cell_mask.shape or target_colors.ndim != 3:
        raise ValueError("target_colors and target_cell_mask must match in 3-D.")
    mask = jnp.asarray(target_cell_mask, dtype=jnp.float32)
    gate_targets = (target_colors != 0).astype(jnp.int32)
    gate_ids = jnp.arange(2, dtype=jnp.int32)
    gate_counts = jnp.sum(
        mask[..., None] * (gate_targets[..., None] == gate_ids), axis=(1, 2)
    )
    roots = jnp.sqrt(jnp.sqrt(gate_counts))
    gate_totals = roots / jnp.maximum(jnp.sum(roots, axis=-1, keepdims=True), 1.0)
    gate_mass = jnp.where(
        gate_counts > 0.0,
        gate_totals / jnp.maximum(gate_counts, 1.0),
        0.0,
    )
    color_ids = jnp.arange(1, COLOR_COUNT, dtype=jnp.int32)
    color_counts = jnp.sum(
        mask[..., None] * (target_colors[..., None] == color_ids), axis=(1, 2)
    )
    color_present = (color_counts > 0.0).astype(jnp.float32)
    per_color_total = color_present / jnp.maximum(
        jnp.sum(color_present, axis=-1, keepdims=True), 1.0
    )
    color_mass = jnp.where(
        color_counts > 0.0,
        per_color_total / jnp.maximum(color_counts, 1.0),
        0.0,
    )
    return gate_mass, color_mass


def _split_output(output: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    if output.shape[-1] != OUTPUT_WIDTH:
        raise ValueError(f"output last dimension must be {OUTPUT_WIDTH}.")
    grid_end = MAX_GRID_SIZE * MAX_GRID_SIZE * COLOR_COUNT
    height_end = grid_end + MAX_GRID_SIZE
    grid = output[..., :grid_end].reshape(
        *output.shape[:-1], MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT
    )
    return grid, output[..., grid_end:height_end], output[..., height_end:]


def paired_spatial_hierarchical_loss(
    output: jax.Array,
    target_colors: jax.Array,
    target_cell_mask: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    gate_mass: jax.Array,
    color_mass: jax.Array,
) -> jax.Array:
    """Score V46 foreground gates, colours, and dimensions at one step.

    Parameters
    ----------
    output : jax.Array
        Batched V46 whole-grid and shape logits.
    target_colors, target_cell_mask : jax.Array
        Batched padded target grids and validity masks.
    target_height, target_width : jax.Array
        Batched zero-based shape labels.
    gate_mass, color_mass : jax.Array
        Per-example masses from :func:`paired_spatial_hierarchical_mass`.

    Returns
    -------
    jax.Array
        Scalar objective with coefficients 0.4, 0.4, 0.1, and 0.1.
    """

    if gate_mass.shape[-1] != 2:
        raise ValueError("gate_mass last dimension must be 2.")
    if color_mass.shape[-1] != COLOR_COUNT - 1:
        raise ValueError(f"color_mass last dimension must be {COLOR_COUNT - 1}.")
    grid_logits, height_logits, width_logits = _split_output(output)
    mask = jnp.asarray(target_cell_mask, dtype=grid_logits.dtype)
    gate_targets = (target_colors != 0).astype(jnp.int32)
    gate_per_cell = optax.sigmoid_binary_cross_entropy(
        grid_logits[..., 0], gate_targets
    )
    selected_gate_mass = jnp.take_along_axis(
        gate_mass[:, None, None, :], gate_targets[..., None], axis=-1
    )[..., 0]
    gate_loss = jnp.mean(jnp.sum(gate_per_cell * mask * selected_gate_mass, axis=(1, 2)))
    conditional_targets = jnp.clip(target_colors - 1, 0, COLOR_COUNT - 2)
    color_per_cell = optax.softmax_cross_entropy_with_integer_labels(
        grid_logits[..., 1:], conditional_targets
    )
    selected_color_mass = jnp.take_along_axis(
        color_mass[:, None, None, :], conditional_targets[..., None], axis=-1
    )[..., 0]
    foreground_mask = mask * gate_targets.astype(mask.dtype)
    color_loss = jnp.mean(
        jnp.sum(
            color_per_cell * foreground_mask * selected_color_mass, axis=(1, 2)
        )
    )
    height_loss = optax.softmax_cross_entropy_with_integer_labels(
        height_logits, target_height
    ).mean()
    width_loss = optax.softmax_cross_entropy_with_integer_labels(
        width_logits, target_width
    ).mean()
    return 0.4 * gate_loss + 0.4 * color_loss + 0.1 * height_loss + 0.1 * width_loss


def _parameter_group(path: tuple[object, ...]) -> str:
    groups = {
        "demo_input_conv": "demo_input",
        "demo_recurrent_conv": "demo_recurrent",
        "query_input_conv": "query_input",
        "query_recurrent_conv": "query_recurrent",
        "color_head": "color",
        "height_head": "height",
        "width_head": "width",
    }
    try:
        return groups[str(path[0])]
    except KeyError as error:
        raise ValueError(f"V46 parameter path {path!r} has no group.") from error


def _array(value: object) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value))


def paired_spatial_parameter_leaf_arrays(
    model: PairedSpatialARC,
) -> dict[str, np.ndarray]:
    """Return every ordered V46 parameter leaf as a copied array.

    Parameters
    ----------
    model : PairedSpatialARC
        Model whose checkpoint-owned leaves are returned.

    Returns
    -------
    dict
        Ordered ``state-path#leaf-index`` to contiguous arrays.
    """

    if not isinstance(model, PairedSpatialARC):
        raise TypeError("model must be a PairedSpatialARC instance.")
    arrays = {}
    for path, state in model.states(brainstate.ParamState).items():
        name = ".".join(map(str, path))
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            arrays[f"{name}#{index}"] = _array(leaf).copy()
    return arrays


def paired_spatial_parameter_arrays(
    model: PairedSpatialARC,
) -> dict[str, np.ndarray]:
    """Return deterministic V46 parameter bytes grouped by model role.

    Parameters
    ----------
    model : PairedSpatialARC
        Model whose trainable parameters are grouped.

    Returns
    -------
    dict
        Contiguous byte arrays for all seven answer-path groups.
    """

    if not isinstance(model, PairedSpatialARC):
        raise TypeError("model must be a PairedSpatialARC instance.")
    chunks = {
        name: []
        for name in (
            "demo_input",
            "demo_recurrent",
            "query_input",
            "query_recurrent",
            "color",
            "height",
            "width",
        )
    }
    for path, state in model.states(brainstate.ParamState).items():
        chunks[_parameter_group(path)].extend(
            _array(leaf).tobytes() for leaf in jax.tree.leaves(state.value)
        )
    return {
        name: np.frombuffer(b"".join(values), dtype=np.uint8).copy()
        for name, values in chunks.items()
    }


def paired_spatial_parameter_digest(model: PairedSpatialARC) -> str:
    """Hash ordered V46 parameter paths, metadata, and bytes.

    Parameters
    ----------
    model : PairedSpatialARC
        Model whose exact checkpoint state is hashed.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """

    if not isinstance(model, PairedSpatialARC):
        raise TypeError("model must be a PairedSpatialARC instance.")
    digest = hashlib.sha256()
    for path, state in model.states(brainstate.ParamState).items():
        digest.update(".".join(map(str, path)).encode("utf-8"))
        for leaf in jax.tree.leaves(state.value):
            array = _array(leaf)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class PairedSpatialPPPropTrainer:
    """Train V46 with compiled single-step PP-prop.

    Parameters
    ----------
    model : PairedSpatialARC
        Model updated in place.
    batch_size : int
        Positive static batch size.
    learning_rate : float, default=0.001
        Positive Adam learning rate.
    trace_decay : float, default=2 ** (-1 / 40)
        PP-prop eligibility decay in ``(0, 1]``.
    """

    algorithm = "pp_prop"
    vjp_method = "single-step"
    loss_version = "paired_spatial_fourth_root_hierarchical_v46"

    def __init__(
        self,
        model: PairedSpatialARC,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
        trace_decay: float = 2.0 ** (-1.0 / 40.0),
    ):
        if not isinstance(model, PairedSpatialARC):
            raise TypeError("model must be a PairedSpatialARC instance.")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        for name, raw in (("learning_rate", learning_rate), ("trace_decay", trace_decay)):
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
                (
                    self.batch_size,
                    MAX_GRID_SIZE,
                    MAX_GRID_SIZE,
                    EVENT_CHANNELS,
                ),
                dtype=jnp.float32,
            ),
            batch_size=self.batch_size,
            vmap=False,
            decay_or_rank=self.trace_decay,
            vjp_method=self.vjp_method,
        )
        self.optimizer = braintools.optim.Adam(lr=self.learning_rate)
        self.optimizer.register_trainable_weights(self.weights)
        grouped = {
            name: []
            for name in (
                "demo_input",
                "demo_recurrent",
                "query_input",
                "query_recurrent",
                "color",
                "height",
                "width",
            )
        }
        for path in self.weights:
            grouped[_parameter_group(path)].append(path)
        self.groups = {name: tuple(paths) for name, paths in grouped.items()}
        if any(not paths for paths in self.groups.values()):
            raise ValueError("Every V46 parameter group must be nonempty.")
        self._train_many = self._make_train_many()

    def _reset(self) -> None:
        brainstate.nn.reset_all_states(self.model, batch_size=self.batch_size)
        self.learner.reset_state(batch_size=self.batch_size)

    def _make_train_many(self):
        learner = self.learner
        optimizer = self.optimizer
        groups = self.groups

        def gradient_norm(gradients, paths):
            leaves = [
                leaf
                for path in paths
                for leaf in jax.tree.leaves(gradients[path])
            ]
            return jnp.sqrt(
                sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves)
            )

        def train_one(events, colors, masks, heights, widths, loss_step_mask):
            self._reset()
            gate_mass, color_mass = paired_spatial_hierarchical_mass(
                colors[-1], masks[-1]
            )
            step_count = events.shape[0]
            gate_steps = jnp.broadcast_to(
                gate_mass[None, ...], (step_count, *gate_mass.shape)
            )
            color_steps = jnp.broadcast_to(
                color_mass[None, ...], (step_count, *color_mass.shape)
            )

            def step_loss(
                event,
                target,
                mask,
                height,
                width,
                step_gate_mass,
                step_color_mass,
            ):
                return paired_spatial_hierarchical_loss(
                    learner(event),
                    target,
                    mask,
                    height,
                    width,
                    step_gate_mass,
                    step_color_mass,
                )

            gradients, objective = learner.etrace_grad(
                events,
                colors,
                masks,
                heights,
                widths,
                gate_steps,
                color_steps,
                step_fn=step_loss,
                mask=loss_step_mask,
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
        def train_many(events, colors, masks, heights, widths, loss_step_mask):
            def body(values):
                return train_one(*values, loss_step_mask)

            return brainstate.transform.for_loop(
                body, (events, colors, masks, heights, widths)
            )

        return train_many

    def train_chunk(
        self, chunk: PairedSpatialTrainingChunk
    ) -> tuple[jax.Array, dict[str, float]]:
        """Run one compiled V46 optimizer chunk.

        Parameters
        ----------
        chunk : PairedSpatialTrainingChunk
            Update-major model inputs and separate labels.

        Returns
        -------
        losses : jax.Array
            One scalar objective per update.
        gradient_norms : dict
            Maximum observed norm for every parameter group.
        """

        if not isinstance(chunk, PairedSpatialTrainingChunk):
            raise TypeError("chunk must be a PairedSpatialTrainingChunk instance.")
        if chunk.events.shape[2] != self.batch_size:
            raise ValueError(
                f"chunk batch axis must equal trainer batch_size {self.batch_size}."
            )
        losses, norms = self._train_many(
            jnp.asarray(chunk.events),
            jnp.asarray(chunk.target_colors),
            jnp.asarray(chunk.target_cell_mask),
            jnp.asarray(chunk.target_heights),
            jnp.asarray(chunk.target_widths),
            jnp.asarray(chunk.loss_step_mask),
        )
        observed = np.asarray(norms)
        return losses, {
            name: float(np.max(observed[:, index]))
            for index, name in enumerate(self.groups)
        }


def _decode_paired_spatial_output(
    output: np.ndarray, parameter_dependencies: tuple[str, ...]
) -> dict[str, object]:
    values = np.asarray(output)
    if values.shape != (OUTPUT_WIDTH,) or not np.all(np.isfinite(values)):
        raise ValueError("V46 decode output must be one finite fixed-width vector.")
    grid_logits, height_logits, width_logits = _split_output(jnp.asarray(values))
    grid_array = np.asarray(grid_logits)
    foreground = grid_array[..., 0] > 0.0
    nonzero = np.asarray(jnp.argmax(grid_logits[..., 1:], axis=-1)) + 1
    colors = np.where(foreground, nonzero, 0).astype(np.int32)
    height = int(np.argmax(np.asarray(height_logits))) + 1
    width = int(np.argmax(np.asarray(width_logits))) + 1
    return {
        "rank": 1,
        "height": height,
        "width": width,
        "grid": colors[:height, :width].tolist(),
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "parameter_dependencies": list(parameter_dependencies),
        "proposal_source": PROPOSAL_SOURCE,
        "selection_role": "greedy_argmax",
        "ranking_source": "none_single_greedy_candidate",
        "answer_head_version": ANSWER_HEAD_VERSION,
    }


def evaluate_paired_spatial_model(
    model: PairedSpatialARC,
    episodes: tuple[PairedSpatialEpisode, ...],
    *,
    trace_decay: float = 2.0 ** (-1.0 / 40.0),
    batch_size: int = 8,
) -> dict[str, object]:
    """Execute target-free V46 recurrence and strict greedy scoring.

    Parameters
    ----------
    model : PairedSpatialARC
        Frozen checkpoint-owned recurrent model.
    episodes : tuple of PairedSpatialEpisode
        Ordered target-free inputs with scorer-side labels.
    trace_decay : float, default=2 ** (-1 / 40)
        Trace setting used only by the forward compiler wrapper.
    batch_size : int, default=8
        Positive static evaluation batch size.

    Returns
    -------
    dict
        Exact memberships, first candidates, digest, and diagnostics.
    """

    if not isinstance(model, PairedSpatialARC):
        raise TypeError("model must be a PairedSpatialARC instance.")
    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    static_batch = min(_positive_integer(batch_size, "batch_size"), len(episodes))
    padding = (-len(episodes)) % static_batch
    padded = episodes + (episodes[-1],) * padding
    batches = tuple(
        stack_paired_spatial_episodes(padded[index : index + static_batch])
        for index in range(0, len(padded), static_batch)
    )
    brainstate.nn.init_all_states(model, batch_size=static_batch)
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros(
            (static_batch, MAX_GRID_SIZE, MAX_GRID_SIZE, EVENT_CHANNELS),
            dtype=jnp.float32,
        ),
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

    raw = np.asarray(
        run_all(jnp.asarray(np.stack([batch.events for batch in batches])))
    )
    ordered = raw.transpose(0, 2, 1, 3).reshape(
        -1, raw.shape[1], raw.shape[3]
    )[: len(episodes)]
    dependencies = tuple(
        ".".join(map(str, path)) for path in model.states(brainstate.ParamState)
    )
    candidates = []
    predictions = []
    targets = []
    task_ids = []
    for episode, outputs in zip(episodes, ordered, strict=True):
        candidate = _decode_paired_spatial_output(outputs[-1], dependencies)
        candidates.append(candidate)
        predictions.append(candidate["grid"])
        height = int(episode.target_heights[-1]) + 1
        width = int(episode.target_widths[-1]) + 1
        targets.append(
            episode.target_colors[-1, :height, :width].astype(np.int32).tolist()
        )
        task_ids.append(episode.task_id)
    score = strict_task_pass_at_1(predictions, targets, task_ids)
    candidate_bytes = first_prediction_bytes(candidates)
    shape_pairs = [
        (np.asarray(prediction), np.asarray(target))
        for prediction, target in zip(predictions, targets, strict=True)
        if np.asarray(prediction).shape == np.asarray(target).shape
    ]
    foreground_total = sum(
        int(np.count_nonzero(target != 0)) for _, target in shape_pairs
    )
    background_total = sum(
        int(np.count_nonzero(target == 0)) for _, target in shape_pairs
    )
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
        **score,
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
                str(color): count
                for color, count in sorted(predicted_colors.items())
            },
        },
        "candidates": candidates,
    }


def save_paired_spatial_checkpoint(
    model: PairedSpatialARC, path: pathlib.Path
) -> str:
    """Save an exact-schema V46 checkpoint.

    Parameters
    ----------
    model : PairedSpatialARC
        Model whose ordered parameters are saved.
    path : pathlib.Path
        Destination ``.npz`` path.

    Returns
    -------
    str
        Ordered parameter digest.
    """

    if not isinstance(model, PairedSpatialARC):
        raise TypeError("model must be a PairedSpatialARC instance.")
    path = pathlib.Path(path)
    arrays = {}
    leaves_metadata = []
    cursor = 0
    for state_path, state in model.states(brainstate.ParamState).items():
        state_name = ".".join(map(str, state_path))
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            key = f"leaf_{cursor:04d}"
            array = _array(leaf)
            arrays[key] = array
            leaves_metadata.append(
                {
                    "key": key,
                    "state_path": state_name,
                    "tree_leaf_index": index,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
            )
            cursor += 1
    digest = paired_spatial_parameter_digest(model)
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


def load_paired_spatial_checkpoint(
    path: pathlib.Path,
) -> tuple[PairedSpatialARC, dict[str, object]]:
    """Load and validate an exact-schema V46 checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        Existing V46 checkpoint path.

    Returns
    -------
    model : PairedSpatialARC
        Restored model.
    metadata : dict
        Validated checkpoint metadata.
    """

    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "__metadata__" not in names:
                raise ValueError("V46 checkpoint metadata is missing.")
            metadata = msgspec.json.decode(bytes(archive["__metadata__"]))
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
                raise ValueError("V46 checkpoint schema is unsupported.")
            architecture = metadata.get("architecture")
            leaves_metadata = metadata.get("leaves")
            if not isinstance(architecture, dict) or not isinstance(
                leaves_metadata, list
            ):
                raise ValueError("V46 checkpoint metadata is invalid.")
            model = PairedSpatialARC(PairedSpatialConfig(**architecture))
            expected = {
                item.get("key") for item in leaves_metadata if isinstance(item, dict)
            }
            if names != expected | {"__metadata__"}:
                raise ValueError("V46 checkpoint leaf set is invalid.")
            cursor = 0
            for state_path, state in model.states(brainstate.ParamState).items():
                state_name = ".".join(map(str, state_path))
                originals, structure = jax.tree.flatten(state.value)
                restored = []
                for index, original in enumerate(originals):
                    item = leaves_metadata[cursor]
                    key = item.get("key")
                    array = np.asarray(archive[key])
                    original_array = _array(original)
                    if (
                        item.get("state_path") != state_name
                        or item.get("tree_leaf_index") != index
                        or item.get("shape") != list(original_array.shape)
                        or item.get("dtype") != original_array.dtype.str
                        or list(array.shape) != item.get("shape")
                        or array.dtype.str != item.get("dtype")
                        or not np.all(np.isfinite(array))
                    ):
                        raise ValueError("V46 checkpoint leaf schema is invalid.")
                    restored.append(jnp.asarray(array))
                    cursor += 1
                state.value = jax.tree.unflatten(structure, restored)
            if cursor != len(leaves_metadata):
                raise ValueError("V46 checkpoint has unexpected leaves.")
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        msgspec.DecodeError,
    ) as error:
        raise ValueError("V46 checkpoint could not be decoded safely.") from error
    expected_digest = metadata.get("parameter_sha256")
    if (
        not isinstance(expected_digest, str)
        or paired_spatial_parameter_digest(model) != expected_digest
    ):
        raise ValueError("V46 checkpoint digest does not match its contents.")
    return model, metadata
